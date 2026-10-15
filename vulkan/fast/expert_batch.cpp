#include "fast/expert_batch.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <ggml.h>
#include <stdexcept>
namespace strata::vulkan::fast {
ExpertBatch::ExpertBatch(Device &device, uint32_t capacity, bool portable, bool gpu_pack, bool vector_weights, uint32_t tile_columns, bool replay, uint32_t tile_rows)
    : device_(device), capacity_(capacity), tile_columns_(tile_columns), tile_rows_(tile_rows), portable_rounding_(portable), gpu_pack_(gpu_pack), vector_weights_(vector_weights), replay_(replay) {
  if (!capacity || capacity > 2048) throw std::invalid_argument("expert batch capacity must be 1..2048");
  if (tile_columns != 16 && tile_columns != 32 && tile_columns != 64)
    throw std::invalid_argument("expert batch column tile must be 16, 32 or 64");
  if (tile_rows != 32 && tile_rows != 64)
    throw std::invalid_argument("expert batch row tile must be 32 or 64");
  if (device.fp64_expert_scale() && (tile_columns != 64 || tile_rows != 64))
    throw std::invalid_argument("FP64 expert scale diagnostic requires 64x64 tiles");
  quantized_row_bytes_ = ggml_row_size(GGML_TYPE_Q8_K, 4096);
  if (ggml_row_size(GGML_TYPE_Q8_K, 256) != 292)
    throw std::invalid_argument("streamed expert shader requires 292-byte Q8_K blocks");
  down_layout_ = layout_for(Q3_K, 2048, 4096);
  gu_weights_.layout = layout_for(Q2_K, 4096, 4096);
  down_weights_.layout = down_layout_;
  gu_weights_.data = device.buffer(gu_weights_.bytes(), Memory::Device);
  down_weights_.data = device.buffer(down_weights_.bytes(), Memory::Device);
  if (gpu_pack_) {
    raw_gu_ = device.buffer(row_bytes(Q2_K, 4096) * 4096, Memory::Device);
    raw_down_ = device.buffer(row_bytes(Q3_K, 2048) * 4096, Memory::Device);
  } else packed_.resize(std::max(gu_weights_.bytes(), down_weights_.bytes()));
  input_ = device.buffer(size_t(capacity) * 4096 * 4, Memory::Host);
  output_ = device.buffer(size_t(capacity) * 4096 * 4, Memory::Host);
  qinput_ = device.buffer(size_t(capacity) * 4096 * 4, Memory::Device);
  gu_ = device.buffer(size_t(capacity) * 4096 * 4, Memory::Device);
  hidden_ = device.buffer(size_t(capacity) * 2048 * 4, Memory::Device);
  qhidden_ = device.buffer(size_t(capacity) * 2048 * 4, Memory::Device);
  input_meta_ = device.buffer(size_t(capacity) * 16 * 17 * 4, Memory::Device);
  hidden_meta_ = device.buffer(size_t(capacity) * 8 * 17 * 4, Memory::Device);
  partial_ = device.buffer(size_t(capacity) * 4096 * 8 * 4, Memory::Device);
  hits_ = device.buffer(256, Memory::Host);
  hits_->as<uint32_t>()[0] = 1;
}

ExpertBatch::~ExpertBatch() {
  // run_impl waits on each command's fence before returning. Drop commands
  // before their descriptor buffers are destroyed.
  for (const auto &[key, command] : commands_) device_.free_command(command);
}

void ExpertBatch::run(const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                      std::span<const float *const> inputs, std::span<float *const> outputs, float limit,
                      ExpertBatchTrace *trace) {
  run_impl(gate, up, down, inputs, {}, outputs, limit, trace);
}
void ExpertBatch::run_quantized(const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                                std::span<const uint8_t *const> inputs, std::span<float *const> outputs,
                                float limit, ExpertBatchTrace *trace) {
  run_impl(gate, up, down, {}, inputs, outputs, limit, trace);
}
void ExpertBatch::run_impl(const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                           std::span<const float *const> inputs, std::span<const uint8_t *const> quantized,
                           std::span<float *const> outputs, float limit, ExpertBatchTrace *trace) {
  const size_t size = outputs.size();
  const bool packed_input = !quantized.empty();
  if (!size || size > capacity_ || (packed_input ? quantized.size() != size || !inputs.empty() : inputs.size() != size))
    throw std::invalid_argument("expert batch input/output count invalid");
  const auto count = uint32_t(size);
  const bool lane_trace = trace && device_.fp64_expert_scale();
  if (lane_trace && !gate_lane_prefix_) gate_lane_prefix_ = device_.buffer(8 * 16 * 4, Memory::Host);
  if (lane_trace && !gate_block_scale_) gate_block_scale_ = device_.buffer(8 * 16 * 8 * 4, Memory::Host);
  const size_t input_stride = packed_input ? quantized_row_bytes_ : 4096 * 4;
  for (uint32_t t = 0; t < count; ++t)
    std::memcpy(static_cast<uint8_t *>(input_->mapped) + size_t(t) * input_stride,
                packed_input ? static_cast<const void *>(quantized[t]) : inputs[t], input_stride);
  stats_.input_bytes += size * input_stride;
  using Clock = std::chrono::steady_clock;
  auto started = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - started).count(); };
  // Read the separate GGUF row ranges directly into their combined block
  // planes. Concatenating already-repacked matrices would order them wrongly.
  if (gpu_pack_) {
    const size_t gate_bytes = row_bytes(Q2_K, 4096) * 2048;
    device_.upload(*raw_gu_, gate, gate_bytes);
    device_.upload(*raw_gu_, up, gate_bytes, gate_bytes);
    device_.upload(*raw_down_, down, row_bytes(Q3_K, 2048) * 4096);
  } else {
    repack_pair_streamed(gu_weights_.layout, gate, gu_weights_.m() / 2, up, packed_.data());
    stats_.pack_ms += elapsed(); started = Clock::now();
    device_.upload(*gu_weights_.data, packed_.data(), gu_weights_.bytes());
    stats_.upload_ms += elapsed(); started = Clock::now();
    repack_streamed(down_layout_, down, packed_.data());
    stats_.pack_ms += elapsed(); started = Clock::now();
    device_.upload(*down_weights_.data, packed_.data(), down_weights_.bytes());
  }
  stats_.upload_ms += elapsed(); started = Clock::now();
  const std::string round_kernel = std::string(portable_rounding_ ? "q8k_codes_portable" : "q8k_codes") +
                                   (device_.fp64_expert_math() ? "_fp64_math" : "");
  struct {uint32_t mode, n, uofs, tokens; float limit;} swiglu{0, 2048, 0, count, limit};
  auto record = [&](Recorder &r) {
    if (gpu_pack_) {
      r.begin_concurrent();
      repack_gpu(r, gu_weights_.layout, View(*raw_gu_), View(*gu_weights_.data));
      repack_gpu(r, down_layout_, View(*raw_down_), View(*down_weights_.data));
      r.end_concurrent();
    }
    auto product = [&](const Matrix &w, const Buffer &codes, const Buffer &metadata, const Buffer &out) {
      const std::string kernel = std::string(w.layout.type == Q2_K ? "expert_gemm_q2k_canon" : "expert_gemm_q3k_canon") +
                                 (vector_weights_ ? "_vector" : "") +
                                 (tile_columns_ == 64 ? "" : "_" + std::to_string(tile_columns_)) +
                                 (tile_rows_ == 64 ? "" : "_m32") +
                                 (device_.fp64_expert_scale() ? "_fp64_scale" : "");
      const std::array<uint32_t, 5> push{w.k(), w.m(), w.layout.mp, count, uint32_t(w.layout.chunk_bytes())};
      if (lane_trace && w.layout.type == Q2_K)
        r.dispatch(kernel + "_trace", {View(*w.data), View(codes), View(metadata), View(*partial_), View(*gate_block_scale_)},
                   push, (w.m() + tile_rows_ - 1) / tile_rows_, (count + tile_columns_ - 1) / tile_columns_, 8);
      else
        r.dispatch(kernel, {View(*w.data), View(codes), View(metadata), View(*partial_)},
                   push, (w.m() + tile_rows_ - 1) / tile_rows_, (count + tile_columns_ - 1) / tile_columns_, 8);
      if (lane_trace && w.layout.type == Q2_K)
        for (size_t lane = 0; lane < 8; ++lane)
          r.copy(View(*partial_, lane * count * w.m() * 4, 16 * 4),
                 View(*gate_lane_prefix_, lane * 16 * 4, 16 * 4), 16 * 4);
      r.dispatch("expert_reduce_canon", {View(*partial_), View(out)},
                 std::array<uint32_t, 1>{count * w.m()}, (count * w.m() + 255) / 256);
    };
    if (packed_input)
      r.dispatch("q8k_unpack", {View(*input_), View(*qinput_), View(*input_meta_)},
                 static_cast<const void *>(nullptr), 0, count * 4096 / 256);
    else
      r.dispatch(round_kernel, {View(*input_), View(*qinput_), View(*hits_), View(*input_meta_)},
                 std::array<uint32_t, 4>{count * 4096, 0, 0, count}, count * 4096 / 256);
    product(gu_weights_, *qinput_, *input_meta_, *gu_);
    r.dispatch(device_.fp64_expert_math() ? "expert_mix_fp64_math" : "expert_mix_canon",
               {View(*gu_), View(*hidden_), View(*hits_)}, swiglu, count * 2048 / 256);
    r.dispatch(round_kernel, {View(*hidden_), View(*qhidden_), View(*hits_), View(*hidden_meta_)},
               std::array<uint32_t, 4>{count * 2048, 0, 0, count}, count * 2048 / 256);
    product(down_weights_, *qhidden_, *hidden_meta_, *output_);
    r.host_visible();
  };
  // Buffer bindings, geometry, input encoding and clamp are fixed in a
  // recorded command. Weight/input bytes are uploaded afresh before replay.
  // Bound the number of retained commands even for many distinct counts.
  const CommandKey key{count, std::bit_cast<uint32_t>(limit), packed_input};
  auto found = commands_.find(key);
  VkCommandBuffer command = VK_NULL_HANDLE;
  bool retained = false;
  if (replay_ && !lane_trace && found != commands_.end()) {
    command = found->second; retained = true; ++stats_.command_replays;
  } else {
    const auto began = Clock::now();
    const bool keep = replay_ && !lane_trace && commands_.size() < 128;
    command = device_.allocate_command();
    try {
      device_.begin(command, !keep);
      Recorder r(device_, command);
      record(r);
      device_.end(command);
      if (keep) {commands_.emplace(key, command); retained = true;}
    } catch (...) {device_.free_command(command); throw;}
    ++stats_.command_records;
    stats_.record_ms += std::chrono::duration<double, std::milli>(Clock::now() - began).count();
  }
  const auto submitted = Clock::now();
  try {device_.submit_wait(command);}
  catch (...) {if (!retained) device_.free_command(command); throw;}
  stats_.submit_wait_ms += std::chrono::duration<double, std::milli>(Clock::now() - submitted).count();
  if (!retained) device_.free_command(command);
  stats_.queued_work_ms += elapsed();
  if (trace) {
    auto capture = [&](const Buffer &buffer, std::vector<float> &values, size_t size) {
      values.resize(size); device_.download(buffer, values.data(), size * 4);
    };
    capture(*qinput_, trace->qinput, size_t(count) * 4096);
    capture(*gu_, trace->gate_up, size_t(count) * 4096);
    capture(*hidden_, trace->hidden, size_t(count) * 2048);
    capture(*qhidden_, trace->qhidden, size_t(count) * 2048);
    trace->gate_lane_prefix.clear();
    if (lane_trace) capture(*gate_lane_prefix_, trace->gate_lane_prefix, 8 * 16);
    trace->gate_block_scale.clear();
    if (lane_trace) capture(*gate_block_scale_, trace->gate_block_scale, 8 * 16 * 8);
    auto dequantize = [&](std::vector<float> &codes, const Buffer &metadata) {
      std::vector<float> meta(codes.size() / 256 * 17);
      device_.download(metadata, meta.data(), meta.size() * 4);
      for (size_t i = 0; i < codes.size(); ++i) codes[i] *= meta[i / 256 * 17];
    };
    dequantize(trace->qinput, *input_meta_);
    dequantize(trace->qhidden, *hidden_meta_);
  }
  for (uint32_t t = 0; t < count; ++t) std::memcpy(outputs[t], output_->as<float>() + size_t(t) * 4096, 4096 * 4);
}
} // namespace strata::vulkan::fast
