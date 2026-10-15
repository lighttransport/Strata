#include "fast/expert_batch.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ggml.h>
#include <stdexcept>
namespace strata::vulkan::fast {
ExpertBatch::ExpertBatch(Device &device, uint32_t capacity, bool portable)
    : device_(device), capacity_(capacity), portable_rounding_(portable) {
  if (!capacity || capacity > 2048) throw std::invalid_argument("expert batch capacity must be 1..2048");
  quantized_row_bytes_ = ggml_row_size(GGML_TYPE_Q8_K, 4096);
  if (ggml_row_size(GGML_TYPE_Q8_K, 256) != 292)
    throw std::invalid_argument("streamed expert shader requires 292-byte Q8_K blocks");
  down_layout_ = layout_for(Q3_K, 2048, 4096);
  gu_weights_.layout = layout_for(Q2_K, 4096, 4096);
  down_weights_.layout = down_layout_;
  gu_weights_.data = device.buffer(gu_weights_.bytes(), Memory::Device);
  down_weights_.data = device.buffer(down_weights_.bytes(), Memory::Device);
  packed_.resize(std::max(gu_weights_.bytes(), down_weights_.bytes()));
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
  repack_pair_streamed(gu_weights_.layout, gate, gu_weights_.m() / 2, up, packed_.data());
  stats_.pack_ms += elapsed(); started = Clock::now();
  device_.upload(*gu_weights_.data, packed_.data(), gu_weights_.bytes());
  stats_.upload_ms += elapsed(); started = Clock::now();
  repack_streamed(down_layout_, down, packed_.data());
  stats_.pack_ms += elapsed(); started = Clock::now();
  device_.upload(*down_weights_.data, packed_.data(), down_weights_.bytes());
  stats_.upload_ms += elapsed(); started = Clock::now();
  const char *round_kernel = portable_rounding_ ? "q8k_codes_portable" : "q8k_codes";
  struct {uint32_t mode, n, uofs, tokens; float limit;} swiglu{0, 2048, 0, count, limit};
  device_.run([&](Recorder &r) {
    auto product = [&](const Matrix &w, const Buffer &codes, const Buffer &metadata, const Buffer &out) {
      const char *kernel = w.layout.type == Q2_K ? "expert_gemm_q2k_canon" : "expert_gemm_q3k_canon";
      r.dispatch(kernel, {View(*w.data), View(codes), View(metadata), View(*partial_)},
                 std::array<uint32_t, 5>{w.k(), w.m(), w.layout.mp, count, uint32_t(w.layout.chunk_bytes())},
                 (w.m() + 63) / 64, (count + 63) / 64, 8);
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
    r.dispatch("expert_mix_canon", {View(*gu_), View(*hidden_), View(*hits_)}, swiglu, count * 2048 / 256);
    r.dispatch(round_kernel, {View(*hidden_), View(*qhidden_), View(*hits_), View(*hidden_meta_)},
               std::array<uint32_t, 4>{count * 2048, 0, 0, count}, count * 2048 / 256);
    product(down_weights_, *qhidden_, *hidden_meta_, *output_);
    r.host_visible();
  });
  stats_.queued_work_ms += elapsed();
  if (trace) {
    auto capture = [&](const Buffer &buffer, std::vector<float> &values, size_t size) {
      values.resize(size); device_.download(buffer, values.data(), size * 4);
    };
    capture(*qinput_, trace->qinput, size_t(count) * 4096);
    capture(*gu_, trace->gate_up, size_t(count) * 4096);
    capture(*hidden_, trace->hidden, size_t(count) * 2048);
    capture(*qhidden_, trace->qhidden, size_t(count) * 2048);
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
