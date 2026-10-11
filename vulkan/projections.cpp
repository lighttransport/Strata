#include "projections.hpp"
#include "task_trace.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ggml-alloc.h>
#include <ggml-vulkan.h>
#include <iostream>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <stdexcept>
namespace strata::vulkan {
using Clock = std::chrono::steady_clock;
#if defined(__AVX2__)
static std::vector<float> vector_activations(std::span<const float> input) {
  std::vector<float> output(input.size());
  const auto sign = _mm256_set1_ps(-0.f), limit = _mm256_set1_ps(127.f);
  for (size_t first = 0; first < input.size(); first += 32) {
    __m256 maximum = _mm256_setzero_ps();
    for (size_t i = first; i < first + 32; i += 8) {
      const auto value =
          _mm256_andnot_ps(sign, _mm256_loadu_ps(input.data() + i));
      if (_mm256_movemask_ps(_mm256_cmp_ps(
              value, _mm256_set1_ps(std::numeric_limits<float>::max()),
              _CMP_LE_OQ)) != 255)
        throw std::invalid_argument("Nonfinite Vulkan activation");
      maximum = _mm256_max_ps(maximum, value);
    }
    float lanes[8], max_value = 0;
    _mm256_storeu_ps(lanes, maximum);
    for (float value : lanes)
      max_value = std::max(max_value, value);
    const float scale = std::min(max_value / 127.f, 65504.f);
    const float stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
    if (!max_value || !stored)
      continue;
    for (size_t i = first; i < first + 32; i += 8) {
      const auto value = _mm256_loadu_ps(input.data() + i);
      const auto absolute = _mm256_min_ps(
          _mm256_div_ps(_mm256_andnot_ps(sign, value), _mm256_set1_ps(scale)),
          limit);
      auto integer = _mm256_cvttps_epi32(absolute);
      const auto fraction =
          _mm256_sub_ps(absolute, _mm256_cvtepi32_ps(integer));
      const auto increment =
          _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(
                               fraction, _mm256_set1_ps(.5f), _CMP_GE_OQ)),
                           _mm256_set1_epi32(1));
      integer = _mm256_add_epi32(integer, increment);
      const auto rounded = _mm256_xor_ps(_mm256_cvtepi32_ps(integer),
                                         _mm256_and_ps(sign, value));
      _mm256_storeu_ps(output.data() + i,
                       _mm256_mul_ps(rounded, _mm256_set1_ps(stored)));
    }
  }
  return output;
}
#endif
std::vector<float> decode_activations(std::span<const float> input) {
  if (input.size() % 32)
    throw std::invalid_argument(
        "Q8_1 activation width must be a multiple of 32");
  const char *vectorized = std::getenv("STRATA_VULKAN_AVX2_ACTIVATIONS");
  if (vectorized && std::string(vectorized) == "1") {
#if defined(__AVX2__)
    return vector_activations(input);
#else
    throw std::invalid_argument(
        "AVX2 activations require a native-host build on an AVX2 CPU");
#endif
  }
  std::vector<float> output(input.size());
  for (size_t first = 0; first < input.size(); first += 32) {
    float maximum = 0;
    for (size_t i = first; i < first + 32; ++i) {
      if (!std::isfinite(input[i]))
        throw std::invalid_argument("Nonfinite Vulkan activation");
      maximum = std::max(maximum, std::abs(input[i]));
    }
    const float scale = std::min(maximum / 127.f, 65504.f);
    const float stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
    for (size_t i = first; i < first + 32; ++i)
      output[i] =
          maximum == 0 || stored == 0
              ? 0
              : std::clamp(std::round(input[i] / scale), -127.f, 127.f) *
                    stored;
  }
  return output;
}
Projections::Projections(int device, size_t budget)
    : budget_(budget), device_(device) {
  // GGML's FP32 batched shader can still convert operands to FP16. Start the
  // experimental runner with the strict path; this does not affect HIP/CUDA.
#ifdef _WIN32
  _putenv_s("GGML_VK_DISABLE_F16", "1");
  _putenv_s("GGML_VK_DISABLE_COOPMAT", "1");
  _putenv_s("GGML_VK_DISABLE_COOPMAT2", "1");
#else
  if (setenv("GGML_VK_DISABLE_F16", "1", 1) ||
      setenv("GGML_VK_DISABLE_COOPMAT", "1", 1) ||
      setenv("GGML_VK_DISABLE_COOPMAT2", "1", 1))
    throw std::runtime_error(
        "Cannot select strict Vulkan projection precision");
#endif
  if (device < 0 || device >= ggml_backend_vk_get_device_count())
    throw std::invalid_argument("Vulkan device index unavailable");
  backend_ = ggml_backend_vk_init(device);
  if (!backend_)
    throw std::runtime_error("Vulkan backend initialization failed");
  const char *async = std::getenv("STRATA_VULKAN_ASYNC_STAGING");
  async_staging_ = async && std::string(async) == "1";
  std::cerr << "VULKAN_PROJECTIONS device=" << device_name()
            << " budget_MiB=" << budget_ / 1048576 << '\n';
}
void Projections::ensure_staging(ggml_backend_buffer_t &buffer, size_t bytes) {
  if (buffer && ggml_backend_buffer_get_size(buffer) >= bytes)
    return;
  auto replacement =
      ggml_backend_buft_alloc_buffer(ggml_backend_vk_host_buffer_type(), bytes);
  if (!replacement)
    throw std::bad_alloc();
  // Products synchronize before reusing or growing pinned host storage.
  const size_t old = buffer ? ggml_backend_buffer_get_size(buffer) : 0;
  const size_t size = ggml_backend_buffer_get_size(replacement);
  if (size > budget_ - (resident_ - old)) {
    ggml_backend_buffer_free(replacement);
    throw std::runtime_error("Vulkan staging allocation exceeds budget");
  }
  if (buffer)
    ggml_backend_buffer_free(buffer);
  buffer = replacement;
  resident_ = resident_ - old + size;
}
Projections::~Projections() {
  // An I/O/allocation exception must not free queued transfer storage.
  if (pending_) {
    try {
      ggml_backend_synchronize(backend_);
    } catch (...) {
      std::_Exit(74);
    }
  }
  for (auto &[name, group] : groups_) {
    ggml_backend_buffer_free(group.buffer);
    ggml_free(group.context);
  }
  if (upload_)
    ggml_backend_buffer_free(upload_);
  if (download_)
    ggml_backend_buffer_free(download_);
  for (auto &[n, w] : weights_) {
    if (w.plan_buffer)
      ggml_backend_buffer_free(w.plan_buffer);
    if (w.plan_context)
      ggml_free(w.plan_context);
    ggml_backend_buffer_free(w.buffer);
    ggml_free(w.context);
  }
  if (backend_)
    ggml_backend_free(backend_);
}
std::string Projections::device_name() const {
  char text[512]{};
  ggml_backend_vk_get_device_description(device_, text, sizeof(text));
  return text;
}
std::vector<float> Projections::mat(const std::string &name,
                                    const strata::core::ArtifactTensor &t,
                                    std::span<const float> x, int heads,
                                    int tokens) {
  if (t.tensor->shape.size() < 2 || t.tensor->shape.size() > 3)
    throw std::invalid_argument("Vulkan matrix rank invalid: " + name);
  if ((heads == 1 && t.tensor->shape.size() != 2) ||
      (heads > 1 &&
       (t.tensor->shape.size() != 3 || t.tensor->shape[2] != uint64_t(heads))))
    throw std::invalid_argument("Vulkan matrix head geometry invalid: " + name);
  return product(name, ggml_type(t.tensor->type), t.data(), t.bytes,
                 int(t.tensor->shape[0]), int(t.tensor->shape[1]), x, tokens,
                 heads);
}
std::vector<std::vector<float>> Projections::joined(
    const std::vector<
        std::pair<std::string, const strata::core::ArtifactTensor *>> &items,
    std::span<const float> input) {
  return multiple(items,
                  std::vector<std::span<const float>>(items.size(), input));
}
std::vector<std::vector<float>> Projections::multiple(
    const std::vector<
        std::pair<std::string, const strata::core::ArtifactTensor *>> &items,
    const std::vector<std::span<const float>> &inputs) {
  if (items.empty() || items.size() > 8 || items.size() != inputs.size())
    throw std::invalid_argument("Vulkan projection group size invalid");
  bool compatible = async_staging_;
  size_t count = 0, upload_bytes = 0;
  std::string key = "group";
  std::vector<size_t> index;
  std::vector<std::span<const float>> unique;
  std::vector<bool> quantized;
  for (size_t i = 0; i < items.size(); ++i) {
    const auto &[name, tensor] = items[i];
    const auto x = inputs[i];
    if (!tensor || !tensor->tensor || tensor->tensor->shape.size() != 2 ||
        x.empty() || tensor->tensor->shape[0] != x.size() ||
        !tensor->tensor->shape[1])
      throw std::invalid_argument("Vulkan grouped matrix geometry invalid: " +
                                  name);
    const auto type = ggml_type(tensor->tensor->type);
    const bool q = ggml_is_quantized(type);
    if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0)
      compatible = false;
    size_t slot = 0;
    for (; slot < unique.size(); ++slot)
      if (unique[slot].data() == x.data() && unique[slot].size() == x.size() &&
          quantized[slot] == q)
        break;
    if (slot == unique.size()) {
      unique.push_back(x);
      quantized.push_back(q);
      upload_bytes += x.size_bytes();
    }
    index.push_back(slot);
    count += tensor->tensor->shape[1];
    key += "/" + std::to_string(name.size()) + ":" + name + ":" +
           std::to_string(slot);
    if (auto found = weights_.find(name); found != weights_.end()) {
      const auto &w = found->second;
      if (w.tensor->type != type || size_t(w.tensor->ne[0]) != x.size() ||
          uint64_t(w.tensor->ne[1]) != tensor->tensor->shape[1] ||
          w.tensor->ne[2] != 1 || w.bytes != tensor->bytes)
        throw std::invalid_argument(
            "Vulkan grouped cached weight geometry changed: " + name);
    }
  }
  std::vector<std::vector<float>> results;
  if (!compatible) {
    for (size_t i = 0; i < items.size(); ++i)
      results.push_back(mat(items[i].first, *items[i].second, inputs[i]));
    return results;
  }
  auto found = groups_.find(key);
  if (found == groups_.end()) {
    // Cold initialization validates weights through the ordinary product path.
    for (size_t i = 0; i < items.size(); ++i)
      if (!weights_.contains(items[i].first))
        mat(items[i].first, *items[i].second, inputs[i]);
    Group g;
    g.context = ggml_init({ggml_tensor_overhead() * (2 * items.size() + 4) +
                               ggml_graph_overhead_custom(64, false) + 4096,
                           nullptr, true});
    if (!g.context)
      throw std::bad_alloc();
    try {
      for (auto input : unique)
        g.inputs.push_back(
            ggml_new_tensor_2d(g.context, GGML_TYPE_F32, input.size(), 1));
      g.graph = ggml_new_graph_custom(g.context, 64, false);
      for (size_t i = 0; i < items.size(); ++i) {
        auto *out = ggml_mul_mat(g.context, weights_.at(items[i].first).tensor,
                                 g.inputs[index[i]]);
        if (!ggml_prec_set_acc(out, GGML_PREC_F32) ||
            !ggml_backend_supports_op(backend_, out))
          throw std::runtime_error("Vulkan grouped projection unsupported: " +
                                   items[i].first);
        g.outputs.push_back(out);
        ggml_build_forward_expand(g.graph, out);
      }
      g.buffer = ggml_backend_alloc_ctx_tensors(g.context, backend_);
      if (!g.buffer)
        throw std::bad_alloc();
      const auto bytes = ggml_backend_buffer_get_size(g.buffer);
      if (bytes > budget_ - resident_)
        throw std::runtime_error("Vulkan projection group exceeds budget");
      found = groups_.emplace(key, std::move(g)).first;
      resident_ += bytes;
    } catch (...) {
      if (g.buffer)
        ggml_backend_buffer_free(g.buffer);
      ggml_free(g.context);
      throw;
    }
  }
  auto &g = found->second;
  if (g.inputs.size() != unique.size() || g.outputs.size() != items.size())
    throw std::invalid_argument("Vulkan cached group geometry changed");
  auto started = Clock::now();
  std::vector<std::vector<float>> encoded(unique.size());
  for (size_t i = 0; i < unique.size(); ++i) {
    if (size_t(g.inputs[i]->ne[0]) != unique[i].size())
      throw std::invalid_argument("Vulkan cached input width changed");
    if (quantized[i])
      encoded[i] = decode_activations(unique[i]);
  }
  ensure_staging(upload_, upload_bytes);
  ensure_staging(download_, count * sizeof(float));
  auto *upload = static_cast<uint8_t *>(ggml_backend_buffer_get_base(upload_));
  size_t offset = 0;
  for (size_t i = 0; i < unique.size(); ++i) {
    auto x = unique[i];
    if (size_t(g.inputs[i]->ne[0]) != x.size())
      throw std::invalid_argument("Vulkan cached input width changed");
    const auto &activation = encoded[i];
    std::memcpy(upload + offset, quantized[i] ? activation.data() : x.data(),
                x.size_bytes());
    pending_ = true;
    ggml_backend_tensor_set_async(backend_, g.inputs[i], upload + offset, 0,
                                  x.size_bytes());
    offset += x.size_bytes();
  }
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  started = Clock::now();
  operation_trace("projection_begin", key);
  auto status = ggml_backend_graph_compute_async(backend_, g.graph);
  auto *download =
      static_cast<float *>(ggml_backend_buffer_get_base(download_));
  offset = 0;
  if (status == GGML_STATUS_SUCCESS)
    for (auto *out : g.outputs) {
      ggml_backend_tensor_get_async(backend_, out, download + offset, 0,
                                    ggml_nbytes(out));
      offset += ggml_nelements(out);
    }
  ggml_backend_synchronize(backend_);
  pending_ = false;
  compute_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  if (status != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Vulkan projection group execution failed");
  operation_trace("projection_complete", key);
  started = Clock::now();
  offset = 0;
  for (auto *out : g.outputs) {
    const size_t elements = ggml_nelements(out);
    results.emplace_back(download + offset, download + offset + elements);
    for (float value : results.back())
      if (!std::isfinite(value))
        throw std::runtime_error(
            "Vulkan grouped projection returned nonfinite values");
    offset += elements;
  }
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  calls += items.size();
  ++submissions;
  return results;
}
std::vector<float>
Projections::synthetic(const std::string &key, ggml_type type,
                       std::span<const uint8_t> bytes, int in, int out,
                       std::span<const float> x, int tokens, int heads) {
  return product(key, type, bytes.data(), bytes.size(), in, out, x, tokens,
                 heads);
}
std::vector<float> Projections::product(const std::string &key, ggml_type type,
                                        const void *source, size_t bytes,
                                        int in, int out,
                                        std::span<const float> x, int tokens,
                                        int heads) {
  if (in <= 0 || out <= 0 || heads <= 0 || tokens <= 0 || tokens > 2048 ||
      x.size() != size_t(in) * tokens * heads)
    throw std::invalid_argument("Vulkan projection geometry invalid: " + key);
  auto found = weights_.find(key);
  if (found == weights_.end()) {
    if (bytes > budget_ - resident_)
      throw std::runtime_error("Vulkan resident projection budget exceeded: " +
                               key);
    Weight w;
    w.context = ggml_init({ggml_tensor_overhead() * 2 + 4096, nullptr, true});
    if (!w.context)
      throw std::bad_alloc();
    try {
      w.tensor = ggml_new_tensor_3d(w.context, type, in, out, heads);
      ggml_set_name(w.tensor, key.c_str());
      if (ggml_nbytes(w.tensor) != bytes)
        throw std::invalid_argument("Vulkan weight byte count differs: " + key);
      w.buffer = ggml_backend_alloc_ctx_tensors(w.context, backend_);
      if (!w.buffer)
        throw std::bad_alloc();
      w.allocated = ggml_backend_buffer_get_size(w.buffer);
      if (w.allocated > budget_ - resident_)
        throw std::runtime_error(
            "Vulkan padded weight allocation exceeds budget: " + key);
      auto start = Clock::now();
      ggml_backend_tensor_set(w.tensor, source, 0, bytes);
      transfer_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      w.bytes = bytes;
      if (heads == 1 && (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0)) {
        const size_t blocks = size_t(in) / 32;
        const size_t block_bytes = ggml_type_size(type);
        const float offset = type == GGML_TYPE_Q4_0 ? -8.f : -16.f;
        w.affine_bias.resize(size_t(out) * blocks);
        for (size_t i = 0; i < w.affine_bias.size(); ++i) {
          ggml_fp16_t scale;
          std::memcpy(&scale,
                      static_cast<const uint8_t *>(source) + i * block_bytes,
                      sizeof(scale));
          w.affine_bias[i] = offset * ggml_fp16_to_fp32(scale);
        }
      }
      found = weights_.emplace(key, w).first;
      resident_ += w.allocated;
    } catch (...) {
      if (w.buffer)
        ggml_backend_buffer_free(w.buffer);
      ggml_free(w.context);
      throw;
    }
  }
  auto &w = found->second;
  if (w.tensor->type != type || w.tensor->ne[0] != in ||
      w.tensor->ne[1] != out || w.tensor->ne[2] != heads || w.bytes != bytes)
    throw std::invalid_argument("Vulkan cached projection geometry changed: " +
                                key);
  if (w.tokens != tokens) {
    if (w.plan_buffer)
      ggml_backend_buffer_free(w.plan_buffer);
    if (w.plan_context)
      ggml_free(w.plan_context);
    resident_ -= w.plan_bytes;
    w.plan_buffer = nullptr;
    w.plan_context = nullptr;
    w.plan_bytes = 0;
    w.tokens = 0;
    ggml_context *ctx =
        ggml_init({ggml_tensor_overhead() * 8 +
                       ggml_graph_overhead_custom(32, false) + 4096,
                   nullptr, true});
    if (!ctx)
      throw std::bad_alloc();
    ggml_backend_buffer_t buffer = nullptr;
    try {
      auto *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, tokens, heads);
      auto *output = ggml_mul_mat(ctx, w.tensor, input);
      if (!ggml_prec_set_acc(output, GGML_PREC_F32))
        throw std::runtime_error(
            "Vulkan projection cannot request FP32 accumulation");
      if (!ggml_backend_supports_op(backend_, output))
        throw std::runtime_error("Vulkan projection operation unsupported: " +
                                 key);
      auto *graph = ggml_new_graph_custom(ctx, 32, false);
      ggml_build_forward_expand(graph, output);
      buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_);
      if (!buffer)
        throw std::bad_alloc();
      const auto plan_bytes = ggml_backend_buffer_get_size(buffer);
      if (plan_bytes > budget_ - resident_)
        throw std::runtime_error(
            "Vulkan projection workspace exceeds budget: " + key);
      w.plan_context = ctx;
      w.plan_buffer = buffer;
      w.input = input;
      w.output = output;
      w.graph = graph;
      w.tokens = tokens;
      w.plan_bytes = plan_bytes;
      resident_ += plan_bytes;
    } catch (...) {
      if (buffer)
        ggml_backend_buffer_free(buffer);
      ggml_free(ctx);
      throw;
    }
  }
  auto start = Clock::now();
  auto quantized = ggml_is_quantized(type) && heads == 1 ? decode_activations(x)
                                                         : std::vector<float>{};
  const auto *activation = quantized.empty() ? x.data() : quantized.data();
  std::vector<float> y(size_t(out) * tokens * heads);
  if (async_staging_) {
    ensure_staging(upload_, x.size_bytes());
    ensure_staging(download_, y.size() * sizeof(float));
    std::memcpy(ggml_backend_buffer_get_base(upload_), activation,
                x.size_bytes());
    pending_ = true;
    ggml_backend_tensor_set_async(backend_, w.input,
                                  ggml_backend_buffer_get_base(upload_), 0,
                                  x.size_bytes());
  } else {
    ggml_backend_tensor_set(w.input, activation, 0, x.size_bytes());
  }
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  start = Clock::now();
  operation_trace("projection_begin", key);
  auto status = async_staging_
                    ? ggml_backend_graph_compute_async(backend_, w.graph)
                    : ggml_backend_graph_compute(backend_, w.graph);
  if (async_staging_) {
    if (status == GGML_STATUS_SUCCESS)
      ggml_backend_tensor_get_async(backend_, w.output,
                                    ggml_backend_buffer_get_base(download_), 0,
                                    y.size() * sizeof(float));
    ggml_backend_synchronize(backend_);
    pending_ = false;
  }
  compute_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (status != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Vulkan projection execution failed: " + key);
  operation_trace("projection_complete", key);
  start = Clock::now();
  if (async_staging_)
    std::memcpy(y.data(), ggml_backend_buffer_get_base(download_),
                y.size() * sizeof(float));
  else
    ggml_backend_tensor_get(w.output, y.data(), 0, y.size() * 4);
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (!w.affine_bias.empty()) {
    // Native Q4_0/Q5_0 dots use the original-input sum stored in Q8_1, not
    // the sum of reconstructed quantized activations. Keep that correction.
    start = Clock::now();
    const int blocks = in / 32;
    std::vector<float> delta(size_t(tokens) * blocks);
    for (int t = 0; t < tokens; ++t)
      for (int b = 0; b < blocks; ++b) {
        std::array<float, 32> sums{};
        float reconstructed = 0;
        const size_t first = size_t(t) * in + b * 32;
        for (int i = 0; i < 32; ++i) {
          sums[i] = x[first + i];
          reconstructed += quantized[first + i];
        }
        // Lane zero's XOR tree needs only these 31 additions; the other
        // lane results were duplicates. Keep the exact operand grouping.
        for (int offset = 16; offset; offset /= 2)
          for (int i = 0; i < offset; ++i)
            sums[i] += sums[i + offset];
        const float original = ggml_fp16_to_fp32(
            ggml_fp32_to_fp16(std::clamp(sums[0], -65504.f, 65504.f)));
        delta[size_t(t) * blocks + b] = original - reconstructed;
      }
    for (int t = 0; t < tokens; ++t)
      for (int r = 0; r < out; ++r) {
        float correction = 0;
        for (int b = 0; b < blocks; ++b)
          correction += w.affine_bias[size_t(r) * blocks + b] *
                        delta[size_t(t) * blocks + b];
        y[size_t(t) * out + r] += correction;
      }
    affine_cpu_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  for (float v : y)
    if (!std::isfinite(v))
      throw std::runtime_error("Vulkan projection returned nonfinite values: " +
                               key);
  ++calls;
  ++submissions;
  return y;
}
} // namespace strata::vulkan
