// Whole streamed experts, including both Q8_K grids and SwiGLU, against the
// canonical CPU implementation. Software Vulkan uses an explicit subgroup-
// independent reduction variant; RX570 keeps the existing wave64 variant.
#include "fast/expert_batch.hpp"
#include "model_checks.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/q23_avx2.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <ggml.h>
#include <random>
namespace fast = strata::vulkan::fast;
namespace cpu = strata::kernels::cpu;
static void decode_q8k(const uint8_t *blocks, float *out, uint32_t n) {
  const size_t stride = ggml_row_size(GGML_TYPE_Q8_K, 256);
  for (uint32_t b = 0; b < n / 256; ++b) {
    float d; std::memcpy(&d, blocks + b * stride, 4);
    const auto *codes = reinterpret_cast<const int8_t *>(blocks + b * stride + 4);
    for (uint32_t i = 0; i < 256; ++i) out[b * 256 + i] = d * float(codes[i]);
  }
}
static std::array<float, 8> scalar_q2_lanes(const uint8_t *w, const uint8_t *a, uint32_t n,
                                           std::vector<float> *block_trace = nullptr) {
  std::array<float, 8> acc{};
  if (block_trace) block_trace->resize(size_t(8) * (n / 256) * 8);
  const size_t stride = ggml_row_size(GGML_TYPE_Q8_K, 256);
  for (uint32_t b = 0; b < n / 256; ++b) {
    const auto *raw = w + b * 84, *act = a + b * stride;
    ggml_fp16_t dh[2]; std::memcpy(dh, raw + 80, 4);
    float yd; std::memcpy(&yd, act, 4);
    const auto *codes = reinterpret_cast<const int8_t *>(act + 4);
    for (uint32_t j = 0; j < 8; ++j) {
      int dot = 0, correction = 0;
      for (uint32_t h = 0; h < 8; ++h) for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t k = 32 * h + 4 * j + i;
        const int q = (raw[16 + k / 128 * 32 + k % 32] >> (2 * ((k % 128) / 32))) & 3;
        dot += int(raw[k / 16] & 15) * q * codes[k];
      }
      for (uint32_t s = 2 * j; s < 2 * j + 2; ++s) {
        int sum = 0; for (uint32_t i = 0; i < 16; ++i) sum += codes[16 * s + i];
        correction += int(raw[s] >> 4) * sum;
      }
      const float dm = -ggml_fp16_to_fp32(dh[1]) * yd, d = ggml_fp16_to_fp32(dh[0]) * yd;
      const size_t at = (size_t(j) * (n / 256) + b) * 8;
      if (block_trace) {
        (*block_trace)[at] = dm; (*block_trace)[at + 1] = float(correction); (*block_trace)[at + 2] = acc[j];
        (*block_trace)[at + 4] = d; (*block_trace)[at + 5] = float(dot);
        (*block_trace)[at + 7] = ggml_fp16_to_fp32(dh[0]);
      }
      acc[j] = std::fma(dm, float(correction), acc[j]);
      if (block_trace) (*block_trace)[at + 3] = acc[j];
      acc[j] = std::fma(d, float(dot), acc[j]);
      if (block_trace) (*block_trace)[at + 6] = acc[j];
    }
  }
  return acc;
}
static float scalar_q2(const uint8_t *w, const uint8_t *a, uint32_t n) {
  const auto acc = scalar_q2_lanes(w, a, n);
  return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
}
int main(int argc, char **argv) {
  try {
    bool software = false, gpu_pack = false, vector_weights = false, boundaries = false, benchmark = false, replay = false;
    uint32_t tile_columns = 64;
    uint32_t tile_rows = 64;
    uint32_t reserve_capacity = 129;
    const char *usage = "usage: glm_vulkan_fast_expert_batch_check MODEL [--cpu] [--gpu-pack] [--vector-weights] [--tile-columns 16|32|64] [--tile-rows 32|64] [--capacity 129..2048] [--boundary-checks] [--benchmark] [--replay]";
    if (argc < 2) throw std::invalid_argument(usage);
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "--cpu" && !software) software = true;
      else if (std::string(argv[i]) == "--gpu-pack" && !gpu_pack) gpu_pack = true;
      else if (std::string(argv[i]) == "--vector-weights" && !vector_weights) vector_weights = true;
      else if (std::string(argv[i]) == "--boundary-checks" && !boundaries) boundaries = true;
      else if (std::string(argv[i]) == "--benchmark" && !benchmark) benchmark = true;
      else if (std::string(argv[i]) == "--replay" && !replay) replay = true;
      else if (std::string(argv[i]) == "--tile-columns" && i + 1 < argc) {
        const std::string value = argv[++i];
        if (value != "16" && value != "32" && value != "64") throw std::invalid_argument(usage);
        tile_columns = uint32_t(std::stoul(value));
      }
      else if (std::string(argv[i]) == "--capacity" && i + 1 < argc) {
        const std::string value = argv[++i]; size_t used = 0;
        const auto count = std::stoul(value, &used);
        if (used != value.size() || count < 129 || count > 2048) throw std::invalid_argument(usage);
        reserve_capacity = uint32_t(count);
      }
      else if (std::string(argv[i]) == "--tile-rows" && i + 1 < argc) {
        const std::string value = argv[++i];
        if (value != "32" && value != "64") throw std::invalid_argument(usage);
        tile_rows = uint32_t(std::stoul(value));
      }
      else throw std::invalid_argument(usage);
    }
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &model = strata::vulkan::checked_model(artifact);
    fast::Device dev(software ? "llvmpipe" : "", 1ULL << 30, software);
    if (dev.fp64_expert_scale()) {
      std::mt19937_64 rng(65271);
      std::vector<uint64_t> patterns;
      for (uint32_t i = 0; i < 8192; ++i) {
        const uint32_t exponent = i % 4 == 0 ? uint32_t(rng() % 2048) : i % 4 == 1 ? 873 + i % 25 :
                                  i % 4 == 2 ? 897 + i % 254 : 1149 + i % 4;
        patterns.push_back((rng() & 0x800fffffffffffffULL) | (uint64_t(exponent) << 52));
      }
      for (uint32_t i = 0; i < 512; ++i) {
        uint32_t bits = uint32_t(rng() % 0x7f7ffffeu), next_bits = bits + 1;
        float a, b;
        std::memcpy(&a, &bits, 4); std::memcpy(&b, &next_bits, 4);
        const double middle = (double(a) + double(b)) * 0.5;
        uint64_t raw;
        std::memcpy(&raw, &middle, 8);
        const uint64_t sign = (i & 1) ? 0x8000000000000000ULL : 0;
        patterns.push_back(sign | (raw - 1)); patterns.push_back(sign | raw); patterns.push_back(sign | (raw + 1));
      }
      for (uint64_t bits : {0ULL, 0x8000000000000000ULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL,
                            0x7ff8000000000001ULL, 1ULL, 0x3690000000000000ULL, 0x36a0000000000000ULL})
        patterns.push_back(bits);
      auto input = dev.buffer(patterns.size() * 8, fast::Memory::Host);
      auto output = dev.buffer(patterns.size() * 4, fast::Memory::Host);
      for (size_t i = 0; i < patterns.size(); ++i) {
        input->as<uint32_t>()[2 * i] = uint32_t(patterns[i]);
        input->as<uint32_t>()[2 * i + 1] = uint32_t(patterns[i] >> 32);
      }
      dev.run([&](fast::Recorder &r) {
        r.dispatch("round_fp64_oracle", {fast::View(*input), fast::View(*output)},
                   std::array<uint32_t, 1>{uint32_t(patterns.size())}, uint32_t((patterns.size() + 255) / 256));
        r.host_visible();
      });
      uint32_t mismatches = 0;
      for (size_t i = 0; i < patterns.size(); ++i) {
        double x;
        std::memcpy(&x, &patterns[i], 8);
        volatile double source = x;
        const float ref = float(source);
        uint32_t bits;
        std::memcpy(&bits, &ref, 4);
        float got;
        std::memcpy(&got, output->as<uint32_t>() + i, 4);
        mismatches += std::isnan(ref) ? !std::isnan(got) : bits != output->as<uint32_t>()[i];
      }
      std::cout << "{\"fp64_to_f32_cases\":" << patterns.size() << ",\"mismatches\":" << mismatches
                << ",\"nan_comparison\":\"classification\",\"finite_comparison\":\"all_bits\"}\n";
      if (mismatches) throw std::runtime_error("explicit FP64toFP32 rounding differs from CPU conversion");
    }
    {
      const uint32_t count = dev.fp64_expert_scale() ? 1792u : 512u;
      auto input = dev.buffer(count * 16, fast::Memory::Host);
      auto output = dev.buffer(count * 16, fast::Memory::Host);
      std::mt19937 probe(733); std::uniform_real_distribution<float> fraction(0.5f, 2.f);
      for (uint32_t i = 0; i < count; ++i) {
        const float a = fraction(probe), b = fraction(probe);
        input->as<float>()[4 * i] = a; input->as<float>()[4 * i + 1] = b;
        input->as<float>()[4 * i + 2] = i < 512 ? -float(a * b) : i < 1024 ? fraction(probe) - 1.25f : 0.f;
        if (i >= 1536) {
          const int index = int(i - 1536), q = index % 17 - 8, exponent = index % 81 - 40;
          const float sign = index & 1 ? -1.f : 1.f;
          input->as<float>()[4 * i] = sign * std::ldexp(0x1.000002p0f, q);
          input->as<float>()[4 * i + 1] = std::ldexp(0x1.fffffcp-1f, exponent - q - 24);
          input->as<float>()[4 * i + 2] = sign * std::ldexp(0x1.000002p0f, exponent);
        }
      }
      dev.run([&](fast::Recorder &r) {
        const std::string oracle = dev.fp64_expert_scale() ?
            std::string("fma_oracle_fp64_scale") + (dev.native_expert_fma() ? "_native_fma" : "") : "fma_oracle";
        r.dispatch(oracle, {fast::View(*input), fast::View(*output)},
                   std::array<uint32_t, 1>{count}, count / 256);
        r.host_visible();
      });
      uint32_t ordinary = 0, precise = 0, separate = 0, wide = 0, wide_cancel = 0, wide_general = 0, multiply = 0;
      uint32_t double_rounding_counterexamples = 0;
      for (uint32_t i = 0; i < count; ++i) {
        const float *v = input->as<float>() + 4 * i;
        const float ref = std::fma(v[0], v[1], v[2]);
        ordinary += output->as<float>()[4 * i] != ref;
        precise += output->as<float>()[4 * i + 1] != ref;
        separate += output->as<float>()[4 * i + 2] != ref;
        if (dev.fp64_expert_scale()) {
          const bool mismatch = output->as<float>()[4 * i + 3] != ref;
          wide += mismatch;
          if (i < 512) wide_cancel += mismatch;
          else wide_general += mismatch;
          if (i >= 1024 && i < 1536) {
            volatile float product = v[0] * v[1];
            multiply += output->as<float>()[4 * i + 2] != product;
          }
          if (i >= 1536) {
            volatile double product = double(v[0]) * double(v[1]);
            volatile double sum = product + double(v[2]);
            double_rounding_counterexamples += float(sum) != ref;
          }
        }
      }
      std::cout << "{\"fma_probe_cases\":" << count << ",\"ordinary_mismatches\":" << ordinary
                << ",\"precise_mismatches\":" << precise << ",\"separate_mismatches\":" << separate;
      if (dev.fp64_expert_scale()) std::cout << ",\"fp64_scale_mismatches\":" << wide
          << ",\"fp64_cancel_mismatches\":" << wide_cancel << ",\"fp64_general_mismatches\":" << wide_general
          << ",\"fp32_multiply_mismatches\":" << multiply
          << ",\"cpu_double_rounding_counterexamples\":" << double_rounding_counterexamples;
      if (dev.native_expert_fma()) std::cout << ",\"fma_implementation\":\"VK_KHR_shader_fma\"";
      std::cout << "}\n";
      if (dev.fp64_expert_scale() && wide) throw std::runtime_error("FP64 scale diagnostic failed FMA cancellation screen");
      if (dev.fp64_expert_scale() && double_rounding_counterexamples != 256)
        throw std::runtime_error("CPU double-rounding negative controls failed");
    }
    const auto &G = artifact.at("blk.3.ffn_gate_exps.weight"), &U = artifact.at("blk.3.ffn_up_exps.weight"),
               &D = artifact.at("blk.3.ffn_down_exps.weight");
    if (G.tensor->type != fast::Q2_K || U.tensor->type != fast::Q2_K || D.tensor->type != fast::Q3_K)
      throw std::invalid_argument("streamed expert check requires Q2_K gate/up and Q3_K down");
    cpu::NativeFmt fmt; std::string error;
    if (!cpu::native_fmt(G.tensor->type, D.tensor->type, 4096, 2048, fmt, error)) throw std::runtime_error(error);
    fmt.canon = true; fmt.swiglu_limit = model.layers[3].swiglu_limit;
    fast::ExpertBatch batch(dev, reserve_capacity, software, gpu_pack, vector_weights, tile_columns, replay, tile_rows);
    std::unique_ptr<fast::ExpertBatch> original;
    if (vector_weights || tile_columns != 64 || tile_rows != 64 || benchmark || reserve_capacity != 129 || replay)
      original = std::make_unique<fast::ExpertBatch>(dev, 129, software, gpu_pack, false);
    std::unique_ptr<fast::ExpertBatch> benchmark_control;
    if (benchmark && vector_weights)
      benchmark_control = std::make_unique<fast::ExpertBatch>(dev, 129, software, gpu_pack, true);
    const std::vector<uint32_t> columns = boundaries ? std::vector<uint32_t>{1, 5, 17, 33, 65, 129}
                                                     : std::vector<uint32_t>{5, 33, 129};
    std::mt19937 rng(7741); std::normal_distribution<float> normal(0.f, 0.5f);
    for (int expert : {5, 77}) {
      const auto *gate = G.data() + size_t(expert) * G.bytes / 144;
      const auto *up = U.data() + size_t(expert) * U.bytes / 144;
      const auto *down = D.data() + size_t(expert) * D.bytes / 144;
      for (uint32_t T : columns) {
        std::vector<float> input(T * 4096), expected(T * 4096), actual(T * 4096), hidden(2048);
        std::vector<float> qinput(T * 4096), expected_hidden(T * 2048), qhidden(T * 2048);
        std::vector<float> gate_up(T * 4096);
        std::vector<const float *> inputs;
        std::vector<float *> outputs;
        for (auto &v : input) v = normal(rng);
        std::vector<uint8_t> act(fmt.act_bytes), hq(fmt.h_bytes);
        std::vector<uint8_t> quantized(size_t(T) * fmt.act_bytes);
        std::vector<const uint8_t *> qinputs;
        for (uint32_t t = 0; t < T; ++t) {
          const float *x = input.data() + size_t(t) * 4096;
          inputs.push_back(x); outputs.push_back(actual.data() + size_t(t) * 4096);
          cpu::native_quant_act(fmt, x, act.data());
          std::memcpy(quantized.data() + size_t(t) * fmt.act_bytes, act.data(), fmt.act_bytes);
          qinputs.push_back(quantized.data() + size_t(t) * fmt.act_bytes);
          decode_q8k(act.data(), qinput.data() + size_t(t) * 4096, 4096);
          const void *ap[] = {act.data()}; float *hp[] = {hidden.data()};
          float *gp[] = {gate_up.data() + size_t(t) * 4096};
          float *uptr[] = {gate_up.data() + size_t(t) * 4096 + 2048};
          cpu::q23_gu_rows(G.tensor->type, gate, nullptr, fmt.gu_row, 4096, ap, 1, gp, 0, 2048, 0, true, 0);
          cpu::q23_gu_rows(G.tensor->type, up, nullptr, fmt.gu_row, 4096, ap, 1, uptr, 0, 2048, 0, true, 0);
          cpu::native_gu_rows(fmt, gate, ap, 1, hp, 0, 2048, up);
          cpu::native_quant_h(fmt, hidden.data(), hq.data());
          std::copy(hidden.begin(), hidden.end(), expected_hidden.begin() + size_t(t) * 2048);
          decode_q8k(hq.data(), qhidden.data() + size_t(t) * 2048, 2048);
          const void *qp[] = {hq.data()}; float *op[] = {expected.data() + size_t(t) * 4096};
          cpu::native_down_rows(fmt, gate, qp, 1, op, 0, 4096, down);
        }
        fast::ExpertBatchTrace trace;
        batch.run(gate, up, down, inputs, outputs, fmt.swiglu_limit, &trace);
        const auto raw_result = actual;
        fast::ExpertBatchTrace packed_trace;
        batch.run_quantized(gate, up, down, qinputs, outputs, fmt.swiglu_limit, &packed_trace);
        if (original) {
          std::vector<float> scalar(actual.size());
          std::vector<float *> scalar_outputs;
          for (uint32_t t = 0; t < T; ++t) scalar_outputs.push_back(scalar.data() + size_t(t) * 4096);
          original->run_quantized(gate, up, down, qinputs, scalar_outputs, fmt.swiglu_limit);
          if (std::memcmp(scalar.data(), actual.data(), actual.size() * 4))
            throw std::runtime_error("expert column tile or vector loads changed whole-expert output bytes");
          std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                    << ",\"column_tile\":" << tile_columns
                    << ",\"row_tile\":" << tile_rows
                    << ",\"tile64_scalar_byte_identical\":true,\"vector_scalar_byte_identical\":true}\n";
        }
        const bool matches_raw = !std::memcmp(raw_result.data(), actual.data(), actual.size() * 4) &&
            !std::memcmp(trace.qinput.data(), packed_trace.qinput.data(), trace.qinput.size() * 4);
        if (std::memcmp(qinput.data(), packed_trace.qinput.data(), qinput.size() * 4))
          throw std::runtime_error("packed Q8_K input differs from CPU quantization");
        if (software && !matches_raw)
          throw std::runtime_error("software packed input differs from software GPU quantization");
        std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                  << ",\"packed_input_output_byte_identical\":" << (matches_raw ? "true" : "false")
                  << ",\"packed_input_cpu_grid_byte_identical\":true,\"fp32_input_bytes\":" << size_t(T) * 4096 * 4
                  << ",\"q8k_input_bytes\":" << size_t(T) * fmt.act_bytes << "}\n";
        auto diagnostic = [&](const char *name, const auto &ref, const auto &got) {
          double e = 0, n = 0; size_t changed = 0;
          for (size_t i = 0; i < ref.size(); ++i) {
            e += (double(ref[i]) - got[i]) * (double(ref[i]) - got[i]); n += double(ref[i]) * ref[i];
            changed += ref[i] != got[i];
          }
          std::cout << "{\"stage\":\"" << name << "\",\"rel_l2\":" << std::sqrt(e / std::max(n, 1e-30))
                    << ",\"changed\":" << changed
                    << ",\"byte_identical\":" << (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) ? "false" : "true")
                    << "}\n";
        };
        diagnostic("qinput", qinput, packed_trace.qinput);
        diagnostic("gate_up", gate_up, packed_trace.gate_up);
        diagnostic("hidden", expected_hidden, packed_trace.hidden);
        diagnostic("qhidden", qhidden, packed_trace.qhidden);
        cpu::native_quant_act(fmt, input.data(), act.data());
        if (!packed_trace.gate_lane_prefix.empty()) {
          uint32_t changed_lanes = 0, changed_reductions = 0;
          for (uint32_t row = 0; row < 16; ++row) {
            const auto ref = scalar_q2_lanes(gate + row * fmt.gu_row, act.data(), 4096);
            std::array<float, 8> lanes{};
            for (uint32_t lane = 0; lane < 8; ++lane) {
              lanes[lane] = packed_trace.gate_lane_prefix[lane * 16 + row];
              changed_lanes += lanes[lane] != ref[lane];
            }
            const float reduced = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) +
                                  ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
            changed_reductions += reduced != packed_trace.gate_up[row];
          }
          std::cout << "{\"gate_lane_prefix_values\":128,\"cpu_lane_mismatches\":" << changed_lanes
                    << ",\"gpu_reduction_mismatches\":" << changed_reductions << "}\n";
          std::vector<float> block_ref;
          scalar_q2_lanes(gate, act.data(), 4096, &block_ref);
          const auto &got = packed_trace.gate_block_scale;
          if (got.size() != block_ref.size()) throw std::runtime_error("GPU block-scale trace size differs");
          std::array<uint32_t, 8> differences{};
          uint32_t min_fma = 0, scale_fma = 0;
          for (size_t at = 0; at < got.size(); at += 8) {
            for (size_t field = 0; field < 8; ++field) differences[field] += got[at + field] != block_ref[at + field];
            min_fma += got[at + 3] != std::fma(got[at], got[at + 1], got[at + 2]);
            scale_fma += got[at + 6] != std::fma(got[at + 4], got[at + 5], got[at + 3]);
          }
          std::cout << "{\"gate_block_scale_values\":" << got.size() << ",\"field_mismatches\":[";
          for (size_t field = 0; field < 8; ++field) std::cout << (field ? "," : "") << differences[field];
          std::cout << "],\"gpu_operands_min_fma_mismatches\":" << min_fma
                    << ",\"gpu_operands_scale_fma_mismatches\":" << scale_fma << "}\n";
        }
        for (uint32_t row = 0; row < 4; ++row) {
          const float scalar = scalar_q2(gate + row * fmt.gu_row, act.data(), 4096);
          if (scalar != gate_up[row]) throw std::runtime_error("CPU scalar canonical dot differs from CPU oracle");
          std::cout << "{\"row\":" << row << ",\"scalar_q2\":\"" << std::hexfloat << scalar
                    << "\",\"cpu\":\"" << gate_up[row] << "\",\"gpu\":\"" << packed_trace.gate_up[row] << std::defaultfloat << "\"}\n";
        }
        double err = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
          if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite streamed expert output");
          err += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
          norm += double(expected[i]) * expected[i];
        }
        const double rel = std::sqrt(err / std::max(norm, 1e-30));
        const bool cpu_bytes = !std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float));
        std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                  << ",\"software\":" << software << ",\"rel_l2_cpu_canonical\":" << rel
                  << ",\"cpu_output_byte_identical\":" << (cpu_bytes ? "true" : "false") << "}\n" << std::flush;
        if (!(rel < 2e-4)) throw std::runtime_error("streamed expert canonical parity failed");
        if (dev.fp64_expert_math() && !cpu_bytes)
          throw std::runtime_error("FP64 expert math differs from canonical CPU output bytes");
        if (replay) {
          const auto fixed = actual;
          std::vector<float> control(actual.size());
          std::vector<float *> control_outputs;
          for (uint32_t t = 0; t < T; ++t) control_outputs.push_back(control.data() + size_t(t) * 4096);
          for (float limit : {0.001f, fmt.swiglu_limit}) {
            batch.run_quantized(gate, up, down, qinputs, outputs, limit);
            original->run_quantized(gate, up, down, qinputs, control_outputs, limit);
            if (std::memcmp(actual.data(), control.data(), actual.size() * 4))
              throw std::runtime_error("replayed expert clamp changed output bytes");
            std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                      << ",\"limit\":" << limit << ",\"replay_limit_byte_identical\":true}\n";
          }
          if (std::memcmp(actual.data(), fixed.data(), actual.size() * 4))
            throw std::runtime_error("expert replay failed to restore the original clamp");
        }
        if (benchmark) {
          const auto fixed_output = actual;
          std::vector<float> baseline(actual.size());
          std::vector<float *> baseline_outputs;
          for (uint32_t t = 0; t < T; ++t) baseline_outputs.push_back(baseline.data() + size_t(t) * 4096);
          // Three warmup pairs, then five alternating pairs. Every invocation
          // checks bytes; the 64-column control uses the same weight loader.
          for (int trial = -3; trial < 5; ++trial) {
            auto run = [&](bool control) {
              auto &selected = control ? (benchmark_control ? *benchmark_control : *original) : batch;
              const auto before = selected.stats();
              const auto started = std::chrono::steady_clock::now();
              selected.run_quantized(gate, up, down, qinputs, control ? baseline_outputs : outputs, fmt.swiglu_limit);
              const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
              const auto &result = control ? baseline : actual;
              if (std::memcmp(result.data(), fixed_output.data(), result.size() * 4))
                throw std::runtime_error("expert benchmark changed output bytes");
              if (trial >= 0)
                std::cout << "{\"expert\":" << expert << ",\"columns\":" << T << ",\"trial\":" << trial
                          << ",\"control\":" << (control ? "true" : "false") << ",\"column_tile\":" << (control ? 64 : tile_columns)
                          << ",\"row_tile\":" << (control ? 64 : tile_rows)
                          << ",\"vector_weights\":" << (vector_weights ? "true" : "false")
                          << ",\"run_wall_ms\":" << wall << ",\"upload_ms\":" << selected.stats().upload_ms - before.upload_ms
                          << ",\"queued_work_ms\":" << selected.stats().queued_work_ms - before.queued_work_ms
                          << ",\"record_ms\":" << selected.stats().record_ms - before.record_ms
                          << ",\"submit_wait_ms\":" << selected.stats().submit_wait_ms - before.submit_wait_ms
                          << ",\"output_byte_identical\":true}\n";
            };
            run(trial % 2 == 0); run(trial % 2 != 0);
          }
        }
      }
    }
    if (replay && !batch.stats().command_replays) throw std::runtime_error("expert replay check did not replay a command");
    std::cout << "{\"command_records\":" << batch.stats().command_records
              << ",\"command_replays\":" << batch.stats().command_replays << "}\n";
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
