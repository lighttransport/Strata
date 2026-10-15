// Whole streamed experts, including both Q8_K grids and SwiGLU, against the
// canonical CPU implementation. Software Vulkan uses an explicit subgroup-
// independent reduction variant; RX570 keeps the existing wave64 variant.
#include "fast/expert_batch.hpp"
#include "model_checks.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/q23_avx2.hpp"
#include <algorithm>
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
static float scalar_q2(const uint8_t *w, const uint8_t *a, uint32_t n) {
  float acc[8]{};
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
      acc[j] = std::fma(-ggml_fp16_to_fp32(dh[1]) * yd, float(correction), acc[j]);
      acc[j] = std::fma(ggml_fp16_to_fp32(dh[0]) * yd, float(dot), acc[j]);
    }
  }
  return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
}
int main(int argc, char **argv) {
  try {
    bool software = false, gpu_pack = false, vector_weights = false;
    const char *usage = "usage: glm_vulkan_fast_expert_batch_check MODEL [--cpu] [--gpu-pack] [--vector-weights]";
    if (argc < 2) throw std::invalid_argument(usage);
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "--cpu" && !software) software = true;
      else if (std::string(argv[i]) == "--gpu-pack" && !gpu_pack) gpu_pack = true;
      else if (std::string(argv[i]) == "--vector-weights" && !vector_weights) vector_weights = true;
      else throw std::invalid_argument(usage);
    }
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &model = strata::vulkan::checked_model(artifact);
    fast::Device dev(software ? "llvmpipe" : "", 1ULL << 30, software);
    {
      constexpr uint32_t count = 512;
      auto input = dev.buffer(count * 16, fast::Memory::Host);
      auto output = dev.buffer(count * 16, fast::Memory::Host);
      std::mt19937 probe(733); std::uniform_real_distribution<float> fraction(0.5f, 2.f);
      for (uint32_t i = 0; i < count; ++i) {
        const float a = fraction(probe), b = fraction(probe);
        input->as<float>()[4 * i] = a; input->as<float>()[4 * i + 1] = b;
        input->as<float>()[4 * i + 2] = -float(a * b);
      }
      dev.run([&](fast::Recorder &r) {
        r.dispatch("fma_oracle", {fast::View(*input), fast::View(*output)},
                   std::array<uint32_t, 1>{count}, count / 256);
        r.host_visible();
      });
      uint32_t ordinary = 0, precise = 0, separate = 0;
      for (uint32_t i = 0; i < count; ++i) {
        const float *v = input->as<float>() + 4 * i;
        const float ref = std::fma(v[0], v[1], v[2]);
        ordinary += output->as<float>()[4 * i] != ref;
        precise += output->as<float>()[4 * i + 1] != ref;
        separate += output->as<float>()[4 * i + 2] != ref;
      }
      std::cout << "{\"fma_probe_cases\":" << count << ",\"ordinary_mismatches\":" << ordinary
                << ",\"precise_mismatches\":" << precise << ",\"separate_mismatches\":" << separate << "}\n";
    }
    const auto &G = artifact.at("blk.3.ffn_gate_exps.weight"), &U = artifact.at("blk.3.ffn_up_exps.weight"),
               &D = artifact.at("blk.3.ffn_down_exps.weight");
    if (G.tensor->type != fast::Q2_K || U.tensor->type != fast::Q2_K || D.tensor->type != fast::Q3_K)
      throw std::invalid_argument("streamed expert check requires Q2_K gate/up and Q3_K down");
    cpu::NativeFmt fmt; std::string error;
    if (!cpu::native_fmt(G.tensor->type, D.tensor->type, 4096, 2048, fmt, error)) throw std::runtime_error(error);
    fmt.canon = true; fmt.swiglu_limit = model.layers[3].swiglu_limit;
    fast::ExpertBatch batch(dev, 129, software, gpu_pack, vector_weights);
    std::unique_ptr<fast::ExpertBatch> original;
    if (vector_weights) original = std::make_unique<fast::ExpertBatch>(dev, 129, software, gpu_pack, false);
    std::mt19937 rng(7741); std::normal_distribution<float> normal(0.f, 0.5f);
    for (int expert : {5, 77}) {
      const auto *gate = G.data() + size_t(expert) * G.bytes / 144;
      const auto *up = U.data() + size_t(expert) * U.bytes / 144;
      const auto *down = D.data() + size_t(expert) * D.bytes / 144;
      for (uint32_t T : {5u, 33u, 129u}) {
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
            throw std::runtime_error("vector weight loads changed whole-expert output bytes");
          std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                    << ",\"vector_scalar_byte_identical\":true}\n";
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
                    << ",\"changed\":" << changed << "}\n";
        };
        diagnostic("qinput", qinput, packed_trace.qinput);
        diagnostic("gate_up", gate_up, packed_trace.gate_up);
        diagnostic("hidden", expected_hidden, packed_trace.hidden);
        diagnostic("qhidden", qhidden, packed_trace.qhidden);
        cpu::native_quant_act(fmt, input.data(), act.data());
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
        std::cout << "{\"expert\":" << expert << ",\"columns\":" << T
                  << ",\"software\":" << software << ",\"rel_l2_cpu_canonical\":" << rel << "}\n" << std::flush;
        if (!(rel < 2e-4)) throw std::runtime_error("streamed expert canonical parity failed");
      }
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
