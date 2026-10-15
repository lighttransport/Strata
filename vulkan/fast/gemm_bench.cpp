// Tiled prefill product checks at model geometries, with sampled CPU FP64 dots.
#include "fast/kernels.hpp"
#include "model_checks.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iostream>
#include <random>

namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: glm_vulkan_fast_gemm_bench MODEL");
    strata::core::ModelArtifact artifact(argv[1]);
    strata::vulkan::checked_model(artifact);
    fast::Device dev("", 15ULL << 30);
    std::mt19937 rng(5417);
    std::normal_distribution<float> normal;
    const std::array<const char *, 8> names{
        "blk.4.attn_q.weight", "blk.4.attn_v.weight", "blk.4.ssm_f_b.weight",
        "blk.3.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight",
        "blk.3.ffn_gate_inp.weight", "blk.3.attn_k_b.weight", "blk.4.hc_attn_fn.weight"};
    for (const char *name : names) {
      const auto &tensor = artifact.at(name);
      const uint32_t k = uint32_t(tensor.tensor->shape[0]), m = uint32_t(tensor.tensor->shape[1]);
      const bool heads_case = std::string(name) == "blk.3.attn_k_b.weight";
      const uint32_t heads = heads_case ? uint32_t(tensor.tensor->shape[2]) : 1;
      auto weights = fast::upload_matrix(dev, int(tensor.tensor->type), k, m, heads, tensor.data());
      const size_t rb = fast::row_bytes(int(tensor.tensor->type), k);
      for (uint32_t columns : {7u, 65u, 256u}) {
        for (uint32_t tile_columns : {64u, 128u, 256u}) {
        const uint32_t xs = k * heads + 32, ys = m * heads + 32, xofs = 16, yofs = 16;
        std::vector<float> input(size_t(columns) * xs + xofs);
        for (float &v : input) v = normal(rng);
        auto x = dev.buffer(input.size() * 4, fast::Memory::Device);
        auto y = dev.buffer((size_t(columns) * ys + yofs) * 4, fast::Memory::Host);
        auto partial = dev.buffer(size_t(16) * m * columns * 4, fast::Memory::Device);
        dev.upload(*x, input.data(), input.size() * 4);
        fast::GemvArgs args;
        args.columns = columns; args.xs = xs; args.ys = ys; args.xofs = xofs; args.yofs = yofs;
        args.scratch = fast::View(*partial);
        auto cmd = dev.allocate_command();
        dev.begin(cmd, false);
        dev.reset_queries(cmd, 0, 2);
        fast::Recorder rec(dev, cmd);
        rec.timestamp(0);
        fast::gemm(rec, weights, fast::View(*x), fast::View(*y), args, tile_columns);
        rec.timestamp(1);
        rec.host_visible();
        dev.end(cmd);
        std::vector<double> milliseconds;
        for (int trial = 0; trial < 6; ++trial) {
          dev.submit_wait(cmd);
          const auto ts = dev.read_timestamps(0, 2);
          if (trial) milliseconds.push_back((ts[1] - ts[0]) / 1e6);
        }
        dev.free_command(cmd);
        double error = 0, norm = 0;
        std::vector<float> row(k);
        for (uint32_t head = 0; head < heads; head += std::max(1u, heads / 4))
          for (uint32_t r = 0; r < m; r += std::max(1u, m / 16)) {
            const uint8_t *source = tensor.data() + (size_t(head) * m + r) * rb;
            if (tensor.tensor->type == fast::F32) std::memcpy(row.data(), source, k * 4);
            else ggml_get_type_traits(ggml_type(tensor.tensor->type))->to_float(source, row.data(), k);
            for (uint32_t n = 0; n < columns; n += std::max(1u, columns / 8)) {
              double ref = 0;
              for (uint32_t i = 0; i < k; ++i) ref += double(row[i]) * input[xofs + size_t(n) * xs + head * k + i];
              const float got = y->as<float>()[yofs + size_t(n) * ys + head * m + r];
              if (!std::isfinite(got)) throw std::runtime_error("nonfinite GEMM output");
              error += (got - ref) * (got - ref); norm += ref * ref;
            }
          }
        const double rel = std::sqrt(error / std::max(norm, 1e-30));
        std::sort(milliseconds.begin(), milliseconds.end());
        const double ms = milliseconds[milliseconds.size() / 2];
        std::cout << "{\"tensor\":\"" << name << "\",\"type\":" << tensor.tensor->type
                  << ",\"k\":" << k << ",\"m\":" << m << ",\"heads\":" << heads << ",\"columns\":" << columns
                  << ",\"tile_columns\":" << tile_columns
                  << ",\"median_ms\":" << ms << ",\"tflops\":" << 2.0 * k * m * heads * columns / ms / 1e9
                  << ",\"rel_l2\":" << rel << "}\n" << std::flush;
        if (!(rel < 2e-5)) throw std::runtime_error("tiled GEMM parity failed");
        }
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "VULKAN_FAST_GEMM_ERROR " << e.what() << '\n';
    return 1;
  }
}
