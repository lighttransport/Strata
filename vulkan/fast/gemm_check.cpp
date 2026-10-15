// Small synthetic quantized product oracle. --cpu with an LVP-only ICD can
// validate shader numerics without submitting work to the physical GPU.
#include "fast/kernels.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iostream>
#include <random>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_gemm_check [--cpu]");
    fast::Device dev(cpu ? "llvmpipe" : "", 256ULL << 20, cpu);
    std::mt19937 rng(182); std::normal_distribution<float> normal(0.f, 0.2f);
    constexpr uint32_t M = 67;
    for (uint32_t K : {256u, 768u}) {
    for (int type : {fast::F32, fast::Q4_K, fast::Q6_K, fast::Q5_0, fast::Q2_K, fast::Q3_K}) {
      for (uint32_t heads : {1u, 2u}) {
      const auto *traits = ggml_get_type_traits(ggml_type(type));
      const size_t rb = fast::row_bytes(type, K);
      std::vector<float> source(heads * M * K), decoded(source.size());
      for (auto &v : source) v = normal(rng);
      std::vector<uint8_t> packed(heads * M * rb);
      if (type == fast::F32) {std::memcpy(packed.data(), source.data(), packed.size()); decoded = source;}
      else for (uint32_t row = 0; row < heads * M; ++row) {
        traits->from_float_ref(source.data() + row * K, packed.data() + row * rb, K);
        traits->to_float(packed.data() + row * rb, decoded.data() + row * K, K);
      }
      auto w = fast::upload_matrix(dev, type, K, M, heads, packed.data());
      for (uint32_t N : {3u, 65u, 257u}) {
        const uint32_t xs = heads * K + 16, ys = heads * M + 16, offset = 4;
        std::vector<float> input(N * xs + offset);
        for (auto &v : input) v = normal(rng);
        auto x = dev.buffer(input.size() * 4, fast::Memory::Host);
        auto y = dev.buffer((N * ys + offset) * 4, fast::Memory::Host);
        auto partial = dev.buffer(size_t(16) * N * M * 4, fast::Memory::Device);
        std::memcpy(x->mapped, input.data(), input.size() * 4);
        for (uint32_t tile : {64u, 128u, 256u}) {
          fast::GemvArgs a; a.columns = N; a.xs = xs; a.ys = ys; a.xofs = offset; a.yofs = offset;
          a.scratch = fast::View(*partial);
          dev.run([&](fast::Recorder &r) {fast::gemm(r, w, fast::View(*x), fast::View(*y), a, tile); r.host_visible();});
          double error = 0, norm = 0;
          for (uint32_t h = 0; h < heads; ++h)
            for (uint32_t n = 0; n < N; ++n)
              for (uint32_t row = 0; row < M; ++row) {
                double ref = 0;
                for (uint32_t k = 0; k < K; ++k) ref += double(decoded[(h * M + row) * K + k]) * input[offset + n * xs + h * K + k];
                const float actual = y->as<float>()[offset + n * ys + h * M + row];
                if (!std::isfinite(actual)) throw std::runtime_error("nonfinite synthetic product");
                error += (actual - ref) * (actual - ref); norm += ref * ref;
              }
          const double rel = std::sqrt(error / std::max(norm, 1e-30));
          std::cout << "{\"type\":" << type << ",\"K\":" << K << ",\"heads\":" << heads << ",\"columns\":" << N << ",\"tile\":" << tile << ",\"rel_l2\":" << rel << "}\n" << std::flush;
          if (!(rel < 2e-5)) throw std::runtime_error("synthetic GEMM parity failed");
        }
      }
      }
    }
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
