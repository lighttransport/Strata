// Large activation slices must retain the per-token shader's output bytes.
// --cpu selects only software Vulkan; this does not execute the full model.
#include "fast/kernels.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_swiglu_check [--cpu]");
    fast::Device dev(cpu ? "llvmpipe" : "", 512ULL << 20, cpu);
    for (const auto shape : {std::array<uint32_t, 2>{2048, 5}, {12288, 1024}, {12288, 1025}, {12288, 2048}}) {
      const auto [n, T] = shape;
      const size_t row = size_t(n) * 4, bytes = size_t(T) * row;
      auto gu = dev.buffer(2 * bytes, fast::Memory::Host);
      auto baseline = dev.buffer(bytes + 512, fast::Memory::Host);
      auto actual = dev.buffer(bytes + 512, fast::Memory::Host);
      for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < n; ++i) {
          gu->as<float>()[size_t(t) * 2 * n + i] = float(int((i + t) % 31) - 15) * .25f;
          gu->as<float>()[size_t(t) * 2 * n + n + i] = float(int((3 * i + t) % 29) - 14) * .25f;
        }
      for (float limit : {0.f, 1.f}) {
        dev.run([&](fast::Recorder &r) {
          r.fill(fast::View(*baseline), baseline->size, 0xa5a5a5a5u);
          for (uint32_t t = 0; t < T; ++t) {
            struct {uint32_t n, tokens; float limit;} push{n, 1, limit};
            r.dispatch("swiglu", {fast::View(*gu, size_t(t) * 2 * row, 2 * row),
                                    fast::View(*baseline, 256 + size_t(t) * row, row)}, push, n / 256);
          }
          r.host_visible();
        });
        dev.run([&](fast::Recorder &r) {
          r.fill(fast::View(*actual), actual->size, 0xa5a5a5a5u);
          fast::swiglu(r, fast::View(*gu), fast::View(*actual, 256, bytes), n, T, limit);
          r.host_visible();
        });
        if (std::memcmp(baseline->mapped, actual->mapped, bytes + 512))
          throw std::runtime_error("windowed SwiGLU differs from per-token shader bytes/sentinels");
        for (uint32_t t = 0; t < T; ++t)
          for (uint32_t i : {0u, n / 2, n - 1}) {
            float g = gu->as<float>()[size_t(t) * 2 * n + i], u = gu->as<float>()[size_t(t) * 2 * n + n + i];
            if (limit > 0) {g = std::min(g, limit); u = std::clamp(u, -limit, limit);}
            const float expected = g * (1.f / (1.f + std::exp(-g))) * u;
            const float got = actual->as<float>()[64 + size_t(t) * n + i];
            if (!std::isfinite(got) || std::abs(got - expected) > 2e-5f * std::max(1.f, std::abs(expected)))
              throw std::runtime_error("windowed SwiGLU CPU oracle failed");
          }
        std::cout << "{\"units\":" << n << ",\"tokens\":" << T << ",\"limit\":" << limit
                  << ",\"byte_identical\":true}\n" << std::flush;
      }
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
