// Invalid blocks must propagate NaN without using an invalid maximum index.
// Finite grid values are checked against the existing canonical CPU encoder.
#include "fast/kernels.hpp"
#include "strata/kernels/canon_expert.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_q8_check [--cpu]");
    fast::Device dev(cpu ? "llvmpipe" : "", 64ULL << 20, cpu);
    constexpr uint32_t blocks = 10, n = blocks * 256;
    auto input = dev.buffer(n * 4, fast::Memory::Host);
    auto codes = dev.buffer(n * 4, fast::Memory::Host);
    auto values = dev.buffer(n * 4, fast::Memory::Host);
    auto meta = dev.buffer(blocks * 17 * 4, fast::Memory::Host);
    auto hits = dev.buffer(256, fast::Memory::Host);
    std::mt19937 rng(952); std::normal_distribution<float> normal(0.f, .5f);
    for (uint32_t i = 0; i < n; ++i) input->as<float>()[i] = normal(rng);
    std::fill(input->as<float>(), input->as<float>() + 256, 0.f);
    for (uint32_t i = 0; i < 256; ++i) input->as<float>()[256 + i] = i % 2 ? -2.f : 2.f;
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    std::fill(input->as<float>() + 4 * 256, input->as<float>() + 5 * 256, nan);
    input->as<float>()[5 * 256] = nan;
    input->as<float>()[6 * 256 + 255] = inf;
    input->as<float>()[7 * 256 + 19] = -inf;
    std::fill(input->as<float>() + 8 * 256, input->as<float>() + 9 * 256, inf);
    std::fill(input->as<float>() + 9 * 256, input->as<float>() + 10 * 256, -inf);
    dev.run([&](fast::Recorder &r) {
      r.dispatch(cpu ? "q8k_codes_portable" : "q8k_codes", {fast::View(*input), fast::View(*codes), fast::View(*hits), fast::View(*meta)},
                 std::array<uint32_t, 4>{n, 0, 0, 1}, blocks);
      r.dispatch(cpu ? "q8k_round_portable" : "q8k_round", {fast::View(*input), fast::View(*values), fast::View(*hits)},
                 std::array<uint32_t, 4>{n, 0, 0, 1}, blocks);
      r.host_visible();
    });
    for (uint32_t b = 0; b < blocks; ++b) {
      const float *q = codes->as<float>() + b * 256, *v = values->as<float>() + b * 256;
      const float *m = meta->as<float>() + b * 17;
      if (b >= 4) {
        if (!std::isnan(m[0])) throw std::runtime_error("invalid Q8 block lacks NaN scale");
        for (uint32_t i = 0; i < 256; ++i)
          if (q[i] != 0.f || !std::isnan(v[i])) throw std::runtime_error("invalid Q8 block not safely propagated");
        for (uint32_t i = 1; i < 17; ++i)
          if (m[i] != 0.f) throw std::runtime_error("invalid Q8 block has nonzero sums");
      } else {
        std::vector<uint8_t> native(ggml_row_size(GGML_TYPE_Q8_K, 256));
        strata::kernels::cpu::canon_quant_q8k(input->as<float>() + b * 256, native.data(), 256);
        float d; std::memcpy(&d, native.data(), 4);
        if (std::memcmp(&d, m, 4)) throw std::runtime_error("finite Q8 scale differs from CPU");
        int16_t sums[16]; std::memcpy(sums, native.data() + 260, sizeof(sums));
        for (uint32_t i = 0; i < 16; ++i)
          if (m[1 + i] != float(sums[i])) throw std::runtime_error("finite Q8 sums differ from CPU");
        for (uint32_t i = 0; i < 256; ++i) {
          const float expected = float(reinterpret_cast<const int8_t *>(native.data() + 4)[i]);
          if (q[i] != expected || v[i] != expected * d) throw std::runtime_error("finite Q8 grid differs from CPU");
        }
      }
      std::cout << "{\"block\":" << b << ",\"invalid\":" << (b >= 4 ? "true" : "false") << ",\"passed\":true}\n";
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
