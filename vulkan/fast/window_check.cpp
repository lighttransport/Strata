// A rejected verifier window can wrap the index ring before its accepted pool
// finishes. Its saved prefix must equal a standalone execution of that prefix.
#include "fast/device.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_window_check [--cpu]");
    fast::Device dev(cpu ? "llvmpipe" : "", 64ULL << 20, cpu);
    constexpr size_t width = 128, ring_floats = 2 * 4 * width;
    constexpr size_t max_tokens = 129, pooled_floats = 64 * width;
    auto stack = dev.buffer(max_tokens * width * 4, fast::Memory::Host);
    auto gates = dev.buffer(max_tokens * width * 4, fast::Memory::Host);
    auto parameters = dev.buffer((2 * width + 4 * width) * 4, fast::Memory::Host);
    auto ring = dev.buffer(ring_floats * 4, fast::Memory::Host);
    auto history = dev.buffer(4 * ring_floats * 4, fast::Memory::Device);
    auto pooled = dev.buffer(pooled_floats * 4, fast::Memory::Host);
    auto step = dev.buffer(max_tokens * 256, fast::Memory::Host);
    std::mt19937 rng(117); std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    for (size_t i = 0; i < max_tokens * width; ++i) {stack->as<float>()[i] = uniform(rng); gates->as<float>()[i] = uniform(rng);}
    for (size_t i = 0; i < 6 * width; ++i) parameters->as<float>()[i] = i < width ? 1.f : uniform(rng);
    std::array<float, ring_floats> initial{}, expected{};
    for (auto &v : initial) v = uniform(rng);
    for (uint32_t position : {1u, 2u, 3u, 5u, 6u, 7u})
      for (uint32_t keep : {1u, 2u, 3u}) {
        auto execute = [&](uint32_t tokens, bool snapshot) {
          dev.run([&](fast::Recorder &r) {
            r.dispatch("index_key", {fast::View(*stack), fast::View(*gates), fast::View(*parameters),
                         fast::View(*ring), fast::View(*pooled), fast::View(*step), fast::View(*history)},
                       std::array<uint32_t, 7>{128, 4, 0, 0, tokens, 128, snapshot ? 1u : 0u}, 1);
            r.host_visible();
          });
        };
        step->as<uint32_t>()[0] = position;
        std::memcpy(ring->mapped, initial.data(), sizeof(initial));
        execute(keep, false); std::memcpy(expected.data(), ring->mapped, sizeof(expected));
        std::memcpy(ring->mapped, initial.data(), sizeof(initial));
        execute(4, true);
        dev.run([&](fast::Recorder &r) {
          r.copy(fast::View(*history, keep * sizeof(initial)), fast::View(*ring), sizeof(initial));
          r.host_visible();
        });
        if (std::memcmp(expected.data(), ring->mapped, sizeof(expected))) throw std::runtime_error("MLA ring prefix restoration failed");
        std::cout << "{\"position\":" << position << ",\"keep\":" << keep << ",\"ring_byte_identical\":true}\n";
      }
    // Bulk prefill must leave the same ring and completed pools as serial
    // indexing, including windows starting and ending inside a pool.
    for (uint32_t position : {0u, 1u, 3u, 7u})
      for (uint32_t tokens : {5u, 32u, 129u}) {
        std::array<float, pooled_floats> expected_pools{};
        for (uint32_t t = 0; t < tokens; ++t) step->as<uint32_t>()[t * 64] = position + t;
        std::memcpy(ring->mapped, initial.data(), sizeof(initial));
        std::memset(pooled->mapped, 0, pooled->size);
        dev.run([&](fast::Recorder &r) {
          for (uint32_t t = 0; t < tokens; ++t)
            r.dispatch("index_key", {fast::View(*stack, t * width * 4), fast::View(*gates, t * width * 4),
                         fast::View(*parameters), fast::View(*ring), fast::View(*pooled),
                         fast::View(*step, t * 256), fast::View(*history)},
                       std::array<uint32_t, 7>{128, 4, 0, 0, 1, 128, 0}, 1);
          r.host_visible();
        });
        std::memcpy(expected.data(), ring->mapped, sizeof(expected));
        std::memcpy(expected_pools.data(), pooled->mapped, sizeof(expected_pools));
        std::memcpy(ring->mapped, initial.data(), sizeof(initial));
        std::memset(pooled->mapped, 0, pooled->size);
        dev.run([&](fast::Recorder &r) {
          r.dispatch("index_key", {fast::View(*stack), fast::View(*gates), fast::View(*parameters),
                       fast::View(*ring), fast::View(*pooled), fast::View(*step), fast::View(*history)},
                     std::array<uint32_t, 7>{128, 4, 0, 0, tokens, 128, 0}, 1);
          r.host_visible();
        });
        if (std::memcmp(expected.data(), ring->mapped, sizeof(expected)) ||
            std::memcmp(expected_pools.data(), pooled->mapped, sizeof(expected_pools)))
          throw std::runtime_error("bulk MLA indexing differs from serial indexing");
        std::cout << "{\"position\":" << position << ",\"tokens\":" << tokens << ",\"bulk_index_byte_identical\":true}\n";
      }
    // Cancellation makes a union-order reduction observably different. Each
    // token gets its own CPU/GPU placement and a different union-row ordering.
    for (uint32_t T : {1u, 2u, 4u}) {
      constexpr uint32_t W = 8, stride = 80, outofs = 16;
      auto mail = dev.buffer(T * stride * 4, fast::Memory::Host);
      auto gpu = dev.buffer(4 * T * W * 4, fast::Memory::Host);
      auto hits = dev.buffer(T * 32 * 4, fast::Memory::Host);
      auto sum = dev.buffer(T * W * 4, fast::Memory::Host);
      std::array<float, 4 * W> reference{};
      for (uint32_t t = 0; t < T; ++t) {
        for (int k = 7; k >= 0; --k) {
          const float weight = k == 7 ? 0.f : (k % 2 ? 0.25f : 1.f);
          mail->as<float>()[t * stride + k] = weight;
          const bool resident = uint32_t(k) % 2 == t % 2;
          const uint32_t u = resident ? (t % 2 ? 3u - uint32_t(k) / 2u : uint32_t(k) / 2u) : 0xffffffffu;
          hits->as<uint32_t>()[t * 32 + 17 + k] = u;
          for (uint32_t d = 0; d < W; ++d) {
            const float row = k % 3 == 0 ? 1e10f : k % 3 == 1 ? -1e10f : float(d + t + 1) / 3.f;
            mail->as<float>()[t * stride + outofs + uint32_t(k) * W + d] = row;
            if (resident) gpu->as<float>()[(u * T + t) * W + d] = row;
            if (weight != 0) reference[t * W + d] = std::fma(weight, row, reference[t * W + d]);
          }
        }
      }
      dev.run([&](fast::Recorder &r) {
        r.dispatch("route_merge", {fast::View(*mail), fast::View(*gpu), fast::View(*hits), fast::View(*sum)},
                   std::array<uint32_t, 8>{W, 8, 0, stride, 0, T, outofs, 0}, 1, T);
        r.host_visible();
      });
      if (std::memcmp(reference.data(), sum->mapped, T * W * 4)) throw std::runtime_error("canonical mixed route sum failed");
      std::cout << "{\"tokens\":" << T << ",\"mixed_route_sum_byte_identical\":true}\n";
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
