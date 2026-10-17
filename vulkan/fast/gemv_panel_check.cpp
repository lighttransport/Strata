// Native 64-lane GEMV panel byte controls and paired GPU timings. The legacy
// control requires 64-lane subgroups; this checker never falls back to a CPU ICD.
#include "fast/kernels.hpp"
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
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    if (argc != 1 && !benchmark)
      throw std::invalid_argument("usage: glm_vulkan_fast_gemv_panel_check [--benchmark]");
    fast::Device dev("", 256ULL << 20);
    if (dev.subgroup_size() != 64)
      throw std::runtime_error("legacy GEMV panel control requires 64-lane subgroups");
    std::mt19937 rng(38119);
    std::normal_distribution<float> normal(0.f, 0.2f);
    uint32_t cases = 0;
    auto check = [&](int type, uint32_t K, uint32_t M, uint32_t heads, uint32_t N, bool accumulate, bool timed) {
      const auto *traits = ggml_get_type_traits(ggml_type(type));
      const size_t rb = fast::row_bytes(type, K);
      std::vector<float> source(size_t(heads) * M * K), decoded(source.size());
      for (float &v : source) v = normal(rng);
      std::vector<uint8_t> packed(size_t(heads) * M * rb);
      if (type == fast::F32) {
        std::memcpy(packed.data(), source.data(), packed.size());
        decoded = source;
      } else for (uint32_t row = 0; row < heads * M; ++row) {
        traits->from_float_ref(source.data() + size_t(row) * K, packed.data() + size_t(row) * rb, K);
        traits->to_float(packed.data() + size_t(row) * rb, decoded.data() + size_t(row) * K, K);
      }
      auto w = fast::upload_matrix(dev, type, K, M, heads, packed.data());
      const uint32_t hx = K + 8, hy = M + 5, xs = heads * hx + 16, ys = heads * hy + 16;
      constexpr uint32_t xofs = 4, yofs = 3;
      std::vector<float> input(size_t(N) * xs + xofs), initial(size_t(N) * ys + yofs, 0.125f);
      for (float &v : input) v = normal(rng);
      auto x = dev.buffer(input.size() * 4, fast::Memory::Device);
      auto y = dev.buffer(initial.size() * 4, fast::Memory::Host);
      auto reset = dev.buffer(initial.size() * 4, fast::Memory::Host);
      auto scratch = dev.buffer(size_t(16) * M * 16 * 4, fast::Memory::Device);
      dev.upload(*x, input.data(), input.size() * 4);
      dev.upload(*reset, initial.data(), initial.size() * 4);
      std::array<VkCommandBuffer, 3> commands{};
      const std::array<uint32_t, 3> widths{4, 8, 16};
      for (size_t i = 0; i < widths.size(); ++i) {
        auto c = dev.allocate_command();
        commands[i] = c;
        dev.begin(c, false);
        dev.reset_queries(c, 0, 2);
        fast::Recorder rec(dev, c);
        rec.copy(fast::View(*reset), fast::View(*y), initial.size() * 4);
        rec.timestamp(0);
        for (uint32_t first = 0; first < N; first += widths[i]) {
          fast::GemvArgs a;
          a.columns = std::min(widths[i], N - first);
          a.xs = xs; a.ys = ys; a.hx = hx; a.hy = hy;
          a.xofs = xofs + first * xs; a.yofs = yofs + first * ys;
          a.accumulate = accumulate;
          a.scratch = fast::View(*scratch, 0, scratch->size);
          fast::gemv(rec, w, fast::View(*x), fast::View(*y), a, widths[i]);
          rec.barrier();
        }
        rec.timestamp(1);
        rec.host_visible();
        dev.end(c);
      }
      dev.submit_wait(commands[0]);
      std::vector<float> expected(initial.size()), actual(initial.size());
      dev.download(*y, expected.data(), expected.size() * 4);
      double error = 0, norm = 0;
      uint32_t samples = 0;
      for (uint32_t h = 0; h < heads; ++h)
        for (uint32_t n = 0; n < N; n += std::max(1u, N / 16))
          for (uint32_t row = 0; row < M; row += std::max(1u, M / 32)) {
            const size_t at = yofs + size_t(n) * ys + h * hy + row;
            double ref = accumulate ? initial[at] : 0;
            for (uint32_t k = 0; k < K; ++k)
              ref += double(decoded[(size_t(h) * M + row) * K + k]) * input[xofs + size_t(n) * xs + h * hx + k];
            const double got = expected[at];
            if (!std::isfinite(got)) throw std::runtime_error("nonfinite GEMV control output");
            error += (got - ref) * (got - ref); norm += ref * ref; ++samples;
          }
      const double rel = std::sqrt(error / std::max(norm, 1e-30));
      if (!(rel < 2e-5)) throw std::runtime_error("legacy GEMV sampled FP64 screen failed");
      std::array<std::vector<double>, 3> times;
      const int first_trial = timed ? -3 : 0, last_trial = timed ? 5 : 1;
      for (int trial = first_trial; trial < last_trial; ++trial)
        for (size_t step = 0; step < widths.size(); ++step) {
          const size_t i = (step + size_t(trial - first_trial)) % widths.size();
          dev.submit_wait(commands[i]);
          dev.download(*y, actual.data(), actual.size() * 4);
          if (std::memcmp(actual.data(), expected.data(), actual.size() * 4))
            throw std::runtime_error("wider GEMV changed output or sentinel bytes");
          if (timed && trial >= 0) {
            const auto ts = dev.read_timestamps(0, 2);
            times[i].push_back((ts[1] - ts[0]) / 1e6);
          }
        }
      for (auto c : commands) dev.free_command(c);
      for (size_t i = 1; i < widths.size(); ++i) {
        std::cout << "{\"type\":" << type << ",\"K\":" << K << ",\"M\":" << M
                  << ",\"heads\":" << heads << ",\"columns\":" << N << ",\"accumulate\":" << accumulate
                  << ",\"panel\":" << widths[i] << ",\"all_output_and_sentinel_bytes_identical\":true"
                  << ",\"sampled_control_fp64_rel_l2\":" << rel << ",\"fp64_samples\":" << samples;
        if (timed) {
          auto median = [](std::vector<double> v) {std::sort(v.begin(), v.end()); return v[v.size() / 2];};
          std::cout << ",\"control_median_gpu_ms\":" << median(times[0])
                    << ",\"candidate_median_gpu_ms\":" << median(times[i])
                    << ",\"gpu_speedup\":" << median(times[0]) / median(times[i]);
        }
        std::cout << "}\n" << std::flush;
        ++cases;
      }
    };
    for (int type : {fast::F32, fast::Q4_K, fast::Q6_K, fast::Q5_0})
      for (uint32_t K : {256u, 768u, 4096u})
        for (uint32_t heads : {1u, 2u}) {
          if (type == fast::F32 && heads != 1) continue;
          for (uint32_t N : {1u, 5u, 17u, 33u, 129u})
            for (bool accumulate : {false, true}) check(type, K, 67, heads, N, accumulate, false);
        }
    if (benchmark)
      for (int type : {fast::F32, fast::Q4_K, fast::Q6_K, fast::Q5_0})
        check(type, type == fast::F32 ? 16384 : 4096, type == fast::F32 ? 144 : 4096, 1, 129, false, true);
    std::cout << "{\"passed_panel_cases\":" << cases << "}\n";
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
