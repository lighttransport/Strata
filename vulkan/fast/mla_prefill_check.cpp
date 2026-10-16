#include "fast/device.hpp"
#include "fast/mla_prefill.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_mla_prefill_check [--cpu]");
    fast::Device device(cpu ? "llvmpipe" : "", 128ULL << 20, cpu);
    constexpr uint32_t heads = 64, latent = 512, queries = 9, stride = 2176, rows = 4096;
    auto cache = device.buffer(size_t(rows) * latent * 4, fast::Memory::Host);
    auto q = device.buffer(size_t(queries) * heads * latent * 4, fast::Memory::Host);
    auto ids = device.buffer(size_t(queries) * stride * 4, fast::Memory::Host);
    auto old = device.buffer(q->size + 256, fast::Memory::Host);
    auto tiled = device.buffer(q->size + 256, fast::Memory::Host);
    auto gemm = device.buffer(q->size + 256, fast::Memory::Host);
    fast::MlaPrefill batch(device);
    std::mt19937 rng(137); std::uniform_real_distribution<float> uniform(-1, 1);
    for (size_t i = 0; i < cache->size / 4; ++i) cache->as<float>()[i] = uniform(rng);
    for (size_t i = 0; i < q->size / 4; ++i) q->as<float>()[i] = uniform(rng);
    struct Push {uint32_t heads, latent; float scale; uint32_t istride;} push{heads, latent, 0.125f, stride};
    for (uint32_t count : {1u, 15u, 16u, 17u, 255u, 2048u, 2112u}) {
      for (uint32_t t = 0; t < queries; ++t) {
        auto *at = ids->as<int>() + size_t(t) * stride; at[0] = int(count);
        for (uint32_t j = 0; j < count; ++j) at[j + 1] = int((j * 17 + t * 23) % rows);
      }
      std::fill(old->as<float>(), old->as<float>() + old->size / 4, 12345.f);
      std::fill(tiled->as<float>(), tiled->as<float>() + tiled->size / 4, 12345.f);
      std::fill(gemm->as<float>(), gemm->as<float>() + gemm->size / 4, 12345.f);
      device.run([&](fast::Recorder &r) {
        r.dispatch("mla_step", {fast::View(*cache), fast::View(*q), fast::View(*ids), fast::View(*old)}, push, heads, queries);
        // Also exercise nonzero descriptor offsets at a panel boundary.
        for (uint32_t first = 0; first < queries; first += 3)
          r.dispatch("mla_prefill_tiled", {fast::View(*cache), fast::View(*q, size_t(first) * heads * latent * 4),
              fast::View(*ids, size_t(first) * stride * 4), fast::View(*tiled, size_t(first) * heads * latent * 4)},
              push, heads / 4, std::min(3u, queries - first));
        // Cross the eight-query batch boundary with explicit descriptor ranges.
        batch.run(r, fast::View(*cache), fast::View(*q, 0, q->size), fast::View(*ids, 0, ids->size),
                  fast::View(*gemm, 0, q->size), queries, 0, push.scale, stride, false);
        r.host_visible();
      });
      double error = 0, norm = 0, oracle_error = 0, gemm_error = 0;
      for (size_t i = 0; i < q->size / 4; ++i) {
        const double a = tiled->as<float>()[i], b = old->as<float>()[i];
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite MLA output");
        const double c = gemm->as<float>()[i];
        if (!std::isfinite(c)) throw std::runtime_error("nonfinite GEMM MLA output");
        error += (a-b)*(a-b); norm += b*b; gemm_error += (c-b)*(c-b);
      }
      // Independent FP64 host oracle for one query/head, including every latent dimension.
      std::vector<double> scores(count); double maximum = -1e300, total = 0;
      for (uint32_t j = 0; j < count; ++j) {
        double sum = 0; const auto *kv = cache->as<float>() + size_t(ids->as<int>()[j + 1]) * latent;
        for (uint32_t k = 0; k < latent; ++k) sum += double(q->as<float>()[k]) * kv[k];
        scores[j] = sum * push.scale; maximum = std::max(maximum, scores[j]);
      }
      for (auto &score : scores) {score = std::exp(score - maximum); total += score;}
      for (uint32_t k = 0; k < latent; ++k) {
        double sum = 0;
        for (uint32_t j = 0; j < count; ++j) sum += scores[j] * cache->as<float>()[size_t(ids->as<int>()[j+1]) * latent + k];
        oracle_error = std::max({oracle_error, std::abs(sum / total - tiled->as<float>()[k]), std::abs(sum / total - gemm->as<float>()[k])});
      }
      for (size_t i = q->size / 4; i < tiled->size / 4; ++i)
        if (tiled->as<float>()[i] != 12345.f || old->as<float>()[i] != 12345.f || gemm->as<float>()[i] != 12345.f)
          throw std::runtime_error("MLA output sentinel overwritten");
      const double relative = std::sqrt(error / std::max(norm, 1e-30));
      std::cout << "{\"keys\":" << count << ",\"relative_l2\":" << relative
                << ",\"gemm_relative_l2\":" << std::sqrt(gemm_error / std::max(norm, 1e-30))
                << ",\"oracle_max_abs\":" << oracle_error << "}\n";
      if (relative > 2e-4 || std::sqrt(gemm_error / std::max(norm, 1e-30)) > 2e-4 || oracle_error > 2e-4) throw std::runtime_error("MLA numerical gate failed");
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
