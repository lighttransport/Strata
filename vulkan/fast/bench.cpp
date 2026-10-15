// Phase-0 measurements for the fast GLM Vulkan engine: GEMV parity and
// bandwidth at the model's real shapes, host/GPU handoff latency, per-dispatch
// cost inside one command buffer, upload bandwidth and host RAM read rate.
// Output is one JSON object per line.
#include "fast/device.hpp"
#include "fast/kernels.hpp"
#include "model_checks.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
namespace fast = strata::vulkan::fast;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static std::vector<float> dequant_rows(const strata::core::ArtifactTensor &t, size_t first_row, size_t rows) {
  const uint32_t k = uint32_t(t.tensor->shape[0]);
  const size_t rb = fast::row_bytes(int(t.tensor->type), k);
  std::vector<float> out(rows * k);
  if (t.tensor->type == 0) {
    std::memcpy(out.data(), t.data() + first_row * rb, rows * rb);
    return out;
  }
  auto *traits = ggml_get_type_traits(ggml_type(t.tensor->type));
  for (size_t r = 0; r < rows; ++r) traits->to_float(t.data() + (first_row + r) * rb, out.data() + r * k, k);
  return out;
}

struct Case {
  std::string name;
  int columns;
};

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::invalid_argument("usage: glm_vulkan_fast_bench MODEL [--quick]");
    const bool quick = argc > 2 && std::string(argv[2]) == "--quick";
    strata::core::ModelArtifact artifact(argv[1]);
    strata::vulkan::checked_model(artifact);
    fast::Device dev("", 15ULL << 30);
    std::cout << std::setprecision(6);
    auto query = [&](auto &&record, int repeats) {
      // Times `record` repeats times inside one command buffer with timestamps.
      auto c = dev.allocate_command();
      dev.begin(c, false);
      dev.reset_queries(c, 0, repeats + 1);
      fast::Recorder r(dev, c);
      r.timestamp(0);
      for (int i = 0; i < repeats; ++i) {
        record(r);
        r.timestamp(i + 1);
      }
      dev.end(c);
      // Warm-up: the GPU clocks down while the host repacks weights.
      for (auto t0 = Clock::now(); since(t0) < 30000;) dev.submit_wait(c);
      dev.submit_wait(c);
      dev.free_command(c);
      auto ts = dev.read_timestamps(0, repeats + 1);
      std::vector<double> us;
      for (int i = 0; i < repeats; ++i) us.push_back((ts[i + 1] - ts[i]) / 1000);
      return us;
    };

    // ---- GEMV parity and bandwidth -----------------------------------------
    const std::vector<std::string> tensors = {
        "blk.4.attn_q.weight",     "blk.4.attn_v.weight",       "blk.4.attn_output.weight",
        "blk.4.ssm_f_b.weight",    "blk.4.ssm_f_a.weight",      "blk.4.ssm_beta.weight",
        "blk.3.attn_q_a.weight",   "blk.3.attn_q_b.weight",     "blk.3.attn_kv_a_mqa.weight",
        "blk.3.attn_output.weight", "blk.3.attn_k_b.weight",    "blk.3.attn_v_b.weight",
        "blk.3.ffn_gate_shexp.weight", "blk.3.ffn_down_shexp.weight", "blk.3.ffn_gate_inp.weight",
        "blk.4.hc_attn_fn.weight", "blk.3.indexer.attn_q_b.weight", "blk.0.ffn_down.weight",
        "output.weight"};
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.f, 1.f);
    // Strict activation rounding must also handle offset, strided windows.
    {
      constexpr uint32_t n = 128, stride = 160, offset = 32;
      std::vector<float> input(offset + 4 * stride);
      for (auto &v : input) v = normal(rng);
      std::fill(input.begin() + offset, input.begin() + offset + 32, 0.f);
      auto x = dev.buffer(input.size() * 4, fast::Memory::Device);
      auto y = dev.buffer(4 * n * 4, fast::Memory::Host);
      dev.upload(*x, input.data(), input.size() * 4);
      for (uint32_t columns : {1u, 2u, 3u, 4u}) {
        dev.run([&](fast::Recorder &r) {
          r.dispatch("q8_round", {fast::View(*x), fast::View(*y)},
                     std::array<uint32_t, 4>{n, offset, stride, n}, n / 32, columns);
          r.host_visible();
        });
        float error = 0;
        for (uint32_t t = 0; t < columns; ++t)
          for (uint32_t b = 0; b < n; b += 32) {
            float maximum = 0;
            for (uint32_t i = 0; i < 32; ++i)
              maximum = std::max(maximum, std::abs(input[offset + t * stride + b + i]));
            const float scale = std::min(maximum / 127.f, 65504.f);
            const float stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
            for (uint32_t i = 0; i < 32; ++i) {
              const float v = input[offset + t * stride + b + i];
              const float expected = maximum == 0 || stored == 0 ? 0.f :
                  std::clamp(std::round(v / scale), -127.f, 127.f) * stored;
              error = std::max(error, std::abs(y->as<float>()[t * n + b + i] - expected));
            }
          }
        std::cout << "{\"kind\":\"q8_round\",\"columns\":" << columns << ",\"max_abs\":" << error << "}\n";
        if (error != 0) throw std::runtime_error("Q8 activation rounding parity failed");
      }
    }
    for (const auto &name : tensors) {
      const auto &t = artifact.at(name);
      auto w = fast::upload_matrix(dev, t);
      const uint32_t k = w.k(), m = w.m(), heads = w.heads;
      const size_t xn = size_t(k) * heads * 4, yn = size_t(m) * heads * 4;
      auto x = dev.buffer(xn * 4, fast::Memory::Device);
      auto y = dev.buffer(yn * 4, fast::Memory::Host);
      auto scratch = dev.buffer(fast::gemv_scratch_floats(w, 4) * 4, fast::Memory::Device);
      std::vector<float> hx(xn);
      for (auto &v : hx) v = normal(rng);
      dev.upload(*x, hx.data(), hx.size() * 4);
      // {waves, ksw}; {0,0} is the automatic choice.
      std::vector<std::pair<uint32_t, uint32_t>> shapes{{0, 0}};
      if (!quick && w.layout.type != fast::F32)
        for (auto s : {std::pair<uint32_t, uint32_t>{1, 1}, {2, 1}, {4, 1}, {2, 4}, {4, 4}, {4, 16}}) shapes.push_back(s);
      for (int columns : {1, 2, 4}) {
        for (const auto &shape : shapes) {
          if (heads > 1 && shape.second > 1) continue;
          fast::GemvArgs a;
          a.columns = columns;
          a.xs = k * heads;
          a.ys = m * heads;
          a.ksw = shape.first;
          a.kparts = shape.second;
          a.scratch = fast::View(*scratch);
          auto us = query([&](fast::Recorder &r) { fast::gemv(r, w, fast::View(*x), fast::View(*y), a); }, 20);
          // Parity on a sample of rows for every column and head.
          const std::vector<float> out(y->as<float>(), y->as<float>() + yn);
          double err = 0, ref = 0;
          const size_t sample_rows = std::min<size_t>(m, 64);
          for (uint32_t h = 0; h < heads; h += std::max<uint32_t>(1, heads / 4)) {
            for (size_t si = 0; si < sample_rows; ++si) {
              const size_t row = (si * 7919) % m;
              auto wr = dequant_rows(t, size_t(h) * m + row, 1);
              for (int n = 0; n < columns; ++n) {
                double s = 0;
                for (uint32_t i = 0; i < k; ++i) s += double(wr[i]) * hx[size_t(n) * k * heads + size_t(h) * k + i];
                const double g = out[size_t(n) * m * heads + size_t(h) * m + row];
                err += (g - s) * (g - s);
                ref += s * s;
              }
            }
          }
          const double rel = std::sqrt(err / std::max(ref, 1e-30));
          const double med = median(us);
          std::cout << "{\"kind\":\"gemv\",\"tensor\":\"" << name << "\",\"type\":" << w.layout.type << ",\"k\":" << k
                    << ",\"m\":" << m << ",\"heads\":" << heads << ",\"columns\":" << columns
                    << ",\"ksw\":" << shape.first << ",\"kparts\":" << shape.second
                    << ",\"bytes\":" << w.bytes() << ",\"median_us\":" << med
                    << ",\"gbps\":" << w.bytes() / med / 1e3 << ",\"rel_l2\":" << rel << "}\n";
          if (!(rel < 2e-5)) throw std::runtime_error("GEMV parity failed for " + name);
        }
        if (quick) break;
      }
    }

    // ---- resident expert products (repacked Q2_K gate/up, Q3_K down) -----------
    {
      const auto &G = artifact.at("blk.3.ffn_gate_exps.weight"), &U = artifact.at("blk.3.ffn_up_exps.weight"),
                 &D = artifact.at("blk.3.ffn_down_exps.weight");
      const auto gl = fast::layout_for(fast::Q2_K, 4096, 2048), dl = fast::layout_for(fast::Q3_K, 2048, 4096);
      const size_t slot = (2 * gl.head_bytes() + dl.head_bytes() + 255) & ~size_t(255);
      const int experts[2] = {5, 77};
      auto arena = dev.buffer(2 * slot, fast::Memory::Device, true);
      std::vector<uint8_t> packed(slot);
      for (int e = 0; e < 2; ++e) {
        const int id = experts[e];
        fast::repack(gl, G.data() + size_t(id) * G.bytes / 144, packed.data());
        fast::repack(gl, U.data() + size_t(id) * U.bytes / 144, packed.data() + gl.head_bytes());
        fast::repack(dl, D.data() + size_t(id) * D.bytes / 144, packed.data() + 2 * gl.head_bytes());
        dev.upload(*arena, packed.data(), slot, e * slot);
      }
      const uint64_t addresses[2] = {dev.address(*arena), dev.address(*arena) + slot};
      auto table = dev.buffer(16, fast::Memory::Device);
      dev.upload(*table, addresses, 16);
      // Exercise the same union layout and column strides as the engine. The
      // two experts are shared by some tokens and absent from others.
      constexpr uint32_t uofs = 128;
      auto hits = dev.buffer((uofs + 192) * 4, fast::Memory::Host);
      std::vector<float> hx(2 * 4 * 4096);
      for (auto &v : hx) v = normal(rng);
      auto x = dev.buffer(hx.size() * 4, fast::Memory::Device);
      dev.upload(*x, hx.data(), hx.size() * 4);
      auto xs = dev.buffer(hx.size() / 16 * 4, fast::Memory::Device);
      auto y = dev.buffer(2 * 4 * 4096 * 4, fast::Memory::Host);
      for (uint32_t columns : {1u, 2u, 3u, 4u}) {
        std::memset(hits->mapped, 0, hits->size);
        auto *hh = hits->as<uint32_t>();
        for (uint32_t t = 0; t < columns; ++t) {
          hh[32 * t] = t % 2 ? 1 : 2;
          hh[32 * t + 1] = t % 2;
          hh[32 * t + 2] = std::bit_cast<uint32_t>(0.25f + t * 0.125f);
          if (!(t % 2)) {
            hh[32 * t + 3] = 1;
            hh[32 * t + 4] = std::bit_cast<uint32_t>(0.75f - t * 0.125f);
          }
        }
        dev.run([&](fast::Recorder &r) {
          r.dispatch("hits_union", {fast::View(*hits)}, std::array<uint32_t, 3>{0, uofs, columns}, 1);
          r.host_visible();
        });
        if (hh[uofs] != 2 || hh[uofs + 1] != 0 || hh[uofs + 2] != 1)
          throw std::runtime_error("hit union slots failed");
        for (uint32_t e = 0; e < 2; ++e)
          for (uint32_t t = 0; t < 4; ++t) {
            const float expected = t >= columns || (e == 0 && t % 2) ? 0.f :
                e == (t % 2) ? 0.25f + t * 0.125f : 0.75f - t * 0.125f;
            if (std::bit_cast<float>(hh[uofs + 33 + 4 * e + t]) != expected)
              throw std::runtime_error("hit union weights failed");
          }
        for (int down = 0; down < 2; ++down) {
          struct {
            uint32_t K, part_rows, parts, Mp, part_bytes, slot_offset, ksw, side,
                     xofs, xhit, yhit, hofs, columns, xcol, ycol;
          } push = down ? decltype(push){2048, 4096, 1, dl.mp, uint32_t(dl.head_bytes()),
                                         uint32_t(2 * gl.head_bytes()), 4, uint32_t(dl.chunk_bytes()),
                                         0, columns * 2048, columns * 4096, uofs, columns, 2048, 4096}
                        : decltype(push){4096, 2048, 2, gl.mp, uint32_t(gl.head_bytes()), 0, 4,
                                         uint32_t(gl.chunk_bytes()), 0, 0, columns * 4096,
                                         uofs, columns, 4096, 4096};
          const uint32_t k = down ? 2048 : 4096;
          auto us = query([&](fast::Recorder &r) {
            r.dispatch("xsum16", {fast::View(*x), fast::View(*xs)},
                       std::array<uint32_t, 2>{uint32_t(hx.size() / 16), 0}, uint32_t(hx.size() / 16 / 256));
            r.dispatch(down ? "gemv_expert_q3k" : "gemv_expert_q2k",
                       {fast::View(*table), fast::View(*x), fast::View(*y), fast::View(*hits), fast::View(*xs)},
                       push, 4096 / 64, 2);
          }, 20);
          double err = 0, ref = 0;
          for (int e = 0; e < 2; ++e)
            for (int part = 0; part < (down ? 1 : 2); ++part) {
              const auto &w = down ? D : (part ? U : G);
              const size_t rows = down ? 4096 : 2048, rb = fast::row_bytes(int(w.tensor->type), k);
              for (size_t row = 0; row < rows; row += 37) {
                std::vector<float> wr(k);
                ggml_get_type_traits(ggml_type(w.tensor->type))
                    ->to_float(w.data() + size_t(experts[e]) * w.bytes / 144 + row * rb, wr.data(), k);
                for (uint32_t t = 0; t < columns; ++t) {
                  double sum = 0;
                  const size_t xo = (down ? size_t(e) * columns * k : 0) + size_t(t) * k;
                  for (uint32_t i = 0; i < k; ++i) sum += double(wr[i]) * hx[xo + i];
                  if (hh[uofs + 33 + 4 * e + t] == 0u) sum = 0;
                  const double got = y->as<float>()[(size_t(e) * columns + t) * 4096 + part * 2048 + row];
                  err += (got - sum) * (got - sum);
                  ref += sum * sum;
                }
              }
            }
          const double rel = std::sqrt(err / ref), bytes = 2.0 * (down ? dl.head_bytes() : 2 * gl.head_bytes());
          std::cout << "{\"kind\":\"expert\",\"part\":\"" << (down ? "down_q3k" : "gate_up_q2k")
                    << "\",\"hits\":2,\"columns\":" << columns
                    << ",\"median_us\":" << median(us) << ",\"gbps\":" << bytes / median(us) / 1e3
                    << ",\"rel_l2\":" << rel << "}\n";
          if (!(rel < 2e-5)) throw std::runtime_error("expert parity failed");
        }
      }
    }

    // ---- dispatch overhead inside one command buffer -------------------------
    auto flag = dev.buffer(4096, fast::Memory::Host);
    for (bool barriers : {true, false}) {
      auto us = query([&](fast::Recorder &r) {
        if (!barriers) r.begin_concurrent();
        for (int i = 0; i < 100; ++i) r.dispatch("touch", {fast::View(*flag)}, std::array<uint32_t, 2>{uint32_t(i), 64}, 1);
        if (!barriers) r.end_concurrent();
      }, 10);
      std::cout << "{\"kind\":\"dispatch_overhead\",\"barriers\":" << barriers
                << ",\"us_per_dispatch\":" << median(us) / 100 << "}\n";
    }

    // ---- host/GPU handoff latency --------------------------------------------
    {
      // Fence round trip: submit one tiny dispatch and wait.
      std::vector<double> fence;
      for (int i = 0; i < 200; ++i) {
        auto c = dev.allocate_command();
        dev.begin(c);
        fast::Recorder r(dev, c);
        r.dispatch("touch", {fast::View(*flag)}, std::array<uint32_t, 2>{uint32_t(i), 1}, 1);
        dev.end(c);
        const auto t0 = Clock::now();
        dev.submit_wait(c);
        fence.push_back(since(t0));
        dev.free_command(c);
      }
      std::cout << "{\"kind\":\"handoff\",\"mode\":\"submit_fence_roundtrip\",\"median_us\":" << median(fence) << "}\n";
      // Pre-submitted wait-before-signal: GPU waits on host value v, runs a tiny
      // dispatch, signals v+1. Host signals and then waits (block) or spins.
      for (int spin = 0; spin < 2; ++spin) {
        auto sem = dev.timeline(0);
        const int rounds = 200;
        std::vector<VkCommandBuffer> cmds;
        for (int i = 0; i < rounds; ++i) {
          auto c = dev.allocate_command();
          dev.begin(c);
          fast::Recorder r(dev, c);
          r.dispatch("touch", {fast::View(*flag)}, std::array<uint32_t, 2>{uint32_t(i + 1), 1}, 1);
          r.host_visible();
          dev.end(c);
          const fast::Wait w{sem, uint64_t(2 * i + 1)}, s{sem, uint64_t(2 * i + 2)};
          dev.submit(c, std::span(&w, 1), std::span(&s, 1));
          cmds.push_back(c);
        }
        std::vector<double> lat, flag_lat;
        for (int i = 0; i < rounds; ++i) {
          const auto t0 = Clock::now();
          dev.signal(sem, uint64_t(2 * i + 1));
          if (spin) {
            // Poll the GPU-written word, then the semaphore value.
            volatile uint32_t *f = flag->as<uint32_t>();
            while (*f != uint32_t(i + 1)) {}
            flag_lat.push_back(since(t0));
            while (dev.value(sem) < uint64_t(2 * i + 2)) {}
          } else if (!dev.wait(sem, uint64_t(2 * i + 2))) throw std::runtime_error("handoff timeout");
          lat.push_back(since(t0));
        }
        dev.wait_idle();
        for (auto c : cmds) dev.free_command(c);
        dev.destroy(sem);
        std::cout << "{\"kind\":\"handoff\",\"mode\":\"" << (spin ? "host_signal_gpu_run_spin" : "host_signal_gpu_run_block")
                  << "\",\"median_us\":" << median(lat);
        if (spin) std::cout << ",\"flag_visible_median_us\":" << median(flag_lat);
        std::cout << "}\n";
      }
    }

    // ---- upload bandwidth and concurrency -------------------------------------
    {
      const size_t bytes = 256ULL << 20;
      auto host = dev.buffer(bytes, fast::Memory::Host);
      auto device = dev.buffer(bytes, fast::Memory::Device);
      std::memset(host->mapped, 0x3c, bytes);
      for (int upload_queue = 0; upload_queue < 2; ++upload_queue) {
        std::vector<double> ms;
        for (int i = 0; i < 5; ++i) {
          auto c = dev.allocate_command(upload_queue);
          dev.begin(c);
          fast::Recorder r(dev, c);
          r.copy(fast::View(*host), fast::View(*device), bytes);
          dev.end(c);
          const auto t0 = Clock::now();
          dev.submit_wait(c, upload_queue);
          ms.push_back(since(t0) / 1000);
          dev.free_command(c, upload_queue);
        }
        std::cout << "{\"kind\":\"upload\",\"queue\":\"" << (upload_queue ? "second" : "main")
                  << "\",\"bytes\":" << bytes << ",\"gbps\":" << bytes / median(ms) / 1e6 << "}\n";
      }
      // CPU write into host-visible VRAM (large BAR) and system memory.
      auto shared = dev.buffer(64ULL << 20, fast::Memory::Shared);
      std::vector<char> src(64ULL << 20, 1);
      for (auto *b : {shared.get(), host.get()}) {
        std::vector<double> ms;
        for (int i = 0; i < 5; ++i) {
          const auto t0 = Clock::now();
          std::memcpy(b->mapped, src.data(), src.size());
          ms.push_back(since(t0) / 1000);
        }
        std::cout << "{\"kind\":\"cpu_write\",\"device_local\":"
                  << bool(b->flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                  << ",\"gbps\":" << src.size() / median(ms) / 1e6 << "}\n";
      }
    }

    // ---- host RAM read bandwidth -------------------------------------------
    {
      const size_t bytes = 4ULL << 30;
      std::vector<uint64_t> data(bytes / 8, 1);
      for (int threads : {8, 12, 16}) {
        std::vector<double> gbps;
        for (int rep = 0; rep < 3; ++rep) {
          std::atomic<uint64_t> sink{0};
          std::vector<std::thread> pool;
          const auto t0 = Clock::now();
          for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
              const size_t n = data.size() / threads, b = t * n;
              uint64_t s = 0;
              for (size_t i = b; i < b + n; i += 8)
                s += data[i] + data[i + 1] + data[i + 2] + data[i + 3] + data[i + 4] + data[i + 5] + data[i + 6] + data[i + 7];
              sink += s;
            });
          for (auto &t : pool) t.join();
          gbps.push_back(bytes / since(t0) / 1e3);
        }
        std::cout << "{\"kind\":\"ram_read\",\"threads\":" << threads << ",\"gbps\":" << median(gbps) << "}\n";
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "VULKAN_FAST_BENCH_ERROR " << e.what() << '\n';
    return 1;
  }
}
