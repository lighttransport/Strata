#include "fast/cpu_prefill.hpp"
#include "strata/kernels/canon_expert.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PREFILL_AVX2 1
#endif
namespace strata::vulkan::fast {
namespace {
void decode_q8(const void *input, float *output, int n) {
  // Q8_K is an activation-only format; ggml's generic to_float slot is null.
  const auto *bytes = static_cast<const uint8_t *>(input);
  for (int block = 0; block < n / 256; ++block) {
    float scale; std::memcpy(&scale, bytes + size_t(block) * 292, 4);
    const auto *codes = reinterpret_cast<const int8_t *>(bytes + size_t(block) * 292 + 4);
    for (int k = 0; k < 256; ++k) output[block * 256 + k] = scale * float(codes[k]);
  }
}
// X and Y are [K][N] / [M][N]. Blocking keeps the active rows and token
// panel in L1 while expanded weights are reused across all tokens of an expert.
#ifdef PREFILL_AVX2
__attribute__((target("avx2,fma")))
void product(const float *w, const float *x, float *y, int m, int k, int n) {
  std::fill(y, y + size_t(m) * n, 0.f);
  for (int row0 = 0; row0 < m; row0 += 32)
    for (int k0 = 0; k0 < k; k0 += 64)
      for (int row = row0; row < std::min(m, row0 + 32); row += 2)
        for (int col = 0; col < n; col += 32) {
          __m256 a[4], b[4];
          for (int j = 0; j < 4; ++j) {
            a[j] = _mm256_loadu_ps(y + size_t(row) * n + col + 8*j);
            b[j] = _mm256_loadu_ps(y + size_t(row+1) * n + col + 8*j);
          }
          for (int kk = k0; kk < std::min(k, k0+64); ++kk) {
            const auto wa = _mm256_set1_ps(w[size_t(row)*k+kk]);
            const auto wb = _mm256_set1_ps(w[size_t(row+1)*k+kk]);
            for (int j = 0; j < 4; ++j) {
              const auto value = _mm256_loadu_ps(x + size_t(kk)*n + col + 8*j);
              a[j] = _mm256_fmadd_ps(wa, value, a[j]); b[j] = _mm256_fmadd_ps(wb, value, b[j]);
            }
          }
          for (int j = 0; j < 4; ++j) {
            _mm256_storeu_ps(y + size_t(row)*n + col + 8*j, a[j]);
            _mm256_storeu_ps(y + size_t(row+1)*n + col + 8*j, b[j]);
          }
        }
}
#endif
struct Group {
  const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
  std::vector<const void *> inputs;
  std::vector<float *> outputs;
};
struct Scratch {
  std::vector<float> gu, down, x, h, output, gates, row;
  std::vector<uint8_t> quantized;
  void execute(const kernels::cpu::NativeFmt &f, const Group &g, int panel) {
#ifdef PREFILL_AVX2
    const int H = int(f.n_embd), F = int(f.n_ff);
    gu.resize(size_t(2)*F*H); down.resize(size_t(H)*F);
    const auto *q2 = ggml_get_type_traits(GGML_TYPE_Q2_K), *q3 = ggml_get_type_traits(GGML_TYPE_Q3_K);
    q2->to_float(g.gate, gu.data(), int64_t(F)*H);
    q2->to_float(g.up, gu.data()+size_t(F)*H, int64_t(F)*H);
    q3->to_float(g.down, down.data(), int64_t(H)*F);
    x.resize(size_t(H)*panel); gates.resize(size_t(2)*F*panel);
    h.resize(size_t(F)*panel); output.resize(size_t(H)*panel); row.resize(H); quantized.resize(f.h_bytes);
    for (size_t first = 0; first < g.inputs.size(); first += panel) {
      const int n = int(std::min<size_t>(panel, g.inputs.size()-first));
      std::fill(x.begin(), x.end(), 0.f);
      for (int t = 0; t < n; ++t) {
        decode_q8(g.inputs[first+t], row.data(), H);
        for (int k = 0; k < H; ++k) x[size_t(k)*panel+t] = row[k];
      }
      product(gu.data(), x.data(), gates.data(), 2*F, H, panel);
      std::fill(h.begin(), h.end(), 0.f);
      for (int t = 0; t < n; ++t) {
        for (int k = 0; k < F; ++k)
          row[k] = kernels::cpu::canon_swiglu(gates[size_t(k)*panel+t], gates[size_t(k+F)*panel+t], f.swiglu_limit, f.gate_skip);
        kernels::cpu::native_quant_h(f, row.data(), quantized.data());
        decode_q8(quantized.data(), row.data(), F);
        for (int k = 0; k < F; ++k) h[size_t(k)*panel+t] = row[k];
      }
      product(down.data(), h.data(), output.data(), H, F, panel);
      for (int t = 0; t < n; ++t)
        for (int k = 0; k < H; ++k) g.outputs[first+t][k] = output[size_t(k)*panel+t];
    }
#else
    throw std::runtime_error("CPU prefill panels require x86 AVX2/FMA");
#endif
  }
};
}
struct CpuPrefill::Impl {
  int panel;
  std::mutex mutex;
  std::condition_variable_any ready;
  std::condition_variable done;
  std::vector<Scratch> scratch;
  std::vector<std::jthread> workers;
  std::atomic<size_t> next{0};
  size_t epoch = 0, remaining = 0;
  std::function<void(size_t,size_t)> work;
  size_t count = 0;
  std::exception_ptr error;
  Impl(int threads, int width) : panel(width), scratch(size_t(threads)) {
    for (int i = 0; i < threads; ++i) workers.emplace_back([this,i](std::stop_token stop) {
      size_t seen = 0;
      for (;;) {
        {std::unique_lock lock(mutex);
         if (!ready.wait(lock, stop, [&] {return epoch != seen;})) return;
         seen = epoch;}
        try {
          for (size_t j = next.fetch_add(1); j < count; j = next.fetch_add(1)) work(size_t(i), j);
        } catch (...) {
          std::lock_guard lock(mutex); if (!error) error = std::current_exception();
        }
        {std::lock_guard lock(mutex); if (--remaining == 0) done.notify_one();}
      }
    });
  }
  ~Impl() {
    for (auto &worker : workers) worker.request_stop();
    ready.notify_all();
    // Join while the mutex, work and condition variables are still alive.
    workers.clear();
  }
};
CpuPrefill::CpuPrefill(int workers, int panel) {
  if (workers < 1 || workers > 64 || (panel != 32 && panel != 64)) throw std::invalid_argument("CPU prefill panel configuration invalid");
#ifdef PREFILL_AVX2
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) throw std::invalid_argument("CPU prefill requires AVX2 and FMA");
#else
  throw std::invalid_argument("CPU prefill requires x86 AVX2/FMA build");
#endif
  impl_ = std::make_unique<Impl>(workers, panel);
}
CpuPrefill::~CpuPrefill() = default;
void CpuPrefill::run(const kernels::cpu::NativeFmt &f, std::span<const kernels::cpu::ExpertJobMulti> jobs) {
  if (!f.canon || f.gu_type != GGML_TYPE_Q2_K || f.d_type != GGML_TYPE_Q3_K ||
      f.gu_act != GGML_TYPE_Q8_K || f.d_act != GGML_TYPE_Q8_K || f.n_embd != 4096 || f.n_ff != 2048 ||
      f.observer || f.hidden_transform || f.q23_layout)
    throw std::invalid_argument("CPU panels require canonical native Q2/Q3 geometry without transforms");
  std::array<Group,144> by_expert;
  for (const auto &job : jobs) {
    if (job.expert_id < 0 || job.expert_id >= 144 || job.nt < 1 || job.nt > kernels::cpu::MAXT)
      throw std::invalid_argument("CPU panel job invalid");
    auto &g = by_expert[job.expert_id];
    if (g.gate && (g.gate != job.blob || g.up != job.native_up || g.down != job.native_down))
      throw std::invalid_argument("inconsistent expert weights");
    g.gate=job.blob; g.up=job.native_up; g.down=job.native_down;
    for (int t=0;t<job.nt;++t) {g.inputs.push_back(job.nact[t]);g.outputs.push_back(job.out[t]);}
  }
  std::vector<const Group *> groups;
  for (const auto &g : by_expert) if (!g.inputs.empty()) groups.push_back(&g);
  if (groups.empty()) return;
  auto &p = *impl_;
  std::unique_lock lock(p.mutex);
  p.work = [&](size_t worker, size_t group) {p.scratch[worker].execute(f,*groups[group],p.panel);};
  p.count=groups.size(); p.next=0; p.error=nullptr; p.remaining=p.workers.size(); ++p.epoch;
  p.ready.notify_all(); p.done.wait(lock,[&] {return p.remaining==0;});
  p.work={}; if (p.error) std::rethrow_exception(p.error);
}
}
