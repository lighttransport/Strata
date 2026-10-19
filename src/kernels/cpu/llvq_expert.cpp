// src/kernels/cpu/llvq_expert.cpp - llvq Tetra experts on the CPU expert lane (include/strata/kernels/cpu/llvq_expert.hpp).
// Built with -mavx2 -mfma: gemv_avx2 of include/strata/llvq/tetra_decode.hpp is the row kernel. The rotation runs in
// float (the GPU's llvq_tetra_prepare does too; the reference llvq::Rotation is double).
#include "strata/kernels/cpu/llvq_expert.hpp"

#include "strata/llvq/tetra_decode.hpp"

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::kernels::cpu {
namespace {

struct LlvqCpu {
    uint64_t seed = 0;
    std::unique_ptr<llvq::Tetra> tetra;
    llvq::DecodeTables tables{};
    std::vector<float> signs;   // the first n of them are Q_n's signs (llvq::Rotation draws them in order)
};
LlvqCpu* g_llvq = nullptr;
std::mutex g_llvq_mutex;

const LlvqCpu& state() {
    if (!g_llvq) throw std::logic_error("llvq experts: llvq_expert_init was not called");
    return *g_llvq;
}

// v <- (1/sqrt n) H v in place, n a power of two
void wht(float* v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; ++j) {
                const float a = v[j], b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
    const float s = 1.0f / std::sqrt(float(n));
    for (int i = 0; i < n; ++i) v[i] *= s;
}

}  // namespace

void llvq_expert_init(uint64_t seed) {
    std::lock_guard lock(g_llvq_mutex);
    if (g_llvq) {
        if (g_llvq->seed != seed) throw std::invalid_argument("llvq experts: units with different rotation seeds");
        return;
    }
#if !(defined(__AVX2__) && defined(__FMA__))
    throw std::runtime_error("llvq experts: this build has no AVX2+FMA kernel");
#else
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
        throw std::runtime_error("llvq experts: the CPU kernels need AVX2 and FMA");
    auto s = std::make_unique<LlvqCpu>();
    s->seed = seed;
    s->tetra = std::make_unique<llvq::Tetra>();
    s->tables = llvq::DecodeTables::build(*s->tetra);
    const llvq::Rotation rot(4096, seed);   // 4096 = 2^12, so no Gram-Schmidt mix (k = 1); Q_2048's signs are the first 2048
    s->signs.assign(rot.signs().begin(), rot.signs().end());
    g_llvq = s.release();
#endif
}

bool llvq_expert_ready() noexcept { return g_llvq != nullptr; }

size_t llvq_record_bytes(int64_t rows, int64_t cols) noexcept {
    const int64_t nb = cols / 24, tail = cols - 24 * nb, stride = (nb * 6 + 7) / 8 * 8;
    return size_t(24 + 4 * rows + stride * rows + 2 * rows * tail);
}

void llvq_prepare(const float* x, int n, float* xp) {
    const LlvqCpu& s = state();
    if (n <= 0 || n & (n - 1) || size_t(n) > s.signs.size()) throw std::invalid_argument("llvq experts: activation width");
    thread_local std::vector<float> v;
    v.resize(size_t(n));
    for (int i = 0; i < n; ++i) v[size_t(i)] = x[i] * s.signs[size_t(i)];
    wht(v.data(), n);
    const int nb = n / 24;
    for (int b = 0; b < nb; ++b)
        for (int t = 0; t < llvq::DIM; ++t) xp[24 * b + t] = v[size_t(24 * b + int(s.tables.order[t]))];
    for (int c = 24 * nb; c < n; ++c) xp[c] = v[size_t(c)];
}

#if defined(__AVX2__) && defined(__FMA__)
namespace {
// gemv_avx2 wants the tokens' activations contiguous (ntok x cols) and writes y[r * ntok + k]
const float* gather(const void* const* act, int nt, uint32_t cols) {
    if (nt == 1) return static_cast<const float*>(act[0]);
    thread_local std::vector<float> xs;
    xs.resize(size_t(nt) * cols);
    for (int t = 0; t < nt; ++t) std::memcpy(xs.data() + size_t(t) * cols, act[t], size_t(cols) * 4);
    return xs.data();
}
}  // namespace

void llvq_gu_rows(const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff, int r0, int r1,
                  float swiglu_limit) {
    const LlvqCpu& s = state();
    if (nt < 1 || nt > 8) throw std::invalid_argument("llvq experts: 1..8 tokens");
    llvq::TetraMatrix g, u;
    g.parse(gate);
    u.parse(up);
    const float* xp = gather(act, nt, g.cols);
    thread_local std::vector<float> yg, yu;   // the kernel writes y[r * nt + k] by absolute row
    const size_t n = size_t(r1) * size_t(nt);
    if (yg.size() < n) { yg.resize(n); yu.resize(n); }
    llvq::gemv_avx2(s.tables, g, xp, nt, uint32_t(r0), uint32_t(r1), yg.data(), true);
    llvq::gemv_avx2(s.tables, u, xp, nt, uint32_t(r0), uint32_t(r1), yu.data(), true);
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) {
            float gv = yg[size_t(r) * size_t(nt) + size_t(t)], uv = yu[size_t(r) * size_t(nt) + size_t(t)];
            if (swiglu_limit > 0) {
                gv = std::fmin(gv, swiglu_limit);
                uv = std::fmax(-swiglu_limit, std::fmin(uv, swiglu_limit));
            }
            ff[t][r] = (gv / (1.f + std::exp(-gv))) * uv;
        }
}

void llvq_down_rows(const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    const LlvqCpu& s = state();
    if (nt < 1 || nt > 8) throw std::invalid_argument("llvq experts: 1..8 tokens");
    llvq::TetraMatrix d;
    d.parse(down);
    const float* xp = gather(hq, nt, d.cols);
    thread_local std::vector<float> y;   // y[r * nt + k] by absolute row
    if (y.size() < size_t(r1) * size_t(nt)) y.resize(size_t(r1) * size_t(nt));
    llvq::gemv_avx2(s.tables, d, xp, nt, uint32_t(r0), uint32_t(r1), y.data(), true);
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) out[t][r] = y[size_t(r) * size_t(nt) + size_t(t)];
}
#else
void llvq_gu_rows(const uint8_t*, const uint8_t*, const void* const*, int, float* const*, int, int, float) {
    throw std::runtime_error("llvq experts: this build has no AVX2+FMA kernel");
}
void llvq_down_rows(const uint8_t*, const void* const*, int, float* const*, int, int) {
    throw std::runtime_error("llvq experts: this build has no AVX2+FMA kernel");
}
#endif

}  // namespace strata::kernels::cpu
