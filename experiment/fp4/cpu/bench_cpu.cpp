// MXFP4 GEMV on CPU (AVX2/FMA, tuned for Zen1 / Threadripper 1950X) with MiMo-V2.6-Flash expert shapes.
//
// Kernels (all read the checkpoint layout directly, see common/mxfp4.h):
//   c0_scalar    scalar fp32 dequant, natural order               (correctness reference only)
//   c1_f32       pshufb -> int8 -> fp32, FMA with fp32 activations (no activation quantization)
//   c2_q8_sign   ggml-style: pshufb LUT, sign trick + vpmaddubsw, q8 activations, per-block cvt+FMA
//   c3_q8_off_rR offset LUT (w+12 >= 0) so vpmaddubsw needs no sign ops; the 12*sum(y) term is a per-token
//                constant per block; R rows share each activation load
//   c3h_q8_off   c3 with an 8-block hadd tree: one cvt + one FMA per 8 blocks instead of per block
//   c4_q8_colsN  N activation columns per weight load (MTP verify / small batch)
//   moe          full MoE layer: q8(x) -> 8x gate/up -> SwiGLU -> q8(h) -> 8x down -> weighted sum
#include "../common/mxfp4.h"
#include "../common/report.h"
#include "../common/weights.h"

#include <immintrin.h>
#include <omp.h>

#include "kq_cpu.h"

#include <cstdlib>
#include <array>
#include <functional>
#include <sstream>

using namespace fp4;

namespace {

float g_e8[256];  // 2^(e-127) as float (e=0 -> 0, treated as zero scale like ggml's e8m0 path never sees it)

void init_tables() {
    for (int e = 0; e < 256; ++e) {
        uint32_t bits = uint32_t(e) << 23;
        float f;
        std::memcpy(&f, &bits, 4);
        g_e8[e] = f;
    }
}

void* amalloc(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, (n + 63) & ~size_t(63))) abort();
    return p;
}

// ---- activation (one token) prepared for every kernel ----------------------------------------------
struct ActCpu {
    int n = 0, nb = 0;
    std::vector<float> xf, xl;        // natural, logical order
    int8_t* qs = nullptr;              // q8 logical, 32B aligned
    float* dh = nullptr;               // 0.5 * d (the 0.5 undoes the x2 in the weight LUT)
    float* corr = nullptr;             // 12 * dh * sum(qs), for the offset LUT
    ActCpu() = default;
    ActCpu(const ActCpu&) = delete;
    ~ActCpu() { free(qs); free(dh); free(corr); }
    void set(const float* x, int n_) {
        if (n_ != n) {
            free(qs); free(dh); free(corr);
            n = n_; nb = n / QK;
            qs = (int8_t*)amalloc(n);
            dh = (float*)amalloc(nb * 4);
            corr = (float*)amalloc(nb * 4);
            xf.resize(n); xl.resize(n);
        }
        std::memcpy(xf.data(), x, n * 4);
        permute_logical(x, n, xl.data());
        std::vector<Q8Block> q(nb);
        quantize_q8_logical(x, n, q.data());
        for (int b = 0; b < nb; ++b) {
            std::memcpy(qs + b * QK, q[b].qs, QK);
            dh[b] = 0.5f * q[b].d;
            corr[b] = 12.f * dh[b] * float(q[b].sum);
        }
    }
};

// Fast q8 quantization of one vector inside the timed MoE path (same math as quantize_q8_logical).
void quantize_act_fast(const float* x, int n, ActCpu& a) {
    a.set(x, n);  // n <= 4096 per token; a few us, measured as part of the moe op
}

inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline __m256i nibbles_lohi(const uint8_t* q) {  // 16 packed bytes -> [lo nibbles | hi nibbles]
    const __m128i m4 = _mm_set1_epi8(15);
    __m128i v = _mm_loadu_si128((const __m128i*)q);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
}

const __m256i& lut_signed() {
    static const __m256i t = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)kE2M1x2));
    return t;
}
const __m256i& lut_offset() {  // kE2M1x2 + 12, all in [0, 24]
    static __m256i t = [] {
        int8_t v[16];
        for (int i = 0; i < 16; ++i) v[i] = kE2M1x2[i] + 12;
        return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)v));
    }();
    return t;
}

// ---- kernels: one call computes `nrows` consecutive rows ------------------------------------------
using RowFn = void (*)(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows);

void c0_scalar(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* qr = q + size_t(r) * cols / 2;
        const uint8_t* er = e + size_t(r) * cols / QK;
        float s = 0;
        for (int b = 0; b < cols / QK; ++b) {
            float bs = 0;
            for (int i = 0; i < QK; ++i) {
                uint8_t byte = qr[b * 16 + i / 2];
                bs += kE2M1[(i & 1) ? byte >> 4 : byte & 15] * a.xf[b * QK + i];
            }
            s += bs * e8m0_to_float(er[b]);
        }
        out[r] = s;
    }
}

void c1_f32(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows) {
    const int nb = cols / QK;
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* qr = q + size_t(r) * cols / 2;
        const uint8_t* er = e + size_t(r) * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m256i w = _mm256_shuffle_epi8(lut_signed(), nibbles_lohi(qr + b * 16));
            const float* x = a.xl.data() + b * QK;
            __m128i w0 = _mm256_castsi256_si128(w), w1 = _mm256_extracti128_si256(w, 1);
            __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)), _mm256_loadu_ps(x));
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8))), _mm256_loadu_ps(x + 8), s);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)), _mm256_loadu_ps(x + 16), s);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8))), _mm256_loadu_ps(x + 24), s);
            acc = _mm256_fmadd_ps(s, _mm256_set1_ps(0.5f * g_e8[er[b]]), acc);
        }
        out[r] = hsum8(acc);
    }
}

void c2_q8_sign(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows) {
    const int nb = cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* qr = q + size_t(r) * cols / 2;
        const uint8_t* er = e + size_t(r) * nb;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int b = 0; b < nb; b += 2) {
            __m256i w0 = _mm256_shuffle_epi8(lut_signed(), nibbles_lohi(qr + b * 16));
            __m256i w1 = _mm256_shuffle_epi8(lut_signed(), nibbles_lohi(qr + b * 16 + 16));
            __m256i y0 = _mm256_load_si256((const __m256i*)(a.qs + b * QK));
            __m256i y1 = _mm256_load_si256((const __m256i*)(a.qs + b * QK + QK));
            __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w0, w0), _mm256_sign_epi8(y0, w0)), ones);
            __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w1, w1), _mm256_sign_epi8(y1, w1)), ones);
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(g_e8[er[b]] * a.dh[b]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(g_e8[er[b + 1]] * a.dh[b + 1]), acc1);
        }
        out[r] = hsum8(_mm256_add_ps(acc0, acc1));
    }
}

template <int R>
void c3_q8_off(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows) {
    const int nb = cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    int r = 0;
    for (; r + R <= nrows; r += R) {
        __m256 acc[R];
        float corr[R];
        for (int k = 0; k < R; ++k) { acc[k] = _mm256_setzero_ps(); corr[k] = 0; }
        for (int b = 0; b < nb; ++b) {
            __m256i y = _mm256_load_si256((const __m256i*)(a.qs + b * QK));
            for (int k = 0; k < R; ++k) {
                const uint8_t* qr = q + size_t(r + k) * cols / 2;
                float sw = g_e8[e[size_t(r + k) * nb + b]];
                __m256i w = _mm256_shuffle_epi8(lut_offset(), nibbles_lohi(qr + b * 16));
                __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, y), ones);
                acc[k] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * a.dh[b]), acc[k]);
                corr[k] += sw * a.corr[b];
            }
        }
        for (int k = 0; k < R; ++k) out[r + k] = hsum8(acc[k]) - corr[k];
    }
    if (r < nrows) c3_q8_off<1>(q + size_t(r) * cols / 2, e + size_t(r) * nb, cols, a, out + r, nrows - r);
}

inline __m256 sum8_epi32(__m256i p0, __m256i p1, __m256i p2, __m256i p3, __m256i p4, __m256i p5, __m256i p6,
                         __m256i p7) {
    __m256i h0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
    __m256i h4567 = _mm256_hadd_epi32(_mm256_hadd_epi32(p4, p5), _mm256_hadd_epi32(p6, p7));
    __m256i s = _mm256_add_epi32(_mm256_permute2x128_si256(h0123, h4567, 0x20),
                                 _mm256_permute2x128_si256(h0123, h4567, 0x31));
    return _mm256_cvtepi32_ps(s);
}

void c3h_q8_off(const uint8_t* q, const uint8_t* e, int cols, const ActCpu& a, float* out, int nrows) {
    const int nb = cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* qr = q + size_t(r) * cols / 2;
        const uint8_t* er = e + size_t(r) * nb;
        __m256 acc = _mm256_setzero_ps(), cacc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b += 8) {
            __m256i p[8];
            for (int k = 0; k < 8; ++k) {
                __m256i w = _mm256_shuffle_epi8(lut_offset(), nibbles_lohi(qr + (b + k) * 16));
                __m256i y = _mm256_load_si256((const __m256i*)(a.qs + (b + k) * QK));
                p[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(w, y), ones);
            }
            __m256 s = sum8_epi32(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
            __m128i e8 = _mm_loadl_epi64((const __m128i*)(er + b));
            __m256 sw = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(e8), 23));
            acc = _mm256_fmadd_ps(s, _mm256_mul_ps(sw, _mm256_loadu_ps(a.dh + b)), acc);
            cacc = _mm256_fmadd_ps(sw, _mm256_loadu_ps(a.corr + b), cacc);
        }
        out[r] = hsum8(_mm256_sub_ps(acc, cacc));
    }
}

// N activation columns; out is [nrows][N].
template <int N>
void c4_q8_cols(const uint8_t* q, const uint8_t* e, int cols, const ActCpu* a, float* out, int nrows) {
    const int nb = cols / QK;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* qr = q + size_t(r) * cols / 2;
        const uint8_t* er = e + size_t(r) * nb;
        __m256 acc[N];
        float corr[N];
        for (int n = 0; n < N; ++n) { acc[n] = _mm256_setzero_ps(); corr[n] = 0; }
        for (int b = 0; b < nb; ++b) {
            __m256i w = _mm256_shuffle_epi8(lut_offset(), nibbles_lohi(qr + b * 16));
            float sw = g_e8[er[b]];
            for (int n = 0; n < N; ++n) {
                __m256i y = _mm256_load_si256((const __m256i*)(a[n].qs + b * QK));
                __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, y), ones);
                acc[n] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * a[n].dh[b]), acc[n]);
                corr[n] += sw * a[n].corr[b];
            }
        }
        for (int n = 0; n < N; ++n) out[size_t(r) * N + n] = hsum8(acc[n]) - corr[n];
    }
}

// ---- NUMA-aware expert buffers ----------------------------------------------------------------------
// Rows are first-touched with the same static row partition the kernels use, so each thread's rows are
// on its own node. Rebuilt for every thread count.
struct Mat { uint8_t* q = nullptr; uint8_t* e = nullptr; int rows = 0, cols = 0; };
struct Set { Mat gu[mimo::kTopK], dn[mimo::kTopK]; float route[mimo::kTopK]; };

void place(const MxView& src, Mat& m, int chunk) {
    m.rows = src.rows; m.cols = src.cols;
    m.q = (uint8_t*)amalloc(src.qbytes());
    m.e = (uint8_t*)amalloc(src.ebytes());
    const size_t qr = src.cols / 2, er = src.cols / QK;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < src.rows / chunk; ++c) {
        std::memcpy(m.q + c * chunk * qr, src.q + c * chunk * qr, chunk * qr);
        std::memcpy(m.e + c * chunk * er, src.e + c * chunk * er, chunk * er);
    }
}
void unplace(Mat& m) { free(m.q); free(m.e); m.q = m.e = nullptr; }

constexpr int kChunk = 16;  // rows per scheduling unit (multiple of every R)

// Parallel GEMV over all top-k matrices of a set: out[k][rows].
void gemv_set(Mat* mats, const ActCpu& a, float* out, RowFn fn) {
    const int rows = mats[0].rows;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < rows / kChunk; ++c)
        for (int k = 0; k < mimo::kTopK; ++k) {
            const Mat& m = mats[k];
            fn(m.q + size_t(c) * kChunk * m.cols / 2, m.e + size_t(c) * kChunk * m.cols / QK, m.cols, a,
               out + size_t(k) * rows + c * kChunk, kChunk);
        }
}

template <int N>
void gemv_set_cols(Mat* mats, const ActCpu* a, float* out) {
    const int rows = mats[0].rows;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < rows / kChunk; ++c)
        for (int k = 0; k < mimo::kTopK; ++k) {
            const Mat& m = mats[k];
            c4_q8_cols<N>(m.q + size_t(c) * kChunk * m.cols / 2, m.e + size_t(c) * kChunk * m.cols / QK, m.cols, a,
                          out + (size_t(k) * rows + c * kChunk) * N, kChunk);
        }
}

// Full MoE layer for one token.
struct MoeScratch { std::vector<float> gu, h, y; ActCpu ax, ah[mimo::kTopK]; };
void moe_layer(Set& s, const float* x, MoeScratch& w, RowFn fn) {
    const int I = mimo::kExpertInter, H = mimo::kHidden;
    quantize_act_fast(x, H, w.ax);
    gemv_set(s.gu, w.ax, w.gu.data(), fn);
#pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < mimo::kTopK; ++k) {
        float* g = w.gu.data() + size_t(k) * 2 * I;
        for (int i = 0; i < I; ++i) w.h[size_t(k) * I + i] = g[i] / (1.f + std::exp(-g[i])) * g[I + i];
        quantize_act_fast(w.h.data() + size_t(k) * I, I, w.ah[k]);
    }
#pragma omp parallel for schedule(static)
    for (int c = 0; c < H / kChunk; ++c) {
        float tmp[kChunk];
        float acc[kChunk] = {};
        for (int k = 0; k < mimo::kTopK; ++k) {
            const Mat& m = s.dn[k];
            fn(m.q + size_t(c) * kChunk * m.cols / 2, m.e + size_t(c) * kChunk * m.cols / QK, m.cols, w.ah[k], tmp, kChunk);
            for (int i = 0; i < kChunk; ++i) acc[i] += s.route[k] * tmp[i];
        }
        std::memcpy(w.y.data() + c * kChunk, acc, sizeof acc);
    }
}

double read_bandwidth_gbps(size_t bytes, int iters) {
    const size_t n = bytes / 32;
    __m256i* buf = (__m256i*)amalloc(bytes);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) buf[i] = _mm256_set1_epi32(int(i));
    std::vector<double> t;
    volatile int sink = 0;
    for (int it = 0; it < iters; ++it) {
        double t0 = now_s();
        int tot = 0;
#pragma omp parallel for schedule(static) reduction(+ : tot)
        for (size_t c = 0; c < n / 64; ++c) {
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            const __m256i* p = buf + c * 64;
            for (int i = 0; i < 64; i += 4) {
                a0 = _mm256_add_epi32(a0, _mm256_load_si256(p + i));
                a1 = _mm256_add_epi32(a1, _mm256_load_si256(p + i + 1));
                a2 = _mm256_add_epi32(a2, _mm256_load_si256(p + i + 2));
                a3 = _mm256_add_epi32(a3, _mm256_load_si256(p + i + 3));
            }
            __m256i s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
            tot += _mm256_extract_epi32(s, 0) + _mm256_extract_epi32(s, 7);
        }
        t.push_back(now_s() - t0);
        sink += tot;
    }
    free(buf);
    (void)sink;
    return bytes / stats_of(t).med / 1e9;
}

// ---- driver -------------------------------------------------------------------------------------------
struct Opts {
    std::string weights = "dummy", model = "/mnt/nvme01/models/m26f", csv;
    int layer = 1, sets = 4, iters = 20, warmup = 3;
    std::vector<int> threads = {1, 2, 4, 8, 16, 32};
    std::vector<int> ncols = {1, 2, 4, 8};
    bool selftest = false, only_moe = false, kquant = true, mx = true;
};

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

struct Kernel { const char* name; RowFn fn; };
const Kernel kKernels[] = {
    {"c1_f32", c1_f32},           {"c2_q8_sign", c2_q8_sign},       {"c3_q8_off_r1", c3_q8_off<1>},
    {"c3_q8_off_r2", c3_q8_off<2>}, {"c3_q8_off_r4", c3_q8_off<4>}, {"c3h_q8_off", c3h_q8_off},
};

int selftest(const std::vector<Expert>& experts) {
    printf("== selftest (fp64 reference, activations N(0,1) with outliers) ==\n");
    int fails = 0;
    for (int which = 0; which < 2; ++which) {
        const MxView W = which == 0 ? experts[0].gu() : experts[0].dn();
        std::vector<float> x = make_activation(W.cols, 7);
        ActCpu a;
        a.set(x.data(), W.cols);
        std::vector<double> ref(W.rows);
        ref_gemv(W, x.data(), ref.data());
        std::vector<float> y(W.rows);
        auto run = [&](const char* name, RowFn fn, double tol) {
            fn(W.q, W.e, W.cols, a, y.data(), W.rows);
            ErrStats e = compare(y.data(), ref.data(), W.rows);
            bool ok = e.finite && e.rel_l2 < tol;
            fails += !ok;
            printf("  %-14s %-4s relL2=%.3e maxabs=%.3e %s\n", name, which ? "down" : "gu", e.rel_l2, e.max_abs,
                   ok ? "ok" : "FAIL");
        };
        run("c0_scalar", c0_scalar, 1e-5);
        for (const Kernel& k : kKernels) run(k.name, k.fn, std::string(k.name) == "c1_f32" ? 1e-5 : 2e-2);
        // multi-column against the same q8 path
        std::vector<ActCpu> as(8);
        std::vector<std::vector<double>> refs(8, std::vector<double>(W.rows));
        for (int n = 0; n < 8; ++n) {
            auto xn = make_activation(W.cols, 100 + n);
            as[n].set(xn.data(), W.cols);
            ref_gemv(W, xn.data(), refs[n].data());
        }
        std::vector<float> y8(size_t(W.rows) * 8);
        c4_q8_cols<8>(W.q, W.e, W.cols, as.data(), y8.data(), W.rows);
        double worst = 0;
        for (int n = 0; n < 8; ++n) {
            std::vector<float> col(W.rows);
            for (int r = 0; r < W.rows; ++r) col[r] = y8[size_t(r) * 8 + n];
            worst = std::max(worst, compare(col.data(), refs[n].data(), W.rows).rel_l2);
        }
        fails += !(worst < 2e-2);
        printf("  %-14s %-4s relL2(worst col)=%.3e %s\n", "c4_q8_cols8", which ? "down" : "gu", worst,
               worst < 2e-2 ? "ok" : "FAIL");
    }
    printf("selftest: %s\n", fails ? "FAILED" : "passed");
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    init_tables();
    Opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value for %s\n", a.c_str()), exit(2), ""); };
        if (a == "--weights") o.weights = next();
        else if (a == "--model") o.model = next();
        else if (a == "--layer") o.layer = std::atoi(next());
        else if (a == "--sets") o.sets = std::atoi(next());
        else if (a == "--iters") o.iters = std::atoi(next());
        else if (a == "--threads") o.threads = parse_list(next());
        else if (a == "--ncols") o.ncols = parse_list(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--moe-only") o.only_moe = true;
        else if (a == "--no-kquant") o.kquant = false;
        else if (a == "--kquant-only") o.mx = false;
        else {
            printf("usage: bench_cpu [--weights dummy|real] [--model DIR] [--layer L] [--sets S] [--iters N]\n"
                   "                 [--threads 1,2,..] [--ncols 1,2,4,8] [--csv FILE] [--selftest] [--moe-only]\n"
                   "                 [--no-kquant | --kquant-only]   (Q2_K/Q3_K: Strata q23 pack and ggml-org Q2_K mixes)\n");
            return a == "--help" ? 0 : 2;
        }
    }

    // Load S sets of top-8 experts. Real: set s uses layer 1+(layer-1+s)%47 and 8 seeded-random experts.
    std::vector<Expert> experts;
    std::vector<std::string> labels;
    {
        double t0 = now_s();
        std::unique_ptr<MimoCheckpoint> ck;
        if (o.weights == "real") ck = std::make_unique<MimoCheckpoint>(o.model);
        std::mt19937_64 rng(1234);
        for (int s = 0; s < o.sets; ++s) {
            int layer = 1 + (o.layer - 1 + s) % mimo::kMoeLayers;
            for (int k = 0; k < mimo::kTopK; ++k) {
                int ex = int(rng() % mimo::kExperts);
                if (ck) experts.push_back(ck->load_expert(layer, ex));
                else experts.push_back(make_dummy_expert(rng()));
                labels.push_back("L" + std::to_string(layer) + "E" + std::to_string(ex));
            }
        }
        printf("loaded %zu experts (%s) in %.1f s, %.1f MB each, first %s\n", experts.size(), o.weights.c_str(),
               now_s() - t0, experts[0].bytes() / 1e6, labels[0].c_str());
    }
    if (o.weights == "real") {  // sanity: E8M0 and code histogram of the first expert
        long hist[16] = {}, emin = 255, emax = 0;
        const Expert& x = experts[0];
        for (size_t i = 0; i < x.guq.size(); ++i) { hist[x.guq[i] & 15]++; hist[x.guq[i] >> 4]++; }
        for (uint8_t e : x.gue) { emin = std::min<long>(emin, e); emax = std::max<long>(emax, e); }
        printf("real-weight check %s: e8m0 range [%ld,%ld], code histogram:", labels[0].c_str(), emin, emax);
        for (int i = 0; i < 16; ++i) printf(" %.3f", hist[i] / double(2 * x.guq.size()));
        printf("\n");
    }
    // K-quant copies of every expert: Q2_K gate/up, Q3_K down (what strata-mimo-q2-pack writes, uncalibrated).
    std::vector<kq::KMat> kgu, kdn;
    if (o.kquant) {
        double t0 = now_s();
        kgu.resize(experts.size());
        kdn.resize(experts.size());
        std::vector<double> e_gu(experts.size()), e_dn(experts.size());
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < int(experts.size()) * 2; ++i) {
            const int e = i / 2;
            if (i & 1) kdn[e] = requant(experts[e].dn(), KType::Q3K, &e_dn[e]);
            else kgu[e] = requant(experts[e].gu(), KType::Q2K, &e_gu[e]);
        }
        double sg = 0, sd = 0;
        for (size_t e = 0; e < experts.size(); ++e) { sg += e_gu[e]; sd += e_dn[e]; }
        printf("requantized %zu experts in %.1f s: weight NMSE vs MXFP4 source: Q2_K gate/up %.4f, Q3_K down %.4f\n",
               experts.size(), now_s() - t0, sg / experts.size(), sd / experts.size());
    }
    if (o.selftest) {
        int fails = selftest(experts);
        if (o.kquant) {  // K-quant kernels against fp64 on the requantized weights
            auto x = make_activation(mimo::kHidden, 7), xi = make_activation(mimo::kExpertInter, 8);
            std::vector<BlockQ8K> qx(mimo::kHidden / QKK), qi(mimo::kExpertInter / QKK);
            quantize_row_q8k(x.data(), qx.data(), mimo::kHidden);
            quantize_row_q8k(xi.data(), qi.data(), mimo::kExpertInter);
            for (int which = 0; which < 2; ++which) {
                const KMat& m = which ? kdn[0] : kgu[0];
                const auto& xv = which ? xi : x;
                std::vector<double> ref(m.rows);
                std::vector<float> got(m.rows);
                for (int r = 0; r < m.rows; ++r) {
                    ref[r] = m.ref_dot(r, xv.data());
                    const uint8_t* p = m.row(r);
                    got[r] = which ? kq::dot_q3k((const BlockQ3K*)p, qi.data(), m.cols / QKK)
                                   : kq::dot_q2k((const BlockQ2K*)p, qx.data(), m.cols / QKK);
                }
                // exactness: the same fp64 dot on the dequantized q8_K activation
                const auto& qv = which ? qi : qx;
                std::vector<float> xq(m.cols);
                for (int b = 0; b < m.cols / QKK; ++b)
                    for (int j = 0; j < QKK; ++j) xq[b * QKK + j] = qv[b].d * qv[b].qs[j];
                std::vector<double> refq(m.rows);
                for (int r = 0; r < m.rows; ++r) refq[r] = m.ref_dot(r, xq.data());
                ErrStats e = compare(got.data(), ref.data(), m.rows), eq = compare(got.data(), refq.data(), m.rows);
                bool ok = e.finite && eq.rel_l2 < 1e-5;
                fails += !ok;
                printf("  %-14s %-4s relL2=%.3e (q8_K per-256 activations), vs fp64 on its quantized input %.2e %s\n",
                       which ? "kq_q3k" : "kq_q2k", which ? "down" : "gu", e.rel_l2, eq.rel_l2, ok ? "EXACT" : "FAIL");
            }
        }
        printf("selftest (incl. K-quant): %s\n", fails ? "FAILED" : "passed");
        return fails ? 1 : 0;
    }

    Csv csv(o.csv);
    const char* wl = o.weights.c_str();
    const int max_threads = omp_get_max_threads();
    printf("omp max threads %d; OMP_PROC_BIND=%s OMP_PLACES=%s\n", max_threads,
           getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "(unset)",
           getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "(unset)");

    for (int T : o.threads) {
        if (T > max_threads) continue;
        omp_set_num_threads(T);
        const double peak = read_bandwidth_gbps(size_t(1) << 30, 5);
        printf("-- threads %d: read bandwidth %.1f GB/s (1 GiB, first-touched)\n", T, peak);
        std::string cfg = "t" + std::to_string(T);
        csv.row("cpu", "read_bw", "stream", wl, 0, cfg, Stats{1e6 * (1 << 30) / (peak * 1e9)}, double(1 << 30), peak, 0, 0, 0, 0);

        std::vector<Set> sets(o.sets);
        for (int s = 0; s < o.sets; ++s)
            for (int k = 0; k < mimo::kTopK; ++k) {
                place(experts[s * mimo::kTopK + k].gu(), sets[s].gu[k], kChunk);
                place(experts[s * mimo::kTopK + k].dn(), sets[s].dn[k], kChunk);
                sets[s].route[k] = 1.f / mimo::kTopK;
            }
        const double gu_bytes = mimo::kTopK * experts[0].gu().bytes();
        const double dn_bytes = mimo::kTopK * experts[0].dn().bytes();
        auto x = make_activation(mimo::kHidden, 11);
        auto xi = make_activation(mimo::kExpertInter, 12);
        ActCpu ax, ai;
        ax.set(x.data(), mimo::kHidden);
        ai.set(xi.data(), mimo::kExpertInter);
        std::vector<float> out(size_t(mimo::kTopK) * mimo::kHidden * 8);

        // Accuracy of each kernel on set 0 / expert 0 against fp64 (cheap: 4096 rows).
        std::vector<double> ref_gu(mimo::kHidden), ref_dn(mimo::kHidden);
        ref_gemv(experts[0].gu(), x.data(), ref_gu.data());
        ref_gemv(experts[0].dn(), xi.data(), ref_dn.data());

        auto time_it = [&](const std::function<void(int)>& f) {
            for (int i = 0; i < o.warmup; ++i) f(i);
            std::vector<double> t;
            for (int i = 0; i < o.iters; ++i) {
                double t0 = now_s();
                f(i);
                t.push_back((now_s() - t0) * 1e6);
            }
            return stats_of(t);
        };

        if (o.mx && !o.only_moe) {
            for (const Kernel& k : kKernels) {
                Stats s = time_it([&](int i) { gemv_set(sets[i % o.sets].gu, ax, out.data(), k.fn); });
                gemv_set(sets[0].gu, ax, out.data(), k.fn);
                ErrStats e = compare(out.data(), ref_gu.data(), mimo::kHidden);
                csv.row("cpu", k.name, "gu8", wl, 1, cfg, s, gu_bytes, peak, 2.0 * gu_bytes / 0.53125, e.rel_l2, e.max_abs,
                        hash_floats(out.data(), mimo::kHidden));
                s = time_it([&](int i) { gemv_set(sets[i % o.sets].dn, ai, out.data(), k.fn); });
                gemv_set(sets[0].dn, ai, out.data(), k.fn);
                e = compare(out.data(), ref_dn.data(), mimo::kHidden);
                csv.row("cpu", k.name, "down8", wl, 1, cfg, s, dn_bytes, peak, 2.0 * dn_bytes / 0.53125, e.rel_l2,
                        e.max_abs, hash_floats(out.data(), mimo::kHidden));
            }
            std::vector<ActCpu> as(8);
            for (int n = 0; n < 8; ++n) {
                auto xn = make_activation(mimo::kHidden, 200 + n);
                as[n].set(xn.data(), mimo::kHidden);
            }
            for (int N : o.ncols) {
                std::function<void(Mat*)> f;
                if (N == 1) f = [&](Mat* m) { gemv_set_cols<1>(m, as.data(), out.data()); };
                else if (N == 2) f = [&](Mat* m) { gemv_set_cols<2>(m, as.data(), out.data()); };
                else if (N == 4) f = [&](Mat* m) { gemv_set_cols<4>(m, as.data(), out.data()); };
                else if (N == 8) f = [&](Mat* m) { gemv_set_cols<8>(m, as.data(), out.data()); };
                else continue;
                Stats s = time_it([&](int i) { f(sets[i % o.sets].gu); });
                csv.row("cpu", "c4_q8_cols" + std::to_string(N), "gu8", wl, N, cfg, s, gu_bytes, peak,
                        N * 2.0 * gu_bytes / 0.53125, 0, 0, hash_floats(out.data(), mimo::kHidden));
            }
        }
        // Full MoE layer with the two best candidates.
        for (const Kernel& k : kKernels) {
            std::string n = k.name;
            if (!o.mx || (n != "c3_q8_off_r2" && n != "c3h_q8_off" && n != "c2_q8_sign")) continue;
            MoeScratch w;
            w.gu.resize(size_t(mimo::kTopK) * 2 * mimo::kExpertInter);
            w.h.resize(size_t(mimo::kTopK) * mimo::kExpertInter);
            w.y.resize(mimo::kHidden);
            Stats s = time_it([&](int i) { moe_layer(sets[i % o.sets], x.data(), w, k.fn); });
            csv.row("cpu", std::string("moe_") + k.name, "moe_layer", wl, 1, cfg, s, gu_bytes + dn_bytes, peak,
                    2.0 * (gu_bytes + dn_bytes) / 0.53125, 0, 0, hash_floats(w.y.data(), mimo::kHidden));
            printf("     -> %s: %.2f ms per MoE layer => %.1f tok/s for 47 MoE layers (experts only)\n", k.name,
                   s.med / 1e3, 1e6 / (s.med * mimo::kMoeLayers));
        }
        if (o.kquant) {
            const int TOPK = mimo::kTopK, H = mimo::kHidden, I = mimo::kExpertInter;
            std::vector<std::array<kq::PMat, 8>> kg(o.sets), kd(o.sets);
            for (int s = 0; s < o.sets; ++s)
                for (int k = 0; k < TOPK; ++k) { kg[s][k] = kq::place(kgu[s * TOPK + k], kChunk); kd[s][k] = kq::place(kdn[s * TOPK + k], kChunk); }
            const double kgu_bytes = TOPK * double(kgu[0].bytes()), kdn_bytes = TOPK * double(kdn[0].bytes());
            std::vector<BlockQ8K> qx(H / QKK), qh(size_t(TOPK) * I / QKK);
            quantize_row_q8k(x.data(), qx.data(), H);
            for (int k = 0; k < TOPK; ++k) quantize_row_q8k(xi.data(), qh.data() + k * (I / QKK), I);
            std::vector<double> rgu(H), rdn(H);
            for (int r = 0; r < H; ++r) { rgu[r] = kgu[0].ref_dot(r, x.data()); rdn[r] = kdn[0].ref_dot(r, xi.data()); }
            Stats st = time_it([&](int i) { kq::gemv_set(kg[i % o.sets].data(), qx.data(), 0, out.data(), kChunk); });
            ErrStats e = compare(out.data(), rgu.data(), H);
            csv.row("cpu", "kq_q2k_ggml", "gu8", wl, 1, cfg, st, kgu_bytes, peak, 2.0 * TOPK * double(H) * H, e.rel_l2, e.max_abs, hash_floats(out.data(), H));
            st = time_it([&](int i) { kq::gemv_set(kd[i % o.sets].data(), qh.data(), I / QKK, out.data(), kChunk); });
            e = compare(out.data(), rdn.data(), H);
            csv.row("cpu", "kq_q3k_ggml", "down8", wl, 1, cfg, st, kdn_bytes, peak, 2.0 * TOPK * double(H) * I, e.rel_l2, e.max_abs, hash_floats(out.data(), H));
            // Full MoE layers: q23 (Q2_K gate/up + Q3_K down) and ggml-org's mix (Q2_K gate/up + MXFP4 down).
            std::vector<float> gu(size_t(TOPK) * 2 * I), h(size_t(TOPK) * I), y(H);
            MoeScratch w;
            w.gu.resize(gu.size()); w.h.resize(h.size()); w.y.resize(H);
            for (int mix = 0; mix < 2; ++mix) {
                auto layer = [&](int i) {
                    const int s = i % o.sets;
                    quantize_row_q8k(x.data(), qx.data(), H);
                    kq::gemv_set(kg[s].data(), qx.data(), 0, gu.data(), kChunk);
#pragma omp parallel for schedule(static, 1)
                    for (int k = 0; k < TOPK; ++k) {
                        float* g = gu.data() + size_t(k) * 2 * I;
                        for (int j = 0; j < I; ++j) h[size_t(k) * I + j] = g[j] / (1.f + std::exp(-g[j])) * g[I + j];
                        if (mix == 0) quantize_row_q8k(h.data() + size_t(k) * I, qh.data() + k * (I / QKK), I);
                        else quantize_act_fast(h.data() + size_t(k) * I, I, w.ah[k]);
                    }
#pragma omp parallel for schedule(static)
                    for (int c = 0; c < H / kChunk; ++c) {
                        float acc[kChunk] = {}, tmp[kChunk];
                        for (int k = 0; k < TOPK; ++k) {
                            if (mix == 0) {
                                for (int r = 0; r < kChunk; ++r) acc[r] += sets[s].route[k] * kq::dot_row(kd[s][k], c * kChunk + r, qh.data() + k * (I / QKK));
                            } else {
                                const Mat& m = sets[s].dn[k];
                                c3_q8_off<1>(m.q + size_t(c) * kChunk * m.cols / 2, m.e + size_t(c) * kChunk * m.cols / QK, m.cols, w.ah[k], tmp, kChunk);
                                for (int r = 0; r < kChunk; ++r) acc[r] += sets[s].route[k] * tmp[r];
                            }
                        }
                        std::memcpy(y.data() + c * kChunk, acc, sizeof acc);
                    }
                };
                st = time_it(layer);
                const double bytes = kgu_bytes + (mix == 0 ? kdn_bytes : dn_bytes);
                const char* name = mix == 0 ? "moe_q23_strata" : "moe_q2k_mxdown_ggmlorg";
                csv.row("cpu", name, "moe_layer", wl, 1, cfg, st, bytes, peak, 2.0 * TOPK * 3.0 * H * I, 0, 0, hash_floats(y.data(), H));
                printf("     -> %s: %.2f ms per MoE layer, %.1f MB => %.1f tok/s for 47 MoE layers (experts only)\n", name,
                       st.med / 1e3, bytes / 1e6, 1e6 / (st.med * mimo::kMoeLayers));
            }
            for (int s = 0; s < o.sets; ++s)
                for (int k = 0; k < TOPK; ++k) { kq::unplace(kg[s][k]); kq::unplace(kd[s][k]); }
        }
        for (auto& st : sets)
            for (int k = 0; k < mimo::kTopK; ++k) { unplace(st.gu[k]); unplace(st.dn[k]); }
    }
    return 0;
}
