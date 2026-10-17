// MXFP4 decode / GEMV / grouped GEMM benchmark for MiMo-V2.6-Flash expert shapes on one NVIDIA GPU
// (developed on RTX 5060 Ti 16 GB, sm_120). Weights stay in the checkpoint layout; see common/mxfp4.h.
//
// Decode (batch-1 and small N, bandwidth bound):
//   g1_dp4a      PRMT LUT E2M1 -> int8, dp4a with q8 activations (llama.cpp mmvq style)
//   g2_hfma2     PRMT LUT E2M1 -> fp16 high byte, hfma2 with fp16 activations, fp16 per-block accumulator
//   g2_hmul_f32  same decode, fp32 accumulation
//   g3_moe       fused MoE layer: [q8(x) + 8x gate/up + SwiGLU + q8(h)] then [8x down + weighted sum], 2 launches
//   mma kinds    the grouped GEMM kernels of mma_native.cu with N tokens per expert (BN = 8/16)
// Prefill (grouped GEMM over all 256 experts of a layer, routed tokens sorted by expert):
//   k0_i8 k1_f8fold k2_f16fold (software, no FP4 hardware) and k3_mxf8 k4_mxf4 k5_msd (sm_120a block-scaled)
#include "../common/mxfp4.h"
#include "../common/report.h"
#include "../common/kquant.h"
#include "../common/weights.h"
#include "common.cuh"
#include "kq_gpu.cuh"

#include <omp.h>

#include <functional>
#include <memory>
#include <sstream>

using namespace fp4;

struct PeakResult { const char* name; double tops; };
std::vector<PeakResult> measure_mma_peaks(int sms);

namespace {

constexpr int H = mimo::kHidden, I = mimo::kExpertInter, TOPK = mimo::kTopK;

// ================================ decode kernels ================================

template <int NT, int RPW>
__global__ void __launch_bounds__(256) k_g1_dp4a(const uint8_t* __restrict__ wq, const uint8_t* __restrict__ we,
                                                 size_t wq_es, size_t we_es, const int* __restrict__ ids, int rows,
                                                 int K, const int8_t* __restrict__ aq, const float* __restrict__ ad,
                                                 int a_per_expert, float* __restrict__ out) {
    const int k = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K / 32;
    const uint8_t* q = wq + size_t(ids[k]) * wq_es;
    const uint8_t* e = we + size_t(ids[k]) * we_es;
    const int tok0 = a_per_expert ? k * NT : 0;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    float acc[RPW][NT] = {};
    for (int b = lane; b < nb; b += 32) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 w = ld_stream(q + size_t(row0 + r) * (K / 2) + b * 16);
            const float ws = e8m0f(e[size_t(row0 + r) * nb + b]);
            const int2 v0 = lut16(w.x, LUT_I8), v1 = lut16(w.y, LUT_I8), v2 = lut16(w.z, LUT_I8), v3 = lut16(w.w, LUT_I8);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int4* ap = (const int4*)(aq + size_t(tok0 + t) * K + b * 32);
                const int4 x0 = __ldg(ap), x1 = __ldg(ap + 1);
                int s = __dp4a(v0.x, x0.x, 0);
                s = __dp4a(v1.x, x0.y, s);
                s = __dp4a(v2.x, x0.z, s);
                s = __dp4a(v3.x, x0.w, s);
                s = __dp4a(v0.y, x1.x, s);
                s = __dp4a(v1.y, x1.y, s);
                s = __dp4a(v2.y, x1.z, s);
                s = __dp4a(v3.y, x1.w, s);
                acc[r][t] += float(s) * (ws * __ldg(ad + size_t(tok0 + t) * nb + b));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) out[size_t(k * NT + t) * rows + row0 + r] = 0.5f * v;
        }
}

template <int NT, int RPW, bool F32ACC>
__global__ void __launch_bounds__(256) k_g2_f16(const uint8_t* __restrict__ wq, const uint8_t* __restrict__ we,
                                                size_t wq_es, size_t we_es, const int* __restrict__ ids, int rows, int K,
                                                const __half* __restrict__ ah, int a_per_expert, float* __restrict__ out) {
    const int k = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K / 32;
    const uint8_t* q = wq + size_t(ids[k]) * wq_es;
    const uint8_t* e = we + size_t(ids[k]) * we_es;
    const int tok0 = a_per_expert ? k * NT : 0;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    float acc[RPW][NT] = {};
    for (int b = lane; b < nb; b += 32) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 w4 = ld_stream(q + size_t(row0 + r) * (K / 2) + b * 16);
            const float ws = e8m0f(e[size_t(row0 + r) * nb + b]);
            const uint32_t wv[4] = {w4.x, w4.y, w4.z, w4.w};
            __half2 hacc[NT];
            float facc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) { hacc[t] = __float2half2_rn(0.f); facc[t] = 0.f; }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int2 v = lut16(wv[i], LUT_F16HI);
                uint4 xv[NT];  // 8 activations (4 half2) per token, one 16-byte load
#pragma unroll
                for (int t = 0; t < NT; ++t) xv[t] = __ldg((const uint4*)(ah + size_t(tok0 + t) * K) + b * 4 + i);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t hb = __byte_perm(v.x, v.y, j | (j << 4) | ((4 + j) << 8) | ((4 + j) << 12)) & 0xff00ff00u;
                    const __half2 w = *(__half2*)&hb;
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const uint32_t xw = j == 0 ? xv[t].x : j == 1 ? xv[t].y : j == 2 ? xv[t].z : xv[t].w;
                        const __half2 x = *(const __half2*)&xw;
                        if constexpr (F32ACC) {
                            const float2 f = __half22float2(__hmul2(w, x));
                            facc[t] += f.x + f.y;
                        } else {
                            hacc[t] = __hfma2(w, x, hacc[t]);
                        }
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float s = F32ACC ? facc[t] : __low2float(hacc[t]) + __high2float(hacc[t]);
                acc[r][t] += s * ws;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) out[size_t(k * NT + t) * rows + row0 + r] = v;
        }
}

__device__ __forceinline__ int logical_pos(int phys) { return (phys & 1) ? 16 + (phys >> 1) : (phys >> 1); }

// Fused MoE, part 1: each CTA quantizes x into smem, computes 32 gate rows + the 32 matching up rows of one
// expert, applies SwiGLU and writes the q8 (logical order) block of h.
__global__ void __launch_bounds__(256) k_moe_gu(const uint8_t* __restrict__ wq, const uint8_t* __restrict__ we,
                                                size_t wq_es, size_t we_es, const int* __restrict__ ids,
                                                const float* __restrict__ x, int8_t* __restrict__ hq,
                                                float* __restrict__ hd) {
    __shared__ __align__(16) int8_t sx[H];
    __shared__ float sd[H / 32];
    __shared__ float sh[32];
    const int k = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < H / 32; b += 8) {
        const float v = x[b * 32 + lane];
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float d = m / 127.f, id = m > 0 ? 127.f / m : 0.f;
        sx[b * 32 + logical_pos(lane)] = (int8_t)__float2int_rn(v * id);
        if (lane == 0) sd[b] = d;
    }
    __syncthreads();
    const uint8_t* q = wq + size_t(ids[k]) * wq_es;
    const uint8_t* e = we + size_t(ids[k]) * we_es;
    constexpr int nb = H / 32;
    const int r0 = blockIdx.x * 32 + warp * 4;
    float acc[8] = {};  // 4 gate rows, 4 up rows
    for (int b = lane; b < nb; b += 32) {
        const int4 x0 = *(const int4*)(sx + b * 32), x1 = *(const int4*)(sx + b * 32 + 16);
        const float db = sd[b];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = (r < 4 ? 0 : I) + r0 + (r & 3);
            const uint4 w = ld_stream(q + size_t(row) * (H / 2) + b * 16);
            const int2 v0 = lut16(w.x, LUT_I8), v1 = lut16(w.y, LUT_I8), v2 = lut16(w.z, LUT_I8), v3 = lut16(w.w, LUT_I8);
            int s = __dp4a(v0.x, x0.x, 0);
            s = __dp4a(v1.x, x0.y, s);
            s = __dp4a(v2.x, x0.z, s);
            s = __dp4a(v3.x, x0.w, s);
            s = __dp4a(v0.y, x1.x, s);
            s = __dp4a(v1.y, x1.y, s);
            s = __dp4a(v2.y, x1.z, s);
            s = __dp4a(v3.y, x1.w, s);
            acc[r] += float(s) * (e8m0f(e[size_t(row) * nb + b]) * db);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.5f * warp_sum(acc[r]);
    if (lane < 4) {
        const float g = lane == 0 ? acc[0] : lane == 1 ? acc[1] : lane == 2 ? acc[2] : acc[3];
        const float u = lane == 0 ? acc[4] : lane == 1 ? acc[5] : lane == 2 ? acc[6] : acc[7];
        sh[warp * 4 + lane] = g / (1.f + __expf(-g)) * u;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = sh[lane];
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float id = m > 0 ? 127.f / m : 0.f;
        hq[size_t(k) * I + blockIdx.x * 32 + logical_pos(lane)] = (int8_t)__float2int_rn(v * id);
        if (lane == 0) hd[k * (I / 32) + blockIdx.x] = m / 127.f;
    }
}

// Fused MoE, part 2: each warp owns RPW output rows and sums all 8 experts' down projections.
template <int RPW>
__global__ void __launch_bounds__(256) k_moe_down(const uint8_t* __restrict__ wq, const uint8_t* __restrict__ we,
                                                  size_t wq_es, size_t we_es, const int* __restrict__ ids,
                                                  const float* __restrict__ route, const int8_t* __restrict__ hq,
                                                  const float* __restrict__ hd, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int nb = I / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    float acc[RPW] = {};
#pragma unroll 2
    for (int k = 0; k < TOPK; ++k) {
        const uint8_t* q = wq + size_t(ids[k]) * wq_es;
        const uint8_t* e = we + size_t(ids[k]) * we_es;
        const float rk = 0.5f * route[k];
#pragma unroll
        for (int bb = 0; bb < nb / 32; ++bb) {
            const int b = lane + 32 * bb;
            const int4* ap = (const int4*)(hq + size_t(k) * I + b * 32);
            const int4 x0 = __ldg(ap), x1 = __ldg(ap + 1);
            const float db = __ldg(hd + k * nb + b) * rk;
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const uint4 w = ld_stream(q + size_t(row0 + r) * (I / 2) + b * 16);
                const int2 v0 = lut16(w.x, LUT_I8), v1 = lut16(w.y, LUT_I8), v2 = lut16(w.z, LUT_I8), v3 = lut16(w.w, LUT_I8);
                int s = __dp4a(v0.x, x0.x, 0);
                s = __dp4a(v1.x, x0.y, s);
                s = __dp4a(v2.x, x0.z, s);
                s = __dp4a(v3.x, x0.w, s);
                s = __dp4a(v0.y, x1.x, s);
                s = __dp4a(v1.y, x1.y, s);
                s = __dp4a(v2.y, x1.z, s);
                s = __dp4a(v3.y, x1.w, s);
                acc[r] += float(s) * (e8m0f(e[size_t(row0 + r) * nb + b]) * db);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) y[row0 + r] = v;
    }
}

__global__ void k_read_bw(const uint4* __restrict__ p, size_t n, uint32_t* sink) {
    uint32_t a = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const uint4 v = ld_stream(p + i);
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x12345678u) *sink = a;
}

// ================================ host side ================================

template <class T>
struct DBuf {
    T* p = nullptr;
    size_t n = 0;
    DBuf() = default;
    explicit DBuf(size_t n_) { alloc(n_); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    DBuf(DBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DBuf& operator=(DBuf&& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
    ~DBuf() { if (p) cudaFree(p); }
    void alloc(size_t n_) { if (p) cudaFree(p); n = n_; CK(cudaMalloc(&p, n * sizeof(T) + 16)); }
    void up(const T* h, size_t cnt) { CK(cudaMemcpy(p, h, cnt * sizeof(T), cudaMemcpyHostToDevice)); }
    void up(const std::vector<T>& h) { if (n < h.size()) alloc(h.size()); up(h.data(), h.size()); }
    std::vector<T> down(size_t cnt) const { std::vector<T> h(cnt); CK(cudaMemcpy(h.data(), p, cnt * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
};

struct Pool {  // P experts, gu and dn matrices each contiguous per expert
    int P = 0;
    DBuf<uint8_t> guq, gue, dnq, dne;
    size_t guq_es = size_t(2) * I * H / 2, gue_es = size_t(2) * I * H / 32, dnq_es = size_t(H) * I / 2, dne_es = size_t(H) * I / 32;
    std::vector<Expert> host;
    void upload() {
        P = int(host.size());
        guq.alloc(P * guq_es); gue.alloc(P * gue_es); dnq.alloc(P * dnq_es); dne.alloc(P * dne_es);
        for (int i = 0; i < P; ++i) {
            CK(cudaMemcpy(guq.p + i * guq_es, host[i].guq.data(), guq_es, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(gue.p + i * gue_es, host[i].gue.data(), gue_es, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dnq.p + i * dnq_es, host[i].dnq.data(), dnq_es, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dne.p + i * dne_es, host[i].dne.data(), dne_es, cudaMemcpyHostToDevice));
        }
    }
};

struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    // per-launch microseconds: `reps` launches between two events, `trials` times
    Stats run(const std::function<void(int)>& f, int reps, int trials) {
        int it = 0;
        for (int i = 0; i < 3; ++i) f(it++);
        CK(cudaDeviceSynchronize());
        std::vector<double> v;
        for (int t = 0; t < trials; ++t) {
            CK(cudaEventRecord(a));
            for (int r = 0; r < reps; ++r) f(it++);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms;
            CK(cudaEventElapsedTime(&ms, a, b));
            v.push_back(1e3 * ms / reps);
        }
        CK(cudaGetLastError());
        return stats_of(v);
    }
};

// ---- activation preparation for every kernel family (host, OpenMP over tokens) ----
enum ActKind { A_Q8, A_F16NAT, A_F8TOK, A_F16LOG, A_MXF8, A_MXF4, A_MSD };

int f16_logical_to_phys(int l) {
    const int h = l / 16, r = l % 16;
    return r < 8 ? 8 * (r / 2) + 2 * (r % 2) + h : 8 * ((r - 8) / 2) + 4 + 2 * (r % 2) + h;
}

struct DevAct {
    DBuf<uint8_t> a, s, a2, s2;
};

DevAct prep_act(ActKind kind, const std::vector<float>& X, int ntok, int K, std::vector<float>* deq = nullptr) {
    const int nb = K / 32;
    std::vector<uint8_t> a, s, a2, s2;
    switch (kind) {
        case A_Q8: a.resize(size_t(ntok) * K); s.resize(size_t(ntok) * nb * 4); break;
        case A_F16NAT: case A_F16LOG: a.resize(size_t(ntok) * K * 2); break;
        case A_F8TOK: a.resize(size_t(ntok) * K); s.resize(size_t(ntok) * 4); break;
        case A_MXF8: a.resize(size_t(ntok) * K); s.resize(size_t(ntok) * nb); break;
        case A_MXF4: a.resize(size_t(ntok) * K / 2); s.resize(size_t(ntok) * nb); break;
        case A_MSD: a.resize(size_t(ntok) * K / 2); s.resize(size_t(ntok) * nb); a2.resize(a.size()); s2.resize(s.size()); break;
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (int t = 0; t < ntok; ++t) {
        const float* x = X.data() + size_t(t) * K;
        switch (kind) {
            case A_Q8: {
                std::vector<Q8Block> qb(nb);
                quantize_q8_logical(x, K, qb.data());
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(&a[size_t(t) * K + b * 32], qb[b].qs, 32);
                    std::memcpy(&s[(size_t(t) * nb + b) * 4], &qb[b].d, 4);
                }
            } break;
            case A_F16NAT:
            case A_F16LOG: {
                __half* h = (__half*)a.data() + size_t(t) * K;
                for (int i = 0; i < K; ++i) {
                    const int src = kind == A_F16NAT ? i : (i / 32) * 32 + f16_logical_to_phys(i % 32);
                    h[i] = __float2half_rn(x[src]);
                }
            } break;
            case A_F8TOK: {
                float amax = 0;
                for (int i = 0; i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
                const float sc = amax > 0 ? amax / 448.f : 1.f;
                std::memcpy(&s[size_t(t) * 4], &sc, 4);
                for (int i = 0; i < K; ++i) a[size_t(t) * K + i] = float_to_e4m3(x[(i / 32) * 32 + logical_to_phys(i % 32)] / sc);
            } break;
            case A_MXF8: quantize_mxfp8_act_logical(x, K, &a[size_t(t) * K], &s[size_t(t) * nb]); break;
            case A_MXF4: quantize_mxfp4_act(x, K, &a[size_t(t) * K / 2], &s[size_t(t) * nb]); break;
            case A_MSD: {
                std::vector<float> deq(K), res(K);
                quantize_mxfp4_act(x, K, &a[size_t(t) * K / 2], &s[size_t(t) * nb], deq.data());
                for (int i = 0; i < K; ++i) res[i] = x[i] - deq[i];
                quantize_mxfp4_act(res.data(), K, &a2[size_t(t) * K / 2], &s2[size_t(t) * nb]);
            } break;
        }
    }
    if (deq) {  // what the kernel actually multiplies, back in natural order (for exactness checks)
        deq->assign(size_t(ntok) * K, 0.f);
#pragma omp parallel for
        for (int t = 0; t < ntok; ++t) {
            float* o = deq->data() + size_t(t) * K;
            for (int i = 0; i < K; ++i) {
                const int b = i / 32, l = i % 32;
                const size_t ti = size_t(t) * K + i;
                switch (kind) {
                    case A_Q8: { float d; std::memcpy(&d, &s[(size_t(t) * nb + b) * 4], 4); o[b * 32 + logical_to_phys(l)] = d * (int8_t)a[ti]; } break;
                    case A_F16NAT: o[i] = __half2float(((const __half*)a.data())[ti]); break;
                    case A_F16LOG: o[b * 32 + f16_logical_to_phys(l)] = __half2float(((const __half*)a.data())[ti]); break;
                    case A_F8TOK: { float sc; std::memcpy(&sc, &s[size_t(t) * 4], 4); o[b * 32 + logical_to_phys(l)] = sc * e4m3_to_float(a[ti]); } break;
                    case A_MXF8: o[b * 32 + logical_to_phys(l)] = e8m0_to_float(s[size_t(t) * nb + b]) * e4m3_to_float(a[ti]); break;
                    case A_MXF4: case A_MSD: {
                        auto dq = [&](const std::vector<uint8_t>& qa, const std::vector<uint8_t>& qs) {
                            const uint8_t byte = qa[(size_t(t) * K + i) / 2];
                            return kE2M1[(i & 1) ? byte >> 4 : byte & 15] * e8m0_to_float(qs[size_t(t) * nb + b]);
                        };
                        o[i] = dq(a, s) + (kind == A_MSD ? dq(a2, s2) : 0.f);
                    } break;
                }
            }
        }
    }
    DevAct d;
    d.a.up(a);
    if (!s.empty()) d.s.up(s);
    if (!a2.empty()) { d.a2.up(a2); d.s2.up(s2); }
    return d;
}

struct GemmKind { const char* name; int kind; ActKind act; int shift; };
std::vector<GemmKind> g_kinds = {
    {"k0_i8_mma", 0, A_Q8, 0},
#ifndef FP4_AMPERE
    {"k1_f8fold_mma", 1, A_F8TOK, 0},
#endif
    {"k2_f16fold_mma", 2, A_F16LOG, 0},
#ifndef FP4_AMPERE
    {"k3_mxf8f6f4", 3, A_MXF8, 2},   {"k4_mxf4", 4, A_MXF4, 0},        {"k5_msd_mxf4x2", 5, A_MSD, 0},
#endif
};

void build_f8_tables() {
    uint32_t tab[16][4];
    for (int d = 0; d < 16; ++d) {
        uint8_t b[16];
        for (int c = 0; c < 16; ++c) b[c] = float_to_e4m3(std::ldexp(kE2M1[c], -d));
        std::memcpy(tab[d], b, 16);
    }
    set_f8_tables(&tab[0][0]);
}

// ---- options ----
struct Opts {
    std::string weights = "dummy", model = "models/m26f", csv, mode = "all";
    int layer = 1, experts = 256, reps = 20, trials = 9;
    std::vector<int> tokens = {512, 2048, 8192};
    std::vector<int> ncols = {1, 2, 4, 8, 16};
    std::vector<int> only_kinds;  // empty = all
};
std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

struct Ctx {
    Opts o;
    Pool pool;
    Csv* csv = nullptr;
    double peak = 0;
    int nsets = 0;
    DBuf<int> ids;           // [nsets][8]
    DBuf<float> route;       // [nsets][8]
    std::vector<std::vector<Tile>> set_tiles;  // filled per N
    Timer tm;
    const char* wl() const { return o.weights.c_str(); }
};

// out[(k*N + t)][rows] reference for set s, matrix gu (shared x[N][H]) or dn (x[8N][I])
std::vector<double> ref_set(Ctx& c, int s, bool gu, const std::vector<float>& X, int N) {
    const int rows = gu ? 2 * I : H, K = gu ? H : I;
    std::vector<double> ref(size_t(TOPK) * N * rows);
#pragma omp parallel for collapse(2)
    for (int k = 0; k < TOPK; ++k)
        for (int t = 0; t < N; ++t) {
            const Expert& ex = c.pool.host[s * TOPK + k];
            const MxView W = gu ? ex.gu() : ex.dn();
            const float* x = X.data() + size_t(gu ? t : k * N + t) * K;
            for (int r = 0; r < rows; ++r) ref[size_t(k * N + t) * rows + r] = ref_dot_row(W, r, x);
        }
    return ref;
}

// ---------------- decode / small-N section ----------------
void run_decode(Ctx& c, bool selftest) {
    printf("== decode / small batch (8 experts per call, rotating over %d expert sets) ==\n", c.nsets);
    for (int N : c.o.ncols) {
        for (int gu = 1; gu >= 0; --gu) {
            const int rows = gu ? 2 * I : H, K = gu ? H : I;
            const int nact = gu ? N : TOPK * N;
            const char* op = gu ? "gu8" : "down8";
            std::vector<float> X(size_t(nact) * K);
            for (int t = 0; t < nact; ++t) {
                auto x = make_activation(K, 1000 + t);
                std::memcpy(&X[size_t(t) * K], x.data(), K * 4);
            }
            const uint8_t* wq = gu ? c.pool.guq.p : c.pool.dnq.p;
            const uint8_t* we = gu ? c.pool.gue.p : c.pool.dne.p;
            const size_t wq_es = gu ? c.pool.guq_es : c.pool.dnq_es, we_es = gu ? c.pool.gue_es : c.pool.dne_es;
            const double bytes = TOPK * (double(rows) * K * 0.53125);
            const double flops = 2.0 * TOPK * double(rows) * K * N;
            std::vector<double> ref = ref_set(c, 0, gu, X, N);
            DBuf<float> out(size_t(TOPK) * N * rows);
            auto check = [&](const std::string& name, const Stats& st, const std::string& cfg) {
                auto h = out.down(size_t(TOPK) * N * rows);
                ErrStats e = compare(h.data(), ref.data(), h.size());
                c.csv->row("gpu", name, op, c.wl(), N, cfg, st, bytes, c.peak, flops, e.rel_l2, e.max_abs, hash_floats(h.data(), h.size()));
            };
            // -- g1 dp4a / g2 f16 GEMV --
            DevAct q8 = prep_act(A_Q8, X, nact, K), f16 = prep_act(A_F16NAT, X, nact, K);
            auto gemv = [&](const std::string& name, auto kern, int rpw, auto&&... act) {
                dim3 grid(rows / (8 * rpw), TOPK);
                auto f = [&](int it) {
                    const int s = selftest ? 0 : it % c.nsets;
                    kern<<<grid, 256>>>(wq, we, wq_es, we_es, c.ids.p + 8 * s, rows, K, act..., gu ? 0 : 1, out.p);
                };
                Stats st = selftest ? Stats{} : c.tm.run(f, c.o.reps, c.o.trials);
                f(0);
                CK(cudaDeviceSynchronize());
                check(name, st, "rpw" + std::to_string(rpw));
            };
#define G1(NT)                                                                                                    \
    if (N == NT) {                                                                                                \
        gemv("g1_dp4a", k_g1_dp4a<NT, 1>, 1, (const int8_t*)q8.a.p, (const float*)q8.s.p);                       \
        gemv("g1_dp4a", k_g1_dp4a<NT, 2>, 2, (const int8_t*)q8.a.p, (const float*)q8.s.p);                       \
        gemv("g2_hfma2", k_g2_f16<NT, 1, false>, 1, (const __half*)f16.a.p);                                      \
        gemv("g2_hfma2", k_g2_f16<NT, 2, false>, 2, (const __half*)f16.a.p);                                      \
        gemv("g2_hmul_f32", k_g2_f16<NT, 2, true>, 2, (const __half*)f16.a.p);                                   \
    }
            G1(1) G1(2) G1(4) G1(8) G1(16)
#undef G1
            // -- tensor-core kinds with N tokens per expert --
            std::vector<DBuf<Tile>> tiles(c.nsets);
            for (int s = 0; s < c.nsets; ++s) {
                std::vector<Tile> tl;
                for (int k = 0; k < TOPK; ++k) {
                    int ex;
                    CK(cudaMemcpy(&ex, c.ids.p + 8 * s + k, 4, cudaMemcpyDeviceToHost));
                    tl.push_back({ex, gu ? 0 : k * N, N, k * N});
                }
                tiles[s].up(tl);
            }
            for (const GemmKind& gk : g_kinds) {
                if (!c.o.only_kinds.empty() && std::find(c.o.only_kinds.begin(), c.o.only_kinds.end(), gk.kind) == c.o.only_kinds.end()) continue;
                std::vector<float> Xq;
                DevAct act = prep_act(gk.act, X, nact, K, selftest ? &Xq : nullptr);
                std::vector<double> refq;
                if (selftest) refq = ref_set(c, 0, gu, Xq, N);
                for (int bn : {8, 16}) {
                    if (bn > 8 && N <= 8) continue;
                    if (bn == 8 && N > 8) continue;
                    GemmArgs a{wq, we, wq_es, we_es, rows, K, nullptr, TOPK, act.a.p, act.s.p, act.a2.p, act.s2.p, out.p};
                    auto f = [&](int it) {
                        a.tiles = tiles[selftest ? 0 : it % c.nsets].p;
                        launch_gemm(gk.kind, bn, gk.shift, a, 0);
                    };
                    Stats st = selftest ? Stats{} : c.tm.run(f, c.o.reps, c.o.trials);
                    f(0);
                    CK(cudaDeviceSynchronize());
                    check(gk.name, st, "bn" + std::to_string(bn));
                    if (selftest) {
                        auto h = out.down(size_t(TOPK) * N * rows);
                        ErrStats e = compare(h.data(), refq.data(), h.size());
                        printf("     exactness vs fp64 on the kernel's own quantized activations: relL2=%.2e %s\n", e.rel_l2,
                               e.rel_l2 < 1e-5 ? "EXACT" : "MISMATCH");
                    }
                }
            }
        }
    }
}

// ---------------- fused MoE layer ----------------
void run_moe(Ctx& c, bool selftest) {
    printf("== fused MoE decode layer (g3: 2 launches) ==\n");
    auto x = make_activation(H, 77);
    DBuf<float> dx(H), dy(H), hd(TOPK * I / 32);
    DBuf<int8_t> hq(TOPK * I);
    dx.up(x);
    auto layer = [&](int s, cudaStream_t st) {
        k_moe_gu<<<dim3(I / 32, TOPK), 256, 0, st>>>(c.pool.guq.p, c.pool.gue.p, c.pool.guq_es, c.pool.gue_es,
                                                      c.ids.p + 8 * s, dx.p, hq.p, hd.p);
        k_moe_down<2><<<H / 16, 256, 0, st>>>(c.pool.dnq.p, c.pool.dne.p, c.pool.dnq_es, c.pool.dne_es, c.ids.p + 8 * s,
                                              c.route.p + 8 * s, hq.p, hd.p, dy.p);
    };
    // reference: fp64 gate/up/down with exact SwiGLU, route weights of set 0
    std::vector<double> ref(H, 0.0);
    {
        std::vector<float> rh(TOPK * 8);
        CK(cudaMemcpy(rh.data(), c.route.p, TOPK * 4, cudaMemcpyDeviceToHost));
        std::vector<double> part(size_t(TOPK) * H);
#pragma omp parallel for
        for (int k = 0; k < TOPK; ++k) {
            const Expert& ex = c.pool.host[k];
            std::vector<double> gu(2 * I);
            for (int r = 0; r < 2 * I; ++r) gu[r] = ref_dot_row(ex.gu(), r, x.data());
            std::vector<float> h(I);
            for (int i = 0; i < I; ++i) h[i] = float(gu[i] / (1 + std::exp(-gu[i])) * gu[I + i]);
            for (int r = 0; r < H; ++r) part[size_t(k) * H + r] = rh[k] * ref_dot_row(ex.dn(), r, h.data());
        }
        for (int k = 0; k < TOPK; ++k)
            for (int r = 0; r < H; ++r) ref[r] += part[size_t(k) * H + r];
    }
    const double bytes = TOPK * (2.0 * I * H + double(H) * I) * 0.53125;
    Stats st{};
    if (!selftest) st = c.tm.run([&](int it) { layer(it % c.nsets, 0); }, c.o.reps, c.o.trials);
    layer(0, 0);
    CK(cudaDeviceSynchronize());
    auto y = dy.down(H);
    ErrStats e = compare(y.data(), ref.data(), H);
    c.csv->row("gpu", "g3_moe_fused", "moe_layer", c.wl(), 1, "2launch", st, bytes, c.peak, 2 * bytes / 0.53125, e.rel_l2, e.max_abs, hash_floats(y.data(), H));
    if (selftest) return;
    // 47 MoE layers in one CUDA graph (rotating expert sets): what decode would see per token
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < mimo::kMoeLayers; ++l) layer(l % c.nsets, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    Stats sg = c.tm.run([&](int) { CK(cudaGraphLaunch(ge, 0)); }, 5, c.o.trials);
    Stats per{sg.med / mimo::kMoeLayers, sg.p10 / mimo::kMoeLayers, sg.p90 / mimo::kMoeLayers, sg.min / mimo::kMoeLayers};
    c.csv->row("gpu", "g3_moe_fused", "moe47_graph", c.wl(), 1, "per_layer", per, bytes, c.peak, 2 * bytes / 0.53125, 0, 0, 0);
    printf("     -> 47 MoE layers in a graph: %.2f ms => %.1f tok/s (expert part only, all experts resident)\n", sg.med / 1e3,
           1e6 / sg.med);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

// ---------------- prefill grouped GEMM ----------------
void run_prefill(Ctx& c, bool selftest) {
    const int P = c.pool.P;
    printf("== prefill grouped GEMM over %d experts ==\n", P);
    std::vector<int> toks = selftest ? std::vector<int>{96} : c.o.tokens;
    for (int T : toks) {
        // routing: 8 distinct experts per token, skewed popularity (lognormal weights)
        std::mt19937_64 rng(4242 + T);
        std::lognormal_distribution<double> ln(0.0, 0.7);
        std::vector<double> pw(P);
        for (auto& v : pw) v = ln(rng);
        std::discrete_distribution<int> pick(pw.begin(), pw.end());
        std::vector<std::vector<int>> by_expert(P);
        for (int t = 0; t < T; ++t) {
            int chosen[TOPK], n = 0;
            while (n < TOPK) {
                int e = pick(rng);
                bool dup = false;
                for (int i = 0; i < n; ++i) dup |= chosen[i] == e;
                if (!dup) chosen[n++] = e;
            }
            for (int i = 0; i < TOPK; ++i) by_expert[chosen[i]].push_back(t);
        }
        const int slots = T * TOPK;
        std::vector<int> slot_expert(slots), slot_tok(slots), off(P + 1, 0);
        for (int e = 0, s = 0; e < P; ++e) {
            off[e] = s;
            for (int t : by_expert[e]) { slot_expert[s] = e; slot_tok[s] = t; ++s; }
            off[e + 1] = s;
        }
        int used = 0, maxc = 0;
        for (int e = 0; e < P; ++e) { used += !by_expert[e].empty(); maxc = std::max<int>(maxc, by_expert[e].size()); }
        printf("-- T=%d tokens: %d slots, %d experts used, max %d tokens per expert, mean %.1f\n", T, slots, used, maxc, slots / double(used));

        for (int gu = 1; gu >= 0; --gu) {
            const int rows = gu ? 2 * I : H, K = gu ? H : I;
            const char* op = gu ? "gu_prefill" : "down_prefill";
            std::vector<float> X(size_t(slots) * K);
            {  // gu: the token's hidden state, gathered; down: an independent h per slot
                std::vector<std::vector<float>> tokx(gu ? T : 0);
#pragma omp parallel for
                for (int t = 0; t < (gu ? T : 0); ++t) tokx[t] = make_activation(K, 50000 + t);
#pragma omp parallel for
                for (int s = 0; s < slots; ++s) {
                    std::vector<float> v = gu ? tokx[slot_tok[s]] : make_activation(K, 90000 + s);
                    std::memcpy(&X[size_t(s) * K], v.data(), K * 4);
                }
            }
            // reference on sampled slots
            const int nsamp = 16;
            std::vector<int> samp(nsamp);
            for (int i = 0; i < nsamp; ++i) samp[i] = int((size_t(i) * 7919 + 13) % slots);
            std::vector<double> ref(size_t(nsamp) * rows);
#pragma omp parallel for collapse(2)
            for (int i = 0; i < nsamp; ++i)
                for (int r = 0; r < rows; ++r) {
                    const Expert& ex = c.pool.host[slot_expert[samp[i]]];
                    ref[size_t(i) * rows + r] = ref_dot_row(gu ? ex.gu() : ex.dn(), r, X.data() + size_t(samp[i]) * K);
                }
            DBuf<float> out(size_t(slots) * rows);
            const double flops = 2.0 * slots * double(rows) * K;
            const double wbytes = used * double(rows) * K * 0.53125;
            for (const GemmKind& gk : g_kinds) {
                if (!c.o.only_kinds.empty() && std::find(c.o.only_kinds.begin(), c.o.only_kinds.end(), gk.kind) == c.o.only_kinds.end()) continue;
                std::vector<float> Xq;
                DevAct act = prep_act(gk.act, X, slots, K, selftest ? &Xq : nullptr);
                std::vector<double> refq;
                if (selftest) {
                    refq.resize(size_t(nsamp) * rows);
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < nsamp; ++i)
                        for (int r = 0; r < rows; ++r) {
                            const Expert& ex = c.pool.host[slot_expert[samp[i]]];
                            refq[size_t(i) * rows + r] = ref_dot_row(gu ? ex.gu() : ex.dn(), r, Xq.data() + size_t(samp[i]) * K);
                        }
                }
                for (int bn : {16, 32, 64}) {
                    std::vector<Tile> tl;
                    for (int e = 0; e < P; ++e)
                        for (int t0 = off[e]; t0 < off[e + 1]; t0 += bn) tl.push_back({e, t0, std::min(bn, off[e + 1] - t0), t0});
                    DBuf<Tile> dt;
                    dt.up(tl);
                    GemmArgs a{gu ? c.pool.guq.p : c.pool.dnq.p, gu ? c.pool.gue.p : c.pool.dne.p,
                               gu ? c.pool.guq_es : c.pool.dnq_es, gu ? c.pool.gue_es : c.pool.dne_es, rows, K, dt.p,
                               int(tl.size()), act.a.p, act.s.p, act.a2.p, act.s2.p, out.p};
                    auto f = [&](int) { launch_gemm(gk.kind, bn, gk.shift, a, 0); };
                    Stats st = selftest ? Stats{} : c.tm.run(f, 3, std::max(3, c.o.trials / 2));
                    f(0);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> got(size_t(nsamp) * rows);
                    for (int i = 0; i < nsamp; ++i)
                        CK(cudaMemcpy(&got[size_t(i) * rows], out.p + size_t(samp[i]) * rows, rows * 4, cudaMemcpyDeviceToHost));
                    ErrStats e = compare(got.data(), ref.data(), got.size());
                    if (selftest) {
                        ErrStats eq = compare(got.data(), refq.data(), got.size());
                        printf("     exactness bn%d: relL2=%.2e %s\n", bn, eq.rel_l2, eq.rel_l2 < 1e-5 ? "EXACT" : "MISMATCH");
                    }
                    c.csv->row("gpu", gk.name, op, c.wl(), T, "bn" + std::to_string(bn), st, wbytes, c.peak, flops, e.rel_l2, e.max_abs, hash_floats(got.data(), got.size()));
                }
            }
        }
    }
}


// ---------------- K-quant experts (Strata q23 pack, ggml-org Q2_K GGUF) ----------------
struct KAct { DBuf<int8_t> q; DBuf<float> d; DBuf<int> s16; };
KAct prep_kact(const std::vector<float>& X, int ntok, int K, std::vector<float>* deq = nullptr) {
    std::vector<int8_t> q(size_t(ntok) * K);
    std::vector<float> d(size_t(ntok) * K / 32);
    std::vector<int> s16(size_t(ntok) * K / 16);
    if (deq) deq->resize(X.size());
    for (size_t b = 0; b < size_t(ntok) * K / 32; ++b) {
        float m = 0;
        for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(X[b * 32 + i]));
        const float id = m > 0 ? 127.f / m : 0.f;
        d[b] = m / 127.f;
        for (int i = 0; i < 32; ++i) {
            const int v = (int)std::nearbyint(X[b * 32 + i] * id);
            q[b * 32 + i] = (int8_t)v;
            s16[b * 2 + i / 16] += v;
            if (deq) (*deq)[b * 32 + i] = v * d[b];
        }
    }
    KAct a;
    a.q.up(q); a.d.up(d); a.s16.up(s16);
    return a;
}

void run_kquant(Ctx& c, bool selftest) {
    const int P = c.pool.P;
    printf("== K-quant experts: Q2_K gate/up, Q3_K down (Strata q23 pack) and Q2_K + MXFP4 down (ggml-org) ==\n");
    double t0 = now_s();
    std::vector<KMat> kgu(P), kdn(P);
    double nmse_gu = 0, nmse_dn = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : nmse_gu, nmse_dn)
    for (int i = 0; i < 2 * P; ++i) {
        double e = 0;
        if (i & 1) { kdn[i / 2] = requant(c.pool.host[i / 2].dn(), KType::Q3K, &e); nmse_dn += e; }
        else { kgu[i / 2] = requant(c.pool.host[i / 2].gu(), KType::Q2K, &e); nmse_gu += e; }
    }
    const size_t gu_es = kgu[0].bytes(), dn_es = kdn[0].bytes();
    DBuf<uint8_t> dgu(size_t(P) * gu_es), ddn(size_t(P) * dn_es);
    for (int e = 0; e < P; ++e) {
        CK(cudaMemcpy(dgu.p + size_t(e) * gu_es, kgu[e].data.data(), gu_es, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ddn.p + size_t(e) * dn_es, kdn[e].data.data(), dn_es, cudaMemcpyHostToDevice));
    }
    printf("requantized %d experts in %.1f s (weight NMSE vs MXFP4: Q2_K gate/up %.4f, Q3_K down %.4f); %.2f + %.2f MB per expert\n",
           P, now_s() - t0, nmse_gu / P, nmse_dn / P, gu_es / 1e6, dn_es / 1e6);

    // GEMV: 8 experts per call, rotating sets
    for (int gu = 1; gu >= 0; --gu) {
        const int rows = gu ? 2 * I : H, K = gu ? H : I, nact = gu ? 1 : TOPK;
        std::vector<float> X(size_t(nact) * K);
        for (int t = 0; t < nact; ++t) { auto x = make_activation(K, 3000 + t); std::memcpy(&X[size_t(t) * K], x.data(), K * 4); }
        std::vector<float> Xq;
        KAct a = prep_kact(X, nact, K, &Xq);
        std::vector<double> ref(size_t(TOPK) * rows), refq(size_t(TOPK) * rows);
#pragma omp parallel for collapse(2)
        for (int k = 0; k < TOPK; ++k)
            for (int r = 0; r < rows; ++r) {
                const KMat& m = gu ? kgu[k] : kdn[k];
                const size_t xo = size_t(gu ? 0 : k) * K;
                ref[size_t(k) * rows + r] = m.ref_dot(r, X.data() + xo);
                refq[size_t(k) * rows + r] = m.ref_dot(r, Xq.data() + xo);
            }
        DBuf<float> out(size_t(TOPK) * rows);
        const int rpw = 32 / (2 * K / 256);
        for (int R : {1, 2, 4}) {
            dim3 grid(rows / (8 * rpw * R), TOPK);
            auto f = [&](int it) {
                const int s = selftest ? 0 : it % c.nsets;
                const int* ids = c.ids.p + 8 * s;
                if (gu) {
                    if (R == 1) kqg::k_kq_gemv<false, 1><<<grid, 256>>>(dgu.p, gu_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 0, out.p);
                    if (R == 2) kqg::k_kq_gemv<false, 2><<<grid, 256>>>(dgu.p, gu_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 0, out.p);
                    if (R == 4) kqg::k_kq_gemv<false, 4><<<grid, 256>>>(dgu.p, gu_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 0, out.p);
                } else {
                    if (R == 1) kqg::k_kq_gemv<true, 1><<<grid, 256>>>(ddn.p, dn_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 1, out.p);
                    if (R == 2) kqg::k_kq_gemv<true, 2><<<grid, 256>>>(ddn.p, dn_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 1, out.p);
                    if (R == 4) kqg::k_kq_gemv<true, 4><<<grid, 256>>>(ddn.p, dn_es, ids, rows, K, a.q.p, a.d.p, a.s16.p, 1, out.p);
                }
            };
            Stats st = selftest ? Stats{} : c.tm.run(f, c.o.reps, c.o.trials);
            f(0);
            CK(cudaDeviceSynchronize());
            auto h = out.down(size_t(TOPK) * rows);
            ErrStats e = compare(h.data(), ref.data(), h.size()), eq = compare(h.data(), refq.data(), h.size());
            c.csv->row("gpu", gu ? "kq_q2k_dp4a" : "kq_q3k_dp4a", gu ? "gu8" : "down8", c.wl(), 1, "rows" + std::to_string(R), st,
                       TOPK * double(gu ? gu_es : dn_es), c.peak, 2.0 * TOPK * rows * double(K), e.rel_l2, e.max_abs, hash_floats(h.data(), h.size()));
            if (selftest) printf("     exactness vs fp64 on its quantized activations: relL2=%.2e %s\n", eq.rel_l2, eq.rel_l2 < 1e-5 ? "EXACT" : "MISMATCH");
        }
    }

    // Fused MoE layers
    auto x = make_activation(H, 77);
    DBuf<float> dx(H), dy(H), hd(TOPK * I / 32);
    DBuf<int8_t> hq(TOPK * I);
    DBuf<int> hs(TOPK * I / 16);
    dx.up(x);
    std::vector<float> rh(TOPK);
    CK(cudaMemcpy(rh.data(), c.route.p, TOPK * 4, cudaMemcpyDeviceToHost));
    for (int mix = 0; mix < 2; ++mix) {
        auto layer = [&](int s, cudaStream_t st) {
            if (mix == 0) {
                kqg::k_moe_gu_q2k<false><<<dim3(I / 32, TOPK), 256, 0, st>>>(dgu.p, gu_es, c.ids.p + 8 * s, dx.p, hq.p, hd.p, hs.p);
                kqg::k_moe_down_q3k<<<H / 32, 256, 0, st>>>(ddn.p, dn_es, c.ids.p + 8 * s, c.route.p + 8 * s, hq.p, hd.p, dy.p);
            } else {
                kqg::k_moe_gu_q2k<true><<<dim3(I / 32, TOPK), 256, 0, st>>>(dgu.p, gu_es, c.ids.p + 8 * s, dx.p, hq.p, hd.p, hs.p);
                k_moe_down<2><<<H / 16, 256, 0, st>>>(c.pool.dnq.p, c.pool.dne.p, c.pool.dnq_es, c.pool.dne_es, c.ids.p + 8 * s,
                                                      c.route.p + 8 * s, hq.p, hd.p, dy.p);
            }
        };
        std::vector<double> ref(H, 0.0);
        {
            std::vector<double> part(size_t(TOPK) * H);
#pragma omp parallel for
            for (int k = 0; k < TOPK; ++k) {
                std::vector<double> g(2 * I);
                for (int r = 0; r < 2 * I; ++r) g[r] = kgu[k].ref_dot(r, x.data());
                std::vector<float> h(I);
                for (int i = 0; i < I; ++i) h[i] = float(g[i] / (1 + std::exp(-g[i])) * g[I + i]);
                for (int r = 0; r < H; ++r) part[size_t(k) * H + r] = rh[k] * (mix == 0 ? kdn[k].ref_dot(r, h.data()) : ref_dot_row(c.pool.host[k].dn(), r, h.data()));
            }
            for (int k = 0; k < TOPK; ++k)
                for (int r = 0; r < H; ++r) ref[r] += part[size_t(k) * H + r];
        }
        const double bytes = TOPK * double(gu_es) + (mix == 0 ? TOPK * double(dn_es) : TOPK * double(H) * I * 0.53125);
        const char* name = mix == 0 ? "moe_q23_strata" : "moe_q2k_mxdown_ggmlorg";
        Stats st{};
        if (!selftest) st = c.tm.run([&](int it) { layer(it % c.nsets, 0); }, c.o.reps, c.o.trials);
        layer(0, 0);
        CK(cudaDeviceSynchronize());
        auto y = dy.down(H);
        ErrStats e = compare(y.data(), ref.data(), H);
        c.csv->row("gpu", name, "moe_layer", c.wl(), 1, "2launch", st, bytes, c.peak, 2.0 * TOPK * 3.0 * H * I, e.rel_l2, e.max_abs, hash_floats(y.data(), H));
        if (selftest) continue;
        cudaStream_t sm;
        CK(cudaStreamCreate(&sm));
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(sm, cudaStreamCaptureModeGlobal));
        for (int l = 0; l < mimo::kMoeLayers; ++l) layer(l % c.nsets, sm);
        CK(cudaStreamEndCapture(sm, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        Stats sg = c.tm.run([&](int) { CK(cudaGraphLaunch(ge, 0)); }, 5, c.o.trials);
        Stats per{sg.med / mimo::kMoeLayers, sg.p10 / mimo::kMoeLayers, sg.p90 / mimo::kMoeLayers, sg.min / mimo::kMoeLayers};
        c.csv->row("gpu", name, "moe47_graph", c.wl(), 1, "per_layer", per, bytes, c.peak, 2.0 * TOPK * 3.0 * H * I, 0, 0, 0);
        printf("     -> %s: 47 MoE layers %.2f ms => %.1f tok/s (expert part, all experts resident), %.1f MB per layer\n", name,
               sg.med / 1e3, 1e6 / sg.med, bytes / 1e6);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        cudaStreamDestroy(sm);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Ctx c;
    Opts& o = c.o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--weights") o.weights = next();
        else if (a == "--model") o.model = next();
        else if (a == "--layer") o.layer = std::atoi(next());
        else if (a == "--experts") o.experts = std::atoi(next());
        else if (a == "--mode") o.mode = next();
        else if (a == "--tokens") o.tokens = parse_list(next());
        else if (a == "--ncols") o.ncols = parse_list(next());
        else if (a == "--kinds") o.only_kinds = parse_list(next());
        else if (a == "--reps") o.reps = std::atoi(next());
        else if (a == "--trials") o.trials = std::atoi(next());
        else if (a == "--csv") o.csv = next();
        else {
            printf("usage: bench_gpu [--mode selftest|peak|decode|moe|prefill|kquant|all] [--weights dummy|real] [--model DIR]\n"
                   "                 [--layer L] [--experts P] [--tokens 512,2048] [--ncols 1,2,4] [--kinds 0,1,..]\n"
                   "                 [--reps R] [--trials T] [--csv FILE]\n");
            return a == "--help" ? 0 : 2;
        }
    }
    if (o.experts < TOPK || o.experts % TOPK || o.reps < 1 || o.trials < 1) {
        fprintf(stderr, "experts must be a positive multiple of 8; reps/trials must be positive\n");
        return 2;
    }
#ifdef FP4_AMPERE
    for (int kind : o.only_kinds) if (kind != 0 && kind != 2) {
        fprintf(stderr, "Ampere supports GEMM kinds 0 and 2 only\n"); return 2;
    }
#endif
    const bool selftest = o.mode == "selftest";
    if (selftest) { o.experts = std::min(o.experts, 16); o.ncols = {1, 2}; }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s, sm_%d%d, %d SMs, L2 %d MB, %.1f GB VRAM\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize >> 20, prop.totalGlobalMem / 1e9);

    {  // expert pool: P experts of one layer
        double t0 = now_s();
        std::unique_ptr<MimoCheckpoint> ck;
        if (o.weights == "real") ck = std::make_unique<MimoCheckpoint>(o.model);
        c.pool.host.resize(o.experts);
#pragma omp parallel for schedule(dynamic)
        for (int e = 0; e < o.experts; ++e)
            c.pool.host[e] = ck ? Expert() : make_dummy_expert(0x5eed0000ull + e);
        if (ck)
            for (int e = 0; e < o.experts; ++e) c.pool.host[e] = ck->load_expert(o.layer, e);
        c.pool.upload();
        printf("pool: %d experts (%s, layer %d) = %.2f GB on device, %.1f s\n", c.pool.P, o.weights.c_str(), o.layer,
               c.pool.P * c.pool.host[0].bytes() / 1e9, now_s() - t0);
    }
    c.nsets = c.pool.P / TOPK;
    {  // expert sets: set s = 8 distinct pool experts (a permutation), route weights sum to 1
        std::mt19937_64 rng(99);
        std::vector<int> perm(c.pool.P);
        for (int i = 0; i < c.pool.P; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int> ids(perm.begin(), perm.begin() + c.nsets * TOPK);
        std::vector<float> rw(ids.size());
        for (size_t i = 0; i < rw.size(); i += TOPK) {
            float sum = 0;
            for (int k = 0; k < TOPK; ++k) sum += (rw[i + k] = 0.2f + (rng() % 1000) / 1000.f);
            for (int k = 0; k < TOPK; ++k) rw[i + k] /= sum;
        }
        // references for decode/moe use pool.host[k] for set 0: keep set 0 = experts 0..7
        for (int k = 0; k < TOPK; ++k) {
            auto it = std::find(ids.begin(), ids.end(), k);
            std::swap(*it, ids[k]);
        }
        c.ids.up(ids);
        c.route.up(rw);
    }
    build_f8_tables();
    {
        DBuf<uint8_t> big(size_t(1) << 30);
        CK(cudaMemset(big.p, 1, big.n));
        DBuf<uint32_t> sink(1);
        Stats s = c.tm.run([&](int) { k_read_bw<<<prop.multiProcessorCount * 8, 512>>>((const uint4*)big.p, big.n / 16, sink.p); }, 5, 7);
        c.peak = big.n / (s.med * 1e-6) / 1e9;
        printf("read bandwidth (1 GiB stream): %.1f GB/s\n", c.peak);
    }
    Csv csv(selftest ? "" : o.csv);
    c.csv = &csv;
    if (selftest) {
        run_decode(c, true);
        run_moe(c, true);
        run_prefill(c, true);
        run_kquant(c, true);
        printf("selftest done: check relL2 column (q8/fp8/fp4 activations ~1e-2, fp16 ~1e-3, HIGH = broken)\n");
        return 0;
    }
    if (o.mode == "peak" || o.mode == "all") {
        printf("== tensor-core peak per instruction (register-only loop, %d SMs) ==\n", prop.multiProcessorCount);
        for (auto& r : measure_mma_peaks(prop.multiProcessorCount)) {
            printf("  %-36s %7.1f TFLOPS/TOPS (dense)\n", r.name, r.tops);
            c.csv->row("gpu", "mma_peak", r.name, "none", 0, "-", Stats{1, 1, 1, 1}, 0, 0, r.tops * 1e6, 0, 0, 0);
        }
    }
    if (o.mode == "decode" || o.mode == "all") run_decode(c, false);
    if (o.mode == "moe" || o.mode == "all") run_moe(c, false);
    if (o.mode == "prefill" || o.mode == "all") run_prefill(c, false);
    if (o.mode == "kquant" || o.mode == "all") run_kquant(c, false);
    return 0;
}
