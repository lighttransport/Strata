// Tensor-core grouped GEMM kernels over MXFP4 weights (A = 16 weight rows per warp, B = tokens).
// Built for sm_120a. Kinds 0..2 are "software" paths that only need sm_80 (int8 mma) or sm_89 (fp8 mma)
// and decode E2M1 in registers; kinds 3..5 use the Blackwell block-scaled instructions.
//
//   0 I8      : E2M1x2 -> int8 (PRMT LUT), q8 activations, mma m16n8k32 s8, per-block float epilogue
//   1 F8FOLD  : E2M1 * 2^(e - e_rowmax) -> e4m3 (exact while the row's exponent spread is small),
//               per-token e4m3 activations, mma m16n8k32 e4m3, ONE epilogue per row at the end
//   2 F16FOLD : E2M1 * 2^(e-127) -> fp16 (always exact for e in [113,142]), fp16 activations,
//               mma m16n8k16 f16, no epilogue
//   3 MXF8    : kind::mxf8f6f4 scale_vec::1X, e2m1 weights (8-bit containers) x MXFP8 activations
//   4 MXF4    : kind::mxf4 scale_vec::2X m16n8k64, raw checkpoint bytes x MXFP4 activations
//   5 MSD     : kind::mxf4 twice, activation split into MXFP4 main + MXFP4 residual (arXiv 2605.13915 idea)
//
// K order: inside each 32-block the 8-bit kinds use "logical" order (evens then odds, see common/mxfp4.h)
// so one 32-bit load of packed weights splits into the two A registers; the fp16 kind uses its own order
// (f16_logical_to_phys in bench_gpu.cu); the fp4 kinds use the natural order.
#include "common.cuh"

namespace {

// Main loop: CTA = 8 warps = 128 weight rows (16 per warp) x BN tokens of one expert tile. K is consumed in
// chunks of KC = 128 (4 MX blocks); raw packed weights, their E8M0 bytes, the activation chunk and its scales
// go through a 3-stage cp.async pipeline in shared memory. Fragments are decoded from shared memory.
constexpr int STAGES = 3, WARPS = 8, BM = 16 * WARPS;
// K chunk: 256 where the smem budget allows (kind::mxf4, or any kind with BN <= 16), else 128.
// Longer chunks halve the barriers per K and matter most when the GEMM is bandwidth bound (small BN).
template <int KIND, int BN> constexpr int kc_of() { return (KIND == 4 || BN <= 16) ? 256 : 128; }

template <int KIND, int BN>
struct Layout {
    static constexpr int KC = kc_of<KIND, BN>();
    static constexpr int A_ROW = KC / 2 + 16;  // packed bytes per weight row per chunk (+16: conflict-free u32 reads)
    static constexpr int b_payload = (KIND == 2) ? KC * 2 : (KIND >= 4 ? KC / 2 : KC);
    static constexpr int b_row = b_payload + 16;
    static constexpr int parts = (KIND == 5) ? 2 : 1;
    static constexpr int s_bytes = (KIND == 0) ? 4 * (KC / 32) : (KIND >= 3 ? KC / 32 : 0);  // per token per chunk
};
template <int KIND, int BN>
struct Smem {
    using L = Layout<KIND, BN>;
    static constexpr int a_q = BM * L::A_ROW, a_e = BM * (L::KC / 32);
    static constexpr int b_q = L::parts * BN * L::b_row, b_s = L::parts * BN * (L::s_bytes ? L::s_bytes : 4);
    static constexpr int stage = a_q + a_e + b_q + b_s;
    static constexpr int total = STAGES * stage;
};

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(d), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifndef FP4_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("trap;"); // Unsupported kinds are rejected by the host dispatcher.
#endif
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_mxf8f6f4_e2m1_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                        uint32_t sa, uint32_t sb) {
#ifndef FP4_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    asm volatile("trap;"); // Unsupported kinds are rejected by the host dispatcher.
#endif
}
__device__ __forceinline__ void mma_mxf4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                         uint32_t sb) {
#ifndef FP4_AMPERE
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    asm volatile("trap;"); // Unsupported kinds are rejected by the host dispatcher.
#endif
}

__device__ __forceinline__ uint32_t bytemax4(uint32_t v) {
    v = __vmaxu4(v, v >> 16);
    v = __vmaxu4(v, v >> 8);
    return v & 0xff;
}

// e4m3 tables for E2M1 * 2^-d, d = 0..15 (built on the host, exact incl. subnormals; underflow -> 0).
__constant__ uint32_t c_f8tab[16][4];

template <int KIND, int BN, int SH>
__global__ void __launch_bounds__(256) k_gemm(GemmArgs a) {
    using L = Layout<KIND, BN>;
    using S = Smem<KIND, BN>;
    constexpr int KC = L::KC, A_ROW = L::A_ROW;
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ uint32_t sTab[16][4];

    // blockIdx.x walks the row tiles of one (expert, token tile) so its activations and the expert's weights
    // stay in L2; blockIdx.y walks the token tiles (sorted by expert).
    const Tile t = a.tiles[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int rowt = blockIdx.x * BM;
    const int K = a.K, nb = K / 32, nchunks = K / KC;
    const uint8_t* q = a.wq + size_t(t.expert) * a.wq_es + size_t(rowt) * (K / 2);
    const uint8_t* e = a.we + size_t(t.expert) * a.we_es + size_t(rowt) * nb;

    auto stage_ptr = [&](int st) { return smem + st * S::stage; };
    auto issue = [&](int c) {
        uint8_t* p = stage_ptr(c % STAGES);
        uint8_t* sAq = p;
        uint8_t* sAe = p + S::a_q;
        uint8_t* sBq = sAe + S::a_e;
        uint8_t* sBs = sBq + S::b_q;
        // A: BM rows x KC/2 bytes
        for (int i = threadIdx.x; i < BM * (KC / 32); i += 256) {
            const int r = i / (KC / 32), part = i % (KC / 32);
            cp16(sAq + r * A_ROW + part * 16, q + size_t(r) * (K / 2) + c * (KC / 2) + part * 16, true);
        }
        for (int i = threadIdx.x; i < BM * (KC / 128); i += 256) {
            const int r = i / (KC / 128), w = i % (KC / 128);
            cp4(sAe + r * (KC / 32) + 4 * w, e + size_t(r) * nb + c * (KC / 32) + 4 * w, true);
        }
#pragma unroll
        for (int pp = 0; pp < L::parts; ++pp) {
            const uint8_t* src = (const uint8_t*)(pp == 0 ? a.act : a.act2);
            constexpr int v16 = L::b_payload / 16;
            const size_t tok_bytes = size_t(K) * L::b_payload / KC;
            for (int i = threadIdx.x; i < BN * v16; i += 256) {
                const int n = i / v16, part = i % v16;
                const bool ok = n < t.ntok;
                cp16(sBq + (pp * BN + n) * L::b_row + part * 16,
                     src + size_t(t.act_tok0 + (ok ? n : 0)) * tok_bytes + size_t(c) * L::b_payload + part * 16, ok);
            }
            if constexpr (L::s_bytes > 0) {
                const uint8_t* ss = (const uint8_t*)(pp == 0 ? a.act_s : a.act2_s);
                const size_t tok_s = size_t(nb) * L::s_bytes / (KC / 32);
                constexpr int w4 = L::s_bytes / 4;
                for (int i = threadIdx.x; i < BN * w4; i += 256) {
                    const int n = i / w4, w = i % w4;
                    const bool ok = n < t.ntok;
                    cp4(sBs + (pp * BN + n) * L::s_bytes + 4 * w,
                        ss + size_t(t.act_tok0 + (ok ? n : 0)) * tok_s + size_t(c) * L::s_bytes + 4 * w, ok);
                }
            }
        }
    };

#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < nchunks) issue(st);
        cp_commit();
    }

    // warp grid: (8/WN) warps along M x WN along N; each warp owns MT = WN m16 tiles and NTW n8 tiles
    constexpr int WN = BN >= 32 ? 2 : 1, MT = WN, NTW = BN / WN / 8;
    const int wm = warp / WN, wn = warp % WN;
    const int wr0 = wm * 16 * MT;          // first weight row of this warp in the CTA
    const int tb = wn * (BN / WN);         // first token of this warp in the tile
    uint32_t rowmax[MT][2] = {};
    if constexpr (KIND == 1) {
        if (threadIdx.x < 64) sTab[threadIdx.x >> 2][threadIdx.x & 3] = c_f8tab[threadIdx.x >> 2][threadIdx.x & 3];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint8_t* er = e + size_t(wr0 + 16 * mt + g + 8 * r) * nb;
                uint32_t m = 0;
                for (int i = tq * 16; i < nb; i += 64) {
                    uint4 v = *(const uint4*)(er + i);
                    m = __vmaxu4(m, __vmaxu4(__vmaxu4(v.x, v.y), __vmaxu4(v.z, v.w)));
                }
                m = bytemax4(m);
                m = max(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = max(m, __shfl_xor_sync(0xffffffffu, m, 2));
                rowmax[mt][r] = m;
            }
    }

    float acc[MT][NTW][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;

    for (int c = 0; c < nchunks; ++c) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (c + STAGES - 1 < nchunks) issue(c + STAGES - 1);
        cp_commit();

        const uint8_t* p = stage_ptr(c % STAGES);
        const uint8_t* sAq = p;
        const uint8_t* sAe = p + S::a_q;
        const uint8_t* sBq = sAe + S::a_e;
        const uint8_t* sBs = sBq + S::b_q;
        // B fragments via ldmatrix.x4: lane supplies the address of row (lane&7) of matrix (lane>>3)
        auto ldsm4 = [&](const uint8_t* addr, uint32_t (&r)[4]) {
            const uint32_t sa = (uint32_t)__cvta_generic_to_shared(addr);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
        };
        auto brow = [&](int pp, int n) { return sBq + (pp * BN + n) * L::b_row; };
        const int lm = lane >> 3, lr = lane & 7;

#pragma unroll
        for (int b = 0; b < KC / 32; ++b) {
            if constexpr (KIND == 4 || KIND == 5) { if (b & 1) continue; }
            // A fragments for MT m-tiles
            uint32_t A[MT][4], A2[MT][4];
            float w[MT][2];
            uint32_t sa[MT];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t raw[2], eb[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = wr0 + 16 * mt + g + 8 * r;
                    raw[r] = *(const uint32_t*)(sAq + row * A_ROW + b * 16 + 4 * tq);
                    eb[r] = sAe[row * (KC / 32) + b];
                }
                if constexpr (KIND == 0) {
                    const int2 v0 = lut16(raw[0], LUT_I8), v1 = lut16(raw[1], LUT_I8);
                    A[mt][0] = v0.x; A[mt][1] = v1.x; A[mt][2] = v0.y; A[mt][3] = v1.y;
                    w[mt][0] = e8m0f(eb[0]); w[mt][1] = e8m0f(eb[1]);
                } else if constexpr (KIND == 1) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const uint32_t d = min(rowmax[mt][r] - eb[r], 15u);
                        const int2 v = lut16(raw[r], sTab[d][0], sTab[d][1], sTab[d][2], sTab[d][3]);
                        A[mt][r] = v.x; A[mt][2 + r] = v.y;
                    }
                } else if constexpr (KIND == 2) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const __half2 hs = __float2half2_rn(e8m0f(eb[r]));
                        const int2 v = lut16(raw[r], LUT_F16HI);
                        auto mk = [&](uint32_t src, uint32_t sel) {
                            uint32_t h = __byte_perm(src, src, sel) & 0xff00ff00u;
                            __half2 x = __hmul2(*(__half2*)&h, hs);
                            return *(uint32_t*)&x;
                        };
                        A[mt][r] = mk(v.x, 0x1100); A[mt][2 + r] = mk(v.x, 0x3322);
                        A2[mt][r] = mk(v.y, 0x1100); A2[mt][2 + r] = mk(v.y, 0x3322);
                    }
                } else if constexpr (KIND == 3) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        A[mt][r] = (raw[r] & 0x0f0f0f0fu) << SH;
                        A[mt][2 + r] = ((raw[r] >> 4) & 0x0f0f0f0fu) << SH;
                    }
                    sa[mt] = (lane & 1) ? eb[1] : eb[0];
                } else {  // mxf4: blocks b and b+1 form one k64 step
                    uint32_t raw2[2], eb2[2];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int row = wr0 + 16 * mt + g + 8 * r;
                        raw2[r] = *(const uint32_t*)(sAq + row * A_ROW + (b + 1) * 16 + 4 * tq);
                        eb2[r] = sAe[row * (KC / 32) + b + 1];
                    }
                    A[mt][0] = raw[0]; A[mt][1] = raw[1]; A[mt][2] = raw2[0]; A[mt][3] = raw2[1];
                    sa[mt] = (lane & 1) ? (eb[1] | (eb2[1] << 8)) : (eb[0] | (eb2[0] << 8));
                }
            }
            // B fragments + mma
#pragma unroll
            for (int pp = 0; pp < L::parts; ++pp) {
#pragma unroll
                for (int j = 0; j < NTW; j += (KIND == 2 ? 1 : 2)) {
                    uint32_t Bf[4];
                    const int n0 = tb + 8 * j;
                    if constexpr (KIND == 2) {
                        ldsm4(brow(0, n0 + lr) + b * 64 + lm * 16, Bf);
                    } else if constexpr (KIND >= 4) {
                        const int jj = (NTW > 1) ? (lm >> 1) : 0;
                        ldsm4(brow(pp, n0 + 8 * jj + lr) + (b + (lm & 1)) * 16, Bf);
                    } else {
                        const int jj = (NTW > 1) ? (lm >> 1) : 0;
                        ldsm4(brow(0, n0 + 8 * jj + lr) + b * 32 + (lm & 1) * 16, Bf);
                    }
#pragma unroll
                    for (int h = 0; h < (KIND == 2 ? 1 : (NTW > 1 ? 2 : 1)); ++h) {
                        const int jt = j + h;
                        const uint32_t b0 = Bf[2 * h], b1 = Bf[2 * h + 1];
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            if constexpr (KIND == 0) {
                                int C[4] = {0, 0, 0, 0};
                                const uint32_t Am[4] = {A[mt][0], A[mt][1], A[mt][2], A[mt][3]};
                                mma_s8(C, Am, b0, b1);
                                const float sa0 = ((const float*)(sBs + (n0 + 8 * h + 2 * tq) * L::s_bytes))[b];
                                const float sa1 = ((const float*)(sBs + (n0 + 8 * h + 2 * tq + 1) * L::s_bytes))[b];
                                acc[mt][jt][0] += float(C[0]) * (w[mt][0] * sa0);
                                acc[mt][jt][1] += float(C[1]) * (w[mt][0] * sa1);
                                acc[mt][jt][2] += float(C[2]) * (w[mt][1] * sa0);
                                acc[mt][jt][3] += float(C[3]) * (w[mt][1] * sa1);
                            } else if constexpr (KIND == 1) {
                                const uint32_t Am[4] = {A[mt][0], A[mt][1], A[mt][2], A[mt][3]};
                                mma_e4m3(acc[mt][jt], Am, b0, b1);
                            } else if constexpr (KIND == 2) {
                                const uint32_t Am[4] = {A[mt][0], A[mt][1], A[mt][2], A[mt][3]};
                                const uint32_t Am2[4] = {A2[mt][0], A2[mt][1], A2[mt][2], A2[mt][3]};
                                mma_f16(acc[mt][jt], Am, Bf[0], Bf[1]);
                                mma_f16(acc[mt][jt], Am2, Bf[2], Bf[3]);
                            } else if constexpr (KIND == 3) {
                                const uint32_t Am[4] = {A[mt][0], A[mt][1], A[mt][2], A[mt][3]};
                                const uint32_t sb = sBs[(n0 + 8 * h + g) * L::s_bytes + b];
                                mma_mxf8f6f4_e2m1_e4m3(acc[mt][jt], Am, b0, b1, sa[mt], sb);
                            } else {
                                const uint32_t Am[4] = {A[mt][0], A[mt][1], A[mt][2], A[mt][3]};
                                const uint32_t sb = *(const uint16_t*)(sBs + (pp * BN + n0 + 8 * h + g) * L::s_bytes + b);
                                mma_mxf4(acc[mt][jt], Am, b0, b1, sa[mt], sb);
                            }
                        }
                    }
                }
            }
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int n = tb + 8 * j + 2 * tq + (i & 1);
                const int r = i >> 1;
                if (n >= t.ntok) continue;
                float v = acc[mt][j][i];
                if constexpr (KIND == 0) v *= 0.5f;
                if constexpr (KIND == 1) v *= e8m0f(rowmax[mt][r]) * ((const float*)a.act_s)[t.act_tok0 + n];
                a.out[size_t(t.out_slot0 + n) * a.rows + rowt + wr0 + 16 * mt + g + 8 * r] = v;
            }
}

template <int KIND, int BN, int SH>
void launch_one(const GemmArgs& a, cudaStream_t s) {
    constexpr int smem = Smem<KIND, BN>::total;
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(k_gemm<KIND, BN, SH>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        init = true;
    }
    k_gemm<KIND, BN, SH><<<dim3(a.rows / BM, a.ntiles), 256, smem, s>>>(a);
}

template <int KIND, int SH>
void launch_bn(int bn, const GemmArgs& a, cudaStream_t s) {
    switch (bn) {
        case 8: launch_one<KIND, 8, SH>(a, s); break;
        case 16: launch_one<KIND, 16, SH>(a, s); break;
        case 32: launch_one<KIND, 32, SH>(a, s); break;
        case 64: launch_one<KIND, 64, SH>(a, s); break;
        default: fprintf(stderr, "bad bn %d\n", bn); exit(2);
    }
}

}  // namespace

void set_f8_tables(const uint32_t* tab64) { CK(cudaMemcpyToSymbol(c_f8tab, tab64, 64 * 4)); }

// kind 0..5 as documented above; container_shift only matters for kind 3.
void launch_gemm(int kind, int bn, int container_shift, const GemmArgs& a, cudaStream_t s) {
    switch (kind) {
        case 0: launch_bn<0, 0>(bn, a, s); break;
#ifndef FP4_AMPERE
        case 1: launch_bn<1, 0>(bn, a, s); break;
#endif
        case 2: launch_bn<2, 0>(bn, a, s); break;
#ifndef FP4_AMPERE
        case 3: container_shift ? launch_bn<3, 2>(bn, a, s) : launch_bn<3, 0>(bn, a, s); break;
        case 4: launch_bn<4, 0>(bn, a, s); break;
        case 5: launch_bn<5, 0>(bn, a, s); break;
#endif
        default: fprintf(stderr, "unsupported GEMM kind %d in this build\n", kind); exit(2);
    }
    CK(cudaGetLastError());
}
