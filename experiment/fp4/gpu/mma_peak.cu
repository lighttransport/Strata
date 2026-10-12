// Raw mma.sync throughput per instruction kind on this GPU (register-only, 4 independent accumulator chains
// per warp). Answers "which matrix units does the card have and how fast are they" with measurements.
#include "common.cuh"

#include <vector>

namespace {

template <int OP>
__global__ void __launch_bounds__(128) k_peak(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x ^ 0x5555u, 0x01010101u};
    uint32_t b[2] = {threadIdx.x * 7u, 0x02020202u};
    // keep E8M0 scales at 127 (x1) and operands small so nothing overflows
    if (OP >= 3) { a[0] &= 0x0f0f0f0fu; a[1] &= 0x0f0f0f0fu; a[2] &= 0x0f0f0f0fu; a[3] &= 0x0f0f0f0fu; b[0] &= 0x0f0f0f0fu; b[1] &= 0x0f0f0f0fu; }
    const uint32_t sc = 0x7f7f7f7fu;
    float c[4][4] = {};
    int ci[4][4] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (OP == 0)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(ci[k][0]), "+r"(ci[k][1]), "+r"(ci[k][2]), "+r"(ci[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (OP == 1)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (OP == 2)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (OP == 3)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %10, {0, 0};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sc));
            else if constexpr (OP == 4)
                asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %10, {0, 0};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sc));
            else if constexpr (OP == 5) {  // f16 inputs, f16 accumulate (2 regs of half2)
                uint32_t* h = (uint32_t*)c[k];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                             : "+r"(h[0]), "+r"(h[1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else if constexpr (OP == 6)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (OP == 7)  // e2m1 x e2m1 through the f8f6f4 path (no block scale)
                asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
    }
    float s = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int i = 0; i < 4; ++i) s += c[k][i] + float(ci[k][i]);
    if (s == 1234.5f) out[threadIdx.x] = s;
}

}  // namespace

struct PeakResult { const char* name; double tops; };

std::vector<PeakResult> measure_mma_peaks(int sms) {
    struct Op { const char* name; int mnk; void (*launch)(float*, int, int); };
    auto L = [](auto kern) { return kern; };
    (void)L;
    float* out;
    CK(cudaMalloc(&out, 4096));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<PeakResult> res;
    auto run = [&](const char* name, auto kern, double macs_per_mma) {
        const int blocks = sms * 8, iters = 4096;
        kern<<<blocks, 128>>>(out, 64);
        CK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int t = 0; t < 5; ++t) {
            CK(cudaEventRecord(a));
            kern<<<blocks, 128>>>(out, iters);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms;
            CK(cudaEventElapsedTime(&ms, a, b));
            best = std::min(best, double(ms));
        }
        CK(cudaGetLastError());
        const double ops = 2.0 * macs_per_mma * 4.0 * iters * (blocks * 4.0);
        res.push_back({name, ops / (best * 1e-3) / 1e12});
    };
    run("s8 m16n8k32 (int8 -> s32)", k_peak<0>, 16 * 8 * 32);
    run("e4m3 m16n8k32 (fp8 -> f32)", k_peak<1>, 16 * 8 * 32);
    run("f16 m16n8k16 (-> f32 acc)", k_peak<2>, 16 * 8 * 16);
    run("f16 m16n8k16 (-> f16 acc)", k_peak<5>, 16 * 8 * 16);
    run("bf16 m16n8k16 (-> f32 acc)", k_peak<6>, 16 * 8 * 16);
    run("f8f6f4 e2m1 m16n8k32 (no scale)", k_peak<7>, 16 * 8 * 32);
    run("mxf8f6f4 e2m1xe4m3 m16n8k32 1X", k_peak<3>, 16 * 8 * 32);
    run("mxf4 e2m1 m16n8k64 2X ue8m0", k_peak<4>, 16 * 8 * 64);
    cudaFree(out);
    return res;
}
