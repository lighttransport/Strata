#include "strata/kernels/glm_mailbox.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr size_t kFlagWords = 32;   // 128 bytes per slot: published at 0, completed at 16
constexpr size_t kControlWords = 64;
constexpr unsigned long long kTimeoutNs = 30ull * 1000 * 1000 * 1000;

size_t round_up(size_t n) { return (n + 255) & ~size_t(255); }

void ok(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("GLM mailbox ") + what + ": " + cudaGetErrorString(e));
}
__device__ unsigned long long now_ns() {
#if defined(STRATA_HIP_GFX906)
    return wall_clock64() * 40ull;   // gfx906: a 25 MHz wall clock
#elif defined(__HIPCC__)
    return wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}
__global__ void begin(const volatile unsigned *control, unsigned *generation) {
    *generation = control[0];
}
__global__ void publish(GlmMailboxView v, int slot, const int *ids, const float *weights, const float *x, int tokens,
                        const unsigned *generation, const int *wanted) {
    const int routes = tokens * v.top_k;
    int *out_ids = v.ids + (size_t)slot * v.max_tokens * v.top_k;
    float *out_weights = v.weights + (size_t)slot * v.max_tokens * v.top_k;
    for (int i = threadIdx.x; i < routes; i += blockDim.x) {
        out_ids[i] = ids[i];
        out_weights[i] = weights[i];
        if (wanted) v.wanted[(size_t)slot * v.max_tokens * v.top_k + i] = wanted[i];
    }
    const float4 *src = reinterpret_cast<const float4 *>(x);
    float4 *dst = reinterpret_cast<float4 *>(v.act + (size_t)slot * v.max_tokens * v.hidden);
    for (int i = threadIdx.x; i < tokens * v.hidden / 4; i += blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile unsigned *)(v.flags + (size_t)slot * kFlagWords) = *generation;
    }
}
// Returns false when the step was aborted or timed out. Thread 0 polls; the block then proceeds together.
__device__ bool wait_done(const GlmMailboxView &v, int slot, const unsigned *generation) {
    __shared__ int ready;
    if (threadIdx.x == 0) {
        const unsigned target = *generation;
        const volatile unsigned *done = v.flags + (size_t)slot * kFlagWords + 16;
        const volatile unsigned *abort = v.control + 32;
        const unsigned long long start = now_ns();
        int status = 1;
        // Poll gently: each read is a PCIe round trip that the CPU's memory system must answer.
        while ((int)(*done - target) < 0) {
            __nanosleep(256);
            if (*abort) { status = 0; break; }
            if (now_ns() - start > kTimeoutNs) {
                atomicCAS(v.control + 16, 0u, (unsigned)slot + 1);
                status = 0;
                break;
            }
        }
        __threadfence_system();
        ready = status;
    }
    __syncthreads();
    return ready;
}
__global__ void wait_add(GlmMailboxView v, int slot, float *out, int n, const unsigned *generation) {
    if (!wait_done(v, slot, generation)) return;
    const volatile float *sum = v.sum + (size_t)slot * v.max_tokens * v.hidden;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) out[i] += sum[i];
}
__global__ void wait_add_resident(GlmMailboxView v, int slot, float *out, int tokens, const unsigned *generation,
                                  const int *ids, const float *weights, const unsigned long long *lookup,
                                  const float *resident) {
    __shared__ float route_weight[64];
    if (!wait_done(v, slot, generation)) return;
    const int routes = tokens * v.top_k;
    for (int j = threadIdx.x; j < routes; j += blockDim.x) route_weight[j] = lookup[ids[j]] ? weights[j] : 0.f;
    __syncthreads();
    const volatile float *sum = v.sum + (size_t)slot * v.max_tokens * v.hidden;
    for (int i = threadIdx.x; i < tokens * v.hidden; i += blockDim.x) {
        const int t = i / v.hidden, row = i % v.hidden;
        float acc = sum[i];
        for (int k = 0; k < v.top_k; ++k) {
            const int j = t * v.top_k + k;
            if (route_weight[j] != 0.f) acc += route_weight[j] * resident[(size_t)j * v.hidden + row];
        }
        out[i] += acc;
    }
}
// Resident routes' rows go to the mailbox (posted writes), then the rows flag: the CPU sums every route.
__global__ void post_rows(GlmMailboxView v, int slot, const int *ids, const float *weights,
                          const unsigned long long *lookup, const float *resident, int tokens) {
    const int routes = tokens * v.top_k;
    float4 *rows = reinterpret_cast<float4 *>(v.rows + (size_t)slot * v.max_tokens * v.top_k * v.hidden);
    const float4 *src = reinterpret_cast<const float4 *>(resident);
    const int quads = v.hidden / 4;
    for (int j = blockIdx.x; j < routes; j += gridDim.x) {
        if (!lookup[ids[j]] || weights[j] == 0.f) continue;
        for (int i = threadIdx.x; i < quads; i += blockDim.x) rows[(size_t)j * quads + i] = src[(size_t)j * quads + i];
    }
}
__global__ void rows_ready(GlmMailboxView v, int slot, const unsigned *generation) {
    __threadfence_system();
    *(volatile unsigned *)(v.flags + (size_t)slot * kFlagWords + 8) = *generation;
}
} // namespace

size_t glm_mailbox_bytes(int slots, int max_tokens, int top_k, int hidden, bool with_rows, bool with_wanted) {
    const size_t routes = (size_t)slots * max_tokens * top_k, rows = (size_t)slots * max_tokens * hidden;
    return round_up(slots * kFlagWords * 4) + round_up(kControlWords * 4) + round_up(routes * 4) * 2 +
           round_up(rows * 4) * 2 + (with_rows ? round_up(rows * top_k * 4) : 0) + (with_wanted ? round_up(routes * 4) : 0);
}

GlmMailboxView glm_mailbox_view(void *base, int slots, int max_tokens, int top_k, int hidden, bool with_rows,
                                bool with_wanted) {
    if (slots < 1 || max_tokens < 1 || top_k < 1 || hidden < 4 || hidden % 4)
        throw std::invalid_argument("GLM mailbox: invalid geometry");
    auto *p = static_cast<char *>(base);
    const size_t routes = (size_t)slots * max_tokens * top_k, rows = (size_t)slots * max_tokens * hidden;
    GlmMailboxView v;
    v.flags = reinterpret_cast<unsigned *>(p); p += round_up(slots * kFlagWords * 4);
    v.control = reinterpret_cast<unsigned *>(p); p += round_up(kControlWords * 4);
    v.ids = reinterpret_cast<int *>(p); p += round_up(routes * 4);
    v.weights = reinterpret_cast<float *>(p); p += round_up(routes * 4);
    v.act = reinterpret_cast<float *>(p); p += round_up(rows * 4);
    v.sum = reinterpret_cast<float *>(p); p += round_up(rows * 4);
    if (with_rows) { v.rows = reinterpret_cast<float *>(p); p += round_up(rows * top_k * 4); }
    if (with_wanted) v.wanted = reinterpret_cast<int *>(p);
    v.slots = slots; v.max_tokens = max_tokens; v.top_k = top_k; v.hidden = hidden;
    return v;
}

void glm_mailbox_begin(GlmMailboxView v, unsigned *generation, void *stream) {
    begin<<<1, 1, 0, (cudaStream_t)stream>>>(v.control, generation);
    ok(cudaGetLastError(), "begin");
}

void glm_mailbox_publish(GlmMailboxView v, int slot, const int *ids, const float *weights, const float *x,
                         int tokens, const unsigned *generation, void *stream, const int *wanted) {
    if (slot < 0 || slot >= v.slots || tokens < 1 || tokens > v.max_tokens || (wanted && !v.wanted))
        throw std::invalid_argument("GLM mailbox: invalid publish");
    publish<<<1, 512, 0, (cudaStream_t)stream>>>(v, slot, ids, weights, x, tokens, generation, wanted);
    ok(cudaGetLastError(), "publish");
}

void glm_mailbox_wait_add(GlmMailboxView v, int slot, float *out, int tokens, const unsigned *generation,
                          void *stream) {
    if (slot < 0 || slot >= v.slots || tokens < 1 || tokens > v.max_tokens)
        throw std::invalid_argument("GLM mailbox: invalid wait");
    wait_add<<<1, 1024, 0, (cudaStream_t)stream>>>(v, slot, out, tokens * v.hidden, generation);
    ok(cudaGetLastError(), "wait");
}

void glm_mailbox_wait_add_resident(GlmMailboxView v, int slot, float *out, int tokens, const unsigned *generation,
                                   const int *ids, const float *weights, const unsigned long long *lookup,
                                   const float *resident, void *stream) {
    if (slot < 0 || slot >= v.slots || tokens < 1 || tokens > v.max_tokens || tokens * v.top_k > 64)
        throw std::invalid_argument("GLM mailbox: invalid resident wait");
    wait_add_resident<<<1, 1024, 0, (cudaStream_t)stream>>>(v, slot, out, tokens, generation, ids, weights, lookup,
                                                             resident);
    ok(cudaGetLastError(), "resident wait");
}

void glm_mailbox_post_rows(GlmMailboxView v, int slot, const int *ids, const float *weights,
                           const unsigned long long *lookup, const float *resident, int tokens,
                           const unsigned *generation, void *stream) {
    if (slot < 0 || slot >= v.slots || tokens < 1 || tokens > v.max_tokens || !v.rows || !lookup || !resident)
        throw std::invalid_argument("GLM mailbox: invalid resident rows");
    post_rows<<<tokens * v.top_k, 256, 0, (cudaStream_t)stream>>>(v, slot, ids, weights, lookup, resident, tokens);
    rows_ready<<<1, 1, 0, (cudaStream_t)stream>>>(v, slot, generation);
    ok(cudaGetLastError(), "resident rows");
}

} // namespace strata::kernels
