#include "strata/kernels/glm_prefill.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels {
namespace {
__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
void check() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void check_expert_mask(unsigned mask, int groups) {
    if (!mask) return;
    int count = 0;
    for (unsigned bits = mask; bits; bits &= bits - 1) ++count;
    if ((mask & ~0xffffu) || count != groups)
        throw std::invalid_argument("GLM: invalid expert bucket mask");
}
__global__ void half_cast(const float *x, uint16_t *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[i] = __half_as_ushort(__float2half(x[i]));
}
__device__ uint16_t bf16_bits(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 64u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__global__ void bf16_cast(const float *x, uint16_t *y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_bits(x[i]);
}
__global__ void gather_half(const float *x, uint16_t *y, const int *routes, int begin,
                            int rows, int width) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * width)
        y[i] = __half_as_ushort(__float2half(x[(int64_t)routes[begin + i / width] * width + i % width]));
}
__global__ void resident_routes(const int *ids, const unsigned long long *lookup,
                                unsigned long long *ptr, int *starts, int *dest,
                                int *tokens, int *count) {
    int n = 0;
    starts[0] = 0;
    for (int j = 0; j < 8; ++j) {
        const auto address = lookup[ids[j]];
        if (!address) continue;
        ptr[n] = address;
        dest[n] = j; tokens[n] = 0;
        ++n; starts[n] = n;
    }
    *count = n;
}
__global__ void resident_routes_batch(const int *ids, const unsigned long long *lookup,
                                      unsigned long long *ptr, int *starts, int *dest,
                                      int *tokens, int *count, int nt) {
    int groups = 0;
    starts[0] = 0;
    for (int j = 0; j < nt * 8; ++j) {
        const auto address = lookup[ids[j]];
        if (!address) continue;
        int g = 0;
        while (g < groups && ptr[g] != address) ++g;
        if (g == groups) { ptr[groups++] = address; starts[g + 1] = 0; }
        ++starts[g + 1];
    }
    int cursor[64];
    for (int g = 0; g < groups; ++g) { starts[g + 1] += starts[g]; cursor[g] = starts[g]; }
    for (int j = 0; j < nt * 8; ++j) {
        const auto address = lookup[ids[j]];
        if (!address) continue;
        int g = 0;
        while (ptr[g] != address) ++g;
        const int slot = cursor[g]++;
        dest[slot] = j; tokens[slot] = j / 8;
    }
    *count = groups;
}
__global__ void moe_reduce_batch(const float *cpu, const float *primary, const float *remote,
                                 const int *owners, const float *weights, float *out, int width, int nt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * nt) return;
    const int t = i / width, col = i % width;
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) {
        const int route = t * 8 + j;
        const float *source = owners[route] == 1 ? primary : owners[route] == 2 ? remote : cpu;
        sum = __fadd_rn(sum, __fmul_rn(weights[route], source[route * width + col]));
    }
    out[i] = __fadd_rn(out[i], sum);
}
__global__ void moe_reduce(const float *cpu, const float *primary, const float *remote,
                           const int *owners, const float *weights, float *out, int width) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) {
        const float *source = owners[j] == 1 ? primary : owners[j] == 2 ? remote : cpu;
        sum = __fadd_rn(sum, __fmul_rn(weights[j], source[j * width + col]));
    }
    out[col] = __fadd_rn(out[col], sum);
}
__global__ void scatter_rows(const float *x, float *y, const int *routes, int begin,
                             int rows, int width) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * width)
        y[(int64_t)routes[begin + i / width] * width + i % width] = x[i];
}
__device__ int masked_expert(int first, unsigned mask, int group) {
    if (!mask) return first + group;
    for (int j = 0; j < group; ++j) mask &= mask - 1;
    return first + __ffs(mask) - 1;
}
__global__ void gather_expert_half(const float *x, uint16_t *y, const int *bounds, const int *source,
                                  int first, int offset, int rows, int width, unsigned mask) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * width) return;
    const int expert = masked_expert(first, mask, blockIdx.y), row = i / width + offset;
    const int begin = bounds[expert], end = bounds[expert + 1];
    y[(int64_t)blockIdx.y * rows * width + i] = row < end - begin
        ? __half_as_ushort(__float2half(x[(int64_t)source[begin + row] * width + i % width])) : 0;
}
__global__ void gather_expert_bf16(const float *x, uint16_t *y, const int *bounds, const int *source,
                                  int first, int offset, int rows, int width, unsigned mask) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * width) return;
    const int expert = masked_expert(first, mask, blockIdx.y), row = i / width + offset;
    const int begin = bounds[expert], end = bounds[expert + 1];
    y[(int64_t)blockIdx.y * rows * width + i] = row < end - begin
        ? bf16_bits(x[(int64_t)source[begin + row] * width + i % width]) : 0;
}
// The same gather from rows already rounded to FP16 (bit-identical to rounding here).
__global__ void gather_expert_half_input(const uint16_t *x, uint16_t *y, const int *bounds, const int *source,
                                         int first, int offset, int rows, int width, unsigned mask) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * width) return;
    const int expert = masked_expert(first, mask, blockIdx.y), row = i / width + offset;
    const int begin = bounds[expert], end = bounds[expert + 1];
    y[(int64_t)blockIdx.y * rows * width + i] = row < end - begin
        ? x[(int64_t)source[begin + row] * width + i % width] : 0;
}
__global__ void scatter_expert_rows(const float *x, float *y, const int *bounds, const int *dest,
                                    int first, int offset, int rows, int width, unsigned mask) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * width) return;
    const int expert = masked_expert(first, mask, blockIdx.y), row = i / width + offset;
    if (row < bounds[expert + 1] - bounds[expert])
        y[(int64_t)dest[bounds[expert] + row] * width + i % width] = x[(int64_t)blockIdx.y * rows * width + i];
}
__global__ void copy_route_rows(const float *source, float *dest, const int *routes, int begin,
                               int rows, int width, bool gather_rows) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * width) return;
    const int row = begin + i / width, column = i % width;
    const int64_t route = (int64_t)routes[row] * width + column;
    const int64_t grouped = (int64_t)row * width + column;
    dest[gather_rows ? grouped : route] = source[gather_rows ? route : grouped];
}
__global__ void convolution(const float *x, const float *w, const float *history, float *y, int n, int kernel,
                            int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, c = i % n;
    float sum = x[i] * w[c * kernel + kernel - 1];
    for (int j = 0; j < kernel - 1; ++j) {
        int previous = t - (kernel - 1) + j;
        float v = previous < 0 ? history[c * (kernel - 1) + previous + kernel - 1] : x[previous * n + c];
        sum += v * w[c * kernel + j];
    }
    y[i] = sum * sigmoid(sum);
}
__global__ void conv_history(const float *x, float *history, int n, int kernel, int tokens) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n)
        return;
    for (int j = 0; j < kernel - 1; ++j) {
        int t = tokens - (kernel - 1) + j;
        history[c * (kernel - 1) + j] = t < 0 ? history[c * (kernel - 1) + t + kernel - 1] : x[t * n + c];
    }
}
__global__ void routing(const float *l, const float *bias, int *ids, float *w, int ne, int top, float scale,
                        const float *bonus) {
    int t = blockIdx.x;
    l += t * ne;
    ids += t * top;
    w += t * top;
    float sum = 0;
    for (int j = 0; j < top; ++j) {
        float best = -INFINITY;
        int chosen = -1;
        for (int e = 0; e < ne; ++e) {
            bool used = false;
            for (int z = 0; z < j; ++z)
                used |= ids[z] == e;
            float score = sigmoid(l[e]) + bias[e];
            if (bonus) score += bonus[e];   // selection only; weights below keep the true probabilities
            if (!used && score > best) {
                best = score;
                chosen = e;
            }
        }
        ids[j] = chosen;
        w[j] = sigmoid(l[chosen]);
        sum += w[j];
    }
    for (int j = 0; j < top; ++j)
        w[j] *= scale / (sum + 1e-20f);
}
__global__ void count_routes(const int *ids, int *b, int count) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count)
        atomicAdd(b + ids[i] + 1, 1);
}
__global__ void prefix_routes(int *b, int *c, int ne) {
    for (int e = 1; e <= ne; ++e)
        b[e] += b[e - 1];
    for (int e = 0; e < ne; ++e)
        c[e] = b[e];
}
__global__ void scatter_routes(const int *ids, int *c, int *d, int *src, int top, int count) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count) {
        int at = atomicAdd(c + ids[i], 1);
        d[at] = i;
        src[at] = i / top;
    }
}
__global__ void route_block_counts(const int *ids, int *counts, int experts, int routes) {
    extern __shared__ int histogram[];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) histogram[e] = 0;
    __syncthreads();
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route < routes) atomicAdd(histogram + ids[route], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        counts[blockIdx.x * experts + e] = histogram[e];
}
__global__ void route_block_prefix(int *counts, int *bounds, int experts, int blocks) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    int total = 0;
    for (int block = 0; block < blocks; ++block) {
        const int index = block * experts + e, count = counts[index];
        counts[index] = total;
        total += count;
    }
    bounds[e + 1] = total;
}
__global__ void stable_route_scatter(const int *ids, const int *bounds, const int *prefix,
                                     int *dest, int *source, int experts, int top, int routes) {
    extern __shared__ int memory[];
    int *warp_counts = memory, *local_ids = memory + 8 * experts;
    const int tid = threadIdx.x, route = blockIdx.x * 256 + tid, warp = tid / 32;
    for (int i = tid; i < 8 * experts; i += 256) warp_counts[i] = 0;
    local_ids[tid] = route < routes ? ids[route] : -1;
    __syncthreads();
    const int expert = local_ids[tid];
    if (expert >= 0) atomicAdd(warp_counts + warp * experts + expert, 1);
    __syncthreads();
    if (expert < 0) return;
    int rank = 0;
    for (int w = 0; w < warp; ++w) rank += warp_counts[w * experts + expert];
    for (int lane = warp * 32; lane < tid; ++lane) rank += local_ids[lane] == expert;
    const int grouped = bounds[expert] + prefix[blockIdx.x * experts + expert] + rank;
    dest[grouped] = route;
    source[grouped] = route / top;
}
__global__ void slice_bounds(const int *global, int *local, int start, int count, int offset, int rows) {
    int i = threadIdx.x;
    if (i <= count)
        local[i] = max(0, min(rows, global[start + i] - offset));
}
__global__ void route_sum(const float *r, const float *w, float *out, int n, int top, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float sum = 0;
    for (int j = 0; j < top; ++j)
        sum += w[t * top + j] * r[(t * top + j) * n + d];
    out[i] += sum;
}
// Weighted sum over the routes whose expert group belongs to `participant` (partition.hpp ownership),
// in increasing route order. The secondary writes its partial; the primary adds its own sum and the
// secondary's partial to the shared-expert output.
__global__ void route_sum_owned(const float *r, const float *w, const int *ids, const float *partial, float *out,
                                int n, int top, int tokens, int primary_groups, int participant) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float sum = 0;
    for (int j = 0; j < top; ++j) {
        const int g = ids[t * top + j] / 16;
        const bool primary = ((g + 1) * primary_groups + 17) / 18 > (g * primary_groups + 17) / 18;
        if (primary == (participant == 0))
            sum += w[t * top + j] * r[(t * top + j) * n + d];
    }
    if (participant == 0) out[i] += sum + (partial ? partial[i] : 0.f);
    else out[i] = sum;
}
__global__ void mhc(const float *p, const float *base, const float *scale, float *c, int iterations,
                    float eps) {
    p += blockIdx.x * 24;
    c += blockIdx.x * 24;
    for (int i = 0; i < 4; ++i) {
        c[i] = sigmoid(p[i] * scale[0] + base[i]) + eps;
        c[4 + i] = 2 * sigmoid(p[4 + i] * scale[1] + base[4 + i]);
    }
    for (int i = 0; i < 4; ++i) {
        float mx = -INFINITY, sum = 0;
        for (int j = 0; j < 4; ++j)
            mx = fmaxf(mx, p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
        for (int j = 0; j < 4; ++j) {
            c[8 + i * 4 + j] = expf(p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j] - mx);
            sum += c[8 + i * 4 + j];
        }
        for (int j = 0; j < 4; ++j)
            c[8 + i * 4 + j] = c[8 + i * 4 + j] / sum + eps;
    }
    for (int it = 0; it < iterations; ++it) {
        if (it > 0)
            for (int i = 0; i < 4; ++i) {
                float sum = eps;
                for (int j = 0; j < 4; ++j)
                    sum += c[8 + i * 4 + j];
                for (int j = 0; j < 4; ++j)
                    c[8 + i * 4 + j] /= sum;
            }
        for (int j = 0; j < 4; ++j) {
            float sum = eps;
            for (int i = 0; i < 4; ++i)
                sum += c[8 + i * 4 + j];
            for (int i = 0; i < 4; ++i)
                c[8 + i * 4 + j] /= sum;
        }
    }
}
__global__ void mhc_read(const float *r, const float *c, float *x, int n, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float sum = 0;
    for (int j = 0; j < 4; ++j)
        sum += c[t * 24 + j] * r[t * 4 * n + j * n + d];
    x[i] = sum;
}
__global__ void mhc_write(const float *r, const float *c, const float *y, float *out, int n, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float in[4];
    for (int j = 0; j < 4; ++j)
        in[j] = r[t * 4 * n + j * n + d];
    for (int j = 0; j < 4; ++j) {
        float sum = c[t * 24 + 4 + j] * y[i];
        for (int z = 0; z < 4; ++z)
            sum += c[t * 24 + 8 + z * 4 + j] * in[z];
        out[t * 4 * n + j * n + d] = sum;
    }
}
__global__ void decay(const float *x, const float *b, const float *a, float *y, int heads, int dim,
                      int tokens, float lower) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < heads * dim * tokens) {
        int d = i % (heads * dim);
        y[i] = lower * sigmoid(-a[d / dim] * (x[i] + b[d]));
    }
}
// snapshots (optional): the state after token t < snapshot_tokens also goes to snapshots + t * snapshot_stride,
// in the state's own layout, so a verify window keeps its rollback history without per-token kernels.
template<int columns, bool prepared = false>
__global__ void kda_chunk(float *state, const float *q, const float *k, const float *v, const float *g,
                          const float *beta, float *out, int heads, int tokens, float *snapshots = nullptr,
                          long long snapshot_stride = 0, int snapshot_tokens = 0,
                          const float *prepared_qi = nullptr) {
    constexpr int dim = 128;
    int h = blockIdx.x, j = blockIdx.y * columns + threadIdx.x;
    state += (size_t)h * dim * dim;
    if (snapshots) snapshots += (size_t)h * dim * dim + j;
    float values[dim];
#pragma unroll
    for (int i = 0; i < dim; ++i)
        values[i] = state[i * dim + j];
    __shared__ float qr[dim], kn[dim], decay[dim], qi, ki, b;
    for (int t = 0; t < tokens; ++t) {
        int offset = (t * heads + h) * dim;
        if (!threadIdx.x) {
            if constexpr (prepared) {
                qi = prepared_qi[t * heads + h];
                b = beta[t * heads + h];
            } else {
                float qs = 1e-6f, ks = 1e-6f;
                for (int i = 0; i < dim; ++i) {
                    qs += q[offset + i] * q[offset + i];
                    ks += k[offset + i] * k[offset + i];
                }
                ki = rsqrtf(ks);
                qi = rsqrtf(qs) * rsqrtf((float)dim);
                b = sigmoid(beta[t * heads + h]);
            }
        }
        if constexpr (!prepared) __syncthreads();
        for (int i = threadIdx.x; i < dim; i += columns) {
            qr[i] = q[offset + i];
            if constexpr (prepared) {
                kn[i] = k[offset + i];
                decay[i] = g[offset + i];
            } else {
                kn[i] = k[offset + i] * ki;
                decay[i] = expf(g[offset + i]);
            }
        }
        __syncthreads();
        float mem = 0;
#pragma unroll
        for (int i = 0; i < dim; ++i) {
            values[i] *= decay[i];
            mem += values[i] * kn[i];
        }
        float delta = (v[offset + j] - mem) * b, sum = 0;
#pragma unroll
        for (int i = 0; i < dim; ++i) {
            values[i] = values[i] + kn[i] * delta;
            sum += values[i] * qr[i] * qi;
        }
        out[offset + j] = sum;
        if (snapshots && t < snapshot_tokens) {
#pragma unroll
            for (int i = 0; i < dim; ++i) snapshots[(size_t)t * snapshot_stride + (size_t)i * dim] = values[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < dim; ++i)
        state[i * dim + j] = values[i];
}
// Partition state rows across warps to bound per-thread register storage.
// Partial dots change FP32 addition order; the original path remains default.
__global__ void kda_prepare(const float *q, float *key, float *decay, float *beta, float *qi) {
    constexpr int dim = 128;
    const int h = blockIdx.x, j = threadIdx.x, offset = h * dim;
    __shared__ float qr[dim], kr[dim], ki;
    qr[j] = q[offset + j]; kr[j] = key[offset + j];
    __syncthreads();
    if (!j) {
        float qs = 1e-6f, ks = 1e-6f;
        for (int i = 0; i < dim; ++i) {
            qs += qr[i] * qr[i]; ks += kr[i] * kr[i];
        }
        ki = rsqrtf(ks); qi[h] = rsqrtf(qs) * rsqrtf((float)dim);
        beta[h] = sigmoid(beta[h]);
    }
    __syncthreads();
    key[offset + j] = kr[j] * ki;
    decay[offset + j] = expf(decay[offset + j]);
}
template<int parts, bool prepared = false>
__global__ void __launch_bounds__(128 * parts, 1)
kda_rows(float *state, const float *q, const float *k, const float *v, const float *g,
         const float *beta, float *out, int heads, int tokens, const float *prepared_qi = nullptr) {
    constexpr int dim = 128, rows = dim / parts;
    const int h = blockIdx.x, j = threadIdx.x % dim, part = threadIdx.x / dim;
    state += (size_t)h * dim * dim;
    float values[rows];
#pragma unroll
    for (int i = 0; i < rows; ++i) values[i] = state[(part * rows + i) * dim + j];
    __shared__ float qr[dim], kn[dim], decay[dim], partial[parts * dim], delta[dim], qi, ki, b;
    for (int t = 0; t < tokens; ++t) {
        const int offset = (t * heads + h) * dim;
        if (!threadIdx.x) {
            if constexpr (prepared) {
                qi = prepared_qi[t * heads + h]; b = beta[t * heads + h];
            } else {
                float qs = 1e-6f, ks = 1e-6f;
                for (int i = 0; i < dim; ++i) {
                    qs += q[offset + i] * q[offset + i];
                    ks += k[offset + i] * k[offset + i];
                }
                ki = rsqrtf(ks); qi = rsqrtf(qs) * rsqrtf((float)dim);
                b = sigmoid(beta[t * heads + h]);
            }
        }
        __syncthreads();
        if (!part) {
            qr[j] = q[offset + j];
            if constexpr (prepared) { kn[j] = k[offset + j]; decay[j] = g[offset + j]; }
            else { kn[j] = k[offset + j] * ki; decay[j] = expf(g[offset + j]); }
        }
        __syncthreads();
        float mem = 0;
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int row = part * rows + i;
            values[i] *= decay[row]; mem += values[i] * kn[row];
        }
        partial[part * dim + j] = mem;
        __syncthreads();
        if (!part) {
            float total = 0;
#pragma unroll
            for (int p = 0; p < parts; ++p) total += partial[p * dim + j];
            delta[j] = (v[offset + j] - total) * b;
        }
        __syncthreads();
        float sum = 0;
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int row = part * rows + i;
            values[i] = values[i] + kn[row] * delta[j];
            sum += values[i] * qr[row] * qi;
        }
        partial[part * dim + j] = sum;
        __syncthreads();
        if (!part) {
            float total = 0;
#pragma unroll
            for (int p = 0; p < parts; ++p) total += partial[p * dim + j];
            out[offset + j] = total;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < rows; ++i) state[(part * rows + i) * dim + j] = values[i];
}
// Each output column is independent; retain the original eight/four partial
// sums and their order while distributing one head across four CTAs.
template <int parts, bool prepared = false, int cols = 32>
__global__ void __launch_bounds__(cols *parts, 1)
    kda_rows_columns(float *state, const float *q, const float *k, const float *v, const float *g,
                     const float *beta, float *out, int heads, int tokens,
                     const float *prepared_qi = nullptr) {
    constexpr int dim = 128, rows = dim / parts;
    const int h = blockIdx.x, lane = threadIdx.x % cols;
    const int j = blockIdx.y * cols + lane, part = threadIdx.x / cols;
    state += (size_t)h * dim * dim;
    float values[rows];
#pragma unroll
    for (int i = 0; i < rows; ++i)
        values[i] = state[(part * rows + i) * dim + j];
    __shared__ float qr[dim], kn[dim], decay[dim], partial[parts * cols], delta[cols], qi, ki, b;
    for (int t = 0; t < tokens; ++t) {
        const int offset = (t * heads + h) * dim;
        if (!threadIdx.x) {
            if constexpr (prepared) {
                qi = prepared_qi[t * heads + h];
                b = beta[t * heads + h];
            } else {
                float qs = 1e-6f, ks = 1e-6f;
                for (int i = 0; i < dim; ++i) {
                    qs += q[offset + i] * q[offset + i];
                    ks += k[offset + i] * k[offset + i];
                }
                ki = rsqrtf(ks);
                qi = rsqrtf(qs) * rsqrtf((float)dim);
                b = sigmoid(beta[t * heads + h]);
            }
        }
        __syncthreads();
        if (threadIdx.x < dim) {
            const int d = threadIdx.x;
            qr[d] = q[offset + d];
            if constexpr (prepared) {
                kn[d] = k[offset + d];
                decay[d] = g[offset + d];
            } else {
                kn[d] = k[offset + d] * ki;
                decay[d] = expf(g[offset + d]);
            }
        }
        __syncthreads();
        float mem = 0;
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int row = part * rows + i;
            values[i] *= decay[row];
            mem += values[i] * kn[row];
        }
        partial[part * cols + lane] = mem;
        __syncthreads();
        if (!part) {
            float total = 0;
#pragma unroll
            for (int p = 0; p < parts; ++p)
                total += partial[p * cols + lane];
            delta[lane] = (v[offset + j] - total) * b;
        }
        __syncthreads();
        float sum = 0;
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int row = part * rows + i;
            values[i] = values[i] + kn[row] * delta[lane];
            sum += values[i] * qr[row] * qi;
        }
        partial[part * cols + lane] = sum;
        __syncthreads();
        if (!part) {
            float total = 0;
#pragma unroll
            for (int p = 0; p < parts; ++p)
                total += partial[p * cols + lane];
            out[offset + j] = total;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < rows; ++i)
        state[(part * rows + i) * dim + j] = values[i];
}
__global__ void kda_output(const float *x, const float *g, const float *w, float *out, int dim, float eps) {
    int h = blockIdx.x, j = threadIdx.x;
    float sum = 0;
    for (int i = 0; i < dim; ++i)
        sum += x[h * dim + i] * x[h * dim + i];
    out[h * dim + j] = x[h * dim + j] * rsqrtf(sum / dim + eps) * w[j] * sigmoid(g[h * dim + j]);
}
__global__ void decode_position(int *position, int value) { *position = value; }
__global__ void cache_rows(const float *rows, float *cache, int width, int tokens, const int *position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * tokens) cache[(size_t)*position * width + i] = rows[i];
}
__global__ void pooling(const float *keys, const float *gates, const float *ape, const float *pk,
                        const float *pg, float *out, int pos, int tokens, int pool, int dim, const int *device_pos, int offset) {
    if (device_pos) pos = *device_pos + offset;
    int d = blockIdx.x * 256 + threadIdx.x, first = pos / pool, last = (pos + tokens) / pool;
    if (d >= (last - first) * dim)
        return;
    int p = first + d / dim, c = d % dim;
    float mx = -INFINITY, sum = 0, value = 0;
    for (int j = 0; j < pool; ++j) {
        int t = p * pool + j - pos;
        float g = t < 0 ? pg[(p * pool + j) % pool * dim + c] : gates[t * dim + c];
        mx = fmaxf(mx, g + ape[j * dim + c]);
    }
    for (int j = 0; j < pool; ++j) {
        int t = p * pool + j - pos;
        float g = t < 0 ? pg[(p * pool + j) % pool * dim + c] : gates[t * dim + c];
        float k = t < 0 ? pk[(p * pool + j) % pool * dim + c] : keys[t * dim + c];
        float w = expf(g + ape[j * dim + c] - mx);
        sum += w;
        value += w * k;
    }
    out[p * dim + c] = value / sum;
}
__global__ void pending(const float *keys, const float *gates, float *pk, float *pg, int pos, int tokens,
                        int pool, int dim, const int *device_pos, int offset) {
    if (device_pos) pos = *device_pos + offset;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= pool * dim)
        return;
    int j = i / dim, last = pos + tokens - 1;
    int absolute = last - ((last - j + pool) % pool), t = absolute - pos;
    if (t >= 0) {
        pk[i] = keys[t * dim + i % dim];
        pg[i] = gates[t * dim + i % dim];
    }
}
__global__ void index_reduce(const float *dots, const float *w, float *scores, int heads, int pools,
                             int queries, int pos, int pool, int dim, const int *device_pos, int offset) {
    if (device_pos) pos = *device_pos + offset;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= pools * queries)
        return;
    int q = i / pools, p = i % pools;
    if (p >= (pos + q + 1) / pool) {
        scores[i] = -INFINITY;
        return;
    }
    float sum = 0;
    for (int h = 0; h < heads; ++h)
        sum += w[q * heads + h] * fmaxf(dots[(q * heads + h) * pools + p] * rsqrtf((float)dim), 0.f);
    scores[i] = sum * rsqrtf((float)heads);
}
__device__ unsigned long long score_key(float score, int index) {
    unsigned bits = __float_as_uint(score == 0 ? 0.f : score),
             ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return ((unsigned long long)ordered << 32) | (0xffffffffu - index);
}
__global__ void selection(const float *scores, int *ids, int *counts, int pools_total, int pos, int pool,
                          int top, int stride, const int *device_pos, int offset) {
    if (device_pos) pos = *device_pos + offset;
    int q = blockIdx.x, tid = threadIdx.x, tokens = pos + q + 1, pools = tokens / pool,
        n = min(top / pool, pools);
    scores += q * pools_total;
    ids += q * stride;
    __shared__ unsigned histogram[256], rank;
    __shared__ unsigned long long prefix, mask, keys[512];
    __shared__ int used;
    if (!tid) {
        prefix = mask = 0;
        rank = n;
        used = 0;
    }
    __syncthreads();
    if (n && n < pools)
        for (int shift = 56; shift >= 0; shift -= 8) {
            histogram[tid] = 0;
            __syncthreads();
            for (int p = tid; p < pools; p += 256) {
                auto key = score_key(scores[p], p);
                if ((key & mask) == prefix)
                    atomicAdd(&histogram[(key >> shift) & 255], 1u);
            }
            __syncthreads();
            if (!tid) {
                for (int bin = 255; bin >= 0; --bin)
                    if (rank > histogram[bin])
                        rank -= histogram[bin];
                    else {
                        prefix |= (unsigned long long)bin << shift;
                        break;
                    }
                mask |= 255ull << shift;
            }
            __syncthreads();
        }
    for (int i = tid; i < 512; i += 256)
        keys[i] = 0;
    __syncthreads();
    for (int p = tid; p < pools; p += 256) {
        auto key = score_key(scores[p], p);
        if (n && key >= prefix) {
            int slot = atomicAdd(&used, 1);
            keys[slot] = key;
        }
    }
    __syncthreads();
    for (int size = 2; size <= 512; size *= 2)
        for (int step = size / 2; step; step /= 2) {
            for (int i = tid; i < 512; i += 256) {
                int other = i ^ step;
                if (other > i) {
                    auto a = keys[i], b = keys[other];
                    bool descending = (i & size) == 0;
                    if ((descending && a < b) || (!descending && a > b)) {
                        keys[i] = b;
                        keys[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int i = tid; i < n * pool; i += 256) {
        int p = 0xffffffffu - (unsigned)keys[i / pool];
        ids[i] = p * pool + i % pool;
    }
    for (int i = tid; i < tokens % pool; i += 256)
        ids[n * pool + i] = tokens - tokens % pool + i;
    if (!tid)
        counts[q] = n * pool + tokens % pool;
}
__global__ void gather(const float *cache, const int *ids, const int *counts, float *out, int queries,
                       int stride, int latent) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (int64_t)queries * stride * latent)
        return;
    int q = i / (stride * latent), key = i / latent % stride, d = i % latent;
    out[i] = key < counts[q] ? cache[(int64_t)ids[q * stride + key] * latent + d] : 0.f;
}
// One block per selected key; preserve float bits while eliminating per-value
// divisions and transferring four contiguous latent values per instruction.
__global__ void gather_vector(const float *cache, const int *ids, const int *counts, float *out,
                              int stride, int latent) {
    const int key = blockIdx.x, query = blockIdx.y;
    float4* destination = reinterpret_cast<float4*>(out + ((int64_t)query * stride + key) * latent);
    if (key < counts[query]) {
        const float4* source = reinterpret_cast<const float4*>(cache + (int64_t)ids[query * stride + key] * latent);
        for (int d = threadIdx.x; d < latent / 4; d += blockDim.x) destination[d] = source[d];
    } else {
        for (int d = threadIdx.x; d < latent / 4; d += blockDim.x) destination[d] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}
// gather_vector with each latent value rounded to FP16 for tensor-core attention products.
__global__ void gather_vector_half(const float *cache, const int *ids, const int *counts, uint16_t *out,
                                   int stride, int latent) {
    const int key = blockIdx.x, query = blockIdx.y;
    uint16_t *destination = out + ((int64_t)query * stride + key) * latent;
    if (key < counts[query]) {
        const float *source = cache + (int64_t)ids[query * stride + key] * latent;
        for (int d = threadIdx.x; d < latent; d += blockDim.x)
            destination[d] = __half_as_ushort(__float2half(source[d]));
    } else {
        for (int d = threadIdx.x; d < latent; d += blockDim.x) destination[d] = 0;
    }
}
__global__ void softmax(float *scores, const int *counts, int heads, int stride, float scale) {
    int q = blockIdx.x / heads;
    float *row = scores + (int64_t)blockIdx.x * stride;
    __shared__ float reduce[256];
    int tid = threadIdx.x, n = counts[q];
    float mx = -INFINITY;
    for (int i = tid; i < n; i += 256)
        mx = fmaxf(mx, row[i] * scale);
    reduce[tid] = mx;
    __syncthreads();
    for (int z = 128; z; z /= 2) {
        if (tid < z)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + z]);
        __syncthreads();
    }
    mx = reduce[0];
    float sum = 0;
    for (int i = tid; i < stride; i += 256) {
        float p = i < n ? expf(row[i] * scale - mx) : 0;
        row[i] = p;
        sum += p;
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int z = 128; z; z /= 2) {
        if (tid < z)
            reduce[tid] += reduce[tid + z];
        __syncthreads();
    }
    for (int i = tid; i < stride; i += 256)
        row[i] /= reduce[0];
}

// One-query MLA decode products (STRATA_GLM_MLA_KERNELS=1) in place of the two strided-batched SGEMMs, whose
// 512 x 64 x 2052 shapes leave most of a GPU idle. Same fp32 products, different summation order.
// scores[h][s] = sum_k keys[s][k] * q[h][k]: a block takes 16 keys into shared memory; a thread is one head and
// four of those keys.
__global__ void __launch_bounds__(256) mla_scores_one(const float *keys, const float *q, float *scores,
                                                      int stride, int latent, int heads) {
    extern __shared__ float4 shared_keys[];
    const int first = blockIdx.x * 16, quads = latent / 4;
    const float4 *k4 = reinterpret_cast<const float4 *>(keys);
    for (int i = threadIdx.x; i < 16 * quads; i += blockDim.x) {
        const int key = first + i / quads;
        shared_keys[i] = key < stride ? k4[(size_t)key * quads + i % quads] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    const int h = threadIdx.x % 64, group = threadIdx.x / 64;
    if (h >= heads) return;
    const float4 *q4 = reinterpret_cast<const float4 *>(q) + (size_t)h * quads;
    float acc[4] = {};
    for (int i = 0; i < quads; ++i) {
        const float4 a = q4[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 b = shared_keys[(group * 4 + j) * quads + i];
            acc[j] = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, acc[j]))));
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int key = first + group * 4 + j;
        if (key < stride) scores[(size_t)h * stride + key] = acc[j];
    }
}
// partial[chunk][h][k] = sum over the chunk's 64 keys s of p[h][s] * keys[s][k]; a block is 128 columns of one chunk.
__global__ void __launch_bounds__(256) mla_values_partial(const float *keys, const float *p, float *partial,
                                                          int stride, int latent, int heads) {
    __shared__ float probability[64][65];
    const int chunk = blockIdx.y, first = chunk * 64, column = blockIdx.x * 128 + threadIdx.x % 128;
    const int half = threadIdx.x / 128;
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int h = i / 64, s = i % 64;
        probability[h][s] = h < heads && first + s < stride ? p[(size_t)h * stride + first + s] : 0.f;
    }
    __syncthreads();
    float acc[32] = {};
    const int count = min(64, stride - first);
    for (int s = 0; s < count; ++s) {
        const float v = keys[(size_t)(first + s) * latent + column];
#pragma unroll
        for (int h = 0; h < 32; ++h) acc[h] = fmaf(probability[half * 32 + h][s], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < 32; ++h)
        if (half * 32 + h < heads) partial[((size_t)chunk * heads + half * 32 + h) * latent + column] = acc[h];
}
__global__ void mla_values_reduce(const float *partial, float *out, int chunks, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.f;
    for (int c = 0; c < chunks; ++c) sum += partial[(size_t)c * n + i];
    out[i] = sum;
}
} // namespace
size_t glm_mla_one_scratch_bytes(int stride, int latent, int heads) {
    return (size_t)((stride + 63) / 64) * heads * latent * sizeof(float);
}
void glm_mla_scores_one(const float *keys, const float *q, float *scores, int stride, int latent, int heads, void *s) {
    if (latent % 4 || latent > 1024 || heads < 1 || heads > 64) throw std::invalid_argument("GLM: MLA scores geometry");
    mla_scores_one<<<(stride + 15) / 16, 256, 16 * latent * sizeof(float), (cudaStream_t)s>>>(keys, q, scores, stride,
                                                                                           latent, heads);
    check();
}
void glm_mla_values_one(const float *keys, const float *p, float *out, float *scratch, int stride, int latent,
                        int heads, void *s) {
    if (latent % 128 || heads < 1 || heads > 64) throw std::invalid_argument("GLM: MLA values geometry");
    const int chunks = (stride + 63) / 64;
    mla_values_partial<<<dim3(latent / 128, chunks), 256, 0, (cudaStream_t)s>>>(keys, p, scratch, stride, latent, heads);
    mla_values_reduce<<<(heads * latent + 255) / 256, 256, 0, (cudaStream_t)s>>>(scratch, out, chunks, heads * latent);
    check();
}
void glm_f16(const float *x, uint16_t *y, int64_t n, void *s) {
    half_cast<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, y, n);
    check();
}
void glm_bf16(const float *x, uint16_t *y, int64_t n, void *s) {
    if (n > 0) bf16_cast<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, y, n);
    check();
}
void glm_gather_expert_bf16(const float *x, uint16_t *y, const int *bounds, const int *source,
                            int first, int groups, int offset, int rows, int width, void *s, unsigned mask) {
    check_expert_mask(mask, groups);
    if (first < 0 || groups < 0 || offset < 0 || rows < 0 || width < 1)
        throw std::invalid_argument("GLM: invalid BF16 expert gather");
    if (groups && rows) gather_expert_bf16<<<dim3(((int64_t)rows * width + 255) / 256, groups), 256, 0,
        (cudaStream_t)s>>>(x, y, bounds, source, first, offset, rows, width, mask);
    check();
}
void glm_resident_routes(const int *ids, const unsigned long long *lookup,
                          unsigned long long *ptr, int *starts, int *dest,
                          int *tokens, int *count, void *s) {
    resident_routes<<<1, 1, 0, (cudaStream_t)s>>>(ids, lookup, ptr, starts, dest, tokens, count);
    check();
}
void glm_resident_routes_batch(const int *ids, const unsigned long long *lookup,
                               unsigned long long *ptr, int *starts, int *dest,
                               int *tokens, int *count, int nt, void *s) {
    if (nt < 1 || nt > 8) throw std::invalid_argument("GLM: invalid resident token count");
    resident_routes_batch<<<1, 1, 0, (cudaStream_t)s>>>(ids, lookup, ptr, starts, dest, tokens, count, nt);
    check();
}
void glm_gather_expert_f16(const float *x, uint16_t *y, const int *bounds, const int *source,
                           int first, int groups, int offset, int rows, int width, void *s, unsigned mask) {
    check_expert_mask(mask, groups);
    if (first < 0 || groups < 0 || offset < 0 || rows < 0 || width < 1)
        throw std::invalid_argument("GLM: invalid expert gather");
    if (groups && rows) gather_expert_half<<<dim3(((int64_t)rows * width + 255) / 256, groups), 256, 0,
        (cudaStream_t)s>>>(x, y, bounds, source, first, offset, rows, width, mask);
    check();
}
void glm_gather_expert_f16_from_half(const uint16_t *x, uint16_t *y, const int *bounds, const int *source,
                                     int first, int groups, int offset, int rows, int width, void *s, unsigned mask) {
    check_expert_mask(mask, groups);
    if (first < 0 || groups < 0 || offset < 0 || rows < 0 || width < 1)
        throw std::invalid_argument("GLM: invalid expert gather");
    if (groups && rows) gather_expert_half_input<<<dim3(((int64_t)rows * width + 255) / 256, groups), 256, 0,
        (cudaStream_t)s>>>(x, y, bounds, source, first, offset, rows, width, mask);
    check();
}
void glm_moe_reduce(const float *cpu, const float *primary, const float *remote,
                     const int *owners, const float *weights, float *out, int width, void *s) {
    if (width < 1) throw std::invalid_argument("GLM: invalid device MoE reduction");
    moe_reduce<<<(width + 255) / 256, 256, 0, (cudaStream_t)s>>>(cpu, primary, remote, owners, weights, out, width);
    check();
}
void glm_moe_reduce_batch(const float *cpu, const float *primary, const float *remote,
                          const int *owners, const float *weights, float *out, int width, int nt, void *s) {
    if (width < 1 || nt < 1 || nt > 8) throw std::invalid_argument("GLM: invalid batched MoE reduction");
    moe_reduce_batch<<<(width * nt + 255) / 256, 256, 0, (cudaStream_t)s>>>(cpu, primary, remote, owners, weights,
                                                                      out, width, nt);
    check();
}
void glm_scatter_expert_rows(const float *x, float *y, const int *bounds, const int *dest,
                             int first, int groups, int offset, int rows, int width, void *s, unsigned mask) {
    check_expert_mask(mask, groups);
    if (first < 0 || groups < 0 || offset < 0 || rows < 0 || width < 1)
        throw std::invalid_argument("GLM: invalid expert scatter");
    if (groups && rows) scatter_expert_rows<<<dim3(((int64_t)rows * width + 255) / 256, groups), 256, 0,
        (cudaStream_t)s>>>(x, y, bounds, dest, first, offset, rows, width, mask);
    check();
}
void glm_gather_f16(const float *x, uint16_t *y, const int *routes, int begin,
                    int rows, int width, void *s) {
    if (begin < 0 || rows < 0 || width < 1) throw std::invalid_argument("GLM: invalid gather");
    const int64_t n = (int64_t)rows * width;
    if (n) gather_half<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, y, routes, begin, rows, width);
    check();
}
void glm_scatter_rows(const float *x, float *y, const int *routes, int begin,
                      int rows, int width, void *s) {
    if (begin < 0 || rows < 0 || width < 1) throw std::invalid_argument("GLM: invalid scatter");
    const int64_t n = (int64_t)rows * width;
    if (n) scatter_rows<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, y, routes, begin, rows, width);
    check();
}
void glm_copy_route_rows(const float *source, float *dest, const int *routes, int begin, int rows,
                         int width, bool gather_rows, void *s) {
    if (begin < 0 || rows < 0 || width < 1) throw std::invalid_argument("GLM: invalid route copy");
    const int64_t n = (int64_t)rows * width;
    if (n) copy_route_rows<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>
        (source, dest, routes, begin, rows, width, gather_rows);
    check();
}
void glm_conv_batch(const float *x, const float *w, float *h, float *y, int n, int kernel, int tokens,
                    void *s) {
    convolution<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, w, h, y, n, kernel, tokens);
    conv_history<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, h, n, kernel, tokens);
    check();
}
void glm_route_batch(const float *l, const float *b, int *ids, float *w, int ne, int top, float scale,
                     int tokens, void *s, const float *bonus) {
    routing<<<tokens, 1, 0, (cudaStream_t)s>>>(l, b, ids, w, ne, top, scale, bonus);
    check();
}
void glm_group_routes(const int *ids, int *b, int *d, int *src, int *cursor, int ne, int top, int tokens,
                      void *s) {
    cudaMemsetAsync(b, 0, (ne + 1) * 4, (cudaStream_t)s);
    count_routes<<<(tokens * top + 255) / 256, 256, 0, (cudaStream_t)s>>>(ids, b, tokens * top);
    prefix_routes<<<1, 1, 0, (cudaStream_t)s>>>(b, cursor, ne);
    scatter_routes<<<(tokens * top + 255) / 256, 256, 0, (cudaStream_t)s>>>(ids, cursor, d, src, top,
                                                                            tokens * top);
    check();
}
void glm_group_routes_stable(const int *ids, int *b, int *d, int *src, int *scratch, int ne, int top,
                            int tokens, void *s) {
    if (ne < 1 || ne > 288 || top < 1 || top > ne || tokens < 1 || (int64_t)tokens * top > INT32_MAX)
        throw std::invalid_argument("GLM: invalid stable grouping geometry");
    const int routes = tokens * top, blocks = (routes + 255) / 256;
    cudaMemsetAsync(b, 0, (ne + 1) * 4, (cudaStream_t)s);
    route_block_counts<<<blocks, 256, ne * 4, (cudaStream_t)s>>>(ids, scratch, ne, routes);
    route_block_prefix<<<(ne + 255) / 256, 256, 0, (cudaStream_t)s>>>(scratch, b, ne, blocks);
    prefix_routes<<<1, 1, 0, (cudaStream_t)s>>>(b, scratch + blocks * ne, ne);
    stable_route_scatter<<<blocks, 256, (8 * ne + 256) * 4, (cudaStream_t)s>>>
        (ids, b, scratch, d, src, ne, top, routes);
    check();
}
void glm_group_bounds(const int *g, int *l, int start, int n, int offset, int rows, void *s) {
    slice_bounds<<<1, 32, 0, (cudaStream_t)s>>>(g, l, start, n, offset, rows);
    check();
}
void glm_route_sum(const float *r, const float *w, float *out, int n, int top, int tokens, void *s) {
    route_sum<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, w, out, n, top, tokens);
    check();
}
void glm_route_sum_owned(const float *r, const float *w, const int *ids, const float *partial, float *out, int n,
                         int top, int tokens, int primary_groups, int participant, void *s) {
    if (primary_groups < 1 || primary_groups > 17 || participant < 0 || participant > 1)
        throw std::invalid_argument("GLM: invalid owned route sum");
    route_sum_owned<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, w, ids, partial, out, n, top, tokens,
                                                                            primary_groups, participant);
    check();
}
void glm_mhc_read_batch(const float *r, const float *p, const float *b, const float *scale, float *c,
                        float *x, int n, int it, float eps, int tokens, void *s) {
    mhc<<<tokens, 1, 0, (cudaStream_t)s>>>(p, b, scale, c, it, eps);
    mhc_read<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, x, n, tokens);
    check();
}
void glm_mhc_write_batch(const float *r, const float *c, const float *y, float *out, int n, int tokens,
                         void *s) {
    mhc_write<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, y, out, n, tokens);
    check();
}
void glm_kda_gate_batch(const float *x, const float *b, const float *a, float *y, int h, int d, float lower,
                        int tokens, void *s) {
    decay<<<(h * d * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, b, a, y, h, d, tokens, lower);
    check();
}
void glm_kda_prepare(const float *q, float *key, float *decay, float *beta, float *qi,
                     int heads, int tokens, void *s) {
    if (heads < 1 || tokens < 1 || tokens > 16384 || !qi)
        throw std::invalid_argument("GLM: invalid KDA preparation");
    kda_prepare<<<heads * tokens, 128, 0, (cudaStream_t)s>>>(q, key, decay, beta, qi);
    check();
}
void glm_kda_chunk(float *state, const float *q, const float *k, const float *v, const float *g,
                   const float *b, float *out, int heads, int dim, int tokens, void *s, int columns,
                   int row_parts, const float *prepared_qi, float *snapshots, long long snapshot_stride,
                   int snapshot_tokens) {
    if (dim != 128 || tokens < 1 || tokens > 2048)
        throw std::invalid_argument("GLM: invalid KDA chunk");
    if (snapshots && (row_parts != 1 || prepared_qi))
        throw std::invalid_argument("GLM: KDA snapshots need the column kernel");
    if (row_parts != 1) {
        if (columns != 128 || (row_parts != 4 && row_parts != 8))
            throw std::invalid_argument("GLM: parallel KDA needs 128 columns and 4 or 8 row parts");
        if (std::getenv("STRATA_GLM_KDA_COLUMN_TILES")) {
            const dim3 grid(heads, 4);
            if (prepared_qi) {
                if (row_parts == 4)
                    kda_rows_columns<4, true><<<grid, 128, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens, prepared_qi);
                else kda_rows_columns<8, true><<<grid, 256, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens, prepared_qi);
            } else {
                if (row_parts == 4)
                    kda_rows_columns<4><<<grid, 128, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens);
                else kda_rows_columns<8><<<grid, 256, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens);
            }
            check(); return;
        }
        if (prepared_qi) {
            if (row_parts == 4)
                kda_rows<4, true><<<heads, 512, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens, prepared_qi);
            else kda_rows<8, true><<<heads, 1024, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens, prepared_qi);
            check(); return;
        }
        if (row_parts == 4)
            kda_rows<4><<<heads, 512, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens);
        else kda_rows<8><<<heads, 1024, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens);
        check(); return;
    }
    if (prepared_qi) {
        // Preparation keeps the original serial norm sums. The recurrence below
        // keeps all 128 state rows in their original accumulation order too.
        if (columns == 32)
            kda_chunk<32, true><<<dim3(heads, 4), 32, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads,
                tokens, nullptr, 0, 0, prepared_qi);
        else if (columns == 64)
            kda_chunk<64, true><<<dim3(heads, 2), 64, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads,
                tokens, nullptr, 0, 0, prepared_qi);
        else if (columns == 128)
            kda_chunk<128, true><<<heads, 128, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads,
                tokens, nullptr, 0, 0, prepared_qi);
        else throw std::invalid_argument("GLM: KDA columns must be 32, 64 or 128");
        check(); return;
    }
    if (columns == 32)
        kda_chunk<32><<<dim3(heads, 4), 32, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens,
                                                                   snapshots, snapshot_stride, snapshot_tokens);
    else if (columns == 64)
        kda_chunk<64><<<dim3(heads, 2), 64, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens,
                                                                   snapshots, snapshot_stride, snapshot_tokens);
    else if (columns == 128)
        kda_chunk<128><<<heads, 128, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens,
                                                            snapshots, snapshot_stride, snapshot_tokens);
    else throw std::invalid_argument("GLM: KDA columns must be 32, 64 or 128");
    check();
}
void glm_kda_output_batch(const float *x, const float *g, const float *w, float *y, int h, int d, float eps,
                          int tokens, void *s) {
    kda_output<<<h * tokens, d, 0, (cudaStream_t)s>>>(x, g, w, y, d, eps);
    check();
}
void glm_decode_position(int *position, int value, void *s) {
    decode_position<<<1, 1, 0, (cudaStream_t)s>>>(position, value);
    check();
}
void glm_cache_rows(const float *rows, float *cache, int width, int tokens, const int *position, void *s) {
    cache_rows<<<(width * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(rows, cache, width, tokens, position);
    check();
}
void glm_index_prepare(const float *k, const float *g, const float *a, float *pk, float *pg, float *out,
                       int pos, int tokens, int pool, int dim, void *s, const int *device_pos, int offset) {
    // The maximum number of completed pools does not depend on the current position.
    int n = (device_pos ? (tokens + pool - 1) / pool : (pos + tokens) / pool - pos / pool) * dim;
    if (n)
        pooling<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(k, g, a, pk, pg, out, pos, tokens, pool, dim, device_pos, offset);
    pending<<<(pool * dim + 255) / 256, 256, 0, (cudaStream_t)s>>>(k, g, pk, pg, pos, tokens, pool, dim, device_pos, offset);
    check();
}
void glm_index_reduce(const float *d, const float *w, float *out, int h, int pools, int q, int pos, int pool,
                      int dim, void *s, const int *device_pos, int offset) {
    if (pools)
        index_reduce<<<(pools * q + 255) / 256, 256, 0, (cudaStream_t)s>>>(d, w, out, h, pools, q, pos, pool,
                                                                           dim, device_pos, offset);
    check();
}
void glm_index_select_batch(const float *scores, int *ids, int *counts, int pools, int pos, int queries,
                            int pool, int top, int stride, void *s, const int *device_pos, int offset) {
    if (top / pool > 512)
        throw std::invalid_argument("GLM: selection exceeds 512 pools");
    selection<<<queries, 256, 0, (cudaStream_t)s>>>(scores, ids, counts, pools, pos, pool, top, stride, device_pos, offset);
    check();
}
void glm_mla_gather(const float *c, const int *ids, const int *counts, float *out, int q, int stride,
                    int latent, void *s) {
    int64_t n = (int64_t)q * stride * latent;
    const char* flag = std::getenv("STRATA_GLM_MLA_VECTOR_GATHER");
    if (flag && std::strcmp(flag, "0") != 0 && std::strcmp(flag, "1") != 0)
        throw std::invalid_argument("GLM: vector MLA gather must be 0 or 1");
    if (flag && std::strcmp(flag, "1") == 0 && q > 0 && q <= 65535 && stride > 0 && latent > 0 && latent % 4 == 0 &&
        reinterpret_cast<uintptr_t>(c) % 16 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0)
        gather_vector<<<dim3(stride, q), 128, 0, (cudaStream_t)s>>>(c, ids, counts, out, stride, latent);
    else gather<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(c, ids, counts, out, q, stride, latent);
    check();
}
void glm_mla_gather_f16(const float *c, const int *ids, const int *counts, uint16_t *out, int q, int stride,
                        int latent, void *s) {
    if (q < 1 || q > 65535 || stride < 1 || latent < 1) throw std::invalid_argument("GLM: invalid FP16 MLA gather");
    gather_vector_half<<<dim3(stride, q), 256, 0, (cudaStream_t)s>>>(c, ids, counts, out, stride, latent);
    check();
}
void glm_mla_softmax(float *scores, const int *counts, int heads, int stride, int q, float scale, void *s) {
    softmax<<<heads * q, 256, 0, (cudaStream_t)s>>>(scores, counts, heads, stride, scale);
    check();
}
} // namespace strata::kernels
