// candidate: a cell decodes its 32 weights once (registers) and applies them to up to ENT entries staged together
template<int ROWS, int ENT, int RWN>
__global__ void cand_kernel(int type, const uint8_t* const* wptr, const int* gstart, const int* eact,
                            const block_q8_K* act, float* out, size_t part_out, int rows, int nb,
                            size_t row_bytes, size_t part_stride) {
    __shared__ int sums[ROWS][ENT][20 * 8];
    __shared__ float scl[ROWS][ENT][20];
    __shared__ float lanes[ROWS][ENT][8];
    __shared__ int xs[ENT][20 * 64];
    __shared__ float xd[ENT][20];
    const int w = threadIdx.x / 32, t = threadIdx.x % 32, g = blockIdx.y, part = blockIdx.z;
    float* o = out + size_t(part) * part_out;
    for (int e0 = gstart[g]; e0 < gstart[g + 1]; e0 += ENT) {
        const int ne = min(ENT, gstart[g + 1] - e0);
        for (int i = threadIdx.x; i < ne * nb * 64; i += blockDim.x) {
            const int k = i / (nb * 64), r = i % (nb * 64);
            xs[k][r] = reinterpret_cast<const int*>(act[size_t(eact[e0 + k]) * nb + r / 64].qs)[r % 64];
        }
        if (threadIdx.x < ne * nb) xd[threadIdx.x / nb][threadIdx.x % nb] = act[size_t(eact[e0 + threadIdx.x / nb]) * nb + threadIdx.x % nb].d;
        __syncthreads();
      for (int rr = 0; rr < RWN; ++rr) {
        const int row = (blockIdx.x * RWN + rr) * ROWS + w;
        const bool live = row < rows;
        const uint8_t* wr = wptr[g] + size_t(part) * part_stride + size_t(live ? row : 0) * row_bytes;
        if (live)
#pragma unroll 2
            for (int cell = t; cell < nb * 8; cell += 32) {
                const int block = cell / 8, lane = cell % 8, gi = lane / 2;
                const uint8_t* p = wr + size_t(block) * (type == 19 ? 50 : 66);
                int sum[ENT];
#pragma unroll
                for (int k = 0; k < ENT; ++k) sum[k] = 0;
                if (type == 19) {
                    unsigned hh[8], ix[8], gr[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) hh[ib] = unsigned(p[34 + 2*ib]) | (unsigned(p[35 + 2*ib]) << 8);
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) ix[ib] = p[2 + 4*ib + gi] | (((hh[ib] >> (3*gi)) & 7) << 8);
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) gr[ib] = iq1s_grid_gpu[ix[ib]];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        const int g4 = int((gr[ib] >> (4 * (lane % 2))) & 0x0F0F0F0Fu);
                        const int sc = 2 * int((hh[ib] >> 12) & 7) + 1;
                        const bool neg = (hh[ib] >> 15) != 0;
#pragma unroll
                        for (int k = 0; k < ENT; ++k) {
                            if (k < ne) {
                                const int xw = xs[k][block * 64 + 8*ib + lane];
                                const int dg = STRATA_DP4A(g4, xw, 0), dx = STRATA_DP4A(0x01010101, xw, 0);
                                sum[k] += (8 * (dg - dx) + (neg ? -dx : dx)) * sc;
                            }
                        }
                    }
                } else {
                    unsigned hw[8], qi[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        const uint8_t* q = p + 2 + 8*ib;
                        hw[ib] = unsigned(q[4]) | (unsigned(q[5]) << 8) | (unsigned(q[6]) << 16) | (unsigned(q[7]) << 24);
                        qi[ib] = q[gi];
                    }
                    uint64_t gv[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) gv[ib] = iq2xxs_grid[qi[ib]];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        unsigned sg = (hw[ib] >> (7*gi)) & 127; sg |= (__popc(sg) & 1) << 7;
                        const unsigned s4 = (sg >> (4 * (lane % 2))) & 15u;
                        const unsigned tt = (s4 * 0x00204081u) & 0x01010101u, m = tt * 0xffu;
                        const int w4 = int((unsigned(gv[ib] >> (32 * (lane % 2))) ^ m) + tt), sc = 2 * int(hw[ib] >> 28) + 1;
#pragma unroll
                        for (int k = 0; k < ENT; ++k)
                            if (k < ne) sum[k] += STRATA_DP4A(w4, xs[k][block * 64 + 8*ib + lane], 0) * sc;
                    }
                }
#pragma unroll
                for (int k = 0; k < ENT; ++k) if (k < ne) sums[w][k][cell] = sum[k];
                if (lane == 0) {
                    const unsigned short bits = unsigned(p[0]) | (unsigned(p[1]) << 8);
                    const float dw = __fmul_rn(__half2float(*reinterpret_cast<const __half*>(&bits)), 0.125f);
#pragma unroll
                    for (int k = 0; k < ENT; ++k) if (k < ne) scl[w][k][block] = __fmul_rn(dw, xd[k][block]);
                }
            }
        __syncthreads();
        if (live && t < 8 * ne) {
            const int k = t / 8, ln = t % 8;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) acc = fmaf(scl[w][k][b], float(sums[w][k][8*b + ln]), acc);
            lanes[w][k][ln] = acc;
        }
        __syncthreads();
        if (live && t < ne) {
            const float* L = lanes[w][t];
            const float p0 = __fadd_rn(L[0],L[4]), p1 = __fadd_rn(L[1],L[5]);
            const float p2 = __fadd_rn(L[2],L[6]), p3 = __fadd_rn(L[3],L[7]);
            o[size_t(e0 + t) * rows + row] = __fadd_rn(__fadd_rn(p0,p1), __fadd_rn(p2,p3));
        }
        __syncthreads();
      }
    }
}
#ifndef CR
#define CR 8
#endif
#ifndef CE
#define CE 2
#endif
#ifndef RW
#define RW 1
#endif
void cand_launch(int type, const uint8_t* const* wptr, const int* gstart, const int* eact, const block_q8_K* act, float* out,
                 size_t part_out, int rows, int nb, size_t row_bytes, size_t part_stride, int groups, int parts, cudaStream_t s) {
    cand_kernel<CR, CE, RW><<<dim3((rows + CR * RW - 1) / (CR * RW), groups, parts), 32 * CR, 0, s>>>(type, wptr, gstart, eact, act, out, part_out, rows, nb, row_bytes, part_stride);
}
