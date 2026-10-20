// Tiled layout adapted from src/prefill/wmma_gfx12.cu (MIT; see LICENSE).
// Mode 0: INT8 x2; 1: FP8; 2: exact split INT4 using two K32 instructions;
// 3: fused BF16 with scales folded into inputs; 5: preexpanded FP8;
// 6,7: unit-scale FP8/INT8, accumulating across all K inside WMMA.
template<int Mode>
__global__ void fp4_gemm(const uint8_t* a, const uint8_t* b,
                         const uint8_t* sa, const uint8_t* sb, float* out,
                         int m, int n, int kp, float alpha) {
    const int lane = threadIdx.x, h = lane / 16, idx = lane % 16;
    const int ar = blockIdx.y * 16 + idx, br = blockIdx.x * 16 + idx;
    F8 total{};
    I8 integers{};
    for (int k = 0; k < kp; k += 16) {
        F8 dot{};
        if constexpr (Mode == 2) {
            I2 av{}, blo{}, bhi{};
            // x=lo+hi, both in [-6,6]. Each K32 concatenates the two A parts.
            // Two instructions compute (Alo+Ahi)*Blo + (Alo+Ahi)*Bhi.
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int x = ar < m ? fp4_x2(nibble(a,size_t(ar)*kp+k+j)) : 0;
                int y = br < n ? fp4_x2(nibble(b,size_t(br)*kp+k+j)) : 0;
                int xa = h ? x-x/2 : x/2;
                av[j/8] |= unsigned(xa & 15) << (4*(j%8));
                blo[j/8] |= unsigned((y/2) & 15) << (4*(j%8));
                bhi[j/8] |= unsigned((y-y/2) & 15) << (4*(j%8));
            }
            I8 d = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,av,true,blo,I8{},false);
            d = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,av,true,bhi,d,false);
            for (int i = 0; i < 8; ++i) dot[i] = float(d[i])*0.25f;
        } else if constexpr (Mode == 3) {
            B8 av{}, bv{};
            float as = ar < m ? scale_value(sa[size_t(ar)*kp/16+k/16]) : 0;
            float bs = br < n ? scale_value(sb[size_t(br)*kp/16+k/16]) : 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (ar < m) av[j] = to_bf16(fp4_x2(nibble(a,size_t(ar)*kp+k+h*8+j))*(0.5f*as));
                if (br < n) bv[j] = to_bf16(fp4_x2(nibble(b,size_t(br)*kp+k+h*8+j))*(0.5f*bs));
            }
            total = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av,bv,total);
        } else {
            I2 av{}, bv{};
            if constexpr (Mode == 5) {
                if (ar < m) av = *reinterpret_cast<const I2*>(a + size_t(ar)*kp+k+h*8);
                if (br < n) bv = *reinterpret_cast<const I2*>(b + size_t(br)*kp+k+h*8);
            } else {
                uint32_t ax = ar < m ? *reinterpret_cast<const uint32_t*>(a+size_t(ar)*kp/2+k/2+h*4) : 0;
                uint32_t bx = br < n ? *reinterpret_cast<const uint32_t*>(b+size_t(br)*kp/2+k/2+h*4) : 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned aq = (ax >> (j*4)) & 15, bq = (bx >> (j*4)) & 15;
                    unsigned x = Mode == 0 || Mode == 7 ? unsigned(fp4_x2(aq)) & 255 : fp4_to_fp8(aq);
                    unsigned y = Mode == 0 || Mode == 7 ? unsigned(fp4_x2(bq)) & 255 : fp4_to_fp8(bq);
                    av[j/4] |= x << ((j%4)*8);
                    bv[j/4] |= y << ((j%4)*8);
                }
            }
            if constexpr (Mode == 0 || Mode == 7) {
                I8 d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,av,true,bv,Mode == 7 ? integers : I8{},false);
                integers = d;
                for (int i = 0; i < 8; ++i) dot[i] = float(d[i])*0.25f;
            } else {
                dot = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av,bv,Mode == 6 ? total : F8{});
            }
        }
        if constexpr (Mode == 6 || Mode == 7) total = dot;
        else if constexpr (Mode != 3) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int row = blockIdx.y * 16 + h * 8 + i;
                float s = row < m && br < n ? scale_value(sa[size_t(row)*kp/16+k/16]) *
                    scale_value(sb[size_t(br)*kp/16+k/16]) : 0;
                total[i] = fmaf(dot[i],s,total[i]);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        int row = blockIdx.y * 16+h*8+i;
        if (row < m && br < n) out[size_t(row)*n+br] = total[i]*alpha;
    }
}

__global__ void expand_fp8(const uint8_t* src, uint32_t* dst, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count/4) return;
    unsigned packed = 0;
    for (int j = 0; j < 4; ++j) packed |= fp4_to_fp8(nibble(src,i*4+j)) << (8*j);
    dst[i] = packed;
}

template<bool Packed>
__device__ __forceinline__ B8 load_bf16(const uint8_t* packed, const uint16_t* decoded,
                                      const uint8_t* scales, int row, int rows, int kp, int k) {
    B8 v{};
    if (row >= rows) return v;
    if constexpr (!Packed) return *reinterpret_cast<const B8*>(decoded+size_t(row)*kp+k);
    else {
        uint32_t word = *reinterpret_cast<const uint32_t*>(packed+size_t(row)*kp/2+k/2);
        float scale = scale_value(scales[size_t(row)*kp/16+k/16]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = to_bf16(fp4_float((word>>(j*4))&15)*scale);
        return v;
    }
}

// 64x64 output, four waves, K32 LDS panels. Decode each operand tile once.
template<bool Packed>
__global__ void bf16_tiled(const uint8_t* a, const uint8_t* b, const uint8_t* sa,
                           const uint8_t* sb, const uint16_t* fa, const uint16_t* fb,
                           float* out, int m, int n, int kp, float alpha) {
    int tid = threadIdx.x, lane = tid%32, wave = tid/32;
    int h = lane/16, idx = lane%16, wm = wave%2, wn = wave/2;
    int m0 = blockIdx.y*64, n0 = blockIdx.x*64;
    __shared__ B8 sx[4*64], sw[4*64];
    F8 acc[2][2]{};
    int row = tid/2, segment = (tid%2)*2;
    for (int k = 0; k < kp; k += 32) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            sx[(segment+j)*64+row] = load_bf16<Packed>(a,fa,sa,m0+row,m,kp,k+(segment+j)*8);
            sw[(segment+j)*64+row] = load_bf16<Packed>(b,fb,sb,n0+row,n,kp,k+(segment+j)*8);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            B8 av[2], bv[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                av[j] = sx[(kk/8+h)*64+wm*32+j*16+idx];
                bv[j] = sw[(kk/8+h)*64+wn*32+j*16+idx];
            }
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                    acc[r][c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av[r],bv[c],acc[r][c]);
        }
        __syncthreads();
    }
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c) for (int i = 0; i < 8; ++i) {
        int rowout = m0+wm*32+r*16+h*8+i, colout = n0+wn*32+c*16+idx;
        if (rowout < m && colout < n) out[size_t(rowout)*n+colout] = acc[r][c][i]*alpha;
    }
}

template<bool Integer>
__device__ __forceinline__ I2 load_bytes(const uint8_t* p, int row, int rows, int kp, int k) {
    I2 v{};
    uint32_t word = row < rows ? *reinterpret_cast<const uint32_t*>(p+size_t(row)*kp/2+k/2) : 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned q = (word>>(j*4))&15;
        unsigned x = Integer ? unsigned(fp4_x2(q))&255 : fp4_to_fp8(q);
        v[j/4] |= x << ((j%4)*8);
    }
    return v;
}

// Same 64x64/K32 topology as BF16; retain K16 subtotals for independent scales.
template<bool Integer>
__global__ void byte_tiled(const uint8_t* a, const uint8_t* b, const uint8_t* sa,
                           const uint8_t* sb, float* out, int m, int n, int kp, float alpha) {
    int tid = threadIdx.x, lane = tid%32, wave = tid/32;
    int h = lane/16, idx = lane%16, wm = wave%2, wn = wave/2;
    int m0 = blockIdx.y*64, n0 = blockIdx.x*64;
    __shared__ I2 sx[4*64], sw[4*64];
    __shared__ float sas[2*64], sbs[2*64];
    F8 acc[2][2]{};
    int row = tid/2, segment = (tid%2)*2;
    for (int k = 0; k < kp; k += 32) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            sx[(segment+j)*64+row] = load_bytes<Integer>(a,m0+row,m,kp,k+(segment+j)*8);
            sw[(segment+j)*64+row] = load_bytes<Integer>(b,n0+row,n,kp,k+(segment+j)*8);
        }
        sas[(segment/2)*64+row] = m0+row < m ? scale_value(sa[size_t(m0+row)*kp/16+k/16+segment/2]) : 0;
        sbs[(segment/2)*64+row] = n0+row < n ? scale_value(sb[size_t(n0+row)*kp/16+k/16+segment/2]) : 0;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            I2 av[2], bv[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                av[j] = sx[(kk/8+h)*64+wm*32+j*16+idx];
                bv[j] = sw[(kk/8+h)*64+wn*32+j*16+idx];
            }
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                F8 scale_a{};
                #pragma unroll
                for (int i = 0; i < 8; ++i) scale_a[i] = sas[(kk/16)*64+wm*32+r*16+h*8+i];
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    F8 dot{};
                    if constexpr (Integer) {
                        I8 d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,av[r],true,bv[c],I8{},false);
                        for (int i = 0; i < 8; ++i) dot[i] = float(d[i])*0.25f;
                    } else dot = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av[r],bv[c],F8{});
                    float scale_b = sbs[(kk/16)*64+wn*32+c*16+idx];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][c][i] = fmaf(dot[i],scale_a[i]*scale_b,acc[r][c][i]);
                }
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c) for (int i = 0; i < 8; ++i) {
        int rowout = m0+wm*32+r*16+h*8+i, colout = n0+wn*32+c*16+idx;
        if (rowout < m && colout < n) out[size_t(rowout)*n+colout] = acc[r][c][i]*alpha;
    }
}
