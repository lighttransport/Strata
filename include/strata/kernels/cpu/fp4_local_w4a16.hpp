#pragma once
#include <cstdint>
namespace strata::kernels::cpu {
// Experimental batch-one local FP4 GEMV. Activations are finite F32 values
// already rounded to BF16 by the caller. E4M3 scales are validated 0..126.
// Decoded E2M1*E4M3 weights are exactly BF16-representable; no Q8 conversion.
// Global weight scale is applied last. SIMD changes FP32 accumulation order
// relative to scalar and GPU WMMA; neither promises bitwise GPU equivalence.
// planar=true requires strata_glm_fp4_repack_k32 codes; storage is unchanged.
void fp4_local_w4a16_rows(const uint8_t* codes,const uint8_t* scales,
    const float* activations,float global,int columns,int rows,int first,int last,
    float* output,bool planar,bool simd);
// Token-major contiguous inputs/outputs, 1..16 tokens. Reuses each decoded
// weight vector across groups of up to four tokens; per-token arithmetic matches rows().
void fp4_local_w4a16_batch_rows(const uint8_t* codes,const uint8_t* scales,
    const float* activations,float global,int columns,int rows,int tokens,
    int first,int last,float* output,bool planar,bool simd);
}
