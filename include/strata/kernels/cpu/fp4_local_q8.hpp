#pragma once
#include <cstdint>
namespace strata::kernels::cpu {
// Experimental ModelOpt layout: adjacent E2M1 nibbles, E4M3 scale per K16.
// Activations are already quantized: signed int8, FP32 scale per K64 and
// signed int16 sum per K16. Q8 activation quantization is lossy and opt-in.
// Callers own correctly sized buffers and validate finite nonnegative E4M3
// scale codes 0..126 and finite activation scales before calling. Weight global scale is applied last.
// With tree_reduce=false each output uses the same sequential K16 FMA order in scalar and SIMD modes.
bool fp4_local_q8_avx2_available();
// Quantize 64 finite F32 values to symmetric Q8 with one scale and four K16
// sums. Requires FE_TONEAREST. Tiny scales are clamped to minimum normal F32.
// Returns false for null/nonfinite input without writing outputs.
bool fp4_local_quantize_q8_64(const float* input, int8_t* output,
    float* scale, int16_t* sums, bool simd);
void fp4_local_q8_rows(const uint8_t* codes, const uint8_t* scales,
    const int8_t* activations, const float* activation_scales, const int16_t* activation_sums,
    float global, int columns, int rows, int tokens, int first, int last,
    float* output, bool simd, bool tree_reduce=false, int row_tile=1, bool wide_k=false);
// tree_reduce=true uses independent FP32 lanes and changes reduction order.
// wide_k=true is an experimental K64 loop, restricted to batch-one tree mode.
// It changes reduction order again; tile four uses one accumulator per row
// to reduce register pressure, while smaller tiles use two.
// Optional CPU scale cache: exact FP16 bits for E4M3/2 (one value per K16).
// Additional storage is one byte per scale. Does not alter checkpoint bytes.
uint16_t fp4_local_half_scale(uint8_t scale);
bool fp4_local_q8_half_available();
void fp4_local_q8_half_rows(const uint8_t* codes,const uint16_t* half_scales,
    const int8_t* activations,const float* activation_scales,float global,
    int columns,int rows,int first,int last,float* output,int row_tile=1);
// Codes repacked with strata_glm_fp4_repack_k32; scales stay row-major E4M3.
// Same arithmetic as the existing tree path, with no resident-size increase.
// wide_k selects the existing K64 reduction order and requires one token.
void fp4_local_q8_planar_rows(const uint8_t* codes,const uint8_t* scales,
    const int8_t* activations,const float* activation_scales,float global,
    int columns,int rows,int tokens,int first,int last,float* output,int row_tile=1,bool wide_k=false);

}
