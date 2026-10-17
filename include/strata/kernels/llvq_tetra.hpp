// llvq Tetra GPU encoder (src/kernels/cuda/llvq_tetra.cu, CUDA and HIP). Create the context from the host tables
// (strata::llvq::make_view: a core::TetraTablesView and its 5^8 row-position table), then encode device arrays.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
struct LlvqTetraGpu;
LlvqTetraGpu* llvq_tetra_gpu_create(const void* host_view, const int16_t* host_pos5);
void llvq_tetra_gpu_destroy(LlvqTetraGpu* g);
void llvq_tetra_gpu_set_ctas(LlvqTetraGpu* g, int ctas);  // grid size (default 16 per SM/CU)
// Encode n blocks of 24 values (device x[n][24]) to 48-bit words (gain bit 0) and t = <x,y>/|y| (`t` may be
// null). f64: bit-identical to llvq's encoder; f32: format-compatible fallback.
void llvq_tetra_encode_f64(LlvqTetraGpu* g, const double* x, int64_t n, uint64_t* words, double* t, void* stream);
void llvq_tetra_encode_f32(LlvqTetraGpu* g, const float* x, int64_t n, uint64_t* words, double* t, void* stream);
}  // namespace strata::kernels

// ---- decode (inference) ---------------------------------------------------------------------------------------
namespace strata::kernels {
struct LlvqTetraDecodeGpu;
// From strata::llvq::DecodeTables (tetra_decode.hpp): pre/suf [64][2], br_c/br_s [64][16], rows [4096],
// vals [2][16], order [24].
LlvqTetraDecodeGpu* llvq_tetra_decode_create(const uint8_t* pre, const uint8_t* suf, const uint8_t* br_c, const uint8_t* br_s,
                                             const uint32_t* rows, const int8_t* vals, const uint32_t* order);
void llvq_tetra_decode_destroy(LlvqTetraDecodeGpu* d);
// One matrix record on the device (TetraMatrix): words rows x stride bytes, f32 row scales, BF16 tail.
struct LlvqTetraMatrixDev {
    const uint8_t* words;
    const float* scales;
    const uint16_t* tail_bf16;
    float c0, c1;
    uint32_t rows, cols, stride;
};
// xp: ntok x cols activations, rotated and trio-permuted (llvq_tetra_prepare). y[r * ntok + k], ntok <= 8.
void llvq_tetra_gemv(const LlvqTetraDecodeGpu* d, const LlvqTetraMatrixDev& m, const float* xp, int ntok, float* y, void* stream);
// W' as FP16 (rows x cols, natural rotated column order) for prefill GEMMs against Q x.
void llvq_tetra_dequant_f16(const LlvqTetraDecodeGpu* d, const LlvqTetraMatrixDev& m, void* out_f16, void* stream);
// xp = trio-permute(Q x) for ntok tokens of a power-of-two width (llvq Rotation with one WHT group); signs: the
// rotation's +-1 per column (device, cols floats).
void llvq_tetra_prepare(const LlvqTetraDecodeGpu* d, const float* signs, const float* x, int ntok, int cols, float* xp, void* stream);
}  // namespace strata::kernels
