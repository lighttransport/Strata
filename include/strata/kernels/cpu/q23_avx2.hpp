#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels::cpu {
// Original GGML Q2_K/Q3_K bytes, Q8_K activations. Float reduction is invariant
// to verification width; it need not match ggml's reduction order.
// Build-only STRATA_Q2_HALF_PAIR_NT1 converts adjacent half scales together
// in direct nt=1 Q2 rows/GU. Default off; the Q2 LUT path is unchanged.
// Optional STRATA_Q3_CLANG_NT4 build: runtime STRATA_Q3_CLANG_NT4=1 selects
// Clang only for original Q3_K rows at nt=4. Read once; default and value 2 off.
// Independently, STRATA_Q3_CLANG_NT2 plus runtime STRATA_Q3_CLANG_NT2=1
// selects original Q3_K rows at nt=2, with the same exact-one semantics.
void q23_rows(int type, const uint8_t* weights, size_t row_bytes, int columns,
              const void* const* act, int nt, float* const* out, int first, int last);
void q23_gu_rows(int type, const uint8_t* gate, const uint8_t* up, size_t row_bytes,
                 int columns, const void* const* act, int nt, float* const* out,
                 int first, int last, float clamp, bool canon = false, float gate_skip = 0.f);
// Explicit experimental layout from strata_q3k_expand_scales (114 bytes/K256).
// Caller must have AVX2/FMA/F16C and provide valid expanded scales, complete
// input/output buffers, and Q8_K activations. No automatic format detection.
// Columns: 256..4096 in steps of 256; tokens: 1..8; padded row strides allowed.
// Invalid geometry/null nonempty buffers throw before writes. Empty ranges
// accept null buffers after geometry validation. GU requires both weight planes.
void q3k_expanded_rows(const uint8_t* weights, size_t row_bytes, int columns,
                      const void* const* act, int nt, float* const* out, int first, int last);
void q3k_expanded_gu_rows(const uint8_t* gate, const uint8_t* up, size_t row_bytes,
                         int columns, const void* const* act, int nt, float* const* out,
                         int first, int last, float clamp, bool canon = false, float gate_skip = 0.f);
// A reversible 32-row x 256-column tile; byte count equals original GGML bytes.
void q23_pack(int type,const uint8_t* source,uint8_t* dest,int columns,int rows);
void q23_unpack(int type,const uint8_t* source,uint8_t* dest,int columns,int rows);
void q23_packed_rows(int type,const uint8_t* weights,int columns,const void* const* act,
                     int nt,float* const* out,int first,int last,bool lookup);
}
