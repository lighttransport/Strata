#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels::cpu {
// Original GGML Q2_K/Q3_K bytes, Q8_K activations. Float reduction is invariant
// to verification width; it need not match ggml's reduction order.
void q23_rows(int type, const uint8_t* weights, size_t row_bytes, int columns,
              const void* const* act, int nt, float* const* out, int first, int last);
void q23_gu_rows(int type, const uint8_t* gate, const uint8_t* up, size_t row_bytes,
                 int columns, const void* const* act, int nt, float* const* out,
                 int first, int last, float clamp);
// A reversible 32-row x 256-column tile; byte count equals original GGML bytes.
void q23_pack(int type,const uint8_t* source,uint8_t* dest,int columns,int rows);
void q23_unpack(int type,const uint8_t* source,uint8_t* dest,int columns,int rows);
void q23_packed_rows(int type,const uint8_t* weights,int columns,const void* const* act,
                     int nt,float* const* out,int first,int last,bool lookup);
}
