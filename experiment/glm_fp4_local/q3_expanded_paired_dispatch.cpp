#include "strata/kernels/cpu/q23_avx2.hpp"
namespace strata::kernels::cpu {
void q23_baseline_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
void q23_baseline_gu_rows(int,const uint8_t*,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float,bool,float);
void q3_expanded_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
// Scratch dispatch: an expanded Q3 row has an explicit, distinct byte extent.
void q23_rows(int ty,const uint8_t* w,size_t stride,int n,const void* const* a,int nt,float* const* o,int lo,int hi) {
    if(ty==11 && stride==size_t(n/256)*114)q3_expanded_rows(ty,w,stride,n,a,nt,o,lo,hi);
    else q23_baseline_rows(ty,w,stride,n,a,nt,o,lo,hi);
}
void q23_gu_rows(int ty,const uint8_t* w,const uint8_t* u,size_t stride,int n,const void* const* a,int nt,float* const* o,int lo,int hi,float clamp,bool canon,float skip) {
    q23_baseline_gu_rows(ty,w,u,stride,n,a,nt,o,lo,hi,clamp,canon,skip);
}
}
