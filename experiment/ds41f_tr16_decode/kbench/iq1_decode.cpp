// Isolate the exact IQ1_S unpack: arithmetic versus predecoded magnitude/sign bytes.
#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
alignas(64) static uint64_t magnitude[2][2048], sign_bytes[2][2048];
static inline __m256i join(const uint64_t* table, const unsigned* q) {
 return _mm256_set_epi64x(table[q[3]],table[q[2]],table[q[1]],table[q[0]]);
}
static inline void original(const unsigned* q,int delta,__m256i& g,__m256i& s) {
 auto v=join(iq1s_grid,q); auto v2=_mm256_add_epi8(v,v); auto v4=_mm256_add_epi8(v2,v2);
 s=_mm256_add_epi8(_mm256_add_epi8(v4,v4),_mm256_set1_epi8(delta?-1:1));
 g=_mm256_abs_epi8(s);
}
static inline void shifted(const unsigned* q,int delta,__m256i& g,__m256i& s) {
 auto v=join(iq1s_grid,q);
 // Mask each byte's low three bits after the 16-bit shift to remove cross-byte carry.
 auto v8=_mm256_and_si256(_mm256_slli_epi16(v,3),_mm256_set1_epi8(char(0xf8)));
 s=_mm256_add_epi8(v8,_mm256_set1_epi8(delta?-1:1));
 g=_mm256_abs_epi8(s);
}
int main() {
 for(int d=0;d<2;++d) for(unsigned i=0;i<2048;++i) {
  uint64_t v=iq1s_grid[i]; uint8_t g[8],s[8];
  for(int j=0;j<8;++j) { int w=8*int(int8_t(v>>(8*j)))+(d?-1:1); s[j]=uint8_t(w); g[j]=uint8_t(w<0?-w:w); }
  std::memcpy(&magnitude[d][i],g,8); std::memcpy(&sign_bytes[d][i],s,8);
 }
 for(int d=0;d<2;++d) for(unsigned i=0;i<2048;++i) {
  unsigned q[4]={i,(i+7)%2048,(i+109)%2048,(i+1337)%2048}; __m256i g,s;
  original(q,d,g,s);
  __m256i gs,ss; shifted(q,d,gs,ss);
  if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(g,gs))!=-1 ||
     _mm256_movemask_epi8(_mm256_cmpeq_epi8(s,ss))!=-1) return 2;
  if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(g,join(magnitude[d],q)))!=-1 ||
     _mm256_movemask_epi8(_mm256_cmpeq_epi8(s,join(sign_bytes[d],q)))!=-1) return 1;
 }
 std::puts("PASS all grid indices and delta signs, exact magnitude/sign bytes");
 unsigned indices[8192]; uint32_t state=137;
 for(auto& x:indices) {state=1664525*state+1013904223; x=(state>>16)&2047;}
 for(int mode=0;mode<3;++mode) {
  __m256i acc=_mm256_setzero_si256(); auto start=std::chrono::steady_clock::now();
  for(int r=0;r<10000;++r) for(int i=0;i<8192;i+=4) {
   __m256i g,s; int d=(i>>2)&1;
   if(mode==2) shifted(indices+i,d,g,s);
   else if(mode==1) {g=join(magnitude[d],indices+i); s=join(sign_bytes[d],indices+i);}
   else original(indices+i,d,g,s);
   acc=_mm256_xor_si256(acc,_mm256_add_epi8(g,s));
  }
  alignas(32) uint64_t out[4]; _mm256_store_si256((__m256i*)out,acc);
  std::printf("%s ms=%.3f checksum=%llu\n",mode==2?"shift-mask":mode?"table":"arithmetic",
   std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),
   (unsigned long long)out[0]);
 }
}
