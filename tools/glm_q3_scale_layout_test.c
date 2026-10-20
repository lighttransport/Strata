#include "strata/host/q3_layout.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

/* Write each of the six encoded bits independently of the converter's masks. */
static void set_scale(uint8_t *b, unsigned position, unsigned value) {
    for (unsigned bit=0; bit<6; ++bit) {
        unsigned byte, shift;
        if (bit<4) { byte=96+position%8; shift=bit+4*(position>=8); }
        else { byte=104+position%4; shift=(bit-4)+2*(position/4); }
        b[byte]=(uint8_t)((b[byte]&~(1u<<shift))|(((value>>bit)&1u)<<shift));
    }
}

static void fixtures(void) {
    uint8_t source[110], expanded[116], restored[112];
    for (unsigned p=0; p<16; ++p) for (unsigned v=0; v<64; ++v) {
        for (unsigned i=0; i<110; ++i) source[i]=(uint8_t)(i*37+v);
        for (unsigned j=0; j<16; ++j) set_scale(source,j,j==p?v:32);
        memset(expanded,0xa5,sizeof expanded); memset(restored,0x5a,sizeof restored);
        CHECK(strata_q3k_expand_scales(source,110,expanded+1,114)==0);
        CHECK(expanded[0]==0xa5 && expanded[115]==0xa5);
        CHECK(memcmp(source,expanded+1,96)==0);
        CHECK(memcmp(source+108,expanded+113,2)==0);
        for (unsigned j=0; j<16; ++j) CHECK(expanded[97+j]==(uint8_t)(j==p?(int)v-32:0));
        CHECK(strata_q3k_compact_scales(expanded+1,114,restored+1,110)==0);
        CHECK(restored[0]==0x5a && restored[111]==0x5a);
        CHECK(memcmp(source,restored+1,110)==0);
    }
    uint8_t src[110*17], dst[114*17], back[110*17];
    uint32_t rng=0x713b9;
    for (size_t i=0;i<sizeof src;++i) { rng=rng*1664525u+1013904223u; src[i]=(uint8_t)(rng>>24); }
    CHECK(strata_q3k_expand_scales(src,sizeof src,dst,sizeof dst)==0);
    CHECK(strata_q3k_compact_scales(dst,sizeof dst,back,sizeof back)==0);
    CHECK(memcmp(src,back,sizeof src)==0);
    for (unsigned p=0;p<16;++p) for (unsigned v=32;v<224;++v) {
        uint8_t saved=dst[16*114+96+p]; dst[16*114+96+p]=(uint8_t)v;
        memset(back,0x5a,sizeof back);
        CHECK(strata_q3k_compact_scales(dst,sizeof dst,back,sizeof back)==-1);
        for (size_t i=0;i<sizeof back;++i) CHECK(back[i]==0x5a);
        dst[16*114+96+p]=saved;
    }
}

typedef int (*convert_fn)(const uint8_t *,size_t,uint8_t *,size_t);
static void invalid(convert_fn fn,size_t si,size_t so) {
    uint8_t a[512], b[512], saved[512];
    memset(a,0xa5,sizeof a); memset(b,0x5a,sizeof b); memcpy(saved,a,sizeof a);
    CHECK(fn(NULL,0,NULL,0)==0);
    CHECK(fn(a,0,b,1)==-1);
    CHECK(fn(a,si,b,0)==-1);
    CHECK(fn(a,si-1,b,so)==-1);
    CHECK(fn(a,si,b,so+1)==-1);
    CHECK(fn(NULL,si,b,so)==-1);
    CHECK(fn(a,si,NULL,so)==-1);
    CHECK(fn(a,si,a,so)==-1);
    CHECK(fn(a,si,a+1,so)==-1);
    CHECK(fn(a+1,si,a,so)==-1);
    CHECK(fn(a,si,a+si-1,so)==-1);
    CHECK(fn(a+so-1,si,a,so)==-1);
    CHECK(fn((const uint8_t *)(UINTPTR_MAX-si+1),si,b,so)==-1);
    CHECK(fn(a,si,(uint8_t *)(UINTPTR_MAX-so+1),so)==-1);
    CHECK(fn(a,SIZE_MAX,b,so)==-1);
    if (so>si) CHECK(fn(a,(SIZE_MAX/si)*si,b,so)==-1);
    CHECK(memcmp(a,saved,sizeof a)==0);
    for(size_t i=0;i<sizeof b;++i) CHECK(b[i]==0x5a);
}
static void experts(void) {
    for(size_t h=256;h<=512;h+=256)for(size_t f=256;f<=512;f+=256) {
        strata_q2q3_layout a,b;
        CHECK(!strata_q2q3_scale_layout(h,f,0,&a));
        CHECK(!strata_q2q3_scale_layout(h,f,1,&b));
        CHECK(a.gu_row==h/256*84 && a.d_row==f/256*110 && b.d_row==f/256*114);
        CHECK(a.down_off==2*(h/256*84)*f && b.down_off==a.down_off);
        CHECK(a.bytes==a.down_off+h*(f/256)*110 && b.bytes==b.down_off+h*(f/256)*114);
        uint8_t *src=malloc(a.bytes),*dst=malloc(b.bytes+2),*back=malloc(a.bytes);
        CHECK(src&&dst&&back);
        for(size_t j=0;j<a.bytes;++j)src[j]=(uint8_t)(j*37+19);
        memset(dst,0xa5,b.bytes+2);
        CHECK(!strata_q2q3_expand_down(h,f,src,a.bytes,dst+1,b.bytes));
        CHECK(dst[0]==0xa5 && dst[b.bytes+1]==0xa5);
        CHECK(!memcmp(src,dst+1,a.down_off));
        CHECK(!strata_q3k_compact_scales(dst+1+b.down_off,b.bytes-b.down_off,back+a.down_off,a.bytes-a.down_off));
        CHECK(!memcmp(src+a.down_off,back+a.down_off,a.bytes-a.down_off));
        memset(dst,0xa5,b.bytes+2);
        CHECK(strata_q2q3_expand_down(h,f,src,a.bytes-1,dst+1,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h,f,src,a.bytes,dst+1,b.bytes-1)==-1);
        CHECK(strata_q2q3_expand_down(h,f,NULL,a.bytes,dst+1,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h,f,src,a.bytes,NULL,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h,f,dst,a.bytes,dst+1,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h,f,dst+1,a.bytes,dst,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h,f,dst,a.bytes,dst,b.bytes)==-1);
        CHECK(strata_q2q3_expand_down(h+1,f,src,a.bytes,dst+1,b.bytes)==-1);
        for(size_t j=0;j<b.bytes+2;++j)CHECK(dst[j]==0xa5);
        free(src);free(dst);free(back);
    }
    strata_q2q3_layout x,before;memset(&x,0xa5,sizeof x);memcpy(&before,&x,sizeof x);
    CHECK(strata_q2q3_scale_layout(0,256,0,&x)==-1);
    CHECK(strata_q2q3_scale_layout(256,257,1,&x)==-1);
    CHECK(strata_q2q3_scale_layout(256,256,2,&x)==-1);
    CHECK(strata_q2q3_scale_layout(SIZE_MAX/256*256,256,1,&x)==-1);
    CHECK(strata_q2q3_scale_layout(256,SIZE_MAX/256*256,0,&x)==-1);
    CHECK(strata_q2q3_scale_layout(256,256,0,NULL)==-1);
    CHECK(!memcmp(&x,&before,sizeof x));
}
int main(void) {
    fixtures();experts();
    invalid(strata_q3k_expand_scales,110,114);
    invalid(strata_q3k_compact_scales,114,110);
    puts("Q3 expanded scales: 1024 fixtures, 3072 invalid scales, round trips and bounds passed");
    return 0;
}
