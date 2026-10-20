#include "strata/host/q3_layout.h"
#include <string.h>
static int valid(const uint8_t *input,size_t ni,uint8_t *output,size_t no,size_t si,size_t so) {
    if(ni%si || ni/si>SIZE_MAX/so || no!=(ni/si)*so)return 0;
    if(!ni)return 1;
    if(!input || !output)return 0;
    const uintptr_t a=(uintptr_t)input,b=(uintptr_t)output;
    if(ni>UINTPTR_MAX-a || no>UINTPTR_MAX-b)return 0;
    return a<=b ? b-a>=ni : a-b>=no;
}
int strata_q3k_expand_scales(const uint8_t *input,size_t ni,uint8_t *output,size_t no) {
    if(!valid(input,ni,output,no,110,114))return -1;
    for(size_t b=0;b<ni/110;++b) {
        const uint8_t *src=input+b*110;uint8_t *dst=output+b*114;
        memcpy(dst,src,96);
        uint32_t a,c,t,v[4];
        memcpy(&a,src+96,4);memcpy(&c,src+100,4);memcpy(&t,src+104,4);
        const uint32_t lo=0x0f0f0f0fu,hi=0x03030303u;
        v[0]=(a&lo)|((t&hi)<<4);
        v[1]=(c&lo)|(((t>>2)&hi)<<4);
        v[2]=((a>>4)&lo)|(((t>>4)&hi)<<4);
        v[3]=((c>>4)&lo)|(((t>>6)&hi)<<4);
        for(unsigned j=0;j<4;++j) {
            // Per byte: preserve low five bits; bit 5 selects sign extension.
            // Each product lane is 0 or 224, so multiplication carries no bits
            // between bytes. Operations are independent of host byte order.
            v[j]=(v[j]&0x1f1f1f1fu)|((~v[j]&0x20202020u)*7u);
        }
        memcpy(dst+96,v,16);
        memcpy(dst+112,src+108,2);
    }
    return 0;
}
int strata_q3k_compact_scales(const uint8_t *input,size_t ni,uint8_t *output,size_t no) {
    if(!valid(input,ni,output,no,114,110))return -1;
    for(size_t b=0;b<ni/114;++b)for(unsigned j=0;j<16;++j) {
        const unsigned s=input[b*114+96+j];
        if(s>31 && s<224)return -1;
    }
    for(size_t b=0;b<ni/114;++b) {
        const uint8_t *src=input+b*114;uint8_t *dst=output+b*110;
        memcpy(dst,src,96);memset(dst+96,0,12);
        for(unsigned j=0;j<16;++j) {
            const unsigned s=(src[96+j]+32)&63;
            dst[96+j%8]|=(uint8_t)((s&15)<<(j/8*4));
            dst[104+j%4]|=(uint8_t)((s>>4)<<(j/4*2));
        }
        memcpy(dst+108,src+112,2);
    }
    return 0;
}

int strata_q2q3_scale_layout(size_t embedding,size_t hidden,int expanded,strata_q2q3_layout *layout) {
    if(!layout || !embedding || !hidden || embedding%256 || hidden%256 ||
       (expanded!=0 && expanded!=1))return -1;
    const size_t block=expanded?STRATA_Q3K_EXPANDED_BLOCK_BYTES:STRATA_Q3K_GGML_BLOCK_BYTES;
    strata_q2q3_layout v;
    if(embedding/256>SIZE_MAX/84 || hidden/256>SIZE_MAX/block)return -1;
    v.gu_row=embedding/256*84;v.d_row=hidden/256*block;
    if(v.gu_row>SIZE_MAX/hidden)return -1;
    v.up_off=v.gu_row*hidden;
    if(v.up_off>SIZE_MAX/2 || v.d_row>SIZE_MAX/embedding)return -1;
    v.down_off=2*v.up_off;
    const size_t down_bytes=v.d_row*embedding;
    if(down_bytes>SIZE_MAX-v.down_off)return -1;
    v.bytes=v.down_off+down_bytes;
    *layout=v;return 0;
}
int strata_q2q3_expand_down(size_t embedding,size_t hidden,const uint8_t *input,size_t ni,uint8_t *output,size_t no) {
    strata_q2q3_layout old,new_layout;
    if(strata_q2q3_scale_layout(embedding,hidden,0,&old) ||
       strata_q2q3_scale_layout(embedding,hidden,1,&new_layout) ||
       ni!=old.bytes || no!=new_layout.bytes || !valid(input,ni,output,no,old.bytes,new_layout.bytes))return -1;
    /* All validation is complete. The down spans are whole blocks and disjoint. */
    if(strata_q3k_expand_scales(input+old.down_off,ni-old.down_off,
                              output+new_layout.down_off,no-new_layout.down_off))return -1;
    memcpy(output,input,old.down_off);
    return 0;
}
