#include "strata/host/glm_fp4.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);return 1;} } while(0)
static int check_payload(const uint8_t *input,size_t bytes) {
    uint8_t output[110],restored[108],inplace[108];
    memset(output,0xa5,sizeof(output));
    CHECK(!strata_glm_nvfp4_repack_k32(input,output+1,bytes));
    CHECK(output[0]==0xa5 && output[bytes+1]==0xa5);
    for(size_t b=0;b<bytes/36;++b) {
        const uint8_t *old=input+b*36,*packed=output+1+b*36;
        CHECK(!memcmp(old,packed,4));
        for(unsigned j=0;j<64;++j) {
            unsigned original=(old[4+(j/16)*8+j%8]>>((j%16)/8*4))&15;
            unsigned changed=(packed[4+(j/32)*16+j%16]>>((j%32)/16*4))&15;
            CHECK(original==changed);
        }
    }
    CHECK(!strata_glm_nvfp4_repack_k32(output+1,restored,bytes));
    CHECK(!memcmp(restored,input,bytes));
    memcpy(inplace,input,bytes);
    CHECK(!strata_glm_nvfp4_repack_k32(inplace,inplace,bytes));
    CHECK(!memcmp(inplace,output+1,bytes));
    CHECK(!strata_glm_nvfp4_repack_k32(inplace,inplace,bytes));
    CHECK(!memcmp(inplace,input,bytes));
    return 0;
}
int main(void) {
    uint8_t input[108],saved[108];
    for(unsigned pos=0;pos<36;++pos)for(unsigned value=0;value<256;++value) {
        for(unsigned j=0;j<108;++j)input[j]=(uint8_t)(j*73+19);
        input[pos]=(uint8_t)value;memcpy(saved,input,sizeof(input));
        CHECK(!check_payload(input,sizeof(input)));
        CHECK(!memcmp(input,saved,sizeof(input)));
    }
    CHECK(!check_payload(input,36));
    CHECK(!strata_glm_nvfp4_repack_k32(NULL,NULL,0));
    uint8_t overlap[110],before[110],output[108],sentinel[108];
    memset(overlap,0x39,sizeof(overlap));memcpy(before,overlap,sizeof(overlap));
    memset(output,0xa5,sizeof(output));memcpy(sentinel,output,sizeof(output));
    CHECK(strata_glm_nvfp4_repack_k32(overlap,overlap+1,108));
    CHECK(strata_glm_nvfp4_repack_k32(overlap+1,overlap,108));
    CHECK(!memcmp(overlap,before,sizeof(overlap)));
    CHECK(strata_glm_nvfp4_repack_k32(input,output,35));
    CHECK(strata_glm_nvfp4_repack_k32(input,output,37));
    CHECK(strata_glm_nvfp4_repack_k32(NULL,output,36));
    CHECK(strata_glm_nvfp4_repack_k32(input,NULL,36));
    CHECK(strata_glm_nvfp4_repack_k32(input,output,SIZE_MAX-SIZE_MAX%36));
    CHECK(!memcmp(output,sentinel,sizeof(output)));
    puts("PASS 9216 payload cases, all byte values/positions, independent nibble decode, scale preservation, guards, inverse/in-place, overlap/null/extent rejection");
    return 0;
}
