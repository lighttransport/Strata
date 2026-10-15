#include "strata/host/glm_layer.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main(void) {
    int b[17], before[17], n, l, s;
    for (n = 1; n <= 16; ++n) {
        CHECK(!strata_glm_layer_bounds("auto",45,n,b));
        CHECK(b[0]==0 && b[n]==45);
        for (s=0;s<n;++s) CHECK(b[s]<b[s+1]);
        for (l=0;l<=45;++l) {
            int owners=0; char name[64];
            snprintf(name,sizeof(name),"blk.%d.attn_norm.weight",l);
            for (s=0;s<n;++s) owners+=strata_glm_layer_tensor_owned(name,b[s],b[s+1],45,1);
            CHECK(owners==1);
        }
    }
    CHECK(!strata_glm_layer_bounds("24",45,2,b) && b[1]==24);
    CHECK(!strata_glm_layer_bounds("3,4,44",45,4,b));
    memset(b,7,sizeof(b)); memcpy(before,b,sizeof(b));
    const char *bad[]={"","0","45","-1","+24"," 24","24 ","24,","24,25","999999999999999999999"};
    for (s=0;s<(int)(sizeof(bad)/sizeof(*bad));++s) {
        CHECK(strata_glm_layer_bounds(bad[s],45,2,b)==-1);
        CHECK(!memcmp(b,before,sizeof(b)));
    }
    CHECK(strata_glm_layer_bounds("4,3",45,3,b)==-1);
    CHECK(strata_glm_layer_bounds("auto",2,3,b)==-1);
    CHECK(strata_glm_layer_tensor_owned("token_embd.weight",0,24,45,1));
    CHECK(!strata_glm_layer_tensor_owned("output.weight",0,24,45,1));
    CHECK(strata_glm_layer_tensor_owned("output.weight",24,45,45,1));
    CHECK(!strata_glm_layer_tensor_owned("blk.45.attn_norm.weight",24,45,45,0));
    CHECK(!strata_glm_layer_tensor_owned("blk.x.bad",24,45,45,1));
    return 0;
}
