#include "strata/host/glm_fp4_worker.h"
#include "strata/host/glm_fp4_layout.h"
#undef NDEBUG
#include <assert.h>
int main(void) {
    uint32_t h[8]={STRATA_GLM_FP4_WORKER_MAGIC,1,1,1,3,0,8,0};
    assert(strata_glm_fp4_worker_command(h,1,213,100,0));
    assert(!strata_glm_fp4_worker_command(h,1,213,100,1));
    h[6]=9;assert(!strata_glm_fp4_worker_command(h,1,213,100,0));h[6]=1;
    h[5]=213;assert(!strata_glm_fp4_worker_command(h,1,213,100,0));h[5]=0;
    h[4]=45;assert(!strata_glm_fp4_worker_command(h,1,213,100,0));h[4]=3;
    assert(!strata_glm_fp4_worker_command(h,2,213,100,0));
    h[2]=2;h[4]=h[5]=h[6]=0;h[7]=99;
    assert(strata_glm_fp4_worker_command(h,1,213,100,1));
    h[7]=100;assert(!strata_glm_fp4_worker_command(h,1,213,100,1));
    h[2]=0;h[7]=0;assert(strata_glm_fp4_worker_command(h,1,213,100,1));
    assert(!strata_glm_fp4_worker_command(h,UINT32_MAX,213,100,1));
    assert(strata_glm_fp4_layout_admit(10,2,12));
    assert(!strata_glm_fp4_layout_admit(UINT64_MAX,1,UINT64_MAX));
    assert(!strata_glm_fp4_layout_span(UINT64_MAX,1,UINT64_MAX));
    assert(strata_glm_fp4_layout_owner(45,287,288)==0);
    assert(strata_glm_fp4_layout_owner(3,288,288)==-1);
    return 0;
}
