#include "strata/host/glm_fp4_cache.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>
static uint64_t hits(const strata_glm_fp4_cache_entry *e, size_t n) {
    uint64_t total=0;
    for(size_t i=0;i<n;++i)if(e[i].owner>=0)total+=e[i].frequency;
    return total;
}
int main(void) {
    /* Hot even experts exceed device 0 capacity. Mode 2 can use the other
     * device's free slots; no selected expert appears on both devices. */
    const strata_glm_fp4_cache_entry input[]={
        {4,0,100,77},{3,2,90,77},{3,4,80,77},{4,1,10,77},{3,3,9,77},{4,5,8,77}};
    strata_glm_fp4_cache_entry e[6], again[6];size_t slots[2]={1,2};
    memcpy(e,input,sizeof(e));assert(strata_glm_fp4_cache_select(e,6,slots,2,1));
    assert(hits(e,6)==119);size_t counts[2]={0,0};
    for(size_t i=0;i<6;++i)if(e[i].owner>=0){assert(e[i].owner==e[i].expert%2);++counts[e[i].owner];}
    assert(counts[0]==1&&counts[1]==2);
    memcpy(e,input,sizeof(e));assert(strata_glm_fp4_cache_select(e,6,slots,2,2));
    assert(hits(e,6)==270&&e[0].owner==0&&e[1].owner==1&&e[2].owner==1);
    memcpy(again,input,sizeof(again));assert(strata_glm_fp4_cache_select(again,6,slots,2,2));
    assert(memcmp(e,again,sizeof(e))==0);
    /* Stable tie rules do not depend on input order. */
    strata_glm_fp4_cache_entry ties[]={{4,2,1,77},{3,5,1,77},{3,1,1,77}};
    assert(strata_glm_fp4_cache_select(ties,3,slots,2,2));
    assert(ties[0].layer==3&&ties[0].expert==1&&ties[1].expert==5&&ties[2].layer==4);
    slots[0]=0;slots[1]=0;
    assert(strata_glm_fp4_cache_select(e,6,slots,2,2)&&hits(e,6)==0);
    slots[0]=100;
    assert(strata_glm_fp4_cache_select(e,6,slots,1,2)&&hits(e,6)==297);
    for(size_t i=0;i<6;++i)assert(e[i].owner==0);
    memcpy(e,input,sizeof(e));
    assert(!strata_glm_fp4_cache_select(e,6,slots,3,2)&&memcmp(e,input,sizeof(e))==0);
    assert(!strata_glm_fp4_cache_select(e,6,slots,2,0)&&memcmp(e,input,sizeof(e))==0);
    e[5].expert=-1;memcpy(again,e,sizeof(e));
    assert(!strata_glm_fp4_cache_select(e,6,slots,2,2)&&memcmp(e,again,sizeof(e))==0);
    assert(strata_glm_fp4_cache_select(NULL,0,slots,1,1));
    assert(!strata_glm_fp4_cache_select(NULL,1,slots,1,1));
    /* Exhaust all small capacities against the frequency-sorted optimum.
     * Mode 1 is optimal within parity; mode 2 uses the combined capacity. */
    for(size_t a=0;a<=6;++a)for(size_t b=0;b<=6;++b)for(int mode=1;mode<=2;++mode) {
        slots[0]=a;slots[1]=b;memcpy(e,input,sizeof(e));
        assert(strata_glm_fp4_cache_select(e,6,slots,2,mode));
        size_t used[2]={0,0},want_used[2]={0,0},admitted=0;uint64_t want=0;
        for(size_t i=0;i<6;++i) {
            if(e[i].owner>=0){assert(e[i].owner<2);++used[e[i].owner];}
            unsigned parity=(unsigned)e[i].expert%2;
            if(mode==2?admitted<a+b:want_used[parity]<slots[parity]) {
                want+=e[i].frequency;++admitted;++want_used[parity];
            }
        }
        assert(used[0]<=a&&used[1]<=b&&hits(e,6)==want);
    }
    uint64_t heat[288]={0};uint8_t keep[288]={0},before[288];int removed[288],added[288];
    keep[0]=keep[2]=keep[4]=1;heat[0]=5;heat[2]=3;heat[4]=1;
    heat[6]=10;heat[8]=8;heat[10]=7;heat[1]=UINT64_MAX;
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,2,2,removed,added)==2);
    assert(removed[0]==4&&removed[1]==2&&added[0]==6&&added[1]==8);
    assert(keep[0]&&keep[6]&&keep[8]&&!keep[1]);
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,2,3,removed,added)==0);
    /* Equal scores do not churn, and integer extremes cannot wrap a threshold. */
    heat[10]=5;assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,2,0,removed,added)==0);
    heat[10]=UINT64_MAX;
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,1,UINT64_MAX,removed,added)==0);
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,1,2,removed,added)==1&&added[0]==10);
    memset(keep,0,sizeof(keep));
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,2,2,removed,added)==0);
    keep[1]=1;memcpy(before,keep,sizeof(keep));
    assert(strata_glm_fp4_cache_adapt(heat,keep,0,2,2,2,removed,added)==-1&&memcmp(keep,before,sizeof(keep))==0);
    keep[1]=2;assert(strata_glm_fp4_cache_adapt(heat,keep,1,2,2,2,removed,added)==-1);
    assert(strata_glm_fp4_cache_adapt(heat,keep,2,2,2,2,removed,added)==-1);
    assert(strata_glm_fp4_cache_adapt(heat,keep,1,2,289,2,removed,added)==-1);
    puts("FP4 global and adaptive cache admission passed");return 0;
}
