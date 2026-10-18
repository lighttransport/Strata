#include "strata/host/glm_route_drop.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
int main(void) {
    float w[8]={1,2,3,4,5,6,7,8};
    double share=99;
    assert(strata_glm_route_drop(w,255,0,2,&share)==0 && share==0);
    assert(strata_glm_route_drop(w,255,0.1,2,&share)==3 && share==3.0/36);
    assert(strata_glm_route_drop(w,255,0.1,1,&share)==1 && share==1.0/36);
    assert(strata_glm_route_drop(w,252,0.1,2,&share)==4 && share==3.0/36);
    assert(strata_glm_route_drop(w,255,0.5,0,&share)==0);
    for(unsigned i=0;i<8;++i)w[i]=1;
    assert(strata_glm_route_drop(w,255,0.25,7,&share)==3 && share==0.25);
    assert(strata_glm_route_drop(w,255,0.124,7,&share)==0);
    w[2]=NAN;assert(strata_glm_route_drop(w,255,0.5,7,&share)==0);
    w[2]=INFINITY;assert(strata_glm_route_drop(w,255,0.5,7,&share)==0);
    w[2]=-1;assert(strata_glm_route_drop(w,255,0.5,7,&share)==0);
    w[2]=0;assert(strata_glm_route_drop(w,255,INFINITY,7,&share)==0);
    float normalized[8]={1,2,3,4,5,6,7,8};
    const double factor=strata_glm_route_renormalize(normalized,3);
    assert(factor==36.0/33 && normalized[0]==0 && normalized[1]==0);
    double sum=0;for(unsigned i=0;i<8;++i)sum+=normalized[i];
    assert(fabs(sum-36)<1.e-5);
    for(unsigned i=2;i<8;++i)assert(normalized[i]==(float)((i+1)*factor));
    float frozen[8];memcpy(frozen,normalized,sizeof frozen);
    assert(strata_glm_route_renormalize(normalized,255)==0);
    assert(!memcmp(frozen,normalized,sizeof frozen));
    uint32_t rng=17;
    for(unsigned trial=0;trial<10000;++trial) {
        double total=0,removed=0;unsigned count=0;
        for(unsigned i=0;i<8;++i) {rng=rng*1664525u+1013904223u;w[i]=(rng%1000)/1000.f;total+=w[i];}
        const uint8_t eligible=(uint8_t)(rng>>12);
        const double budget=(trial%51)/100.0;
        const unsigned limit=trial%8;
        const uint8_t mask=strata_glm_route_drop(w,eligible,budget,limit,&share);
        assert(!(mask & (uint8_t)~eligible));
        for(unsigned i=0;i<8;++i)if(mask&(1u<<i)){removed+=w[i];++count;}
        assert(count<=limit && count<8 && removed<=budget*total);
        assert(share==removed/total);
        float copy[8];memcpy(copy,w,sizeof copy);
        const double multiplier=strata_glm_route_renormalize(copy,mask);
        assert(multiplier>=1 && multiplier<=2.000001);
        double restored=0;
        for(unsigned i=0;i<8;++i) {
            if(mask&(1u<<i))assert(copy[i]==0);
            else assert(copy[i]==(float)(w[i]*multiplier));
            restored+=copy[i];
        }
        assert(fabs(restored-total)<=total*1.e-6);
    }
    return 0;
}
