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
    assert(strata_glm_route_drop(w,255,0.6,7,&share)==0 && share==0);
    assert(strata_glm_route_drop_capped(w,255,0.75,7,0.75,&share)==63 && share==0.75);
    assert(strata_glm_route_drop_capped(w,255,0.6,7,0.5,&share)==0 && share==0);
    assert(strata_glm_route_drop_capped(w,255,0.6,7,0.8,&share)==0 && share==0);
    assert(strata_glm_route_drop_capped(w,255,0.6,7,NAN,&share)==0 && share==0);
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
        const double extended=(trial%76)/100.0;
        const uint8_t extra=strata_glm_route_drop_capped(w,eligible,extended,limit,0.75,&share);
        total=removed=0;count=0;
        for(unsigned i=0;i<8;++i) {
            total+=w[i];
            if(extra&(1u<<i)){removed+=w[i];++count;}
        }
        assert(!(extra & (uint8_t)~eligible));
        assert(count<=limit && count<8 && removed<=extended*total && share==removed/total);
        memcpy(copy,w,sizeof copy);
        const double extra_multiplier=strata_glm_route_renormalize(copy,extra);
        assert(extra_multiplier>=1 && extra_multiplier<=4.000001);
    }
    /* Batch rule: one row equals the capped rule; an expert shared by two rows is dropped for both or neither. */
    {
        float bw[16]={1,2,3,4,5,6,7,8, 1,2,3,4,5,6,7,8};
        int bid[16]={10,11,12,13,14,15,16,17, 10,21,22,23,24,25,26,27};
        uint8_t be[2]={255,255}, bm[2];double bs[2];
        strata_glm_route_drop_batch(bw,bid,be,1,0.1,2,0.5,bm,bs);
        assert(bm[0]==strata_glm_route_drop_capped(bw,255,0.1,2,0.5,&share) && bs[0]==share);
        strata_glm_route_drop_batch(bw,bid,be,2,0.1,2,0.5,bm,bs);
        assert(bm[0]==3 && bm[1]==3 && bs[0]==3.0/36 && bs[1]==3.0/36);
        be[1]=254;  /* expert 10 is not eligible in row 1: it stays on the GPU there, so row 0 may still drop it */
        strata_glm_route_drop_batch(bw,bid,be,2,0.1,2,0.5,bm,bs);
        assert(bm[0]==3 && bm[1]==2);
        be[1]=255;bw[8]=4;  /* row 1 cannot afford expert 10 (4 > 3.9): both rows keep it */
        strata_glm_route_drop_batch(bw,bid,be,2,0.1,2,0.5,bm,bs);
        assert(bm[0]==2 && bm[1]==2);
        strata_glm_route_drop_batch(bw,bid,be,2,0.6,7,0.5,bm,bs);
        assert(bm[0]==0 && bm[1]==0);
        float many_w[(STRATA_GLM_ROUTE_DROP_BATCH_ROWS+1)*8];int many_id[(STRATA_GLM_ROUTE_DROP_BATCH_ROWS+1)*8];
        uint8_t many_e[STRATA_GLM_ROUTE_DROP_BATCH_ROWS+1],many_m[STRATA_GLM_ROUTE_DROP_BATCH_ROWS+1];
        for(unsigned r=0;r<=STRATA_GLM_ROUTE_DROP_BATCH_ROWS;++r) {
            many_e[r]=255;many_m[r]=99;for(unsigned i=0;i<8;++i){many_w[r*8+i]=1;many_id[r*8+i]=(int)i;}
        }
        strata_glm_route_drop_batch(many_w,many_id,many_e,STRATA_GLM_ROUTE_DROP_BATCH_ROWS+1,0.5,7,0.5,many_m,0);
        for(unsigned r=0;r<=STRATA_GLM_ROUTE_DROP_BATCH_ROWS;++r)assert(many_m[r]==0);
    }
    /* Duplicate IDs invalidate the entire batch, even in one row or an
     * ineligible route: otherwise a group can exceed both per-row budgets. */
    for(unsigned rows=1;rows<=STRATA_GLM_ROUTE_DROP_BATCH_ROWS;++rows) {
        float weights[STRATA_GLM_ROUTE_DROP_BATCH_ROWS*8];
        int ids[STRATA_GLM_ROUTE_DROP_BATCH_ROWS*8];
        uint8_t eligible[STRATA_GLM_ROUTE_DROP_BATCH_ROWS],masks[STRATA_GLM_ROUTE_DROP_BATCH_ROWS];
        double shares[STRATA_GLM_ROUTE_DROP_BATCH_ROWS];
        for(unsigned r=0;r<rows;++r) {
            eligible[r]=3;masks[r]=99;shares[r]=99;
            for(unsigned i=0;i<8;++i){weights[r*8+i]=1;ids[r*8+i]=(int)i;}
        }
        ids[(rows-1)*8+1]=ids[(rows-1)*8];
        strata_glm_route_drop_batch(weights,ids,eligible,rows,.2,1,.5,masks,shares);
        for(unsigned r=0;r<rows;++r)assert(masks[r]==0 && shares[r]==0);
        eligible[rows-1]=1;
        strata_glm_route_drop_batch(weights,ids,eligible,rows,.2,1,.5,masks,shares);
        for(unsigned r=0;r<rows;++r)assert(masks[r]==0 && shares[r]==0);
    }
    for(unsigned trial=0;trial<2000;++trial) {
        enum {ROWS=32};
        float bw[ROWS*8];int bid[ROWS*8];uint8_t be[ROWS],bm[ROWS],again[ROWS];double bs[ROWS];
        const double budget=0.05+(trial%46)/100.0;
        const unsigned limit=1+trial%7;
        for(unsigned r=0;r<ROWS;++r) {
            rng=rng*1664525u+1013904223u;be[r]=(uint8_t)(rng>>9);
            for(unsigned i=0;i<8;++i) {
                rng=rng*1664525u+1013904223u;bw[r*8+i]=(rng%1000)/1000.f;
                int id;unsigned j;
                do {rng=rng*1664525u+1013904223u;id=(int)((rng>>8)%48);for(j=0;j<i && bid[r*8+j]!=id;++j){}} while(j<i);
                bid[r*8+i]=id;
            }
        }
        strata_glm_route_drop_batch(bw,bid,be,ROWS,budget,limit,0.5,bm,bs);
        strata_glm_route_drop_batch(bw,bid,be,ROWS,budget,limit,0.5,again,bs);
        assert(!memcmp(bm,again,sizeof bm));
        for(unsigned r=0;r<ROWS;++r) {
            double total=0,removed=0;unsigned count=0;
            for(unsigned i=0;i<8;++i){total+=bw[r*8+i];if(bm[r]&(1u<<i)){removed+=bw[r*8+i];++count;}}
            assert(!(bm[r]&(uint8_t)~be[r]) && count<=limit && removed<=budget*total && bs[r]==removed/total);
        }
        for(int e=0;e<48;++e) {  /* every eligible route of a dropped expert is dropped */
            int any=0,all=1;
            for(unsigned r=0;r<ROWS;++r)for(unsigned i=0;i<8;++i)if(bid[r*8+i]==e && (be[r]&(1u<<i))) {
                if(bm[r]&(1u<<i))any=1;else all=0;
            }
            assert(!any || all);
        }
    }
    return 0;
}
