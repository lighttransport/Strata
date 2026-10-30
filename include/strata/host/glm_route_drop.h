/* Optional approximation for eight routed expert contributions. The budget
 * is a fraction of ORIGINAL total routing weight; selection does not renormalize. A
 * routing-mass bound is not an output-error bound. Portable C11 policy. */
#ifndef STRATA_HOST_GLM_ROUTE_DROP_H_
#define STRATA_HOST_GLM_ROUTE_DROP_H_
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
/* An explicit experimental cap may extend the original 0.5 bound, up to 0.75.
 * The original entry point below always retains its original cap. */
static inline uint8_t strata_glm_route_drop_capped(const float weights[8], uint8_t eligible,
                                                  double fraction, unsigned max_drop,
                                                  double cap, double *dropped_share) {
    unsigned order[8], count=0, dropped=0;
    double total=0, removed=0;
    uint8_t mask=0;
    if(dropped_share)*dropped_share=0;
    if(!weights || !isfinite(cap) || cap<=0 || cap>0.75 ||
       !isfinite(fraction) || fraction<=0 || fraction>cap || !max_drop)return 0;
    if(max_drop>7)max_drop=7;
    for(unsigned i=0;i<8;++i) {
        if(!isfinite(weights[i]) || weights[i]<0)return 0;
        total+=weights[i];
        if(eligible&(1u<<i)) {
            unsigned j=count++;
            while(j && weights[order[j-1]]>weights[i]) {order[j]=order[j-1];--j;}
            order[j]=i;
        }
    }
    if(total<=0)return 0;
    for(unsigned j=0;j<count && dropped<max_drop;++j) {
        const unsigned i=order[j];
        if(removed+weights[i]>fraction*total)break;
        removed+=weights[i];mask|=(uint8_t)(1u<<i);++dropped;
    }
    if(dropped_share)*dropped_share=removed/total;
    return mask;
}
static inline uint8_t strata_glm_route_drop(const float weights[8], uint8_t eligible,
                                           double fraction, unsigned max_drop,
                                           double *dropped_share) {
    return strata_glm_route_drop_capped(weights,eligible,fraction,max_drop,0.5,dropped_share);
}
/* Batch-aware selection for rows decoded together (opt-in): every row keeps the same budget as
 * strata_glm_route_drop_capped (a fraction of its ORIGINAL total weight, at most max_drop routes, eligible routes
 * only), but an expert is dropped for all of its rows or for none, since a CPU expert that keeps any row still
 * costs a full weight pass. Experts go in ascending order of their largest share of a user row's budget, then fewer
 * users, then id; one is dropped only if every user row can still afford it. One row gives exactly the capped result.
 * weights and ids are rows x 8; nonnegative expert IDs must be unique within a row.
 * masks (and shares, if given) receive one entry per row. Invalid input drops nothing. */
#define STRATA_GLM_ROUTE_DROP_BATCH_ROWS 64u
typedef struct { int expert; unsigned route; double share; } strata_glm_route_drop_entry;
typedef struct { int expert; unsigned first, count; double share; } strata_glm_route_drop_group;
static inline int strata_glm_route_drop_entry_order(const void *a, const void *b) {
    const strata_glm_route_drop_entry *x=(const strata_glm_route_drop_entry*)a, *y=(const strata_glm_route_drop_entry*)b;
    if(x->expert!=y->expert)return x->expert<y->expert?-1:1;
    return x->route<y->route?-1:x->route>y->route;
}
static inline int strata_glm_route_drop_group_order(const void *a, const void *b) {
    const strata_glm_route_drop_group *x=(const strata_glm_route_drop_group*)a, *y=(const strata_glm_route_drop_group*)b;
    if(x->share!=y->share)return x->share<y->share?-1:1;
    if(x->count!=y->count)return x->count<y->count?-1:1;
    return x->expert<y->expert?-1:x->expert>y->expert;
}
static inline void strata_glm_route_drop_batch(const float *weights, const int *ids, const uint8_t *eligible,
                                               unsigned rows, double fraction, unsigned max_drop, double cap,
                                               uint8_t *masks, double *shares) {
    strata_glm_route_drop_entry entries[STRATA_GLM_ROUTE_DROP_BATCH_ROWS*8];
    strata_glm_route_drop_group groups[STRATA_GLM_ROUTE_DROP_BATCH_ROWS*8];
    double total[STRATA_GLM_ROUTE_DROP_BATCH_ROWS], removed[STRATA_GLM_ROUTE_DROP_BATCH_ROWS];
    unsigned dropped[STRATA_GLM_ROUTE_DROP_BATCH_ROWS], count=0, group_count=0;
    if(!masks)return;
    for(unsigned r=0;r<rows;++r) {masks[r]=0;if(shares)shares[r]=0;}
    if(!weights || !ids || !eligible || !rows || rows>STRATA_GLM_ROUTE_DROP_BATCH_ROWS)return;
    for(unsigned r=0;r<rows;++r)for(unsigned i=0;i<8;++i)if(ids[r*8+i]>=0)
        for(unsigned j=0;j<i;++j)if(ids[r*8+i]==ids[r*8+j])return;
    if(rows==1) {masks[0]=strata_glm_route_drop_capped(weights,eligible[0],fraction,max_drop,cap,shares);return;}
    if(!isfinite(cap) || cap<=0 || cap>0.75 || !isfinite(fraction) || fraction<=0 || fraction>cap || !max_drop)return;
    if(max_drop>7)max_drop=7;
    for(unsigned r=0;r<rows;++r) {
        total[r]=0;removed[r]=0;dropped[r]=0;
        for(unsigned i=0;i<8;++i) {
            const float w=weights[r*8+i];
            if(!isfinite(w) || w<0)return;
            total[r]+=w;
        }
    }
    for(unsigned r=0;r<rows;++r)if(total[r]>0)for(unsigned i=0;i<8;++i)
        if((eligible[r]&(1u<<i)) && ids[r*8+i]>=0) {
            entries[count].expert=ids[r*8+i];entries[count].route=r*8+i;
            entries[count].share=weights[r*8+i]/(fraction*total[r]);++count;
        }
    qsort(entries,count,sizeof entries[0],strata_glm_route_drop_entry_order);
    for(unsigned j=0;j<count;) {
        unsigned k=j;double share=0;
        while(k<count && entries[k].expert==entries[j].expert) {if(entries[k].share>share)share=entries[k].share;++k;}
        groups[group_count].expert=entries[j].expert;groups[group_count].first=j;
        groups[group_count].count=k-j;groups[group_count].share=share;++group_count;
        j=k;
    }
    qsort(groups,group_count,sizeof groups[0],strata_glm_route_drop_group_order);
    for(unsigned g=0;g<group_count;++g) {
        const strata_glm_route_drop_group *group=&groups[g];
        int fits=1;
        for(unsigned j=group->first;j<group->first+group->count && fits;++j) {
            const unsigned r=entries[j].route/8;
            if(dropped[r]>=max_drop || removed[r]+weights[entries[j].route]>fraction*total[r])fits=0;
        }
        if(!fits)continue;
        for(unsigned j=group->first;j<group->first+group->count;++j) {
            const unsigned r=entries[j].route/8, i=entries[j].route%8;
            removed[r]+=weights[entries[j].route];++dropped[r];masks[r]|=(uint8_t)(1u<<i);
        }
    }
    if(shares)for(unsigned r=0;r<rows;++r)shares[r]=total[r]>0?removed[r]/total[r]:0;
}
/* Optional compensation after selection: restore the original total routing
 * weight across retained routes. The drop budget always refers to the original
 * weights. Return the multiplier, or zero without changing invalid input. */
static inline double strata_glm_route_renormalize(float weights[8], uint8_t discarded) {
    double total=0,kept=0;
    if(!weights)return 0;
    for(unsigned i=0;i<8;++i) {
        if(!isfinite(weights[i]) || weights[i]<0)return 0;
        total+=weights[i];
        if(!(discarded&(1u<<i)))kept+=weights[i];
    }
    if(kept<=0)return 0;
    const double factor=total/kept;
    for(unsigned i=0;i<8;++i)
        if(!(discarded&(1u<<i)) && weights[i]*factor>FLT_MAX)return 0;
    for(unsigned i=0;i<8;++i)weights[i]=(discarded&(1u<<i))?0:(float)(weights[i]*factor);
    return factor;
}
#endif
