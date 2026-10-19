/* Optional approximation for eight routed expert contributions. The budget
 * is a fraction of ORIGINAL total routing weight; selection does not renormalize. A
 * routing-mass bound is not an output-error bound. Portable C11 policy. */
#ifndef STRATA_HOST_GLM_ROUTE_DROP_H_
#define STRATA_HOST_GLM_ROUTE_DROP_H_
#include <math.h>
#include <float.h>
#include <stdint.h>
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
