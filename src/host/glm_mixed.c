#include "strata/host/glm_mixed.h"
#include <math.h>
#include <string.h>

_Static_assert(offsetof(struct strata_mixed_buffer, activation)==offsetof(struct strata_mixed_buffer_v5, activation), "mixed activation prefix");
_Static_assert(offsetof(struct strata_mixed_buffer, output)==offsetof(struct strata_mixed_buffer_v5, output), "mixed output prefix");
_Static_assert(sizeof(struct strata_mixed_buffer)==143413600, "v4 wire ABI");
uint32_t strata_mixed_protocol(const char *value) {
    if (!value || !strcmp(value,"4")) return STRATA_MIXED_VERSION;
    return !strcmp(value,"5") ? STRATA_MIXED_VERSION_FAST : 0;
}
const struct strata_mixed_layout *strata_mixed_protocol_layout(uint32_t version) {
#define OFFSET(type, field) offsetof(struct type, field)
    static const struct strata_mixed_layout v4={4,512,sizeof(struct strata_mixed_buffer),
        OFFSET(strata_mixed_buffer,expert), OFFSET(strata_mixed_buffer,destination),
        OFFSET(strata_mixed_buffer,route_weight), OFFSET(strata_mixed_buffer,primary_owned),
        OFFSET(strata_mixed_buffer,worker_owned), OFFSET(strata_mixed_buffer,chunk_destination),
        OFFSET(strata_mixed_buffer,activation), OFFSET(strata_mixed_buffer,output), 0};
    static const struct strata_mixed_layout v5={5,8192,sizeof(struct strata_mixed_buffer_v5),
        OFFSET(strata_mixed_buffer_v5,expert), OFFSET(strata_mixed_buffer_v5,destination),
        OFFSET(strata_mixed_buffer_v5,route_weight), OFFSET(strata_mixed_buffer_v5,primary_owned),
        OFFSET(strata_mixed_buffer_v5,worker_owned), OFFSET(strata_mixed_buffer_v5,chunk_destination),
        OFFSET(strata_mixed_buffer_v5,activation), OFFSET(strata_mixed_buffer_v5,output),
        OFFSET(strata_mixed_buffer_v5,activation_half)};
#undef OFFSET
    return version==4 ? &v4 : version==5 ? &v5 : NULL;
}

int strata_mixed_select_cache(uint32_t experts, uint32_t capacity,
    const int *order, const double *score, const float *counts,
    const uint8_t *previous, const uint8_t *primary, uint32_t max_replacements,
    float minimum_routes, double minimum_gain_percent, uint8_t *selected) {
    uint8_t seen[288]={0}, chosen[288]={0};
    uint32_t used=0, replaced=0;
    if (!experts || experts>288 || capacity>experts || !order || !score || !counts ||
        !previous || !primary || !selected || !isfinite(minimum_routes) || minimum_routes<0 ||
        !isfinite(minimum_gain_percent) || minimum_gain_percent<0 || minimum_gain_percent>1000) return 0;
    for (uint32_t i=0;i<experts;++i) {
        const int e=order[i];
        if (e<0 || (uint32_t)e>=experts || seen[e] || !isfinite(score[e]) || score[e]<0 ||
            !isfinite(counts[e]) || counts[e]<0 || (i && score[order[i-1]]<score[e])) return 0;
        seen[e]=1;
    }
    /* Keep the strongest old owners if the requested capacity shrank. */
    for (uint32_t i=0;i<experts && used<capacity;++i) {
        const int e=order[i];
        if (previous[e] && !primary[e]) {chosen[e]=1;++used;}
    }
    /* Empty/displaced slots need no eviction: fill them without a count gate. */
    for (uint32_t i=0;i<experts && used<capacity;++i) {
        const int e=order[i];
        if (!chosen[e] && !primary[e]) {chosen[e]=1;++used;}
    }
    for (uint32_t i=0;i<experts && replaced<max_replacements;++i) {
        const int candidate=order[i];
        if (chosen[candidate] || primary[candidate] || counts[candidate]<minimum_routes) continue;
        int victim=-1;
        for (uint32_t j=experts;j>0;--j) {
            const int e=order[j-1];
            if (chosen[e] && previous[e]) {victim=e;break;}
        }
        if (victim<0 || score[candidate]/(1.+minimum_gain_percent/100.)<=score[victim]) continue;
        chosen[victim]=0;chosen[candidate]=1;++replaced;
    }
    memcpy(selected,chosen,experts);
    return 1;
}
int strata_mixed_prompt_valid(const struct strata_mixed_buffer *b, uint32_t experts) {
    if (!b || b->prefill!=2 || b->routes || b->chunk_tag>1 || !experts || experts>288) return 0;
    for (uint32_t e=0;e<experts;++e)
        if (!isfinite(b->route_weight[e]) || b->route_weight[e]<0) return 0;
    return 1;
}
int strata_mixed_valid_sized(const void *buffer, size_t bytes, uint32_t layers, uint32_t experts) {
    if (!buffer || bytes<sizeof(struct strata_mixed_buffer)) return 0;
    const struct strata_mixed_buffer *b=buffer;
    const struct strata_mixed_layout *layout=strata_mixed_protocol_layout(b->version);
    if (!layout || bytes!=layout->bytes) return 0;
    if (layout->activation_half) {
        const struct strata_mixed_buffer_v5 *v5=buffer;
        if (v5->activation_pad || v5->activation_half>1 ||
            (v5->activation_half && (b->canon || (b->prefill!=1 && b->prefill!=STRATA_MIXED_PREFILL_PARTIAL)))) return 0;
    }
    if (b->magic != STRATA_MIXED_MAGIC ||
        !b->sequence || !b->generation || b->layer >= layers || !b->nt || b->nt > STRATA_MIXED_TOKENS ||
        ((!b->prefill || b->prefill==STRATA_MIXED_DECODE_PARTIAL) && b->nt > 8) || b->prefill > STRATA_MIXED_DECODE_PARTIAL || (!b->routes && b->prefill != 2) || b->routes > b->nt * 8 || b->canon > 1 || experts > 288 ||
        !isfinite(b->swiglu_limit) || !isfinite(b->gate_skip)) return 0;
    uint8_t seen[STRATA_MIXED_ROUTES] = {0};
    for (uint32_t i = 0; i < b->routes; ++i) {
        int d = b->destination[i], e = b->expert[i];
        if (e < 0 || (uint32_t)e >= experts || d < 0 || (uint32_t)d >= b->nt * 8 ||
            ((!b->prefill || b->prefill==STRATA_MIXED_DECODE_PARTIAL) && b->primary_owned[e]) || seen[d]) return 0;
        if ((b->prefill==STRATA_MIXED_PREFILL_PARTIAL || b->prefill==STRATA_MIXED_DECODE_PARTIAL) && (!isfinite(b->route_weight[d]) || b->route_weight[d]<0)) return 0;
        seen[d] = 1;
    }
    if (b->prefill==STRATA_MIXED_DECODE_PARTIAL)
        for (uint32_t j=0;j<b->nt*8;++j)
            if (!isfinite(b->route_weight[j]) || b->route_weight[j]<0 || (!seen[j] && b->route_weight[j]!=0)) return 0;
    return 1;
}
int strata_mixed_valid(const struct strata_mixed_buffer *b, uint32_t layers, uint32_t experts) {
    return strata_mixed_valid_sized(b,sizeof *b,layers,experts);
}

int strata_mixed_chunk_valid_sized(const void *buffer, size_t bytes, const uint8_t *requested, uint8_t *seen) {
    uint8_t local[STRATA_MIXED_ROUTES] = {0};
    if (!buffer || bytes<sizeof(struct strata_mixed_buffer)) return 0;
    const struct strata_mixed_buffer *b=buffer;
    const struct strata_mixed_layout *layout=strata_mixed_protocol_layout(b->version);
    if (!layout || bytes!=layout->bytes || !requested || !seen || !b->nt || b->nt>STRATA_MIXED_TOKENS ||
        b->chunk_rows>layout->output_rows || (!b->chunk_rows && !(b->chunk_tag&STRATA_MIXED_FINAL))) return 0;
    const int32_t *destinations=(const int32_t *)((const char *)buffer+layout->chunk_destination);
    for (uint32_t i=0;i<b->chunk_rows;++i) {
        const int d=destinations[i];
        if (d<0 || (uint32_t)d>=(b->prefill==STRATA_MIXED_PREFILL_PARTIAL ? b->nt : b->nt*8) || !requested[d] || seen[d] || local[d]) return 0;
        local[d]=1;
    }
    for (uint32_t i=0;i<b->chunk_rows;++i) seen[destinations[i]]=1;
    return 1;
}
int strata_mixed_chunk_valid(const struct strata_mixed_buffer *b, const uint8_t *requested, uint8_t *seen) {
    return strata_mixed_chunk_valid_sized(b,sizeof *b,requested,seen);
}
