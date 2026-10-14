#ifndef STRATA_GLM_MIXED_H
#define STRATA_GLM_MIXED_H
#include <stdint.h>
#define STRATA_MIXED_MAGIC UINT32_C(0x474d5831)
#define STRATA_MIXED_VERSION 4
#define STRATA_MIXED_TOKENS 8192
#define STRATA_MIXED_ROUTES 65536
#define STRATA_MIXED_OUTPUT_ROWS 512
#define STRATA_MIXED_FINAL UINT32_C(0x80000000)
#define STRATA_MIXED_HIDDEN 4096
struct strata_mixed_message { uint64_t sequence; uint32_t tag, kind; };
#define STRATA_MIXED_CHUNK 1
#define STRATA_MIXED_ACK 2
#define STRATA_MIXED_PREFILL_PARTIAL 3
#define STRATA_MIXED_DECODE_PARTIAL 4
/* One in-flight, stateless expert request. Sequence is also the buffer lease. */
struct strata_mixed_buffer {
    uint32_t magic, version, layer, nt, routes, canon, prefill, reserved;
    uint64_t sequence, generation;
    float swiglu_limit, gate_skip;
    uint32_t chunk_rows, chunk_tag;
    int32_t expert[STRATA_MIXED_ROUTES], destination[STRATA_MIXED_ROUTES];
    float route_weight[STRATA_MIXED_ROUTES];
    uint8_t primary_owned[288];
    uint8_t worker_owned[64 * 288];
    int32_t chunk_destination[STRATA_MIXED_OUTPUT_ROWS];
    float activation[STRATA_MIXED_TOKENS * STRATA_MIXED_HIDDEN];
    float output[STRATA_MIXED_OUTPUT_ROWS * STRATA_MIXED_HIDDEN];
};
#ifdef __cplusplus
extern "C" {
#endif
int strata_mixed_valid(const struct strata_mixed_buffer *b, uint32_t layers, uint32_t experts);
/* Optional ownership request: route_weight holds counts; chunk_tag 1 refreshes. */
int strata_mixed_prompt_valid(const struct strata_mixed_buffer *b, uint32_t experts);
int strata_mixed_chunk_valid(const struct strata_mixed_buffer *b, const uint8_t *requested, uint8_t *seen);
/* Opt-in cache hysteresis. Fill primary-displaced holes, then replace at most
   max_replacements old owners only when both count and relative-gain tests pass.
   order is a complete best-first permutation; score includes the caller's prior.
   Returns 0 on invalid input, leaving selected unchanged. */
int strata_mixed_select_cache(uint32_t experts, uint32_t capacity,
    const int *order, const double *score, const float *counts,
    const uint8_t *previous, const uint8_t *primary, uint32_t max_replacements,
    float minimum_routes, double minimum_gain_percent, uint8_t *selected);
#ifdef __cplusplus
}
#endif
#endif
