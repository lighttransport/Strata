/* Experimental prompt-frequency admission. Equal-sized expert slots; the
 * caller separately reserves draft weights and execution scratch. */
#ifndef STRATA_HOST_GLM_FP4_CACHE_H_
#define STRATA_HOST_GLM_FP4_CACHE_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct strata_glm_fp4_cache_entry {
    int layer, expert;
    uint64_t frequency;
    int owner; /* output: -1 for CPU, otherwise device rank */
} strata_glm_fp4_cache_entry;
/* Sort entries by frequency descending, then layer/expert ascending. Mode 1
 * preserves expert % devices ownership. Mode 2 admits the hottest combined
 * working set, retaining parity where capacity permits. Returns 0 on invalid
 * arguments; validates before changing entries. One or two devices supported. */
int strata_glm_fp4_cache_select(strata_glm_fp4_cache_entry *entries, size_t count,
                               const size_t slots[2], unsigned devices, int mode);
/* Replace at most limit entries in one device's fixed-size layer set. Only a
 * strictly hotter expert whose score exceeds the cold score by hysteresis can
 * enter. Arrays have 288 elements; removed/added return the changed prefix.
 * Returns -1 on invalid input, otherwise the number of replacements. */
int strata_glm_fp4_cache_adapt(const uint64_t frequency[288], uint8_t keep[288],
                              unsigned rank, unsigned devices, unsigned limit,
                              uint64_t hysteresis, int removed[288], int added[288]);
#ifdef __cplusplus
}
#endif
#endif
