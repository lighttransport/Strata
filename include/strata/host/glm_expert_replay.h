/* Bounded diagnostic replay records, little-endian uint32/float32. No weight payloads.
 * Header, followed by ids[nt*K], weights[nt*K], cold[nt*K] (uint8), activation[nt*H].
 * Preserve record order: positions repeat after speculative rejection. */
#ifndef STRATA_HOST_GLM_EXPERT_REPLAY_H
#define STRATA_HOST_GLM_EXPERT_REPLAY_H
#include <stdint.h>
enum { STRATA_REPLAY_MAGIC = 0x31525047, STRATA_REPLAY_WORDS = 12 };
enum { STRATA_REPLAY_CANON = 1, STRATA_REPLAY_NUMA = 2, STRATA_REPLAY_FUSE_QUANT = 4 };
static inline int strata_replay_header_valid(const uint32_t *h) {
    return h && h[0] == STRATA_REPLAY_MAGIC && h[1] == 1 &&
           h[2] >= 3 && h[2] <= 44 && h[3] <= 1048576 && h[4] < 92 &&
           h[5] >= 1 && h[5] <= 8 && h[6] == 4096 && h[7] == 8 &&
           h[8] >= 16 && h[8] <= 288 && h[8] % 16 == 0 && (h[9] & ~7u) == 0 &&
           (!(h[9] & STRATA_REPLAY_FUSE_QUANT) || (h[9] & STRATA_REPLAY_NUMA));
}
#endif
