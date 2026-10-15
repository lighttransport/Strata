#include "strata/host/glm_expert_replay.h"
#include <assert.h>
#include <string.h>
int main(void) {
    uint32_t h[STRATA_REPLAY_WORDS] = {STRATA_REPLAY_MAGIC, 1, 13, 8192, 59, 2, 4096, 8, 224, 3, 0, 0};
    const unsigned fields[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const uint32_t bad[] = {0, 2, 45, 1048577, 92, 9, 0, 64, 225, 8};
    if (!strata_replay_header_valid(h) || strata_replay_header_valid(0)) return 1;
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        uint32_t copy[STRATA_REPLAY_WORDS];
        memcpy(copy, h, sizeof(h)); copy[fields[i]] = bad[i];
        if (strata_replay_header_valid(copy)) return 2;
    }
    h[5] = 0;
    return strata_replay_header_valid(h) ? 3 : 0;
}
