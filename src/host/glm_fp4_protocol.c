#include "strata/host/glm_fp4_protocol.h"
int strata_glm_fp4_command_valid(const uint32_t h[7], uint32_t sequence,
                                 unsigned role, int direct) {
    if (!h || !sequence || role < 1 || role > 2 || (direct != 0 && direct != 1)) return 0;
    if (h[0] != STRATA_GLM_FP4_WIRE_MAGIC || h[1] != STRATA_GLM_FP4_WIRE_VERSION || h[6] != sequence) return 0;
    if (h[2] > 1 || h[3] >= 2 || h[4] > 16 || h[5] > 16) return 0;
    if (!h[2] && (h[3] || h[4] || h[5])) return 0;
    if (role == 2 && h[4]) return 0;
    if (role == 1 && direct && h[5]) return 0;
    return 1;
}

int strata_glm_fp4_batch_command_valid(const uint32_t h[9], uint32_t sequence) {
    unsigned i;
    if (!h || !sequence || h[0] != STRATA_GLM_FP4_BATCH_MAGIC ||
        h[1] != STRATA_GLM_FP4_BATCH_VERSION || h[5] != sequence) return 0;
    if (h[2] > 1 || h[3] >= 2 || h[4] > STRATA_GLM_FP4_BATCH_MAX_EXPERTS) return 0;
    if (!h[2] && (h[3] || h[4])) return 0;
    for (i = 0; i < STRATA_GLM_FP4_BATCH_MAX_EXPERTS; ++i) {
        if (i < h[4]) {
            if (h[6+i] >= STRATA_GLM_FP4_BATCH_MAX_EXPERTS) return 0;
        } else if (h[6+i]) return 0;
    }
    return 1;
}
