#include "strata/host/glm_fp4_worker.h"
#include "strata/host/glm_fp4_layout.h"
int strata_glm_fp4_worker_command(const uint32_t h[8], uint32_t sequence,
                                 unsigned remote_count, unsigned tensor_count, int fetch_only) {
    if (!h || h[0] != STRATA_GLM_FP4_WORKER_MAGIC || h[1] != STRATA_GLM_FP4_WORKER_VERSION ||
        !sequence || h[3] != sequence || sequence == UINT32_MAX || remote_count > 288) return 0;
    if (!h[2]) return !(h[4] | h[5] | h[6] | h[7]);
    if (h[2] == 2) return !(h[4] | h[5] | h[6]) && h[7] < tensor_count;
    return !fetch_only && h[2] == 1 && h[6] >= 1 && h[6] <= 8 && !h[7] &&
           strata_glm_fp4_layout_owner(h[4], h[5], remote_count) == 1;
}
