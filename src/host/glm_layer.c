#include "strata/host/glm_layer.h"
#include "strata/host/glm_names.h"
#include <limits.h>
#include <string.h>

int strata_glm_layer_bounds(const char *split, int layers, int stages, int out[STRATA_GLM_MAX_STAGES + 1]) {
    int tmp[STRATA_GLM_MAX_STAGES + 1], i;
    const char *p = split;
    if (!split || !out || layers < 1 || stages < 1 || stages > STRATA_GLM_MAX_STAGES || stages > layers) return -1;
    tmp[0] = 0;
    tmp[stages] = layers;
    if (!strcmp(split, "auto")) {
        for (i = 1; i < stages; ++i) tmp[i] = (int)((long long)layers * i / stages);
    } else {
        for (i = 1; i < stages; ++i) {
            int n = 0;
            if (*p < '0' || *p > '9') return -1;
            do {
                if (n > (INT_MAX - (*p - '0')) / 10) return -1;
                n = n * 10 + (*p++ - '0');
            } while (*p >= '0' && *p <= '9');
            if (n <= tmp[i - 1] || n >= layers) return -1;
            tmp[i] = n;
            if (i < stages - 1 && *p++ != ',') return -1;
        }
        if (*p) return -1;
    }
    memcpy(out, tmp, (size_t)(stages + 1) * sizeof(*out));
    return 0;
}
int strata_glm_layer_tensor_owned(const char *name, int first, int last, int layers, int include_draft) {
    int layer = 0, parsed;
    const char *suffix = NULL;
    if (!name || first < 0 || first >= last || last > layers) return 0;
    parsed = strata_parse_blk(name, &layer, &suffix);
    if (parsed < 0) return 0;
    if (parsed) return (layer >= first && layer < last) ||
        (include_draft && last == layers && layer == layers);
    if (!strcmp(name, "token_embd.weight")) return first == 0;
    return last == layers;
}
