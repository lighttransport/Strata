#include "strata/host/glm_draft.h"
#include <math.h>

int strata_glm_draft_depth(const float *gaps, int count, int minimum, float threshold) {
    int i;
    if (!gaps || count < 1 || count > 7 || minimum < 1 || minimum > 7 ||
        !isfinite(threshold) || threshold < 0 || threshold > 32) return -1;
    for (i = 0; i < count; ++i) if (!isfinite(gaps[i])) return -1;
    for (i = minimum; i < count; ++i) if (gaps[i] < threshold) return i;
    return count;
}
