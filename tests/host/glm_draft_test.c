#include "strata/host/glm_draft.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    float gaps[7] = {4, 3, 2, 1, .5f, -1, 8}, original[7];
    int count, minimum, weak;
    memcpy(original, gaps, sizeof(gaps));
    assert(strata_glm_draft_depth(gaps, 7, 1, 1) == 4);
    assert(strata_glm_draft_depth(gaps, 7, 1, 0) == 5);
    assert(strata_glm_draft_depth(gaps, 7, 7, 32) == 7);
    assert(strata_glm_draft_depth(gaps, 2, 7, 32) == 2);
    assert(memcmp(gaps, original, sizeof(gaps)) == 0);
    /* Exhaust every possible first uncertain row and request-tail length.
     * A weak guaranteed row cannot discard the guaranteed prefix. */
    for (count = 1; count <= 7; ++count) {
        for (minimum = 1; minimum <= 7; ++minimum) {
            for (weak = 0; weak <= count; ++weak) {
                int i, want = count;
                for (i = 0; i < count; ++i) gaps[i] = i >= weak ? .5f : 2;
                if (minimum < count && weak < count) want = weak > minimum ? weak : minimum;
                assert(strata_glm_draft_depth(gaps, count, minimum, 1) == want);
            }
        }
    }
    assert(strata_glm_draft_depth(NULL, 7, 1, 1) == -1);
    assert(strata_glm_draft_depth(gaps, 0, 1, 1) == -1);
    assert(strata_glm_draft_depth(gaps, 8, 1, 1) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 0, 1) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 8, 1) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 1, -1) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 1, 33) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 1, NAN) == -1);
    assert(strata_glm_draft_depth(gaps, 7, 1, INFINITY) == -1);
    gaps[6] = NAN;
    assert(strata_glm_draft_depth(gaps, 7, 1, 1) == -1);
    gaps[6] = INFINITY;
    assert(strata_glm_draft_depth(gaps, 7, 7, 1) == -1);
    puts("glm_draft_test: passed");
    return 0;
}
