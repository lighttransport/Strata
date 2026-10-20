/* Optional confidence bounds for a seven-token parallel draft. */
#ifndef STRATA_HOST_GLM_DRAFT_H
#define STRATA_HOST_GLM_DRAFT_H

#ifdef __cplusplus
extern "C" {
#endif

/* gaps[i] is the selected token's unary logit minus its strongest other
 * candidate. Retain at least minimum tokens (or count at the request tail),
 * then stop before the first gap below threshold. Equal gaps are retained.
 * count/minimum are 1..7 and threshold is finite 0..32. Returns -1 for
 * invalid bounds, a null input or any nonfinite gap. No input is modified. */
int strata_glm_draft_depth(const float *gaps, int count, int minimum, float threshold);

#ifdef __cplusplus
}
#endif
#endif
