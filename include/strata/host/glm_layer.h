#ifndef STRATA_HOST_GLM_LAYER_H
#define STRATA_HOST_GLM_LAYER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_GLM_MAX_STAGES 16
/* Cuts are the first global layer on each later device. Empty stages, trailing
   commas, signs, whitespace and duplicate cuts are errors. auto is a balanced
   initial placement, not a claim of a measured optimum. */
int strata_glm_layer_bounds(const char *split, int layers, int stages,
                            int bounds[STRATA_GLM_MAX_STAGES + 1]);
/* Embedding belongs to the first stage, head and draft to the tail. */
int strata_glm_layer_tensor_owned(const char *name, int first, int last,
                                 int layers, int include_draft);
#ifdef __cplusplus
}
#endif
#endif
