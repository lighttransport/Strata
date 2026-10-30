#ifndef STRATA_GLM_FP4_LAYOUT_H
#define STRATA_GLM_FP4_LAYOUT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Native text checkpoint: routed layers 3..44, 288 whole experts per layer.
 * Layer 45 is the draft and always belongs to the decoder. */
int strata_glm_fp4_layout_role(unsigned layer, unsigned expert, unsigned first_moe,
                               unsigned main_layers, unsigned expert_count, unsigned remote_count);
int strata_glm_fp4_layout_owner(unsigned layer, unsigned expert, unsigned remote_count);
/* Bounds and arithmetic shared by layout writers, readers and workers. */
int strata_glm_fp4_layout_span(uint64_t offset, uint64_t bytes, uint64_t file_bytes);
int strata_glm_fp4_layout_admit(uint64_t weights, uint64_t reserve, uint64_t allowance);
#ifdef __cplusplus
}
#endif
#endif
