#ifndef STRATA_GLM_FP4_WORKER_H
#define STRATA_GLM_FP4_WORKER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_GLM_FP4_WORKER_MAGIC UINT32_C(0x57465034)
#define STRATA_GLM_FP4_WORKER_VERSION UINT32_C(1)
/* Eight u32 words: magic, version, op, sequence, layer, expert, tokens, tensor.
 * op 0: stop; 1: exact expert; 2: read one complete owned tensor.
 * Stop and tensor reads have no input payload. Expert inputs/outputs are F32.
 * All unused fields are zero. Contexts and sequence spaces are independent. */
int strata_glm_fp4_worker_command(const uint32_t h[8], uint32_t sequence,
                                 unsigned remote_count, unsigned tensor_count, int fetch_only);
#ifdef __cplusplus
}
#endif
#endif
