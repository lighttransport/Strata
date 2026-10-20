#ifndef STRATA_GLM_FP4_PROTOCOL_H
#define STRATA_GLM_FP4_PROTOCOL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_GLM_FP4_WIRE_MAGIC UINT32_C(0x33504653)
#define STRATA_GLM_FP4_WIRE_VERSION UINT32_C(1)
/* Hello word 6 is a feature mask. Zero/EXTENDED preserve the original
 * W4A16 handshake. All peers must match the complete seven-word hello. */
#define STRATA_GLM_FP4_HELLO_EXTENDED UINT32_C(1)
#define STRATA_GLM_FP4_HELLO_NATIVE_Q8 UINT32_C(2)
#define STRATA_GLM_FP4_HELLO_PRESERVE_SMALL UINT32_C(4)
/* Fixed seven-word command: magic, version, operation, layer, M1, M2, sequence.
 * Roles 1/2 are workers. Validation precedes any activation allocation/read. */
int strata_glm_fp4_command_valid(const uint32_t h[7], uint32_t sequence,
                                 unsigned role, int direct);
/* Separate native-Q8 grouped-expert protocol: one common input token,
 * preserved-small hidden scaling, two layer slots and three resident experts.
 * Nine words: magic, version, operation, layer, expert count, sequence, IDs[3].
 * Operation 0 stops; 1 executes (count zero is an idle request). Only count IDs
 * are meaningful; unused IDs must be zero. Repeated IDs are allowed and outputs
 * retain request order. Validate before allocating or reading any payload.
 * Non-idle work has one WIDTH-float input and count WIDTH-float outputs;
 * stop/idle has no activation payload. Existing tree commands are unaffected. */
#define STRATA_GLM_FP4_BATCH_MAGIC UINT32_C(0x42504653)
#define STRATA_GLM_FP4_BATCH_VERSION UINT32_C(1)
#define STRATA_GLM_FP4_BATCH_WIDTH UINT32_C(4096)
#define STRATA_GLM_FP4_BATCH_MAX_EXPERTS UINT32_C(3)
int strata_glm_fp4_batch_command_valid(const uint32_t h[9], uint32_t sequence);
#ifdef __cplusplus
}
#endif
#endif
