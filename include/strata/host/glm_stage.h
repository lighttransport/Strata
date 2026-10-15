#ifndef STRATA_GLM_STAGE_H
#define STRATA_GLM_STAGE_H
/* Layer-split stage protocol (STRATA_GLM_LAYER_RANGE): the primary decoder runs layers [0, K) plus the output
   head and MTP; a stage decoder in another process (other GPU vendor) runs layers [K, end). The residual streams
   (4 x hidden floats per token) cross through one shared mapping; a SOCK_SEQPACKET channel carries sequences.
   One request is outstanding at a time. The stage mirrors every state-changing decoder call of the primary. */
#include <stdint.h>
#include <stddef.h>
#define STRATA_STAGE_MAGIC UINT32_C(0x474d5354)
#define STRATA_STAGE_VERSION 1
#define STRATA_STAGE_TOKENS 8192
#define STRATA_STAGE_HIDDEN 4096
#define STRATA_STAGE_ROW_FLOATS (4 * STRATA_STAGE_HIDDEN)
enum strata_stage_command {
    STRATA_STAGE_BATCH = 1,              /* tokens[nt], prefill flag; residual in, residual out in place */
    STRATA_STAGE_VERIFY = 2,             /* as BATCH, capturing verify history */
    STRATA_STAGE_RESTORE_VERIFIED = 3,   /* count */
    STRATA_STAGE_ENABLE_HISTORY = 4,     /* count = slots; result = slots granted */
    STRATA_STAGE_INVALIDATE_HISTORY = 5,
    STRATA_STAGE_CHECKPOINT = 6,         /* result = id */
    STRATA_STAGE_SAVE = 7,               /* id */
    STRATA_STAGE_RESTORE = 8,            /* id */
    STRATA_STAGE_RESET = 9,
    STRATA_STAGE_PREPARE_DECODE_CACHE = 10,
    STRATA_STAGE_FREEZE_CACHE = 11,
    STRATA_STAGE_STOP = 12,
    STRATA_STAGE_COMMAND_LAST = 12
};
struct strata_stage_buffer {
    uint32_t magic, version, command, status;   /* status: 0 ok, 1 error (message in error) */
    uint32_t nt, prefill, reserved0, reserved1;
    int32_t position, count, id, result;
    uint64_t sequence;
    char error[256];
    int32_t tokens[STRATA_STAGE_TOKENS];
    float residual[(size_t)STRATA_STAGE_TOKENS * STRATA_STAGE_ROW_FLOATS];
};
#ifdef __cplusplus
extern "C" {
#endif
/* A request the stage may execute: known command, matching magic/version/sequence, bounded token count,
   tokens in [0, vocab), and decode batches at most max_decode_tokens wide. */
int strata_stage_valid(const struct strata_stage_buffer *b, uint64_t sequence, uint32_t vocab, uint32_t max_decode_tokens);
/* Parses "first:end" with 0 <= first < end <= layers. Returns 1 and fills both on success. */
int strata_stage_range(const char *text, uint32_t layers, uint32_t *first, uint32_t *end);
#ifdef __cplusplus
}
#endif
#endif
