/* Versioned whole-expert prefill messages. Legacy STD1/STD2 are unchanged. */
#ifndef STRATA_DS41_PREFILL_WIRE_H
#define STRATA_DS41_PREFILL_WIRE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_DS41_PREFILL_MAGIC UINT32_C(0x33445354)
#define STRATA_DS41_PREFILL_VERSION UINT32_C(1)
#define STRATA_DS41_PREFILL_MAX_TOKENS UINT32_C(8192)
#define STRATA_DS41_PREFILL_MAX_ROUTES UINT32_C(131072)
#define STRATA_DS41_PREFILL_CAP_Q8 UINT32_C(1)
#define STRATA_DS41_PREFILL_CAP_WEIGHTED_F32 UINT32_C(2)
#define STRATA_DS41_PREFILL_CAP_PHASES UINT32_C(4)
#define STRATA_DS41_PREFILL_REQUIRED_CAPS UINT32_C(7)
typedef struct strata_ds41_prefill_capabilities {
    uint32_t magic, version, flags, reserved;
    uint32_t max_tokens, max_routes, max_hidden, max_top_k;
} strata_ds41_prefill_capabilities;
/* Required limits are checked before accepting or transferring weights. */
int strata_ds41_prefill_negotiate(
    const strata_ds41_prefill_capabilities *peer,
    uint32_t tokens, uint32_t hidden, uint32_t top_k);
enum strata_ds41_prefill_op {
    STRATA_DS41_PREFILL_BEGIN = 1,
    STRATA_DS41_PREFILL_EXECUTE = 2,
    STRATA_DS41_PREFILL_END = 3,
    STRATA_DS41_PREFILL_DECODE_RAW = 4
};
/* Q8 payloads are row-major logical 128-value blocks: 16 metadata bytes,
 * followed by 128 signed quant bytes. Metadata retains its original bits:
 * DS4 = four fp16 (scale,sum) pairs; D4 = four fp32 scales;
 * D2S6 = two fp16 scales then six fp16 sums. Never convert these layouts. */
enum strata_ds41_prefill_activation {
    STRATA_DS41_PREFILL_NO_ACTIVATION = 0,
    STRATA_DS41_PREFILL_Q8_1 = 1,
    STRATA_DS41_PREFILL_F32 = 2,
    STRATA_DS41_PREFILL_Q8_D4 = 3,
    STRATA_DS41_PREFILL_Q8_D2S6 = 4
};
typedef struct strata_ds41_prefill_request {
    uint32_t version, op;
    uint64_t sequence;
    uint32_t layer, first_token, tokens, routes;
    uint32_t hidden, top_k, activation, reserved;
    uint64_t activation_bytes;
} strata_ds41_prefill_request;
/* Returns zero for invalid geometry; does not inspect or allocate payloads. */
int strata_ds41_prefill_validate_request(
    const strata_ds41_prefill_request *r, uint32_t layers, uint32_t hidden,
    uint32_t context, uint64_t expected_sequence);

/* Commit a transition only after its GPU work and transfers have completed.
 * Rejected requests leave state unchanged. Sequence numbers never wrap. */
typedef struct strata_ds41_prefill_state {
    uint64_t sequence;
    uint32_t active, reserved;
} strata_ds41_prefill_state;
int strata_ds41_prefill_advance(
    strata_ds41_prefill_state *state, const strata_ds41_prefill_request *r,
    uint32_t layers, uint32_t hidden, uint32_t context);

/* CPU reference for the GPU transport permutation. Native blocks are indexed
 * [column_block][row], wire blocks [row][column_block]. Both contain 144 raw
 * bytes per block; buffers must be disjoint. The native row padding and extra
 * MMQ tile tail are excluded from the wire. Unpack clears both deterministically.
 * Returns zero without writing on invalid geometry or insufficient capacity. */
int strata_ds41_prefill_q8_repack(
    void *dst, size_t dst_bytes, const void *src, size_t src_bytes,
    uint32_t rows, uint32_t hidden, uint32_t padded_hidden, int to_wire);

#ifdef __cplusplus
}
static_assert(sizeof(strata_ds41_prefill_request) == 56, "prefill wire layout");
#else
_Static_assert(sizeof(strata_ds41_prefill_request) == 56, "prefill wire layout");
#endif
#endif
