/* Buffered, ordered, one-direction message relay between two independent contexts.
 * Drive both contexts from the same thread. For duplex operation create one relay
 * per direction and alternate progress calls. This is an application relay, not
 * transparent verbs/QP routing. Existing ucomm defaults and wire format are unchanged. */
#ifndef UCOMM_RELAY_H_
#define UCOMM_RELAY_H_
#include "ucomm.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ucomm_relay ucomm_relay_t;

/* Forward count messages, preserving lengths and order, from src_peer/src_tag to
 * dst_peer/dst_tag. cap is the maximum individual message size; depth is 1..64.
 * Allocates min(depth, count) slots in one pool registered once in each context.
 * Zero count allocates no payload memory. No requests are posted
 * until progress is called. Contexts must outlive the relay. */
ucomm_status ucomm_relay_create(ucomm_t *src, int src_peer, uint32_t src_tag,
                               ucomm_t *dst, int dst_peer, uint32_t dst_tag,
                               size_t cap, int depth, uint64_t count, ucomm_relay_t **out);
/* Nonblocking; progresses BOTH contexts even when one direction is stalled.
 * done becomes 1 after every outgoing send completes. Errors leave the relay
 * allocated; on transport/truncation errors terminate the communicating session.
 * No implicit timeout: caller owns the overall deadline. */
ucomm_status ucomm_relay_progress(ucomm_relay_t *r, int *done);
/* Opt-in duplex path: forward/reverse must use the same two contexts in opposite
 * directions. Polls each context once, then advances both rings without per-slot
 * CQ polling. fd/rd must be distinct outputs. Context ownership is unchanged. */
ucomm_status ucomm_relay_progress_pair(ucomm_relay_t *forward, ucomm_relay_t *reverse,
                                      int *forward_done, int *reverse_done);
/* Payload bytes whose outgoing sends have completed (not merely been queued). */
uint64_t ucomm_relay_bytes(const ucomm_relay_t *r);
/* Valid before progress, after completion, or after an error with no outstanding
 * requests. Otherwise returns ERR_ARG
 * without freeing buffers that may still be referenced by verbs. */
ucomm_status ucomm_relay_destroy(ucomm_relay_t *r);
#ifdef __cplusplus
}
#endif
#endif
