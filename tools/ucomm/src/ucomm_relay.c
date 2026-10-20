#define _POSIX_C_SOURCE 200112L
#include "ucomm_relay.h"
#include "ucomm_internal.h"
#include <stdlib.h>

enum { EMPTY, RECEIVING, READY, SENDING };
typedef struct {
  void *buf;
  ucomm_req_t *req;
  size_t len;
  uint64_t seq;
  int state;
} relay_slot;

struct ucomm_relay {
  ucomm_t *src, *dst;
  int src_peer, dst_peer, depth, started;
  uint32_t src_tag, dst_tag;
  size_t cap;
  uint64_t count, posted, forwarded, completed, bytes;
  ucomm_status error;
  void *pool;
  ucomm_mr_t *src_mr, *dst_mr;
  relay_slot slot[64];
};

ucomm_status ucomm_relay_destroy(ucomm_relay_t *r) {
  if (!r) return UCOMM_OK;
  for (int i = 0; i < r->depth; i++) if (r->slot[i].req) return UCOMM_ERR_ARG;
  if (r->started && !r->error && r->completed != r->count) return UCOMM_ERR_ARG;
  if (r->dst_mr) ucomm_mr_dereg(r->dst_mr);
  if (r->src_mr) ucomm_mr_dereg(r->src_mr);
  free(r->pool);
  free(r);
  return UCOMM_OK;
}

ucomm_status ucomm_relay_create(ucomm_t *src, int sp, uint32_t st,
                               ucomm_t *dst, int dp, uint32_t dt,
                               size_t cap, int depth, uint64_t count, ucomm_relay_t **out) {
  if (!out) return UCOMM_ERR_ARG;
  *out = NULL;
  if (!src || !dst || src == dst || sp < 0 || sp >= src->world || sp == src->rank ||
      dp < 0 || dp >= dst->world || dp == dst->rank || st >= UCOMM_TAG_RESERVED ||
      dt >= UCOMM_TAG_RESERVED || !cap || depth < 1 || depth > 64 || cap > SIZE_MAX / (size_t)depth)
    return UCOMM_ERR_ARG;
  ucomm_relay_t *r = calloc(1, sizeof *r);
  if (!r) return UCOMM_ERR_NOMEM;
  r->src = src; r->dst = dst; r->src_peer = sp; r->dst_peer = dp;
  r->src_tag = st; r->dst_tag = dt; r->cap = cap; r->depth = depth; r->count = count;
  if (count < (uint64_t)depth) r->depth = (int)count;
  if (!count) { *out = r; return UCOMM_OK; }
  ucomm_status s = UCOMM_OK;
  size_t bytes = cap * (size_t)r->depth;
  if (posix_memalign(&r->pool, 4096, bytes)) { free(r); return UCOMM_ERR_NOMEM; }
  s = ucomm_mr_reg(src, r->pool, bytes, &r->src_mr);
  if (s == UCOMM_OK) s = ucomm_mr_reg(dst, r->pool, bytes, &r->dst_mr);
  if (s != UCOMM_OK) { ucomm_relay_destroy(r); return s; }
  for (int i = 0; i < r->depth; i++) r->slot[i].buf = (char *)r->pool + cap * (size_t)i;
  *out = r;
  return UCOMM_OK;
}

static ucomm_status relay_progress(ucomm_relay_t *r, int *done, int already_polled) {
  if (!r || !done) return UCOMM_ERR_ARG;
  *done = 0;
  if (r->error) return r->error;
  if (r->completed == r->count) { *done = 1; return UCOMM_OK; }
  r->started = 1;
  ucomm_status s = UCOMM_OK;
  if (!already_polled) {
    s = r->src->ops->progress(r->src, 0);
    if (s != UCOMM_OK) goto fail;
    s = r->dst->ops->progress(r->dst, 0);
    if (s != UCOMM_OK) goto fail;
  }
  for (int i = 0; i < r->depth; i++) {
    relay_slot *b = &r->slot[i];
    if (b->state != RECEIVING && b->state != SENDING) continue;
    if (already_polled && !b->req->done) continue;
    int ready = 0;
    size_t len = 0;
    s = ucomm_test(b->req, &ready, &len);
    if (ready) b->req = NULL;
    if (s != UCOMM_OK) goto fail;
    if (!ready) continue;
    if (b->state == RECEIVING) { b->len = len; b->state = READY; }
    else { b->state = EMPTY; r->completed++; r->bytes += b->len; }
  }
  /* Forward in receive-post order, independent of completion order. */
  while (r->forwarded < r->posted) {
    relay_slot *b = &r->slot[r->forwarded % (uint64_t)r->depth];
    if (b->state != READY || b->seq != r->forwarded) break;
    s = ucomm_isend(r->dst, r->dst_peer, r->dst_tag, b->buf, b->len, &b->req);
    if (s != UCOMM_OK) goto fail;
    b->state = SENDING;
    r->forwarded++;
  }
  while (r->posted < r->count) {
    relay_slot *b = &r->slot[r->posted % (uint64_t)r->depth];
    if (b->state != EMPTY) break;
    b->seq = r->posted;
    s = ucomm_irecv(r->src, r->src_peer, r->src_tag, b->buf, r->cap, &b->req);
    if (s != UCOMM_OK) goto fail;
    b->state = RECEIVING;
    r->posted++;
  }
  *done = r->completed == r->count;
  return UCOMM_OK;
fail:
  r->error = s;
  return s;
}

uint64_t ucomm_relay_bytes(const ucomm_relay_t *r) { return r ? r->bytes : 0; }

ucomm_status ucomm_relay_progress(ucomm_relay_t *r, int *done) { return relay_progress(r, done, 0); }

ucomm_status ucomm_relay_progress_pair(ucomm_relay_t *f, ucomm_relay_t *r, int *fd, int *rd) {
  if (!f || !r || !fd || !rd || fd == rd || f->src != r->dst || f->dst != r->src)
    return UCOMM_ERR_ARG;
  *fd = *rd = 0;
  if (f->error) return f->error;
  if (r->error) return r->error;
  if (f->completed != f->count || r->completed != r->count) {
    ucomm_status s = f->src->ops->progress(f->src, 0);
    if (s == UCOMM_OK) s = f->dst->ops->progress(f->dst, 0);
    if (s != UCOMM_OK) { f->error = r->error = s; return s; }
  }
  ucomm_status s = relay_progress(f, fd, 1);
  if (s != UCOMM_OK) return s;
  return relay_progress(r, rd, 1);
}
