/* Core: config, init, tag matching, requests, collectives. Backend-agnostic. */
#define _GNU_SOURCE
#include "ucomm_internal.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

uint64_t uc_now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

void uc_log(int rank, const char *fmt, ...) {
  char line[512];
  int n = snprintf(line, sizeof line, "[ucomm %d] ", rank);
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(line + n, sizeof line - (size_t)n - 1, fmt, ap);
  va_end(ap);
  strcat(line, "\n");
  fputs(line, stderr); /* one write per line so ranks sharing a terminal do not interleave */
}

static int env_int(const char *k, int def) {
  const char *v = getenv(k);
  return v && *v ? atoi(v) : def;
}

void ucomm_config_default(ucomm_config *cfg) {
  memset(cfg, 0, sizeof *cfg);
  cfg->rank = env_int("UCOMM_RANK", 0);
  cfg->world = env_int("UCOMM_WORLD", 1);
  cfg->root_addr = getenv("UCOMM_ROOT") ? getenv("UCOMM_ROOT") : "127.0.0.1:29500";
  const char *be = getenv("UCOMM_BACKEND");
  cfg->backend = UCOMM_BACKEND_AUTO;
  if (be && !strcmp(be, "ib")) cfg->backend = UCOMM_BACKEND_IB;
  if (be && !strcmp(be, "udp")) cfg->backend = UCOMM_BACKEND_UDP;
  cfg->ib_dev = getenv("UCOMM_IB_DEV");
  cfg->ib_port = env_int("UCOMM_IB_PORT", 1);
  cfg->ib_gid_index = env_int("UCOMM_IB_GID_INDEX", 0);
  cfg->udp_mtu = env_int("UCOMM_UDP_MTU", 1400);
  cfg->timeout_ms = env_int("UCOMM_TIMEOUT_MS", 60000);
  cfg->verbose = env_int("UCOMM_VERBOSE", 0);
}

const char *ucomm_strerror(ucomm_status s) {
  switch (s) {
    case UCOMM_OK: return "ok";
    case UCOMM_ERR_NOBACKEND: return "backend not available";
    case UCOMM_ERR_IO: return "I/O error";
    case UCOMM_ERR_TIMEOUT: return "timeout";
    case UCOMM_ERR_ARG: return "invalid argument";
    case UCOMM_ERR_NOMEM: return "out of memory";
    case UCOMM_ERR_TRUNC: return "message truncated";
  }
  return "unknown";
}

int ucomm_rank(const ucomm_t *c) { return c->rank; }
int ucomm_world(const ucomm_t *c) { return c->world; }
const char *ucomm_backend_name(const ucomm_t *c) { return c->ops ? c->ops->name : "none"; }

ucomm_status ucomm_init(const ucomm_config *cfg, ucomm_t **out) {
  if (!cfg || !out || cfg->world < 1 || cfg->world > UCOMM_MAX_WORLD || cfg->rank < 0 || cfg->rank >= cfg->world)
    return UCOMM_ERR_ARG;
  ucomm_t *c = calloc(1, sizeof *c);
  if (!c) return UCOMM_ERR_NOMEM;
  c->cfg = *cfg;
  if (c->cfg.ib_port <= 0) c->cfg.ib_port = 1;
  if (c->cfg.udp_mtu <= 0) c->cfg.udp_mtu = 1400;
  c->rank = cfg->rank;
  c->world = cfg->world;

  ucomm_status s = uc_boot_init(c);
  if (s != UCOMM_OK) {
    free(c);
    return s;
  }

  /* Round 1: which backends does every rank have; rank 0 also hands out the session id. */
  struct {
    uint32_t have_ib, session;
  } mine, all[UCOMM_MAX_WORLD];
  mine.have_ib = cfg->backend != UCOMM_BACKEND_UDP ? (uint32_t)uc_ib_probe(c) : 0;
  mine.session = (uint32_t)(uc_now_us() ^ ((uint64_t)getpid() << 16)) | 1u;
  s = uc_boot_allgather(c, &mine, all, sizeof mine);
  if (s != UCOMM_OK) goto fail;
  c->session = all[0].session;
  int every_ib = 1;
  for (int r = 0; r < c->world; r++) every_ib &= all[r].have_ib != 0;

  if (cfg->backend == UCOMM_BACKEND_IB && !every_ib) {
    if (c->rank == 0) fprintf(stderr, "[ucomm 0] ib requested but not every rank has an active HCA port\n");
    s = UCOMM_ERR_NOBACKEND;
    goto fail;
  }
  if (every_ib) {
    c->ops = &uc_ib_ops;
    s = uc_ib_init(c);
  } else {
    if (cfg->backend == UCOMM_BACKEND_AUTO) UC_LOG(c, "no InfiniBand on every rank, using udp");
    c->ops = &uc_udp_ops;
    s = uc_udp_init(c);
  }
  if (s != UCOMM_OK) {
    c->ops = NULL;
    goto fail;
  }
  UC_LOG(c, "rank %d/%d up, backend %s", c->rank, c->world, c->ops->name);
  *out = c;
  return UCOMM_OK;
fail:
  uc_boot_close(c);
  free(c);
  return s;
}

void ucomm_finalize(ucomm_t *c) {
  if (!c) return;
  while (!c->ops->idle(c)) c->ops->progress(c, 1);
  uc_boot_drain_barrier(c);
  c->ops->finalize(c);
  uc_boot_close(c);
  for (int p = 0; p < c->world; p++) {
    uc_unexp *u = c->unexp[p];
    while (u) {
      uc_unexp *n = u->next;
      free(u->data);
      free(u);
      u = n;
    }
  }
  while (c->mrs) {
    ucomm_mr_t *m = c->mrs;
    c->mrs = m->next;
    free(m);
  }
  free(c);
}

ucomm_status ucomm_mr_reg(ucomm_t *c, void *ptr, size_t len, ucomm_mr_t **out) {
  if (!c || !ptr || !len || !out) return UCOMM_ERR_ARG;
  ucomm_mr_t *m = calloc(1, sizeof *m);
  if (!m) return UCOMM_ERR_NOMEM;
  m->c = c;
  m->ptr = ptr;
  m->len = len;
  ucomm_status s = c->ops->mr_reg ? c->ops->mr_reg(c, m) : UCOMM_OK;
  if (s != UCOMM_OK) {
    free(m);
    return s;
  }
  m->next = c->mrs;
  c->mrs = m;
  *out = m;
  return UCOMM_OK;
}

void ucomm_mr_dereg(ucomm_mr_t *m) {
  if (!m) return;
  ucomm_t *c = m->c;
  for (ucomm_mr_t **pp = &c->mrs; *pp; pp = &(*pp)->next)
    if (*pp == m) {
      *pp = m->next;
      break;
    }
  if (c->ops->mr_dereg) c->ops->mr_dereg(c, m);
  free(m);
}

/* ---- matching ---- */

ucomm_req_t *uc_match_posted(ucomm_t *c, int peer, uint32_t tag) {
  ucomm_req_t *prev = NULL;
  for (ucomm_req_t *r = c->posted[peer]; r; prev = r, r = r->next) {
    if (r->tag != tag) continue;
    if (prev)
      prev->next = r->next;
    else
      c->posted[peer] = r->next;
    if (c->posted_tail[peer] == r) c->posted_tail[peer] = prev;
    r->next = NULL;
    return r;
  }
  return NULL;
}

uc_unexp *uc_unexp_new(ucomm_t *c, int peer, uint32_t tag, size_t len) {
  uc_unexp *u = calloc(1, sizeof *u);
  if (!u) return NULL;
  u->peer = peer;
  u->tag = tag;
  u->len = len;
  if (c->unexp_tail[peer])
    c->unexp_tail[peer]->next = u;
  else
    c->unexp[peer] = u;
  c->unexp_tail[peer] = u;
  return u;
}

static void unexp_unlink(ucomm_t *c, uc_unexp *u) {
  uc_unexp *prev = NULL;
  for (uc_unexp *x = c->unexp[u->peer]; x; prev = x, x = x->next) {
    if (x != u) continue;
    if (prev)
      prev->next = x->next;
    else
      c->unexp[u->peer] = x->next;
    if (c->unexp_tail[u->peer] == x) c->unexp_tail[u->peer] = prev;
    return;
  }
}

static void copy_into(ucomm_req_t *r, const void *data, size_t len) {
  r->nbytes = len;
  size_t n = len < r->len ? len : r->len;
  if (n) memcpy(r->buf, data, n);
  uc_req_complete(r, len > r->len ? UCOMM_ERR_TRUNC : UCOMM_OK);
}

void uc_unexp_done(ucomm_t *c, uc_unexp *u) {
  u->complete = 1;
  if (!u->waiter) return;
  copy_into(u->waiter, u->data, u->len);
  unexp_unlink(c, u);
  free(u->data);
  free(u);
}

void uc_req_complete(ucomm_req_t *r, ucomm_status s) {
  r->status = s;
  r->done = 1;
}

/* ---- point to point ---- */

static ucomm_req_t *req_new(ucomm_t *c, int kind, int peer, uint32_t tag, void *buf, size_t len) {
  ucomm_req_t *r = calloc(1, sizeof *r);
  if (!r) return NULL;
  r->c = c;
  r->kind = kind;
  r->peer = peer;
  r->tag = tag;
  r->buf = buf;
  r->len = len;
  return r;
}

static ucomm_status isend_tag(ucomm_t *c, int peer, uint32_t tag, const void *buf, size_t len, ucomm_req_t **out) {
  if (!c || !out || peer < 0 || peer >= c->world || (len && !buf)) return UCOMM_ERR_ARG;
  ucomm_req_t *r = req_new(c, UC_REQ_SEND, peer, tag, (void *)buf, len);
  if (!r) return UCOMM_ERR_NOMEM;
  *out = r;
  if (peer == c->rank) { /* self: deliver locally */
    ucomm_req_t *rr = uc_match_posted(c, peer, tag);
    if (rr) {
      copy_into(rr, buf, len);
    } else {
      uc_unexp *u = uc_unexp_new(c, peer, tag, len);
      if (!u || !(u->data = malloc(len ? len : 1))) return UCOMM_ERR_NOMEM;
      if (len) memcpy(u->data, buf, len);
      u->complete = 1;
    }
    uc_req_complete(r, UCOMM_OK);
    return UCOMM_OK;
  }
  return c->ops->isend(c, r);
}

static ucomm_status irecv_tag(ucomm_t *c, int peer, uint32_t tag, void *buf, size_t cap, ucomm_req_t **out) {
  if (!c || !out || peer < 0 || peer >= c->world || (cap && !buf)) return UCOMM_ERR_ARG;
  ucomm_req_t *r = req_new(c, UC_REQ_RECV, peer, tag, buf, cap);
  if (!r) return UCOMM_ERR_NOMEM;
  *out = r;
  for (uc_unexp *u = c->unexp[peer]; u; u = u->next) {
    if (u->tag != tag || u->waiter) continue;
    if (u->rndv) {
      unexp_unlink(c, u);
      ucomm_status s = c->ops->rndv_recv(c, r, u);
      free(u);
      return s;
    }
    if (u->complete) {
      copy_into(r, u->data, u->len);
      unexp_unlink(c, u);
      free(u->data);
      free(u);
    } else {
      u->waiter = r;
    }
    return UCOMM_OK;
  }
  if (c->posted_tail[peer])
    c->posted_tail[peer]->next = r;
  else
    c->posted[peer] = r;
  c->posted_tail[peer] = r;
  return UCOMM_OK;
}

ucomm_status ucomm_isend(ucomm_t *c, int peer, uint32_t tag, const void *buf, size_t len, ucomm_req_t **out) {
  if (tag >= UCOMM_TAG_RESERVED) return UCOMM_ERR_ARG;
  return isend_tag(c, peer, tag, buf, len, out);
}

ucomm_status ucomm_irecv(ucomm_t *c, int peer, uint32_t tag, void *buf, size_t cap, ucomm_req_t **out) {
  if (tag >= UCOMM_TAG_RESERVED) return UCOMM_ERR_ARG;
  return irecv_tag(c, peer, tag, buf, cap, out);
}

ucomm_status ucomm_test(ucomm_req_t *r, int *done, size_t *nbytes) {
  if (!r) return UCOMM_ERR_ARG;
  if (!r->done) r->c->ops->progress(r->c, 0);
  if (done) *done = r->done;
  if (!r->done) return UCOMM_OK;
  ucomm_status s = r->status;
  if (nbytes) *nbytes = r->nbytes;
  free(r);
  return s;
}

ucomm_status ucomm_wait(ucomm_req_t *r, size_t *nbytes) {
  if (!r) return UCOMM_ERR_ARG;
  ucomm_t *c = r->c;
  uint64_t deadline = c->cfg.timeout_ms > 0 ? uc_now_us() + (uint64_t)c->cfg.timeout_ms * 1000 : 0;
  while (!r->done) {
    ucomm_status s = c->ops->progress(c, 10);
    if (s != UCOMM_OK) return s;
    if (deadline && !r->done && uc_now_us() > deadline) return UCOMM_ERR_TIMEOUT;
  }
  ucomm_status s = r->status;
  if (nbytes) *nbytes = r->nbytes;
  free(r);
  return s;
}

ucomm_status ucomm_waitall(int n, ucomm_req_t **reqs) {
  ucomm_status first = UCOMM_OK;
  for (int i = 0; i < n; i++) {
    if (!reqs[i]) continue;
    ucomm_status s = ucomm_wait(reqs[i], NULL);
    if (s == UCOMM_ERR_TIMEOUT) return s; /* remaining requests stay valid */
    reqs[i] = NULL;
    if (s != UCOMM_OK && first == UCOMM_OK) first = s;
  }
  return first;
}

ucomm_status ucomm_send(ucomm_t *c, int peer, uint32_t tag, const void *buf, size_t len) {
  ucomm_req_t *r;
  ucomm_status s = ucomm_isend(c, peer, tag, buf, len, &r);
  return s != UCOMM_OK ? s : ucomm_wait(r, NULL);
}

ucomm_status ucomm_recv(ucomm_t *c, int peer, uint32_t tag, void *buf, size_t cap, size_t *nbytes) {
  ucomm_req_t *r;
  ucomm_status s = ucomm_irecv(c, peer, tag, buf, cap, &r);
  return s != UCOMM_OK ? s : ucomm_wait(r, nbytes);
}

/* ---- collectives ---- */

static uint32_t coll_tag(ucomm_t *c) { return UCOMM_TAG_RESERVED | (c->coll_seq++ & 0x7fffffffu); }

static ucomm_status sendrecv(ucomm_t *c, uint32_t tag, int to, const void *sb, size_t sn, int from, void *rb,
                             size_t rn) {
  ucomm_req_t *rq[2];
  ucomm_status s = irecv_tag(c, from, tag, rb, rn, &rq[0]);
  if (s != UCOMM_OK) return s;
  s = isend_tag(c, to, tag, sb, sn, &rq[1]);
  if (s != UCOMM_OK) {
    ucomm_wait(rq[0], NULL);
    return s;
  }
  return ucomm_waitall(2, rq);
}

ucomm_status ucomm_barrier(ucomm_t *c) {
  /* dissemination barrier: log2(world) rounds of zero-byte messages */
  uint32_t tag = coll_tag(c);
  for (int k = 1; k < c->world; k <<= 1) {
    ucomm_status s = sendrecv(c, tag, (c->rank + k) % c->world, NULL, 0, (c->rank - k + c->world) % c->world,
                              NULL, 0);
    if (s != UCOMM_OK) return s;
  }
  return UCOMM_OK;
}

ucomm_status ucomm_bcast(ucomm_t *c, void *buf, size_t len, int root) {
  if (root < 0 || root >= c->world) return UCOMM_ERR_ARG;
  uint32_t tag = coll_tag(c);
  ucomm_req_t *rq[UCOMM_MAX_WORLD];
  int n = 0;
  ucomm_status s = UCOMM_OK;
  if (c->rank != root) {
    s = irecv_tag(c, root, tag, buf, len, &rq[n]);
    if (s == UCOMM_OK) n++;
  } else {
    for (int p = 0; p < c->world && s == UCOMM_OK; p++) /* world <= 8: flat fan-out is fine */
      if (p != root && (s = isend_tag(c, p, tag, buf, len, &rq[n])) == UCOMM_OK) n++;
  }
  ucomm_status w = ucomm_waitall(n, rq);
  return s != UCOMM_OK ? s : w;
}

static size_t dt_size(ucomm_dtype dt) { return dt == UCOMM_F16 || dt == UCOMM_BF16 ? 2 : 4; }

static float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
  if (e == 0) {
    if (!m) {
      u = s;
    } else { /* subnormal */
      e = 113;
      while (!(m & 0x400)) {
        m <<= 1;
        e--;
      }
      u = s | (e << 23) | ((m & 0x3ff) << 13);
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | (m << 13);
  } else {
    u = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static uint16_t f2h(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  uint16_t s = (uint16_t)((u >> 16) & 0x8000);
  int32_t e = (int32_t)((u >> 23) & 0xff) - 112;
  uint32_t m = u & 0x7fffff;
  if (((u >> 23) & 0xff) == 0xff) return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
  if (e >= 31) return (uint16_t)(s | 0x7c00);
  if (e <= 0) {
    if (e < -10) return s;
    m |= 0x800000;
    uint32_t sh = (uint32_t)(14 - e);
    uint32_t hm = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hm & 1))) hm++;
    return (uint16_t)(s | hm);
  }
  uint32_t hm = m >> 13, rem = m & 0x1fff;
  uint32_t out = ((uint32_t)e << 10) | hm;
  if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) out++; /* may carry into exponent: correct */
  return (uint16_t)(s | out);
}

static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

static void reduce_add(void *dst, const void *src, size_t n, ucomm_dtype dt) {
  switch (dt) {
    case UCOMM_F32: {
      float *d = dst;
      const float *s = src;
      for (size_t i = 0; i < n; i++) d[i] += s[i];
    } break;
    case UCOMM_I32: {
      int32_t *d = dst;
      const int32_t *s = src;
      for (size_t i = 0; i < n; i++) d[i] = (int32_t)((uint32_t)d[i] + (uint32_t)s[i]);
    } break;
    case UCOMM_F16: {
      uint16_t *d = dst;
      const uint16_t *s = src;
      for (size_t i = 0; i < n; i++) d[i] = f2h(h2f(d[i]) + h2f(s[i]));
    } break;
    case UCOMM_BF16: {
      uint16_t *d = dst;
      const uint16_t *s = src;
      for (size_t i = 0; i < n; i++) d[i] = f2bf(bf2f(d[i]) + bf2f(s[i]));
    } break;
  }
}

/* Ring allreduce: reduce-scatter then allgather, world-1 steps each. Segment i covers [seg_off(i), seg_off(i+1)). */
ucomm_status ucomm_allreduce_sum(ucomm_t *c, void *buf, size_t count, ucomm_dtype dt) {
  if (!c || (count && !buf) || (int)dt < 0 || dt > UCOMM_I32) return UCOMM_ERR_ARG;
  int W = c->world, me = c->rank;
  if (W == 1 || !count) return UCOMM_OK;
  size_t es = dt_size(dt);
  size_t seg[UCOMM_MAX_WORLD + 1];
  for (int i = 0; i <= W; i++) seg[i] = count * (size_t)i / (size_t)W;
  size_t maxseg = 0;
  for (int i = 0; i < W; i++)
    if (seg[i + 1] - seg[i] > maxseg) maxseg = seg[i + 1] - seg[i];
  char *tmp = malloc(maxseg * es + 1);
  if (!tmp) return UCOMM_ERR_NOMEM;
  char *b = buf;
  int right = (me + 1) % W, left = (me - 1 + W) % W;
  uint32_t tag = coll_tag(c);
  ucomm_status s = UCOMM_OK;
  for (int k = 0; k < W - 1 && s == UCOMM_OK; k++) {
    int si = (me - k + W) % W, ri = (me - k - 1 + W) % W;
    s = sendrecv(c, tag, right, b + seg[si] * es, (seg[si + 1] - seg[si]) * es, left, tmp,
                 (seg[ri + 1] - seg[ri]) * es);
    if (s == UCOMM_OK) reduce_add(b + seg[ri] * es, tmp, seg[ri + 1] - seg[ri], dt);
  }
  for (int k = 0; k < W - 1 && s == UCOMM_OK; k++) {
    int si = (me + 1 - k + W) % W, ri = (me - k + W) % W;
    s = sendrecv(c, tag, right, b + seg[si] * es, (seg[si + 1] - seg[si]) * es, left, b + seg[ri] * es,
                 (seg[ri + 1] - seg[ri]) * es);
  }
  free(tmp);
  return s;
}

ucomm_status ucomm_allgather(ucomm_t *c, const void *in, void *out, size_t n) {
  if (!c || (n && (!in || !out))) return UCOMM_ERR_ARG;
  int W = c->world, me = c->rank;
  char *o = out;
  if ((const char *)in != o + (size_t)me * n && n) memmove(o + (size_t)me * n, in, n);
  uint32_t tag = coll_tag(c);
  int right = (me + 1) % W, left = (me - 1 + W) % W;
  for (int k = 0; k < W - 1; k++) {
    int si = (me - k + W) % W, ri = (me - k - 1 + W) % W;
    ucomm_status s = sendrecv(c, tag, right, o + (size_t)si * n, n, left, o + (size_t)ri * n, n);
    if (s != UCOMM_OK) return s;
  }
  return UCOMM_OK;
}
