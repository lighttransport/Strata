/* InfiniBand / RoCE backend over dlsym'd verbs (see ibvew.h).
 *
 * One RC queue pair per peer, one shared completion queue, busy polling.
 *   - Eager: messages up to EAGER_MAX are copied into a registered send slot and SENT into one of SLOTS
 *     pre-posted receive slots on the peer.
 *   - Rendezvous: larger messages send an RTS {addr, rkey, len}; the receiver RDMA-READs straight into the
 *     user buffer in READ_CHUNK pieces and answers with FIN, which completes the send.
 *   - Flow control: the receiver RDMA-WRITEs its running count of reposted slots into a credit word on the
 *     sender, so the sender never overruns the receive ring (no RNR) and credits do not use receive slots.
 * Buffers passed through ucomm_mr_reg are used as is; other rendezvous buffers are registered for the
 * duration of one transfer. */
#define _GNU_SOURCE
#include "ibvew.h"
#include "ucomm_internal.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SLOTS 64
#define SLOT_SZ 8192
#define MHDR 40
#define EAGER_MAX (SLOT_SZ - MHDR)
#define READ_CHUNK (1u << 20)
#define SQ_DEPTH 256
#define FINQ 128

enum { M_EAGER = 1, M_RTS, M_FIN };
enum { W_RECV = 1, W_SEND, W_READ, W_CREDIT };

#define WRID(type, peer, val) (((uint64_t)(type) << 60) | ((uint64_t)(peer) << 52) | (uint64_t)(val))
#define WR_TYPE(id) ((int)((id) >> 60))
#define WR_PEER(id) ((int)(((id) >> 52) & 0xff))
#define WR_VAL(id) ((id) & ((1ull << 52) - 1))

typedef struct {
  uint8_t recv[SLOTS][SLOT_SZ];
  uint8_t send[SLOTS][SLOT_SZ];
  volatile uint64_t credit_in; /* written by the peer: slots it has reposted for us */
  uint64_t credit_out;         /* source of our credit RDMA write */
} ibblk;

typedef struct {
  struct ibv_qp *qp;
  ibblk *blk;
  struct ibv_mr *mr;
  uint64_t r_credit_addr;
  uint32_t r_rkey;
  uint64_t sent, reposted, credit_sent_at;
  int free_slot[SLOTS], nfree;
  int sq_used;
  ucomm_req_t *sq, *sq_tail; /* sends waiting for a slot / credit */
  ucomm_req_t *rq, *rq_tail; /* rendezvous receives with reads left to post */
  uint64_t finq[FINQ];
  int fin_n;
} ibpeer;

typedef struct {
  ibvew_api api;
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_port_attr port;
  union ibv_gid gid;
  int port_num, roce;
  ibpeer peer[UCOMM_MAX_WORLD];
} ib_be;

typedef struct {
  uint32_t type, tag;
  uint64_t len, raddr;
  uint32_t rkey, pad;
  uint64_t sreq;
} mhdr;

/* Open the configured (or first active) device. Returns NULL if none. */
static struct ibv_context *open_active(ucomm_t *c, ibvew_api *api, struct ibv_port_attr *pa) {
  int n = 0;
  struct ibv_device **list = api->get_device_list(&n);
  if (!list) return NULL;
  struct ibv_context *found = NULL;
  for (int i = 0; i < n && !found; i++) {
    const char *name = api->get_device_name(list[i]);
    if (c->cfg.ib_dev && *c->cfg.ib_dev && (!name || strcmp(name, c->cfg.ib_dev))) continue;
    struct ibv_context *ctx = api->open_device(list[i]);
    if (!ctx) continue;
    memset(pa, 0, sizeof *pa);
    if (api->query_port(ctx, (uint8_t)c->cfg.ib_port, pa) == 0 && pa->state == IBV_PORT_ACTIVE) {
      found = ctx;
      UC_LOG(c, "ib device %s port %d active (lid %u, %s)", name ? name : "?", c->cfg.ib_port, pa->lid,
             pa->link_layer == IBV_LINK_LAYER_ETHERNET ? "RoCE" : "IB");
    } else {
      api->close_device(ctx);
    }
  }
  api->free_device_list(list);
  return found;
}

int uc_ib_probe(ucomm_t *c) {
  ibvew_api api;
  if (ibvew_load(&api) != 0) {
    UC_LOG(c, "libibverbs not found");
    return 0;
  }
  struct ibv_port_attr pa;
  struct ibv_context *ctx = open_active(c, &api, &pa);
  if (!ctx) UC_LOG(c, "libibverbs loaded but no active HCA port");
  if (ctx) api.close_device(ctx);
  ibvew_unload(&api);
  return ctx != NULL;
}

static struct ibv_mr *user_mr(ucomm_t *c, const void *p, size_t n) {
  for (ucomm_mr_t *m = c->mrs; m; m = m->next)
    if ((const char *)p >= (char *)m->ptr && (const char *)p + n <= (char *)m->ptr + m->len) return m->bmr;
  return NULL;
}

/* Registration for one transfer. *owned set when the caller must deregister. */
static struct ibv_mr *get_mr(ucomm_t *c, void *p, size_t n, int *owned) {
  ib_be *b = c->be;
  struct ibv_mr *mr = user_mr(c, p, n);
  *owned = 0;
  if (mr) return mr;
  mr = b->api.reg_mr(b->pd, p, n, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
  *owned = mr != NULL;
  return mr;
}

/* Send side: r->bmr is the registration, r->rkey == 1 when it was made for this transfer only. */
static void put_send_mr(ucomm_t *c, ucomm_req_t *r) {
  ib_be *b = c->be;
  if (r->bmr && r->rkey == 1) b->api.dereg_mr(r->bmr);
  r->bmr = NULL;
}

static int can_post(ib_be *b, int pi) {
  ibpeer *p = &b->peer[pi];
  int64_t credits = (int64_t)(SLOTS + p->blk->credit_in - p->sent);
  return credits > 0 && p->nfree > 0 && p->sq_used < SQ_DEPTH;
}

/* Post one message into a send slot; caller checked can_post. */
static int post_msg(ib_be *b, int pi, const mhdr *h, const void *payload, size_t plen) {
  ibpeer *p = &b->peer[pi];
  int slot = p->free_slot[--p->nfree];
  uint8_t *dst = p->blk->send[slot];
  memcpy(dst, h, MHDR);
  if (plen) memcpy(dst + MHDR, payload, plen);
  struct ibv_sge sge = {(uint64_t)(uintptr_t)dst, (uint32_t)(MHDR + plen), p->mr->lkey};
  struct ibv_send_wr wr, *bad;
  memset(&wr, 0, sizeof wr);
  wr.wr_id = WRID(W_SEND, pi, slot);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibvew_post_send(p->qp, &wr, &bad) != 0) {
    p->free_slot[p->nfree++] = slot;
    return -1;
  }
  p->sq_used++;
  p->sent++;
  return 0;
}

static int post_recv_slot(ib_be *b, int pi, int slot) {
  ibpeer *p = &b->peer[pi];
  struct ibv_sge sge = {(uint64_t)(uintptr_t)p->blk->recv[slot], SLOT_SZ, p->mr->lkey};
  struct ibv_recv_wr wr, *bad;
  memset(&wr, 0, sizeof wr);
  wr.wr_id = WRID(W_RECV, pi, slot);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  return ibvew_post_recv(p->qp, &wr, &bad);
}

static int write_credit(ib_be *b, int pi) {
  ibpeer *p = &b->peer[pi];
  if (p->sq_used >= SQ_DEPTH) return 1;
  p->blk->credit_out = p->reposted;
  struct ibv_sge sge = {(uint64_t)(uintptr_t)&p->blk->credit_out, 8, p->mr->lkey};
  struct ibv_send_wr wr, *bad;
  memset(&wr, 0, sizeof wr);
  wr.wr_id = WRID(W_CREDIT, pi, 0);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = p->r_credit_addr;
  wr.wr.rdma.rkey = p->r_rkey;
  if (ibvew_post_send(p->qp, &wr, &bad) != 0) return -1;
  p->sq_used++;
  p->credit_sent_at = p->reposted;
  return 0;
}

/* Try to emit a queued send (eager or RTS). 0 posted, 1 no resources, -1 error. */
static int try_send(ucomm_t *c, ucomm_req_t *r) {
  ib_be *b = c->be;
  if (!can_post(b, r->peer)) return 1;
  mhdr h;
  memset(&h, 0, sizeof h);
  h.tag = r->tag;
  h.len = r->len;
  if (r->len <= EAGER_MAX) {
    h.type = M_EAGER;
    if (post_msg(b, r->peer, &h, r->buf, r->len) != 0) return -1;
    uc_req_complete(r, UCOMM_OK); /* data copied into the slot */
    return 0;
  }
  int owned;
  struct ibv_mr *mr = get_mr(c, r->buf, r->len, &owned);
  if (!mr) return -1;
  r->bmr = mr;
  r->rkey = owned ? 1u : 0u;
  h.type = M_RTS;
  h.raddr = (uint64_t)(uintptr_t)r->buf;
  h.rkey = mr->rkey;
  h.sreq = (uint64_t)(uintptr_t)r;
  if (post_msg(b, r->peer, &h, NULL, 0) != 0) {
    put_send_mr(c, r);
    return -1;
  }
  return 0; /* completes on FIN */
}

static ucomm_status ib_isend(ucomm_t *c, ucomm_req_t *r) {
  ib_be *b = c->be;
  ibpeer *p = &b->peer[r->peer];
  if (!p->sq) {
    int rc = try_send(c, r);
    if (rc < 0) return UCOMM_ERR_IO;
    if (rc == 0) return UCOMM_OK;
  }
  if (p->sq_tail)
    p->sq_tail->next = r;
  else
    p->sq = r;
  p->sq_tail = r;
  return UCOMM_OK;
}

static void start_rndv(ucomm_t *c, ucomm_req_t *r, uint64_t raddr, uint32_t rkey, size_t len, uint64_t sreq) {
  ib_be *b = c->be;
  ibpeer *p = &b->peer[r->peer];
  r->nbytes = len;
  r->raddr = raddr;
  r->sreq = sreq;
  r->off = 0;
  r->pending = 0;
  r->rkey = rkey; /* receive side: the sender's rkey */
  r->bmr = NULL;
  r->next = NULL;
  if (p->rq_tail)
    p->rq_tail->next = r;
  else
    p->rq = r;
  p->rq_tail = r;
}

static ucomm_status ib_rndv_recv(ucomm_t *c, ucomm_req_t *r, uc_unexp *u) {
  start_rndv(c, r, u->raddr, u->rkey, u->len, u->sreq);
  return UCOMM_OK;
}

/* Owned registrations for receives are tracked separately because r->rkey holds the remote key. */
typedef struct {
  struct ibv_mr *mr;
  int owned;
} recv_mr;

static void finish_recv(ucomm_t *c, int pi, ucomm_req_t *r) {
  ib_be *b = c->be;
  ibpeer *p = &b->peer[pi];
  recv_mr *rm = r->bmr;
  if (rm) {
    if (rm->owned) b->api.dereg_mr(rm->mr);
    free(rm);
    r->bmr = NULL;
  }
  if (p->fin_n < FINQ) p->finq[p->fin_n++] = r->sreq;
  uc_req_complete(r, r->nbytes > r->len ? UCOMM_ERR_TRUNC : UCOMM_OK);
}

/* Post RDMA reads for queued rendezvous receives. */
static int pump_reads(ucomm_t *c, int pi) {
  ib_be *b = c->be;
  ibpeer *p = &b->peer[pi];
  while (p->rq) {
    ucomm_req_t *r = p->rq;
    size_t want = r->nbytes < r->len ? r->nbytes : r->len;
    if (want && !r->bmr) {
      recv_mr *rm = calloc(1, sizeof *rm);
      if (!rm) return -1;
      rm->mr = get_mr(c, r->buf, want, &rm->owned);
      if (!rm->mr) {
        free(rm);
        return -1;
      }
      r->bmr = rm;
    }
    while (r->off < want && p->sq_used < SQ_DEPTH) {
      size_t n = want - r->off < READ_CHUNK ? want - r->off : READ_CHUNK;
      recv_mr *rm = r->bmr;
      struct ibv_sge sge = {(uint64_t)(uintptr_t)((char *)r->buf + r->off), (uint32_t)n, rm->mr->lkey};
      struct ibv_send_wr wr, *bad;
      memset(&wr, 0, sizeof wr);
      wr.wr_id = WRID(W_READ, pi, (uint64_t)(uintptr_t)r);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_READ;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = r->raddr + r->off;
      wr.wr.rdma.rkey = r->rkey;
      if (ibvew_post_send(p->qp, &wr, &bad) != 0) return -1;
      p->sq_used++;
      r->pending++;
      r->off += n;
    }
    if (r->off < want) return 0; /* send queue full */
    p->rq = r->next;
    if (!p->rq) p->rq_tail = NULL;
    r->next = NULL;
    if (r->pending == 0) finish_recv(c, pi, r); /* zero-byte read */
  }
  return 0;
}

static void on_recv(ucomm_t *c, int pi, int slot) {
  ib_be *b = c->be;
  ibpeer *p = &b->peer[pi];
  mhdr h;
  memcpy(&h, p->blk->recv[slot], MHDR);
  const uint8_t *payload = p->blk->recv[slot] + MHDR;
  if (h.type == M_EAGER) {
    ucomm_req_t *r = uc_match_posted(c, pi, h.tag);
    if (r) {
      r->nbytes = h.len;
      size_t n = h.len < r->len ? h.len : r->len;
      if (n) memcpy(r->buf, payload, n);
      uc_req_complete(r, h.len > r->len ? UCOMM_ERR_TRUNC : UCOMM_OK);
    } else {
      uc_unexp *u = uc_unexp_new(c, pi, h.tag, h.len);
      if (u && (u->data = malloc(h.len ? h.len : 1))) {
        if (h.len) memcpy(u->data, payload, h.len);
        uc_unexp_done(c, u);
      }
    }
  } else if (h.type == M_RTS) {
    ucomm_req_t *r = uc_match_posted(c, pi, h.tag);
    if (r) {
      start_rndv(c, r, h.raddr, h.rkey, h.len, h.sreq);
    } else {
      uc_unexp *u = uc_unexp_new(c, pi, h.tag, h.len);
      if (u) {
        u->rndv = 1;
        u->raddr = h.raddr;
        u->rkey = h.rkey;
        u->sreq = h.sreq;
      }
    }
  } else if (h.type == M_FIN) {
    ucomm_req_t *s = (ucomm_req_t *)(uintptr_t)h.sreq;
    put_send_mr(c, s);
    uc_req_complete(s, UCOMM_OK);
  }
  post_recv_slot(b, pi, slot);
  p->reposted++;
}

static ucomm_status ib_progress(ucomm_t *c, int timeout_ms) {
  ib_be *b = c->be;
  uint64_t end = uc_now_us() + (uint64_t)(timeout_ms > 0 ? timeout_ms : 0) * 1000;
  for (;;) {
    struct ibv_wc wc[32];
    int n = ibvew_poll_cq(b->cq, 32, wc);
    if (n < 0) return UCOMM_ERR_IO;
    for (int i = 0; i < n; i++) {
      int pi = WR_PEER(wc[i].wr_id), t = WR_TYPE(wc[i].wr_id);
      uint64_t v = WR_VAL(wc[i].wr_id);
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "[ucomm %d] ib completion error %d (vendor 0x%x) on op %d peer %d\n", c->rank,
                wc[i].status, wc[i].vendor_err, t, pi);
        return UCOMM_ERR_IO;
      }
      ibpeer *p = &b->peer[pi];
      switch (t) {
        case W_RECV: on_recv(c, pi, (int)v); break;
        case W_SEND:
          p->free_slot[p->nfree++] = (int)v;
          p->sq_used--;
          break;
        case W_CREDIT: p->sq_used--; break;
        case W_READ: {
          ucomm_req_t *r = (ucomm_req_t *)(uintptr_t)v;
          p->sq_used--;
          size_t want = r->nbytes < r->len ? r->nbytes : r->len;
          if (--r->pending == 0 && r->off == want) finish_recv(c, pi, r);
        } break;
      }
    }
    for (int pi = 0; pi < c->world; pi++) {
      if (pi == c->rank) continue;
      ibpeer *p = &b->peer[pi];
      if (p->reposted - p->credit_sent_at >= SLOTS / 4 && write_credit(b, pi) < 0) return UCOMM_ERR_IO;
      while (p->fin_n && can_post(b, pi)) {
        mhdr h;
        memset(&h, 0, sizeof h);
        h.type = M_FIN;
        h.sreq = p->finq[0];
        if (post_msg(b, pi, &h, NULL, 0) != 0) return UCOMM_ERR_IO;
        memmove(p->finq, p->finq + 1, (size_t)--p->fin_n * sizeof p->finq[0]);
      }
      while (p->sq) {
        int rc = try_send(c, p->sq);
        if (rc < 0) return UCOMM_ERR_IO;
        if (rc > 0) break;
        ucomm_req_t *r = p->sq;
        p->sq = r->next;
        if (!p->sq) p->sq_tail = NULL;
        r->next = NULL;
      }
      if (pump_reads(c, pi) < 0) return UCOMM_ERR_IO;
    }
    if (n > 0 || uc_now_us() >= end) break;
  }
  return UCOMM_OK;
}

static ucomm_status ib_mr_reg(ucomm_t *c, ucomm_mr_t *m) {
  ib_be *b = c->be;
  m->bmr = b->api.reg_mr(b->pd, m->ptr, m->len,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
  return m->bmr ? UCOMM_OK : UCOMM_ERR_IO;
}

static void ib_mr_dereg(ucomm_t *c, ucomm_mr_t *m) {
  ib_be *b = c->be;
  if (m->bmr) b->api.dereg_mr(m->bmr);
  m->bmr = NULL;
}

static int ib_idle(ucomm_t *c) {
  ib_be *b = c->be;
  for (int pi = 0; pi < c->world; pi++) {
    ibpeer *p = &b->peer[pi];
    if (p->sq || p->rq || p->fin_n || p->sq_used) return 0;
  }
  return 1;
}

static void ib_finalize(ucomm_t *c) {
  ib_be *b = c->be;
  if (!b) return;
  for (int pi = 0; pi < c->world; pi++) {
    ibpeer *p = &b->peer[pi];
    if (p->qp) b->api.destroy_qp(p->qp);
    if (p->mr) b->api.dereg_mr(p->mr);
    free(p->blk);
  }
  for (ucomm_mr_t *m = c->mrs; m; m = m->next) ib_mr_dereg(c, m);
  if (b->cq) b->api.destroy_cq(b->cq);
  if (b->pd) b->api.dealloc_pd(b->pd);
  if (b->ctx) b->api.close_device(b->ctx);
  ibvew_unload(&b->api);
  free(b);
  c->be = NULL;
}

const uc_ops uc_ib_ops = {"ib", ib_isend, ib_rndv_recv, ib_progress, ib_mr_reg, ib_mr_dereg, ib_idle, ib_finalize};

typedef struct {
  union ibv_gid gid;
  uint16_t lid, pad;
  uint32_t qpn[UCOMM_MAX_WORLD];
  uint32_t psn[UCOMM_MAX_WORLD];
  uint64_t credit_addr[UCOMM_MAX_WORLD];
  uint32_t rkey[UCOMM_MAX_WORLD];
} ib_info;

static int qp_connect(ucomm_t *c, ib_be *b, int pi, const ib_info *mine, const ib_info *theirs) {
  ibpeer *p = &b->peer[pi];
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = b->port.active_mtu < IBV_MTU_2048 ? b->port.active_mtu : IBV_MTU_2048;
  a.dest_qp_num = theirs->qpn[c->rank];
  a.rq_psn = theirs->psn[c->rank];
  a.max_dest_rd_atomic = 4;
  a.min_rnr_timer = 12;
  a.ah_attr.dlid = theirs->lid;
  a.ah_attr.port_num = (uint8_t)b->port_num;
  if (b->roce) {
    a.ah_attr.is_global = 1;
    a.ah_attr.grh.dgid = theirs->gid;
    a.ah_attr.grh.sgid_index = (uint8_t)c->cfg.ib_gid_index;
    a.ah_attr.grh.hop_limit = 64;
  }
  if (b->api.modify_qp(p->qp, &a,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                           IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    return -1;
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = mine->psn[pi];
  a.max_rd_atomic = 4;
  return b->api.modify_qp(p->qp, &a,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                              IBV_QP_MAX_QP_RD_ATOMIC);
}

ucomm_status uc_ib_init(ucomm_t *c) {
  ib_be *b = calloc(1, sizeof *b);
  if (!b) return UCOMM_ERR_NOMEM;
  c->be = b;
  ib_info mine, all[UCOMM_MAX_WORLD];
  memset(&mine, 0, sizeof mine);
  if (ibvew_load(&b->api) != 0) goto fail;
  b->ctx = open_active(c, &b->api, &b->port);
  if (!b->ctx) goto fail;
  b->port_num = c->cfg.ib_port;
  b->roce = b->port.link_layer == IBV_LINK_LAYER_ETHERNET;
  if (b->api.query_gid(b->ctx, (uint8_t)b->port_num, c->cfg.ib_gid_index, &b->gid) != 0 && b->roce) goto fail;
  b->pd = b->api.alloc_pd(b->ctx);
  if (!b->pd) goto fail;
  b->cq = b->api.create_cq(b->ctx, c->world * (SLOTS + SQ_DEPTH), NULL, NULL, 0);
  if (!b->cq) goto fail;

  mine.gid = b->gid;
  mine.lid = b->port.lid;
  for (int pi = 0; pi < c->world; pi++) {
    if (pi == c->rank) continue;
    ibpeer *p = &b->peer[pi];
    if (posix_memalign((void **)&p->blk, 4096, sizeof *p->blk) != 0) {
      p->blk = NULL;
      goto fail;
    }
    memset(p->blk, 0, sizeof *p->blk);
    p->mr = b->api.reg_mr(b->pd, p->blk, sizeof *p->blk,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!p->mr) goto fail;
    for (int s = 0; s < SLOTS; s++) p->free_slot[s] = s;
    p->nfree = SLOTS;
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.send_cq = b->cq;
    qa.recv_cq = b->cq;
    qa.cap.max_send_wr = SQ_DEPTH;
    qa.cap.max_recv_wr = SLOTS;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.qp_type = IBV_QPT_RC;
    p->qp = b->api.create_qp(b->pd, &qa);
    if (!p->qp) goto fail;
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT;
    a.port_num = (uint8_t)b->port_num;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (b->api.modify_qp(p->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
      goto fail;
    for (int s = 0; s < SLOTS; s++)
      if (post_recv_slot(b, pi, s) != 0) goto fail;
    mine.qpn[pi] = p->qp->qp_num;
    mine.psn[pi] = (uint32_t)(uc_now_us() + (uint64_t)pi * 7919) & 0xffffff;
    mine.credit_addr[pi] = (uint64_t)(uintptr_t)&p->blk->credit_in;
    mine.rkey[pi] = p->mr->rkey;
  }
  if (uc_boot_allgather(c, &mine, all, sizeof mine) != UCOMM_OK) goto fail;
  for (int pi = 0; pi < c->world; pi++) {
    if (pi == c->rank) continue;
    if (qp_connect(c, b, pi, &mine, &all[pi]) != 0) {
      fprintf(stderr, "[ucomm %d] ib: connecting QP to rank %d failed\n", c->rank, pi);
      goto fail;
    }
    b->peer[pi].r_credit_addr = all[pi].credit_addr[c->rank];
    b->peer[pi].r_rkey = all[pi].rkey[c->rank];
  }
  /* Every QP must be RTS before anyone sends. */
  uint32_t dummy = 0, dall[UCOMM_MAX_WORLD];
  if (uc_boot_allgather(c, &dummy, dall, sizeof dummy) != UCOMM_OK) goto fail;
  return UCOMM_OK;
fail:
  fprintf(stderr, "[ucomm %d] ib backend init failed\n", c->rank);
  ib_finalize(c);
  return UCOMM_ERR_IO;
}
