/* Reliable UDP backend.
 *
 * One datagram socket per rank. Each message is cut into fragments of at most udp_mtu payload bytes; every
 * fragment gets a per-peer sequence number. The receiver delivers fragments in sequence order (holding up to
 * one window of out-of-order fragments) and acknowledges with a cumulative ack plus a 64-bit SACK bitmap,
 * piggybacked on data or sent alone. The sender keeps a congestion window (AIMD), retransmits on RTO and on
 * SACK holes. Fragments are sent straight from the user buffer, so a send completes only when every fragment
 * is acknowledged. */
#define _GNU_SOURCE
#include "ucomm_internal.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

enum { PK_DATA = 1, PK_ACK = 2 };
enum { FL_FIRST = 1, FL_LAST = 2 };

#define HDR_SZ 24  /* session u32, type u8, src u8, flags u8, pad u8, seq u32, ack u32, sack u64 */
#define FIRST_SZ 16 /* tag u32, pad u32, len u64 */
#define WIN_MAX 4096
#define BATCH 64
#define EAGER_MAX (16 << 10) /* sends up to this size are copied and complete at once */

typedef struct {
  uint32_t session;
  uint8_t type, src, flags, pad;
  uint32_t seq, ack;
  uint64_t sack;
} pk_hdr;

typedef struct {
  ucomm_req_t *req;
  size_t off;
  uint32_t len;
  uint8_t flags, acked;
  uint16_t nsent;
  uint64_t t_sent;
} sfrag;

typedef struct {
  struct sockaddr_in addr;
  uint32_t rwnd; /* peer's receive window in fragments */
  /* send side */
  ucomm_req_t *sq, *sq_tail; /* queued sends; head may be partially fragmented */
  sfrag *win;                /* [WIN_MAX] indexed by seq % WIN_MAX */
  uint32_t una, nxt;         /* oldest unacked, next to send */
  double cwnd;
  double srtt, rttvar;
  uint64_t rto_us;
  /* receive side */
  uint32_t rnext;       /* next expected seq */
  uint8_t **ooo;        /* [rwnd_local] stored packets, NULL if empty */
  uint32_t *ooo_len;
  int ack_pending;
  /* reassembly of the message currently arriving */
  int rx_active;
  size_t rx_len, rx_got;
  char *rx_dst;
  size_t rx_cap;
  ucomm_req_t *rx_req;
  uc_unexp *rx_unexp;
} upeer;

typedef struct {
  int fd;
  int mtu;       /* payload bytes per datagram */
  int spin_us;   /* busy-poll this long before sleeping in poll() (UCOMM_SPIN_US) */
  uint32_t rwnd; /* our receive window in fragments */
  upeer peer[UCOMM_MAX_WORLD];
  uint8_t *rxbuf; /* BATCH * (HDR_SZ + FIRST_SZ + mtu) */
  size_t rxslot;
  double drop;    /* UCOMM_UDP_DROP test hook: send-side drop probability */
  uint64_t rng;
  uint64_t retrans, sent;
} udp_be;

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static uint32_t get32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
static uint64_t get64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

static void hdr_pack(uint8_t *p, const pk_hdr *h) {
  put32(p, h->session);
  p[4] = h->type;
  p[5] = h->src;
  p[6] = h->flags;
  p[7] = 0;
  put32(p + 8, h->seq);
  put32(p + 12, h->ack);
  put64(p + 16, h->sack);
}

static void hdr_unpack(const uint8_t *p, pk_hdr *h) {
  h->session = get32(p);
  h->type = p[4];
  h->src = p[5];
  h->flags = p[6];
  h->seq = get32(p + 8);
  h->ack = get32(p + 12);
  h->sack = get64(p + 16);
}

static int seq_lt(uint32_t a, uint32_t b) { return (int32_t)(a - b) < 0; }

static uint64_t sack_bits(udp_be *u, upeer *p) {
  uint64_t bits = 0;
  for (uint32_t i = 0; i < 64 && i + 1 < u->rwnd; i++)
    if (p->ooo[(p->rnext + 1 + i) % u->rwnd]) bits |= 1ull << i;
  return bits;
}

static int drop_now(udp_be *u) {
  if (u->drop <= 0) return 0;
  u->rng ^= u->rng << 13;
  u->rng ^= u->rng >> 7;
  u->rng ^= u->rng << 17;
  return (double)(u->rng >> 11) / 9007199254740992.0 < u->drop;
}

/* Returns 0 sent (or dropped by the test hook), 1 would block, -1 error. */
static int send_frag(ucomm_t *c, udp_be *u, int pi, uint32_t seq) {
  upeer *p = &u->peer[pi];
  sfrag *f = &p->win[seq % WIN_MAX];
  uint8_t h[HDR_SZ + FIRST_SZ];
  pk_hdr ph = {c->session, PK_DATA, (uint8_t)c->rank, f->flags, 0, seq, p->rnext, sack_bits(u, p)};
  hdr_pack(h, &ph);
  size_t hl = HDR_SZ;
  if (f->flags & FL_FIRST) {
    put32(h + HDR_SZ, f->req->tag);
    put32(h + HDR_SZ + 4, 0);
    put64(h + HDR_SZ + 8, (uint64_t)f->req->len);
    hl += FIRST_SZ;
  }
  struct iovec iov[2] = {{h, hl}, {(char *)f->req->buf + f->off, f->len}};
  struct msghdr m;
  memset(&m, 0, sizeof m);
  m.msg_name = &p->addr;
  m.msg_namelen = sizeof p->addr;
  m.msg_iov = iov;
  m.msg_iovlen = f->len ? 2 : 1;
  f->t_sent = uc_now_us();
  f->nsent++;
  u->sent++;
  if (f->nsent > 1) u->retrans++;
  p->ack_pending = 0;
  if (drop_now(u)) return 0;
  if (sendmsg(u->fd, &m, MSG_DONTWAIT) < 0) {
    f->nsent--;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS || errno == EINTR) return 1;
    return -1;
  }
  return 0;
}

static void send_ack(ucomm_t *c, udp_be *u, int pi) {
  upeer *p = &u->peer[pi];
  uint8_t h[HDR_SZ];
  pk_hdr ph = {c->session, PK_ACK, (uint8_t)c->rank, 0, 0, 0, p->rnext, sack_bits(u, p)};
  hdr_pack(h, &ph);
  p->ack_pending = 0;
  if (drop_now(u)) return;
  sendto(u->fd, h, HDR_SZ, MSG_DONTWAIT, (struct sockaddr *)&p->addr, sizeof p->addr);
}

static void frag_acked(upeer *p, sfrag *f, uint64_t now) {
  if (f->acked || !f->req) return;
  f->acked = 1;
  if (f->nsent == 1) { /* Karn: only sample unambiguous fragments */
    double r = (double)(now - f->t_sent);
    if (p->srtt == 0) {
      p->srtt = r;
      p->rttvar = r / 2;
    } else {
      p->rttvar = 0.75 * p->rttvar + 0.25 * (r > p->srtt ? r - p->srtt : p->srtt - r);
      p->srtt = 0.875 * p->srtt + 0.125 * r;
    }
    double rto = p->srtt + 4 * p->rttvar + 200;
    p->rto_us = rto < 1000 ? 1000 : rto > 200000 ? 200000 : (uint64_t)rto;
  }
  ucomm_req_t *r = f->req;
  f->req = NULL;
  if (--r->pending == 0 && r->off == r->len) {
    if (r->kind == UC_REQ_INTERNAL)
      free(r);
    else
      uc_req_complete(r, UCOMM_OK);
  }
}

static void on_ack(ucomm_t *c, udp_be *u, int pi, uint32_t ack, uint64_t sack) {
  upeer *p = &u->peer[pi];
  uint64_t now = uc_now_us();
  if (seq_lt(p->nxt, ack) || seq_lt(ack, p->una)) return; /* stale or bogus */
  uint32_t newly = 0;
  while (seq_lt(p->una, ack)) {
    sfrag *f = &p->win[p->una % WIN_MAX];
    if (!f->acked) {
      frag_acked(p, f, now);
      newly++;
    }
    f->acked = 0;
    f->req = NULL;
    p->una++;
  }
  uint32_t hi = ack;
  for (uint32_t i = 0; i < 64; i++) {
    uint32_t s = ack + 1 + i;
    if (!seq_lt(s, p->nxt)) break;
    if (sack & (1ull << i)) {
      sfrag *f = &p->win[s % WIN_MAX];
      if (!f->acked) {
        frag_acked(p, f, now);
        newly++;
      }
      hi = s;
    }
  }
  if (newly) {
    p->cwnd += (double)newly / (p->cwnd > 1 ? p->cwnd : 1) * 4; /* grow ~4 fragments per window */
    if (p->cwnd > p->rwnd) p->cwnd = p->rwnd;
  }
  /* SACK hole repair: anything below the highest sacked seq that is still missing and older than srtt. */
  if (hi != ack) {
    uint64_t age = p->srtt > 0 ? (uint64_t)p->srtt : 500;
    for (uint32_t s = ack; seq_lt(s, hi); s++) {
      sfrag *f = &p->win[s % WIN_MAX];
      if (!f->acked && f->req && now - f->t_sent > age) {
        if (send_frag(c, u, pi, s) != 0) break;
      }
    }
  }
}

/* Deliver one in-order data fragment into the current message. */
static void deliver(ucomm_t *c, udp_be *u, int pi, const pk_hdr *h, const uint8_t *pl, size_t n) {
  upeer *p = &u->peer[pi];
  if (h->flags & FL_FIRST) {
    if (n < FIRST_SZ) return;
    uint32_t tag = get32(pl);
    uint64_t len = get64(pl + 8);
    pl += FIRST_SZ;
    n -= FIRST_SZ;
    p->rx_active = 1;
    p->rx_len = (size_t)len;
    p->rx_got = 0;
    p->rx_req = uc_match_posted(c, pi, tag);
    p->rx_unexp = NULL;
    if (p->rx_req) {
      p->rx_dst = p->rx_req->buf;
      p->rx_cap = p->rx_req->len;
    } else {
      p->rx_unexp = uc_unexp_new(c, pi, tag, (size_t)len);
      p->rx_unexp->data = malloc(len ? (size_t)len : 1);
      p->rx_dst = p->rx_unexp->data;
      p->rx_cap = (size_t)len;
    }
  }
  if (!p->rx_active) return;
  if (p->rx_got < p->rx_cap) {
    size_t k = n < p->rx_cap - p->rx_got ? n : p->rx_cap - p->rx_got;
    memcpy(p->rx_dst + p->rx_got, pl, k);
  }
  p->rx_got += n;
  if (h->flags & FL_LAST) {
    p->rx_active = 0;
    if (p->rx_req) {
      p->rx_req->nbytes = p->rx_len;
      uc_req_complete(p->rx_req, p->rx_len > p->rx_req->len ? UCOMM_ERR_TRUNC : UCOMM_OK);
    } else {
      uc_unexp_done(c, p->rx_unexp);
    }
    p->rx_req = NULL;
    p->rx_unexp = NULL;
  }
}

static void on_packet(ucomm_t *c, udp_be *u, const uint8_t *pkt, size_t n) {
  if (n < HDR_SZ) return;
  pk_hdr h;
  hdr_unpack(pkt, &h);
  if (h.session != c->session || h.src >= c->world || h.src == c->rank) return;
  int pi = h.src;
  upeer *p = &u->peer[pi];
  on_ack(c, u, pi, h.ack, h.sack);
  if (h.type != PK_DATA) return;
  p->ack_pending++;
  if (seq_lt(h.seq, p->rnext)) return; /* duplicate */
  if (h.seq - p->rnext >= u->rwnd) return; /* beyond window: sender misbehaving, drop */
  if (h.seq != p->rnext) {
    uint32_t slot = h.seq % u->rwnd;
    if (!p->ooo[slot]) {
      p->ooo[slot] = malloc(n);
      if (!p->ooo[slot]) return;
      memcpy(p->ooo[slot], pkt, n);
      p->ooo_len[slot] = (uint32_t)n;
    }
    p->ack_pending += 64; /* hole: ack immediately so the sender can repair */
    return;
  }
  deliver(c, u, pi, &h, pkt + HDR_SZ, n - HDR_SZ);
  p->rnext++;
  for (;;) { /* drain now-in-order fragments */
    uint32_t slot = p->rnext % u->rwnd;
    uint8_t *q = p->ooo[slot];
    if (!q) break;
    pk_hdr qh;
    hdr_unpack(q, &qh);
    p->ooo[slot] = NULL;
    if (qh.seq == p->rnext) {
      deliver(c, u, pi, &qh, q + HDR_SZ, p->ooo_len[slot] - HDR_SZ);
      p->rnext++;
    }
    free(q);
  }
}

/* Move queued sends into the window as far as cwnd/rwnd allow. Returns 1 if the socket would block. */
static int pump(ucomm_t *c, udp_be *u, int pi) {
  upeer *p = &u->peer[pi];
  uint32_t lim = (uint32_t)p->cwnd;
  if (lim > p->rwnd) lim = p->rwnd;
  if (lim < 1) lim = 1;
  while (p->sq && p->nxt - p->una < lim) {
    ucomm_req_t *r = p->sq;
    sfrag *f = &p->win[p->nxt % WIN_MAX];
    size_t room = (size_t)u->mtu - (r->off == 0 && !r->sreq ? FIRST_SZ : 0);
    size_t left = r->len - r->off;
    f->req = r;
    f->off = r->off;
    f->len = (uint32_t)(left < room ? left : room);
    f->flags = (uint8_t)((!r->sreq ? FL_FIRST : 0) | (f->len == left ? FL_LAST : 0));
    f->acked = 0;
    f->nsent = 0;
    int rc = send_frag(c, u, pi, p->nxt);
    if (rc < 0) return -1;
    if (rc > 0) {
      f->req = NULL;
      return 1;
    }
    r->sreq = 1; /* marks "first fragment already emitted" */
    r->off += f->len;
    r->pending++;
    p->nxt++;
    if (r->off == r->len) {
      p->sq = r->next;
      if (!p->sq) p->sq_tail = NULL;
      r->next = NULL;
    }
  }
  return 0;
}

static void check_rto(ucomm_t *c, udp_be *u, int pi, uint64_t now) {
  upeer *p = &u->peer[pi];
  if (p->una == p->nxt) return;
  sfrag *f0 = &p->win[p->una % WIN_MAX];
  if (f0->acked || now - f0->t_sent < p->rto_us) return;
  /* timeout: shrink window and resend every expired unacked fragment */
  p->cwnd = p->cwnd / 2 < 8 ? 8 : p->cwnd / 2;
  p->rto_us = p->rto_us * 2 > 200000 ? 200000 : p->rto_us * 2;
  for (uint32_t s = p->una; seq_lt(s, p->nxt); s++) {
    sfrag *f = &p->win[s % WIN_MAX];
    if (f->acked || !f->req || now - f->t_sent < p->rto_us / 2) continue;
    if (send_frag(c, u, pi, s) != 0) break;
  }
}

static ucomm_status udp_isend(ucomm_t *c, ucomm_req_t *r) {
  udp_be *u = c->be;
  upeer *p = &u->peer[r->peer];
  if (r->len <= EAGER_MAX) {
    ucomm_req_t *cp = malloc(sizeof *cp + (r->len ? r->len : 1));
    if (!cp) return UCOMM_ERR_NOMEM;
    *cp = *r;
    cp->kind = UC_REQ_INTERNAL;
    cp->buf = cp + 1;
    if (r->len) memcpy(cp->buf, r->buf, r->len);
    uc_req_complete(r, UCOMM_OK);
    r = cp;
  }
  r->next = NULL;
  r->off = 0;
  r->sreq = 0;
  r->pending = 0;
  if (p->sq_tail)
    p->sq_tail->next = r;
  else
    p->sq = r;
  p->sq_tail = r;
  if (pump(c, u, r->peer) < 0) return UCOMM_ERR_IO;
  return UCOMM_OK;
}

static ucomm_status udp_progress(ucomm_t *c, int timeout_ms) {
  udp_be *u = c->be;
  struct mmsghdr mm[BATCH];
  struct iovec iov[BATCH];
  int blocked = 0;
  uint64_t spin_until = 0;
  int got = 0;
  for (int round = 0;; round++) {
    for (int i = 0; i < BATCH; i++) {
      iov[i].iov_base = u->rxbuf + (size_t)i * u->rxslot;
      iov[i].iov_len = u->rxslot;
      memset(&mm[i].msg_hdr, 0, sizeof mm[i].msg_hdr);
      mm[i].msg_hdr.msg_iov = &iov[i];
      mm[i].msg_hdr.msg_iovlen = 1;
    }
    int n = recvmmsg(u->fd, mm, BATCH, MSG_DONTWAIT, NULL);
    for (int i = 0; i < n; i++) on_packet(c, u, iov[i].iov_base, mm[i].msg_len);
    if (n > 0) got = 1;

    uint64_t now = uc_now_us();
    int busy = 0;
    blocked = 0;
    for (int pi = 0; pi < c->world; pi++) {
      if (pi == c->rank) continue;
      check_rto(c, u, pi, now);
      int rc = pump(c, u, pi);
      if (rc < 0) return UCOMM_ERR_IO;
      blocked |= rc;
      if (u->peer[pi].ack_pending) send_ack(c, u, pi);
      if (u->peer[pi].sq || u->peer[pi].una != u->peer[pi].nxt) busy = 1;
    }
    if (n > 0) continue; /* keep draining while packets flow */
    if (timeout_ms <= 0 || got) break; /* traffic handled: let the caller re-check its request */
    if (u->spin_us > 0) { /* busy-poll first: a sleeping receiver adds a scheduler wakeup to every round trip */
      if (!spin_until) spin_until = now + (uint64_t)u->spin_us;
      if (now < spin_until) continue;
    }
    if (round > 0 && !u->spin_us) break;
    /* nothing arrived: sleep until traffic, writability or the next retransmit deadline */
    int wait = timeout_ms;
    if (busy) {
      int rto_ms = 1;
      if (wait > rto_ms) wait = rto_ms;
    }
    struct pollfd pf = {u->fd, (short)(POLLIN | (blocked ? POLLOUT : 0)), 0};
    if (poll(&pf, 1, wait) <= 0) break;
  }
  return UCOMM_OK;
}

static int udp_idle(ucomm_t *c) {
  udp_be *u = c->be;
  for (int pi = 0; pi < c->world; pi++)
    if (u->peer[pi].sq || u->peer[pi].una != u->peer[pi].nxt) return 0;
  return 1;
}

static void udp_finalize(ucomm_t *c) {
  udp_be *u = c->be;
  if (!u) return;
  if (c->cfg.verbose) UC_LOG(c, "udp: %llu datagrams sent, %llu retransmitted", (unsigned long long)u->sent,
                             (unsigned long long)u->retrans);
  for (int pi = 0; pi < c->world; pi++) {
    upeer *p = &u->peer[pi];
    if (p->ooo)
      for (uint32_t i = 0; i < u->rwnd; i++) free(p->ooo[i]);
    free(p->ooo);
    free(p->ooo_len);
    free(p->win);
  }
  if (u->fd >= 0) close(u->fd);
  free(u->rxbuf);
  free(u);
  c->be = NULL;
}

const uc_ops uc_udp_ops = {"udp", udp_isend, NULL, udp_progress, NULL, NULL, udp_idle, udp_finalize};

ucomm_status uc_udp_init(ucomm_t *c) {
  udp_be *u = calloc(1, sizeof *u);
  if (!u) return UCOMM_ERR_NOMEM;
  c->be = u;
  u->mtu = c->cfg.udp_mtu;
  if (u->mtu < 256) u->mtu = 256;
  if (u->mtu > 65000) u->mtu = 65000;
  const char *d = getenv("UCOMM_UDP_DROP");
  u->drop = d ? atof(d) : 0;
  const char *sp = getenv("UCOMM_SPIN_US");
  u->spin_us = sp ? atoi(sp) : 0;
  u->rng = 0x9e3779b97f4a7c15ull ^ (uint64_t)c->rank * 0x100000001b3ull ^ uc_now_us();
  u->fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (u->fd < 0) goto fail;
  int sz = 32 << 20;
  if (setsockopt(u->fd, SOL_SOCKET, SO_RCVBUFFORCE, &sz, sizeof sz) != 0)
    setsockopt(u->fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
  if (setsockopt(u->fd, SOL_SOCKET, SO_SNDBUFFORCE, &sz, sizeof sz) != 0)
    setsockopt(u->fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
  int rcv = 0;
  socklen_t rl = sizeof rcv;
  getsockopt(u->fd, SOL_SOCKET, SO_RCVBUF, &rcv, &rl);
  /* Window sized so all peers bursting at once still fit the kernel buffer (skb overhead ~ 1 KiB). */
  long per = (long)rcv / (u->mtu + HDR_SZ + 1024) / (c->world > 1 ? c->world - 1 : 1);
  u->rwnd = per < 16 ? 16 : per > WIN_MAX ? WIN_MAX : (uint32_t)per;

  struct sockaddr_in sa;
  memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = c->boot.local_ip;
  if (bind(u->fd, (struct sockaddr *)&sa, sizeof sa) != 0) goto fail;
  socklen_t sl = sizeof sa;
  getsockname(u->fd, (struct sockaddr *)&sa, &sl);

  struct {
    uint32_t ip;
    uint16_t port, pad;
    uint32_t rwnd, mtu;
  } mine = {sa.sin_addr.s_addr, sa.sin_port, 0, u->rwnd, (uint32_t)u->mtu}, all[UCOMM_MAX_WORLD];
  if (uc_boot_allgather(c, &mine, all, sizeof mine) != UCOMM_OK) goto fail;

  u->rxslot = HDR_SZ + FIRST_SZ + (size_t)u->mtu;
  for (int r = 0; r < c->world; r++) {
    if ((int)all[r].mtu > u->mtu + 0 && (size_t)all[r].mtu + HDR_SZ + FIRST_SZ > u->rxslot)
      u->rxslot = (size_t)all[r].mtu + HDR_SZ + FIRST_SZ; /* peers may use a larger MTU */
  }
  u->rxbuf = malloc(u->rxslot * BATCH);
  if (!u->rxbuf) goto fail;
  for (int r = 0; r < c->world; r++) {
    upeer *p = &u->peer[r];
    p->addr.sin_family = AF_INET;
    p->addr.sin_addr.s_addr = all[r].ip;
    p->addr.sin_port = all[r].port;
    p->rwnd = all[r].rwnd;
    p->cwnd = 32;
    p->rto_us = 20000;
    if (r == c->rank) continue;
    p->win = calloc(WIN_MAX, sizeof *p->win);
    p->ooo = calloc(u->rwnd, sizeof *p->ooo);
    p->ooo_len = calloc(u->rwnd, sizeof *p->ooo_len);
    if (!p->win || !p->ooo || !p->ooo_len) goto fail;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &p->addr.sin_addr, ip, sizeof ip);
    UC_LOG(c, "udp peer %d at %s:%u, window %u x %u B", r, ip, ntohs(p->addr.sin_port), p->rwnd, all[r].mtu);
  }
  return UCOMM_OK;
fail:
  udp_finalize(c);
  return UCOMM_ERR_IO;
}
