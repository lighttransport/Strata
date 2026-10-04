#ifndef UCOMM_INTERNAL_H_
#define UCOMM_INTERNAL_H_

#include "ucomm.h"

#include <stdio.h>

enum { UC_REQ_SEND = 1, UC_REQ_RECV = 2, UC_REQ_INTERNAL = 3 /* backend-owned copy of a small send */ };

struct ucomm_req {
  ucomm_t *c;
  struct ucomm_req *next; /* posted-recv list / backend send queue */
  int kind;
  int peer;
  uint32_t tag;
  void *buf;
  size_t len;    /* send: message size; recv: capacity */
  size_t nbytes; /* recv: message size */
  int done;
  ucomm_status status;
  /* backend scratch */
  size_t off;
  uint32_t pending;
  void *bmr;
  uint64_t raddr;
  uint32_t rkey;
  uint64_t sreq;
};

/* A message that arrived (or started arriving) before a matching receive was posted. */
typedef struct uc_unexp {
  struct uc_unexp *next;
  int peer;
  uint32_t tag;
  size_t len;
  void *data;          /* eager / udp: malloc'd copy */
  int complete;        /* all bytes are in data */
  ucomm_req_t *waiter; /* receive matched while data was still arriving */
  int rndv;            /* ib: RTS, data still at the sender */
  uint64_t raddr, sreq;
  uint32_t rkey;
} uc_unexp;

typedef struct uc_ops {
  const char *name;
  ucomm_status (*isend)(ucomm_t *c, ucomm_req_t *r);
  /* Called when a receive matches an unexpected rendezvous message (ib only). */
  ucomm_status (*rndv_recv)(ucomm_t *c, ucomm_req_t *r, uc_unexp *u);
  /* Make progress; may block up to timeout_ms waiting for traffic (0: just poll). */
  ucomm_status (*progress)(ucomm_t *c, int timeout_ms);
  ucomm_status (*mr_reg)(ucomm_t *c, ucomm_mr_t *mr);
  void (*mr_dereg)(ucomm_t *c, ucomm_mr_t *mr);
  int (*idle)(ucomm_t *c); /* no outgoing data in flight */
  void (*finalize)(ucomm_t *c);
} uc_ops;

struct ucomm_mr {
  ucomm_t *c;
  struct ucomm_mr *next;
  void *ptr;
  size_t len;
  void *bmr;
};

typedef struct {
  int fd[UCOMM_MAX_WORLD]; /* rank 0: one per peer; others: fd[0] */
  uint32_t local_ip;       /* network order, address used to reach the root */
} uc_boot;

struct ucomm {
  ucomm_config cfg;
  int rank, world;
  const uc_ops *ops;
  void *be;
  uc_boot boot;
  uint32_t session;
  ucomm_req_t *posted[UCOMM_MAX_WORLD]; /* FIFO of posted receives per peer */
  ucomm_req_t *posted_tail[UCOMM_MAX_WORLD];
  uc_unexp *unexp[UCOMM_MAX_WORLD]; /* FIFO of unexpected messages per peer */
  uc_unexp *unexp_tail[UCOMM_MAX_WORLD];
  ucomm_mr_t *mrs;
  uint32_t coll_seq;
};

#define UC_LOG(c, ...)                         \
  do {                                         \
    if ((c)->cfg.verbose) uc_log((c)->rank, __VA_ARGS__); \
  } while (0)
void uc_log(int rank, const char *fmt, ...);

uint64_t uc_now_us(void);

/* Matching, used by backends when a message header arrives. */
ucomm_req_t *uc_match_posted(ucomm_t *c, int peer, uint32_t tag);
uc_unexp *uc_unexp_new(ucomm_t *c, int peer, uint32_t tag, size_t len); /* appended to the queue */
void uc_unexp_done(ucomm_t *c, uc_unexp *u); /* all data arrived into u->data */
void uc_req_complete(ucomm_req_t *r, ucomm_status s);

/* Bootstrap over TCP (ucomm_boot.c). */
ucomm_status uc_boot_init(ucomm_t *c);
ucomm_status uc_boot_allgather(ucomm_t *c, const void *mine, void *all, size_t n);
void uc_boot_close(ucomm_t *c);
/* Finalize barrier that keeps calling backend progress while waiting. */
ucomm_status uc_boot_drain_barrier(ucomm_t *c);

/* Backends. probe: 1 when usable on this host. */
extern const uc_ops uc_udp_ops;
ucomm_status uc_udp_init(ucomm_t *c);
extern const uc_ops uc_ib_ops;
int uc_ib_probe(ucomm_t *c);
ucomm_status uc_ib_init(ucomm_t *c);

#endif
