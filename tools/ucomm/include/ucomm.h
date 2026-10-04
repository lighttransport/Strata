/*
 * ucomm - small inter-node communication library for LLM inference on 2..8 local nodes.
 *
 * Backends:
 *   ib  : InfiniBand / RoCE verbs (ConnectX-3 era, mlx4 and newer). libibverbs.so.1 is loaded at run time
 *         with dlopen/dlsym, so no verbs headers or libraries are needed to build.
 *   udp : reliable UDP (sliding window, SACK, retransmit) for machines without an HCA.
 *
 * Model:
 *   - One context per process, one rank per process. All ranks call ucomm_init with the same world size and
 *     the same root address; rank 0 listens there for a short TCP bootstrap.
 *   - Two-sided tagged messages. Messages between one pair of ranks with the same tag are matched in order.
 *     User tags must be below UCOMM_TAG_RESERVED.
 *   - Nonblocking calls return a request. ucomm_wait / a successful ucomm_test (done=1) release it.
 *     Send buffers must stay untouched, and receive buffers alive, until the request completes.
 *   - A context is not thread safe; drive it from one thread. Progress happens inside ucomm_* calls.
 */
#ifndef UCOMM_H_
#define UCOMM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UCOMM_MAX_WORLD 8
#define UCOMM_TAG_RESERVED 0x80000000u

typedef struct ucomm ucomm_t;
typedef struct ucomm_req ucomm_req_t;
typedef struct ucomm_mr ucomm_mr_t;

typedef enum { UCOMM_BACKEND_AUTO = 0, UCOMM_BACKEND_IB, UCOMM_BACKEND_UDP } ucomm_backend;

typedef enum {
  UCOMM_OK = 0,
  UCOMM_ERR_NOBACKEND, /* requested backend not available on every rank */
  UCOMM_ERR_IO,        /* socket / verbs failure */
  UCOMM_ERR_TIMEOUT,   /* wait timed out; the request stays valid */
  UCOMM_ERR_ARG,
  UCOMM_ERR_NOMEM,
  UCOMM_ERR_TRUNC      /* incoming message larger than the receive buffer */
} ucomm_status;

typedef enum { UCOMM_F32 = 0, UCOMM_F16, UCOMM_BF16, UCOMM_I32 } ucomm_dtype;

typedef struct {
  int rank;              /* 0..world-1 */
  int world;             /* 1..UCOMM_MAX_WORLD */
  const char *root_addr; /* "host:port" of rank 0, e.g. "192.168.100.22:29500" */
  ucomm_backend backend; /* AUTO picks ib when every rank has an active HCA port, else udp */
  const char *ib_dev;    /* NULL: first device with an active port */
  int ib_port;           /* 0: default 1 */
  int ib_gid_index;      /* RoCE GID index (ignored on native IB) */
  int udp_mtu;           /* UDP payload bytes per datagram; 0: 1400 */
  int timeout_ms;        /* bootstrap and wait timeout; <=0: wait forever */
  int verbose;           /* log backend choice etc. to stderr */
} ucomm_config;

/* Fills defaults, then applies env: UCOMM_RANK, UCOMM_WORLD, UCOMM_ROOT, UCOMM_BACKEND (auto|ib|udp),
 * UCOMM_IB_DEV, UCOMM_IB_PORT, UCOMM_IB_GID_INDEX, UCOMM_UDP_MTU, UCOMM_TIMEOUT_MS, UCOMM_VERBOSE. */
void ucomm_config_default(ucomm_config *cfg);

ucomm_status ucomm_init(const ucomm_config *cfg, ucomm_t **out);
/* Waits until every rank calls finalize, so in-flight traffic is drained. */
void ucomm_finalize(ucomm_t *c);

int ucomm_rank(const ucomm_t *c);
int ucomm_world(const ucomm_t *c);
const char *ucomm_backend_name(const ucomm_t *c);
const char *ucomm_strerror(ucomm_status s);

/* Pre-register a buffer that will be used often (IB: avoids registration on the fly; udp: no-op). */
ucomm_status ucomm_mr_reg(ucomm_t *c, void *ptr, size_t len, ucomm_mr_t **mr);
void ucomm_mr_dereg(ucomm_mr_t *mr);

/* Point to point. peer may equal own rank. */
ucomm_status ucomm_isend(ucomm_t *c, int peer, uint32_t tag, const void *buf, size_t len, ucomm_req_t **req);
ucomm_status ucomm_irecv(ucomm_t *c, int peer, uint32_t tag, void *buf, size_t cap, ucomm_req_t **req);
/* nbytes (optional) receives the message size for receives. Releases req on completion. */
ucomm_status ucomm_test(ucomm_req_t *req, int *done, size_t *nbytes);
ucomm_status ucomm_wait(ucomm_req_t *req, size_t *nbytes);
ucomm_status ucomm_waitall(int n, ucomm_req_t **reqs);
ucomm_status ucomm_send(ucomm_t *c, int peer, uint32_t tag, const void *buf, size_t len);
ucomm_status ucomm_recv(ucomm_t *c, int peer, uint32_t tag, void *buf, size_t cap, size_t *nbytes);

/* Collectives; every rank must call them in the same order. */
ucomm_status ucomm_barrier(ucomm_t *c);
ucomm_status ucomm_bcast(ucomm_t *c, void *buf, size_t len, int root);
ucomm_status ucomm_allreduce_sum(ucomm_t *c, void *buf, size_t count, ucomm_dtype dt); /* in place */
ucomm_status ucomm_allgather(ucomm_t *c, const void *in, void *out, size_t bytes_per_rank);

#ifdef __cplusplus
}
#endif
#endif
