/* ucomm_bench: p2p latency / bandwidth between rank 0 and 1, and allreduce over all ranks.
 *   ucomm_bench -r RANK -w WORLD -a ROOT_HOST:PORT [-b auto|ib|udp] [-n ITERS]
 * Run one process per node with the same -w and -a. */
#define _GNU_SOURCE
#include "ucomm.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define OK(x)                                                                    \
  do {                                                                           \
    ucomm_status s_ = (x);                                                       \
    if (s_ != UCOMM_OK) {                                                        \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, ucomm_strerror(s_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

int main(int argc, char **argv) {
  ucomm_config cfg;
  ucomm_config_default(&cfg);
  int iters = 1000, opt;
  while ((opt = getopt(argc, argv, "r:w:a:b:n:v")) != -1) {
    switch (opt) {
      case 'r': cfg.rank = atoi(optarg); break;
      case 'w': cfg.world = atoi(optarg); break;
      case 'a': cfg.root_addr = optarg; break;
      case 'b':
        cfg.backend = !strcmp(optarg, "ib") ? UCOMM_BACKEND_IB
                      : !strcmp(optarg, "udp") ? UCOMM_BACKEND_UDP
                                               : UCOMM_BACKEND_AUTO;
        break;
      case 'n': iters = atoi(optarg); break;
      case 'v': cfg.verbose = 1; break;
      default:
        fprintf(stderr, "usage: %s -r RANK -w WORLD -a HOST:PORT [-b auto|ib|udp] [-n ITERS] [-v]\n", argv[0]);
        return 2;
    }
  }
  ucomm_t *c;
  OK(ucomm_init(&cfg, &c));
  int me = ucomm_rank(c), W = ucomm_world(c);
  if (me == 0) printf("ucomm_bench: world %d, backend %s\n", W, ucomm_backend_name(c));

  size_t maxsz = 64u << 20;
  char *buf = malloc(maxsz);
  memset(buf, me, maxsz);
  ucomm_mr_t *mr;
  OK(ucomm_mr_reg(c, buf, maxsz, &mr));

  if (W >= 2 && me < 2) {
    int peer = 1 - me;
    /* latency: ping-pong */
    size_t lat_sizes[] = {8, 1024, 8192};
    for (size_t k = 0; k < sizeof lat_sizes / sizeof lat_sizes[0]; k++) {
      size_t sz = lat_sizes[k];
      for (int warm = 0; warm < 2; warm++) {
        int n = warm ? iters : iters / 10 + 1;
        double t0 = now_s();
        for (int i = 0; i < n; i++) {
          if (me == 0) {
            OK(ucomm_send(c, peer, 1, buf, sz));
            OK(ucomm_recv(c, peer, 1, buf, sz, NULL));
          } else {
            OK(ucomm_recv(c, peer, 1, buf, sz, NULL));
            OK(ucomm_send(c, peer, 1, buf, sz));
          }
        }
        double t = now_s() - t0;
        if (warm && me == 0) printf("latency %7zu B : %8.1f us (half round trip)\n", sz, t / n / 2 * 1e6);
      }
    }
    /* bandwidth: rank 0 streams, a window of 4 messages in flight */
    for (size_t sz = 1024; sz <= maxsz; sz *= 4) {
      int n = (int)((256u << 20) / sz);
      if (n < 4) n = 4;
      if (n > 2000) n = 2000;
      ucomm_req_t *rq[4] = {0};
      double t0 = now_s();
      for (int i = 0; i < n; i++) {
        int s = i % 4;
        if (rq[s]) OK(ucomm_wait(rq[s], NULL));
        rq[s] = NULL;
        if (me == 0)
          OK(ucomm_isend(c, peer, 2, buf, sz, &rq[s]));
        else
          OK(ucomm_irecv(c, peer, 2, buf + 0, sz, &rq[s]));
      }
      OK(ucomm_waitall(4, rq));
      char ack = 0;
      if (me == 1)
        OK(ucomm_send(c, peer, 3, &ack, 1));
      else
        OK(ucomm_recv(c, peer, 3, &ack, 1, NULL));
      double t = now_s() - t0;
      if (me == 0) printf("bandwidth %9zu B : %8.1f MB/s\n", sz, (double)sz * n / t / 1e6);
    }
  }
  OK(ucomm_barrier(c));

  /* allreduce of a 16 MiB f32 buffer (a mid-size activation) and a 16 KiB one (one decode token's hidden) */
  size_t counts[] = {4096, 4u << 20};
  for (size_t k = 0; k < 2; k++) {
    float *f = (float *)buf;
    for (size_t i = 0; i < counts[k]; i++) f[i] = 1.0f;
    OK(ucomm_allreduce_sum(c, f, counts[k], UCOMM_F32));
    int n = k == 0 ? iters : 10;
    OK(ucomm_barrier(c));
    double t0 = now_s();
    for (int i = 0; i < n; i++) OK(ucomm_allreduce_sum(c, f, counts[k], UCOMM_F32));
    double t = (now_s() - t0) / n;
    if (me == 0)
      printf("allreduce f32 %8zu B x %d ranks: %9.1f us  (%.1f MB/s algbw)\n", counts[k] * 4, W, t * 1e6,
             (double)counts[k] * 4 / t / 1e6);
  }
  ucomm_mr_dereg(mr);
  ucomm_finalize(c);
  free(buf);
  return 0;
}
