/* stream_pattern: the decoder's owned-expert stream against ucomm alone. Rank 1 (the TR16 worker) holds a registered region; rank 0 (the B550) posts N irecvs into a
 * registered buffer, sends {hdr, ids}, waits for all, repeats. Prints GB/s over the whole loop.
 *   rank 0: ./stream_pattern -r 0 -a 192.0.2.1:PORT -n ITEMS -k EXPERTS_PER_ITEM -s BYTES [-d DEPTH]
 *   rank 1: same with -r 1 */
#define _GNU_SOURCE
#include "ucomm.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define OK(x) do { ucomm_status s_ = (x); if (s_ != UCOMM_OK) { fprintf(stderr, "%d %s: %s\n", __LINE__, #x, ucomm_strerror(s_)); exit(1); } } while (0)
int main(int argc, char** argv) {
  ucomm_config cfg; ucomm_config_default(&cfg);
  int items = 100, k = 12, depth = 1, opt; size_t sz = 7500000; int thp = 0; double big_gb = 0;
  while ((opt = getopt(argc, argv, "r:a:n:k:s:d:tR:")) != -1) {
    switch (opt) { case 'r': cfg.rank = atoi(optarg); break; case 'a': cfg.root_addr = optarg; break; case 'n': items = atoi(optarg); break;
      case 'k': k = atoi(optarg); break; case 's': sz = (size_t)atol(optarg); break; case 'd': depth = atoi(optarg); break; case 't': thp = 1; break; case 'R': big_gb = atof(optarg); break; }
  }
  cfg.world = 2; cfg.backend = UCOMM_BACKEND_IB; cfg.timeout_ms = 0;
  ucomm_t* c; OK(ucomm_init(&cfg, &c));
  const int slots = 6;                      /* host slots on the receiver */
  size_t region = (size_t)k * sz * (cfg.rank == 0 ? slots : 8);   /* rank 1: 8 items' worth of "rows" */
  if (cfg.rank == 1 && big_gb > 0) region = (size_t)(big_gb * 1e9);
  void* mem = mmap(NULL, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (thp) madvise(mem, region, MADV_HUGEPAGE);
  memset(mem, 1, region);
  ucomm_mr_t* mr; OK(ucomm_mr_reg(c, mem, region, &mr));
  if (cfg.rank == 1) {
    int32_t hdr[4], ids[64];
    for (;;) {
      OK(ucomm_recv(c, 0, 60, hdr, sizeof hdr, NULL));
      if (hdr[0] < 0) break;
      OK(ucomm_recv(c, 0, 61, ids, hdr[1] * 4, NULL));
      for (int i = 0; i < hdr[1]; i++) { static unsigned long long x = 88172645463325252ull; x ^= x << 13; x ^= x >> 7; x ^= x << 17; OK(ucomm_send(c, 0, 62, (char*)mem + (x % ((region - sz) / sz)) * sz, sz)); }
    }
  } else {
    int32_t ids[64]; for (int i = 0; i < 64; i++) ids[i] = i;
    ucomm_req_t* rx[depth][64];
    double t0 = now_s();
    for (int u = 0; u < items; u++) {
      char* base = (char*)mem + (size_t)(u % slots) * k * sz;
      for (int i = 0; i < k; i++) OK(ucomm_irecv(c, 1, 62, base + (size_t)i * sz, sz, &rx[0][i]));
      int32_t hdr[4] = {u % 40, k, 0, 0};
      OK(ucomm_send(c, 1, 60, hdr, sizeof hdr));
      OK(ucomm_send(c, 1, 61, ids, k * 4));
      for (int i = 0; i < k; i++) OK(ucomm_wait(rx[0][i], NULL));
    }
    double t = now_s() - t0;
    printf("items=%d k=%d size=%zu: %.2f GB in %.3f s = %.2f GB/s (%.1f ms per item)\n", items, k, sz, (double)items * k * sz / 1e9, t, (double)items * k * sz / 1e9 / t, t / items * 1e3);
    int32_t end[4] = {-1, 0, 0, 0}; OK(ucomm_send(c, 1, 60, end, sizeof end));
  }
  ucomm_mr_dereg(mr); ucomm_finalize(c);
  return 0;
}
