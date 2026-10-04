/* Runs N ranks as processes on this host (127.0.0.1) and checks p2p, matching and collectives.
 *   test_loopback [world=2] [port]
 * Env as for ucomm_config_default (e.g. UCOMM_BACKEND, UCOMM_UDP_MTU, UCOMM_UDP_DROP=0.05). */
#define _GNU_SOURCE
#include "ucomm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails;
#define CHECK(cond, ...)                              \
  do {                                                \
    if (!(cond)) {                                    \
      fprintf(stderr, "rank %d FAIL %s:%d: ", me, __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                   \
      fputc('\n', stderr);                            \
      fails++;                                        \
    }                                                 \
  } while (0)

static uint8_t pat(int from, size_t sz, size_t i) { return (uint8_t)(from * 31 + sz * 7 + i * 13 + (i >> 9)); }

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16); /* exact for the small integers used here */
}
static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static int run_rank(int me, int world, const char *root) {
  ucomm_config cfg;
  ucomm_config_default(&cfg);
  cfg.rank = me;
  cfg.world = world;
  cfg.root_addr = root;
  cfg.timeout_ms = 120000;
  ucomm_t *c;
  ucomm_status s = ucomm_init(&cfg, &c);
  if (s != UCOMM_OK) {
    fprintf(stderr, "rank %d init: %s\n", me, ucomm_strerror(s));
    return 1;
  }
  if (me == 0) printf("world %d backend %s\n", world, ucomm_backend_name(c));
  int right = (me + 1) % world, left = (me - 1 + world) % world;

  /* 1. ring p2p over sizes that straddle fragment / eager boundaries */
  size_t sizes[] = {0, 1, 100, 1383, 1384, 1385, 1400, 8151, 8152, 8153, 100000, 1 << 20, 16 << 20, 64 << 20};
  for (size_t k = 0; k < sizeof sizes / sizeof sizes[0]; k++) {
    size_t sz = sizes[k];
    uint8_t *sb = malloc(sz + 1), *rb = malloc(sz + 1);
    for (size_t i = 0; i < sz; i++) sb[i] = pat(me, sz, i);
    memset(rb, 0xee, sz + 1);
    ucomm_req_t *rq[2];
    ucomm_irecv(c, left, 7, rb, sz, &rq[0]);
    ucomm_isend(c, right, 7, sb, sz, &rq[1]);
    size_t got = 12345;
    s = ucomm_wait(rq[0], &got);
    CHECK(s == UCOMM_OK && got == sz, "recv %zu: %s got %zu", sz, ucomm_strerror(s), got);
    CHECK(ucomm_wait(rq[1], NULL) == UCOMM_OK, "send %zu", sz);
    size_t bad = sz;
    for (size_t i = 0; i < sz; i++)
      if (rb[i] != pat(left, sz, i)) {
        bad = i;
        break;
      }
    CHECK(bad == sz, "data mismatch size %zu at %zu", sz, bad);
    free(sb);
    free(rb);
  }

  /* 2. unexpected messages and tag matching out of order (eager and large) */
  {
    size_t big = 3 << 20;
    char a[64], b2[64];
    uint8_t *bs = malloc(big), *br = malloc(big);
    for (size_t i = 0; i < big; i++) bs[i] = pat(me, big, i);
    snprintf(a, sizeof a, "tag1 from %d", me);
    snprintf(b2, sizeof b2, "tag2 from %d", me);
    ucomm_req_t *sq[3];
    ucomm_isend(c, right, 1, a, strlen(a) + 1, &sq[0]);
    ucomm_isend(c, right, 3, bs, big, &sq[1]);
    ucomm_isend(c, right, 2, b2, strlen(b2) + 1, &sq[2]);
    ucomm_barrier(c); /* messages now sit in the unexpected queue (or are in flight) */
    char r1[64], r2[64], ex1[64], ex2[64];
    size_t n;
    CHECK(ucomm_recv(c, left, 2, r2, sizeof r2, &n) == UCOMM_OK, "recv tag2");
    CHECK(ucomm_recv(c, left, 3, br, big, &n) == UCOMM_OK && n == big, "recv tag3");
    CHECK(ucomm_recv(c, left, 1, r1, sizeof r1, &n) == UCOMM_OK, "recv tag1");
    snprintf(ex1, sizeof ex1, "tag1 from %d", left);
    snprintf(ex2, sizeof ex2, "tag2 from %d", left);
    CHECK(!strcmp(r1, ex1) && !strcmp(r2, ex2), "tag payloads '%s' '%s'", r1, r2);
    int ok = 1;
    for (size_t i = 0; i < big; i++) ok &= br[i] == pat(left, big, i);
    CHECK(ok, "unexpected large payload");
    CHECK(ucomm_waitall(3, sq) == UCOMM_OK, "sends");
    free(bs);
    free(br);
  }

  /* 3. truncation reports an error but still consumes the message */
  {
    char big[200], small[50];
    memset(big, 'x', sizeof big);
    ucomm_req_t *sr;
    ucomm_isend(c, right, 9, big, sizeof big, &sr);
    size_t n = 0;
    s = ucomm_recv(c, left, 9, small, sizeof small, &n);
    CHECK(s == UCOMM_ERR_TRUNC && n == sizeof big, "trunc: %s %zu", ucomm_strerror(s), n);
    ucomm_wait(sr, NULL);
  }

  /* 4. collectives */
  {
    size_t cnt = 1000003;
    float *f = malloc(cnt * 4);
    int32_t *iv = malloc(cnt * 4);
    uint16_t *bf = malloc(cnt * 2);
    for (size_t i = 0; i < cnt; i++) {
      f[i] = (float)(me + (int)(i % 7));
      iv[i] = me * 1000 + (int32_t)i;
      bf[i] = f2bf((float)(me + (int)(i % 3)));
    }
    CHECK(ucomm_allreduce_sum(c, f, cnt, UCOMM_F32) == UCOMM_OK, "allreduce f32");
    CHECK(ucomm_allreduce_sum(c, iv, cnt, UCOMM_I32) == UCOMM_OK, "allreduce i32");
    CHECK(ucomm_allreduce_sum(c, bf, cnt, UCOMM_BF16) == UCOMM_OK, "allreduce bf16");
    int rs = world * (world - 1) / 2, ef = 0, ei = 0, eb = 0;
    for (size_t i = 0; i < cnt; i++) {
      ef += f[i] != (float)(rs + world * (int)(i % 7));
      ei += iv[i] != 1000 * rs + world * (int32_t)i;
      eb += bf2f(bf[i]) != (float)(rs + world * (int)(i % 3));
    }
    CHECK(!ef && !ei && !eb, "allreduce mismatches f32 %d i32 %d bf16 %d", ef, ei, eb);
    free(f);
    free(iv);
    free(bf);

    int mine = me * 11 + 1, all[UCOMM_MAX_WORLD];
    CHECK(ucomm_allgather(c, &mine, all, sizeof mine) == UCOMM_OK, "allgather");
    for (int r = 0; r < world; r++) CHECK(all[r] == r * 11 + 1, "allgather[%d]=%d", r, all[r]);

    size_t bl = 5 << 20;
    uint8_t *bb = malloc(bl);
    for (size_t i = 0; i < bl; i++) bb[i] = me == world - 1 ? pat(99, bl, i) : 0;
    CHECK(ucomm_bcast(c, bb, bl, world - 1) == UCOMM_OK, "bcast");
    int ok = 1;
    for (size_t i = 0; i < bl; i++) ok &= bb[i] == pat(99, bl, i);
    CHECK(ok, "bcast payload");
    free(bb);
    for (int i = 0; i < 100; i++) CHECK(ucomm_barrier(c) == UCOMM_OK, "barrier");
  }

  /* 5. self send */
  {
    int x = 42, y = 0;
    ucomm_req_t *sr;
    ucomm_isend(c, me, 5, &x, sizeof x, &sr);
    CHECK(ucomm_recv(c, me, 5, &y, sizeof y, NULL) == UCOMM_OK && y == 42, "self");
    ucomm_wait(sr, NULL);
  }

  ucomm_finalize(c);
  if (!fails) printf("rank %d ok\n", me);
  return fails ? 1 : 0;
}

int main(int argc, char **argv) {
  int world = argc > 1 ? atoi(argv[1]) : 2;
  int port = argc > 2 ? atoi(argv[2]) : 30000 + (getpid() % 20000);
  char root[64];
  snprintf(root, sizeof root, "127.0.0.1:%d", port);
  fflush(stdout);
  pid_t pids[UCOMM_MAX_WORLD];
  for (int r = 0; r < world; r++) {
    pids[r] = fork();
    if (pids[r] == 0) _exit(run_rank(r, world, root));
  }
  int rc = 0;
  for (int r = 0; r < world; r++) {
    int st = 0;
    waitpid(pids[r], &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st)) rc = 1;
  }
  printf("%s (world %d)\n", rc ? "FAILED" : "PASSED", world);
  return rc;
}
