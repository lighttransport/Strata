/* Variable-length and zero-byte messages, order across slot reuse, API bounds. */
#define _POSIX_C_SOURCE 200809L
#include "ucomm_relay.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "relay API check failed at line %d\n", __LINE__); exit(1); } } while (0)
#define CAP 8193
#define COUNT 37
static size_t length(int i) {
  const size_t sizes[] = {0, 1, 8152, 8153, CAP, 7};
  return sizes[i % 6];
}
static ucomm_t *init(int rank, const char *root) {
  ucomm_config cfg; ucomm_config_default(&cfg);
  cfg.backend = UCOMM_BACKEND_UDP; cfg.rank = rank; cfg.world = 2;
  cfg.root_addr = root; cfg.timeout_ms = 10000;
  ucomm_t *c; CHECK(ucomm_init(&cfg, &c) == UCOMM_OK); return c;
}
static void endpoint(int source, const char *root) {
  ucomm_t *c = init(1, root);
  unsigned char buf[4][CAP]; ucomm_req_t *req[4] = {0};
  for (int i = 0; i < COUNT; i++) {
    int s = i % 4; size_t len = length(i);
    if (source) {
      if (req[s]) CHECK(ucomm_wait(req[s], NULL) == UCOMM_OK);
      memset(buf[s], i + 1, len);
      CHECK(ucomm_isend(c, 0, 30, buf[s], len, &req[s]) == UCOMM_OK);
    } else {
      size_t got = SIZE_MAX;
      CHECK(ucomm_recv(c, 0, 31, buf[s], CAP, &got) == UCOMM_OK);
      CHECK(got == len);
      for (size_t k = 0; k < got; k++) CHECK(buf[s][k] == (unsigned char)(i + 1));
    }
  }
  if (source) {
    CHECK(ucomm_waitall(4, req) == UCOMM_OK);
    CHECK(ucomm_send(c, 0, 40, buf[0], CAP) == UCOMM_OK);
  }
  ucomm_finalize(c);
  exit(0);
}
int main(void) {
  char left[64], right[64];
  int port = 32000 + (getpid() % 10000) * 2;
  snprintf(left, sizeof left, "127.0.0.1:%d", port);
  snprintf(right, sizeof right, "127.0.0.1:%d", port + 1);
  pid_t l = fork(); CHECK(l >= 0); if (!l) endpoint(1, left);
  pid_t r = fork(); CHECK(r >= 0); if (!r) endpoint(0, right);
  ucomm_t *a = init(0, left), *b = init(0, right);
  ucomm_relay_t *relay = NULL;
  CHECK(ucomm_relay_create(a, 1, 30, a, 1, 31, CAP, 3, COUNT, &relay) == UCOMM_ERR_ARG);
  CHECK(ucomm_relay_create(a, 1, 30, b, 1, 31, CAP, 65, COUNT, &relay) == UCOMM_ERR_ARG);
  CHECK(ucomm_relay_create(a, 1, UCOMM_TAG_RESERVED, b, 1, 31, CAP, 3, COUNT, &relay) == UCOMM_ERR_ARG);
  CHECK(ucomm_relay_create(a, 0, 30, b, 1, 31, CAP, 3, COUNT, &relay) == UCOMM_ERR_ARG);
  CHECK(ucomm_relay_create(a, 1, 32, b, 1, 33, CAP, 1, 0, &relay) == UCOMM_OK);
  int done = 0;
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_OK && done);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK);
  CHECK(ucomm_relay_create(a, 1, 30, b, 1, 31, CAP, 3, COUNT, &relay) == UCOMM_OK);
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_OK && !done);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_ERR_ARG);
  time_t start = time(NULL);
  while (!done) {
    CHECK(ucomm_relay_progress(relay, &done) == UCOMM_OK);
    CHECK(time(NULL) - start < 15);
  }
  uint64_t bytes = 0; for (int i = 0; i < COUNT; i++) bytes += length(i);
  CHECK(ucomm_relay_bytes(relay) == bytes);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK);
  CHECK(ucomm_relay_create(a, 1, 40, b, 1, 41, 1, 1, 1, &relay) == UCOMM_OK);
  ucomm_status trunc = UCOMM_OK;
  start = time(NULL);
  while (trunc == UCOMM_OK) {
    trunc = ucomm_relay_progress(relay, &done);
    CHECK(time(NULL) - start < 15);
  }
  CHECK(trunc == UCOMM_ERR_TRUNC && !done && ucomm_relay_bytes(relay) == 0);
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_ERR_TRUNC && !done);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK);
  ucomm_finalize(a); ucomm_finalize(b);
  int status;
  CHECK(waitpid(l, &status, 0) == l && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(waitpid(r, &status, 0) == r && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  puts("relay API: PASS");
  return 0;
}
