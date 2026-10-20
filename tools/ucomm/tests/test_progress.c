/* Fault injection for request lifetime and relay errors; no network or hardware. */
#include "ucomm_internal.h"
#include "ucomm_relay.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "progress check failed at line %d\n", __LINE__); exit(1); } } while (0)
static int polls, registrations, deregistrations, fail_registration;
static ucomm_status progress_error(ucomm_t *c, int timeout) {
  (void)c; (void)timeout; polls++; return UCOMM_ERR_IO;
}
static ucomm_status register_mr(ucomm_t *c, ucomm_mr_t *m) {
  (void)c; (void)m;
  registrations++;
  return registrations == fail_registration ? UCOMM_ERR_IO : UCOMM_OK;
}
static void deregister_mr(ucomm_t *c, ucomm_mr_t *m) {
  (void)c; (void)m; deregistrations++;
}
static const uc_ops ops = { .progress = progress_error, .mr_reg = register_mr, .mr_dereg = deregister_mr };
static ucomm_req_t *request(ucomm_t *c) {
  ucomm_req_t *r = calloc(1, sizeof *r); CHECK(r); r->c = c; return r;
}
int main(void) {
  ucomm_t a = { .rank = 0, .world = 2, .ops = &ops };
  ucomm_t b = { .rank = 0, .world = 2, .ops = &ops };
  ucomm_req_t *r = request(&a);
  int done = 123; size_t nbytes = 99;
  CHECK(ucomm_test(r, &done, &nbytes) == UCOMM_ERR_IO && !done && nbytes == 99);
  CHECK(ucomm_wait(r, NULL) == UCOMM_ERR_IO);
  r->done = 1; r->status = UCOMM_OK; r->nbytes = 7;
  CHECK(ucomm_test(r, &done, &nbytes) == UCOMM_OK && done && nbytes == 7);
  ucomm_req_t *req[2] = { request(&a), request(&a) };
  ucomm_req_t *original = req[0], *second = req[1];
  CHECK(ucomm_waitall(2, req) == UCOMM_ERR_IO && req[0] == original && req[1] == second);
  req[0]->done = req[1]->done = 1;
  req[0]->status = UCOMM_ERR_TRUNC;
  CHECK(ucomm_waitall(2, req) == UCOMM_ERR_TRUNC && !req[0] && !req[1]);
  CHECK(ucomm_waitall(-1, NULL) == UCOMM_ERR_ARG);
  CHECK(ucomm_waitall(1, NULL) == UCOMM_ERR_ARG);
  CHECK(ucomm_waitall(0, NULL) == UCOMM_OK);

  ucomm_relay_t *relay;
  fail_registration = 2;
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 8193, 3, 7, &relay) == UCOMM_ERR_IO);
  CHECK(!relay && registrations == 2 && deregistrations == 1 && !a.mrs && !b.mrs);
  registrations = deregistrations = fail_registration = 0;
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 8193, 3, 7, &relay) == UCOMM_OK);
  CHECK(registrations == 2 && a.mrs->len == 8193 * 3 && b.mrs->len == 8193 * 3);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK && deregistrations == 2 && !a.mrs && !b.mrs);
  registrations = deregistrations = 0;
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 8193, 64, 1, &relay) == UCOMM_OK);
  CHECK(registrations == 2 && a.mrs->len == 8193 && b.mrs->len == 8193);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK);
  registrations = 0; polls = 0;
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 8193, 64, 0, &relay) == UCOMM_OK);
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_OK && done && !polls && !registrations);
  CHECK(ucomm_relay_bytes(relay) == 0 && ucomm_relay_destroy(relay) == UCOMM_OK);
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 8193, 3, 7, &relay) == UCOMM_OK);
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_ERR_IO && !done);
  int previous = polls;
  CHECK(ucomm_relay_progress(relay, &done) == UCOMM_ERR_IO && !done && polls == previous);
  CHECK(ucomm_relay_bytes(relay) == 0);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK && !a.mrs && !b.mrs);
  ucomm_relay_t *reverse;
  CHECK(ucomm_relay_create(&a, 1, 1, &b, 1, 2, 64, 1, 1, &relay) == UCOMM_OK);
  CHECK(ucomm_relay_create(&b, 1, 2, &a, 1, 1, 64, 1, 0, &reverse) == UCOMM_OK);
  int rd = 0;
  CHECK(ucomm_relay_progress_pair(relay, relay, &done, &rd) == UCOMM_ERR_ARG);
  CHECK(ucomm_relay_progress_pair(relay, reverse, &done, &done) == UCOMM_ERR_ARG);
  previous = polls;
  CHECK(ucomm_relay_progress_pair(relay, reverse, &done, &rd) == UCOMM_ERR_IO && !done && !rd);
  CHECK(polls == previous + 1);
  CHECK(ucomm_relay_progress_pair(relay, reverse, &done, &rd) == UCOMM_ERR_IO && polls == previous + 1);
  CHECK(ucomm_relay_destroy(relay) == UCOMM_OK);
  CHECK(ucomm_relay_destroy(reverse) == UCOMM_OK);
  puts("progress and registration failures: PASS");
  return 0;
}
