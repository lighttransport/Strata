/* Three-process relay exercise. All roles must use the same size/count/depth.
 * TCP addresses are bootstrap only; -b ib forces verbs for every payload. */
#define _POSIX_C_SOURCE 200809L
#include "ucomm_relay.h"
#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OK(expr) do { ucomm_status s_ = (expr); if (s_ != UCOMM_OK) { \
  fprintf(stderr, "%s: %s\n", #expr, ucomm_strerror(s_)); exit(1); } } while (0)

static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void die(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
static unsigned long long number(const char *s) {
  char *end; errno = 0;
  unsigned long long n = strtoull(s, &end, 10);
  if (errno || !*s || *s == '-' || *end) die("invalid numeric argument");
  return n;
}
static void receive(ucomm_t *c, uint32_t tag, void *p, size_t n) {
  size_t got; OK(ucomm_recv(c, 0, tag, p, n, &got));
  if (got != n) die("unexpected message length");
}
static void ready(ucomm_t *c) {
  uint64_t token = 1;
  OK(ucomm_send(c, 0, 3, &token, sizeof token));
  receive(c, 4, &token, sizeof token);
  if (token != 1) die("invalid ready acknowledgement");
}
static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static void pump(ucomm_t *a, ucomm_t *b, size_t cap, int depth,
                 uint64_t count, uint64_t replies, uint32_t tag, int timeout, int json, int paired) {
  ucomm_relay_t *f, *r;
  OK(ucomm_relay_create(a, 1, tag, b, 1, tag, cap, depth, count, &f));
  OK(ucomm_relay_create(b, 1, tag + 1, a, 1, tag + 1, 8, tag == 20 ? 1 : depth, replies, &r));
  if (tag == 20) {
    uint64_t token; size_t got;
    OK(ucomm_recv(a, 1, 3, &token, sizeof token, &got));
    if (got != sizeof token || token != 1) die("invalid source ready message");
    OK(ucomm_recv(b, 1, 3, &token, sizeof token, &got));
    if (got != sizeof token || token != 1) die("invalid destination ready message");
    /* Register all pools before starting the timed payload phase. */
    OK(ucomm_send(b, 1, 4, &token, sizeof token));
    OK(ucomm_send(a, 1, 4, &token, sizeof token));
  }
  int fd = 0, rd = 0;
  double start = now();
  while (!fd || !rd) {
    if (paired) OK(ucomm_relay_progress_pair(f, r, &fd, &rd));
    else {
      if (!fd) OK(ucomm_relay_progress(f, &fd));
      if (!rd) OK(ucomm_relay_progress(r, &rd));
    }
    if (timeout > 0 && (now() - start) * 1000 > timeout) die("relay phase deadline exceeded");
  }
  printf(json ? "{\"event\":\"relay\",\"tag\":%u,\"forward_bytes\":%llu,\"reverse_bytes\":%llu}\n"
              : "relay: tag %u, forward %llu bytes, reverse %llu bytes\n", tag,
         (unsigned long long)ucomm_relay_bytes(f), (unsigned long long)ucomm_relay_bytes(r));
  OK(ucomm_relay_destroy(r)); OK(ucomm_relay_destroy(f));
}
static void usage(const char *p) {
  fprintf(stderr, "usage: %s -r relay|left|right -a LEFT_ROOT:PORT -z RIGHT_ROOT:PORT\n"
          "  [-b ib|udp] [-p LOCAL_PORT] [-q RELAY_RIGHT_PORT] [-s BYTES]\n"
          "  [-n MESSAGES] [-l LATENCY_ITERS] [-d DEPTH] [-R] [-V] [-J] [-P]\n"
          "Defaults: ib, port 1 (relay right 2), 1048576 bytes, 100 messages, depth 4.\n"
          "Latency defaults to 1000 exchanges, independent of bulk message count.\n"
          "-R reverses bandwidth direction; -V checks every received payload byte;\n"
          "-J emits JSON lines on stdout (diagnostics remain on stderr).\n"
          "-P uses paired duplex polling on the relay (opt-in).\n", p);
}
int main(int argc, char **argv) {
  ucomm_config cfg; ucomm_config_default(&cfg);
  cfg.backend = UCOMM_BACKEND_IB; cfg.world = 2;
  const char *role = NULL, *left = NULL, *right = NULL;
  size_t size = 1u << 20;
  uint64_t count = 100, lat_iters = 1000;
  int depth = 4, reverse = 0, verify = 0, rp = 2, json = 0, paired = 0, opt;
  while ((opt = getopt(argc, argv, "r:a:z:b:p:q:s:n:l:d:RVJPh")) != -1) {
    switch (opt) {
      case 'r': role = optarg; break;
      case 'a': left = optarg; break;
      case 'z': right = optarg; break;
      case 'b':
        if (!strcmp(optarg, "ib")) cfg.backend = UCOMM_BACKEND_IB;
        else if (!strcmp(optarg, "udp")) cfg.backend = UCOMM_BACKEND_UDP;
        else die("backend must be ib or udp (no automatic fallback)");
        break;
      case 'p': { unsigned long long v = number(optarg); if (v < 1 || v > 255) die("bad port"); cfg.ib_port = (int)v; break; }
      case 'q': { unsigned long long v = number(optarg); if (v < 1 || v > 255) die("bad right port"); rp = (int)v; break; }
      case 's': { unsigned long long v = number(optarg); if (!v || v > (1u << 28)) die("size must be 1..268435456"); size = (size_t)v; break; }
      case 'n': count = number(optarg); if (!count || count > 10000000) die("count must be 1..10000000"); break;
      case 'l': lat_iters = number(optarg); if (!lat_iters || lat_iters > 1000000) die("latency iterations must be 1..1000000"); break;
      case 'd': { unsigned long long v = number(optarg); if (v < 1 || v > 64) die("depth must be 1..64"); depth = (int)v; break; }
      case 'R': reverse = 1; break;
      case 'V': verify = 1; break;
      case 'J': json = 1; break;
      case 'P': paired = 1; break;
      default: usage(argv[0]); return opt == 'h' ? 0 : 2;
    }
  }
  if (!role || !left || !right || optind != argc ||
      (strcmp(role, "relay") && strcmp(role, "left") && strcmp(role, "right"))) {
    usage(argv[0]); return 2;
  }
  setvbuf(stdout, NULL, _IOLBF, 0);
  int relay = !strcmp(role, "relay"), isleft = !strcmp(role, "left");
  ucomm_t *a = NULL, *b = NULL, *c = NULL;
  cfg.rank = relay ? 0 : 1;
  cfg.root_addr = relay || isleft ? left : right;
  OK(ucomm_init(&cfg, &c));
  if (relay) {
    a = c; cfg.root_addr = right; cfg.ib_port = rp;
    OK(ucomm_init(&cfg, &b));
  }
  /* Check protocol parameters before posting payloads. */
  uint64_t params[7] = {1, size, count, (uint64_t)depth, (uint64_t)reverse, (uint64_t)verify, lat_iters};
  if (relay) {
    uint64_t peer[7]; size_t got;
    OK(ucomm_recv(a, 1, 1, peer, sizeof peer, &got));
    if (got != sizeof peer || memcmp(peer, params, sizeof peer)) die("left settings mismatch");
    OK(ucomm_recv(b, 1, 1, peer, sizeof peer, &got));
    if (got != sizeof peer || memcmp(peer, params, sizeof peer)) die("right settings mismatch");
    OK(ucomm_send(a, 1, 2, params, sizeof params));
    OK(ucomm_send(b, 1, 2, params, sizeof params));
    if (!json) printf("relay: left/right backends %s/%s, size %zu, depth %d\n",
           ucomm_backend_name(a), ucomm_backend_name(b), size, depth);
    /* Small ping-pongs first, then pipelined bulk traffic and a terminal ack. */
    pump(a, b, 8, depth, lat_iters + 10, lat_iters + 10, 10, cfg.timeout_ms, json, paired);
    pump(reverse ? b : a, reverse ? a : b, size, depth, count, 1, 20, cfg.timeout_ms, json, paired);
    ucomm_finalize(a); ucomm_finalize(b);
    return 0;
  }
  OK(ucomm_send(c, 0, 1, params, sizeof params));
  uint64_t confirm[7]; receive(c, 2, confirm, sizeof confirm);
  if (memcmp(confirm, params, sizeof params)) die("relay settings mismatch");
  uint64_t token = 0;
  double start = 0, sum = 0, min = 1e30, max = 0;
  double *lat = isleft ? malloc((size_t)lat_iters * sizeof *lat) : NULL;
  if (isleft && !lat) die("latency allocation failed");
  for (uint64_t i = 0; i < lat_iters + 10; i++) {
    double t = now();
    if (isleft) {
      token = i; OK(ucomm_send(c, 0, 10, &token, 8)); receive(c, 11, &token, 8);
      if (token != i) die("latency payload mismatch");
    } else {
      receive(c, 10, &token, 8); OK(ucomm_send(c, 0, 11, &token, 8));
    }
    double us = (now() - t) * 1e6;
    if (isleft && i >= 10) { lat[i - 10] = us; sum += us; if (us < min) min = us; if (us > max) max = us; }
  }
  if (isleft) {
    qsort(lat, (size_t)lat_iters, sizeof *lat, compare_double);
    double p50 = lat[(lat_iters - 1) / 2], p99 = lat[(lat_iters * 99 + 99) / 100 - 1];
    if (json) printf("{\"event\":\"latency\",\"backend\":\"%s\",\"iterations\":%llu,"
                     "\"rtt_min_us\":%.6f,\"rtt_mean_us\":%.6f,\"rtt_max_us\":%.6f,"
                     "\"rtt_p50_us\":%.6f,\"rtt_p99_us\":%.6f}\n", ucomm_backend_name(c),
                     (unsigned long long)lat_iters, min, sum / lat_iters, max, p50, p99);
    else printf("latency: 8 B, RTT min/mean/max %.3f/%.3f/%.3f us, p50/p99 %.3f/%.3f us, mean half RTT %.3f us\n",
                min, sum / lat_iters, max, p50, p99, sum / lat_iters / 2);
  }
  free(lat);
  if (size > SIZE_MAX / (size_t)depth) die("buffer size overflow");
  unsigned char *buf = malloc(size * (size_t)depth);
  if (!buf) die("buffer allocation failed");
  ucomm_mr_t *mr;
  ucomm_status registration = ucomm_mr_reg(c, buf, size * (size_t)depth, &mr);
  if (registration != UCOMM_OK) {
    fprintf(stderr, "bulk registration failed (%zu bytes): %s; check ulimit -l and HCA resources\n",
            size * (size_t)depth, ucomm_strerror(registration));
    return 1;
  }
  ucomm_req_t *req[64] = {0};
  int source = reverse ? !isleft : isleft;
  for (int i = 0; i < depth; i++) memset(buf + size * (size_t)i, i + 1, size);
  ready(c);
  start = now();
  if (source) {
    for (uint64_t i = 0; i < count; i++) {
      int s = (int)(i % (uint64_t)depth);
      if (req[s]) { OK(ucomm_wait(req[s], NULL)); req[s] = NULL; }
      if (size >= sizeof i) memcpy(buf + size * (size_t)s, &i, sizeof i);
      OK(ucomm_isend(c, 0, 20, buf + size * (size_t)s, size, &req[s]));
    }
    OK(ucomm_waitall(depth, req));
    receive(c, 21, &token, 8);
    if (token != count) die("terminal ack mismatch");
    double secs = now() - start;
    if (json) printf("{\"event\":\"bandwidth\",\"backend\":\"%s\",\"reverse\":%d,\"verify\":%d,"
                     "\"messages\":%llu,\"size\":%zu,\"depth\":%d,\"seconds\":%.9f,\"MBps\":%.6f,\"Gbps\":%.6f}\n",
                     ucomm_backend_name(c), reverse, verify, (unsigned long long)count, size, depth, secs,
                     (double)size * count / secs / 1e6, (double)size * count * 8 / secs / 1e9);
    else printf("bandwidth: %s -> %s, %llu x %zu B, %.3f MB/s, %.3f Gbit/s, verify=%d\n",
           reverse ? "right" : "left", reverse ? "left" : "right",
           (unsigned long long)count, size, (double)size * count / secs / 1e6,
           (double)size * count * 8 / secs / 1e9, verify);
  } else {
    for (uint64_t i = 0; i < count; i++) {
      int s = (int)(i % (uint64_t)depth);
      receive(c, 20, buf + size * (size_t)s, size);
      if (verify) {
        size_t first = 0;
        if (size >= sizeof i) {
          uint64_t seq; memcpy(&seq, buf + size * (size_t)s, sizeof seq);
          if (seq != i) die("bandwidth sequence mismatch");
          first = sizeof i;
        }
        for (size_t k = first; k < size; k++)
          if (buf[size * (size_t)s + k] != (unsigned char)(s + 1)) die("bandwidth payload mismatch");
      }
    }
    token = count; OK(ucomm_send(c, 0, 21, &token, 8));
  }
  ucomm_mr_dereg(mr); free(buf); ucomm_finalize(c);
  return 0;
}
