/* ucomm_file over two ranks on 127.0.0.1 (processes): a whole odd-sized range, with and without CRC32C, a resume
 * from a half-filled bitmap, a sub-range at an offset, and CRC32C known-answer values.
 *   test_file [port]      env as for ucomm_config_default (e.g. UCOMM_BACKEND=udp, UCOMM_UDP_DROP=0.02) */
#define _GNU_SOURCE
#include "ucomm.h"
#include "ucomm_file.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static const size_t SIZE = (24u << 20) + 12345;

static int compare(const char *a, const char *b, uint64_t a_off, uint64_t len) {
  int fa = open(a, O_RDONLY), fb = open(b, O_RDONLY);
  uint8_t *x = malloc(len), *y = malloc(len);
  int ok = fa >= 0 && fb >= 0 && pread(fa, x, len, (off_t)a_off) == (ssize_t)len && pread(fb, y, len, 0) == (ssize_t)len &&
           !memcmp(x, y, len);
  close(fa);
  close(fb);
  free(x);
  free(y);
  return ok;
}

static int rank_main(int me, const char *root, const char *src, const char *dst) {
  ucomm_config cfg;
  ucomm_config_default(&cfg);
  cfg.rank = me;
  cfg.world = 2;
  cfg.root_addr = root;
  ucomm_t *c;
  if (ucomm_init(&cfg, &c) != UCOMM_OK) return 1;
  int fails = 0;
  ucomm_file_opts o;
  ucomm_file_opts_default(&o);
  o.chunk_bytes = 1u << 20;
  o.depth = 3;
  for (int round = 0; round < 4; round++) {
    o.verify = round >= 1 ? UCOMM_FILE_VERIFY_CRC32C : UCOMM_FILE_VERIFY_NONE;
    const uint64_t off = round == 3 ? 3000000 : 0, len = round == 3 ? 5000001 : SIZE;
    ucomm_file_stats st;
    if (me == 0) {
      int fd = open(src, O_RDONLY);
      fails += ucomm_file_send(c, 1, 10 + 2 * round, fd, off, len, &o, &st) != UCOMM_OK;
      close(fd);
    } else {
      int fd = open(dst, O_RDWR | O_CREAT | O_TRUNC, 0644);
      size_t bm = ucomm_file_bitmap_bytes(len, &o);
      uint8_t *have = calloc(bm, 1);
      if (round == 2) { /* resume: the first half is already there */
        const uint64_t n = ucomm_file_chunks(len, &o);
        int sfd = open(src, O_RDONLY);
        uint8_t *buf = malloc(o.chunk_bytes);
        for (uint64_t i = 0; i < n / 2; i++) {
          ssize_t k = pread(sfd, buf, o.chunk_bytes, (off_t)(i * o.chunk_bytes));
          if (k > 0 && pwrite(fd, buf, (size_t)k, (off_t)(i * o.chunk_bytes)) != k) fails++;
          have[i / 8] |= (uint8_t)(1u << (i % 8));
        }
        free(buf);
        close(sfd);
      }
      ucomm_status s = ucomm_file_recv(c, 0, 10 + 2 * round, fd, 0, len, &o, have, &st);
      close(fd);
      const int ok = s == UCOMM_OK && compare(src, dst, off, len);
      printf("round %d (%s, %s): %llu bytes, %llu chunks sent, %llu skipped, %.3f s  %s\n", round,
             o.verify ? "crc32c" : "no verify", round == 2 ? "resume" : round == 3 ? "sub-range" : "whole",
             (unsigned long long)st.bytes, (unsigned long long)st.chunks, (unsigned long long)st.skipped, st.seconds,
             ok ? "ok" : "FAIL");
      fails += !ok;
      if (round == 2 && st.skipped != ucomm_file_chunks(len, &o) / 2) { printf("resume skipped the wrong count\n"); fails++; }
      free(have);
    }
  }
  ucomm_finalize(c);
  return fails;
}

int main(int argc, char **argv) {
  /* CRC32C check values (RFC 3720 / iSCSI): "123456789" -> 0xe3069283, 32 zero bytes -> 0x8a9136aa */
  uint8_t zeros[32] = {0};
  const uint32_t a = ucomm_crc32c(0, "123456789", 9), b = ucomm_crc32c(0, zeros, 32);
  const uint32_t split = ucomm_crc32c(ucomm_crc32c(0, "1234", 4), "56789", 5);
  printf("crc32c: %08x %08x %08x  %s\n", a, b, split, a == 0xe3069283u && b == 0x8a9136aau && split == a ? "ok" : "FAIL");
  int fails = !(a == 0xe3069283u && b == 0x8a9136aau && split == a);
  char root[64], src[256], dst[256];
  snprintf(root, sizeof root, "127.0.0.1:%s", argc > 1 ? argv[1] : "29655");
  const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
  snprintf(src, sizeof src, "%s/ucomm_test_file_%d.src", tmp, (int)getpid());
  snprintf(dst, sizeof dst, "%s/ucomm_test_file_%d.dst", tmp, (int)getpid());
  int fd = open(src, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  uint8_t *buf = malloc(SIZE);
  uint64_t x = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < SIZE; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; buf[i] = (uint8_t)x; }
  if (write(fd, buf, SIZE) != (ssize_t)SIZE) return 1;
  close(fd);
  free(buf);
  pid_t pid = fork();
  if (pid == 0) {
    const int r = rank_main(1, root, src, dst);
    fflush(stdout);
    _exit(r);
  }
  fails += rank_main(0, root, src, dst);
  int status = 0;
  waitpid(pid, &status, 0);
  fails += !WIFEXITED(status) || WEXITSTATUS(status);
  unlink(src);
  unlink(dst);
  printf("test_file: %s\n", fails ? "FAIL" : "ok");
  return fails ? 1 : 0;
}
