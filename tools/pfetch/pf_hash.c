/* pf_hash.c - CRC32C (slice-by-8, SSE4.2 or ARMv8 CRC when available) and SHA-256 for strata-pfetch.
 * The CRC table code is the one in tools/ucomm/src/ucomm_file.c. */
#define _GNU_SOURCE
#include "pf_hash.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

/* ---------------------------------------------------------------- CRC32C */

static uint32_t crc_table[8][256];
static int crc_ready;

static void crc_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82f63b78u & (0u - (c & 1)));
    crc_table[0][i] = c;
  }
  for (uint32_t i = 0; i < 256; i++)
    for (int t = 1; t < 8; t++) crc_table[t][i] = (crc_table[t - 1][i] >> 8) ^ crc_table[0][crc_table[t - 1][i] & 255];
  crc_ready = 1;
}

static uint32_t crc_sw(uint32_t crc, const uint8_t *p, size_t n) {
  if (!crc_ready) crc_init();
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    v ^= crc;
    crc = crc_table[7][v & 255] ^ crc_table[6][(v >> 8) & 255] ^ crc_table[5][(v >> 16) & 255] ^
          crc_table[4][(v >> 24) & 255] ^ crc_table[3][(v >> 32) & 255] ^ crc_table[2][(v >> 40) & 255] ^
          crc_table[1][(v >> 48) & 255] ^ crc_table[0][v >> 56];
    p += 8;
    n -= 8;
  }
  while (n--) crc = (crc >> 8) ^ crc_table[0][(crc ^ *p++) & 255];
  return crc;
}

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("sse4.2"))) static uint32_t crc_hw(uint32_t crc, const uint8_t *p, size_t n) {
  uint64_t c = crc;
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    c = __builtin_ia32_crc32di(c, v);
    p += 8;
    n -= 8;
  }
  uint32_t c32 = (uint32_t)c;
  while (n--) c32 = __builtin_ia32_crc32qi(c32, *p++);
  return c32;
}
static int have_hw(void) {
  static int v = -1;
  if (v < 0) v = __builtin_cpu_supports("sse4.2") ? 1 : 0;
  return v;
}
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
static uint32_t crc_hw(uint32_t crc, const uint8_t *p, size_t n) {
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
    p += 8;
    n -= 8;
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return crc;
}
static int have_hw(void) { return 1; }
#endif

uint32_t pf_crc32c(uint32_t crc, const void *p, size_t n) {
  crc = ~crc;
#if (defined(__x86_64__) && defined(__GNUC__)) || (defined(__aarch64__) && defined(__ARM_FEATURE_CRC32))
  if (have_hw()) return ~crc_hw(crc, p, n);
#endif
  return ~crc_sw(crc, p, n);
}

/* ---------------------------------------------------------------- SHA-256 */

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha_block(uint32_t h[8], const uint8_t *p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
    uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
}

void pf_sha256_init(pf_sha256 *s) {
  static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  memcpy(s->h, iv, sizeof iv);
  s->bytes = 0;
  s->fill = 0;
}

void pf_sha256_update(pf_sha256 *s, const void *data, size_t n) {
  const uint8_t *p = data;
  s->bytes += n;
  if (s->fill) {
    size_t take = 64 - s->fill < n ? 64 - s->fill : n;
    memcpy(s->buf + s->fill, p, take);
    s->fill += take, p += take, n -= take;
    if (s->fill < 64) return;
    sha_block(s->h, s->buf);
    s->fill = 0;
  }
  for (; n >= 64; p += 64, n -= 64) sha_block(s->h, p);
  memcpy(s->buf, p, n);
  s->fill = n;
}

void pf_sha256_hex(pf_sha256 *s, char out[65]) {
  uint64_t bits = s->bytes * 8;
  uint8_t pad[72] = {0x80};
  size_t padn = (s->fill < 56 ? 56 : 120) - s->fill;
  for (int i = 0; i < 8; i++) pad[padn + i] = (uint8_t)(bits >> (56 - 8 * i));
  pf_sha256_update(s, pad, padn + 8);
  for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", s->h[i]);
}

int pf_sha256_file(const char *path, char out[65]) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  const size_t step = 8u << 20;
  uint8_t *buf = malloc(step);
  if (!buf) {
    close(fd);
    return -1;
  }
  pf_sha256 s;
  pf_sha256_init(&s);
  off_t off = 0;
  for (;;) {
    ssize_t r = pread(fd, buf, step, off);
    if (r < 0) {
      free(buf);
      close(fd);
      return -1;
    }
    if (r == 0) break;
    pf_sha256_update(&s, buf, (size_t)r);
    posix_fadvise(fd, off, r, POSIX_FADV_DONTNEED);
    off += r;
  }
  free(buf);
  close(fd);
  pf_sha256_hex(&s, out);
  return 0;
}
