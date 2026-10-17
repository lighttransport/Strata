/* ucomm_file.c - chunked byte-range file transfer over a ucomm context (include/ucomm_file.h).
 *
 * Protocol for one range (ctrl = tag, data = tag + 1):
 *   receiver -> sender  ctrl  bitmap of chunks already held
 *   sender   -> receiver data chunk payload, then ctrl {u64 index, u64 crc32c} when verifying (per chunk, in order)
 *   receiver -> sender  ctrl  {u64 count, u64 index[count]}: chunks that failed verification (count 0: done)
 *   ... the failed chunks are sent again, up to 3 rounds.
 * Both sides walk the same chunk list in the same order, so the receiver can post its data receives ahead. */
#define _GNU_SOURCE
#include "ucomm_file.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define FILE_ROUNDS 4 /* the first send plus 3 resends */

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
static int have_sse42(void) {
  static int v = -1;
  if (v < 0) v = __builtin_cpu_supports("sse4.2") ? 1 : 0;
  return v;
}
#endif

uint32_t ucomm_crc32c(uint32_t crc, const void *p, size_t n) {
  crc = ~crc;
#if defined(__x86_64__) && defined(__GNUC__)
  if (have_sse42()) return ~crc_hw(crc, p, n);
#endif
  return ~crc_sw(crc, p, n);
}

/* ---------------------------------------------------------------- helpers */

void ucomm_file_opts_default(ucomm_file_opts *o) {
  memset(o, 0, sizeof *o);
  o->chunk_bytes = 16u << 20;
  o->depth = 4;
  o->verify = UCOMM_FILE_VERIFY_NONE;
}

static size_t chunk_of(const ucomm_file_opts *o) { return o && o->chunk_bytes ? o->chunk_bytes : 16u << 20; }
static int depth_of(const ucomm_file_opts *o) { return o && o->depth > 0 ? (o->depth > 16 ? 16 : o->depth) : 4; }

uint64_t ucomm_file_chunks(uint64_t length, const ucomm_file_opts *o) {
  const size_t ch = chunk_of(o);
  return (length + ch - 1) / ch;
}
size_t ucomm_file_bitmap_bytes(uint64_t length, const ucomm_file_opts *o) {
  return (size_t)((ucomm_file_chunks(length, o) + 7) / 8);
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static int full_pread(int fd, void *buf, size_t n, uint64_t off) {
  uint8_t *p = buf;
  while (n) {
    ssize_t r = pread(fd, p, n, (off_t)off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    p += r;
    off += (uint64_t)r;
    n -= (size_t)r;
  }
  return 0;
}
static int full_pwrite(int fd, const void *buf, size_t n, uint64_t off) {
  const uint8_t *p = buf;
  while (n) {
    ssize_t r = pwrite(fd, p, n, (off_t)off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    p += r;
    off += (uint64_t)r;
    n -= (size_t)r;
  }
  return 0;
}

typedef struct {
  int depth;
  size_t chunk;
  uint8_t *buf[16];
  ucomm_mr_t *mr[16];
} slots_t;

static ucomm_status slots_alloc(ucomm_t *c, slots_t *s, int depth, size_t chunk) {
  memset(s, 0, sizeof *s);
  s->depth = depth;
  s->chunk = chunk;
  for (int i = 0; i < depth; i++) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, chunk)) return UCOMM_ERR_NOMEM;
    s->buf[i] = p;
    if (ucomm_mr_reg(c, p, chunk, &s->mr[i]) != UCOMM_OK) s->mr[i] = NULL; /* registered on the fly instead */
  }
  return UCOMM_OK;
}
static void slots_free(slots_t *s) {
  for (int i = 0; i < s->depth; i++) {
    if (s->mr[i]) ucomm_mr_dereg(s->mr[i]);
    free(s->buf[i]);
  }
}

static size_t chunk_len(uint64_t index, uint64_t length, size_t chunk) {
  const uint64_t start = index * chunk;
  return (size_t)(length - start < chunk ? length - start : chunk);
}

/* ---------------------------------------------------------------- sender */

ucomm_status ucomm_file_send(ucomm_t *c, int peer, uint32_t tag, int fd, uint64_t offset, uint64_t length,
                             const ucomm_file_opts *o, ucomm_file_stats *st) {
  ucomm_file_stats local = {0};
  if (!st) st = &local;
  memset(st, 0, sizeof *st);
  if (!c || fd < 0 || tag + 1 >= UCOMM_TAG_RESERVED) return UCOMM_ERR_ARG;
  if (!length) return UCOMM_OK;
  const double t0 = now_s();
  const size_t chunk = chunk_of(o);
  const int depth = depth_of(o), verify = o && o->verify == UCOMM_FILE_VERIFY_CRC32C;
  const uint64_t n = ucomm_file_chunks(length, o);
  const size_t bm_bytes = ucomm_file_bitmap_bytes(length, o);
  uint8_t *have = calloc(bm_bytes ? bm_bytes : 1, 1);
  uint64_t *list = malloc((n + 1) * sizeof *list);
  slots_t s;
  ucomm_status rc = (have && list) ? slots_alloc(c, &s, depth, chunk) : UCOMM_ERR_NOMEM;
  if (rc != UCOMM_OK) goto out_nomem;
  size_t got = 0;
  rc = ucomm_recv(c, peer, tag, have, bm_bytes, &got);
  if (rc == UCOMM_OK && got != bm_bytes) rc = UCOMM_ERR_ARG; /* the two sides disagree on length or chunk size */
  if (rc != UCOMM_OK) goto out;
  uint64_t count = 0;
  for (uint64_t i = 0; i < n; i++)
    if (have[i / 8] & (1u << (i % 8))) st->skipped++;
    else list[count++] = i;
  for (int round = 0; round < FILE_ROUNDS; round++) {
    ucomm_req_t *data[16] = {0}, *trailer[16] = {0};
    uint64_t trail[16][2];
    for (uint64_t j = 0; j < count && rc == UCOMM_OK; j++) {
      const int k = (int)(j % (uint64_t)depth);
      if (data[k] && (rc = ucomm_wait(data[k], NULL)) != UCOMM_OK) break;
      data[k] = NULL;
      if (trailer[k] && (rc = ucomm_wait(trailer[k], NULL)) != UCOMM_OK) break;
      trailer[k] = NULL;
      const uint64_t index = list[j];
      const size_t len = chunk_len(index, length, chunk);
      if (full_pread(fd, s.buf[k], len, offset + index * chunk)) {
        rc = UCOMM_ERR_IO;
        break;
      }
      if ((rc = ucomm_isend(c, peer, tag + 1, s.buf[k], len, &data[k])) != UCOMM_OK) break;
      if (verify) {
        trail[k][0] = index;
        trail[k][1] = ucomm_crc32c(0, s.buf[k], len);
        if ((rc = ucomm_isend(c, peer, tag, trail[k], sizeof trail[k], &trailer[k])) != UCOMM_OK) break;
      }
      st->bytes += len;
      if (round) st->resent++;
      else st->chunks++;
    }
    for (int k = 0; k < depth; k++) {
      if (data[k]) { ucomm_status w = ucomm_wait(data[k], NULL); if (rc == UCOMM_OK) rc = w; }
      if (trailer[k]) { ucomm_status w = ucomm_wait(trailer[k], NULL); if (rc == UCOMM_OK) rc = w; }
    }
    if (rc != UCOMM_OK) break;
    /* the receiver's verdict: the chunks to send again */
    rc = ucomm_recv(c, peer, tag, list, (n + 1) * sizeof *list, &got);
    if (rc != UCOMM_OK) break;
    count = list[0];
    if (got != (count + 1) * sizeof *list || count > n) { rc = UCOMM_ERR_IO; break; }
    memmove(list, list + 1, count * sizeof *list);
    if (!count) break;
    if (round == FILE_ROUNDS - 1) rc = UCOMM_ERR_IO;
  }
out:
  slots_free(&s);
out_nomem:
  free(have);
  free(list);
  st->seconds = now_s() - t0;
  return rc;
}

/* ---------------------------------------------------------------- receiver */

ucomm_status ucomm_file_recv(ucomm_t *c, int peer, uint32_t tag, int fd, uint64_t offset, uint64_t length,
                             const ucomm_file_opts *o, uint8_t *have, ucomm_file_stats *st) {
  ucomm_file_stats local = {0};
  if (!st) st = &local;
  memset(st, 0, sizeof *st);
  if (!c || fd < 0 || tag + 1 >= UCOMM_TAG_RESERVED) return UCOMM_ERR_ARG;
  if (!length) return UCOMM_OK;
  const double t0 = now_s();
  const size_t chunk = chunk_of(o);
  const int depth = depth_of(o), verify = o && o->verify == UCOMM_FILE_VERIFY_CRC32C;
  const uint64_t n = ucomm_file_chunks(length, o);
  const size_t bm_bytes = ucomm_file_bitmap_bytes(length, o);
  uint8_t *bm = have ? have : calloc(bm_bytes ? bm_bytes : 1, 1);
  uint64_t *list = malloc((n + 1) * sizeof *list), *failed = malloc((n + 1) * sizeof *failed);
  slots_t s;
  ucomm_status rc = (bm && list && failed) ? slots_alloc(c, &s, depth, chunk) : UCOMM_ERR_NOMEM;
  if (rc != UCOMM_OK) goto out_nomem;
  if ((rc = ucomm_send(c, peer, tag, bm, bm_bytes)) != UCOMM_OK) goto out;
  uint64_t count = 0, done = 0;
  for (uint64_t i = 0; i < n; i++)
    if (bm[i / 8] & (1u << (i % 8))) st->skipped++, done += chunk_len(i, length, chunk);
    else list[count++] = i;
  for (int round = 0; round < FILE_ROUNDS; round++) {
    ucomm_req_t *data[16] = {0};
    uint64_t nfail = 0;
    for (uint64_t j = 0; j < count && j < (uint64_t)depth && rc == UCOMM_OK; j++)
      rc = ucomm_irecv(c, peer, tag + 1, s.buf[j], chunk_len(list[j], length, chunk), &data[j]);
    for (uint64_t j = 0; j < count && rc == UCOMM_OK; j++) {
      const int k = (int)(j % (uint64_t)depth);
      const uint64_t index = list[j];
      const size_t len = chunk_len(index, length, chunk);
      size_t got = 0;
      if ((rc = ucomm_wait(data[k], &got)) != UCOMM_OK) break;
      data[k] = NULL;
      if (got != len) { rc = UCOMM_ERR_IO; break; }
      int good = 1;
      if (verify) {
        uint64_t trail[2];
        if ((rc = ucomm_recv(c, peer, tag, trail, sizeof trail, &got)) != UCOMM_OK) break;
        if (got != sizeof trail || trail[0] != index) { rc = UCOMM_ERR_IO; break; }
        good = (uint32_t)trail[1] == ucomm_crc32c(0, s.buf[k], len);
      }
      if (good) {
        if (full_pwrite(fd, s.buf[k], len, offset + index * chunk)) { rc = UCOMM_ERR_IO; break; }
        bm[index / 8] |= (uint8_t)(1u << (index % 8));
        st->bytes += len;
        done += len;
        if (round) st->resent++;
        else st->chunks++;
        if (o && o->on_chunk) o->on_chunk(o->user, index, done);
      } else failed[1 + nfail++] = index;
      if (j + depth < count)
        rc = ucomm_irecv(c, peer, tag + 1, s.buf[k], chunk_len(list[j + depth], length, chunk), &data[k]);
    }
    for (int k = 0; k < depth; k++)
      if (data[k]) ucomm_wait(data[k], NULL);
    if (rc != UCOMM_OK) break;
    failed[0] = nfail;
    if ((rc = ucomm_send(c, peer, tag, failed, (nfail + 1) * sizeof *failed)) != UCOMM_OK) break;
    if (!nfail) break;
    if (round == FILE_ROUNDS - 1) { rc = UCOMM_ERR_IO; break; }
    memcpy(list, failed + 1, nfail * sizeof *list);
    count = nfail;
  }
out:
  slots_free(&s);
out_nomem:
  if (!have) free(bm);
  free(list);
  free(failed);
  st->seconds = now_s() - t0;
  return rc;
}
