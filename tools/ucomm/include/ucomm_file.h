/*
 * ucomm_file - byte-range file transfer between two ranks of a ucomm context.
 *
 * One call per side moves [offset, offset + length) of an open file descriptor from the sender to the receiver's
 * descriptor (pread / pwrite, so the descriptors may be shared with other ranges). The range is cut into chunks;
 * up to `depth` chunks are in flight, each in its own registered buffer, so on the ib backend the receiver
 * RDMA-reads straight into it. There is no compression.
 *
 * Verification (opt-in): the sender sends a CRC32C of every chunk after its data; the receiver checks it before
 * writing. Chunks that fail are listed back to the sender and sent again (up to 3 rounds), then the call fails.
 *
 * Resume: the receiver passes a bitmap of the chunks it already holds (one bit per chunk, bit i of byte i/8);
 * it is sent to the sender first, so only the missing chunks travel. The receiver's bitmap is updated as
 * chunks land; the caller persists it.
 *
 * Both sides must pass the same length, chunk size and verify setting. A transfer uses two tags: tag (control)
 * and tag + 1 (data); both must be below UCOMM_TAG_RESERVED.
 */
#ifndef UCOMM_FILE_H_
#define UCOMM_FILE_H_

#include "ucomm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { UCOMM_FILE_VERIFY_NONE = 0, UCOMM_FILE_VERIFY_CRC32C = 1 } ucomm_file_verify;

typedef struct {
  size_t chunk_bytes; /* 0: 16 MiB; a multiple of 4096 */
  int depth;          /* chunks in flight; 0: 4 */
  int verify;         /* ucomm_file_verify */
  /* Receiver only: called after each chunk is written (progress, persisting the bitmap). May be NULL. */
  void (*on_chunk)(void *user, uint64_t chunk, uint64_t done_bytes);
  void *user;
} ucomm_file_opts;

typedef struct {
  uint64_t bytes;   /* payload bytes moved this call */
  uint64_t chunks;  /* chunks moved (first sends) */
  uint64_t skipped; /* chunks the receiver already held */
  uint64_t resent;  /* chunks sent again after a CRC mismatch */
  double seconds;
} ucomm_file_stats;

void ucomm_file_opts_default(ucomm_file_opts *o);
uint64_t ucomm_file_chunks(uint64_t length, const ucomm_file_opts *o);
size_t ucomm_file_bitmap_bytes(uint64_t length, const ucomm_file_opts *o);

ucomm_status ucomm_file_send(ucomm_t *c, int peer, uint32_t tag, int fd, uint64_t offset, uint64_t length,
                             const ucomm_file_opts *o, ucomm_file_stats *st);
/* have: ucomm_file_bitmap_bytes() bytes, or NULL (nothing held yet). */
ucomm_status ucomm_file_recv(ucomm_t *c, int peer, uint32_t tag, int fd, uint64_t offset, uint64_t length,
                             const ucomm_file_opts *o, uint8_t *have, ucomm_file_stats *st);

/* CRC32C (Castagnoli), the SSE4.2 instruction when the CPU has it. crc: 0 to start. */
uint32_t ucomm_crc32c(uint32_t crc, const void *p, size_t n);

#ifdef __cplusplus
}
#endif
#endif
