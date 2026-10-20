/* pf_hash.h - checksums for strata-pfetch: CRC32C per message, SHA-256 for staging sidecars. */
#ifndef PF_HASH_H
#define PF_HASH_H
#include <stddef.h>
#include <stdint.h>

uint32_t pf_crc32c(uint32_t crc, const void *p, size_t n);

typedef struct {
  uint32_t h[8];
  uint64_t bytes;
  uint8_t buf[64];
  size_t fill;
} pf_sha256;
void pf_sha256_init(pf_sha256 *s);
void pf_sha256_update(pf_sha256 *s, const void *p, size_t n);
void pf_sha256_hex(pf_sha256 *s, char out[65]);

/* SHA-256 of a whole file, read with pread in 8 MiB steps (page cache dropped behind). 0 on success. */
int pf_sha256_file(const char *path, char out[65]);
#endif
