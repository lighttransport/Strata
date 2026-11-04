#ifndef STRATA_ENGRAM_WIRE_H
#define STRATA_ENGRAM_WIRE_H
#include <stdint.h>
#define EG_MAGIC 0x31474553u
#define EG_HELLO 110u
#define EG_REQUEST 111u
#define EG_STATUS 112u
#define EG_DATA 113u
#define EG_MAX_ENTRIES 8192u
#define EG_MAX_ROW 16384u
#define EG_MAX_BYTES (16u*1024u*1024u)
struct eg_hello { uint32_t magic, version; uint64_t size; unsigned char sha256[32]; };
struct eg_request { uint64_t offset; uint32_t length, reserved; };
struct eg_status { uint32_t code, bytes; };
#ifdef __cplusplus
static_assert(sizeof(eg_hello)==48 && sizeof(eg_request)==16 && sizeof(eg_status)==8, "Engram wire ABI mismatch");
#else
_Static_assert(sizeof(struct eg_hello)==48 && sizeof(struct eg_request)==16 && sizeof(struct eg_status)==8, "Engram wire ABI mismatch");
#endif
#endif
