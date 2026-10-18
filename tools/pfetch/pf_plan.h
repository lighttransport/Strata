/* pf_plan.h - strata-pfetch manifest and cooperative read plan (pure functions, no I/O besides what the caller passes).
 *
 * Manifest v1, one record per line, whitespace separated, '#' comments:
 *   PFETCH 1 [block=BYTES]
 *   G <gid> <name> <priority>                    group; gids are 0,1,2,... in order; lower priority is read first
 *   D <did> <rank|*> <path> <size> <sparse|atomic> [sidecar=<path>]
 *                                                 destination; dids are 0,1,2,... in order; '*' = one copy per rank
 *   S <rank|*> <linkpath> <target>               symlink made by that rank
 *   C <did> <dst_off> <src_path> <src_off> <len> [gid]   copy bytes of a source file into a destination
 * Paths may start with "@L/" (the --local-root) and contain "{rank}" / "{rank2}" (rank, two digits).
 * sparse: the file is created at full size and only the copied ranges are written (holes elsewhere);
 * atomic: written as <path>.pfpart, renamed when complete; copies must tile [0, size) exactly; skipped when <path>
 * already exists with that size (and its sidecar, if any).
 *
 * The plan cuts every source file into aligned blocks; one block is read once by one reader rank, even when several
 * destinations (on several ranks) need its bytes, and is sent to each destination rank as one message holding the
 * union of the ranges that rank needs. */
#ifndef PF_PLAN_H
#define PF_PLAN_H
#include <stddef.h>
#include <stdint.h>

enum { PF_SPARSE = 0, PF_ATOMIC = 1 };
enum { PF_ASSIGN_LOCAL = 0, PF_ASSIGN_SPREAD = 1 };

typedef struct { char *name; int prio; } pf_group;
typedef struct {
  char *path, *sidecar;
  uint64_t size, covered; /* covered: bytes the copies write (after skips) */
  int rank, mode, skip;
} pf_dest;
typedef struct { int rank; char *link, *target; } pf_link;
typedef struct { uint64_t dst_off, src_off, len; uint32_t tmpl, src, group; } pf_copy;
typedef struct {
  uint64_t src_off, dst_off, len, msg_off, blkno;
  uint32_t dest, group, src, block;
  int32_t rank;
} pf_piece;
typedef struct { uint64_t off, len; } pf_span;
typedef struct { int32_t rank; uint32_t piece0, npiece, span0, nspan; uint64_t bytes; } pf_msg;
typedef struct {
  uint64_t index, read_bytes;
  uint32_t src, piece0, npiece, rspan0, nrspan, msg0, nmsg;
  int32_t reader, prio;
} pf_block;

typedef struct {
  int ranks;
  uint64_t block;
  char **srcs;
  uint32_t nsrc, cap_src;
  pf_group *groups;
  uint32_t ngroup;
  pf_dest *dests;
  uint32_t ndest;
  uint32_t *tmpl_first, *tmpl_count, ntmpl; /* manifest did -> expanded dests */
  pf_link *links;
  uint32_t nlink;
  pf_copy *copies;
  size_t ncopy;
  /* built by pf_build */
  pf_piece *pieces;
  size_t npiece;
  pf_block *blocks;
  size_t nblock;
  pf_span *rspans, *mspans;
  size_t nrspan, nmspan;
  pf_msg *msgs;
  size_t nmsg;
  uint64_t *load; /* bytes read per reader rank */
  uint64_t total_read, total_needed;
  /* private: source path hash */
  uint32_t *src_hash;
  uint32_t hash_cap;
} pf_plan;

/* Parse a manifest for `ranks` ranks. local_root may be NULL when no path uses "@L/". 0 on success, else -1 + err. */
int pf_parse(pf_plan *p, const char *text, size_t n, int ranks, const char *local_root, char *err, size_t errn);
/* Build pieces/blocks/messages for the destinations not marked skip, and assign readers. */
int pf_build(pf_plan *p, int assign, char *err, size_t errn);
/* The blocks rank `rank` reads, in read order; out holds nblock entries. Returns the count. */
size_t pf_queue(const pf_plan *p, int rank, int sequential, uint32_t *out);
/* The message for destination rank `rank` of block b, or NULL. */
const pf_msg *pf_block_msg(const pf_plan *p, const pf_block *b, int rank);
/* Expand "@L/" and "{rank}"/"{rank2}" in a path (malloc'ed; NULL + err on failure; rank < 0 forbids {rank}). */
char *pf_expand(const char *s, const char *local_root, int rank, char *err, size_t errn);
void pf_free(pf_plan *p);
#endif
