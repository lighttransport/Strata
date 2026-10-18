/* strata_pfetch.c - cooperative parallel staging between a shared filesystem and node-local disks.
 *
 * Built with MPI (strata-pfetch) every rank of one mpiexec reads its share of the manifest's source blocks with
 * K threads, sends each block's bytes to the ranks whose destinations need them (MPI point-to-point, bounded windows,
 * CRC32C per message) and writes what it receives with W writer threads; finisher threads rename atomic files and
 * write sha256 sidecars. Built with -DPFETCH_NO_MPI (strata-pput) it is one rank: the same engine, plus --write.
 *
 *   strata-pfetch --manifest=FILE [--local-root=DIR] [--streams=16] [--assign=local|spread] [--order=interleave|sequential]
 *                 [--writers=2] [--finishers=4] [--pool-mib=512] [--recv-slots=8] [--send-window=8]
 *                 [--dest-sync=none|file] [--coverage-out=PATH] [--status-dir=DIR] [--deadline=EPOCH]
 *                 [--progress-s=30] [--inject=corrupt:N|eio:N]
 *   strata-pfetch --plan-dump --ranks=R --manifest=FILE [--assign=..] [--order=..] [--streams=K]
 *   strata-pfetch --measure=read,write,mpi,local [--src-dir=DIR] [--scratch-shared=DIR] [--local-root=DIR]
 *                 [--point-s=45] [--region-mib=512] [--streams-list=1,4,16,32] [--min-file-gib=4]
 *   strata-pput   --write=LIST [--streams=8] [--skip-existing] [--readback] [--block-mib=16]
 *       LIST lines: W <src> <dst> [after=<line index>]   (dst is renamed into place only after that line's dst)
 *
 * See pf_plan.h for the manifest. Every result line is greppable: PFETCH_PROGRESS, PFETCH_AGG, PFETCH_MEASURE,
 * SENTINEL PFETCH_BEGIN / PFETCH_GROUP_READY / PFETCH_END state=ok|partial|failed, SENTINEL PPUT_END. */
#define _GNU_SOURCE
#include "pf_hash.h"
#include "pf_plan.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#ifndef PFETCH_NO_MPI
#include <mpi.h>
#endif

#define TRAILER 64
#define MAGIC 0x50464d31u /* "PFM1" */
enum { TAG_DATA = 101, TAG_DONE = 102, TAG_STAT = 103 };

static int g_rank, g_ranks = 1;

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void nap_us(long us) {
  struct timespec t = {us / 1000000, (us % 1000000) * 1000};
  nanosleep(&t, NULL);
}

static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "pfetch rank %d: ", g_rank);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  printf("SENTINEL PFETCH_END state=failed rank=%d\n", g_rank);
  fflush(stdout);
#ifndef PFETCH_NO_MPI
  int init = 0;
  MPI_Initialized(&init);
  if (init) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  exit(1);
}

static int mkdir_p(const char *path) {
  char buf[4096];
  snprintf(buf, sizeof buf, "%s", path);
  for (char *s = buf + 1; *s; s++)
    if (*s == '/') {
      *s = 0;
      if (mkdir(buf, 0755) && errno != EEXIST) return -1;
      *s = '/';
    }
  return mkdir(buf, 0755) && errno != EEXIST ? -1 : 0;
}
static int mkdir_parent(const char *path) {
  char buf[4096];
  snprintf(buf, sizeof buf, "%s", path);
  char *s = strrchr(buf, '/');
  if (!s || s == buf) return 0;
  *s = 0;
  return mkdir_p(buf);
}

static int pread_full(int fd, uint8_t *p, size_t n, uint64_t off) {
  while (n) {
    ssize_t r = pread(fd, p, n, (off_t)off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return r == 0 ? -2 : -1;
    p += r, n -= (size_t)r, off += (uint64_t)r;
  }
  return 0;
}
static int pwrite_full(int fd, const uint8_t *p, size_t n, uint64_t off) {
  while (n) {
    ssize_t r = pwrite(fd, p, n, (off_t)off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    p += r, n -= (size_t)r, off += (uint64_t)r;
  }
  return 0;
}

static void raise_nofile(void) {
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
    rl.rlim_cur = rl.rlim_max;
    setrlimit(RLIMIT_NOFILE, &rl);
  }
}

static char *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  size_t cap = 1 << 16, len = 0;
  char *buf = malloc(cap);
  size_t r;
  while ((r = fread(buf + len, 1, cap - len, f)) > 0) {
    len += r;
    if (len == cap) buf = realloc(buf, cap *= 2);
  }
  fclose(f);
  *n = len;
  return buf;
}

/* ------------------------------------------------------------------ options */

typedef struct {
  const char *manifest, *local_root, *coverage_out, *status_dir, *measure, *src_dir, *scratch, *write_list;
  const char *streams_list;
  int streams, writers, finishers, assign, sequential, recv_slots, send_window, dest_sync, progress_s, plan_dump;
  int dump_ranks, skip_existing, readback, point_s, region_mib, min_file_gib;
  size_t pool_mib, block_mib;
  double deadline;
  long inject_corrupt, inject_eio;
} opts;

static void parse_args(int argc, char **argv, opts *o) {
  memset(o, 0, sizeof *o);
  o->streams = 16, o->writers = 2, o->finishers = 4, o->recv_slots = 8, o->send_window = 8, o->progress_s = 30;
  o->pool_mib = 512, o->block_mib = 16, o->point_s = 45, o->region_mib = 512, o->min_file_gib = 4;
  o->streams_list = "1,4,16,32";
  o->inject_corrupt = o->inject_eio = -1;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i], *v = strchr(a, '=');
    v = v ? v + 1 : "";
#define IS(k) (strncmp(a, k "=", sizeof k) == 0)
    if (IS("--manifest")) o->manifest = v;
    else if (IS("--local-root")) o->local_root = v;
    else if (IS("--coverage-out")) o->coverage_out = v;
    else if (IS("--status-dir")) o->status_dir = v;
    else if (IS("--streams")) o->streams = atoi(v);
    else if (IS("--writers")) o->writers = atoi(v);
    else if (IS("--finishers")) o->finishers = atoi(v);
    else if (IS("--assign")) {
      if (strcmp(v, "local") && strcmp(v, "spread")) die("--assign is local or spread");
      o->assign = strcmp(v, "spread") == 0 ? PF_ASSIGN_SPREAD : PF_ASSIGN_LOCAL;
    } else if (IS("--order")) {
      if (strcmp(v, "interleave") && strcmp(v, "sequential")) die("--order is interleave or sequential");
      o->sequential = strcmp(v, "sequential") == 0;
    } else if (IS("--pool-mib")) o->pool_mib = (size_t)atol(v);
    else if (IS("--block-mib")) o->block_mib = (size_t)atol(v);
    else if (IS("--recv-slots")) o->recv_slots = atoi(v);
    else if (IS("--send-window")) o->send_window = atoi(v);
    else if (IS("--dest-sync")) {
      if (strcmp(v, "none") && strcmp(v, "file")) die("--dest-sync is none or file");
      o->dest_sync = strcmp(v, "file") == 0;
    } else if (IS("--deadline")) o->deadline = atof(v);
    else if (IS("--progress-s")) o->progress_s = atoi(v);
    else if (IS("--inject")) {
      if (strncmp(v, "corrupt:", 8) == 0) o->inject_corrupt = atol(v + 8);
      else if (strncmp(v, "eio:", 4) == 0) o->inject_eio = atol(v + 4);
      else die("--inject is corrupt:N or eio:N");
    } else if (strcmp(a, "--plan-dump") == 0) o->plan_dump = 1;
    else if (IS("--ranks")) o->dump_ranks = atoi(v);
    else if (IS("--measure")) o->measure = v;
    else if (IS("--src-dir")) o->src_dir = v;
    else if (IS("--scratch-shared")) o->scratch = v;
    else if (IS("--point-s")) o->point_s = atoi(v);
    else if (IS("--region-mib")) o->region_mib = atoi(v);
    else if (IS("--streams-list")) o->streams_list = v;
    else if (IS("--min-file-gib")) o->min_file_gib = atoi(v);
    else if (IS("--write")) o->write_list = v;
    else if (strcmp(a, "--skip-existing") == 0) o->skip_existing = 1;
    else if (strcmp(a, "--readback") == 0) o->readback = 1;
    else die("unknown argument %s", a);
#undef IS
  }
  if (o->streams < 1 || o->streams > 256 || o->writers < 1 || o->finishers < 1 || o->recv_slots < 1 || o->send_window < 1)
    die("thread and window counts must be positive");
}

/* ------------------------------------------------------------------ communication */

#ifndef PFETCH_NO_MPI
static void comm_init(int *argc, char ***argv) {
  int provided;
  MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) die("MPI does not provide MPI_THREAD_FUNNELED");
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_ranks);
}
static void comm_fini(void) { MPI_Finalize(); }
static void barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }
static void bcast(void *p, size_t n, int root) {
  for (size_t o = 0; o < n; o += 1u << 30) {
    size_t k = n - o < (1u << 30) ? n - o : 1u << 30;
    MPI_Bcast((char *)p + o, (int)k, MPI_BYTE, root, MPI_COMM_WORLD);
  }
}
static void allsum_u64(uint64_t *v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD); }
static void allmax_d(double *v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); }
static void gather_d(double v, double *all) { MPI_Gather(&v, 1, MPI_DOUBLE, all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); }
#else
static void comm_init(int *argc, char ***argv) { (void)argc, (void)argv; }
static void comm_fini(void) {}
static void barrier(void) {}
static void bcast(void *p, size_t n, int root) { (void)p, (void)n, (void)root; }
static void allsum_u64(uint64_t *v, int n) { (void)v, (void)n; }
static void allmax_d(double *v, int n) { (void)v, (void)n; }
static void gather_d(double v, double *all) { all[0] = v; }
#endif

/* ------------------------------------------------------------------ buffers and queues */

typedef struct pfbuf {
  uint8_t *p;
  struct pfbuf *next;
  int pool, peer;
  uint32_t len; /* payload + trailer */
} pfbuf;

typedef struct {
  uint32_t magic, block;
  int32_t src, dst;
  uint64_t len;
  uint32_t crc, pad;
  uint8_t rest[TRAILER - 32];
} trailer;

typedef struct {
  pthread_mutex_t mu;
  pthread_cond_t cv;
  pfbuf *head, *tail;
  size_t count;
  int busy, closed;
} queue;

static volatile int g_failed;
static char g_fail_msg[512];
static pthread_mutex_t g_fail_mu = PTHREAD_MUTEX_INITIALIZER;
static queue pools[2]; /* 0: outbound/local messages, 1: receive slots */

static void q_init(queue *q) {
  memset(q, 0, sizeof *q);
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
}
static void q_push(queue *q, pfbuf *b) {
  pthread_mutex_lock(&q->mu);
  b->next = NULL;
  if (q->tail) q->tail->next = b;
  else q->head = b;
  q->tail = b;
  q->count++;
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->mu);
}
static pfbuf *q_take_locked(queue *q) {
  pfbuf *b = q->head;
  q->head = b->next;
  if (!q->head) q->tail = NULL;
  q->count--;
  return b;
}
/* Blocking pop; worker=1 marks the caller busy until its next pop. NULL when closed (or failed) and empty. */
static pfbuf *q_pop(queue *q, int worker) {
  pthread_mutex_lock(&q->mu);
  if (worker) q->busy--;
  while (!q->head && !q->closed && !g_failed) pthread_cond_wait(&q->cv, &q->mu);
  pfbuf *b = q->head && !g_failed ? q_take_locked(q) : NULL;
  if (b && worker) q->busy++;
  pthread_mutex_unlock(&q->mu);
  return b;
}
#ifndef PFETCH_NO_MPI
static pfbuf *q_trypop(queue *q) {
  pthread_mutex_lock(&q->mu);
  pfbuf *b = q->head ? q_take_locked(q) : NULL;
  pthread_mutex_unlock(&q->mu);
  return b;
}
#endif
static int q_idle(queue *q) {
  pthread_mutex_lock(&q->mu);
  int idle = !q->head && q->busy == 0;
  pthread_mutex_unlock(&q->mu);
  return idle;
}
static void q_close(queue *q) {
  pthread_mutex_lock(&q->mu);
  q->closed = 1;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}
static void fail_at(const char *fmt, ...) {
  pthread_mutex_lock(&g_fail_mu);
  if (!g_failed) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_fail_msg, sizeof g_fail_msg, fmt, ap);
    va_end(ap);
    g_failed = 1;
  }
  pthread_mutex_unlock(&g_fail_mu);
  for (int i = 0; i < 2; i++) {
    pthread_mutex_lock(&pools[i].mu);
    pthread_cond_broadcast(&pools[i].cv);
    pthread_mutex_unlock(&pools[i].mu);
  }
}
static void pool_fill(queue *q, int id, size_t count, size_t bytes) {
  q_init(q);
  for (size_t i = 0; i < count; i++) {
    pfbuf *b = calloc(1, sizeof *b);
    if (posix_memalign((void **)&b->p, 4096, bytes)) die("out of memory for the buffer pool");
    b->pool = id;
    q_push(q, b);
  }
}

/* ------------------------------------------------------------------ fetch engine */

typedef struct {
  const opts *o;
  pf_plan plan;
  uint32_t *queue_ids;
  size_t queue_n;
  uint64_t next_q; /* atomic */
  int readers_alive; /* atomic */
  int partial;
  int *src_fd;
  pthread_mutex_t src_mu;
  uint64_t *dest_left; /* per destination (only this rank's are used); atomic */
  uint64_t *group_left; /* per group, this rank; atomic */
  uint8_t *piece_done;
  queue sendq, writeq, finq;
  uint64_t st_read, st_sent, st_recv, st_written, st_blocks, st_dests_done; /* atomic */
  long sent_msgs; /* atomic, for --inject=corrupt */
  long eio_left;  /* atomic */
} engine;

static engine E;

#define A_ADD(p, v) __atomic_add_fetch((p), (v), __ATOMIC_ACQ_REL)
#define A_SUB(p, v) __atomic_sub_fetch((p), (v), __ATOMIC_ACQ_REL)
#define A_GET(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)

static int write_path_open(const pf_dest *d) {
  char tmp[4200];
  if (d->mode == PF_ATOMIC) snprintf(tmp, sizeof tmp, "%s.pfpart", d->path);
  return open(d->mode == PF_ATOMIC ? tmp : d->path, O_WRONLY);
}

static int src_open(uint32_t s) {
  if (E.src_fd) {
    int fd = A_GET(&E.src_fd[s]);
    if (fd >= 0) return fd;
    pthread_mutex_lock(&E.src_mu);
    fd = E.src_fd[s];
    if (fd < 0) fd = E.src_fd[s] = open(E.plan.srcs[s], O_RDONLY);
    pthread_mutex_unlock(&E.src_mu);
    return fd;
  }
  return open(E.plan.srcs[s], O_RDONLY);
}
static void src_release(int fd) {
  if (!E.src_fd) close(fd);
}

static void mark_group_ready(uint32_t g) {
  if (E.o->local_root) {
    char path[4200];
    snprintf(path, sizeof path, "%s/.pfetch/%s.ready", E.o->local_root, E.plan.groups[g].name);
    mkdir_parent(path);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) close(fd);
  }
  printf("SENTINEL PFETCH_GROUP_READY rank=%d group=%s\n", g_rank, E.plan.groups[g].name);
  fflush(stdout);
}

static void *reader_main(void *arg) {
  (void)arg;
  const uint64_t B = E.plan.block;
  uint8_t *rbuf;
  if (posix_memalign((void **)&rbuf, 4096, B)) die("out of memory");
  for (;;) {
    uint64_t i = A_ADD(&E.next_q, 1) - 1;
    if (i >= E.queue_n || g_failed) break;
    if (E.o->deadline > 0 && (double)time(NULL) >= E.o->deadline) {
      E.partial = 1;
      break;
    }
    const pf_block *b = &E.plan.blocks[E.queue_ids[i]];
    const uint64_t base = b->index * B;
    int fd = src_open(b->src);
    if (fd < 0) {
      fail_at("open %s: %s", E.plan.srcs[b->src], strerror(errno));
      break;
    }
    for (uint32_t k = 0; k < b->nrspan && !g_failed; k++) {
      const pf_span *s = &E.plan.rspans[b->rspan0 + k];
      int rc = -1;
      for (int attempt = 0; attempt < 4; attempt++) {
        if (E.o->inject_eio >= 0 && A_SUB(&E.eio_left, 1) >= 0) rc = -1, errno = EIO;
        else rc = pread_full(fd, rbuf + (s->off - base), s->len, s->off);
        if (rc == 0 || rc == -2) break;
        fprintf(stderr, "pfetch rank %d: read %s at %" PRIu64 ": %s (retry %d)\n", g_rank, E.plan.srcs[b->src], s->off,
                strerror(errno), attempt + 1);
        if (attempt < 3) nap_us(1000000L << attempt);
      }
      if (rc) fail_at("read %s at %" PRIu64 " len %" PRIu64 ": %s", E.plan.srcs[b->src], s->off, s->len,
                      rc == -2 ? "file is shorter than the manifest says" : strerror(errno));
      posix_fadvise(fd, (off_t)s->off, (off_t)s->len, POSIX_FADV_DONTNEED);
    }
    src_release(fd);
    if (g_failed) break;
    A_ADD(&E.st_read, b->read_bytes);
    A_ADD(&E.st_blocks, 1);
    for (uint32_t k = 0; k < b->nmsg; k++) {
      const pf_msg *m = &E.plan.msgs[b->msg0 + k];
      pfbuf *mb = q_pop(&pools[0], 0);
      if (!mb) break;
      uint64_t o = 0;
      for (uint32_t j = 0; j < m->nspan; j++) {
        const pf_span *s = &E.plan.mspans[m->span0 + j];
        memcpy(mb->p + o, rbuf + (s->off - base), s->len);
        o += s->len;
      }
      trailer t;
      memset(&t, 0, sizeof t);
      t.magic = MAGIC;
      t.block = E.queue_ids[i];
      t.src = g_rank;
      t.dst = m->rank;
      t.len = m->bytes;
      t.crc = pf_crc32c(0, mb->p, m->bytes);
      memcpy(mb->p + m->bytes, &t, sizeof t);
      mb->len = (uint32_t)(m->bytes + TRAILER);
      mb->peer = m->rank;
      if (m->rank != g_rank) {
        long n = A_ADD(&E.sent_msgs, 1);
        if (E.o->inject_corrupt >= 0 && n == E.o->inject_corrupt + 1) mb->p[0] ^= 0x5a;
        q_push(&E.sendq, mb);
      } else
        q_push(&E.writeq, mb);
    }
  }
  free(rbuf);
  A_SUB(&E.readers_alive, 1);
  return NULL;
}

static void *writer_main(void *arg) {
  (void)arg;
  pfbuf *mb;
  pthread_mutex_lock(&E.writeq.mu);
  E.writeq.busy++; /* balanced by the first q_pop */
  pthread_mutex_unlock(&E.writeq.mu);
  while ((mb = q_pop(&E.writeq, 1))) {
    trailer t;
    if (mb->len < TRAILER) {
      fail_at("short message (%u bytes)", mb->len);
      break;
    }
    memcpy(&t, mb->p + mb->len - TRAILER, sizeof t);
    if (t.magic != MAGIC || t.dst != g_rank || t.block >= E.plan.nblock || t.len + TRAILER != mb->len) {
      fail_at("malformed message from rank %d", (int)t.src);
      break;
    }
    if (pf_crc32c(0, mb->p, t.len) != t.crc) {
      fail_at("CRC mismatch on block %u from rank %d", t.block, (int)t.src);
      break;
    }
    const pf_block *b = &E.plan.blocks[t.block];
    const pf_msg *m = pf_block_msg(&E.plan, b, g_rank);
    if (!m || m->bytes != t.len) {
      fail_at("unexpected message for block %u", t.block);
      break;
    }
    for (uint32_t k = 0; k < m->npiece; k++) {
      const uint32_t pi = m->piece0 + k;
      const pf_piece *pc = &E.plan.pieces[pi];
      const pf_dest *d = &E.plan.dests[pc->dest];
      int fd = write_path_open(d);
      if (fd < 0 || pwrite_full(fd, mb->p + pc->msg_off, pc->len, pc->dst_off)) {
        fail_at("write %s: %s", d->path, strerror(errno));
        if (fd >= 0) close(fd);
        break;
      }
      close(fd);
      E.piece_done[pi] = 1;
      A_ADD(&E.st_written, pc->len);
      if (A_SUB(&E.group_left[pc->group], pc->len) == 0) mark_group_ready(pc->group);
      if (A_SUB(&E.dest_left[pc->dest], pc->len) == 0) {
        pfbuf *f = calloc(1, sizeof *f);
        f->peer = (int)pc->dest;
        f->pool = -1;
        q_push(&E.finq, f);
      }
    }
    q_push(&pools[mb->pool], mb);
  }
  return NULL;
}

static void finish_dest(uint32_t di) {
  const pf_dest *d = &E.plan.dests[di];
  char tmp[4200];
  snprintf(tmp, sizeof tmp, "%s.pfpart", d->path);
  const char *cur = d->mode == PF_ATOMIC ? tmp : d->path;
  int fd = open(cur, O_RDONLY);
  if (fd < 0) {
    fail_at("finish %s: %s", cur, strerror(errno));
    return;
  }
  if (E.o->dest_sync && fdatasync(fd)) {
    fail_at("fdatasync %s: %s", cur, strerror(errno));
    close(fd);
    return;
  }
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  if (d->mode == PF_ATOMIC && rename(tmp, d->path)) {
    fail_at("rename %s: %s", tmp, strerror(errno));
    return;
  }
  if (d->sidecar) {
    char hex[65], stmp[4200];
    if (pf_sha256_file(d->path, hex)) {
      fail_at("hash %s: %s", d->path, strerror(errno));
      return;
    }
    snprintf(stmp, sizeof stmp, "%s.pfpart", d->sidecar);
    FILE *f = fopen(stmp, "w");
    if (!f || fprintf(f, "%s\n", hex) < 0 || fflush(f) || fsync(fileno(f)) || fclose(f) || rename(stmp, d->sidecar)) {
      fail_at("sidecar %s: %s", d->sidecar, strerror(errno));
      return;
    }
  }
  A_ADD(&E.st_dests_done, 1);
}

static void *finisher_main(void *arg) {
  (void)arg;
  pfbuf *f;
  pthread_mutex_lock(&E.finq.mu);
  E.finq.busy++;
  pthread_mutex_unlock(&E.finq.mu);
  while ((f = q_pop(&E.finq, 1))) {
    finish_dest((uint32_t)f->peer);
    free(f);
  }
  return NULL;
}

static void make_link(const pf_link *l) {
  char tmp[4200];
  snprintf(tmp, sizeof tmp, "%s.pflink", l->link);
  mkdir_parent(l->link);
  unlink(tmp);
  if (symlink(l->target, tmp) || rename(tmp, l->link)) die("symlink %s -> %s: %s", l->link, l->target, strerror(errno));
}

static int dest_complete_already(const pf_dest *d) {
  struct stat st;
  if (d->mode != PF_ATOMIC || stat(d->path, &st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size != d->size) return 0;
  return !d->sidecar || access(d->sidecar, F_OK) == 0;
}

/* Exchange which destinations their owners already hold, so every rank plans the same remaining work. */
static void share_skips(pf_plan *p) {
  uint32_t *mine = malloc((p->ndest ? p->ndest : 1) * sizeof *mine);
  int n = 0;
  for (uint32_t d = 0; d < p->ndest; d++)
    if (p->dests[d].rank == g_rank && dest_complete_already(&p->dests[d])) mine[n++] = d;
#ifndef PFETCH_NO_MPI
  int *counts = malloc((size_t)g_ranks * sizeof *counts), *displs = malloc((size_t)g_ranks * sizeof *displs);
  MPI_Allgather(&n, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  int total = 0;
  for (int r = 0; r < g_ranks; r++) displs[r] = total, total += counts[r];
  uint32_t *all = malloc((size_t)(total ? total : 1) * sizeof *all);
  MPI_Allgatherv(mine, n, MPI_UINT32_T, all, counts, displs, MPI_UINT32_T, MPI_COMM_WORLD);
  for (int i = 0; i < total; i++) p->dests[all[i]].skip = 1;
  free(all), free(counts), free(displs);
#else
  for (int i = 0; i < n; i++) p->dests[mine[i]].skip = 1;
#endif
  free(mine);
}

static void write_status(const char *state, double secs) {
  if (!E.o->status_dir) return;
  char path[4200];
  snprintf(path, sizeof path, "%s/pfetch.rank%02d.status", E.o->status_dir, g_rank);
  mkdir_p(E.o->status_dir);
  FILE *f = fopen(path, "w");
  if (!f) return;
  fprintf(f, "PFETCH_RANK_END rank=%d state=%s read_bytes=%" PRIu64 " written_bytes=%" PRIu64 " seconds=%.1f read_MBps=%.1f\n",
          g_rank, state, E.st_read, E.st_written, secs, secs > 0 ? E.st_read / secs / 1e6 : 0.0);
  fclose(f);
}

static void write_coverage(void) {
  if (!E.o->coverage_out) return;
  char tmp[4300], err[512];
  char *path = pf_expand(E.o->coverage_out, E.o->local_root, g_rank, err, sizeof err);
  if (!path) die("--coverage-out: %s", err);
  snprintf(tmp, sizeof tmp, "%s.pfpart", path);
  mkdir_parent(path);
  FILE *f = fopen(tmp, "w");
  if (!f) die("coverage %s: %s", tmp, strerror(errno));
  /* pieces in destination order, adjacent ranges merged */
  size_t n = 0;
  uint32_t *ids = malloc((E.plan.npiece ? E.plan.npiece : 1) * sizeof *ids);
  for (size_t i = 0; i < E.plan.npiece; i++)
    if (E.piece_done[i]) ids[n++] = (uint32_t)i;
  /* sort ids by (dest, dst_off): simple insertion into buckets per dest via counting */
  uint32_t *cnt = calloc(E.plan.ndest + 1, sizeof *cnt);
  for (size_t i = 0; i < n; i++) cnt[E.plan.pieces[ids[i]].dest + 1]++;
  for (uint32_t d = 0; d < E.plan.ndest; d++) cnt[d + 1] += cnt[d];
  uint32_t *by = malloc((n ? n : 1) * sizeof *by), *fill = calloc(E.plan.ndest, sizeof *fill);
  for (size_t i = 0; i < n; i++) {
    uint32_t d = E.plan.pieces[ids[i]].dest;
    by[cnt[d] + fill[d]++] = ids[i];
  }
  for (uint32_t d = 0; d < E.plan.ndest; d++) {
    uint32_t a = cnt[d], z = cnt[d + 1];
    for (uint32_t i = a + 1; i < z; i++) { /* insertion sort by dst_off; pieces of one dest are few per block */
      uint32_t v = by[i], j = i;
      while (j > a && E.plan.pieces[by[j - 1]].dst_off > E.plan.pieces[v].dst_off) by[j] = by[j - 1], j--;
      by[j] = v;
    }
    uint64_t off = 0, len = 0;
    for (uint32_t i = a; i < z; i++) {
      const pf_piece *pc = &E.plan.pieces[by[i]];
      if (len && pc->dst_off == off + len) len += pc->len;
      else {
        if (len) fprintf(f, "%s %" PRIu64 " %" PRIu64 "\n", E.plan.dests[d].path, off, len);
        off = pc->dst_off, len = pc->len;
      }
    }
    if (len) fprintf(f, "%s %" PRIu64 " %" PRIu64 "\n", E.plan.dests[d].path, off, len);
  }
  free(ids), free(cnt), free(by), free(fill);
  if (fflush(f) || fsync(fileno(f)) || fclose(f) || rename(tmp, path)) die("coverage %s: %s", path, strerror(errno));
  free(path);
}

static int run_fetch(const opts *o) {
  memset(&E, 0, sizeof E);
  E.o = o;
  const double t0 = now_s();
  raise_nofile();
  size_t mlen = 0;
  char *text = NULL;
  if (g_rank == 0 && !(text = read_file(o->manifest, &mlen))) die("cannot read manifest %s", o->manifest);
  uint64_t len64 = mlen;
  bcast(&len64, sizeof len64, 0);
  if (g_rank) text = malloc(len64 + 1);
  bcast(text, len64, 0);
  char err[512];
  if (pf_parse(&E.plan, text, len64, g_ranks, o->local_root, err, sizeof err)) die("manifest: %s", err);
  free(text);
  for (uint32_t d = 0; d < E.plan.ndest; d++)
    if (E.plan.dests[d].rank == g_rank && mkdir_parent(E.plan.dests[d].path)) die("mkdir for %s: %s", E.plan.dests[d].path, strerror(errno));
  share_skips(&E.plan);
  if (pf_build(&E.plan, o->assign, err, sizeof err)) die("plan: %s", err);
  const pf_plan *P = &E.plan;

  /* destinations: create at full size */
  uint32_t mine = 0, skipped = 0;
  E.dest_left = calloc(P->ndest ? P->ndest : 1, sizeof *E.dest_left);
  E.group_left = calloc(P->ngroup, sizeof *E.group_left);
  E.piece_done = calloc(P->npiece ? P->npiece : 1, 1);
  for (uint32_t d = 0; d < P->ndest; d++) {
    const pf_dest *x = &P->dests[d];
    if (x->rank != g_rank) continue;
    mine++;
    if (x->skip) {
      skipped++;
      continue;
    }
    char tmp[4200];
    snprintf(tmp, sizeof tmp, "%s.pfpart", x->path);
    const char *path = x->mode == PF_ATOMIC ? tmp : x->path;
    unlink(path); /* never write through a stale symlink or into an older file */
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, (off_t)x->size)) die("create %s: %s", path, strerror(errno));
    close(fd);
    E.dest_left[d] = x->covered;
  }
  for (size_t i = 0; i < P->npiece; i++)
    if (P->pieces[i].rank == g_rank) E.group_left[P->pieces[i].group] += P->pieces[i].len;
  for (uint32_t l = 0; l < P->nlink; l++)
    if (P->links[l].rank == g_rank) make_link(&P->links[l]);

  E.queue_ids = malloc((P->nblock ? P->nblock : 1) * sizeof *E.queue_ids);
  E.queue_n = pf_queue(P, g_rank, o->sequential, E.queue_ids);
  uint64_t expected_recv = 0, my_read = 0;
  for (size_t b = 0; b < P->nblock; b++) {
    if (P->blocks[b].reader == g_rank) my_read += P->blocks[b].read_bytes;
    else if (pf_block_msg(P, &P->blocks[b], g_rank)) expected_recv++;
  }
  if (P->nsrc <= 512) {
    E.src_fd = malloc(P->nsrc * sizeof *E.src_fd);
    for (uint32_t s = 0; s < P->nsrc; s++) E.src_fd[s] = -1;
  }
  pthread_mutex_init(&E.src_mu, NULL);
  E.eio_left = o->inject_eio;
  const size_t msg_bytes = P->block + TRAILER;
  size_t npool = o->pool_mib * (1u << 20) / msg_bytes;
  if (npool < (size_t)o->streams + 2) npool = (size_t)o->streams + 2;
  pool_fill(&pools[0], 0, npool, msg_bytes);
  pool_fill(&pools[1], 1, (size_t)o->recv_slots, msg_bytes);
  q_init(&E.sendq), q_init(&E.writeq), q_init(&E.finq);
  if (g_rank == 0)
    printf("SENTINEL PFETCH_BEGIN ranks=%d blocks=%zu read_bytes=%" PRIu64 " needed_bytes=%" PRIu64 " dests=%u links=%u "
           "streams=%d assign=%s order=%s block=%" PRIu64 "\n",
           g_ranks, P->nblock, P->total_read, P->total_needed, P->ndest, P->nlink, o->streams,
           o->assign == PF_ASSIGN_SPREAD ? "spread" : "local", o->sequential ? "sequential" : "interleave", P->block);
  printf("PFETCH_RANK rank=%d dests=%u skipped=%u blocks=%zu read_bytes=%" PRIu64 " expect_msgs=%" PRIu64 "\n", g_rank,
         mine, skipped, E.queue_n, my_read, expected_recv);
  fflush(stdout);
  for (uint32_t g = 0; g < P->ngroup; g++)
    if (E.group_left[g] == 0) mark_group_ready(g);
  for (uint32_t d = 0; d < P->ndest; d++)
    if (P->dests[d].rank == g_rank && !P->dests[d].skip && E.dest_left[d] == 0) {
      pfbuf *f = calloc(1, sizeof *f);
      f->peer = (int)d;
      q_push(&E.finq, f);
    }

  const int nthreads = o->streams + o->writers + o->finishers;
  pthread_t *th = malloc((size_t)nthreads * sizeof *th);
  E.readers_alive = o->streams;
  int ti = 0;
  for (int i = 0; i < o->streams; i++) pthread_create(&th[ti++], NULL, reader_main, NULL);
  for (int i = 0; i < o->writers; i++) pthread_create(&th[ti++], NULL, writer_main, NULL);
  for (int i = 0; i < o->finishers; i++) pthread_create(&th[ti++], NULL, finisher_main, NULL);

  double last_p = now_s();
  uint64_t last_read = 0;
#ifndef PFETCH_NO_MPI
  const int R = g_ranks;
  MPI_Request *sreq = malloc((size_t)o->send_window * sizeof *sreq), *rreq = malloc((size_t)o->recv_slots * sizeof *rreq);
  pfbuf **sbuf = calloc((size_t)o->send_window, sizeof *sbuf), **rbuf = calloc((size_t)o->recv_slots, sizeof *rbuf);
  for (int i = 0; i < o->send_window; i++) sreq[i] = MPI_REQUEST_NULL;
  for (int i = 0; i < o->recv_slots; i++) rreq[i] = MPI_REQUEST_NULL;
  uint64_t *sent_to = calloc((size_t)R, sizeof *sent_to), *recv_from = calloc((size_t)R, sizeof *recv_from);
  uint64_t *announced = calloc((size_t)R, sizeof *announced), *done_in = calloc((size_t)R, sizeof *done_in);
  int *got_done = calloc((size_t)R, sizeof *got_done);
  MPI_Request *dreq_out = malloc((size_t)R * sizeof *dreq_out), *dreq_in = malloc((size_t)R * sizeof *dreq_in);
  for (int r = 0; r < R; r++) dreq_out[r] = dreq_in[r] = MPI_REQUEST_NULL;
  for (int k = 0; k < R - 1; k++) MPI_Irecv(&done_in[k], 1, MPI_UINT64_T, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &dreq_in[k]);
  int ndone_in = 0, announced_out = 0, posted = 0;
  uint64_t received = 0;
  uint64_t stat_out[6], *stat_all = g_rank == 0 ? calloc((size_t)R * 6, sizeof *stat_all) : NULL, stat_in[6];
  MPI_Request stat_sreq = MPI_REQUEST_NULL, stat_rreq = MPI_REQUEST_NULL;
  uint64_t stats_sent = 0, stats_got = 0, agg_last = 0;
  double agg_t = now_s();
  if (g_rank == 0 && R > 1) MPI_Irecv(stat_in, 6, MPI_UINT64_T, MPI_ANY_SOURCE, TAG_STAT, MPI_COMM_WORLD, &stat_rreq);
  for (;;) {
    int progress = 0, flag;
    MPI_Status st;
    for (int i = 0; i < o->send_window; i++)
      if (sbuf[i]) {
        MPI_Test(&sreq[i], &flag, MPI_STATUS_IGNORE);
        if (flag) {
          A_ADD(&E.st_sent, sbuf[i]->len - TRAILER);
          q_push(&pools[0], sbuf[i]);
          sbuf[i] = NULL;
          progress = 1;
        }
      }
    for (int i = 0; i < o->send_window; i++)
      if (!sbuf[i]) {
        pfbuf *b = q_trypop(&E.sendq);
        if (!b) break;
        MPI_Isend(b->p, (int)b->len, MPI_BYTE, b->peer, TAG_DATA, MPI_COMM_WORLD, &sreq[i]);
        sent_to[b->peer]++;
        sbuf[i] = b;
        progress = 1;
      }
    for (int i = 0; i < o->recv_slots; i++)
      if (rbuf[i]) {
        MPI_Test(&rreq[i], &flag, &st);
        if (flag) {
          int count;
          MPI_Get_count(&st, MPI_BYTE, &count);
          rbuf[i]->len = (uint32_t)count;
          recv_from[st.MPI_SOURCE]++;
          received++;
          posted--;
          A_ADD(&E.st_recv, (uint64_t)count - TRAILER);
          q_push(&E.writeq, rbuf[i]);
          rbuf[i] = NULL;
          progress = 1;
        }
      }
    for (int i = 0; i < o->recv_slots && posted + received < expected_recv; i++)
      if (!rbuf[i]) {
        pfbuf *b = q_trypop(&pools[1]);
        if (!b) break;
        MPI_Irecv(b->p, (int)msg_bytes, MPI_BYTE, MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &rreq[i]);
        rbuf[i] = b;
        posted++;
      }
    if (!announced_out && A_GET(&E.readers_alive) == 0 && q_idle(&E.sendq)) {
      int busy = 0;
      for (int i = 0; i < o->send_window; i++) busy |= sbuf[i] != NULL;
      if (!busy) {
        for (int r = 0; r < R; r++)
          if (r != g_rank) MPI_Isend(&sent_to[r], 1, MPI_UINT64_T, r, TAG_DONE, MPI_COMM_WORLD, &dreq_out[r]);
        announced_out = 1;
      }
    }
    for (int k = 0; k < R - 1; k++)
      if (dreq_in[k] != MPI_REQUEST_NULL) {
        MPI_Test(&dreq_in[k], &flag, &st);
        if (flag) {
          got_done[st.MPI_SOURCE] = 1;
          announced[st.MPI_SOURCE] = done_in[k];
          ndone_in++;
          progress = 1;
        }
      }
    int recv_complete = ndone_in == R - 1;
    for (int r = 0; r < R && recv_complete; r++)
      if (r != g_rank && recv_from[r] != announced[r]) recv_complete = 0;
    if (g_rank == 0 && stat_rreq != MPI_REQUEST_NULL) {
      MPI_Test(&stat_rreq, &flag, &st);
      if (flag) {
        memcpy(&stat_all[6 * st.MPI_SOURCE], stat_in, sizeof stat_in);
        stats_got++;
        MPI_Irecv(stat_in, 6, MPI_UINT64_T, MPI_ANY_SOURCE, TAG_STAT, MPI_COMM_WORLD, &stat_rreq);
      }
    }
    if (g_failed) die("%s", g_fail_msg);
    if (announced_out && recv_complete && q_idle(&E.writeq) && q_idle(&E.finq)) break;
    const double t = now_s();
    if (t - last_p >= o->progress_s) {
      uint64_t rd = A_GET(&E.st_read);
      printf("PFETCH_PROGRESS rank=%d read_MBps=%.1f read_GB=%.2f/%.2f sent_GB=%.2f recv_GB=%.2f written_GB=%.2f blocks=%" PRIu64 "/%zu\n",
             g_rank, (rd - last_read) / (t - last_p) / 1e6, rd / 1e9, my_read / 1e9, A_GET(&E.st_sent) / 1e9,
             A_GET(&E.st_recv) / 1e9, A_GET(&E.st_written) / 1e9, A_GET(&E.st_blocks), E.queue_n);
      fflush(stdout);
      last_p = t, last_read = rd;
      stat_out[0] = rd, stat_out[1] = A_GET(&E.st_sent), stat_out[2] = A_GET(&E.st_recv), stat_out[3] = A_GET(&E.st_written);
      stat_out[4] = my_read, stat_out[5] = (uint64_t)((t - t0) * 1000);
      if (g_rank == 0) {
        memcpy(stat_all, stat_out, sizeof stat_out);
        uint64_t sum = 0, need = 0;
        for (int r = 0; r < R; r++) sum += stat_all[6 * r], need += stat_all[6 * r + 4];
        printf("PFETCH_AGG agg_MBps=%.1f read_GB=%.2f/%.2f elapsed_s=%.0f\n", (sum - agg_last) / (t - agg_t) / 1e6, sum / 1e9,
               need / 1e9, t - t0);
        fflush(stdout);
        agg_last = sum, agg_t = t;
      } else {
        flag = 1;
        if (stat_sreq != MPI_REQUEST_NULL) MPI_Test(&stat_sreq, &flag, MPI_STATUS_IGNORE);
        if (flag) {
          static uint64_t sbufstat[6];
          memcpy(sbufstat, stat_out, sizeof sbufstat);
          MPI_Isend(sbufstat, 6, MPI_UINT64_T, 0, TAG_STAT, MPI_COMM_WORLD, &stat_sreq);
          stats_sent++;
        }
      }
    }
    if (!progress) nap_us(100);
  }
  for (int i = 0; i < o->recv_slots; i++)
    if (rbuf[i]) {
      MPI_Cancel(&rreq[i]);
      MPI_Wait(&rreq[i], MPI_STATUS_IGNORE);
    }
  MPI_Waitall(R, dreq_out, MPI_STATUSES_IGNORE);
  if (stat_sreq != MPI_REQUEST_NULL) MPI_Wait(&stat_sreq, MPI_STATUS_IGNORE);
  uint64_t total_stats = stats_sent;
  MPI_Reduce(g_rank == 0 ? MPI_IN_PLACE : &total_stats, &total_stats, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  if (g_rank == 0 && stat_rreq != MPI_REQUEST_NULL) {
    while (stats_got < total_stats) {
      MPI_Wait(&stat_rreq, MPI_STATUS_IGNORE);
      if (++stats_got < total_stats) MPI_Irecv(stat_in, 6, MPI_UINT64_T, MPI_ANY_SOURCE, TAG_STAT, MPI_COMM_WORLD, &stat_rreq);
      else stat_rreq = MPI_REQUEST_NULL;
    }
    if (stat_rreq != MPI_REQUEST_NULL) {
      MPI_Cancel(&stat_rreq);
      MPI_Wait(&stat_rreq, MPI_STATUS_IGNORE);
    }
  }
#else
  for (;;) {
    if (g_failed) die("%s", g_fail_msg);
    if (A_GET(&E.readers_alive) == 0 && q_idle(&E.writeq) && q_idle(&E.finq)) break;
    const double t = now_s();
    if (t - last_p >= o->progress_s) {
      uint64_t rd = A_GET(&E.st_read);
      printf("PFETCH_PROGRESS rank=0 read_MBps=%.1f read_GB=%.2f/%.2f written_GB=%.2f\n", (rd - last_read) / (t - last_p) / 1e6,
             rd / 1e9, my_read / 1e9, A_GET(&E.st_written) / 1e9);
      fflush(stdout);
      last_p = t, last_read = rd;
    }
    nap_us(1000);
  }
#endif
  q_close(&E.writeq), q_close(&E.finq), q_close(&E.sendq);
  for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
  if (g_failed) die("%s", g_fail_msg);
  const double secs = now_s() - t0;
  write_coverage();
  int partial = E.partial;
  uint64_t tot[4] = {E.st_read, E.st_written, (uint64_t)partial, E.st_dests_done};
  allsum_u64(tot, 4);
  double tmax[1] = {secs};
  allmax_d(tmax, 1);
  double *rates = g_rank == 0 ? malloc((size_t)g_ranks * sizeof *rates) : NULL;
  gather_d(secs > 0 ? E.st_read / secs / 1e6 : 0, rates);
  const char *state = tot[2] ? "partial" : "ok";
  write_status(tot[2] ? "partial" : "ok", secs);
  printf("PFETCH_RANK_END rank=%d state=%s read_bytes=%" PRIu64 " written_bytes=%" PRIu64 " seconds=%.1f read_MBps=%.1f\n", g_rank,
         partial ? "partial" : "ok", E.st_read, E.st_written, secs, secs > 0 ? E.st_read / secs / 1e6 : 0.0);
  if (g_rank == 0) {
    double mn = 1e30, mx = 0;
    for (int r = 0; r < g_ranks; r++) mn = rates[r] < mn ? rates[r] : mn, mx = rates[r] > mx ? rates[r] : mx;
    printf("SENTINEL PFETCH_END state=%s ranks=%d read_bytes=%" PRIu64 " written_bytes=%" PRIu64 " dests_done=%" PRIu64
           " seconds=%.1f agg_MBps=%.1f rank_MBps_min=%.1f rank_MBps_max=%.1f\n",
           state, g_ranks, tot[0], tot[1], tot[3], tmax[0], tmax[0] > 0 ? tot[0] / tmax[0] / 1e6 : 0.0, mn, mx);
  }
  fflush(stdout);
  free(rates);
  return tot[2] ? 2 : 0;
}

/* ------------------------------------------------------------------ plan dump (tests and what-if) */

static int run_plan_dump(const opts *o) {
  size_t n;
  char *text = read_file(o->manifest, &n), err[512];
  if (!text) die("cannot read manifest %s", o->manifest);
  pf_plan p;
  const int R = o->dump_ranks > 0 ? o->dump_ranks : 1;
  if (pf_parse(&p, text, n, R, o->local_root ? o->local_root : "/L", err, sizeof err) || pf_build(&p, o->assign, err, sizeof err)) {
    fprintf(stderr, "plan: %s\n", err);
    return 1;
  }
  uint64_t remote = 0;
  for (size_t b = 0; b < p.nblock; b++)
    for (uint32_t k = 0; k < p.blocks[b].nmsg; k++)
      if (p.msgs[p.blocks[b].msg0 + k].rank != p.blocks[b].reader) remote += p.msgs[p.blocks[b].msg0 + k].bytes;
  printf("PLAN ranks=%d blocks=%zu read_bytes=%" PRIu64 " needed_bytes=%" PRIu64 " msgs=%zu remote_bytes=%" PRIu64 " srcs=%u dests=%u\n", R,
         p.nblock, p.total_read, p.total_needed, p.nmsg, remote, p.nsrc, p.ndest);
  uint32_t *q = malloc((p.nblock ? p.nblock : 1) * sizeof *q);
  for (int r = 0; r < R; r++) {
    size_t qn = pf_queue(&p, r, o->sequential, q);
    printf("LOAD rank=%d read=%" PRIu64 " blocks=%zu first=", r, p.load[r], qn);
    for (size_t i = 0; i < qn && i < (size_t)o->streams; i++) printf("%s%u", i ? "," : "", p.blocks[q[i]].src);
    printf("\n");
  }
  for (size_t b = 0; b < p.nblock; b++) {
    const pf_block *x = &p.blocks[b];
    printf("BLOCK %zu src=%u index=%" PRIu64 " reader=%d read=%" PRIu64 " prio=%d msgs=", b, x->src, x->index, x->reader, x->read_bytes, x->prio);
    for (uint32_t k = 0; k < x->nmsg; k++) printf("%s%d:%" PRIu64, k ? "," : "", p.msgs[x->msg0 + k].rank, p.msgs[x->msg0 + k].bytes);
    printf("\n");
  }
  for (uint32_t s = 0; s < p.nsrc; s++) printf("SRC %u %s\n", s, p.srcs[s]);
  free(q);
  free(text);
  pf_free(&p);
  return 0;
}

/* ------------------------------------------------------------------ write direction (strata-pput) */

typedef struct {
  char *src, *dst;
  long after;
  uint64_t size, left;
  int data_done, renamed, skipped;
  uint32_t *crc;
} put_file;
typedef struct { uint32_t file; uint64_t off, len; } put_block;

static struct {
  const opts *o;
  put_file *f;
  size_t nf;
  put_block *b;
  size_t nb;
  uint64_t next; /* atomic */
  pthread_mutex_t mu;
  uint64_t bytes;
  int failed;
  char msg[512];
} W;

static void put_fail(const char *fmt, ...) {
  pthread_mutex_lock(&W.mu);
  if (!W.failed) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(W.msg, sizeof W.msg, fmt, ap);
    va_end(ap);
    W.failed = 1;
  }
  pthread_mutex_unlock(&W.mu);
}

/* Rename every data-complete file whose predecessor is in place; called with W.mu held. */
static void put_publish_locked(void) {
  for (int again = 1; again && !W.failed;) {
    again = 0;
    for (size_t i = 0; i < W.nf; i++) {
      put_file *f = &W.f[i];
      if (f->renamed || !f->data_done || (f->after >= 0 && !W.f[f->after].renamed)) continue;
      char tmp[4200];
      snprintf(tmp, sizeof tmp, "%s.pfpart", f->dst);
      int fd = open(tmp, O_RDONLY);
      if (fd < 0 || fdatasync(fd)) {
        put_fail("sync %s: %s", tmp, strerror(errno));
        if (fd >= 0) close(fd);
        return;
      }
      if (W.o->readback) {
        uint8_t *buf = malloc(W.o->block_mib << 20);
        for (size_t k = 0; k < W.nb && !W.failed; k++)
          if (W.b[k].file == i) {
            posix_fadvise(fd, (off_t)W.b[k].off, (off_t)W.b[k].len, POSIX_FADV_DONTNEED);
            if (pread_full(fd, buf, W.b[k].len, W.b[k].off) || pf_crc32c(0, buf, W.b[k].len) != f->crc[k])
              put_fail("readback mismatch in %s at %" PRIu64, tmp, W.b[k].off);
          }
        free(buf);
      }
      close(fd);
      if (W.failed) return;
      if (rename(tmp, f->dst)) {
        put_fail("rename %s: %s", tmp, strerror(errno));
        return;
      }
      f->renamed = 1;
      again = 1;
    }
  }
}

static void *put_main(void *arg) {
  (void)arg;
  uint8_t *buf;
  if (posix_memalign((void **)&buf, 4096, W.o->block_mib << 20)) return NULL;
  for (;;) {
    uint64_t k = A_ADD(&W.next, 1) - 1;
    if (k >= W.nb || W.failed) break;
    put_block *b = &W.b[k];
    put_file *f = &W.f[b->file];
    char tmp[4200];
    snprintf(tmp, sizeof tmp, "%s.pfpart", f->dst);
    int in = open(f->src, O_RDONLY), out = open(tmp, O_WRONLY);
    if (in < 0 || out < 0 || pread_full(in, buf, b->len, b->off) || pwrite_full(out, buf, b->len, b->off)) {
      put_fail("copy %s -> %s at %" PRIu64 ": %s", f->src, tmp, b->off, strerror(errno));
      if (in >= 0) close(in);
      if (out >= 0) close(out);
      break;
    }
    posix_fadvise(in, (off_t)b->off, (off_t)b->len, POSIX_FADV_DONTNEED);
    close(in), close(out);
    if (W.o->readback) f->crc[k] = pf_crc32c(0, buf, b->len);
    pthread_mutex_lock(&W.mu);
    W.bytes += b->len;
    f->left -= b->len;
    if (f->left == 0) {
      f->data_done = 1;
      put_publish_locked();
    }
    pthread_mutex_unlock(&W.mu);
  }
  free(buf);
  return NULL;
}

static int run_put(const opts *o) {
  memset(&W, 0, sizeof W);
  W.o = o;
  pthread_mutex_init(&W.mu, NULL);
  const double t0 = now_s();
  size_t n;
  char *text = read_file(o->write_list, &n);
  if (!text) die("cannot read %s", o->write_list);
  size_t cap = 0, capb = 0;
  for (char *line = strtok(text, "\n"); line; line = strtok(NULL, "\n")) {
    char src[4096], dst[4096], extra[64] = "";
    if (line[0] == '#' || line[0] == 0) continue;
    if (sscanf(line, "W %4095s %4095s %63s", src, dst, extra) < 2) die("bad write line: %s", line);
    if (W.nf == cap) W.f = realloc(W.f, (cap = cap ? cap * 2 : 64) * sizeof *W.f);
    put_file *f = &W.f[W.nf];
    memset(f, 0, sizeof *f);
    f->src = strdup(src), f->dst = strdup(dst), f->after = -1;
    if (extra[0]) {
      if (strncmp(extra, "after=", 6)) die("bad field %s", extra);
      f->after = atol(extra + 6);
      if (f->after < 0 || (size_t)f->after >= W.nf) die("after=%ld must name an earlier line", f->after);
    }
    struct stat st, ds;
    if (stat(src, &st)) die("stat %s: %s", src, strerror(errno));
    f->size = f->left = (uint64_t)st.st_size;
    if (o->skip_existing && stat(dst, &ds) == 0 && ds.st_size == st.st_size) f->skipped = f->renamed = f->data_done = 1;
    W.nf++;
  }
  const uint64_t B = (uint64_t)o->block_mib << 20;
  for (size_t i = 0; i < W.nf; i++) {
    put_file *f = &W.f[i];
    if (f->skipped) continue;
    char tmp[4200];
    snprintf(tmp, sizeof tmp, "%s.pfpart", f->dst);
    mkdir_parent(f->dst);
    unlink(tmp);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, (off_t)f->size)) die("create %s: %s", tmp, strerror(errno));
    close(fd);
    for (uint64_t off = 0; off < f->size; off += B) {
      if (W.nb == capb) W.b = realloc(W.b, (capb = capb ? capb * 2 : 256) * sizeof *W.b);
      W.b[W.nb].file = (uint32_t)i, W.b[W.nb].off = off, W.b[W.nb].len = f->size - off < B ? f->size - off : B;
      W.nb++;
    }
    if (f->size == 0) f->data_done = 1;
  }
  for (size_t i = 0; i < W.nf; i++) W.f[i].crc = o->readback ? calloc(W.nb ? W.nb : 1, sizeof(uint32_t)) : NULL;
  pthread_mutex_lock(&W.mu);
  put_publish_locked();
  pthread_mutex_unlock(&W.mu);
  pthread_t *th = malloc((size_t)o->streams * sizeof *th);
  for (int i = 0; i < o->streams; i++) pthread_create(&th[i], NULL, put_main, NULL);
  for (int i = 0; i < o->streams; i++) pthread_join(th[i], NULL);
  pthread_mutex_lock(&W.mu);
  put_publish_locked();
  pthread_mutex_unlock(&W.mu);
  size_t renamed = 0, skipped = 0;
  for (size_t i = 0; i < W.nf; i++) renamed += W.f[i].renamed, skipped += W.f[i].skipped;
  const double secs = now_s() - t0;
  if (W.failed || renamed != W.nf) {
    fprintf(stderr, "pput: %s\n", W.failed ? W.msg : "not every file was published");
    printf("SENTINEL PPUT_END state=failed files=%zu published=%zu\n", W.nf, renamed);
    return 1;
  }
  printf("SENTINEL PPUT_END state=ok files=%zu skipped=%zu bytes=%" PRIu64 " seconds=%.1f MBps=%.1f streams=%d\n", W.nf, skipped,
         W.bytes, secs, secs > 0 ? W.bytes / secs / 1e6 : 0.0, o->streams);
  free(text);
  return 0;
}

/* ------------------------------------------------------------------ measurement mode */

typedef struct { char *path; uint64_t size; uint64_t cursor; } mfile;
typedef struct {
  const char *path;
  uint64_t off, len, req;
  int write;
  double deadline;
  uint64_t bytes;
  double secs;
} mjob;

static void *mjob_main(void *arg) {
  mjob *j = arg;
  const double t0 = now_s();
  uint8_t *buf;
  if (posix_memalign((void **)&buf, 4096, j->req)) return NULL;
  memset(buf, 0xa5, j->req);
  int fd = open(j->path, j->write ? O_WRONLY : O_RDONLY);
  if (fd >= 0) {
    for (uint64_t o = 0; o < j->len && now_s() < j->deadline; o += j->req) {
      uint64_t n = j->len - o < j->req ? j->len - o : j->req;
      if ((j->write ? pwrite_full(fd, buf, n, j->off + o) : pread_full(fd, buf, n, j->off + o))) break;
      if (!j->write) posix_fadvise(fd, (off_t)(j->off + o), (off_t)n, POSIX_FADV_DONTNEED);
      j->bytes += n;
    }
    if (j->write) fdatasync(fd);
    close(fd);
  }
  free(buf);
  j->secs = now_s() - t0;
  return NULL;
}

/* Run jobs concurrently on this rank; report bytes and the wall time of the slowest. */
static void mrun(mjob *jobs, int n, uint64_t *bytes, double *secs) {
  pthread_t *th = malloc((size_t)n * sizeof *th);
  barrier();
  const double t0 = now_s();
  for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, mjob_main, &jobs[i]);
  for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
  *secs = now_s() - t0;
  *bytes = 0;
  for (int i = 0; i < n; i++) *bytes += jobs[i].bytes;
  free(th);
}

static int cmp_d(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static void mreport(const char *kind, const char *layout, int streams, uint64_t req, int active, uint64_t bytes, double secs,
                    const char *extra) {
  uint64_t tot[1] = {bytes};
  allsum_u64(tot, 1);
  double wall[1] = {secs};
  allmax_d(wall, 1);
  double *all = g_rank == 0 ? malloc((size_t)g_ranks * sizeof *all) : NULL;
  gather_d(secs > 0 && bytes ? bytes / secs / 1e6 : 0, all);
  if (g_rank == 0) {
    double *act = malloc((size_t)g_ranks * sizeof *act);
    int na = 0;
    for (int r = 0; r < g_ranks; r++)
      if (all[r] > 0) act[na++] = all[r];
    qsort(act, (size_t)na, sizeof *act, cmp_d);
    printf("PFETCH_MEASURE kind=%s layout=%s streams=%d req_mib=%" PRIu64 " ranks=%d seconds=%.1f GB=%.2f agg_MBps=%.1f "
           "rank_MBps_min=%.1f rank_MBps_median=%.1f rank_MBps_max=%.1f%s%s\n",
           kind, layout, streams, req >> 20, active, wall[0], tot[0] / 1e9, wall[0] > 0 ? tot[0] / wall[0] / 1e6 : 0.0,
           na ? act[0] : 0.0, na ? act[na / 2] : 0.0, na ? act[na - 1] : 0.0, extra ? " " : "", extra ? extra : "");
    fflush(stdout);
    free(act);
  }
  free(all);
}

static int parse_list(const char *s, int *out, int max) {
  int n = 0;
  for (const char *p = s; *p && n < max;) {
    out[n++] = atoi(p);
    p = strchr(p, ',');
    if (!p) break;
    p++;
  }
  return n;
}

static void measure_read(const opts *o) {
  DIR *dir = opendir(o->src_dir);
  if (!dir) die("--src-dir %s: %s", o->src_dir ? o->src_dir : "(none)", strerror(errno));
  mfile *f = NULL;
  int nf = 0, cap = 0;
  struct dirent *de;
  char **names = NULL;
  while ((de = readdir(dir))) {
    char path[4200];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s", o->src_dir, de->d_name);
    if (de->d_name[0] == '.' || stat(path, &st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size < ((uint64_t)o->min_file_gib << 30)) continue;
    if (nf == cap) f = realloc(f, (size_t)(cap = cap ? cap * 2 : 64) * sizeof *f), names = realloc(names, (size_t)cap * sizeof *names);
    f[nf].path = strdup(path), f[nf].size = (uint64_t)st.st_size, f[nf].cursor = 0;
    nf++;
  }
  closedir(dir);
  /* sort by path so every rank sees the same list */
  for (int i = 1; i < nf; i++)
    for (int j = i; j > 0 && strcmp(f[j - 1].path, f[j].path) > 0; j--) {
      mfile t = f[j];
      f[j] = f[j - 1], f[j - 1] = t;
    }
  free(names);
  if (nf == 0) die("no files of at least %d GiB in %s", o->min_file_gib, o->src_dir);
  int big = 0;
  for (int i = 1; i < nf; i++)
    if (f[i].size > f[big].size) big = i;
  const uint64_t region = (uint64_t)o->region_mib << 20;
  int streams[16], ns = parse_list(o->streams_list, streams, 16);
  struct { const char *layout; int streams; uint64_t req; int alone, reread; } pts[64];
  int np = 0;
  const char *layouts[3] = {"own-file", "all-one-file", "per-stream-file"};
  for (int l = 0; l < 3; l++)
    for (int s = 0; s < ns; s++) pts[np].layout = layouts[l], pts[np].streams = streams[s], pts[np].req = 16u << 20, pts[np].alone = 0, pts[np].reread = 0, np++;
  pts[np].layout = "own-file", pts[np].streams = 1, pts[np].req = 4u << 20, pts[np].alone = 0, pts[np].reread = 0, np++;
  pts[np].layout = "own-file", pts[np].streams = 1, pts[np].req = 16u << 20, pts[np].alone = 0, pts[np].reread = 1, np++;
  pts[np].layout = "own-file", pts[np].streams = 1, pts[np].req = 16u << 20, pts[np].alone = 1, pts[np].reread = 0, np++;
  pts[np].layout = "own-file", pts[np].streams = 16, pts[np].req = 16u << 20, pts[np].alone = 1, pts[np].reread = 0, np++;
  mjob *jobs = calloc(256, sizeof *jobs), *prev = calloc(256, sizeof *prev);
  int nprev = 0;
  for (int p = 0; p < np; p++) {
    const int S = pts[p].streams;
    int nj = 0, wrapped = 0;
    if (pts[p].reread) {
      memcpy(jobs, prev, (size_t)nprev * sizeof *jobs);
      nj = nprev;
    } else
      for (int r = 0; r < g_ranks; r++)
        for (int s = 0; s < S; s++) {
          int fi = pts[p].layout == layouts[0] ? r % nf : pts[p].layout == layouts[1] ? big : (r * S + s) % nf;
          if (f[fi].cursor + region > f[fi].size) f[fi].cursor = 0, wrapped = 1;
          uint64_t off = f[fi].cursor;
          f[fi].cursor += region;
          if (r != g_rank || (pts[p].alone && r != 0) || nj >= 256) continue;
          jobs[nj].path = f[fi].path, jobs[nj].off = off, jobs[nj].len = region < f[fi].size ? region : f[fi].size;
          nj++;
        }
    const double dl = now_s() + o->point_s;
    for (int i = 0; i < nj; i++) jobs[i].req = pts[p].req, jobs[i].write = 0, jobs[i].deadline = dl, jobs[i].bytes = 0, jobs[i].secs = 0;
    uint64_t bytes;
    double secs;
    mrun(jobs, nj, &bytes, &secs);
    if (!pts[p].reread) memcpy(prev, jobs, (size_t)nj * sizeof *jobs), nprev = nj;
    char extra[96];
    snprintf(extra, sizeof extra, "files=%d wrapped=%d%s%s", nf, wrapped, pts[p].alone ? " alone=1" : "", pts[p].reread ? " reread=1" : "");
    mreport("read", pts[p].layout, S, pts[p].req, pts[p].alone ? 1 : g_ranks, bytes, nj ? secs : 0, extra);
  }
  free(jobs), free(prev);
  for (int i = 0; i < nf; i++) free(f[i].path);
  free(f);
}

static void measure_write(const opts *o, const char *dir, const char *kind, const int *streams, int ns, uint64_t region) {
  char sub[4200];
  snprintf(sub, sizeof sub, "%s/pfm-rank%02d", dir, g_rank);
  if (mkdir_p(sub)) die("mkdir %s: %s", sub, strerror(errno));
  mjob jobs[64];
  char paths[64][4300];
  for (int l = 0; l < 2; l++)
    for (int s = 0; s < ns; s++) {
      const int S = streams[s] < 64 ? streams[s] : 64;
      for (int i = 0; i < S; i++) {
        snprintf(paths[i], sizeof paths[i], "%s/w-%d-%d-%d", sub, l, S, l == 0 ? i : 0);
        if (l == 0 || i == 0) {
          int fd = open(paths[i], O_WRONLY | O_CREAT | O_TRUNC, 0644);
          if (fd < 0) die("create %s: %s", paths[i], strerror(errno));
          close(fd);
        }
        memset(&jobs[i], 0, sizeof jobs[i]);
        jobs[i].path = paths[i], jobs[i].off = l == 0 ? 0 : (uint64_t)i * region, jobs[i].len = region, jobs[i].req = 16u << 20;
        jobs[i].write = 1, jobs[i].deadline = now_s() + o->point_s;
      }
      uint64_t bytes;
      double secs;
      mrun(jobs, S, &bytes, &secs);
      mreport(kind, l == 0 ? "file-per-stream" : "one-file", S, 16u << 20, g_ranks, bytes, secs, NULL);
      if (strcmp(kind, "local-write") == 0) { /* read it back, page cache dropped */
        for (int i = 0; i < S; i++) {
          int fd = open(paths[l == 0 ? i : 0], O_RDONLY);
          if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), close(fd);
          jobs[i].write = 0, jobs[i].bytes = 0, jobs[i].deadline = now_s() + o->point_s;
        }
        mrun(jobs, S, &bytes, &secs);
        mreport("local-read", l == 0 ? "file-per-stream" : "one-file", S, 16u << 20, g_ranks, bytes, secs, NULL);
      }
      for (int i = 0; i < S; i++) unlink(paths[i]);
    }
  rmdir(sub);
}

static void measure_sparse(const opts *o) {
  char path[4200];
  snprintf(path, sizeof path, "%s/pfm-sparse-rank%02d", o->local_root, g_rank);
  struct statvfs v0, v1;
  statvfs(o->local_root, &v0);
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die("create %s: %s", path, strerror(errno));
  const uint64_t size = 128ull << 30, chunk = 16u << 20;
  uint8_t *buf = calloc(1, chunk);
  memset(buf, 1, chunk);
  int ok = ftruncate(fd, (off_t)size) == 0 && pwrite_full(fd, buf, chunk, size - chunk) == 0 && fsync(fd) == 0;
  struct stat st;
  fstat(fd, &st);
  off_t hole = lseek(fd, 0, SEEK_HOLE);
  statvfs(o->local_root, &v1);
  close(fd);
  unlink(path);
  free(buf);
  uint64_t alloc = (uint64_t)st.st_blocks * 512;
  const int sparse = ok && alloc < (1ull << 30);
  uint64_t v[2] = {(uint64_t)sparse, 1};
  allsum_u64(v, 2);
  if (g_rank == 0)
    printf("PFETCH_MEASURE kind=sparse ranks=%d sparse_ranks=%" PRIu64 " rank0_allocated_bytes=%" PRIu64 " rank0_hole_at_0=%d "
           "rank0_free_GiB=%.1f local_sparse=%s\n",
           g_ranks, v[0], alloc, hole == 0, (double)v0.f_bavail * v0.f_frsize / (1u << 30), v[0] == v[1] ? "yes" : "no");
  (void)v1;
  fflush(stdout);
}

#ifndef PFETCH_NO_MPI
static void measure_mpi(const opts *o) {
  (void)o;
  const size_t M = 16u << 20;
  uint8_t *a, *b;
  if (posix_memalign((void **)&a, 4096, M * 16) || posix_memalign((void **)&b, 4096, M * 16)) die("out of memory");
  memset(a, 7, M * 16), memset(b, 0, M * 16);
  const int R = g_ranks;
  if (R < 2) return;
  /* ring: everyone sends 16 x 16 MiB to the next rank while receiving from the previous */
  barrier();
  double t0 = now_s();
  MPI_Request rq[32];
  for (int i = 0; i < 16; i++) {
    MPI_Irecv(b + i * M, (int)M, MPI_BYTE, (g_rank + R - 1) % R, 7, MPI_COMM_WORLD, &rq[2 * i]);
    MPI_Isend(a + i * M, (int)M, MPI_BYTE, (g_rank + 1) % R, 7, MPI_COMM_WORLD, &rq[2 * i + 1]);
  }
  MPI_Waitall(32, rq, MPI_STATUSES_IGNORE);
  double s = now_s() - t0;
  mreport("mpi", "ring", 1, M, R, 16 * M, s, NULL);
  /* fan-in: every rank sends 4 x 16 MiB to the last rank */
  barrier();
  t0 = now_s();
  if (g_rank == R - 1) {
    MPI_Request *r2 = malloc((size_t)4 * (R - 1) * sizeof *r2);
    int k = 0;
    for (int src = 0; src < R - 1; src++)
      for (int i = 0; i < 4; i++) MPI_Irecv(b + (size_t)(k % 16) * M, (int)M, MPI_BYTE, src, 8, MPI_COMM_WORLD, &r2[k]), k++;
    MPI_Waitall(k, r2, MPI_STATUSES_IGNORE);
    free(r2);
  } else {
    for (int i = 0; i < 4; i++) MPI_Isend(a + i * M, (int)M, MPI_BYTE, R - 1, 8, MPI_COMM_WORLD, &rq[i]);
    MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
  }
  s = now_s() - t0;
  uint64_t got = g_rank == R - 1 ? (uint64_t)4 * (R - 1) * M : 0;
  mreport("mpi", "fan-in", 1, M, R, got, g_rank == R - 1 ? s : 0, NULL);
  /* broadcast of 256 MiB: MPI_Bcast, then rank 0 sending to each rank */
  barrier();
  t0 = now_s();
  MPI_Bcast(a, (int)(16 * M), MPI_BYTE, 0, MPI_COMM_WORLD);
  s = now_s() - t0;
  mreport("mpi", "bcast", 1, 16 * M, R, g_rank ? 16 * M : 0, g_rank ? s : 0, NULL);
  barrier();
  t0 = now_s();
  if (g_rank == 0) {
    MPI_Request *r3 = malloc((size_t)R * sizeof *r3);
    for (int d = 1; d < R; d++) MPI_Isend(a, (int)(16 * M), MPI_BYTE, d, 9, MPI_COMM_WORLD, &r3[d - 1]);
    MPI_Waitall(R - 1, r3, MPI_STATUSES_IGNORE);
    free(r3);
  } else
    MPI_Recv(b, (int)(16 * M), MPI_BYTE, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  s = now_s() - t0;
  mreport("mpi", "fan-out", 1, 16 * M, R, g_rank ? 16 * M : 0, g_rank ? s : 0, NULL);
  free(a), free(b);
}
#endif

static int run_measure(const opts *o) {
  const char *m = o->measure;
  int streams[16], ns = parse_list(o->streams_list, streams, 16);
  if (g_rank == 0) {
    printf("SENTINEL PFETCH_MEASURE_BEGIN ranks=%d kinds=%s point_s=%d region_mib=%d streams=%s\n", g_ranks, m, o->point_s, o->region_mib,
           o->streams_list);
    fflush(stdout);
  }
  if (strstr(m, "read")) measure_read(o);
  if (strstr(m, "write")) {
    if (!o->scratch) die("--measure=write needs --scratch-shared");
    int ws[3] = {1, 4, 16};
    measure_write(o, o->scratch, "write", ws, 3, (uint64_t)o->region_mib << 20);
  }
#ifndef PFETCH_NO_MPI
  if (strstr(m, "mpi")) measure_mpi(o);
#endif
  if (strstr(m, "local")) {
    if (!o->local_root) die("--measure=local needs --local-root");
    int ls[2] = {1, 4};
    measure_write(o, o->local_root, "local-write", ls, 2, 1ull << 30);
    measure_sparse(o);
  }
  (void)ns;
  if (g_rank == 0) printf("SENTINEL PFETCH_MEASURE_END state=ok\n");
  fflush(stdout);
  return 0;
}

int main(int argc, char **argv) {
  opts o;
  setvbuf(stdout, NULL, _IOLBF, 0);
  parse_args(argc, argv, &o);
  if (o.write_list) return run_put(&o);
  if (o.plan_dump) return run_plan_dump(&o);
  if (!o.manifest && !o.measure) die("need --manifest, --measure, --plan-dump or --write (see the header of strata_pfetch.c)");
  comm_init(&argc, &argv);
  printf("PFETCH_WORLD rank=%d ranks=%d\n", g_rank, g_ranks); /* which world a launched process joined */
  fflush(stdout);
  int rc = o.measure ? run_measure(&o) : run_fetch(&o);
  comm_fini();
  return rc;
}
