/* src/cpu_runner/comm_utofu.c - optional uTofu all-reduce for glm_cpu_runner (A64FX / Tofu-D).
 *
 * Two deterministic sum all-reduces, both leaving the same bits on every rank (the runner computes routers and norms
 * on every rank from the result):
 *   rsag (default): reduce-scatter + all-gather over up to six TNIs, ported from the gemm repository's
 *        a64fx/glm5/glm53f_collective_12n.c: each rank Puts shard d of its vector to rank d, rank d sums its shard in
 *        rank order 0..N-1 and Puts the sum back to every peer. Payload and an 8-byte token travel in one Put; slots
 *        need no double buffering (a peer cannot start call n+1 before this rank consumed call n's scatter slots).
 *   tree: the deterministic binomial reduce + broadcast of the gemm repository's a64fx/utofu-tests/tp_allreduce.h
 *        (include path UTOFU_DIR at build time); TP_AR_DETERMINISTIC=0 TP_AR_A2A=1 selects its rank-ordered
 *        all-to-all instead.
 * STRATA_RUNNER_UTOFU=tree selects the second. MPI only bootstraps: each rank's VCQ ids and region addresses are
 * exchanged with MPI_Allgather. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <utofu.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include "tp_allreduce.h"

static int mode_tree;
static tp_comm comm;
static utofu_vcq_hdl_t vcq;

static void barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }

/* ------------------------------------------------------------------------------------------------ rsag */
enum { RS_MAXT = 6, RS_MAXN = 64, RS_STAG = 9 };
static int rs_n, rs_nt, rs_me;
static int rs_late_drain;   /* one completion drain per call, at the end (STRATA_RUNNER_RSAG=1: also after the scatter) */
static utofu_vcq_hdl_t rs_vcq[RS_MAXT];
static utofu_vcq_id_t rs_peer[RS_MAXT][RS_MAXN];
static uint64_t rs_seq;
/* two registered regions on the same VCQs: the small one for vectors up to its count (decode), the big one for longer
   vectors (prefill passes: fewer, larger Puts); a call uses one region throughout */
struct rs_region_t {
    char *region;
    size_t slot, rsize;
    int max;
    utofu_stadd_t base[RS_MAXT], pbase[RS_MAXT][RS_MAXN];
};
static struct rs_region_t rs_r[2];
static int rs_nreg;
static struct rs_region_t *C = &rs_r[0];   /* the region of the current call */
/* region: send[d] (N) | recv[s] (N) | gsend (1) | grecv[s] (N), each C->slot bytes */
static size_t send_off(int d) { return (size_t) d * C->slot; }
static size_t recv_off(int s) { return (size_t) (rs_n + s) * C->slot; }
static size_t gsend_off(void) { return (size_t) (2 * rs_n) * C->slot; }
static size_t grecv_off(int s) { return (size_t) (2 * rs_n + 1 + s) * C->slot; }

static void drain_mrq(void)
{
    struct utofu_mrq_notice nt;
    for (int k = 0; k < rs_nt; ++k) while (utofu_poll_mrq(rs_vcq[k], 0, &nt) == UTOFU_SUCCESS) {}
}

static int put(int k, int dst, size_t src, size_t dst_off, size_t len)
{
    int rc;
    void *cb;
    for (;;) {
        rc = utofu_put(rs_vcq[k], rs_peer[k][dst], C->base[k] + src, C->pbase[k][dst] + dst_off, len, 0,
                       UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        utofu_poll_tcq(rs_vcq[k], 0, &cb);
    }
    return rc == UTOFU_SUCCESS ? 0 : -1;
}

static int drain_tcq(const int *issued)
{
    void *cb;
    for (int k = 0; k < rs_nt; ++k)
        for (int j = 0; j < issued[k]; ++j) {
            int rc;
            do rc = utofu_poll_tcq(rs_vcq[k], 0, &cb); while (rc == UTOFU_ERR_NOT_FOUND);
            if (rc != UTOFU_SUCCESS) return -1;
        }
    return 0;
}

/* wait until every other source's slot carries `tok` behind its payload (one dc civac per pending line, one dsb) */
static int wait_all(size_t (*slot)(int), const size_t *trailer, uint64_t tok)
{
    unsigned char done[RS_MAXN];
    int pending = 0;
    for (int s = 0; s < rs_n; ++s) { done[s] = s == rs_me; pending += s != rs_me; }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long passes = 0;
    while (pending) {
        for (int s = 0; s < rs_n; ++s)
            if (!done[s]) __asm__ __volatile__("dc civac, %0" ::"r"(C->region + slot(s) + trailer[s]) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        for (int s = 0; s < rs_n; ++s)
            if (!done[s] && *(volatile uint64_t *) (C->region + slot(s) + trailer[s]) == tok) { done[s] = 1; --pending; }
        if ((++passes & 0x3FFF) == 0) {
            drain_mrq();
            struct timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (t1.tv_sec - t0.tv_sec > 60) return -1;
        }
    }
    return 0;
}

static int rsag_init(int rank, int world, int max_count)
{
    if (world > RS_MAXN) return -10;
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni < 1) return -11;
    rs_nt = (int) ntni < RS_MAXT ? (int) ntni : RS_MAXT;
    rs_n = world, rs_me = rank;
    /* STRATA_RUNNER_AR_BIG: the big region's count (default 262144 floats; 0: no big region) */
    const char *eb = getenv("STRATA_RUNNER_AR_BIG");
    const int big = eb ? atoi(eb) : 262144;
    rs_nreg = big > max_count ? 2 : 1;
    for (int g = 0; g < rs_nreg; ++g) {
        struct rs_region_t *R = &rs_r[g];
        R->max = g ? big : max_count;
        const size_t shard = ((size_t) R->max / world + 1) * sizeof(float);
        R->slot = (shard + 16 + 255) & ~(size_t) 255;
        R->rsize = (size_t) (3 * world + 1) * R->slot;
        if (posix_memalign((void **) &R->region, 256, R->rsize)) return -12;
        memset(R->region, 0, R->rsize);
        for (size_t off = 0; off < R->rsize; off += 256) __asm__ __volatile__("dc civac, %0" ::"r"(R->region + off) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (int k = 0; k < rs_nt; ++k) {
        if (utofu_create_vcq(tnis[k], 0, &rs_vcq[k]) != UTOFU_SUCCESS) return -13;
        for (int g = 0; g < rs_nreg; ++g)
            if (utofu_reg_mem_with_stag(rs_vcq[k], rs_r[g].region, rs_r[g].rsize, RS_STAG + g, 0, &rs_r[g].base[k]) != UTOFU_SUCCESS) return -14;
        utofu_vcq_id_t me;
        if (utofu_query_vcq_id(rs_vcq[k], &me) != UTOFU_SUCCESS) return -15;
        MPI_Allgather(&me, (int) sizeof me, MPI_BYTE, rs_peer[k], (int) sizeof me, MPI_BYTE, MPI_COMM_WORLD);
        for (int r = 0; r < world; ++r) utofu_set_vcq_id_path(&rs_peer[k][r], NULL);
        for (int g = 0; g < rs_nreg; ++g)
            MPI_Allgather(&rs_r[g].base[k], (int) sizeof rs_r[g].base[k], MPI_BYTE, rs_r[g].pbase[k], (int) sizeof rs_r[g].base[k], MPI_BYTE, MPI_COMM_WORLD);
    }
    free(tnis);
    MPI_Barrier(MPI_COMM_WORLD);
    rs_seq = 0;
    { const char *e = getenv("STRATA_RUNNER_RSAG"); rs_late_drain = !(e && e[0] == '1'); }   /* default late (17.3 vs 19.0 us) */
    return 0;
}

/* staged: the payload is already in the send slots (shard d in send slot d, this rank's own in send slot me), and
   the results stay in the region (this rank's shard in gsend, shard s in grecv slot s): no copies in or out */
static int rsag_run(float *x, int count, int staged)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    if (count < 1 || count > C->max) return -1;
    const uint64_t tok = 0x5A5A5A5A00000000ull | (++rs_seq & 0xFFFFFFFFull);
    int off[RS_MAXN + 1];
    off[0] = 0;
    for (int r = 0; r < n; ++r) off[r + 1] = off[r] + count / n + (r < count % n);
    int issued[RS_MAXT] = {0};
    drain_mrq();
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {   /* 1. scatter: shard d -> rank d */
        const int d = (me + j) % n, len = off[d + 1] - off[d], k = idx % nt;
        const size_t pl = ((size_t) len * sizeof(float) + 7) & ~(size_t) 7;
        char *sb = C->region + send_off(d);
        if (!staged) memcpy(sb, x + off[d], (size_t) len * sizeof(float));
        *(volatile uint64_t *) (sb + pl) = tok;
        if (put(k, d, send_off(d), recv_off(me), pl + 8)) return -1;
        issued[k]++;
    }
    const int mylen = off[me + 1] - off[me];
    const size_t mypl = ((size_t) mylen * sizeof(float) + 7) & ~(size_t) 7;
    size_t trl[RS_MAXN];
    for (int s = 0; s < n; ++s) trl[s] = mypl;
    if (wait_all(recv_off, trl, tok)) return -1;
    float *red = (float *) (C->region + gsend_off());   /* 2. my shard summed in rank order */
#if defined(__ARM_FEATURE_SVE)
    for (int i = 0; i < mylen; i += (int) svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, mylen);
        svfloat32_t acc = svdup_f32(0);
        for (int r = 0; r < n; ++r) {
            const float *src = r == me ? (staged ? (const float *) (C->region + send_off(me)) : x + off[me]) : (const float *) (C->region + recv_off(r));
            acc = svadd_f32_x(pg, acc, svld1_f32(pg, src + i));
        }
        svst1_f32(pg, red + i, acc);
    }
#else
    for (int i = 0; i < mylen; ++i) {
        float acc = 0;
        for (int r = 0; r < n; ++r) acc += (r == me ? (staged ? (const float *) (C->region + send_off(me)) : x + off[me]) : (const float *) (C->region + recv_off(r)))[i];
        red[i] = acc;
    }
#endif
    *(volatile uint64_t *) ((char *) red + mypl) = tok;
    if (!rs_late_drain) {
        if (drain_tcq(issued)) return -1;
        memset(issued, 0, sizeof issued);
    }   /* late drain: the scatter Puts' completions are polled at the end with the gather's (a peer that sent its
           gather slot to me already read my scatter slot; the send slots are reused only by the next call) */
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {   /* 3. gather: my sum -> every peer */
        const int d = (me + j) % n, k = idx % nt;
        if (put(k, d, gsend_off(), grecv_off(me), mypl + 8)) return -1;
        issued[k]++;
    }
    if (!staged) memcpy(x + off[me], red, (size_t) mylen * sizeof(float));
    for (int s = 0; s < n; ++s) trl[s] = (((size_t) (off[s + 1] - off[s]) * sizeof(float)) + 7) & ~(size_t) 7;
    if (wait_all(grecv_off, trl, tok)) return -1;
    if (!staged)
        for (int s = 0; s < n; ++s)
            if (s != me) memcpy(x + off[s], C->region + grecv_off(s), (size_t) (off[s + 1] - off[s]) * sizeof(float));
    if (drain_tcq(issued)) return -1;
    if (!rs_late_drain) drain_mrq();
    return 0;
}

static int rsag_allreduce(float *x, int count) { return rsag_run(x, count, 0); }

/* banks: further regions for staged all-reduces (one per pipelined pass slot: an all-reduce in flight or queued never
   shares slots with another), registered on the same VCQs (stags RS_STAG + 2 + bank) */
enum { RS_MAXB = 8 };
static struct rs_region_t rs_bank[RS_MAXB];
static int rs_nbank;
static int region_init(struct rs_region_t *R, int max, int stag)
{
    R->max = max;
    const size_t shard = ((size_t) max / rs_n + 1) * sizeof(float);
    R->slot = (shard + 16 + 255) & ~(size_t) 255;
    R->rsize = (size_t) (3 * rs_n + 1) * R->slot;
    if (posix_memalign((void **) &R->region, 256, R->rsize)) return -1;
    memset(R->region, 0, R->rsize);
    for (size_t off = 0; off < R->rsize; off += 256) __asm__ __volatile__("dc civac, %0" ::"r"(R->region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (int k = 0; k < rs_nt; ++k) {
        if (utofu_reg_mem_with_stag(rs_vcq[k], R->region, R->rsize, stag, 0, &R->base[k]) != UTOFU_SUCCESS) return -2;
        MPI_Allgather(&R->base[k], (int) sizeof R->base[k], MPI_BYTE, R->pbase[k], (int) sizeof R->base[k], MPI_BYTE, MPI_COMM_WORLD);
    }
    return 0;
}
/* collective: nb banks of max floats each; 0 on success */
int cpu_utofu_banks(int nb, int max)
{
    if (mode_tree || nb > RS_MAXB) return -1;
    for (int b = 0; b < nb; ++b)
        if (region_init(&rs_bank[b], max, RS_STAG + 2 + b)) return -2;
    MPI_Barrier(MPI_COMM_WORLD);
    rs_nbank = nb;
    return 0;
}
int cpu_utofu_bank_max(void) { return rs_nbank ? rs_bank[0].max : 0; }
/* the shard bounds of a count-float vector (off[0..n]) and the bank's slot pointers for staging and results */
void cpu_utofu_shards(int count, int *off)
{
    off[0] = 0;
    for (int r = 0; r < rs_n; ++r) off[r + 1] = off[r] + count / rs_n + (r < count % rs_n);
}
/* (the slot addresses from the bank itself: these run on the main thread while the comm thread's call holds C) */
float *cpu_utofu_bank_send(int bank, int d) { return (float *) (rs_bank[bank].region + (size_t) d * rs_bank[bank].slot); }
const float *cpu_utofu_bank_result(int bank, int d)
{
    const struct rs_region_t *R = &rs_bank[bank];
    return (const float *) (R->region + (size_t) (d == rs_me ? 2 * rs_n : 2 * rs_n + 1 + d) * R->slot);
}
void cpu_utofu_bank_allreduce(int bank, int count)
{
    C = &rs_bank[bank];
    if (rsag_run(NULL, count, 1)) { fprintf(stderr, "RUNNER uTofu staged all-reduce failed\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
}
int cpu_utofu_world(void) { return rs_n; }
int cpu_utofu_rank(void) { return rs_me; }

/* ------------------------------------------------------------------------------------------------ entry points */
/* 0 on success, a negative step number otherwise */
int cpu_utofu_init(int rank, int world, int max_count)
{
    const char *m = getenv("STRATA_RUNNER_UTOFU");
    mode_tree = m && !strcmp(m, "tree");
    if (!mode_tree) return rsag_init(rank, world, max_count);
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni < 1) return -1;
    const int rc = utofu_create_vcq(tnis[0], 0, &vcq);
    free(tnis);
    if (rc != UTOFU_SUCCESS) return -2;
    utofu_vcq_id_t me;
    if (utofu_query_vcq_id(vcq, &me) != UTOFU_SUCCESS) return -3;
    utofu_vcq_id_t *peers = malloc(sizeof(*peers) * (size_t) world);
    if (!peers) return -4;
    MPI_Allgather(&me, (int) sizeof me, MPI_BYTE, peers, (int) sizeof me, MPI_BYTE, MPI_COMM_WORLD);
    for (int r = 0; r < world; ++r) utofu_set_vcq_id_path(&peers[r], NULL);
    setenv("TP_AR_DETERMINISTIC", "1", 0);   /* keep a caller's TP_AR_* choice */
    const int ok = tp_comm_init(&comm, vcq, peers, rank, world, max_count, barrier);
    free(peers);
    return ok == 0 ? 0 : -5;
}

const char *cpu_utofu_mode(void) { return mode_tree ? "utofu tree" : rs_nreg == 2 ? "utofu rsag (16K + big region)" : "utofu rsag"; }
int cpu_utofu_max(void) { return mode_tree ? 0 : rs_r[rs_nreg - 1].max; }

void cpu_utofu_allreduce(float *x, int n)
{
    if (mode_tree) { tp_allreduce_sum(&comm, x, n); return; }
    /* the small region for vectors that fit it, else the big one, in pieces of its count (each element is still
       summed in rank order) */
    C = &rs_r[n <= rs_r[0].max || rs_nreg == 1 ? 0 : 1];
    for (int o = 0; o < n; o += C->max) {
        const int len = n - o < C->max ? n - o : C->max;
        if (rsag_allreduce(x + o, len)) { fprintf(stderr, "RUNNER uTofu rsag all-reduce failed\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    }
}
