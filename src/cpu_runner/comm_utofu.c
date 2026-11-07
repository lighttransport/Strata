/* src/cpu_runner/comm_utofu.c - optional uTofu all-reduce for glm_cpu_runner (A64FX / Tofu-D).
 *
 * Deterministic sum all-reduces, both leaving the same bits on every rank (the runner computes routers and norms
 * on every rank from the result):
 *   rsag (default): reduce-scatter + all-gather over up to six TNIs, ported from the gemm repository's
 *        a64fx/glm5/glm53f_collective_12n.c: each rank Puts shard d of its vector to rank d, rank d sums its shard in
 *        rank order 0..N-1 and Puts the sum back to every peer. Payload and an 8-byte token travel in one Put; slots
 *        need no double buffering (a peer cannot start call n+1 before this rank consumed call n's scatter slots).
 *   a2a (STRATA_RUNNER_A2A_MAX floats and shorter): one round, every rank Puts its whole vector to every peer and sums
 *        all N in rank order itself (the same bits as rsag).
 *   tree: the deterministic binomial reduce + broadcast of the gemm repository's a64fx/utofu-tests/tp_allreduce.h
 *        (include path UTOFU_DIR at build time); TP_AR_DETERMINISTIC=0 TP_AR_A2A=1 selects its rank-ordered
 *        all-to-all instead.
 * STRATA_RUNNER_UTOFU=tree selects the tree. MPI only bootstraps: each rank's VCQ ids and region addresses are
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
#include "strata/host/glm_env.h"

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
    { const char *e = getenv("STRATA_RUNNER_TNIS"); if (e && atoi(e) >= 1 && atoi(e) < rs_nt) rs_nt = atoi(e); }   /* (TNIs used: default all, up to 6) */
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
    rs_late_drain = !strata_env_truthy_one("STRATA_RUNNER_RSAG");   /* default late (17.3 vs 19.0 us) */
    return 0;
}

/* cpu_utofu_set_par: a parallel for (the caller's thread team) for the copies and the shard sum of long vectors
   (STRATA_RUNNER_AR_PAR_MIN floats and longer, default 20000: above the 16K region): each element's sum keeps the rank order */
static void (*par_for)(void (*fn)(void *, int, int), void *arg, int n);
static int par_min = 20000;
void cpu_utofu_set_par(void (*p)(void (*fn)(void *, int, int), void *arg, int n))
{
    par_for = p;
    const char *e = getenv("STRATA_RUNNER_AR_PAR_MIN");
    if (e) par_min = atoi(e);
}
struct par_ctx { float *x; const int *off; int staged, n, me, mylen, phase; float *red; };
static void par_body(void *v, int a, int b)
{
    struct par_ctx *c = (struct par_ctx *) v;
    if (c->phase == 0) {   /* scatter copies: shards [a, b) */
        for (int d = a; d < b; ++d)
            if (d != c->me) memcpy(C->region + send_off(d), c->x + c->off[d], (size_t) (c->off[d + 1] - c->off[d]) * sizeof(float));
    } else if (c->phase == 1) {   /* my shard's elements [16 a, 16 b), summed in rank order */
        const int i0 = 16 * a, i1 = 16 * b < c->mylen ? 16 * b : c->mylen;
#if defined(__ARM_FEATURE_SVE)
        for (int i = i0; i < i1; i += (int) svcntw()) {
            const svbool_t pg = svwhilelt_b32(i, i1);
            svfloat32_t acc = svdup_f32(0);
            for (int r = 0; r < c->n; ++r) {
                const float *src = r == c->me ? c->x + c->off[c->me] : (const float *) (C->region + recv_off(r));
                acc = svadd_f32_x(pg, acc, svld1_f32(pg, src + i));
            }
            svst1_f32(pg, c->red + i, acc);
        }
#else
        for (int i = i0; i < i1; ++i) {
            float acc = 0;
            for (int r = 0; r < c->n; ++r) acc += (r == c->me ? c->x + c->off[c->me] : (const float *) (C->region + recv_off(r)))[i];
            c->red[i] = acc;
        }
#endif
    } else {   /* gathered shards [a, b) back into x */
        for (int s = a; s < b; ++s)
            if (s != c->me) memcpy(c->x + c->off[s], C->region + grecv_off(s), (size_t) (c->off[s + 1] - c->off[s]) * sizeof(float));
    }
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
    const int par = !staged && par_for && count >= par_min;
    struct par_ctx pc = {x, off, staged, n, me, off[me + 1] - off[me], 0, (float *) (C->region + gsend_off())};
    if (par) par_for(par_body, &pc, n);   /* the scatter copies over the team, before the Puts */
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {   /* 1. scatter: shard d -> rank d */
        const int d = (me + j) % n, len = off[d + 1] - off[d], k = idx % nt;
        const size_t pl = ((size_t) len * sizeof(float) + 7) & ~(size_t) 7;
        char *sb = C->region + send_off(d);
        if (!staged && !par) memcpy(sb, x + off[d], (size_t) len * sizeof(float));
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
    if (par) { pc.phase = 1; par_for(par_body, &pc, (mylen + 15) / 16); }
    else {
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
    }
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
    if (par) { pc.phase = 2; par_for(par_body, &pc, n); }
    else if (!staged)
        for (int s = 0; s < n; ++s)
            if (s != me) memcpy(x + off[s], C->region + grecv_off(s), (size_t) (off[s + 1] - off[s]) * sizeof(float));
    if (drain_tcq(issued)) return -1;
    if (!rs_late_drain) drain_mrq();
    return 0;
}

static int rsag_allreduce(float *x, int count) { return rsag_run(x, count, 0); }

/* STRATA_RUNNER_AR_BF16=1 (opt-in, lossy): long vectors (par_min and longer, with the team's parallel for) travel as
   bf16 (round to nearest even) both ways: every rank's contribution rounded, each shard summed in fp32 in rank order,
   the sum rounded and sent back, so every rank holds the same values (half the bytes of rsag) */
static int ar_bf16 = -1;
static inline uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static inline float from_bf16(uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
struct bctx { float *x; const int *off; int n, me, mylen, phase; };
static void bf16_body(void *v, int a, int b)
{
    struct bctx *c = (struct bctx *) v;
    if (c->phase == 0) {   /* every shard (mine too) to bf16 in its send slot */
        for (int d = a; d < b; ++d) {
            uint16_t *o = (uint16_t *) (C->region + send_off(d));
            for (int i = c->off[d]; i < c->off[d + 1]; ++i) o[i - c->off[d]] = to_bf16(c->x[i]);
        }
    } else if (c->phase == 1) {   /* my shard: fp32 sums in rank order, rounded; mine in x too */
        const int i0 = 16 * a, i1 = 16 * b < c->mylen ? 16 * b : c->mylen;
        uint16_t *red = (uint16_t *) (C->region + gsend_off());
        for (int i = i0; i < i1; ++i) {
            float acc = 0;
            for (int r = 0; r < c->n; ++r)
                acc += from_bf16(((const uint16_t *) (C->region + (r == c->me ? send_off(c->me) : recv_off(r))))[i]);
            red[i] = to_bf16(acc);
            c->x[c->off[c->me] + i] = from_bf16(red[i]);
        }
    } else {
        for (int s = a; s < b; ++s)
            if (s != c->me) {
                const uint16_t *g = (const uint16_t *) (C->region + grecv_off(s));
                for (int i = c->off[s]; i < c->off[s + 1]; ++i) c->x[i] = from_bf16(g[i - c->off[s]]);
            }
    }
}
static int rsag_bf16(float *x, int count)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    if (count < 1 || count > C->max) return -1;
    const uint64_t tok = 0x5A5A5A5A00000000ull | (++rs_seq & 0xFFFFFFFFull);
    int off[RS_MAXN + 1];
    off[0] = 0;
    for (int r = 0; r < n; ++r) off[r + 1] = off[r] + count / n + (r < count % n);
    int issued[RS_MAXT] = {0};
    drain_mrq();
    struct bctx bc = {x, off, n, me, off[me + 1] - off[me], 0};
    par_for(bf16_body, &bc, n);
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, len = off[d + 1] - off[d], k = idx % nt;
        const size_t pl = ((size_t) len * 2 + 7) & ~(size_t) 7;
        *(volatile uint64_t *) (C->region + send_off(d) + pl) = tok;
        if (put(k, d, send_off(d), recv_off(me), pl + 8)) return -1;
        issued[k]++;
    }
    const int mylen = off[me + 1] - off[me];
    const size_t mypl = ((size_t) mylen * 2 + 7) & ~(size_t) 7;
    size_t trl[RS_MAXN];
    for (int s = 0; s < n; ++s) trl[s] = mypl;
    if (wait_all(recv_off, trl, tok)) return -1;
    bc.phase = 1;
    par_for(bf16_body, &bc, (mylen + 15) / 16);
    *(volatile uint64_t *) (C->region + gsend_off() + mypl) = tok;
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, k = idx % nt;
        if (put(k, d, gsend_off(), grecv_off(me), mypl + 8)) return -1;
        issued[k]++;
    }
    for (int s = 0; s < n; ++s) trl[s] = (((size_t) (off[s + 1] - off[s]) * 2) + 7) & ~(size_t) 7;
    if (wait_all(grecv_off, trl, tok)) return -1;
    bc.phase = 2;
    par_for(bf16_body, &bc, n);
    return drain_tcq(issued);
}

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


/* ------------------------------------------------------------------------------------------------ a2a */
/* one round for short vectors (STRATA_RUNNER_A2A_MAX floats, default 0: off): each rank Puts its whole vector (with
   the token) to every peer, then every rank sums the N vectors in rank order 0..N-1 itself -- the same order as rsag's
   shard sums, so the same bits. Receive slots alternate by call parity: a rank in call n + 2 has received every peer's
   call n + 1 vector, which each peer sends only after summing call n. */
static struct {
    char *region;
    size_t slot, rsize;
    int max;
    utofu_stadd_t base[RS_MAXT], pbase[RS_MAXT][RS_MAXN];
} A;
static uint64_t a2a_seq;
static size_t a_recv(int p, int s) { return ((size_t) p * rs_n + s) * A.slot; }
static size_t a_send(void) { return (size_t) 2 * rs_n * A.slot; }

static int a2a_init(int max)
{
    A.max = max;
    A.slot = ((size_t) max * sizeof(float) + 16 + 255) & ~(size_t) 255;
    A.rsize = (size_t) (2 * rs_n + 1) * A.slot;
    if (posix_memalign((void **) &A.region, 256, A.rsize)) return -1;
    memset(A.region, 0, A.rsize);
    for (size_t off = 0; off < A.rsize; off += 256) __asm__ __volatile__("dc civac, %0" ::"r"(A.region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (int k = 0; k < rs_nt; ++k) {
        if (utofu_reg_mem_with_stag(rs_vcq[k], A.region, A.rsize, RS_STAG + 20, 0, &A.base[k]) != UTOFU_SUCCESS) return -2;
        MPI_Allgather(&A.base[k], (int) sizeof A.base[k], MPI_BYTE, A.pbase[k], (int) sizeof A.base[k], MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

static int a2a_run(float *x, int count)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    if (count < 1 || count > A.max) return -1;
    const uint64_t seq = ++a2a_seq, tok = 0xA2A2A2A200000000ull | (seq & 0xFFFFFFFFull);
    const int p = (int) (seq & 1);
    const size_t pl = ((size_t) count * sizeof(float) + 7) & ~(size_t) 7;
    char *sb = A.region + a_send();
    memcpy(sb, x, (size_t) count * sizeof(float));
    *(volatile uint64_t *) (sb + pl) = tok;
    int issued[RS_MAXT] = {0};
    drain_mrq();
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, k = idx % nt;
        int rc;
        void *cb;
        for (;;) {
            rc = utofu_put(rs_vcq[k], rs_peer[k][d], A.base[k] + a_send(), A.pbase[k][d] + a_recv(p, me), pl + 8, 0,
                           UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
            if (rc != UTOFU_ERR_BUSY) break;
            utofu_poll_tcq(rs_vcq[k], 0, &cb);
        }
        if (rc != UTOFU_SUCCESS) return -1;
        issued[k]++;
    }
    unsigned char done[RS_MAXN];
    int pending = 0;
    for (int s = 0; s < n; ++s) { done[s] = s == me; pending += s != me; }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long passes = 0;
    while (pending) {
        for (int s = 0; s < n; ++s)
            if (!done[s]) __asm__ __volatile__("dc civac, %0" ::"r"(A.region + a_recv(p, s) + pl) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        for (int s = 0; s < n; ++s)
            if (!done[s] && *(volatile uint64_t *) (A.region + a_recv(p, s) + pl) == tok) { done[s] = 1; --pending; }
        if ((++passes & 0x3FFF) == 0) {
            drain_mrq();
            struct timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (t1.tv_sec - t0.tv_sec > 60) return -1;
        }
    }
    const float *src[RS_MAXN];
    for (int r = 0; r < n; ++r) src[r] = r == me ? (const float *) sb : (const float *) (A.region + a_recv(p, r));
#if defined(__ARM_FEATURE_SVE)
    for (int i = 0; i < count; i += (int) svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, count);
        svfloat32_t acc = svdup_f32(0);
        for (int r = 0; r < n; ++r) acc = svadd_f32_x(pg, acc, svld1_f32(pg, src[r] + i));
        svst1_f32(pg, x + i, acc);
    }
#else
    for (int i = 0; i < count; ++i) {
        float acc = 0;
        for (int r = 0; r < n; ++r) acc += src[r][i];
        x[i] = acc;
    }
#endif
    return drain_tcq(issued);
}

/* ------------------------------------------------------------------------------------------------ xc */
/* reduce-scatter and all-gather with caller-given shard bounds off[0..N] (shard r: elements [off[r], off[r+1])), on a
   region of their own: send[d] (N) | rs recv[p][s] (2N) | ag send[p] (2) | ag recv[p][s] (2N), each xc.slot bytes.
   Unlike rsag, a call of either kind ends without hearing from every peer after this rank read its slots, so a peer
   may already send the next call: the receive slots alternate by call (a peer cannot start the call after that
   before this rank sent its part of the next one, i.e. before it finished reading this one).
   reduce-scatter: this rank's shard summed in rank order 0..N-1 from 0 (rsag's order: the same bits as the all-reduce).
   all-gather: every shard to every rank, each value + 0.0f (a -0.0 becomes +0.0, as rsag's sum of zeros and the one
   rank's value gives: the same bits as the all-reduce of a vector that is zero outside each rank's shard). */
static struct rs_region_t xc;
static int xc_par_min = 20000;
static uint64_t xc_rs_calls, xc_ag_calls;
static int xc_p;   /* the current call's parity */
static size_t xc_send(int d) { return (size_t) d * xc.slot; }
static size_t xc_rrecv(int s) { return (size_t) (rs_n + xc_p * rs_n + s) * xc.slot; }
static size_t xc_gsend(void) { return (size_t) (3 * rs_n + xc_p) * xc.slot; }
static size_t xc_grecv(int s) { return (size_t) (3 * rs_n + 2 + xc_p * rs_n + s) * xc.slot; }
int cpu_utofu_xc_init(int shard_cap)   /* collective; floats per shard at most shard_cap */
{
    if (mode_tree || xc.region) return mode_tree ? -1 : 0;
    xc.max = shard_cap;
    xc.slot = ((size_t) shard_cap * sizeof(float) + 16 + 255) & ~(size_t) 255;
    xc.rsize = (size_t) (5 * rs_n + 2) * xc.slot;
    if (posix_memalign((void **) &xc.region, 256, xc.rsize)) return -2;
    memset(xc.region, 0, xc.rsize);
    for (size_t off = 0; off < xc.rsize; off += 256) __asm__ __volatile__("dc civac, %0" ::"r"(xc.region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (int k = 0; k < rs_nt; ++k) {
        if (utofu_reg_mem_with_stag(rs_vcq[k], xc.region, xc.rsize, RS_STAG + 12, 0, &xc.base[k]) != UTOFU_SUCCESS) return -3;
        MPI_Allgather(&xc.base[k], (int) sizeof xc.base[k], MPI_BYTE, xc.pbase[k], (int) sizeof xc.base[k], MPI_BYTE, MPI_COMM_WORLD);
    }
    { const char *e = getenv("STRATA_RUNNER_AR_PAR_MIN"); if (e) xc_par_min = atoi(e); }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
int cpu_utofu_xc_cap(void) { return xc.region ? xc.max : 0; }
struct xc_ctx { const float *x; float *y; const int *off; int me, n, len; };
static void xc_rs_copy(void *v, int a, int b)
{
    const struct xc_ctx *c = (const struct xc_ctx *) v;
    for (int d = a; d < b; ++d)
        if (d != c->me) memcpy(xc.region + xc_send(d), c->x + c->off[d], (size_t) (c->off[d + 1] - c->off[d]) * sizeof(float));
}
static void xc_rs_sum(void *v, int a, int b)   /* elements [16 a, 16 b) of this rank's shard */
{
    const struct xc_ctx *c = (const struct xc_ctx *) v;
    const int i0 = 16 * a, i1 = 16 * b < c->len ? 16 * b : c->len;
#if defined(__ARM_FEATURE_SVE)
    for (int i = i0; i < i1; i += (int) svcntw()) {
        const svbool_t pg = svwhilelt_b32(i, i1);
        svfloat32_t acc = svdup_f32(0);
        for (int r = 0; r < c->n; ++r) {
            const float *src = r == c->me ? c->x + c->off[c->me] : (const float *) (xc.region + xc_rrecv(r));
            acc = svadd_f32_x(pg, acc, svld1_f32(pg, src + i));
        }
        svst1_f32(pg, c->y + i, acc);
    }
#else
    for (int i = i0; i < i1; ++i) {
        float acc = 0;
        for (int r = 0; r < c->n; ++r) acc += (r == c->me ? c->x + c->off[c->me] : (const float *) (xc.region + xc_rrecv(r)))[i];
        c->y[i] = acc;
    }
#endif
}
static void xc_ag_copy(void *v, int a, int b)
{
    const struct xc_ctx *c = (const struct xc_ctx *) v;
    for (int s = a; s < b; ++s)
        if (s != c->me) memcpy(c->y + c->off[s], xc.region + xc_grecv(s), (size_t) (c->off[s + 1] - c->off[s]) * sizeof(float));
}
static size_t xc_pl(int len) { return ((size_t) len * sizeof(float) + 7) & ~(size_t) 7; }
/* mine = this rank's shard of the sum of every rank's x (x itself unchanged); 0 on success */
int cpu_utofu_reduce_scatter(const float *x, const int *off, float *mine)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    for (int r = 0; r < n; ++r) if (off[r + 1] - off[r] > xc.max || off[r + 1] < off[r]) return -1;
    struct rs_region_t *const C0 = C;
    C = &xc;
    xc_p = (int) (xc_rs_calls++ & 1);
    const uint64_t tok = 0x3C3C3C3C00000000ull | (++rs_seq & 0xFFFFFFFFull);
    const int total = off[n], par = par_for && total >= xc_par_min;
    struct xc_ctx c = {x, mine, off, me, n, off[me + 1] - off[me]};
    int issued[RS_MAXT] = {0}, rc = 0;
    drain_mrq();
    if (par) par_for(xc_rs_copy, &c, n);
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, len = off[d + 1] - off[d], k = idx % nt;
        char *sb = xc.region + xc_send(d);
        if (!par) memcpy(sb, x + off[d], (size_t) len * sizeof(float));
        *(volatile uint64_t *) (sb + xc_pl(len)) = tok;
        if (put(k, d, xc_send(d), xc_rrecv(me), xc_pl(len) + 8)) { rc = -2; goto out; }
        issued[k]++;
    }
    {
        size_t trl[RS_MAXN];
        for (int s = 0; s < n; ++s) trl[s] = xc_pl(c.len);
        if (wait_all(xc_rrecv, trl, tok)) { rc = -3; goto out; }
    }
    if (par) par_for(xc_rs_sum, &c, (c.len + 15) / 16);
    else xc_rs_sum(&c, 0, (c.len + 15) / 16);
    if (drain_tcq(issued)) rc = -4;
out:
    C = C0;
    return rc;
}
/* x: this rank's shard at x + off[me] in, every shard out (canon: each value + 0.0f; else the bits as sent); 0 on success */
int cpu_utofu_allgather(float *x, const int *off, int canon)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    for (int r = 0; r < n; ++r) if (off[r + 1] - off[r] > xc.max || off[r + 1] < off[r]) return -1;
    struct rs_region_t *const C0 = C;
    C = &xc;
    xc_p = (int) (xc_ag_calls++ & 1);
    const uint64_t tok = 0x6C6C6C6C00000000ull | (++rs_seq & 0xFFFFFFFFull);
    const int len = off[me + 1] - off[me], par = par_for && off[n] >= xc_par_min;
    float *gs = (float *) (xc.region + xc_gsend()), *mine = x + off[me];
    if (canon) for (int i = 0; i < len; ++i) gs[i] = mine[i] + 0.0f, mine[i] = gs[i];
    else memcpy(gs, mine, (size_t) len * sizeof(float));
    *(volatile uint64_t *) ((char *) gs + xc_pl(len)) = tok;
    int issued[RS_MAXT] = {0}, rc = 0;
    drain_mrq();
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, k = idx % nt;
        if (put(k, d, xc_gsend(), xc_grecv(me), xc_pl(len) + 8)) { rc = -2; goto out; }
        issued[k]++;
    }
    {
        size_t trl[RS_MAXN];
        for (int s = 0; s < n; ++s) trl[s] = xc_pl(off[s + 1] - off[s]);
        if (wait_all(xc_grecv, trl, tok)) { rc = -3; goto out; }
    }
    {
        struct xc_ctx c = {NULL, x, off, me, n, len};
        if (par) par_for(xc_ag_copy, &c, n);
        else xc_ag_copy(&c, 0, n);
    }
    if (drain_tcq(issued)) rc = -4;
out:
    C = C0;
    return rc;
}

/* The all-to-all's copies can use the same caller team as reduce-scatter. Each
   segment has its own staging/receive slot; notification and completion stay serial. */
struct xc_a2a_ctx { const float *send; float *recv; const int *soff, *roff; int me; };
static void xc_a2a_send_copy(void *v, int a, int b)
{
    const struct xc_a2a_ctx *c = (const struct xc_a2a_ctx *) v;
    for (int d = a; d < b; ++d) {
        const size_t bytes = (size_t) (c->soff[d + 1] - c->soff[d]) * sizeof(float);
        if (d != c->me) memcpy(xc.region + xc_send(d), c->send + c->soff[d], bytes);
    }
}
static void xc_a2a_recv_copy(void *v, int a, int b)
{
    const struct xc_a2a_ctx *c = (const struct xc_a2a_ctx *) v;
    for (int s = a; s < b; ++s)
        if (s != c->me) memcpy(c->recv + c->roff[s], xc.region + xc_rrecv(s),
                              (size_t) (c->roff[s + 1] - c->roff[s]) * sizeof(float));
}

/* all-to-all: segment d of send ([soff[d], soff[d+1])) to rank d, segment s of recv ([roff[s], roff[s+1])) from rank s
   (this rank's own segment copied); the bits as sent. Uses the reduce-scatter's slots (the same alternation) */
int cpu_utofu_alltoall(const float *send, const int *soff, float *recv, const int *roff)
{
    const int n = rs_n, nt = rs_nt, me = rs_me;
    for (int r = 0; r < n; ++r)
        if (soff[r + 1] - soff[r] > xc.max || roff[r + 1] - roff[r] > xc.max || soff[r + 1] < soff[r] || roff[r + 1] < roff[r]) return -1;
    if (soff[me + 1] - soff[me] != roff[me + 1] - roff[me]) return -1;
    struct rs_region_t *const C0 = C;
    C = &xc;
    xc_p = (int) (xc_rs_calls++ & 1);
    const uint64_t tok = 0x5C5C5C5C00000000ull | (++rs_seq & 0xFFFFFFFFull);
    int issued[RS_MAXT] = {0}, rc = 0;
    drain_mrq();
    /* STRATA_RUNNER_A2A_PAR=1: opt-in only; existing small-copy/default path is unchanged. */
    static int a2a_par = -1;
    if (a2a_par < 0) a2a_par = strata_env_truthy_one("STRATA_RUNNER_A2A_PAR");
    const int par = a2a_par && par_for && (soff[n] >= xc_par_min || roff[n] >= xc_par_min);
    struct xc_a2a_ctx ctx = {send, recv, soff, roff, me};
    if (par) par_for(xc_a2a_send_copy, &ctx, n);
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, len = soff[d + 1] - soff[d], k = idx % nt;
        char *sb = xc.region + xc_send(d);
        if (!par) memcpy(sb, send + soff[d], (size_t) len * sizeof(float));
        *(volatile uint64_t *) (sb + xc_pl(len)) = tok;
        if (put(k, d, xc_send(d), xc_rrecv(me), xc_pl(len) + 8)) { rc = -2; goto out; }
        issued[k]++;
    }
    memcpy(recv + roff[me], send + soff[me], (size_t) (soff[me + 1] - soff[me]) * sizeof(float));
    {
        size_t trl[RS_MAXN];
        for (int s = 0; s < n; ++s) trl[s] = xc_pl(roff[s + 1] - roff[s]);
        if (wait_all(xc_rrecv, trl, tok)) { rc = -3; goto out; }
    }
    if (par) par_for(xc_a2a_recv_copy, &ctx, n);
    else for (int s = 0; s < n; ++s)
        if (s != me) memcpy(recv + roff[s], xc.region + xc_rrecv(s), (size_t) (roff[s + 1] - roff[s]) * sizeof(float));
    if (drain_tcq(issued)) rc = -4;
out:
    C = C0;
    return rc;
}

/* ------------------------------------------------------------------------------------------------ entry points */
/* 0 on success, a negative step number otherwise */
int cpu_utofu_init(int rank, int world, int max_count)
{
    const char *m = getenv("STRATA_RUNNER_UTOFU");
    mode_tree = m && !strcmp(m, "tree");
    if (!mode_tree) {
        const int rc = rsag_init(rank, world, max_count);
        const char *ea = getenv("STRATA_RUNNER_A2A_MAX");
        const int am = ea ? atoi(ea) : 0;
        if (rc || am <= 0) return rc;
        return a2a_init(am) ? -20 : 0;
    }
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

const char *cpu_utofu_mode(void) { return mode_tree ? "utofu tree" : A.max ? "utofu a2a (short) + rsag (16K + big region)" : rs_nreg == 2 ? "utofu rsag (16K + big region)" : "utofu rsag"; }
int cpu_utofu_max(void) { return mode_tree ? 0 : rs_r[rs_nreg - 1].max; }

void cpu_utofu_allreduce(float *x, int n)
{
    if (mode_tree) { tp_allreduce_sum(&comm, x, n); return; }
    if (n <= A.max) {
        if (a2a_run(x, n)) { fprintf(stderr, "RUNNER uTofu a2a all-reduce failed\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
        return;
    }
    /* the small region for vectors that fit it, else the big one, in pieces of its count (each element is still
       summed in rank order) */
    C = &rs_r[n <= rs_r[0].max || rs_nreg == 1 ? 0 : 1];
    if (ar_bf16 < 0) { ar_bf16 = strata_env_truthy_one("STRATA_RUNNER_AR_BF16"); }
    const int half = ar_bf16 && par_for && n >= par_min;
    for (int o = 0; o < n; o += C->max) {
        const int len = n - o < C->max ? n - o : C->max;
        if (half ? rsag_bf16(x + o, len) : rsag_allreduce(x + o, len)) { fprintf(stderr, "RUNNER uTofu rsag all-reduce failed\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    }
}
