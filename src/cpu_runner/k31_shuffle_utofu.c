/* src/cpu_runner/k31_shuffle_utofu.c - uTofu Put engine for k31pro_runner's weight load (A64FX / Tofu-D).
 *
 * Every rank registers its weight arena (in chunks) and a send buffer on every one-sided TNI; MPI only bootstraps
 * (VCQ ids and region addresses through MPI_Allgather). A source rank packs a destination's bytes into its send
 * buffer and Puts them straight to the destination's arena offset (split at arena chunk edges and the Put size cap,
 * round-robin over the TNIs). k31x_flush waits for every issued Put's local completion before the send buffer is
 * reused. Destinations do not poll: the caller ends the exchange with a barrier and checks a hash of every piece.
 * Environment: K31X_CHUNK_MB (arena chunk per registration, default 1024), K31X_TNIS (TNIs used, default all, up
 * to 6), K31X_PUT_MAX (bytes per Put, default the TNI cap, at most 16 MiB). */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utofu.h>

enum { X_MAXT = 6 };
static int x_nt, x_world, x_me, x_nch;
static size_t x_chunk, x_putmax;
static utofu_vcq_hdl_t x_vcq[X_MAXT];
static utofu_vcq_id_t *x_peer[X_MAXT];        /* [rank] */
static utofu_stadd_t *x_rbase[X_MAXT];        /* [rank * x_nch + chunk]: every rank's arena chunks */
static utofu_stadd_t x_lbase[X_MAXT][256];    /* my arena chunks (unused locally; kept for dereg) */
static utofu_stadd_t x_sbase[X_MAXT];         /* my send buffer */
static int x_nlocal;
static char *x_arena, *x_send;
static size_t x_arena_bytes, x_send_bytes;
static long x_out[X_MAXT];                    /* Puts issued since the last flush */
static int x_rr;
static unsigned long long x_puts, x_bytes;

static void flush_lines(char *p, size_t n)
{
    for (size_t off = 0; off < n; off += 256) __asm__ __volatile__("dc civac, %0" ::"r"(p + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}

/* collective. arena: this rank's weight arena (already touched); send: the send buffer. 0 on success. */
int k31x_init(void *arena, size_t arena_bytes, void *send, size_t send_bytes)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &x_me);
    MPI_Comm_size(MPI_COMM_WORLD, &x_world);
    x_arena = (char *) arena; x_arena_bytes = arena_bytes; x_send = (char *) send; x_send_bytes = send_bytes;
    const char *e = getenv("K31X_CHUNK_MB");
    x_chunk = (size_t) (e && atoi(e) > 0 ? atoi(e) : 1024) << 20;
    unsigned long long mx = arena_bytes, gmx = 0;
    MPI_Allreduce(&mx, &gmx, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    x_nch = (int) ((gmx + x_chunk - 1) / x_chunk);
    if (x_nch < 1) x_nch = 1;
    if (x_nch > 256) return -1;
    x_nlocal = (int) ((arena_bytes + x_chunk - 1) / x_chunk);
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni < 1) return -2;
    x_nt = (int) ntni < X_MAXT ? (int) ntni : X_MAXT;
    e = getenv("K31X_TNIS");
    if (e && atoi(e) >= 1 && atoi(e) < x_nt) x_nt = atoi(e);
    struct utofu_onesided_caps *caps = NULL;
    if (utofu_query_onesided_caps(tnis[0], &caps) != UTOFU_SUCCESS) return -3;
    x_putmax = caps->max_putget_size;
    if (x_putmax > (16u << 20)) x_putmax = 16u << 20;
    e = getenv("K31X_PUT_MAX");
    if (e && (size_t) atol(e) > 0 && (size_t) atol(e) < x_putmax) x_putmax = (size_t) atol(e);
    flush_lines(x_arena, arena_bytes);
    for (int k = 0; k < x_nt; ++k) {
        if (utofu_create_vcq(tnis[k], 0, &x_vcq[k]) != UTOFU_SUCCESS) return -4;
        utofu_stadd_t mine[256];
        for (int c = 0; c < x_nch; ++c) {
            mine[c] = 0;
            if (c < x_nlocal) {
                const size_t off = (size_t) c * x_chunk, n = arena_bytes - off < x_chunk ? arena_bytes - off : x_chunk;
                if (utofu_reg_mem(x_vcq[k], x_arena + off, n, 0, &mine[c]) != UTOFU_SUCCESS) {
                    fprintf(stderr, "k31x rank %d: utofu_reg_mem chunk %d (%zu bytes) failed\n", x_me, c, n);
                    return -5;
                }
                x_lbase[k][c] = mine[c];
            }
        }
        if (utofu_reg_mem(x_vcq[k], x_send, send_bytes, 0, &x_sbase[k]) != UTOFU_SUCCESS) return -6;
        utofu_vcq_id_t me;
        if (utofu_query_vcq_id(x_vcq[k], &me) != UTOFU_SUCCESS) return -7;
        x_peer[k] = (utofu_vcq_id_t *) malloc(sizeof(utofu_vcq_id_t) * (size_t) x_world);
        x_rbase[k] = (utofu_stadd_t *) malloc(sizeof(utofu_stadd_t) * (size_t) x_world * (size_t) x_nch);
        MPI_Allgather(&me, (int) sizeof me, MPI_BYTE, x_peer[k], (int) sizeof me, MPI_BYTE, MPI_COMM_WORLD);
        for (int r = 0; r < x_world; ++r) utofu_set_vcq_id_path(&x_peer[k][r], NULL);
        MPI_Allgather(mine, (int) (sizeof(utofu_stadd_t) * (size_t) x_nch), MPI_BYTE, x_rbase[k],
                      (int) (sizeof(utofu_stadd_t) * (size_t) x_nch), MPI_BYTE, MPI_COMM_WORLD);
        x_out[k] = 0;
    }
    free(tnis);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

static int put1(int k, int dst, size_t src_off, size_t dst_off, size_t len)
{
    const int c = (int) (dst_off / x_chunk);
    const size_t in = dst_off - (size_t) c * x_chunk;
    void *cb;
    int rc;
    for (;;) {
        rc = utofu_put(x_vcq[k], x_peer[k][dst], x_sbase[k] + src_off, x_rbase[k][(size_t) dst * x_nch + c] + in, len, 0,
                       UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        if (utofu_poll_tcq(x_vcq[k], 0, &cb) == UTOFU_SUCCESS) --x_out[k];
    }
    if (rc != UTOFU_SUCCESS) return -1;
    ++x_out[k];
    ++x_puts;
    x_bytes += len;
    return 0;
}

/* Put send[src_off, +len) to rank dst's arena at dst_off. 0 on success. */
int k31x_put(int dst, size_t src_off, size_t dst_off, size_t len)
{
    while (len) {
        const size_t edge = (dst_off / x_chunk + 1) * x_chunk - dst_off;
        size_t n = len < edge ? len : edge;
        if (n > x_putmax) n = x_putmax;
        const int k = x_rr; x_rr = (x_rr + 1) % x_nt;
        if (put1(k, dst, src_off, dst_off, n)) return -1;
        src_off += n; dst_off += n; len -= n;
    }
    return 0;
}

/* wait for the local completion of every Put issued since the last flush */
int k31x_flush(void)
{
    void *cb;
    for (int k = 0; k < x_nt; ++k)
        while (x_out[k] > 0) {
            const int rc = utofu_poll_tcq(x_vcq[k], 0, &cb);
            if (rc == UTOFU_ERR_NOT_FOUND) continue;
            if (rc != UTOFU_SUCCESS) return -1;
            --x_out[k];
        }
    return 0;
}

/* after the closing barrier: drop cached lines of the arena so reads see the Put data */
void k31x_settle(void) { flush_lines(x_arena, x_arena_bytes); }

void k31x_stats(unsigned long long *puts, unsigned long long *bytes, int *tnis, size_t *put_max)
{
    *puts = x_puts; *bytes = x_bytes; *tnis = x_nt; *put_max = x_putmax;
}

void k31x_fini(void)
{
    for (int k = 0; k < x_nt; ++k) {
        for (int c = 0; c < x_nlocal; ++c) utofu_dereg_mem(x_vcq[k], x_lbase[k][c], 0);
        utofu_dereg_mem(x_vcq[k], x_sbase[k], 0);
        utofu_free_vcq(x_vcq[k]);
        free(x_peer[k]); free(x_rbase[k]);
    }
    x_nt = 0;
}

/* ------------------------------------------------------------------------------------------------ 2-level all-reduce */
/* k31ar: deterministic sum all-reduce over W = R x C ranks (rank = r C + c). 1. row reduce-scatter: rank sends shard j
 * of its vector to row peer j, which sums the C shards in column order; 2. column exchange: every rank Puts its row-shard
 * sum to its R - 1 column peers and sums the R row sums in row order; 3. row all-gather of the summed shards. 29 Puts per
 * call at 8 x 12 instead of rsag's 2 (W - 1). Every payload carries an 8-byte call token after it (one Put). */
enum { AR_MAXT = 6, AR_MAXW = 256 };
static int ar_R, ar_C, ar_r, ar_c, ar_W, ar_me, ar_nt, ar_max, ar_shard;
static size_t ar_slot;
static char *ar_region;
static size_t ar_rsize;
static utofu_vcq_hdl_t ar_vcq[AR_MAXT];
static utofu_vcq_id_t ar_peer[AR_MAXT][AR_MAXW];
static utofu_stadd_t ar_base[AR_MAXT], ar_pbase[AR_MAXT][AR_MAXW];
static uint64_t ar_seq;
static long ar_out[AR_MAXT];
static int ar_pflush;   /* K31AR_PAYLOAD_FLUSH=1: also drop payload lines before reading (default: token line only, as rsag) */
/* region: two receive banks by call parity (a column peer can be one call ahead: it finishes call n with its own row
 * and Puts call n + 1's column data before this rank read call n's) | send buffers. bank: s1[C] | s2[R] | s3[C] */
static int ar_par;
static size_t ar_bank(void) { return (size_t) ar_par * (size_t) (2 * ar_C + ar_R) * ar_slot; }
static size_t s1_off(int j) { return ar_bank() + (size_t) j * ar_slot; }
static size_t s2_off(int i) { return ar_bank() + (size_t) (ar_C + i) * ar_slot; }
static size_t s3_off(int j) { return ar_bank() + (size_t) (ar_C + ar_R + j) * ar_slot; }
static size_t sb_off(int k) { return (size_t) (2 * (2 * ar_C + ar_R) + k) * ar_slot; }

int k31ar_init(int R, int max_count)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &ar_me);
    MPI_Comm_size(MPI_COMM_WORLD, &ar_W);
    if (R < 1 || ar_W % R || ar_W > AR_MAXW) return -1;
    ar_R = R; ar_C = ar_W / R; ar_r = ar_me / ar_C; ar_c = ar_me % ar_C; ar_max = max_count;
    ar_shard = (max_count + ar_C - 1) / ar_C;
    ar_slot = (((size_t) ar_shard * sizeof(float) + 7) / 8 * 8 + 8 + 255) & ~(size_t) 255;
    ar_rsize = (size_t) (2 * (2 * ar_C + ar_R) + ar_C + 2) * ar_slot;
    if (posix_memalign((void **) &ar_region, 256, ar_rsize)) return -2;
    memset(ar_region, 0, ar_rsize);
    flush_lines(ar_region, ar_rsize);
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni < 1) return -3;
    ar_nt = (int) ntni < AR_MAXT ? (int) ntni : AR_MAXT;
    const char *e = getenv("K31AR_TNIS");
    if (e && atoi(e) >= 1 && atoi(e) < ar_nt) ar_nt = atoi(e);
    for (int k = 0; k < ar_nt; ++k) {
        if (utofu_create_vcq(tnis[k], 0, &ar_vcq[k]) != UTOFU_SUCCESS) return -4;
        if (utofu_reg_mem(ar_vcq[k], ar_region, ar_rsize, 0, &ar_base[k]) != UTOFU_SUCCESS) return -5;
        utofu_vcq_id_t me;
        if (utofu_query_vcq_id(ar_vcq[k], &me) != UTOFU_SUCCESS) return -6;
        MPI_Allgather(&me, (int) sizeof me, MPI_BYTE, ar_peer[k], (int) sizeof me, MPI_BYTE, MPI_COMM_WORLD);
        for (int r = 0; r < ar_W; ++r) utofu_set_vcq_id_path(&ar_peer[k][r], NULL);
        MPI_Allgather(&ar_base[k], (int) sizeof ar_base[k], MPI_BYTE, ar_pbase[k], (int) sizeof ar_base[k], MPI_BYTE, MPI_COMM_WORLD);
        ar_out[k] = 0;
    }
    free(tnis);
    ar_seq = 0;
    { const char *pf = getenv("K31AR_PAYLOAD_FLUSH"); ar_pflush = pf && pf[0] == '1'; }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

static int ar_k;
static int ar_put(int dst, size_t src, size_t dst_off, size_t len)
{
    const int k = ar_k; ar_k = (ar_k + 1) % ar_nt;
    void *cb;
    int rc;
    for (;;) {
        rc = utofu_put(ar_vcq[k], ar_peer[k][dst], ar_base[k] + src, ar_pbase[k][dst] + dst_off, len, 0,
                       UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        if (utofu_poll_tcq(ar_vcq[k], 0, &cb) == UTOFU_SUCCESS) --ar_out[k];
    }
    if (rc != UTOFU_SUCCESS) return -1;
    ++ar_out[k];
    return 0;
}
static void ar_drain(void)
{
    void *cb;
    for (int k = 0; k < ar_nt; ++k)
        while (ar_out[k] > 0) { if (utofu_poll_tcq(ar_vcq[k], 0, &cb) == UTOFU_SUCCESS) --ar_out[k]; }
}
/* wait until slots off(0..n-1) except `skip` carry tok after pl payload bytes; then drop their payload lines */
static int ar_wait(size_t (*off)(int), int n, int skip, size_t pl, uint64_t tok)
{
    int pending = 0;
    unsigned char done[AR_MAXW];
    for (int s = 0; s < n; ++s) { done[s] = s == skip; pending += s != skip; }
    unsigned long passes = 0;
    while (pending) {
        for (int s = 0; s < n; ++s)
            if (!done[s]) __asm__ __volatile__("dc civac, %0" ::"r"(ar_region + off(s) + pl) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        for (int s = 0; s < n; ++s)
            if (!done[s] && *(volatile uint64_t *) (ar_region + off(s) + pl) == tok) {
                done[s] = 1; --pending;
                if (ar_pflush) for (size_t o = 0; o < pl; o += 256) __asm__ __volatile__("dc civac, %0" ::"r"(ar_region + off(s) + o) : "memory");
            }
        if ((++passes & 0xFFFFF) == 0 && passes > (1ul << 26)) {
            for (int s = 0; s < n; ++s)
                if (!done[s]) fprintf(stderr, "k31ar rank %d (r %d c %d) seq %llu: slot %d off %zu pl %zu has %016llx want %016llx\n",
                                      ar_me, ar_r, ar_c, (unsigned long long) ar_seq, s, off(s), pl,
                                      (unsigned long long) *(volatile uint64_t *) (ar_region + off(s) + pl), (unsigned long long) tok);
            return -1;
        }
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    return 0;
}

/* sum all-reduce of x[0..n), n <= max_count; 0 on success */
int k31ar_allreduce(float *x, int n)
{
    if (n > ar_max) return -1;
    const uint64_t tok = 0xA5A5000000000000ull | (++ar_seq & 0xFFFFFFFFFFull);
    ar_par = (int) (ar_seq & 1);
    const int C = ar_C, R = ar_R, c = ar_c, r = ar_r;
    int off[AR_MAXW + 1];
    off[0] = 0;
    for (int j = 0; j < C; ++j) off[j + 1] = off[j] + n / C + (j < n % C);
    const int my = off[c + 1] - off[c];
    /* 1. row reduce-scatter */
    for (int jj = 1; jj < C; ++jj) {
        const int j = (c + jj) % C, len = off[j + 1] - off[j];
        const size_t pl = ((size_t) len * 4 + 7) & ~(size_t) 7;
        char *sb = ar_region + sb_off(j);
        memcpy(sb, x + off[j], (size_t) len * 4);
        *(uint64_t *) (sb + pl) = tok;
        if (ar_put(r * C + j, sb_off(j), s1_off(c), pl + 8)) { fprintf(stderr, "k31ar rank %d: put failed\n", ar_me); return -2; }
    }
    const size_t mypl = ((size_t) my * 4 + 7) & ~(size_t) 7;
    if (ar_wait(s1_off, C, c, mypl, tok)) { fprintf(stderr, "k31ar rank %d: stage 1 timeout\n", ar_me); return -3; }
    float *rs = (float *) (ar_region + sb_off(C));   /* my row-shard sum (also the column send buffer) */
    /* slot by slot over contiguous shards (vectorizes); each element still summed in column order 0..C-1 */
    for (int j = 0; j < C; ++j) {
        const float *v = j == c ? x + off[c] : (const float *) (ar_region + s1_off(j));
        if (j == 0) for (int i = 0; i < my; ++i) rs[i] = 0.0f + v[i];
        else for (int i = 0; i < my; ++i) rs[i] += v[i];
    }
    /* 2. column exchange of the row-shard sums */
    *(uint64_t *) ((char *) rs + mypl) = tok;
    for (int ii = 1; ii < R; ++ii) {
        const int i = (r + ii) % R;
        if (ar_put(i * C + c, sb_off(C), s2_off(r), mypl + 8)) return -4;
    }
    if (R > 1 && ar_wait(s2_off, R, r, mypl, tok)) { fprintf(stderr, "k31ar rank %d: stage 2 timeout\n", ar_me); return -5; }
    float *tot = (float *) (ar_region + sb_off(C + 1));
    for (int i = 0; i < R; ++i) {
        const float *v = i == r ? rs : (const float *) (ar_region + s2_off(i));
        if (i == 0) for (int k = 0; k < my; ++k) tot[k] = 0.0f + v[k];
        else for (int k = 0; k < my; ++k) tot[k] += v[k];
    }
    /* 3. row all-gather */
    *(uint64_t *) ((char *) tot + mypl) = tok;
    for (int jj = 1; jj < C; ++jj) {
        const int j = (c + jj) % C;
        if (ar_put(r * C + j, sb_off(C + 1), s3_off(c), mypl + 8)) return -6;
    }
    memcpy(x + off[c], tot, (size_t) my * 4);
    /* every peer's shard length differs by at most one: wait per slot with its own payload size */
    for (int jj = 1; jj < C; ++jj) {
        const int j = (c + jj) % C, len = off[j + 1] - off[j];
        const size_t pl = ((size_t) len * 4 + 7) & ~(size_t) 7;
        while (*(volatile uint64_t *) (ar_region + s3_off(j) + pl) != tok)
            __asm__ __volatile__("dc civac, %0\n dsb sy" ::"r"(ar_region + s3_off(j) + pl) : "memory");
        if (ar_pflush) for (size_t o = 0; o < pl; o += 256) __asm__ __volatile__("dc civac, %0" ::"r"(ar_region + s3_off(j) + o) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        memcpy(x + off[j], ar_region + s3_off(j), (size_t) len * 4);
    }
    ar_drain();
    return 0;
}
