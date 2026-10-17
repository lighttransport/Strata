/* tools/a64fx_probe/node_probe.c - per-node constants of the A64FX simulator (tools/sim/a64fx.py):
 *   1. HBM read stream (SVE loads, every thread its own first-touched slice) at 12 threads (one CMG) and 48 (node);
 *   2. barrier latency: flat OpenMP barrier at 12 and 48 threads, and the libhwb EL0 hardware barrier (one CMG at
 *      12 threads; hierarchical 4 x HW + software leader combine at 48 threads, as gemm's ring_attn_async does);
 *   3. fork-join: an empty `omp parallel` region at 48 threads.
 * Build: fcc -Nclang -O3 -fopenmp -march=armv8.2-a+sve node_probe.c -o node_probe [-DHAVE_HWB -lhwb]
 * Output lines start with A64FX_NODE. */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static int cores[64], ncores;

static void pin(int tid) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid % ncores], &s); sched_setaffinity(0, sizeof s, &s); }

static double stream_gbps(int nthr, size_t bytes_per_thread, int reps)
{
    uint8_t **buf = calloc(nthr, sizeof *buf);
    double best = 1e30; volatile uint64_t sink = 0;
    #pragma omp parallel num_threads(nthr)
    {
        const int tid = omp_get_thread_num(); pin(tid);
        buf[tid] = aligned_alloc(256, bytes_per_thread);
        memset(buf[tid], tid + 1, bytes_per_thread);              /* first touch: CMG-local pages */
        for (int r = 0; r < reps; ++r) {
            #pragma omp barrier
            double t0 = 0;
            #pragma omp master
            t0 = now();
            const svbool_t p = svptrue_b64();
            svuint64_t a0 = svdup_u64(0), a1 = a0, a2 = a0, a3 = a0;
            const uint64_t *q = (const uint64_t *) buf[tid];
            const size_t n = bytes_per_thread / 8, step = svcntd();
            for (size_t i = 0; i + 4 * step <= n; i += 4 * step) {
                a0 = svadd_u64_x(p, a0, svld1_u64(p, q + i));
                a1 = svadd_u64_x(p, a1, svld1_u64(p, q + i + step));
                a2 = svadd_u64_x(p, a2, svld1_u64(p, q + i + 2 * step));
                a3 = svadd_u64_x(p, a3, svld1_u64(p, q + i + 3 * step));
            }
            const uint64_t s = svaddv_u64(p, svadd_u64_x(p, svadd_u64_x(p, a0, a1), svadd_u64_x(p, a2, a3)));
            #pragma omp atomic
            sink += s;
            #pragma omp barrier
            #pragma omp master
            { const double dt = now() - t0; if (dt < best) best = dt; }
        }
        free(buf[tid]);
    }
    free(buf);
    return (double) nthr * bytes_per_thread / best / 1e9;
}

static double omp_barrier_us(int nthr, int iters)
{
    double dt = 0;
    #pragma omp parallel num_threads(nthr)
    {
        pin(omp_get_thread_num());
        for (int i = 0; i < 1000; ++i) {
            #pragma omp barrier
        }
        double t0 = now();
        for (int i = 0; i < iters; ++i) {
            #pragma omp barrier
        }
        #pragma omp master
        dt = now() - t0;
    }
    return dt / iters * 1e6;
}

static double fork_join_us(int nthr, int iters)
{
    volatile int x = 0;
    #pragma omp parallel num_threads(nthr)
    pin(omp_get_thread_num());
    const double t0 = now();
    for (int i = 0; i < iters; ++i) {
        #pragma omp parallel num_threads(nthr)
        { if (omp_get_thread_num() == 0) x++; }
    }
    return (now() - t0) / iters * 1e6;
}

#ifdef HAVE_HWB
extern int  vhbm_bar_init(uint64_t core_bitmask);
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint);
extern void vhbm_bar(long bb);
extern int  vhbm_bar_unassign(int bd_mask);

static volatile int g_lcount = 0; static char pad1[60] __attribute__((unused));
static volatile int g_lsense = 0; static char pad2[60] __attribute__((unused));

/* nthr = 12: one CMG, pure HW. nthr = 48: per-CMG HW arrival, 4-way software leader combine, HW release. */
static double hwb_us(int nthr, int iters)
{
    uint64_t mask = 0; for (int t = 0; t < nthr; ++t) mask |= 1ull << cores[t];
    const int bd = vhbm_bar_init(mask);
    if (bd < 0) { printf("A64FX_NODE hwb_init_failed threads=%d rc=%d\n", nthr, bd); return -1; }
    const int tpc = 12, ncmg = nthr / tpc; volatile int failed = 0; double dt = 0;
    #pragma omp parallel num_threads(nthr)
    {
        const int tid = omp_get_thread_num(); pin(tid);
        long bb = vhbm_bar_assign(bd, NULL); if (bb < 0) failed = 1;
        int ls = 0;
        #pragma omp barrier
        if (!failed) {
            double t0 = 0;
            for (int i = 0; i < iters + 1000; ++i) {
                if (i == 1000 && tid == 0) t0 = now();
                vhbm_bar(bb);
                if (ncmg > 1) {
                    if (tid % tpc == 0) {
                        const int my = !ls;
                        if (__sync_add_and_fetch(&g_lcount, 1) == ncmg) { g_lcount = 0; __sync_synchronize(); g_lsense = my; }
                        else while (g_lsense != my) __asm__ volatile("yield");
                        ls = my;
                    }
                    vhbm_bar(bb);
                }
            }
            if (tid == 0) dt = now() - t0;
        }
        #pragma omp barrier
    }
    vhbm_bar_unassign(bd);
    return failed ? -1 : dt / iters * 1e6;
}
#endif

int main(int argc, char **argv)
{
    cpu_set_t s; sched_getaffinity(0, sizeof s, &s);
    for (int c = 0; c < 64; ++c) if (CPU_ISSET(c, &s)) cores[ncores++] = c;
    const size_t mb = argc > 1 ? (size_t) atol(argv[1]) : 256;   /* per thread */
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("A64FX_NODE cores=%d first=%d last=%d sve_bytes=%d\n", ncores, cores[0], cores[ncores - 1], (int) svcntb());
    printf("A64FX_NODE stream_read threads=12 MB_per_thread=%zu GBps=%.1f\n", mb, stream_gbps(12, mb << 20, 5));
    printf("A64FX_NODE stream_read threads=48 MB_per_thread=%zu GBps=%.1f\n", mb, stream_gbps(48, mb << 20, 5));
    printf("A64FX_NODE omp_barrier threads=12 us=%.2f\n", omp_barrier_us(12, 100000));
    printf("A64FX_NODE omp_barrier threads=48 us=%.2f\n", omp_barrier_us(48, 100000));
    printf("A64FX_NODE omp_fork_join threads=48 us=%.2f\n", fork_join_us(48, 20000));
#ifdef HAVE_HWB
    printf("A64FX_NODE hw_barrier threads=12 us=%.2f\n", hwb_us(12, 100000));
    printf("A64FX_NODE hw_barrier_hier threads=48 us=%.2f\n", hwb_us(48, 100000));
#else
    printf("A64FX_NODE hw_barrier skipped (built without HAVE_HWB)\n");
#endif
    return 0;
}
