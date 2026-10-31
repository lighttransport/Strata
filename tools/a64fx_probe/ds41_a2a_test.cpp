// Bit transport and timing probe for the CPU runner's variable all-to-all.
// Run on the same 12-node allocation with STRATA_RUNNER_A2A_PAR=0 and =1.
// Build with infra.hpp's MPI/uTofu flags, comm_utofu.c, glm_env.c and -ltofucom.
#include "strata/cpu_runner/infra.hpp"
#include <array>
#include <cstdio>

extern "C" void cpu_utofu_set_par(void (*)(void (*)(void *, int, int), void *, int));
using namespace strata::cpu_runner;

static int copy_calls = 0;
static void copy_team(void (*fn)(void *, int, int), void *arg, int n) {
    ++copy_calls;
    team_run([&] {
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        fn(arg, int(int64_t(n) * t / nt), int(int64_t(n) * (t + 1) / nt));
    });
}

static uint32_t word(int src, int dst, int i, int generation) {
    static constexpr uint32_t edges[] = {
        0, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x7fa12345u, 1, 0x80000001u
    };
    if (i == 0) return 0x7fc00000u | uint32_t(src << 14) | uint32_t(dst << 8) | uint32_t(generation);
    if (i <= 8) return edges[(i - 1 + generation) % 8];
    return 0x9e3779b9u * uint32_t(i + 1) ^ uint32_t(src * 977 + dst * 131 + generation * 65537);
}

static int count(int base, int src, int dst) {
    return base < 0 ? 0 : base + (src * 7 + dst * 11) % 13;
}

int main(int argc, char **argv) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    Comm comm;
    comm.init(&argc, &argv);
    cpu_utofu_set_par(copy_team);
    const int expect_parallel = argc > 1 ? std::atoi(argv[1]) : -1;
    g_bar.init(omp_get_max_threads(), cmg_threads());
    if (!comm.xc_init(65552)) {
        if (!comm.rank) std::printf("A2A_INIT_FAILED\n");
        comm.finish();
        return 1;
    }
    int local_bad = 0;
    with_team([&] {
        for (int base : {-1, 0, 1, 63, 256, 4096, 16384, 65536}) {
            std::vector<int> send_off(size_t(comm.world) + 1), recv_off(size_t(comm.world) + 1);
            for (int r = 0; r < comm.world; ++r) {
                send_off[size_t(r) + 1] = send_off[size_t(r)] + count(base, comm.rank, r);
                recv_off[size_t(r) + 1] = recv_off[size_t(r)] + count(base, r, comm.rank);
            }
            constexpr int guard = 16;
            constexpr uint32_t sentinel = 0xdeadbeefu;
            std::vector<float> send(size_t(send_off.back()) + 2 * guard), recv(size_t(recv_off.back()) + 2 * guard);
            for (auto *v : {&send, &recv})
                for (size_t i = 0; i < v->size(); ++i) std::memcpy(v->data() + i, &sentinel, sizeof sentinel);
            float *const sp = send.data() + guard, *const rp = recv.data() + guard;
            int bad = 0;
            // Change every payload between calls so stale notifications/slots cannot pass a bit comparison.
            for (int gen = 0; gen < 6; ++gen) {
                for (int r = 0; r < comm.world; ++r)
                    for (int i = 0; i < send_off[size_t(r) + 1] - send_off[size_t(r)]; ++i) {
                        const uint32_t bits = word(comm.rank, r, i, gen);
                        std::memcpy(sp + send_off[size_t(r)] + i, &bits, sizeof bits);
                    }
                comm.alltoall(sp, send_off.data(), rp, recv_off.data());
                for (int r = 0; r < comm.world; ++r)
                    for (int i = 0; i < recv_off[size_t(r) + 1] - recv_off[size_t(r)]; ++i) {
                        uint32_t got;
                        std::memcpy(&got, rp + recv_off[size_t(r)] + i, sizeof got);
                        bad += got != word(r, comm.rank, i, gen);
                    }
                for (auto *v : {&send, &recv})
                    for (int i = 0; i < guard; ++i) {
                        uint32_t lo, hi;
                        std::memcpy(&lo, v->data() + i, sizeof lo);
                        std::memcpy(&hi, v->data() + v->size() - guard + i, sizeof hi);
                        bad += lo != sentinel || hi != sentinel;
                    }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            constexpr int reps = 30;
            const double start = now();
            for (int rep = 0; rep < reps; ++rep) comm.alltoall(sp, send_off.data(), rp, recv_off.data());
            const double elapsed = now() - start;
            double worst = 0;
            int all_bad = 0;
            MPI_Allreduce(&elapsed, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&bad, &all_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            local_bad += bad;
            if (!comm.rank) {
                std::printf("A2A base=%d ranks=%d threads=%d bytes=%zu us=%.2f fails=%d\n", base, comm.world,
                            omp_get_num_threads(), size_t(send_off.back()) * sizeof(float), 1e6 * worst / reps, all_bad);
                std::fflush(stdout);
            }
        }
    });
    if (expect_parallel >= 0 && (copy_calls > 0) != (expect_parallel != 0)) ++local_bad;
    int all_bad = 0;
    MPI_Allreduce(&local_bad, &all_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (!comm.rank) std::printf("A2A_TEST fails=%d par_calls=%d\n", all_bad, copy_calls);
    comm.finish();
    return all_bad != 0;
}
