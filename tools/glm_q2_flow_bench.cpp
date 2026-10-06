// Expert-phase throughput of the decoder's CPU path without the model or the GPU: synthetic Q2_K gate/up and
// Q3_K down experts in node-owned rows, random routes, ExpertPool::run_layer_native (STRATA_GLM_LAYER_FLOW=1) or
// the phased run_split_multi_native. Reports unique expert bytes per second.
// usage: strata-glm-q2-flow-bench [tokens=1] [experts=96] [layers=300] [flow|phased] [workers=15]
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels::cpu;

int main(int argc, char** argv) {
    try {
        const int nt = argc > 1 ? std::atoi(argv[1]) : 1, count = argc > 2 ? std::atoi(argv[2]) : 96;
        const int layers = argc > 3 ? std::atoi(argv[3]) : 300;
        const bool flow = argc <= 4 || std::string(argv[4]) == "flow";
        const int workers = argc > 5 ? std::atoi(argv[5]) : 15;
        const int K = 8, H = 4096, FF = 2048;
        if (nt < 1 || nt > 4 || count < K * nt || layers < 1) throw std::invalid_argument("invalid arguments");
        ExpertPool pool(workers, true);
        if (!pool.numa_rows_available()) throw std::runtime_error("node-owned rows unavailable");
        const auto previous = pin_current_thread(physical_cores(false).front());
        NativeFmt f;
        std::string error;
        if (!native_fmt(10, 11, H, FF, f, error)) throw std::runtime_error(error);
        f.swiglu_limit = 10;
        f.fuse_h_quant = true;
        std::mt19937 rng(2026);
        const size_t gu_bytes = f.up_off, d_bytes = f.bytes - f.down_off;
        std::vector<uint8_t> g((size_t)count * gu_bytes), u(g.size()), d((size_t)count * d_bytes);
        auto fill = [&](std::vector<uint8_t>& v, size_t block, size_t scale_at, int scales) {
            for (auto& b : v) b = (uint8_t)rng();
            const uint16_t h = 0x2400;
            for (size_t o = 0; o + block <= v.size(); o += block)
                for (int s = 0; s < scales; ++s) std::memcpy(v.data() + o + scale_at + 2 * s, &h, 2);
        };
        fill(g, 84, 80, 2); fill(u, 84, 80, 2); fill(d, 110, 108, 1);
        NumaTensor ng(g.data(), f.gu_row, FF, count, pool.numa_cores()), nu(u.data(), f.gu_row, FF, count, pool.numa_cores()),
            nd(d.data(), f.d_row, H, count, pool.numa_cores());
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.act_bytes));
        std::vector<float> input(H);
        for (int t = 0; t < nt; ++t) {
            for (auto& v : input) v = float(int(rng() % 2001) - 1000) * .001f;
            native_quant_act(f, input.data(), act[t].data());
        }
        std::vector<float> results((size_t)nt * K * H), sum((size_t)nt * H), weights(nt * K, .125f);
        std::vector<ExpertJobMulti> jobs;
        std::vector<int> selected(nt * K), job_of(count);
        double seconds = 0, bytes = 0, experts = 0;
        for (int layer = -20; layer < layers; ++layer) {   // 20 warm-up layers
            for (int t = 0; t < nt; ++t) {
                std::vector<int> ids(count);
                for (int e = 0; e < count; ++e) ids[e] = e;
                for (int k = 0; k < K; ++k) std::swap(ids[k], ids[k + rng() % (count - k)]);
                for (int k = 0; k < K; ++k) selected[t * K + k] = ids[k];
            }
            jobs.clear();
            std::fill(job_of.begin(), job_of.end(), -1);
            for (int j = 0; j < nt * K; ++j) {
                const int e = selected[j];
                if (job_of[e] < 0) {
                    job_of[e] = (int)jobs.size();
                    auto& job = jobs.emplace_back();
                    job.expert_id = e;
                    job.blob = g.data() + (size_t)e * gu_bytes;
                    job.native_up = u.data() + (size_t)e * gu_bytes;
                    job.native_down = d.data() + (size_t)e * d_bytes;
                    for (int node = 0; node < 2; ++node) job.numa[node] = {ng.shard(e, node), nu.shard(e, node), nd.shard(e, node)};
                }
                auto& job = jobs[job_of[e]];
                const int slot = job.nt++;
                job.nact[slot] = act[j / K].data();
                job.out[slot] = results.data() + (size_t)j * H;
            }
            const auto start = std::chrono::steady_clock::now();
            if (flow) pool.run_layer_native(f, jobs.data(), (int)jobs.size(), results.data(), weights.data(), sum.data(), nt, K);
            else {
                pool.run_split_multi_native(f, jobs.data(), (int)jobs.size());
                pool.reduce_routed(results.data(), weights.data(), sum.data(), nt, K, H);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (layer >= 0) { seconds += s; bytes += double(jobs.size()) * f.bytes; experts += jobs.size(); }
        }
        restore_thread_affinity(previous);
        std::printf("FLOW_BENCH mode=%s tokens=%d workers=%d experts_per_layer=%.1f us_per_layer=%.1f GB_s=%.2f\n",
                    flow ? "flow" : "phased", nt, workers, experts / layers, seconds / layers * 1e6, bytes / seconds / 1e9);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
