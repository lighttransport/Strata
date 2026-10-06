// run_layer_native against run_split_multi_native + reduce_routed: plain and NUMA-owned rows, IQ and fused
// Q2_K/Q3_K formats, shared experts across tokens, and routes left to the GPU (zero weight and rows).
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "ggml.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
using namespace strata::kernels::cpu;
struct Weights {
    NativeFmt f;
    int count = 0;
    std::vector<uint8_t> g, u, d;
    std::unique_ptr<NumaTensor> ng, nu, nd;
    void view() {
        ng = std::make_unique<NumaTensor>(NumaTensor::View{}, g.data(), f.gu_row, f.n_ff, count);
        nu = std::make_unique<NumaTensor>(NumaTensor::View{}, u.data(), f.gu_row, f.n_ff, count);
        nd = std::make_unique<NumaTensor>(NumaTensor::View{}, d.data(), f.d_row, f.n_embd, count);
    }
};
Weights make(int gu_type, int down_type, int hidden, int ff, int count, bool quantized, std::mt19937 &rng,
             const ExpertPool &pool) {
    Weights w;
    std::string error;
    if (!native_fmt(gu_type, down_type, hidden, ff, w.f, error)) throw std::runtime_error(error);
    w.f.swiglu_limit = 10;
    w.count = count;
    w.g.resize((size_t)count * w.f.up_off);
    w.u.resize(w.g.size());
    w.d.resize((size_t)count * (w.f.bytes - w.f.down_off));
    if (quantized) {
        std::vector<float> values((size_t)hidden * ff);
        for (auto *dest : {&w.g, &w.u, &w.d})
            for (int e = 0; e < count; ++e) {
                for (auto &v : values) v = float(int(rng() % 2001) - 1000) * .0001f;
                const bool down = dest == &w.d;
                ggml_quantize_chunk(ggml_type(down ? down_type : gu_type), values.data(),
                                    dest->data() + (size_t)e * (down ? w.f.bytes - w.f.down_off : w.f.up_off), 0,
                                    down ? hidden : ff, down ? ff : hidden, nullptr);
            }
    } else {
        for (auto *v : {&w.g, &w.u, &w.d})
            for (auto &b : *v) b = (uint8_t)rng();
        // Small FP16 scales keep random IQ codes finite.
        for (int e = 0; e < count; ++e)
            for (int part = 0; part < 3; ++part) {
                auto &v = part == 0 ? w.g : part == 1 ? w.u : w.d;
                const int rows = part == 2 ? hidden : ff;
                const size_t stride = part == 2 ? w.f.d_row : w.f.gu_row;
                const size_t base = (size_t)e * (part == 2 ? w.f.bytes - w.f.down_off : w.f.up_off);
                for (int row = 0; row < rows; ++row)
                    for (size_t block = 0; block < stride; block += (part == 2 ? 98 : 74)) {
                        const uint16_t scale = 0x2400;
                        if (block + 2 <= stride) std::memcpy(v.data() + base + row * stride + block, &scale, 2);
                    }
            }
    }
    w.ng = std::make_unique<NumaTensor>(w.g.data(), w.f.gu_row, ff, count, pool.numa_cores());
    w.nu = std::make_unique<NumaTensor>(w.u.data(), w.f.gu_row, ff, count, pool.numa_cores());
    w.nd = std::make_unique<NumaTensor>(w.d.data(), w.f.d_row, hidden, count, pool.numa_cores());
    return w;
}
int run(ExpertPool &pool, Weights &w, bool owned, bool fuse, int nt, int skip_expert, std::mt19937 &rng) {
    const int K = 8, H = (int)w.f.n_embd;
    w.f.fuse_h_quant = fuse;
    std::vector<int> selected(nt * K);
    for (int t = 0; t < nt; ++t) {
        std::vector<int> pool_ids(w.count);
        for (int e = 0; e < w.count; ++e) pool_ids[e] = e;
        std::shuffle(pool_ids.begin(), pool_ids.end(), rng);
        for (int k = 0; k < K; ++k) selected[t * K + k] = pool_ids[k];
    }
    std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(w.f.act_bytes));
    std::vector<float> input(H);
    for (int t = 0; t < nt; ++t) {
        for (auto &v : input) v = float(int(rng() % 2001) - 1000) * .001f;
        native_quant_act(w.f, input.data(), act[t].data());
    }
    std::vector<float> weights(nt * K);
    for (auto &v : weights) v = float(int(rng() % 201) - 100) * .01f;
    auto build = [&](std::vector<float> &results, std::vector<ExpertJobMulti> &jobs) {
        jobs.clear();
        std::vector<int> job_of(w.count, -1);
        for (int j = 0; j < nt * K; ++j) {
            const int e = selected[j];
            if (e == skip_expert) {
                std::fill_n(results.data() + (size_t)j * H, H, 0.f);
                continue;
            }
            if (job_of[e] < 0) {
                job_of[e] = (int)jobs.size();
                auto &job = jobs.emplace_back();
                job.expert_id = e;
                job.blob = w.g.data() + (size_t)e * w.f.up_off;
                job.native_up = w.u.data() + (size_t)e * w.f.up_off;
                job.native_down = w.d.data() + (size_t)e * (w.f.bytes - w.f.down_off);
                if (owned)
                    for (int node = 0; node < 2; ++node)
                        job.numa[node] = {w.ng->shard(e, node), w.nu->shard(e, node), w.nd->shard(e, node)};
            }
            auto &job = jobs[job_of[e]];
            const int slot = job.nt++;
            job.nact[slot] = act[j / K].data();
            job.out[slot] = results.data() + (size_t)j * H;
        }
    };
    std::vector<float> masked = weights;
    for (int j = 0; j < nt * K; ++j)
        if (selected[j] == skip_expert) masked[j] = 0.f;
    std::vector<float> ref_results((size_t)nt * K * H), ref_sum((size_t)nt * H);
    std::vector<ExpertJobMulti> jobs;
    build(ref_results, jobs);
    pool.run_split_multi_native(w.f, jobs.data(), (int)jobs.size());
    pool.reduce_routed(ref_results.data(), masked.data(), ref_sum.data(), nt, K, H);
    for (int repeat = 0; repeat < 6; ++repeat) {
        std::vector<float> results((size_t)nt * K * H, std::numeric_limits<float>::quiet_NaN()),
            sum((size_t)nt * H, std::numeric_limits<float>::quiet_NaN());
        build(results, jobs);
        pool.run_layer_native(w.f, jobs.data(), (int)jobs.size(), results.data(), masked.data(), sum.data(), nt, K);
        if (std::memcmp(results.data(), ref_results.data(), results.size() * 4) ||
            std::memcmp(sum.data(), ref_sum.data(), sum.size() * 4))
            throw std::runtime_error("layer dataflow differs: owned=" + std::to_string(owned) + " fuse=" +
                                     std::to_string(fuse) + " nt=" + std::to_string(nt));
    }
    return 6;
}
int main() {
    try {
        ExpertPool pool(15, true);
        if (!pool.numa_rows_available()) return 77;
        const auto cores = physical_cores(false);
        const auto previous = pin_current_thread(cores.front());
        std::mt19937 rng(2026);
        int checks = 0;
        auto iq = make(17, 18, 4096, 2048, 12, false, rng, pool);
        for (bool owned : {false, true})
            for (int nt : {1, 2, 4})
                for (int skip : {-1, 3}) checks += run(pool, iq, owned, false, nt, skip, rng);
        auto q23 = make(10, 11, 4096, 2048, 10, true, rng, pool);
        for (bool fuse : {false, true})
            for (int nt : {1, 3}) checks += run(pool, q23, true, fuse, nt, 5, rng);
        // Shards viewing the source buffers (no copies) must give the same results as owned copies.
        iq.view();
        for (int nt : {1, 4}) checks += run(pool, iq, true, false, nt, 3, rng);
        q23.view();
        for (int nt : {1, 3}) checks += run(pool, q23, true, true, nt, -1, rng);
        restore_thread_affinity(previous);
        std::cout << "Layer dataflow PASS " << checks << " runs bitwise equal to phased execution\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
