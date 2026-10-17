// tools/muse_split_check.cpp - the two-node expert row split on a real Muse GGUF, without a GPU: for every layer, random
// latents and routes go through this node's full rows (one node) and through the split (rows [0, cut) here, the rest
// on strata-muse-expert-worker over ucomm); the two must agree within float rounding.
//
//     muse_split_check MODEL.gguf [SHARD...] --worker-root HOST:PORT [--share-b 0.375] [--threads T] [--tokens N]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/muse_model.hpp"
#include "strata/host/muse_names.h"
#include "strata/muse/row_split.hpp"
#include "strata/muse/split_lane.hpp"

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace muse = strata::muse;

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    std::string root;
    double share_b = 0.375;
    int threads = 15, tokens = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--worker-root") root = next();
        else if (a == "--share-b") share_b = std::stod(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--tokens") tokens = std::stoi(next());
        else shards.push_back(a);
    }
    if (shards.empty() || root.empty()) { std::fprintf(stderr, "usage: muse_split_check MODEL.gguf --worker-root HOST:PORT\n"); return 2; }
    const strata::GgufModel gm(shards);
    muse::MuseConfig c;
    if (const std::string e = muse::read_config(gm.meta(), c); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    const int L = (int) c.latent, FF = (int) c.expert_ff, K = c.top_k, cut = (int) strata_muse_row_split(FF, 256, share_b);
    auto data = [&](const std::string& name, const strata::TensorInfo** info = nullptr) {
        size_t shard = 0;
        const strata::TensorInfo* p = gm.find(name, &shard);
        if (!p) { std::fprintf(stderr, "missing %s\n", name.c_str()); std::exit(1); }
        if (info) *info = p;
        return gm.shard(shard).tensor_data(*p);
    };
    const strata::TensorInfo *gi = nullptr, *di = nullptr;
    data("blk.0.ffn_gate_exps.weight", &gi);
    data("blk.0.ffn_down_exps.weight", &di);
    auto fmt = [&](int ff) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) gi->type, (int) di->type, L, ff, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
        f.iq1s_rows = (int) gi->type == GGML_TYPE_IQ1_S;
        return f;
    };
    std::vector<muse::ExpertShare> share;
    std::vector<std::vector<float>> post((size_t) c.layers);
    std::vector<muse::LaneLayer> full, local;
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const uint8_t *g = data(p + "ffn_gate_exps.weight"), *u = data(p + "ffn_up_exps.weight"), *dn = data(p + "ffn_down_exps.weight");
        const float* pe = reinterpret_cast<const float*>(data(p + "ffn_post_exp_norm.weight"));
        post[(size_t) l].assign(pe, pe + L);
        share.push_back(muse::slice_experts(g, u, dn, (int) c.experts, L, FF, (int) gi->type, (int) di->type, 0, cut));
        full.push_back({g, u, dn, post[(size_t) l].data()});
    }
    for (int l = 0; l < c.layers; ++l) local.push_back({share[(size_t) l].gate.data(), share[(size_t) l].up.data(), share[(size_t) l].down.data(), post[(size_t) l].data()});
    muse::ExpertLane whole(fmt(FF), L, FF, (float) c.post_eps, threads), part(fmt(cut), L, cut, (float) c.post_eps, threads);
    ucomm_config cfg;
    ucomm_config_default(&cfg);
    cfg.rank = 0; cfg.world = 2; cfg.root_addr = root.c_str();
    ucomm_t* comm = nullptr;
    if (ucomm_init(&cfg, &comm) != UCOMM_OK) { std::fprintf(stderr, "ucomm init failed\n"); return 1; }
    muse::SplitLane split(part, comm, 1, L, (float) c.post_eps);
    std::mt19937 rng(62);
    std::normal_distribution<float> nd;
    int fails = 0;
    double ms_full = 0, ms_split = 0;
    for (int l = 0; l < c.layers; ++l)
        for (int nt : {1, tokens}) {
            std::vector<float> z((size_t) nt * L), w((size_t) nt * K), y0((size_t) nt * L), y1((size_t) nt * L);
            std::vector<int32_t> ids((size_t) nt * K);
            for (auto& v : z) v = nd(rng);
            for (int t = 0; t < nt; ++t) {
                float s = 0;
                for (int k = 0; k < K; ++k) { ids[t * K + k] = (int) (rng() % c.experts); s += w[t * K + k] = 0.05f + (rng() % 100) / 100.f; }
                for (int k = 0; k < K; ++k) w[t * K + k] /= s;
            }
            auto t0 = std::chrono::steady_clock::now();
            whole.run(full[(size_t) l], z.data(), nt, ids.data(), w.data(), K, y0.data());
            auto t1 = std::chrono::steady_clock::now();
            split.run(l, local[(size_t) l], z.data(), nt, ids.data(), w.data(), K, y1.data());
            auto t2 = std::chrono::steady_clock::now();
            ms_full += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_split += std::chrono::duration<double, std::milli>(t2 - t1).count();
            double err = 0, scale = 0;
            for (size_t i = 0; i < y0.size(); ++i) { err = std::max(err, (double) std::fabs(y0[i] - y1[i])); scale = std::max(scale, (double) std::fabs(y0[i])); }
            const bool ok = err / scale < 1e-5;
            std::printf("layer %d, %d token(s): split vs one node relative %.2e  %s\n", l, nt, err / scale, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    split.stop();
    std::printf("over %s: rows [0, %d) here, [%d, %d) on the worker; %.1f ms one node, %.1f ms split (untimed run, shared machines)\n",
                ucomm_backend_name(comm), cut, cut, FF, ms_full, ms_split);
    ucomm_finalize(comm);
    std::printf("muse_split_check: %d failures\n", fails);
    return fails ? 1 : 0;
}
