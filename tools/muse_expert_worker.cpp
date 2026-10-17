// tools/muse_expert_worker.cpp - strata-muse-expert-worker: node 1 of Muse Spark's two-node expert row split.
//
//     strata-muse-expert-worker MODEL.gguf [SHARD.gguf...] --root HOST:PORT [--share-b 0.375] [--threads T]
//
// Holds ff rows [cut, ff) of every routed expert (cut = strata_muse_row_split(ff, 256, share_b), the same rule the
// primary uses), as ucomm rank 1 of 2 with rank 0 (strata-muse-decode --worker-root HOST:PORT) bootstrapping at HOST:PORT,
// and answers each MoE layer's request with its partial expert outputs (strata/muse/split_lane.hpp) until the primary
// stops it. The share is copied out of the GGUF into RAM at start (the release's 0.375 share is 54.5 GiB; a packer
// that fetches only these rows is not written yet). No GPU.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/muse_model.hpp"
#include "strata/host/muse_names.h"
#include "strata/muse/row_split.hpp"
#include "strata/muse/split_lane.hpp"

#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace muse = strata::muse;

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    std::string root;
    double share_b = 0.375;
    int threads = 15;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--root") root = next();
        else if (a == "--share-b") share_b = std::stod(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        else shards.push_back(a);
    }
    if (shards.empty() || root.empty()) {
        std::fprintf(stderr, "usage: strata-muse-expert-worker MODEL.gguf [SHARD...] --root HOST:PORT [--share-b 0.375] [--threads T]\n");
        return 2;
    }
    const strata::GgufModel gm(shards);
    muse::MuseConfig c;
    const std::string err = muse::read_config(gm.meta(), c);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int L = (int) c.latent, FF = (int) c.expert_ff;
    const int cut = (int) strata_muse_row_split(FF, 256, share_b);
    if (cut < 0 || cut >= FF) { std::fprintf(stderr, "share %.3f leaves node 1 no rows\n", share_b); return 1; }
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
    cpu::NativeFmt f;
    std::string ferr;
    if (!cpu::native_fmt((int) gi->type, (int) di->type, L, FF - cut, f, ferr)) { std::fprintf(stderr, "%s\n", ferr.c_str()); return 1; }
    f.iq1s_rows = (int) gi->type == GGML_TYPE_IQ1_S;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<muse::ExpertShare> share;
    std::vector<std::vector<float>> post((size_t) c.layers);
    std::vector<muse::LaneLayer> layers;
    size_t bytes = 0;
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        share.push_back(muse::slice_experts(data(p + "ffn_gate_exps.weight"), data(p + "ffn_up_exps.weight"), data(p + "ffn_down_exps.weight"),
                                            (int) c.experts, L, FF, (int) gi->type, (int) di->type, cut, FF));
        const strata::TensorInfo* pi = nullptr;
        const float* pe = reinterpret_cast<const float*>(data(p + "ffn_post_exp_norm.weight", &pi));
        if (pi->type != GGML_TYPE_F32) { std::fprintf(stderr, "ffn_post_exp_norm must be F32\n"); return 1; }
        post[(size_t) l].assign(pe, pe + L);
        bytes += share.back().gate.size() * 3;   // gate + up + down are the same size per expert row
    }
    for (int l = 0; l < c.layers; ++l)
        layers.push_back({share[(size_t) l].gate.data(), share[(size_t) l].up.data(), share[(size_t) l].down.data(), post[(size_t) l].data()});
    std::fprintf(stderr, "muse expert worker: ff rows [%d, %d) of %lld experts x %d layers, %.2f GB in RAM (%.1f s)\n", cut, FF,
                 (long long) c.experts, c.layers, bytes / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    muse::ExpertLane lane(f, L, FF - cut, (float) c.post_eps, threads);
    ucomm_config cfg;
    ucomm_config_default(&cfg);
    cfg.rank = 1;
    cfg.world = 2;
    cfg.root_addr = root.c_str();
    ucomm_t* comm = nullptr;
    const ucomm_status s = ucomm_init(&cfg, &comm);
    if (s != UCOMM_OK) { std::fprintf(stderr, "ucomm: %s\n", ucomm_strerror(s)); return 1; }
    std::fprintf(stderr, "muse expert worker: connected to %s (%s)\n", root.c_str(), ucomm_backend_name(comm));
    const int rc = muse::serve_expert_share(comm, 0, lane, layers, L);
    ucomm_finalize(comm);
    std::fprintf(stderr, "muse expert worker: %s\n", rc ? "transport error" : "stopped");
    return rc;
}
