// src/artifact/muse_inspect.cpp - strata-muse-inspect: check a Muse Spark Open GGUF (every shard, or just the first
// megabyte of each: only headers are read) and print its geometry, footprint and the two-node expert row split.
//
//     strata-muse-inspect SHARD... [--b-ram-gib 56]
//
// Exit status 0 when the file is one the Muse engine can bind, 1 with the list of problems otherwise.
#include "strata/artifact/muse_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    double b_ram_gib = 56.0;   // the B550's 62 GiB less 6 for the OS and buffers (tools/sim/muse.py)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--b-ram-gib" && i + 1 < argc) b_ram_gib = std::atof(argv[++i]);
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "usage: strata-muse-inspect SHARD... [--b-ram-gib 56]\n");
            return 2;
        } else paths.push_back(a);
    }
    if (paths.empty()) {
        std::fprintf(stderr, "usage: strata-muse-inspect SHARD... [--b-ram-gib 56]\n");
        return 2;
    }
    try {
        const strata::GgufModel m(paths);
        strata::muse::MuseConfig c;
        const std::string err = strata::muse::read_config(m.meta(), c);
        if (!err.empty()) {
            std::printf("not a Muse Spark model the engine reads: %s\n", err.c_str());
            return 1;
        }
        std::printf("muse-spark: %d layers (%d full attention, %d sliding window %lld), hidden %lld, latent %lld,\n"
                    "  %lld experts of %lld x %lld (top-%d, sigmoid router%s), %lld/%lld heads of %lld, vocab %lld, "
                    "context %lld\n", c.layers, c.full_layers(), c.layers - c.full_layers(), (long long) c.window,
                    (long long) c.hidden, (long long) c.latent, (long long) c.experts, (long long) c.latent,
                    (long long) c.expert_ff, c.top_k, c.expert_weights_norm ? ", weights renormalized" : "",
                    (long long) c.q_heads, (long long) c.kv_heads, (long long) c.head_dim, (long long) c.vocab,
                    (long long) c.context);
        std::printf("  rope base %.0f (sliding-window layers only), rms eps %g, post-norm eps %g, logit scale %g, "
                    "softcap %g\n", c.rope_base_swa, c.rms_eps, c.post_eps, c.logit_scale, c.softcap);
        strata::muse::Footprint fp;
        const auto problems = strata::muse::check_tensors(m, c, fp);
        const double gb = 1e9;
        std::printf("footprint: %.2f GB = experts %.2f (%.2f MB each) + block trunk %.2f (%.1f MB per layer) + "
                    "output %.2f + token_embd %.2f\n", fp.total() / gb, fp.experts / gb, fp.expert_bytes / 1e6,
                    fp.fixed_layers / gb, fp.fixed_layer_max / 1e6, fp.lm_head / gb, fp.token_embd / gb);
        std::printf("  per decoded token: %.2f GB of trunk + output on the GPUs, %.2f GB of routed experts (%d x %d)\n",
                    (fp.fixed_layers + fp.lm_head) / gb, (double) fp.expert_bytes * c.top_k * c.layers / gb, c.top_k,
                    c.layers);
        const auto split = strata::muse::plan_row_split(c, fp, (uint64_t) (b_ram_gib * 1073741824.0));
        std::printf("two nodes, %.0f GiB for node B's expert rows: node B holds rows [%lld, %lld) of every expert "
                    "(share %.3f), %.1f GiB; node A keeps %.1f GiB of expert rows\n", b_ram_gib,
                    (long long) split.first_b, (long long) c.expert_ff, split.share_b, split.bytes_b / 1073741824.0,
                    split.bytes_a / 1073741824.0);
        for (const auto& p : problems) std::printf("PROBLEM: %s\n", p.c_str());
        std::printf("%s\n", problems.empty() ? "ok" : "not ok");
        return problems.empty() ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("strata-muse-inspect: %s\n", e.what());
        return 1;
    }
}
