// tests/core/muse_row_split_test.cpp - Muse Spark's two-node expert row split (strata/muse/row_split.hpp,
// strata/muse/split_lane.hpp) on synthetic IQ1_S experts of the release's shape (latent 4096, ff 4096):
//   A. in one process: the two shares' partials added and combined, against one node's full rows (summation order
//      only, so within float rounding), for the plan's cut (B550 share 0.375) and an uneven one;
//   B. two processes over ucomm on 127.0.0.1: the primary's SplitLane with a worker serving the other share gives
//      A's result bit for bit, for 1 and 3 tokens over several layers, then a clean stop.
// Env as for ucomm_config_default (UCOMM_BACKEND=udp etc.).
#include "strata/host/muse_names.h"
#include "strata/muse/row_split.hpp"
#include "strata/muse/split_lane.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace muse = strata::muse;

namespace {
constexpr int L = 4096, FF = 4096, E = 12, K = 8, LAYERS = 3;
constexpr float EPS = 1e-8f;

struct Synthetic {
    std::vector<std::vector<uint8_t>> gate, up, down;
    std::vector<std::vector<float>> post;
};
Synthetic make(const cpu::NativeFmt& f) {
    std::mt19937 rng(20261017);
    auto blob = [&](size_t bytes) {
        std::vector<uint8_t> b(bytes);
        for (auto& v : b) v = (uint8_t) rng();
        for (size_t o = 0; o < b.size(); o += 50) {   // IQ1_S blocks: small finite fp16 scales
            const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f + 0.02f * (rng() % 1000) / 1000.f);
            std::memcpy(&b[o], &d, 2);
        }
        return b;
    };
    Synthetic s;
    std::normal_distribution<float> nd;
    for (int l = 0; l < LAYERS; ++l) {
        s.gate.push_back(blob((size_t) E * FF * f.gu_row));
        s.up.push_back(blob((size_t) E * FF * f.gu_row));
        s.down.push_back(blob((size_t) E * L * f.d_row));
        std::vector<float> p(L);
        for (auto& v : p) v = 1.f + 0.1f * nd(rng);
        s.post.push_back(std::move(p));
    }
    return s;
}
cpu::NativeFmt fmt(int ff) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, L, ff, f, err)) { std::printf("%s\n", err.c_str()); std::exit(1); }
    f.iq1s_rows = true;
    return f;
}
struct Shares {
    std::vector<muse::ExpertShare> a, b;
    std::vector<muse::LaneLayer> la, lb;
};
Shares shares(const Synthetic& s, int cut) {
    Shares r;
    for (int l = 0; l < LAYERS; ++l) {
        r.a.push_back(muse::slice_experts(s.gate[l].data(), s.up[l].data(), s.down[l].data(), E, L, FF, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, 0, cut));
        r.b.push_back(muse::slice_experts(s.gate[l].data(), s.up[l].data(), s.down[l].data(), E, L, FF, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, cut, FF));
    }
    for (int l = 0; l < LAYERS; ++l) {
        r.la.push_back({r.a[l].gate.data(), r.a[l].up.data(), r.a[l].down.data(), s.post[l].data()});
        r.lb.push_back({r.b[l].gate.data(), r.b[l].up.data(), r.b[l].down.data(), s.post[l].data()});
    }
    return r;
}
struct Request { int layer, nt; std::vector<float> z, w; std::vector<int32_t> ids; };
std::vector<Request> requests() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<Request> out;
    for (int l = 0; l < LAYERS; ++l)
        for (int nt : {1, 3}) {
            Request q{l, nt, std::vector<float>((size_t) nt * L), std::vector<float>((size_t) nt * K), std::vector<int32_t>((size_t) nt * K)};
            for (auto& v : q.z) v = nd(rng);
            for (int t = 0; t < nt; ++t) {
                float sum = 0;
                for (int k = 0; k < K; ++k) {
                    q.ids[t * K + k] = (t * 5 + k * 3 + l) % E;
                    sum += q.w[t * K + k] = 0.1f + (float) (rng() % 100) / 100.f;
                }
                for (int k = 0; k < K; ++k) q.w[t * K + k] /= sum;
            }
            out.push_back(std::move(q));
        }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    const cpu::NativeFmt full = fmt(FF);
    const Synthetic s = make(full);
    const auto reqs = requests();
    int fails = 0;
    // A. in-process split against the full rows
    const int cut = (int) strata_muse_row_split(FF, 256, 0.375);   // the plan's B550 share
    std::vector<std::vector<float>> split_out;
    for (int c : {cut, 1024}) {
        const Shares sh = shares(s, c);
        muse::ExpertLane whole(full, L, FF, EPS, 4), a(fmt(c), L, c, EPS, 4), b(fmt(FF - c), L, FF - c, EPS, 4);
        double worst = 0;
        for (const Request& q : reqs) {
            std::vector<float> want((size_t) q.nt * L), got((size_t) q.nt * L);
            const muse::LaneLayer lw{s.gate[q.layer].data(), s.up[q.layer].data(), s.down[q.layer].data(), s.post[q.layer].data()};
            whole.run(lw, q.z.data(), q.nt, q.ids.data(), q.w.data(), K, want.data());
            a.run_raw(sh.la[q.layer], q.z.data(), q.nt, q.ids.data(), K);
            b.run_raw(sh.lb[q.layer], q.z.data(), q.nt, q.ids.data(), K);
            float* raw = a.raw();
            for (size_t i = 0; i < (size_t) q.nt * K * L; ++i) raw[i] += b.raw()[i];
            muse::ExpertLane::combine(raw, s.post[q.layer].data(), EPS, L, q.nt, q.w.data(), K, got.data());
            double err = 0, scale = 0;
            for (size_t i = 0; i < got.size(); ++i) { err = std::max(err, (double) std::fabs(got[i] - want[i])); scale = std::max(scale, (double) std::fabs(want[i])); }
            worst = std::max(worst, err / scale);
            if (c == cut) split_out.push_back(got);
        }
        const bool ok = worst < 1e-5;
        std::printf("A. cut at ff row %d (node 1: %.3f of each expert): split vs one node, worst relative %.2e  %s\n", c,
                    (FF - c) / (double) FF, worst, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    // B. the same split over ucomm, the other share in a second process
    const std::string root = std::string("127.0.0.1:") + (argc > 1 ? argv[1] : "29671");
    const pid_t pid = fork();
    if (pid == 0) {
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 1; cfg.world = 2; cfg.root_addr = root.c_str();
        ucomm_t* c = nullptr;
        if (ucomm_init(&cfg, &c) != UCOMM_OK) _exit(3);
        const Shares sh = shares(s, cut);
        muse::ExpertLane b(fmt(FF - cut), L, FF - cut, EPS, 4);
        const int rc = muse::serve_expert_share(c, 0, b, sh.lb, L);
        ucomm_finalize(c);
        _exit(rc);
    }
    ucomm_config cfg;
    ucomm_config_default(&cfg);
    cfg.rank = 0; cfg.world = 2; cfg.root_addr = root.c_str();
    ucomm_t* c = nullptr;
    if (ucomm_init(&cfg, &c) != UCOMM_OK) { std::printf("B. ucomm init failed\n"); return 1; }
    {
        const Shares sh = shares(s, cut);
        muse::ExpertLane a(fmt(cut), L, cut, EPS, 4);
        muse::SplitLane lane(a, c, 1, L, EPS);
        int same = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            const Request& q = reqs[i];
            std::vector<float> got((size_t) q.nt * L);
            lane.run(q.layer, sh.la[q.layer], q.z.data(), q.nt, q.ids.data(), q.w.data(), K, got.data());
            same += !std::memcmp(got.data(), split_out[i].data(), got.size() * 4);
        }
        lane.stop();
        const bool ok = same == (int) reqs.size();
        std::printf("B. over ucomm (%s): %d of %zu layer calls bit-identical to the in-process split  %s\n", ucomm_backend_name(c), same,
                    reqs.size(), ok ? "ok" : "FAIL");
        fails += !ok;
    }
    ucomm_finalize(c);
    int status = 0;
    waitpid(pid, &status, 0);
    const bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    std::printf("B. worker %s\n", clean ? "stopped cleanly" : "FAILED");
    fails += !clean;
    std::printf("muse_row_split_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
