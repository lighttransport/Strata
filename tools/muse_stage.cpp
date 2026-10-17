// tools/muse_stage.cpp - strata-muse-stage: Muse Spark Open on two nodes, both GPUs and both CPUs.
//
//     node 0: strata-muse-stage MODEL... --rank 0 --root HOST:PORT --split-layer S (--ids 1,2,3 | --dump DIR) [--tokens N]
//     node 1: strata-muse-stage MODEL... --rank 1 --root HOST:PORT --split-layer S
//     both:   [--share-b 0.375] [--threads T] [--graphs] [--splits N] [--context N]
//
// Node 0 runs the embedding and the trunk of layers [0, S) on its GPU; node 1 runs layers [S, layers) and the head on its
// GPU. Every routed expert is split by rows between the two CPUs (strata/muse/row_split.hpp: node 0 keeps ff rows
// [0, cut), node 1 the rest), so each MoE layer asks the other node for its partial outputs (strata/muse/split_lane.hpp).
// Per token node 0 drives its layers while node 1 serves expert requests, hands the hidden state over, then serves
// node 1's expert requests until node 1 sends back the next token. Both nodes alternate between driving and serving;
// nothing runs concurrently. HOST:PORT is an address of node 0 (ucomm's bootstrap root).
//
// Messages besides the expert ones (ucomm tags):
//   kTagHandoff (0 -> 1): int32 {op, pos} (op 1: run, 0: stop), then float x[hidden] for op 1
//   kTagResult  (1 -> 0): int32 {next token}, float {node 1's stage milliseconds}
#include "muse_check_common.hpp"
#include "strata/host/muse_names.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/muse/gpu_trunk.hpp"
#include "strata/muse/row_split.hpp"
#include "strata/muse/split_lane.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

using namespace muse_check;
namespace cpu = strata::kernels::cpu;
namespace sm = strata::muse;

namespace {
constexpr uint32_t kTagHandoff = 0x4d530010u, kTagResult = 0x4d530011u;

sm::HostTensor host(const Model& M, const std::string& name) {
    const strata::TensorInfo& t = M.t(name);
    sm::HostTensor h;
    h.data = M.data(name);
    h.type = (int) t.type;
    h.ne0 = (int64_t) t.shape[0];
    h.ne1 = 1;
    for (size_t k = 1; k < t.shape.size(); ++k) h.ne1 *= (int64_t) t.shape[k];
    h.bytes = (size_t) strata::tensor_payload_bytes(t);
    return h;
}
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void ok(ucomm_status s, const char* what) { if (s != UCOMM_OK) throw std::runtime_error(std::string(what) + ": " + ucomm_strerror(s)); }

// Serves the other node's expert requests until a message with `tag` arrives (into buf, cap bytes). The expert-request
// header receive stays posted across calls (the other node sends requests only while it drives).
struct Server {
    ucomm_t* c;
    int peer;
    sm::ExpertLane& lane;
    const std::vector<sm::LaneLayer>& layers;
    int latent;
    int32_t head[4] = {0, 0, 0, 0};
    ucomm_req_t* pending = nullptr;
    std::vector<uint8_t> scratch;
    double serve_ms = 0;
    void post() { ok(ucomm_irecv(c, peer, sm::kTagRequest, head, sizeof head, &pending), "post request"); }
    size_t until(uint32_t tag, void* buf, size_t cap) {
        ucomm_req_t* ctrl = nullptr;
        ok(ucomm_irecv(c, peer, tag, buf, cap, &ctrl), "post control");
        for (;;) {
            int done = 0;
            size_t n = 0;
            ok(ucomm_test(pending, &done, &n), "test request");
            if (done) {
                pending = nullptr;
                if (n != sizeof head || head[0] != 1) throw std::runtime_error("unexpected expert request");
                const double t0 = now_ms();
                if (sm::serve_expert_request(c, peer, lane, layers, latent, head, scratch)) throw std::runtime_error("serving an expert request");
                serve_ms += now_ms() - t0;
                post();
            }
            ok(ucomm_test(ctrl, &done, &n), "test control");
            if (done) return n;
        }
    }
    // the posted request header is completed by the other node's stop header
    void finish() { if (pending) ok(ucomm_wait(pending, nullptr), "final request"); }
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    std::string ids_arg, dump, root;
    double share_b = 0.375;
    int rank = -1, split_layer = -1, gen = 16, threads = 15, splits = 1, context = 4096;
    bool graphs = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--ids") ids_arg = next();
        else if (a == "--dump") dump = next();
        else if (a == "--root") root = next();
        else if (a == "--rank") rank = std::stoi(next());
        else if (a == "--split-layer") split_layer = std::stoi(next());
        else if (a == "--share-b") share_b = std::stod(next());
        else if (a == "--tokens") gen = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--splits") splits = std::stoi(next());
        else if (a == "--context") context = std::stoi(next());
        else if (a == "--graphs") graphs = true;
        else if (a.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        else shards.push_back(a);
    }
    if (shards.empty() || root.empty() || (rank != 0 && rank != 1) || split_layer < 1 || (rank == 0 && ids_arg.empty() == dump.empty())) {
        std::fprintf(stderr, "usage: strata-muse-stage MODEL... --rank 0|1 --root HOST:PORT --split-layer S [rank 0: --ids 1,2,3 | --dump DIR]\n");
        return 2;
    }
    const strata::GgufModel gm(shards);
    sm::MuseConfig c;
    if (const std::string e = sm::read_config(gm.meta(), c); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    sm::Footprint fp;
    if (const auto problems = sm::check_tensors(gm, c, fp); !problems.empty()) { std::fprintf(stderr, "%s\n", problems[0].c_str()); return 1; }
    if (split_layer >= c.layers) { std::fprintf(stderr, "--split-layer must leave node 1 at least one layer\n"); return 1; }
    const Model M{gm};
    const int H = (int) c.hidden, L = (int) c.latent, FF = (int) c.expert_ff, K = c.top_k;
    const int first = rank == 0 ? 0 : split_layer, last = rank == 0 ? split_layer : c.layers;   // this node's trunk layers

    // ---- this node's trunk on its GPU (node 1 also has the head)
    sm::TrunkConfig tc;
    tc.hidden = H; tc.latent = L; tc.experts = (int) c.experts; tc.top_k = K; tc.q_heads = (int) c.q_heads;
    tc.kv_heads = (int) c.kv_heads; tc.head_dim = (int) c.head_dim; tc.window = (int) c.window; tc.context = context;
    tc.rms_eps = (float) c.rms_eps; tc.post_eps = (float) c.post_eps; tc.rope_base = (float) c.rope_base_swa;
    tc.expert_weights_scale = (float) c.expert_weights_scale; tc.logit_scale = (float) c.logit_scale;
    tc.softcap = (float) c.softcap; tc.graphs = graphs; tc.attention_splits = splits;
    std::vector<sm::TrunkLayerHost> lh;
    for (int l = first; l < last; ++l) {
        auto n = [&](const char* s) { return host(M, "blk." + std::to_string(l) + "." + s); };
        sm::TrunkLayerHost x;
        x.swa = c.swa[(size_t) l] != 0;
        x.attn_norm = n("attn_norm.weight"); x.attn_q = n("attn_q.weight"); x.attn_k = n("attn_k.weight");
        x.attn_v = n("attn_v.weight"); x.attn_gate = n("attn_gate.weight"); x.attn_output = n("attn_output.weight");
        x.attn_q_norm = n("attn_q_norm.weight"); x.attn_k_norm = n("attn_k_norm.weight");
        x.attn_alpha = n("attn_resid_alpha.weight"); x.attn_beta = n("attn_resid_beta.weight");
        x.ffn_norm = n("ffn_norm.weight"); x.post_ffw_norm = n("post_ffw_norm.weight");
        x.ffn_alpha = n("ffn_resid_alpha.weight"); x.ffn_beta = n("ffn_resid_beta.weight");
        x.router = n("ffn_gate_inp.weight"); x.router_bias = n("exp_probs_b.bias");
        x.latent_down = n("ffn_latent_down.weight"); x.latent_up = n("ffn_latent_up.weight");
        x.pre_exp_norm = n("ffn_pre_exp_norm.weight");
        lh.push_back(x);
    }
    const auto out_norm = rank == 1 ? host(M, "output_norm.weight") : sm::HostTensor{};
    const auto output = rank == 1 ? host(M, "output.weight") : sm::HostTensor{};
    sm::GpuTrunk gpu(tc, lh, rank == 1 ? &out_norm : nullptr, rank == 1 ? &output : nullptr);

    // ---- this node's row share of every layer's experts
    cpu::NativeFmt nf;
    std::string ferr;
    const int gt = (int) M.t("blk.0.ffn_gate_exps.weight").type, dt = (int) M.t("blk.0.ffn_down_exps.weight").type;
    const int cut = (int) strata_muse_row_split(FF, 256, share_b);
    if (cut <= 0 || cut >= FF) { std::fprintf(stderr, "share %.3f leaves a node no rows\n", share_b); return 1; }
    const int f0 = rank == 0 ? 0 : cut, f1 = rank == 0 ? cut : FF;
    if (!cpu::native_fmt(gt, dt, L, f1 - f0, nf, ferr)) { std::fprintf(stderr, "%s\n", ferr.c_str()); return 1; }
    nf.iq1s_rows = gt == GGML_TYPE_IQ1_S;
    std::vector<sm::ExpertShare> share;
    std::vector<std::vector<float>> post((size_t) c.layers);
    std::vector<sm::LaneLayer> layers;
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        post[(size_t) l] = M.all(p + "ffn_post_exp_norm.weight");
        share.push_back(sm::slice_experts(M.data(p + "ffn_gate_exps.weight"), M.data(p + "ffn_up_exps.weight"), M.data(p + "ffn_down_exps.weight"),
                                          (int) c.experts, L, FF, gt, dt, f0, f1));
    }
    for (int l = 0; l < c.layers; ++l)
        layers.push_back({share[(size_t) l].gate.data(), share[(size_t) l].up.data(), share[(size_t) l].down.data(), post[(size_t) l].data()});
    sm::ExpertLane lane(nf, L, f1 - f0, (float) c.post_eps, threads);
    std::fprintf(stderr, "muse stage %d: trunk layers [%d, %d)%s on the GPU (%.2f GB), expert rows [%d, %d) of every layer on %d CPU threads\n",
                 rank, first, last, rank == 1 ? " + head" : "", gpu.device_bytes() / 1e9, f0, f1, threads);

    ucomm_config cfg;
    ucomm_config_default(&cfg);
    cfg.rank = rank;
    cfg.world = 2;
    cfg.root_addr = root.c_str();
    ucomm_t* comm = nullptr;
    ok(ucomm_init(&cfg, &comm), "ucomm init");
    const int peer = 1 - rank;
    sm::SplitLane split(lane, comm, peer, L, (float) c.post_eps);
    Server server{comm, peer, lane, layers, L};
    server.post();
    std::vector<float> x, y((size_t) L), logits;
    // runs this node's trunk layers with the experts split across both nodes; x: the stream in and out (host)
    auto run_layers = [&](int pos) {
        gpu.begin(x.data(), pos);
        for (int l = first; l < last; ++l) {
            gpu.attention_and_route(l - first);
            split.run(l, layers[(size_t) l], gpu.z().data(), 1, gpu.ids().data(), gpu.weights().data(), K, y.data());
            gpu.finish_moe(l - first, y.data());
        }
    };
    if (rank == 1) {
        int32_t hand[2];
        x.resize((size_t) H);
        for (;;) {
            server.until(kTagHandoff, hand, sizeof hand);
            if (hand[0] == 0) break;
            size_t n = 0;
            ok(ucomm_recv(comm, peer, kTagHandoff, x.data(), x.size() * 4, &n), "receive hidden");
            const double t0 = now_ms();
            run_layers(hand[1]);
            gpu.head(logits);
            struct { int32_t token; float ms; } result{(int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin()), (float) (now_ms() - t0)};
            ok(ucomm_send(comm, peer, kTagResult, &result, sizeof result), "send result");
        }
    } else {
        std::vector<int> prompt;
        if (!dump.empty()) { std::ifstream tf(dump + "/tokens.txt"); for (int t; tf >> t;) prompt.push_back(t); }
        else { std::stringstream ss(ids_arg); for (std::string t; std::getline(ss, t, ',');) prompt.push_back(std::stoi(t)); }
        const int P = (int) prompt.size();
        const ref::Dims d = dims(c);
        const Dump llama = dump.empty() ? Dump{} : load_dump(dump + "/result_output.bin");
        const int64_t V = (int64_t) M.t("output.weight").shape[1];
        std::vector<int> out;
        int agree = 0, token = prompt.at(0);
        double gen_ms = 0, stage1_ms = 0, own_ms = 0;
        for (int pos = 0; pos < P + gen - 1; ++pos) {
            token = pos < P ? prompt[(size_t) pos] : token;
            const double t0 = now_ms();
            M.rows("token_embd.weight", token, 1, x);
            ref::rms_norm(x.data(), H, d.rms_eps);
            run_layers(pos);
            gpu.trace(sm::TrunkTrace::Hidden, x);
            const double t1 = now_ms();
            const int32_t hand[2] = {1, pos};
            ok(ucomm_send(comm, peer, kTagHandoff, hand, sizeof hand), "send handoff");
            ok(ucomm_send(comm, peer, kTagHandoff, x.data(), x.size() * 4), "send hidden");
            struct { int32_t token; float ms; } result{};
            server.until(kTagResult, &result, sizeof result);
            const int next = result.token;
            if (pos < P && llama.ok() && llama.ne[0] == V)
                agree += next == (int) (std::max_element(llama.f.data() + (size_t) pos * V, llama.f.data() + (size_t) (pos + 1) * V) -
                                        (llama.f.data() + (size_t) pos * V));
            if (pos >= P - 1) {
                out.push_back(next);
                token = next;
                if (pos >= P) { gen_ms += now_ms() - t0; stage1_ms += result.ms; own_ms += t1 - t0; }
            }
        }
        const int32_t stop[2] = {0, 0};
        ok(ucomm_send(comm, peer, kTagHandoff, stop, sizeof stop), "send stop");
        for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? " " : "", out[i]);
        std::printf("\n");
        const int timed = (int) out.size() - 1;
        if (timed > 0)
            std::fprintf(stderr, "two nodes (%s): %d tokens, %.2f tok/s; per token %.1f ms = node 0 stage %.1f + node 1 stage %.1f + transfers\n",
                         ucomm_backend_name(comm), timed, 1e3 * timed / gen_ms, gen_ms / timed, own_ms / timed, stage1_ms / timed);
        if (llama.ok()) std::fprintf(stderr, "prompt argmax agrees with llama.cpp at %d of %d positions\n", agree, P);
    }
    // both nodes stop the other's posted request receive, then drain it
    const int32_t stop_head[4] = {0, 0, 0, 0};
    ok(ucomm_send(comm, peer, sm::kTagRequest, stop_head, sizeof stop_head), "send request stop");
    server.finish();
    ucomm_finalize(comm);
    return 0;
}
