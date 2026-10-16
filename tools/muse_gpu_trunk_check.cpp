// tools/muse_gpu_trunk_check.cpp - the GPU block trunk (strata/muse/gpu_trunk.hpp) with the CPU IQ1_S expert lane,
// against strata/muse/reference.hpp, on a few-layer slice of the release (tools/muse_layer_slice.py).
//
//     muse_gpu_trunk_check MODEL.gguf DUMPDIR [--f32] [--graphs] [--splits N] [--bench N]
//
// --graphs replays each layer as captured graphs (TrunkConfig::graphs); the check is the same.  --bench N then decodes
// N more tokens (the last token repeated) with no reference and no traces, and prints the GPU calls' wall time per
// token apart from the CPU expert lane's.
// DUMPDIR/tokens.txt gives the tokens (tools/muse_llama_dump.cpp writes it; its result_output.bin, when present, is
// compared too).  The reference rounds its activations as this path does unless --f32: the GPU projections' inputs to
// Q8_1, the experts' to Q8_K (the CPU lane: native_quant_act / native_quant_h), K and V to f16; the router (F32
// weights) and the attention's query and softmax weights stay f32 on both.  What is left is summation order and the
// GPU's expf / rsqrtf.
#include "muse_check_common.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/muse/gpu_trunk.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace muse_check;
namespace cpu = strata::kernels::cpu;

namespace {
strata::muse::HostTensor host(const Model& M, const std::string& name) {
    const strata::TensorInfo& t = M.t(name);
    strata::muse::HostTensor h;
    h.data = M.data(name);
    h.type = (int) t.type;
    h.ne0 = (int64_t) t.shape[0];
    h.ne1 = 1;
    for (size_t k = 1; k < t.shape.size(); ++k) h.ne1 *= (int64_t) t.shape[k];
    h.bytes = (size_t) strata::tensor_payload_bytes(t);
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: muse_gpu_trunk_check MODEL.gguf DUMPDIR [--f32] [--graphs] [--splits N] [--bench N]\n");
        return 2;
    }
    const std::string dir = argv[2];
    bool f32 = false, graphs = false;
    int bench = 0, splits = 1;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--f32") f32 = true;
        else if (a == "--graphs") graphs = true;
        else if (a == "--bench" && i + 1 < argc) bench = std::atoi(argv[++i]);
        else if (a == "--splits" && i + 1 < argc) splits = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    const strata::GgufModel gm(std::vector<std::string>{argv[1]});
    strata::muse::MuseConfig c;
    const std::string err = strata::muse::read_config(gm.meta(), c);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    strata::muse::Footprint fp;
    const auto problems = strata::muse::check_tensors(gm, c, fp);
    if (!problems.empty()) { std::fprintf(stderr, "%s\n", problems[0].c_str()); return 1; }
    const Model M{gm};
    std::vector<int> tokens;
    {
        std::ifstream tf(dir + "/tokens.txt");
        for (int t; tf >> t;) tokens.push_back(t);
    }
    const int T = (int) tokens.size(), H = (int) c.hidden, L = (int) c.latent, FF = (int) c.expert_ff;

    // ---- the GPU trunk
    strata::muse::TrunkConfig tc;
    tc.hidden = H; tc.latent = L; tc.experts = (int) c.experts; tc.top_k = c.top_k; tc.q_heads = (int) c.q_heads;
    tc.kv_heads = (int) c.kv_heads; tc.head_dim = (int) c.head_dim; tc.window = (int) c.window; tc.context = std::max(T + bench, 256);
    tc.rms_eps = (float) c.rms_eps; tc.post_eps = (float) c.post_eps; tc.rope_base = (float) c.rope_base_swa;
    tc.expert_weights_scale = (float) c.expert_weights_scale; tc.logit_scale = (float) c.logit_scale;
    tc.softcap = (float) c.softcap;
    tc.graphs = graphs;
    tc.attention_splits = splits;
    std::vector<strata::muse::TrunkLayerHost> lh((size_t) c.layers);
    for (int l = 0; l < c.layers; ++l) {
        auto n = [&](const char* s) { return host(M, "blk." + std::to_string(l) + "." + s); };
        auto& x = lh[(size_t) l];
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
    }
    const auto out_norm = host(M, "output_norm.weight"), output = host(M, "output.weight");
    strata::muse::GpuTrunk gpu(tc, lh, &out_norm, &output);
    std::fprintf(stderr, "GPU trunk: %.2f GB on the device\n", gpu.device_bytes() / 1e9);

    // ---- the CPU expert lane: Strata's IQ1_S rows (NativeFmt::iq1s_rows) on the GGUF's own expert tensors
    cpu::NativeFmt nf;
    std::string ferr;
    const int etype = (int) M.t("blk.0.ffn_gate_exps.weight").type;
    if (!cpu::native_fmt(etype, (int) M.t("blk.0.ffn_down_exps.weight").type, L, FF, nf, ferr)) {
        std::fprintf(stderr, "%s\n", ferr.c_str());
        return 1;
    }
    nf.iq1s_rows = true;
    std::vector<std::vector<float>> post_exp((size_t) c.layers);
    for (int l = 0; l < c.layers; ++l) post_exp[(size_t) l] = M.all("blk." + std::to_string(l) + ".ffn_post_exp_norm.weight");
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
    std::vector<float> ffv((size_t) FF), outv((size_t) L);
    auto expert_lane = [&](int l, const std::vector<float>& z, const std::vector<int32_t>& ids, const std::vector<float>& w) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const uint8_t *gate = M.data(p + "ffn_gate_exps.weight"), *up = M.data(p + "ffn_up_exps.weight"),
                      *down = M.data(p + "ffn_down_exps.weight");
        std::vector<float> y((size_t) L, 0.f);
        cpu::native_quant_act(nf, z.data(), act.data());
        const void* ap = act.data();
        const void* hp = hq.data();
        float* ffp = ffv.data();
        float* op = outv.data();
        for (int k = 0; k < c.top_k; ++k) {
            const size_t e = (size_t) ids[(size_t) k];
            const uint8_t* g = gate + e * nf.gu_row * (size_t) FF;
            cpu::native_gu_rows(nf, g, &ap, 1, &ffp, 0, FF, up + e * nf.gu_row * (size_t) FF);
            cpu::native_quant_h(nf, ffv.data(), hq.data());
            cpu::native_down_rows(nf, g, &hp, 1, &op, 0, L, down + e * nf.d_row * (size_t) L);
            for (int j = 0; j < L; ++j) outv[(size_t) j] *= post_exp[(size_t) l][(size_t) j];
            ref::rms_norm(outv.data(), L, (float) c.post_eps);
            for (int j = 0; j < L; ++j) y[(size_t) j] += w[(size_t) k] * outv[(size_t) j];
        }
        return y;
    };

    // ---- the reference, rounded as this path rounds
    ref::Dims d = dims(c);
    std::map<std::string, int> site;
    for (const char* s : {"attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "ffn_latent_down", "ffn_latent_up"})
        site[s] = kQ8_1;
    site["ffn_gate_inp"] = GGML_TYPE_F32;
    for (const char* s : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) site[s] = GGML_TYPE_Q8_K;
    site["cache_k"] = site["cache_v"] = GGML_TYPE_F16;
    site["kq_q"] = site["kq"] = GGML_TYPE_F32;
    if (!f32) d.round = [&site](const char* s, float* x, int n) { round_type(site.at(s), x, n); };
    const auto loaded = load_blocks(M, c);
    std::vector<ref::KvCache> kv((size_t) c.layers);

    // ---- run both, token by token
    std::vector<Stat> s_attn((size_t) c.layers), s_inp((size_t) c.layers), s_moe((size_t) c.layers),
        s_post((size_t) c.layers), s_out((size_t) c.layers);
    std::vector<int> same_routes((size_t) c.layers, 0);
    Stat s_logits;
    int agree = 0, agree_llama = 0;
    const Dump llama = load_dump(dir + "/result_output.bin");
    std::vector<float> trace, logits_gpu;
    const int64_t V = (int64_t) output.ne1;
    for (int p = 0; p < T; ++p) {
        std::vector<float> x;
        M.rows("token_embd.weight", tokens[(size_t) p], 1, x);
        ref::rms_norm(x.data(), H, d.rms_eps);
        gpu.begin(x.data(), p);
        for (int l = 0; l < c.layers; ++l) {
            ref::Routes r;
            ref::block_step(d, loaded->blocks[(size_t) l], kv[(size_t) l], p, x, &r);
            gpu.attention_and_route(l);
            gpu.trace(strata::muse::TrunkTrace::AttnPostNorm, trace);
            s_attn[(size_t) l].add(trace.data(), r.attn_post_norm.data(), (size_t) H);
            gpu.trace(strata::muse::TrunkTrace::FfnInp, trace);
            s_inp[(size_t) l].add(trace.data(), r.ffn_inp.data(), (size_t) H);
            for (int k = 0; k < c.top_k; ++k)
                same_routes[(size_t) l] += std::find(r.experts.begin(), r.experts.end(), gpu.ids()[(size_t) k]) != r.experts.end();
            const std::vector<float> y = expert_lane(l, gpu.z(), gpu.ids(), gpu.weights());
            s_moe[(size_t) l].add(y.data(), r.moe_out.data(), (size_t) L);
            gpu.finish_moe(l, y.data());
            gpu.trace(strata::muse::TrunkTrace::FfnPostNorm, trace);
            s_post[(size_t) l].add(trace.data(), r.ffn_post_norm.data(), (size_t) H);
            gpu.trace(strata::muse::TrunkTrace::Hidden, trace);
            s_out[(size_t) l].add(trace.data(), x.data(), (size_t) H);
        }
        gpu.head(logits_gpu);
        // the reference head on this token (rounded like the GPU's Q8_1 input unless --f32)
        std::vector<float> hn = x, chunk, logits_ref((size_t) V);
        const std::vector<float> on = M.all("output_norm.weight");
        ref::rms_norm(hn.data(), H, d.rms_eps, on.data());
        if (!f32) round_q8_1(hn.data(), H);
        for (int64_t r0 = 0; r0 < V; r0 += 4096) {
            const int64_t rows = std::min<int64_t>(4096, V - r0);
            M.rows("output.weight", r0, rows, chunk);
            for (int64_t r = 0; r < rows; ++r) {
                double acc = 0;
                for (int i = 0; i < H; ++i) acc += (double) chunk[(size_t) (r * H + i)] * hn[(size_t) i];
                const float lg = (float) acc * (float) c.logit_scale;
                logits_ref[(size_t) (r0 + r)] = c.softcap > 0 ? (float) (c.softcap * std::tanh(lg / c.softcap)) : lg;
            }
        }
        s_logits.add(logits_gpu.data(), logits_ref.data(), (size_t) V);
        const auto am = [&](const float* v) { return std::max_element(v, v + V) - v; };
        agree += am(logits_gpu.data()) == am(logits_ref.data());
        if (llama.ok() && llama.ne[0] == V) agree_llama += am(logits_gpu.data()) == am(llama.f.data() + (size_t) p * V);
        std::fprintf(stderr, "\rtoken %d/%d", p + 1, T);
    }
    std::fprintf(stderr, "\n");

    int failures = 0;
    const double tol = f32 ? 0.08 : 0.01;
    auto row = [&](const std::string& name, const Stat& s) {
        const bool ok = s.max_rel <= tol;
        std::printf("  %-22s rel %.2e  cos %.6f  %s\n", name.c_str(), s.max_rel, s.min_cos, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    std::printf("%d tokens, %d layers: GPU trunk + CPU IQ1_S experts vs the reference%s\n", T, c.layers,
                f32 ? " in f32" : " rounded as the GPU path");
    for (int l = 0; l < c.layers; ++l) {
        const std::string s = "-" + std::to_string(l);
        row("attn_post_norm" + s, s_attn[(size_t) l]);
        row("ffn_inp" + s, s_inp[(size_t) l]);
        const bool routes_ok = same_routes[(size_t) l] >= T * c.top_k - T;
        std::printf("  %-22s %d of %d routes the same  %s\n", ("routes" + s).c_str(), same_routes[(size_t) l], T * c.top_k,
                    routes_ok ? "ok" : "FAIL");
        failures += !routes_ok;
        row("ffn_moe_out" + s, s_moe[(size_t) l]);
        row("ffn_post_norm" + s, s_post[(size_t) l]);
        row("l_out" + s, s_out[(size_t) l]);
    }
    row("logits", s_logits);
    std::printf("  %-22s argmax agrees at %d of %d positions  %s\n", "logits", agree, T, agree == T ? "ok" : "FAIL");
    failures += agree != T;
    if (llama.ok()) std::printf("  %-22s argmax agrees with llama.cpp at %d of %d positions\n", "logits", agree_llama, T);
    if (bench > 0) {
        using clk = std::chrono::steady_clock;
        double gpu_s = 0, lane_s = 0;
        std::vector<float> x;
        for (int i = 0; i < bench; ++i) {
            M.rows("token_embd.weight", tokens.back(), 1, x);
            ref::rms_norm(x.data(), H, d.rms_eps);
            auto t0 = clk::now();
            gpu.begin(x.data(), T + i);
            for (int l = 0; l < c.layers; ++l) {
                gpu.attention_and_route(l);
                const auto t1 = clk::now();
                gpu_s += std::chrono::duration<double>(t1 - t0).count();
                const std::vector<float> y = expert_lane(l, gpu.z(), gpu.ids(), gpu.weights());
                t0 = clk::now();
                lane_s += std::chrono::duration<double>(t0 - t1).count();
                gpu.finish_moe(l, y.data());
            }
            gpu.head(logits_gpu);
            gpu_s += std::chrono::duration<double>(clk::now() - t0).count();
        }
        std::printf("bench %d tokens%s: GPU trunk %.3f ms per token (%d layers + head), CPU expert lane %.3f ms\n", bench,
                    graphs ? " (graphs)" : "", 1e3 * gpu_s / bench, c.layers, 1e3 * lane_s / bench);
    }
    std::printf("muse_gpu_trunk_check: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
