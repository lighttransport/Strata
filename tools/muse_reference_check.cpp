// tools/muse_reference_check.cpp - strata/muse/reference.hpp against llama.cpp on real Muse Spark weights.
//
// Takes a few-layer slice of the release (tools/muse_layer_slice.py) and the tensors llama.cpp computed for a prompt
// on it (tools/muse_llama_dump.cpp), runs the same tokens through the f32 reference one at a time with the weights
// dequantized by ggml's own to_float, and compares every dumped stage per token: relative error, cosine, the experts
// the router chose, and the logits' argmax.  llama.cpp's CPU matmuls quantize their activations (Q8_K / Q8_0), the
// reference does not, so agreement is expected to ~1e-2 relative, not bit for bit.
//
//     muse_reference_check MODEL.gguf DUMPDIR [--like-llama]
//
// --like-llama: round the reference's activations where ggml-cpu does (Dims::round): each quantized matmul's input to
// the weight type's vec_dot_type (Q8_K for Q6_K / Q4_K / IQ1_S, Q8_0 for Q8_0) and back, the stored K and V and the
// attention matmuls' inputs to f16.  ggml's quantized dot equals the dot of the dequantized weight with that rounded
// input, so what is left is float summation order: a semantic difference would stand out against it.
#include "strata/artifact/muse_model.hpp"
#include "strata/muse/reference.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ref = strata::muse::ref;

namespace {

// ggml's quantize_row_q8_K_ref then dequantization (d * q), in place: d = max / -127 from the signed largest value
void round_q8k(float* x, int n) {
    for (int b = 0; b + 256 <= n; b += 256) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[b + j]);
            if (ax > amax) { amax = ax; max = x[b + j]; }
        }
        if (!amax) { for (int j = 0; j < 256; ++j) x[b + j] = 0; continue; }
        const float iscale = -127.f / max, d = 1 / iscale;
        for (int j = 0; j < 256; ++j) x[b + j] = d * (float) std::min(127, (int) std::nearbyint(iscale * x[b + j]));
    }
}

void round_type(int vec_dot_type, float* x, int n) {
    if (vec_dot_type == GGML_TYPE_Q8_K) return round_q8k(x, n);
    const auto* tr = ggml_get_type_traits((ggml_type) vec_dot_type);
    std::vector<uint8_t> q(ggml_row_size((ggml_type) vec_dot_type, n));
    tr->from_float_ref(x, q.data(), n);
    tr->to_float(q.data(), x, n);
}

// the activation type ggml-cpu quantizes to for a weight type (ggml-cpu.c type_traits_cpu[].vec_dot_type)
int vec_dot_type(int weight_type) {
    switch (weight_type) {
        case GGML_TYPE_F32: return GGML_TYPE_F32;
        case GGML_TYPE_F16: return GGML_TYPE_F16;
        case GGML_TYPE_Q8_0: return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_Q8_K;   // the K-quants and i-quants of this model
    }
}

struct Dump {
    int32_t type = -1;
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<float> f;
    std::vector<int32_t> i;
    bool ok() const { return type >= 0; }
};

Dump load_dump(const std::string& path) {
    Dump d;
    std::ifstream in(path, std::ios::binary);
    if (!in) return d;
    in.read((char*) &d.type, 4);
    in.read((char*) d.ne, 32);
    const size_t n = (size_t) (d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3]);
    if (d.type == 0) { d.f.resize(n); in.read((char*) d.f.data(), (std::streamsize) (n * 4)); }
    else { d.i.resize(n); in.read((char*) d.i.data(), (std::streamsize) (n * 4)); }
    return d;
}

struct Model {
    const strata::GgufModel& m;
    const strata::TensorInfo& t(const std::string& name) const {
        const strata::TensorInfo* p = m.find(name);
        if (!p) throw std::runtime_error("missing " + name);
        return *p;
    }
    /// rows [r0, r0 + rows) of a tensor whose rows are its dimension 0, as f32
    void rows(const std::string& name, int64_t r0, int64_t rows, std::vector<float>& out) const {
        size_t shard = 0;
        const strata::TensorInfo* p = m.find(name, &shard);
        if (!p) throw std::runtime_error("missing " + name);
        const int64_t ne0 = (int64_t) p->shape[0];
        const size_t row_bytes = ggml_row_size((ggml_type) p->type, ne0);
        out.resize((size_t) (rows * ne0));
        const auto* tr = ggml_get_type_traits((ggml_type) p->type);
        const uint8_t* base = m.shard(shard).tensor_data(*p) + (size_t) r0 * row_bytes;
        if (p->type == GGML_TYPE_F32) std::memcpy(out.data(), base, out.size() * 4);
        else tr->to_float(base, out.data(), rows * ne0);
    }
    std::vector<float> all(const std::string& name) const {
        const strata::TensorInfo& p = t(name);
        int64_t rows = 1;
        for (size_t k = 1; k < p.shape.size(); ++k) rows *= (int64_t) p.shape[k];
        std::vector<float> v;
        this->rows(name, 0, rows, v);
        return v;
    }
};

struct Stat {
    double max_rel = 0, min_cos = 1;
    void add(const float* a, const float* b, size_t n) {
        double num = 0, den = 0, ab = 0, aa = 0;
        for (size_t i = 0; i < n; ++i) {
            num += (double) (a[i] - b[i]) * (a[i] - b[i]);
            den += (double) b[i] * b[i];
            ab += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
        }
        max_rel = std::max(max_rel, std::sqrt(num / std::max(den, 1e-30)));
        min_cos = std::min(min_cos, ab / std::sqrt(std::max(aa * den, 1e-30)));
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: muse_reference_check MODEL.gguf DUMPDIR\n");
        return 2;
    }
    const std::string dir = argv[2];
    const strata::GgufModel gm(std::vector<std::string>{argv[1]});
    strata::muse::MuseConfig c;
    const std::string err = strata::muse::read_config(gm.meta(), c);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    strata::muse::Footprint fp;
    const auto problems = strata::muse::check_tensors(gm, c, fp);
    if (!problems.empty()) { std::fprintf(stderr, "%s\n", problems[0].c_str()); return 1; }
    const Model M{gm};

    ref::Dims d;
    d.hidden = (int) c.hidden; d.latent = (int) c.latent; d.expert_ff = (int) c.expert_ff; d.experts = (int) c.experts;
    d.top_k = c.top_k; d.q_heads = (int) c.q_heads; d.kv_heads = (int) c.kv_heads; d.head_dim = (int) c.head_dim;
    d.window = (int) c.window; d.rms_eps = (float) c.rms_eps; d.post_eps = (float) c.post_eps;
    d.rope_base = (float) c.rope_base_swa; d.expert_weights_scale = (float) c.expert_weights_scale;
    d.expert_weights_norm = c.expert_weights_norm;
    const bool like_llama = argc > 3 && std::string(argv[3]) == "--like-llama";
    std::map<std::string, int> site_type;
    for (const char* site : {"attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "ffn_latent_down", "ffn_gate_inp",
                             "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_latent_up"})
        site_type[site] = vec_dot_type((int) M.t(std::string("blk.0.") + site + ".weight").type);
    for (const char* site : {"cache_k", "cache_v", "kq_q", "kq"}) site_type[site] = GGML_TYPE_F16;
    if (like_llama)
        d.round = [&site_type](const char* site, float* x, int n) {
            const int t = site_type.at(site);
            if (t != GGML_TYPE_F32) round_type(t, x, n);
        };

    std::vector<ref::Block> blocks((size_t) c.layers);
    std::vector<std::map<int, std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>>> cache((size_t) c.layers);
    for (int l = 0; l < c.layers; ++l) {
        ref::Block& b = blocks[(size_t) l];
        auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
        b.swa = c.swa[(size_t) l] != 0;
        b.attn_norm = M.all(n("attn_norm.weight")); b.wq = M.all(n("attn_q.weight")); b.wk = M.all(n("attn_k.weight"));
        b.wv = M.all(n("attn_v.weight")); b.wgate = M.all(n("attn_gate.weight")); b.wo = M.all(n("attn_output.weight"));
        b.q_norm = M.all(n("attn_q_norm.weight")); b.k_norm = M.all(n("attn_k_norm.weight"));
        b.attn_alpha = M.all(n("attn_resid_alpha.weight")); b.attn_beta = M.all(n("attn_resid_beta.weight"));
        b.ffn_norm = M.all(n("ffn_norm.weight")); b.latent_down = M.all(n("ffn_latent_down.weight"));
        b.pre_exp_norm = M.all(n("ffn_pre_exp_norm.weight")); b.router = M.all(n("ffn_gate_inp.weight"));
        b.router_bias = M.all(n("exp_probs_b.bias")); b.post_exp_norm = M.all(n("ffn_post_exp_norm.weight"));
        b.latent_up = M.all(n("ffn_latent_up.weight")); b.post_ffw_norm = M.all(n("post_ffw_norm.weight"));
        b.ffn_alpha = M.all(n("ffn_resid_alpha.weight")); b.ffn_beta = M.all(n("ffn_resid_beta.weight"));
        b.expert = [&M, &c, &cache, l](int e, std::vector<float>& g, std::vector<float>& u, std::vector<float>& dn) {
            auto& slot = cache[(size_t) l];
            auto it = slot.find(e);
            if (it == slot.end()) {
                const std::string p = "blk." + std::to_string(l) + ".";
                std::vector<float> a, bb, cc;
                M.rows(p + "ffn_gate_exps.weight", (int64_t) e * c.expert_ff, c.expert_ff, a);
                M.rows(p + "ffn_up_exps.weight", (int64_t) e * c.expert_ff, c.expert_ff, bb);
                M.rows(p + "ffn_down_exps.weight", (int64_t) e * c.latent, c.latent, cc);
                it = slot.emplace(e, std::make_tuple(std::move(a), std::move(bb), std::move(cc))).first;
            }
            g = std::get<0>(it->second);
            u = std::get<1>(it->second);
            dn = std::get<2>(it->second);
        };
        std::fprintf(stderr, "layer %d (%s) loaded\n", l, b.swa ? "sliding window, RoPE" : "full attention, NoPE");
    }

    // Negative controls (STRATA_MUSE_CONTROL): plausible mistakes the check must catch.
    //   rope_full: RoPE on the full-attention layers too;  no_bias: the router bias left out of the choice.
    if (const char* ctl = std::getenv("STRATA_MUSE_CONTROL")) {
        const std::string what = ctl;
        for (int l = 0; l < c.layers; ++l) {
            ref::Block& b = blocks[(size_t) l];
            if (what == "rope_full" && !b.swa) { b.swa = true; d.window = 1 << 30; }
            if (what == "no_bias") std::fill(b.router_bias.begin(), b.router_bias.end(), 0.f);
        }
        std::printf("negative control: %s\n", what.c_str());
    }

    std::vector<int> tokens;
    {
        std::ifstream tf(dir + "/tokens.txt");
        for (int t; tf >> t;) tokens.push_back(t);
    }
    const int T = (int) tokens.size(), H = d.hidden;
    std::vector<ref::KvCache> kv((size_t) c.layers);
    std::vector<std::vector<float>> embd(T), finals(T);
    std::vector<std::vector<ref::Routes>> tr((size_t) c.layers, std::vector<ref::Routes>((size_t) T));
    std::vector<std::vector<std::vector<float>>> lout((size_t) c.layers, std::vector<std::vector<float>>((size_t) T));
    for (int p = 0; p < T; ++p) {
        std::vector<float> x;
        M.rows("token_embd.weight", tokens[(size_t) p], 1, x);
        ref::rms_norm(x.data(), H, d.rms_eps);
        embd[(size_t) p] = x;
        for (int l = 0; l < c.layers; ++l) {
            ref::block_step(d, blocks[(size_t) l], kv[(size_t) l], p, x, &tr[(size_t) l][(size_t) p]);
            lout[(size_t) l][(size_t) p] = x;
        }
        finals[(size_t) p] = x;
        std::fprintf(stderr, "\rtoken %d/%d", p + 1, T);
    }
    std::fprintf(stderr, "\n");
    // the head: output_norm, then each vocab row dequantized once for all tokens
    const std::vector<float> onorm = M.all("output_norm.weight");
    std::vector<std::vector<float>> rnorm((size_t) T);
    for (int p = 0; p < T; ++p) {
        rnorm[(size_t) p] = finals[(size_t) p];
        ref::rms_norm(rnorm[(size_t) p].data(), H, d.rms_eps, onorm.data());
    }
    std::vector<std::vector<float>> head_in = rnorm;   // what the head's matmul sees
    if (like_llama)
        for (auto& v : head_in) round_type(vec_dot_type((int) M.t("output.weight").type), v.data(), H);
    const int64_t V = (int64_t) gm.find("output.weight")->shape[1];
    std::vector<float> logits((size_t) (V * T)), chunk;
    for (int64_t r0 = 0; r0 < V; r0 += 4096) {
        const int64_t rows = std::min<int64_t>(4096, V - r0);
        M.rows("output.weight", r0, rows, chunk);
        for (int64_t r = 0; r < rows; ++r)
            for (int p = 0; p < T; ++p) {
                double acc = 0;
                for (int i = 0; i < H; ++i) acc += (double) chunk[(size_t) (r * H + i)] * head_in[(size_t) p][(size_t) i];
                float lg = (float) acc * (float) c.logit_scale;
                if (c.softcap > 0) lg = (float) (c.softcap * std::tanh(lg / c.softcap));
                logits[(size_t) (p * V + r0 + r)] = lg;
            }
    }

    int failures = 0;
    std::printf("%d tokens, %d layers; reference%s vs llama.cpp (per token: max relative error, min cosine)\n", T, c.layers,
                like_llama ? " with ggml-cpu's activation rounding" : " in f32");
    auto compare = [&](const std::string& name, auto get, int dim, double rel_ok) {
        const Dump dd = load_dump(dir + "/" + name + ".bin");
        if (!dd.ok()) { std::printf("  %-22s not dumped\n", name.c_str()); return; }
        if (dd.ne[0] != dim || dd.ne[1] != T) {
            std::printf("  %-22s shape [%lld, %lld], expected [%d, %d]\n", name.c_str(), (long long) dd.ne[0], (long long) dd.ne[1], dim, T);
            ++failures;
            return;
        }
        Stat s;
        for (int p = 0; p < T; ++p) s.add(get(p), dd.f.data() + (size_t) p * dim, (size_t) dim);
        const bool ok = s.max_rel <= (like_llama ? 0.015 : rel_ok);   // matched rounding: 9e-3 at worst, measured
        std::printf("  %-22s rel %.2e  cos %.6f  %s\n", name.c_str(), s.max_rel, s.min_cos, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    compare("embd_norm", [&](int p) { return embd[(size_t) p].data(); }, H, 1e-5);
    for (int l = 0; l < c.layers; ++l) {
        const std::string s = "-" + std::to_string(l);
        compare("attn_post_norm" + s, [&](int p) { return tr[(size_t) l][(size_t) p].attn_post_norm.data(); }, H, 0.05);
        compare("ffn_inp" + s, [&](int p) { return tr[(size_t) l][(size_t) p].ffn_inp.data(); }, H, 0.05);
        const Dump topk = load_dump(dir + "/ffn_moe_topk" + s + ".bin");
        if (topk.ok()) {
            int same = 0, total = 0;
            for (int p = 0; p < T; ++p) {
                const auto& mine = tr[(size_t) l][(size_t) p].experts;
                for (int k = 0; k < c.top_k; ++k) {
                    const int e = topk.i[(size_t) (p * topk.ne[0] + k)];
                    same += std::find(mine.begin(), mine.end(), e) != mine.end();
                    ++total;
                }
            }
            const bool ok = same >= total - T;   // at most one near-tie swap per token
            std::printf("  %-22s %d of %d routes the same  %s\n", ("ffn_moe_topk" + s).c_str(), same, total, ok ? "ok" : "FAIL");
            failures += !ok;
        }
        compare("ffn_moe_out" + s, [&](int p) { return tr[(size_t) l][(size_t) p].moe_out.data(); }, d.latent, 0.1);
        compare("ffn_post_norm" + s, [&](int p) { return tr[(size_t) l][(size_t) p].ffn_post_norm.data(); }, H, 0.1);
        compare("l_out" + s, [&](int p) { return lout[(size_t) l][(size_t) p].data(); }, H, 0.05);
    }
    compare("result_norm", [&](int p) { return rnorm[(size_t) p].data(); }, H, 0.05);
    compare("result_output", [&](int p) { return logits.data() + (size_t) p * V; }, (int) V, 0.05);
    const Dump lg = load_dump(dir + "/result_output.bin");
    if (lg.ok()) {
        int agree = 0;
        for (int p = 0; p < T; ++p) {
            const float* a = logits.data() + (size_t) p * V;
            const float* b = lg.f.data() + (size_t) p * V;
            agree += std::max_element(a, a + V) - a == std::max_element(b, b + V) - b;
        }
        const bool ok = agree == T;
        std::printf("  %-22s argmax agrees at %d of %d positions  %s\n", "logits", agree, T, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    std::printf("muse_reference_check: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
