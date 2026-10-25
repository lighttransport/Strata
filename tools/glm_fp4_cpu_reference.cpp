// tools/glm_fp4_cpu_reference.cpp - golden CPU forward of GLM-5.3-Flash FP4-Local (the ModelOpt safetensors directory):
// every weight in its checkpoint value (NVFP4 e2m1 x e4m3 x global, MXFP8, block FP8, BF16) as F32, every dot product
// in double, F32 activations. It follows vulkan/fast/reference.cpp (the CPU reference of the GGUF runner) step by step;
// only the routed experts differ (dequantized from the checkpoint instead of the GGUF expert pool). Diagnostic only:
// slow, one token at a time, no MTP. Gate reference for the A64FX runner (docs/GLM_A64FX_FP4_LOCAL.md).
//
//   c++ -std=c++20 -O2 -fopenmp -Iinclude tools/glm_fp4_cpu_reference.cpp src/host/glm_fp4.c src/host/glm_names.c
//       src/host/glm_env.c src/host/glm_expert_pack.c -o glm_fp4_cpu_reference
//   glm_fp4_cpu_reference MODEL_DIR IDS OUTDIR [--tokens N] [--dump-at T,T,...]
//
// With -DSTRATA_GOLDEN_GGML (link ggml, add its include dir) the same forward also runs on a GLM-5.3-Flash GGUF
// (quantized trunk and routed experts dequantized by ggml): a semantic cross-check of the FP4-Local path.
//
// OUTDIR: logits.bin (tokens x vocab F32, every position), routes.bin (tokens x 42 x 8 int32 expert ids then the
// same count of F32 weights), dump-<t>/attn-<l>.bin and layer-<l>.bin (the 4 x hidden residual after each half-layer)
// for the positions in --dump-at, tokens.txt, summary.txt (argmax per position).
#include "strata/core/model.hpp"
#include "strata/kernels/glm_reference.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#ifdef STRATA_GOLDEN_GGML
#include <ggml.h>
#endif

namespace ref = strata::kernels::glm_reference;
using Vec = std::vector<float>;
using strata::artifact::GlmModelOpt;
using strata::artifact::ModelOptPacked;

namespace {

float bf16(const uint8_t *p) {
    uint32_t v = (uint32_t(p[0]) | (uint32_t(p[1]) << 8)) << 16;
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// one row of `k` values of a stored tensor (F32, BF16, or a ggml quantized type) as F32
void row_f32(int type, const uint8_t *src, size_t k, float *dst) {
    if (type == 0) std::memcpy(dst, src, k * 4);
    else if (type == 30) for (size_t i = 0; i < k; ++i) dst[i] = bf16(src + 2 * i);
    else {
#ifdef STRATA_GOLDEN_GGML
        ggml_internal_get_type_traits(ggml_type(type)).to_float(src, dst, int64_t(k));
#else
        throw std::runtime_error("golden: quantized GGUF tensors need a -DSTRATA_GOLDEN_GGML build");
#endif
    }
}

struct Golden {
    strata::core::ModelArtifact &a;
    const strata::core::ModelDescriptor &m;
    const GlmModelOpt *opt;   // FP4-Local only
    int position = 0;
    struct State {
        std::array<Vec, 3> conv;
        Vec keys, gates, pooled, cache, kda;
    };
    std::vector<State> states;
    std::vector<int32_t> route_ids;
    Vec route_weights;

    Golden(strata::core::ModelArtifact &artifact, int capacity)
        : a(artifact), m(artifact.descriptor()), opt(artifact.is_modelopt() ? artifact.exl3().modelopt.get() : nullptr) {
        states.resize(m.layers.size());
        for (size_t l = 0; l < states.size(); ++l) {
            auto &s = states[l];
            if (m.layers[l].mixer == strata::core::MixerKind::Kda) {
                for (auto &v : s.conv) v.assign(m.linear_heads * m.linear_dim * (m.conv_kernel - 1), 0);
                s.kda.assign(size_t(m.linear_heads) * m.linear_dim * m.linear_dim, 0);
            } else {
                s.keys.assign(m.index_pool * m.index_dim, 0);
                s.gates.assign(m.index_pool * m.index_dim, 0);
                s.pooled.assign(size_t(capacity / m.index_pool + 1) * m.index_dim, 0);
                s.cache.assign(size_t(capacity) * m.kv_rank, 0);
            }
        }
    }
    Vec f32(const std::string &n) {
        const auto &t = a.at(n);
        Vec out(t.tensor->elements());
        row_f32(int(t.tensor->type), t.data(), out.size(), out.data());
        return out;
    }
    // y[h][row] = sum_k W[h][row][k] * x[h][k], double accumulation
    Vec mat(const std::string &n, const Vec &x, int heads = 1) {
        const auto &t = a.at(n);
        const size_t k = t.tensor->shape[0], rows = t.tensor->shape[1];
        if (x.size() != k * heads) throw std::runtime_error("golden width mismatch: " + n);
        const uint8_t *d = t.data();   // materializes once, outside the parallel region
        const int type = int(t.tensor->type);
        const long total = long(rows * heads);
        const size_t rb = t.bytes / size_t(total);
        Vec y(total);
#pragma omp parallel
        {
            Vec w(k);
#pragma omp for schedule(static)
            for (long r = 0; r < total; ++r) {
                const size_t h = r / rows;
                row_f32(type, d + size_t(r) * rb, k, w.data());
                double s = 0;
                for (size_t i = 0; i < k; ++i) s += double(w[i]) * x[h * k + i];
                y[r] = float(s);
            }
        }
        return y;
    }
    // y = W x for one checkpoint matrix (NVFP4 experts), in the checkpoint's values
    Vec packed(const ModelOptPacked &p, const Vec &x) {
        if (x.size() != p.columns) throw std::runtime_error("golden expert width mismatch");
        Vec y(p.rows);
        const uint8_t *w = p.data();
#pragma omp parallel for schedule(static)
        for (long r = 0; r < long(p.rows); ++r) {
            double s = 0;
            for (size_t c = 0; c < p.columns; c += 16) {
                const double scale = double(strata_fp4_e4m3(p.scales[r * (p.columns / 16) + c / 16])) * p.global;
                double g = 0;
                for (size_t i = 0; i < 16; ++i) {
                    const size_t e = r * p.columns + c + i;
                    g += double(strata_fp4_e2m1((w[e / 2] >> ((e % 2) * 4)) & 15)) * x[c + i];
                }
                s += g * scale;
            }
            y[r] = float(s);
        }
        return y;
    }
    // GGUF routed experts: expert `e` of a stacked [experts][rows][k] tensor
    Vec expert(const std::string &n, int e, const Vec &x) {
        const auto &t = a.at(n);
        const size_t k = t.tensor->shape[0], rows = t.tensor->shape[1];
        if (x.size() != k) throw std::runtime_error("golden expert width mismatch: " + n);
        const size_t rb = t.bytes / (rows * size_t(m.experts));
        const uint8_t *d = t.data() + size_t(e) * rows * rb;
        Vec y(rows);
#pragma omp parallel
        {
            Vec w(k);
#pragma omp for schedule(static)
            for (long r = 0; r < long(rows); ++r) {
                row_f32(int(t.tensor->type), d + size_t(r) * rb, k, w.data());
                double s = 0;
                for (size_t i = 0; i < k; ++i) s += double(w[i]) * x[i];
                y[r] = float(s);
            }
        }
        return y;
    }
    Vec norm(const Vec &x, const Vec &w) {
        double s = 0;
        for (float v : x) s += double(v) * v;
        const float inv = float(1 / std::sqrt(s / x.size() + m.rms_epsilon));
        Vec y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * (w.empty() ? 1 : w[i]);
        return y;
    }
    std::pair<Vec, Vec> read(const std::string &p, const char *kind, const Vec &r) {
        auto projected = mat(p + "hc_" + kind + "_fn.weight", norm(r, {}));
        Vec c(24), x(m.hidden);
        ref::mhc(projected.data(), f32(p + "hc_" + kind + "_base.weight").data(), f32(p + "hc_" + kind + "_scale.weight").data(),
                 c.data(), m.sinkhorn_iterations, m.hc_epsilon);
        for (int d = 0; d < m.hidden; ++d)
            for (int i = 0; i < 4; ++i) x[d] += c[i] * r[i * m.hidden + d];
        return {x, c};
    }
    void write(Vec &r, const Vec &c, const Vec &y) {
        Vec out(r.size());
        for (int j = 0; j < 4; ++j)
            for (int d = 0; d < m.hidden; ++d) {
                float v = c[4 + j] * y[d];
                for (int i = 0; i < 4; ++i) v += c[8 + i * 4 + j] * r[i * m.hidden + d];
                out[j * m.hidden + d] = v;
            }
        r.swap(out);
    }
    Vec ffn(const std::string &p, const Vec &x, const std::string &suffix, float limit) {
        auto g = mat(p + "ffn_gate" + suffix + ".weight", x), u = mat(p + "ffn_up" + suffix + ".weight", x);
        for (size_t i = 0; i < g.size(); ++i) g[i] = ref::swiglu(g[i], u[i], limit);
        return mat(p + "ffn_down" + suffix + ".weight", g);
    }
    Vec kda(const std::string &p, int layer, const Vec &x) {
        const int n = m.linear_heads * m.linear_dim, K = m.conv_kernel;
        auto &s = states[layer];
        std::array<Vec, 3> qkv;
        const char *parts[] = {"q", "k", "v"};
        for (int part = 0; part < 3; ++part) {
            auto raw = mat(p + "attn_" + parts[part] + ".weight", x);
            auto w = f32(p + "ssm_conv1d_" + parts[part] + ".weight");
            auto &hist = s.conv[part];
            qkv[part].resize(n);
            for (int i = 0; i < n; ++i) {
                float sum = raw[i] * w[i * K + K - 1];
                for (int j = 0; j < K - 1; ++j) sum += hist[i * (K - 1) + j] * w[i * K + j];
                for (int j = 0; j < K - 2; ++j) hist[i * (K - 1) + j] = hist[i * (K - 1) + j + 1];
                hist[i * (K - 1) + K - 2] = raw[i];
                qkv[part][i] = sum * ref::sigmoid(sum);
            }
        }
        auto decay = mat(p + "ssm_f_b.weight", mat(p + "ssm_f_a.weight", x));
        auto bias = f32(p + "ssm_dt.bias"), av = f32(p + "ssm_a");
        if (std::all_of(av.begin(), av.end(), [](float v) { return v > 0; }))
            for (auto &v : av) v = -v;
        for (int i = 0; i < n; ++i) decay[i] = m.gate_lower_bound * ref::sigmoid(-av[i / m.linear_dim] * (decay[i] + bias[i]));
        auto beta = mat(p + "ssm_beta.weight", x);
        Vec y(n);
        ref::kda(s.kda.data(), qkv[0].data(), qkv[1].data(), qkv[2].data(), decay.data(), beta.data(), y.data(), m.linear_heads,
                 m.linear_dim);
        auto gate = mat(p + "ssm_g_b.weight", mat(p + "ssm_g_a.weight", x));
        auto w = f32(p + "ssm_norm.weight");
        for (int h = 0; h < m.linear_heads; ++h) {
            double sum = 0;
            for (int i = 0; i < m.linear_dim; ++i) sum += double(y[h * m.linear_dim + i]) * y[h * m.linear_dim + i];
            const float inv = float(1 / std::sqrt(sum / m.linear_dim + m.rms_epsilon));
            for (int i = 0; i < m.linear_dim; ++i) {
                const int j = h * m.linear_dim + i;
                y[j] = y[j] * inv * w[i] * ref::sigmoid(gate[j]);
            }
        }
        return mat(p + "attn_output.weight", y);
    }
    Vec mla(const std::string &p, int layer, const Vec &x) {
        auto qr = norm(mat(p + "attn_q_a.weight", x), f32(p + "attn_q_a_norm.weight"));
        auto q = mat(p + "attn_q_b.weight", qr), iq = mat(p + "indexer.attn_q_b.weight", qr);
        auto row = norm(mat(p + "attn_kv_a_mqa.weight", x), f32(p + "attn_kv_a_norm.weight"));
        auto iw = mat(p + "indexer.proj.weight", x), ik = mat(p + "indexer.attn_k.weight", x),
             gate = mat(p + "indexer_compressor_gate.weight", x);
        auto &s = states[layer];
        const int dim = m.index_dim, P = m.index_pool, tokens = position + 1, pools = tokens / P, tail = position % P;
        auto nw = f32(p + "indexer.k_norm.weight"), nb = f32(p + "indexer.k_norm.bias");
        double mean = 0, var = 0;
        for (float v : ik) mean += v;
        mean /= dim;
        for (float v : ik) var += (v - mean) * (v - mean);
        const float inv = float(1 / std::sqrt(var / dim + 1e-6));
        for (int i = 0; i < dim; ++i) {
            s.keys[tail * dim + i] = float(ik[i] - mean) * inv * nw[i] + nb[i];
            s.gates[tail * dim + i] = gate[i];
        }
        if (tokens % P == 0)
            ref::pool(s.keys.data(), s.gates.data(), f32(p + "indexer_compressor_ape.weight").data(), s.pooled.data() + (pools - 1) * dim,
                      P, dim);
        Vec scores(pools);
        for (int b = 0; b < pools; ++b) {
            float score = 0;
            for (int h = 0; h < m.index_heads; ++h) {
                double dot = 0;
                for (int d = 0; d < dim; ++d) dot += double(iq[h * dim + d]) * s.pooled[b * dim + d];
                score += iw[h] * std::max(float(dot) / std::sqrt(float(dim)), 0.f);
            }
            scores[b] = score / std::sqrt(float(m.index_heads));
        }
        std::vector<int> order(pools);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int x1, int x2) { return scores[x1] > scores[x2]; });
        order.resize(std::min(int(m.index_top_k / P), pools));
        std::sort(order.begin(), order.end());
        std::vector<int> ids;
        for (int b : order)
            for (int i = 0; i < P; ++i) ids.push_back(b * P + i);
        for (int i = pools * P; i < tokens; ++i) ids.push_back(i);
        std::copy(row.begin(), row.end(), s.cache.begin() + size_t(position) * m.kv_rank);
        auto aq = mat(p + "attn_k_b.weight", q, m.attention_heads);
        const int latent = m.kv_rank, head_dim = int(a.at(p + "attn_k_b.weight").tensor->shape[0]);
        const float scale = 1 / std::sqrt(float(head_dim));
        Vec av(size_t(m.attention_heads) * latent);
#pragma omp parallel for schedule(static)
        for (int h = 0; h < m.attention_heads; ++h) {
            std::vector<double> prob(ids.size());
            double mx = -1e300, total = 0;
            for (size_t si = 0; si < ids.size(); ++si) {
                double dot = 0;
                for (int i = 0; i < latent; ++i) dot += double(aq[h * latent + i]) * s.cache[size_t(ids[si]) * latent + i];
                prob[si] = dot * scale;
                mx = std::max(mx, prob[si]);
            }
            for (auto &v : prob) { v = std::exp(v - mx); total += v; }
            for (int i = 0; i < latent; ++i) {
                double sum = 0;
                for (size_t si = 0; si < ids.size(); ++si) sum += prob[si] * s.cache[size_t(ids[si]) * latent + i];
                av[h * latent + i] = float(sum / total);
            }
        }
        return mat(p + "attn_output.weight", mat(p + "attn_v_b.weight", av, m.attention_heads));
    }
    Vec moe(int layer, const std::string &p, const Vec &x, const strata::core::LayerDescriptor &desc) {
        auto logits = mat(p + "ffn_gate_inp.weight", x);
        std::vector<int> ids(m.top_k);
        Vec routing(m.top_k);
        ref::router(logits.data(), f32(p + "exp_probs_b.bias").data(), m.experts, m.top_k, m.expert_scale, ids.data(), routing.data());
        std::vector<double> sum(m.hidden, 0.0);
        const std::string src = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
        for (int i = 0; i < m.top_k; ++i) {
            Vec g, u, y;
            if (a.is_modelopt()) {
                const auto e = src + std::to_string(ids[i]) + ".";
                g = packed(opt->load(e + "gate_proj.weight"), x), u = packed(opt->load(e + "up_proj.weight"), x);
                for (size_t j = 0; j < g.size(); ++j) g[j] = ref::swiglu(g[j], u[j], desc.swiglu_limit);
                y = packed(opt->load(e + "down_proj.weight"), g);
            } else {
                g = expert(p + "ffn_gate_exps.weight", ids[i], x), u = expert(p + "ffn_up_exps.weight", ids[i], x);
                for (size_t j = 0; j < g.size(); ++j) g[j] = ref::swiglu(g[j], u[j], desc.swiglu_limit);
                y = expert(p + "ffn_down_exps.weight", ids[i], g);
            }
            for (int d = 0; d < m.hidden; ++d) sum[d] += double(routing[i]) * y[d];
            route_ids.push_back(ids[i]);
            route_weights.push_back(routing[i]);
        }
        auto shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
        Vec out(m.hidden);
        for (int d = 0; d < m.hidden; ++d) out[d] = float(sum[d] + shared[d]);
        return out;
    }
    Vec step(int token, const std::filesystem::path &dump) {
        const auto &e = a.at("token_embd.weight");
        Vec emb(m.hidden);
        row_f32(int(e.tensor->type), e.data() + size_t(token) * (e.bytes / m.vocab), m.hidden, emb.data());
        Vec r(4 * m.hidden);
        for (int i = 0; i < 4; ++i) std::copy(emb.begin(), emb.end(), r.begin() + i * m.hidden);
        auto save = [&](const char *kind, size_t l) {
            if (dump.empty()) return;
            std::ofstream o(dump / (std::string(kind) + "-" + std::to_string(l) + ".bin"), std::ios::binary);
            o.write(reinterpret_cast<const char *>(r.data()), r.size() * 4);
        };
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const auto p = "blk." + std::to_string(l) + ".";
            auto [cx, c] = read(p, "attn", r);
            auto x = norm(cx, f32(p + "attn_norm.weight"));
            auto y = m.layers[l].mixer == strata::core::MixerKind::Kda ? kda(p, int(l), x) : mla(p, int(l), x);
            write(r, c, y);
            save("attn", l);
            auto f = read(p, "ffn", r);
            x = norm(f.first, f32(p + "ffn_norm.weight"));
            const auto &desc = m.layers[l];
            y = desc.ffn == strata::core::FfnKind::Dense ? ffn(p, x, "", desc.swiglu_limit) : moe(int(l), p, x, desc);
            write(r, f.second, y);
            save("layer", l);
        }
        Vec collapsed(m.hidden);
        for (int d = 0; d < m.hidden; ++d) collapsed[d] = (r[d] + r[m.hidden + d] + r[2 * m.hidden + d] + r[3 * m.hidden + d]) * .25f;
        ++position;
        return mat("output.weight", norm(collapsed, f32("output_norm.weight")));
    }
};

}   // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 4) throw std::invalid_argument("usage: glm_fp4_cpu_reference MODEL_DIR IDS OUTDIR [--tokens N] [--dump-at T,...]");
        size_t limit = SIZE_MAX;
        std::set<size_t> dump_at;
        for (int i = 4; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--tokens") limit = std::stoul(v);
            else if (k == "--dump-at") {
                std::istringstream is(v);
                for (std::string t; std::getline(is, t, ',');) dump_at.insert(std::stoul(t));
            } else throw std::invalid_argument("unknown option " + k);
        }
#ifdef STRATA_GOLDEN_GGML
        {   // older ggml fills its F16 conversion table in ggml_init
            ggml_init_params params{1 << 20, nullptr, true};
            ggml_free(ggml_init(params));
        }
#endif
        strata::core::ModelArtifact artifact{std::filesystem::path(argv[1])};
        if (artifact.is_exl3() && !artifact.is_modelopt()) throw std::invalid_argument("golden reference: FP4-Local directory or a GGUF");
        std::ifstream in(argv[2]);
        std::string text((std::istreambuf_iterator<char>(in)), {});
        std::replace(text.begin(), text.end(), ',', ' ');
        std::istringstream is(text);
        std::vector<int> tokens;
        for (int t; is >> t && tokens.size() < limit;) tokens.push_back(t);
        if (tokens.empty()) throw std::invalid_argument("empty prompt");
        const std::filesystem::path out(argv[3]);
        std::filesystem::create_directories(out);
        Golden g(artifact, int((tokens.size() + 64) / 4 * 4));
        std::ofstream logits_out(out / "logits.bin", std::ios::binary), summary(out / "summary.txt"), tok(out / "tokens.txt");
        for (int t : tokens) tok << t << '\n';
        std::vector<int32_t> ids_all;
        Vec weights_all;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            std::filesystem::path dump;
            if (dump_at.count(i)) { dump = out / ("dump-" + std::to_string(i)); std::filesystem::create_directories(dump); }
            const auto logits = g.step(tokens[i], dump);
            logits_out.write(reinterpret_cast<const char *>(logits.data()), logits.size() * 4);
            logits_out.flush();
            const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            summary << i << ' ' << tokens[i] << ' ' << best << ' ' << logits[best] << '\n';
            summary.flush();
            std::cerr << "GOLDEN token " << i << " id " << tokens[i] << " argmax " << best << " " << sec << " s\n";
        }
        std::ofstream routes(out / "routes.bin", std::ios::binary);
        routes.write(reinterpret_cast<const char *>(g.route_ids.data()), g.route_ids.size() * 4);
        routes.write(reinterpret_cast<const char *>(g.route_weights.data()), g.route_weights.size() * 4);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "GOLDEN_ERROR " << e.what() << '\n';
        return 1;
    }
}
