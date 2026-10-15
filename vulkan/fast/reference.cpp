// CPU reference for the GLM-5.3f step: FP32 activations, dequantized weights,
// double-precision dot products, routed experts through the canonical CPU pool.
// It follows Decoder::step in decode.cpp (unfused KDA, CPU mHC, CPU indexer) and
// writes per-layer residuals and logits for the last token of the prompt.
// Diagnostic only: slow (all dense weights are dequantized for every token).
#include "model_checks.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/glm_reference.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ggml.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
namespace ref = strata::kernels::glm_reference;
namespace kcpu = strata::kernels::cpu;
using Vec = std::vector<float>;

struct Reference {
  strata::core::ModelArtifact &a;
  const strata::core::ModelDescriptor &m;
  kcpu::ExpertPool pool;
  int threads, position = 0;
  struct State {
    std::array<Vec, 3> conv;
    Vec keys, gates, pooled, cache, kda;
  };
  std::vector<State> states;
  Reference(strata::core::ModelArtifact &artifact, const strata::core::ModelDescriptor &model, int t, int capacity)
      : a(artifact), m(model), pool(t, true, true), threads(t) {
    states.resize(m.layers.size());
    for (size_t l = 0; l < states.size(); ++l) {
      auto &s = states[l];
      if (m.layers[l].mixer == strata::core::MixerKind::Kda) {
        for (auto &v : s.conv) v.assign(m.linear_heads * m.linear_dim * (m.conv_kernel - 1), 0);
        s.kda.assign(size_t(m.linear_heads) * m.linear_dim * m.linear_dim, 0);
      } else {
        s.keys.assign(m.index_pool * m.index_dim, 0);
        s.gates.assign(m.index_pool * m.index_dim, 0);
        s.pooled.assign(size_t(capacity / m.index_pool) * m.index_dim, 0);
        s.cache.assign(size_t(capacity) * m.kv_rank, 0);
      }
    }
  }
  Vec f32(const std::string &n) {
    const auto &t = a.at(n);
    Vec out(t.tensor->elements());
    if (t.tensor->type == 0) std::memcpy(out.data(), t.data(), out.size() * 4);
    else ggml_get_type_traits(ggml_type(t.tensor->type))->to_float(t.data(), out.data(), out.size());
    return out;
  }
  // y[h][row] = sum_k W[h][row][k] * x[h][k]
  Vec mat(const std::string &n, const Vec &x, int heads = 1) {
    const auto &t = a.at(n);
    const size_t k = t.tensor->shape[0], rows = t.tensor->shape[1];
    if (x.size() != k * heads) throw std::runtime_error("reference width mismatch: " + n);
    const size_t rb = t.bytes / (rows * heads);
    Vec y(rows * heads);
    auto *traits = ggml_get_type_traits(ggml_type(t.tensor->type));
    std::vector<std::thread> pool_threads;
    const size_t total = rows * heads;
    for (int w = 0; w < threads; ++w)
      pool_threads.emplace_back([&, w] {
        Vec row(k);
        for (size_t r = w; r < total; r += threads) {
          const uint8_t *src = t.data() + r * rb;
          if (t.tensor->type == 0) std::memcpy(row.data(), src, k * 4);
          else traits->to_float(src, row.data(), k);
          const size_t h = r / rows;
          double s = 0;
          for (size_t i = 0; i < k; ++i) s += double(row[i]) * x[h * k + i];
          y[r] = float(s);
        }
      });
    for (auto &t2 : pool_threads) t2.join();
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
    ref::mhc(projected.data(), f32(p + "hc_" + kind + "_base.weight").data(),
             f32(p + "hc_" + kind + "_scale.weight").data(), c.data(), m.sinkhorn_iterations, m.hc_epsilon);
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
    for (int i = 0; i < n; ++i)
      decay[i] = m.gate_lower_bound * ref::sigmoid(-av[i / m.linear_dim] * (decay[i] + bias[i]));
    auto beta = mat(p + "ssm_beta.weight", x);
    Vec y(n);
    ref::kda(s.kda.data(), qkv[0].data(), qkv[1].data(), qkv[2].data(), decay.data(), beta.data(), y.data(),
             m.linear_heads, m.linear_dim);
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
      ref::pool(s.keys.data(), s.gates.data(), f32(p + "indexer_compressor_ape.weight").data(),
                s.pooled.data() + (pools - 1) * dim, P, dim);
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
    const int latent = m.kv_rank, head_dim = a.at(p + "attn_k_b.weight").tensor->shape[0];
    const float scale = 1 / std::sqrt(float(head_dim));
    Vec av(size_t(m.attention_heads) * latent);
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
  Vec moe(const std::string &p, const Vec &x, const strata::core::LayerDescriptor &desc) {
    auto logits = mat(p + "ffn_gate_inp.weight", x);
    std::vector<int> ids(m.top_k);
    Vec routing(m.top_k);
    ref::router(logits.data(), f32(p + "exp_probs_b.bias").data(), m.experts, m.top_k, m.expert_scale, ids.data(),
                routing.data());
    const auto &G = a.at(p + "ffn_gate_exps.weight"), &U = a.at(p + "ffn_up_exps.weight"),
               &D = a.at(p + "ffn_down_exps.weight");
    kcpu::NativeFmt format;
    std::string error;
    if (!kcpu::native_fmt(G.tensor->type, D.tensor->type, m.hidden, desc.intermediate, format, error))
      throw std::runtime_error(error);
    format.swiglu_limit = desc.swiglu_limit;
    format.canon = (G.tensor->type == 10 || G.tensor->type == 11) && (D.tensor->type == 10 || D.tensor->type == 11);
    std::vector<uint8_t> act(format.act_bytes);
    kcpu::native_quant_act(format, x.data(), act.data());
    Vec results(m.top_k * m.hidden), sum(m.hidden);
    std::vector<kcpu::ExpertJobMulti> jobs(m.top_k);
    for (int i = 0; i < m.top_k; ++i) {
      auto &j = jobs[i];
      j.expert_id = ids[i];
      j.nt = 1;
      j.blob = G.data() + size_t(ids[i]) * G.bytes / m.experts;
      j.native_up = U.data() + size_t(ids[i]) * U.bytes / m.experts;
      j.native_down = D.data() + size_t(ids[i]) * D.bytes / m.experts;
      j.nact[0] = act.data();
      j.out[0] = results.data() + i * m.hidden;
    }
    pool.run_layer_native(format, jobs.data(), m.top_k, results.data(), routing.data(), sum.data(), 1, m.top_k);
    auto shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
    for (int i = 0; i < m.hidden; ++i) sum[i] += shared[i];
    return sum;
  }
  Vec step(int token, const std::string &dump) {
    const auto &e = a.at("token_embd.weight");
    Vec emb(m.hidden);
    ggml_get_type_traits(ggml_type(e.tensor->type))->to_float(e.data() + size_t(token) * e.bytes / m.vocab, emb.data(), m.hidden);
    Vec r(4 * m.hidden);
    for (int i = 0; i < 4; ++i) std::copy(emb.begin(), emb.end(), r.begin() + i * m.hidden);
    for (size_t l = 0; l < m.layers.size(); ++l) {
      const auto p = "blk." + std::to_string(l) + ".";
      auto [cx, c] = read(p, "attn", r);
      auto x = norm(cx, f32(p + "attn_norm.weight"));
      auto y = m.layers[l].mixer == strata::core::MixerKind::Kda ? kda(p, int(l), x) : mla(p, int(l), x);
      write(r, c, y);
      if (!dump.empty()) {
        std::ofstream o(std::filesystem::path(dump) / ("attn-" + std::to_string(l) + ".bin"), std::ios::binary);
        o.write(reinterpret_cast<const char *>(r.data()), r.size() * 4);
      }
      auto f = read(p, "ffn", r);
      x = norm(f.first, f32(p + "ffn_norm.weight"));
      const auto &desc = m.layers[l];
      y = desc.ffn == strata::core::FfnKind::Dense ? ffn(p, x, "", desc.swiglu_limit) : moe(p, x, desc);
      write(r, f.second, y);
      if (!dump.empty()) {
        std::ofstream o(std::filesystem::path(dump) / ("layer-" + std::to_string(l) + ".bin"), std::ios::binary);
        o.write(reinterpret_cast<const char *>(r.data()), r.size() * 4);
      }
    }
    Vec collapsed(m.hidden);
    for (int d = 0; d < m.hidden; ++d)
      collapsed[d] = (r[d] + r[m.hidden + d] + r[2 * m.hidden + d] + r[3 * m.hidden + d]) * .25f;
    ++position;
    return mat("output.weight", norm(collapsed, f32("output_norm.weight")));
  }
};

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: glm_vulkan_fast_ref MODEL IDS OUTDIR");
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &m = strata::vulkan::checked_model(artifact);
    std::ifstream in(argv[2]);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream is(text);
    std::vector<int> tokens;
    for (int t; is >> t;) tokens.push_back(t);
    if (tokens.empty()) throw std::invalid_argument("empty prompt");
    std::filesystem::create_directories(argv[3]);
    Reference ref(artifact, m, 12, int((tokens.size() + 64) / 4 * 4));
    Vec logits;
    for (size_t i = 0; i < tokens.size(); ++i) {
      logits = ref.step(tokens[i], i + 1 == tokens.size() ? argv[3] : "");
      std::cerr << "REF token " << i << " argmax " << (std::max_element(logits.begin(), logits.end()) - logits.begin()) << '\n';
    }
    std::ofstream o(std::filesystem::path(argv[3]) / "logits.bin", std::ios::binary);
    o.write(reinterpret_cast<const char *>(logits.data()), logits.size() * 4);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "REF_ERROR " << e.what() << '\n';
    return 1;
  }
}
