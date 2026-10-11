// Experimental Vulkan GLM-5.3f runner: GPU projections/mHC/KDA/MLA, existing
// CPU routed experts.
#include "cpu_executor.hpp"
#include "model_checks.hpp"
#include "projections.hpp"
#include "runtime.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/glm_reference.hpp"
#include "task_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <ggml.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif
namespace cpu = strata::kernels::cpu;
namespace ref = strata::kernels::glm_reference;
using Vec = std::vector<float>;
using Clock = std::chrono::steady_clock;
void trace(const std::string &phase, int pos, int layer = -1) {
  const char *path = std::getenv("STRATA_GLM_TASK_TRACE");
  if (!path || !*path)
    return;
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
  std::ofstream f(path, std::ios::app);
  if (!f)
    throw std::runtime_error("Vulkan task trace unavailable");
  f << "{\"unix_ms\":" << time << ",\"phase\":\"" << phase
    << "\",\"position\":" << pos << ",\"layer\":" << layer << "}\n";
  f.flush();
  if (!f)
    throw std::runtime_error("Vulkan task trace write failed");
#ifdef __linux__
  int fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0 || fsync(fd)) {
    if (fd >= 0)
      close(fd);
    throw std::runtime_error("Vulkan task trace sync failed");
  }
  close(fd);
#endif
}
Vec dequant(const strata::core::ArtifactTensor &t) {
  size_t count = t.tensor->elements();
  Vec out(count);
  auto type = ggml_type(t.tensor->type);
  if (type == GGML_TYPE_F32)
    std::memcpy(out.data(), t.data(), count * 4);
  else {
    auto *traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float)
      throw std::runtime_error(
          "Vulkan scalar tensor dequantization unavailable: " + t.tensor->name);
    traits->to_float(t.data(), out.data(), count);
  }
  return out;
}
Vec norm(std::span<const float> x, std::span<const float> weight, float eps) {
  float sum = 0;
  for (float v : x)
    sum += v * v;
  float inv = 1 / std::sqrt(sum / x.size() + eps);
  Vec out(x.size());
  for (size_t i = 0; i < x.size(); i++)
    out[i] = x[i] * inv * (weight.empty() ? 1 : weight[i]);
  return out;
}
class Decoder {
  strata::core::ModelArtifact artifact_;
  const strata::core::ModelDescriptor &m_;
  strata::vulkan::Projections projections_;
  strata::vulkan::Compute compute_;
  cpu::ExpertPool pool_;
  std::unique_ptr<strata::vulkan::CpuExecutor> executor_;
  cpu::ThreadAffinity gpu_affinity_;
  int capacity_, position_ = 0;
  bool overlap_experts_ = false, cpu_mhc_ = false, joined_projections_ = false;
  double cpu_expert_ms_ = 0;
  bool phased_experts_ = false, fused_kda_ = false;
  std::map<std::string, Vec> scalars_;
  struct State {
    std::array<Vec, 3> conv;
    Vec keys, gates, pooled, fused_weights;
  };
  std::vector<State> states_;
  const Vec &scalar(const std::string &name) {
    auto i = scalars_.find(name);
    if (i != scalars_.end())
      return i->second;
    auto v = dequant(artifact_.at(name));
    if (name.ends_with(".ssm_a") &&
        std::all_of(v.begin(), v.end(), [](float a) { return a > 0; }))
      for (float &a : v)
        a = -a;
    return scalars_.emplace(name, std::move(v)).first->second;
  }
  Vec mat(const std::string &name, std::span<const float> x, int heads = 1) {
    return projections_.mat(name, artifact_.at(name), x, heads);
  }
  std::vector<Vec> mats(const std::vector<std::string> &names, const Vec &x) {
    std::vector<std::pair<std::string, const strata::core::ArtifactTensor *>>
        items;
    for (const auto &name : names)
      items.emplace_back(name, &artifact_.at(name));
    if (joined_projections_)
      return projections_.joined(items, x);
    std::vector<Vec> results;
    for (const auto &name : names)
      results.push_back(mat(name, x));
    return results;
  }
  std::vector<Vec> low_gates(const std::string &p, const Vec &f, const Vec &g) {
    const std::string fn = p + "ssm_f_b.weight", gn = p + "ssm_g_b.weight";
    if (joined_projections_)
      return projections_.multiple(
          {{fn, &artifact_.at(fn)}, {gn, &artifact_.at(gn)}}, {f, g});
    return {mat(fn, f), mat(gn, g)};
  }
  Vec normalize(const std::string &name, std::span<const float> x) {
    return norm(x, scalar(name), m_.rms_epsilon);
  }
  std::pair<Vec, Vec> read(const std::string &p, const char *kind,
                           const Vec &r) {
    auto normalized = norm(r, {}, m_.rms_epsilon);
    auto projected = mat(p + "hc_" + kind + "_fn.weight", normalized);
    Vec c(24), x(m_.hidden);
    if (cpu_mhc_) {
      ref::mhc(projected.data(),
               scalar(p + "hc_" + kind + "_base.weight").data(),
               scalar(p + "hc_" + kind + "_scale.weight").data(), c.data(),
               m_.sinkhorn_iterations, m_.hc_epsilon);
      for (int d = 0; d < m_.hidden; ++d)
        for (int i = 0; i < 4; ++i)
          x[d] += c[i] * r[i * m_.hidden + d];
    } else {
      compute_.mhc_read(r, projected, scalar(p + "hc_" + kind + "_base.weight"),
                        scalar(p + "hc_" + kind + "_scale.weight"), c, x,
                        m_.sinkhorn_iterations, m_.hc_epsilon);
    }
    return {std::move(x), std::move(c)};
  }
  void write(Vec &r, const Vec &c, const Vec &y) {
    if (cpu_mhc_) {
      Vec out(r.size());
      for (int j = 0; j < 4; ++j)
        for (int d = 0; d < m_.hidden; ++d) {
          float value = c[4 + j] * y[d];
          for (int i = 0; i < 4; ++i)
            value += c[8 + i * 4 + j] * r[i * m_.hidden + d];
          out[j * m_.hidden + d] = value;
        }
      r.swap(out);
    } else {
      compute_.mhc_write(r, c, y, r);
    }
  }
  Vec ffn(const std::string &p, const Vec &x, const std::string &suffix,
          float limit) {
    auto gu = mats({p + "ffn_gate" + suffix + ".weight",
                    p + "ffn_up" + suffix + ".weight"},
                   x);
    auto &g = gu[0];
    auto &u = gu[1];
    if (g.size() != u.size())
      throw std::runtime_error("Vulkan FFN geometry differs");
    for (size_t i = 0; i < g.size(); i++)
      g[i] = ref::swiglu(g[i], u[i], limit);
    return mat(p + "ffn_down" + suffix + ".weight", g);
  }
  Vec kda(const std::string &p, int layer, const Vec &x) {
    int n = m_.linear_heads * m_.linear_dim;
    auto &s = states_[layer];
    std::array<Vec, 3> qkv;
    const std::array<std::string, 3> parts{"q", "k", "v"};
    auto qk = mats({p + "attn_q.weight", p + "attn_k.weight",
                    p + "attn_v.weight", p + "ssm_f_a.weight",
                    p + "ssm_g_a.weight", p + "ssm_beta.weight"},
                   x);
    if (fused_kda_) {
      Vec raw(3 * n);
      for (int part = 0; part < 3; ++part) {
        std::copy(qk[part].begin(), qk[part].end(),
                  raw.begin() + size_t(part) * n);
      }
      if (s.fused_weights.empty())
        for (const auto &part : parts) {
          const auto &weight = scalar(p + "ssm_conv1d_" + part + ".weight");
          s.fused_weights.insert(s.fused_weights.end(), weight.begin(),
                                 weight.end());
        }
      auto fg = low_gates(p, qk[3], qk[4]);
      auto &decay = fg[0];
      auto &gate = fg[1];
      Vec y(n);
      compute_.kda_fused(layer, raw, decay, qk[5], gate, s.fused_weights,
                         scalar(p + "ssm_dt.bias"), scalar(p + "ssm_a"),
                         scalar(p + "ssm_norm.weight"), y, m_.linear_heads,
                         m_.conv_kernel, m_.gate_lower_bound, m_.rms_epsilon);
      return mat(p + "attn_output.weight", y);
    }
    for (int part = 0; part < 3; part++) {
      const auto &raw = qk[part];
      if (raw.size() != size_t(n))
        throw std::runtime_error("Vulkan KDA projection geometry differs");
      const auto &w = scalar(p + "ssm_conv1d_" + parts[part] + ".weight");
      auto &history = s.conv[part];
      qkv[part].resize(n);
      for (int i = 0; i < n; i++) {
        float sum = raw[i] * w[i * m_.conv_kernel + m_.conv_kernel - 1];
        for (int j = 0; j < m_.conv_kernel - 1; j++)
          sum +=
              history[i * (m_.conv_kernel - 1) + j] * w[i * m_.conv_kernel + j];
        for (int j = 0; j < m_.conv_kernel - 2; j++)
          history[i * (m_.conv_kernel - 1) + j] =
              history[i * (m_.conv_kernel - 1) + j + 1];
        if (m_.conv_kernel > 1)
          history[i * (m_.conv_kernel - 1) + m_.conv_kernel - 2] = raw[i];
        qkv[part][i] = sum * ref::sigmoid(sum);
      }
    }
    auto low = std::move(qk[3]);
    auto decay = mat(p + "ssm_f_b.weight", low);
    const auto &bias = scalar(p + "ssm_dt.bias");
    const auto &a = scalar(p + "ssm_a");
    for (int i = 0; i < n; i++)
      decay[i] = m_.gate_lower_bound *
                 ref::sigmoid(-a[i / m_.linear_dim] * (decay[i] + bias[i]));
    auto beta = std::move(qk[5]);
    Vec y(n);
    compute_.kda(layer, qkv[0], qkv[1], qkv[2], decay, beta, y, m_.linear_heads,
                 m_.linear_dim);
    low = std::move(qk[4]);
    auto gate = mat(p + "ssm_g_b.weight", low);
    const auto &w = scalar(p + "ssm_norm.weight");
    for (int h = 0; h < m_.linear_heads; h++) {
      float sum = 0;
      for (int i = 0; i < m_.linear_dim; i++)
        sum += y[h * m_.linear_dim + i] * y[h * m_.linear_dim + i];
      float inv = 1 / std::sqrt(sum / m_.linear_dim + m_.rms_epsilon);
      for (int i = 0; i < m_.linear_dim; i++) {
        int j = h * m_.linear_dim + i;
        y[j] = y[j] * inv * w[i] * ref::sigmoid(gate[j]);
      }
    }
    return mat(p + "attn_output.weight", y);
  }
  Vec mla(const std::string &p, int layer, const Vec &x) {
    auto initial = mats({p + "attn_q_a.weight", p + "attn_kv_a_mqa.weight",
                         p + "indexer.proj.weight", p + "indexer.attn_k.weight",
                         p + "indexer_compressor_gate.weight"},
                        x);
    auto qr = normalize(p + "attn_q_a_norm.weight", initial[0]);
    auto queries =
        mats({p + "attn_q_b.weight", p + "indexer.attn_q_b.weight"}, qr);
    auto q = std::move(queries[0]), iq = std::move(queries[1]);
    auto row = normalize(p + "attn_kv_a_norm.weight", initial[1]);
    auto iw = std::move(initial[2]), ik = std::move(initial[3]),
         gate = std::move(initial[4]);
    auto &s = states_[layer];
    const int dim = m_.index_dim, pool = m_.index_pool, tokens = position_ + 1,
              pools = tokens / pool, tail = position_ % pool;
    const auto &nw = scalar(p + "indexer.k_norm.weight");
    const auto &nb = scalar(p + "indexer.k_norm.bias");
    float mean = 0, sum = 0;
    for (float v : ik)
      mean += v;
    mean /= dim;
    for (float v : ik)
      sum += (v - mean) * (v - mean);
    float inv = 1 / std::sqrt(sum / dim + 1e-6f);
    for (int i = 0; i < dim; i++) {
      s.keys[tail * dim + i] = (ik[i] - mean) * inv * nw[i] + nb[i];
      s.gates[tail * dim + i] = gate[i];
    }
    if (tokens % pool == 0)
      ref::pool(s.keys.data(), s.gates.data(),
                scalar(p + "indexer_compressor_ape.weight").data(),
                s.pooled.data() + (pools - 1) * dim, pool, dim);
    Vec scores(pools);
    for (int b = 0; b < pools; b++) {
      float score = 0;
      for (int h = 0; h < m_.index_heads; h++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
          dot += iq[h * dim + d] * s.pooled[b * dim + d];
        score += iw[h] * std::max(dot / std::sqrt(float(dim)), 0.f);
      }
      scores[b] = score / std::sqrt(float(m_.index_heads));
    }
    std::vector<int> order(pools);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });
    int selected = std::min(int(m_.index_top_k / pool), pools);
    order.resize(selected);
    std::sort(order.begin(), order.end());
    std::vector<int> ids;
    for (int b : order)
      for (int i = 0; i < pool; i++)
        ids.push_back(b * pool + i);
    for (int i = pools * pool; i < tokens; i++)
      ids.push_back(i);
    auto aq = mat(p + "attn_k_b.weight", q, m_.attention_heads);
    Vec av(aq.size());
    int head_dim = artifact_.at(p + "attn_k_b.weight").tensor->shape[0];
    compute_.mla(layer, row, position_, capacity_, aq, ids, av,
                 m_.attention_heads, 1 / std::sqrt(float(head_dim)));
    auto value = mat(p + "attn_v_b.weight", av, m_.attention_heads);
    return mat(p + "attn_output.weight", value);
  }
  Vec moe(const std::string &p, const Vec &x,
          const strata::core::LayerDescriptor &desc) {
    auto logits = mat(p + "ffn_gate_inp.weight", x);
    std::vector<int> ids(m_.top_k);
    Vec routing(m_.top_k);
    ref::router(logits.data(), scalar(p + "exp_probs_b.bias").data(),
                m_.experts, m_.top_k, m_.expert_scale, ids.data(),
                routing.data());
    const auto &G = artifact_.at(p + "ffn_gate_exps.weight");
    const auto &U = artifact_.at(p + "ffn_up_exps.weight");
    const auto &D = artifact_.at(p + "ffn_down_exps.weight");
    cpu::NativeFmt format;
    std::string error;
    if (G.tensor->type != U.tensor->type ||
        !cpu::native_fmt(G.tensor->type, D.tensor->type, m_.hidden,
                         desc.intermediate, format, error))
      throw std::runtime_error("Vulkan CPU expert geometry unsupported: " +
                               error);
    format.swiglu_limit = desc.swiglu_limit;
    format.canon = (G.tensor->type == 10 || G.tensor->type == 11) &&
                   (D.tensor->type == 10 || D.tensor->type == 11);
    std::vector<uint8_t> act(format.act_bytes);
    cpu::native_quant_act(format, x.data(), act.data());
    Vec results(m_.top_k * m_.hidden), sum(m_.hidden);
    std::vector<cpu::ExpertJobMulti> jobs(m_.top_k);
    for (int i = 0; i < m_.top_k; i++) {
      auto &j = jobs[i];
      j.expert_id = ids[i];
      j.nt = 1;
      j.blob = G.data() + size_t(ids[i]) * G.bytes / m_.experts;
      j.native_up = U.data() + size_t(ids[i]) * U.bytes / m_.experts;
      j.native_down = D.data() + size_t(ids[i]) * D.bytes / m_.experts;
      j.nact[0] = act.data();
      j.out[0] = results.data() + i * m_.hidden;
    }
    auto routed = [&] {
      const auto started = Clock::now();
      if (phased_experts_) {
        pool_.run_split_multi_native(format, jobs.data(), jobs.size());
        pool_.reduce_routed(results.data(), routing.data(), sum.data(), 1,
                            m_.top_k, m_.hidden, format.canon);
      } else {
        pool_.run_layer_native(format, jobs.data(), jobs.size(), results.data(),
                               routing.data(), sum.data(), 1, m_.top_k);
      }
      cpu_expert_ms_ +=
          std::chrono::duration<double, std::milli>(Clock::now() - started)
              .count();
    };
    Vec shared;
    if (overlap_experts_) {
      // Only the main thread owns Vulkan. The worker uses layer-local CPU
      // storage, which stays alive until get(), also during exception unwind.
      auto pending = executor_ ? executor_->submit(routed)
                               : std::async(std::launch::async, routed);
      try {
        shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
      } catch (...) {
        // Persistent tasks need an explicit join on exceptional GPU paths too.
        pending.wait();
        throw;
      }
      pending.get();
    } else {
      routed();
      shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
    }
    for (size_t i = 0; i < sum.size(); i++)
      sum[i] += shared[i];
    return sum;
  }

public:
  Decoder(const std::string &model, int device, int threads, int capacity,
          size_t gpu_budget)
      : artifact_(model), m_(strata::vulkan::checked_model(artifact_)),
        projections_(device, gpu_budget - 1536ULL * 1024 * 1024),
        compute_(projections_.device_name(), 1024ULL * 1024 * 1024),
        pool_(threads, true, true), capacity_(capacity) {
    if (m_.architecture != "glm5next" || m_.streams != 4 || m_.rope_dim != 0 ||
        m_.linear_dim != 128 || m_.top_k > 16 || m_.experts != 144)
      throw std::invalid_argument(
          "Vulkan initial runner requires GLM5next NoPE REAP50, four streams, "
          "KDA dimension 128 and <=16 routes");
    states_.resize(m_.layers.size());
    for (size_t l = 0; l < states_.size(); l++) {
      if (m_.layers[l].mixer == strata::core::MixerKind::Kda)
        for (auto &v : states_[l].conv)
          v.resize(m_.linear_heads * m_.linear_dim * (m_.conv_kernel - 1));
      else {
        states_[l].keys.resize(m_.index_pool * m_.index_dim);
        states_[l].gates.resize(m_.index_pool * m_.index_dim);
        states_[l].pooled.resize(size_t(capacity / m_.index_pool) *
                                 m_.index_dim);
      }
    }
    const char *fused = std::getenv("STRATA_VULKAN_FUSED_KDA");
    fused_kda_ = fused && std::string(fused) == "1";
    const char *placement = std::getenv("STRATA_VULKAN_CPU_PLACEMENT");
    if (placement && std::string(placement) != "split" &&
        std::string(placement) != "default")
      throw std::invalid_argument(
          "Vulkan CPU placement must be split or default");
    if (placement && std::string(placement) == "split") {
      const auto topology =
          cpu::detect_cpu_topology(true, cpu::PoolAffinity::All);
      if (topology.host_core < 0 ||
          topology.worker_cores.size() <= size_t(pool_.workers()))
        throw std::invalid_argument(
            "Vulkan split placement needs a spare physical core");
      executor_ =
          std::make_unique<strata::vulkan::CpuExecutor>(topology.host_core);
      const int gpu_cpu = topology.worker_cores.back();
      gpu_affinity_ = cpu::pin_current_thread(gpu_cpu);
      std::cerr << "VULKAN_CPU_PLACEMENT expert_host=" << topology.host_core
                << " gpu_host=" << gpu_cpu << " persistent=1\n";
    }
    const char *schedule = std::getenv("STRATA_VULKAN_EXPERT_SCHEDULE");
    if (schedule && std::string(schedule) != "phased" &&
        std::string(schedule) != "flow")
      throw std::invalid_argument(
          "Vulkan expert schedule must be phased or flow");
    phased_experts_ = schedule && std::string(schedule) == "phased";
    const char *overlap = std::getenv("STRATA_VULKAN_OVERLAP_EXPERTS");
    overlap_experts_ = overlap && std::string(overlap) == "1";
    const char *mhc = std::getenv("STRATA_VULKAN_CPU_MHC");
    cpu_mhc_ = mhc && std::string(mhc) == "1";
    const char *joined = std::getenv("STRATA_VULKAN_JOINED_PROJECTIONS");
    joined_projections_ = joined && std::string(joined) == "1";
    std::cerr
        << "VULKAN_MODEL layers=" << m_.layers.size() << " hidden=" << m_.hidden
        << " experts=" << m_.experts << " cpu_workers=" << pool_.workers()
        << " expert_cache_slots=0 prefill=sequential speculative=unsupported\n";
  }
  ~Decoder() {
    executor_.reset();
    cpu::restore_thread_affinity(gpu_affinity_);
  }
  int vocabulary() const { return m_.vocab; }
  int position() const { return position_; }
  Vec step(int token) {
    if (token < 0 || token >= m_.vocab || position_ >= capacity_)
      throw std::out_of_range("Vulkan token/context out of bounds");
    trace("token_begin", position_);
    strata::vulkan::set_operation_context(position_, -1);
    const auto &embedding = artifact_.at("token_embd.weight");
    Vec e(m_.hidden);
    const uint8_t *src =
        embedding.data() + size_t(token) * embedding.bytes / m_.vocab;
    auto *traits = ggml_get_type_traits(ggml_type(embedding.tensor->type));
    if (embedding.tensor->type == 0)
      std::memcpy(e.data(), src, e.size() * 4);
    else if (traits && traits->to_float)
      traits->to_float(src, e.data(), e.size());
    else
      throw std::runtime_error("Vulkan embedding type unsupported");
    Vec r(4 * m_.hidden);
    for (int i = 0; i < 4; i++)
      std::copy(e.begin(), e.end(), r.begin() + i * m_.hidden);
    for (size_t l = 0; l < m_.layers.size(); l++) {
      strata::vulkan::set_operation_context(position_, int(l));
      trace("layer_begin", position_, l);
      auto p = "blk." + std::to_string(l) + ".";
      auto [collapsed, c] = read(p, "attn", r);
      auto x = normalize(p + "attn_norm.weight", collapsed);
      Vec y = m_.layers[l].mixer == strata::core::MixerKind::Kda ? kda(p, l, x)
                                                                 : mla(p, l, x);
      write(r, c, y);
      trace("mixer_complete", position_, l);
      auto f = read(p, "ffn", r);
      x = normalize(p + "ffn_norm.weight", f.first);
      auto &desc = m_.layers[l];
      y = desc.ffn == strata::core::FfnKind::Dense
              ? ffn(p, x, "", desc.swiglu_limit)
              : moe(p, x, desc);
      write(r, f.second, y);
      trace("ffn_complete", position_, l);
      if (const char *directory = std::getenv("STRATA_GLM_STATE_TRACE_DIR");
          directory && *directory && position_ == 0) {
        const auto path = std::filesystem::path(directory) /
                          ("layer-" + std::to_string(l) + ".bin");
        std::filesystem::create_directories(path.parent_path());
        if (std::filesystem::exists(path))
          throw std::runtime_error("Vulkan: refusing to overwrite layer trace");
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(r.data()),
                r.size() * sizeof(float));
        if (!f)
          throw std::runtime_error("Vulkan: layer trace write failed");
      }
    }
    Vec collapsed(m_.hidden);
    for (size_t d = 0; d < collapsed.size(); d++)
      collapsed[d] = (r[d] + r[m_.hidden + d] + r[2 * m_.hidden + d] +
                      r[3 * m_.hidden + d]) *
                     .25f;
    auto x = normalize("output_norm.weight", collapsed);
    strata::vulkan::set_operation_context(position_, -1);
    auto logits = mat("output.weight", x);
    position_++;
    trace("token_complete", position_);
    return logits;
  }
  void reset() {
    compute_.reset();
    for (auto &s : states_) {
      for (auto &v : s.conv)
        std::fill(v.begin(), v.end(), 0);
      std::fill(s.keys.begin(), s.keys.end(), 0);
      std::fill(s.gates.begin(), s.gates.end(), 0);
      std::fill(s.pooled.begin(), s.pooled.end(), 0);
    }
    position_ = 0;
  }
  void report() {
    std::cerr << "VULKAN_WORK projection_calls=" << projections_.calls
              << " projection_submissions=" << projections_.submissions
              << " projection_compute_ms=" << projections_.compute_ms
              << " projection_transfer_ms=" << projections_.transfer_ms
              << " affine_cpu_ms=" << projections_.affine_cpu_ms
              << " cpu_expert_ms=" << cpu_expert_ms_
              << " custom_dispatch_ms=" << compute_.dispatch_ms
              << " projection_resident_MiB="
              << projections_.resident_bytes() / 1048576.0
              << " custom_peak_MiB=" << compute_.peak_bytes() / 1048576.0
              << '\n';
  }
};
int main(int argc, char **argv) {
  try {
    std::string model, prompt, output, dump;
    std::vector<int> stops;
    int device = 0, threads = 12, capacity = 4096, generate = 32, trials = 1;
    size_t budget = 15360ULL * 1024 * 1024;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto value = [&] {
        if (++i >= argc)
          throw std::invalid_argument("missing argument: " + a);
        return std::string(argv[i]);
      };
      if (a == "--model")
        model = value();
      else if (a == "--prompt-file")
        prompt = value();
      else if (a == "--output")
        output = value();
      else if (a == "--dump-prefill")
        dump = value();
      else if (a == "--stop-ids") {
        auto ids = value();
        std::replace(ids.begin(), ids.end(), ',', ' ');
        std::istringstream stream(ids);
        int id;
        while (stream >> id) {
          if (id < 0)
            throw std::invalid_argument("Negative stop token");
          stops.push_back(id);
        }
        if (!stream.eof() || stops.empty())
          throw std::invalid_argument("Invalid stop token list");
      } else if (a == "--device")
        device = std::stoi(value());
      else if (a == "--threads")
        threads = std::stoi(value());
      else if (a == "--context")
        capacity = std::stoi(value());
      else if (a == "--generate")
        generate = std::stoi(value());
      else if (a == "--trials")
        trials = std::stoi(value());
      else if (a == "--gpu-budget-mib")
        budget = std::stoull(value()) * 1024 * 1024;
      else
        throw std::invalid_argument("unknown argument: " + a);
    }
    if (model.empty() || prompt.empty() || output.empty() || threads < 1 ||
        threads > 32 || capacity < 2 || capacity > 32768 || generate < 2 ||
        generate > 4096 || trials < 1 || trials > 5 ||
        budget < 2048ULL * 1024 * 1024 || budget > 15360ULL * 1024 * 1024)
      throw std::invalid_argument(
          "usage: strata-glm-vulkan --model MODEL --prompt-file IDS --output "
          "JSON [--generate 32 --trials 1 --context 4096 --threads 12 "
          "--gpu-budget-mib 15360]");
    std::ifstream pf(prompt);
    if (!pf)
      throw std::runtime_error("prompt unavailable");
    std::string text((std::istreambuf_iterator<char>(pf)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    std::vector<int> tokens;
    int token;
    while (input >> token)
      tokens.push_back(token);
    if (!input.eof() || tokens.empty() ||
        tokens.size() + generate > size_t(capacity))
      throw std::invalid_argument("prompt invalid or exceeds context");
    if (std::filesystem::exists(output) ||
        (!dump.empty() && std::filesystem::exists(dump)))
      throw std::invalid_argument("refusing to overwrite Vulkan evidence");
    trace("model_load_begin", 0);
    Decoder decoder(model, device, threads, capacity, budget);
    trace("model_load_complete", 0);
    std::vector<double> prefill_rates, decode_rates;
    std::vector<std::vector<int>> outputs;
    Vec logits;
    if (trials > 1) {
      decoder.reset();
      for (int id : tokens)
        logits = decoder.step(id);
    }
    for (int trial = 0; trial < trials; trial++) {
      decoder.reset();
      trace("prefill_begin", 0);
      auto start = Clock::now();
      for (int id : tokens)
        logits = decoder.step(id);
      double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      prefill_rates.push_back(tokens.size() * 1000 / ms);
      std::cerr << "PREFILL trial=" << trial << " tokens=" << tokens.size()
                << " batch=1 ms=" << ms << " tok_s=" << prefill_rates.back()
                << '\n';
      trace("prefill_complete", tokens.size());
      if (trial == 0 && !dump.empty()) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char *>(logits.data()),
                logits.size() * 4);
        if (!f)
          throw std::runtime_error("prefill dump failed");
      }
      std::vector<int> ids;
      start = Clock::now();
      for (int i = 0; i < generate; i++) {
        int id =
            std::max_element(logits.begin(), logits.end()) - logits.begin();
        ids.push_back(id);
        std::cout << id << '\n' << std::flush;
        if (std::find(stops.begin(), stops.end(), id) != stops.end())
          break;
        if (i + 1 < generate)
          logits = decoder.step(id);
      }
      ms = std::chrono::duration<double, std::milli>(Clock::now() - start)
               .count();
      decode_rates.push_back((ids.size() - 1) * 1000 / ms);
      outputs.push_back(ids);
      std::cerr << "DECODE steps=" << ids.size() - 1 << " ms=" << ms
                << " tok_s=" << decode_rates.back() << '\n';
      decoder.report();
    }
    auto median = [](std::vector<double> x) {
      std::sort(x.begin(), x.end());
      return x[x.size() / 2];
    };
    std::ofstream f(output);
    f << "{\"backend\":\"vulkan\",\"prefill_mode\":\"sequential\",\"expert_"
         "cache_slots\":0,\"prefill_tok_s\":"
      << median(prefill_rates) << ",\"decode_tok_s\":" << median(decode_rates)
      << ",\"output_ids\":[";
    for (size_t t = 0; t < outputs.size(); t++) {
      if (t)
        f << ',';
      f << '[';
      for (size_t i = 0; i < outputs[t].size(); i++) {
        if (i)
          f << ',';
        f << outputs[t][i];
      }
      f << ']';
    }
    f << "]}\n";
    if (!f)
      throw std::runtime_error("result write failed");
    trace("run_complete", decoder.position());
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "VULKAN_ERROR " << e.what() << '\n';
    return 1;
  }
}
