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
#include <set>
#include "strata/kernels/canon_expert.hpp"
#include "strata/host/glm_env.h"
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
  size_t expert_budget_ = 0, expert_resident_ = 0;
  bool cache_ready_ = false;
  uint64_t expert_hits_ = 0, expert_routes_ = 0;
  std::map<int, uint64_t> route_counts_;
  std::set<int> cached_experts_;
  bool mtp_enabled_ = false, prefill_gpu_ = false, verification_ = false;
  int mtp_position_ = 0;
  Vec mtp_hidden_;
  std::vector<Vec> target_hidden_;
  std::map<std::string, Vec> scalars_;
  std::map<std::string, std::vector<Vec>> batch_projections_;
  int batch_slot_ = -1;
  std::vector<Vec> batch_mla_rows_, batch_mla_q_;
  std::vector<std::vector<int>> batch_mla_ids_;
  std::vector<Vec> batch_index_queries_,batch_index_weights_;
  std::vector<int> batch_pool_counts_;
  bool gpu_indexer_ = strata_env_is_one("STRATA_VULKAN_GPU_INDEXER");
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
    if (batch_slot_ >= 0) {
      auto found=batch_projections_.find(name);
      if(found!=batch_projections_.end()) return found->second.at(batch_slot_);
    }
    return projections_.mat(name, artifact_.at(name), x, heads);
  }
  std::vector<Vec> mats(const std::vector<std::string> &names, const Vec &x) {
    if(batch_slot_>=0) {
      std::vector<Vec> result;
      for(const auto &name:names)result.push_back(mat(name,x));
      return result;
    }

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
    if (joined_projections_ && batch_slot_<0)
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
  Vec kda(const std::string &p, int layer, const Vec &x, bool project = true) {
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
      return project ? mat(p + "attn_output.weight", y) : y;
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
    return project ? mat(p + "attn_output.weight", y) : y;
  }
  Vec mla(const std::string &p, int layer, const Vec &x, bool project = true, bool collect = false) {
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
    if(collect && gpu_indexer_) {
      batch_mla_rows_.push_back(row);
      batch_mla_q_.push_back(mat(p+"attn_k_b.weight",q,m_.attention_heads));
      batch_index_queries_.push_back(iq);batch_index_weights_.push_back(iw);batch_pool_counts_.push_back(pools);
      batch_mla_ids_.emplace_back();
      return Vec(size_t(m_.attention_heads)*m_.kv_rank);
    }
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
    if(collect) {
      batch_mla_rows_.push_back(row);batch_mla_q_.push_back(aq);batch_mla_ids_.push_back(ids);
      return av;
    }
    compute_.mla(layer, row, position_, capacity_, aq, ids, av,
                 m_.attention_heads, 1 / std::sqrt(float(head_dim)));
    if (!project) return av;
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
    const int layer = std::stoi(p.substr(4));
    std::vector<cpu::ExpertJobMulti> missing;
    std::vector<int> hit_indices;
    for (int i = 0; i < m_.top_k; ++i) {
      const int key = layer * m_.experts + ids[i];
      if (!cache_ready_ && expert_budget_ && layer<int(m_.layers.size())) route_counts_[key]++;
      if (cache_ready_) expert_routes_++;
      if (cache_ready_ && format.canon && cached_experts_.contains(key)) {
        hit_indices.push_back(i); expert_hits_++;
      } else missing.push_back(jobs[i]);
    }
    const bool mixed = !hit_indices.empty();
    auto gpu_routed = [&] {
      for (int i : hit_indices) {
        const int key = layer * m_.experts + ids[i];
        auto gu = compute_.expert_dot(key, act, m_.hidden, 2*desc.intermediate, G.tensor->type, 0);
        for (int j=0;j<desc.intermediate;++j)
          gu[j]=cpu::canon_swiglu(gu[j],gu[desc.intermediate+j],desc.swiglu_limit);
        std::vector<uint8_t> hidden(format.h_bytes);
        cpu::native_quant_h(format,gu.data(),hidden.data());
        auto output=compute_.expert_dot(key,hidden,desc.intermediate,m_.hidden,D.tensor->type,2*G.bytes/m_.experts);
        std::copy(output.begin(),output.end(),results.begin()+i*m_.hidden);
      }
    };
    auto routed = [&] {
      const auto started = Clock::now();
      if (mixed) {
        if (!missing.empty()) pool_.run_split_multi_native(format, missing.data(), missing.size());
      } else if (phased_experts_) {
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
        gpu_routed();
        shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
      } catch (...) {
        // Persistent tasks need an explicit join on exceptional GPU paths too.
        pending.wait();
        throw;
      }
      pending.get();
    } else {
      routed();
      gpu_routed();
      shared = ffn(p, x, "_shexp", desc.shared_swiglu_limit);
    }
    if (mixed) pool_.reduce_routed(results.data(), routing.data(), sum.data(), 1,
                                    m_.top_k, m_.hidden, format.canon);
    for (size_t i = 0; i < sum.size(); i++)
      sum[i] += shared[i];
    return sum;
  }

public:
  Decoder(const std::string &model, int device, int threads, int capacity,
          size_t gpu_budget, size_t expert_budget = 0, bool mtp_enabled = false, bool prefill_gpu = false)
      : artifact_(model), m_(strata::vulkan::checked_model(artifact_)),
        projections_(device, gpu_budget - expert_budget - 1536ULL * 1024 * 1024),
        compute_(projections_.device_name(), 1024ULL * 1024 * 1024 + expert_budget),
        pool_(threads, true, true), capacity_(capacity), expert_budget_(expert_budget), mtp_enabled_(mtp_enabled), prefill_gpu_(prefill_gpu) {
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
    if(mtp_enabled_) {
      if(m_.draft_layers.size()!=1 || m_.draft_layers[0].mixer!=strata::core::MixerKind::SparseMla)
        throw std::invalid_argument("Vulkan MTP requires exactly one MLA draft block");
      artifact_.shape("blk.45.nextn.enorm.weight",{uint64_t(m_.hidden)});
      artifact_.shape("blk.45.nextn.hnorm.weight",{uint64_t(m_.hidden)});
      artifact_.shape("blk.45.nextn.eh_proj.weight",{uint64_t(2*m_.hidden),uint64_t(m_.hidden)});
      artifact_.shape("blk.45.nextn.shared_head_norm.weight",{uint64_t(m_.hidden)});
      State state;state.keys.resize(m_.index_pool*m_.index_dim);state.gates.resize(state.keys.size());
      state.pooled.resize(size_t(capacity/m_.index_pool)*m_.index_dim);states_.push_back(std::move(state));
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
        << " expert_cache_budget_MiB=" << expert_budget_/1048576
        << " prefill_experts=" << (prefill_gpu_?"gpu":"cpu")
        << " speculative_modes=none,lookup,mtp\n";
  }
  ~Decoder() {
    executor_.reset();
    cpu::restore_thread_affinity(gpu_affinity_);
  }
  std::vector<Vec> batch_product(const std::string &name, const std::vector<Vec> &inputs, int heads=1) {
    if(inputs.empty())throw std::invalid_argument("Empty Vulkan projection batch");
    const auto &tensor=artifact_.at(name);
    const size_t in=tensor.tensor->shape[0],out=tensor.tensor->shape[1],nt=inputs.size();
    Vec packed(in*heads*nt);
    for(size_t t=0;t<nt;++t) {
      if(inputs[t].size()!=in*heads)throw std::invalid_argument("Vulkan batched input width differs");
      for(int h=0;h<heads;++h)std::copy_n(inputs[t].data()+h*in,in,packed.data()+(h*nt+t)*in);
    }
    Vec product;
    const bool mmv=strata_env_is_one("STRATA_VULKAN_BATCH_MMV");
    if(strata_env_is_one("STRATA_VULKAN_BATCH_GEMV") || (mmv && heads>1)) {
      product.resize(out*heads*nt);
      for(size_t t=0;t<nt;++t) {
        auto y=projections_.mat(name,tensor,inputs[t],heads,1);
        for(int h=0;h<heads;++h)std::copy_n(y.data()+h*out,out,product.data()+(h*nt+t)*out);
      }
    } else if(mmv) {
      // GGML's vector path accepts at most eight columns. Larger or per-head
      // batches select a different GEMM reduction and can change greedy IDs.
      product.resize(out*nt);
      for(size_t offset=0;offset<nt;offset+=8) {
        const size_t count=std::min<size_t>(8,nt-offset);
        auto y=projections_.mat(name,tensor,{packed.data()+offset*in,count*in},1,count);
        std::copy(y.begin(),y.end(),product.begin()+offset*out);
      }
    } else product=projections_.mat(name,tensor,packed,heads,nt);
    std::vector<Vec> result(nt,Vec(out*heads));
    for(size_t t=0;t<nt;++t)for(int h=0;h<heads;++h)
      std::copy_n(product.data()+(h*nt+t)*out,out,result[t].data()+h*out);
    batch_projections_[name]=result;return result;
  }
  std::vector<Vec> ffn_batch(const std::string &p,const std::vector<Vec> &x,
                             const std::string &suffix,float limit) {
    auto gate=batch_product(p+"ffn_gate"+suffix+".weight",x);
    auto up=batch_product(p+"ffn_up"+suffix+".weight",x);
    for(size_t t=0;t<x.size();++t)for(size_t i=0;i<gate[t].size();++i)
      gate[t][i]=ref::swiglu(gate[t][i],up[t][i],limit);
    return batch_product(p+"ffn_down"+suffix+".weight",gate);
  }
  std::vector<Vec> moe_batch(const std::string &p,const std::vector<Vec> &x,
                             const strata::core::LayerDescriptor &desc) {
    auto logits=batch_product(p+"ffn_gate_inp.weight",x);
    const size_t nt=x.size();const int top=m_.top_k,H=m_.hidden;
    std::vector<int> ids(nt*top);Vec routing(nt*top),results(nt*top*H);
    const auto &G=artifact_.at(p+"ffn_gate_exps.weight"), &U=artifact_.at(p+"ffn_up_exps.weight"),
               &D=artifact_.at(p+"ffn_down_exps.weight");
    cpu::NativeFmt format;std::string error;
    if(!cpu::native_fmt(G.tensor->type,D.tensor->type,H,desc.intermediate,format,error))throw std::runtime_error(error);
    format.canon=(G.tensor->type==10 || G.tensor->type==11)&&(D.tensor->type==10 || D.tensor->type==11);
    format.swiglu_limit=desc.swiglu_limit;
    std::vector<uint8_t> activations(nt*format.act_bytes);
    std::map<int,std::vector<int>> groups;
    const int layer=std::stoi(p.substr(4));
    for(size_t t=0;t<nt;++t) {
      ref::router(logits[t].data(),scalar(p+"exp_probs_b.bias").data(),m_.experts,top,m_.expert_scale,
                  ids.data()+t*top,routing.data()+t*top);
      cpu::native_quant_act(format,x[t].data(),activations.data()+t*format.act_bytes);
      for(int k=0;k<top;++k) {
        const int id=ids[t*top+k];groups[id].push_back(t*top+k);
        if(expert_budget_ && !cache_ready_)route_counts_[layer*m_.experts+id]++;
      }
    }
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> gpu_groups;
    for(const auto &[id,entries]:groups) {
      const bool cached=cache_ready_ && cached_experts_.contains(layer*m_.experts+id);
      const bool gpu=format.canon && ((prefill_gpu_ && !verification_) || (verification_ && cached));
      if(gpu) {gpu_groups.push_back(id);if(verification_)expert_hits_+=entries.size();continue;}
      for(size_t offset=0;offset<entries.size();offset+=cpu::MAXT) {
        cpu::ExpertJobMulti job;job.expert_id=id;job.nt=std::min<size_t>(cpu::MAXT,entries.size()-offset);
        job.blob=G.data()+id*G.bytes/m_.experts;job.native_up=U.data()+id*U.bytes/m_.experts;
        job.native_down=D.data()+id*D.bytes/m_.experts;
        for(int k=0;k<job.nt;++k) {
          const int entry=entries[offset+k];job.nact[k]=activations.data()+(entry/top)*format.act_bytes;
          job.out[k]=results.data()+size_t(entry)*H;
        }
        jobs.push_back(job);
      }
    }
    if(verification_ && cache_ready_)expert_routes_+=nt*top;
    auto cpu_work=[&] {
      const auto started=Clock::now();
      if(!jobs.empty())pool_.run_split_multi_native(format,jobs.data(),jobs.size());
      cpu_expert_ms_+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    };
    auto gpu_work=[&] {
      for(int id:gpu_groups) {
        const auto &entries=groups.at(id);
        const int cache_key=layer*m_.experts+id;
        const bool cached=cache_ready_ && cached_experts_.contains(cache_key);
        const int key=cached?cache_key:1000000;
        const size_t gb=G.bytes/m_.experts,db=D.bytes/m_.experts;
        if(!cached)compute_.upload_expert(key,{G.data()+id*gb,gb},{U.data()+id*gb,gb},{D.data()+id*db,db});
        for(size_t offset=0;offset<entries.size();offset+=cpu::MAXT) {
          const int count=std::min<size_t>(cpu::MAXT,entries.size()-offset);
          std::vector<uint8_t> acts(count*format.act_bytes),hidden(count*format.h_bytes);
          for(int t=0;t<count;++t)std::memcpy(acts.data()+t*format.act_bytes,
                        activations.data()+(entries[offset+t]/top)*format.act_bytes,format.act_bytes);
          auto gu=compute_.expert_dot(key,acts,H,2*desc.intermediate,G.tensor->type,0,count);
          for(int t=0;t<count;++t) {
            float *row=gu.data()+size_t(t)*2*desc.intermediate;
            for(int i=0;i<desc.intermediate;++i)row[i]=cpu::canon_swiglu(row[i],row[desc.intermediate+i],desc.swiglu_limit);
            cpu::native_quant_h(format,row,hidden.data()+t*format.h_bytes);
          }
          auto output=compute_.expert_dot(key,hidden,desc.intermediate,H,D.tensor->type,2*gb,count);
          for(int t=0;t<count;++t)std::copy_n(output.data()+size_t(t)*H,H,
                                           results.data()+size_t(entries[offset+t])*H);
        }
        if(!cached)compute_.release_expert(key);
      }
    };
    std::vector<Vec> shared;
    if(overlap_experts_ && !jobs.empty()) {
      auto pending=executor_?executor_->submit(cpu_work):std::async(std::launch::async,cpu_work);
      try {gpu_work();shared=ffn_batch(p,x,"_shexp",desc.shared_swiglu_limit);}
      catch(...){pending.wait();throw;}
      pending.get();
    } else {cpu_work();gpu_work();shared=ffn_batch(p,x,"_shexp",desc.shared_swiglu_limit);}
    std::vector<Vec> sum(nt,Vec(H));
    for(size_t t=0;t<nt;++t) {
      pool_.reduce_routed(results.data()+t*top*H,routing.data()+t*top,sum[t].data(),1,top,H,format.canon);
      for(int i=0;i<H;++i)sum[t][i]+=shared[t][i];
    }
    return sum;
  }
  Vec prefill(std::span<const int> tokens,int chunk, std::vector<Vec> *all_logits = nullptr) {
    if(chunk==1) {Vec logits;for(int id:tokens){logits=step(id);if(all_logits)all_logits->push_back(logits);}return logits;}
    if(chunk<1 || chunk>1024 || tokens.empty() || size_t(position_)+tokens.size()>size_t(capacity_))
      throw std::invalid_argument("Vulkan prefill geometry invalid");
    Vec logits;
    try {
      for(size_t offset=0;offset<tokens.size();offset+=chunk) {
        const int nt=std::min<size_t>(chunk,tokens.size()-offset),base=position_;
        std::vector<Vec> residual(nt,Vec(4*m_.hidden));
        const auto &embedding=artifact_.at("token_embd.weight");
        auto *traits=ggml_get_type_traits(ggml_type(embedding.tensor->type));
        for(int t=0;t<nt;++t) {
          const int id=tokens[offset+t];if(id<0 || id>=m_.vocab)throw std::out_of_range("Invalid Vulkan prefill token");
          Vec e(m_.hidden);const auto *src=embedding.data()+size_t(id)*embedding.bytes/m_.vocab;
          if(embedding.tensor->type==0)std::memcpy(e.data(),src,e.size()*4);
          else if(traits && traits->to_float)traits->to_float(src,e.data(),e.size());
          else throw std::runtime_error("Vulkan prefill embedding type unsupported");
          for(int h=0;h<4;++h)std::copy(e.begin(),e.end(),residual[t].begin()+h*m_.hidden);
        }
        for(size_t layer=0;layer<m_.layers.size();++layer) {
          const std::string p="blk."+std::to_string(layer)+".";
          const auto &desc=m_.layers[layer];std::vector<Vec> x(nt),coefficients(nt);
          strata::vulkan::set_operation_context(base,int(layer));
          trace("prefill_layer_begin",base,layer);
          auto hc_inputs=residual;
          for(auto &v:hc_inputs)v=norm(v,{},m_.rms_epsilon);
          batch_product(p+"hc_attn_fn.weight",hc_inputs);
          for(int t=0;t<nt;++t) {
            batch_slot_=t;auto readout=read(p,"attn",residual[t]);coefficients[t]=std::move(readout.second);
            x[t]=normalize(p+"attn_norm.weight",readout.first);
          }
          batch_slot_=-1;
          std::vector<Vec> mixed(nt);
          if(desc.mixer==strata::core::MixerKind::Kda) {
            for(const char *name:{"attn_q","attn_k","attn_v","ssm_f_a","ssm_g_a","ssm_beta"})
              batch_product(p+name+".weight",x);
            batch_product(p+"ssm_f_b.weight",batch_projections_.at(p+"ssm_f_a.weight"));
            batch_product(p+"ssm_g_b.weight",batch_projections_.at(p+"ssm_g_a.weight"));
            for(int t=0;t<nt;++t) {batch_slot_=t;position_=base+t;mixed[t]=kda(p,layer,x[t],false);}
          } else {
            for(const char *name:{"attn_q_a","attn_kv_a_mqa","indexer.proj","indexer.attn_k","indexer_compressor_gate"})
              batch_product(p+name+".weight",x);
            auto qr=batch_projections_.at(p+"attn_q_a.weight");
            for(auto &v:qr)v=normalize(p+"attn_q_a_norm.weight",v);
            auto q=batch_product(p+"attn_q_b.weight",qr);
            batch_product(p+"indexer.attn_q_b.weight",qr);
            batch_product(p+"attn_k_b.weight",q,m_.attention_heads);
            batch_mla_rows_.clear();batch_mla_q_.clear();batch_mla_ids_.clear();
            batch_index_queries_.clear();batch_index_weights_.clear();batch_pool_counts_.clear();
            for(int t=0;t<nt;++t) {batch_slot_=t;position_=base+t;mla(p,layer,x[t],false,true);}
            if(gpu_indexer_) {
              Vec iq,iw;
              for(const auto &v:batch_index_queries_)iq.insert(iq.end(),v.begin(),v.end());
              for(const auto &v:batch_index_weights_)iw.insert(iw.end(),v.begin(),v.end());
              const int max_pools=(base+nt)/m_.index_pool;
              auto scores=compute_.index_scores({states_[layer].pooled.data(),size_t(max_pools)*m_.index_dim},
                                               iq,iw,batch_pool_counts_,m_.index_heads,m_.index_dim);
              for(int t=0;t<nt;++t) {
                const int pools=batch_pool_counts_[t];std::vector<int> order(pools);std::iota(order.begin(),order.end(),0);
                std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[t*max_pools+a]>scores[t*max_pools+b];});
                order.resize(std::min<int>(m_.index_top_k/m_.index_pool,pools));std::sort(order.begin(),order.end());
                auto &ids=batch_mla_ids_[t];
                for(int pool:order)for(int i=0;i<m_.index_pool;++i)ids.push_back(pool*m_.index_pool+i);
                for(int i=pools*m_.index_pool;i<=base+t;++i)ids.push_back(i);
              }
            }
            Vec rows,queries;
            for(auto &v:batch_mla_rows_)rows.insert(rows.end(),v.begin(),v.end());
            for(auto &v:batch_mla_q_)queries.insert(queries.end(),v.begin(),v.end());
            const auto head_dim=artifact_.at(p+"attn_k_b.weight").tensor->shape[0];
            auto av=compute_.mla_batch(layer,rows,base,capacity_,queries,batch_mla_ids_,m_.attention_heads,
                                       1/std::sqrt(float(head_dim)));
            const size_t width=av.size()/nt;
            for(int t=0;t<nt;++t)mixed[t]=Vec(av.begin()+t*width,av.begin()+(t+1)*width);
            batch_slot_=-1;mixed=batch_product(p+"attn_v_b.weight",mixed,m_.attention_heads);
          }
          batch_slot_=-1;mixed=batch_product(p+"attn_output.weight",mixed);
          for(int t=0;t<nt;++t)write(residual[t],coefficients[t],mixed[t]);
          hc_inputs=residual;
          for(auto &v:hc_inputs)v=norm(v,{},m_.rms_epsilon);
          batch_product(p+"hc_ffn_fn.weight",hc_inputs);
          for(int t=0;t<nt;++t) {
            batch_slot_=t;auto f=read(p,"ffn",residual[t]);coefficients[t]=std::move(f.second);
            x[t]=normalize(p+"ffn_norm.weight",f.first);
          }
          batch_slot_=-1;
          auto ffn=desc.ffn==strata::core::FfnKind::Dense?ffn_batch(p,x,"",desc.swiglu_limit):moe_batch(p,x,desc);
          for(int t=0;t<nt;++t)write(residual[t],coefficients[t],ffn[t]);
          if(base==0)save_layer_trace(int(layer),residual[0]);
          batch_projections_.clear();projections_.release_plans();
          trace("prefill_layer_complete",base,layer);
        }
        std::vector<Vec> normalized;
        const int first=(all_logits || mtp_enabled_)?0:nt-1;
        if(mtp_enabled_)target_hidden_.resize(base+nt);
        for(int t=first;t<nt;++t) {
          Vec collapsed(m_.hidden);const auto &r=residual[t];
          for(int d=0;d<m_.hidden;++d)collapsed[d]=(r[d]+r[m_.hidden+d]+r[2*m_.hidden+d]+r[3*m_.hidden+d])*.25f;
          auto x=normalize("output_norm.weight",collapsed);
          if(mtp_enabled_)target_hidden_[base+t]=x;
          normalized.push_back(std::move(x));
        }
        if(all_logits) {
          auto heads=batch_product("output.weight",normalized);
          logits=heads.back();all_logits->insert(all_logits->end(),heads.begin(),heads.end());
          batch_projections_.clear();projections_.release_plans();
        } else logits=mat("output.weight",normalized.back());
        position_=base+nt;
      }
      batch_mla_rows_.clear();batch_mla_q_.clear();batch_mla_ids_.clear();
      compute_.release_working_buffers();
      return logits;
    } catch(...) {batch_slot_=-1;batch_projections_.clear();throw;}
  }
  void prepare_decode_cache() {
    if (!expert_budget_) return;
    const auto started=Clock::now();
    std::vector<std::pair<uint64_t,int>> ranked;
    for (auto [key,count] : route_counts_) ranked.emplace_back(count,key);
    std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    for (auto [count,key] : ranked) {
      const int layer=key/m_.experts,id=key%m_.experts;
      const std::string p="blk."+std::to_string(layer)+".";
      const auto &G=artifact_.at(p+"ffn_gate_exps.weight"), &U=artifact_.at(p+"ffn_up_exps.weight"),
                 &D=artifact_.at(p+"ffn_down_exps.weight");
      if((G.tensor->type!=10 && G.tensor->type!=11) || (D.tensor->type!=10 && D.tensor->type!=11))continue;
      const size_t gb=G.bytes/m_.experts, db=D.bytes/m_.experts,bytes=gb*2+db;
      // Vulkan memory requirements can add alignment beyond the packed bytes.
      const size_t charged=(bytes+4095)&~size_t(4095);
      if(charged>expert_budget_-expert_resident_)continue;
      const size_t allocated=compute_.upload_expert(key,{G.data()+id*gb,gb},{U.data()+id*gb,gb},
                                                    {D.data()+id*db,db},expert_budget_-expert_resident_);
      cached_experts_.insert(key);expert_resident_+=allocated;
    }
    cache_ready_=true;
    std::cerr << "VULKAN_CACHE slots=" << cached_experts_.size()
              << " resident_MiB=" << expert_resident_/1048576.0
              << " setup_ms=" << std::chrono::duration<double,std::milli>(Clock::now()-started).count() << '\n';
  }
  struct Snapshot {
    strata::vulkan::Compute::Snapshot gpu;
    std::vector<State> states;
    size_t hidden_count;
    int position;
  };
  Snapshot checkpoint() {return {compute_.checkpoint(),states_,target_hidden_.size(),position_};}
  void restore(const Snapshot &snapshot) {
    compute_.restore(snapshot.gpu);states_=snapshot.states;
    target_hidden_.resize(snapshot.hidden_count);position_=snapshot.position;
  }
  Vec mtp_step(int token,const Vec &previous,bool head=true) {
    if(!mtp_enabled_ || token<0 || token>=m_.vocab || previous.size()!=size_t(m_.hidden) || mtp_position_>=capacity_)
      throw std::invalid_argument("Vulkan MTP input invalid");
    struct RestorePosition {int &p;int old;~RestorePosition(){p=old;}} restore_position{position_,position_};
    position_=mtp_position_;
    const std::string p="blk.45.";const auto &embedding=artifact_.at("token_embd.weight");
    Vec e(m_.hidden);const auto *src=embedding.data()+size_t(token)*embedding.bytes/m_.vocab;
    if(embedding.tensor->type==0)std::memcpy(e.data(),src,e.size()*4);
    else ggml_get_type_traits(ggml_type(embedding.tensor->type))->to_float(src,e.data(),e.size());
    Vec joined=normalize(p+"nextn.enorm.weight",e);
    auto prev=normalize(p+"nextn.hnorm.weight",previous);joined.insert(joined.end(),prev.begin(),prev.end());
    auto cur=mat(p+"nextn.eh_proj.weight",joined);
    auto x=normalize(p+"attn_norm.weight",cur),y=mla(p,45,x);
    for(size_t i=0;i<cur.size();++i)cur[i]+=y[i];
    x=normalize(p+"ffn_norm.weight",cur);y=moe(p,x,m_.draft_layers[0]);
    for(size_t i=0;i<cur.size();++i)cur[i]+=y[i];
    mtp_hidden_=normalize(p+"nextn.shared_head_norm.weight",cur);++mtp_position_;
    return head?mat("output.weight",mtp_hidden_):Vec{};
  }
  void prime_mtp(std::span<const int> tokens,int start=0) {
    if(!mtp_enabled_)return;
    const auto begin=Clock::now();Vec zero(m_.hidden);
    for(size_t i=0;i<tokens.size();++i) {
      const int previous=start+int(i)-1;
      mtp_step(tokens[i],previous<0?zero:target_hidden_.at(previous),false);
    }
    std::cerr << "VULKAN_MTP_SYNC tokens=" << tokens.size() << " ms="
              << std::chrono::duration<double,std::milli>(Clock::now()-begin).count() << '\n';
  }
  static std::vector<int> lookup_draft(const std::vector<int> &history,int anchor,int depth) {
    auto context=history;context.push_back(anchor);
    for(int n=std::min<int>(4,context.size());n>=1;--n) {
      for(int i=int(history.size())-n;i>=0;--i) {
        if(std::equal(context.end()-n,context.end(),history.begin()+i)) {
          std::vector<int> result;
          for(int k=i+n;k<int(history.size()) && int(result.size())<depth;++k)result.push_back(history[k]);
          if(!result.empty())return result;
        }
      }
    }
    return {};
  }
  std::vector<int> speculative(Vec &logits,const std::vector<int> &prompt,int maximum,
                                const std::vector<int> &stops,const std::string &source,
                                int depth,bool batch_verify) {
    auto argmax=[](const Vec &v){return int(std::max_element(v.begin(),v.end())-v.begin());};
    auto stop=[&](int token){return std::find(stops.begin(),stops.end(),token)!=stops.end();};
    std::vector<int> history=prompt,output;
    uint64_t proposed=0,accepted=0,rounds=0,rejected=0,target_steps=0;
    double draft_ms=0,verify_ms=0,rollback_ms=0;
    while(int(output.size())<maximum) {
      const int anchor=argmax(logits),start=position_,remaining=maximum-int(output.size());
      if(stop(anchor) || remaining==1) {output.push_back(anchor);std::cout << anchor << '\n' << std::flush;break;}
      ++rounds;std::vector<int> drafts;
      State draft_state;Vec draft_hidden;int draft_position=mtp_position_;
      auto began=Clock::now();
      if(source=="mtp") {
        draft_state=states_.at(45);draft_hidden=mtp_hidden_;
        Vec predicted=mtp_step(anchor,target_hidden_.back());
        for(int i=0;i<std::min(depth,remaining-1);++i) {
          const int token=argmax(predicted);drafts.push_back(token);
          if(stop(token) || i+1>=std::min(depth,remaining-1))break;
          predicted=mtp_step(token,mtp_hidden_);
        }
      } else drafts=lookup_draft(history,anchor,std::min(depth,remaining-1));
      draft_ms+=std::chrono::duration<double,std::milli>(Clock::now()-began).count();
      proposed+=drafts.size();std::vector<int> window{anchor},committed;
      window.insert(window.end(),drafts.begin(),drafts.end());
      began=Clock::now();
      if(batch_verify && !drafts.empty()) {
        auto saved=checkpoint();std::vector<Vec> verified;
        verification_=true;
        try {prefill(window,window.size(),&verified);} catch(...) {verification_=false;throw;}
        verification_=false;target_steps+=window.size();
        committed.push_back(anchor);
        for(size_t i=0;i<drafts.size();++i) {
          if(argmax(verified[i])!=drafts[i]) {++rejected;break;}
          committed.push_back(drafts[i]);++accepted;
          if(stop(drafts[i]))break;
        }
        verify_ms+=std::chrono::duration<double,std::milli>(Clock::now()-began).count();
        if(committed.size()!=window.size()) {
          began=Clock::now();restore(saved);logits=prefill(committed,1);target_steps+=committed.size();
          rollback_ms+=std::chrono::duration<double,std::milli>(Clock::now()-began).count();
        } else logits=verified.back();
      } else {
        committed.push_back(anchor);logits=step(anchor);++target_steps;
        for(int token:drafts) {
          if(argmax(logits)!=token){++rejected;break;}
          committed.push_back(token);++accepted;
          if(stop(token))break;
          logits=step(token);++target_steps;
        }
        verify_ms+=std::chrono::duration<double,std::milli>(Clock::now()-began).count();
      }
      for(int token:committed) {output.push_back(token);history.push_back(token);std::cout << token << '\n' << std::flush;}
      if(source=="mtp" && !stop(output.back()) && int(output.size())<maximum) {
        states_[45]=std::move(draft_state);mtp_hidden_=std::move(draft_hidden);mtp_position_=draft_position;
        compute_.rewind_mla(45,draft_position);prime_mtp(committed,start);
      }
      if(stop(output.back()))break;
    }
    std::cerr << "VULKAN_SPEC source=" << source << " rounds=" << rounds << " proposed=" << proposed
              << " accepted=" << accepted << " rejected_windows=" << rejected << " target_steps=" << target_steps
              << " draft_ms=" << draft_ms << " verify_ms=" << verify_ms << " rollback_ms=" << rollback_ms
              << " verify_batch=" << batch_verify << '\n';
    return output;
  }
  size_t cache_slots() const { return cached_experts_.size(); }
  int vocabulary() const { return m_.vocab; }
  int position() const { return position_; }
  void save_layer_trace(int layer,const Vec &r) {
    const char *directory=std::getenv("STRATA_GLM_STATE_TRACE_DIR");
    if(!directory || !*directory)return;
    const auto path=std::filesystem::path(directory)/("layer-"+std::to_string(layer)+".bin");
    std::filesystem::create_directories(path.parent_path());
    if(std::filesystem::exists(path))throw std::runtime_error("Vulkan: refusing to overwrite layer trace");
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(r.data()),r.size()*4);
    if(!file)throw std::runtime_error("Vulkan layer trace write failed");
  }
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
    if(mtp_enabled_) {target_hidden_.resize(position_+1);target_hidden_[position_]=x;}
    strata::vulkan::set_operation_context(position_, -1);
    auto logits = mat("output.weight", x);
    position_++;
    trace("token_complete", position_);
    return logits;
  }
  void reset() {
    mtp_position_=0;mtp_hidden_.clear();target_hidden_.clear();
    compute_.clear_experts();
    cached_experts_.clear();route_counts_.clear();expert_resident_=0;cache_ready_=false;
    expert_hits_=0;expert_routes_=0;
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
  std::array<double, 7> work_snapshot() const {
    return {projections_.compute_ms, projections_.transfer_ms, cpu_expert_ms_,
            compute_.dispatch_ms, double(projections_.calls), double(projections_.submissions),compute_.transfer_ms};
  }
  void phase_report(const char *phase, int trial, const std::array<double,7> &before) {
    const auto after = work_snapshot();
    std::cerr << "VULKAN_PHASE phase=" << phase << " trial=" << trial
              << " projection_compute_ms=" << after[0]-before[0]
              << " projection_transfer_ms=" << after[1]-before[1]
              << " cpu_expert_ms=" << after[2]-before[2]
              << " custom_dispatch_ms=" << after[3]-before[3]
              << " projection_calls=" << after[4]-before[4]
              << " projection_submissions=" << after[5]-before[5]
              << " custom_transfer_ms=" << after[6]-before[6] << '\n';
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
              << " expert_cache_hits=" << expert_hits_ << " expert_routes=" << expert_routes_
              << " custom_peak_MiB=" << compute_.peak_bytes() / 1048576.0
              << '\n';
  }
};
int main(int argc, char **argv) {
  try {
    std::string model, prompt, output, dump, speculative = "none", prefill_experts = "cpu";
    int draft_depth = 1;
    bool verify_batch = false;
    std::vector<int> stops;
    int device = 0, threads = 12, capacity = 4096, generate = 32, trials = 1, prefill_batch = 1;
    size_t budget = 15360ULL * 1024 * 1024, expert_budget = 0;
    auto mib=[](const std::string &text) {
      size_t consumed=0;const auto number=std::stoull(text,&consumed);
      if(consumed!=text.size() || number>15360)throw std::invalid_argument("MiB value must be 0..15360");
      return size_t(number)*1024*1024;
    };
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
      else if (a == "--prefill-experts")
        prefill_experts = value();
      else if (a == "--speculative")
        speculative = value();
      else if (a == "--draft-depth")
        draft_depth = std::stoi(value());
      else if (a == "--verify-batch")
        verify_batch = true;
      else if (a == "--prefill-batch")
        prefill_batch = std::stoi(value());
      else if (a == "--decode-cache-mib")
        expert_budget = mib(value());
      else if (a == "--gpu-budget-mib")
        budget = mib(value());
      else
        throw std::invalid_argument("unknown argument: " + a);
    }
    if ((prefill_experts!="cpu" && prefill_experts!="gpu") || (prefill_experts=="gpu" && prefill_batch==1) ||
        (speculative!="none" && speculative!="lookup" && speculative!="mtp") || draft_depth<1 || draft_depth>3 ||
        prefill_batch<1 || prefill_batch>1024 || model.empty() || prompt.empty() || output.empty() || threads < 1 ||
        threads > 32 || capacity < 2 || capacity > 32768 || generate < 2 ||
        generate > 4096 || trials < 1 || trials > 5 ||
        budget < 2048ULL * 1024 * 1024 || budget > 15360ULL * 1024 * 1024 ||
        expert_budget > 8192ULL * 1024 * 1024 || expert_budget > budget - 2048ULL*1024*1024)
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
    Decoder decoder(model, device, threads, capacity, budget, expert_budget, speculative=="mtp", prefill_experts=="gpu");
    trace("model_load_complete", 0);
    std::vector<double> prefill_rates, decode_rates;
    std::vector<std::vector<int>> outputs;
    Vec logits;
    if (trials > 1) {
      decoder.reset();
      logits = decoder.prefill(tokens,prefill_batch);
    }
    for (int trial = 0; trial < trials; trial++) {
      decoder.reset();
      trace("prefill_begin", 0);
      auto phase_start = decoder.work_snapshot();
      auto start = Clock::now();
      logits = decoder.prefill(tokens,prefill_batch);
      double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      prefill_rates.push_back(tokens.size() * 1000 / ms);
      std::cerr << "PREFILL trial=" << trial << " tokens=" << tokens.size()
                << " batch=" << prefill_batch << " ms=" << ms << " tok_s=" << prefill_rates.back()
                << '\n';
      decoder.phase_report("prefill", trial, phase_start);
      trace("prefill_complete", tokens.size());
      if (trial == 0 && !dump.empty()) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char *>(logits.data()),
                logits.size() * 4);
        if (!f)
          throw std::runtime_error("prefill dump failed");
      }
      decoder.prime_mtp(tokens);
      decoder.prepare_decode_cache();
      std::vector<int> ids;
      phase_start = decoder.work_snapshot();
      start = Clock::now();
      if(speculative!="none")
        ids=decoder.speculative(logits,tokens,generate,stops,speculative,draft_depth,verify_batch);
      else {
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
      }
      ms = std::chrono::duration<double, std::milli>(Clock::now() - start)
               .count();
      decode_rates.push_back((ids.size() - 1) * 1000 / ms);
      outputs.push_back(ids);
      if(speculative=="none")
        std::cerr << "DECODE steps=" << ids.size() - 1 << " ms=" << ms << " tok_s=" << decode_rates.back() << '\n';
      else std::cerr << "SPECULATIVE source=" << speculative << " generated=" << ids.size()
                     << " ms=" << ms << " tok_s=" << decode_rates.back() << '\n';
      decoder.phase_report("decode", trial, phase_start);
      decoder.report();
    }
    auto median = [](std::vector<double> x) {
      std::sort(x.begin(), x.end());
      return x[x.size() / 2];
    };
    std::ofstream f(output);
    f << "{\"backend\":\"vulkan\",\"prefill_mode\":\"" << (prefill_batch==1?"sequential":"batched") << "\",\"expert_"
         "cache_slots\":" << decoder.cache_slots() << ",\"prefill_tok_s\":"
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
