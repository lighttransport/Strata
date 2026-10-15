#include "fast/engine.hpp"
#include "fast/expert_batch.hpp"
#include "strata/kernels/glm_reference.hpp"
#include "strata/host/glm_env.h"
#include "strata/artifact/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <ggml.h>
#include <iostream>
#include <stdexcept>
namespace kcpu = strata::kernels::cpu;
namespace strata::vulkan::fast {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
void require(bool v, const char *message) {
  if (!v) throw std::invalid_argument(message);
}
// Host mailbox per MoE layer and token, in floats: x[4096] | ids[16] | weights[16] | pad | out[4096].
constexpr size_t MB_IDS = 4096, MB_WEIGHTS = 4112, MB_OUT = 4160;
// Per-token strides and the per-layer hit union (count, 32 slots, 32x4 weights).
constexpr uint32_t QK_STACK = 2 * 8192 + 256 + 64, IDS_STRIDE = 2176, UNION_STRIDE = 192;
// Fixed sizes of the GLM-5.3f REAP50 geometry validated in the constructor.
constexpr uint32_t H = 4096, KDA_N = 8192, MLA_INIT = 2208, Q_RANK = 1536, KV_RANK = 512, IDX_PROJ = 2048,
                   IDX_KEY = 2080;
} // namespace

struct Engine::Layer {
  int index = 0, moe_index = -1;
  bool kda = false, moe = false;
  bool resident_eligible = false;
  Matrix hc[2];
  size_t hc_par[2]{}, norm[2]{};
  // KDA
  Matrix qk, v, fb, gb, out;
  std::unique_ptr<Buffer> kda_params, state, history;
  // MLA
  Matrix init, comp_gate, qb, iqb, kb, vb;
  size_t q_norm = 0, kv_norm = 0, idx_par = 0;
  std::unique_ptr<Buffer> cache, ring, pooled, ring_history;
  float mla_scale = 1;
  // MTP draft block
  Matrix eh_proj;
  size_t enorm = 0, hnorm = 0, head_norm = 0;
  // FFN
  Matrix gu, down, router;
  size_t probs_bias = 0;
  float limit = 0;
  const uint8_t *gate = nullptr, *up = nullptr, *down_experts = nullptr;
  size_t gate_stride = 0, down_stride = 0;
  kcpu::NativeFmt fmt;
};

Engine::Engine(core::ModelArtifact &artifact, const core::ModelDescriptor &m, const EngineOptions &o)
    : artifact_(artifact), m_(m), options_(o), dev_(o.device, o.budget), pool_(o.threads, true, true) {
  require(m.hidden == H && m.streams == 4 && m.linear_heads * m.linear_dim == KDA_N && m.linear_dim == 128 &&
              m.q_rank == Q_RANK && m.kv_rank == KV_RANK && m.attention_heads == 64 && m.index_heads == 32 &&
              m.index_dim == 128 && m.index_pool == 4 && m.experts == 144 && m.top_k == 8 && m.conv_kernel == 4 &&
              o.capacity >= 8 && o.capacity % 4 == 0 && m.index_top_k / m.index_pool <= 512,
          "Vulkan fast engine requires the GLM-5.3f REAP50 geometry");
  const auto started = Clock::now();
  if (const char *dir = std::getenv("STRATA_GLM_STATE_TRACE_DIR"); dir && *dir) trace_dir_ = dir;
  trace_position_ = strata_env_int_or("STRATA_VULKAN_FAST_TRACE_POSITION", 0);
  prefill_chunk_ = strata_env_int_or("STRATA_VULKAN_FAST_PREFILL_CHUNK", 4);
  require(prefill_chunk_ >= 1 && prefill_chunk_ <= 2048, "STRATA_VULKAN_FAST_PREFILL_CHUNK must be 1..2048");
  if (prefill_chunk_ > int(NT)) hidden_slots_ = uint32_t(prefill_chunk_) + NT;
  q8_emulate_ = strata_env_is_one("STRATA_VULKAN_FAST_Q8_EMULATE");
  expert_q8_ = strata_env_is_one("STRATA_VULKAN_FAST_EXPERT_Q8");
  canonical_sum_ = strata_env_is_one("STRATA_VULKAN_FAST_CANONICAL_SUM");
  prefill_gpu_ = strata_env_is_one("STRATA_VULKAN_FAST_PREFILL_GPU");
  prefill_gpu_pack_ = strata_env_is_one("STRATA_VULKAN_FAST_PREFILL_GPU_PACK");
  prefill_gpu_vector_ = strata_env_is_one("STRATA_VULKAN_FAST_PREFILL_GPU_VECTOR");
  prefill_match_gemv_ = strata_env_is_one("STRATA_VULKAN_FAST_PREFILL_MATCH_GEMV");
  prefill_progress_ = strata_env_is_one("STRATA_VULKAN_FAST_PREFILL_PROGRESS");
  prefill_tile_ = uint32_t(strata_env_int_or("STRATA_VULKAN_FAST_PREFILL_TILE", 0));
  require(prefill_tile_ == 0 || prefill_tile_ == 64 || prefill_tile_ == 128 || prefill_tile_ == 256,
          "Vulkan prefill tile must be 0 (automatic), 64, 128 or 256");
  if (prefill_gpu_) {
    prefill_gpu_max_experts_ = strata_env_int_or("STRATA_VULKAN_FAST_PREFILL_GPU_MAX_EXPERTS", 72);
    prefill_gpu_min_tokens_ = strata_env_int_or("STRATA_VULKAN_FAST_PREFILL_GPU_MIN_TOKENS", 32);
    require(prefill_gpu_max_experts_ >= 0 && prefill_gpu_max_experts_ <= 144 &&
                prefill_gpu_min_tokens_ >= 1 && prefill_gpu_min_tokens_ <= 2048,
            "Vulkan prefill GPU experts require max experts 0..144 and min tokens 1..2048");
  }
  require(!canonical_sum_ || expert_q8_, "Vulkan canonical sum requires STRATA_VULKAN_FAST_EXPERT_Q8=1");
  require(!prefill_gpu_ || (canonical_sum_ && prefill_chunk_ > int(NT)),
          "Vulkan prefill GPU experts require canonical expert sums and a prefill chunk greater than 4");
  require(!prefill_gpu_pack_ || prefill_gpu_, "Vulkan GPU packing requires streamed prefill GPU experts");
  require(!prefill_gpu_vector_ || prefill_gpu_, "Vulkan vector expert loads require streamed prefill GPU experts");
  if (canonical_sum_) mailbox_stride_ = MB_OUT + 8 * H;
  // Lossy, opt-in: the HIP path's routing affinity and min-share switches.
  auto fraction = [](const char *name, float hi) {
    const char *v = std::getenv(name);
    if (!v || !*v) return 0.f;
    char *end = nullptr;
    const float x = std::strtof(v, &end);
    if (*end || !(x >= 0.f && x <= hi)) throw std::invalid_argument(std::string(name) + " out of range");
    return x;
  };
  affinity_ = fraction("STRATA_GLM_ROUTE_AFFINITY", 2.f);
  min_share_ = fraction("STRATA_GLM_ROUTE_MIN_SHARE", 1.f);
  const int profile_level = strata_env_int_or("STRATA_VULKAN_FAST_PROFILE", 0);
  profile_ = profile_level >= 1;
  detail_ = profile_level >= 2;
  profile_t_ = uint32_t(strata_env_int_or("STRATA_VULKAN_FAST_PROFILE_T", 1));
  require(profile_t_ >= 1 && profile_t_ <= std::max(NT, uint32_t(prefill_chunk_)),
          "STRATA_VULKAN_FAST_PROFILE_T must fit the configured token window");
  load();
  allocate();
  allocate_tier();
  timeline_ = dev_.timeline(0);
  for (uint32_t T = 1; T <= NT; ++T) {
    record(T);
    if (mtp_)
      for (bool head : {false, true}) record_mtp(T, head);
  }
  reset();
  std::cerr << "VULKAN_FAST_MODEL layers=" << layers_.size() << " moe_layers=" << moe_layers_.size()
            << " tier_slots=" << tier_capacity_ << " route_affinity=" << affinity_ << " route_min_share=" << min_share_
            << " canonical_sum=" << canonical_sum_ << " segments=" << segments_[0].size() << " resident_MiB=" << dev_.live_bytes() / 1048576.0
            << " cpu_workers=" << pool_.workers() << " load_ms=" << ms_since(started) << '\n';
}
Engine::~Engine() {
  dev_.wait_idle();
  release_prefill();
  for (auto &set : segments_)
    for (auto &s : set) dev_.free_command(s.command);
  for (auto &heads : mtp_segments_)
    for (auto &set : heads)
      for (auto &s : set) dev_.free_command(s.command);
  dev_.destroy(timeline_);
}

size_t Engine::constant(std::span<const float> v) {
  while (constants_host_.size() % 4) constants_host_.push_back(0);
  const size_t at = constants_host_.size();
  constants_host_.insert(constants_host_.end(), v.begin(), v.end());
  return at;
}
size_t Engine::constant(const std::string &name) {
  const auto &t = artifact_.at(name);
  require(t.tensor->type == F32, "Vulkan fast constant must be F32");
  return constant(std::span(reinterpret_cast<const float *>(t.data()), t.tensor->elements()));
}

void Engine::load() {
  auto at = [&](const std::string &n) -> const core::ArtifactTensor & { return artifact_.at(n); };
  auto f32 = [&](const std::string &n) {
    const auto &t = at(n);
    return std::vector<float>(reinterpret_cast<const float *>(t.data()),
                              reinterpret_cast<const float *>(t.data()) + t.tensor->elements());
  };
  const size_t total = m_.layers.size() + (options_.mtp ? 1 : 0);
  if (options_.mtp)
    require(m_.draft_layers.size() == 1 && m_.draft_layers[0].mixer == core::MixerKind::SparseMla &&
                m_.draft_layers[0].ffn == core::FfnKind::Moe,
            "Vulkan fast MTP needs one MLA/MoE draft block");
  for (size_t l = 0; l < total; ++l) {
    const auto p = "blk." + std::to_string(l) + ".";
    const bool draft = l == m_.layers.size();
    const auto &desc = draft ? m_.draft_layers[0] : m_.layers[l];
    auto L = std::make_unique<Layer>();
    L->index = int(l);
    L->kda = desc.mixer == core::MixerKind::Kda;
    L->moe = desc.ffn == core::FfnKind::Moe;
    if (draft) {  // single-stream block: plain norms instead of mHC
      L->norm[0] = constant(p + "attn_norm.weight");
      L->norm[1] = constant(p + "ffn_norm.weight");
      L->enorm = constant(p + "nextn.enorm.weight");
      L->hnorm = constant(p + "nextn.hnorm.weight");
      L->head_norm = constant(p + "nextn.shared_head_norm.weight");
      L->eh_proj = upload_matrix(dev_, at(p + "nextn.eh_proj.weight"));
      require(L->eh_proj.k() == 2 * H && L->eh_proj.m() == H, "Vulkan fast MTP projection geometry");
    }
    for (int k = 0; k < 2 && !draft; ++k) {
      const std::string kind = k ? "ffn" : "attn";
      L->hc[k] = upload_matrix(dev_, at(p + "hc_" + kind + "_fn.weight"));
      auto par = f32(p + "hc_" + kind + "_base.weight");
      auto scale = f32(p + "hc_" + kind + "_scale.weight");
      require(par.size() == 24 && scale.size() == 3, "Vulkan fast mHC parameter geometry");
      par.insert(par.end(), scale.begin(), scale.end());
      L->hc_par[k] = constant(par);
      L->norm[k] = constant(p + kind + "_norm.weight");
    }
    if (L->kda) {
      const core::ArtifactTensor *parts[] = {&at(p + "attn_q.weight"), &at(p + "attn_k.weight"),
                                             &at(p + "ssm_f_a.weight"), &at(p + "ssm_g_a.weight"),
                                             &at(p + "ssm_beta.weight")};
      L->qk = upload_stacked(dev_, parts);
      require(L->qk.m() == 2 * KDA_N + 256 + 64, "Vulkan fast KDA stack geometry");
      L->v = upload_matrix(dev_, at(p + "attn_v.weight"));
      L->fb = upload_matrix(dev_, at(p + "ssm_f_b.weight"));
      L->gb = upload_matrix(dev_, at(p + "ssm_g_b.weight"));
      L->out = upload_matrix(dev_, at(p + "attn_output.weight"));
      std::vector<float> params;
      for (const char *part : {"q", "k", "v"}) {
        auto w = f32(p + "ssm_conv1d_" + part + ".weight");
        params.insert(params.end(), w.begin(), w.end());
      }
      auto bias = f32(p + "ssm_dt.bias");
      auto a = f32(p + "ssm_a");
      if (std::all_of(a.begin(), a.end(), [](float v) { return v > 0; }))
        for (auto &v : a) v = -v;
      auto norm = f32(p + "ssm_norm.weight");
      require(params.size() == size_t(3) * KDA_N * 4 && bias.size() == KDA_N && a.size() == 64 && norm.size() == 128,
              "Vulkan fast KDA parameter geometry");
      for (auto *v : {&bias, &a, &norm}) params.insert(params.end(), v->begin(), v->end());
      L->kda_params = dev_.buffer(params.size() * 4, Memory::Device);
      dev_.upload(*L->kda_params, params.data(), params.size() * 4);
    } else {
      const core::ArtifactTensor *parts[] = {&at(p + "attn_q_a.weight"), &at(p + "attn_kv_a_mqa.weight"),
                                             &at(p + "indexer.proj.weight"), &at(p + "indexer.attn_k.weight")};
      L->init = upload_stacked(dev_, parts);
      require(L->init.m() == MLA_INIT, "Vulkan fast MLA stack geometry");
      L->comp_gate = upload_matrix(dev_, at(p + "indexer_compressor_gate.weight"));
      require(L->comp_gate.m() == 128 && L->comp_gate.layout.type == F32, "Vulkan fast index gate geometry");
      L->qb = upload_matrix(dev_, at(p + "attn_q_b.weight"));
      L->iqb = upload_matrix(dev_, at(p + "indexer.attn_q_b.weight"));
      L->kb = upload_matrix(dev_, at(p + "attn_k_b.weight"));
      L->vb = upload_matrix(dev_, at(p + "attn_v_b.weight"));
      L->out = upload_matrix(dev_, at(p + "attn_output.weight"));
      require(L->kb.heads == 64 && L->kb.k() == 256 && L->kb.m() == KV_RANK && L->vb.k() == KV_RANK,
              "Vulkan fast MLA head geometry");
      L->mla_scale = 1 / std::sqrt(float(L->kb.k()));
      L->q_norm = constant(p + "attn_q_a_norm.weight");
      L->kv_norm = constant(p + "attn_kv_a_norm.weight");
      auto w = f32(p + "indexer.k_norm.weight"), b = f32(p + "indexer.k_norm.bias"),
           ape = f32(p + "indexer_compressor_ape.weight");
      require(w.size() == 128 && b.size() == 128 && ape.size() == 512, "Vulkan fast indexer parameter geometry");
      w.insert(w.end(), b.begin(), b.end());
      w.insert(w.end(), ape.begin(), ape.end());
      L->idx_par = constant(w);
      L->cache = dev_.buffer(size_t(options_.capacity) * KV_RANK * 4, Memory::Device);
      L->ring = dev_.buffer(2 * 4 * 128 * 4, Memory::Device);
      L->ring_history = dev_.buffer(NT * 2 * 4 * 128 * 4, Memory::Device);
      L->pooled = dev_.buffer(size_t(options_.capacity / 4) * 128 * 4, Memory::Device);
    }
    if (!L->moe) {
      const core::ArtifactTensor *parts[] = {&at(p + "ffn_gate.weight"), &at(p + "ffn_up.weight")};
      L->gu = upload_stacked(dev_, parts);
      L->down = upload_matrix(dev_, at(p + "ffn_down.weight"));
      L->limit = desc.swiglu_limit;
    } else {
      L->router = upload_matrix(dev_, at(p + "ffn_gate_inp.weight"));
      require(L->router.m() == 144 && L->router.layout.type == F32, "Vulkan fast router geometry");
      L->probs_bias = constant(p + "exp_probs_b.bias");
      const core::ArtifactTensor *parts[] = {&at(p + "ffn_gate_shexp.weight"), &at(p + "ffn_up_shexp.weight")};
      L->gu = upload_stacked(dev_, parts);
      L->down = upload_matrix(dev_, at(p + "ffn_down_shexp.weight"));
      L->limit = desc.shared_swiglu_limit;
      const auto &G = at(p + "ffn_gate_exps.weight"), &U = at(p + "ffn_up_exps.weight"),
                 &D = at(p + "ffn_down_exps.weight");
      std::string error;
      require(G.tensor->type == U.tensor->type, "Vulkan fast expert gate/up types differ");
      if (!kcpu::native_fmt(G.tensor->type, D.tensor->type, m_.hidden, desc.intermediate, L->fmt, error))
        throw std::runtime_error("Vulkan fast CPU expert geometry unsupported: " + error);
      L->fmt.swiglu_limit = desc.swiglu_limit;
      L->fmt.canon = (G.tensor->type == 10 || G.tensor->type == 11) && (D.tensor->type == 10 || D.tensor->type == 11);
      L->resident_eligible = G.tensor->type == Q2_K && D.tensor->type == Q3_K;
      L->gate = G.data();
      L->up = U.data();
      L->down_experts = D.data();
      L->gate_stride = G.bytes / m_.experts;
      L->down_stride = D.bytes / m_.experts;
      L->moe_index = int(moe_layers_.size());
      if (!draft) moe_layers_.push_back(int(l));
    }
    if (draft) mtp_ = std::move(L);
    else layers_.push_back(std::move(L));
  }
  output_norm_ = constant("output_norm.weight");
  head_ = upload_matrix(dev_, artifact_.at("output.weight"));
  constants_ = dev_.buffer(constants_host_.size() * 4, Memory::Device);
  dev_.upload(*constants_, constants_host_.data(), constants_host_.size() * 4);
}

void Engine::allocate() {
  // Activations hold up to NT tokens (verification windows and prefill chunks).
  auto device = [&](size_t floats) { return dev_.buffer(floats * 4, Memory::Device); };
  r_ = device(NT * 4 * H);
  proj_ = device(NT * 32);
  c_attn_ = device(NT * 32);
  c_ffn_ = device(NT * 32);
  xn_ = device(NT * H);
  y_ = device(NT * H);
  s_ = device(NT * QK_STACK + 64);
  v_ = device(NT * KDA_N);
  d_ = device(NT * KDA_N);
  g_ = device(NT * KDA_N);
  k_ = device(NT * KDA_N);
  m0_ = device(NT * MLA_INIT + 64);
  mq_ = device(NT * Q_RANK);
  mqb_ = device(NT * 64 * 256);
  miq_ = device(NT * 32 * 128);
  mg_ = device(NT * 128);
  maq_ = device(NT * 64 * KV_RANK);
  mav_ = device(NT * 64 * KV_RANK);
  mval_ = device(NT * 64 * 256);
  score_stride_ = uint32_t(options_.capacity / 4 + 64);
  scores_ = device(NT * score_stride_);
  ids_ = device(NT * IDS_STRIDE);
  gu_ = device(NT * 2 * 12288);
  h_ = device(NT * 12288);
  f_ = device(NT * H);
  logits144_ = device(NT * 144 + 16);
  part_ = device(size_t(16) * 16384 * NT);
  dummy_ = device(64);
  slot_of_ = dev_.buffer((m_.layers.size() + 1) * 144 * 4, Memory::Device);
  hits_ = device((m_.layers.size() + 1) * (128 + UNION_STRIDE));
  step_ = dev_.buffer(256, Memory::Host);
  embed_ = dev_.buffer(NT * H * 4, Memory::Host);
  mailbox_ = dev_.buffer((moe_layers_.size() + 1) * NT * mailbox_stride_ * 4, Memory::Host);
  cur_ = device(NT * H);
  join_ = device(NT * 2 * H);
  hidden_ring_ = device((hidden_slots_ + 1) * H); // final slot is the chained draft hidden
  if (mtp_) mtp_ring_saved_ = dev_.buffer(mtp_->ring->size, Memory::Device);
  mtp_step_ = dev_.buffer(256, Memory::Host);
  mtp_embed_ = dev_.buffer(NT * H * 4, Memory::Host);
  mtp_logits_ = dev_.buffer(size_t(m_.vocab) * 4, Memory::Host);
  logits_ = dev_.buffer(NT * size_t(m_.vocab) * 4, Memory::Host);
  qx_ = device(4 * 16448 * NT);
  if (!trace_dir_.empty()) trace_ = dev_.buffer(2 * m_.layers.size() * 4 * H * 4, Memory::Host);
  // KDA rollback: slots 1..NT-1 hold the state after each verified prefix.
  for (auto &L : layers_)
    if (L->kda) {
      L->state = dev_.buffer(size_t(NT) * KDA_N * 128 * 4, Memory::Device);
      L->history = dev_.buffer(size_t(NT) * 3 * KDA_N * 3 * 4, Memory::Device);
    }
  dev_.run([&](Recorder &r) {
    r.fill(View(*slot_of_), slot_of_->size, 0xffffffffu);
    r.fill(View(*hidden_ring_), hidden_ring_->size, 0);
  });
}

// Quantized single-head products optionally see Q8_1-rounded activations, the
// strict runner's numerics (diagnostic; the fast engine keeps FP32 by default).
void Engine::mv(Recorder &r, const Matrix &w, View x, View y, GemvArgs a) {
  if (a.columns > NT && prefill_match_gemv_) {
    const uint32_t xs = a.xs ? a.xs : w.k() * w.heads, ys = a.ys ? a.ys : w.m() * w.heads;
    for (uint32_t first = 0; first < a.columns; first += NT) {
      auto sub = a; sub.columns = std::min(NT, a.columns - first);
      sub.xofs += first * xs; sub.yofs += first * ys;
      mv(r, w, x, y, sub);
      // Split-K and activation rounding scratch can be reused by the next
      // tile, including inside a caller's concurrent projection group.
      r.barrier();
    }
    return;
  }
  if (a.columns > NT && w.layout.type == F32 && w.m() <= 256) a.scratch = View(*part_);
  if (!q8_emulate_ || w.layout.type == F32 || w.heads > 1) {
    if (a.columns > NT) gemm(r, w, x, y, a, prefill_tile_ ? prefill_tile_ : a.columns >= 128 ? 128 : 64);
    else gemv(r, w, x, y, a);
    return;
  }
  // Separate scratch slots: concurrent groups may hold several rounded inputs.
  const size_t offset = size_t(q8_slot_++ % 4) * (recording_prefill_ ? prefill_chunk_ : NT) * 16448 * 4;
  r.dispatch("q8_round", {x, View(*qx_, offset)},
             std::array<uint32_t, 4>{w.k(), a.xofs, a.xs ? a.xs : w.k(), w.k()}, w.k() / 32, a.columns);
  r.barrier();
  a.xofs = 0;
  a.xs = w.k();
  if (a.columns > NT) gemm(r, w, View(*qx_, offset), y, a, prefill_tile_ ? prefill_tile_ : a.columns >= 128 ? 128 : 64);
  else gemv(r, w, View(*qx_, offset), y, a);
}

void Engine::allocate_tier() {
  gate_layout_ = layout_for(Q2_K, H, 2048);
  down_layout_ = layout_for(Q3_K, 2048, H);
  slot_bytes_ = (2 * gate_layout_.head_bytes() + down_layout_.head_bytes() + 255) & ~size_t(255);
  slot_of_host_.assign((m_.layers.size() + 1) * 144, -1);
  route_counts_.assign((m_.layers.size() + 1) * 144, 0);
  if (const char *path = std::getenv("STRATA_GLM_EXPERT_PRIOR"); path && *path) {
    expert_prior_.assign(route_counts_.size(), 0);
    std::vector<bool> found(m_.layers.size(), false);
    const auto prior = strata::artifact::read_json(path);
    for (const auto &entry : prior.at("projections").array) {
      const auto &name = entry.at("name").string();
      if (!name.starts_with("blk.") || !name.ends_with(".gu")) continue;
      size_t consumed = 0;
      const auto text = name.substr(4, name.size() - 7);
      const size_t layer = std::stoul(text, &consumed);
      const auto &counts = entry.at("counts").array;
      require(consumed == text.size() && layer < m_.layers.size() && counts.size() == 144 && !found[layer],
              "Vulkan expert prior geometry/duplicate layer invalid");
      double total = 0;
      for (const auto &count : counts) total += count.integer(); // parser rejects negative integers
      require(total > 0, "Vulkan expert prior has an empty layer");
      for (size_t e = 0; e < 144; ++e) expert_prior_[layer * 144 + e] = double(counts[e].integer()) / total;
      found[layer] = true;
    }
    for (size_t l = 0; l < layers_.size(); ++l)
      require(!layers_[l]->moe || found[l], "Vulkan expert prior lacks a routed layer");
  }
  const size_t vectors = size_t(8) * NT * NT;  // union entries x tokens
  egu_ = dev_.buffer(vectors * 2 * 2048 * 4, Memory::Device);
  eh_ = dev_.buffer(vectors * 2048 * 4, Memory::Device);
  eout_ = dev_.buffer(vectors * H * 4, Memory::Device);
  esum_ = dev_.buffer(NT * H * 4, Memory::Device);
  xsum_ = dev_.buffer((NT * H / 16 + vectors * 2048 / 16) * 4 + 64, Memory::Device);
  if (expert_q8_) {
    expert_x_ = dev_.buffer(NT * H * 4, Memory::Device);
    expert_h_ = dev_.buffer(vectors * 2048 * 4, Memory::Device);
  }
  tier_capacity_ = options_.tier_bytes / slot_bytes_;
  slots_per_arena_ = (2ULL << 30) / slot_bytes_;
  std::vector<uint64_t> addresses;
  for (size_t left = prefill_chunk_ > int(NT) ? 0 : tier_capacity_; left;) {
    const size_t n = std::min(left, slots_per_arena_);
    arenas_.push_back(dev_.buffer(n * slot_bytes_, Memory::Device, true));
    const uint64_t base = dev_.address(*arenas_.back());
    for (size_t i = 0; i < n; ++i) addresses.push_back(base + i * slot_bytes_);
    left -= n;
  }
  slot_table_ = dev_.buffer(std::max<size_t>(1, tier_capacity_) * 8, Memory::Device);
  if (!addresses.empty()) dev_.upload(*slot_table_, addresses.data(), addresses.size() * 8);
}

void Engine::prepare_tier() {
  if (!tier_capacity_) return;
  const auto started = Clock::now();
  if (arenas_.empty()) {
    std::vector<uint64_t> addresses;
    for (size_t left = tier_capacity_; left;) {
      const size_t n = std::min(left, slots_per_arena_);
      arenas_.push_back(dev_.buffer(n * slot_bytes_, Memory::Device, true));
      const uint64_t base = dev_.address(*arenas_.back());
      for (size_t i = 0; i < n; ++i) addresses.push_back(base + i * slot_bytes_);
      left -= n;
    }
    dev_.upload(*slot_table_, addresses.data(), addresses.size() * 8);
  }
  std::vector<std::pair<double, int>> ranked;
  for (size_t key = 0; key < layers_.size() * 144; ++key)  // main layers; draft experts stay on the CPU
    if (layers_[key / 144]->moe && layers_[key / 144]->resident_eligible) {
      const double score = double(route_counts_[key]) + (expert_prior_.empty() ? 0 : 256.0 * m_.top_k * expert_prior_[key]);
      if (score > 0) ranked.emplace_back(score, int(key));
    }
  std::sort(ranked.begin(), ranked.end(),
            [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  std::fill(slot_of_host_.begin(), slot_of_host_.end(), -1);
  tier_used_ = 0;
  std::vector<uint8_t> packed(slot_bytes_);
  for (auto [count, key] : ranked) {
    if (tier_used_ >= tier_capacity_) break;
    const auto &L = *layers_[key / 144];
    const int id = key % 144;
    repack(gate_layout_, L.gate + size_t(id) * L.gate_stride, packed.data());
    repack(gate_layout_, L.up + size_t(id) * L.gate_stride, packed.data() + gate_layout_.head_bytes());
    repack(down_layout_, L.down_experts + size_t(id) * L.down_stride, packed.data() + 2 * gate_layout_.head_bytes());
    const size_t slot = tier_used_++;
    dev_.upload(*arenas_[slot / slots_per_arena_], packed.data(), slot_bytes_, (slot % slots_per_arena_) * slot_bytes_);
    slot_of_host_[key] = int(slot);
  }
  dev_.upload(*slot_of_, slot_of_host_.data(), slot_of_host_.size() * 4);
  tier_ready_ = true;
  std::cerr << "VULKAN_FAST_TIER slots=" << tier_used_ << " MiB=" << tier_used_ * slot_bytes_ / 1048576.0
            << " setup_ms=" << ms_since(started) << '\n';
}

void Engine::record_experts(Recorder &r, int l, uint32_t T) {
  const auto &L = *layers_[l];  // main layers only: draft experts stay on the CPU
  if (recording_prefill_ || !tier_capacity_ || !L.resident_eligible) return;
  const uint32_t hofs = uint32_t(l * 128), uofs = uint32_t((m_.layers.size() + 1) * 128 + l * UNION_STRIDE);
  const uint32_t xsum_h = T * H / 16;  // eh sums follow the xn sums
  r.dispatch("hits_union", {View(*hits_)}, std::array<uint32_t, 3>{hofs, uofs, T}, 1);
  const auto &expert_x = expert_q8_ ? *expert_x_ : *xn_;
  const auto &expert_h = expert_q8_ ? *expert_h_ : *eh_;
  if (expert_q8_)
    r.dispatch("q8k_round", {View(*xn_), View(expert_x), View(*hits_)},
               std::array<uint32_t, 4>{T * H, uofs, 0, T}, T * H / 256);
  r.dispatch("xsum16", {View(expert_x), View(*xsum_)}, std::array<uint32_t, 2>{T * H / 16, 0}, (T * H / 16 + 255) / 256);
  struct Push {
    uint32_t K, part_rows, parts, Mp, part_bytes, slot_offset, ksw, side, xofs, xhit, yhit, hofs, columns, xcol, ycol;
  };
  const Push gu{H, 2048, 2, gate_layout_.mp, uint32_t(gate_layout_.head_bytes()), 0, 4,
                uint32_t(gate_layout_.chunk_bytes()), 0, 0, T * 2 * 2048, uofs, T, H, 2 * 2048};
  const Push down{2048, H, 1, down_layout_.mp, uint32_t(down_layout_.head_bytes()),
                  uint32_t(2 * gate_layout_.head_bytes()), 4, uint32_t(down_layout_.chunk_bytes()), 0, T * 2048, T * H,
                  uofs, T, 2048, H};
  r.dispatch("gemv_expert_q2k", {View(*slot_table_), View(expert_x), View(*egu_), View(*hits_), View(*xsum_)}, gu,
             2 * 2048 / 64, 8 * T);
  struct {
    uint32_t mode, n, uofs, tokens;
    float limit;
  } swi{0, 2048, uofs, T, m_.layers[l].swiglu_limit}, sum{1, H, uofs, T, 0};
  r.dispatch(expert_q8_ ? "expert_mix_canon" : "expert_mix", {View(*egu_), View(*eh_), View(*hits_)}, swi, 8 * T * T * 2048 / 256);
  if (expert_q8_)
    r.dispatch("q8k_round", {View(*eh_), View(expert_h), View(*hits_)},
               std::array<uint32_t, 4>{8 * T * T * 2048, uofs, 2048, T}, 8 * T * T * 2048 / 256);
  r.dispatch("xsum16", {View(expert_h), View(*xsum_, xsum_h * 4)}, std::array<uint32_t, 2>{8 * T * T * 2048 / 16, 0},
             8 * T * T * 2048 / 16 / 256);
  r.dispatch("gemv_expert_q3k", {View(*slot_table_), View(expert_h), View(*eout_), View(*hits_), View(*xsum_, xsum_h * 4)},
             down, H / 64, 8 * T);
  if (!canonical_sum_) r.dispatch("expert_mix", {View(*eout_), View(*esum_), View(*hits_)}, sum, H / 256, T);
}

float *Engine::mailbox(int i, int t) const {
  return mailbox_->as<float>() + ((running_prefill_ ? 0 : size_t(i) * NT) + t) * mailbox_stride_;
}

namespace {
struct NormPush {
  uint32_t n, xofs, wofs, yofs, flags;
  float eps;
  uint32_t xstride, ystride;
};
} // namespace

void Engine::record_attention(Recorder &r, int l, uint32_t T) {
  const auto &L = *layers_[l];
  read_mhc(r, L, 0, T);
  GemvArgs cols;
  cols.columns = T;
  if (L.kda) {
    r.begin_concurrent();
    mv(r, L.qk, View(*xn_), View(*s_), cols);
    mv(r, L.v, View(*xn_), View(*v_), cols);
    r.end_concurrent();
    r.begin_concurrent();
    GemvArgs fa = cols, ga = cols;
    fa.xofs = 2 * KDA_N;
    ga.xofs = 2 * KDA_N + 128;
    fa.xs = ga.xs = L.qk.m();
    mv(r, L.fb, View(*s_), View(*d_), fa);
    mv(r, L.gb, View(*s_), View(*g_), ga);
    r.end_concurrent();
    struct {
      uint32_t heads, kernel, tokens, bofs;
      float lower, eps;
      uint32_t qk_stride, v_stride, d_stride, g_stride, y_stride, snapshot;
    } kp{64, uint32_t(m_.conv_kernel), T, 2 * KDA_N + 256, m_.gate_lower_bound, m_.rms_epsilon, L.qk.m(),
         KDA_N, KDA_N, KDA_N, KDA_N, T > 1 && !recording_prefill_ ? 1u : 0u};
    r.dispatch("kda_step", {View(*L.state), View(*s_), View(*v_), View(*d_), View(*g_), View(*L.kda_params),
                            View(*k_), View(*L.history)},
               kp, 64);
    mv(r, L.out, View(*k_), View(*y_), cols);
  } else
    record_mla(r, L, T, *step_);
  write_mhc(r, *c_attn_, View(*y_), {}, 1, 0, T);
}

void Engine::record_mla(Recorder &r, const Layer &L, uint32_t T, const Buffer &step, bool cache_only) {
  GemvArgs cols;
  cols.columns = T;
    r.begin_concurrent();
    GemvArgs split = cols;
    split.scratch = View(*part_);
    mv(r, L.init, View(*xn_), View(*m0_), split);
    mv(r, L.comp_gate, View(*xn_), View(*mg_), cols);
    r.end_concurrent();
    r.begin_concurrent();
    if (!cache_only)
      r.dispatch("rmsnorm", {View(*m0_), View(*constants_), View(*mq_), View(step)},
                 NormPush{Q_RANK, 0, uint32_t(L.q_norm), 0, 1, m_.rms_epsilon, MLA_INIT, Q_RANK}, 1, T);
    r.dispatch("rmsnorm", {View(*m0_), View(*constants_), View(*L.cache), View(step)},
               NormPush{KV_RANK, Q_RANK, uint32_t(L.kv_norm), 0, 1 | 4, m_.rms_epsilon, MLA_INIT, 0}, 1, T);
    r.dispatch("index_key", {View(*m0_), View(*mg_), View(*constants_), View(*L.ring), View(*L.pooled), View(step), View(*L.ring_history)},
               std::array<uint32_t, 7>{128, 4, IDX_KEY, uint32_t(L.idx_par), T, MLA_INIT,
                                      !recording_prefill_ && &step == step_.get() && T > 1 ? 1u : 0u}, 1);
    r.end_concurrent();
    if (cache_only) return;
    r.begin_concurrent();
    mv(r, L.qb, View(*mq_), View(*mqb_), cols);
    mv(r, L.iqb, View(*mq_), View(*miq_), cols);
    r.end_concurrent();
    r.begin_concurrent();
    r.dispatch("index_scores_step", {View(*L.pooled), View(*miq_), View(*m0_), View(*scores_), View(step)},
               std::array<uint32_t, 8>{32, 128, 4, 0, IDX_PROJ, 32 * 128, MLA_INIT, score_stride_},
               (uint32_t(options_.capacity) / 4 + 7) / 8, T);
    GemvArgs heads = cols;
    heads.xs = 64 * 256;
    heads.ys = 64 * KV_RANK;
    mv(r, L.kb, View(*mqb_), View(*maq_), heads);
    r.end_concurrent();
    r.dispatch("select_pools", {View(*scores_), View(*ids_), View(step)},
               std::array<uint32_t, 4>{4, uint32_t(m_.index_top_k / m_.index_pool), score_stride_, IDS_STRIDE}, T);
    struct {
      uint32_t heads, latent;
      float scale;
      uint32_t istride;
    } mp{64, KV_RANK, L.mla_scale, IDS_STRIDE};
    r.dispatch("mla_step", {View(*L.cache), View(*maq_), View(*ids_), View(*mav_)}, mp, 64, T);
    GemvArgs vheads = cols;
    vheads.xs = 64 * KV_RANK;
    vheads.ys = 64 * 256;
    mv(r, L.vb, View(*mav_), View(*mval_), vheads);
    mv(r, L.out, View(*mval_), View(*y_), cols);
  }

void Engine::record_ffn_head(Recorder &r, int l, uint32_t T) {
  const auto &L = *layers_[l];
  read_mhc(r, L, 1, T);
  GemvArgs cols;
  cols.columns = T;
  if (!L.moe) {
    record_shared(r, L, T);
    write_mhc(r, *c_ffn_, View(*f_), {}, 1, 0, T);
    return;
  }
  record_route(r, L, T);
}

void Engine::record_route(Recorder &r, const Layer &L, uint32_t T) {
  GemvArgs cols;
  cols.columns = T;
  const int l = L.index;
  const uint32_t stride = mailbox_stride_;
  mv(r, L.router, View(*xn_), View(*logits144_), cols);
  const uint32_t mofs = recording_prefill_ ? 0 : uint32_t(size_t(L.moe_index) * NT * mailbox_stride_);
  struct {
    uint32_t experts, topk, mofs, bofs, rofs, hofs;
    float scale, affinity, min_share;
    uint32_t lstride, mstride;
  } rp{144, uint32_t(m_.top_k), mofs, uint32_t(L.probs_bias), uint32_t(l * 144), uint32_t(l * 128), m_.expert_scale,
       affinity_, min_share_, 144, stride};
  r.begin_concurrent();
  r.dispatch("router", {View(*logits144_), View(*constants_), View(*mailbox_), View(*slot_of_), View(*hits_)}, rp, T);
  for (uint32_t t = 0; t < T; ++t)
    r.copy(View(*xn_, size_t(t) * H * 4), View(*mailbox_, (size_t(mofs) + t * stride) * 4), H * 4);
  r.end_concurrent();
  r.host_visible();
}

void Engine::record_shared(Recorder &r, const Layer &L, uint32_t T) {
  GemvArgs cols;
  cols.columns = T;
  mv(r, L.gu, View(*xn_), View(*gu_), cols);
  const uint32_t n = L.gu.m() / 2;
  swiglu(r, View(*gu_), View(*h_), n, T, L.limit);
  mv(r, L.down, View(*h_), View(*f_), cols);
}

void Engine::record_ffn_tail(Recorder &r, int l, uint32_t T) {
  const auto &L = *layers_[l];
  if (canonical_sum_ && L.fmt.canon) {
    const uint32_t mofs = recording_prefill_ ? 0 : uint32_t(size_t(L.moe_index) * NT * mailbox_stride_);
    r.dispatch("route_merge", {View(*mailbox_), View(*eout_), View(*hits_), View(*esum_)},
               std::array<uint32_t, 8>{H, uint32_t(m_.top_k), mofs, mailbox_stride_, uint32_t(l * 128), T, MB_OUT, MB_WEIGHTS}, H / 256, T);
    write_mhc(r, *c_ffn_, View(*f_), View(*esum_), 2, 0, T, {}, H);
    return;
  }
  const bool tier = !recording_prefill_ && tier_capacity_ && L.resident_eligible;
  write_mhc(r, *c_ffn_, View(*f_), View(*mailbox_), tier ? 3 : 2,
            uint32_t((recording_prefill_ ? 0 : size_t(L.moe_index) * NT * mailbox_stride_) + MB_OUT), T, tier ? View(*esum_) : View());
}

void Engine::read_mhc(Recorder &r, const Layer &L, int kind, uint32_t T) {
  GemvArgs cols;
  cols.columns = T;
  cols.xs = 4 * H;
  cols.ys = 32;
  mv(r, L.hc[kind], View(*r_), View(*proj_), cols);
  struct {
    uint32_t width, iterations, pofs, wofs;
    float hc_eps, rms_eps;
  } push{H, uint32_t(m_.sinkhorn_iterations), uint32_t(L.hc_par[kind]), uint32_t(L.norm[kind]), m_.hc_epsilon,
         m_.rms_epsilon};
  r.dispatch("mhc_read", {View(*r_), View(*proj_), View(*constants_), View(kind ? *c_ffn_ : *c_attn_), View(*xn_)},
             push, T);
}

void Engine::write_mhc(Recorder &r, const Buffer &c, View ya, View yb, uint32_t inputs, uint32_t ybofs, uint32_t T,
                       View yc, uint32_t ybstride) {
  struct {
    uint32_t width, inputs, ybofs, ycofs, ybstride;
  } push{H, inputs, ybofs, 0, ybstride ? ybstride : mailbox_stride_};
  r.dispatch("mhc_write", {View(*r_), View(c), ya, yb.buffer ? yb : View(*dummy_), yc.buffer ? yc : View(*dummy_)},
             push, H / 256, T);
}

void Engine::record_head(Recorder &r, uint32_t T) {
  r.dispatch("rmsnorm", {View(*r_), View(*constants_), View(*xn_), View(*step_)},
             NormPush{H, 0, uint32_t(output_norm_), 0, 1 | 2, m_.rms_epsilon, 4 * H, H}, 1, T);
  GemvArgs cols;
  cols.columns = recording_prefill_ ? 1 : T;
  if (recording_prefill_) cols.xofs = (T - 1) * H;
  r.begin_concurrent();
  mv(r, head_, View(*xn_), View(*logits_), cols);
  if (mtp_)
    r.dispatch("hidden_store", {View(*xn_), View(*hidden_ring_), View(*step_)}, std::array<uint32_t, 2>{H, hidden_slots_},
               H / 256, T);
  r.end_concurrent();
  r.host_visible();
}

// Draft block for a window of T tokens at the MTP position: joins the token
// embedding with the previous verified hidden state, runs its MLA and MoE with
// plain residuals, and (with head) the shared head on the last token.
void Engine::record_mtp(uint32_t T, bool head) {
  auto &segments = mtp_segments_[head][T - 1];
  const auto &L = *mtp_;
  auto open = [&] {
    auto c = dev_.allocate_command();
    dev_.begin(c, false);
    return c;
  };
  VkCommandBuffer c = open();
  auto r = std::make_unique<Recorder>(dev_, c);
  r->begin_concurrent();
  r->dispatch("rmsnorm", {View(*mtp_embed_), View(*constants_), View(*join_), View(*mtp_step_)},
              NormPush{H, 0, uint32_t(L.enorm), 0, 1, m_.rms_epsilon, H, 2 * H}, 1, T);
  r->dispatch("rmsnorm", {View(*hidden_ring_), View(*constants_), View(*join_), View(*mtp_step_)},
                NormPush{H, 0, uint32_t(L.hnorm), H, 1 | 8 | 16, m_.rms_epsilon, hidden_slots_, 2 * H}, 1, T);
  r->end_concurrent();
  GemvArgs cols;
  cols.columns = T;
  mv(*r, L.eh_proj, View(*join_), View(*cur_), cols);
  r->dispatch("rmsnorm", {View(*cur_), View(*constants_), View(*xn_), View(*mtp_step_)},
              NormPush{H, 0, uint32_t(L.norm[0]), 0, 1, m_.rms_epsilon, H, H}, 1, T);
  record_mla(*r, L, T, *mtp_step_, !head);
  if (!head) {
    // Priming only needs the draft block's keys and index pools. Its FFN
    // outputs are discarded: the next token reads verified target hiddens.
    dev_.end(c);
    segments.push_back({c, {}, {timeline_, 3}});
    return;
  }
  struct {
    uint32_t width, inputs, ybofs, ybstride;
  } add1{H, 1, 0, 0}, add2{H, 2, uint32_t(size_t(L.moe_index) * NT * mailbox_stride_ + MB_OUT), mailbox_stride_};
  r->dispatch("residual_add", {View(*cur_), View(*y_), View(*dummy_), View(*dummy_)}, add1, H / 256, T);
  r->dispatch("rmsnorm", {View(*cur_), View(*constants_), View(*xn_), View(*mtp_step_)},
              NormPush{H, (T - 1) * H, uint32_t(L.norm[1]), 0, 1, m_.rms_epsilon, H, H}, 1, 1);
  record_route(*r, L, 1);
  dev_.end(c);
  segments.push_back({c, {}, {timeline_, 1}});
  c = open();
  r = std::make_unique<Recorder>(dev_, c);
  record_shared(*r, L, 1);
  dev_.end(c);
  segments.push_back({c, {}, {}});
  c = open();
  r = std::make_unique<Recorder>(dev_, c);
  r->dispatch("residual_add", {View(*cur_, size_t(T - 1) * H * 4), View(*f_), View(*mailbox_), View(*dummy_)}, add2, H / 256, 1);
  if (head) {
    r->dispatch("rmsnorm", {View(*cur_), View(*constants_), View(*xn_), View(*mtp_step_)},
                NormPush{H, (T - 1) * H, uint32_t(L.head_norm), 0, 1, m_.rms_epsilon, 0, 0}, 1, 1);
    mv(*r, head_, View(*xn_), View(*mtp_logits_));
    r->copy(View(*xn_), View(*hidden_ring_, size_t(hidden_slots_) * H * 4), H * 4);
    r->host_visible();
  }
  dev_.end(c);
  segments.push_back({c, {timeline_, 2}, {timeline_, 3}});
}

const float *Engine::mtp_forward(std::span<const int> tokens, bool head, bool chain) {
  const uint32_t T = uint32_t(tokens.size());
  require(mtp_ && T >= 1 && T <= NT && mtp_position_ + int(T) <= options_.capacity, "Vulkan fast MTP window invalid");
  require(!chain || (head && T == 1), "Vulkan chained draft needs one token and a head");
  const auto started = Clock::now();
  const auto &embedding = artifact_.at("token_embd.weight");
  for (uint32_t t = 0; t < T; ++t) {
    require(tokens[t] >= 0 && tokens[t] < m_.vocab, "Vulkan fast MTP token out of range");
    const uint8_t *src = embedding.data() + size_t(tokens[t]) * embedding.bytes / m_.vocab;
    float *e = mtp_embed_->as<float>() + size_t(t) * H;
    if (embedding.tensor->type == F32) std::memcpy(e, src, H * 4);
    else ggml_get_type_traits(ggml_type(embedding.tensor->type))->to_float(src, e, H);
  }
  mtp_step_->as<uint32_t>()[0] = uint32_t(mtp_position_);
  mtp_step_->as<uint32_t>()[1] = chain ? 1 : 0;
  std::vector<Submission> batch(mtp_segments_[head][T - 1]);
  for (auto &s : batch) {
    if (s.wait.semaphore) s.wait.value += timeline_base_;
    if (s.signal.semaphore) s.signal.value += timeline_base_;
  }
  dev_.submit_batch(batch);
  if (head) {
    if (!dev_.wait(timeline_, timeline_base_ + 1)) fatal("timed out waiting for draft router output");
    try {
      cpu_experts(*mtp_, 1);
    } catch (...) {
      dev_.signal(timeline_, timeline_base_ + 64);
      dev_.wait_idle();
      throw;
    }
    dev_.signal(timeline_, timeline_base_ + 2);
  }
  if (!dev_.wait(timeline_, timeline_base_ + 3)) fatal("timed out waiting for draft logits");
  timeline_base_ += 64;
  mtp_position_ += int(T);
  draft_ms_ += ms_since(started);
  return mtp_logits_->as<float>();
}

void Engine::record_trace(Recorder &r, int l) {
  if (!trace_) return;
  r.dispatch("trace_window", {View(*r_), View(*trace_), View(*step_)},
             std::array<uint32_t, 4>{4 * H, uint32_t(trace_position_), recording_width_, uint32_t(l) * 4 * H}, 4 * H / 256);
}
void Engine::record_trace_attention(Recorder &r, int l) {
  if (!trace_) return;
  r.dispatch("trace_window", {View(*r_), View(*trace_), View(*step_)},
             std::array<uint32_t, 4>{4 * H, uint32_t(trace_position_), recording_width_, uint32_t(layers_.size() + l) * 4 * H}, 4 * H / 256);
}

void Engine::record(uint32_t T) {
  record_window(T, segments_[T - 1]);
}
void Engine::record_window(uint32_t T, std::vector<Submission> &segments) {
  recording_width_ = T;
  if (recording_prefill_ && profile_ && T == profile_t_) {
    // Prefill scratch is released after each prompt; recreated commands must
    // reuse the query layout rather than append stale, unwritten queries.
    queries_used_ = 0; query_names_.clear(); segment_names_.clear();
  }
  uint32_t &query = queries_used_;
  auto mark = [&](const std::string &name) { query_names_.push_back(name); };
  auto open = [&] {
    auto c = dev_.allocate_command();
    dev_.begin(c, false);
    if (profile_ && T == profile_t_) {
      require(query < Device::query_capacity, "Vulkan profile timestamp capacity exceeded");
      vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dev_.queries(), query++);
      mark("@begin");
    }
    return c;
  };
  auto close = [&](VkCommandBuffer c, Recorder &r, const std::string &name) {
    if (profile_ && T == profile_t_) {
      r.timestamp(query++);
      mark("@end");
      segment_names_.push_back(name);
    }
    dev_.end(c);
  };
  auto attach = [&](Recorder &r) {
    if (!detail_ || T != profile_t_) return;
    r.names = &query_names_;
    r.next = &query;
  };
  VkCommandBuffer c = open();
  auto rec = std::make_unique<Recorder>(dev_, c);
  attach(*rec);
  rec->begin_concurrent();
  for (uint32_t t = 0; t < T; ++t)
    for (int s = 0; s < 4; ++s)
      rec->copy(View(*embed_, size_t(t) * H * 4), View(*r_, (size_t(t) * 4 + s) * H * 4), H * 4);
  rec->end_concurrent();
  const bool trace = bool(trace_);
  Wait wait{};
  for (size_t l = 0; l < layers_.size(); ++l) {
    const auto &L = *layers_[l];
    record_attention(*rec, int(l), T);
    if (trace) record_trace_attention(*rec, int(l));
    record_ffn_head(*rec, int(l), T);
    if (!L.moe) {
      if (trace) record_trace(*rec, int(l));
      continue;
    }
    const uint64_t ready = 2 * uint64_t(L.moe_index) + 1, done = ready + 1;
    close(c, *rec, "attn" + std::to_string(l));
    segments.push_back({c, wait, {timeline_, ready}});
    c = open();
    rec = std::make_unique<Recorder>(dev_, c);
    attach(*rec);
    record_experts(*rec, int(l), T);
    record_shared(*rec, L, T);
    close(c, *rec, "shared" + std::to_string(l));
    segments.push_back({c, {}, {}});
    c = open();
    rec = std::make_unique<Recorder>(dev_, c);
    attach(*rec);
    record_ffn_tail(*rec, int(l), T);
    if (trace) record_trace(*rec, int(l));
    wait = {timeline_, done};
  }
  record_head(*rec, T);
  close(c, *rec, "head");
  segments.push_back({c, wait, {timeline_, 2 * moe_layers_.size() + 1}});
}

void Engine::reset() {
  dev_.run([&](Recorder &r) {
    for (auto &L : layers_)
      for (auto *b : {L->state.get(), L->history.get(), L->cache.get(), L->ring.get(), L->pooled.get()})
        if (b) r.fill(View(*b), b->size, 0);
    if (mtp_)
      for (auto *b : {mtp_->cache.get(), mtp_->ring.get(), mtp_->pooled.get(), hidden_ring_.get()})
        r.fill(View(*b), b->size, 0);
  });
  position_ = 0;
  mtp_position_ = 0;
  if (tier_capacity_) {
    if (prefill_chunk_ > int(NT)) arenas_.clear();
    std::fill(slot_of_host_.begin(), slot_of_host_.end(), -1);
    std::fill(route_counts_.begin(), route_counts_.end(), 0);
    dev_.upload(*slot_of_, slot_of_host_.data(), slot_of_host_.size() * 4);
    tier_ready_ = false;
    tier_used_ = 0;
  }
}

void Engine::allocate_prefill() {
  if (!prefill_buffers_.empty()) return;
  const size_t T = size_t(prefill_chunk_);
  auto add = [&](std::unique_ptr<Buffer> &buffer, size_t floats, Memory memory = Memory::Device) {
    prefill_buffers_.emplace_back(&buffer, dev_.buffer(floats * 4, memory));
  };
  add(r_, T * 4 * H); add(proj_, T * 32); add(c_attn_, T * 32); add(c_ffn_, T * 32);
  add(xn_, T * H); add(y_, T * H); add(s_, T * QK_STACK + 64);
  add(v_, T * KDA_N); add(d_, T * KDA_N); add(g_, T * KDA_N); add(k_, T * KDA_N);
  add(m0_, T * MLA_INIT + 64); add(mq_, T * Q_RANK); add(mqb_, T * 64 * 256);
  add(miq_, T * 32 * 128); add(mg_, T * 128); add(maq_, T * 64 * KV_RANK);
  add(mav_, T * 64 * KV_RANK); add(mval_, T * 64 * 256);
  add(scores_, T * score_stride_); add(ids_, T * IDS_STRIDE);
  add(gu_, T * 2 * 12288); add(h_, T * 12288); add(f_, T * H); add(logits144_, T * 144 + 16);
  add(embed_, T * H, Memory::Host); add(mailbox_, T * mailbox_stride_, Memory::Host);
  if (canonical_sum_) add(esum_, T * H);
  add(part_, prefill_match_gemv_ ? std::max(size_t(16) * 256 * T, size_t(16) * 16384 * NT) : 16 * 256 * T);
  add(hits_, T * 32 + (m_.layers.size() + 1) * 128 + 64);
  if (q8_emulate_) add(qx_, 4 * 16448 * T);
}

void Engine::release_prefill() {
  require(!running_prefill_, "cannot release an active Vulkan prefill");
  for (auto &[width, segments] : prefill_segments_)
    for (const auto &segment : segments) dev_.free_command(segment.command);
  prefill_segments_.clear();
  prefill_buffers_.clear();
}

void Engine::cpu_experts_prefill(const Layer &L, uint32_t T) {
  const int top = int(m_.top_k);
  std::vector<uint8_t> acts(size_t(T) * L.fmt.act_bytes);
  std::vector<float> results(size_t(T) * top * H, 0.f), weights(size_t(T) * top, 0.f), sums(size_t(T) * H);
  std::vector<kcpu::ExpertJobMulti> jobs;
  int last_job[144]; std::fill(std::begin(last_job), std::end(last_job), -1);
  const size_t key0 = size_t(L.index) * 144;
  for (uint32_t t = 0; t < T; ++t) {
    float *mb = mailbox(L.moe_index, int(t));
    int ids[16]; float routing[16];
    std::memcpy(ids, mb + MB_IDS, top * 4); std::memcpy(routing, mb + MB_WEIGHTS, top * 4);
    uint8_t *act = acts.data() + size_t(t) * L.fmt.act_bytes;
    for (uint32_t d = 0; d < H; ++d)
      if (!std::isfinite(mb[d]))
        throw std::runtime_error("nonfinite Vulkan prefill expert input: layer=" + std::to_string(L.index) +
                                 " token=" + std::to_string(t) + " column=" + std::to_string(d));
    kcpu::native_quant_act(L.fmt, mb, act);
    for (int k = 0; k < top; ++k) {
      require(ids[k] >= 0 && ids[k] < 144, "Vulkan prefill router produced an invalid expert");
      ++route_counts_[key0 + ids[k]]; ++routes_;
      if (routing[k] == 0.f) continue;
      int &index = last_job[ids[k]];
      if (index < 0 || jobs[size_t(index)].nt == kcpu::MAXT) {
        index = int(jobs.size()); jobs.emplace_back();
        auto &job = jobs.back(); job.expert_id = ids[k];
        job.blob = L.gate + size_t(ids[k]) * L.gate_stride;
        job.native_up = L.up + size_t(ids[k]) * L.gate_stride;
        job.native_down = L.down_experts + size_t(ids[k]) * L.down_stride;
      }
      auto &job = jobs[size_t(index)];
      job.nact[job.nt] = act; job.out[job.nt] = results.data() + (size_t(t) * top + k) * H;
      ++job.nt; weights[size_t(t) * top + k] = routing[k];
    }
  }
  std::stable_sort(jobs.begin(), jobs.end(), [](const auto &a, const auto &b) { return a.expert_id < b.expert_id; });
  if (prefill_gpu_ && L.resident_eligible) {
    std::array<int, 144> counts{};
    std::array<bool, 144> on_gpu{};
    for (const auto &job : jobs) counts[size_t(job.expert_id)] += job.nt;
    std::array<int, 144> order{};
    for (int e = 0; e < 144; ++e) order[size_t(e)] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return counts[size_t(a)] > counts[size_t(b)];});
    uint32_t capacity = 0;
    for (size_t i = 0; i < size_t(prefill_gpu_max_experts_) && counts[size_t(order[i])] >= prefill_gpu_min_tokens_; ++i) {
      const int e = order[i]; on_gpu[size_t(e)] = true;
      capacity = std::max(capacity, uint32_t(counts[size_t(e)]));
    }
    if (capacity) {
      std::vector<kcpu::ExpertJobMulti> cpu_jobs;
      for (const auto &job : jobs) if (!on_gpu[size_t(job.expert_id)]) cpu_jobs.push_back(job);
      ExpertBatch gpu(dev_, capacity, false, prefill_gpu_pack_, prefill_gpu_vector_);
      // The pool writes only the non-GPU result rows. Keep its future inside
      // this scope so exceptions still join before any referenced storage dies.
      std::future<void> cpu;
      if (!cpu_jobs.empty()) cpu = std::async(std::launch::async, [&] {
        pool_.run_split_multi_native(L.fmt, cpu_jobs.data(), int(cpu_jobs.size()));
      });
      uint32_t gpu_routes = 0, gpu_experts = 0;
      for (int expert : order) {
        if (!on_gpu[size_t(expert)]) continue;
        std::vector<const uint8_t *> inputs;
        std::vector<float *> outputs;
        for (const auto &job : jobs) if (job.expert_id == expert)
          for (int j = 0; j < job.nt; ++j) {
            inputs.push_back(static_cast<const uint8_t *>(job.nact[j]));
            outputs.push_back(job.out[j]);
          }
        gpu.run_quantized(L.gate + size_t(expert) * L.gate_stride, L.up + size_t(expert) * L.gate_stride,
                          L.down_experts + size_t(expert) * L.down_stride, inputs, outputs, L.fmt.swiglu_limit);
        gpu_routes += uint32_t(inputs.size()); ++gpu_experts;
      }
      if (cpu.valid()) cpu.get();
      const auto &stats = gpu.stats();
      std::cerr << "PREFILL_EXPERT_SPLIT layer=" << L.index << " tokens=" << T
                << " gpu_experts=" << gpu_experts << " gpu_routes=" << gpu_routes
                << " gpu_pack=" << prefill_gpu_pack_
                << " vector_weights=" << prefill_gpu_vector_
                << " pack_ms=" << stats.pack_ms << " upload_ms=" << stats.upload_ms
                << " input_bytes=" << stats.input_bytes
                << " queued_work_ms=" << stats.queued_work_ms << '\n';
    } else pool_.run_split_multi_native(L.fmt, jobs.data(), int(jobs.size()));
  } else pool_.run_split_multi_native(L.fmt, jobs.data(), int(jobs.size()));
  if (canonical_sum_ && L.fmt.canon) {
    // Use the same GPU reduction as small windows. GLSL fma is not required
    // to round like CPU std::fma; switching devices here can change later Q8
    // grids and routes even when the expert rows match exactly.
    for (uint32_t t = 0; t < T; ++t)
      std::memcpy(mailbox(L.moe_index, int(t)) + MB_OUT, results.data() + size_t(t) * top * H, size_t(top) * H * 4);
  } else {
    for (uint32_t t = 0; t < T; t += kcpu::MAXT) {
      const int n = int(std::min<uint32_t>(kcpu::MAXT, T - t));
      pool_.reduce_routed(results.data() + size_t(t) * top * H, weights.data() + size_t(t) * top,
                          sums.data() + size_t(t) * H, n, top, H, L.fmt.canon);
    }
    for (uint32_t t = 0; t < T; ++t) std::memcpy(mailbox(L.moe_index, int(t)) + MB_OUT, sums.data() + size_t(t) * H, H * 4);
  }
}

std::vector<float> Engine::forward_prefill(std::span<const int> tokens) {
  const uint32_t T = uint32_t(tokens.size());
  require(T > NT && T <= uint32_t(prefill_chunk_) && position_ + int(T) <= options_.capacity && !tier_ready_,
          "Vulkan large prefill window/context invalid");
  if (prefill_progress_)
    std::cerr << "PREFILL_PROGRESS stage=allocate position=" << position_ << " tokens=" << T << std::endl;
  allocate_prefill();
  struct Swap {
    Engine &engine;
    Swap(Engine &e) : engine(e) {
      for (auto &[original, alternate] : e.prefill_buffers_) original->swap(alternate);
      e.running_prefill_ = true;
    }
    ~Swap() {
      for (auto &[original, alternate] : engine.prefill_buffers_) original->swap(alternate);
      engine.running_prefill_ = false; engine.recording_prefill_ = false;
    }
  } swap(*this);
  const auto started = Clock::now();
  const auto &embedding = artifact_.at("token_embd.weight");
  for (uint32_t t = 0; t < T; ++t) {
    require(tokens[t] >= 0 && tokens[t] < m_.vocab, "Vulkan prefill token out of range");
    const auto *src = embedding.data() + size_t(tokens[t]) * embedding.bytes / m_.vocab;
    auto *out = embed_->as<float>() + size_t(t) * H;
    if (embedding.tensor->type == F32) std::memcpy(out, src, H * 4);
    else ggml_get_type_traits(ggml_type(embedding.tensor->type))->to_float(src, out, H);
  }
  auto &segments = prefill_segments_[T];
  if (segments.empty()) {recording_prefill_ = true; record_window(T, segments); recording_prefill_ = false;}
  step_->as<uint32_t>()[0] = uint32_t(position_);
  std::vector<Submission> batch(segments);
  for (auto &s : batch) {
    if (s.wait.semaphore) s.wait.value += timeline_base_;
    if (s.signal.semaphore) s.signal.value += timeline_base_;
  }
  // Submit through this layer's shared FFN, leaving the host-waiting tail
  // unqueued until the CPU work finishes. Large windows can make that host
  // work long; keeping future waits out of the submitted GPU work also
  // avoids a long queue job stalled on the host during CPU-only prefill.
  require(batch.size() == 2 * moe_layers_.size() + 1, "Vulkan prefill segment geometry invalid");
  if (profile_ && T == profile_t_) dev_.host_reset_queries(0, queries_used_);
  for (size_t i = 0; i < moe_layers_.size(); ++i) {
    if (prefill_progress_)
      std::cerr << "PREFILL_PROGRESS stage=submit layer=" << moe_layers_[i] << " tokens=" << T << std::endl;
    dev_.submit_batch(std::span<const Submission>(batch.data() + 2 * i, 2));
    const auto wait = Clock::now();
    if (!dev_.wait(timeline_, timeline_base_ + 2 * i + 1)) fatal("timed out waiting for prefill router");
    wait_ms_ += ms_since(wait);
    if (prefill_progress_)
      std::cerr << "PREFILL_PROGRESS stage=experts layer=" << moe_layers_[i] << " tokens=" << T << std::endl;
    const auto cpu = Clock::now();
    try {cpu_experts_prefill(*layers_[moe_layers_[i]], T);}
    catch (...) {dev_.signal(timeline_, timeline_base_ + 2 * moe_layers_.size() + 64); dev_.wait_idle(); throw;}
    cpu_expert_ms_ += ms_since(cpu);
    dev_.signal(timeline_, timeline_base_ + 2 * i + 2);
    if (prefill_progress_)
      std::cerr << "PREFILL_PROGRESS stage=ready layer=" << moe_layers_[i] << " tokens=" << T << std::endl;
  }
  dev_.submit_batch(std::span<const Submission>(batch.data() + 2 * moe_layers_.size(), 1));
  const auto wait = Clock::now();
  if (!dev_.wait(timeline_, timeline_base_ + 2 * moe_layers_.size() + 1)) fatal("timed out waiting for prefill logits");
  wait_ms_ += ms_since(wait);
  timeline_base_ += 2 * moe_layers_.size() + 64;
  if (profile_ && T == profile_t_) collect_profile(T);
  if (trace_ && trace_position_ >= position_ && trace_position_ < position_ + int(T)) write_trace();
  for (int i = 0; i < m_.vocab; ++i)
    if (!std::isfinite(logits_->as<float>()[i]))
      throw std::runtime_error("nonfinite Vulkan prefill logit at index=" + std::to_string(i));
  position_ += int(T); steps_ += T; step_ms_ += ms_since(started);
  return {logits_->as<float>(), logits_->as<float>() + m_.vocab};
}

void Engine::cpu_experts(const Layer &L, uint32_t T) {
  const int i = L.moe_index;
  const int top = int(m_.top_k);
  const size_t key0 = size_t(L.index) * 144;
  thread_local std::vector<uint8_t> act;
  thread_local std::vector<float> results, weights, sums;
  act.resize(size_t(T) * L.fmt.act_bytes);
  results.assign(size_t(T) * top * H, 0.f);
  weights.assign(size_t(T) * top, 0.f);
  sums.resize(size_t(T) * H);
  // Union of the window's CPU routes: one job per expert, one slot per token routed to it.
  int expert_job[144];
  std::fill(std::begin(expert_job), std::end(expert_job), -1);
  kcpu::ExpertJobMulti jobs[8 * NT];
  int n = 0;
  for (uint32_t t = 0; t < T; ++t) {
    float *mb = mailbox(i, int(t));
    int ids[16];
    float w[16];
    std::memcpy(ids, mb + MB_IDS, top * 4);
    std::memcpy(w, mb + MB_WEIGHTS, top * 4);
    uint8_t *a = act.data() + size_t(t) * L.fmt.act_bytes;
    bool quantized = false;
    for (int k = 0; k < top; ++k) {
      require(ids[k] >= 0 && ids[k] < 144, "Vulkan fast router produced an invalid expert");
      if (!tier_ready_) route_counts_[key0 + ids[k]]++;
      ++routes_;
      if (w[k] == 0.f) continue;  // dropped by min-share
      if (slot_of_host_[key0 + ids[k]] >= 0) {
        ++hits_count_;
        continue;
      }
      if (!quantized) {
        kcpu::native_quant_act(L.fmt, mb, a);
        quantized = true;
      }
      int &j = expert_job[ids[k]];
      if (j < 0) {
        j = n++;
        auto &job = jobs[j];
        job = {};
        job.expert_id = ids[k];
        job.blob = L.gate + size_t(ids[k]) * L.gate_stride;
        job.native_up = L.up + size_t(ids[k]) * L.gate_stride;
        job.native_down = L.down_experts + size_t(ids[k]) * L.down_stride;
      }
      auto &job = jobs[j];
      job.nact[job.nt] = a;
      job.out[job.nt] = results.data() + (size_t(t) * top + k) * H;
      ++job.nt;
      weights[size_t(t) * top + k] = w[k];
    }
  }
  if (n) pool_.run_layer_native(L.fmt, jobs, n, results.data(), weights.data(), sums.data(), int(T), top);
  else std::fill(sums.begin(), sums.end(), 0.f);
  for (uint32_t t = 0; t < T; ++t) {
    if (canonical_sum_ && L.fmt.canon && L.index < int(layers_.size()))
      std::memcpy(mailbox(i, int(t)) + MB_OUT, results.data() + size_t(t) * top * H, size_t(top) * H * 4);
    else std::memcpy(mailbox(i, int(t)) + MB_OUT, sums.data() + size_t(t) * H, H * 4);
  }
}

const float *Engine::forward(std::span<const int> tokens) {
  const uint32_t T = uint32_t(tokens.size());
  require(T >= 1 && T <= NT && position_ + int(T) <= options_.capacity, "Vulkan fast window/context out of bounds");
  const auto started = Clock::now();
  const auto &embedding = artifact_.at("token_embd.weight");
  for (uint32_t t = 0; t < T; ++t) {
    require(tokens[t] >= 0 && tokens[t] < m_.vocab, "Vulkan fast token out of range");
    const uint8_t *src = embedding.data() + size_t(tokens[t]) * embedding.bytes / m_.vocab;
    float *e = embed_->as<float>() + size_t(t) * H;
    if (embedding.tensor->type == F32) std::memcpy(e, src, H * 4);
    else ggml_get_type_traits(ggml_type(embedding.tensor->type))->to_float(src, e, H);
  }
  step_->as<uint32_t>()[0] = uint32_t(position_);
  std::vector<Submission> batch(segments_[T - 1]);
  for (auto &s : batch) {
    if (s.wait.semaphore) s.wait.value += timeline_base_;
    if (s.signal.semaphore) s.signal.value += timeline_base_;
  }
  const bool profiled = profile_ && T == profile_t_;
  if (profiled) dev_.host_reset_queries(0, queries_used_);
  dev_.submit_batch(batch);
  for (size_t i = 0; i < moe_layers_.size(); ++i) {
    const auto w0 = Clock::now();
    if (!dev_.wait(timeline_, timeline_base_ + 2 * i + 1)) fatal("timed out waiting for router output");
    wait_ms_ += ms_since(w0);
    const auto c0 = Clock::now();
    try {
      cpu_experts(*layers_[moe_layers_[i]], T);
    } catch (...) {
      // Release the pre-submitted GPU work before reporting the failure.
      dev_.signal(timeline_, timeline_base_ + 2 * moe_layers_.size() + 64);
      dev_.wait_idle();
      throw;
    }
    cpu_expert_ms_ += ms_since(c0);
    dev_.signal(timeline_, timeline_base_ + 2 * i + 2);
  }
  const auto w0 = Clock::now();
  if (!dev_.wait(timeline_, timeline_base_ + 2 * moe_layers_.size() + 1)) fatal("timed out waiting for logits");
  wait_ms_ += ms_since(w0);
  timeline_base_ += 2 * moe_layers_.size() + 64;
  if (profiled) collect_profile(T);
  if (trace_ && trace_position_ >= position_ && trace_position_ < position_ + int(T)) write_trace();
  window_start_ = position_;
  window_tokens_ = int(T);
  position_ += int(T);
  steps_ += T;
  step_ms_ += ms_since(started);
  return logits_->as<float>();
}

void Engine::collect_profile(uint32_t T) {
  profiled_tokens_ += T;
  const auto ts = dev_.read_timestamps(0, queries_used_);
  segment_gpu_ms_.resize(segment_names_.size());
  segment_gap_ms_.resize(segment_names_.size());
  int segment = -1;
  double begin = 0, previous_end = 0;
  for (uint32_t q = 0; q < queries_used_; ++q) {
    const auto &n = query_names_[q];
    if (n == "@begin") {
      ++segment;
      begin = ts[q];
      if (segment) segment_gap_ms_[segment] += (ts[q] - previous_end) / 1e6;
    } else if (n == "@end") {
      segment_gpu_ms_[segment] += (ts[q] - begin) / 1e6;
      previous_end = ts[q];
    } else kernel_ms_[n] += (ts[q] - ts[q - 1]) / 1e6;
  }
}

void Engine::write_trace() {
    for (size_t l = 0; l < layers_.size(); ++l) {
      const auto path = std::filesystem::path(trace_dir_) / ("layer-" + std::to_string(l) + ".bin");
      std::filesystem::create_directories(path.parent_path());
      if (std::filesystem::exists(path)) throw std::runtime_error("Vulkan fast: refusing to overwrite layer trace");
      std::ofstream f(path, std::ios::binary);
      f.write(reinterpret_cast<const char *>(trace_->as<float>() + l * 4 * H), 4 * H * 4);
      std::ofstream g(std::filesystem::path(trace_dir_) / ("attn-" + std::to_string(l) + ".bin"), std::ios::binary);
      g.write(reinterpret_cast<const char *>(trace_->as<float>() + (layers_.size() + l) * 4 * H), 4 * H * 4);
      if (!f || !g) throw std::runtime_error("Vulkan fast: layer trace write failed");
    }
}

std::vector<float> Engine::step(int token) {
  const float *l = forward(std::span(&token, 1));
  return std::vector<float>(l, l + m_.vocab);
}

void Engine::rollback(int keep) {
  require(keep >= 1 && keep <= window_tokens_ && position_ == window_start_ + window_tokens_,
          "Vulkan fast rollback outside the last window");
  if (keep == window_tokens_) return;
  // KDA state after `keep` tokens of the window sits in rollback slot `keep`.
  dev_.run([&](Recorder &r) {
    r.begin_concurrent();
    for (auto &L : layers_)
      if (L->kda) {
        const size_t s = size_t(KDA_N) * 128 * 4, h = size_t(3) * KDA_N * 3 * 4;
        r.copy(View(*L->state, keep * s), View(*L->state), s);
        r.copy(View(*L->history, keep * h), View(*L->history), h);
      } else {
        r.copy(View(*L->ring_history, size_t(keep) * L->ring->size), View(*L->ring), L->ring->size);
      }
    r.end_concurrent();
  });
  // MLA caches/pools append by position. The four-slot index ring also needs
  // restoration when rejected tokens wrapped around an unfinished pool.
  position_ = window_start_ + keep;
  window_tokens_ = keep;
}

std::vector<float> Engine::prefill(std::span<const int> tokens, std::vector<std::vector<float>> *all) {
  std::vector<float> logits;
  const size_t chunk = all ? 1 : size_t(prefill_chunk_);
  for (size_t i = 0; i < tokens.size(); i += chunk) {
    const size_t n = std::min(chunk, tokens.size() - i);
    if (n > NT) {
      logits = forward_prefill(tokens.subspan(i, n));
      if (mtp_)
        for (size_t t = 0; t < n; t += NT) mtp_forward(tokens.subspan(i + t, std::min<size_t>(NT, n - t)), false);
      continue;
    }
    const float *l = forward(tokens.subspan(i, n));
    if (all) all->emplace_back(l, l + m_.vocab);
    if (i + n == tokens.size()) logits.assign(l + (n - 1) * m_.vocab, l + n * m_.vocab);
    if (mtp_) mtp_forward(tokens.subspan(i, n), false);  // the draft block sees every verified token
  }
  release_prefill(); // release scratch before allocating the decode expert tier
  return logits;
}

std::vector<int> Engine::generate_mtp(std::vector<float> logits, int maximum, const std::vector<int> &stops, int depth) {
  require(depth >= 1 && depth < int(NT), "Vulkan fast MTP depth must be 1..3");
  if (depth > 1) return generate_mtp_deep(std::move(logits), maximum, stops, depth);
  require(mtp_ && mtp_position_ == position_, "Vulkan fast MTP generation needs a primed draft block");
  auto argmax = [&](const float *v) { return int(std::max_element(v, v + m_.vocab) - v); };
  auto stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
  std::vector<int> out, pending;
  int anchor = argmax(logits.data());
  while (int(out.size()) < maximum) {
    out.push_back(anchor);
    std::cout << anchor << '\n' << std::flush;
    if (stop(anchor) || int(out.size()) >= maximum) break;
    std::vector<int> window = pending;
    window.push_back(anchor);
    const int draft = argmax(mtp_forward(window, true));
    pending.clear();
    ++rounds_;
    if (int(out.size()) + 1 >= maximum) {  // room for the anchor only
      anchor = argmax(forward(std::span(&anchor, 1)));
      continue;
    }
    const int pair[2] = {anchor, draft};
    const float *l = forward(pair);
    ++proposed_;
    if (argmax(l) == draft) {
      ++accepted_;
      out.push_back(draft);
      std::cout << draft << '\n' << std::flush;
      if (stop(draft)) break;
      pending.push_back(draft);
      anchor = argmax(l + m_.vocab);
    } else {
      rollback(1);
      anchor = argmax(l);
    }
  }
  return out;
}

std::vector<int> Engine::generate_mtp_deep(std::vector<float> logits, int maximum, const std::vector<int> &stops, int depth) {
  require(mtp_ && mtp_position_ == position_, "Vulkan fast MTP generation needs a primed draft block");
  auto argmax = [&](const float *v) { return int(std::max_element(v, v + m_.vocab) - v); };
  auto stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
  std::vector<int> out, pending;
  int anchor = argmax(logits.data());
  while (int(out.size()) < maximum) {
    out.push_back(anchor); std::cout << anchor << '\n' << std::flush;
    if (stop(anchor) || int(out.size()) >= maximum) break;
    std::vector<int> prime = pending; prime.push_back(anchor);
    int draft = argmax(mtp_forward(prime, true)); pending.clear();
    const int saved_position = mtp_position_;
    // Ring history is mutable; caches/pools append by position. Restore the
    // ring after speculative drafting, then replay accepted drafts next round
    // against their verified target hidden states.
    dev_.run([&](Recorder &r) { r.copy(View(*mtp_->ring), View(*mtp_ring_saved_), mtp_->ring->size); });
    const int count = std::min(depth, maximum - int(out.size()));
    std::vector<int> window{anchor};
    for (int i = 0; i < count; ++i) {
      window.push_back(draft); ++proposed_;
      if (stop(draft) || i + 1 == count) break;
      draft = argmax(mtp_forward(std::span(&draft, 1), true, true));
    }
    ++rounds_;
    const float *verified = forward(window);
    int keep = 1;
    for (size_t i = 1; i < window.size(); ++i) {
      if (argmax(verified + (i - 1) * m_.vocab) != window[i]) break;
      out.push_back(window[i]); std::cout << window[i] << '\n' << std::flush;
      ++accepted_; ++keep; pending.push_back(window[i]);
      if (stop(window[i])) break;
    }
    anchor = argmax(verified + (keep - 1) * m_.vocab);
    rollback(keep);
    dev_.run([&](Recorder &r) { r.copy(View(*mtp_ring_saved_), View(*mtp_->ring), mtp_->ring->size); });
    mtp_position_ = saved_position;
    if (!out.empty() && stop(out.back())) break;
  }
  return out;
}

void Engine::report(const char *phase, int trial) {
  std::cerr << "VULKAN_FAST_PHASE phase=" << phase << " trial=" << trial << " steps=" << steps_
            << " step_ms=" << (steps_ ? step_ms_ / steps_ : 0) << " cpu_expert_ms=" << (steps_ ? cpu_expert_ms_ / steps_ : 0)
            << " wait_ms=" << (steps_ ? wait_ms_ / steps_ : 0) << " routes=" << routes_ << " gpu_hits=" << hits_count_
            << " draft_ms=" << draft_ms_ << " rounds=" << rounds_ << " proposed=" << proposed_ << " accepted=" << accepted_
            << '\n';
  draft_ms_ = 0;
  rounds_ = proposed_ = accepted_ = 0;
  if (profile_ && profiled_tokens_) {
    double attn = 0, shared = 0, head = 0, gap = 0;
    for (size_t i = 0; i < segment_names_.size(); ++i) {
      const auto &n = segment_names_[i];
      (n.starts_with("attn") ? attn : n.starts_with("shared") ? shared : head) += segment_gpu_ms_[i] / profiled_tokens_;
      gap += segment_gap_ms_[i] / profiled_tokens_;
    }
    std::cerr << "VULKAN_FAST_GPU phase=" << phase << " attn_segments_ms=" << attn << " shared_segments_ms=" << shared
              << " head_ms=" << head << " gaps_ms=" << gap << " profiled_tokens=" << profiled_tokens_
              << " window=" << profile_t_ << " first_segment_ms=" << segment_gpu_ms_[0] / profiled_tokens_ << '\n';
    for (auto &[name, ms] : kernel_ms_) std::cerr << "VULKAN_FAST_KERNEL " << name << " ms_per_token=" << ms / profiled_tokens_ << '\n';
    kernel_ms_.clear();
    std::fill(segment_gpu_ms_.begin(), segment_gpu_ms_.end(), 0);
    std::fill(segment_gap_ms_.begin(), segment_gap_ms_.end(), 0);
  }
  profiled_tokens_ = 0;
  steps_ = 0;
  step_ms_ = cpu_expert_ms_ = wait_ms_ = 0;
  routes_ = hits_count_ = 0;
}

} // namespace strata::vulkan::fast
