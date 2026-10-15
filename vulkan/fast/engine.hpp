#pragma once
// Opt-in fast GLM-5.3f engine (strata-glm-vulkan --engine fast): one Vulkan
// device, repacked resident dense weights, command buffers recorded once per
// layer segment and resubmitted per token, and one host handoff per MoE layer
// for the CPU routed experts (timeline semaphores).
#include "fast/device.hpp"
#include "fast/kernels.hpp"
#include "strata/core/model.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <map>
#include <memory>
#include <span>
#include <vector>
namespace strata::vulkan::fast {

class ExpertBatch;

struct EngineOptions {
  int threads = 12;
  int capacity = 4096;
  size_t budget = 15ULL << 30;
  size_t tier_bytes = 0;  // GPU-resident routed experts (decode cache)
  bool mtp = false;       // load the draft block for speculative decoding
  std::string device;
};

class Engine {
public:
  Engine(core::ModelArtifact &, const core::ModelDescriptor &, const EngineOptions &);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  static constexpr uint32_t NT = 4;  // largest token window per pass
  // Runs a window of 1..NT tokens from the current position; returns logits
  // [T][vocab] valid until the next call.
  const float *forward(std::span<const int> tokens);
  // Keeps the first `keep` tokens of the last window (verification rejection).
  void rollback(int keep);
  // Decodes one token at the current position and returns its logits.
  std::vector<float> step(int token);
  std::vector<float> prefill(std::span<const int> tokens, std::vector<std::vector<float>> *all = nullptr);
  // Runs the draft block over a window at the draft position; with head,
  // returns the logits of the last token's next-token prediction.
  const float *mtp_forward(std::span<const int> tokens, bool head, bool chain = false);
  // Greedy decoding with one MTP draft per round, verified by the target (lossless).
  std::vector<int> generate_mtp(std::vector<float> logits, int maximum, const std::vector<int> &stops, int depth = 1);
  void reset();
  // Fills the GPU expert tier from the routes counted since reset (prompt).
  void prepare_tier();
  size_t tier_slots() const { return tier_used_; }
  int position() const { return position_; }
  void report(const char *phase, int trial);

private:
  struct Layer;
  core::ModelArtifact &artifact_;
  const core::ModelDescriptor &m_;
  EngineOptions options_;
  Device dev_;
  kernels::cpu::ExpertPool pool_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::unique_ptr<Layer> mtp_;
  std::unique_ptr<Buffer> mtp_ring_saved_;
  std::unique_ptr<Buffer> cur_, join_, hidden_ring_, mtp_step_, mtp_embed_, mtp_logits_;
  std::vector<Submission> mtp_segments_[2][4];
  int mtp_position_ = 0;
  double draft_ms_ = 0;
  uint64_t rounds_ = 0, proposed_ = 0, accepted_ = 0;
  std::vector<float> constants_host_;
  std::unique_ptr<Buffer> constants_;
  Matrix head_;
  size_t output_norm_ = 0;
  // Activations (device) and host-shared buffers.
  std::unique_ptr<Buffer> r_, proj_, c_attn_, c_ffn_, xn_, y_, s_, v_, d_, g_, k_, m0_, mq_, mqb_,
      miq_, mg_, maq_, mav_, mval_, scores_, ids_, gu_, h_, f_, logits144_, part_, dummy_, slot_of_, hits_;
  std::unique_ptr<Buffer> step_, embed_, mailbox_, logits_, trace_;
  std::string trace_dir_;
  int trace_position_ = 0;
  uint32_t recording_width_ = 1;
  std::unique_ptr<Buffer> qx_;
  bool q8_emulate_ = false, profile_ = false, detail_ = false;
  bool expert_q8_ = false;
  bool canonical_sum_ = false;
  bool prefill_gpu_ = false;
  bool prefill_gpu_pack_ = false;
  bool prefill_gpu_vector_ = false;
  bool prefill_reuse_expert_ = false;
  bool prefill_match_gemv_ = false;
  bool prefill_progress_ = false;
  bool prefill_split_prefix_ = false;
  uint32_t prefill_submit_limit_ms_ = 500;
  uint32_t prefill_tile_ = 0;
  int prefill_gpu_max_experts_ = 72, prefill_gpu_min_tokens_ = 32;
  uint32_t prefill_gpu_tile_ = 64;
  uint32_t prefill_gpu_rows_ = 64;
  uint32_t mailbox_stride_ = 8448;
  std::unique_ptr<Buffer> expert_x_, expert_h_;
  uint32_t q8_slot_ = 0;
  uint32_t profile_t_ = 1;
  uint32_t hidden_slots_ = 8;
  bool recording_prefill_ = false, running_prefill_ = false;
  std::vector<std::pair<std::unique_ptr<Buffer> *, std::unique_ptr<Buffer>>> prefill_buffers_;
  std::map<uint32_t, std::vector<Submission>> prefill_segments_;
  std::unique_ptr<ExpertBatch> prefill_expert_batch_;
  uint64_t profiled_tokens_ = 0;
  std::vector<double> segment_gpu_ms_, segment_gap_ms_;
  std::vector<std::string> segment_names_, query_names_;
  std::map<std::string, double> kernel_ms_;
  uint32_t queries_used_ = 0;
  void collect_profile(uint32_t T);
  void mv(Recorder &, const Matrix &, View x, View y, GemvArgs = {});
  VkSemaphore timeline_ = VK_NULL_HANDLE;
  uint64_t timeline_base_ = 0;
  std::vector<Submission> segments_[NT];  // per window size, in order; semaphore values relative to the base
  uint32_t score_stride_ = 0;
  int window_start_ = 0, window_tokens_ = 0, prefill_chunk_ = 4;
  std::vector<int> moe_layers_;
  int position_ = 0;
  // Statistics.
  double cpu_expert_ms_ = 0, wait_ms_ = 0, step_ms_ = 0;
  uint64_t steps_ = 0, routes_ = 0, hits_count_ = 0;

  size_t constant(std::span<const float>);
  size_t constant(const std::string &tensor);
  void load();
  void allocate();
  void record(uint32_t T);
  void record_window(uint32_t T, std::vector<Submission> &);
  void allocate_prefill();
  void release_prefill();
  std::vector<float> forward_prefill(std::span<const int>);
  void cpu_experts_prefill(const Layer &, uint32_t T);
  void record_attention(Recorder &, int layer, uint32_t T);
  void record_ffn_head(Recorder &, int layer, uint32_t T);  // through the router (MoE) or the write (dense)
  void record_ffn_tail(Recorder &, int layer, uint32_t T);  // MoE: combine and write
  void record_shared(Recorder &, const Layer &, uint32_t T);
  void record_mla(Recorder &, const Layer &, uint32_t T, const Buffer &step, bool cache_only = false);
  void record_route(Recorder &, const Layer &, uint32_t T);
  void record_mtp(uint32_t T, bool head);
  std::vector<int> generate_mtp_deep(std::vector<float>, int maximum, const std::vector<int> &stops, int depth);
  void record_head(Recorder &, uint32_t T);
  void record_trace(Recorder &, int layer);
  void read_mhc(Recorder &, const Layer &, int kind, uint32_t T);
  void write_mhc(Recorder &, const Buffer &c, View ya, View yb, uint32_t inputs, uint32_t ybofs, uint32_t T,
                 View yc = {}, uint32_t ybstride = 0);
  void record_trace_attention(Recorder &, int layer);
  void write_trace();
  void cpu_experts(const Layer &, uint32_t T);
  // Expert tier.
  Layout gate_layout_, down_layout_;
  size_t slot_bytes_ = 0, slots_per_arena_ = 0, tier_capacity_ = 0, tier_used_ = 0;
  std::vector<std::unique_ptr<Buffer>> arenas_;
  std::unique_ptr<Buffer> slot_table_, egu_, eh_, eout_, esum_, xsum_;
  std::vector<int> slot_of_host_;
  std::vector<uint64_t> route_counts_;
  std::vector<double> expert_prior_;
  bool tier_ready_ = false;
  float affinity_ = 0, min_share_ = 0;
  void allocate_tier();
  void record_experts(Recorder &, int layer, uint32_t T);
  float *mailbox(int moe_index, int token) const;
};

} // namespace strata::vulkan::fast
