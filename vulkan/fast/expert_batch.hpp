#pragma once
#include "fast/kernels.hpp"
#include <span>
namespace strata::vulkan::fast {

struct ExpertBatchTrace {
  std::vector<float> qinput, gate_up, hidden, qhidden;
};
struct ExpertBatchStats {
  double pack_ms = 0, upload_ms = 0, queued_work_ms = 0;
  uint64_t input_bytes = 0;
};

// Scratch for one Q2_K gate/up (4096 -> 2048), Q3_K down (2048 -> 4096)
// frequently routed expert at a time during prefill.
// Inputs/outputs retain the router's token order; the caller combines them
// after both CPU and GPU work have completed. Optional GPU packing copies
// native weight bytes first and converts their layout in the product command.
class ExpertBatch {
public:
  ExpertBatch(Device &, uint32_t capacity, bool portable_rounding = false, bool gpu_pack = false,
              bool vector_weights = false);
  void run(const uint8_t *gate, const uint8_t *up, const uint8_t *down,
           std::span<const float *const> inputs, std::span<float *const> outputs,
           float swiglu_limit, ExpertBatchTrace *trace = nullptr);
  // Inputs are canonical native Q8_K rows of 4096 elements, already prepared
  // by the CPU pool. Only their 4672-byte payload is copied per token.
  void run_quantized(const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                     std::span<const uint8_t *const> inputs, std::span<float *const> outputs,
                     float swiglu_limit, ExpertBatchTrace *trace = nullptr);
  const ExpertBatchStats &stats() const { return stats_; }
private:
  Device &device_;
  uint32_t capacity_;
  bool portable_rounding_, gpu_pack_, vector_weights_;
  ExpertBatchStats stats_;
  size_t quantized_row_bytes_ = 0;
  Layout down_layout_;
  std::vector<uint8_t> packed_;
  Matrix gu_weights_, down_weights_;
  std::unique_ptr<Buffer> input_, qinput_, gu_, hidden_, qhidden_, output_, hits_;
  std::unique_ptr<Buffer> input_meta_, hidden_meta_, partial_;
  std::unique_ptr<Buffer> raw_gu_, raw_down_;
  void run_impl(const uint8_t *, const uint8_t *, const uint8_t *,
                std::span<const float *const>, std::span<const uint8_t *const>,
                std::span<float *const>, float, ExpertBatchTrace *);
};
} // namespace strata::vulkan::fast
