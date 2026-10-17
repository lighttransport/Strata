#pragma once
#include "fast/kernels.hpp"
namespace strata::vulkan::fast {
// Gather a small query batch once, then share keys across all 64 heads in
// two FP32 GEMMs, reading the same buffer transposed for values. Scratch is
// independent of the logical prefill length.
class MlaPrefill {
public:
  explicit MlaPrefill(Device &device) : device_(device) {}
  void run(Recorder &, View cache, View queries, View ids, View output,
           uint32_t tokens, uint32_t position, float scale, uint32_t istride = 2176,
           bool bounded_keys = true);
private:
  Device &device_;
  Matrix keys_;
  std::unique_ptr<Buffer> scores_;
};
}
