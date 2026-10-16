#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <memory>
#include <span>
namespace strata::vulkan::fast {
// Prefill-only FP32 matrix panels; the existing native/LUT executor remains
// the default. Owns persistent workers and releases expanded weights on exit.
class CpuPrefill {
public:
  CpuPrefill(int workers, int panel);
  ~CpuPrefill();
  void run(const kernels::cpu::NativeFmt &, std::span<const kernels::cpu::ExpertJobMulti>);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
