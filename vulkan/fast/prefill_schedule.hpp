#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::vulkan::fast {
struct PrefillCosts {
  double cpu_route_us, gpu_route_us, upload_us, shared_us;
};
struct PrefillAssignment {
  std::vector<int> gpu;
  double cpu_ms = 0, gpu_ms = 0;
};
// Fixed route assignment, two upload slots, one compute queue. The candidate
// order is deterministic. Every prefix is evaluated, including CPU-only.
inline PrefillAssignment schedule_prefill(const std::array<uint32_t,144> &counts,
                                         uint32_t maximum, uint32_t minimum, PrefillCosts cost) {
  if (maximum > 144 || !minimum || !std::isfinite(cost.cpu_route_us) || cost.cpu_route_us <= 0 ||
      !std::isfinite(cost.gpu_route_us) || cost.gpu_route_us <= 0 ||
      !std::isfinite(cost.upload_us) || cost.upload_us < 0 ||
      !std::isfinite(cost.shared_us) || cost.shared_us < 0)
    throw std::invalid_argument("invalid prefill scheduling costs");
  std::array<int,144> order{};
  double cpu = 0;
  for (int e=0;e<144;++e) {order[e]=e;cpu+=counts[e]*cost.cpu_route_us;}
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return counts[a]>counts[b];});
  PrefillAssignment best{{},cpu/1000,cost.shared_us/1000};
  double best_end=std::max(cpu,cost.shared_us), transfer=0, compute=cost.shared_us;
  std::array<double,144> complete{};
  for (uint32_t i=0;i<maximum && counts[order[i]]>=minimum;++i) {
    transfer=std::max(transfer,i>=2?complete[i-2]:0)+cost.upload_us;
    compute=std::max(compute,transfer)+counts[order[i]]*cost.gpu_route_us;
    complete[i]=compute; cpu-=counts[order[i]]*cost.cpu_route_us;
    const double end=std::max(cpu,compute);
    if(end<best_end) {best_end=end;best.gpu.assign(order.begin(),order.begin()+i+1);best.cpu_ms=std::max(0.,cpu)/1000;best.gpu_ms=compute/1000;}
  }
  return best;
}
}
