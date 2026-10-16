#include "fast/prefill_schedule.hpp"
#include <iostream>
#include <limits>
using namespace strata::vulkan::fast;
int main() {
  try {
    auto require=[](bool x){if(!x)throw std::runtime_error("prefill scheduler assertion failed");};
    std::array<uint32_t,144> counts{};
    require(schedule_prefill(counts,144,1,{10,1,0,0}).gpu.empty());
    counts[0]=counts[1]=counts[2]=counts[3]=100;
    const auto balanced=schedule_prefill(counts,144,1,{1,1,0,0});
    require(balanced.gpu==std::vector<int>({0,1}));
    require(balanced.cpu_ms==0.2 && balanced.gpu_ms==0.2);
    require(schedule_prefill(counts,0,1,{1,1,0,0}).gpu.empty());
    require(schedule_prefill(counts,144,1,{1,1,10000,0}).gpu.empty());
    require(schedule_prefill(counts,144,1,{100,0.01,0,0}).gpu.size()==4);
    counts[9]=10000;
    require(schedule_prefill(counts,1,1,{100,0.01,0,0}).gpu==std::vector<int>({9}));
    for(double bad:{-1.,0.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected=false;try{schedule_prefill(counts,144,1,{bad,1,0,0});}catch(const std::invalid_argument&){rejected=true;}
      require(rejected);
    }
    std::cout<<"prefill scheduler checks passed\n";
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
