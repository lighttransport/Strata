// Small, bounded RX570 calibration workloads. Timings include submission/waits.
#include "projections.hpp"
#include "runtime.hpp"
#include "model_checks.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <iomanip>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: glm_vulkan_bench MODEL");
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &model = strata::vulkan::checked_model(artifact);
    strata::vulkan::Projections projections(0, 768ULL*1024*1024);
    strata::vulkan::Compute compute(projections.device_name());
    std::cout << std::setprecision(10);
    for (auto &m : compute.benchmark_transfers(64ULL*1024*1024, 7)) {
      std::cout << "{\"kind\":\"transfer\",\"direction\":\"" << m.direction
                << "\",\"payload_bytes\":" << m.bytes << ",\"source_flags\":" << m.source_flags
                << ",\"destination_flags\":" << m.destination_flags << ",\"milliseconds\":[";
      for (size_t i=0;i<m.milliseconds.size();++i) std::cout << (i?",":"") << m.milliseconds[i];
      std::cout << "]}\n";
    }
    std::set<int> types;
    for (const auto &[name,t] : artifact.tensors()) {
      if (!name.starts_with("blk.") || t.tensor->shape.size()<2 ||
          t.tensor->shape[0]<256 || t.tensor->shape[1]<128) continue;
      const auto type=ggml_type(t.tensor->type);
      if (!types.insert(int(type)).second) continue;
      const int in=int(t.tensor->shape[0]), out=std::min<int>(4096,t.tensor->shape[1]);
      const size_t bytes=ggml_row_size(type,in)*out;
      for (int nt : {1,8,64,256}) {
        std::vector<float> x(size_t(in)*nt);
        for(size_t i=0;i<x.size();++i) x[i]=float(int(i%101)-50)/100.f;
        for (int repeat=-1;repeat<5;++repeat) {
          const auto start=Clock::now();
          const double c=projections.compute_ms, tr=projections.transfer_ms;
          auto y=projections.synthetic("bench-"+std::to_string(int(type)),type,
                   {t.data(),bytes},in,out,x,nt);
          const double ms=elapsed(start);
          if(!std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);}))
            throw std::runtime_error("nonfinite projection benchmark output");
          if(repeat>=0) std::cout << "{\"kind\":\"projection\",\"type\":" << int(type)
             << ",\"in\":" << in << ",\"out\":" << out << ",\"tokens\":" << nt
             << ",\"weight_bytes\":" << bytes << ",\"repeat\":" << repeat
             << ",\"milliseconds\":" << ms << ",\"compute_ms\":" << projections.compute_ms-c
             << ",\"transfer_ms\":" << projections.transfer_ms-tr << "}\n";
        }
      }
    }
    for (int layer : {3,11,12,44}) {
      const std::string p="blk."+std::to_string(layer)+".";
      const auto &g=artifact.at(p+"ffn_gate_exps.weight"), &u=artifact.at(p+"ffn_up_exps.weight"),
                 &d=artifact.at(p+"ffn_down_exps.weight");
      strata::kernels::cpu::NativeFmt fmt; std::string error;
      if(!strata::kernels::cpu::native_fmt(g.tensor->type,d.tensor->type,model.hidden,
                                           model.layers[layer].intermediate,fmt,error)) throw std::runtime_error(error);
      fmt.canon=(g.tensor->type==10 || g.tensor->type==11)&&(d.tensor->type==10 || d.tensor->type==11);
      fmt.swiglu_limit=model.layers[layer].swiglu_limit;
      std::vector<float> x(model.hidden,.01f), y(model.hidden*model.top_k);
      std::vector<uint8_t> act(fmt.act_bytes);
      strata::kernels::cpu::native_quant_act(fmt,x.data(),act.data());
      for(int workers : {8,10,12,14}) {
        strata::kernels::cpu::ExpertPool pool(workers,true,true);
        std::vector<strata::kernels::cpu::ExpertJobMulti> jobs(model.top_k);
        for(int repeat=-8;repeat<7;++repeat) {
          // Rotate slices to avoid turning the test into an L3-cache benchmark.
          for(int j=0;j<model.top_k;++j) {
            auto &job=jobs[j]; const int id=((repeat+8)*model.top_k+j)%(8*model.top_k);
            job.expert_id=id; job.nt=1; job.blob=g.data()+id*g.bytes/model.experts;
            job.native_up=u.data()+id*u.bytes/model.experts; job.native_down=d.data()+id*d.bytes/model.experts;
            job.nact[0]=act.data(); job.out[0]=y.data()+j*model.hidden;
          }
          const auto start=Clock::now();pool.run_split_multi_native(fmt,jobs.data(),jobs.size());
          const double ms=elapsed(start);
          if(!std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);})) throw std::runtime_error("nonfinite CPU expert output");
          if(repeat>=0) std::cout << "{\"kind\":\"cpu_experts\",\"layer\":" << layer
             << ",\"workers\":" << workers << ",\"bytes\":" << (g.bytes+u.bytes+d.bytes)/model.experts*model.top_k
             << ",\"milliseconds\":" << ms << ",\"repeat\":" << repeat << "}\n";
        }
      }
    }
    return 0;
  } catch(const std::exception &e) {std::cerr << e.what() << '\n';return 1;}
}
