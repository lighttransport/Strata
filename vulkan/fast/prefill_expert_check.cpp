#include "fast/expert_batch.hpp"
#include "fast/cpu_prefill.hpp"
#include "model_checks.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <ggml.h>
namespace fast = strata::vulkan::fast;
namespace cpu = strata::kernels::cpu;
int main(int argc, char **argv) {
  try {
    int panel = 0;
    if (argc == 4 && std::string(argv[2]) == "--cpu-panel") panel = std::stoi(argv[3]);
    else if (argc != 2) throw std::invalid_argument("usage: prefill_expert_check MODEL [--cpu-panel 32|64]");
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &model = strata::vulkan::checked_model(artifact);
    const auto &G = artifact.at("blk.3.ffn_gate_exps.weight"), &U = artifact.at("blk.3.ffn_up_exps.weight"),
               &D = artifact.at("blk.3.ffn_down_exps.weight");
    cpu::NativeFmt fmt; std::string error;
    if (!cpu::native_fmt(G.tensor->type, D.tensor->type, 4096, 2048, fmt, error)) throw std::runtime_error(error);
    fmt.canon = true; fmt.swiglu_limit = model.layers[3].swiglu_limit;
    std::unique_ptr<fast::Device> device;
    std::unique_ptr<fast::ExpertBatch> slots[2];
    std::unique_ptr<fast::CpuPrefill> host;
    if (panel) host = std::make_unique<fast::CpuPrefill>(2, panel);
    else {
      device = std::make_unique<fast::Device>("", 512ULL << 20);
      for (auto &slot : slots) slot = std::make_unique<fast::ExpertBatch>(*device, 256, false, true, true, 64, true, 64, true);
    }
    std::mt19937 rng(7741); std::normal_distribution<float> normal(0, 0.5f);
    size_t failures = 0;
    for (int expert : {5,77}) for (int T : {1,5,17,33,65,129,257}) {
      const auto *gate = G.data()+size_t(expert)*G.bytes/144, *up = U.data()+size_t(expert)*U.bytes/144,
                 *down = D.data()+size_t(expert)*D.bytes/144;
      std::vector<float> input(size_t(T)*4096), actual(size_t(T)*4096+64,12345.f), expected(size_t(T)*4096), hidden(2048);
      std::vector<uint8_t> quantized(size_t(T)*fmt.act_bytes), hq(fmt.h_bytes);
      std::vector<const uint8_t *> inputs;
      std::vector<float *> outputs;
      std::vector<cpu::ExpertJobMulti> jobs;
      for (auto &value : input) value = normal(rng);
      for (int t=0;t<T;++t) {
        auto *act = quantized.data()+size_t(t)*fmt.act_bytes;
        cpu::native_quant_act(fmt,input.data()+size_t(t)*4096,act);
        const void *ap[] = {act}; float *hp[] = {hidden.data()};
        cpu::native_gu_rows(fmt,gate,ap,1,hp,0,2048,up);
        cpu::native_quant_h(fmt,hidden.data(),hq.data());
        const void *hqp[] = {hq.data()}; float *out[] = {expected.data()+size_t(t)*4096};
        cpu::native_down_rows(fmt,gate,hqp,1,out,0,4096,down);
        inputs.push_back(act); outputs.push_back(actual.data()+size_t(t)*4096);
        if (jobs.empty() || jobs.back().nt == cpu::MAXT) {
          jobs.emplace_back(); auto &j=jobs.back();j.expert_id=expert;j.blob=gate;j.native_up=up;j.native_down=down;
        }
        auto &j=jobs.back();j.nact[j.nt]=act;j.out[j.nt]=outputs.back();++j.nt;
      }
      if (host) host->run(fmt,jobs);
      else {
        auto &slot = slots[(expert+T)%2];
        slot->prepare_weights_async(gate,up,down);
        slot->run_quantized_panels(gate,up,down,inputs,outputs,fmt.swiglu_limit,true);
      }
      double norm=0, error2=0;size_t unequal=0;
      for (size_t i=0;i<expected.size();++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite expert output");
        norm+=double(expected[i])*expected[i]; error2+=double(actual[i]-expected[i])*(actual[i]-expected[i]);
        unequal+=std::memcmp(&actual[i],&expected[i],4)!=0;
      }
      for (size_t i=expected.size();i<actual.size();++i) if(actual[i]!=12345.f) throw std::runtime_error("expert output overrun");
      const double relative=std::sqrt(error2/std::max(norm,1e-30));
      std::cout << "{\"expert\":" << expert << ",\"tokens\":" << T << ",\"relative_l2\":" << relative << ",\"unequal\":" << unequal << "}" << std::endl;
      failures += relative > 2e-4 || (!host && unequal);
    }
    if(failures) throw std::runtime_error("prefill expert numerical gate failed");
    return 0;
  } catch(const std::exception &e) {std::cerr << e.what() << '\n';return 1;}
}
