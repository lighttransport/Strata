// Bounded rotating expert benchmark: exact W4A16 or opt-in Q8 activations.
#include "strata/artifact/glm_modelopt.hpp"
#include "strata/kernels/cpu/glm_fp4_exact.hpp"
#include "strata/net/glm_fp4_worker.hpp"
#include <chrono>
#include <iostream>
#include <memory>
int main(int argc,char** argv){try{
    if(argc<2||argc>7)throw std::invalid_argument("usage: exact-bench MODEL [threads=14] [spin=0] [repeats=64] [experts=16] [remote-root]");
    const unsigned threads=argc>2?std::stoul(argv[2]):14,spin=argc>3?std::stoul(argv[3]):0;
    const unsigned repeats=argc>4?std::stoul(argv[4]):64,count=argc>5?std::stoul(argv[5]):16;
    if(!threads||threads>32||spin>1||!repeats||repeats>4096||!count||count>42)throw std::invalid_argument("invalid benchmark bounds");
    strata::artifact::Safetensors tensors(argv[1],true);
    auto config=strata::artifact::read_json((std::filesystem::path(argv[1])/"config.json").string());
    strata::artifact::GlmModelOpt model(tensors,config);
    using Expert=strata::kernels::cpu::GlmFp4ExactExpert;
    const bool q8=strata_env_is_one("STRATA_GLM_FP4_BENCH_Q8");
    if(q8&&argc==7)throw std::invalid_argument("Q8 benchmark is local only; wire arithmetic remains exact");
    const auto load_begin=std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<Expert>> experts;
    for(unsigned i=0;i<count;++i)experts.push_back(std::make_unique<Expert>(model,3+i,0,false,q8||strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE")));
    std::cout<<"{\"load_expert_us\":"<<std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-load_begin).count()/count<<"}\n";
    const bool reload=strata_env_is_one("STRATA_GLM_FP4_BENCH_RELOAD");
    size_t cached=0;for(auto& e:experts)cached+=e->scale_bytes;
    std::cout<<"{\"scale_cache_bytes\":"<<cached<<",\"experts\":"<<count<<"}\n";
    std::vector<float> input(4096,1.f/4096);
    if(q8)for(size_t j=0;j<input.size();++j)input[j]=0.1f*std::sin(float(j)*0.173f);
    const float clamp=config.at("text_config").at("swiglu_limit").real();
    std::vector<std::vector<float>> reference;
    {strata::cpu::exl3::Pool scalar(1,false);for(unsigned i=0;i<count;++i){Expert logical(model,3+i,0,false,false);reference.push_back(logical.run(scalar,input,1,clamp));}}
    std::unique_ptr<strata::net::GlmFp4WorkerConnection> remote;
    std::unique_ptr<strata::cpu::exl3::Pool> pool;
    if(argc==7){auto report=strata::artifact::read_json((std::filesystem::path(argv[1])/"partition.json").string());
        remote=std::make_unique<strata::net::GlmFp4WorkerConnection>(argv[6],report.at("layout_sha256").string(),report.at("index_sha256").string(),false,false);
    }else pool=std::make_unique<strata::cpu::exl3::Pool>(threads,true,spin);
    auto run=[&](unsigned i){if(remote)return remote->run(3+i,0,input);
        if(reload){Expert fresh(model,3+i,0,false,q8||strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE"));return q8?fresh.run_q8(*pool,input,1,clamp):fresh.run(*pool,input,1,clamp);}
        return q8?experts[i]->run_q8(*pool,input,1,clamp):experts[i]->run(*pool,input,1,clamp);};
    double squared_error=0,squared_reference=0,max_error=0;
    for(unsigned i=0;i<count;++i){auto out=run(i);
        if(!q8&&std::memcmp(out.data(),reference[i].data(),out.size()*4))throw std::runtime_error("thread profile changed exact output");
        for(size_t j=0;j<out.size();++j){double d=double(out[j])-reference[i][j];squared_error+=d*d;squared_reference+=double(reference[i][j])*reference[i][j];max_error=std::max(max_error,std::abs(d));}
    }
    std::cout<<"{\"q8\":"<<q8<<",\"relative_l2_error\":"<<std::sqrt(squared_error/std::max(squared_reference,1e-300))<<",\"max_absolute_error\":"<<max_error<<"}\n";
    auto now=[](){return std::chrono::steady_clock::now();};
    auto us=[&](auto start,unsigned n){return std::chrono::duration<double,std::micro>(now()-start).count()/n;};
    auto begin=now();for(unsigned i=0;i<256;++i)if(pool)pool->run([](size_t){});
    std::cout<<"{\"threads\":"<<threads<<",\"spin\":"<<spin<<",\"wake_us\":"<<us(begin,256)<<"}\n";
    double checksum=0;
    for(unsigned sample=0;sample<3;++sample){
        begin=now();for(unsigned i=0;i<repeats;++i){auto out=run(i%count);checksum+=out[0];}
        std::cout<<"{\"threads\":"<<threads<<",\"spin\":"<<spin<<",\"sample\":"<<sample<<",\"expert_us\":"<<us(begin,repeats)
                 <<",\"reload\":"<<reload<<",\"remote\":"<<bool(remote)<<",\"rotating_experts\":"<<count<<",\"checkpoint_codes_bytes\":"<<uint64_t(count)*3*4096*2048/2<<",\"exact_output\":"<<(!q8)<<"}\n";
    }
    std::cerr<<"checksum="<<checksum<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
