#include "strata/kernels/cpu/fp4_local.hpp"
#include <bit>
#include <iostream>
#include <limits>
#include <chrono>
#include <algorithm>
using strata::artifact::GlmFp4Matrix;
using strata::kernels::cpu::Fp4LocalCpu;
template<class F> void rejects(F fn) {
    try{fn();}catch(const std::invalid_argument&){return;}
    throw std::runtime_error("invalid input was accepted");
}
void check(const GlmFp4Matrix& m,int tokens) {
    std::vector<float> x(size_t(tokens)*m.cols);
    for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+i/m.cols*19)%257)-128)/128;
    Fp4LocalCpu serial(m,1),parallel(m,4);
    auto expected=serial.run(x,tokens,true),got=parallel.run(x,tokens,true);
    if(std::memcmp(expected.data(),got.data(),got.size()*sizeof(float)))throw std::runtime_error("thread count changed output");
    // Original adjacent-layout data independently checks constructor padding and packing.
    for(int t=0;t<tokens;++t)for(size_t r=0;r<m.rows;++r) {
        double ref=0,norm=0;
        for(size_t j=0;j<m.cols;++j) {
            const size_t at=r*m.cols+j;
            const uint8_t code=(m.codes[at/2]>>((j%2)*4))&15;
            auto bits=std::bit_cast<uint32_t>(x[size_t(t)*m.cols+j]);bits+=0x7fff+((bits>>16)&1);
            const double product=double(strata_glm_e2m1(code))*strata_glm_e4m3(m.scales[at/16])*std::bit_cast<float>(bits&0xffff0000u);
            ref+=product;norm+=std::abs(product);
        }
        ref*=m.global;norm*=m.global;
        const double steps=((m.cols+31)/32*32)+2,eps=std::numeric_limits<float>::epsilon()/2;
        if(std::abs(double(got[size_t(t)*m.rows+r])-ref)>2*steps*eps/(1-steps*eps)*norm+1e-7)
            throw std::runtime_error("original-layout reference mismatch");
    }
    // Per-token calls exercise padded input strides independently of the batch.
    for(int t=0;t<tokens;++t) {
        std::vector<float> one(x.begin()+size_t(t)*m.cols,x.begin()+size_t(t+1)*m.cols);
        auto y=serial.run(one,1,false);
        if(std::memcmp(y.data(),got.data()+size_t(t)*m.rows,y.size()*sizeof(float)))throw std::runtime_error("batch changed output");
    }
    if(m.cols<=128) {
        auto other=m;other.rows+=1;other.global*=2;
        other.codes.resize(other.rows*other.cols/2,0x71);other.scales.resize(other.rows*other.cols/16,39);
        Fp4LocalCpu second(other,4);
        auto pair=parallel.run_pair(second,x,tokens,true);
        if(std::memcmp(pair[0].data(),got.data(),got.size()*sizeof(float)))throw std::runtime_error("paired first output mismatch");
    }
    std::cout<<"PASS projection rows="<<m.rows<<" cols="<<m.cols<<" tokens="<<tokens<<" threads=1,4\n";
}
void bench(const char* model) {
    strata::artifact::Safetensors ts(model);
    for(int layer:{3,4})for(const char* projection:{"gate_proj","up_proj","down_proj"}) {
        auto w=strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+projection);
        for(int tokens:{1,4}) {
            std::vector<float> x(size_t(tokens)*w.cols);
            for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+i/w.cols*19)%257)-128)/128;
            Fp4LocalCpu reference(w,1);
            const auto expected=reference.run(x,tokens,true);
            const int threads[]={1,4,16};
            std::vector<Fp4LocalCpu> backends;
            for(int n:threads)backends.emplace_back(w,n);
            std::vector<double> wall[3],kernel[3];
            for(auto& backend:backends)if(backend.run(x,tokens,false)!=expected)throw std::runtime_error("benchmark warmup mismatch");
            for(int repeat=0;repeat<7;++repeat)for(int order=0;order<3;++order) {
                int mode=(repeat+order)%3;float ms=-1;
                // Establish this OpenMP team size before timing a steady-state call.
                backends[mode].run(x,tokens,false);
                const auto start=std::chrono::steady_clock::now();
                auto got=backends[mode].run(x,tokens,false,&ms);
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                if(!std::isfinite(ms)||ms<0||std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)))
                    throw std::runtime_error("benchmark timing or output mismatch");
                wall[mode].push_back(elapsed);kernel[mode].push_back(ms);
            }
            for(int mode=0;mode<3;++mode) {
                std::cout<<"layer="<<layer<<" projection="<<projection<<" tokens="<<tokens<<" threads="<<threads[mode]<<" wall_ms=";
                for(double ms:wall[mode])std::cout<<ms<<",";
                std::cout<<" kernel_ms=";for(double ms:kernel[mode])std::cout<<ms<<",";
                std::sort(wall[mode].begin(),wall[mode].end());std::sort(kernel[mode].begin(),kernel[mode].end());
                std::cout<<" median_wall_ms="<<wall[mode][3]<<" median_kernel_ms="<<kernel[mode][3]<<"\n";
            }
        }
    }
    std::cout<<"PASS measured CPU projections match verified outputs; no full-model throughput claim\n";
}
void pair_bench(const char* model) {
    strata::artifact::Safetensors ts(model);
    for(int layer:{3,4}) {
        auto prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.0.";
        auto g=strata::artifact::glm_load_fp4(ts,prefix+"gate_proj"),u=strata::artifact::glm_load_fp4(ts,prefix+"up_proj");
        for(int tokens:{1,4}) {
            std::vector<float> x(size_t(tokens)*g.cols);
            for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+i/g.cols*19)%257)-128)/128;
            const auto expected=Fp4LocalCpu(g,1).run_pair(Fp4LocalCpu(u,1),x,tokens,true);
            for(int threads:{4,16}) {
                Fp4LocalCpu gate(g,threads),up(u,threads);
                auto execute=[&](int mode) {
                    if(mode)return gate.run_pair(up,x,tokens,false);
                    return std::array<std::vector<float>,2>{gate.run(x,tokens,false),up.run(x,tokens,false)};
                };
                std::vector<double> samples[2];
                for(int mode=0;mode<2;++mode)execute(mode);
                for(int r=0;r<7;++r)for(int order=0;order<2;++order) {
                    int mode=(r+order)%2;const auto start=std::chrono::steady_clock::now();
                    auto got=execute(mode);
                    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    for(int p=0;p<2;++p)if(std::memcmp(got[p].data(),expected[p].data(),got[p].size()*sizeof(float)))throw std::runtime_error("pair benchmark output mismatch");
                    samples[mode].push_back(ms);
                }
                for(int mode=0;mode<2;++mode) {
                    std::cout<<"layer="<<layer<<" tokens="<<tokens<<" threads="<<threads<<" paired="<<mode<<" ms=";
                    for(double ms:samples[mode])std::cout<<ms<<",";
                    std::sort(samples[mode].begin(),samples[mode].end());std::cout<<" median_ms="<<samples[mode][3]<<"\n";
                }
            }
        }
    }
    std::cout<<"PASS paired real-weight projections preserve separate-call output bits\n";
}
void expert_bench(const char* model) {
    strata::artifact::Safetensors ts(model);
    for(int layer:{3,4}) {
        const auto prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.0.";
        std::array<GlmFp4Matrix,3> w;
        const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int i=0;i<3;++i)w[i]=strata::artifact::glm_load_fp4(ts,prefix+names[i]);
        for(int tokens:{1,2,3,4,7,16}) {
            std::vector<float>x(size_t(tokens)*w[0].cols);
            for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+i/w[0].cols*19)%257)-128)/128;
            const auto expected=strata::kernels::cpu::Fp4LocalCpuExpert(w,1).run(x,tokens,false,true);
            for(int threads:{4,16}) {
                strata::kernels::cpu::Fp4LocalCpuExpert expert(w,threads);
                std::vector<double> samples[4];
                auto execute=[&](int mode){return mode>=2?expert.run_fused(x,tokens,false,mode==3):expert.run(x,tokens,mode==1,false);};
                for(int mode=0;mode<4;++mode)execute(mode);
                for(int r=0;r<9;++r)for(int order=0;order<4;++order) {
                    const int mode=(r+order)%4;
                    const auto start=std::chrono::steady_clock::now();
                    auto got=execute(mode);
                    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    if(std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)))throw std::runtime_error("expert benchmark output mismatch");
                    samples[mode].push_back(ms);
                }
                for(int mode=0;mode<4;++mode) {
                    std::cout<<"layer="<<layer<<" tokens="<<tokens<<" threads="<<threads<<" mode="<<(mode==0?"separate":mode==1?"paired":mode==2?"fused":"reuse")<<" ms=";
                    for(double ms:samples[mode])std::cout<<ms<<",";
                    std::sort(samples[mode].begin(),samples[mode].end());std::cout<<" median_ms="<<samples[mode][4]<<"\n";
                }
            }
        }
    }
    std::cout<<"PASS complete CPU experts preserve verified output bits across pairing/thread modes\n";
}
void expert_checks() {
    std::array<GlmFp4Matrix,3> w;
    for(int p=0;p<3;++p) {
        auto& m=w[p];m.rows=p==2?16:32;m.cols=p==2?32:16;m.global=p==0?.375f:p==1?.25f:.125f;
        m.codes.assign(m.rows*m.cols/2,0x11);m.scales.assign(m.rows*m.cols/16,56);
    }
    auto rounded=[](float x){auto b=std::bit_cast<uint32_t>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);};
    for(float input:{0.f,1.f,20.f,-20.f}) {
        const float g=std::min(3*input,10.f),u=std::clamp(2*input,-10.f,10.f);
        const float expected=2*rounded(rounded(g/(1+std::exp(-g)))*u);
        for(int threads:{1,4})for(int nt:{1,3,16})for(bool paired:{false,true}) {
            auto got=strata::kernels::cpu::Fp4LocalCpuExpert(w,threads).run(std::vector<float>(size_t(nt)*16,input),nt,paired,true);
            for(float value:got)if(value!=expected)throw std::runtime_error("analytic CPU expert mismatch");
            auto fused=strata::kernels::cpu::Fp4LocalCpuExpert(w,threads).run_fused(std::vector<float>(size_t(nt)*16,input),nt,true);
            if(std::memcmp(fused.data(),got.data(),got.size()*sizeof(float)))throw std::runtime_error("analytic fused CPU expert mismatch");
        }
    }
    for(int H:{16,48,80}) {
        const int F=H+32;auto tail=w;
        for(int p=0;p<3;++p) {
            auto& m=tail[p];m.rows=p==2?H:F;m.cols=p==2?F:H;
            m.codes.resize(m.rows*m.cols/2);m.scales.resize(m.rows*m.cols/16);
            for(size_t i=0;i<m.codes.size();++i)m.codes[i]=uint8_t(i*73+19);
            for(size_t i=0;i<m.scales.size();++i)m.scales[i]=uint8_t(i%80);
        }
        for(int nt:{1,2,3,4,7,16})for(int threads:{1,4}) {
            std::vector<float> input(size_t(nt)*H);
            for(size_t i=0;i<input.size();++i)input[i]=float(int(i%13)-6)/8;
            strata::kernels::cpu::Fp4LocalCpuExpert expert(tail,threads);
            expert.run_fused(input,nt,true);
            expert.run_fused(input,nt,true,true);
            rejects([&]{expert.run_fused({},0);});
            rejects([&]{expert.run_fused({},17);});
            rejects([&]{expert.run_fused(std::vector<float>(input.size(),std::numeric_limits<float>::max()),nt);});
        }
    }
    auto huge=w;
    for(auto& m:huge){m.codes.assign(m.codes.size(),0x77);m.scales.assign(m.scales.size(),126);}
    bool overflow_rejected=false;
    try{strata::kernels::cpu::Fp4LocalCpuExpert(huge,4).run_fused(std::vector<float>(16,1e35f),1,false);}
    catch(const std::runtime_error&){overflow_rejected=true;}
    if(!overflow_rejected)throw std::runtime_error("fused nonfinite intermediate was accepted");
    w[2].rows=32;
    rejects([&]{strata::kernels::cpu::Fp4LocalCpuExpert invalid(w);});
    std::cout<<"PASS analytic complete expert and incompatible shape rejection\n";
}
int main(int argc,char** argv){try {
    if(argc==3&&std::string(argv[1])=="--expert-bench"){expert_bench(argv[2]);return 0;}
    if(argc==3&&std::string(argv[1])=="--pair-bench"){pair_bench(argv[2]);return 0;}
    if(argc==3&&std::string(argv[1])=="--bench"){bench(argv[2]);return 0;}
    if(argc>2)throw std::runtime_error("usage: glm-fp4-cpu-projection-test [PARTIAL_MODEL_DIR | --bench PARTIAL_MODEL_DIR | --pair-bench PARTIAL_MODEL_DIR | --expert-bench PARTIAL_MODEL_DIR]");
    expert_checks();
    for(size_t n:{size_t(1),size_t(17),size_t(65)})for(size_t k:{size_t(16),size_t(32),size_t(48),size_t(128)}) {
        GlmFp4Matrix m;m.rows=n;m.cols=k;m.global=.0001f;m.codes.resize(n*k/2);m.scales.resize(n*k/16);
        for(size_t i=0;i<m.codes.size();++i)m.codes[i]=uint8_t(i*73+19);
        for(size_t i=0;i<m.scales.size();++i)m.scales[i]=uint8_t(i%127);
        for(int tokens:{1,3,16})check(m,tokens);
    }
    GlmFp4Matrix m;m.rows=1;m.cols=16;m.global=.375f;m.codes.assign(8,0x11);m.scales.assign(1,56);
    Fp4LocalCpu owned(m);m.codes.assign(8,0x77);
    if(owned.run(std::vector<float>(16,1),1).at(0)!=3.f)throw std::runtime_error("ownership or analytic result mismatch");
    Fp4LocalCpu different_threads(m,2);
    rejects([&]{owned.run_pair(different_threads,std::vector<float>(16,1),1);});
    auto wide=m;wide.cols=32;wide.codes.resize(16);wide.scales.resize(2,56);
    Fp4LocalCpu wider(wide);
    rejects([&]{owned.run_pair(wider,std::vector<float>(16,1),1);});
    rejects([&]{owned.run_pair(owned,{},0);});
    rejects([&]{owned.run_pair(owned,{},17);});
    rejects([&]{owned.run_pair(owned,{},1);});
    rejects([&]{owned.run_pair(owned,std::vector<float>(16,std::numeric_limits<float>::infinity()),1);});
    rejects([&]{owned.run({},1);});rejects([&]{owned.run({},0);});rejects([&]{owned.run({},17);});
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::max()})
        rejects([&]{owned.run(std::vector<float>(16,bad),1);});
    rejects([&]{Fp4LocalCpu bad(m,0);});m.scales[0]=127;rejects([&]{Fp4LocalCpu bad(m);});
    if(argc==2) {
        strata::artifact::Safetensors ts(argv[1]);
        for(int layer:{3,4})for(const char* p:{"gate_proj","up_proj","down_proj"}) {
            auto w=strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+p);
            for(int tokens:{1,3})check(w,tokens);
        }
    }
    std::cout<<"PASS CPU W4A16 projection, K16 padding, batch/thread parity and rejection checks\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
