// Cold-cache, full-token-size CPU kernel probe; never treats MCG weights as mul1 inference.
#include "strata/kernels/cpu/exl3/pool.hpp"
#include "strata/kernels/cpu/exl3/mul1_probe.hpp"
#include <chrono>
#include <iostream>
#include <random>
using namespace strata::cpu::exl3;
using Clock=std::chrono::steady_clock;
void probe(Pool& pool,const Pool::Matrix& m,const float* x,float* y,bool wide) {
    std::vector<float> h(m.in);for(size_t i=0;i<m.in;++i)h[i]=x[i]*half_value(m.suh[i]);hadamard(h.data(),m.in);
    float peak=0;for(float v:h)peak=std::max(peak,std::abs(v));
    float scale=peak?(peak/(wide?16129.f:127.f)):1.f;
    std::vector<int8_t> q(m.in),low(m.in);int64_t sum=0,low_sum=0;
    for(size_t i=0;i<m.in;++i){int v=std::clamp(int(std::round(h[i]/scale)),wide?-16129:-127,wide?16129:127);int a=wide?int(std::round(v/127.f)):v;int b=wide?v-a*127:0;q[i]=a;low[i]=b;sum+=a;low_sum+=b;}
    pool.run([&](size_t rank){const auto& r=m.rows[rank];
        for(size_t ot=r.first/16;ot<r.last/16;++ot){int32_t acc[16]={},residual[16]={};
            for(size_t it=0;it<m.in/16;++it){const auto* packed=r.packed.data()+((ot-r.first/16)*(m.in/16)+it)*64;mul1_tile_dot(packed,q.data()+it*16,acc);if(wide)mul1_tile_dot(packed,low.data()+it*16,residual);}
            for(unsigned c=0;c<16;++c){double value=double(acc[c])-510*sum;if(wide)value=value*127+double(residual[c])-510*low_sum;y[ot*16+c]=float(value)*scale*half_value(0x1eee);}
        }
        hadamard(y+r.first,r.last-r.first);for(size_t i=r.first;i<r.last;++i)y[i]*=half_value(m.svh[i]);
    });
}
int main(int argc,char** argv){try{
    if(argc<2)throw std::runtime_error("usage: strata-glm-exl3-kernel-bench MODEL_DIR [experts=336] [rounds=3]");
    size_t experts=argc>2?std::stoul(argv[2]):336,rounds=argc>3?std::stoul(argv[3]):3;
    if(!experts||experts>336||!rounds||rounds>20)throw std::runtime_error("invalid benchmark extent");
    std::mt19937 rng(42);for(unsigned test=0;test<256;++test){uint8_t p[64];int8_t x[16];int32_t a[16]={},b[16]={};for(auto&v:p)v=rng();for(auto&v:x)v=int(rng()%255)-127;
        for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)a[c]+=int32_t(mul1_bytesum(state(p,exl3_offsets[r*16+c])))*x[r];
        mul1_tile_dot(p,x,b);for(unsigned c=0;c<16;++c)if(a[c]!=b[c])throw std::runtime_error("integer tile parity failed");}
    std::cout<<"integer_tile_parity=256_PASS scope=synthetic_mul1_codebook_no_model_quality_claim\n";
    strata::artifact::GlmExl3 model(argv[1]);Pool pool(15,true);
    struct Expert{Pool::Matrix gate,up,down;};std::vector<Expert> corpus;size_t bytes=0;
    for(size_t e=0;e<experts;++e){Expert x{pool.load(model,3,17,"gate_proj"),pool.load(model,3,17,"up_proj"),pool.load(model,3,17,"down_proj")};for(auto* m:{&x.gate,&x.up,&x.down})bytes+=m->in*m->out/4;corpus.push_back(std::move(x));}
    std::cout<<"experts="<<experts<<" packed_bytes="<<bytes<<" workers=15 source=replicated_real_expert_shapes first_touch=worker_local\n";
    std::vector<float>x(4096),g(2048),u(2048),y(4096);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.017f);
    // Alternate order across rounds to expose drift. Each corpus is ~2 GiB at full size.
    for(size_t round=0;round<rounds;++round)for(int order=0;order<3;++order){int mode=(order+round)%3;auto start=Clock::now();double checksum=0;
        for(auto& e:corpus){if(!mode){pool.apply(e.gate,x.data(),g.data());pool.apply(e.up,x.data(),u.data());}else{probe(pool,e.gate,x.data(),g.data(),mode==2);probe(pool,e.up,x.data(),u.data(),mode==2);}
            for(size_t i=0;i<g.size();++i){float a=std::min(g[i],10.f),b=std::clamp(u[i],-10.f,10.f);g[i]=a/(1+std::exp(-a))*b;}
            if(!mode)pool.apply(e.down,g.data(),y.data());else probe(pool,e.down,g.data(),y.data(),mode==2);
            for(float v:y){if(!std::isfinite(v))throw std::runtime_error("nonfinite benchmark output");checksum+=v;}}
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        std::cout<<"mode="<<(mode==0?"mcg_fp32":mode==1?"mul1_int8_probe":"mul1_wide_probe")<<" round="<<round<<" ms="<<ms<<" equivalent_packed_GB_s="<<bytes/ms/1e6<<" expert_only_tok_s="<<1000/ms<<" checksum="<<checksum<<'\n';
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
