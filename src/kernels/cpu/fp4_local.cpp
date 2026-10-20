#include <algorithm>
#include "strata/kernels/cpu/fp4_local.hpp"
#include "strata/kernels/cpu/fp4_local_w4a16.hpp"
#include "strata/host/glm_fp4.h"
#include <bit>
#include <chrono>
#include <limits>
#include <atomic>

namespace strata::kernels::cpu {
namespace {
const std::array<artifact::GlmFp4Matrix,3>& expert_shape(const std::array<artifact::GlmFp4Matrix,3>& w) {
    if(w[0].rows!=w[1].rows||w[0].cols!=w[1].cols||w[2].rows!=w[0].cols||w[2].cols!=w[0].rows)
        throw std::invalid_argument("incompatible CPU FP4 expert shapes");
    return w;
}
float bf16(float x) {
    auto bits=std::bit_cast<uint32_t>(x);bits+=0x7fff+((bits>>16)&1);
    return std::bit_cast<float>(bits&0xffff0000u);
}
}
Fp4LocalCpu::Fp4LocalCpu(const artifact::GlmFp4Matrix& m,int threads)
    :rows_(0),columns_(0),global_(0),padded_columns_(0),threads_(threads) {
    if(!m.rows||!m.cols||m.rows>16384||m.cols>16384||m.cols%16||
       m.codes.size()!=m.rows*m.cols/2||m.scales.size()!=m.rows*m.cols/16||
       !std::isfinite(m.global)||m.global<=0||threads<1||threads>32)
        throw std::invalid_argument("invalid CPU FP4 projection extent");
    for(auto s:m.scales)if(s>126)throw std::invalid_argument("invalid CPU FP4 scale");
    rows_=m.rows;columns_=m.cols;global_=m.global;
    padded_columns_=int((m.cols+31)/32*32);
    codes_.resize(m.rows*size_t(padded_columns_)/2,0);
    scales_.resize(m.rows*size_t(padded_columns_)/16,0);
    for(size_t r=0;r<m.rows;++r) {
        std::copy_n(m.codes.data()+r*m.cols/2,m.cols/2,codes_.data()+r*padded_columns_/2);
        std::copy_n(m.scales.data()+r*m.cols/16,m.cols/16,scales_.data()+r*padded_columns_/16);
    }
    if(strata_glm_fp4_repack_k32(codes_.data(),codes_.data(),codes_.size(),0))
        throw std::runtime_error("CPU FP4 packing failed");
}

std::vector<float> Fp4LocalCpu::run(const std::vector<float>& input,int tokens,bool verify,float* kernel_ms) const {
    if(tokens<1||tokens>16||input.size()!=size_t(tokens)*columns_)
        throw std::invalid_argument("invalid CPU FP4 input extent");
    std::vector<float> x(size_t(tokens)*padded_columns_,0),out(size_t(tokens)*rows_);
    for(int t=0;t<tokens;++t)for(size_t j=0;j<columns_;++j) {
        float v=input[size_t(t)*columns_+j];
        if(!std::isfinite(v))throw std::invalid_argument("nonfinite CPU FP4 input");
        auto bits=std::bit_cast<uint32_t>(v);bits+=0x7fff+((bits>>16)&1);
        v=std::bit_cast<float>(bits&0xffff0000u);
        if(!std::isfinite(v))throw std::invalid_argument("CPU FP4 input overflows BF16");
        x[size_t(t)*padded_columns_+j]=v;
    }
    const int chunks=int((rows_+63)/64);
    const auto start=std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static) num_threads(threads_)
    for(int job=0;job<tokens*chunks;++job) {
        const int t=job/chunks,first=job%chunks*64;
        fp4_local_w4a16_rows(codes_.data(),scales_.data(),x.data()+size_t(t)*padded_columns_,global_,
            padded_columns_,int(rows_),first,std::min(first+64,int(rows_)),out.data()+size_t(t)*rows_,true,true);
    }
    if(kernel_ms)*kernel_ms=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count();
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite CPU FP4 output");
    if(verify) {
        const double eps=std::numeric_limits<float>::epsilon()/2;
        // Also covers the portable sequential scalar fallback.
        const double steps=padded_columns_+2;
        for(int t=0;t<tokens;++t)for(size_t r=0;r<rows_;++r) {
            double ref=0,norm=0;
            for(size_t j=0;j<columns_;++j) {
                const size_t at=r*size_t(padded_columns_)/2+(j/32)*16+j%16;
                const uint8_t code=(codes_[at]>>((j%32)/16*4))&15;
                const double product=double(strata_glm_e2m1(code))*strata_glm_e4m3(scales_[r*size_t(padded_columns_)/16+j/16])*x[size_t(t)*padded_columns_+j];
                ref+=product;norm+=std::abs(product);
            }
            ref*=global_;norm*=global_;
            const double bound=2*steps*eps/(1-steps*eps)*norm+1e-7;
            if(std::abs(double(out[size_t(t)*rows_+r])-ref)>bound)
                throw std::runtime_error("CPU FP4 projection exceeds FP64 rounding bound");
        }
    }
    return out;
}

std::array<std::vector<float>,2> Fp4LocalCpu::run_pair(const Fp4LocalCpu& other,
        const std::vector<float>& input,int tokens,bool verify) const {
    if(columns_!=other.columns_||threads_!=other.threads_||tokens<1||tokens>16||
       input.size()!=size_t(tokens)*columns_)
        throw std::invalid_argument("invalid CPU FP4 projection pair");
    std::vector<float> x(size_t(tokens)*padded_columns_,0);
    for(int t=0;t<tokens;++t)for(size_t j=0;j<columns_;++j) {
        float v=input[size_t(t)*columns_+j];
        if(!std::isfinite(v))throw std::invalid_argument("nonfinite CPU FP4 input");
        auto bits=std::bit_cast<uint32_t>(v);bits+=0x7fff+((bits>>16)&1);
        v=std::bit_cast<float>(bits&0xffff0000u);
        if(!std::isfinite(v))throw std::invalid_argument("CPU FP4 input overflows BF16");
        x[size_t(t)*padded_columns_+j]=v;
    }
    const Fp4LocalCpu* projections[]={this,&other};
    std::array<std::vector<float>,2> out={std::vector<float>(size_t(tokens)*rows_),
                                       std::vector<float>(size_t(tokens)*other.rows_)};
    const int first_chunks=int((rows_+63)/64),chunks=first_chunks+int((other.rows_+63)/64);
    #pragma omp parallel for schedule(static) num_threads(threads_)
    for(int job=0;job<tokens*chunks;++job) {
        const int t=job/chunks,part=job%chunks,p=part>=first_chunks;
        const auto& w=*projections[p];const int row=(part-(p?first_chunks:0))*64;
        fp4_local_w4a16_rows(w.codes_.data(),w.scales_.data(),x.data()+size_t(t)*padded_columns_,w.global_,
            padded_columns_,int(w.rows_),row,std::min(row+64,int(w.rows_)),out[p].data()+size_t(t)*w.rows_,true,true);
    }
    for(const auto& y:out)for(float v:y)if(!std::isfinite(v))throw std::runtime_error("nonfinite CPU FP4 output");
    if(verify)for(int p=0;p<2;++p) {
        const auto reference=projections[p]->run(input,tokens,true);
        if(std::memcmp(reference.data(),out[p].data(),reference.size()*sizeof(float)))
            throw std::runtime_error("CPU FP4 pair changed projection output");
    }
    return out;
}
Fp4LocalCpuExpert::Fp4LocalCpuExpert(const std::array<artifact::GlmFp4Matrix,3>& w,int threads)
    :gate_(expert_shape(w)[0],threads),up_(w[1],threads),down_(w[2],threads) {}

std::vector<float> Fp4LocalCpuExpert::run(const std::vector<float>& input,int tokens,bool paired,bool verify) const {
    auto gu=paired?gate_.run_pair(up_,input,tokens,verify):
        std::array<std::vector<float>,2>{gate_.run(input,tokens,verify),up_.run(input,tokens,verify)};
    for(size_t i=0;i<gu[0].size();++i) {
        const float g=std::min(bf16(gu[0][i]),10.f),u=std::clamp(bf16(gu[1][i]),-10.f,10.f);
        gu[0][i]=bf16(bf16(g/(1+std::exp(-g)))*u);
    }
    return down_.run(gu[0],tokens,verify);
}
std::vector<float> Fp4LocalCpuExpert::run_fused(const std::vector<float>& input,int tokens,bool verify,bool reuse_weights) const {
    const int H=int(gate_.columns_),F=int(gate_.rows_);
    if(tokens<1||tokens>16||input.size()!=size_t(tokens)*H)
        throw std::invalid_argument("invalid fused CPU FP4 input extent");
    std::vector<float> x(size_t(tokens)*gate_.padded_columns_,0);
    for(int t=0;t<tokens;++t)for(int j=0;j<H;++j) {
        const float v=input[size_t(t)*H+j];
        if(!std::isfinite(v)||!std::isfinite(bf16(v)))throw std::invalid_argument("nonfinite fused CPU FP4 input");
        x[size_t(t)*gate_.padded_columns_+j]=bf16(v);
    }
    std::vector<float> g(size_t(tokens)*F),u(g.size()),h(size_t(tokens)*down_.padded_columns_,0),out(size_t(tokens)*H);
    std::atomic<bool> invalid{false};
    const int gu_chunks=(F+63)/64,d_chunks=(H+63)/64;
    const int tile=reuse_weights?4:1,groups=(tokens+tile-1)/tile;
    #pragma omp parallel num_threads(gate_.threads_)
    {
        #pragma omp for schedule(static)
        for(int job=0;job<groups*gu_chunks;++job) {
            const int t=(job/gu_chunks)*tile,count=std::min(tile,tokens-t),first=(job%gu_chunks)*64,last=std::min(first+64,F);
            auto project=[&](const Fp4LocalCpu& w,float* y) {
                if(reuse_weights)fp4_local_w4a16_batch_rows(w.codes_.data(),w.scales_.data(),x.data()+size_t(t)*w.padded_columns_,w.global_,
                    w.padded_columns_,F,count,first,last,y,true,true);
                else fp4_local_w4a16_rows(w.codes_.data(),w.scales_.data(),x.data()+size_t(t)*w.padded_columns_,w.global_,
                    w.padded_columns_,F,first,last,y,true,true);
            };
            project(gate_,g.data()+size_t(t)*F);project(up_,u.data()+size_t(t)*F);
            for(int tt=t;tt<t+count;++tt)for(int r=first;r<last;++r) {
                const size_t i=size_t(tt)*F+r;
                if(!std::isfinite(g[i])||!std::isfinite(u[i])){invalid.store(true,std::memory_order_relaxed);continue;}
                const float gv=std::min(bf16(g[i]),10.f),uv=std::clamp(bf16(u[i]),-10.f,10.f);
                const float value=bf16(bf16(gv/(1+std::exp(-gv)))*uv);
                if(!std::isfinite(value)){invalid.store(true,std::memory_order_relaxed);continue;}
                h[size_t(tt)*down_.padded_columns_+r]=value;
            }
        }
        // The implicit barrier above makes every hidden row visible to down.
        #pragma omp for schedule(static)
        for(int job=0;job<groups*d_chunks;++job) {
            const int t=(job/d_chunks)*tile,count=std::min(tile,tokens-t),first=(job%d_chunks)*64;
            if(reuse_weights)fp4_local_w4a16_batch_rows(down_.codes_.data(),down_.scales_.data(),h.data()+size_t(t)*down_.padded_columns_,down_.global_,
                down_.padded_columns_,H,count,first,std::min(first+64,H),out.data()+size_t(t)*H,true,true);
            else fp4_local_w4a16_rows(down_.codes_.data(),down_.scales_.data(),h.data()+size_t(t)*down_.padded_columns_,down_.global_,
                down_.padded_columns_,H,first,std::min(first+64,H),out.data()+size_t(t)*H,true,true);
        }
    }
    if(invalid.load(std::memory_order_relaxed))throw std::runtime_error("nonfinite fused CPU FP4 intermediate");
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite fused CPU FP4 output");
    if(verify) {
        const auto reference=run(input,tokens,false,true);
        if(std::memcmp(reference.data(),out.data(),out.size()*sizeof(float)))throw std::runtime_error("fused CPU FP4 expert changed output");
    }
    return out;
}
}
