#include "strata/artifact/glm_fp4.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include <memory>
#include <chrono>
#include "ggml.h"
#include "ggml-cpu.h"
#include <array>
#include <iostream>
#include <algorithm>
#include <stdexcept>
using namespace strata::kernels::cpu;
using strata::artifact::GlmFp4Matrix;
// All FP4 codes, extreme signed Q8 values and a -128 in the final block.
// This exercises the opt-in correction and its raw-input fallback against GGML.
void check_raw_q8() {
    if (!cpu_avx2_ok()) return;
    for (int k : {64,128,640,4096}) {
        constexpr int rows = 16;
        std::vector<block_nvfp4> w(rows * k / 64);
        std::vector<block_q8_0> q(k / 32);
        for (int r = 0; r < rows; ++r) for (int b = 0; b < k/64; ++b) {
            auto& v = w[r*(k/64)+b];
            for (int j = 0; j < 4; ++j) v.d[j] = (r*11+b*7+j)%127;
            std::memset(v.qs, r | (r << 4), sizeof(v.qs));
        }
        // Distinct up rows make a swapped/misaddressed gate/up load observable.
        auto gu = w;
        for (int r = rows - 1; r >= 0; --r)
            gu.insert(gu.end(), w.begin() + r*(k/64), w.begin() + (r+1)*(k/64));
        // Zero, subnormal boundaries, normal boundaries, and a signed scale.
        for (uint16_t scale : {0x0000,0x0001,0x03ff,0x0400,0x3c00,0x7bff,0xbc00})
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int b = 0; b < k/32; ++b) {
                q[b].d = scale;
                for (int j = 0; j < 32; ++j) {
                    const int v = pattern == 0 ? (j%2 ? 127 : -127) : (b*32+j)%255-127;
                    q[b].qs[j] = pattern == 3 ? -128 : v;
                }
            }
            if (pattern == 2) q.back().qs[31] = -128;
            float out[rows], expected[rows];
            const void* ap[] = {q.data()}; float* op[] = {out};
            nvfp4_256_rows(reinterpret_cast<const uint8_t*>(w.data()), k/64*sizeof(block_nvfp4), k, ap, 1, op, 0, rows);
            for (int r = 0; r < rows; ++r)
                ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot(k, expected+r, 0, w.data()+r*(k/64), 0, q.data(), 0, 1);
            if (std::memcmp(out, expected, sizeof(out))) throw std::runtime_error("raw Q8 NVFP4 parity mismatch");
            // Odd-length range, nonzero first row, and untouched output guards.
            std::fill(std::begin(out), std::end(out), -12345.f);
            nvfp4_256_rows(reinterpret_cast<const uint8_t*>(w.data()), k/64*sizeof(block_nvfp4), k, ap, 1, op, 1, rows-2);
            if (out[0] != -12345.f || out[rows-2] != -12345.f || out[rows-1] != -12345.f ||
                std::memcmp(out+1, expected+1, (rows-3)*sizeof(float)))
                throw std::runtime_error("partial-row Q8 NVFP4 parity or guard mismatch");
            const size_t row_bytes = k/64*sizeof(block_nvfp4);
            for (float limit : {0.f, 7.f}) for (bool separate : {false, true}) {
                std::fill(std::begin(out), std::end(out), -12345.f);
                const auto* blob = reinterpret_cast<const uint8_t*>(gu.data());
                nvfp4_256_gu_rows_clamped(blob, row_bytes, rows*row_bytes, k, ap, 1, op,
                                          1, rows-2, .0001f, -.0002f, limit,
                                          separate ? blob + rows*row_bytes : nullptr);
                for (int r = 1; r < rows-2; ++r) {
                    float gs = expected[r]*.0001f, us = expected[rows-1-r]*-.0002f;
                    if (limit > 0) {
                        gs = std::fmin(gs, limit);
                        us = std::fmax(-limit, std::fmin(us, limit));
                    }
                    const float ref = (gs / (1.f + std::exp(-gs))) * us;
                    if (std::memcmp(out+r, &ref, sizeof(ref)))
                        throw std::runtime_error("raw Q8 gate/up parity mismatch");
                }
                if (out[0] != -12345.f || out[rows-2] != -12345.f || out[rows-1] != -12345.f)
                    throw std::runtime_error("raw Q8 gate/up row guard mismatch");
            }
        }
    }
    std::cout << "PASS raw NVFP4 all codes, signed Q8 extremes, final-block -128, four shapes, seven finite scales, partial rows, gate/up clamp/storage parity bit_exact=1\n";
}

void check(const std::array<GlmFp4Matrix,3>& w,int nt,float amplitude,float limit,bool benchmark=false,bool owned=false,bool require_clamp=false) {
    NativeFmt f;std::string error;
    if(!native_fmt(kNvfp4Type,kNvfp4Type,w[0].cols,w[0].rows,f,error))throw std::runtime_error(error);
    f.swiglu_limit=limit;
    const int H=int(f.n_embd),F=int(f.n_ff);
    if(w[1].rows!=size_t(F)||w[1].cols!=size_t(H)||w[2].rows!=size_t(H)||w[2].cols!=size_t(F))throw std::runtime_error("bad shape");
    std::vector<uint8_t> blob(f.bytes);
    const size_t offsets[]={0,f.up_off,f.down_off};
    for(int p=0;p<3;++p) {
        const auto& m=w[p];
        if(strata_glm_fp4_pack_nvfp4(m.rows,m.cols,m.codes.data(),m.codes.size(),m.scales.data(),m.scales.size(),blob.data()+offsets[p],m.rows*m.cols/64*36))throw std::runtime_error("packing failed");
    }
    const float tail[]={w[0].global,w[1].global,w[2].global,0};
    std::memcpy(blob.data()+f.tail_off,tail,sizeof(tail));
    std::vector<std::vector<uint8_t>> aq(nt,std::vector<uint8_t>(f.act_bytes)),hq(nt,std::vector<uint8_t>(f.h_bytes));
    std::vector<std::vector<float>> hidden(nt,std::vector<float>(F)),reference(nt,std::vector<float>(F)),out(nt,std::vector<float>(H));
    std::vector<const void*> ap(nt),hp(nt);std::vector<float*> fp(nt),op(nt);std::vector<float>x(H);
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot;
    size_t clamped=0,gate_high=0,up_high=0,up_low=0;
    for(int t=0;t<nt;++t) {
        for(int j=0;j<H;++j)x[j]=amplitude*(1.f+float((j+t)%7)/16);
        native_quant_act(f,x.data(),aq[t].data());ap[t]=aq[t].data();fp[t]=hidden[t].data();op[t]=out[t].data();
        for(int r=0;r<F;++r) {
            float g=0,u=0;
            dot(H,&g,0,blob.data()+size_t(r)*f.gu_row,0,ap[t],0,1);
            dot(H,&u,0,blob.data()+f.up_off+size_t(r)*f.gu_row,0,ap[t],0,1);
            g*=tail[0];u*=tail[1];
            if(limit>0){gate_high+=g>limit;up_high+=u>limit;up_low+=u<-limit;clamped+=g>limit||std::abs(u)>limit;g=std::fmin(g,limit);u=std::fmax(-limit,std::fmin(u,limit));}
            reference[t][r]=(g/(1.f+std::exp(-g)))*u;
        }
    }
    if(require_clamp&&(!gate_high||!up_high||!up_low))throw std::runtime_error("real fixture did not exercise all clamp boundaries");
    native_gu_rows(f,blob.data(),ap.data(),nt,fp.data(),0,F);
    for(int t=0;t<nt;++t) {
        if(std::memcmp(hidden[t].data(),reference[t].data(),F*sizeof(float)))throw std::runtime_error("native gate/up differs from clamped GGML reference");
        native_quant_h(f,hidden[t].data(),hq[t].data());hp[t]=hq[t].data();
    }
    native_down_rows(f,blob.data(),hp.data(),nt,op.data(),0,H);
    for(int t=0;t<nt;++t)for(int r=0;r<H;++r) {
        float v=0;dot(F,&v,0,blob.data()+f.down_off+size_t(r)*f.d_row,0,hp[t],0,1);v*=tail[2];
        if(std::memcmp(&v,&out[t][r],sizeof(float)))throw std::runtime_error("native down differs from GGML reference");
    }
    // Independently allocated half-row shards must use the full blob's scale tail.
    const auto expected_out=out;
    std::array<std::vector<uint8_t>,2> gate_shards,up_shards,down_shards;
    for(int node=0;node<2;++node) {
        const size_t gu_half=size_t(F/2)*f.gu_row,d_half=size_t(H/2)*f.d_row;
        gate_shards[node].assign(blob.begin()+node*gu_half,blob.begin()+(node+1)*gu_half);
        up_shards[node].assign(blob.begin()+f.up_off+node*gu_half,blob.begin()+f.up_off+(node+1)*gu_half);
        down_shards[node].assign(blob.begin()+f.down_off+node*d_half,blob.begin()+f.down_off+(node+1)*d_half);
        std::vector<float*> ptr(nt);
        for(int t=0;t<nt;++t)ptr[t]=hidden[t].data()+node*(F/2);
        native_gu_rows(f,gate_shards[node].data(),ap.data(),nt,ptr.data(),0,F/2,up_shards[node].data(),blob.data());
    }
    for(int t=0;t<nt;++t) {
        if(std::memcmp(hidden[t].data(),reference[t].data(),F*sizeof(float)))throw std::runtime_error("sharded gate/up mismatch");
        native_quant_h(f,hidden[t].data(),hq[t].data());
    }
    for(int node=0;node<2;++node) {
        std::vector<float*> ptr(nt);
        for(int t=0;t<nt;++t)ptr[t]=out[t].data()+node*(H/2);
        native_down_rows(f,gate_shards[node].data(),hp.data(),nt,ptr.data(),0,H/2,down_shards[node].data(),blob.data());
    }
    for(int t=0;t<nt;++t)if(std::memcmp(out[t].data(),expected_out[t].data(),H*sizeof(float)))throw std::runtime_error("sharded down mismatch");
    if(benchmark) {
        ExpertPool pool(15,true);
        const auto previous=pin_current_thread(planned_host_core());
        std::vector<float> flat(size_t(nt)*H),weights(nt,1),sum(flat.size());
        ExpertJobMulti job;job.blob=blob.data();job.nt=nt;
        std::unique_ptr<NumaTensor> ng,nu,nd;
        if(owned) {
            if(!pool.numa_rows_available())throw std::runtime_error("NUMA benchmark needs workers on both nodes");
            ng=std::make_unique<NumaTensor>(blob.data(),f.gu_row,F,1,pool.numa_cores(),false);
            nu=std::make_unique<NumaTensor>(blob.data()+f.up_off,f.gu_row,F,1,pool.numa_cores(),false);
            nd=std::make_unique<NumaTensor>(blob.data()+f.down_off,f.d_row,H,1,pool.numa_cores(),false);
            for(int node=0;node<2;++node)job.numa[node]={ng->shard(0,node),nu->shard(0,node),nd->shard(0,node)};
        }
        for(int t=0;t<nt;++t) {
            job.nact[t]=ap[t];job.out[t]=flat.data()+size_t(t)*H;
            std::copy(out[t].begin(),out[t].end(),job.out[t]);
        }
        if(owned) {
            job.blob=nullptr;
            for(bool flow:{false,true}) {
                bool rejected=false;
                try {
                    if(flow)pool.run_layer_native(f,&job,1,flat.data(),weights.data(),sum.data(),nt,1);
                    else pool.run_split_multi_native(f,&job,1);
                } catch(const std::invalid_argument&) { rejected=true; }
                if(!rejected)throw std::runtime_error("missing scale-tail owner was accepted");
            }
            job.blob=blob.data();
        }
        const auto expected=flat;
        pool.reduce_routed(flat.data(),weights.data(),sum.data(),nt,1,H);
        const auto expected_sum=sum;
        auto run=[&](int flow) {
            if(flow)pool.run_layer_native(f,&job,1,flat.data(),weights.data(),sum.data(),nt,1);
            else {
                pool.run_split_multi_native(f,&job,1);
                pool.reduce_routed(flat.data(),weights.data(),sum.data(),nt,1,H);
            }
        };
        std::vector<double> times[2];run(0);run(1);
        for(int r=0;r<9;++r)for(int order=0;order<2;++order) {
            const int flow=(r+order)%2;
            const auto start=std::chrono::steady_clock::now();run(flow);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(std::memcmp(flat.data(),expected.data(),flat.size()*sizeof(float)) ||
               std::memcmp(sum.data(),expected_sum.data(),sum.size()*sizeof(float)))throw std::runtime_error("native pool output mismatch");
            times[flow].push_back(ms);
        }
        for(int flow=0;flow<2;++flow) {
            std::sort(times[flow].begin(),times[flow].end());
            std::cout<<"BENCH tokens="<<nt<<" workers=15 owned="<<owned<<" mode="<<(flow?"flow":"split")<<" blob_bytes="<<blob.size()<<" median_ms="<<times[flow][4]<<" samples=";
            for(double ms:times[flow])std::cout<<ms<<",";
            std::cout<<"\n";
        }
        if(previous.valid)restore_thread_affinity(previous);
    }
    std::cout<<"PASS native NVFP4 H="<<H<<" F="<<F<<" tokens="<<nt<<" amplitude="<<amplitude<<" limit="<<limit<<" clamped_rows="<<clamped<<" gate_high="<<gate_high<<" up_high="<<up_high<<" up_low="<<up_low<<" bit_exact=1\n";
}
int main(int argc,char**argv){try {
    const bool owned=argc==3&&std::string(argv[1])=="--bench-numa";
    const bool benchmark=owned||(argc==3&&std::string(argv[1])=="--bench");
    if(argc>2&&!benchmark)throw std::runtime_error("usage: glm-fp4-native-test [PARTIAL_MODEL_DIR | --bench PARTIAL_MODEL_DIR | --bench-numa PARTIAL_MODEL_DIR]");
    if(!native_experts_available())return 77;
    ggml_cpu_init();check_raw_q8();std::array<GlmFp4Matrix,3>w;
    for(auto& m:w){m.rows=m.cols=64;m.global=.375f;m.codes.assign(2048,0x11);m.scales.assign(256,56);}
    w[1].global=.25f;w[2].global=.125f;
    if(!benchmark)for(float limit:{0.f,10.f})for(float a:{0.f,1.f,20.f,-20.f})for(int nt:{1,2,3,4,5,6,7,8,9,16})check(w,nt,a,limit);
    if(argc==2||benchmark){strata::artifact::Safetensors ts(argv[benchmark?2:1]);for(int l:{3,4}) {
        int p=0;for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(l)+".mlp.experts.0."+name);
        for(int nt:{1,4}){if(benchmark)std::cout<<"BENCH_LAYER layer="<<l<<" tokens="<<nt<<"\n";check(w,nt,1,10,benchmark,owned);}
        if(!benchmark)for(float amplitude:{4096.f,-4096.f})for(int nt:{1,4})check(w,nt,amplitude,10,false,false,true);
    }}
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
