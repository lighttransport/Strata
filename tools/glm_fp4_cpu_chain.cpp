// CPU-only partial MoE chain: real local NVFP4 weights, opt-in Q8 activations.
// Compare separate projection dispatches with one persistent team per chain.
#include "strata/artifact/glm_fp4.hpp"
#include "strata/kernels/cpu/fp4_local_q8.hpp"
#include "strata/kernels/cpu/fp4_local_w4a16.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <sys/mman.h>
#include <cstring>
#include <memory>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <cerrno>
#include <map>
#include <random>
struct LocalRows {
    uint8_t *codes=nullptr,*scales=nullptr;size_t nc=0,ns=0;
    LocalRows(size_t c,size_t s):nc(c),ns(s) {
        codes=static_cast<uint8_t*>(mmap(nullptr,nc,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(codes==MAP_FAILED){codes=nullptr;throw std::runtime_error("codes mmap failed");}
        scales=static_cast<uint8_t*>(mmap(nullptr,ns,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(scales==MAP_FAILED){scales=nullptr;munmap(codes,nc);codes=nullptr;throw std::runtime_error("scales mmap failed");}
        madvise(codes,nc,MADV_NOHUGEPAGE);madvise(scales,ns,MADV_NOHUGEPAGE);
    }
    ~LocalRows(){if(codes)munmap(codes,nc);if(scales)munmap(scales,ns);}
    LocalRows(const LocalRows&)=delete;LocalRows& operator=(const LocalRows&)=delete;
};
struct Matrix: strata::artifact::GlmFp4Matrix {
    using strata::artifact::GlmFp4Matrix::operator=;
    std::unique_ptr<LocalRows> local;
    std::vector<int> row_nodes;
    void copy_rows(int row) {
        unsigned node=0;
        row_nodes[row/64]=syscall(SYS_getcpu,nullptr,&node,nullptr)==0?int(node):-1;
        const size_t count=std::min(size_t(64),rows-row);
        std::memcpy(local->codes+size_t(row)*cols/2,codes.data()+size_t(row)*cols/2,count*cols/2);
        std::memcpy(local->scales+size_t(row)*cols/16,scales.data()+size_t(row)*cols/16,count*cols/16);
    }
};
constexpr int H=4096,F=2048,E=3,L=2;
float bf16(float x) {auto b=std::bit_cast<uint32_t>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
// Numerical oracle only: BF16 operands, FP64 accumulation, global scale last.
// It is not a WMMA instruction-order emulator or a performance implementation.
void reference_projection(const Matrix& w,const std::vector<float>& x,std::vector<float>& y) {
    static const auto weights=[] {
        std::array<float,127*16> values{};
        for(int s=0;s<127;++s)for(int q=0;q<16;++q) {
            const float v=strata_glm_e4m3(uint8_t(s))*strata_glm_e2m1(uint8_t(q));
            if(bf16(v)!=v)throw std::runtime_error("FP4 scaled weight is not exactly BF16");
            values[s*16+q]=v;
        }
        return values;
    }();
    #pragma omp parallel for schedule(static)
    for(int r=0;r<int(w.rows);++r) {
        double sum=0;
        for(int j=0;j<int(w.cols);++j) {
            const size_t at=size_t(r)*w.cols+j;
            const unsigned q=(w.codes[at/2]>>((j%2)*4))&15;
            const unsigned scale=w.scales[at/16];
            sum+=double(bf16(x[j]))*double(weights[scale*16+q]);
        }
        y[r]=float(sum*double(w.global));
    }
}
void error_report(const std::string& tag,const std::vector<float>& ref,const std::vector<float>& got) {
    double err=0,norm=0,gotnorm=0,dot=0,max_abs=0;
    if(ref.size()!=got.size()||ref.empty())throw std::runtime_error("invalid error comparison extent");
    for(size_t i=0;i<ref.size();++i) {
        if(!std::isfinite(ref[i])||!std::isfinite(got[i]))throw std::runtime_error("nonfinite quality comparison");
        const double d=double(got[i])-ref[i];err+=d*d;norm+=double(ref[i])*ref[i];
        gotnorm+=double(got[i])*got[i];dot+=double(got[i])*ref[i];max_abs=std::max(max_abs,std::abs(d));
    }
    std::cout<<tag<<" relative_l2="<<std::sqrt(err/std::max(norm,1e-30))
             <<" max_abs="<<max_abs<<" reference_rms="<<std::sqrt(norm/ref.size())
             <<" cosine="<<(norm&&gotnorm?dot/std::sqrt(norm*gotnorm):err==0?1:0)<<"\n";
}
struct Quant {
    std::vector<int8_t> values;std::vector<float> scales;std::vector<int16_t> sums;
    explicit Quant(int n):values(n),scales(n/64),sums(n/16){}
    bool simd=false;
    void block(const float* x,int j) {
        if(!strata::kernels::cpu::fp4_local_quantize_q8_64(x,values.data()+j,scales.data()+j/64,sums.data()+j/16,simd))
            throw std::runtime_error("nonfinite chain quantizer input");
    }
};
struct Expert {
    Matrix gate,up,down;
    std::vector<float> g=std::vector<float>(F),u=g,h=g,y=std::vector<float>(H);
    Quant qh{F};
};
struct Chain {
    std::array<std::array<Expert,E>,L> experts;
    std::vector<float> input=std::vector<float>(H),initial=input;
    Quant qx{H};
    explicit Chain(const char* dir) {
        strata::artifact::Safetensors ts(dir);
        for(int l=0;l<L;++l)for(int e=0;e<E;++e) {
            auto prefix="model.language_model.layers."+std::to_string(l+3)+".mlp.experts."+std::to_string(e)+".";
            auto& w=experts[l][e];
            w.gate=strata::artifact::glm_load_fp4(ts,prefix+"gate_proj");
            w.up=strata::artifact::glm_load_fp4(ts,prefix+"up_proj");
            w.down=strata::artifact::glm_load_fp4(ts,prefix+"down_proj");
            if(w.gate.rows!=F||w.gate.cols!=H||w.up.rows!=F||w.up.cols!=H||w.down.rows!=H||w.down.cols!=F)
                throw std::runtime_error("unexpected expert shape");
        }
        for(int i=0;i<H;++i)initial[i]=bf16(std::sin(float(i)*.013f)*.5f);
    }
    bool local_rows=false,planar=false;
    void set_planar(bool enabled) {
        if(planar==enabled)return;
        if(local_rows)throw std::runtime_error("layout conversion requires loader-owned chain weights");
        for(auto& layer:experts)for(auto& e:layer)for(Matrix* w:{&e.gate,&e.up,&e.down})
            if(strata_glm_fp4_repack_k32(w->codes.data(),w->codes.data(),w->codes.size(),enabled?0:1))throw std::runtime_error("chain layout conversion failed");
        planar=enabled;
    }
    void place_rows() {
        // Match the persistent chain's flattened expert/row ownership. Placement
        // and exact copy verification are outside timing, repeated per team size.
        for(auto& layer:experts)for(auto& e:layer)for(Matrix* w:{&e.gate,&e.up,&e.down}) {
            w->row_nodes.assign((w->rows+63)/64,-1);
            w->local=std::make_unique<LocalRows>(w->codes.size(),w->scales.size());
        }
        #pragma omp parallel
        {
            for(int l=0;l<L;++l) {
                #pragma omp for schedule(static)
                for(int job=0;job<E*(F/64);++job) {
                    auto& w=experts[l][job/(F/64)];int row=(job%(F/64))*64;
                    w.gate.copy_rows(row);w.up.copy_rows(row);
                }
                #pragma omp for schedule(static)
                for(int job=0;job<E*(H/64);++job)
                    experts[l][job/(H/64)].down.copy_rows((job%(H/64))*64);
            }
        }
        for(auto& layer:experts)for(auto& e:layer)for(Matrix* w:{&e.gate,&e.up,&e.down})
            if(std::memcmp(w->local->codes,w->codes.data(),w->codes.size())||std::memcmp(w->local->scales,w->scales.data(),w->scales.size()))
                throw std::runtime_error("row placement changed weight bytes");
    }
    void placement_report() {
        std::vector<int> cpus(omp_get_max_threads(),-1);
        #pragma omp parallel
        cpus[omp_get_thread_num()]=sched_getcpu();
        std::cout<<"worker_cpus=";for(int cpu:cpus)std::cout<<cpu<<",";std::cout<<"\n";
        // Query sampled pages only; do not migrate or change memory policy.
        std::vector<void*> pages;std::vector<int> expected_nodes;
        const size_t stride=size_t(sysconf(_SC_PAGESIZE))*16;
        for(auto& layer:experts)for(auto& e:layer)for(Matrix* w:{&e.gate,&e.up,&e.down}) {
            for(size_t i=0;i<w->local->nc;i+=stride){pages.push_back(w->local->codes+i);expected_nodes.push_back(w->row_nodes[i/(w->cols/2)/64]);}
            for(size_t i=0;i<w->local->ns;i+=stride){pages.push_back(w->local->scales+i);expected_nodes.push_back(w->row_nodes[i/(w->cols/16)/64]);}
        }
        std::vector<int> nodes(pages.size(),-1);
        const long result=syscall(SYS_move_pages,0,pages.size(),pages.data(),nullptr,nodes.data(),0);
        if(result<0){std::cout<<"page_query_unavailable_errno="<<errno<<" sampled_pages="<<pages.size()<<"\n";return;}
        size_t mismatches=0;for(size_t i=0;i<nodes.size();++i)if(nodes[i]!=expected_nodes[i]||nodes[i]<0)++mismatches;
        std::cout<<"sampled_owner_node_mismatches="<<mismatches<<"/"<<nodes.size()<<"\n";
        std::map<int,size_t> counts;for(int node:nodes)++counts[node];
        std::cout<<"sampled_page_nodes=";for(auto [node,count]:counts)std::cout<<node<<":"<<count<<",";std::cout<<"\n";
    }
    int row_tile=4;
    void projection(const Matrix& w,const Quant& q,std::vector<float>& y,int row,bool fast) {
        if(planar) {
            if(!fast)throw std::runtime_error("planar chain requires tree arithmetic");
            strata::kernels::cpu::fp4_local_q8_planar_rows(w.codes.data(),w.scales.data(),q.values.data(),q.scales.data(),w.global,w.cols,w.rows,1,row,std::min(row+64,int(w.rows)),y.data(),row_tile);
            return;
        }
        strata::kernels::cpu::fp4_local_q8_rows(local_rows?w.local->codes:w.codes.data(),local_rows?w.local->scales:w.scales.data(),q.values.data(),q.scales.data(),q.sums.data(),w.global,w.cols,w.rows,1,row,std::min(row+64,int(w.rows)),y.data(),true,fast,fast?row_tile:1);
    }
    void hidden(Expert& w,int row,bool quantize=true) {
        for(int j=row;j<row+64;++j) {
            float g=std::min(bf16(w.g[j]),10.f),u=std::clamp(bf16(w.u[j]),-10.f,10.f);
            w.h[j]=bf16(bf16(g/(1.f+std::exp(-g)))*u);
        }
        if(quantize)w.qh.block(w.h.data()+row,row);
    }
    void reduce(int l,int j) {
        float sum=0;for(int e=0;e<E;++e)sum+=experts[l][e].y[j]/float(E);
        input[j]=bf16(std::clamp(input[j]+sum*.125f,-4.f,4.f));
    }
    void w4a16_projection(const Matrix& w,const std::vector<float>& x,std::vector<float>& y,int row) {
        strata::kernels::cpu::fp4_local_w4a16_rows(local_rows?w.local->codes:w.codes.data(),local_rows?w.local->scales:w.scales.data(),x.data(),w.global,w.cols,w.rows,row,std::min(row+64,int(w.rows)),y.data(),planar,true);
    }
    void run_w4a16() {
        input=initial;
        #pragma omp parallel
        {
            for(int l=0;l<L;++l) {
                #pragma omp for schedule(static)
                for(int job=0;job<E*(F/64);++job) {
                    auto& w=experts[l][job/(F/64)];const int row=(job%(F/64))*64;
                    w4a16_projection(w.gate,input,w.g,row);w4a16_projection(w.up,input,w.u,row);
                    hidden(w,row,false);
                }
                #pragma omp for schedule(static)
                for(int job=0;job<E*(H/64);++job) {
                    auto& w=experts[l][job/(H/64)];w4a16_projection(w.down,w.h,w.y,(job%(H/64))*64);
                }
                #pragma omp for schedule(static)
                for(int j=0;j<H;++j)reduce(l,j);
            }
        }
    }
    void run_reference() {
        if(planar||local_rows)throw std::runtime_error("reference requires original adjacent weights");
        input=initial;
        for(int l=0;l<L;++l) {
            for(auto& w:experts[l]) {
                reference_projection(w.gate,input,w.g);reference_projection(w.up,input,w.u);
                #pragma omp parallel for schedule(static)
                for(int row=0;row<F;row+=64)hidden(w,row,false);
                reference_projection(w.down,w.h,w.y);
            }
            #pragma omp parallel for schedule(static)
            for(int j=0;j<H;++j)reduce(l,j);
        }
    }
    void run(bool persistent,bool fast=true,bool vector_quant=false,int tile=4,bool fuse_hidden=false) {
        input=initial;qx.simd=vector_quant;row_tile=tile;
        for(auto& layer:experts)for(auto& expert:layer)expert.qh.simd=vector_quant;
        if(persistent) {
            #pragma omp parallel
            {
                for(int l=0;l<L;++l) {
                    #pragma omp for schedule(static)
                    for(int j=0;j<H;j+=64)qx.block(input.data()+j,j);
                    #pragma omp for schedule(static)
                    for(int job=0;job<E*(F/64);++job) {
                        auto& w=experts[l][job/(F/64)];int row=(job%(F/64))*64;
                        projection(w.gate,qx,w.g,row,fast);projection(w.up,qx,w.u,row,fast);
                        if(fuse_hidden)hidden(w,row);
                    }
                    if(!fuse_hidden) {
                        #pragma omp for schedule(static)
                        for(int job=0;job<E*(F/64);++job)hidden(experts[l][job/(F/64)],(job%(F/64))*64);
                    }
                    #pragma omp for schedule(static)
                    for(int job=0;job<E*(H/64);++job) {
                        auto& w=experts[l][job/(H/64)];projection(w.down,w.qh,w.y,(job%(H/64))*64,fast);
                    }
                    #pragma omp for schedule(static)
                    for(int j=0;j<H;++j)reduce(l,j);
                }
            }
        } else for(int l=0;l<L;++l) {
            #pragma omp parallel for schedule(static)
            for(int j=0;j<H;j+=64)qx.block(input.data()+j,j);
            for(auto& w:experts[l]) {
                #pragma omp parallel for schedule(static)
                for(int r=0;r<F;r+=64)projection(w.gate,qx,w.g,r,fast);
                #pragma omp parallel for schedule(static)
                for(int r=0;r<F;r+=64)projection(w.up,qx,w.u,r,fast);
                #pragma omp parallel for schedule(static)
                for(int r=0;r<F;r+=64)hidden(w,r);
                #pragma omp parallel for schedule(static)
                for(int r=0;r<H;r+=64)projection(w.down,w.qh,w.y,r,fast);
            }
            #pragma omp parallel for schedule(static)
            for(int j=0;j<H;++j)reduce(l,j);
        }
    }
};
int main(int argc,char** argv) {try {
    const bool check_only=argc==3&&std::string(argv[2])=="--check-only";
    const bool placement=argc==3&&std::string(argv[2])=="--placement-bench";
    const bool layout_bench=argc==3&&std::string(argv[2])=="--layout-bench";
    const bool quality=argc==3&&std::string(argv[2])=="--quality-check";
    const bool precision_bench=argc==3&&std::string(argv[2])=="--precision-bench";
    if(argc!=2&&!check_only&&!placement&&!layout_bench&&!quality&&!precision_bench)throw std::runtime_error("usage: glm-fp4-cpu-chain PARTIAL_MODEL_DIR [--check-only|--placement-bench|--layout-bench|--quality-check|--precision-bench]");
    if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("FE_TONEAREST required");
    omp_set_dynamic(0);Chain c(argv[1]);
    if(quality) {
        omp_set_num_threads(16);
        Matrix known;known.rows=3;known.cols=64;known.global=.375f;
        known.codes.assign(3*32,0x11);known.scales.assign(3*4,56);
        std::vector<float> x(64,1.f),y(3);
        reference_projection(known,x,y);
        if(y!=std::vector<float>(3,12.f))throw std::runtime_error("reference analytic fixture failed");
        std::fill(x.begin(),x.end(),1.00390625f);reference_projection(known,x,y);
        if(y!=std::vector<float>(3,12.f))throw std::runtime_error("reference BF16 tie fixture failed");
        known.codes.assign(3*32,0x99);reference_projection(known,x,y);
        if(y!=std::vector<float>(3,-12.f))throw std::runtime_error("reference sign fixture failed");
        std::cout<<"reference=BF16_operands_FP64_sum_global_last inputs=synthetic layers=2 experts_per_layer=3 residual=synthetic not_model_quality=true\n";
        for(const std::string pattern:{"sine","normal","outlier","sparse"}) {
            std::mt19937 rng(71);std::normal_distribution<float> normal(0,.5f);
            for(int i=0;i<H;++i) {
                float v=pattern=="sine"?.5f*std::sin(float(i)*.013f):normal(rng);
                if(pattern=="outlier"&&i%64==0)v=16.f;
                if(pattern=="sparse"&&i%64)v=0;
                c.initial[i]=bf16(v);
            }
            uint64_t input_hash=14695981039346656037ull;
            for(float v:c.initial){const auto bits=std::bit_cast<uint32_t>(v);for(int j=0;j<4;++j){input_hash^=(bits>>(j*8))&255;input_hash*=1099511628211ull;}}
            std::cout<<"pattern="<<pattern<<" input_fnv1a64="<<input_hash<<"\n";
            c.run_reference();const auto reference=c.input;
            std::array<std::vector<float>,L> down_reference;
            for(int l=0;l<L;++l)for(const auto& e:c.experts[l])down_reference[l].insert(down_reference[l].end(),e.y.begin(),e.y.end());
            c.run_w4a16();
            error_report("pattern="+pattern+" mode=w4a16_simd scope=final_residual",reference,c.input);
            for(int l=0;l<L;++l) {
                std::vector<float> down;for(const auto& e:c.experts[l])down.insert(down.end(),e.y.begin(),e.y.end());
                error_report("pattern="+pattern+" mode=w4a16_simd scope=expert_down layer="+std::to_string(l+3),down_reference[l],down);
            }
            const auto w4a16_adjacent=c.input;
            c.set_planar(true);c.run_w4a16();
            if(c.input!=w4a16_adjacent)throw std::runtime_error("packed W4A16 quality output differs");
            c.set_planar(false);
            std::vector<float> strict;
            for(bool tree:{false,true}) {
                c.run(true,tree,true,tree?4:1,true);
                const std::string tag="pattern="+pattern+" mode="+(tree?"q8_tree":"q8_strict");
                error_report(tag+" scope=final_residual",reference,c.input);
                for(int l=0;l<L;++l) {
                    std::vector<float> down;
                    for(const auto& e:c.experts[l])down.insert(down.end(),e.y.begin(),e.y.end());
                    error_report(tag+" scope=expert_down layer="+std::to_string(l+3),down_reference[l],down);
                }
                if(!tree)strict=c.input;
                else error_report("pattern="+pattern+" scope=tree_vs_strict",strict,c.input);
            }
            const auto adjacent=c.input;
            c.set_planar(true);c.run(true,true,true,4,true);
            if(c.input!=adjacent)throw std::runtime_error("packed quality output differs");
            c.set_planar(false);
        }
        std::cout<<"PASS: finite reference comparisons and exact packed parity; no model-quality acceptance gate\n";
        return 0;
    }
    omp_set_num_threads(1);c.run(false,false);const auto strict=c.input;
    c.run(false,true);const auto expected=c.input;
    double error=0,norm=0;
    for(int j=0;j<H;++j){double d=double(expected[j])-strict[j];error+=d*d;norm+=double(strict[j])*strict[j];}
    std::cout<<"partial_layers=2 experts_per_layer=3 tokens=1 Q8=lossy global_scale=last BF16_clamp=10 residual=synthetic\n"
             <<"tree_vs_strict_q8_chain_relative_l2="<<std::sqrt(error/std::max(norm,1e-30))<<"\n";
    if(precision_bench) {
        c.set_planar(true);c.run_w4a16();const auto expected_w4a16=c.input;
        auto execute=[&](int mode){if(mode==2)c.run_w4a16();else c.run(true,true,true,mode==0?1:4,true);};
        const char* names[]={"q8_planar_r1","q8_planar_r4","w4a16_planar_r1"};
        for(int threads:{8,16}) {
            omp_set_num_threads(threads);std::vector<double> samples[3];
            for(int mode=0;mode<3;++mode){execute(mode);if(c.input!=(mode==2?expected_w4a16:expected))throw std::runtime_error("precision warmup mismatch");}
            for(int repeat=0;repeat<7;++repeat)for(int order=0;order<3;++order) {
                const int mode=(repeat+order)%3;const double start=omp_get_wtime();
                for(int step=0;step<10;++step)execute(mode);
                samples[mode].push_back((omp_get_wtime()-start)*1000/10);
                if(c.input!=(mode==2?expected_w4a16:expected))throw std::runtime_error("precision output changed across repeats/threads");
            }
            for(int mode=0;mode<3;++mode) {
                std::cout<<"threads="<<threads<<" precision_mode="<<names[mode]<<" ms=";
                for(double ms:samples[mode])std::cout<<ms<<",";
                std::sort(samples[mode].begin(),samples[mode].end());std::cout<<" median_ms="<<samples[mode][3]<<"\n";
            }
        }
        c.set_planar(false);
        std::cout<<"PASS: per-mode output invariant across threads and repeats; precision modes have different arithmetic\n";
        return 0;
    }
    if(layout_bench) {
        for(int threads:{8,16}) {
            omp_set_num_threads(threads);std::vector<double> samples[4];
            for(int repeat=0;repeat<7;++repeat)for(int order=0;order<4;++order) {
                const int mode=(repeat+order)%4;const int tile=mode%2?4:1;
                c.set_planar(mode>=2);c.run(true,true,true,tile,true);
                if(c.input!=expected)throw std::runtime_error("layout warmup changed chain output");
                const double start=omp_get_wtime();
                for(int step=0;step<10;++step)c.run(true,true,true,tile,true);
                samples[mode].push_back((omp_get_wtime()-start)*1000/10);
                if(c.input!=expected)throw std::runtime_error("layout changed chain output");
            }
            for(int mode=0;mode<4;++mode) {
                std::cout<<"threads="<<threads<<" layout="<<(mode>=2?"planar_k32":"adjacent")<<" tile="<<(mode%2?4:1)<<" ms=";
                for(double ms:samples[mode])std::cout<<ms<<",";
                std::sort(samples[mode].begin(),samples[mode].end());std::cout<<" median_ms="<<samples[mode][3]<<"\n";
            }
        }
        c.set_planar(false);c.run(true,true,true,4,true);
        if(c.input!=expected)throw std::runtime_error("restored chain output mismatch");
        std::cout<<"PASS: in-place layout transitions and complete chain outputs match original tree reference\n";
        return 0;
    }
    if(placement) {
        for(int threads:{8,16}) {
            omp_set_num_threads(threads);c.place_rows();c.placement_report();
            std::vector<double> samples[4];
            for(int repeat=-1;repeat<7;++repeat)for(int order=0;order<4;++order) {
                const int mode=repeat<0?order:(repeat+order)%4;
                c.local_rows=mode>=2;const int tile=mode%2?4:1;
                const double start=omp_get_wtime();
                for(int step=0;step<10;++step)c.run(true,true,true,tile,true);
                const double ms=(omp_get_wtime()-start)*1000/10;
                if(c.input!=expected)throw std::runtime_error("placement changed chain output");
                if(repeat>=0)samples[mode].push_back(ms);
            }
            for(int mode=0;mode<4;++mode) {
                std::cout<<"threads="<<threads<<" placement="<<(mode>=2?"worker_first_touch":"loader")<<" tile="<<(mode%2?4:1)<<" ms=";
                for(double ms:samples[mode])std::cout<<ms<<",";
                std::sort(samples[mode].begin(),samples[mode].end());std::cout<<" median_ms="<<samples[mode][3]<<"\n";
            }
        }
        std::cout<<"PASS: placed weight bytes and full chain outputs match loader reference\n";
        return 0;
    }
    const char* names[]={"separate_r4","persistent_r4","persistent_vector_r4","persistent_vector_r1","persistent_vector_r2","fused_vector_r1","fused_vector_r4"};
    constexpr int modes=7;
    auto execute=[&](int mode){c.run(mode!=0,true,mode>=2,mode==3||mode==5?1:mode==4?2:4,mode>=5);};
    for(int threads:{1,8,16}) {
        omp_set_num_threads(threads);std::vector<double> samples[modes];
        for(int mode=0;mode<modes;++mode){execute(mode);if(c.input!=expected)throw std::runtime_error("dispatch/thread count changed output");}
        if(check_only)continue;
        for(int repeat=0;repeat<7;++repeat)for(int order=0;order<modes;++order) {
            int mode=(repeat+order)%modes;double start=omp_get_wtime();
            for(int step=0;step<10;++step)execute(mode);
            samples[mode].push_back((omp_get_wtime()-start)*1000/10);
            if(c.input!=expected)throw std::runtime_error("repeat output mismatch");
        }
        for(int mode=0;mode<modes;++mode) {
            std::cout<<"threads="<<threads<<" mode="<<names[mode]<<" ms=";
            for(double ms:samples[mode])std::cout<<ms<<",";
            std::sort(samples[mode].begin(),samples[mode].end());
            std::cout<<" median_ms="<<samples[mode][3]<<"\n";
        }
    }
    std::cout<<"PASS: complete Q8 partial-chain output invariant across dispatch modes and thread counts\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
