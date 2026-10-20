// CPU-only local NVFP4/Q8 parity and optional real-checkpoint microbenchmark.
#include "strata/kernels/cpu/fp4_local_q8.hpp"
#include "strata/artifact/glm_fp4.hpp"
#include <algorithm>
#include <array>
#include <ctime>
#include <limits>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <omp.h>
#include <sys/mman.h>
#include <memory>
using strata::kernels::cpu::fp4_local_q8_rows;
struct Input {
    int k,m;std::vector<int8_t> x;std::vector<float> scales;std::vector<int16_t> sums;
    Input(int K,int M,int pattern=0):k(K),m(M),x(K*M),scales(K/64*M),sums(K/16*M) {
        std::mt19937 rng(71);
        for(auto& v:x)v=pattern==1?-128:pattern==2?127:int(rng()%256)-128;
        for(auto& v:scales)v=float(1+rng()%100)/1024;
        for(int t=0;t<m;++t)for(int g=0;g<k/16;++g)
            for(int j=0;j<16;++j)sums[t*(k/16)+g]+=x[t*k+g*16+j];
    }
};
void run(const strata::artifact::GlmFp4Matrix& w,const Input& a,std::vector<float>& y,bool simd,int lo,int hi,bool tree=false,int tile=1,bool wide=false) {
    fp4_local_q8_rows(w.codes.data(),w.scales.data(),a.x.data(),a.scales.data(),a.sums.data(),w.global,w.cols,w.rows,a.m,lo,hi,y.data(),simd,tree,tile,wide);
}
void check(const strata::artifact::GlmFp4Matrix& w,int m,int pattern=0,bool tree=false,int tile=1,bool wide=false) {
    Input a(w.cols,m,pattern);std::vector<float> ref(w.rows*m,-123),got=ref;
    run(w,a,ref,false,1,w.rows-1);run(w,a,got,true,1,w.rows-1,tree,tile,wide);
    if(m==1&&tree&&!wide&&(tile==1||tile==4)&&strata::kernels::cpu::fp4_local_q8_half_available()) {
        std::vector<uint16_t> hs(w.scales.size());
        for(size_t i=0;i<hs.size();++i)hs[i]=strata::kernels::cpu::fp4_local_half_scale(w.scales[i]);
        std::vector<float> half_out(w.rows,-123);
        strata::kernels::cpu::fp4_local_q8_half_rows(w.codes.data(),hs.data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,1,w.rows-1,half_out.data(),tile);
        if(half_out!=got)throw std::runtime_error("half scale cache changed tree bits");
    }
    if(tree&&(tile==1||tile==4)) {
        std::vector<uint8_t> packed(w.codes.size());
        if(strata_glm_fp4_repack_k32(w.codes.data(),packed.data(),packed.size(),0))throw std::runtime_error("packing failed");
        std::vector<float> planar(w.rows*m,-123);
        strata::kernels::cpu::fp4_local_q8_planar_rows(packed.data(),w.scales.data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,m,1,w.rows-1,planar.data(),tile,wide);
        if(planar!=got)throw std::runtime_error("planar layout changed tree bits");
    }
    for(int t=0;t<m;++t)if(got[t*w.rows]!=-123||got[(t+1)*w.rows-1]!=-123)throw std::runtime_error("row guard changed");
    if(!tree&&std::memcmp(ref.data(),got.data(),ref.size()*4))throw std::runtime_error("scalar/AVX2 bits differ");
    // Independent FP64 dot against the exact Q8-dequantized operands.
    for(int t=0;t<m;++t)for(size_t r=1;r+1<w.rows;++r) {
        double sum=0,norm=0;
        for(size_t j=0;j<w.cols;++j) {
            size_t i=r*w.cols+j;
            double v=double(strata_glm_e2m1((w.codes[i/2]>>((i%2)*4))&15))*strata_glm_e4m3(w.scales[i/16]);
            v*=double(a.x[t*w.cols+j])*a.scales[t*(w.cols/64)+j/64];sum+=v;norm+=std::abs(v);
        }
        if(!std::isfinite(got[t*w.rows+r])||std::abs(got[t*w.rows+r]-sum*w.global)>1e-5*norm*w.global+1e-6)
            throw std::runtime_error("FP64 bound failed");
    }
}
void quantizer_checks() {
    using strata::kernels::cpu::fp4_local_quantize_q8_64;
    std::mt19937 rng(913);
    for(int pattern=0;pattern<1024;++pattern) {
        std::array<float,64> x{};
        for(auto& v:x)v=pattern==0?0.f:std::ldexp(float(int(rng()%20001)-10000),int(rng()%240)-140);
        if(pattern==1){x.fill(0);x[0]=.5f;x[1]=1.5f;x[2]=-1.5f;x[3]=-126.5f;x[4]=125.5f;x[63]=127;}
        if(pattern==2)x.fill(std::numeric_limits<float>::max());
        if(pattern==3)x.fill(std::numeric_limits<float>::denorm_min());
        std::array<int8_t,66> a,b;a.fill(85);b=a;
        std::array<int16_t,6> sa,sb;sa.fill(1234);sb=sa;float da=-1,db=-1;
        if(!fp4_local_quantize_q8_64(x.data(),a.data()+1,&da,sa.data()+1,false)||
           !fp4_local_quantize_q8_64(x.data(),b.data()+1,&db,sb.data()+1,true)||a!=b||sa!=sb||std::memcmp(&da,&db,4))
            throw std::runtime_error("Q8 quantizer parity failed");
        if(a.front()!=85||a.back()!=85||sa.front()!=1234||sa.back()!=1234)throw std::runtime_error("Q8 quantizer guard changed");
        if(pattern==1&&(da!=1||a[1]!=0||a[2]!=2||a[3]!=-2||a[4]!=-126||a[5]!=126))throw std::runtime_error("Q8 nearest-even ties failed");
    }
    for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})for(bool simd:{false,true}) {
        std::array<float,64> x{};x.back()=bad;
        std::array<int8_t,64> out;out.fill(85);std::array<int16_t,4> sums;sums.fill(1234);float scale=-1;
        if(fp4_local_quantize_q8_64(x.data(),out.data(),&scale,sums.data(),simd)||scale!=-1||
           std::any_of(out.begin(),out.end(),[](auto v){return v!=85;})||std::any_of(sums.begin(),sums.end(),[](auto v){return v!=1234;}))
            throw std::runtime_error("Q8 nonfinite rejection wrote output");
    }
    std::cout<<"PASS: Q8 scalar/AVX2 scale, byte and sum parity; ties, extremes, guards and nonfinite rejection\n";
}
void quantizer_bench() {
    constexpr int blocks=256,passes=512;
    std::vector<float> x(blocks*64);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.013f)*3.f;
    std::vector<int8_t> out(blocks*64),reference(out.size());std::vector<float> scales(blocks),ref_scales(blocks);
    std::vector<int16_t> sums(blocks*4),ref_sums(sums.size());
    std::vector<double> wall[2],cpu[2];
    for(int repeat=-1;repeat<7;++repeat)for(int order=0;order<2;++order) {
        int mode=repeat<0?order:(repeat+order)%2;
        const double start=omp_get_wtime();const auto ticks=std::clock();
        for(int pass=0;pass<passes;++pass)for(int b=0;b<blocks;++b)
            if(!strata::kernels::cpu::fp4_local_quantize_q8_64(x.data()+64*b,out.data()+64*b,scales.data()+b,sums.data()+4*b,mode))
                throw std::runtime_error("quantizer benchmark rejected input");
        const double cpu_ms=1000.*double(std::clock()-ticks)/CLOCKS_PER_SEC,wall_ms=1000*(omp_get_wtime()-start);
        if(repeat==-1&&mode==0){reference=out;ref_scales=scales;ref_sums=sums;}
        if(out!=reference||scales!=ref_scales||sums!=ref_sums)throw std::runtime_error("quantizer benchmark changed bytes");
        if(repeat>=0){wall[mode].push_back(wall_ms);cpu[mode].push_back(cpu_ms);}
    }
    std::cout<<"quantizer blocks_per_sample="<<blocks*passes<<" working_set_bytes="<<blocks*(64*4+64+4+8)<<" threads=1 repeats=7\n";
    for(int mode=0;mode<2;++mode) {
        std::cout<<"quantizer mode="<<(mode?"avx2":"scalar")<<" wall_ms=";for(auto t:wall[mode])std::cout<<t<<",";
        std::cout<<" cpu_ms=";for(auto t:cpu[mode])std::cout<<t<<",";
        std::sort(wall[mode].begin(),wall[mode].end());std::sort(cpu[mode].begin(),cpu[mode].end());
        std::cout<<" median_wall_ms="<<wall[mode][3]<<" median_cpu_ms="<<cpu[mode][3]<<"\n";
    }
}
// Pinned single-core kernel study for perf stat. Keep setup small and report
// process CPU time as well as wall time; no model throughput interpretation.
void kernel_bench(int mode,int repeats,int batch=1) {
    if(batch<1||batch>8||(batch>1&&mode!=1&&mode!=2&&mode!=10))throw std::runtime_error("batched kernel bench requires mode 1/2 and batch 1..8");
    if(mode<1||mode>13||repeats<1||repeats>100000)throw std::runtime_error("kernel bench mode 1..13, repeats 1..100000");
    strata::artifact::GlmFp4Matrix w;w.rows=128;w.cols=4096;w.global=.375f;
    w.codes.resize(w.rows*w.cols/2);w.scales.resize(w.rows*w.cols/16);
    std::mt19937 rng(19);for(auto& v:w.codes)v=rng()%256;for(auto& v:w.scales)v=rng()%127;
    Input a(w.cols,batch);std::vector<float> y(w.rows*batch),ref(w.rows*batch);
    run(w,a,ref,false,0,w.rows);
    std::vector<uint16_t> hs(w.scales.size());for(size_t i=0;i<hs.size();++i)hs[i]=strata::kernels::cpu::fp4_local_half_scale(w.scales[i]);
    std::vector<uint8_t> planar(w.codes.size());
    if(strata_glm_fp4_repack_k32(w.codes.data(),planar.data(),planar.size(),0))throw std::runtime_error("kernel packing failed");
    auto execute=[&]{
        if(mode>=10)strata::kernels::cpu::fp4_local_q8_planar_rows(planar.data(),w.scales.data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,batch,0,w.rows,y.data(),mode==11||mode==13?4:1,mode>=12);
        else if(mode>=8)strata::kernels::cpu::fp4_local_q8_half_rows(w.codes.data(),hs.data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,0,w.rows,y.data(),mode==9?4:1);
        else run(w,a,y,true,0,w.rows,mode>=2,mode==3||mode==6?2:mode==4||mode==7?4:1,mode>=5);
    };
    execute();double err=0,norm=0;for(size_t i=0;i<y.size();++i){double d=double(y[i])-ref[i];err+=d*d;norm+=double(ref[i])*ref[i];}
    if(mode==1&&y!=ref)throw std::runtime_error("kernel bench strict mismatch");
    const double rel=std::sqrt(err/std::max(norm,1e-30));
    if(!std::isfinite(rel)||rel>1e-5)throw std::runtime_error("kernel bench error bound");
    const auto cpu0=std::clock();const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;++i)execute();
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double cpu=double(std::clock()-cpu0)/CLOCKS_PER_SEC;
    double checksum=0;for(float v:y)checksum+=v;
    std::cout<<"kernel_bench mode="<<mode<<" tokens="<<batch<<" rows="<<w.rows<<" columns="<<w.cols<<" repeats="<<repeats
             <<" weight_scale_bytes="<<w.bytes()+((mode==8||mode==9)?w.scales.size():0)<<" wall_s="<<wall<<" cpu_s="<<cpu<<" relative_l2="<<rel<<" checksum="<<checksum<<"\n";
}
int main(int argc,char** argv) {try {
    if(!strata::kernels::cpu::fp4_local_q8_avx2_available())throw std::runtime_error("AVX2/FMA required for this test");
    if((argc==4||argc==5)&&std::string(argv[1])=="--kernel-bench"){kernel_bench(std::stoi(argv[2]),std::stoi(argv[3]),argc==5?std::stoi(argv[4]):1);return 0;}
    for(uint8_t invalid:{uint8_t(127),uint8_t(255)}) {
        bool rejected=false;try{(void)strata::kernels::cpu::fp4_local_half_scale(invalid);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid half-scale code accepted");
    }
    quantizer_checks();
    if(argc==2&&std::string(argv[1])=="--quant-bench"){quantizer_bench();return 0;}
    std::mt19937 rng(19);
    for(int k:{64,128,2048,4096})for(int m:{1,2,3,4,5,6,7,8})for(int pattern:{0,1,2}) {
        strata::artifact::GlmFp4Matrix w;w.rows=7;w.cols=k;w.global=.375;
        w.codes.resize(w.rows*k/2);w.scales.resize(w.rows*k/16);
        for(auto& v:w.codes)v=rng()%256;
        for(size_t i=0;i<w.scales.size();++i)w.scales[i]=i%127;
        check(w,m,pattern);check(w,m,pattern,true);
        if(m==1)for(int tile:{1,2,4}){check(w,m,pattern,true,tile);check(w,m,pattern,true,tile,true);}
    }
    // Exhaust all scale/code combinations, including negative zero.
    strata::artifact::GlmFp4Matrix all;all.rows=129;all.cols=64;all.global=1;
    all.codes.resize(129*32);all.scales.resize(129*4);
    for(int r=0;r<129;++r) {
        for(int j=0;j<64;++j)all.codes[r*32+j/2]|=uint8_t((j%16)<<((j%2)*4));
        for(int g=0;g<4;++g)all.scales[r*4+g]=std::max(0,r-1)%127;
    }
    check(all,1,1);check(all,1,2);check(all,1,1,true);check(all,1,2,true);
    for(int tile:{1,2,4})for(int pattern:{1,2})check(all,1,pattern,true,tile,true);
    std::cout<<"PASS: scalar/AVX2 bit parity, FP64 bounds, row guards, all scale/code pairs, -128 and +127 activations\n";
    if(argc==1)return 0;
    bool local=false;int batch=1;std::vector<int> thread_counts={1,8,16};
    for(int i=2;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--local-rows")local=true;
        else if(arg.starts_with("--batch="))batch=std::stoi(arg.substr(8));
        else if(arg.starts_with("--threads="))thread_counts.assign(1,std::stoi(arg.substr(10)));
        else throw std::runtime_error("unknown benchmark option");
    }
    if(batch<1||batch>8||thread_counts[0]<1||thread_counts[0]>32)throw std::runtime_error("batch 1..8 and threads 1..32 required");
    std::vector<int> modes={0,1,2};
    if(batch==1){for(int mode=3;mode<8;++mode)modes.push_back(mode);if(strata::kernels::cpu::fp4_local_q8_half_available()){modes.push_back(8);modes.push_back(9);}}
    modes.push_back(10);if(batch==1){modes.push_back(11);modes.push_back(12);modes.push_back(13);}
    strata::artifact::Safetensors ts(argv[1]);std::vector<strata::artifact::GlmFp4Matrix> matrices;
    for(int layer:{3,4})for(int e:{0,1,2})for(const char* projection:{"gate_proj","up_proj","down_proj"})
        matrices.push_back(strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(e)+"."+projection));
    size_t bytes=0,extra_half_bytes=0;for(auto& w:matrices){bytes+=w.bytes();extra_half_bytes+=w.scales.size();}
    std::vector<Input> inputs;std::vector<std::vector<float>> out,reference;
    for(auto& w:matrices){inputs.emplace_back(w.cols,batch);out.emplace_back(w.rows*batch);reference.emplace_back(w.rows*batch);}
    std::cout<<"real_matrices="<<matrices.size()<<" packed_bytes="<<bytes<<" repeats=7 warmup=1 tokens="<<batch<<" affinity=OMP_ENV allocation="<<(local?"parallel_row_first_touch":"loader_first_touch")<<"\n";
    std::vector<std::vector<uint16_t>> half_scales;
    for(const auto& w:matrices){auto& hs=half_scales.emplace_back(w.scales.size());for(size_t i=0;i<hs.size();++i)hs[i]=strata::kernels::cpu::fp4_local_half_scale(w.scales[i]);}
    std::vector<std::vector<uint8_t>> planar_weights;
    for(const auto& w:matrices){auto& packed=planar_weights.emplace_back(w.codes.size());
        std::vector<uint8_t> unpacked(packed.size());
        if(strata_glm_fp4_repack_k32(w.codes.data(),packed.data(),packed.size(),0)||strata_glm_fp4_repack_k32(packed.data(),unpacked.data(),packed.size(),1)||unpacked!=w.codes)throw std::runtime_error("real planar roundtrip failed");}
    for(int threads:thread_counts) {
        omp_set_num_threads(threads);
        struct LocalMatrix {
            uint8_t *w=nullptr,*s=nullptr,*pw=nullptr;uint16_t* hs=nullptr;size_t nw=0,ns=0;
            ~LocalMatrix(){if(w)munmap(w,nw);if(s)munmap(s,ns);if(hs)munmap(hs,ns*2);if(pw)munmap(pw,nw);}
        };
        std::vector<std::unique_ptr<LocalMatrix>> placed;
        if(local)for(const auto& matrix:matrices) {
            auto p=std::make_unique<LocalMatrix>();p->nw=matrix.codes.size();p->ns=matrix.scales.size();
            p->w=static_cast<uint8_t*>(mmap(nullptr,p->nw,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
            if(p->w==MAP_FAILED){p->w=nullptr;throw std::runtime_error("weight mmap failed");}
            p->s=static_cast<uint8_t*>(mmap(nullptr,p->ns,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
            if(p->s==MAP_FAILED){p->s=nullptr;throw std::runtime_error("scale mmap failed");}
            p->hs=static_cast<uint16_t*>(mmap(nullptr,p->ns*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
            if(p->hs==MAP_FAILED){p->hs=nullptr;throw std::runtime_error("half scale mmap failed");}
            p->pw=static_cast<uint8_t*>(mmap(nullptr,p->nw,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
            if(p->pw==MAP_FAILED){p->pw=nullptr;throw std::runtime_error("planar mmap failed");}
            #pragma omp parallel for schedule(static)
            for(int r=0;r<int(matrix.rows);r+=64) {
                size_t count=std::min(size_t(64),matrix.rows-r);
                for(size_t i=size_t(r)*matrix.cols/16;i<(r+count)*matrix.cols/16;++i)p->hs[i]=strata::kernels::cpu::fp4_local_half_scale(matrix.scales[i]);
                strata_glm_fp4_repack_k32(matrix.codes.data()+size_t(r)*matrix.cols/2,p->pw+size_t(r)*matrix.cols/2,count*matrix.cols/2,0);
                std::memcpy(p->w+size_t(r)*matrix.cols/2,matrix.codes.data()+size_t(r)*matrix.cols/2,count*matrix.cols/2);
                std::memcpy(p->s+size_t(r)*matrix.cols/16,matrix.scales.data()+size_t(r)*matrix.cols/16,count*matrix.cols/16);
            }
            if(std::memcmp(p->w,matrix.codes.data(),p->nw)||std::memcmp(p->s,matrix.scales.data(),p->ns))throw std::runtime_error("NUMA copy changed weights");
            placed.push_back(std::move(p));
        }
        std::vector<double> times[14];double worst_tree_l2=0;
        std::vector<std::vector<float>> tree_reference,wide_reference[2];
        auto timed=[&](int mode){
            bool simd=mode!=0;
            double start=omp_get_wtime();
            for(size_t i=0;i<matrices.size();++i) {
                int rows=matrices[i].rows;
                #pragma omp parallel for schedule(static)
                for(int r=0;r<rows;r+=64) {
                    if(mode>=10) {
                        auto& w=matrices[i];auto& a=inputs[i];
                        strata::kernels::cpu::fp4_local_q8_planar_rows(local?placed[i]->pw:planar_weights[i].data(),local?placed[i]->s:w.scales.data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,batch,r,std::min(r+64,rows),out[i].data(),mode==11||mode==13?4:1,mode>=12);
                    }
                    else if(mode>=8) {
                        auto& w=matrices[i];auto& a=inputs[i];
                        strata::kernels::cpu::fp4_local_q8_half_rows(local?placed[i]->w:w.codes.data(),local?placed[i]->hs:half_scales[i].data(),a.x.data(),a.scales.data(),w.global,w.cols,w.rows,r,std::min(r+64,rows),out[i].data(),mode==9?4:1);
                    }
                    else if(!local)run(matrices[i],inputs[i],simd?out[i]:reference[i],simd,r,std::min(r+64,rows),mode>=2,mode==3||mode==6?2:mode==4||mode==7?4:1,mode>=5);
                    else {
                        auto& w=matrices[i];auto& a=inputs[i];
                        fp4_local_q8_rows(placed[i]->w,placed[i]->s,a.x.data(),a.scales.data(),a.sums.data(),w.global,w.cols,w.rows,batch,r,std::min(r+64,rows),(simd?out[i]:reference[i]).data(),simd,mode>=2,mode==3||mode==6?2:mode==4||mode==7?4:1,mode>=5);
                    }
                }
            }
            return omp_get_wtime()-start;
        };
        for(int mode:modes){timed(mode);if(mode==2)tree_reference=out;if(mode==5)wide_reference[0]=out;if(mode==7)wide_reference[1]=out;}
        for(int repeat=0;repeat<7;++repeat) {
            for(int order=0;order<int(modes.size());++order){int mode=modes[(order+repeat)%modes.size()];times[mode].push_back(timed(mode));
                if(mode>=12&&out!=wide_reference[mode-12])throw std::runtime_error("real planar K64 output bits changed");
                if(mode>=8&&mode<12&&out!=tree_reference)throw std::runtime_error("real half-scale output bits changed");
                if(mode==1)for(size_t i=0;i<matrices.size();++i)if(out[i]!=reference[i])throw std::runtime_error("real strict output mismatch");
                if(mode>=2)for(size_t i=0;i<matrices.size();++i) {
                    double err=0,norm=0;
                    for(size_t j=0;j<out[i].size();++j){double d=double(out[i][j])-reference[i][j];err+=d*d;norm+=double(reference[i][j])*reference[i][j];}
                    const double rel=norm?std::sqrt(err/norm):std::sqrt(err);
                    if(!std::isfinite(rel)||rel>1e-5)throw std::runtime_error("real tree reduction error bound");
                    worst_tree_l2=std::max(worst_tree_l2,rel);
                }
            }

        }
        std::cout<<"threads="<<threads<<" tree_worst_relative_l2_vs_scalar_q8="<<worst_tree_l2<<"\n";
        for(int mode:modes){
            std::cout<<"threads="<<threads<<" mode="<<(mode==13?"avx2_planar_wide_r4":mode==12?"avx2_planar_wide_r1":mode==11?"avx2_planar_r4":mode==10?"avx2_planar_r1":mode==9?"avx2_half_r4":mode==8?"avx2_half_r1":mode==7?"avx2_wide_r4":mode==6?"avx2_wide_r2":mode==5?"avx2_wide_r1":mode==4?"avx2_tree_r4":mode==3?"avx2_tree_r2":mode==2?"avx2_tree":mode==1?"avx2":"scalar")<<" seconds=";
            for(auto t:times[mode])std::cout<<t<<",";
            std::sort(times[mode].begin(),times[mode].end());
            std::cout<<" median_ms="<<times[mode][3]*1000<<" ms_per_token="<<times[mode][3]*1000/batch<<" stored_weight_scale_bytes="<<bytes+((mode==8||mode==9)?extra_half_bytes:0)<<" payload_GB_s="<<bytes/times[mode][3]/1e9<<"\n";
        }
    }
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
