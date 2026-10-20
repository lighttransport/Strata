// CPU-only real partial-weight benchmark; no full-model routing or throughput claim.
#include "strata/artifact/glm_fp4.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <array>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <memory>
using namespace strata::kernels::cpu;
using strata::artifact::GlmFp4Matrix;
int main(int argc,char**argv){try {
    if(argc<2||argc>9)throw std::runtime_error("usage: glm-fp4-native-pool-bench PARTIAL_MODEL_DIR [EXPERTS(1..3) [WORKERS(1..31) [fixed|task-sweep [BANKS(1..16) [ggml|k32|paired [TOKENS(1|2|4) [CALLS(1..64)]]]]]]]");
    const int E=argc>=3?std::stoi(argv[2]):3;
    if(E<1||E>3)throw std::runtime_error("expert count must be 1..3");
    const int workers=argc>=4?std::stoi(argv[3]):15;
    if(workers<1||workers>31)throw std::runtime_error("workers must be 1..31");
    const std::string benchmark_mode=argc>=5?argv[4]:"fixed";
    const bool sweep=benchmark_mode=="task-sweep";
    if(!sweep&&benchmark_mode!="fixed")throw std::runtime_error("invalid benchmark mode");
    const int banks=argc>=6?std::stoi(argv[5]):1;
    if(banks<1||banks>16)throw std::runtime_error("banks must be 1..16");
    const std::string layout=argc>=7?argv[6]:"ggml";
    if(layout!="ggml"&&layout!="k32"&&layout!="paired")throw std::runtime_error("invalid weight layout");
    const bool paired=layout=="paired",packed_k32=layout!="ggml";
    const int selected_tokens=argc>=8?std::stoi(argv[7]):0;
    if(argc>=8&&selected_tokens!=1&&selected_tokens!=2&&selected_tokens!=4)
        throw std::runtime_error("tokens must be 1, 2 or 4");
    const int latency_calls=argc==9?std::stoi(argv[8]):1;
    if(latency_calls<1||latency_calls>64)throw std::runtime_error("calls must be 1..64");
    if(!native_experts_available()||(packed_k32&&!cpu_avx2_ok()))return 77;
    ggml_cpu_init();ExpertPool pool(workers,true);
    const int modes=pool.numa_rows_available()?6:2;
    constexpr int tasks[]={1,2,3,4,6};
    const int variants=(sweep?modes*int(std::size(tasks)):modes)*(paired?2:1);
    auto base_variant=[&](int variant){return paired?variant/2:variant;};
    auto variant_mode=[&](int variant){return base_variant(variant)%modes;};
    const auto previous=pin_current_thread(planned_host_core());
    strata::artifact::Safetensors ts(argv[1]);
    constexpr int H=4096,F=2048;
    NativeFmt f;std::string error;
    if(!native_fmt(kNvfp4Type,kNvfp4Type,H,F,f,error))throw std::runtime_error(error);
    f.swiglu_limit=10;
    for(int layer:{3,4}) {
        std::vector<std::vector<uint8_t>> blobs(E);
        std::vector<uint8_t> gate(E*f.up_off),up(gate.size()),down(E*(f.tail_off-f.down_off));
        for(int e=0;e<E;++e) {
            auto& b=blobs[e];b.resize(f.bytes);float tail[4]={};
            int p=0;const size_t offsets[]={0,f.up_off,f.down_off};
            for(const char* name:{"gate_proj","up_proj","down_proj"}) {
                const auto m=strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(e)+"."+name);
                if(m.rows!=size_t(p==2?H:F)||m.cols!=size_t(p==2?F:H))throw std::runtime_error("unexpected real expert geometry");
                if(strata_glm_fp4_pack_nvfp4(m.rows,m.cols,m.codes.data(),m.codes.size(),m.scales.data(),m.scales.size(),b.data()+offsets[p],m.rows*m.cols/64*36))throw std::runtime_error("native packing failed");
                tail[p++]=m.global;
            }
            std::memcpy(b.data()+f.tail_off,tail,sizeof(tail));
            std::memcpy(gate.data()+e*f.up_off,b.data(),f.up_off);
            std::memcpy(up.data()+e*f.up_off,b.data()+f.up_off,f.up_off);
            std::memcpy(down.data()+e*(f.tail_off-f.down_off),b.data()+f.down_off,f.tail_off-f.down_off);
        }
        std::vector<uint8_t> original_gate,original_up,original_down;
        if(paired){original_gate=gate;original_up=up;original_down=down;}
        std::vector<std::vector<uint8_t>> reference_blobs;
        if(packed_k32) {
            reference_blobs=blobs;
            for(int e=0;e<E;++e) {
                auto& b=blobs[e];
                if(strata_glm_nvfp4_repack_k32(b.data(),b.data(),f.tail_off))throw std::runtime_error("native K32 conversion failed");
                std::memcpy(gate.data()+e*f.up_off,b.data(),f.up_off);
                std::memcpy(up.data()+e*f.up_off,b.data()+f.up_off,f.up_off);
                std::memcpy(down.data()+e*(f.tail_off-f.down_off),b.data()+f.down_off,f.tail_off-f.down_off);
            }
        }
        f.nvfp4_k32=packed_k32;
        std::vector<std::vector<std::vector<uint8_t>>> extra_banks;
        for(int bank=1;bank<banks;++bank)extra_banks.push_back(blobs);
        std::vector<std::unique_ptr<NumaTensor>> ng,nu,nd;
        if(modes==6)for(int bank=0;bank<banks;++bank) {
            ng.push_back(std::make_unique<NumaTensor>(gate.data(),f.gu_row,F,E,pool.numa_cores(),false));
            nu.push_back(std::make_unique<NumaTensor>(up.data(),f.gu_row,F,E,pool.numa_cores(),false));
            nd.push_back(std::make_unique<NumaTensor>(down.data(),f.d_row,H,E,pool.numa_cores(),false));
        }
        std::vector<std::vector<uint8_t>> alternate_blobs;
        std::vector<std::vector<std::vector<uint8_t>>> alternate_banks;
        std::vector<std::unique_ptr<NumaTensor>> alternate_ng,alternate_nu,alternate_nd;
        if(paired) {
            alternate_blobs=reference_blobs;
            for(int bank=1;bank<banks;++bank)alternate_banks.push_back(alternate_blobs);
            if(modes==6)for(int bank=0;bank<banks;++bank) {
                alternate_ng.push_back(std::make_unique<NumaTensor>(original_gate.data(),f.gu_row,F,E,pool.numa_cores(),false));
                alternate_nu.push_back(std::make_unique<NumaTensor>(original_up.data(),f.gu_row,F,E,pool.numa_cores(),false));
                alternate_nd.push_back(std::make_unique<NumaTensor>(original_down.data(),f.d_row,H,E,pool.numa_cores(),false));
            }
        }
        for(int nt:(selected_tokens?std::vector<int>{selected_tokens}:std::vector<int>{1,4})) {
            f.fuse_h_quant=false;
            std::vector<std::vector<uint8_t>> aq(nt,std::vector<uint8_t>(f.act_bytes)),hq(nt,std::vector<uint8_t>(f.h_bytes));
            std::vector<std::vector<float>> hidden(nt,std::vector<float>(F));
            std::vector<const void*> ap(nt),hp(nt);std::vector<float*> fp(nt),op(nt);
            std::vector<float> x(H),output(size_t(nt)*E*H),weights(nt*E),sum(size_t(nt)*H);
            for(int t=0;t<nt;++t) {
                for(int j=0;j<H;++j)x[j]=float((j*19+t*37)%257-128)/128;
                native_quant_act(f,x.data(),aq[t].data());ap[t]=aq[t].data();hp[t]=hq[t].data();fp[t]=hidden[t].data();
                for(int e=0;e<E;++e)weights[t*E+e]=float(e+1)/6;
            }
            std::vector<ExpertJobMulti> jobs(E);
            // Sequential complete-blob reference; a separate process checks the GGML fallback too.
            for(int e=0;e<E;++e) {
                auto& j=jobs[e];j.blob=blobs[e].data();j.nt=nt;
                for(int t=0;t<nt;++t){j.nact[t]=ap[t];j.out[t]=op[t]=output.data()+(t*E+e)*H;}
                NativeFmt reference_fmt=f;reference_fmt.nvfp4_k32=false;
                const auto* reference_blob=packed_k32?reference_blobs[e].data():j.blob;
                native_gu_rows(reference_fmt,reference_blob,ap.data(),nt,fp.data(),0,F);
                for(int t=0;t<nt;++t)native_quant_h(f,hidden[t].data(),hq[t].data());
                native_down_rows(reference_fmt,reference_blob,hp.data(),nt,op.data(),0,H);
            }
            for(float v:output)if(!std::isfinite(v))throw std::runtime_error("nonfinite reference output");
            pool.reduce_routed(output.data(),weights.data(),sum.data(),nt,E,H);
            const auto expected=output,expected_sum=sum;
            uint64_t hash=14695981039346656037ULL;
            for(const auto* v:{&expected,&expected_sum})for(size_t i=0;i<v->size()*sizeof(float);++i){hash^=reinterpret_cast<const uint8_t*>(v->data())[i];hash*=1099511628211ULL;}
            auto select=[&](int variant) {
                const int mode=variant_mode(variant);
                if(sweep)pool.set_native_tasks_per_thread(tasks[base_variant(variant)/modes]);
                const bool wanted_k32=paired?(variant%2!=0):packed_k32;
                if(wanted_k32!=f.nvfp4_k32) {
                    blobs.swap(alternate_blobs);extra_banks.swap(alternate_banks);
                    ng.swap(alternate_ng);nu.swap(alternate_nu);nd.swap(alternate_nd);
                    f.nvfp4_k32=wanted_k32;
                }
                for(int e=0;e<E;++e)jobs[e].blob=blobs[e].data();
                f.fuse_h_quant=mode>=4;
                for(int e=0;e<E;++e)for(int node=0;node<2;++node)
                    jobs[e].numa[node]=mode>=2?ExpertJobMulti::NumaRows{ng[0]->shard(e,node),nu[0]->shard(e,node),nd[0]->shard(e,node)}:ExpertJobMulti::NumaRows{};
            };
            size_t next_bank[2]={};
            auto run=[&](int mode) {
                if(banks>1) {
                    const size_t bank=next_bank[f.nvfp4_k32]++%banks;
                    auto& selected=bank?extra_banks[bank-1]:blobs;
                    for(int e=0;e<E;++e) {
                        // The complete blob owns global scales; shards own packed rows.
                        jobs[e].blob=selected[e].data();
                        if(mode>=2)for(int node=0;node<2;++node)
                            jobs[e].numa[node]={ng[bank]->shard(e,node),nu[bank]->shard(e,node),nd[bank]->shard(e,node)};
                    }
                }
                if(mode%2)pool.run_layer_native(f,jobs.data(),E,output.data(),weights.data(),sum.data(),nt,E);
                else {pool.run_split_multi_native(f,jobs.data(),E);pool.reduce_routed(output.data(),weights.data(),sum.data(),nt,E,H);}
            };
            std::vector<std::vector<double>> times(variants);
            std::vector<std::array<double,4>> phases(variants);
            for(int r=0;r<9;++r)for(int order=0;order<variants;++order) {
                // Keep each layout pair adjacent and reverse which goes first each round.
                const int variant=paired?((r+order/2)%(variants/2))*2+((order%2)^(r%2)):(r+order)%variants;
                const int mode=variant_mode(variant);select(variant);run(mode);
                const double gu_before=pool.ms_multi_gu,q_before=pool.ms_multi_q,down_before=pool.ms_multi_down;
                const auto start=std::chrono::steady_clock::now();
                for(int call=0;call<latency_calls;++call)run(mode);
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/latency_calls;
                if(std::memcmp(output.data(),expected.data(),output.size()*sizeof(float))||std::memcmp(sum.data(),expected_sum.data(),sum.size()*sizeof(float)))throw std::runtime_error("native routed expert mismatch");
                times[variant].push_back(ms);
                if(!(mode%2)) {
                    const double gu=(pool.ms_multi_gu-gu_before)/latency_calls,q=(pool.ms_multi_q-q_before)/latency_calls,
                                 down=(pool.ms_multi_down-down_before)/latency_calls;
                    phases[variant][0]+=gu;phases[variant][1]+=q;phases[variant][2]+=down;
                    phases[variant][3]+=ms-gu-q-down;
                }
            }
            for(int variant=0;variant<variants;++variant) {
                const int mode=variant_mode(variant);
                const auto ordered_times=times[variant];
                std::sort(times[variant].begin(),times[variant].end());
                // Process-clock sampling is too coarse/expensive for individual sub-ms calls.
                // Keep the latency samples above separate from this longer CPU-cost batch.
                constexpr int cpu_calls=32;
                select(variant);run(mode);
                const auto cpu_start=std::clock();
                const auto batch_start=std::chrono::steady_clock::now();
                for(int call=0;call<cpu_calls;++call)run(mode);
                const double batch_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-batch_start).count();
                const double cpu_ms=1000.0*(std::clock()-cpu_start)/CLOCKS_PER_SEC;
                if(std::memcmp(output.data(),expected.data(),output.size()*sizeof(float))||std::memcmp(sum.data(),expected_sum.data(),sum.size()*sizeof(float)))throw std::runtime_error("CPU batch routed expert mismatch");
                if(sweep)std::cout<<"tasks_per_thread="<<tasks[base_variant(variant)/modes]<<" ";
                std::cout<<"layer="<<layer<<" experts="<<E<<" tokens="<<nt<<" workers="<<workers<<" owned="<<(mode>=2)<<" flow="<<(mode%2)<<" fuse_h_quant="<<(mode>=4)<<" layout="<<(f.nvfp4_k32?"k32":"ggml")<<" paired="<<paired<<" banks="<<banks<<" rotating_bytes="<<size_t(banks)*E*f.bytes<<" packed_bytes="<<E*f.bytes<<" median_ms="<<times[variant][4]<<" cpu_batch_calls="<<cpu_calls<<" cpu_avg_ms="<<cpu_ms/cpu_calls<<" cpu_batch_active_cores="<<cpu_ms/batch_ms<<" hash="<<hash<<" samples=";
                for(double ms:times[variant])std::cout<<ms<<",";
                std::cout<<" ordered_samples=";
                for(double ms:ordered_times)std::cout<<ms<<",";
                std::cout<<" latency_batch_calls="<<latency_calls;
                if(!(mode%2)) {
                    // Same timed calls, arithmetic means; phase medians need not sum.
                    std::cout<<" gu_avg_ms="<<phases[variant][0]/9
                             <<" hidden_quant_avg_ms="<<phases[variant][1]/9
                             <<" down_avg_ms="<<phases[variant][2]/9
                             <<" other_avg_ms="<<phases[variant][3]/9;
                }
                std::cout<<"\n";
            }
        }
    }
    if(previous.valid)restore_thread_affinity(previous);
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
