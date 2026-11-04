#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <limits>
using namespace strata::kernels::cpu;
int main(){try {
    ExpertPool pool(15,true);if(!pool.numa_rows_available())return 77;
    NativeFmt f;std::string error;if(!native_fmt(17,18,256,256,f,error))throw std::runtime_error(error);f.swiglu_limit=10;
    const int count=8;std::vector<uint8_t> g(count*f.up_off),u(g.size()),d(count*(f.bytes-f.down_off));std::mt19937 rng(7);
    for(auto* v:{&g,&u,&d})for(auto& b:*v)b=rng();
    for(int e=0;e<count;++e)for(int row=0;row<256;++row)for(auto part:{0,1,2}) {
        auto& v=part==0?g:part==1?u:d;size_t stride=part==2?f.d_row:f.gu_row;uint16_t scale=0x2400;
        std::memcpy(v.data()+(e*256+row)*stride,&scale,2);
    }
    NumaTensor ng(g.data(),f.gu_row,256,count,pool.numa_cores()),nu(u.data(),f.gu_row,256,count,pool.numa_cores()),nd(d.data(),f.d_row,256,count,pool.numa_cores());
    std::vector<uint8_t> copied(f.up_off);for(int e=0;e<count;++e){ng.copy(e,copied.data());if(std::memcmp(copied.data(),g.data()+e*f.up_off,f.up_off))throw std::runtime_error("NUMA copy changed packed bytes");}
    auto cores=physical_cores(false);auto affinity=pin_current_thread(cores[0]);
    int batches=0;
    for(int nt:{1,2,4})for(int n:{1,2,8}) {
        std::vector<std::vector<uint8_t>> act(nt,std::vector<uint8_t>(f.act_bytes));std::vector<float> a(256);
        for(int t=0;t<nt;++t){for(auto& v:a)v=float(int(rng()%201)-100)*.01f;native_quant_act(f,a.data(),act[t].data());}
        std::vector<float> reference(n*nt*256),actual(reference.size());std::vector<ExpertJobMulti> jobs(n);
        for(int e=0;e<n;++e){auto& j=jobs[e];j.nt=nt;j.blob=g.data()+e*f.up_off;j.native_up=u.data()+e*f.up_off;j.native_down=d.data()+e*256*f.d_row;
            for(int t=0;t<nt;++t){j.nact[t]=act[t].data();j.out[t]=reference.data()+(e*nt+t)*256;}}
        pool.run_split_multi_native(f,jobs.data(),n);
        for(int e=0;e<n;++e){auto& j=jobs[e];for(int node=0;node<2;++node)j.numa[node]={ng.shard(e,node),nu.shard(e,node),nd.shard(e,node)};for(int t=0;t<nt;++t)j.out[t]=actual.data()+(e*nt+t)*256;}
        for(int repetition=0;repetition<16;++repetition){std::fill(actual.begin(),actual.end(),std::numeric_limits<float>::quiet_NaN());pool.run_split_multi_native(f,jobs.data(),n);if(actual!=reference)throw std::runtime_error("NUMA pool row coverage/arithmetic mismatch");++batches;}
        jobs[0].numa[1].up=nullptr;bool rejected=false;try{pool.run_split_multi_native(f,jobs.data(),n);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("partial NUMA shards accepted");
    }
    restore_thread_affinity(affinity);std::cout<<"NUMA pool PASS "<<batches<<" batches, packed byte parity, complete rows, partial-shard rejection\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
