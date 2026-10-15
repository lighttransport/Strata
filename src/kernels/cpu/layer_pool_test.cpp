#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include <algorithm>
#include <array>
#include <future>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::kernels::cpu;
int main() { try {
    auto cores=physical_cores(false);
    if(cores.size()<4)return 77;
    CpuTopology a,b;
    a.worker_cores={cores[0],cores[1]};b.worker_cores={cores[2],cores[3]};
    ExpertPool first(2,true,true,PoolAffinity::All,0,&a);
    {
        ExpertPool second(2,true,true,PoolAffinity::All,0,&b);
        auto run=[](ExpertPool &pool,int salt) {
            constexpr int nt=4,k=8,h=256;
            std::vector<float> source(nt*k*h),weights(nt*k),actual(nt*h),expected(nt*h);
            for(size_t i=0;i<source.size();++i)source[i]=float(int((i+salt)%37)-18)*.125f;
            for(size_t i=0;i<weights.size();++i)weights[i]=float((i+salt)%7)*.0625f;
            for(int t=0;t<nt;++t)for(int r=0;r<h;++r)for(int e=0;e<k;++e)
                expected[t*h+r]+=weights[t*k+e]*source[(t*k+e)*h+r];
            for(int n=0;n<100;++n) {
                pool.reduce_routed(source.data(),weights.data(),actual.data(),nt,k,h);
                if(actual!=expected)throw std::runtime_error("concurrent pool reduction differs");
            }
        };
        auto one=std::async(std::launch::async,[&]{run(first,1);});
        auto two=std::async(std::launch::async,[&]{run(second,3);});
        one.get();two.get();
        std::FILE *file=std::tmpfile();if(!file)throw std::runtime_error("tmpfile failed");
        auto diag=strata::core::diag_pool_fn().load();if(!diag)throw std::runtime_error("missing diagnostics");
        diag(file);std::rewind(file);
        char line[1024];int count=0;
        while(std::fgets(line,sizeof(line),file))if(std::string(line).find("expert pool:")!=std::string::npos)++count;
        std::fclose(file);if(count!=2)throw std::runtime_error("diagnostics omit a live pool");
    }
    first.reduce_routed(std::array<float,1>{2}.data(),std::array<float,1>{3}.data(),std::array<float,1>{0}.data(),1,1,1);
    std::cout<<"layer pools: concurrent reductions and diagnostics passed\n";
} catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
