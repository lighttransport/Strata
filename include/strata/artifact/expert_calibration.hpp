#pragma once
#include "strata/kernels/cpu/expert_observer.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace strata::artifact {
// Per-expert diagonal activation moments. Calibration and trace are independent:
// a trace retains real inputs/routing for the CPU benchmark, moments fit in RAM.
class ExpertCalibration final:public kernels::cpu::ExpertObserver {
    struct Moments {int columns=0;std::vector<double> sum;std::vector<uint64_t> count;};
    int experts_=288;
    std::map<std::pair<int,bool>,Moments> data_;
    std::filesystem::path directory_;
    std::ofstream trace_;
    int trace_positions_=24,first_position_=-1;
    void add(int layer,int expert,const float* x,int n,bool down) {
        if(directory_.empty())return;
        if(layer<3||layer>44||expert<0||expert>=experts_||n!=(down?2048:4096))throw std::runtime_error("calibration geometry mismatch");
        auto& m=data_[{layer,down}];if(m.sum.empty()){
            m.columns=n;m.sum.resize(size_t(n)*experts_);m.count.assign(experts_,0);
#ifdef __linux__
            if(mlock(m.sum.data(),m.sum.size()*sizeof(double)))throw std::runtime_error("cannot lock calibration moments within the process memlock allowance");
#endif
        }
        for(int k=0;k<n;++k){if(!std::isfinite(x[k]))throw std::runtime_error("nonfinite calibration activation");m.sum[size_t(expert)*n+k]+=double(x[k])*x[k];}
        ++m.count[expert];
    }
public:
    bool collects_moments()const{return !directory_.empty();}
    ~ExpertCalibration(){
#ifdef __linux__
        for(auto& [key,m]:data_)if(!m.sum.empty())munlock(m.sum.data(),m.sum.size()*sizeof(double));
#endif
    }
    ExpertCalibration(std::string directory,std::string trace,int experts=288):experts_(experts),directory_(std::move(directory)) {
        if(experts_<1||experts_>512)throw std::invalid_argument("calibration: invalid expert count");
        if(!directory_.empty())std::filesystem::create_directories(directory_);
        if(!trace.empty()){trace_.open(trace,std::ios::binary|std::ios::trunc);if(!trace_)throw std::runtime_error("cannot create expert activation trace");}
    }
    void input(int layer,int position,int nt,int n,int topk,const int* ids,const float* values) {
        if(layer<3||layer>44)return;
        if(first_position_<0)first_position_=position;
        if(trace_.is_open()&&position-first_position_<trace_positions_) {
            const uint32_t header[]={0x31435247,uint32_t(layer),uint32_t(position),uint32_t(nt),uint32_t(n),uint32_t(topk)};
            trace_.write(reinterpret_cast<const char*>(header),sizeof header);
            trace_.write(reinterpret_cast<const char*>(ids),nt*topk*sizeof(int));trace_.write(reinterpret_cast<const char*>(values),nt*n*sizeof(float));
            if(!trace_)throw std::runtime_error("expert trace write failed");
        }
        for(int t=0;t<nt;++t)for(int k=0;k<topk;++k)add(layer,ids[t*topk+k],values+t*n,n,false);
    }
    void hidden(int layer,int expert,const float* values,int count)override {add(layer,expert,values,count,true);}
    void flush() {
        if(trace_.is_open()){trace_.flush();if(!trace_)throw std::runtime_error("expert trace flush failed");}
        if(directory_.empty())return;
        std::ofstream report(directory_/"coverage.json");report<<"{\"projections\":[";bool comma=false;
        for(const auto& [key,m]:data_) {
            const int n=m.columns;std::vector<float> mean(size_t(n)*experts_);std::vector<double> aggregate(n);uint64_t total=0;
            for(int e=0;e<experts_;++e){total+=m.count[e];for(int k=0;k<n;++k)aggregate[k]+=m.sum[size_t(e)*n+k];}
            for(int e=0;e<experts_;++e)for(int k=0;k<n;++k)mean[size_t(e)*n+k]=std::max(1e-12,double(m.count[e]?m.sum[size_t(e)*n+k]/m.count[e]:aggregate[k]/std::max<uint64_t>(1,total)));
            const auto name="blk."+std::to_string(key.first)+(key.second?".down":".gu");const auto path=directory_/(name+".f32"),temp=directory_/(name+".f32.partial");
            std::ofstream output(temp,std::ios::binary|std::ios::trunc);output.write(reinterpret_cast<const char*>(mean.data()),mean.size()*4);output.close();if(!output)throw std::runtime_error("calibration write failed");std::filesystem::rename(temp,path);
            if(comma)report<<',';
            comma=true;report<<"{\"name\":\""<<name<<"\",\"counts\":[";for(int e=0;e<experts_;++e){if(e)report<<',';report<<m.count[e];}report<<"]}";
        }
        report<<"]}\n";report.close();if(!report)throw std::runtime_error("calibration coverage write failed");
    }
};
}
