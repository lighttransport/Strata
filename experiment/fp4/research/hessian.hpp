#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
namespace research {
struct HessianBlock {
    int first=0,width=0;
    std::vector<double> h,l,d;
};
inline std::vector<HessianBlock> factor_hessian(const std::vector<float>& calibration,int in,int width=128,double reg=.01) {
    if(in<=0||width<=0||in%width||calibration.empty()||calibration.size()%in||!std::isfinite(reg)||reg<=0)
        throw std::runtime_error("invalid Hessian calibration extent or regularization");
    for(float v:calibration)if(!std::isfinite(v))throw std::runtime_error("nonfinite calibration");
    size_t tokens=calibration.size()/in;std::vector<HessianBlock> blocks;
    for(int first=0;first<in;first+=width){HessianBlock b{first,width,std::vector<double>(size_t(width)*width),std::vector<double>(size_t(width)*width),std::vector<double>(width)};
        for(int i=0;i<width;++i)for(int j=0;j<=i;++j){double value=0;for(size_t t=0;t<tokens;++t)value+=double(calibration[t*in+first+i])*calibration[t*in+first+j]/tokens;
            b.h[size_t(i)*width+j]=b.h[size_t(j)*width+i]=value;}
        double mean=0;for(int i=0;i<width;++i)mean+=b.h[size_t(i)*width+i]/width;
        double damp=reg*std::max(mean,1e-12);for(int i=0;i<width;++i)b.h[size_t(i)*width+i]+=damp;
        for(int j=0;j<width;++j){double pivot=b.h[size_t(j)*width+j];for(int k=0;k<j;++k)pivot-=b.l[size_t(j)*width+k]*b.l[size_t(j)*width+k]*b.d[k];
            if(!std::isfinite(pivot)||pivot<=0)throw std::runtime_error("nonpositive Hessian pivot");
            b.d[j]=pivot;b.l[size_t(j)*width+j]=1;
            for(int i=j+1;i<width;++i){double value=b.h[size_t(i)*width+j];for(int k=0;k<j;++k)value-=b.l[size_t(i)*width+k]*b.l[size_t(j)*width+k]*b.d[k];
                b.l[size_t(i)*width+j]=value/pivot;}
        }blocks.push_back(std::move(b));
    }return blocks;
}
} // namespace research
