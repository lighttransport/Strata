#pragma once
namespace strata::kernels::cpu {
struct ExpertObserver {
    virtual ~ExpertObserver()=default;
    virtual void hidden(int layer,int expert,const float* values,int count)=0;
};
}
