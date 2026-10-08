#pragma once
namespace strata::kernels::cpu {
struct ExpertObserver {
    virtual ~ExpertObserver()=default;
    virtual void hidden(int layer,int expert,const float* values,int count)=0;
};
// A diagnostic transform after full gate/up execution, before hidden quantization.
// It cannot establish a speedup: all gate/up rows have already been evaluated.
struct ExpertHiddenTransform {
    virtual ~ExpertHiddenTransform()=default;
    virtual void apply(int layer,int expert,int token,float* values,int count)=0;
};
}
