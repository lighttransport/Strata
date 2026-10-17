#pragma once
// Experimental encoders/decoders, not model converters. See research/README.md.
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/kernels/cpu/exl3/mul1_probe.hpp"
#include <algorithm>
#include <random>
#include <string>
#include <limits>
#include "trellis.hpp"
#include "cpu_decode.hpp"
#include "hessian.hpp"
#include <unordered_map>
namespace research {
enum Kind { MCG, MUL1, XOR, HIGGS4, QIFT2 };
inline const char* name(Kind k) {
    return k == MCG ? "exl3_mcg" : k == MUL1 ? "exl3_mul1" : k == XOR ? "xor_trellis" : k == HIGGS4 ? "higgs_scalar4" : "qift2";
}
inline float sign(unsigned i) {
    uint32_t v=(i+42u)*0x9e3779b9u;v^=v>>16;v*=0x85ebca6bu;v^=v>>13;
    return (v&1)?-1.f:1.f;
}
inline void transform(float* x,int n,Kind kind) {
    if(kind==XOR)return;
    if(kind==HIGGS4)for(int i=0;i<n;++i)x[i]*=sign(i);
    strata::cpu::exl3::hadamard(x,n);
}
inline float xor_value(uint16_t s) {
    constexpr unsigned masks[] = {0x1089,0x2112,0x0a24,0x2441};
    unsigned h=s>>2, p=0,q=0;
    for(unsigned i=0;i<2;++i) { p|=(std::popcount(h&masks[i])&1)<<i; q|=(std::popcount(h&masks[i+2])&1)<<i; }
    constexpr float palette[4][4]={{-6,-1.5,0,2},{-4,-1,.5,3},{-3,-.5,1,4},{-2,0,1.5,6}};
    return palette[p][(s&3)^q];
}
inline float e5m3(unsigned b) { return b?std::ldexp(1.f+float(b&7)/8, int(b>>3)-15):0.f; }
inline uint8_t scale_byte(float v) {
    unsigned best=0;float error=INFINITY;
    for(unsigned b=0;b<256;++b) if(float e=std::abs(e5m3(b)-v);e<error){error=e;best=b;}
    return uint8_t(best);
}
inline std::vector<float> grid(Kind k) {
    if(k==QIFT2)return {-1.5f,-.5f,.5f,1.5f};
    // Lloyd-Max stationary grid for N(0,1); scalar HIGGS-style baseline.
    std::vector<float> c(16);for(int i=0;i<16;++i)c[i]=-3.f+6.f*i/15;
    auto pdf=[](double x){return std::exp(-x*x/2)/std::sqrt(2*3.141592653589793);};
    auto cdf=[](double x){return .5*(1+std::erf(x/std::sqrt(2.)));};
    for(int it=0;it<200;++it){auto old=c;for(int i=0;i<16;++i){double a=i?(old[i-1]+old[i])/2:-12,b=i<15?(old[i]+old[i+1])/2:12;c[i]=float((pdf(a)-pdf(b))/(cdf(b)-cdf(a)));}}
    return c;
}
// Alternating nearest assignment and least-squares scale, retaining the best
// actual reconstruction. Unlike RMS-only scaling, clipping participates in fit.
inline float fit_scale(const float* values,int n,const std::vector<float>& levels,float initial) {
    auto loss=[&](float scale){double error=0;for(int i=0;i<n;++i){double best=INFINITY;
        for(float q:levels){double e=double(values[i])-double(scale)*q;best=std::min(best,e*e);}error+=best;}return error;};
    float best_scale=initial;double best_loss=loss(initial);
    for(float factor:{.5f,.75f,1.f,1.25f,1.5f,2.f}){
        float scale=std::max(1e-8f,initial*factor);
        for(int iter=0;iter<16;++iter){double dot=0,norm=0;
            for(int i=0;i<n;++i){float chosen=levels[0];double best=INFINITY;
                for(float q:levels){double e=double(values[i])-double(scale)*q;if(e*e<best){best=e*e;chosen=q;}}
                dot+=double(values[i])*chosen;norm+=double(chosen)*chosen;}
            scale=float(std::max(1e-8,dot/std::max(norm,1e-30)));
            double error=loss(scale);if(error<best_loss){best_loss=error;best_scale=scale;}
        }
    }return best_scale;
}
struct Matrix {
    Kind kind; int in,out,group;
    std::vector<uint8_t> packed,scale8;
    std::vector<float> scales,levels;
    size_t bytes()const{return packed.size()+scale8.size()+4*(scales.size()+levels.size())+(kind==HIGGS4?4:0);}
    float scale(int row,int col)const {size_t n=size_t(row)*(in/group)+col/group;return kind==XOR?e5m3(scale8[n]):scales[n];}
    float value(int row,int col,const uint8_t* data=nullptr)const {
        if(!data)data=packed.data();
        if(kind<=XOR){auto p=data+(size_t(row/16)*(in/16)+col/16)*64;
            uint16_t s=strata::cpu::exl3::state(p,strata::cpu::exl3::exl3_offsets[(col%16)*16+row%16]);
            return scale(row,col)*(kind==XOR?xor_value(s):strata::cpu::exl3::codebook(s,kind==MUL1));}
        size_t i=size_t(row)*in+col;unsigned bits=kind==HIGGS4?4:2;
        return scale(row,col)*levels[(data[i*bits/8]>>(i*bits%8))&((1<<bits)-1)];
    }
};
inline Matrix encode(Kind k,const std::vector<float>& source,int in,int out,bool use_viterbi=false,
                     const std::vector<float>& diagonal={},bool fitted=false,unsigned boundary_trials=1,
                     bool xor_scale_search=false) {
    if(in<=0||out<=0||in%128||out%16||source.size()!=size_t(in)*out ||
       (!diagonal.empty()&&diagonal.size()!=size_t(in)))throw std::runtime_error("invalid matrix extent");
    for(float v:source)if(!std::isfinite(v))throw std::runtime_error("nonfinite source");
    for(float v:diagonal)if(!std::isfinite(v)||v<0)throw std::runtime_error("invalid sensitivity diagonal");
    if(boundary_trials<1||boundary_trials>8||(!use_viterbi&&boundary_trials!=1))throw std::runtime_error("invalid boundary search extent");
    Matrix m{k,in,out,k==XOR?32:k==HIGGS4?128:in,{},{},{},{}};
    m.packed.resize(size_t(in)*out/(k==HIGGS4?2:4));
    m.levels=k>=HIGGS4?grid(k):std::vector<float>{};
    std::vector<float> w=source;
    for(int r=0;r<out;++r)transform(w.data()+size_t(r)*in,in,k);
    for(float v:w)if(!std::isfinite(v))throw std::runtime_error("source transform overflow");
    for(int r=0;r<out;++r)for(int g=0;g<in;g+=m.group){double energy=0;for(int i=0;i<m.group;++i)energy+=double(w[size_t(r)*in+g+i])*w[size_t(r)*in+g+i];
        float s=float(std::sqrt(energy/m.group));if(s>0&&s<1e-8f)s=1e-8f;
        if(!std::isfinite(s))throw std::runtime_error("source scale overflow");
        if(k==XOR)m.scale8.push_back(scale_byte(s/2));else {
            if(fitted&&k>=HIGGS4)s=fit_scale(w.data()+size_t(r)*in+g,m.group,m.levels,s);
            m.scales.push_back(s);
        }
    }
    if(k>=HIGGS4){unsigned bits=k==HIGGS4?4:2;for(int r=0;r<out;++r)for(int c=0;c<in;++c){float scale=m.scale(r,c),v=scale?w[size_t(r)*in+c]/scale:0,err=INFINITY;unsigned best=0;
        for(unsigned j=0;j<m.levels.size();++j)if(float e=std::abs(v-m.levels[j]);e<err){err=e;best=j;}
        size_t i=size_t(r)*in+c;m.packed[i*bits/8]|=uint8_t(best<<(i*bits%8));}return m;}
    // Bounded greedy cyclic search. Each codebook encodes the SAME source independently.
    // Three sweeps; final wrap-around states are decoded from the ring, never fabricated.
    std::vector<float> book(65536);for(unsigned s=0;s<65536;++s)book[s]=k==XOR?xor_value(s):strata::cpu::exl3::codebook(s,k==MUL1);
    unsigned inv[256];for(unsigned i=0;i<256;++i)inv[strata::cpu::exl3::exl3_offsets[i]]=i;
    for(int ot=0;ot<out/16;++ot)for(int it=0;it<in/16;++it){uint8_t branches[256]={};uint16_t state=0;
        for(int pass=0;pass<3;++pass)for(unsigned t=0;t<256;++t){unsigned r=ot*16+inv[t]%16,c=it*16+inv[t]/16;float scale=m.scale(r,c),target=scale?w[size_t(r)*in+c]/scale:0,err=INFINITY;unsigned best=0;
            for(unsigned b=0;b<4;++b){uint16_t next=uint16_t((state<<2)|b);float e=std::abs(book[next]-target);if(e<err){err=e;best=b;}}
            branches[t]=uint8_t(best);state=uint16_t((state<<2)|best);}
        if(use_viterbi){std::array<float,256> target,importance;
            for(unsigned t=0;t<256;++t){unsigned r=ot*16+inv[t]%16,c=it*16+inv[t]/16;
                float scale=m.scale(r,c);target[t]=scale?w[size_t(r)*in+c]/scale:0;
                importance[t]=scale*scale*(diagonal.empty()?1.f:diagonal[c]);}
            auto path=search_boundaries(target,importance,book,state,boundary_trials);std::copy(path.begin(),path.end(),branches);
        }
        uint8_t* p=m.packed.data()+(size_t(ot)*(in/16)+it)*64;pack_ring(branches,p);
    }
    if(k==XOR&&xor_scale_search){
        for(int r=0;r<out;++r)for(int first=0;first<in;first+=32){float q[32];double dot=0,norm=0;
            for(int c=0;c<32;++c){int col=first+c;auto p=m.packed.data()+(size_t(r/16)*(in/16)+col/16)*64;
                q[c]=xor_value(strata::cpu::exl3::state(p,strata::cpu::exl3::exl3_offsets[(col%16)*16+r%16]));
                double d=diagonal.empty()?1:diagonal[col];dot+=d*q[c]*source[size_t(r)*in+col];norm+=d*q[c]*q[c];}
            size_t index=size_t(r)*(in/32)+first/32;unsigned old=m.scale8[index],center=scale_byte(float(std::max(0.,dot/std::max(norm,1e-30))));
            auto loss=[&](unsigned byte){double error=0;float scale=e5m3(byte);for(int c=0;c<32;++c){double diff=double(q[c]*scale)-source[size_t(r)*in+first+c];error+=(diagonal.empty()?1:diagonal[first+c])*diff*diff;}return error;};
            double best_loss=loss(old);unsigned best=old;
            for(int offset=-2;offset<=2;++offset){int candidate=int(center)+offset;if(candidate<0||candidate>255)continue;double error=loss(candidate);if(error<best_loss){best_loss=error;best=candidate;}}
            m.scale8[index]=uint8_t(best);
        }
    }return m;
}
inline std::vector<double> per_vector_loss(const Matrix& m,const std::vector<float>& source,const std::vector<float>& calibration) {
    if(m.kind!=XOR||source.size()!=size_t(m.in)*m.out||calibration.empty()||calibration.size()%m.in)
        throw std::runtime_error("invalid XOR projection loss extent");
    size_t tokens=calibration.size()/m.in;std::vector<double> losses(tokens,0);
    for(int r=0;r<m.out;++r){std::vector<double> error(m.in);for(int c=0;c<m.in;++c)error[c]=double(source[size_t(r)*m.in+c])-m.value(r,c);
        for(size_t t=0;t<tokens;++t){double projection=0;for(int c=0;c<m.in;++c)projection+=error[c]*calibration[t*m.in+c];losses[t]+=projection*projection;}}
    return losses;
}
inline double calibration_loss(const Matrix& m,const std::vector<float>& source,const std::vector<float>& calibration) {
    auto losses=per_vector_loss(m,source,calibration);double total=0;for(double v:losses)total+=v/losses.size();return total;
}
// Blockwise scalar LDL, with feedback between 16-input trellis tiles. The
// Viterbi objective remains diagonal inside each tile, so this is an LDLQ-style
// approximation, not full within-tile Hessian path optimization.
struct FeedbackMetrics {double baseline_loss=NAN,candidate_loss=NAN;bool selected=false;};
inline Matrix encode_ldl_feedback(const std::vector<float>& source,int in,int out,
                                  const std::vector<float>& calibration,double reg=.01,FeedbackMetrics* metrics=nullptr,
                                  Matrix* baseline_output=nullptr,int hessian_width=128) {
    if(hessian_width<128||hessian_width>512||hessian_width%128)throw std::runtime_error("Hessian width must be 128, 256, 384 or 512");
    auto blocks=factor_hessian(calibration,in,hessian_width,reg);
    std::vector<float> diagonal(in);for(const auto& b:blocks)for(int i=0;i<b.width;++i)diagonal[b.first+i]=float(b.d[i]);
    auto baseline=encode(XOR,source,in,out,true,diagonal);
    auto candidate=baseline;std::vector<float> book(65536);for(unsigned s=0;s<65536;++s)book[s]=xor_value(s);
    unsigned inv[256];for(unsigned i=0;i<256;++i)inv[strata::cpu::exl3::exl3_offsets[i]]=i;
    for(int ot=0;ot<out/16;++ot)for(const auto& b:blocks){
        // Corrections are in source units. Input tiles run from high to low.
        std::vector<double> correction(size_t(16)*b.width,0);
        for(int tile=b.width/16-1;tile>=0;--tile){int first=b.first+tile*16,it=first/16;
            std::array<float,256> targets,importance;
            for(unsigned t=0;t<256;++t){int r=ot*16+inv[t]%16,c=first+inv[t]/16;float scale=candidate.scale(r,c);
                double target=source[size_t(r)*in+c]+correction[(inv[t]%16)*b.width+c-b.first];
                targets[t]=scale?float(target/scale):0;importance[t]=float(scale*scale*b.d[c-b.first]);}
            uint8_t* p=candidate.packed.data()+(size_t(ot)*(in/16)+it)*64;
            auto boundary=strata::cpu::exl3::state(p,255);auto path=viterbi(targets,importance,book,boundary);pack_ring(path.data(),p);
            // Propagate each finalized tile's error to lower input coordinates.
            for(int r=0;r<16;++r)for(int j=tile*16;j<tile*16+16;++j){
                double error=source[size_t(ot*16+r)*in+b.first+j]-candidate.value(ot*16+r,b.first+j);
                for(int i=0;i<tile*16;++i)correction[size_t(r)*b.width+i]+=b.l[size_t(j)*b.width+i]*error;
            }
        }
    }
    // This guard uses calibration data only. It promises no held-out improvement.
    double old_loss=calibration_loss(baseline,source,calibration),new_loss=calibration_loss(candidate,source,calibration);
    if(!std::isfinite(old_loss)||!std::isfinite(new_loss))throw std::runtime_error("nonfinite calibration objective");
    if(metrics)*metrics={old_loss,new_loss,new_loss<old_loss};
    if(baseline_output)*baseline_output=baseline;
    return new_loss<old_loss?candidate:baseline;
}
struct ProjectionMetrics {double mse=NAN,rel_l2=NAN,worst_rel_l2=NAN;size_t tokens=0;};
inline ProjectionMetrics evaluate_projection(const Matrix& m,const std::vector<float>& source,
                                             const std::vector<float>& activations) {
    if(source.size()!=size_t(m.in)*m.out||activations.empty()||activations.size()%m.in)
        throw std::runtime_error("invalid projection evaluation extent");
    ProjectionMetrics metrics;metrics.tokens=activations.size()/m.in;double error=0,energy=0,worst=0;
    // Expanded weights are used ONLY by this untimed diagnostic evaluator.
    std::vector<float> decoded(size_t(m.in)*m.out);for(int r=0;r<m.out;++r)for(int c=0;c<m.in;++c)decoded[size_t(r)*m.in+c]=m.value(r,c);
    for(size_t t=0;t<metrics.tokens;++t){std::vector<float> x(activations.begin()+t*m.in,activations.begin()+(t+1)*m.in),h=x;transform(h.data(),m.in,m.kind);
        double te=0,tn=0;for(int r=0;r<m.out;++r){double original=0,actual=0;for(int c=0;c<m.in;++c){original+=double(source[size_t(r)*m.in+c])*x[c];actual+=double(decoded[size_t(r)*m.in+c])*h[c];}
            double diff=actual-original;te+=diff*diff;tn+=original*original;}
        error+=te;energy+=tn;worst=std::max(worst,std::sqrt(te/std::max(tn,1e-30)));
    }
    metrics.mse=error/metrics.tokens;metrics.rel_l2=std::sqrt(error/std::max(energy,1e-30));metrics.worst_rel_l2=worst;
    if(!std::isfinite(metrics.mse)||!std::isfinite(metrics.rel_l2)||!std::isfinite(worst))throw std::runtime_error("nonfinite held-out metric");
    return metrics;
}
inline void require_disjoint(const std::vector<float>& train,const std::vector<float>& heldout,int in) {
    if(in<=0||train.size()%in||heldout.empty()||heldout.size()%in)throw std::runtime_error("invalid held-out extent");
    auto hash=[&](const float* p){uint64_t value=14695981039346656037ull;for(int i=0;i<in;++i){uint32_t bits=p[i]==0?0:std::bit_cast<uint32_t>(p[i]);value^=bits;value*=1099511628211ull;}return value;};
    std::unordered_multimap<uint64_t,size_t> index;for(size_t t=0;t<train.size()/in;++t)index.emplace(hash(train.data()+t*in),t);
    for(size_t t=0;t<heldout.size()/in;++t){auto range=index.equal_range(hash(heldout.data()+t*in));for(auto it=range.first;it!=range.second;++it)
        if(std::equal(heldout.begin()+t*in,heldout.begin()+(t+1)*in,train.begin()+it->second*in))throw std::runtime_error("held-out vector also appears in calibration");}
}
struct TuningMetrics {
    double baseline_validation_loss=NAN,selected_validation_loss=NAN,selected_reg=NAN;
    double validation_gain=0,validation_gain_stderr=0;unsigned guard_rejections=0;
    bool feedback=false;unsigned trials=0;
};
struct PairedGain {double mean=0,stderr=0;};
inline PairedGain paired_gain(const std::vector<double>& baseline,const std::vector<double>& candidate) {
    if(baseline.size()<2||baseline.size()!=candidate.size())throw std::runtime_error("paired guard requires at least two validation vectors");
    double mean=0,m2=0;size_t count=0;
    for(size_t i=0;i<baseline.size();++i){if(!std::isfinite(baseline[i])||!std::isfinite(candidate[i]))throw std::runtime_error("nonfinite validation losses");
        double gain=baseline[i]-candidate[i],delta=gain-mean;mean+=delta/++count;m2+=delta*(gain-mean);}
    return {mean,std::sqrt(std::max(m2,0.)/(count-1)/count)};
}
struct ScaleCalibrationMetrics {
    double train_baseline=NAN,train_candidate=NAN,validation_baseline=NAN,validation_candidate=NAN;
    double gain=NAN,stderr=NAN;unsigned changed=0;bool selected=false;
};
inline Matrix calibrate_xor_scales(const Matrix& baseline,const std::vector<float>& source,
                                   const std::vector<float>& train,const std::vector<float>& validation,
                                   ScaleCalibrationMetrics* metrics=nullptr) {
    if(baseline.kind!=XOR||source.size()!=size_t(baseline.in)*baseline.out||train.empty()||train.size()%baseline.in)
        throw std::runtime_error("invalid scale calibration extent");
    require_disjoint(train,validation,baseline.in);
    if(validation.size()/baseline.in<2)throw std::runtime_error("scale validation requires at least two vectors");
    Matrix candidate=baseline;int groups=baseline.in/32;size_t tokens=train.size()/baseline.in;
    for(int r=0;r<baseline.out;++r){
        std::vector<double> features(size_t(groups)*tokens),target(tokens,0),prediction(tokens,0);
        for(int c=0;c<baseline.in;++c){auto p=baseline.packed.data()+(size_t(r/16)*(baseline.in/16)+c/16)*64;
            float q=xor_value(strata::cpu::exl3::state(p,strata::cpu::exl3::exl3_offsets[(c%16)*16+r%16]));
            for(size_t t=0;t<tokens;++t){double x=train[t*baseline.in+c];features[size_t(c/32)*tokens+t]+=q*x;target[t]+=double(source[size_t(r)*baseline.in+c])*x;}}
        for(int g=0;g<groups;++g)for(size_t t=0;t<tokens;++t)prediction[t]+=e5m3(candidate.scale8[size_t(r)*groups+g])*features[size_t(g)*tokens+t];
        for(int sweep=0;sweep<3;++sweep)for(int g=0;g<groups;++g){size_t index=size_t(r)*groups+g;unsigned old=candidate.scale8[index];double old_scale=e5m3(old),dot=0,norm=0;
            for(size_t t=0;t<tokens;++t){double z=features[size_t(g)*tokens+t];dot+=z*(target[t]-prediction[t]+old_scale*z);norm+=z*z;}
            if(norm==0)continue;unsigned center=scale_byte(float(std::max(0.,dot/norm))),best=old;double best_loss=0;
            for(size_t t=0;t<tokens;++t){double error=target[t]-prediction[t];best_loss+=error*error;}
            for(int offset=-2;offset<=2;++offset){int byte=int(center)+offset;if(byte<0||byte>255)continue;double delta=e5m3(byte)-old_scale,loss=0;
                for(size_t t=0;t<tokens;++t){double error=target[t]-prediction[t]-delta*features[size_t(g)*tokens+t];loss+=error*error;}
                if(loss<best_loss){best_loss=loss;best=byte;}}
            if(best!=old){double delta=e5m3(best)-old_scale;candidate.scale8[index]=uint8_t(best);
                for(size_t t=0;t<tokens;++t)prediction[t]+=delta*features[size_t(g)*tokens+t];}
        }
    }
    ScaleCalibrationMetrics result;for(size_t i=0;i<baseline.scale8.size();++i)result.changed+=candidate.scale8[i]!=baseline.scale8[i];
    result.train_baseline=calibration_loss(baseline,source,train);result.train_candidate=calibration_loss(candidate,source,train);
    auto before=per_vector_loss(baseline,source,validation),after=per_vector_loss(candidate,source,validation);
    auto gain=paired_gain(before,after);result.gain=gain.mean;result.stderr=gain.stderr;
    result.validation_baseline=calibration_loss(baseline,source,validation);result.validation_candidate=calibration_loss(candidate,source,validation);
    result.selected=result.train_candidate<=result.train_baseline && gain.mean>2*gain.stderr;
    if(metrics)*metrics=result;return result.selected?candidate:baseline;
}
inline Matrix tune_ldl_feedback(const std::vector<float>& source,int in,int out,
                               const std::vector<float>& calibration,const std::vector<float>& validation,
                               TuningMetrics* metrics=nullptr,Matrix* baseline_output=nullptr,bool paired_guard=false,int hessian_width=128) {
    require_disjoint(calibration,validation,in);
    Matrix selected{};TuningMetrics result;std::vector<double> reference_losses;
    if(paired_guard&&validation.size()/in<2)throw std::runtime_error("paired guard requires at least two validation vectors");
    // A fixed, declared sweep. The final held-out set is not an argument.
    for(double reg:{.01,.1,1.}){
        FeedbackMetrics feedback;Matrix baseline;
        auto candidate=encode_ldl_feedback(source,in,out,calibration,reg,&feedback,&baseline,hessian_width);
        double base_loss=calibration_loss(baseline,source,validation);
        double candidate_loss=calibration_loss(candidate,source,validation);
        if(!std::isfinite(base_loss)||!std::isfinite(candidate_loss))throw std::runtime_error("nonfinite validation objective");
        if(!result.trials){selected=baseline;result.baseline_validation_loss=base_loss;result.selected_validation_loss=base_loss;result.selected_reg=reg;
            reference_losses=per_vector_loss(baseline,source,validation);
            if(baseline_output)*baseline_output=baseline;}
        result.trials+=2;
        auto consider=[&](const Matrix& choice,double loss,bool feedback_choice){
            if(loss>=result.selected_validation_loss)return;
            PairedGain gain{result.baseline_validation_loss-loss,NAN};
            if(reference_losses.size()>=2)gain=paired_gain(reference_losses,per_vector_loss(choice,source,validation));
            if(paired_guard){
                if(gain.mean<=2*gain.stderr){++result.guard_rejections;return;}}
            selected=choice;result.selected_validation_loss=loss;result.selected_reg=reg;result.feedback=feedback_choice;
            result.validation_gain=gain.mean;result.validation_gain_stderr=gain.stderr;
        };
        consider(baseline,base_loss,false);consider(candidate,candidate_loss,feedback.selected);
    }
    if(metrics)*metrics=result;return selected;
}
inline std::vector<float> source(int in,int out,unsigned seed) {
    std::mt19937 rng(seed);std::normal_distribution<float> normal(0,1);std::vector<float>w(size_t(in)*out);
    for(int r=0;r<out;++r)for(int c=0;c<in;++c)w[size_t(r)*in+c]=normal(rng)*(c%97==0?8.f:1.f);
    return w;
}
inline void cpu_apply(const Matrix& m,const std::vector<float>& x,std::vector<float>& y,bool q8,const uint8_t* data,bool simd=false) {
    if(x.size()!=size_t(m.in)||y.size()!=size_t(m.out)||!data||(q8&&m.kind!=MUL1))throw std::runtime_error("invalid CPU apply arguments");
    auto h=x;transform(h.data(),m.in,m.kind);
    float a_scale=1;std::vector<int8_t> aq(m.in);
    if(q8){float peak=0;for(float v:h)peak=std::max(peak,std::abs(v));a_scale=peak?peak/127:1;for(int i=0;i<m.in;++i)aq[i]=int8_t(std::round(h[i]/a_scale));}
    if(q8){int32_t xs=0;for(auto v:aq)xs+=v;
#pragma omp parallel for schedule(static)
        for(int ot=0;ot<m.out/16;++ot){int32_t values[16]={};
            for(int it=0;it<m.in/16;++it)strata::cpu::exl3::mul1_tile_dot(data+(size_t(ot)*(m.in/16)+it)*64,aq.data()+it*16,values);
            float unit=strata::cpu::exl3::half_value(0x1eee),bias=1024*unit+strata::cpu::exl3::half_value(0xc931);
            for(int c=0;c<16;++c)y[ot*16+c]=(float(values[c])*unit+float(xs)*bias)*a_scale*m.scale(ot*16+c,0);
        }return;
    }
    if(simd&&m.kind<=XOR){
#pragma omp parallel for schedule(static)
        for(int ot=0;ot<m.out/16;++ot){alignas(32)float acc[16]={};
            for(int it=0;it<m.in/16;++it){alignas(32)float tile[16]={};
                tile_dot_simd(data+(size_t(ot)*(m.in/16)+it)*64,h.data()+it*16,tile,int(m.kind));
                for(int c=0;c<16;++c)acc[c]+=tile[c]*m.scale(ot*16+c,it*16);
            }for(int c=0;c<16;++c)y[ot*16+c]=acc[c];
        }return;
    }
#pragma omp parallel for schedule(static)
    for(int r=0;r<m.out;++r){alignas(32) float lanes[8]={};
        for(int c=0;c<m.in;c+=8){alignas(32)float decoded[8];for(int j=0;j<8;++j)decoded[j]=m.value(r,c+j,data);
            auto a=_mm256_load_ps(lanes);a=_mm256_fmadd_ps(_mm256_load_ps(decoded),_mm256_loadu_ps(h.data()+c),a);_mm256_store_ps(lanes,a);}
        float acc=0;for(float v:lanes)acc+=v;y[r]=acc;
    }
}
} // namespace research
