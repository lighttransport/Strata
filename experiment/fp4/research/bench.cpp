#include "formats.hpp"
#include "input.hpp"
#include <iomanip>
#include "../common/report.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#ifdef RESEARCH_GPU
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer{T* p=nullptr;Buffer(size_t n){ck(cudaMalloc(&p,std::max<size_t>(n,1)*sizeof(T)));}~Buffer(){cudaFree(p);}Buffer(const Buffer&)=delete;};
void research_gemv(const uint8_t*,const float*,const uint8_t*,const float*,const float*,int8_t*,float*,float*,int,int,int,int,bool,bool);
void research_xor_decode_probe(float*);
#endif
int main(int argc,char** argv){try{
    bool smoke=true,use_viterbi=false,curvature=false,check_gpu=false,ldl_feedback=false,tune_ldl=false,xor_scale_search=false,calibrate_scales=false;double hessian_reg=.01;int threads=2,rows=0,cols=0,boundary_trials=1,hessian_block=128;std::string csv_path,only="all",weight_path,act_path,cal_path,report_path,scale_mode="rms",cpu_mode="reference",gpu_mode="reference",heldout_path,validation_path,validation_guard="mean";
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--benchmark")smoke=false;
        else if(a=="--curvature")curvature=true;
        else if(a=="--check-gpu-decoder")check_gpu=true;
        else if(a=="--calibrate-xor-scales"){calibrate_scales=true;use_viterbi=true;}
        else if(a=="--xor-scale-search")xor_scale_search=true;
        else if(a=="--tune-ldl"){tune_ldl=true;ldl_feedback=true;curvature=true;use_viterbi=true;}
        else if(a=="--ldl-feedback"){ldl_feedback=true;curvature=true;use_viterbi=true;}
        else if((a=="--threads"||a=="--csv"||a=="--encoder"||a=="--format"||a=="--rows"||a=="--cols"||a=="--weights-f32"||a=="--activation-f32"||a=="--calibration-f32"||a=="--report-csv"||a=="--scale"||a=="--cpu-kernel"||a=="--gpu-kernel"||a=="--hessian-reg"||a=="--heldout-f32"||a=="--validation-f32"||a=="--validation-guard"||a=="--boundary-trials"||a=="--hessian-block")&&i+1<argc){
            auto value=std::string(argv[++i]);if(a=="--threads")threads=std::stoi(value);else if(a=="--rows")rows=std::stoi(value);else if(a=="--cols")cols=std::stoi(value);
            else if(a=="--weights-f32")weight_path=value;else if(a=="--activation-f32")act_path=value;
            else if(a=="--calibration-f32")cal_path=value;else if(a=="--report-csv")report_path=value;else if(a=="--scale")scale_mode=value;else if(a=="--cpu-kernel")cpu_mode=value;else if(a=="--gpu-kernel")gpu_mode=value;else if(a=="--hessian-reg")hessian_reg=std::stod(value);else if(a=="--heldout-f32")heldout_path=value;else if(a=="--validation-f32")validation_path=value;else if(a=="--validation-guard")validation_guard=value;else if(a=="--boundary-trials")boundary_trials=std::stoi(value);else if(a=="--hessian-block")hessian_block=std::stoi(value);
            else if(a=="--encoder"){if(value!="greedy"&&value!="viterbi")throw std::runtime_error("invalid encoder");use_viterbi=value=="viterbi";}
            else if(a=="--format")only=value;else csv_path=value;}
        else throw std::runtime_error("usage: bench_research [--benchmark] [--threads N] [--csv FILE] [--encoder greedy|viterbi] [--format NAME] [--rows N] [--curvature] [--scale rms|fit|both] [--weights-f32 FILE --cols N --rows N] [--activation-f32 FILE] [--calibration-f32 FILE] [--report-csv FILE] [--cpu-kernel reference|simd|both] [--gpu-kernel reference|packed|both] [--check-gpu-decoder] [--ldl-feedback] [--hessian-reg X] [--heldout-f32 FILE] [--validation-f32 FILE --tune-ldl] [--validation-guard mean|paired] [--boundary-trials 1..8] [--xor-scale-search] [--calibrate-xor-scales] [--hessian-block 128|256|384|512]");}
    if(threads<1||threads>256|| (smoke&&!csv_path.empty()))throw std::runtime_error("invalid threads or smoke CSV");
    if(rows<0||rows%16||rows>4096)throw std::runtime_error("rows must be 0 or a multiple of 16 up to 4096");
    if(cols<0||cols%128||cols>16384)throw std::runtime_error("cols must be a multiple of 128 up to 16384");
    if(!weight_path.empty()&&(!cols||!rows))throw std::runtime_error("weight input requires cols and rows");
    if(cols&&!weight_path.size())throw std::runtime_error("cols requires weight input");
    if(!cal_path.empty()&&!curvature&&!calibrate_scales)throw std::runtime_error("calibration input requires --curvature");
    if(boundary_trials<1||boundary_trials>8||(!use_viterbi&&boundary_trials!=1))throw std::runtime_error("boundary search requires Viterbi and 1..8 trials");
    if(ldl_feedback&&(boundary_trials!=1||xor_scale_search))throw std::runtime_error("boundary/scale search currently requires the independent encoder");
    if(validation_guard!="mean"&&validation_guard!="paired")throw std::runtime_error("invalid validation guard");
    if(validation_guard=="paired"&&!tune_ldl&&!calibrate_scales)throw std::runtime_error("paired guard requires tuning");
    if(tune_ldl&&validation_path.empty())throw std::runtime_error("LDL tuning requires supplied validation vectors");
    if(!validation_path.empty()&&!tune_ldl&&!calibrate_scales)throw std::runtime_error("validation requires --tune-ldl");
    if(calibrate_scales&&(cal_path.empty()||validation_path.empty()||ldl_feedback))throw std::runtime_error("scale calibration requires train/validation and the independent encoder");
    if(ldl_feedback&&cal_path.empty())throw std::runtime_error("LDL feedback requires supplied calibration vectors");
    if(!std::isfinite(hessian_reg)||hessian_reg<=0)throw std::runtime_error("Hessian regularization must be positive");
    if(hessian_block<128||hessian_block>512||hessian_block%128||(!ldl_feedback&&hessian_block!=128))throw std::runtime_error("invalid Hessian block or missing LDL feedback");
    if(ldl_feedback&&!use_viterbi)throw std::runtime_error("LDL feedback requires Viterbi encoder");
    if(scale_mode!="rms"&&scale_mode!="fit"&&scale_mode!="both")throw std::runtime_error("invalid scale mode");
    if(cpu_mode!="reference"&&cpu_mode!="simd"&&cpu_mode!="both")throw std::runtime_error("invalid CPU kernel");
    if(gpu_mode!="reference"&&gpu_mode!="packed"&&gpu_mode!="both")throw std::runtime_error("invalid GPU kernel");
    if(check_gpu){
#ifdef RESEARCH_GPU
        Buffer<float> out(65536);research_xor_decode_probe(out.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<float> decoded(65536);ck(cudaMemcpy(decoded.data(),out.p,decoded.size()*4,cudaMemcpyDeviceToHost));
        for(unsigned s=0;s<65536;++s)if(decoded[s]!=research::xor_value(uint16_t(s)))throw std::runtime_error("CUDA XOR byte decoder mismatch");
        std::cout<<"CUDA XOR byte decoder exhaustive parity PASS (65536 states)\n";return 0;
#else
        throw std::runtime_error("GPU decoder check requires the CUDA binary");
#endif
    }
    std::ofstream report;if(!report_path.empty()){report.open(report_path);if(!report)throw std::runtime_error("cannot write diagnostic CSV");
        report<<"scope,device,kernel,in,out,total_bytes,effective_bpw,weight_nmse,own_weight_rel_l2,source_projection_rel_l2,source,encoder,curvature,scale_fit,calibration_projection_mse,ldl_baseline_mse,ldl_candidate_mse,ldl_selected,heldout_tokens,heldout_projection_mse,heldout_rel_l2,heldout_worst_rel_l2,ldl_baseline_heldout_rel_l2,validation_baseline_mse,validation_selected_mse,selected_hessian_reg,selected_feedback,tuning_trials,validation_guard,validation_gain,validation_gain_stderr,guard_rejections,boundary_trials,xor_scale_search,scale_cal_train_baseline,scale_cal_train_candidate,scale_cal_validation_baseline,scale_cal_validation_candidate,scale_cal_gain,scale_cal_stderr,scale_cal_changed,scale_cal_selected,scale_cal_baseline_heldout_rel_l2\n";report<<std::setprecision(9);}
    bool known=only=="all";for(auto k:{research::MCG,research::MUL1,research::XOR,research::HIGGS4,research::QIFT2})known|=only==research::name(k);
    if(!known)throw std::runtime_error("unknown format");
    omp_set_num_threads(threads);fp4::Csv csv(csv_path);int sets=smoke?1:64,repeats=smoke?1:20;
    std::cout<<(smoke?"SMOKE ONLY: timings unqualified\n":"BENCHMARK: verify idle machine\n")<<"encoder="<<(use_viterbi?"viterbi_fixed_greedy_boundary":"greedy_3_sweep")<<" curvature="<<(ldl_feedback?("block"+std::to_string(hessian_block)+"_ldl_feedback"):curvature?"calibration_diagonal_proxy":"none")<<" source="<<(weight_path.empty()?"seeded_gaussian_outliers":"provided_f32")<<" sets=replicated_"<<sets<<'\n';
    for(bool down:{false,true}){int in=smoke?128:(down?2048:4096),out=smoke?128:(down?4096:2048);
        if(down&&!weight_path.empty())break;if(rows)out=rows;if(cols)in=cols;
        auto source=weight_path.empty()?research::source(in,out,down?43:42):research::read_f32(weight_path,size_t(in)*out);
        std::vector<float>x(in),y(out);for(int i=0;i<in;++i)x[i]=std::sin(i*.017f);if(!act_path.empty())x=research::read_f32(act_path,in);
        std::vector<float> calibration_values;if(!cal_path.empty()){calibration_values=research::read_f32(cal_path);if(calibration_values.size()%in)throw std::runtime_error("calibration input must contain whole vectors");}
        std::vector<float> heldout_values;if(!heldout_path.empty()){heldout_values=research::read_f32(heldout_path);research::require_disjoint(calibration_values,heldout_values,in);}
        std::vector<float> validation_values;if(!validation_path.empty()){validation_values=research::read_f32(validation_path);research::require_disjoint(calibration_values,validation_values,in);if(!heldout_values.empty())research::require_disjoint(validation_values,heldout_values,in);}
        std::vector<double> original(out);for(int r=0;r<out;++r)for(int c=0;c<in;++c)original[r]+=double(source[size_t(r)*in+c])*x[c];
        for(auto kind:{research::MCG,research::MUL1,research::XOR,research::HIGGS4,research::QIFT2})for(bool fitted:{false,true}){
            if(kind<research::HIGGS4&&fitted)continue;
            if(kind>=research::HIGGS4&&((scale_mode=="rms"&&fitted)||(scale_mode=="fit"&&!fitted)))continue;
            if(only!="all"&&only!=research::name(kind))continue;
            std::vector<float> diagonal;if(curvature&&kind==research::XOR){diagonal.assign(in,1e-4f);std::mt19937 calibration(991);std::normal_distribution<float> normal(0,1);
                if(!cal_path.empty()){const auto& supplied=calibration_values;size_t tokens=supplied.size()/in;for(size_t t=0;t<tokens;++t)for(int c=0;c<in;++c)diagonal[c]+=supplied[t*in+c]*supplied[t*in+c]/tokens;}
                else for(int t=0;t<8;++t)for(int c=0;c<in;++c){float v=normal(calibration)*(c%97==0?8.f:1.f);diagonal[c]+=v*v/8;}}
            research::FeedbackMetrics feedback;research::TuningMetrics tuning;research::Matrix ldl_baseline{};auto m=tune_ldl&&kind==research::XOR?research::tune_ldl_feedback(source,in,out,calibration_values,validation_values,&tuning,&ldl_baseline,validation_guard=="paired",hessian_block):ldl_feedback&&kind==research::XOR?research::encode_ldl_feedback(source,in,out,calibration_values,hessian_reg,&feedback,heldout_values.empty()?nullptr:&ldl_baseline,hessian_block):research::encode(kind,source,in,out,use_viterbi,diagonal,fitted,boundary_trials,xor_scale_search);auto h=x;research::transform(h.data(),in,kind);
            research::ScaleCalibrationMetrics scale_cal;research::Matrix scale_baseline{};if(calibrate_scales&&kind==research::XOR){scale_baseline=m;m=research::calibrate_xor_scales(m,source,calibration_values,validation_values,&scale_cal);}
            research::ProjectionMetrics heldout,baseline_heldout,scale_baseline_heldout;if(!heldout_values.empty()){heldout=research::evaluate_projection(m,source,heldout_values);if(ldl_feedback&&kind==research::XOR)baseline_heldout=research::evaluate_projection(ldl_baseline,source,heldout_values);if(calibrate_scales&&kind==research::XOR)scale_baseline_heldout=research::evaluate_projection(scale_baseline,source,heldout_values);}
            double calibration_error=calibration_values.empty()||kind!=research::XOR?NAN:research::calibration_loss(m,source,calibration_values);
            double we=0,wn=0;for(int r=0;r<out;++r){std::vector<float>w(source.begin()+size_t(r)*in,source.begin()+size_t(r+1)*in);research::transform(w.data(),in,kind);
                for(int c=0;c<in;++c){double v=m.value(r,c)-w[c];we+=v*v;wn+=double(w[c])*w[c];}}
            std::vector<double>ref(out);for(int r=0;r<out;++r)for(int c=0;c<in;++c)ref[r]+=double(m.value(r,c))*h[c];
            // Distinct physical copies exceed cache, with identical quantized source.
            std::vector<uint8_t> corpus(m.packed.size()*sets);for(int s=0;s<sets;++s)std::copy(m.packed.begin(),m.packed.end(),corpus.begin()+s*m.packed.size());
            const char* device="cpu";
#ifdef RESEARCH_GPU
            device="gpu";cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));std::cout<<"GPU "<<prop.name<<" sm_"<<prop.major<<prop.minor<<'\n';
            Buffer<uint8_t> dp(corpus.size()),ds8(m.scale8.size());Buffer<float> ds(m.scales.size()),dg(m.levels.size()),dx(in),dh(in),dy(out),as(1);Buffer<int8_t> aq(in);Buffer<uint16_t>unit(in);
            ck(cudaMemcpy(dp.p,corpus.data(),corpus.size(),cudaMemcpyHostToDevice));
            if(!m.scale8.empty())ck(cudaMemcpy(ds8.p,m.scale8.data(),m.scale8.size(),cudaMemcpyHostToDevice));
            if(!m.scales.empty())ck(cudaMemcpy(ds.p,m.scales.data(),m.scales.size()*4,cudaMemcpyHostToDevice));
            if(!m.levels.empty())ck(cudaMemcpy(dg.p,m.levels.data(),m.levels.size()*4,cudaMemcpyHostToDevice));
            std::vector<uint16_t>units(in,0x3c00);if(kind==research::HIGGS4)for(int i=0;i<in;++i)if(research::sign(i)<0)units[i]=0xbc00;ck(cudaMemcpy(unit.p,units.data(),in*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dx.p,x.data(),in*4,cudaMemcpyHostToDevice));
#endif
            for(bool q8:{false,true})for(bool simd:{false,true}){if(q8&&kind!=research::MUL1)continue;
#ifdef RESEARCH_GPU
                if((q8||(kind!=research::XOR&&kind!=research::QIFT2))&&simd)continue;
                if(!q8&&(kind==research::XOR||kind==research::QIFT2)&&((gpu_mode=="reference"&&simd)||(gpu_mode=="packed"&&!simd)))continue;
#else
                if((q8||kind>=research::HIGGS4)&&simd)continue;
                if(!q8&&kind<=research::XOR&&((cpu_mode=="reference"&&simd)||(cpu_mode=="simd"&&!simd)))continue;
#endif
                auto run=[&](int set){
#ifdef RESEARCH_GPU
                    const float* input=dx.p;if(kind!=research::XOR){strata::kernels::exl3_hadamard(dx.p,dh.p,unit.p,in,1,true,nullptr);input=dh.p;}
                    research_gemv(dp.p+set*m.packed.size(),ds.p,ds8.p,dg.p,input,aq.p,as.p,dy.p,in,out,m.group,int(kind),q8,simd);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
#else
                    // CPU uses the same physically rotated corpus, without encoding in the timed region.
                    research::cpu_apply(m,x,y,q8,corpus.data()+set*m.packed.size(),simd);
#endif
                };
                run(0);std::vector<double>time;for(int r=0;r<repeats;++r){double start=fp4::now_s();for(int s=0;s<sets;++s)run(s);time.push_back((fp4::now_s()-start)*1e6/sets);}
#ifdef RESEARCH_GPU
                ck(cudaMemcpy(y.data(),dy.p,out*4,cudaMemcpyDeviceToHost));
#endif
                double error=0,energy=0,checksum=0,maxerr=0,source_error=0,source_energy=0;for(int r=0;r<out;++r){if(!std::isfinite(y[r]))throw std::runtime_error("nonfinite output");double d=y[r]-ref[r];error+=d*d;energy+=ref[r]*ref[r];maxerr=std::max(maxerr,std::abs(d));checksum+=y[r];double sd=y[r]-original[r];source_error+=sd*sd;source_energy+=original[r]*original[r];}
                auto st=fp4::stats_of(time);std::string label=std::string(research::name(kind))+(q8?"_q8_affine_probe":"_fp32")+(kind>=research::HIGGS4?(fitted?"_fit":"_rms"):"")+(simd?(std::string(device)=="gpu"?"_packed":"_avx2"):"");
                std::cout<<device<<" kernel="<<label<<" in="<<in<<" out="<<out<<" total_bytes="<<m.bytes()<<" effective_bpw="<<8.0*m.bytes()/(in*out)<<" weight_nmse="<<we/std::max(wn,1e-30)<<" validation_baseline_mse="<<tuning.baseline_validation_loss<<" validation_selected_mse="<<tuning.selected_validation_loss<<" selected_hessian_reg="<<tuning.selected_reg<<" selected_feedback="<<tuning.feedback<<" validation_guard="<<validation_guard<<" validation_gain="<<tuning.validation_gain<<" validation_gain_stderr="<<tuning.validation_gain_stderr<<" guard_rejections="<<tuning.guard_rejections<<" tuning_trials="<<tuning.trials<<" boundary_trials="<<boundary_trials<<" xor_scale_search="<<xor_scale_search<<" scale_cal_selected="<<scale_cal.selected<<" scale_cal_gain="<<scale_cal.gain<<" scale_cal_stderr="<<scale_cal.stderr<<" heldout_tokens="<<heldout.tokens<<" heldout_rel_l2="<<heldout.rel_l2<<" heldout_worst_rel_l2="<<heldout.worst_rel_l2<<" ldl_baseline_heldout_rel_l2="<<baseline_heldout.rel_l2<<" ldl_baseline_mse="<<feedback.baseline_loss<<" ldl_candidate_mse="<<feedback.candidate_loss<<" ldl_selected="<<feedback.selected<<" calibration_projection_mse="<<calibration_error<<" source_projection_rel_l2="<<std::sqrt(source_error/std::max(source_energy,1e-30))<<" own_weight_rel_l2="<<std::sqrt(error/std::max(energy,1e-30))<<" checksum="<<checksum<<" us_med="<<st.med<<'\n';
                if(report)report<<(smoke?"smoke_diagnostic":"benchmark_diagnostic")<<','<<device<<','<<label<<','<<in<<','<<out<<','<<m.bytes()<<','<<8.0*m.bytes()/(in*out)<<','<<we/std::max(wn,1e-30)<<','<<std::sqrt(error/std::max(energy,1e-30))<<','<<std::sqrt(source_error/std::max(source_energy,1e-30))<<','<<(weight_path.empty()?"synthetic":"provided_f32")<<','<<(use_viterbi?"fixed_viterbi":"greedy")<<','<<(kind==research::XOR?(ldl_feedback?("block"+std::to_string(hessian_block)+"_ldl_feedback"):curvature?"diagonal_proxy":"none"):"none")<<','<<(fitted?"fit":"rms")<<','<<calibration_error<<','<<feedback.baseline_loss<<','<<feedback.candidate_loss<<','<<feedback.selected<<','<<heldout.tokens<<','<<heldout.mse<<','<<heldout.rel_l2<<','<<heldout.worst_rel_l2<<','<<baseline_heldout.rel_l2<<','<<tuning.baseline_validation_loss<<','<<tuning.selected_validation_loss<<','<<tuning.selected_reg<<','<<tuning.feedback<<','<<tuning.trials<<','<<validation_guard<<','<<tuning.validation_gain<<','<<tuning.validation_gain_stderr<<','<<tuning.guard_rejections<<','<<boundary_trials<<','<<xor_scale_search<<','<<scale_cal.train_baseline<<','<<scale_cal.train_candidate<<','<<scale_cal.validation_baseline<<','<<scale_cal.validation_candidate<<','<<scale_cal.gain<<','<<scale_cal.stderr<<','<<scale_cal.changed<<','<<scale_cal.selected<<','<<scale_baseline_heldout.rel_l2<<'\n';
                if(!smoke)csv.row(device,label,down?"down":"gate_or_up",weight_path.empty()?(use_viterbi?"synthetic_fixed_viterbi":"synthetic_greedy"):"provided_f32",1,"t"+std::to_string(threads),st,m.bytes(),0,2.0*in*out,std::sqrt(error/std::max(energy,1e-30)),maxerr,0);
            }
        }
    }return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
