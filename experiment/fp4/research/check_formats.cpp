#include "formats.hpp"
#include "input.hpp"
#include <filesystem>
#include <chrono>
#include <iostream>
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){try{
    std::mt19937 rng(31415);
    // Independent bit-by-bit extraction crosses every word and cyclic boundary.
    for(int trial=0;trial<32;++trial){uint8_t branches[256],packed[64];for(auto& b:branches)b=rng()%4;research::pack_ring(branches,packed);
        for(unsigned t=0;t<256;++t){uint16_t expected=0;for(int i=-7;i<=0;++i)expected=uint16_t((expected<<2)|branches[(int(t)+i+256)%256]);
            require(strata::cpu::exl3::state(packed,t)==expected,"cyclic state extraction");}}
    const float golden[8]={-6,-1.5,0,2,.5,3,-4,-1};for(unsigned s=0;s<8;++s)require(research::xor_value(s)==golden[s],"XOR palette golden");
    // Exhaustive vector parity verifies all history masks and byte sign extension.
    for(unsigned s=0;s<65536;s+=8){alignas(32)uint32_t states[8];alignas(32)float decoded[8];for(unsigned i=0;i<8;++i)states[i]=s+i;
        _mm256_store_ps(decoded,research::xor_decode8(_mm256_load_si256(reinterpret_cast<const __m256i*>(states))));
        for(unsigned i=0;i<8;++i)require(decoded[i]==research::xor_value(uint16_t(s+i)),"XOR AVX2 exhaustive parity");}
    // Closed-path Viterbi must recover an exactly representable known ring.
    std::vector<float> book(65536);for(unsigned s=0;s<65536;++s)book[s]=research::xor_value(s);
    uint8_t branches[256],packed[64];for(auto& b:branches)b=rng()%4;research::pack_ring(branches,packed);
    std::array<float,256> targets,weights;weights.fill(1);for(unsigned t=0;t<256;++t)targets[t]=book[strata::cpu::exl3::state(packed,t)];
    auto path=research::viterbi(targets,weights,book,strata::cpu::exl3::state(packed,255));research::pack_ring(path.data(),packed);
    for(unsigned t=0;t<256;++t)require(book[strata::cpu::exl3::state(packed,t)]==targets[t],"Viterbi exact representable ring");
    auto c=research::grid(research::HIGGS4);for(unsigned i=0;i<c.size();++i){require(std::isfinite(c[i]),"finite Lloyd-Max grid");require(std::abs(c[i]+c[c.size()-1-i])<1e-5,"symmetric Lloyd-Max grid");if(i)require(c[i]>c[i-1],"ordered Lloyd-Max grid");}
    // RHT must preserve the dot product and use the same signs on W and x.
    auto a=research::source(128,16,5);std::vector<float> x(128);for(auto& v:x)v=float(int(rng()%2001)-1000)/1000;
    double before=0,after=0;for(int i=0;i<128;++i)before+=double(a[i])*x[i];research::transform(a.data(),128,research::HIGGS4);research::transform(x.data(),128,research::HIGGS4);
    for(int i=0;i<128;++i)after+=double(a[i])*x[i];require(std::abs(before-after)<1e-4,"RHT projection invariance");
    // Validate the optimized MUL1 integer path independently of floating codebook rounding.
    for(int trial=0;trial<32;++trial){for(auto& b:branches)b=rng()%4;research::pack_ring(branches,packed);int8_t x8[16];int32_t actual[16]={},expected[16]={};for(auto& v:x8)v=int(rng()%255)-127;
        for(int r=0;r<16;++r)for(int col=0;col<16;++col){uint16_t s=strata::cpu::exl3::state(packed,strata::cpu::exl3::exl3_offsets[r*16+col]);uint32_t v=uint32_t(s)*0x83dcd12du;
            expected[col]+=int((v&255)+((v>>8)&255)+((v>>16)&255)+(v>>24))*x8[r];}
        strata::cpu::exl3::mul1_tile_dot(packed,x8,actual);for(int c=0;c<16;++c)require(actual[c]==expected[c],"MUL1 AVX2 integer parity");}
    bool rejected=false;try{research::encode(research::XOR,{},127,16);}catch(...){rejected=true;}require(rejected,"invalid geometry rejected");
    // Reconstruct a rank-deficient, damped covariance and verify the LDL identity.
    std::vector<float> calibration(3*8);for(int t=0;t<3;++t)for(int i=0;i<8;++i)calibration[t*8+i]=(i+1)*(t+1)*.1f;
    auto factors=research::factor_hessian(calibration,8,8,.01);const auto& factor=factors[0];
    for(int i=0;i<8;++i)for(int j=0;j<8;++j){double value=0;for(int k=0;k<8;++k)value+=factor.l[i*8+k]*factor.d[k]*factor.l[j*8+k];
        require(std::abs(value-factor.h[i*8+j])<1e-12,"LDL covariance reconstruction");}
    std::vector<double> error(8);for(int i=0;i<8;++i)error[i]=std::sin(i);
    double quadratic=0,residual_energy=0;for(int i=0;i<8;++i){double residual=0;for(int j=0;j<8;++j){quadratic+=error[i]*factor.h[i*8+j]*error[j];residual+=factor.l[j*8+i]*error[j];}
        residual_energy+=factor.d[i]*residual*residual;}
    require(std::abs(quadratic-residual_energy)<1e-12,"LDL objective identity");
    rejected=false;try{research::factor_hessian(calibration,8,8,0);}catch(...){rejected=true;}require(rejected,"zero damping rejected");
    std::vector<float> cross_block(4*256,0);
    for(int t=0;t<4;++t)cross_block[t*256]=cross_block[t*256+128]=float(t+1);
    auto full=research::factor_hessian(cross_block,256,256,.01);
    auto split=research::factor_hessian(cross_block,256,128,.01);
    require(full.size()==1&&split.size()==2,"Hessian block geometry");
    require(full[0].h[128]==7.5,"cross-128 covariance retained");
    double reconstructed=full[0].l[128*256]*full[0].d[0];
    require(std::abs(reconstructed-7.5)<1e-12,"cross-128 LDL reconstruction");
    rejected=false;try{research::require_disjoint(calibration,calibration,8);}catch(...){rejected=true;}require(rejected,"held-out leakage rejected");
    auto heldout=calibration;for(auto& v:heldout)v+=.01f;research::require_disjoint(calibration,heldout,8);
    auto strong=research::paired_gain({10,10,10,10},{5,5,5,5});require(strong.mean==5&&strong.stderr==0,"paired constant improvement");
    auto noisy=research::paired_gain({10,10,10,10},{0,20,0,19});require(noisy.mean>0&&noisy.mean<=2*noisy.stderr,"paired noisy gain rejected");
    auto tie=research::paired_gain({1,2},{1,2});require(tie.mean==0&&tie.stderr==0,"paired ties");
    rejected=false;try{research::paired_gain({1},{0});}catch(...){rejected=true;}require(rejected,"paired singleton rejected");
    // Fitting must not worsen the stored reconstruction relative to RMS.
    auto src=research::source(128,16,901);
    // Projection parity exercises varying per-row/per-group scales and packed boundaries.
    std::vector<float> ax(128),ref(16),simd(16);for(int i=0;i<128;++i)ax[i]=std::sin(float(i));
    for(auto kind:{research::MCG,research::MUL1,research::XOR}){auto m=research::encode(kind,src,128,16);
        research::cpu_apply(m,ax,ref,false,m.packed.data());research::cpu_apply(m,ax,simd,false,m.packed.data(),true);
        for(int r=0;r<16;++r)require(std::abs(ref[r]-simd[r])<2e-5f*std::max(1.f,std::abs(ref[r])),"SIMD projection parity");}
    // Zero scales represent exact zero weights, including XOR's reserved scale byte.
    std::vector<float> zeros(128*16,0);
    for(auto kind:{research::MCG,research::MUL1,research::XOR,research::HIGGS4,research::QIFT2}){
        auto m=research::encode(kind,zeros,128,16);for(int r=0;r<16;++r)for(int c=0;c<128;++c)require(m.value(r,c)==0,"zero weight decode");
        research::cpu_apply(m,ax,ref,false,m.packed.data(),true);for(float v:ref)require(v==0,"zero projection");
        auto metrics=research::evaluate_projection(m,zeros,ax);require(metrics.mse==0&&metrics.rel_l2==0,"zero held-out projection");
    }
    auto huge=zeros;for(auto& v:huge)v=std::numeric_limits<float>::max();
    rejected=false;try{research::encode(research::HIGGS4,huge,128,16);}catch(...){rejected=true;}require(rejected,"transform overflow rejected");
    for(auto kind:{research::QIFT2,research::HIGGS4}){
        auto rms=research::encode(kind,src,128,16),fit=research::encode(kind,src,128,16,false,{},true);
        double old_loss=0,new_loss=0;for(int r=0;r<16;++r){std::vector<float>w(src.begin()+r*128,src.begin()+(r+1)*128);research::transform(w.data(),128,kind);
            for(int col=0;col<128;++col){double a=rms.value(r,col)-w[col],b=fit.value(r,col)-w[col];old_loss+=a*a;new_loss+=b*b;}}
        require(new_loss<=old_loss+1e-5,"scale fitting must not worsen RMS error");
    }
    // Boundary search contains the original boundary, and discrete scale fitting
    // contains the original byte. Compare actual decoded source reconstruction.
    auto baseline_ring=research::encode(research::XOR,src,128,16,true);
    auto searched_ring=research::encode(research::XOR,src,128,16,true,{},false,2,true);
    double baseline_error=0,searched_error=0;
    for(int r=0;r<16;++r)for(int c=0;c<128;++c){double a=baseline_ring.value(r,c)-src[r*128+c],b=searched_ring.value(r,c)-src[r*128+c];baseline_error+=a*a;searched_error+=b*b;}
    require(searched_error<=baseline_error+1e-5,"ring and scale search source regression");
    // A zero target cannot be improved; calibrated fitting must retain its exact
    // zero baseline, regardless of nonzero training/validation activations.
    std::vector<float> scale_train(4*128),scale_validation(4*128);for(size_t i=0;i<scale_train.size();++i){scale_train[i]=std::sin(float(i));scale_validation[i]=std::cos(float(i));}
    auto zero_scale_matrix=research::encode(research::XOR,zeros,128,16);
    research::ScaleCalibrationMetrics scale_metrics;
    auto calibrated_zero=research::calibrate_xor_scales(zero_scale_matrix,zeros,scale_train,scale_validation,&scale_metrics);
    require(!scale_metrics.selected && scale_metrics.train_candidate==0,"zero scale-calibration fallback");
    require(calibrated_zero.scale8==zero_scale_matrix.scale8&&calibrated_zero.packed==zero_scale_matrix.packed,"zero scale-calibration unchanged");
    // Fixture belongs to this test; remove it on success or exceptions.
    struct Scratch {std::filesystem::path p;~Scratch(){std::error_code e;std::filesystem::remove(p,e);}} scratch{
        std::filesystem::temp_directory_path()/("strata-research-input-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".f32")};
    {std::ofstream f(scratch.p,std::ios::binary);float input[2]={1,-2};f.write(reinterpret_cast<char*>(input),sizeof(input));}
    auto input=research::read_f32(scratch.p.string(),2);require(input[0]==1&&input[1]==-2,"raw float32 round trip");
    rejected=false;try{research::read_f32(scratch.p.string(),3);}catch(...){rejected=true;}require(rejected,"raw size mismatch rejected");
    {std::ofstream f(scratch.p,std::ios::binary);float value=NAN;f.write(reinterpret_cast<char*>(&value),4);}
    rejected=false;try{research::read_f32(scratch.p.string());}catch(...){rejected=true;}require(rejected,"raw NaN rejected");
    std::cout<<"format checks PASS: ring, XOR golden/AVX2, Viterbi, LDL identities, grid, RHT, MUL1 parity, SIMD projection, zero/overflow, scale fitting, raw reader\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
