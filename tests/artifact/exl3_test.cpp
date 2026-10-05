#include "strata/kernels/cpu/exl3/pool.hpp"
#include "strata/kernels/cpu/exl3/dense_q8.hpp"
#include "strata/core/model.hpp"
#include <iostream>
#include <bit>
#include <random>
#include <filesystem>
using namespace strata::cpu::exl3;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void explicit_had(std::vector<float>& x) {
    auto input=x;for(size_t b=0;b<x.size();b+=128)for(unsigned i=0;i<128;++i){double sum=0;for(unsigned j=0;j<128;++j)sum+=(std::popcount(i&j)&1)?-input[b+j]:input[b+j];x[b+i]=sum*0.088388347648f;}
}
int main(int argc,char** argv) {
 try {
    for(unsigned i=0;i<65536;++i) {
        float f=half_value(i);if(std::isfinite(f))require(half_round(f)==i,"FP16 round trip");
        unsigned r=i%16,c=(i/16)%16;
        unsigned offset=(r&1)+((r&6)<<2)+((r&8)>>2)+((c&7)<<5)+((c&8)>>1);
        require(offset==exl3_offsets[r*16+c],"GPU fragment permutation");
    }
    // Golden hashes from clair 6ff3a17's independent C implementation, K2/MCG.
    auto hash=[](const float* values,size_t n){uint64_t h=14695981039346656037ULL;for(size_t i=0;i<n;++i){uint32_t bits=std::bit_cast<uint32_t>(values[i]);for(unsigned b=0;b<4;++b){h^=(bits>>(8*b))&255;h*=1099511628211ULL;}}return h;};
    require(hash(lookup().data(),65536)==15812111532333246046ULL,"MCG codebook golden");
    uint8_t packed[64];float golden[256];for(unsigned i=0;i<64;++i)packed[i]=i*53+17;tile(packed,golden);
    require(hash(golden,256)==15277630358373401808ULL,"K2 cyclic tile golden");
    // Exercise every cyclic word boundary with independent scalar tile decode.
    std::mt19937 rng(12345);
    for(unsigned trial=0;trial<256;++trial) {
        float x[16],expected[16],actual[16],w[256];
        for(unsigned i=0;i<64;++i)packed[i]=rng();
        for(unsigned i=0;i<16;++i){x[i]=float(int(rng()%2001)-1000)*.001f;actual[i]=expected[i]=float(int(rng()%101)-50);}
        tile(packed,w);
        for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)expected[c]+=x[r]*w[r*16+c];
        tile_dot(packed,x,actual);
        for(unsigned i=0;i<16;++i)require(actual[i]==expected[i],"vector tile accumulation parity");
    }
    for(size_t n:{8u,128u,4096u,12288u}) {
        std::vector<uint16_t>w(n);std::vector<float>x(n);float lanes[8]={};double reference=0,absolute=0;
        for(size_t i=0;i<n;++i){w[i]=uint16_t((rng()&0x807f)|0x3f00);x[i]=float(int(rng()%2001)-1000)*.001f;float product=std::bit_cast<float>(uint32_t(w[i])<<16)*x[i];lanes[i%8]+=product;reference+=double(std::bit_cast<float>(uint32_t(w[i])<<16))*x[i];absolute+=std::abs(double(product));}
        float expected=0;for(float v:lanes)expected+=v;
        float actual=bf16_dot(w.data(),x.data(),n);require(actual==expected,"BF16 vector accumulator parity");
        require(std::abs(actual-reference)<1e-5*absolute+1e-6,"BF16 double reference");
    }
    for(size_t n:{32u,128u,4096u,12288u}) {
        std::vector<uint16_t>source(n);std::vector<DenseQ8Block>packed(n/32);
        for(size_t i=0;i<n;++i)source[i]=std::bit_cast<uint32_t>(float(int(rng()%2001)-1000)*.001f)>>16;
        dense_q8_row(source.data(),packed.data(),n);
        for(size_t i=0;i<n;++i) {
            const auto& b=packed[i/32];float scale=half_value(b.scale),value=std::bit_cast<float>(uint32_t(source[i])<<16);
            require(std::abs(value-scale*b.values[i%32])<=scale*.501f+1e-7f,"dense Q8 quantization bound");
        }
    }
    for(size_t n:{32u,128u,4096u,12288u}) {
        std::vector<uint16_t> source(n); std::vector<DenseQ6Block> packed(n/32);
        for(size_t i=0;i<n;++i)source[i]=std::bit_cast<uint32_t>(float(int(rng()%2001)-1000)*.001f)>>16;
        dense_q6_row(source.data(),packed.data(),n);
        for(size_t i=0;i<n;++i) {
            float scale=half_value(packed[i/32].scale),value=std::bit_cast<float>(uint32_t(source[i])<<16);
            require(std::abs(value-dense_q6_value(packed[i/32],i%32))<=scale*.501f+1e-7f,"dense Q6 quantization bound");
        }
    }
    uint16_t zero_weights[32]={};DenseQ8Block zero_block;dense_q8_row(zero_weights,&zero_block,32);
    require(zero_block.scale==0,"dense Q8 zero scale");for(int8_t v:zero_block.values)require(v==0,"dense Q8 zero values");
    zero_weights[0]=0x7f80;bool finite_rejected=false;
    try{dense_q8_row(zero_weights,&zero_block,32);}catch(...){finite_rejected=true;}
    require(finite_rejected,"dense Q8 nonfinite rejected");
    std::vector<float> impulse(128);impulse[0]=1;hadamard(impulse.data(),impulse.size());
    double had_energy=0;for(float v:impulse){require(std::abs(v-1/std::sqrt(128.f))<1e-7,"upstream normalized Hadamard impulse");had_energy+=double(v)*v;}
    require(std::abs(had_energy-1)<1e-6,"Hadamard must preserve energy");
    std::vector<float> h(256);for(size_t i=0;i<h.size();++i)h[i]=std::sin(float(i));auto reference=h;explicit_had(reference);hadamard(h.data(),h.size());
    for(size_t i=0;i<h.size();++i)require(std::abs(h[i]-reference[i])<1e-5,"Hadamard reference");
    require(strata::artifact::parse_json("9007199254740993").integer()==9007199254740993ULL,"integer precision");
    for(const auto* bad:{"{\"a\":1,\"a\":2}","01","[1,]","\"\\ud800\"","1 true"}) {
        bool rejected=false;try{strata::artifact::parse_json(bad);}catch(...){rejected=true;}require(rejected,"malformed JSON accepted");
    }
    if(argc>=2) {
        if(argc==3&&std::string(argv[2])=="--huge")huge_pages=true;
        strata::core::ModelArtifact artifact(argv[1]);auto& model=artifact.exl3();Pool pool(15,true);
        require(model.token_ids.size()==154880 && model.token_ids[154855] && !model.token_ids[154879],"padded output vocabulary");
        require(artifact.descriptor().layers.size()==45,"native descriptor");
        artifact.shape("blk.0.ssm_conv1d_q.weight",{4,1,8192});artifact.shape("blk.3.attn_k_b.weight",{256,512,64});
        const auto& a=artifact.at("blk.0.ssm_a");const auto& source=model.tensors.at("model.language_model.layers.0.self_attn.A_log");
        float original;model.tensors.read(source,0,&original,4);float transformed;std::memcpy(&transformed,a.data(),4);
        require(transformed==-std::exp(original),"A_log conversion");
        const auto& kv=model.tensors.at("model.language_model.layers.3.self_attn.kv_b_proj.weight");
        const auto* kb=artifact.at("blk.3.attn_k_b.weight").data();const auto* vb=artifact.at("blk.3.attn_v_b.weight").data();
        for(size_t head:{0u,32u,63u})for(size_t row:{0u,255u})for(size_t col:{0u,511u}) {
            uint16_t value;model.tensors.read(kv,((head*512+row)*512+col)*2,&value,2);uint16_t converted;std::memcpy(&converted,kb+(head*256*512+col*256+row)*2,2);require(value==converted,"K transpose");
            model.tensors.read(kv,((head*512+256+row)*512+col)*2,&value,2);std::memcpy(&converted,vb+(head*256*512+row*512+col)*2,2);require(value==converted,"V extraction");
        }
        for(unsigned layer:{3u,7u,44u})for(unsigned expert:{0u,17u,287u})for(const auto* projection:{"gate_proj","up_proj","down_proj"}) {
            auto raw=model.load(layer,expert,projection);auto sharded=pool.load(model,layer,expert,projection);
            std::vector<float>x(raw.in),y(raw.out),z(raw.out);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.01f);
            raw.apply(x.data(),y.data());pool.apply(sharded,x.data(),z.data());
            for(size_t i=0;i<y.size();++i)require(y[i]==z[i],"NUMA sharded projection parity");
        }
    }
    std::cout<<"EXL3 CPU PASS\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
