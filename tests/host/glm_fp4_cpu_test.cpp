#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
namespace {
struct GuardedCodes {
    void* mapping=nullptr;size_t bytes=0;uint8_t* data=nullptr;
    explicit GuardedCodes(size_t count) {
        const long page=sysconf(_SC_PAGESIZE);
        if(page<=0)throw std::runtime_error("cannot read guard page size");
        const size_t payload=(count+size_t(page)-1)/size_t(page)*size_t(page);
        bytes=payload+2*size_t(page);
        mapping=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(mapping==MAP_FAILED){mapping=nullptr;throw std::runtime_error("cannot map CPU scale guards");}
        auto* begin=static_cast<uint8_t*>(mapping)+page;
        if(mprotect(begin,payload,PROT_READ|PROT_WRITE)) {
            munmap(mapping,bytes);mapping=nullptr;throw std::runtime_error("cannot enable CPU scale fixture");
        }
        data=begin+payload-count;
    }
    GuardedCodes(const GuardedCodes&)=delete;
    GuardedCodes& operator=(const GuardedCodes&)=delete;
    ~GuardedCodes(){if(mapping)munmap(mapping,bytes);}
};
}
#endif
int main(int argc,char** argv){try {
    {
        std::vector<float> original(65536+137),rounded;
        for(unsigned i=0;i<65536;++i)original[i]=strata::cpu::exl3::half_value(uint16_t(i));
        for(size_t i=65536;i<original.size();++i)original[i]=std::sin(float(i))*70000.f;
        rounded=original;
        strata::kernels::cpu::glm_fp4_round_f16(rounded.data()+1,rounded.size()-2);
        for(size_t i=0;i<rounded.size();++i) {
            const float expected=(i==0||i+1==rounded.size())?original[i]:
                strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(original[i]));
            if(!(std::isnan(expected)&&std::isnan(rounded[i]))&&std::memcmp(&expected,&rounded[i],sizeof(float)))
                throw std::runtime_error("FP16 conversion differs from scalar codec");
        }
        for(size_t count=0;count<=129;++count) {
            auto values=original;strata::kernels::cpu::glm_fp4_round_f16(values.data()+1,count);
            for(size_t i=0;i<values.size();++i)if(i<1||i>count)
                if(std::memcmp(&values[i],&original[i],sizeof(float)))throw std::runtime_error("FP16 conversion wrote outside extent");
        }
    }
    constexpr int N=32,K=256;
    std::vector<uint8_t> weights(N*K/2),scales(N*K/16);
    for(size_t i=0;i<weights.size();++i)weights[i]=uint8_t(31*i+19);
    for(size_t i=0;i<scales.size();++i)scales[i]=uint8_t(8+i%100);
    std::vector<float> x(K*8),reference(N*8),actual(N*8),one;
    for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.13f)*(i%107==0?8000.f:1.f)));
    constexpr float global=.00061f;
    for(bool qpn:{false,true}) {
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,qpn,table);
        const float g=qpn?strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(global*16384.f)):global;
        for(int t=0;t<8;++t)for(int r=0;r<N;++r) {
            double sum=0;
            for(int j=0;j<K;++j) {
                float scale=strata_fp4_e4m3(scales[r*(K/16)+j/16])*g;
                if(qpn)scale=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(scale))/16384.f;
                const int q=(weights[r*(K/2)+j/2]>>((j&1)*4))&15;
                const float w=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(strata_fp4_e2m1(q)*scale));
                sum+=double(w)*x[t*K+j];
            }
            reference[t*N+r]=float(sum);
        }
        for(int M=1;M<=8;++M) {
            std::fill(actual.begin(),actual.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows(weights.data(),scales.data(),global,x.data(),actual.data(),K,N,M,0,13,qpn);
            strata::kernels::cpu::glm_fp4_rows_table(weights.data(),scales.data(),table,x.data(),actual.data(),K,N,M,13,N);
            double error=0,energy=0;
            for(int i=0;i<M*N;++i){error+=double(actual[i]-reference[i])*(actual[i]-reference[i]);energy+=double(reference[i])*reference[i];}
            const double rel=std::sqrt(error/std::max(energy,1e-30));
            if(!std::isfinite(rel)||rel>1e-5)throw std::runtime_error("CPU W4A16 parity failed");
            if(M==1)one.assign(actual.begin(),actual.begin()+N);
            else if(std::memcmp(one.data(),actual.data(),N*sizeof(float)))throw std::runtime_error("CPU W4A16 first row changes with verify width");
        }
    }
    std::cout<<"FP4 CPU W4A16 PASS (1..8 tokens, disjoint rows, outliers, fixed weight scales)\n";
    for(int groups:{4,16,64,128,256})for(int rows:{128,256}) {
        const int columns=groups*16;
        std::vector<uint8_t> bases(rows),codes(size_t(rows)*groups/2),logical(size_t(rows)*groups),w(size_t(rows)*columns/2);
        for(int i=0;i<rows;++i)bases[i]=30+i%20;
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*53+7);
        const size_t tiles=size_t(rows/128)*(groups/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(rows/32)*(groups/2)+pages);
        std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(rows,groups,bases.data(),bases.size(),codes.data(),codes.size(),
            offsets.data(),offsets.size(),page_bases.data(),pages,nullptr,0,logical.data(),logical.size()))
            throw std::runtime_error("CSF32 test fixture failed host validation");
        const auto clean=logical;
        auto empty_sparse=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        if(!empty_sparse.exceptions.empty())throw std::runtime_error("exception-free sparse fixture has records");
        // Dense exception rows, empty rows and exceptions on mask boundaries.
        for(int row=0;row<rows;++row)for(int group=0;group<groups;++group)
            if(row%3==0||group==0||group==63||group==64||group==127)
                logical[size_t(row)*groups+group]=uint8_t(8+(row+group*7)%110);
        auto compact=strata::kernels::cpu::glm_fp4_csf32_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        auto sparse=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        const auto source_bases=bases,source_codes=codes;
        const auto borrowed=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),source_bases.data(),source_codes.data(),rows,groups,true);
        if(borrowed.bytes()+size_t(rows)*(1+groups/2)!=sparse.bytes())throw std::runtime_error("borrowed CSF32 layout copied checkpoint bytes");
        std::fill(bases.begin(),bases.end(),0);std::fill(codes.begin(),codes.end(),0);
        std::vector<float> input(size_t(columns)*8),expected(size_t(rows)*8),got(expected.size()),first;
        for(size_t i=0;i<input.size();++i)input[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.37f)));
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,true,table);
        for(int tokens=1;tokens<=8;++tokens) {
            std::fill(expected.begin(),expected.end(),-777.f);std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_table(w.data(),logical.data(),table,input.data(),expected.data(),columns,rows,tokens,13,rows-1);
            strata::kernels::cpu::glm_fp4_rows_csf32_table(w.data(),compact,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("CSF32 CPU rows differ from logical scale planes");
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),sparse,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("sparse CSF32 CPU rows differ from logical scale planes");
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),borrowed,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("borrowed CSF32 CPU rows differ from logical scale planes");
            if(tokens==1)first.assign(got.begin(),got.begin()+rows);
            else if(std::memcmp(first.data(),got.data(),rows*4))throw std::runtime_error("CSF32 CPU first row changes with verification width");
        }
        strata::kernels::cpu::glm_fp4_rows_table(w.data(),clean.data(),table,input.data(),expected.data(),columns,rows,4,0,rows);
        strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),empty_sparse,table,input.data(),got.data(),columns,rows,4,0,rows);
        if(std::memcmp(expected.data(),got.data(),size_t(rows)*4*4))throw std::runtime_error("exception-free sparse CPU rows differ");
        auto qinput=input;
        qinput[0]=8192.f;qinput[63]=-4096.f;
        std::fill(qinput.begin()+128,qinput.begin()+192,0.f);
        const auto q8=strata::kernels::cpu::glm_fp4_quantize_q8(qinput.data(),columns,8);
        for(size_t i=0;i<qinput.size();++i) {
            const float scale=q8.scales[i/64];
            float maximum=0;
            for(size_t j=i/64*64;j<i/64*64+64;++j)maximum=std::max(maximum,std::abs(qinput[j]));
            float reference_scale=maximum/127.f;
            if(maximum>0&&reference_scale==0)reference_scale=maximum;
            const int reference=reference_scale==0?0:int(std::clamp(std::nearbyint(qinput[i]/reference_scale),-127.f,127.f));
            if(scale!=reference_scale||int(q8.values[i])!=reference)throw std::runtime_error("Q8 quantizer differs from scalar reference");
            if(std::abs(qinput[i]-float(q8.values[i])*scale)>scale*.5001f+1.e-6f)
                throw std::runtime_error("Q8 activation exceeds half-scale error bound");
        }
        for(int mode:{0,1,2,3}) {
            auto source=q8;
            if(mode&1)strata::kernels::cpu::glm_fp4_prepare_q8(source);
            if(mode&2)strata::kernels::cpu::glm_fp4_prepare_q8_bias(source);
            const int order[]={7,1,7,0};std::vector<float> picked(size_t(columns)*4);
            for(int t=0;t<4;++t)std::copy_n(qinput.data()+size_t(order[t])*columns,columns,picked.data()+size_t(t)*columns);
            auto expected=strata::kernels::cpu::glm_fp4_quantize_q8(picked.data(),columns,4);
            if(mode&1)strata::kernels::cpu::glm_fp4_prepare_q8(expected);
            if(mode&2)strata::kernels::cpu::glm_fp4_prepare_q8_bias(expected);
            const auto actual=strata::kernels::cpu::glm_fp4_gather_q8(source,order,4);
            if(actual.values!=expected.values||actual.scales!=expected.scales||actual.magnitudes!=expected.magnitudes||
               actual.negative!=expected.negative||actual.dot_bias!=expected.dot_bias)
                throw std::runtime_error("Q8 input reuse changes duplicate/reordered rows");
            for(int bad:{-1,8}) {
                bool rejected=false;try{strata::kernels::cpu::glm_fp4_gather_q8(source,&bad,1);}catch(const std::invalid_argument&){rejected=true;}
                if(!rejected)throw std::runtime_error("Q8 gather accepts an invalid row");
            }
        }
        if(q8.scales[2]!=0)throw std::runtime_error("Q8 zero block has nonzero scale");
        std::vector<float> qfirst;
        for(int tokens=1;tokens<=8;++tokens) {
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),sparse,table,q8,got.data(),columns,rows,tokens,13,rows-1);
            auto borrowed_output=got;
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),borrowed,table,q8,borrowed_output.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(got.data(),borrowed_output.data(),got.size()*4))throw std::runtime_error("borrowed Q8 rows differ from owning scales");
            for(bool owned:{false,true})for(bool cached:{false,true})for(int mode:{0,1,2,3})for(int tile:{1,2,4})for(bool local:{false,true}) {
                auto layout=owned?sparse:borrowed;
                if(cached)strata::kernels::cpu::glm_fp4_cache_scale_quartets(layout);
                auto operands=q8;
                if(mode&1)strata::kernels::cpu::glm_fp4_prepare_q8(operands);
                if(mode&2)strata::kernels::cpu::glm_fp4_prepare_q8_bias(operands);
                std::vector<float> tiled(got.size(),-777.f);
                strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),layout,table,operands,tiled.data(),columns,rows,tokens,13,rows-1,tile,local);
                if(std::memcmp(got.data(),tiled.data(),got.size()*4))
                    throw std::runtime_error("tiled/prepared/biased/local Q8 differs from original row kernel");
            }
            double error=0,norm=0;
            for(int t=0;t<tokens;++t)for(int row=13;row<rows-1;++row) {
                double sum=0;
                for(int j=0;j<columns;++j) {
                    const auto packed=w[size_t(row)*(columns/2)+j/2];
                    const auto code=(packed>>((j%2)*4))&15;
                    const auto scale=logical[size_t(row)*groups+j/16];
                    const float factor=table[size_t(scale)*8+1]*q8.scales[size_t(t)*(columns/64)+j/64];
                    sum+=double(strata_fp4_e2m1(code)*2)*q8.values[size_t(t)*columns+j]*factor;
                }
                const double delta=got[size_t(t)*rows+row]-sum;
                error+=delta*delta;norm+=sum*sum;
            }
            if(std::sqrt(error/std::max(norm,1.e-30))>1.e-5)throw std::runtime_error("Q8 dot differs from decoded FP64 reference");
            for(int t=0;t<8;++t)for(int row=0;row<rows;++row)
                if((t>=tokens||row<13||row==rows-1)&&got[size_t(t)*rows+row]!=-777.f)
                    throw std::runtime_error("Q8 product wrote outside requested rows");
            if(tokens==1)qfirst.assign(got.begin(),got.begin()+rows);
            else if(std::memcmp(qfirst.data(),got.data(),rows*4))throw std::runtime_error("Q8 first row changes with verification width");
        }
        // Exercise the full signed-byte range, including -128, and the extrema
        // of the int16 correction. Both VNNI and scalar preparation must agree.
        auto extreme=q8;
        for(size_t i=0;i<extreme.values.size();++i)extreme.values[i]=int8_t(int(i%256)-128);
        std::fill_n(extreme.values.data(),64,int8_t(-128));
        std::fill_n(extreme.values.data()+64,64,int8_t(127));
        auto biased=extreme;strata::kernels::cpu::glm_fp4_prepare_q8_bias(biased);
        for(size_t i=0;i<extreme.values.size();i+=4) {
            int sum=0;for(int j=0;j<4;++j)sum+=int(extreme.values[i+j]);
            if(biased.dot_bias[i/4]!=-12*sum)throw std::runtime_error("Q8 dot bias differs from independent signed sum");
        }
        if(biased.values!=extreme.values||biased.scales!=extreme.scales)
            throw std::runtime_error("Q8 bias preparation changes quantization");
        for(int tokens=1;tokens<=8;++tokens)for(int tile:{1,2,4})for(bool local:{false,true}) {
            std::fill(expected.begin(),expected.end(),-777.f);std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),empty_sparse,table,extreme,expected.data(),columns,rows,tokens,1,rows-1);
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),empty_sparse,table,biased,got.data(),columns,rows,tokens,1,rows-1,tile,local);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("biased Q8 extreme operands change outputs or guards");
        }
#if defined(__unix__)
        // End the code span at a protected page, or 1..3 bytes before it to
        // exercise tensor bases with every alignment. Never read a word from
        // the last CSF byte; gather its containing logical word instead.
        for(int padding:{0,1,2,3})for(bool exceptions:{false,true}) {
            GuardedCodes guarded(source_codes.size()+padding);
            std::copy(source_codes.begin(),source_codes.end(),guarded.data);
            const auto layout=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(
                (exceptions?logical:clean).data(),source_bases.data(),guarded.data,rows,groups,true);
            for(int tokens=1;tokens<=8;++tokens)for(int tile:{1,2,4})for(int first:{0,1,2,3,rows-1}) {
                const int last=std::min(first+5,rows);
                std::fill(expected.begin(),expected.end(),-777.f);std::fill(got.begin(),got.end(),-777.f);
                strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),exceptions?sparse:empty_sparse,
                    table,q8,expected.data(),columns,rows,tokens,first,last);
                strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),layout,table,q8,got.data(),columns,rows,tokens,first,last,tile,true);
                if(std::memcmp(expected.data(),got.data(),got.size()*4))
                    throw std::runtime_error("local scales differ at guarded or unaligned code boundary");
            }
        }
#endif
        biased.dot_bias.pop_back();
        bool bad_gather=false,bad_product=false;
        const int pick=0;
        try{strata::kernels::cpu::glm_fp4_gather_q8(biased,&pick,1);}catch(const std::invalid_argument&){bad_gather=true;}
        try{strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),sparse,table,biased,got.data(),columns,rows,1,0,rows);}catch(const std::invalid_argument&){bad_product=true;}
        if(!bad_gather||!bad_product)throw std::runtime_error("Q8 accepts a truncated bias buffer");
        auto invalid=extreme;invalid.columns-=1;
        bool bad_prepare=false;
        try{strata::kernels::cpu::glm_fp4_prepare_q8_bias(invalid);}catch(const std::invalid_argument&){bad_prepare=true;}
        if(!bad_prepare)throw std::runtime_error("Q8 bias preparation accepts an invalid extent");
        qinput[0]=std::numeric_limits<float>::infinity();
        bool rejected=false;try{strata::kernels::cpu::glm_fp4_quantize_q8(qinput.data(),columns,1);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Q8 accepted nonfinite activation");
    }
    std::cout<<"FP4 CPU CSF32 rows PASS (host codec, sparse/dense exceptions, mask boundaries, widths 1..8)\n";
    std::cout<<"FP4 CPU W4A8 PASS (FP64 dots, quantization error bounds, partial rows, fixed first row) mode="
             <<(strata::kernels::cpu::glm_fp4_q8_vnni_available()?"vnni":"scalar")<<'\n';
    // The dispatch policy is specific to the two native expert shapes. Check
    // their actual dimensions, every width, scale layout and boundary rows.
    for(int rows:{2048,4096}) {
        const int columns=rows==2048?4096:2048,groups=columns/16;
        std::vector<uint8_t> bases(rows,40),codes(size_t(rows)*groups/2),logical(size_t(rows)*groups),w(size_t(rows)*columns/2);
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*53+7);
        const size_t tiles=size_t(rows/128)*(groups/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(rows/32)*(groups/2)+pages);std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(rows,groups,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),page_bases.data(),pages,nullptr,0,logical.data(),logical.size()))throw std::runtime_error("policy CSF32 fixture");
        for(size_t i=0;i<logical.size();i+=1021)logical[i]=uint8_t(8+i%110);
        std::vector<float> input(size_t(columns)*8),reference(size_t(rows)*8),out(reference.size());
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.37f);
        const auto q8=strata::kernels::cpu::glm_fp4_quantize_q8(input.data(),columns,8);
        auto biased=q8;strata::kernels::cpu::glm_fp4_prepare_q8_bias(biased);
        alignas(64)float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,true,table);
        for(bool borrow:{false,true})for(bool cached:{false,true}) {
            auto layout=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),bases.data(),codes.data(),rows,groups,borrow);
            if(cached)strata::kernels::cpu::glm_fp4_cache_scale_quartets(layout);
            for(int tokens=1;tokens<=8;++tokens)for(int first:{0,1,2,3,rows-5}) {
                std::fill(reference.begin(),reference.end(),-777.f);std::fill(out.begin(),out.end(),-777.f);
                strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),layout,table,q8,reference.data(),columns,rows,tokens,first,first+5);
                strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),layout,table,biased,out.data(),columns,rows,tokens,first,first+5,1,false,true);
                if(std::memcmp(reference.data(),out.data(),out.size()*4))throw std::runtime_error("exact CPU dispatch changes product bits or output guards");
            }
            bool rejected=false;
            try{strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),layout,table,q8,out.data(),columns,rows,1,0,1,1,false,true);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("CPU policy accepts missing bias operands");
        }
    }
    std::cout<<"FP4 CPU exact dispatch PASS (both native expert shapes, widths1..8, owned/borrowed/cached and output guards)\n";
    if(argc==2 && std::string(argv[1])=="--bench") {
        constexpr int rows=2048,columns=4096,copies=16,repeats=64;
        const size_t wb=size_t(rows)*columns/2,sb=size_t(rows)*columns/16;
        std::vector<uint8_t> w(wb*copies),s(sb*copies);
        std::vector<float> input(columns*8),baseline(rows*8),output(rows*8);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*59+7);
        for(size_t i=0;i<s.size();++i)s[i]=uint8_t(24+i%100);
        for(size_t i=0;i<input.size();++i)input[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.17f)));
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,true,table);
        for(int tokens:{1,4,8}) {
            strata::kernels::cpu::glm_fp4_rows(w.data(),s.data(),global,input.data(),baseline.data(),columns,rows,tokens,0,rows,true);
            strata::kernels::cpu::glm_fp4_rows_table(w.data(),s.data(),table,input.data(),output.data(),columns,rows,tokens,0,rows);
            if(std::memcmp(baseline.data(),output.data(),size_t(tokens)*rows*4))throw std::runtime_error("cached weight lookup changes CPU outputs");
            for(bool lookup:{false,true}) {
                auto start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i) {
                    const auto* codes=w.data()+(i%copies)*wb;const auto* scales=s.data()+(i%copies)*sb;
                    if(lookup)strata::kernels::cpu::glm_fp4_rows_table(codes,scales,table,input.data(),output.data(),columns,rows,tokens,0,rows);
                    else strata::kernels::cpu::glm_fp4_rows(codes,scales,global,input.data(),output.data(),columns,rows,tokens,0,rows,true);
                }
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                std::cout<<"CPU_KERNEL_BENCH N="<<rows<<" K="<<columns<<" M="<<tokens<<" lookup="<<lookup
                         <<" us="<<us<<" packed_GB_s="<<(wb+sb)/(us*1000.)<<" working_set_bytes="<<copies*(wb+sb)<<" threads=1\n";
            }
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
