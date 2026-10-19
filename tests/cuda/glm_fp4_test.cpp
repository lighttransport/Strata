#include "strata/kernels/glm_fp4.hpp"
#include "strata/host/glm_env.h"
#include "strata/artifact/glm_modelopt.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include <cuda_runtime.h>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
namespace k=strata::kernels;
namespace a=strata::artifact;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {void* p=nullptr;explicit Buffer(size_t bytes){check(cudaMalloc(&p,std::max<size_t>(bytes,1)));}~Buffer(){cudaFree(p);}};
static void routing() {
    constexpr int tokens=32,count=17,stride=8;
    for(int columns:{37,4096}) {
        std::vector<float> input(tokens*columns),values(count*columns),weights(tokens*stride),sum(tokens*columns),got(sum.size());
        std::vector<int> routes(count);
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.13f)*(i%107==0?8000.f:1.f);
        for(size_t i=0;i<values.size();++i)values[i]=std::cos(float(i)*.31f)*2.17f;
        for(size_t i=0;i<weights.size();++i)weights[i]=i%13?std::cos(float(i)*.19f)*.7f:0.f;
        for(size_t i=0;i<sum.size();++i)sum[i]=std::sin(float(i)*.17f);
        for(int j=0;j<count;++j)routes[j]=((j*5)%tokens)*stride+j%stride;
        Buffer x(input.size()*4),v(values.size()*4),w(weights.size()*4),s(sum.size()*4),r(routes.size()*4),half(values.size()*2);
        check(cudaMemcpy(x.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(v.p,values.data(),values.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(w.p,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.p,sum.data(),sum.size()*4,cudaMemcpyHostToDevice));
        for(int pass=0;pass<2;++pass) {
            if(pass)std::reverse(routes.begin(),routes.end());
            check(cudaMemcpy(r.p,routes.data(),routes.size()*4,cudaMemcpyHostToDevice));
            k::glm_fp4_route_gather(static_cast<float*>(x.p),static_cast<int*>(r.p),static_cast<uint16_t*>(half.p),count,columns,stride,nullptr);
            std::vector<uint16_t> gathered(count*columns);
            check(cudaMemcpy(gathered.data(),half.p,gathered.size()*2,cudaMemcpyDeviceToHost));
            for(int j=0;j<count;++j)for(int h=0;h<columns;++h)
                if(gathered[j*columns+h]!=strata::cpu::exl3::half_round(input[(routes[j]/stride)*columns+h]))
                    throw std::runtime_error("routed FP16 gather differs from host reference");
            k::glm_fp4_route_scatter(static_cast<float*>(v.p),static_cast<int*>(r.p),static_cast<float*>(w.p),static_cast<float*>(s.p),count,columns,stride,nullptr);
            for(int j=0;j<count;++j)for(int h=0;h<columns;++h) {
                volatile float product=weights[routes[j]]*values[j*columns+h];
                sum[(routes[j]/stride)*columns+h]+=product;
            }
        }
        check(cudaMemcpy(got.data(),s.p,got.size()*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(sum.data(),got.data(),sum.size()*4))throw std::runtime_error("ordered GPU route sum differs from host reference");
        k::glm_fp4_route_gather(nullptr,nullptr,nullptr,0,columns,stride,nullptr);
        k::glm_fp4_route_scatter(nullptr,nullptr,nullptr,nullptr,0,columns,stride,nullptr);
    }
    std::cout<<"device routing gather and ordered sum PASS (bitwise, partial tiles, reversed routes)\n";
}
static void csf32() {
    // Independent encoder, including multiple exception pages and two
    // exceptions in every 32x2 block. Compare packed output byte for byte.
    constexpr int N=4096,K=2048,C=K/16;
    std::vector<uint8_t> bases(N,40),codes,logical(N*C),decoded(N*C);
    std::vector<uint16_t> offsets,exceptions;
    std::vector<uint32_t> pages;
    int tile=0;
    for(int r0=0;r0<N;r0+=128)for(int c0=0;c0<C;c0+=4,++tile) {
        if(tile%32==0){pages.push_back(exceptions.size()*2);offsets.push_back(0);}
        for(int col=c0;col<c0+4;col+=2)for(int row=r0;row<r0+128;row+=32) {
            for(int rr=0;rr<32;++rr) {
                const unsigned lo=(row+rr+col)%16,hi=(row+rr+col+1)%16;
                codes.push_back(lo|(hi<<4));logical[(row+rr)*C+col]=40+lo;logical[(row+rr)*C+col+1]=40+hi;
            }
            exceptions.push_back(0|(81<<8));logical[row*C+col]=81;
            exceptions.push_back(63|(11<<8));logical[(row+31)*C+col+1]=11;
            offsets.push_back(exceptions.size()-pages.back()/2);
        }
    }
    if(strata_fp4_csf32_decode(N,C,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),
        pages.data(),pages.size(),exceptions.data(),exceptions.size(),decoded.data(),decoded.size()) || decoded!=logical)
        throw std::runtime_error("CSF32 fixture differs from C11 decoder");
    Buffer b(bases.size()),c(codes.size()),o(offsets.size()*2),p(pages.size()*4),e(exceptions.size()*2);
    Buffer result(logical.size()),flag(4),raw(size_t(N)*K/2),packed(size_t(N)*K/2),want(logical.size()),s(logical.size());
    check(cudaMemcpy(b.p,bases.data(),bases.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(c.p,codes.data(),codes.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(o.p,offsets.data(),offsets.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.p,pages.data(),pages.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(e.p,exceptions.data(),exceptions.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.p,logical.data(),logical.size(),cudaMemcpyHostToDevice));
    check(cudaMemset(raw.p,0,size_t(N)*K/2));
    k::glm_fp4_prepack(static_cast<uint8_t*>(raw.p),static_cast<uint8_t*>(s.p),static_cast<uint8_t*>(packed.p),static_cast<uint8_t*>(want.p),N,K,nullptr);
    k::glm_fp4_csf32_scales(static_cast<uint8_t*>(b.p),static_cast<uint8_t*>(c.p),static_cast<uint16_t*>(o.p),
        static_cast<uint32_t*>(p.p),static_cast<uint16_t*>(e.p),static_cast<uint8_t*>(result.p),static_cast<int*>(flag.p),N,K,nullptr);
    int invalid=-1;check(cudaMemcpy(&invalid,flag.p,4,cudaMemcpyDeviceToHost));
    std::vector<uint8_t> expected(logical.size()),got(logical.size());
    check(cudaMemcpy(expected.data(),want.p,expected.size(),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(),result.p,got.size(),cudaMemcpyDeviceToHost));
    if(invalid || got!=expected)throw std::runtime_error("GPU CSF32 packing differs from C11 reference");
    exceptions[0]=0|(127<<8);check(cudaMemcpy(e.p,exceptions.data(),exceptions.size()*2,cudaMemcpyHostToDevice));
    k::glm_fp4_csf32_scales(static_cast<uint8_t*>(b.p),static_cast<uint8_t*>(c.p),static_cast<uint16_t*>(o.p),
        static_cast<uint32_t*>(p.p),static_cast<uint16_t*>(e.p),static_cast<uint8_t*>(result.p),static_cast<int*>(flag.p),N,K,nullptr);
    check(cudaMemcpy(&invalid,flag.p,4,cudaMemcpyDeviceToHost));
    if(invalid!=1)throw std::runtime_error("GPU CSF32 accepted NaN scale");
    std::cout<<"CSF32 GPU scale packing PASS\n";
}
static void real_scales(const a::ModelOptCsf32& h,int N,int K,const void* expected) {
    Buffer b(N),c(h.codes_count),o(h.offsets.size()*2),p(h.pages.size()*4),e(h.exceptions.size()*2);
    Buffer output(size_t(N)*K/16),flag(4);
    check(cudaMemcpy(b.p,h.bases,N,cudaMemcpyHostToDevice));
    check(cudaMemcpy(c.p,h.codes,h.codes_count,cudaMemcpyHostToDevice));
    check(cudaMemcpy(o.p,h.offsets.data(),h.offsets.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.p,h.pages.data(),h.pages.size()*4,cudaMemcpyHostToDevice));
    if(!h.exceptions.empty())check(cudaMemcpy(e.p,h.exceptions.data(),h.exceptions.size()*2,cudaMemcpyHostToDevice));
    k::glm_fp4_csf32_scales(static_cast<uint8_t*>(b.p),static_cast<uint8_t*>(c.p),static_cast<uint16_t*>(o.p),
        static_cast<uint32_t*>(p.p),static_cast<uint16_t*>(e.p),static_cast<uint8_t*>(output.p),static_cast<int*>(flag.p),N,K,nullptr);
    std::vector<uint8_t> want(size_t(N)*K/16),got(want.size());
    check(cudaMemcpy(want.data(),expected,want.size(),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(),output.p,got.size(),cudaMemcpyDeviceToHost));
    int invalid=-1;check(cudaMemcpy(&invalid,flag.p,4,cudaMemcpyDeviceToHost));
    if(invalid || got!=want)throw std::runtime_error("real GPU CSF32 scales differ from host reference");
    std::cout<<"real CSF32 scale packing PASS\n";
}
static void head_partition() {
    // Both halves remain above the dense-split threshold, as on the real head.
    // Check the separate QPN code/scale offsets and every verification width.
    constexpr int N=16448,K=256,half=N/2;
    std::vector<uint8_t> codes(size_t(N)*K),scales(size_t(N)*K/32,125);
    for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*7%127)|uint8_t((i&1)*128);
    std::vector<uint16_t> input(size_t(8)*K);
    for(size_t i=0;i<input.size();++i)input[i]=strata::cpu::exl3::half_round(std::sin(float(i)*.17f)*2);
    Buffer raw(codes.size()),rs(scales.size()),packed(codes.size()),ps(scales.size()*2),
        tail(codes.size()/2),tail_scales(scales.size()),
        x(input.size()*2),whole(size_t(8)*N*4),part(size_t(8)*half*4);
    check(cudaMemcpy(raw.p,codes.data(),codes.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(rs.p,scales.data(),scales.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
    k::glm_fp8_prepack(k::GlmPackedFormat::Mxfp8,static_cast<uint8_t*>(raw.p),rs.p,
        static_cast<uint8_t*>(packed.p),ps.p,N,K,nullptr);
    k::glm_fp8_prepack(k::GlmPackedFormat::Mxfp8,static_cast<uint8_t*>(raw.p)+size_t(half)*K,
        static_cast<uint8_t*>(rs.p)+size_t(half)*K/32,
        static_cast<uint8_t*>(tail.p),tail_scales.p,half,K,nullptr);
    std::vector<float> expected(size_t(8)*N),got(size_t(8)*half);
    for(int acc:{1,2,4})for(int M=1;M<=8;++M) {
        k::glm_fp4_gemm(k::GlmPackedFormat::Mxfp8Qpn,packed.p,ps.p,1,
            static_cast<uint16_t*>(x.p),static_cast<float*>(whole.p),N,K,M,true,nullptr,nullptr,acc);
        check(cudaMemcpy(expected.data(),whole.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
        for(int rank=0;rank<2;++rank) {
            k::glm_fp4_gemm(k::GlmPackedFormat::Mxfp8Qpn,
                rank?tail.p:packed.p,rank?tail_scales.p:ps.p,1,
                static_cast<uint16_t*>(x.p),static_cast<float*>(part.p),half,K,M,true,nullptr,nullptr,acc);
            check(cudaMemcpy(got.data(),part.p,size_t(M)*half*4,cudaMemcpyDeviceToHost));
            for(int t=0;t<M;++t)if(std::memcmp(got.data()+size_t(t)*half,
                expected.data()+size_t(t)*N+rank*half,size_t(half)*4))
                throw std::runtime_error("head row partition changes FP8 output bits");
        }
    }
    std::cout<<"FP8 head partition bits=identical widths=1..8 accumulators=1,2,4 PASS\n";
}
static void run(a::ModelOptPacked h,bool outliers,bool benchmark=false,const a::ModelOptCsf32* csf=nullptr) {
    const int N=h.rows,K=h.columns;
    const size_t bytes=h.weights.size();
    const size_t packed_scale_bytes=h.scales.size()*(h.format==a::ModelOptFormat::Mxfp8?2:1);
    Buffer raw(bytes),scales(h.scales.size()),packed(bytes),ps(packed_scale_bytes);
    check(cudaMemcpy(raw.p,h.data(),bytes,cudaMemcpyHostToDevice));
    if(!h.scales.empty())check(cudaMemcpy(scales.p,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice));
    const auto fmt=static_cast<k::GlmPackedFormat>(h.format);
    const bool packed_fp8=(fmt==k::GlmPackedFormat::Mxfp8||fmt==k::GlmPackedFormat::BlockFp8)&&N%32==0;
    const auto cached_format=fmt==k::GlmPackedFormat::Bf16?k::GlmPackedFormat::F16:
        fmt==k::GlmPackedFormat::Mxfp8?k::GlmPackedFormat::Mxfp8Qpn:k::GlmPackedFormat::BlockFp8Qpn;
    const void* w=raw.p;const void* s=scales.p;
    if(h.format==a::ModelOptFormat::Nvfp4) {
        k::glm_fp4_prepack(static_cast<uint8_t*>(raw.p),static_cast<uint8_t*>(scales.p),static_cast<uint8_t*>(packed.p),static_cast<uint8_t*>(ps.p),N,K,nullptr);
        w=packed.p;s=ps.p;
        if(csf)real_scales(*csf,N,K,s);
    }
    Buffer x(size_t(K)*8*2),y(size_t(N)*8*4),dh(size_t(N)*K*2);
    std::vector<uint16_t> hx(size_t(K)*8),expected_half(size_t(N)*K),actual_half(expected_half.size());
    for(int t=0;t<8;++t)for(int j=0;j<K;++j) {
        float value=std::sin(float(j+13*t)*.37f)*2;
        if(outliers&&j%901==0)value=8192.f*(t&1?-1:1);
        hx[size_t(t)*K+j]=strata::cpu::exl3::half_round(value);
    }
    for(int r=0;r<N;++r)for(int j=0;j<K;++j)
        expected_half[size_t(r)*K+j]=strata::cpu::exl3::half_round(a::GlmModelOpt::value(h,r,j));
    k::glm_fp4_dequant_f16(fmt,w,s,h.global,static_cast<uint16_t*>(dh.p),N,K,nullptr);
    check(cudaMemcpy(actual_half.data(),dh.p,actual_half.size()*2,cudaMemcpyDeviceToHost));
    if(actual_half!=expected_half)throw std::runtime_error("FP4 native-format reconstruction differs");
    if(fmt==k::GlmPackedFormat::Nvfp4) {
        k::glm_fp4_dequant_f16(fmt,w,s,h.global,static_cast<uint16_t*>(dh.p),N,K,nullptr,true);
        check(cudaMemcpy(actual_half.data(),dh.p,actual_half.size()*2,cudaMemcpyDeviceToHost));
        const float global=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(h.global*16384.f));
        for(int r=0;r<N;++r)for(int j=0;j<K;++j) {
            const float scale=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(strata_fp4_e4m3(h.scales[size_t(r)*(K/16)+j/16])*global))/16384.f;
            const unsigned q=(h.data()[size_t(r)*(K/2)+j/2]>>((j&1)*4))&15;
            const auto want=strata::cpu::exl3::half_round(strata_fp4_e2m1(q)*scale);
            if(actual_half[size_t(r)*K+j]!=want)throw std::runtime_error("SM70 prefill operand differs from skinny decoder");
        }
    }
    if(fmt==k::GlmPackedFormat::Bf16) {
        check(cudaMemcpy(packed.p,raw.p,bytes,cudaMemcpyDeviceToDevice));
        k::glm_fp4_dequant_f16(fmt,packed.p,nullptr,1,static_cast<uint16_t*>(packed.p),N,K,nullptr);
        check(cudaMemcpy(actual_half.data(),packed.p,actual_half.size()*2,cudaMemcpyDeviceToHost));
        if(actual_half!=expected_half)throw std::runtime_error("in-place BF16 operand cache differs");
    }
    if(packed_fp8) {
        k::glm_fp8_prepack(fmt,static_cast<uint8_t*>(raw.p),scales.p,static_cast<uint8_t*>(packed.p),ps.p,N,K,nullptr);
        k::glm_fp4_dequant_f16(cached_format,packed.p,ps.p,h.global,static_cast<uint16_t*>(dh.p),N,K,nullptr);
        check(cudaMemcpy(actual_half.data(),dh.p,actual_half.size()*2,cudaMemcpyDeviceToHost));
        if(actual_half!=expected_half)throw std::runtime_error("packed FP8 cache differs");
    }
    check(cudaMemcpy(x.p,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    std::vector<float> ref(size_t(N)*8),got(ref.size()),one;
    for(int t=0;t<8;++t)for(int r=0;r<N;++r) {
        double v=0;
        for(int j=0;j<K;++j)v+=double(strata::cpu::exl3::half_value(expected_half[size_t(r)*K+j]))*
            strata::cpu::exl3::half_value(hx[size_t(t)*K+j]);
        ref[size_t(t)*N+r]=float(v);
    }
    Buffer partials(k::glm_fp4_partial_bytes(N));
    const bool small=fmt!=k::GlmPackedFormat::Nvfp4 && N<=8192 && K>=4096;
    if(packed_fp8) {
        Buffer half_cache(size_t(N)*K*2);
        std::vector<float> cache_reference(ref.size());
        for(bool cached:{false,true}) {
            const auto active=cached?cached_format:fmt;
            const void* operand=cached?packed.p:w;
            const void* active_scales=cached?ps.p:s;
            k::glm_fp8_cache_f16_qpn(active,operand,active_scales,
                static_cast<uint16_t*>(half_cache.p),N,K,nullptr);
            k::glm_fp4_dequant_f16(k::GlmPackedFormat::F16Qpn,half_cache.p,nullptr,1,
                static_cast<uint16_t*>(dh.p),N,K,nullptr);
            check(cudaMemcpy(actual_half.data(),dh.p,actual_half.size()*2,cudaMemcpyDeviceToHost));
            if(actual_half!=expected_half)throw std::runtime_error("fixed FP16 lane cache changes reconstructed operands");
            for(bool split:{false,true}) {
                if(split&&!small)continue;
                float* scratch=split?static_cast<float*>(partials.p):nullptr;
                for(int M=1;M<=8;++M) {
                    k::glm_fp4_gemm(active,operand,active_scales,h.global,static_cast<uint16_t*>(x.p),
                        static_cast<float*>(y.p),N,K,M,true,nullptr,scratch);
                    check(cudaMemcpy(cache_reference.data(),y.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
                    k::glm_fp4_gemm(k::GlmPackedFormat::F16Qpn,half_cache.p,nullptr,1,static_cast<uint16_t*>(x.p),
                        static_cast<float*>(y.p),N,K,M,true,nullptr,scratch);
                    check(cudaMemcpy(got.data(),y.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
                    if(std::memcmp(cache_reference.data(),got.data(),size_t(M)*N*4))
                        throw std::runtime_error("fixed FP16 lane cache changes skinny output bits");
                }
            }
        }
        std::cout<<"fixed FP16 lane cache operands and widths1..8 bitwise PASS\n";
    }
    for(bool cached:{false,true})for(bool split:{false,true}) {
    if((cached && fmt!=k::GlmPackedFormat::Bf16 && !packed_fp8) || (split && !small))continue;
    const auto active=cached?cached_format:fmt;
    const void* operand=cached?packed.p:w;
    const void* active_scales=cached&&packed_fp8?ps.p:s;
    for(int accumulators:{1,2,4})for(bool sm70:{false,true})for(int M=1;M<=8;++M) {
        if(accumulators!=1&&(!sm70||(fmt!=k::GlmPackedFormat::Mxfp8&&fmt!=k::GlmPackedFormat::BlockFp8)))continue;
        if(split&&!sm70)continue;
        k::glm_fp4_gemm(active,operand,active_scales,h.global,static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),N,K,M,sm70,nullptr,split?static_cast<float*>(partials.p):nullptr,accumulators);
        check(cudaMemcpy(got.data(),y.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
        double error=0,energy=0;
        for(int i=0;i<M*N;++i) {
            if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite FP4 output");
            error+=double(got[i]-ref[i])*(got[i]-ref[i]);energy+=double(ref[i])*ref[i];
        }
        const double rel=std::sqrt(error/std::max(energy,1e-30));
        if(rel>(sm70?.002:1e-5))throw std::runtime_error("FP4 GEMM parity: rel="+std::to_string(rel));
        if(sm70) {
            if(M==1)one.assign(got.begin(),got.begin()+N);
            else if(std::memcmp(one.data(),got.data(),N*sizeof(float)))throw std::runtime_error("verification width changes first-row MMA arithmetic");
        }
        std::cout<<"format="<<int(active)<<" N="<<N<<" K="<<K<<" M="<<M<<" sm70="<<sm70<<" split="<<split<<" accumulators="<<accumulators<<" outliers="<<outliers<<" rel_l2="<<rel<<'\n';
    }
    }
    const size_t stride=(bytes+255)/256*256,scale_stride=(packed_scale_bytes+255)/256*256;
    const size_t copies=benchmark?std::min<size_t>(32,std::max<size_t>(1,128*1024*1024/(stride+scale_stride))):1;
    Buffer ring(benchmark?stride*copies:1),ring_scales(benchmark?scale_stride*copies:1);
    if(benchmark)for(bool cached:{false,true}) {
    if(cached && fmt!=k::GlmPackedFormat::Bf16 && !packed_fp8)continue;
    const auto active=cached?cached_format:fmt;
    const void* operand=cached?packed.p:w;
    const void* active_scales=cached&&packed_fp8?ps.p:s;
    const size_t scale_bytes=cached&&packed_fp8?packed_scale_bytes:h.scales.size();
    for(size_t i=0;i<copies;++i) {
        check(cudaMemcpy(static_cast<char*>(ring.p)+i*stride,operand,bytes,cudaMemcpyDeviceToDevice));
        if(scale_bytes)check(cudaMemcpy(static_cast<char*>(ring_scales.p)+i*scale_stride,active_scales,scale_bytes,cudaMemcpyDeviceToDevice));
    }
    for(int accumulators:{1,2,4})for(bool rotate:{false,true})for(bool split:{false,true})for(bool sm70:{false,true})for(int M:{1,4,8}) {
        if(accumulators!=1&&(!sm70||(fmt!=k::GlmPackedFormat::Mxfp8&&fmt!=k::GlmPackedFormat::BlockFp8)))continue;
        if((rotate&&copies==1) || (split&&(!small||!sm70)))continue;
        float* scratch=split?static_cast<float*>(partials.p):nullptr;
        for(int i=0;i<10;++i)k::glm_fp4_gemm(active,operand,active_scales,h.global,static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),N,K,M,sm70,nullptr,scratch,accumulators);
        check(cudaDeviceSynchronize());
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<100;++i) {
            const size_t slot=rotate?i%copies:0;
            k::glm_fp4_gemm(active,static_cast<char*>(ring.p)+slot*stride,static_cast<char*>(ring_scales.p)+slot*scale_stride,
                h.global,static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),N,K,M,sm70,stream,scratch,accumulators);
        }
        cudaGraph_t graph;check(cudaStreamEndCapture(stream,&graph));
        cudaGraphExec_t exec;check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int i=0;i<10;++i)check(cudaGraphLaunch(exec,stream));
        check(cudaStreamSynchronize(stream));
        cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
        check(cudaEventRecord(start,stream));for(int i=0;i<10;++i)check(cudaGraphLaunch(exec,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float ms=0;check(cudaEventElapsedTime(&ms,start,end));const double us=ms;
        std::cout<<"KERNEL_BENCH format="<<int(active)<<" N="<<N<<" K="<<K<<" M="<<M<<" sm70="<<sm70<<" split="<<split<<" accumulators="<<accumulators
                 <<" us="<<us<<" compressed_GB_s="<<(bytes+scale_bytes)/(us*1000.)
                 <<" operand_copies="<<(rotate?copies:1)<<" working_set_bytes="<<(rotate?copies:1)*(bytes+scale_bytes)
                 <<" launches=1000 cuda_graph=1\n";
        check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    }
    }
}
int main(int argc,char** argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    cudaDeviceProp properties{};
    if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess||properties.major<7)return 77;
    try {
        routing();
        if(argc==2 && std::string(argv[1])=="--routing-only")return 0;
        head_partition();
        if(argc==2 && std::string(argv[1])=="--head-partition-only")return 0;
        csf32();
        const bool dense_split=argc==2 && std::string(argv[1])=="--dense-split";
        if(argc==3 || argc==4) {
            a::Safetensors tensors(argv[1],true,false,strata_env_is_one("STRATA_GLM_FP4_RAM_FDS"));
            const auto cfg=a::read_json((std::filesystem::path(argv[1])/"config.json").string());
            a::GlmModelOpt model(tensors,cfg);const auto m=model.matrix(argv[2]);
            auto c=model.has_csf32(m)?std::make_unique<a::ModelOptCsf32>(model.csf32(m)):nullptr;
            run(model.load(m),true,argc==4&&std::string(argv[3])=="--bench",c.get());
        }else for(int format=0;format<5;++format)for(int K:dense_split?std::vector<int>{4096}:std::vector<int>{64,256,512,4096})
        for(int N:dense_split?std::vector<int>{1024,2048,4096,8192}:std::vector<int>{format==4?24:128})for(bool outliers:{false,true}) {
            if(dense_split&&(format==0||format==4))continue;
            a::ModelOptPacked h{static_cast<a::ModelOptFormat>(format),size_t(N),size_t(K),.00061f,{},{}};
            h.weights.resize(size_t(N)*K*(format==0?1:format==3?4:format==4?8:2)/2);
            if(format==0) {h.scales.resize(N*K/16);for(size_t i=0;i<h.weights.size();++i)h.weights[i]=uint8_t(i*53+17);for(size_t i=0;i<h.scales.size();++i)h.scales[i]=uint8_t(24+i%80);}
            if(format==1) {h.scales.resize(N*K/32);for(size_t i=0;i<h.weights.size();++i)h.weights[i]=uint8_t((i*7)%127)|uint8_t((i&1)*128);for(size_t i=0;i<h.scales.size();++i)h.scales[i]=uint8_t(K==512?(i%33?105+i%30:0):120+i%7);}
            if(format==2) {h.scales.resize(((N+127)/128)*((K+127)/128)*4);for(size_t i=0;i<h.weights.size();++i)h.weights[i]=uint8_t((i*7)%120)|uint8_t((i&1)*128);for(size_t i=0;i<h.scales.size();i+=4){float s=.003f*(1+i);std::memcpy(h.scales.data()+i,&s,4);}}
            if(format>=3)for(size_t i=0;i<size_t(N)*K;++i) {
                const float v=.03f*std::cos(float(i)*.1f);
                if(format==3) {const uint16_t b=std::bit_cast<uint32_t>(v)>>16;std::memcpy(h.weights.data()+i*2,&b,2);}
                else std::memcpy(h.weights.data()+i*4,&v,4);
            }
            run(std::move(h),outliers);
        }
        std::cout<<"native FP4 / scaled FP8 CUDA PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
