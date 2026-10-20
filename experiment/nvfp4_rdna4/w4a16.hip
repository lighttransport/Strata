// Tiled layout adapted from src/prefill/wmma_gfx12.cu (MIT; see LICENSE).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "numeric.hpp"

static void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
#define HIP(call) do { hipError_t status_ = (call); if (status_ != hipSuccess) \
    throw std::runtime_error(std::string(#call)+": "+hipGetErrorString(status_)); } while (0)
using B8 = unsigned short __attribute__((ext_vector_type(8)));
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));

template<class T> struct Buffer {
    T* p = nullptr;
    size_t n;
    explicit Buffer(size_t count) : n(count) { HIP(hipMalloc(&p,n*sizeof(T))); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& v) {
        require(v.size()==n,"upload size mismatch");
        HIP(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        HIP(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));
        return v;
    }
};
struct Event {
    hipEvent_t e;
    Event() { HIP(hipEventCreate(&e)); }
    ~Event() { (void)hipEventDestroy(e); }
};

template<bool Half> __host__ __device__ float decode16(uint16_t bits) {
    if constexpr (Half) {
        _Float16 x;
        __builtin_memcpy(&x,&bits,2);
        return float(x);
    } else {
        uint32_t u = uint32_t(bits)<<16;
        float x;
        __builtin_memcpy(&x,&u,4);
        return x;
    }
}
template<bool Half> __host__ __device__ uint16_t encode16(float x) {
    if constexpr (Half) {
        _Float16 v = _Float16(x);
        uint16_t bits;
        __builtin_memcpy(&bits,&v,2);
        return bits;
    } else return to_bf16(x);
}
template<bool Half> __device__ __forceinline__ F8 mma(B8 a, B8 b, F8 c) {
    if constexpr (Half)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(H8,a),__builtin_bit_cast(H8,b),c);
    else return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c);
}
template<bool Half, bool Packed>
__device__ __forceinline__ B8 load_weight(const uint8_t* w, const uint8_t* s,
                                         const uint16_t* expanded, int row, int rows, int kp, int k) {
    B8 v{};
    if (row >= rows) return v;
    if constexpr (!Packed) return *reinterpret_cast<const B8*>(expanded+size_t(row)*kp+k);
    else {
        uint32_t word = *reinterpret_cast<const uint32_t*>(w+size_t(row)*kp/2+k/2);
        float scale = scale_value(s[size_t(row)*kp/16+k/16]);
        #pragma unroll
        for (int i=0;i<8;++i) v[i]=encode16<Half>(fp4_float((word>>(4*i))&15)*scale);
        return v;
    }
}

template<bool Half, bool Packed>
__global__ void simple(const uint16_t* a, const uint8_t* w, const uint8_t* s,
                        const uint16_t* expanded, float* out, int m, int n, int kp, float alpha) {
    int lane=threadIdx.x, h=lane/16, idx=lane%16;
    int row=blockIdx.y*16+idx, col=blockIdx.x*16+idx;
    F8 acc{};
    for (int k=0;k<kp;k+=16) {
        B8 av{};
        if (row<m) av=*reinterpret_cast<const B8*>(a+size_t(row)*kp+k+h*8);
        B8 bv=load_weight<Half,Packed>(w,s,expanded,col,n,kp,k+h*8);
        acc=mma<Half>(av,bv,acc);
    }
    for (int i=0;i<8;++i) {
        int r=blockIdx.y*16+h*8+i;
        if(r<m && col<n) out[size_t(r)*n+col]=acc[i]*alpha;
    }
}

template<bool Half, bool Packed, int BM>
__global__ void tiled(const uint16_t* a, const uint8_t* w, const uint8_t* s,
                       const uint16_t* expanded, float* out, int m, int n, int kp, float alpha) {
    constexpr int R=BM==64?2:1, C=BM==64?2:1;
    int tid=threadIdx.x, lane=tid%32, wave=tid/32, h=lane/16, idx=lane%16;
    int wm=BM==64?wave%2:0, wn=BM==64?wave/2:wave;
    int m0=blockIdx.y*BM, n0=blockIdx.x*64;
    __shared__ B8 sa[4*BM], sw[4*64];
    F8 acc[R][C]{};
    for(int k=0;k<kp;k+=32) {
        for(int i=tid;i<4*BM;i+=128) {
            int row=i%BM, kk=(i/BM)*8;
            B8 v{};
            if(m0+row<m) v=*reinterpret_cast<const B8*>(a+size_t(m0+row)*kp+k+kk);
            sa[i]=v;
        }
        for(int i=tid;i<4*64;i+=128)
            sw[i]=load_weight<Half,Packed>(w,s,expanded,n0+i%64,n,kp,k+(i/64)*8);
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<32;kk+=16) {
            B8 av[R],bv[C];
            #pragma unroll
            for(int r=0;r<R;++r) av[r]=sa[(kk/8+h)*BM+wm*(R*16)+r*16+idx];
            #pragma unroll
            for(int c=0;c<C;++c) bv[c]=sw[(kk/8+h)*64+wn*(C*16)+c*16+idx];
            #pragma unroll
            for(int r=0;r<R;++r)
                #pragma unroll
                for(int c=0;c<C;++c) acc[r][c]=mma<Half>(av[r],bv[c],acc[r][c]);
        }
        __syncthreads();
    }
    for(int r=0;r<R;++r) for(int c=0;c<C;++c) for(int i=0;i<8;++i) {
        int row=m0+wm*R*16+r*16+h*8+i, col=n0+wn*C*16+c*16+idx;
        if(row<m && col<n) out[size_t(row)*n+col]=acc[r][c][i]*alpha;
    }
}

template<bool Half>
__global__ void expand(const uint8_t* w, const uint8_t* s, uint16_t* out, size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) out[i]=encode16<Half>(fp4_float(nibble(w,i))*scale_value(s[i/16]));
}
template<bool Half>
__global__ void reference(const uint16_t* a, const uint8_t* w, const uint8_t* s,
                           double* out, double* absolute, int m,int n,int kp,float alpha) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(m)*n) return;
    size_t row=i/n,col=i%n;
    double total=0, magnitude=0;
    for(int k=0;k<kp;++k) {
        double weight=fp4_x2(nibble(w,col*kp+k))*0.5*scale_value(s[col*kp/16+k/16]);
        double p=double(decode16<Half>(a[row*kp+k]))*weight;
        total+=p; magnitude+=fabs(p);
    }
    out[i]=total*alpha; absolute[i]=magnitude*fabs(double(alpha));
}

template<bool Half> struct Case {
    int m,n,k,kp;
    float alpha;
    std::vector<uint16_t> a,expanded;
    std::vector<uint8_t> w,s;
    Case(int mm,int nn,int kk,const std::string& pattern,bool unit=false)
        :m(mm),n(nn),k(kk),kp((kk+63)&~63),alpha(unit?1.0f:0.75f),
         a(size_t(m)*kp),expanded(size_t(n)*kp),w(size_t(n)*kp/2),s(size_t(n)*kp/16) {
        std::mt19937 rng(98765);
        constexpr int mantissa=Half?10:7, bias=Half?15:127;
        const uint8_t scales[]={0,1,7,8,0x38,0x39,0x7e,0x5d};
        for(int r=0;r<m;++r) for(int j=0;j<k;++j) {
            unsigned bits=(rng()&1)<<15 | unsigned(bias-4+rng()%8)<<mantissa | (rng()&((1u<<mantissa)-1));
            if(pattern=="basis") bits=encode16<Half>(j==r?1.0f:0.0f);
            if(pattern=="cancel") bits=encode16<Half>(Half?1.0009765625f:1.0078125f);
            if(pattern=="range") {
                int exponent=Half?int(rng()%30)+1:int(rng()%121)+67;
                bits=(rng()&1)<<15 | unsigned(exponent)<<mantissa | (rng()&((1u<<mantissa)-1));
            }
            a[size_t(r)*kp+j]=uint16_t(bits);
        }
        for(int r=0;r<n;++r) {
            for(int g=0;g<kp/16;++g)
                s[size_t(r)*kp/16+g]=unit?0x38:pattern=="all_scales"?r%127:scales[(r+g*3)%8];
            for(int j=0;j<kp;++j) {
                unsigned q=j<k?rng()&15:0;
                if(j<k && pattern=="codebook") q=r&15;
                if(j<k && pattern=="zeros") q=(r+j)&1?8:0;
                if(j<k && pattern=="cancel") q=j&1?15:7;
                w[size_t(r)*kp/2+j/2]|=q<<((j&1)*4);
                float x=fp4_x2(q)*0.5f*scale_value(s[size_t(r)*kp/16+j/16]);
                uint16_t bits=encode16<Half>(x);
                require(decode16<Half>(bits)==x,"weight expansion changed value");
                expanded[size_t(r)*kp+j]=bits;
            }
        }
    }
};

template<class F> static float time_ms(F launch) {
    Event start,end;
    HIP(hipEventRecord(start.e));
    for(int i=0;i<20;++i) launch();
    HIP(hipGetLastError()); HIP(hipEventRecord(end.e)); HIP(hipEventSynchronize(end.e));
    float ms; HIP(hipEventElapsedTime(&ms,start.e,end.e)); return ms/20;
}

template<bool Half> static void run(const Case<Half>& c,bool timing,std::ofstream* csv) {
    const char* type=Half?"fp16":"bf16";
    const char* names[]={"simple_packed","simple_predecoded","tile16_packed","tile16_predecoded","tile64_packed","tile64_predecoded"};
    size_t count=size_t(c.m)*c.n;
    Buffer<uint16_t> a(c.a.size()),expanded(c.expanded.size());
    Buffer<uint8_t> w(c.w.size()),s(c.s.size());
    Buffer<float> out(count+32);
    Buffer<double> ref(count),absolute(count);
    a.put(c.a);w.put(c.w);s.put(c.s);expanded.put(c.expanded);
    auto prepare=[&] { expand<Half><<<(expanded.n+255)/256,256>>>(w.p,s.p,expanded.p,expanded.n); };
    prepare(); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    auto decoded=expanded.get();
    for(size_t i=0;i<decoded.size();++i) require(decode16<Half>(decoded[i])==decode16<Half>(c.expanded[i]),"GPU weight expansion mismatch");
    reference<Half><<<(count+127)/128,128>>>(a.p,w.p,s.p,ref.p,absolute.p,c.m,c.n,c.kp,c.alpha);
    HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    auto expected=ref.get(),absref=absolute.get();
    size_t samples=timing?std::min(count,size_t(64)):count;
    for(size_t j=0;j<samples;++j) {
        size_t i=timing && samples>1?j*(count-1)/(samples-1):j;
        double sum=0;
        for(int k=0;k<c.kp;++k)
            sum+=double(decode16<Half>(c.a[(i/c.n)*c.kp+k]))*decode16<Half>(c.expanded[(i%c.n)*c.kp+k]);
        require(sum*c.alpha==expected[i],"CPU/GPU FP64 mismatch");
    }
    auto launch=[&](int mode) {
        dim3 small((c.n+15)/16,(c.m+15)/16), t16((c.n+63)/64,(c.m+15)/16), t64((c.n+63)/64,(c.m+63)/64);
        switch(mode) {
        case 0:simple<Half,true><<<small,32>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        case 1:simple<Half,false><<<small,32>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        case 2:tiled<Half,true,16><<<t16,128>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        case 3:tiled<Half,false,16><<<t16,128>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        case 4:tiled<Half,true,64><<<t64,128>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        case 5:tiled<Half,false,64><<<t64,128>>>(a.p,w.p,s.p,expanded.p,out.p,c.m,c.n,c.kp,c.alpha);break;
        }
    };
    std::vector<float> sentinel(count+32,123.25f);
    out.put(sentinel); launch(1);HIP(hipGetLastError());HIP(hipDeviceSynchronize());
    auto baseline=out.get();
    auto check=[&](int mode) {
        auto got=out.get();double maxerr=0,num=0,den=0;
        double u=std::numeric_limits<float>::epsilon()/2, gamma=((c.kp+1)*u)/(1-(c.kp+1)*u);
        for(size_t i=0;i<count;++i) {
            double e=std::abs(double(got[i])-expected[i]);
            if(!std::isfinite(got[i]) || got[i]!=baseline[i] || e>gamma*absref[i]) {
                std::fprintf(stderr,"FAIL %s %s %dx%dx%d i=%zu got=%g baseline=%g FP64=%.17g bound=%g\n",type,names[mode],c.m,c.n,c.k,i,got[i],baseline[i],expected[i],gamma*absref[i]);
                throw std::runtime_error("W4A16 mismatch");
            }
            maxerr=std::max(maxerr,e);num+=e*e;den+=expected[i]*expected[i];
        }
        for(size_t i=count;i<count+32;++i) require(got[i]==123.25f,"output canary");
        return std::make_pair(maxerr,den?std::sqrt(num/den):0);
    };
    for(int mode=0;mode<6;++mode) {
        out.put(sentinel);launch(mode);HIP(hipGetLastError());HIP(hipDeviceSynchronize());check(mode);
    }
    if(timing) {
        for(int j=0;j<5;++j) for(int mode=0;mode<6;++mode) launch(mode);
        HIP(hipDeviceSynchronize());
        std::vector<float> times[6];
        for(int round=0;round<5;++round) {
            for(int step=0;step<6;++step) {
                int mode=(step+round)%6;
                float ms=time_ms([&]{launch(mode);});auto error=check(mode);times[mode].push_back(ms);
                *csv<<type<<','<<c.m<<','<<c.n<<','<<c.k<<','<<names[mode]<<','<<round<<','<<ms<<','<<2.0*c.m*c.n*c.k/(ms*1e9)<<','<<error.first<<','<<error.second<<'\n';
            }
            float ms=time_ms(prepare);
            *csv<<type<<','<<c.m<<','<<c.n<<','<<c.k<<",expand_weights,"<<round<<','<<ms<<",0,0,0\n";
        }
        for(int mode=0;mode<6;++mode) {
            auto& t=times[mode];std::sort(t.begin(),t.end());
            std::printf("%s %dx%dx%d %s median=%.6f ms range=%.6f..%.6f TOPS=%.4f\n",type,c.m,c.n,c.k,names[mode],t[2],t[0],t[4],2.0*c.m*c.n*c.k/(t[2]*1e9));
        }
        csv->flush();
    } else std::printf("GPU PASS %s %dx%dx%d (6 modes)\n",type,c.m,c.n,c.k);
}

template<bool Half> static void tests() {
    for(unsigned q=0;q<16;++q) for(unsigned s=0;s<127;++s) {
        float v=fp4_x2(q)*0.5f*scale_value(s);
        require(decode16<Half>(encode16<Half>(v))==v,"codebook representability");
    }
    run(Case<Half>(16,16,16,"codebook",true),false,nullptr);
    run(Case<Half>(16,16,16,"codebook"),false,nullptr);
    run(Case<Half>(17,127,33,"all_scales"),false,nullptr);
    for(int k:{16,32,64}) run(Case<Half>(k,19,k,"basis"),false,nullptr);
    for(const char* p:{"zeros","cancel","range","random"}) run(Case<Half>(17,19,33,p),false,nullptr);
    for(int k:{1,15,17,63,65,129,256}) run(Case<Half>(3,7,k,"random"),false,nullptr);
    run(Case<Half>(67,131,129,"random"),false,nullptr);
}

int main(int argc,char** argv) {
    try {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        bool bench=false;std::string output="build/w4a16.csv";
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--bench") bench=true;
            else if(arg=="--test") {}
            else if(arg=="--output" && i+1<argc) output=argv[++i];
            else throw std::runtime_error("usage: w4a16 [--test|--bench] [--output FILE]");
        }
        HIP(hipSetDevice(0));hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));
        require(std::string(p.gcnArchName).find("gfx1201")==0,"requires gfx1201");
        std::printf("GPU %s %s; genuine BF16/FP16 activations; packed E2M1 + K16 E4M3 weights\n",p.name,p.gcnArchName);
        tests<false>();tests<true>();std::puts("All 36 W4A16 correctness cases passed");
        if(bench) {
            std::ofstream csv(output);require(bool(csv),"cannot open CSV");csv.precision(10);
            csv<<"activation,m,n,k,kernel,round,ms,tops,max_abs_fp64_error,rel_l2_fp64\n";
            for(int m:{1,16,64,512}) for(int direction=0;direction<2;++direction) {
                int n=direction?4096:2048,k=direction?2048:4096;
                run(Case<false>(m,n,k,"random"),true,&csv);
                run(Case<true>(m,n,k,"random"),true,&csv);
            }
            run(Case<false>(1024,1024,1024,"random"),true,&csv);
            run(Case<true>(1024,1024,1024,"random"),true,&csv);
            require(bool(csv),"CSV write failure");
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what());return 1; }
}
