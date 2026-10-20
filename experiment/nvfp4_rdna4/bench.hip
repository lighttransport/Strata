#ifndef CPU_ONLY
#include <hip/hip_runtime.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "numeric.hpp"

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static uint16_t bf16_exact(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    require((bits & 65535) == 0, "decoded input is not exact BF16");
    return uint16_t(bits >> 16);
}

static void numeric_tests() {
    const int values[] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    int folded_exact = 0;
    for (unsigned q = 0; q < 16; ++q) {
        require(fp4_x2(q) == values[q], "FP4 decode");
        require(fp4_float(q) == values[q]*0.5f,"FP4 float bit mapping");
        unsigned f = fp4_to_fp8(q);
        require((f & 128 ? -1 : 1)*scale_value(f&127) == values[q]*0.5f, "FP8 mapping");
        int lo = values[q]/2, hi = values[q]-lo;
        require(lo >= -6 && lo <= 6 && hi >= -6 && hi <= 6 && lo+hi == values[q], "INT4 split");
        for (unsigned s = 0; s < 127; ++s) {
            float ref = s >> 3 ? std::ldexp(1.0f+(s&7)*0.125f,int(s>>3)-7) : (s&7)/512.0f;
            require(scale_value(s) == ref,"scale decode");
            float v = std::abs(values[q]*0.5f*ref);
            bf16_exact(v);
            bool exact = false;
            for (unsigned f8 = 0; f8 < 127; ++f8) exact |= v == scale_value(f8);
            folded_exact += exact;
        }
        for (int y : values) {
            int ly=y/2, hy=y-ly;
            require((lo+hi)*ly+(lo+hi)*hy == values[q]*y,"split product");
        }
    }
    std::printf("CPU PASS: all 16 codes, 256 split products, 2032 exact BF16 scaled values; only %d/2032 scaled values fit FP8 exactly\n",folded_exact);
}

#ifndef CPU_ONLY
#define HIP(call) do { hipError_t hip_status_ = (call); if (hip_status_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(hip_status_)); } while (0)

using I8 = int __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));
using F8 = float __attribute__((ext_vector_type(8)));
using B8 = unsigned short __attribute__((ext_vector_type(8)));

template<class T> struct Buffer {
    T* p = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { HIP(hipMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) {
        require(h.size() == count, "buffer size mismatch");
        HIP(hipMemcpy(p, h.data(), count * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> h(count);
        HIP(hipMemcpy(h.data(), p, count * sizeof(T), hipMemcpyDeviceToHost));
        return h;
    }
};

struct Event {
    hipEvent_t e;
    Event() { HIP(hipEventCreate(&e)); }
    ~Event() { (void)hipEventDestroy(e); }
};

#include "kernels.hpp"

__global__ void bf16_gemm(const uint16_t* a, const uint16_t* b, float* out,
                         int m, int n, int kp, float alpha) {
    int lane = threadIdx.x, h = lane / 16, idx = lane % 16;
    int ar = blockIdx.y * 16 + idx, br = blockIdx.x * 16 + idx;
    F8 acc{};
    for (int k = 0; k < kp; k += 16) {
        B8 av{}, bv{};
        if (ar < m) av = *reinterpret_cast<const B8*>(a + size_t(ar)*kp + k + h*8);
        if (br < n) bv = *reinterpret_cast<const B8*>(b + size_t(br)*kp + k + h*8);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
    }
    for (int i = 0; i < 8; ++i) {
        int row = blockIdx.y * 16 + h*8+i;
        if (row < m && br < n) out[size_t(row)*n+br] = acc[i]*alpha;
    }
}

// Independent scalar reference: no WMMA, packed-word conversion, or fragment mapping.
__global__ void reference(const uint8_t* a, const uint8_t* b, const uint8_t* sa,
                          const uint8_t* sb, float* grouped, double* exact, double* absolute,
                          int m, int n, int kp, float alpha) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= size_t(m)*n) return;
    int row = i/n, col = i%n;
    float f = 0;
    double d = 0, magnitude = 0;
    for (int k = 0; k < kp; k += 16) {
        int dot = 0, absdot = 0;
        for (int j = 0; j < 16; ++j) {
            int p = fp4_x2(nibble(a, size_t(row)*kp+k+j)) * fp4_x2(nibble(b, size_t(col)*kp+k+j));
            dot += p;
            absdot += p < 0 ? -p : p;
        }
        float s = scale_value(sa[size_t(row)*kp/16+k/16]) * scale_value(sb[size_t(col)*kp/16+k/16]);
        f = fmaf(float(dot)*0.25f, s, f);
        d += double(dot)*0.25*s;
        magnitude += double(absdot)*0.25*s;
    }
    grouped[i] = f*alpha; exact[i] = d*alpha; absolute[i] = magnitude*std::abs(alpha);
}

struct Case {
    int m, n, k, kp;
    float alpha;
    std::vector<uint8_t> a, b, sa, sb;
    std::vector<uint16_t> fa, fb;
    Case(int mm, int nn, int kk, const std::string& pattern, bool unit)
        : m(mm), n(nn), k(kk), kp((kk+63)&~63), alpha(unit ? 1.0f : 0.9375f), a(size_t(m)*kp/2), b(size_t(n)*kp/2),
          sa(size_t(m)*kp/16), sb(size_t(n)*kp/16), fa(size_t(m)*kp), fb(size_t(n)*kp) {
        std::mt19937 rng(12345);
        const uint8_t scales[] = {0,1,7,8,0x38,0x7e,0x20,0x5d};
        for (int side = 0; side < 2; ++side) {
            int rows = side ? n : m;
            auto& codes = side ? b : a;
            auto& sc = side ? sb : sa;
            auto& decoded = side ? fb : fa;
            for (int r = 0; r < rows; ++r) {
                for (int g = 0; g < kp/16; ++g)
                    sc[size_t(r)*kp/16+g] = unit ? 0x38 : pattern == "all_scales" ? r % 127 : scales[(r+g*3+side*2)%8];
                for (int j = 0; j < kp; ++j) {
                    unsigned q = j >= k ? 0 : rng() & 15;
                    if (j < k && pattern == "codebook") q = r & 15;
                    if (j < k && pattern == "extrema") q = side ? 15 : 7;
                    if (j < k && pattern == "zeros") q = (r+j)&1 ? 8 : 0;
                    if (j < k && pattern == "basis" && side == 0) q = j == r ? 1 : 0;
                    if (j < k && pattern == "cancel") q = side ? (j&1 ? 15 : 7) : 7;
                    codes[size_t(r)*kp/2+j/2] |= q << ((j&1)*4);
                    decoded[size_t(r)*kp+j] = bf16_exact(0.5f*fp4_x2(q)*scale_value(sc[size_t(r)*kp/16+j/16]));
                }
            }
        }
    }
};

static void cpu_reference(const Case& c, size_t i, float& f, double& d) {
    int row = i/c.n, col = i%c.n;
    f = 0; d = 0;
    for (int k = 0; k < c.kp; k += 16) {
        int dot = 0;
        for (int j = 0; j < 16; ++j)
            dot += fp4_x2(nibble(c.a.data(), size_t(row)*c.kp+k+j)) *
                   fp4_x2(nibble(c.b.data(), size_t(col)*c.kp+k+j));
        float s = scale_value(c.sa[size_t(row)*c.kp/16+k/16]) * scale_value(c.sb[size_t(col)*c.kp/16+k/16]);
        f = std::fma(float(dot)*0.25f, s, f);
        d += double(dot)*0.25*s;
    }
    f *= c.alpha; d *= c.alpha;
}

template<class F> static float time_ms(F launch, int iterations) {
    Event start, end;
    HIP(hipEventRecord(start.e));
    for (int i = 0; i < iterations; ++i) launch();
    HIP(hipGetLastError());
    HIP(hipEventRecord(end.e));
    HIP(hipEventSynchronize(end.e));
    float ms;
    HIP(hipEventElapsedTime(&ms, start.e, end.e));
    return ms / iterations;
}

static void run_case(const Case& c, bool unit, bool timing, std::ofstream* csv) {
    size_t count = size_t(c.m)*c.n;
    Buffer<uint8_t> a(c.a.size()), b(c.b.size()), sa(c.sa.size()), sb(c.sb.size());
    Buffer<uint8_t> ia(c.fa.size()), ib(c.fb.size());
    Buffer<uint16_t> fa(c.fa.size()), fb(c.fb.size());
    Buffer<float> out(count+32), ref(count);
    Buffer<double> exact(count), absolute(count);
    a.put(c.a); b.put(c.b); sa.put(c.sa); sb.put(c.sb); fa.put(c.fa); fb.put(c.fb);
    auto convert = [&] {
        expand_fp8<<<(ia.count/4+255)/256,256>>>(a.p,reinterpret_cast<uint32_t*>(ia.p),ia.count);
        expand_fp8<<<(ib.count/4+255)/256,256>>>(b.p,reinterpret_cast<uint32_t*>(ib.p),ib.count);
    };
    convert();
    reference<<<(count+127)/128,128>>>(a.p,b.p,sa.p,sb.p,ref.p,exact.p,absolute.p,c.m,c.n,c.kp,c.alpha);
    HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    auto expected = ref.get(); auto fp64 = exact.get(); auto abs64 = absolute.get();
    size_t samples = timing ? std::min(count,size_t(64)) : count;
    for (size_t j = 0; j < samples; ++j) {
        size_t i = timing && samples > 1 ? j*(count-1)/(samples-1) : j;
        float f; double d; cpu_reference(c,i,f,d);
        require(f == expected[i] && d == fp64[i], "CPU/GPU reference mismatch");
    }
    dim3 grid((c.n+15)/16,(c.m+15)/16);
    dim3 tiled_grid((c.n+63)/64,(c.m+63)/64);
    auto launch = [&](int mode) {
        switch (mode) {
        case 0: fp4_gemm<0><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 1: fp4_gemm<1><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 2: fp4_gemm<2><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 3: fp4_gemm<3><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 4: bf16_gemm<<<grid,32>>>(fa.p,fb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 5: fp4_gemm<5><<<grid,32>>>(ia.p,ib.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 6: fp4_gemm<6><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 7: fp4_gemm<7><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 8: bf16_tiled<true><<<tiled_grid,128>>>(a.p,b.p,sa.p,sb.p,fa.p,fb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 9: bf16_tiled<false><<<tiled_grid,128>>>(a.p,b.p,sa.p,sb.p,fa.p,fb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 10: byte_tiled<false><<<tiled_grid,128>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        case 11: byte_tiled<true><<<tiled_grid,128>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp,c.alpha); break;
        }
    };
    const char* names[] = {"int8_fused","fp8_fused","int4_split_k32","bf16_fused","bf16_predecoded","fp8_preexpanded","fp8_unit","int8_unit","bf16_tiled_fused","bf16_tiled_predecoded","fp8_tiled_fused","int8_tiled_fused"};
    auto check = [&](int mode) {
        auto got = out.get(); double max_error = 0;
        double u = std::numeric_limits<float>::epsilon()/2;
        double gamma = ((c.kp+1)*u)/(1-(c.kp+1)*u);
        for (size_t i = 0; i < count; ++i) {
            double error = std::abs(double(got[i])-fp64[i]);
            if (!std::isfinite(got[i]) || (mode != 3 && mode != 4 && mode != 8 && mode != 9 && got[i] != expected[i]) || error > gamma*abs64[i]) {
                std::fprintf(stderr,"FAIL %s %dx%dx%d index=%zu got=%.9g grouped=%.9g fp64=%.17g bound=%g\n",
                             names[mode],c.m,c.n,c.k,i,got[i],expected[i],fp64[i],gamma*abs64[i]);
                throw std::runtime_error("GEMM validation failed");
            }
            max_error = std::max(max_error,error);
        }
        for (size_t i = count; i < count+32; ++i) require(got[i] == 123.25f,"output canary modified");
        return max_error;
    };
    std::vector<int> modes = {0,1,2,3,4,5,8,9,10,11};
    if (unit) { modes.push_back(6); modes.push_back(7); }
    std::vector<float> sentinel(count+32,123.25f);
    for (int mode : modes) {
        out.put(sentinel); launch(mode); HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); check(mode);
    }
    if (timing) {
        for (int j = 0; j < 5; ++j) for (int mode : modes) launch(mode);
        HIP(hipDeviceSynchronize());
        std::vector<float> times[12];
        for (int round = 0; round < 5; ++round) {
            for (size_t step = 0; step < modes.size(); ++step) {
                int mode = modes[(step+round)%modes.size()];
                float ms = time_ms([&] { launch(mode); },20);
                double error = check(mode);
                times[mode].push_back(ms);
                *csv << c.m << ',' << c.n << ',' << c.k << ',' << (unit ? "unit" : "varying") << ','
                     << names[mode] << ',' << round << ',' << ms << ',' << 2.0*c.m*c.n*c.k/(ms*1e9) << ',' << error << '\n';
            }
            float ms = time_ms(convert,20);
            *csv << c.m << ',' << c.n << ',' << c.k << ',' << (unit ? "unit" : "varying")
                 << ",expand_fp8_both," << round << ',' << ms << ",0,0\n";
        }
        for (int mode : modes) {
            auto& t = times[mode]; std::sort(t.begin(),t.end());
            std::printf("%dx%dx%d %s %s median=%.6f ms range=%.6f..%.6f TOPS=%.4f\n",
                        c.m,c.n,c.k,unit?"unit":"varying",names[mode],t[2],t.front(),t.back(),2.0*c.m*c.n*c.k/(t[2]*1e9));
        }
        csv->flush();
    } else std::printf("GPU PASS %dx%dx%d %s (%d modes)\n",c.m,c.n,c.k,unit?"unit":"varying",int(modes.size()));
}

static void gpu_tests() {
    run_case(Case(16,16,16,"codebook",true),true,false,nullptr);
    run_case(Case(16,16,16,"codebook",false),false,false,nullptr);
    run_case(Case(127,127,16,"all_scales",false),false,false,nullptr);
    for (int k : {16,32,64}) run_case(Case(k,16,k,"basis",true),true,false,nullptr);
    for (const char* pattern : {"zeros","extrema","cancel","random"})
        for (bool unit : {false,true}) run_case(Case(17,19,33,pattern,unit),unit,false,nullptr);
    for (int k : {1,15,17,63,65,129,256}) run_case(Case(3,7,k,"random",false),false,false,nullptr);
    run_case(Case(31,33,127,"random",false),false,false,nullptr);
    run_case(Case(67,131,129,"random",false),false,false,nullptr);
    std::puts("All GPU correctness cases passed");
}
#endif

int main(int argc, char** argv) {
    try {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
#ifdef CPU_ONLY
        (void)argc; (void)argv;
        numeric_tests();
#else
        bool bench = false;
        std::string output = "build/timings.csv";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--bench") bench = true;
            else if (arg == "--test") {}
            else if (arg == "--output" && i+1 < argc) output = argv[++i];
            else throw std::runtime_error("usage: nvfp4_rdna4 [--test|--bench] [--output FILE]");
        }
        numeric_tests();
        hipDeviceProp_t p{}; HIP(hipGetDeviceProperties(&p,0));
        require(std::string(p.gcnArchName).find("gfx1201") == 0,"requires gfx1201 RX 9070 XT");
        HIP(hipSetDevice(0));
        std::printf("GPU: %s arch=%s HIP_multiprocessors=%d wave=%d HIP=%d\n",p.name,p.gcnArchName,p.multiProcessorCount,p.warpSize,HIP_VERSION);
        gpu_tests();
        if (bench) {
            std::ofstream csv(output);
            require(bool(csv),"cannot open output CSV");
            csv.precision(10);
            csv << "m,n,k,scales,kernel,round,ms,tops,max_abs_fp64_error\n";
            for (int m : {1,16,64,512}) {
                run_case(Case(m,2048,4096,"random",false),false,true,&csv);
                run_case(Case(m,4096,2048,"random",false),false,true,&csv);
            }
            run_case(Case(1024,1024,1024,"random",false),false,true,&csv);
            run_case(Case(1024,1024,1024,"random",true),true,true,&csv);
            require(bool(csv),"failed writing output CSV");
        }
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"ERROR: %s\n",e.what());
        return 1;
    }
}
