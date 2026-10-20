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
    for (unsigned p = 0; p < 65536; ++p) {
        uint32_t word = p | (p << 16), out = recode(word);
        for (int i = 0; i < 8; ++i) {
            unsigned q = (out >> (4 * i)) & 15;
            require((q & 8 ? int(q) - 16 : int(q)) == e0_value(word >> (4 * i)),
                    "packed recode mismatch");
        }
    }
    for (unsigned q = 0; q < 16; ++q)
        for (unsigned s = 0; s < 127; ++s)
            bf16_exact(e0_value(q) * scale_value(s));
    for (unsigned sa = 0; sa < 127; ++sa)
        for (unsigned sb = 0; sb < 127; ++sb) {
            double s = double(scale_value(sa)) * scale_value(sb);
            require(double(float(s)) == s, "scale product rounding");
            for (int a = -7; a <= 7; ++a)
                for (int b = -7; b <= 7; ++b) {
                    double p = a * b * s;
                    require(double(float(p)) == p, "scaled product rounding");
                }
            // K16 dot magnitude <= 784; four-bit scale significands give <= 18 bits.
            require(double(float(784 * s)) == 784 * s, "K16 endpoint rounding");
        }
    std::puts("CPU PASS: 65536 packed patterns, 2032 exact BF16 inputs, all code/scale products");
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

// One wave owns a 16x16 output tile. K fragments are contiguous in each half-wave.
template<bool Convert, bool K32>
__global__ void int4_gemm(const uint8_t* a, const uint8_t* b,
                          const uint8_t* sa, const uint8_t* sb, float* out,
                          int m, int n, int kp) {
    int lane = threadIdx.x, h = lane / 16, idx = lane % 16;
    int ar = blockIdx.y * 16 + idx, br = blockIdx.x * 16 + idx;
    F8 total{};
    I8 whole{};
    for (int k = 0; k < kp; k += K32 ? 32 : 16) {
        I2 av{}, bv{};
        for (int j = 0; j < (K32 ? 2 : 1); ++j) {
            int pos = k + h * (K32 ? 16 : 8) + j * 8;
            uint32_t x = ar < m ? *reinterpret_cast<const uint32_t*>(a + size_t(ar)*kp/2 + pos/2) : 0;
            uint32_t y = br < n ? *reinterpret_cast<const uint32_t*>(b + size_t(br)*kp/2 + pos/2) : 0;
            av[j] = Convert ? recode(x) : x;
            bv[j] = Convert ? recode(y) : y;
        }
        if constexpr (K32) {
            whole = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, av, true, bv, whole, false);
        } else {
            I8 dot = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, av[0], true, bv[0], I8{}, false);
            for (int i = 0; i < 8; ++i) {
                int row = blockIdx.y * 16 + h * 8 + i;
                float s = row < m && br < n ? scale_value(sa[size_t(row)*kp/16+k/16]) *
                    scale_value(sb[size_t(br)*kp/16+k/16]) : 0;
                total[i] = fmaf(float(dot[i]), s, total[i]);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        int row = blockIdx.y * 16 + h * 8 + i;
        if (row < m && br < n) out[size_t(row)*n+br] = K32 ? float(whole[i]) : total[i];
    }
}

__global__ void bf16_gemm(const uint16_t* a, const uint16_t* b, float* out,
                         int m, int n, int kp) {
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
        if (row < m && br < n) out[size_t(row)*n+br] = acc[i];
    }
}

__global__ void convert_codes(const uint32_t* src, uint32_t* dst, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < count) dst[i] = recode(src[i]);
}

// Independent scalar reference: no WMMA, packed-word conversion, or fragment mapping.
__global__ void reference(const uint8_t* a, const uint8_t* b, const uint8_t* sa,
                          const uint8_t* sb, float* grouped, double* exact, double* absolute,
                          int m, int n, int kp) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= size_t(m)*n) return;
    int row = i/n, col = i%n;
    float f = 0;
    double d = 0, magnitude = 0;
    for (int k = 0; k < kp; k += 16) {
        int dot = 0, absdot = 0;
        for (int j = 0; j < 16; ++j) {
            int p = e0_value(nibble(a, size_t(row)*kp+k+j)) * e0_value(nibble(b, size_t(col)*kp+k+j));
            dot += p;
            absdot += p < 0 ? -p : p;
        }
        float s = scale_value(sa[size_t(row)*kp/16+k/16]) * scale_value(sb[size_t(col)*kp/16+k/16]);
        f = fmaf(float(dot), s, f);
        d += double(dot)*s;
        magnitude += double(absdot)*s;
    }
    grouped[i] = f; exact[i] = d; absolute[i] = magnitude;
}

struct Case {
    int m, n, k, kp;
    std::vector<uint8_t> a, b, sa, sb;
    std::vector<uint16_t> fa, fb;
    Case(int mm, int nn, int kk, const std::string& pattern, bool unit)
        : m(mm), n(nn), k(kk), kp((kk+63)&~63), a(size_t(m)*kp/2), b(size_t(n)*kp/2),
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
                    decoded[size_t(r)*kp+j] = bf16_exact(e0_value(q)*scale_value(sc[size_t(r)*kp/16+j/16]));
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
            dot += e0_value(nibble(c.a.data(), size_t(row)*c.kp+k+j)) *
                   e0_value(nibble(c.b.data(), size_t(col)*c.kp+k+j));
        float s = scale_value(c.sa[size_t(row)*c.kp/16+k/16]) * scale_value(c.sb[size_t(col)*c.kp/16+k/16]);
        f = std::fma(float(dot), s, f);
        d += double(dot)*s;
    }
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

static void run_case(const Case& c, bool unit, bool timing, std::ofstream* csv, bool negative = false) {
    size_t count = size_t(c.m)*c.n;
    Buffer<uint8_t> a(c.a.size()), b(c.b.size()), sa(c.sa.size()), sb(c.sb.size());
    Buffer<uint8_t> ia(c.a.size()), ib(c.b.size());
    Buffer<uint16_t> fa(c.fa.size()), fb(c.fb.size());
    Buffer<float> out(count+32), ref(count);
    Buffer<double> exact(count), absolute(count);
    a.put(c.a); b.put(c.b); sa.put(c.sa); sb.put(c.sb); fa.put(c.fa); fb.put(c.fb);
    auto convert = [&] {
        convert_codes<<<(c.a.size()/4+255)/256,256>>>(reinterpret_cast<uint32_t*>(a.p), reinterpret_cast<uint32_t*>(ia.p), c.a.size()/4);
        convert_codes<<<(c.b.size()/4+255)/256,256>>>(reinterpret_cast<uint32_t*>(b.p), reinterpret_cast<uint32_t*>(ib.p), c.b.size()/4);
    };
    convert();
    reference<<<(count+127)/128,128>>>(a.p,b.p,sa.p,sb.p,ref.p,exact.p,absolute.p,c.m,c.n,c.kp);
    HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    auto expected = ref.get(); auto fp64 = exact.get(); auto abs64 = absolute.get();
    size_t samples = timing ? std::min(count,size_t(64)) : count;
    for (size_t j = 0; j < samples; ++j) {
        size_t i = timing && samples > 1 ? j*(count-1)/(samples-1) : j;
        float f; double d; cpu_reference(c,i,f,d);
        require(f == expected[i] && d == fp64[i], "CPU/GPU reference mismatch");
    }
    dim3 grid((c.n+15)/16,(c.m+15)/16);
    auto launch = [&](int mode) {
        switch (mode) {
        case 0: int4_gemm<true,false><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp); break;
        case 1: int4_gemm<false,false><<<grid,32>>>(ia.p,ib.p,sa.p,sb.p,out.p,c.m,c.n,c.kp); break;
        case 2: bf16_gemm<<<grid,32>>>(fa.p,fb.p,out.p,c.m,c.n,c.kp); break;
        case 3: int4_gemm<true,true><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp); break;
        default: int4_gemm<false,false><<<grid,32>>>(a.p,b.p,sa.p,sb.p,out.p,c.m,c.n,c.kp);
        }
    };
    const char* names[] = {"fused_k16","preconverted_k16","bf16_k16","fused_k32_unit"};
    auto check = [&](int mode) {
        auto got = out.get(); double max_error = 0;
        double u = std::numeric_limits<float>::epsilon()/2;
        double gamma = (c.kp*u)/(1-c.kp*u);
        for (size_t i = 0; i < count; ++i) {
            double error = std::abs(double(got[i])-fp64[i]);
            if (!std::isfinite(got[i]) || (mode != 2 && got[i] != expected[i]) || error > gamma*abs64[i]) {
                std::fprintf(stderr,"FAIL %s %dx%dx%d index=%zu got=%.9g grouped=%.9g fp64=%.17g bound=%g\n",
                             names[mode],c.m,c.n,c.k,i,got[i],expected[i],fp64[i],gamma*abs64[i]);
                throw std::runtime_error("GEMM validation failed");
            }
            max_error = std::max(max_error,error);
        }
        for (size_t i = count; i < count+32; ++i) require(got[i] == 123.25f,"output canary modified");
        return max_error;
    };
    int modes = unit ? 4 : 3;
    std::vector<float> sentinel(count+32,123.25f);
    for (int mode = 0; mode < modes; ++mode) {
        out.put(sentinel); launch(mode); HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); check(mode);
    }
    if (negative) {
        launch(4); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
        auto got = out.get(); size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) mismatches += got[i] != expected[i];
        require(mismatches > 0, "raw reinterpretation negative control unexpectedly passed");
        std::printf("Negative control: %zu/%zu mismatches (expected)\n",mismatches,count);
    }
    if (timing) {
        for (int j = 0; j < 5; ++j) for (int mode = 0; mode < modes; ++mode) launch(mode);
        HIP(hipDeviceSynchronize());
        std::vector<float> times[4];
        for (int round = 0; round < 5; ++round) {
            for (int step = 0; step < modes; ++step) {
                int mode = (step+round)%modes;
                float ms = time_ms([&] { launch(mode); },20);
                double error = check(mode);
                times[mode].push_back(ms);
                *csv << c.m << ',' << c.n << ',' << c.k << ',' << (unit ? "unit" : "varying") << ','
                     << names[mode] << ',' << round << ',' << ms << ',' << 2.0*c.m*c.n*c.k/(ms*1e9) << ',' << error << '\n';
            }
            float ms = time_ms(convert,20);
            *csv << c.m << ',' << c.n << ',' << c.k << ',' << (unit ? "unit" : "varying")
                 << ",recode_both," << round << ',' << ms << ",0,0\n";
        }
        for (int mode = 0; mode < modes; ++mode) {
            auto& t = times[mode]; std::sort(t.begin(),t.end());
            std::printf("%dx%dx%d %s %s median=%.6f ms range=%.6f..%.6f TOPS=%.4f\n",
                        c.m,c.n,c.k,unit?"unit":"varying",names[mode],t[2],t.front(),t.back(),2.0*c.m*c.n*c.k/(t[2]*1e9));
        }
        csv->flush();
    } else std::printf("GPU PASS %dx%dx%d %s (%d modes)\n",c.m,c.n,c.k,unit?"unit":"varying",modes);
}

static void gpu_tests() {
    run_case(Case(16,16,16,"codebook",true),true,false,nullptr,true);
    run_case(Case(16,16,16,"codebook",false),false,false,nullptr);
    run_case(Case(127,127,16,"all_scales",false),false,false,nullptr);
    for (int k : {16,32,64}) run_case(Case(k,16,k,"basis",true),true,false,nullptr);
    for (const char* pattern : {"zeros","extrema","cancel","random"})
        for (bool unit : {false,true}) run_case(Case(17,19,33,pattern,unit),unit,false,nullptr);
    for (int k : {1,15,17,63,65,129,256}) run_case(Case(3,7,k,"random",false),false,false,nullptr);
    run_case(Case(31,33,127,"random",false),false,false,nullptr);
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
            else throw std::runtime_error("usage: e0m3_rdna4 [--test|--bench] [--output FILE]");
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
