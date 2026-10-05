// Linux/x86 AVX2 streaming RAM benchmark; no libnuma dependency.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>
#include <numeric>
#include <map>
#include <random>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

static std::string read(const std::string& path) {
    std::ifstream f(path); std::string s; std::getline(f, s); return s;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Separate compiled loops keep the no-prefetch path free of a per-iteration branch.
template<int U, bool PF>
__attribute__((noinline)) static double scan(const double* src, size_t count, size_t ahead) {
    __m256d acc[U];
    for (int k = 0; k < U; ++k) acc[k] = _mm256_setzero_pd();
    for (size_t i = 0; i < count; i += 4 * U) {
        if constexpr (PF) {
            if (i + ahead + 4 * U <= count)
                for (int k = 0; k < U; k += 2)
                    _mm_prefetch(reinterpret_cast<const char*>(src + i + ahead + 4*k), _MM_HINT_T0);
        }
        for (int k = 0; k < U; ++k)
            acc[k] = _mm256_add_pd(acc[k], _mm256_load_pd(src + i + 4*k));
    }
    __m256d total = acc[0];
    for (int k = 1; k < U; ++k) total = _mm256_add_pd(total, acc[k]);
    alignas(32) double out[4]; _mm256_store_pd(out, total);
    return out[0] + out[1] + out[2] + out[3];
}

// Synthetic batch-one packed int4 x int8 GEMV: 4096 weights per output row.
// Shuffled blocks stand in for separately selected expert/matrix segments.
// This deliberately omits format-specific scales, lookup tables and GPU traffic.
__attribute__((noinline)) static void decode(const double* src, size_t bytes,
        const std::vector<size_t>& order, size_t block_bytes, size_t ahead,
        const int8_t* activation, int32_t* output, size_t begin, size_t finish) {
    const auto* w = reinterpret_cast<const uint8_t*>(src);
    const __m256i mask = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    for (size_t index = begin; index < finish; ++index) {
        size_t block = order[index];
        size_t end = std::min(bytes, (block + 1) * block_bytes);
        for (size_t row = block * block_bytes; row < end; row += 2048) {
            __m256i loacc = _mm256_setzero_si256(), hiacc = loacc;
            for (size_t i = 0; i < 2048; i += 32) {
                if (ahead && row + i + ahead < end)
                    _mm_prefetch(reinterpret_cast<const char*>(w + row + i + ahead), _MM_HINT_T0);
                __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + row + i));
                __m256i lo = _mm256_and_si256(packed, mask);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(activation + 2*i));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(activation + 2*i + 32));
                loacc = _mm256_add_epi32(loacc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a), ones));
                hiacc = _mm256_add_epi32(hiacc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, b), ones));
            }
            alignas(32) int32_t lanes[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_add_epi32(loacc, hiacc));
            int32_t sum = 0; for (int v : lanes) sum += v;
            output[row/2048] = sum;
        }
    }
}
static double run_scan(const double* src, size_t count, int unroll, size_t ahead) {
    if (unroll == 4) return ahead ? scan<4,true>(src,count,ahead) : scan<4,false>(src,count,0);
    if (unroll == 8) return ahead ? scan<8,true>(src,count,ahead) : scan<8,false>(src,count,0);
    return ahead ? scan<16,true>(src,count,ahead) : scan<16,false>(src,count,0);
}

template<bool SSE>
__attribute__((noinline)) static double scan_integer(const double* src, size_t count) {
    const auto* words = reinterpret_cast<const uint64_t*>(src);
    alignas(32) uint64_t out[4];
    if constexpr (SSE) {
        __m128i a[8]; for (auto& v : a) v = _mm_setzero_si128();
        for (size_t i = 0; i < count; i += 16)
            for (int k = 0; k < 8; ++k) a[k] = _mm_add_epi64(a[k], _mm_load_si128(reinterpret_cast<const __m128i*>(words+i+2*k)));
        for (int k = 1; k < 8; ++k) a[0] = _mm_add_epi64(a[0],a[k]);
        _mm_store_si128(reinterpret_cast<__m128i*>(out),a[0]);
        return double(out[0]+out[1]);
    } else {
        __m256i a[8]; for (auto& v : a) v = _mm256_setzero_si256();
        for (size_t i = 0; i < count; i += 32)
            for (int k = 0; k < 8; ++k) a[k] = _mm256_add_epi64(a[k], _mm256_load_si256(reinterpret_cast<const __m256i*>(words+i+4*k)));
        for (int k = 1; k < 8; ++k) a[0] = _mm256_add_epi64(a[0],a[k]);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out),a[0]);
        return double(out[0]+out[1]+out[2]+out[3]);
    }
}

static double* allocate(size_t bytes, bool huge) {
    size_t alignment = huge ? 2*1024*1024 : 4096;
    // Trim over-allocation so transparent huge pages can cover the whole mapping.
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { std::cerr << "mmap failed\n"; std::abort(); }
    uintptr_t base = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
    if (aligned > base) munmap(raw, aligned - base);
    size_t tail = base + bytes + alignment - aligned - bytes;
    if (tail) munmap(reinterpret_cast<void*>(aligned + bytes), tail);
    auto* result = reinterpret_cast<double*>(aligned);
    if (madvise(result, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE)) {
        std::cerr << "madvise failed\n"; std::abort();
    }
    return result;
}

static std::vector<int> spread_cpus(const std::vector<int>& cpus) {
    std::map<std::string, size_t> indices;
    std::vector<std::vector<int>> groups;
    for (int cpu : cpus) {
        std::string key = read("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index3/shared_cpu_list");
        if (!indices.count(key)) { indices[key] = groups.size(); groups.emplace_back(); }
        groups[indices[key]].push_back(cpu);
    }
    std::vector<int> result;
    for (size_t i = 0; result.size() < cpus.size(); ++i)
        for (const auto& group : groups) if (i < group.size()) result.push_back(group[i]);
    return result;
}
static void pin_cpu(int cpu) {
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask)) { std::cerr << "CPU pinning failed\n"; std::abort(); }
}
int main(int argc, char** argv) {
    int threads = 0, node = -1, repeats = 10;
    size_t mib = 1024, prefetch = 0, block_kib = 256;
    int unroll = 4, layers = 1; bool huge = false, spread = false;
    std::string kernel = "all";
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--spread") { spread = true; continue; }
            if (a == "--huge") { huge = true; continue; }
            if (a == "--kernel" && i + 1 < argc) { kernel = argv[++i]; continue; }
            if (a == "--help") {
                std::cout << "Usage: membw [--threads N] [--node N] [--mib N] [--repeats N]\n"
                          << "       [--huge] [--spread] [--unroll 4|8|16] [--prefetch BYTES] [--block-kib N] [--layers N]\n"
                          << "       [--kernel all|read|read-int|read-sse|copy|decode|decode-stream]\n"
                          << "Defaults: physical cores in current affinity, all nodes, 1024 MiB per array, 10 repeats.\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing option value");
            std::string value = argv[++i]; size_t end = 0;
            long v = std::stol(value, &end);
            if (end != value.size() || v > INT_MAX || v <= 0) {
                if (!((a == "--node" || a == "--prefetch") && value == "0")) throw std::runtime_error("expected positive integer (node may be zero)");
            }
            if (a == "--threads") threads = v;
            else if (a == "--node") node = v;
            else if (a == "--mib") mib = v;
            else if (a == "--repeats") repeats = v;
            else if (a == "--unroll") unroll = v;
            else if (a == "--prefetch") prefetch = v;
            else if (a == "--block-kib") block_kib = v;
            else if (a == "--layers") layers = v;
            else throw std::runtime_error("unknown option: " + a);
        }
        if (layers > 4096) throw std::runtime_error("layers must be at most 4096");
        if (unroll != 4 && unroll != 8 && unroll != 16) throw std::runtime_error("unroll must be 4, 8 or 16");
        if (prefetch % 64 || prefetch > 1048576) throw std::runtime_error("prefetch must be a multiple of 64, at most 1 MiB");
        if (block_kib < 4 || block_kib > 65536 || block_kib % 4) throw std::runtime_error("block-kib must be a multiple of 4, at most 65536");
        if (kernel != "all" && kernel != "read" && kernel != "read-int" && kernel != "read-sse" && kernel != "copy" && kernel != "decode" && kernel != "decode-stream") throw std::runtime_error("unknown kernel");
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("cannot read CPU affinity");
        std::vector<int> cpus, siblings; std::set<std::string> cores;
        std::set<int> nodes;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &allowed)) continue;
            std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
            int n = -1;
            for (const auto& p : std::filesystem::directory_iterator(base)) {
                std::string name = p.path().filename();
                if (name.rfind("node", 0) == 0) n = std::stoi(name.substr(4));
            }
            if (node >= 0 && node != n) continue;
            nodes.insert(n);
            std::string key = read(base + "/topology/physical_package_id") + ":" + read(base + "/topology/core_id");
            if (cores.insert(key).second) cpus.push_back(c); else siblings.push_back(c);
        }
        if (spread) { cpus = spread_cpus(cpus); siblings = spread_cpus(siblings); }
        if (threads == 0) threads = cpus.size();
        cpus.insert(cpus.end(), siblings.begin(), siblings.end());
        if (threads < 1 || threads > static_cast<int>(cpus.size())) throw std::runtime_error("requested threads exceed eligible CPUs, or node has no eligible CPUs");
        if (mib > (size_t(1) << 20) || repeats > 10000) throw std::runtime_error("size or repeats too large");
        size_t quantum = huge ? 262144 : 512;
        size_t count = (mib * 1024 * 1024 / sizeof(double) / threads / quantum) * quantum;
        if (count == 0) throw std::runtime_error("array too small for thread count");
        size_t bytes = count * sizeof(double);
        std::vector<double*> src(threads), dst(threads);
        std::vector<double> sums(threads);
        std::vector<std::vector<size_t>> orders(threads);
        std::vector<std::vector<int32_t>> outputs(threads);
        alignas(32) int8_t activation[4096];
        for (int i = 0; i < 4096; ++i) activation[i] = 1 + i % 8;
        int32_t reference[256] = {};
        for (int row = 0; row < 256; ++row)
            for (size_t i = 0; i < 2048; i += 32)
                for (size_t j = 0; j < 32; ++j) {
                    uint8_t packed = (i + j + row) & 255;
                    reference[row] += (packed & 15)*activation[2*i+j] + (packed >> 4)*activation[2*i+32+j];
                }
        omp_set_dynamic(0); omp_set_num_threads(threads);
        std::cout << "Threads: " << threads << "; eligible NUMA nodes:";
        for (int n : nodes) std::cout << ' ' << n;
        std::cout << "; worker CPUs:";
        for (int t = 0; t < threads; ++t) std::cout << ' ' << cpus[t];
        std::cout << "\nArray size: " << bytes * threads / 1048576.0 << " MiB each (two arrays); worker first-touch placement\n"
                  << "Page advice: " << (huge ? "huge" : "4 KiB") << "; unroll: " << unroll
                  << "; prefetch: " << prefetch << " bytes; decode block: " << block_kib << " KiB; layers: " << layers << "\n";
        #pragma omp parallel
        {
            if (omp_get_num_threads() != threads) {
                std::cerr << "OpenMP could not create the requested worker count\n"; std::abort();
            }
            int t = omp_get_thread_num(); pin_cpu(cpus[t]);
            src[t] = allocate(bytes, huge); dst[t] = allocate(bytes, huge);
            for (size_t i = 0; i < count; ++i) { src[t][i] = 1.0; dst[t][i] = 0.0; }
            outputs[t].resize(bytes/2048);
            orders[t].resize((bytes + block_kib*1024 - 1)/(block_kib*1024));
            std::iota(orders[t].begin(), orders[t].end(), 0);
        }
        std::vector<std::string> modes = kernel == "all" ? std::vector<std::string>{"read", "copy", "decode"} : std::vector<std::string>{kernel};
        for (const auto& mode : modes) {
            bool isdecode = mode == "decode" || mode == "decode-stream";
            bool integer = mode == "read-int" || mode == "read-sse";
            if (integer) {
                #pragma omp parallel
                {
                    int t = omp_get_thread_num();
                    for (size_t i = 0; i < count; ++i) { uint64_t value = i % 512 + 1; std::memcpy(src[t]+i, &value, 8); }
                }
            }
            if (isdecode) {
                #pragma omp parallel
                {
                    int t = omp_get_thread_num();
                    auto* packed = reinterpret_cast<uint8_t*>(src[t]);
                    for (size_t row = 0; row < bytes/2048; ++row)
                        for (size_t i = 0; i < 2048; ++i) packed[row*2048+i] = (i + row) & 255;
                    if (mode == "decode") {
                        std::mt19937 rng(42+t);
                        std::shuffle(orders[t].begin(), orders[t].end(), rng);
                    }
                }
            }
            std::vector<double> rates;
            double start = 0;
            // A persistent worker team matches an inference thread pool.
            #pragma omp parallel shared(start, rates)
            {
                int t = omp_get_thread_num(); pin_cpu(cpus[t]);
                for (int r = -1; r < repeats; ++r) {
                    #pragma omp barrier
                    #pragma omp single
                    { start = now(); }
                    if (mode == "read") sums[t] = run_scan(src[t], count, unroll, prefetch/8);
                    else if (integer) sums[t] = mode == "read-sse" ? scan_integer<true>(src[t],count) : scan_integer<false>(src[t],count);
                    else if (isdecode) {
                        for (int layer = 0; layer < layers; ++layer) {
                            decode(src[t], bytes, orders[t], block_kib*1024, prefetch, activation, outputs[t].data(),
                                   orders[t].size()*layer/layers, orders[t].size()*(layer+1)/layers);
                            // A token cannot advance to the next layer until its current layer finishes.
                            if (layers > 1) {
                                #pragma omp barrier
                            }
                        }
                    }
                    else {
                        for (size_t i = 0; i < count; i += 4)
                            _mm256_stream_pd(dst[t] + i, _mm256_load_pd(src[t] + i));
                        _mm_sfence();
                    }
                    #pragma omp barrier
                    #pragma omp single
                    {
                        double elapsed = now() - start;
                        if (r >= 0) rates.push_back(double(bytes) * threads * (mode == "copy" ? 2 : 1) / elapsed / 1e9);
                    }
                }
            }
            int errors = 0;
            #pragma omp parallel reduction(+:errors)
            {
                int t = omp_get_thread_num();
                if (mode == "read") { if (sums[t] != double(count)) ++errors; }
                else if (integer) { if (sums[t] != double(count/2)*513) ++errors; }
                else if (isdecode) {
                    // Check every output against independent scalar references with varied nibbles.
                    for (size_t row = 0; row < outputs[t].size(); ++row)
                        if (outputs[t][row] != reference[row % 256]) ++errors;
                } else for (size_t i = 0; i < count; ++i) if (dst[t][i] != 1.0) ++errors;
            }
            if (errors) throw std::runtime_error("data validation failed");
            std::sort(rates.begin(), rates.end());
            double median = (rates[(rates.size()-1)/2] + rates[rates.size()/2]) / 2;
            std::cout << mode << ": median " << median << ", best " << rates.back() << ", min " << rates.front()
                      << " GB/s (decimal), " << repeats << " passes; validation OK\n";
        }
        for (int t = 0; t < threads; ++t) { munmap(src[t], bytes); munmap(dst[t], bytes); }
    } catch (const std::exception& e) { std::cerr << "membw: " << e.what() << '\n'; return 1; }
}
