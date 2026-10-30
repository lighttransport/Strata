// src/program/ds41_decode.cpp - DeepSeek-V4.1-Flash in strata-glm-decode (a GGUF with general.architecture
// deepseek41, llama.cpp PR #30817's schema). strata-glm-decode dispatches here by architecture; the GLM decoder is
// untouched.
//
// One token at a time: the trunk (attention, compressor, indexer, hyper-connections, shared expert, head) runs on the
// GPU from the GGUF's own quantized tensors (native MMVQ, BF16/F32 GEMVs and the kernels of strata/kernels/ds41.hpp);
// the routed experts run on the CPU lane (strata/ds41/expert_lane.hpp, IQ1_S on Fmt32<19>); the router's top-k, the
// indexer's top-k and the Engram rows (read from the mapped table) are host work. The operations follow PR #30817
// (src/models/deepseek41.cpp), the reference tools/ds41f_llama_dump.cpp captures.
//
//   strata-glm-decode MODEL-00001-of-0000N.gguf <token ids|@file> [steps] [--threads=N] [--context=N]
//                     [--check=DUMPDIR]   compare every layer of every prompt position with a --serial dump
//                     [--dump-logits=PATH] the prompt's last-position logits (f32)
//                     [--eval-corpus=IDS --eval-reference=LOGITS | --eval-save-logits=LOGITS]
#include "strata/core/model.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/decode.hpp"
#include "strata/ds41/engram_wire.hpp"
#include "strata/ds41/expert_lane.hpp"
#include "strata/ds41/stream_config.hpp"
#include "strata/host/glm_env.h"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/net/expert_tp.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"
#include "ucomm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <deque>
#include <initializer_list>
#include <barrier>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds41 {
namespace k = strata::kernels;
namespace cpu = strata::kernels::cpu;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41: ") + what + ": " + cudaGetErrorString(e));
}

float cpu_trim_threshold() {
    const char* raw = std::getenv("STRATA_DS41_CPU_TRIM");
    if (!raw) return 0.0f;
    char* end = nullptr;
    (void)std::strtod(raw, &end);
    const double v = strata_env_value_double("STRATA_DS41_CPU_TRIM", 0.0);
    if (end == raw || *end || !std::isfinite(v) || v < 0 || v > 0.25)
        throw std::invalid_argument("ds41: CPU_TRIM must be in [0, 0.25]");
    return float(v);
}

int dspark_cache_slots() {
    const int slots = strata_env_int_or("STRATA_DS41_DSPARK_CACHE_SLOTS", 45);
    if (slots < 1 || slots > 100) throw std::invalid_argument("ds41: DSpark cache slots must be in [1, 100]");
    return slots;
}

// STRATA_DS41_DESKTOP_MIB (2048): VRAM this process leaves free for the desktop. Every device allocation checks it
// and refuses with a message instead of starving the compositor (a shared desktop loses its session when VRAM runs out).
size_t vram_reserve = [] {
    const char* v = std::getenv("STRATA_DS41_DESKTOP_MIB");
    return size_t(v ? std::max(0, std::atoi(v)) : 2048) << 20;
}();

#if defined(__HIP_PLATFORM_AMD__)
// HIP's reported free memory can exclude allocations owned by the desktop. Read
// physical VRAM before sizing any tier or making any individual allocation.
size_t physical_vram_free() {
    int device = 0;
    char bus[32] = {};
    ck(cudaGetDevice(&device), "current device");
    ck(cudaDeviceGetPCIBusId(bus, sizeof bus, device), "device PCI address");
    std::string id(bus);
    for (char& ch : id) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    const std::string base = "/sys/bus/pci/devices/" + id + "/mem_info_vram_";
    unsigned long long total = 0, used = 0;
    std::ifstream tf(base + "total"), uf(base + "used");
    if (!(tf >> total) || !(uf >> used) || used > total)
        throw std::runtime_error("ds41: cannot read physical AMD VRAM usage");
    return size_t(total - used);
}

size_t physical_vram_keep() {
    static const size_t keep = [] {
        double headroom_mib = 1024;
        if (const char* raw = std::getenv("STRATA_DS41_GPU_HEADROOM_MIB")) {
            char* end = nullptr;
            headroom_mib = std::strtod(raw, &end);
            if (end == raw || *end || !std::isfinite(headroom_mib) || headroom_mib < 0 || headroom_mib > 4096)
                throw std::invalid_argument("ds41: GPU headroom must be in [0, 4096] MiB");
        }
        const size_t headroom = size_t(headroom_mib) << 20;
        if (vram_reserve > SIZE_MAX - headroom) throw std::runtime_error("ds41: GPU reserve plus headroom overflow");
        return std::max(vram_reserve + headroom, size_t(2560) << 20);
    }();
    return keep;
}
#endif

// STRATA_DS41_CANON_STREAM_THREADS: helper threads that copy streamed experts from the mapped files into pinned
// slots and enqueue each part's DMA as soon as its copy is done (pageable copies run at about half the link's rate)
class CanonStager {
public:
    struct Job { const uint8_t* src; uint8_t* pinned; void* dev; size_t bytes; };
    CanonStager(int threads, cudaStream_t stream) : stream_(stream) {
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { work(); });
    }
    ~CanonStager() {
        { std::lock_guard<std::mutex> g(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    // runs every job (memcpy, then its DMA on the stream) and returns once all are enqueued
    void run(const std::vector<Job>& jobs) {
        if (jobs.empty()) return;
        {
            std::lock_guard<std::mutex> g(m_);
            jobs_ = &jobs; next_ = 0; done_ = 0; error_.clear(); ++gen_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> g(m_);
        done_cv_.wait(g, [&] { return done_ == jobs.size(); });
        jobs_ = nullptr;
        if (!error_.empty()) throw std::runtime_error(error_);
    }
private:
    void work() {
        uint64_t seen = 0;
        for (;;) {
            const std::vector<Job>* jobs = nullptr;
            {
                std::unique_lock<std::mutex> g(m_);
                cv_.wait(g, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                jobs = jobs_;
            }
            for (;;) {
                size_t i;
                {
                    std::lock_guard<std::mutex> g(m_);
                    if (!jobs || next_ >= jobs->size()) break;
                    i = next_++;
                }
                const Job& j = (*jobs)[i];
                std::string err;
                std::memcpy(j.pinned, j.src, j.bytes);
                const cudaError_t e = cudaMemcpyAsync(j.dev, j.pinned, j.bytes, cudaMemcpyHostToDevice, stream_);
                if (e != cudaSuccess) err = std::string("ds41: canonical staged copy: ") + cudaGetErrorString(e);
                std::lock_guard<std::mutex> g(m_);
                if (!err.empty() && error_.empty()) error_ = err;
                if (++done_ == jobs->size()) done_cv_.notify_all();
            }
        }
    }
    cudaStream_t stream_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::vector<Job>* jobs_ = nullptr;
    size_t next_ = 0, done_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::string error_;
};

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    bool borrowed = false;   // a view of another Dev's memory (a second decoder in the process shares the first one's weights and tier): never freed here
    Dev() = default;
    static Dev view(const Dev& o) { Dev d; d.p = o.p; d.bytes = o.bytes; d.borrowed = true; return d; }
    explicit Dev(size_t n) : bytes(n) {
        const size_t allocation = std::max<size_t>(n, 16);
        static const bool probe = std::getenv("STRATA_DS41_ALLOC_PROBE") != nullptr;
#if defined(__HIP_PLATFORM_AMD__)
        const size_t physical_free = physical_vram_free();
        const size_t physical_keep = physical_vram_keep();
        if (physical_free < physical_keep || allocation > physical_free - physical_keep)
            throw std::runtime_error("ds41: device allocation of " + std::to_string(allocation >> 20) +
                                     " MiB would leave " + std::to_string(physical_free > allocation ? (physical_free - allocation) >> 20 : 0) +
                                     " MiB free below the " + std::to_string(physical_keep >> 20) + " MiB physical AMD VRAM reserve");
#endif
        if (vram_reserve > 0) {   // STRATA_DS41_DESKTOP_MIB: never take VRAM the desktop needs
            size_t f = 0, t = 0;
            ck(cudaMemGetInfo(&f, &t), "memory info before allocation");
            if (f < vram_reserve || allocation > f - vram_reserve)
                throw std::runtime_error("ds41: cudaMalloc of " + std::to_string(n >> 20) + " MiB would leave " +
                                         std::to_string((f > n ? f - n : 0) >> 20) + " MiB free, below the " +
                                         std::to_string(vram_reserve >> 20) + " MiB desktop reserve (STRATA_DS41_DESKTOP_MIB); " +
                                         "lower STRATA_DS41_PROMPT_BATCH or a tile size");
        }
        if (probe) {
            size_t f = 0, t = 0;
            cudaMemGetInfo(&f, &t);
            std::fprintf(stderr, "DS41 alloc %zu MiB (free %zu MiB)\n", n >> 20, f >> 20);
        }
        ck(cudaMalloc(&p, allocation), "cudaMalloc");
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), bytes(o.bytes), borrowed(o.borrowed) { o.p = nullptr; o.borrowed = false; }
    Dev& operator=(Dev&& o) noexcept { std::swap(p, o.p); std::swap(bytes, o.bytes); std::swap(borrowed, o.borrowed); return *this; }
    ~Dev() { if (p && !borrowed) cudaFree(p); }
    float* f(size_t at = 0) const { return static_cast<float*>(p) + at; }
    int* i() const { return static_cast<int*>(p); }
    uint16_t* h() const { return static_cast<uint16_t*>(p); }
    uint8_t* b(size_t at = 0) const { return static_cast<uint8_t*>(p) + at; }
};

size_t checked_size_mul(size_t a, size_t b, const char* what) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error(std::string("ds41: size overflow for ") + what);
    return a * b;
}

void require_device_range(const Dev& owner, size_t offset, size_t bytes, const char* what) {
    if (!owner.p || offset > owner.bytes || bytes > owner.bytes - offset)
        throw std::runtime_error(std::string("ds41: device buffer bounds for ") + what);
}

void* device_range_ptr(const Dev& owner, size_t offset, size_t bytes, const char* what) {
    require_device_range(owner, offset, bytes, what);
    return static_cast<uint8_t*>(owner.p) + offset;
}

struct Weight {
    Dev data;
    int type = 0;
    int in = 0, out = 0;
    size_t row_bytes = 0;
};

// ggml's YaRN rotation table for one position (ggml-cpu/ops.cpp ggml_rope_cache_init + rope_yarn), pairs {cos, sin}
void rope_table(const Rope& r, int n_dims, int64_t pos, float sign, float* cs) {
    const float theta_scale = std::pow(r.base, -2.0f / float(n_dims));
    float corr[2] = {0, 0};
    if (r.ext_factor != 0.0f) {
        auto dim = [&](float rot) {
            return float(n_dims) * std::log(float(r.ctx_orig) / (rot * 2.0f * float(M_PI))) / (2.0f * std::log(r.base));
        };
        corr[0] = std::max(0.0f, std::floor(dim(r.beta_fast)));
        corr[1] = std::min(float(n_dims - 1), std::ceil(dim(r.beta_slow)));
    }
    float theta = float(pos);
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const float extrap = theta;
        const float interp = r.freq_scale * extrap;
        float t = interp, mscale = r.attn_factor;
        if (r.ext_factor != 0.0f) {
            const float y = (float(i0 / 2) - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
            const float ramp = (1 - std::min(1.0f, std::max(0.0f, y))) * r.ext_factor;
            t = interp * (1 - ramp) + extrap * ramp;
            mscale *= 1.0f + 0.1f * std::log(1.0f / r.freq_scale);
        }
        cs[i0] = std::cos(t) * mscale;
        cs[i0 + 1] = std::sin(t) * mscale * sign;
        theta *= theta_scale;
    }
}

// STRATA_DS41_LOW_PRIORITY (1): the decoder's streams at the device's lowest priority, so the desktop's GPU work
// (compositor, browser) is scheduled first
cudaStream_t make_stream(unsigned flags) {
    cudaStream_t s = nullptr;
    const bool low = !(std::getenv("STRATA_DS41_LOW_PRIORITY") && std::string(std::getenv("STRATA_DS41_LOW_PRIORITY")) == "0");
    int least = 0, greatest = 0;
#ifdef STRATA_USE_HIP
    if (low && hipDeviceGetStreamPriorityRange(&least, &greatest) == hipSuccess &&
        hipStreamCreateWithPriority(&s, flags, least) == hipSuccess)
        return s;
#else
    if (low && cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess &&
        cudaStreamCreateWithPriority(&s, flags, least) == cudaSuccess)
        return s;
#endif
    (void)cudaGetLastError();
    ck(cudaStreamCreateWithFlags(&s, flags), "stream");
    return s;
}

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// indices of the k largest values, ties to the lower index, sorted by value (descending)
std::vector<int> top_k(const float* v, int n, int k) {
    std::vector<int> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); });
    idx.resize(size_t(k));
    return idx;
}

// an I32 / I64 tensor's values (the GSQ-RCO files' Engram hash tables)
std::vector<int64_t> tensor_ints(const core::ArtifactTensor& t) {
    const size_t n = static_cast<size_t>(t.tensor->elements());
    std::vector<int64_t> out(n);
    if (t.tensor->type == 26) {
        std::vector<int32_t> v(n);
        std::memcpy(v.data(), t.data(), n * 4);
        for (size_t i = 0; i < n; ++i) out[i] = v[i];
    } else if (t.tensor->type == 27) {
        std::memcpy(out.data(), t.data(), n * 8);
    } else
        throw std::runtime_error("ds41: " + t.tensor->name + " is not an integer tensor");
    return out;
}

std::vector<float> to_float(int type, const uint8_t* src, int64_t n) {
    std::vector<float> out(static_cast<size_t>(n));
    if (type == GGML_TYPE_F32) std::memcpy(out.data(), src, size_t(n) * 4);
    else ggml_get_type_traits(ggml_type(type))->to_float(src, out.data(), n);
    return out;
}

}  // namespace

// The process's CPU set, captured before the expert lane pins the main thread to one core: helper threads (staging
// copies, Engram reads, uploads) start from it instead of inheriting that single core.
const cpu_set_t& process_cpus() {
    static const cpu_set_t set = [] {
        cpu_set_t s;
        CPU_ZERO(&s);
        if (sched_getaffinity(0, sizeof s, &s) != 0) for (int i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i, &s);
        // STRATA_DS41_HELPER_CPUS=a,b,...: the helper threads run on these CPUs only, away from the lane's pinned cores (the lane's
        // threads spin at barriers, so one helper sharing a core delays a whole pass)
        if (const char* v = std::getenv("STRATA_DS41_HELPER_CPUS"); v && *v) {
            cpu_set_t h;
            CPU_ZERO(&h);
            for (const char* q = v; *q;) {
                char* end = nullptr;
                const long c = std::strtol(q, &end, 10);
                if (end == q) break;
                if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(int(c), &s)) CPU_SET(int(c), &h);
                q = *end == ',' ? end + 1 : end;
            }
            if (CPU_COUNT(&h) > 0) s = h;
        }
        return s;
    }();
    return set;
}
inline void unpin_helper() { pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &process_cpus()); }

// dst (32-byte aligned) = src, n bytes: AVX2 loads, non-temporal stores (no read of the destination's lines)
__attribute__((target("avx2"))) inline void copy_nt_avx2(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 31) == 0) {
        for (; i + 128 <= n; i += 128) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 32));
            const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 64));
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 96));
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), a);
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 32), b);
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 64), c);
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 96), d);
        }
        _mm_sfence();
    }
    if (i < n) std::memcpy(dst + i, src + i, n - i);
}

// The other node's rows of every routed expert (STRATA_DS41_REMOTE_TP, include/strata/net/expert_tp.hpp): a
// strata-glm-tp-worker in DS4.1 mode holds FFN rows [fl, ff) of each expert (and the matching down columns) in its RAM.
// A decode pass sends each layer's routes and z there before its own work and adds the reply after it; a streamed
// prompt fetches the rows back into the VRAM slots beside the local ones.
// STRATA_DS41_ENGRAM_REMOTE=HOST:PORT: the Engram table pages are read on the node that holds the model (strata-ds41-cpu-worker
// --engram-port) instead of from local files; rank 0 of its own ucomm context, one batched request per call
class EngramRemote {
  public:
    explicit EngramRemote(const std::string& addr) {
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 0;
        cfg.world = 2;
        cfg.root_addr = addr.c_str();
        if (!std::getenv("UCOMM_TIMEOUT_MS")) cfg.timeout_ms = 0;
        ok(ucomm_init(&cfg, &c_), "engram connect");
        request_.resize(strata::ds41::engram_wire::kMaxRequestBytes);
        std::cerr << "DS41 engram remote backend=" << ucomm_backend_name(c_) << "\n";
    }
    ~EngramRemote() {
        if (!c_) return;
        const strata::ds41::engram_wire::Header end{0xFFFFFFFFu, 0};
        (void)ucomm_send(c_, 1, strata::ds41::engram_wire::kTagRequest, &end, sizeof end);
        ucomm_finalize(c_);
    }
    EngramRemote(const EngramRemote&) = delete;
    EngramRemote& operator=(const EngramRemote&) = delete;
    // reads every entry (shard, offset, len) in order into `out`; callers serialize on the lock
    void read(const std::vector<strata::ds41::engram_wire::Entry>& entries, std::vector<uint8_t>& out) {
        namespace ew = strata::ds41::engram_wire;
        std::lock_guard<std::mutex> lock(m_);
        size_t total = 0;
        for (const auto& e : entries) total += e.len;
        out.resize(total);
        for (size_t first = 0, bytes_at = 0; first < entries.size();) {
            const size_t n = std::min<size_t>(ew::kMaxEntries, entries.size() - first);
            size_t chunk = 0;
            for (size_t i = 0; i < n; ++i) chunk += entries[first + i].len;
            const ew::Header h{uint32_t(n), 0};
            std::memcpy(request_.data(), &h, sizeof h);
            std::memcpy(request_.data() + sizeof h, entries.data() + first, n * sizeof(ew::Entry));
            ucomm_req_t* rx = nullptr;
            ok(ucomm_irecv(c_, 1, ew::kTagReply, out.data() + bytes_at, chunk, &rx), "engram receive");
            ok(ucomm_send(c_, 1, ew::kTagRequest, request_.data(), sizeof h + n * sizeof(ew::Entry)), "engram send");
            size_t got = 0;
            ok(ucomm_wait(rx, &got), "engram wait");
            if (got != chunk) throw std::runtime_error("ds41 engram remote: short reply");
            first += n;
            bytes_at += chunk;
        }
    }

  private:
    static void ok(ucomm_status s, const char* what) {
        if (s != UCOMM_OK) throw std::runtime_error(std::string("ds41 ") + what + ": " + ucomm_strerror(s));
    }
    ucomm_t* c_ = nullptr;
    std::mutex m_;
    std::vector<uint8_t> request_;
};

struct Remote {
    ucomm_t* comm = nullptr;
    int hidden = 0;
    std::vector<uint8_t> request;   // registered once (IB)
    std::vector<float> reply;
    std::vector<ucomm_mr_t*> regions;
    ucomm_req_t *sending = nullptr, *receiving = nullptr;
    bool reply_pinned = false;   // registered with the GPU runtime too (the streamed pass uploads it asynchronously)
    double wait_ms = 0;
    int64_t calls = 0;
    struct Geo { size_t rows, gu_row, d_keep, d_rest, bytes, stride; };   // a layer's worker blob: [gate | up | down]
    std::vector<Geo> geo;
    static void ok(ucomm_status s, const char* what) {
        if (s != UCOMM_OK) throw std::runtime_error(std::string("ds41 remote ") + what + ": " + ucomm_strerror(s));
    }
    // reply_rows: hidden-wide rows of the reply (STD1: nt summed rows; STD2: one raw row per owned route)
    void start(int layer, int nt, int K, const int* ids, const float* w, const float* z, int reply_rows = -1) {
        const strata::net::TpRequest head{layer, nt, nt * K, 4 * hidden};
        const size_t rows = reply_rows >= 0 ? size_t(reply_rows) : size_t(nt);
        const size_t jobs = size_t(nt) * K * sizeof(strata::net::TpJob), bytes = sizeof head + jobs + size_t(nt) * hidden * 4;
        if (bytes > request.size() || rows * hidden > reply.size())
            throw std::runtime_error("ds41 remote: a pass larger than the request buffers");
        std::memcpy(request.data(), &head, sizeof head);
        auto* list = reinterpret_cast<strata::net::TpJob*>(request.data() + sizeof head);
        for (int j = 0; j < nt * K; ++j) list[j] = {ids[j], j / K, w[j]};
        std::memcpy(request.data() + sizeof head + jobs, z, size_t(nt) * hidden * 4);
        ok(ucomm_irecv(comm, 1, strata::net::kTagReply, reply.data(), rows * hidden * 4, &receiving), "receive");
        ok(ucomm_isend(comm, 1, strata::net::kTagRequest, request.data(), bytes, &sending), "send");
    }
    void finish_raw(int) {   // the reply in `reply`
        const auto t0 = std::chrono::steady_clock::now();
        ok(ucomm_wait(sending, nullptr), "send");
        ok(ucomm_wait(receiving, nullptr), "reply");
        sending = receiving = nullptr;
        wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++calls;
    }
    void finish(float* sum, int nt) {   // sum[nt][hidden] += the worker's partial
        const auto t0 = std::chrono::steady_clock::now();
        ok(ucomm_wait(sending, nullptr), "send");
        ok(ucomm_wait(receiving, nullptr), "reply");
        sending = receiving = nullptr;
        wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++calls;
        for (size_t i = 0; i < size_t(nt) * hidden; ++i) sum[i] += reply[i];
    }
    ~Remote() {
        if (!comm) return;
        const strata::net::TpRequest quit{-1, 0, 0, 0};
        ucomm_send(comm, 1, strata::net::kTagRequest, &quit, sizeof quit);
        std::cerr << "DS41 remote backend=" << ucomm_backend_name(comm) << " calls=" << calls
                  << " wait_ms_per_call=" << (calls ? wait_ms / calls : 0.0) << "\n";
        for (auto* r : regions) ucomm_mr_dereg(r);
        if (reply_pinned) cudaHostUnregister(reply.data());
        ucomm_finalize(comm);
    }
};

// A second GPU holding whole experts (STRATA_DS41_RAM_COPY's residency, STRATA_DS41_PEER_GPU=1): per layer, its experts
// contiguous at a block-aligned stride; decode routes to them run MMVQ there, prompt passes MMQ, and the partial sums
// come back through pinned host memory. Every call selects its device and restores device 0.
// The NUMA nodes of the socket a GPU hangs off (its PCI device's node and the nodes whose CPUs share its package),
// as an mbind mask; 0 when unknown
inline unsigned long gpu_socket_nodes(int dev) {
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    std::string id(bus);
    for (auto& ch : id) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    auto first_int = [](const std::string& path) {
        std::ifstream f(path);
        long v = -1;
        f >> v;
        return v;
    };
    auto package_of = [&](long node) {
        std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
        long cpu = -1;
        f >> cpu;
        return cpu < 0 ? -1L : first_int("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/physical_package_id");
    };
    const long node = first_int("/sys/bus/pci/devices/" + id + "/numa_node");
    if (node < 0) return 0;
    const long pkg = package_of(node);
    unsigned long mask = 0;
    for (long n = 0; n < 64; ++n)
        if (std::filesystem::exists("/sys/devices/system/node/node" + std::to_string(n)) && package_of(n) == pkg) mask |= 1ul << n;
    return mask;
}

// STRATA_DS41_DECODE_GROUPED=1: a short pass's routes to one GPU's resident experts in one grouped launch
// (native_expert_grouped: gate/up, SwiGLU, down for every (expert, token) entry), then the weighted rows added
// into `sum`. Expert blobs at device pointers in the tier layout [gate | up | down].
struct GroupedRunner {
    Dev table, scratch, out, xq;
    // the tables' pinned (portable) staging, two buffers by call parity: a buffer is refilled once its upload two calls
    // back is done (an event), so the host never waits for the stream's other work (it had synchronized the stream,
    // on GPU 0 the whole decode stream, every layer)
    uint8_t* stagings[2] = {nullptr, nullptr};
    size_t staging_sizes[2] = {0, 0};
    cudaEvent_t uploaded[2] = {nullptr, nullptr};
    unsigned calls = 0;
    int cap = 0;
    // groups: (expert blob, its routes (token, weight)); x: [nt][hidden] floats on this GPU
    bool run(int gu_type, int d_type, int hidden, int ff, float clamp, const std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>>& groups,
             const float* x, int nt, float* sum, cudaStream_t stream) {
        if (groups.empty()) return false;
        int E = 0;
        for (const auto& g : groups) E += int(g.second.size());
        const int G = int(groups.size());
        if (E > cap) {
            cap = std::max(E, 64);
            scratch = Dev(k::native_expert_scratch_bytes(cap, ff));
            out = Dev(size_t(cap) * hidden * 4);
            table = Dev(size_t(cap) * 64);
        }
        if (xq.bytes < k::native_q8_1_bytes(hidden, 1) * size_t(nt)) xq = Dev(k::native_q8_1_bytes(hidden, 1) * size_t(nt) * 2);
        // tables: ptr[G] (u64), start[G + 1], n_groups, dst[E], tok[E], w[E]
        const size_t need = size_t(G) * 8 + size_t(G + 1) * 4 + 4 + size_t(E) * 12 + 64;
        const int sb = int(calls++ & 1u);
        if (uploaded[sb]) ck(cudaEventSynchronize(uploaded[sb]), "grouped staging");   // its upload two calls back
        else ck(cudaEventCreateWithFlags(&uploaded[sb], cudaEventDisableTiming), "grouped event");
        if (need > staging_sizes[sb]) {
            if (stagings[sb]) cudaFreeHost(stagings[sb]);
            ck(cudaHostAlloc(reinterpret_cast<void**>(&stagings[sb]), need * 2, cudaHostAllocPortable), "grouped staging");
            staging_sizes[sb] = need * 2;
        }
        uint8_t* staging = stagings[sb];
        auto* ptr = reinterpret_cast<unsigned long long*>(staging);
        auto* start = reinterpret_cast<int32_t*>(ptr + G);
        int32_t* ng = start + G + 1;
        int32_t* dst = ng + 1;
        int32_t* tok = dst + E;
        float* w = reinterpret_cast<float*>(tok + E);
        int e = 0;
        for (int g = 0; g < G; ++g) {
            ptr[g] = reinterpret_cast<unsigned long long>(groups[size_t(g)].first);
            start[g] = e;
            for (const auto& [t, wt] : groups[size_t(g)].second) { dst[e] = e; tok[e] = t; w[e] = wt; ++e; }
        }
        start[G] = e;
        *ng = G;
        const size_t bytes = reinterpret_cast<uint8_t*>(w + E) - staging;
        if (table.bytes < bytes) table = Dev(bytes * 2);
        ck(cudaMemcpyAsync(table.p, staging, bytes, cudaMemcpyHostToDevice, stream), "grouped tables");
        ck(cudaEventRecord(uploaded[sb], stream), "grouped event");
        const uint8_t* tb = table.b();
        const size_t o_start = size_t(G) * 8, o_ng = o_start + size_t(G + 1) * 4, o_dst = o_ng + 4, o_tok = o_dst + size_t(E) * 4,
                     o_w = o_tok + size_t(E) * 4;
        for (int j = 0; j < nt; j += 8)
            k::native_quantize_q8_1(x + size_t(j) * hidden, xq.b(k::native_q8_1_bytes(hidden, 1) * size_t(j)), hidden, std::min(8, nt - j), stream);
        auto L = k::native_expert_layout(gu_type, d_type, hidden, ff);
        L.swiglu_limit = clamp;
        k::native_expert_grouped(L, reinterpret_cast<const unsigned long long*>(tb), reinterpret_cast<const int32_t*>(tb + o_start),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), reinterpret_cast<const int32_t*>(tb + o_dst),
                                 reinterpret_cast<const int32_t*>(tb + o_tok), G, E, xq.p, scratch.p, out.f(), stream);
        k::ds41_scatter_axpy(out.f(), reinterpret_cast<const int32_t*>(tb + o_tok), reinterpret_cast<const float*>(tb + o_w), E, hidden, sum, stream);
        return true;
    }
    // STRATA_DS41_GPU_ROUTE: the same kernels on a table the GPU built (k::ds41_route_table, capacity cap), added to sum
    void run_device(int gu_type, int d_type, int hidden, int ff, float clamp, const uint8_t* tb, int cap, const float* x, int nt, float* sum,
                    cudaStream_t stream) {
        if (cap > this->cap) {
            this->cap = std::max(cap, 64);
            scratch = Dev(k::native_expert_scratch_bytes(this->cap, ff));
            out = Dev(size_t(this->cap) * hidden * 4);
        }
        if (xq.bytes < k::native_q8_1_bytes(hidden, 1) * size_t(nt)) xq = Dev(k::native_q8_1_bytes(hidden, 1) * size_t(nt) * 2);
        const size_t o_start = size_t(cap) * 8, o_ng = o_start + size_t(cap + 1) * 4, o_dst = o_ng + 4, o_tok = o_dst + size_t(cap) * 4,
                     o_w = o_tok + size_t(cap) * 4;
        for (int j = 0; j < nt; j += 8)
            k::native_quantize_q8_1(x + size_t(j) * hidden, xq.b(k::native_q8_1_bytes(hidden, 1) * size_t(j)), hidden, std::min(8, nt - j), stream);
        auto L = k::native_expert_layout(gu_type, d_type, hidden, ff);
        L.swiglu_limit = clamp;
        k::native_expert_grouped(L, reinterpret_cast<const unsigned long long*>(tb), reinterpret_cast<const int32_t*>(tb + o_start),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), reinterpret_cast<const int32_t*>(tb + o_dst),
                                 reinterpret_cast<const int32_t*>(tb + o_tok), cap, cap, xq.p, scratch.p, out.f(), stream);
        k::ds41_scatter_axpy_dev(out.f(), reinterpret_cast<const int32_t*>(tb + o_tok), reinterpret_cast<const float*>(tb + o_w),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), hidden, sum, stream);
    }
};

struct PeerGpu {
    int dev = 1, hidden = 0, ff = 0;
    cudaStream_t stream = nullptr;
    std::vector<uint8_t*> base;       // [layer] the layer's resident experts
    std::vector<size_t> stride;       // [layer]
    std::vector<std::vector<int>> pos;   // [layer][expert] -> position in base, or -1
    std::vector<int> gu_type, d_type;
    std::vector<float> clamp;
    Dev arena, x, xq8, g_gate, g_up, g_h, g_hq, g_out, out, xq, gu, hf, hq, dout, idx, w, bound, iota;
    std::unique_ptr<strata::prefill::mmq::Context> mmq;
    float* host_out = nullptr;        // pinned [maxT][hidden]
    int rcap = 8192, maxT = 0;
    bool busy = false;
    // STRATA_DS41_PEER_STREAM: per layer, experts streamed over this GPU's own link from the packed pinned host copy
    // (sbase[l], scount[l] experts at stride[l]; spos[l][e] -> their position or -1), through slots (copy stream cs)
    std::vector<const uint8_t*> sbase;
    std::vector<int> scount;
    std::vector<std::vector<int>> spos;
    struct Item { int layer, first, n; };
    std::vector<Item> items;
    std::vector<int> item0;
    std::vector<Dev> slots;
    std::vector<cudaEvent_t> ready, freed;
    cudaStream_t cs = nullptr;
    size_t slot_bytes = 0;
    int issued = 0, computed = 0;
    bool active = false;
    struct Guard { Guard(int d) { cudaSetDevice(d); } ~Guard() { cudaSetDevice(0); } };

    void init(int maxT_, int rcap_) {
        Guard g(dev);
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "peer stream");
        sized(maxT_, rcap_);
    }
    // the buffers sized by the pass width (maxT tokens) and the MMQ row cap (rcap); after the prompt release_prompt()
    // shrinks them (a prefill product splits at rcap, so a smaller one stays correct; a wider pass regrows them)
    void sized(int maxT_, int rcap_) {
        Guard g(dev);
        ck(cudaStreamSynchronize(stream), "peer resize");
        maxT = maxT_;
        rcap = rcap_;
        for (Dev* d : {&x, &out, &xq, &gu, &hf, &hq, &dout, &idx, &w, &iota}) *d = Dev();
        if (host_out) { cudaFreeHost(host_out); host_out = nullptr; }
        x = Dev(size_t(maxT) * hidden * 4);
        out = Dev(size_t(maxT) * hidden * 4);
        xq8 = Dev(k::native_q8_1_bytes(hidden, 1));
        g_gate = Dev(size_t(ff) * 4); g_up = Dev(size_t(ff) * 4); g_h = Dev(size_t(ff) * 4);
        g_hq = Dev(k::native_q8_1_bytes(ff, 1));
        g_out = Dev(size_t(hidden) * 4);
        namespace mq = strata::prefill::mmq;
        if (mq::built()) {
            xq = Dev(mq::q8_bytes(rcap, hidden));
            gu = Dev(size_t(rcap) * 2 * ff * 4);
            hf = Dev(size_t(rcap) * ff * 4);
            hq = Dev(mq::q8_bytes(rcap, ff));
            dout = Dev(size_t(rcap) * hidden * 4);
            idx = Dev(size_t(maxT) * 16 * 4);
            w = Dev(size_t(maxT) * 16 * 4);
            bound = Dev(size_t(4096) * 4);
            iota = Dev((size_t(rcap) + 128) * 4);
            mq::iota(iota.i(), rcap + 128, stream);
            mmq = std::make_unique<mq::Context>();
        }
        void* h = nullptr;
        ck(cudaHostAlloc(&h, size_t(maxT) * hidden * 4, cudaHostAllocPortable), "peer output");
        host_out = static_cast<float*>(h);
        ck(cudaStreamSynchronize(stream), "peer init");
    }
    // STRATA_DS41_REFILL: the prompt's streaming slots freed and the pass buffers shrunk (decode's widths)
    void release_prompt(int maxT_, int rcap_) {
        Guard g(dev);
        ck(cudaStreamSynchronize(stream), "peer release");
        if (cs) { ck(cudaStreamSynchronize(cs), "peer release"); cudaStreamDestroy(cs); cs = nullptr; }
        slots.clear();
        for (auto e : ready) cudaEventDestroy(e);
        for (auto e : freed) cudaEventDestroy(e);
        ready.clear();
        freed.clear();
        items.clear();
        sbase.clear();
        scount.clear();
        spos.clear();
        slot_bytes = 0;
        active = false;
        sized(maxT_, rcap_);
    }
    // grown experts (STRATA_DS41_REFILL, where_ 4): copies of RAM experts in their own arena, computed by a GEMV chain
    // per route after the call's resident products. gpos[l][e] -> slot, gbase[slot] the copy
    Dev garena;
    std::vector<std::vector<int>> gpos;
    std::vector<const uint8_t*> gbase;
    bool grown(int l, int e) const { return !gpos.empty() && gpos[size_t(l)][size_t(e)] >= 0; }
    // the grown routes of a call into out (x uploaded, out zeroed by the caller)
    void grown_routes(int l, int nt, int K, const int* ids, const float* wts) {
        if (gpos.empty()) return;
        // STRATA_DS41_GROWN_GROUPED=1: all of them in one grouped launch (a GEMV chain per route is launch-bound)
        static const bool grouped_on = std::getenv("STRATA_DS41_GROWN_GROUPED") && std::string(std::getenv("STRATA_DS41_GROWN_GROUPED")) == "1";
        if (grouped_on && k::native_expert_supported(gu_type[size_t(l)], d_type[size_t(l)], hidden, ff)) {
            std::map<int, std::vector<std::pair<int, float>>> by;
            for (int i = 0; i < nt * K; ++i)
                if (ids[i] >= 0 && grown(l, ids[i])) by[ids[i]].push_back({i / K, wts[i]});
            std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
            for (auto& [e, r] : by) groups.push_back({gbase[size_t(gpos[size_t(l)][size_t(e)])], std::move(r)});
            grown_runner.run(gu_type[size_t(l)], d_type[size_t(l)], hidden, ff, clamp[size_t(l)], groups, x.f(), nt, out.f(), stream);
            return;
        }
        const size_t gb = ggml_row_size(ggml_type(gu_type[size_t(l)]), hidden) * size_t(ff);
        for (int t = 0; t < nt; ++t) {
            bool quantized = false;
            for (int kk = 0; kk < K; ++kk) {
                const int e = ids[t * K + kk];
                if (e < 0 || !grown(l, e)) continue;
                if (!quantized) { k::native_quantize_q8_1(x.f(size_t(t) * hidden), xq8.p, hidden, 1, stream); quantized = true; }
                const uint8_t* b = gbase[size_t(gpos[size_t(l)][size_t(e)])];
                k::native_mmvq(gu_type[size_t(l)], b, xq8.p, g_gate.f(), hidden, ff, 1, stream);
                k::native_mmvq(gu_type[size_t(l)], b + gb, xq8.p, g_up.f(), hidden, ff, 1, stream);
                k::glm_swiglu(g_gate.f(), g_up.f(), g_h.f(), ff, clamp[size_t(l)], stream);
                k::native_quantize_q8_1(g_h.f(), g_hq.p, ff, 1, stream);
                k::native_mmvq(d_type[size_t(l)], b + 2 * gb, g_hq.p, g_out.f(), ff, hidden, 1, stream);
                k::ds41_axpy(g_out.f(), wts[t * K + kk], out.f(size_t(t) * hidden), hidden, stream);
            }
        }
    }
    int grown_count(int l, int nt, int K, const int* ids) const {
        if (gpos.empty()) return 0;
        int n = 0;
        for (int i = 0; i < nt * K; ++i) if (ids[i] >= 0 && grown(l, ids[i])) ++n;
        return n;
    }
    void ensure(int nt) { if (nt > maxT) sized(std::max(nt, 2 * maxT), rcap); }
    void stream_setup(int nslots, size_t bytes) {
        Guard g(dev);
        slot_bytes = bytes;
        ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "peer copy stream");
        for (int i = 0; i < nslots; ++i) {
            slots.emplace_back(bytes);
            cudaEvent_t a, b;
            ck(cudaEventCreateWithFlags(&a, cudaEventDisableTiming), "peer event");
            ck(cudaEventCreateWithFlags(&b, cudaEventDisableTiming), "peer event");
            ready.push_back(a);
            freed.push_back(b);
        }
    }
    // a streamed pass starts: the items of every layer, the first slots' copies issued now
    void stream_begin() {
        if (slots.empty()) return;
        Guard g(dev);
        ck(cudaStreamSynchronize(cs), "peer copies");
        ck(cudaStreamSynchronize(stream), "peer");
        items.clear();
        item0.assign(sbase.size() + 1, 0);
        for (size_t l = 0; l < sbase.size(); ++l) {
            item0[l] = int(items.size());
            const int per = int(std::max<size_t>(1, slot_bytes / stride[l]));
            for (int e = 0; e < scount[l]; e += per) items.push_back({int(l), e, std::min(per, scount[l] - e)});
        }
        item0[sbase.size()] = int(items.size());
        issued = computed = 0;
        active = true;
        issue(int(items.size()));
    }
    void stream_end() { active = false; }
    // copies up to item `limit` (exclusive), each once its slot's previous item has been computed (enqueued)
    void issue(int limit) {
        const int NS = int(slots.size());
        while (issued < int(items.size()) && issued < limit && issued < computed + NS) {
            const Item& it = items[size_t(issued)];
            const int k = issued % NS;
            if (issued >= NS) ck(cudaStreamWaitEvent(cs, freed[size_t(k)], 0), "peer slot free");
            ck(cudaMemcpyAsync(slots[size_t(k)].p, sbase[size_t(it.layer)] + size_t(it.first) * stride[size_t(it.layer)],
                               size_t(it.n) * stride[size_t(it.layer)], cudaMemcpyHostToDevice, cs), "peer stream copy");
            ck(cudaEventRecord(ready[size_t(k)], cs), "peer slot ready");
            ++issued;
        }
    }
    size_t available_bytes() const {
        Guard g(dev);
        size_t f = 0, t = 0;
        ck(cudaMemGetInfo(&f, &t), "peer memory info");
#if defined(__HIP_PLATFORM_AMD__)
        f = std::min(f, physical_vram_free());
#endif
        return f;
    }
    // STRATA_DS41_DECODE_STREAM: experts of the host's pinned copy computed here in this call (copied in, then a GEMV
    // chain per route), set by the caller before decode()/prefill(): {expert bytes on the host, token, weight}, grouped by expert
    uint8_t* staging = nullptr;   // pinned, portable: prefill's row tables
    size_t staging_bytes = 0;
    const int32_t* idx_dev = nullptr;   // a call's tables in `tab`
    const float* w_dev = nullptr;
    struct Extra { const uint8_t* host; int token; float w; };
    std::vector<Extra> extra;
    std::vector<Dev> xslots;
    // runs the extras into out (uploading z and zeroing out first when nothing else did); returns whether any ran
    bool extras(int l, int nt, const float* z, bool started) {
        if (extra.empty()) return false;
        if (!started) {
            ck(cudaMemcpyAsync(x.p, z, size_t(nt) * hidden * 4, cudaMemcpyHostToDevice, stream), "peer z");
            ck(cudaMemsetAsync(out.p, 0, size_t(nt) * hidden * 4, stream), "peer sum");
        }
        const size_t gb = ggml_row_size(ggml_type(gu_type[size_t(l)]), hidden) * size_t(ff);
        // STRATA_DS41_EXTRAS_GROUPED=1: the copied experts' routes in one grouped launch (S27 with STRATA_EXPERT_V3)
        // instead of a GEMV chain per route
        static const bool grouped_on = std::getenv("STRATA_DS41_EXTRAS_GROUPED") && std::string(std::getenv("STRATA_DS41_EXTRAS_GROUPED")) == "1";
        if (grouped_on && k::native_expert_supported(gu_type[size_t(l)], d_type[size_t(l)], hidden, ff)) {
            std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
            const uint8_t* last = nullptr;
            int slot = -1;
            for (const Extra& e : extra) {
                if (e.host != last) {
                    last = e.host;
                    if (++slot >= int(xslots.size())) xslots.emplace_back(stride[size_t(l)] > 0 ? *std::max_element(stride.begin(), stride.end()) : 0);
                    ck(cudaMemcpyAsync(xslots[size_t(slot)].p, e.host, stride[size_t(l)], cudaMemcpyHostToDevice, stream), "peer streamed expert");
                    groups.push_back({xslots[size_t(slot)].b(), {}});
                }
                groups.back().second.push_back({e.token, e.w});
            }
            extras_runner.run(gu_type[size_t(l)], d_type[size_t(l)], hidden, ff, clamp[size_t(l)], groups, x.f(), nt, out.f(), stream);
            extra.clear();
            return true;
        }
        int slot = -1;
        const uint8_t* last = nullptr;
        for (const Extra& e : extra) {
            if (e.host != last) {
                last = e.host;
                if (++slot >= int(xslots.size())) xslots.emplace_back(stride[size_t(l)] > 0 ? *std::max_element(stride.begin(), stride.end()) : 0);
                ck(cudaMemcpyAsync(xslots[size_t(slot)].p, e.host, stride[size_t(l)], cudaMemcpyHostToDevice, stream), "peer streamed expert");
            }
            const uint8_t* b = xslots[size_t(slot)].b();
            k::native_quantize_q8_1(x.f(size_t(e.token) * hidden), xq8.p, hidden, 1, stream);
            k::native_mmvq(gu_type[size_t(l)], b, xq8.p, g_gate.f(), hidden, ff, 1, stream);
            k::native_mmvq(gu_type[size_t(l)], b + gb, xq8.p, g_up.f(), hidden, ff, 1, stream);
            k::glm_swiglu(g_gate.f(), g_up.f(), g_h.f(), ff, clamp[size_t(l)], stream);
            k::native_quantize_q8_1(g_h.f(), g_hq.p, ff, 1, stream);
            k::native_mmvq(d_type[size_t(l)], b + 2 * gb, g_hq.p, g_out.f(), ff, hidden, 1, stream);
            k::ds41_axpy(g_out.f(), e.w, out.f(size_t(e.token) * hidden), hidden, stream);
        }
        extra.clear();
        return true;
    }
    GroupedRunner grouped_runner, grown_runner, extras_runner;
    bool grouped(int l, int nt, int K, const int* ids, const float* wts, const float* z) {
        Guard g(dev);
        std::map<int, std::vector<std::pair<int, float>>> by;
        ensure(nt);
        for (int i = 0; i < nt * K; ++i) {
            const int e = ids[i];
            if (e >= 0 && (pos[size_t(l)][size_t(e)] >= 0 || grown(l, e))) by[e].push_back({i / K, wts[i]});
        }
        bool any = !by.empty() || !extra.empty();
        if (!any) { busy = false; return false; }
        mark_begin();
        ck(cudaMemcpyAsync(x.p, z, size_t(nt) * hidden * 4, cudaMemcpyHostToDevice, stream), "peer z");
        ck(cudaMemsetAsync(out.p, 0, size_t(nt) * hidden * 4, stream), "peer sum");
        std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
        for (auto& [e, r] : by)
            groups.push_back({pos[size_t(l)][size_t(e)] >= 0 ? base[size_t(l)] + size_t(pos[size_t(l)][size_t(e)]) * stride[size_t(l)]
                                                            : gbase[size_t(gpos[size_t(l)][size_t(e)])],
                              std::move(r)});
        grouped_runner.run(gu_type[size_t(l)], d_type[size_t(l)], hidden, ff, clamp[size_t(l)], groups, x.f(), nt, out.f(), stream);
        if (!extra.empty()) extras(l, nt, z, true);
        result(nt);
        busy = true;
        return true;
    }
    // decode: the routes resident here (ids[t K + k] with pos >= 0), z host [nt][hidden]; result into host_out (zeroed
    // first); returns whether any ran (then `collect` waits)
    bool decode(int l, int nt, int K, const int* ids, const float* wts, const float* z) {
        Guard g(dev);
        ensure(nt);
        bool any = false;
        for (int t = 0; t < nt; ++t) {
            bool quantized = false;
            for (int kk = 0; kk < K; ++kk) {
                const int e = ids[t * K + kk];
                if (e < 0 || (pos[size_t(l)][size_t(e)] < 0 && !grown(l, e))) continue;
                if (!any) {
                    mark_begin();
                    ck(cudaMemcpyAsync(x.p, z, size_t(nt) * hidden * 4, cudaMemcpyHostToDevice, stream), "peer z");
                    ck(cudaMemsetAsync(out.p, 0, size_t(nt) * hidden * 4, stream), "peer sum");
                    any = true;
                }
                if (!quantized) { k::native_quantize_q8_1(x.f(size_t(t) * hidden), xq8.p, hidden, 1, stream); quantized = true; }
                const uint8_t* b = pos[size_t(l)][size_t(e)] >= 0 ? base[size_t(l)] + size_t(pos[size_t(l)][size_t(e)]) * stride[size_t(l)]
                                                                  : gbase[size_t(gpos[size_t(l)][size_t(e)])];
                const size_t gb = ggml_row_size(ggml_type(gu_type[size_t(l)]), hidden) * size_t(ff);
                k::native_mmvq(gu_type[size_t(l)], b, xq8.p, g_gate.f(), hidden, ff, 1, stream);
                k::native_mmvq(gu_type[size_t(l)], b + gb, xq8.p, g_up.f(), hidden, ff, 1, stream);
                k::glm_swiglu(g_gate.f(), g_up.f(), g_h.f(), ff, clamp[size_t(l)], stream);
                k::native_quantize_q8_1(g_h.f(), g_hq.p, ff, 1, stream);
                k::native_mmvq(d_type[size_t(l)], b + 2 * gb, g_hq.p, g_out.f(), ff, hidden, 1, stream);
                k::ds41_axpy(g_out.f(), wts[t * K + kk], out.f(size_t(t) * hidden), hidden, stream);
            }
        }
        if (!extra.empty()) {
            if (!any) mark_begin();
            any = extras(l, nt, z, any) || any;
        }
        if (any) result(nt);
        busy = any;
        return any;
    }
    // a prompt pass: every route of layer l resident here, and in a streamed pass those of the experts streamed here,
    // by MMQ; result into host_out
    // per layer: the resident experts by position (the row order of a prefill product); per call: the routes by expert
    std::vector<std::vector<int>> order_of;
    std::vector<std::vector<int>> tok;
    std::vector<std::vector<float>> wt;
    std::vector<int> touched, order, hidx, hb, row0;
    std::vector<float> hw;
    Dev tab;   // idx | w | bounds of a call, one upload
    bool prefill(int l, int nt, int K, const int* ids, const float* wts, const float* z) {
        namespace mq = strata::prefill::mmq;
        // a layer whose expert types MMQ lacks (IQ1_M without third_party/ggml/patches/apply_mmq_iq1m.py): GEMV chains
        if (!mq::supported(gu_type[size_t(l)]) || !mq::supported(d_type[size_t(l)])) {
            if (active && !spos.empty() && scount[size_t(l)] > 0)
                throw std::runtime_error("ds41: the peer's streamed experts need MMQ for layer " + std::to_string(l) + "'s types");
            return decode(l, nt, K, ids, wts, z);
        }
        Guard g(dev);
        ensure(nt);
        const int E = int(pos[size_t(l)].size());
        const bool streamed = active && !spos.empty() && scount[size_t(l)] > 0;
        if (tok.size() != size_t(E)) { tok.assign(size_t(E), {}); wt.assign(size_t(E), {}); }
        for (int e : touched) { tok[size_t(e)].clear(); wt[size_t(e)].clear(); }
        touched.clear();
        // rows by segment: the resident experts by position, then the streamed ones by position
        if (order_of.size() != pos.size()) order_of.assign(pos.size(), {});
        auto& resident = order_of[size_t(l)];
        if (resident.empty()) {
            for (int e = 0; e < E; ++e) if (pos[size_t(l)][size_t(e)] >= 0) resident.push_back(e);
            std::sort(resident.begin(), resident.end(), [&](int a, int b2) { return pos[size_t(l)][size_t(a)] < pos[size_t(l)][size_t(b2)]; });
        }
        order = resident;
        const int R = int(order.size());
        if (streamed) {
            std::vector<int> so(size_t(scount[size_t(l)]), -1);
            for (int e = 0; e < E; ++e) if (spos[size_t(l)][size_t(e)] >= 0) so[size_t(spos[size_t(l)][size_t(e)])] = e;
            order.insert(order.end(), so.begin(), so.end());
        }
        int routes = 0;
        for (int i = 0; i < nt * K; ++i) {
            const int e = ids[i];
            if (e < 0 || (pos[size_t(l)][size_t(e)] < 0 && !(streamed && spos[size_t(l)][size_t(e)] >= 0))) continue;
            if (tok[size_t(e)].empty()) touched.push_back(e);
            tok[size_t(e)].push_back(i / K);
            wt[size_t(e)].push_back(wts[i]);
            ++routes;
        }
        hidx.clear(); hb.clear(); hw.clear();
        row0.assign(order.size() + 1, 0);
        for (size_t q = 0; q < order.size(); ++q) {
            row0[q + 1] = row0[q] + int(tok[size_t(order[q])].size());
            hidx.insert(hidx.end(), tok[size_t(order[q])].begin(), tok[size_t(order[q])].end());
            hw.insert(hw.end(), wt[size_t(order[q])].begin(), wt[size_t(order[q])].end());
        }
        // products: runs within the row cap inside one segment (item -1: the resident area, else a streamed item)
        struct Prod { int item, j0, j1, first, rows, b0, mx; };
        std::vector<Prod> prods;
        // STRATA_DS41_MMQ_COMPACT=1: products over runs of consecutive experts that have rows only (the empty ones
        // between are skipped), instead of one product spanning every resident expert
        static const bool compact = std::getenv("STRATA_DS41_MMQ_COMPACT") && std::string(std::getenv("STRATA_DS41_MMQ_COMPACT")) == "1";
        auto group = [&](int item, int base, int count) {
            for (int j0 = base; j0 < base + count;) {
                if (compact && tok[size_t(order[size_t(j0)])].empty()) { ++j0; continue; }
                int j1 = j0, r = 0, mx = 0;
                while (j1 < base + count && (j1 == j0 || r + int(tok[size_t(order[size_t(j1)])].size()) <= rcap) &&
                       !(compact && tok[size_t(order[size_t(j1)])].empty())) {
                    r += int(tok[size_t(order[size_t(j1)])].size());
                    mx = std::max(mx, int(tok[size_t(order[size_t(j1)])].size()));
                    ++j1;
                }
                Prod pr{item, j0 - base, j1 - base, row0[size_t(j0)], r, int(hb.size()), mx};
                for (int j = j0; j <= j1; ++j) hb.push_back(row0[size_t(j)] - pr.first);
                prods.push_back(pr);
                j0 = j1;
            }
        };
        group(-1, 0, R);
        if (streamed)
            for (int i = item0[size_t(l)]; i < item0[size_t(l) + 1]; ++i) group(i, R + items[size_t(i)].first, items[size_t(i)].n);
        const int groutes = active ? 0 : grown_count(l, nt, K, ids);   // (a streamed pass streams them like RAM experts)
        busy = routes + groutes > 0;
        const int32_t* bounds_dev = bound.i();
        if (busy) {
            mark_begin();
            // the tables through a pinned (portable) staging buffer (a pageable upload blocks until it is staged), in one
            // upload: idx | w | bounds, each 256-byte aligned
            const size_t a_idx = 0, a_w = (hidx.size() * 4 + 255) & ~size_t(255), a_b = a_w + ((hw.size() * 4 + 255) & ~size_t(255));
            const size_t need = a_b + hb.size() * 4;
            if (need > staging_bytes) {
                if (staging) cudaFreeHost(staging);
                ck(cudaHostAlloc(reinterpret_cast<void**>(&staging), need * 2, cudaHostAllocPortable), "peer staging");
                staging_bytes = need * 2;
            }
            if (need > tab.bytes) tab = Dev(need * 2);
            // the previous call's uploads: done before this call in the threaded mode (GPU 0 waited on `done`, then the
            // decoder synchronized before posting); the synchronous mode waits here
            if (!threaded) ck(cudaStreamSynchronize(stream), "peer staging");
            uint8_t* st = staging;
            std::memcpy(st + a_idx, hidx.data(), hidx.size() * 4);
            std::memcpy(st + a_w, hw.data(), hw.size() * 4);
            std::memcpy(st + a_b, hb.data(), hb.size() * 4);
            ck(cudaMemcpyAsync(x.p, z, size_t(nt) * hidden * 4, cudaMemcpyHostToDevice, stream), "peer z");
            ck(cudaMemcpyAsync(tab.p, st, need, cudaMemcpyHostToDevice, stream), "peer tables");
            ck(cudaMemsetAsync(out.p, 0, size_t(nt) * hidden * 4, stream), "peer sum");
            split_mark(0);
            idx_dev = reinterpret_cast<const int32_t*>(tab.b(a_idx));
            w_dev = reinterpret_cast<const float*>(tab.b(a_w));
            bounds_dev = reinterpret_cast<const int32_t*>(tab.b(a_b));
        }
        const size_t gb = ggml_row_size(ggml_type(gu_type[size_t(l)]), hidden) * size_t(ff);
        const int NS = int(slots.size());
        int at = -1;   // the streamed item whose slot the products read
        auto done = [&](int i) {   // its slot is free once its products are enqueued
            ck(cudaEventRecord(freed[size_t(i % NS)], stream), "peer slot freed");
            computed = i + 1;
            issue(computed + NS);
        };
        for (const Prod& p : prods) {
            if (p.item >= 0 && p.item != at) {
                if (at >= 0) done(at);
                issue(p.item + 1);
                ck(cudaStreamWaitEvent(stream, ready[size_t(p.item % NS)], 0), "peer slot ready");
                at = p.item;
            }
            if (!p.rows) continue;
            const uint8_t* st = p.item < 0 ? base[size_t(l)] + size_t(p.j0) * stride[size_t(l)]
                                           : static_cast<const uint8_t*>(slots[size_t(p.item % NS)].p) + size_t(p.j0) * stride[size_t(l)];
            const int32_t* rtok = idx_dev + p.first;
            mq::quantize(x.f(), rtok, xq.p, gu_type[size_t(l)], hidden, hidden, p.rows, stream);
            mq::Product a;
            a.w = st; a.type = gu_type[size_t(l)]; a.w_rows = 2 * ff; a.w_cols = hidden; a.expert_bytes = stride[size_t(l)];
            a.n = p.j1 - p.j0; a.xq = xq.p; a.bounds = bounds_dev + p.b0; a.ids = iota.i(); a.total_rows = p.rows; a.max_rows = p.mx;
            a.dst = gu.f(); a.ld_dst = 2 * ff; a.dst_rows = p.rows;
            mmq->run(a, stream);
            split_mark(1);
            mq::swiglu(gu.f(), hf.f(), p.rows, ff, false, stream, clamp[size_t(l)]);
            mq::quantize(hf.f(), nullptr, hq.p, d_type[size_t(l)], ff, ff, p.rows, stream);
            mq::Product d = a;
            d.w = st + 2 * gb; d.type = d_type[size_t(l)]; d.w_rows = hidden; d.w_cols = ff; d.xq = hq.p; d.dst = dout.f(); d.ld_dst = hidden;
            split_mark(2);
            mmq->run(d, stream);
            split_mark(3);
            k::ds41_scatter_axpy(dout.f(), rtok, w_dev + p.first, p.rows, hidden, out.f(), stream);
        }
        if (at >= 0) done(at);
        if (groutes) grown_routes(l, nt, K, ids, wts);
        if (!extra.empty()) {
            if (!busy) mark_begin();
            busy = extras(l, nt, z, busy) || busy;
        }
        if (busy) result(nt);
        return busy;
    }
    // a call's result: D2H into host_out (to_host), or left in `out` for the worker thread's copy to GPU 0
    bool to_host = true;
    void result(int nt) {
        if (to_host) ck(cudaMemcpyAsync(host_out, out.p, size_t(nt) * hidden * 4, cudaMemcpyDeviceToHost, stream), "peer result");
        mark_end();
    }
    // STRATA_DS41_DECODE_TIMING: the peer's GPU time per call (from its first upload to its result)
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    bool timing = false;
    double gpu_ms = 0;
    void mark_begin() {
        if (!timing) return;
        if (!t0) { cudaEventCreate(&t0); cudaEventCreate(&t1); }
        cudaEventRecord(t0, stream);
    }
    void mark_end() { if (timing && t1) cudaEventRecord(t1, stream); }   // after a call's last operation
    // DECODE_TIMING: a prefill call's GPU time split [begin, uploads done, gate/up product, swiglu + quantize, down product,
    // end] (the first product of the call), summed by collect into split_ms
    cudaEvent_t se[4] = {};
    bool split_on = false;
    double split_ms[5] = {};
    void split_mark(int i) {
        if (!timing) return;
        if (!se[i]) cudaEventCreate(&se[i]);
        if (i == 0) split_on = true;
        if (split_on) cudaEventRecord(se[i], stream);
    }
    void collect() {   // the last call's result is in host_out
        if (!busy) return;
        Guard g(dev);
        ck(cudaStreamSynchronize(stream), "peer");
        if (timing) {
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            gpu_ms += ms;
            if (split_on) {
                cudaEvent_t pts[6] = {t0, se[0], se[1], se[2], se[3], t1};
                for (int i = 0; i < 5; ++i) {
                    float m = 0;
                    if (cudaEventElapsedTime(&m, pts[i], pts[i + 1]) == cudaSuccess) split_ms[i] += m;
                    else (void)cudaGetLastError();
                }
                split_on = false;
            }
        }
        busy = false;
    }
    // STRATA_DS41_PEER_THREAD=1: a worker thread enqueues this GPU's work. The decoder posts a layer's routes right
    // after routing (before the CPU lane and GPU 0's experts); the thread builds the tables, enqueues decode / prefill
    // / grouped, copies the result into the decoder's GPU 0 buffer (`dst`) and records `done` on this stream, which
    // GPU 0's stream waits on. The host waits only for the enqueue (wait_enqueued), never for this GPU.
    struct Job { int l, nt, K, mode; const int* ids; const float* wts; const float* z; void* dst; };   // mode: 0 decode, 1 prefill, 2 grouped
    bool threaded = false;
    std::thread worker;
    std::mutex jm;
    std::condition_variable jcv;
    Job job{};
    int posted = 0, enqueued = 0;
    bool stop = false;
    cudaEvent_t done = nullptr;
    // STRATA_DS41_PEER_THREAD_CPU=n: the worker pinned to CPU n (a core the lane does not use: --threads one below the
    // core count leaves the last one), else wherever the scheduler puts it (a lane core's sibling slows that lane thread)
    void start_thread() {
        threaded = true;
        to_host = false;
        { Guard g(dev); ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "peer done"); }
        const int cpu = std::getenv("STRATA_DS41_PEER_THREAD_CPU") ? std::atoi(std::getenv("STRATA_DS41_PEER_THREAD_CPU")) : -1;
        worker = std::thread([this, cpu] {
            if (cpu >= 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpu, &set);
                pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            }
            run_jobs();
        });
    }
    void run_jobs() {
        int seen = 0;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lock(jm);
                jcv.wait(lock, [&] { return stop || posted > seen; });
                if (stop) return;
                j = job;
                seen = posted;
            }
            // the previous job's time (its events are complete: GPU 0 waited on `done`, then the decoder synchronized)
            if (timing && busy && t0) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess) gpu_ms += ms;
                else (void)cudaGetLastError();
            }
            try {
                const bool ran = j.mode == 2 ? grouped(j.l, j.nt, j.K, j.ids, j.wts, j.z)
                               : j.mode == 1 ? prefill(j.l, j.nt, j.K, j.ids, j.wts, j.z)
                                             : decode(j.l, j.nt, j.K, j.ids, j.wts, j.z);
                if (ran) {
                    Guard g(dev);
                    ck(cudaMemcpyPeerAsync(j.dst, 0, out.p, dev, size_t(j.nt) * hidden * 4, stream), "peer result copy");
                    ck(cudaEventRecord(done, stream), "peer done");
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(jm);
                error = std::current_exception();   // rethrown by wait_enqueued
                busy = false;
            }
            { std::lock_guard<std::mutex> lock(jm); enqueued = seen; }
            jcv.notify_all();
        }
    }
    void post(const Job& j) {
        { std::lock_guard<std::mutex> lock(jm); job = j; ++posted; }
        jcv.notify_all();
    }
    std::exception_ptr error;
    void wait_enqueued() {   // the posted job is on this GPU's stream; `busy` then says whether it ran
        std::unique_lock<std::mutex> lock(jm);
        jcv.wait(lock, [&] { return enqueued == posted; });
        if (error) { auto e = error; error = nullptr; std::rethrow_exception(e); }
    }
    ~PeerGpu() {
        if (worker.joinable()) {
            { std::lock_guard<std::mutex> lock(jm); stop = true; }
            jcv.notify_all();
            worker.join();
        }
        if (done) cudaEventDestroy(done);
    }
};

int engram_profile_run(const std::string& model, const std::string& streams, const std::string& out);

class Decoder {
public:
    friend int engram_profile_run(const std::string& model, const std::string& streams, const std::string& out);
    Decoder(const std::string& path, int context, int threads, int chunk = 0, const Decoder* borrow = nullptr)
        : borrow_(borrow), art_(path), cfg_(Config::read(*art_.metadata(), art_.descriptor(), [this](const std::string& n) {
              try { art_.at(n); return true; } catch (...) { return false; }
          }, [this](const std::string& n) { return tensor_ints(art_.at(n)); })), context_(context) {
        if (chunk > 0) chunk_ = chunk;
        const auto& c = cfg_;
        if (context_ < 1) throw std::invalid_argument("ds41: context must be positive");
        strata::ds41::validate_owned_stream_settings();
        // STRATA_DS41_PREFILL_CHUNK: tokens per forward pass of the prompt (default 256)
        if (const char* v = std::getenv("STRATA_DS41_PREFILL_CHUNK"); v && chunk <= 0) chunk_ = std::max(1, std::atoi(v));
        ring_ = c.window + chunk_;
        stream_ = make_stream(0);   // the default (blocking) flags
        // STRATA_DS41_MULTI_ROWS=1|2|4: rows per block of the exact multi-column dense GEMV (every choice is bitwise the
        // single-column result, mmvq_multi_parity; speed only)
        if (const char* rows = std::getenv("STRATA_DS41_MULTI_ROWS")) k::native_mmvq_set_multi_rows(std::atoi(rows));
        ck(cudaEventCreateWithFlags(&router_ev_, cudaEventDisableTiming), "router event");
        // STRATA_DS41_IQ2XXS_EXACT (1): IQ2_XXS GPU dots scale in floats like the CPU reference (the truncating
        // integer scale cost +2.5 % PPL on the IQ2_XXSE4 pack)
        k::native_mmvq_set_iq2xxs_exact(!(std::getenv("STRATA_DS41_IQ2XXS_EXACT") && std::string(std::getenv("STRATA_DS41_IQ2XXS_EXACT")) == "0"));
        // STRATA_DS41_REMOTE_TP=HOST:PORT (this node's address the worker connects to): the worker takes FFN rows
        // [fl_, ff) of every routed expert, STRATA_DS41_REMOTE_SHARE of them (default 2/9, whole 256-row blocks)
        fl_ = c.ff;
        ram_copy_ = std::getenv("STRATA_DS41_RAM_COPY") && std::string(std::getenv("STRATA_DS41_RAM_COPY")) == "1";
        stage_cpu_expert_ = strata_env_is_one("STRATA_DS41_STAGE_CPU_EXPERT");
        stage_cpu_order_ = strata_env_is_one("STRATA_DS41_STAGE_CPU_ORDER");
        stage_cpu_collect_ = strata_env_is_one("STRATA_DS41_STAGE_CPU_COLLECT");
        stage_cpu_host_act_ = strata_env_is_one("STRATA_DS41_STAGE_CPU_HOST_ACT");
        stage_cpu_batch_act_ = strata_env_is_one("STRATA_DS41_STAGE_BATCH_ACT");
        stage_cpu_timing_ = strata_env_is_one("STRATA_DS41_STAGE_TIMING");
        if (const char* slots = std::getenv("STRATA_DS41_STAGE_CACHE_SLOTS")) {
            char* end = nullptr;
            const long n = std::strtol(slots, &end, 10);
            if (!*slots || *end || n < 0 || n > 192)
                throw std::invalid_argument("ds41: staging cache slots must be 0 through 192");
            stage_cache_slots_requested_ = int(n);
        }
        if (const char* quota = std::getenv("STRATA_DS41_STAGE_CPU_QUOTA")) {
            if (std::strlen(quota) != 1 || quota[0] < '1' || quota[0] > '4')
                throw std::invalid_argument("ds41: CPU staging quota must be 1 through 4");
            stage_cpu_quota_ = quota[0] - '0';
        }
        if (stage_cpu_order_ && (!stage_cpu_expert_ || strata_env_is_one("STRATA_IQ1S_FAST")))
            throw std::invalid_argument("ds41: CPU-order staging requires checked staging and default IQ1_S arithmetic");
        if (stage_cpu_collect_ && !stage_cpu_order_)
            throw std::invalid_argument("ds41: CPU-order collection requires CPU-order staging");
        if (stage_cpu_host_act_ && !stage_cpu_collect_)
            throw std::invalid_argument("ds41: host activation requires CPU-order collection");
        if (stage_cpu_batch_act_ && (!stage_cpu_host_act_ || stage_cpu_quota_ != 1 || strata_env_is_one("STRATA_DS41_STAGE_CPU_ORDER_CHECK")))
            throw std::invalid_argument("ds41: batched host activation requires one staged expert and disables the per-layer diagnostic");
        if (stage_cpu_quota_ > 1 && !stage_cpu_collect_)
            throw std::invalid_argument("ds41: multiple staged experts require ordered collection");
        if (stage_cache_slots_requested_ && (!stage_cpu_collect_ || stage_cache_slots_requested_ < stage_cpu_quota_))
            throw std::invalid_argument("ds41: staging cache requires ordered collection and at least one slot per staged expert");
        if (strata_env_is_one("STRATA_DS41_STAGE_CACHE_FREQUENCY")) {
            if (!stage_cache_slots_requested_ || c.layers <= 0 || c.experts <= 0 ||
                size_t(c.layers) > (size_t(1) << 20) / size_t(c.experts))
                throw std::invalid_argument("ds41: staging frequency requires a bounded staging cache and model geometry");
            stage_cache_frequency_.resize(size_t(c.layers) * size_t(c.experts), 0);
        }
        if (stage_cpu_expert_) {
            const char* rank = std::getenv("STRATA_DS41_DEFER_RANK");
            const char* promote = std::getenv("STRATA_DS41_TIER_PROMOTE");
            if (!strata_env_is_one("STRATA_DS41_VERIFY_EXACT") || !strata_env_is_one("STRATA_DS41_STAGE_CPU_CHECK") ||
                ram_copy_ || strata_env_is_one("STRATA_DS41_GPU_ROUTE") || strata_env_is_one("STRATA_DS41_DECODE_GROUPED") ||
                strata_env_is_one("STRATA_DS41_DEFER_ALL") || (rank && std::string(rank) != "0") ||
                !promote || std::string(promote) != "0" || !std::getenv("STRATA_DS41_EXPERT_PROFILE") ||
                std::getenv("STRATA_DS41_CPU_TOPK") || std::getenv("STRATA_DS41_CPU_TRIM") ||
                std::getenv("STRATA_DS41_GPU_SUBST") || std::getenv("STRATA_DS41_DECODE_STREAM"))
                throw std::invalid_argument("ds41: staged CPU expert requires checked exact-mode CPU routing with no lossy or packed routes");
        }
        if (const char* addr = std::getenv("STRATA_DS41_REMOTE_TP")) {
            remote_addr_ = addr;
            const char* v = std::getenv("STRATA_DS41_REMOTE_SHARE");
            const std::string text = v ? v : "2/9";
            const auto slash = text.find('/');
            const double share = slash == std::string::npos ? std::stod(text) : std::stod(text.substr(0, slash)) / std::stod(text.substr(slash + 1));
            const int block = 256, rows = std::clamp(int(std::lround(share * c.ff / block)) * block, block, c.ff - block);
            if (c.ff % block) throw std::runtime_error("ds41 remote: the expert width is not whole 256-row blocks");
            fl_ = c.ff - rows;
        }
        // every non-expert tensor except the two host tables goes to the GPU as stored
        size_t bytes = 0;
        for (int l = 0; l < c.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            for (const char* n : {"attn_norm.weight", "attn_sinks.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
                                  "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight", "attn_output_a.weight",
                                  "attn_output_b.weight", "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                                  "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight", "ffn_gate_inp.weight",
                                  "ffn_norm.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
                bytes += upload(p + n);
            if (c.kv_src[size_t(l)] == l) {
                for (const char* n : {"attn_compressor_kv.weight", "attn_compressor_norm.weight", "indexer.attn_k.weight",
                                      "indexer.k_norm.weight"})
                    bytes += upload(p + n);
                if (c.ratio[size_t(l)] > 1) bytes += upload(p + "attn_compressor_gate.weight");
            }
            if (c.topk_src[size_t(l)] == l)
                for (const char* n : {"indexer.attn_q_b.weight", "indexer.proj.weight"}) bytes += upload(p + n);
            if (c.engram_index(l) >= 0)
                if (!c.engram_per_layer) {
                    for (const char* n : {"engram_kv.weight", "engram_gate.weight"}) bytes += upload(p + n);
                } else {   // #28696: engram_wkv is engram_kv; the gate is engram_q * engram_k (only their product is used)
                    bytes += upload(p + "engram_wkv.weight", p + "engram_kv.weight");
                    const auto &q = art_.at(p + "engram_q.weight"), &kk = art_.at(p + "engram_k.weight");
                    const int64_t n = int64_t(c.hidden) * c.streams;
                    if (int64_t(q.tensor->shape[0]) * int64_t(q.tensor->shape[1]) != n || q.tensor->shape != kk.tensor->shape)
                        throw std::runtime_error("ds41: Engram q/k geometry");
                    auto gq = to_float(int(q.tensor->type), q.data(), n), gk = to_float(int(kk.tensor->type), kk.data(), n);
                    for (int64_t i = 0; i < n; ++i) gq[size_t(i)] *= gk[size_t(i)];
                    bytes += upload_f32(p + "engram_gate.weight", gq, c.hidden, c.streams);
                }
            // the router bias stays on the host: the top-k runs there
            const auto& b = art_.at(p + "exp_probs_b.bias");
            bias_.push_back(to_float(int(b.tensor->type), b.data(), c.experts));
            // experts: mapped in place, one NativeFmt per layer
            LaneLayer lane;
            const auto& g = art_.at(p + "ffn_gate_exps.weight");
            const auto& u = art_.at(p + "ffn_up_exps.weight");
            const auto& d = art_.at(p + "ffn_down_exps.weight");
            if (g.tensor->type != u.tensor->type) throw std::runtime_error("ds41: gate and up experts differ in type");
            std::string err;
            if (!cpu::native_fmt(int(g.tensor->type), int(d.tensor->type), c.hidden, c.ff, lane.f, err))
                throw std::runtime_error("ds41: " + p + " experts: " + err);
            if (stage_cpu_expert_) {
                if (c.hidden <= 0 || c.hidden > 32768 || c.ff <= 0 || c.ff > 32768 ||
                    c.experts <= 0 || c.experts > 4096 || c.layers <= 0 || c.layers > 256 ||
                    !lane.f.gu_row || !lane.f.d_row ||
                    size_t(c.ff) > SIZE_MAX / lane.f.gu_row || size_t(c.hidden) > SIZE_MAX / lane.f.d_row ||
                    g.tensor->shape != std::vector<uint64_t>{uint64_t(c.hidden), uint64_t(c.ff), uint64_t(c.experts)} ||
                    u.tensor->shape != g.tensor->shape ||
                    d.tensor->shape != std::vector<uint64_t>{uint64_t(c.ff), uint64_t(c.hidden), uint64_t(c.experts)} ||
                    !k::native_mmvq_supported(int(g.tensor->type)) || !k::native_mmvq_supported(int(d.tensor->type)))
                    throw std::runtime_error("ds41: staged CPU expert tensor geometry or type is unsupported");
                const size_t gate_bytes = lane.f.gu_row * size_t(c.ff), down_bytes = lane.f.d_row * size_t(c.hidden);
                if (gate_bytes > SIZE_MAX / 2 || down_bytes > SIZE_MAX - 2 * gate_bytes ||
                    gate_bytes > g.bytes / size_t(c.experts) || gate_bytes > u.bytes / size_t(c.experts) ||
                    down_bytes > d.bytes / size_t(c.experts))
                    throw std::runtime_error("ds41: staged CPU expert tensor span exceeds its source");
                const size_t bytes_per_expert = 2 * gate_bytes + down_bytes;
                if (bytes_per_expert == 0 || bytes_per_expert > (size_t(16) << 20) ||
                    k::native_mmvq_weight_bytes(int(g.tensor->type), c.hidden, c.ff) != gate_bytes ||
                    k::native_mmvq_weight_bytes(int(d.tensor->type), c.ff, c.hidden) != down_bytes)
                    throw std::runtime_error("ds41: staged CPU expert packed size is invalid");
                stage_cpu_capacity_ = std::max(stage_cpu_capacity_, bytes_per_expert);
            }
            if (stage_cpu_order_ && (c.hidden <= 0 || c.hidden > 5120 || c.hidden % 256 ||
                                     c.ff <= 0 || c.ff > 5120 || c.ff % 256 ||
                                     (lane.f.gu_type != 16 && lane.f.gu_type != 19) ||
                                     (lane.f.d_type != 16 && lane.f.d_type != 19) ||
                                     !(c.clamp[size_t(l)] > 0.f) || !std::isfinite(c.clamp[size_t(l)])))
                throw std::runtime_error("ds41: CPU-order staging requires bounded IQ1_S/IQ2_XXS and a positive clamp");
            lane.f.swiglu_limit = c.clamp[size_t(l)];
            lane.f.iq1s_rows = true;
            full_.push_back(lane.f);
            lane.gate_stride = lane.f.gu_row * size_t(c.ff);
            lane.down_stride = lane.f.d_row * size_t(c.hidden);
            if (fl_ != c.ff) {   // this node's rows [0, fl_): the full row strides, the first fl_ down columns
                lane.f.n_ff = fl_;
                lane.f.h_bytes = ggml_row_size(ggml_type(lane.f.d_act), fl_);
            }
            // a read-only shared mapping of the shard: first touches are about twice as fast to copy as the
            // loader's private mapping (11.8 vs 20.4 GB/s on TR16), for the streamed prompt experts and the lane
            static const bool lane_shared = !(std::getenv("STRATA_DS41_LANE_SHARED") && std::string(std::getenv("STRATA_DS41_LANE_SHARED")) == "0");
            const bool split = fl_ != c.ff;
            const uint8_t *G = shared(g, !split && !ram_copy_), *U = shared(u, !split && !ram_copy_), *D = shared(d, !ram_copy_);
            if (split) {   // populate only this node's gate/up rows of each expert
                const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE));
                for (const uint8_t* base : {G, U})
                    for (int e = 0; e < c.experts; ++e) {
                        const uintptr_t a = reinterpret_cast<uintptr_t>(base + size_t(e) * lane.gate_stride);
                        const uintptr_t a0 = a / pg * pg, a1 = (a + lane.f.gu_row * size_t(fl_) + pg - 1) / pg * pg;
                        populate_.push_back({reinterpret_cast<void*>(a0), size_t(a1 - a0)});
                    }
            }
            lane.gate = lane_shared ? G : g.data();
            lane.up = lane_shared ? U : u.data();
            lane.down = lane_shared ? D : d.data();
            stream_src_.push_back({G, U, D});
            experts_.push_back(lane);
        }
        for (const char* n : {"output_norm.weight", "output.weight"}) bytes += upload(n);
        // STRATA_DS41_POPULATE (1): read every expert page in now (MADV_POPULATE_READ, 8 threads), so the first prompt
        // does not stream the uncached part from the SSD
        if (!(std::getenv("STRATA_DS41_POPULATE") && std::string(std::getenv("STRATA_DS41_POPULATE")) == "0")) {
            const auto p0 = std::chrono::steady_clock::now();
            std::atomic<size_t> next{0};
            std::vector<std::thread> workers;
            for (int w = 0; w < 8; ++w)
                workers.emplace_back([&] {
                    for (size_t i; (i = next++) < populate_.size();)
                        for (size_t o = 0; o < populate_[i].second; o += size_t(64) << 20)
                            madvise(static_cast<uint8_t*>(populate_[i].first) + o, std::min(populate_[i].second - o, size_t(64) << 20),
                                    22 /* MADV_POPULATE_READ */);
                });
            for (auto& w : workers) w.join();
            std::cerr << "DS41 experts populated s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count() << "\n";
        }
        populate_.clear();
        std::cerr << "DS41 model layers=" << c.layers << " hidden=" << c.hidden << " experts=" << c.experts << " top_k="
                  << c.top_k << " gpu_weights_MiB=" << bytes / double(1 << 20) << " context=" << context_
                  << " prefill_chunk=" << chunk_ << "\n";
        // persistent state: compressed rows, Engram table, the CPU lane, GEMM handles, streaming machinery
        const size_t H = size_t(c.hidden);
        caches_.resize(size_t(c.layers));
        stash_v_.resize(size_t(c.layers));
        stash_s_.resize(size_t(c.layers));
        for (int l = 0; l < c.layers; ++l)
            if (c.kv_src[size_t(l)] == l) {
                caches_[size_t(l)] = Dev(size_t(context_ / c.ratio[size_t(l)] + 1) * kv_row_bytes());
                if (c.ratio[size_t(l)] > 1) { stash_v_[size_t(l)] = Dev(size_t(c.head_dim) * 4); stash_s_[size_t(l)] = Dev(size_t(c.head_dim) * 4); }
            }
        score_stride_ = context_ + 1;
        attn_scores_ = Dev(16 * k::ds41_attention_scratch(c.heads));   // up to 8 tokens of ds41_attention_multi, and the split parts
        comp_ = Dev(size_t(c.head_dim) * 4);
        ik_ = Dev(size_t(c.index_dim) * 4);
        // the Engram tables (one shared, or one per Engram layer), read with O_DIRECT
        for (size_t e = 0; e < c.engram_layers.size(); ++e) {
            const auto& t = art_.at(c.engram_per_layer ? "blk." + std::to_string(c.engram_layers[e]) + ".engram_embd.weight"
                                                       : std::string("per_layer_token_embd.weight"));
            int fd = -1;
            for (size_t j = 0; j < engram_tables_.size(); ++j)
                if (engram_tables_[j]->file->path() == t.file->path()) fd = engram_fds_[j];
            // STRATA_DS41_ENGRAM_DIRECT (1): O_DIRECT, so the table never fills the page cache the experts live in.
            // 0: buffered reads - on a shared filesystem (Lustre) many buffered readers reach ~13x the O_DIRECT
            // rate of random 4 KiB rows (raise STRATA_DS41_ENGRAM_THREADS to 128-256 there)
            static const bool direct = !(std::getenv("STRATA_DS41_ENGRAM_DIRECT") && std::string(std::getenv("STRATA_DS41_ENGRAM_DIRECT")) == "0");
            if (fd < 0) {
                fd = ::open(t.file->path().c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
                if (fd < 0) throw std::runtime_error("ds41: cannot open the Engram table: " + t.file->path());
                owned_fds_.push_back(fd);
            }
            engram_tables_.push_back(&t);
            engram_fds_.push_back(fd);
            // the loader's mapping may have pulled table pages in: drop them from the page cache
            if (direct) posix_fadvise(fd, off_t(t.file->data_start() + t.tensor->offset), off_t(t.bytes), POSIX_FADV_DONTNEED);
            else posix_fadvise(fd, off_t(t.file->data_start() + t.tensor->offset), off_t(t.bytes), POSIX_FADV_RANDOM);
        }
        if (const char* v = std::getenv("STRATA_DS41_ENGRAM_REMOTE"); v && *v) engram_remote_ = std::make_unique<EngramRemote>(v);
        if (const char* v = std::getenv("STRATA_DS41_ENGRAM_THREADS")) engram_threads_ = std::max(1, std::atoi(v));
        if (const char* v = std::getenv("STRATA_DS41_ENGRAM_RAM_GIB")) load_engram_ram(std::atof(v));
        (void)process_cpus();   // before the lane pins this thread
        lane_ = std::make_unique<ExpertLane>(c.hidden, fl_, threads, true);
        // STRATA_DS41_LANE_BENCH=map: the lane alone on 64 experts of a middle layer, one token on 6 of them a call,
        // read in place from the mapping (4 KiB pages) and then from an anonymous huge-page copy of the same experts
        if (const char* lb = std::getenv("STRATA_DS41_LANE_BENCH"); lb && std::string(lb) == "map") {
            const int l = c.layers / 2, n = 64, K = c.top_k;
            const LaneLayer& src = experts_[size_t(l)];
            const size_t gs = src.gate_stride ? src.gate_stride : src.f.gu_row * size_t(fl_);
            const size_t ds = src.down_stride ? src.down_stride : src.f.d_row * size_t(c.hidden);
            const size_t bytes = size_t(n) * (2 * gs + ds);
            uint8_t* buf = static_cast<uint8_t*>(mmap(nullptr, bytes + (size_t(2) << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            uint8_t* a = reinterpret_cast<uint8_t*>((uintptr_t(buf) + (size_t(2) << 20) - 1) & ~((uintptr_t(2) << 20) - 1));
            madvise(a, bytes, MADV_HUGEPAGE);
            for (int e = 0; e < n; ++e) {
                std::memcpy(a + size_t(e) * gs, src.gate + size_t(e) * gs, gs);
                std::memcpy(a + size_t(n) * gs + size_t(e) * gs, src.up + size_t(e) * gs, gs);
                std::memcpy(a + 2 * size_t(n) * gs + size_t(e) * ds, src.down + size_t(e) * ds, ds);
            }
            LaneLayer copy = src;
            copy.gate = a;
            copy.up = a + size_t(n) * gs;
            copy.down = a + 2 * size_t(n) * gs;
            copy.gate_stride = gs;
            copy.down_stride = ds;
            std::vector<float> z(c.hidden, 0.01f), y(c.hidden), wv(size_t(K), 0.1f);
            std::vector<int32_t> ids(static_cast<size_t>(K), 0);
            const LaneLayer* order[4] = {&src, &copy, &src, &copy};
            for (int fast = 0; fast < 4; ++fast)
            for (int kk : {K, 2})
            for (int mode = 0; mode < 2; ++mode) {   // run(), then begin() + finish() (the deferred jobs' form)
                const LaneLayer* L = order[0];
                cpu::iq1s_fast_set(fast % 2);
                const int reps = 400;
                for (int k = 0; k < K; ++k) ids[size_t(k)] = -1;
                const auto b0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) {
                    for (int k = 0; k < kk; ++k) ids[size_t(k)] = (r * kk + k) % n;
                    if (mode == 0) lane_->run(*L, z.data(), 1, ids.data(), wv.data(), K, y.data());
                    else { lane_->begin(*L, z.data(), 1, ids.data(), wv.data(), K, y.data()); lane_->finish(); }
                }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count();
                std::cerr << "DS41 lane_bench mapped iq1s_fast=" << fast % 2 << " experts=" << kk << (mode ? " begin/finish" : " run") << " ms_per_call=" << sec * 1e3 / reps
                          << " GB/s=" << double(reps) * kk * (2 * gs + ds) / sec / 1e9 << "\n";
            }
            munmap(buf, bytes + (size_t(2) << 20));
        }
        // STRATA_DS41_LANE_BANDS=1: the mapped experts' bands moved to their NUMA nodes (the page cache keeps them
        // there across runs), 8 threads over the layers
        if (lane_->bands() && !std::getenv("STRATA_DS41_RAM_COPY")) {
            const auto b0 = std::chrono::steady_clock::now();
            std::atomic<size_t> asked{0};
            std::atomic<int> next{0};
            std::vector<std::thread> movers;
            for (int t = 0; t < 8; ++t)
                movers.emplace_back([&] {
                    unpin_helper();
                    for (int l; (l = next.fetch_add(1)) < c.layers;)
                        if (experts_[size_t(l)].gate) asked += ExpertLane::place_bands(experts_[size_t(l)], c.experts, c.hidden, fl_, lane_->nodes());
                });
            for (auto& m : movers) m.join();
            std::cerr << "DS41 lane_bands nodes=" << lane_->nodes() << " GiB=" << double(asked.load()) / double(1ull << 30)
                      << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count() << "\n";
        }
        // STRATA_DS41_GEMM_MIN (32): passes of at least this many tokens take GEMMs for the trunk
        if (const char* v = std::getenv("STRATA_DS41_GEMM_MIN")) gemm_min_ = std::max(2, std::atoi(v));
        if (cublasCreate(&blas_) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas_, stream_) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("ds41: cuBLAS");
        gemm_ = std::make_unique<strata::prefill::Gemm>();
        {
            std::string err;
            if (!gemm_->init(stream_, int64_t(c.heads) * c.head_dim * c.q_rank, err)) throw std::runtime_error("ds41: " + err);
        }
        if (const char* v = std::getenv("STRATA_DS41_STREAM_MIN")) stream_min_ = std::max(2, std::atoi(v));
        {
            const char* v = std::getenv("STRATA_DS41_STAGE_MIB");
            stage_bytes_ = size_t(v ? std::atoi(v) : 128) << 20;   // (128: the slot ring fits beside the desktop reserve)
            if (const char* t = std::getenv("STRATA_DS41_STAGE_THREADS")) stage_threads_ = std::max(1, std::atoi(t));
            for (int i = 0; i < 2; ++i) {
                ck(cudaEventCreateWithFlags(&stage_uploaded_[i], cudaEventDisableTiming), "stage event");
                ck(cudaEventCreateWithFlags(&stage_copied_[i], cudaEventDisableTiming), "stage event");
                ck(cudaEventCreateWithFlags(&stage_free_[i], cudaEventDisableTiming), "stage event");
            }
            copy_s_ = make_stream(cudaStreamNonBlocking);
            use_mmq_ = strata::prefill::mmq::built() && !(std::getenv("STRATA_DS41_MMQ") && std::string(std::getenv("STRATA_DS41_MMQ")) == "0");
            if (use_mmq_) mmq_ = std::make_unique<strata::prefill::mmq::Context>();
            if (const char* r = std::getenv("STRATA_DS41_MMQ_ROWS")) {
                size_t consumed = 0;
                const long long requested = std::stoll(r, &consumed, 10);
                if (consumed != std::strlen(r) || requested > 8192)
                    throw std::invalid_argument("ds41: STRATA_DS41_MMQ_ROWS must be an integer no greater than 8192");
                rcap_min_ = requested <= 256 ? 256 : int(requested);
            }
        }
        (void)H;
        score_.assign(size_t(c.layers) * c.experts, 0.0f);
        if (!remote_addr_.empty()) connect_remote();
        order_.assign(size_t(c.layers), {});
        n_stream_.assign(size_t(c.layers), c.experts);
        for (auto& o : order_) { o.resize(size_t(c.experts)); std::iota(o.begin(), o.end(), 0); }
        resize(chunk_);
        // STRATA_DS41_DSPARK=draft.gguf: the DSpark drafter on GPU 0, loaded before the experts take the free VRAM
        if (const char* dp = std::getenv("STRATA_DS41_DSPARK"); dp && *dp) load_dspark(dp);
        // a long first prompt borrows the tier's VRAM; prefill_done() shrinks the buffers and loads the tier
        if (ram_copy_) {
            if (!remote_addr_.empty()) throw std::runtime_error("ds41: STRATA_DS41_RAM_COPY with a remote worker is not supported");
            residency();
            tier_lent_ = false;
            return;
        }
        tier_lent_ = chunk_ >= stream_min_;
        if (!tier_lent_) load_tier();
        else pin_experts();   // at load, not in the first prompt's time
        if (stage_cpu_expert_) {
            if (tier_lent_ || fl_ != c.ff || peer_ || !remote_addr_.empty() || experts_.size() != size_t(c.layers) ||
                !stage_cpu_capacity_ || !gx_q8_.p || gx_q8_.bytes < k::native_q8_1_bytes(c.hidden, 1) ||
                !g_h_q8_.p || g_h_q8_.bytes < k::native_q8_1_bytes(c.ff, 1) ||
                !g_gate_.p || g_gate_.bytes < size_t(c.ff) * sizeof(float) ||
                !g_up_.p || g_up_.bytes < size_t(c.ff) * sizeof(float) ||
                !g_h_.p || g_h_.bytes < size_t(c.ff) * sizeof(float) ||
                !g_out_.p || g_out_.bytes < size_t(c.hidden) * sizeof(float) ||
                !g_sum_.p || chunk_ < 1 || g_sum_.bytes < size_t(std::min(chunk_, 8)) * size_t(c.hidden) * sizeof(float))
                throw std::runtime_error("ds41: staged CPU expert scratch or execution mode is unsupported");
            // All staging is bounded and allocated before any decoder forward pass.
            if (stage_cpu_order_) {
                stage_cpu_xq_ = Dev(size_t(c.hidden / 256) * 292);
                stage_cpu_hq_ = Dev(size_t(c.ff / 256) * 292);
                if (stage_cpu_collect_)
                    stage_cpu_results_ = reinterpret_cast<float*>(pinned_keep(size_t(8) * stage_cpu_quota_ * c.hidden * sizeof(float)));
                if (stage_cpu_host_act_) {
                    stage_cpu_gateup_ = reinterpret_cast<float*>(pinned_keep(size_t(2) * c.ff * sizeof(float)));
                    stage_cpu_hq_host_ = reinterpret_cast<uint8_t*>(pinned_keep(size_t(c.ff / 256) * 292));
                    stage_cpu_hidden_.resize(size_t(c.ff));
                    if (stage_cpu_batch_act_) {
                        stage_batch_gateup_ = reinterpret_cast<float*>(pinned_keep(size_t(16) * c.ff * sizeof(float)));
                        stage_batch_hq_ = reinterpret_cast<uint8_t*>(pinned_keep(size_t(8) * size_t(c.ff / 256) * 292));
                    }
                }
            }
            stage_cpu_device_ = Dev(stage_cpu_capacity_);
            if (stage_cache_slots_requested_) {
                const size_t cache_bytes = checked_size_mul(size_t(stage_cache_slots_requested_), stage_cpu_capacity_, "CPU staging cache");
                if (cache_bytes > (size_t(3072) << 20))
                    throw std::runtime_error("ds41: CPU staging cache exceeds 3 GiB");
                stage_cache_device_ = Dev(cache_bytes);
                stage_cache_.resize(size_t(stage_cache_slots_requested_));
            }
            stage_cpu_host_ = reinterpret_cast<uint8_t*>(pinned_keep(stage_cpu_capacity_));
            stage_cpu_check_host_ = reinterpret_cast<uint8_t*>(pinned_keep(stage_cpu_capacity_));
            stage_cpu_checked_.assign(size_t(c.layers), 0);
            if (stage_cpu_order_) stage_cpu_order_checked_.assign(size_t(c.layers), 0);
            ck(cudaEventCreateWithFlags(&stage_cpu_copy_done_, cudaEventDisableTiming), "CPU expert stage event");
        }
        if (const char* v = std::getenv("STRATA_DS41_CANON_MIB"); v && *v && std::string(v) != "0") canon_load(v);
        if (const char* v = std::getenv("STRATA_DS41_REMOTE_OWN"); v && *v) connect_own(v);
    }

    // ---- STRATA_DS41_CANON_MIB=N (opt-in): a static GPU expert tier with the CPU lane's arithmetic.  The most-used
    // experts of STRATA_DS41_CANON_PROFILE (a STRATA_DS41_ROUTES trace; hits per byte) fill up to N MiB of VRAM once at
    // load.  In passes of up to 8 tokens, the CPU routes whose expert is held run on the GPU with the CPU-order IQ dots
    // (Q8_K activations, the CPU's integer lanes and float order; the SwiGLU exponential in double), all of a layer's in
    // one grouped launch per matrix, and their raw outputs join the lane's sum in its rank order
    // (finish_with_external).  STRATA_DS41_CANON_CHECK=1 compares each layer's first entry with the CPU rows.
    void canon_load(const char* mib_text) {
        const auto& c = cfg_;
        char* end = nullptr;
        const long mib = std::strtol(mib_text, &end, 10);
        if (*end || mib < 1 || mib > 12288) throw std::invalid_argument("ds41: STRATA_DS41_CANON_MIB must be 1 through 12288");
        const char* profile = std::getenv("STRATA_DS41_CANON_PROFILE");
        if (!profile || !*profile) throw std::invalid_argument("ds41: STRATA_DS41_CANON_MIB needs STRATA_DS41_CANON_PROFILE");
        if (stage_cpu_expert_ || ram_copy_ || peer_ || !remote_addr_.empty() || !ram_pos_.empty() || !packed_.empty() ||
            fl_ != c.ff || experts_.size() != size_t(c.layers) || strata_env_is_one("STRATA_IQ1S_FAST") ||
            strata_env_is_one("STRATA_DS41_GPU_ROUTE") || c.hidden % 256 || c.ff % 256 || c.hidden > 5120 || c.ff > 5120 ||
            c.top_k < 1 || c.top_k > 16)
            throw std::invalid_argument("ds41: the canonical tier needs plain CPU routing (no staging, RAM copy, peer, remote, GPU_ROUTE or IQ1S_FAST)");
        for (int l = 0; l < c.layers; ++l) {
            const auto& f = experts_[size_t(l)].f;
            if ((f.gu_type != 16 && f.gu_type != 19) || (f.d_type != 16 && f.d_type != 19) ||
                f.gu_row != size_t(c.hidden / 256) * size_t(f.gu_type == 19 ? 50 : 66) ||
                f.d_row != size_t(c.ff / 256) * size_t(f.d_type == 19 ? 50 : 66))
                throw std::invalid_argument("ds41: the canonical tier supports IQ1_S / IQ2_XXS experts only");
        }
        std::vector<double> count(size_t(c.layers) * c.experts, 0);
        {
            std::ifstream g(profile, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the canonical tier profile " + std::string(profile));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int k = 0; k < c.top_k; ++k) {
                        const int e = v[size_t(l) * c.top_k + k];
                        if (e >= 0 && e < c.experts) count[size_t(l) * c.experts + e] += 1;
                    }
        }
        std::vector<std::tuple<double, int, int>> order;
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (count[size_t(l) * c.experts + e] > 0)
                    order.emplace_back(count[size_t(l) * c.experts + e] / double(expert_bytes(l)), l, e);
        std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        // STRATA_DS41_CANON_TRAFFIC=path (a STRATA_DS41_ROUTES trace of the workload; needs STRATA_DS41_REMOTE_OWN_SKIP_MIB = the node's kept MiB): the tier
        // is ranked by that trace instead, among the experts this node holds (the first 90 % of the kept MiB in the profile's order)
        if (const char* tp = std::getenv("STRATA_DS41_CANON_TRAFFIC"); tp && *tp) {
            const char* skip = std::getenv("STRATA_DS41_REMOTE_OWN_SKIP_MIB");
            if (!skip || std::atol(skip) <= 0) throw std::invalid_argument("ds41: STRATA_DS41_CANON_TRAFFIC needs STRATA_DS41_REMOTE_OWN_SKIP_MIB");
            std::vector<double> tcount(size_t(c.layers) * c.experts, 0);
            std::ifstream g(tp, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the traffic trace " + std::string(tp));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int k = 0; k < c.top_k; ++k) {
                        const int e = v[size_t(l) * c.top_k + k];
                        if (e >= 0 && e < c.experts) tcount[size_t(l) * c.experts + e] += 1;
                    }
            std::vector<char> kept(size_t(c.layers) * c.experts, 0);
            const double limit = 0.9 * double(std::atol(skip)) * 1048576.0;
            double acc = 0;
            for (const auto& [score, l, e] : order) {
                (void)score;
                acc += double(expert_bytes(l));
                if (acc > limit) break;
                kept[size_t(l) * c.experts + e] = 1;
            }
            std::vector<std::tuple<double, int, int>> torder;
            for (int l = 0; l < c.layers; ++l)
                for (int e = 0; e < c.experts; ++e)
                    if (kept[size_t(l) * c.experts + e] && tcount[size_t(l) * c.experts + e] > 0)
                        torder.emplace_back(tcount[size_t(l) * c.experts + e] / double(expert_bytes(l)), l, e);
            std::stable_sort(torder.begin(), torder.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
            std::cerr << "DS41 canon_traffic ranked=" << torder.size() << " of " << order.size() << " profile experts\n";
            order = std::move(torder);
        }
        // the budget: the request, less a 512 MiB margin above the physical floor for later scratch
        size_t budget = size_t(mib) << 20;
#if defined(__HIP_PLATFORM_AMD__)
        const size_t pf = physical_vram_free(), keep = physical_vram_keep() + (size_t(512) << 20);
        budget = std::min(budget, pf > keep ? pf - keep : size_t(0));
#endif
        canon_off_.assign(size_t(c.layers) * c.experts, -1);
        size_t used = 0;
        std::vector<std::pair<int, int>> picks;
        for (const auto& [score, l, e] : order) {
            (void)score;
            const size_t b = (expert_bytes(l) + 255) / 256 * 256;
            if (used + b > budget) continue;
            canon_off_[size_t(l) * c.experts + e] = int64_t(used);
            picks.emplace_back(l, e);
            used += b;
        }
        if (picks.empty()) throw std::runtime_error("ds41: no VRAM left for the canonical tier");
        if (borrow_) {
            if (borrow_->canon_arena_.bytes != used)
                throw std::runtime_error("ds41: the borrowed canonical tier has another size than this decoder's (same STRATA_DS41_CANON_MIB / profile needed)");
            canon_off_ = borrow_->canon_off_;   // (the first decoder may have settled its tier since: adapt_settle)
            canon_arena_ = Dev::view(borrow_->canon_arena_);
        } else
        canon_arena_ = Dev(used);
        size_t max_b = 0;
        for (int l = 0; l < c.layers; ++l) max_b = std::max(max_b, expert_bytes(l));
        uint8_t* host = reinterpret_cast<uint8_t*>(pinned_keep(max_b));
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& [l, e] : picks) {
            if (borrow_) break;   // the first decoder's arena is already filled
            slice_to_host(l, e, host);
            ck(cudaMemcpyAsync(canon_arena_.b(size_t(canon_off_[size_t(l) * c.experts + e])), host, expert_bytes(l),
                               cudaMemcpyHostToDevice, stream_), "canonical tier upload");
            ck(cudaStreamSynchronize(stream_), "canonical tier upload sync");
        }
        // scratch for canon_tmax_ (8; STRATA_DS41_CANON_TOKENS, batched sessions) positions x top_k entries
        if (const char* v = std::getenv("STRATA_DS41_CANON_TOKENS")) canon_tmax_ = std::clamp(std::atoi(v), 8, 64);
        const int E = canon_tmax_ * c.top_k;
        const size_t q8k = 292;
        canon_xq_ = Dev(size_t(canon_tmax_) * (c.hidden / 256) * q8k);
        canon_gu_ = Dev(size_t(2) * E * c.ff * sizeof(float));
        canon_h_ = Dev(size_t(E) * c.ff * sizeof(float));
        canon_hq_ = Dev(size_t(E) * (c.ff / 256) * q8k);
        canon_out_ = Dev(size_t(E) * c.hidden * sizeof(float));
        canon_meta_ = Dev(size_t(2) * E * sizeof(uint64_t) + size_t(3 * E + 2) * sizeof(int));
        canon_meta_host_ = reinterpret_cast<uint8_t*>(pinned_keep(canon_meta_.bytes));
        canon_res_ = pinned_keep(size_t(E) * c.hidden * sizeof(float));
        canon_checked_.assign(size_t(c.layers), 0);
        if (const char* q = std::getenv("STRATA_DS41_CANON_STREAM"); q && *q) {
            char* qe = nullptr;
            const long v = std::strtol(q, &qe, 10);
            if (*qe || v < 0 || v > 32) throw std::invalid_argument("ds41: STRATA_DS41_CANON_STREAM must be 0 through 32");
            canon_stream_ = int(v);
            if (canon_stream_) {
                canon_ring_slot_ = (max_b + 255) / 256 * 256;
                canon_ring_ = Dev(canon_ring_slot_ * size_t(canon_stream_));
                if (const char* th = std::getenv("STRATA_DS41_CANON_STREAM_THREADS"); th && *th) {
                    char* te = nullptr;
                    const long tv = std::strtol(th, &te, 10);
                    if (*te || tv < 0 || tv > 8) throw std::invalid_argument("ds41: STRATA_DS41_CANON_STREAM_THREADS must be 0 through 8");
                    canon_stage_threads_ = int(tv);
                }
                if (canon_stage_threads_)
                    canon_pinned_ = reinterpret_cast<uint8_t*>(pinned_keep(canon_ring_slot_ * size_t(canon_stream_)));
            }
        }
        if (const char* pv = std::getenv("STRATA_DS41_CANON_PIN_MIB"); pv && *pv && std::atol(pv) > 0) {
            // the experts after the tier in profile order: all are kept on this node when tier + pin <= STRATA_DS41_REMOTE_OWN_SKIP_MIB
            const long pin_mib = std::atol(pv);
            const char* skip = std::getenv("STRATA_DS41_REMOTE_OWN_SKIP_MIB");
            if (!canon_stream_ || !skip || std::atol(skip) < mib + pin_mib)
                throw std::invalid_argument("ds41: STRATA_DS41_CANON_PIN_MIB needs STRATA_DS41_CANON_STREAM and STRATA_DS41_REMOTE_OWN_SKIP_MIB >= canon MiB + pin MiB");
            if (borrow_) {   // a second decoder in the process reads the first one's pinned level (host memory, read-only)
                canon_pin_ = borrow_->canon_pin_;
                canon_pin_off_ = borrow_->canon_pin_off_;
                canon_pin_used_ = borrow_->canon_pin_used_;
                canon_pin_cap_ = borrow_->canon_pin_cap_;
            } else {
            canon_pin_off_.assign(size_t(c.layers) * c.experts, -1);
            size_t pused = 0;
            std::vector<std::pair<int, int>> ppicks;
            for (const auto& [score, l, e] : order) {
                (void)score;
                if (canon_off_[size_t(l) * c.experts + e] >= 0) continue;
                const size_t b = (expert_bytes(l) + 255) / 256 * 256;
                if (pused + b > (size_t(pin_mib) << 20)) continue;
                canon_pin_off_[size_t(l) * c.experts + e] = int64_t(pused);
                ppicks.emplace_back(l, e);
                pused += b;
            }
            const long owned_mib = std::getenv("STRATA_DS41_CANON_PIN_OWNED_MIB") ? std::atol(std::getenv("STRATA_DS41_CANON_PIN_OWNED_MIB")) : 0;
            canon_pin_cap_ = pused + (size_t(std::max(0L, owned_mib)) << 20);
            canon_pin_ = reinterpret_cast<uint8_t*>(pinned_keep(canon_pin_cap_));
            canon_pin_used_ = pused;
            for (const auto& [score, l, e] : order) { (void)score; canon_order_.emplace_back(l, e); }
            for (const auto& [l, e] : ppicks) slice_to_host(l, e, canon_pin_ + canon_pin_off_[size_t(l) * c.experts + e]);
            std::cerr << "DS41 canon_pin experts=" << ppicks.size() << " MiB=" << (pused >> 20) << "\n";
            }
        }
        std::cerr << "DS41 canon_tier experts=" << picks.size() << " MiB=" << (used >> 20) << " of_requested=" << mib
                  << " load_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }

    // Connects to the worker (it dials STRATA_DS41_REMOTE_TP) and makes it hold rows [fl_, ff) of every expert: kept
    // ones (same fingerprint in its --weights-dir) or streamed now, read with pread so they never enter this node's
    // mapping, then dropped from its page cache.
    void connect_remote() {
        const auto& c = cfg_;
        namespace net = strata::net;
        auto r = std::make_unique<Remote>();
        r->hidden = c.hidden;
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 0;
        cfg.world = 2;
        cfg.root_addr = remote_addr_.c_str();
        if (!std::getenv("UCOMM_TIMEOUT_MS")) cfg.timeout_ms = 0;   // wait for the worker (it may load for minutes)
        std::cerr << "DS41 remote waiting for the worker at " << remote_addr_ << "\n";
        Remote::ok(ucomm_init(&cfg, &r->comm), "connect");
        const int most = std::max(chunk_, 1024);   // the largest CPU-lane pass (prompt chunks, decode)
        r->request.resize(sizeof(net::TpRequest) + size_t(most) * c.top_k * sizeof(net::TpJob) + size_t(most) * c.hidden * 4);
        r->reply.resize(size_t(most) * c.hidden);
        for (const auto& [ptr, bytes] : {std::pair<void*, size_t>{r->request.data(), r->request.size()}, {r->reply.data(), r->reply.size() * 4}}) {
            ucomm_mr_t* region = nullptr;
            if (ucomm_mr_reg(r->comm, ptr, bytes, &region) == UCOMM_OK) r->regions.push_back(region);
        }
        std::vector<uint8_t> setup(sizeof(net::TpSetup));
        uint64_t h = 14695981039346656037ull;
        auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull; };
        for (int l = 0; l < c.layers; ++l) {
            const auto& F = full_[size_t(l)];
            const net::TpLayer spec{l, int32_t(F.gu_type), int32_t(F.d_type), c.ff, fl_, c.experts, c.clamp[size_t(l)], 0};
            const auto* b = reinterpret_cast<const uint8_t*>(&spec);
            setup.insert(setup.end(), b, b + sizeof spec);
            const size_t rows = size_t(c.ff - fl_), d_keep = ggml_row_size(ggml_type(F.d_type), fl_);
            const size_t d_rest = F.d_row - d_keep, bytes = 2 * rows * F.gu_row + d_rest * size_t(c.hidden);
            r->geo.push_back({rows, F.gu_row, d_keep, d_rest, bytes, net::tp_blob_stride(bytes)});
            for (const char* part : {"gate", "up", "down"}) {
                const auto& t = art_.at("blk." + std::to_string(l) + ".ffn_" + part + "_exps.weight");
                const int64_t geometry[4] = {l, fl_, int64_t(t.tensor->type), int64_t(t.bytes)};
                mix(geometry, sizeof geometry);
                mix(t.file->path().c_str(), t.file->path().size());
                const int64_t where[2] = {int64_t(t.file->file_size()), int64_t(t.tensor->offset)};
                mix(where, sizeof where);
            }
        }
        const net::TpSetup head{net::kTpMagicDs41, uint32_t(c.layers), uint32_t(c.hidden), net::kReplyF32, net::kWeightsStreamed, 0, h ? h : 1};
        std::memcpy(setup.data(), &head, sizeof head);
        Remote::ok(ucomm_send(r->comm, 1, net::kTagSetup, setup.data(), setup.size()), "setup");
        net::TpAccept accept{};
        Remote::ok(ucomm_recv(r->comm, 1, net::kTagAccept, &accept, sizeof accept, nullptr), "accept");
        if (!accept.ok)
            throw std::runtime_error("ds41 remote: the worker refused: " + std::string(accept.reason, strnlen(accept.reason, sizeof accept.reason)));
        double sent = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (!accept.have) {
            size_t largest = 0;
            for (const auto& g : r->geo) largest = std::max(largest, g.bytes);
            std::array<std::vector<uint8_t>, 2> staging;
            std::array<ucomm_req_t*, 2> inflight{};
            std::array<ucomm_mr_t*, 2> registered{};
            for (int k = 0; k < 2; ++k) {
                staging[size_t(k)].resize(largest);
                if (ucomm_mr_reg(r->comm, staging[size_t(k)].data(), largest, &registered[size_t(k)]) != UCOMM_OK) registered[size_t(k)] = nullptr;
            }
            std::map<std::string, int> fds;
            auto fd_of = [&](const core::ArtifactTensor& t) {
                auto it = fds.find(t.file->path());
                if (it == fds.end()) {
                    const int fd = ::open(t.file->path().c_str(), O_RDONLY);
                    if (fd < 0) throw std::runtime_error("ds41 remote: cannot open " + t.file->path());
                    it = fds.emplace(t.file->path(), fd).first;
                }
                return it->second;
            };
            auto read_at = [](int fd, uint8_t* dst, size_t n, uint64_t off) {
                for (size_t got = 0; got < n;) {
                    const ssize_t k = pread(fd, dst + got, n - got, off_t(off + got));
                    if (k <= 0) throw std::runtime_error("ds41 remote: read failed");
                    got += size_t(k);
                }
            };
            std::vector<uint8_t> down;
            int which = 0;
            for (int l = 0; l < c.layers; ++l) {
                const std::string p = "blk." + std::to_string(l) + ".";
                const auto &G = art_.at(p + "ffn_gate_exps.weight"), &U = art_.at(p + "ffn_up_exps.weight"), &D = art_.at(p + "ffn_down_exps.weight");
                const auto& g = r->geo[size_t(l)];
                const auto& F = full_[size_t(l)];
                const uint64_t go = G.file->data_start() + G.tensor->offset, uo = U.file->data_start() + U.tensor->offset;
                const uint64_t dof = D.file->data_start() + D.tensor->offset;
                const size_t gstride = F.gu_row * size_t(c.ff), dstride = F.d_row * size_t(c.hidden);
                down.resize(dstride);
                for (int e = 0; e < c.experts; ++e) {
                    if (inflight[size_t(which)]) Remote::ok(ucomm_wait(inflight[size_t(which)], nullptr), "weights");
                    inflight[size_t(which)] = nullptr;
                    uint8_t* blob = staging[size_t(which)].data();
                    read_at(fd_of(G), blob, g.rows * g.gu_row, go + size_t(e) * gstride + size_t(fl_) * g.gu_row);
                    read_at(fd_of(U), blob + g.rows * g.gu_row, g.rows * g.gu_row, uo + size_t(e) * gstride + size_t(fl_) * g.gu_row);
                    read_at(fd_of(D), down.data(), dstride, dof + size_t(e) * dstride);
                    for (int row = 0; row < c.hidden; ++row)
                        std::memcpy(blob + 2 * g.rows * g.gu_row + size_t(row) * g.d_rest, down.data() + size_t(row) * F.d_row + g.d_keep, g.d_rest);
                    Remote::ok(ucomm_isend(r->comm, 1, net::kTagWeights, blob, g.bytes, &inflight[size_t(which)]), "weights");
                    which ^= 1;
                    sent += double(g.bytes);
                }
                // the worker's gate/up rows leave this node's page cache (the mapped local rows stay)
                posix_fadvise(fd_of(G), off_t(go), off_t(G.bytes), POSIX_FADV_DONTNEED);
                posix_fadvise(fd_of(U), off_t(uo), off_t(U.bytes), POSIX_FADV_DONTNEED);
            }
            for (auto* q : inflight) if (q) Remote::ok(ucomm_wait(q, nullptr), "weights");
            for (auto* m : registered) if (m) ucomm_mr_dereg(m);
            for (const auto& [path, fd] : fds) ::close(fd);
        }
        net::TpReady ready{};
        Remote::ok(ucomm_recv(r->comm, 1, net::kTagReady, &ready, sizeof ready, nullptr), "ready");
        // a GPU worker (strata-ds41-worker) computes its rows of streamed passes too: no rows are fetched back;
        // STRATA_DS41_REMOTE_PREFILL=fetch fetches them anyway
        remote_gpu_ = ready.ok == 2 && !(std::getenv("STRATA_DS41_REMOTE_PREFILL") && std::string(std::getenv("STRATA_DS41_REMOTE_PREFILL")) == "fetch");
        if (remote_gpu_ && cudaHostRegister(r->reply.data(), r->reply.size() * 4, 0) == cudaSuccess) r->reply_pinned = true;
        else (void)cudaGetLastError();
        std::cerr << "DS41 remote backend=" << ucomm_backend_name(r->comm) << " rows=" << c.ff - fl_ << "/" << c.ff
                  << " weights=" << (accept.have ? "kept" : "sent") << " prompt=" << (remote_gpu_ ? "worker_gpu" : "fetch")
                  << " GiB=" << sent / double(1ull << 30) << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
        remote_ = std::move(r);
    }

    // ---- STRATA_DS41_RAM_COPY=1: every expert in VRAM or in this process's memory (a shared filesystem such as
    // Lustre keeps little page cache). The most-routed experts (STRATA_DS41_EXPERT_PROFILE, else spread by index over
    // the layers) fill GPU0's free VRAM less STRATA_DS41_GPU_RESERVE_MIB (1024), then a second GPU's
    // (STRATA_DS41_PEER_GPU, default on with two devices); they are read from the files straight to the GPU, laid out
    // per layer at the MMQ stride. The rest are read into anonymous memory (STRATA_DS41_RAM_THREADS, 16) and dropped
    // from the page cache. The tier is static then (no promotions; nothing could be promoted from).
    void residency() {
        const auto& c = cfg_;
        {   // the token embeddings too: a prompt reads thousands of scattered rows (each a filesystem round trip)
            const auto& t = art_.at("token_embd.weight");
            embd_copy_.resize(t.bytes);
            const int fd = ::open(t.file->path().c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
            const uint64_t base = t.file->data_start() + t.tensor->offset;
            for (size_t got = 0; got < t.bytes;) {
                const ssize_t r = pread(fd, embd_copy_.data() + got, t.bytes - got, off_t(base + got));
                if (r <= 0) { ::close(fd); throw std::runtime_error("ds41: token embedding read failed"); }
                got += size_t(r);
            }
            ::close(fd);
        }
        auto env = [](const char* name, double fallback) { const char* v = std::getenv(name); return v ? std::stod(v) : fallback; };
        const auto t0 = std::chrono::steady_clock::now();
        promote_ = 0;
        pinned_tried_ = true;
        pinned_layer_.assign(size_t(c.layers), 0);
        gu_pin_ = down_pin_ = pinned_layer_;
        slot_.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
        if (score_.size() != size_t(c.layers) * c.experts) score_.assign(size_t(c.layers) * c.experts, 0.0f);
        gx_q8_ = Dev(k::native_q8_1_bytes(c.hidden, 1));
        g_gate_ = Dev(size_t(c.ff) * 4);
        g_up_ = Dev(size_t(c.ff) * 4);
        g_h_ = Dev(size_t(c.ff) * 4);
        g_h_q8_ = Dev(k::native_q8_1_bytes(c.ff, 1));
        g_out_ = Dev(size_t(c.hidden) * 4);
        size_t keep = std::max<size_t>(size_t(env("STRATA_DS41_GPU_RESERVE_MIB", 1024)) << 20, vram_reserve + (128 << 20));
#if defined(__HIP_PLATFORM_AMD__)
        keep = std::max(keep, physical_vram_keep());
        const size_t physical_free = physical_vram_free();
        const size_t physical_budget = physical_free > keep ? physical_free - keep : 0;
#endif
        size_t free0 = 0, total0 = 0;
        ck(cudaMemGetInfo(&free0, &total0), "memory info");
        size_t budget0 = free0 > keep ? free0 - keep : 0, budget1 = 0;
#if defined(__HIP_PLATFORM_AMD__)
        budget0 = std::min(budget0, physical_budget);
#endif
        if (std::getenv("STRATA_DS41_GPU_EXPERT_MIB")) budget0 = std::min(budget0, size_t(env("STRATA_DS41_GPU_EXPERT_MIB", 0)) << 20);
        int devices = 0;
        cudaGetDeviceCount(&devices);
        if (devices > 1 && env("STRATA_DS41_PEER_GPU", 1) != 0) {
            peer_ = std::make_unique<PeerGpu>();
            peer_->hidden = c.hidden;
            peer_->ff = c.ff;
            peer_->init(chunk_, rcap_);
            peer_->timing = dtiming_;
            if (env("STRATA_DS41_PEER_THREAD", 0) != 0) peer_->start_thread();
            const size_t f1 = peer_->available_bytes();
            size_t peer_keep = keep;
#if defined(__HIP_PLATFORM_AMD__)
            peer_keep = std::max(peer_keep, physical_vram_keep());
#endif
            budget1 = f1 > peer_keep ? f1 - peer_keep : 0;
        }
        // STRATA_DS41_RAM_PINNED=1: the RAM experts in one pinned buffer per layer, in streaming order at the stream
        // stride (each streamed item is one DMA, no host staging). STRATA_DS41_PEER_STREAM=f (with it and the peer
        // GPU): the peer streams the last fraction f of each layer's RAM experts over its own link
        // (STRATA_DS41_PEER_SLOTS slots, 4, of the stream slot size, taken from its expert budget)
        packed_on_ = env("STRATA_DS41_RAM_PINNED", 0) != 0;
        // STRATA_DS41_RAM_NUMA (1): place each GPU's part of that copy on its socket's NUMA nodes
        if (packed_on_ && env("STRATA_DS41_RAM_NUMA", 1) != 0) {
            numa_[0] = gpu_socket_nodes(0);
            numa_[1] = peer_ ? gpu_socket_nodes(peer_->dev) : 0;
            std::cerr << "DS41 numa gpu0_nodes=0x" << std::hex << numa_[0] << " gpu1_nodes=0x" << numa_[1] << std::dec << "\n";
        }
        peer_frac_ = packed_on_ && peer_ && slot_bytes_ ? std::min(1.0, std::max(0.0, env("STRATA_DS41_PEER_STREAM", 0))) : 0.0;
        if (peer_frac_ > 0) {
            const int ns = std::max(2, int(env("STRATA_DS41_PEER_SLOTS", 4)));
            peer_->stream_setup(ns, slot_bytes_);
            const size_t f1 = peer_->available_bytes();
            size_t peer_keep = keep;
#if defined(__HIP_PLATFORM_AMD__)
            peer_keep = std::max(peer_keep, physical_vram_keep());
#endif
            budget1 = f1 > peer_keep ? f1 - peer_keep : 0;
        }
        // ranking: profile counts, ties by expert index then layer (spread over the layers)
        std::vector<double> count(size_t(c.layers) * c.experts, 0);
        if (const char* profile = std::getenv("STRATA_DS41_EXPERT_PROFILE")) {
            std::ifstream g(profile, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the expert profile " + std::string(profile));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int kk = 0; kk < c.top_k; ++kk) {
                        const int e = v[size_t(l) * c.top_k + kk];
                        if (e >= 0 && e < c.experts) count[size_t(l) * c.experts + e] += 1;
                    }
        }
        std::vector<int> order(count.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (count[size_t(a)] != count[size_t(b)]) return count[size_t(a)] > count[size_t(b)];
            return a % c.experts != b % c.experts ? a % c.experts < b % c.experts : a < b;
        });
        where_.assign(size_t(c.layers), std::vector<char>(size_t(c.experts), 0));
        std::vector<size_t> stride(size_t(c.layers));
        for (int l = 0; l < c.layers; ++l) stride[size_t(l)] = stream_stride(l);
        size_t used0 = 0, used1 = 0, ram = 0;
        for (int i : order) {
            const int l = i / c.experts, e = i % c.experts;
            const size_t b = stride[size_t(l)];
            if (used0 + b <= budget0) { where_[size_t(l)][size_t(e)] = 1; used0 += b; }
            else if (peer_ && used1 + b <= budget1) { where_[size_t(l)][size_t(e)] = 2; used1 += b; }
            else ram += expert_bytes(l);
        }
        // layout: per layer, GPU0's experts contiguous (positions after the streamed ones), the peer's contiguous
        if (used0) {
#if defined(__HIP_PLATFORM_AMD__)
            const size_t physical_free_now = physical_vram_free();
            if (physical_free_now < keep || used0 > physical_free_now - keep)
                throw std::runtime_error("ds41: resident expert arena would cross the physical AMD VRAM reserve");
#endif
            void* p = nullptr;
            ck(cudaMalloc(&p, used0), "resident experts");
            arena_.p = p;
            arena_.bytes = used0;
        }
        res0_base_.assign(size_t(c.layers), nullptr);
        res0_stride_ = stride;
        max_stride_ = *std::max_element(stride.begin(), stride.end());
        order_.assign(size_t(c.layers), {});
        n_stream_.assign(size_t(c.layers), 0);
        size_t off0 = 0, off1 = 0;
        Dev peer_arena;
        if (peer_ && used1) {
            PeerGpu::Guard gd(peer_->dev);
            peer_arena = Dev(used1);
        }
        if (peer_) {
            peer_->base.assign(size_t(c.layers), nullptr);
            peer_->stride = stride;
            peer_->pos.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
            for (int l = 0; l < c.layers; ++l) {
                peer_->gu_type.push_back(full_[size_t(l)].gu_type);
                peer_->d_type.push_back(full_[size_t(l)].d_type);
                peer_->clamp.push_back(c.clamp[size_t(l)]);
            }
        }
        struct Task { int l, e, where; uint8_t* dst; };
        std::vector<Task> uploads;
        s0_.assign(size_t(c.layers), 0);
        if (packed_on_) {
            packed_.assign(size_t(c.layers), nullptr);
            ram_pos_.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
        }
        for (int l = 0; l < c.layers; ++l) {
            auto& ord = order_[size_t(l)];
            for (int e = 0; e < c.experts; ++e) if (where_[size_t(l)][size_t(e)] == 0) ord.push_back(e);
            n_stream_[size_t(l)] = int(ord.size());
            const int S = int(ord.size());
            s0_[size_t(l)] = S - int(std::lround(S * peer_frac_));
            if (packed_on_ && S && numa_[0] && (peer_frac_ == 0 || numa_[1])) {
                // each GPU's part on the NUMA nodes of its socket (interleaved over them), pinned after the fill
                const size_t bytes = size_t(S) * stride[size_t(l)], page = size_t(sysconf(_SC_PAGESIZE));
                void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (m == MAP_FAILED) throw std::runtime_error("ds41: cannot allocate the expert copy");
                madvise(m, bytes, MADV_HUGEPAGE);
                const size_t cut = std::min(bytes, size_t(s0_[size_t(l)]) * stride[size_t(l)] / page * page);
                auto bind = [&](size_t a, size_t n, unsigned long mask) {
                    if (n && syscall(SYS_mbind, static_cast<uint8_t*>(m) + a, n, 3 /* MPOL_INTERLEAVE */, &mask, 65, 0) != 0)
                        std::cerr << "DS41 mbind failed; the expert copy follows the process policy\n";
                };
                bind(0, cut, numa_[0]);
                bind(cut, bytes - cut, numa_[peer_frac_ > 0 ? 1 : 0]);
                packed_[size_t(l)] = static_cast<uint8_t*>(m);
                packed_mapped_ = true;
                for (int q = 0; q < S; ++q) ram_pos_[size_t(l)][size_t(ord[size_t(q)])] = q;
            } else if (packed_on_ && S) {
                void* h = nullptr;
                ck(cudaHostAlloc(&h, size_t(S) * stride[size_t(l)], cudaHostAllocPortable), "pinned expert copy");
                packed_[size_t(l)] = static_cast<uint8_t*>(h);
                for (int q = 0; q < S; ++q) ram_pos_[size_t(l)][size_t(ord[size_t(q)])] = q;
            }
            res0_base_[size_t(l)] = static_cast<uint8_t*>(arena_.p) + off0;
            int n1 = 0;
            if (peer_) peer_->base[size_t(l)] = peer_arena.b(off1);
            for (int e = 0; e < c.experts; ++e) {
                if (where_[size_t(l)][size_t(e)] == 1) {
                    uint8_t* dst = static_cast<uint8_t*>(arena_.p) + off0;
                    slot_[size_t(l)][size_t(e)] = int(slots_.size());
                    slots_.push_back({dst, expert_bytes(l), l * c.experts + e, false, nullptr});
                    ord.push_back(e);
                    uploads.push_back({l, e, 1, dst});
                    off0 += stride[size_t(l)];
                } else if (where_[size_t(l)][size_t(e)] == 2) {
                    peer_->pos[size_t(l)][size_t(e)] = n1++;
                    uploads.push_back({l, e, 2, peer_arena.b(off1)});
                    off1 += stride[size_t(l)];
                }
            }
        }
        if (peer_) peer_->arena = std::move(peer_arena);
        if (peer_frac_ > 0) {   // the peer's streamed experts: positions [s0, S) of each layer's packed copy
            peer_->sbase.assign(size_t(c.layers), nullptr);
            peer_->scount.assign(size_t(c.layers), 0);
            peer_->spos.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
            for (int l = 0; l < c.layers; ++l) {
                const int S = n_stream_[size_t(l)], s0 = s0_[size_t(l)];
                if (S == s0) continue;
                peer_->sbase[size_t(l)] = packed_[size_t(l)] + size_t(s0) * stride[size_t(l)];
                peer_->scount[size_t(l)] = S - s0;
                for (int q = s0; q < S; ++q) peer_->spos[size_t(l)][size_t(order_[size_t(l)][size_t(q)])] = q - s0;
            }
        }
        // file reads: resident experts to their GPU, the others into anonymous memory
        std::map<std::string, int> fds;
        auto fd_of = [&](const core::ArtifactTensor& t) {
            auto it = fds.find(t.file->path());
            if (it == fds.end()) {
                const int fd = ::open(t.file->path().c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
                it = fds.emplace(t.file->path(), fd).first;
            }
            return it->second;
        };
        auto read_at = [](int fd, uint8_t* dst, size_t n, uint64_t off) {
            for (size_t got = 0; got < n;) {
                const ssize_t r = pread(fd, dst + got, n - got, off_t(off + got));
                if (r <= 0) throw std::runtime_error("ds41: expert read failed");
                got += size_t(r);
            }
        };
        struct Part { int fd; uint64_t off; size_t bytes; };   // gate, up, down of layer l
        std::vector<std::array<Part, 3>> parts(size_t(c.layers));
        for (int l = 0; l < c.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            int j = 0;
            for (const char* n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const auto& t = art_.at(p + n);
                parts[size_t(l)][size_t(j++)] = {fd_of(t), t.file->data_start() + t.tensor->offset, t.bytes};
            }
        }
        // anonymous copies, laid out as the tensors (untouched pages of resident experts cost nothing)
        for (int l = 0; l < c.layers && packed_on_; ++l) {   // the lane reads expert e at position ram_pos_[l][e]
            auto& L = experts_[size_t(l)];
            const size_t g = L.f.gu_row * size_t(c.ff);
            L.gate = packed_[size_t(l)];
            L.up = packed_[size_t(l)] ? packed_[size_t(l)] + g : nullptr;
            L.down = packed_[size_t(l)] ? packed_[size_t(l)] + 2 * g : nullptr;
            L.gate_stride = L.down_stride = stride[size_t(l)];
            pinned_layer_[size_t(l)] = gu_pin_[size_t(l)] = down_pin_[size_t(l)] = 1;   // no host staging
        }
        for (int l = 0; l < c.layers && !packed_on_; ++l) {
            auto& L = experts_[size_t(l)];
            uint8_t* region[3];
            for (int j = 0; j < 3; ++j) {
                void* m = mmap(nullptr, parts[size_t(l)][size_t(j)].bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (m == MAP_FAILED) throw std::runtime_error("ds41: cannot allocate the expert copy");
                madvise(m, parts[size_t(l)][size_t(j)].bytes, MADV_HUGEPAGE);
                ram_maps_.push_back({m, parts[size_t(l)][size_t(j)].bytes});
                region[j] = static_cast<uint8_t*>(m);
            }
            L.gate = region[0];
            L.up = region[1];
            L.down = region[2];
            stream_src_[size_t(l)] = {region[0], region[1], region[2]};
        }
        const int threads = int(env("STRATA_DS41_RAM_THREADS", 16));
        std::atomic<size_t> next_up{0};
        std::atomic<int> next_layer{0};
        std::atomic<bool> failed{false};
        std::string error;
        std::mutex error_m;
        std::vector<std::thread> workers;
        for (int w = 0; w < threads; ++w)
            workers.emplace_back([&] {
                unpin_helper();
                try {
                    std::vector<uint8_t> buf;
                    for (size_t i; !failed && (i = next_up++) < uploads.size();) {   // resident experts
                        const Task& tk = uploads[i];
                        const auto& L = experts_[size_t(tk.l)];
                        const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
                        buf.resize(2 * g + d);
                        const auto& P = parts[size_t(tk.l)];
                        read_at(P[0].fd, buf.data(), g, P[0].off + size_t(tk.e) * g);
                        read_at(P[1].fd, buf.data() + g, g, P[1].off + size_t(tk.e) * g);
                        read_at(P[2].fd, buf.data() + 2 * g, d, P[2].off + size_t(tk.e) * d);
                        cudaSetDevice(tk.where == 2 ? peer_->dev : 0);
                        ck(cudaMemcpy(tk.dst, buf.data(), buf.size(), cudaMemcpyHostToDevice), "resident upload");
                    }
                    cudaSetDevice(0);
                    for (int l; !failed && (l = next_layer++) < c.layers;) {   // the others, per layer
                        const auto& L = experts_[size_t(l)];
                        const auto& P = parts[size_t(l)];
                        const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
                        for (int e = 0; e < c.experts; ++e) {
                            if (where_[size_t(l)][size_t(e)]) continue;
                            if (packed_on_) {
                                uint8_t* dst = packed_[size_t(l)] + size_t(ram_pos_[size_t(l)][size_t(e)]) * stride[size_t(l)];
                                read_at(P[0].fd, dst, g, P[0].off + size_t(e) * g);
                                read_at(P[1].fd, dst + g, g, P[1].off + size_t(e) * g);
                                read_at(P[2].fd, dst + 2 * g, d, P[2].off + size_t(e) * d);
                                continue;
                            }
                            read_at(P[0].fd, const_cast<uint8_t*>(L.gate) + size_t(e) * g, g, P[0].off + size_t(e) * g);
                            read_at(P[1].fd, const_cast<uint8_t*>(L.up) + size_t(e) * g, g, P[1].off + size_t(e) * g);
                            read_at(P[2].fd, const_cast<uint8_t*>(L.down) + size_t(e) * d, d, P[2].off + size_t(e) * d);
                        }
                        for (const auto& q : P) posix_fadvise(q.fd, off_t(q.off), off_t(q.bytes), POSIX_FADV_DONTNEED);
                    }
                } catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lock(error_m);
                    error = ex.what();
                    failed = true;
                }
            });
        for (auto& w : workers) w.join();
        for (const auto& [path, fd] : fds) ::close(fd);
        if (failed) throw std::runtime_error(error);
        if (packed_mapped_) {   // the NUMA-placed copy: pinned now (portable: both GPUs DMA from it)
            const auto r0 = std::chrono::steady_clock::now();
            for (int l = 0; l < c.layers; ++l)
                if (packed_[size_t(l)])
                    ck(cudaHostRegister(packed_[size_t(l)], size_t(n_stream_[size_t(l)]) * stride[size_t(l)], cudaHostRegisterPortable), "pin the expert copy");
            std::cerr << "DS41 expert copy pinned s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count() << "\n";
        }
        if (std::getenv("STRATA_DS41_LANE_BENCH") && lane_) {   // the CPU lane alone: n tokens on n distinct RAM experts of a layer
            const int l = c.layers / 2;
            std::vector<int> ram;
            for (int e = 0; e < c.experts; ++e) if (!where_[size_t(l)][size_t(e)]) ram.push_back(packed_on_ ? ram_pos_[size_t(l)][size_t(e)] : e);
            std::vector<float> z(size_t(8) * c.hidden, 0.01f), y(size_t(8) * c.hidden), wv(size_t(8) * c.top_k, 0.1f);
            for (int n : {1, 2, 4, 8}) {
                std::vector<int32_t> ids(size_t(n) * c.top_k, -1);
                for (int t = 0; t < n; ++t) ids[size_t(t) * c.top_k] = ram[size_t((t * 7 + 3) % int(ram.size()))];
                const int reps = 50;
                lane_->run(experts_[size_t(l)], z.data(), n, ids.data(), wv.data(), c.top_k, y.data());
                const auto b0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) lane_->run(experts_[size_t(l)], z.data(), n, ids.data(), wv.data(), c.top_k, y.data());
                const double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count() / reps;
                std::cerr << "DS41 lane_bench layer=" << l << " experts=" << n << " ms=" << s1 * 1e3 << " GB/s="
                          << double(n) * expert_bytes(l) / s1 / 1e9 << "\n";
            }
        }
        if (std::getenv("STRATA_DS41_REBALANCE")) rt_count_.assign(size_t(c.layers) * c.experts, 0.0f);
        std::cerr << "DS41 residency gpu0_GiB=" << used0 / double(1ull << 30) << " gpu1_GiB=" << used1 / double(1ull << 30)
                  << " ram_GiB=" << ram / double(1ull << 30) << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << (std::getenv("STRATA_DS41_EXPERT_PROFILE") ? " (profile)" : " (no profile: spread by index)") << "\n";
    }

    // STRATA_DS41_MISS_TEST_DROP=f (a test of MISS_READ): MADV_PAGEOUT on a fraction f of every layer's mapped experts
    // before decode, so the lane meets page-cache misses (mapped experts only: not RAM_COPY's)
    void miss_test_drop() {
        if (lane_) { lane_->miss_reset(); lane_->async_reset(); }
        substituted_ = dropped_ = affinity_swaps_ = 0;
        const char* v = std::getenv("STRATA_DS41_MISS_TEST_DROP");
        if (!v || !ram_pos_.empty()) return;
        const double f = std::atof(v);
        size_t asked = 0;
        for (int l = 0; l < cfg_.layers; ++l)
            if (experts_[size_t(l)].gate) asked += ExpertLane::page_out(experts_[size_t(l)], cfg_.experts, cfg_.hidden, fl_, f, 1234u + unsigned(l));
        std::cerr << "DS41 miss_test_drop fraction=" << f << " paged_out_GiB=" << double(asked) / double(1ull << 30) << "\n";
    }
    void hybrid_line() const {
        if (hybrid_cpu_s_ > 0 || hybrid_skipped_gb_ > 0)
            std::cerr << "DS41 prefill_hybrid cpu_lane_s=" << hybrid_cpu_s_ << " not_copied_GB=" << hybrid_skipped_gb_ << "\n";
    }
    void miss_line(double tokens) const {
        if (substituted_ || dropped_ || affinity_swaps_)
            std::cerr << "DS41 lossy_routes gpu_subst_per_token=" << double(substituted_) / std::max(tokens, 1.0)
                      << " cpu_dropped_per_token=" << double(dropped_) / std::max(tokens, 1.0)
                      << " affinity_swaps_per_token=" << double(affinity_swaps_) / std::max(tokens, 1.0) << "\n";
        if (!lane_) return;
        if (lane_->async_jobs())
            std::cerr << "DS41 lane_async jobs=" << lane_->async_jobs() << " busy_ms_per_token=" << lane_->async_busy_s() * 1e3 / std::max(tokens, 1.0) << "\n";
        const auto s = lane_->miss_stats();
        if (s.calls == 0) return;
        std::cerr << "DS41 miss_read calls=" << s.calls << " cold=" << s.cold << " read_pieces=" << s.read_pieces
                  << " read_MB_per_token=" << double(s.read_bytes) / 1e6 / std::max(tokens, 1.0) << " reader_s=" << s.read_s
                  << " lane_wait_ms_per_token=" << s.wait_s * 1e3 / std::max(tokens, 1.0) / std::max(1, lane_->threads()) << "\n";
    }

    /// After a long prompt: the chunk buffers shrink to STRATA_DS41_DECODE_CHUNK (64) and the GPU expert tier takes
    /// the VRAM, seeded from the profile file or else from the prompt's own routing.
    void prefill_done() {
        if (peer_) { std::fill(std::begin(peer_->split_ms), std::end(peer_->split_ms), 0.0); peer_->split_on = false; }   // decode's own split
        // (DECODE_GROUPED computes grown experts with the resident ones; DECODE_MMQ by GEMV chains after its product)
        static const bool grow_on = std::getenv("STRATA_DS41_GROW") && std::string(std::getenv("STRATA_DS41_GROW")) == "1" &&
                                    ((std::getenv("STRATA_DS41_DECODE_GROUPED") && std::string(std::getenv("STRATA_DS41_DECODE_GROUPED")) == "1") ||
                                     (std::getenv("STRATA_DS41_DECODE_MMQ") && std::string(std::getenv("STRATA_DS41_DECODE_MMQ")) == "1"));
        // STRATA_DS41_REFILL=1 (RAM_COPY, REBALANCE): after the prompt, both GPUs' prompt-only buffers are released (GPU 0's
        // pass buffers and stream ring, the peer's stream slots and wide buffers) and the VRAM refilled with copies of
        // the prompt's most-routed RAM experts: GPU 0 by grow() (where_ 3), the peer by grow_peer() (where_ 4)
        static const bool refill = std::getenv("STRATA_DS41_REFILL") && std::string(std::getenv("STRATA_DS41_REFILL")) == "1";
        if (refill) {
            ck(cudaStreamSynchronize(stream_), "refill");
            if (chunk_ > 128) resize(128);
            if (peer_) peer_->release_prompt(128, 512);
            grow();
            grow_peer();
        } else if (grow_on) grow();
        rebalance();
        if (!tier_lent_) return;
        ck(cudaStreamSynchronize(stream_), "sync");
        const char* v = std::getenv("STRATA_DS41_DECODE_CHUNK");
        resize(v ? std::max(1, std::atoi(v)) : 64);
        tier_lent_ = false;
        load_tier();
        // decode's own counters from here
        tier_hits_ = tier_all_ = 0;
        std::fill(std::begin(prof_), std::end(prof_), 0.0);
    }

    /// (Re)allocates every buffer sized by the tokens of one pass; the window ring keeps its last rows.
    void resize(int tokens) {
        const auto& c = cfg_;
        const size_t T = size_t(tokens), H = size_t(c.hidden);
        chunk_ = tokens;
        // STRATA_DS41_MIX_TILE=n: the hyper-connection mixes run n tokens at a time (flat_ / mixes_ hold n rows)
        mix_tile_ = std::getenv("STRATA_DS41_MIX_TILE") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_MIX_TILE"))) : 0;
        // STRATA_DS41_PREFILL_HALVES=1 with STRATA_DS41_PREFILL_TILE (4096): a prompt pass wider than the tile runs each
        // layer's trunk over the tiles and its FFN once over the whole pass (one expert stream per pass). The
        // attention-local buffers hold A = tile rows; the residual / FFN arrays stay at the pass's T.
        halves_ = std::getenv("STRATA_DS41_PREFILL_HALVES") && std::string(std::getenv("STRATA_DS41_PREFILL_HALVES")) == "1";
        pftile_ = std::getenv("STRATA_DS41_PREFILL_TILE") ? std::max(256, std::atoi(std::getenv("STRATA_DS41_PREFILL_TILE"))) : 4096;
        halves_skip_from_ = std::getenv("STRATA_DS41_HALVES_SKIP_FROM") ? std::atoi(std::getenv("STRATA_DS41_HALVES_SKIP_FROM")) : -1;
        const size_t A = halves_ && size_t(pftile_) < T ? size_t(pftile_) : T;
        (void) A;
        // the window ring: slot = position % ring; re-laid when its size changes. A prompt pass writes one attention
        // batch (STRATA_DS41_ATTN_SUB, 256) at a time, so it needs window + that many rows, not window + the pass.
        const int ring = c.window + int(std::min<size_t>(T, size_t(attn_sub_)));
        std::vector<Dev> windows(size_t(c.layers));
        for (auto& w : windows) {   // zeroed: enc-skip's dropped tokens may read slots never written
            w = Dev(size_t(ring) * c.head_dim * 4);
            ck(cudaMemsetAsync(w.p, 0, w.bytes, stream_), "ring");
        }
        if (!windows_.empty()) {
            const int next = int(history_.size());
            for (int l = 0; l < c.layers; ++l)
                for (int p = std::max(0, next - c.window); p < next; ++p)
                    ck(cudaMemcpyAsync(windows[size_t(l)].f(size_t(p % ring) * c.head_dim), windows_[size_t(l)].f(size_t(p % ring_) * c.head_dim),
                                       size_t(c.head_dim) * 4, cudaMemcpyDeviceToDevice, stream_), "ring");
            ck(cudaStreamSynchronize(stream_), "ring");
        }
        windows_ = std::move(windows);
        ring_ = ring;
        // free first, then allocate (VRAM is tight)
        for (Dev* d : {&h_, &h2_, &ht_, &h2t_, &yt_, &xnt_, &coeff_ffn_full_, &x_, &xn_, &flat_, &mixes_, &coeff_attn_, &coeff_ffn_, &qa_, &qr_, &q_, &kv_, &kvn_, &att_, &oa_, &oa_part_, &y_,
                       &q8_, &q8g_, &cs_, &comp_v_, &comp_s_, &iq_, &iw_, &logits_, &sh_gate_, &sh_up_, &sh_h_, &moe_, &scores_,
                       &engram_rows_, &engram_kv_, &xh_, &g_sum_, &stage_[0], &stage_[1], &s_x_, &s_xq_, &s_gate_, &s_up_, &s_h_, &s_hq_,
                       &s_out_, &s_idx_, &s_w_, &s_bound_, &m_xq_, &s_gu_, &s_hf_, &m_hq_, &s_dout_, &s_iota_})
            *d = Dev();
        for (auto& d : sel_) d = Dev();
        for (auto& d : sel_cnt_) d = Dev();
        cand_dev_ = Dev();
        pinned_bufs_.clear();
        h_ = Dev(T * 4 * H * 4);
        h2_ = Dev(T * 4 * H * 4);
        if (A < T) {   // the trunk's views: swapped in for one tile at a time
            ht_ = Dev(A * 4 * H * 4);
            h2t_ = Dev(A * 4 * H * 4);
            yt_ = Dev(A * H * 4);
            xnt_ = Dev(A * H * 4);
        }
        x_ = Dev(T * H * 4);
        xn_ = Dev(T * H * 4);
        flat_ = Dev(std::min(A, mix_tile_ > 0 ? size_t(mix_tile_) : A) * 4 * H * 4);
        mixes_ = Dev(std::min(A, mix_tile_ > 0 ? size_t(mix_tile_) : A) * 24 * 4);
        coeff_attn_ = Dev(A * 24 * 4);
        coeff_ffn_ = Dev(A * 24 * 4);
        if (A < T) coeff_ffn_full_ = Dev(T * 24 * 4);   // the FFN post runs over the whole pass
        qa_ = Dev(A * size_t(c.q_rank) * 4);
        qr_ = Dev(A * size_t(c.q_rank) * 4);
        q_ = Dev(A * size_t(c.heads) * c.head_dim * 4);
        kv_ = Dev(A * size_t(c.head_dim) * 4);
        kvn_ = Dev(A * size_t(c.head_dim) * 4);
        att_ = Dev(A * size_t(c.heads) * c.head_dim * 4);
        oa_ = Dev(A * size_t(c.out_groups) * c.out_rank * 4);
        oa_part_ = Dev(A * size_t(c.out_rank) * 4);
        y_ = Dev(T * H * 4);
        // q8_ only holds passes shorter than gemm_min (all other widths go through xh_), so a wide pass does not size it
        q8_ = Dev(k::native_q8_1_bytes(std::max(c.heads * c.head_dim, 5 * c.hidden), 1) * std::min<size_t>(T, std::max(64, gemm_min_)));
        q8x_ = Dev(k::native_q8_1_bytes(c.hidden, 1) * A);
        q8g_ = Dev(k::native_q8_1_bytes(c.heads / c.out_groups * c.head_dim, 1) * A);
        cs_ = Dev(T * kTables * 64 * 4);
        rope_host_.assign(T * kTables * 64, 0.0f);
        comp_v_ = Dev(A * size_t(c.head_dim) * 4);
        comp_s_ = Dev(A * size_t(c.head_dim) * 4);
        iq_ = Dev(A * size_t(c.index_heads) * c.index_dim * 4);
        iw_ = Dev(A * size_t(c.index_heads) * 4);
        logits_ = Dev(std::max(T * size_t(c.experts), size_t(c.vocab)) * 4);   // the router's, or one position's logits (more on request)
        // STRATA_DS41_SHEXP_TILE=n: the shared expert's intermediates hold n rows (a 8192-token pass saves 0.2 GiB at 1024)
        shexp_tile_ = std::getenv("STRATA_DS41_SHEXP_TILE") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_SHEXP_TILE"))) : 0;
        const size_t SH = std::min(A, shexp_tile_ > 0 ? std::min<size_t>(A, size_t(shexp_tile_)) : A);
        sh_rows_ = SH;
        sh_gate_ = Dev(SH * size_t(c.ff) * 4);
        sh_up_ = Dev(SH * size_t(c.ff) * 4);
        sh_h_ = Dev(SH * size_t(c.ff) * 4);
        moe_ = Dev(T * H * 4);
        // the indexer's scores: score_tile_ tokens at a time (STRATA_DS41_SCORE_MIB, 1024: a pass of T tokens at a long
        // context would need T x context floats, 17 GB for 4096 tokens at 1M)
        {
            const size_t budget = size_t(std::getenv("STRATA_DS41_SCORE_MIB") ? std::max(16, std::atoi(std::getenv("STRATA_DS41_SCORE_MIB"))) : 1024) << 20;
            score_tile_ = int(std::max<size_t>(16, std::min<size_t>(A, budget / (size_t(score_stride_) * 4))));
            scores_ = Dev(size_t(score_tile_) * size_t(score_stride_) * 4);
        }
        if (kv_half_) {
            const size_t rw = size_t(row_width()), g = std::min<size_t>(A, kGatherTokens);
            kv_rows_ = Dev(A * rw * 4);
            kv_sel_ = Dev(g * size_t(c.index_top_k) * rw * 4);
            std::vector<int> io(g * size_t(c.index_top_k));
            for (size_t i = 0; i < io.size(); ++i) io[i] = int(i);
            sel_iota_ = Dev(io.size() * 4);
            ck(cudaMemcpy(sel_iota_.p, io.data(), io.size() * 4, cudaMemcpyHostToDevice), "sel iota");
        }
        sel_.resize(size_t(c.layers));
        sel_count_.assign(size_t(c.layers), std::vector<int>(halves_ ? T : A, 0));
        for (int l = 0; l < c.layers; ++l)
            if (c.topk_src[size_t(l)] == l) sel_[size_t(l)] = Dev((halves_ ? T : A) * size_t(c.index_top_k) * 4);
        sel_cnt_.resize(size_t(c.layers));
        sel_on_gpu_.assign(size_t(c.layers), 0);
        for (int l = 0; l < c.layers; ++l)
            if (c.topk_src[size_t(l)] == l) sel_cnt_[size_t(l)] = Dev(T * 4);
        cand_stride_ = c.candidate_block > 0 ? (score_stride_ + c.candidate_block - 1) / c.candidate_block : 1;
        cand_dev_ = Dev(A * size_t(cand_stride_));
        candidates_.assign(A, {});
        if (!c.engram_layers.empty()) {
            engram_rows_ = Dev(c.engram_layers.size() * T * c.engram_cols() * c.engram_dim * 4);
            if (engram_host_pinned_) { cudaHostUnregister(engram_host_.data()); engram_host_pinned_ = false; }
            engram_host_.assign(c.engram_layers.size() * T * c.engram_cols() * c.engram_dim, 0.0f);
            // STRATA_DS41_ENGRAM_PINNED=1: the rows' host buffer pinned, so their uploads do not wait for the stream
            if (std::getenv("STRATA_DS41_ENGRAM_PINNED") && std::string(std::getenv("STRATA_DS41_ENGRAM_PINNED")) == "1")
                engram_host_pinned_ = cudaHostRegister(engram_host_.data(), engram_host_.size() * 4, 0) == cudaSuccess || ((void)cudaGetLastError(), false);
            // STRATA_DS41_ENGRAM_TILE=n: the e-th Engram's projection runs n tokens at a time (its scratch is n rows
            // instead of the pass's; a 8192-token pass saves 0.75 GiB at n = 256)
            engram_tile_ = std::getenv("STRATA_DS41_ENGRAM_TILE") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_ENGRAM_TILE"))) : 0;
            engram_kv_ = Dev((engram_tile_ > 0 ? std::min(A, size_t(engram_tile_)) : A) * 5 * H * 4);
        }
        // STRATA_DS41_OUTA_TILE=n: the attention output projection's activation conversion (T x heads x head_dim halves,
        // 0.5 GiB at 8192) runs n rows at a time, so xh_ holds the widest single call instead of the whole pass.
        outa_tile_ = std::getenv("STRATA_DS41_OUTA_TILE") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_OUTA_TILE"))) : 0;
        if (tokens >= gemm_min_) {
            const size_t wide = std::max(T * H, (outa_tile_ > 0 ? std::min<size_t>(T, size_t(outa_tile_)) : T) *
                                                   (size_t(c.heads) * c.head_dim + size_t(c.heads / c.out_groups) * c.head_dim));
            xh_ = Dev(wide * 2);
        }
        g_sum_ = Dev(T * H * 4);
        attn_info_dev_ = Dev(A * 16);
        attn_info_ = reinterpret_cast<int*>(pinned(A * 16));
        attn_info_host_bytes_ = A * 16;
        if (attn_bounds_check_ && !attn_error_.p) attn_error_ = Dev(sizeof(int));
        pinned_[0] = pinned(T * H * 4);
        pinned_[1] = pinned(T * H * 4);
        pinned_[2] = pinned(T * H * 4);
        dring_.clear();
        if (tokens >= stream_min_) {   // the streamed experts' rings: VRAM slots from what is left, pinned host slots
            slot_bytes_ = stage_bytes_;
            // (not with RAM_COPY + RAM_PINNED: every layer is pinned and streams from the packed copy, so the 6 x 256 MiB
            // of pinned host staging would never be used; stream_begin allocates it if a layer turns out unpinned)
            static const bool all_pinned = ram_copy_ && std::getenv("STRATA_DS41_RAM_PINNED") && std::string(std::getenv("STRATA_DS41_RAM_PINNED")) == "1";
            if (hring_.empty() && !all_pinned) alloc_hring();
            if (remote_ && remote_gpu_) {   // the worker's partial sums; h's columns [fl_, ff) stay zero (local rows only)
                remote_sum_ = Dev(T * H * 4);
            }
            if (remote_ && !remote_gpu_ && rbuf_.empty()) {   // the worker's part of an item: at most slot_bytes_ worth of experts
                size_t most = 0;
                for (int l = 0; l < c.layers; ++l)
                    most = std::max(most, slot_bytes_ / std::max<size_t>(1, stream_stride(l)) * remote_->geo[size_t(l)].stride);
                // STRATA_DS41_FETCH_BUFS (16): the fetches run this many items ahead of the uploads (the link stays
                // busy while the VRAM slots wait for the compute)
                const char* fb = std::getenv("STRATA_DS41_FETCH_BUFS");
                const int bufs = fb ? std::max(2, std::atoi(fb)) : 16;
                for (int i = 0; i < bufs; ++i) {
                    void* hb = nullptr;
                    ck(cudaHostAlloc(&hb, most, cudaHostAllocDefault), "fetch buffer");
                    rbuf_.push_back(static_cast<uint8_t*>(hb));
                    ucomm_mr_t* region = nullptr;
                    if (ucomm_mr_reg(remote_->comm, hb, most, &region) == UCOMM_OK) rbuf_regions_.push_back(region);
                    cudaEvent_t e;
                    ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
                    ev_rdma_.push_back(e);
                }
                rscratch_ = Dev(most);
            }
            s_x_ = Dev(size_t(8) * H * 4);
            s_xq_ = Dev(k::native_q8_1_bytes(c.hidden, 8));
            s_gate_ = Dev(size_t(8) * c.ff * 4);
            s_up_ = Dev(size_t(8) * c.ff * 4);
            s_h_ = Dev(size_t(8) * c.ff * 4);
            s_hq_ = Dev(k::native_q8_1_bytes(c.ff, 8));
            s_out_ = Dev(size_t(8) * H * 4);
            s_idx_ = Dev(T * size_t(c.top_k) * 4);
            s_w_ = Dev(T * size_t(c.top_k) * 4);
            // An expert can exceed the row cap and be split into several products. Each split
            // adds two bounds; 256 is the smallest permitted cap.
            const size_t max_bounds = size_t(c.experts) * 2 + 8 + 2 * ((T * size_t(c.top_k) + 255) / 256);
            s_bound_ = Dev(max_bounds * 4);
            s_hidx_ = reinterpret_cast<int*>(pinned(T * size_t(c.top_k) * 8 + max_bounds * 4));
            s_hw_ = reinterpret_cast<float*>(s_hidx_ + T * size_t(c.top_k));
            s_hbound_ = reinterpret_cast<int32_t*>(s_hw_ + T * size_t(c.top_k));
            rcap_ = std::max(rcap_min_, std::min(int(T), 4096));   // larger experts are split across products
            if (use_mmq_) {
                namespace mq = strata::prefill::mmq;
                m_xq_ = Dev(mq::q8_bytes(rcap_, c.hidden));
                s_gu_ = Dev(size_t(rcap_) * 2 * c.ff * 4);
                s_hf_ = Dev(size_t(rcap_) * c.ff * 4);
                if (remote_ && remote_gpu_) ck(cudaMemsetAsync(s_hf_.p, 0, s_hf_.bytes, stream_), "padded h");
                m_hq_ = Dev(mq::q8_bytes(rcap_, c.ff));
                s_dout_ = Dev(size_t(rcap_) * H * 4);
                s_iota_ = Dev((size_t(rcap_) + 128) * 4);
                mq::iota(s_iota_.i(), rcap_ + 128, stream_);
            }
            size_t free_b = 0, total_b = 0;
            ck(cudaMemGetInfo(&free_b, &total_b), "memory info");
            const char* dv = std::getenv("STRATA_DS41_STAGE_SLOTS");
            const char* dm = std::getenv("STRATA_DS41_DESKTOP_MIB");
            const size_t keep = std::max<size_t>(size_t(dm ? std::max(0, std::atoi(dm)) : 0) << 20, vram_reserve);
            // the desktop reserve is a hard floor: if the configured slot size leaves no room for two of them, halve the
            // slots (down to 32 MiB) rather than take the desktop's VRAM
            // beyond the reserve, leave the driver / runtime scratch (the MMQ stream-k fixup pool grows to ~256 MiB on
            // the first wide expert product, and the runtime allocates a little more)
            const size_t margin = size_t(640) << 20;
            auto fit_for = [&](size_t bytes) { return free_b > keep + margin ? (free_b - keep - margin) / bytes : 0; };
            while (fit_for(slot_bytes_) < 2 && slot_bytes_ > (size_t(32) << 20)) slot_bytes_ = std::max<size_t>(size_t(32) << 20, slot_bytes_ / 2);
            const size_t fit = fit_for(slot_bytes_);
            dslots_ = int(std::min<size_t>(dv ? size_t(std::max(2, std::atoi(dv))) : 8, fit));
            if (dslots_ < 2)
                throw std::runtime_error("ds41: no VRAM for the streamed experts' slots (shorter STRATA_DS41_PROMPT_BATCH; " +
                                         std::to_string(free_b >> 20) + " MiB free, " + std::to_string(keep >> 20) +
                                         " MiB kept for the desktop, slots are " + std::to_string(slot_bytes_ >> 20) + " MiB each)");
            for (int i = 0; i < dslots_; ++i) dring_.emplace_back(slot_bytes_);
            for (auto* v : {&ev_ready_, &ev_free_}) {
                for (auto e : *v) cudaEventDestroy(e);
                v->clear();
                for (int i = 0; i < dslots_; ++i) {
                    cudaEvent_t e;
                    ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
                    v->push_back(e);
                }
            }
            std::cerr << "DS41 stream slots VRAM=" << dslots_ << " host=" << hslots_ << " x " << (slot_bytes_ >> 20) << " MiB\n";
        }
    }

    void alloc_hring() {
        const char* hv = std::getenv("STRATA_DS41_STAGE_HOST_SLOTS");
        hslots_ = hv ? std::max(2, std::atoi(hv)) : 6;
        for (int i = 0; i < hslots_; ++i) {
            void* hs = nullptr;
            ck(cudaHostAlloc(&hs, slot_bytes_, cudaHostAllocDefault), "host stage");
            hring_.push_back(static_cast<uint8_t*>(hs));
            cudaEvent_t e;
            ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
            ev_dma_.push_back(e);
        }
    }
    // pinned host memory owned by the decoder (released by resize)
    // STRATA_DS41_VERIFY_EXACT=1 rounds a multi-token pass's trunk (dense GEMVs, the fused pre-mix) one column at a
    // time, as the one-token passes do; STRATA_DS41_VERIFY_FAST_TRUNK=1 (opt-in) keeps the exact expert path but lets
    // the trunk batch its columns (not bitwise equal to the one-token passes)
    static bool trunk_exact() {
        static const bool e = strata_env_is_one("STRATA_DS41_VERIFY_EXACT") && !strata_env_is_one("STRATA_DS41_VERIFY_FAST_TRUNK");
        return e;
    }
    // pinned memory that resize() keeps (the drafter's, the decode tables')
    std::vector<std::unique_ptr<void, void (*)(void*)>> keep_bufs_;
    float* pinned_keep(size_t bytes) {
        void* h = nullptr;
        ck(cudaHostAlloc(&h, std::max<size_t>(bytes, 16), cudaHostAllocPortable), "pinned");
        keep_bufs_.emplace_back(h, [](void* q) { cudaFreeHost(q); });
        return static_cast<float*>(h);
    }
    float* pinned(size_t bytes) {
        void* h = nullptr;
        ck(cudaHostAlloc(&h, std::max<size_t>(bytes, 16), cudaHostAllocPortable), "pinned");   // (portable: the peer GPU reads some)
        pinned_bufs_.emplace_back(h, [](void* q) { cudaFreeHost(q); });
        return static_cast<float*>(h);
    }
    ~Decoder() {
        if (ost_comm_) {
            const int32_t end[4] = {-1, 0, 0, 0};
            (void)ucomm_send(ost_comm_, 1, 60, end, sizeof end);
            for (auto* mr : ost_mrs_) ucomm_mr_dereg(mr);
            ucomm_finalize(ost_comm_);
            ost_comm_ = nullptr;
        }
        adapt_stop();
        promote_stop();
        // the promotions' uploader first: its queued copies read the mapped experts unmapped below
        if (uploader_.joinable()) {
            { std::lock_guard<std::mutex> lock(jobs_mutex_); stop_uploads_ = true; }
            jobs_cv_.notify_all();
            uploader_.join();
        }
        { std::lock_guard<std::mutex> lock(pf_m_); pf_stop_ = true; }
        pf_cv_.notify_all();
        for (auto& t : pf_threads_) t.join();
        if (engram_later_.joinable()) engram_later_.join();
        stream_end();
        // Registered DSpark weight pages remain valid until all transfers on the decode stream finish.
        if (stream_) (void)cudaStreamSynchronize(stream_);
        for (auto* r : rbuf_regions_) ucomm_mr_dereg(r);
        for (const auto& [m, n] : ram_maps_) munmap(m, n);
        for (void* r : registered_) cudaHostUnregister(r);
        for (const auto& [file, m] : shared_maps_) munmap(m.first, m.second);
        for (int fd : owned_fds_) ::close(fd);
        for (auto e : peer_late_ev_) if (e) cudaEventDestroy(e);
        if (router_ev_) cudaEventDestroy(router_ev_);
        if (stage_cpu_copy_done_) cudaEventDestroy(stage_cpu_copy_done_);
        if (stage_cpu_expert_)
            std::cerr << "DS41 staged_cpu_experts=" << stage_cpu_transfers_ << " checked_layers="
                      << std::count(stage_cpu_checked_.begin(), stage_cpu_checked_.end(), uint8_t(1)) << "\n";
        if (stage_cpu_timing_)
            std::cerr << "DS41 stage_timing groups=" << stage_timed_groups_ << " reuse_wait_s=" << stage_timing_[0]
                      << " host_pack_s=" << stage_timing_[1] << " gate_wait_s=" << stage_timing_[2]
                      << " host_activation_s=" << stage_timing_[3] << " final_wait_collect_s=" << stage_timing_[4] << "\n";
        if (stage_cpu_batch_act_)
            std::cerr << "DS41 stage_batch_activation groups=" << stage_batch_groups_ << " routes=" << stage_batch_routes_ << "\n";
        if (stage_cache_slots_requested_)
            std::cerr << "DS41 stage_cache slots=" << stage_cache_.size() << " hits=" << stage_cache_hits_
                      << " misses=" << stage_cache_misses_ << "\n";
        if (stream_) cudaStreamDestroy(stream_);
    }

    const Config& config() const { return cfg_; }

    /// STRATA_DS41_GPU_BENCH=1: the GPU expert kernels of both GPUs on the same synthetic routes of one layer (the real
    /// resident experts), then the program exits: raw copy bandwidth, one expert's gate GEMV, and per call the MMQ
    /// product, the GEMV chains and the grouped kernel on GPU 0 and on the peer GPU. GPU time by events on each device's
    /// own stream; "host" is the wall time of enqueue + wait.
    void gpu_bench() {
        const auto& c = cfg_;
        const int l = c.layers / 2, K = c.top_k, T = 6;
        const auto& L = experts_[size_t(l)];
        if (std::string(std::getenv("STRATA_DS41_GPU_BENCH")) == "dense") {   // the trunk's one-token GEMVs of one layer
            const std::string p = "blk." + std::to_string(l) + ".";
            std::vector<float> xh(size_t(1) << 15, 0.01f);
            Dev xd(xh.size() * 4), yd(size_t(1) << 18);
            ck(cudaMemcpy(xd.p, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "bench x");
            double all_ms = 0, all_b = 0;
            for (const char* n : {"attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
                                  "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}) {
                const Weight& W = weight(p + n);
                if (W.type == GGML_TYPE_F32 || W.type == GGML_TYPE_BF16) continue;
                quantize(xd.f(), q8_.p, W.in, 1);
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                mmvq(W.type, static_cast<const uint8_t*>(W.data.p), q8_.p, yd.f(), W.in, W.out, 1);
                cudaEventRecord(a, stream_);
                const int reps = 200;
                for (int r = 0; r < reps; ++r) mmvq(W.type, static_cast<const uint8_t*>(W.data.p), q8_.p, yd.f(), W.in, W.out, 1);
                cudaEventRecord(b, stream_);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                ms /= reps;
                const double bytes = double(W.row_bytes) * W.out;
                all_ms += ms;
                all_b += bytes;
                std::cerr << "DS41 gpu_bench dense " << n << " type=" << W.type << " [" << W.out << " x " << W.in << "] us=" << ms * 1e3
                          << " GB/s=" << bytes / (ms * 1e-3) / 1e9 << "\n";
            }
            std::cerr << "DS41 gpu_bench dense layer=" << l << " us=" << all_ms * 1e3 << " GB/s=" << all_b / (all_ms * 1e-3) / 1e9 << "\n";
            return;
        }
        auto attrs = [&](int dev) {
            int clk = 0, mclk = 0, bus = 0, sm = 0;
            cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
#if !defined(__HIP_PLATFORM_AMD__)
            cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
            cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
#endif
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
            char pci[32] = {};
            cudaDeviceGetPCIBusId(pci, sizeof pci, dev);
            std::cerr << "DS41 gpu_bench dev=" << dev << " pci=" << pci << " sm=" << sm << " clock_MHz=" << clk / 1000 << " mem_clock_MHz=" << mclk / 1000
                      << " bus_bits=" << bus << "\n";
        };
        // GPU time of `reps` calls of fn on (dev, stream), ms per call
        auto gpu_ms = [&](int dev, cudaStream_t st, int reps, const std::function<void()>& fn) {
            cudaSetDevice(dev);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            fn();
            cudaStreamSynchronize(st);
            cudaEventRecord(a, st);
            for (int r = 0; r < reps; ++r) fn();
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            cudaSetDevice(0);
            return ms / reps;
        };
        // prompt shapes: one GPU-0 resident expert of the layer, M tokens through its gate/up [2 ff x hidden] - MMQ (q8_1,
        // dp4a on sm_70) against a dequantization to FP16 + a cuBLAS tensor-core GEMM (the down matrix likewise)
        auto bench_prefill = [&](const uint8_t* wexp) {
            namespace mq = strata::prefill::mmq;
            const size_t g = slice_gate(l);
            const int Mmax = 256;
            Dev xf(size_t(Mmax) * std::max(c.hidden, c.ff) * 4), xq(mq::q8_bytes(Mmax, std::max(c.hidden, c.ff))),
                dst(size_t(Mmax) * std::max(c.hidden, 2 * c.ff) * 4),
                w16(size_t(2) * c.ff * c.hidden * 2), xh(size_t(Mmax) * std::max(c.hidden, c.ff) * 2), idsd((size_t(Mmax) + 128) * 4), bnd(64);
            std::vector<float> xr(size_t(Mmax) * std::max(c.hidden, c.ff));
            std::vector<uint16_t> hr(size_t(Mmax) * std::max(c.hidden, c.ff));
            uint32_t rs = 777u;
            for (auto& v : xr) { rs = rs * 1664525u + 1013904223u; v = (float(rs >> 8) / 16777216.0f - 0.5f) * 2.0f; }
            for (auto& v : hr) { rs = rs * 1664525u + 1013904223u; v = uint16_t(0x3400 + ((rs >> 16) & 0x3ff)) | uint16_t((rs >> 15) & 0x8000); }
            ck(cudaMemcpy(xf.p, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice), "bench x");
            ck(cudaMemcpy(xh.p, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice), "bench xh");
            mq::iota(idsd.i(), Mmax + 128, stream_);
            const float one = 1.0f, zero = 0.0f;
            for (int part = 0; part < 2; ++part) {   // 0: gate/up, 1: down
                const int ty = part ? L.f.d_type : L.f.gu_type;
                const int rows = part ? c.hidden : 2 * c.ff, cols = part ? c.ff : c.hidden;
                const uint8_t* wm = part ? wexp + 2 * g : wexp;
                const double dq = gpu_ms(0, stream_, 20, [&] { k::iq_dequant_f16(ty, wm, int64_t(rows) * cols, w16.h(), stream_); });
                std::cerr << "DS41 gpu_bench prefill " << (part ? "down" : "gate/up") << " type=" << ty << " [" << rows << " x " << cols
                          << "] dequant_f16_ms=" << dq;
                for (int M : {16, 32, 64, 128, 256}) {
                    const int32_t b[2] = {0, M};
                    ck(cudaMemcpy(bnd.p, b, sizeof b, cudaMemcpyHostToDevice), "bench bounds");
                    const double mm = gpu_ms(0, stream_, 20, [&] {
                        mq::quantize(xf.f(), nullptr, xq.p, ty, cols, cols, M, stream_);
                        mq::Product pr;
                        pr.w = wm; pr.type = ty; pr.w_rows = rows; pr.w_cols = cols; pr.expert_bytes = mq::matrix_bytes(ty, rows, cols); pr.n = 1;
                        pr.xq = xq.p; pr.bounds = bnd.i(); pr.ids = idsd.i(); pr.total_rows = M; pr.max_rows = M; pr.dst = dst.f(); pr.ld_dst = rows;
                        pr.dst_rows = M;
                        mmq_->run(pr, stream_);
                    });
                    const double gm = gpu_ms(0, stream_, 20, [&] {
                        if (cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, M, cols, &one, w16.p, CUDA_R_16F, cols, xh.p, CUDA_R_16F, cols, &zero,
                                         dst.p, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                            throw std::runtime_error("ds41: bench GEMM");
                    });
                    std::cerr << " M" << M << ":mmq=" << mm << ",f16gemm=" << gm;
                }
                std::cerr << "\n";
            }
        };
        if (std::string(std::getenv("STRATA_DS41_GPU_BENCH")) == "prefill") {
            if (tier_lent_ || slot_.empty() || !use_mmq_ || !mmq_)
                throw std::runtime_error("ds41: prefill GPU bench needs a loaded tier and MMQ; use a prompt shorter than STRATA_DS41_STREAM_MIN");
            for (int e = 0; e < c.experts; ++e) {
                const int si = slot_[size_t(l)][size_t(e)];
                if (si >= 0) { bench_prefill(slots_[size_t(si)].p); return; }
            }
            throw std::runtime_error("ds41: prefill GPU bench has no resident expert in its test layer");
        }
        if (where_.empty() || res0_base_.empty())
            throw std::runtime_error("ds41: GPU bench requires contiguous resident experts; use STRATA_DS41_GPU_BENCH=prefill for a loaded tier");
        std::vector<int> devs{0};
        if (peer_) devs.push_back(peer_->dev);
        for (int dev : devs) {
            attrs(dev);
            cudaSetDevice(dev);
            cudaStream_t st;
            ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "bench stream");
            try {   // raw device copy (64 MiB: the VRAM is full of experts)
                const size_t n = size_t(64) << 20;
                Dev a(n), b(n);
                const double ms = gpu_ms(dev, st, 50, [&] { cudaMemcpyAsync(b.p, a.p, n, cudaMemcpyDeviceToDevice, st); });
                std::cerr << "DS41 gpu_bench dev=" << dev << " d2d_GBps=" << 2.0 * n / (ms * 1e-3) / 1e9 << "\n";
            } catch (const std::exception& e) {
                std::cerr << "DS41 gpu_bench dev=" << dev << " d2d skipped: " << e.what() << "\n";
            }
            cudaSetDevice(dev);
            // one resident expert's gate rows: native_mmvq at 1 and 6 columns
            const uint8_t* w = nullptr;
            if (dev == 0) {
                const int S = n_stream_[size_t(l)];
                if (int(order_[size_t(l)].size()) > S) w = res0_base_[size_t(l)];
            } else {
                w = peer_->base[size_t(l)];
            }
            if (w) {
                Dev xq(k::native_q8_1_bytes(c.hidden, 8)), y(size_t(8) * c.ff * 4), x(size_t(8) * c.hidden * 4);
                cudaMemset(x.p, 0, x.bytes);
                k::native_quantize_q8_1(x.f(), xq.p, c.hidden, 8, st);
                for (int nc : {1, 6}) {
                    const double ms = gpu_ms(dev, st, 200, [&] { k::native_mmvq(L.f.gu_type, w, xq.p, y.f(), c.hidden, c.ff, nc, st); });
                    std::cerr << "DS41 gpu_bench dev=" << dev << " gate_gemv cols=" << nc << " us=" << ms * 1e3 << " GBps="
                              << double(L.f.gu_row) * c.ff / (ms * 1e-3) / 1e9 << "\n";
                }
            }
            cudaSetDevice(0);
            cudaStreamDestroy(st);
        }
        // the decode paths on synthetic routes: `routes` of T x K route slots, each to a distinct expert resident on the GPU
        std::vector<int> e0, e1;
        for (int e = 0; e < c.experts; ++e) {
            if (where_[size_t(l)][size_t(e)] == 1) e0.push_back(e);
            if (where_[size_t(l)][size_t(e)] == 2) e1.push_back(e);
        }
        std::cerr << "DS41 gpu_bench layer=" << l << " resident gpu0=" << e0.size() << " gpu1=" << e1.size() << "\n";
        ck(cudaMemsetAsync(xn_.p, 0, size_t(T) * c.hidden * 4, stream_), "bench x");
        std::fill(pinned_[0], pinned_[0] + size_t(T) * c.hidden, 0.0f);
        ck(cudaStreamSynchronize(stream_), "bench");
        const bool timing0 = peer_ ? peer_->timing : false;
        if (peer_) peer_->timing = true;
        for (int routes : {4, 8, 16, 36}) {
            std::vector<int> ids(size_t(T) * K, -1), ids1(size_t(T) * K, -1);
            std::vector<float> wts(size_t(T) * K, 0.1f);
            for (int r = 0; r < routes && r < T * K; ++r) {
                if (!e0.empty()) ids[size_t(r)] = e0[size_t(r * 7) % e0.size()];
                if (!e1.empty()) ids1[size_t(r)] = e1[size_t(r * 7) % e1.size()];
            }
            tokens_ = T;
            auto host_ms = [&](int reps, const std::function<void()>& fn) {
                fn();
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) fn();
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / reps;
            };
            const int reps = 40;
            std::cerr << "DS41 gpu_bench routes=" << routes;
            if (!e0.empty() && !res0_base_.empty()) {
                const double g = gpu_ms(0, stream_, reps, [&] { resident_mmq(l, T, K, ids.data(), wts.data()); });
                const double h = host_ms(reps, [&] { resident_mmq(l, T, K, ids.data(), wts.data()); ck(cudaStreamSynchronize(stream_), "b"); });
                std::cerr << " gpu0_mmq_ms=" << g << " (host " << h << ")";
                std::vector<int> gi;
                std::vector<float> gw;
                const double c0 = gpu_ms(0, stream_, reps, [&] {
                    for (int i = 0; i < T; ++i) {
                        int ii[16], n = 0;
                        float ww[16];
                        for (int k2 = 0; k2 < K; ++k2) if (ids[size_t(i) * K + k2] >= 0) { ii[n] = ids[size_t(i) * K + k2]; ww[n] = 0.1f; ++n; }
                        if (n) gpu_experts(l, i, ii, ww, n);
                    }
                });
                std::cerr << " gpu0_gemv_ms=" << c0;
                if (k::native_expert_supported(L.f.gu_type, L.f.d_type, c.hidden, c.ff) && fl_ == c.ff) {
                    std::map<int, std::vector<std::pair<int, float>>> by;
                    for (int i = 0; i < T * K; ++i) if (ids[size_t(i)] >= 0) by[ids[size_t(i)]].push_back({i / K, 0.1f});
                    std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
                    for (auto& [e, r] : by) groups.push_back({slots_[size_t(slot_[size_t(l)][size_t(e)])].p, r});
                    const double gg = gpu_ms(0, stream_, reps, [&] {
                        grouped0_.run(L.f.gu_type, L.f.d_type, c.hidden, c.ff, c.clamp[size_t(l)], groups, xn_.f(), T, g_sum_.f(), stream_);
                    });
                    std::cerr << " gpu0_grouped_ms=" << gg;
                    // correctness on random activations: the GEMV chain is the reference; MMQ and grouped against it
                    {
                        std::vector<float> xr(size_t(T) * c.hidden);
                        uint32_t rs = 12345u;
                        for (auto& v : xr) { rs = rs * 1664525u + 1013904223u; v = (float(rs >> 8) / 16777216.0f - 0.5f) * 4.0f; }
                        ck(cudaMemcpy(xn_.p, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice), "bench x");
                        const size_t nsum = size_t(T) * c.hidden;
                        auto result = [&](const std::function<void()>& fn) {
                            ck(cudaMemsetAsync(g_sum_.p, 0, nsum * 4, stream_), "bench sum");
                            fn();
                            std::vector<float> o(nsum);
                            ck(cudaMemcpyAsync(o.data(), g_sum_.p, nsum * 4, cudaMemcpyDeviceToHost, stream_), "bench sum");
                            ck(cudaStreamSynchronize(stream_), "bench sum");
                            return o;
                        };
                        const auto ref = result([&] {
                            for (int i = 0; i < T; ++i) {
                                int ii[16], n = 0;
                                float ww[16];
                                for (int k2 = 0; k2 < K; ++k2) if (ids[size_t(i) * K + k2] >= 0) { ii[n] = ids[size_t(i) * K + k2]; ww[n] = 0.1f; ++n; }
                                if (n) gpu_experts(l, i, ii, ww, n);
                            }
                        });
                        auto rel = [&](const std::vector<float>& o) {
                            double num = 0, den = 0;
                            for (size_t i = 0; i < nsum; ++i) { num += double(o[i] - ref[i]) * (o[i] - ref[i]); den += double(ref[i]) * ref[i]; }
                            return den > 0 ? std::sqrt(num / den) : 0.0;
                        };
                        const auto mm = result([&] { resident_mmq(l, T, K, ids.data(), wts.data()); });
                        const auto gr = result([&] { grouped0_.run(L.f.gu_type, L.f.d_type, c.hidden, c.ff, c.clamp[size_t(l)], groups, xn_.f(), T, g_sum_.f(), stream_); });
                        std::cerr << " rel_err_mmq=" << rel(mm) << " rel_err_grouped=" << rel(gr);
                        ck(cudaMemsetAsync(xn_.p, 0, nsum * 4, stream_), "bench x");
                    }
                }
            }
            if (peer_ && !e1.empty()) {
                const bool grouped_ok = k::native_expert_supported(L.f.gu_type, L.f.d_type, c.hidden, c.ff);
                for (int mode = 0; mode < (grouped_ok ? 3 : 2); ++mode) {
                    const double g0 = peer_->gpu_ms;
                    const double h = host_ms(reps, [&] {
                        const bool ran = mode == 1 ? peer_->prefill(l, T, K, ids1.data(), wts.data(), pinned_[0])
                                       : mode == 2 ? peer_->grouped(l, T, K, ids1.data(), wts.data(), pinned_[0])
                                                   : peer_->decode(l, T, K, ids1.data(), wts.data(), pinned_[0]);
                        if (ran) peer_->collect();
                    });
                    std::cerr << " gpu1_" << (mode == 1 ? "mmq" : mode == 2 ? "grouped" : "gemv") << "_ms=" << (peer_->gpu_ms - g0) / (reps + 1) << " (host " << h << ")";
                }
            }
            std::cerr << "\n";
        }
        if (peer_) peer_->timing = timing0;
        if (!e0.empty() && use_mmq_ && mmq_)
            bench_prefill(slots_[size_t(slot_[size_t(l)][size_t(e0[0])])].p);
    }
    int chunk() const { return chunk_; }

    /// Comparison hooks for --check (single-token passes only): (name, layer, device pointer, floats) / a host pointer.
    std::function<void(const std::string&, int, const float*, size_t)> observe, observe_host;
    /// Teacher forcing for --check-force: fills a host vector with the reference tensor (name, layer); false if none.
    std::function<bool(const std::string&, int, std::vector<float>&)> reference;

    /// Batched sessions (STRATA_DS41_SESSIONS): S independent sequences decoded together, token t of a pass being
    /// session t's next token. Each session's window ring (window + 8 slots), compressed caches and ratio-2 stash live
    /// in per-layer buffers [S][...]; a session's prompt runs through the single-sequence state (forward), then park(s)
    /// copies that state in and clears it for the next prompt. Any KV storage (STRATA_DS41_KV).
    void sessions_init(int S) {
        const auto& c = cfg_;
        if (S < 1) throw std::invalid_argument("ds41: sessions");
        sessions_ = S;
        ring_b_ = c.window + 8;
        const size_t hd = size_t(c.head_dim);
        size_t bytes = 0;
        for (auto* v : {&bwin_, &bcache_, &bstash_v_, &bstash_s_}) {
            v->clear();
            v->resize(size_t(c.layers));
        }
        for (int l = 0; l < c.layers; ++l) {
            bwin_[size_t(l)] = Dev(size_t(S) * ring_b_ * hd * 4);
            ck(cudaMemsetAsync(bwin_[size_t(l)].p, 0, bwin_[size_t(l)].bytes, stream_), "session rings");
            bytes += bwin_[size_t(l)].bytes;
            if (c.kv_src[size_t(l)] != l || c.ratio[size_t(l)] < 1) continue;
            bcache_[size_t(l)] = Dev(size_t(S) * rows_b(l) * kv_row_bytes());
            bytes += bcache_[size_t(l)].bytes;
            if (c.ratio[size_t(l)] > 1) {
                bstash_v_[size_t(l)] = Dev(size_t(S) * hd * 4);
                bstash_s_[size_t(l)] = Dev(size_t(S) * hd * 4);
                bytes += 2 * size_t(S) * hd * 4;
            }
        }
        shist_.assign(size_t(S), {});
        tpos_dev_ = Dev(size_t(4) * S * 4);
        tpos_host_ = reinterpret_cast<int*>(pinned_keep(size_t(4) * S * 4));
        const size_t scratch = size_t(2 * S + 2) * k::ds41_attention_scratch(c.heads);   // the scores, then the split parts
        if (attn_scores_.bytes < scratch) attn_scores_ = Dev(scratch);
        ck(cudaStreamSynchronize(stream_), "sessions");
        std::cerr << "DS41 sessions=" << S << " state_MiB=" << bytes / double(1 << 20) << " ring=" << ring_b_ << "\n";
    }
    /// The single-sequence state (a prompt just run) into session s; then that state starts empty.
    void park(int s) {
        const auto& c = cfg_;
        const int P = int(history_.size());
        const size_t hd = size_t(c.head_dim);
        for (int l = 0; l < c.layers; ++l) {
            for (int q = std::max(0, P - c.window); q < P;) {   // the window's positions, in runs contiguous in both rings
                const int a = q % ring_, b = q % ring_b_, run = std::min({P - q, ring_ - a, ring_b_ - b});
                ck(cudaMemcpyAsync(bwin_[size_t(l)].f((size_t(s) * ring_b_ + size_t(b)) * hd), windows_[size_t(l)].f(size_t(a) * hd),
                                   size_t(run) * hd * 4, cudaMemcpyDeviceToDevice, stream_), "park window");
                q += run;
            }
            if (c.kv_src[size_t(l)] != l || c.ratio[size_t(l)] < 1) continue;
            const int r = c.ratio[size_t(l)];
            if (P / r > 0)
                ck(cudaMemcpyAsync(bcache_[size_t(l)].b(size_t(s) * rows_b(l) * kv_row_bytes()), caches_[size_t(l)].p,
                                   size_t(P / r) * kv_row_bytes(), cudaMemcpyDeviceToDevice, stream_), "park cache");
            if (r > 1) {
                ck(cudaMemcpyAsync(bstash_v_[size_t(l)].f(size_t(s) * hd), stash_v_[size_t(l)].p, hd * 4, cudaMemcpyDeviceToDevice, stream_), "park stash");
                ck(cudaMemcpyAsync(bstash_s_[size_t(l)].f(size_t(s) * hd), stash_s_[size_t(l)].p, hd * 4, cudaMemcpyDeviceToDevice, stream_), "park stash");
            }
        }
        shist_[size_t(s)] = history_;
        reset_sequence();
    }
    /// An empty single-sequence state (position 0 next).
    void reset_sequence() {
        ck(cudaStreamSynchronize(stream_), "reset");
        history_.clear();
        for (auto& w : windows_) ck(cudaMemsetAsync(w.p, 0, w.bytes, stream_), "reset ring");
        ck(cudaStreamSynchronize(stream_), "reset");
    }
    int session_pos(int s) const { return int(shist_[size_t(s)].size()); }
    /// One token for each of the S sessions (tokens[s] at session s's next position): every session's logits, [S][vocab].
    const std::vector<float>& forward_sessions(const int* tokens, int S) {
        if (S != sessions_) throw std::invalid_argument("ds41: session count");
        if (chunk_ < S) resize(S);
        for (int i = 0; i < S; ++i) {
            const int q = int(shist_[size_t(i)].size());
            if (q + 1 > context_) throw std::runtime_error("ds41: context exhausted");
            tpos_host_[i] = q;
            tpos_host_[S + i] = i * ring_b_ + q % ring_b_;                                        // window slot
            tpos_host_[2 * S + i] = i * (context_ + 1) + q;                                      // ratio-1 cache row
            tpos_host_[3 * S + i] = q % 2 ? i * (context_ / 2 + 1) + q / 2 : -1;                 // ratio-2 row (odd positions)
            shist_[size_t(i)].push_back(tokens[i]);
        }
        ck(cudaMemcpyAsync(tpos_dev_.p, tpos_host_, size_t(4) * S * 4, cudaMemcpyHostToDevice, stream_), "session positions");
        sess_ = true;
        struct Off { bool& b; ~Off() { b = false; } } off{sess_};
        return forward(tokens, S, 0, true);
    }

    /// One token at position pos (positions must arrive in order from 0). Returns the logits on the host.
    const std::vector<float>& step(int token, int pos) { return forward(&token, 1, pos, false); }

    /// `count` tokens (at most chunk()) at positions pos .. pos + count - 1. Returns the last token's logits, or with
    /// all_logits every token's ([count][vocab]).
    const std::vector<float>& forward(const int* tokens, int count, int pos, bool all_logits) {
        return forward_impl(tokens, count, pos, all_logits, false);
    }

    /// Speculative verification only: return each row's target-greedy token without copying full logits to host.
    const std::vector<int>& forward_argmax(const int* tokens, int count, int pos) {
        (void)forward_impl(tokens, count, pos, true, true);
        return verify_argmax_host_;
    }

    const std::vector<float>& forward_impl(const int* tokens, int count, int pos, bool all_logits, bool argmax_only) {
        const auto& c = cfg_;
        const int T = count;
        const auto step0 = std::chrono::steady_clock::now();
        struct Done { double& t; std::chrono::steady_clock::time_point s; ~Done() { t += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count(); } } done{prof_[0], step0};
        if (!tokens || T < 1 || T > chunk_ || (spec_save_ && T > 8) || pos < 0 || T > context_ || pos > context_ - T)
            throw std::invalid_argument("ds41: chunk or context bounds");
        if (argmax_only && !all_logits) throw std::invalid_argument("ds41: argmax-only forward requires every output row");
        if (attn_error_.p) ck(cudaMemsetAsync(attn_error_.p, 0, sizeof(int), stream_), "attention bounds reset");
        for (int i = 0; i < T; ++i)
            if (tokens[i] < 0 || tokens[i] >= c.vocab)
                throw std::invalid_argument("ds41: token id outside vocabulary");
        if (!sess_ && pos != int(history_.size())) throw std::logic_error("ds41: positions must be consecutive from 0");
        pos0_ = pos;
        tokens_ = T;
        attn_info_key_ = -1;
        q8x_ready_ = false;
        const bool hook = T == 1 && !sess_;
        // embeddings: Q6_K rows dequantized on the host
        {
            const auto& t = art_.at("token_embd.weight");
            const size_t row = ggml_row_size(ggml_type(t.tensor->type), c.hidden);
            const uint8_t* table = embd_copy_.empty() ? t.data() : embd_copy_.data();
            for (int i = 0; i < T; ++i) {
                const auto e = to_float(int(t.tensor->type), table + size_t(tokens[i]) * row, c.hidden);
                std::copy(e.begin(), e.end(), pinned_[1] + size_t(i) * c.hidden);
            }
            ck(cudaMemcpyAsync(x_.p, pinned_[1], size_t(T) * c.hidden * 4, cudaMemcpyHostToDevice, stream_), "embed");
            k::ds41_broadcast_streams(x_.f(), h_.f(), c.hidden, stream_, T);
        }
        // the rotation tables of each position: [0] plain forward, [1] plain back, [2] compressed forward, [3] back,
        // [4] compressed forward at pos - 1 (a ratio-2 group's row stands for its first position)
        for (int i = 0; i < T; ++i) {
            const int p = sess_ ? tpos_host_[i] : pos + i;
            float* cs = rope_host_.data() + size_t(i) * kTables * 64;
            rope_table(c.rope_plain, c.rope_dim, p, 1.0f, cs);
            rope_table(c.rope_plain, c.rope_dim, p, -1.0f, cs + 64);
            rope_table(c.rope_compressed, c.rope_dim, p, 1.0f, cs + 128);
            rope_table(c.rope_compressed, c.rope_dim, p, -1.0f, cs + 192);
            rope_table(c.rope_compressed, c.rope_dim, std::max(p - 1, 0), 1.0f, cs + 256);
        }
        ck(cudaMemcpyAsync(cs_.p, rope_host_.data(), size_t(T) * kTables * 64 * 4, cudaMemcpyHostToDevice, stream_), "rope");
        if (!sess_) history_.insert(history_.end(), tokens, tokens + T);
        if (T >= stream_min_) stream_begin();   // the producer starts on layer 0's experts while the trunk runs
        // every Engram layer's rows, [engram][token][cols x dim]
        {
            const auto g0 = std::chrono::steady_clock::now();
            for (size_t e = 0; e < c.engram_layers.size(); ++e)
                for (int i = 0; i < T; ++i)
                    if (sess_) gather_engram(int(e), i, tpos_host_[i], shist_[size_t(i)]);
                    else gather_engram(int(e), i, pos + i, history_);
            // STRATA_DS41_ENGRAM_ASYNC (1): the first Engram layer's rows now, the later layers' rows on a background
            // thread that the layer waits for (their reads overlap the layers before it)
            static const bool async = !(std::getenv("STRATA_DS41_ENGRAM_ASYNC") && std::string(std::getenv("STRATA_DS41_ENGRAM_ASYNC")) == "0");
            if (async && c.engram_layers.size() > 1) {
                std::vector<EngramReq> later;
                std::vector<EngramReq> first;
                for (const auto& r : engram_reqs_) (r.table == 0 ? first : later).push_back(r);
                engram_reqs_ = std::move(first);
                read_engram_rows();
                engram_later_ = std::thread([this, later = std::move(later)]() mutable { engram_reqs_bg_ = std::move(later); read_rows(engram_reqs_bg_); });
                upload_engram(0);
            } else {
                read_engram_rows();
                if (!c.engram_layers.empty())
                    ck(cudaMemcpyAsync(engram_rows_.p, engram_host_.data(), engram_host_.size() * 4, cudaMemcpyHostToDevice, stream_),
                       "engram rows");
            }
            prof_[4] += std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
        }
        if (hook && observe) observe("hc_init", -1, h_.f(), size_t(4) * c.hidden);
        route_log_.assign(size_t(T) * c.layers * c.top_k, -1);
        bool have_pre = false;
        skip_ = 0;
        const char* attention_trace = std::getenv("STRATA_DS41_ATTN_TRACE");
        auto trace_attention = [&](const char* label, const float* dev, size_t n, bool first = false) {
            if (!attention_trace || !*attention_trace) return;
            if (T != 1 || c.hidden != 5120 || n > 32768 || fuse_ || sess_ || producer_.joinable() || peer_ || remote_)
                throw std::runtime_error("ds41: attention trace requires the bounded one-token single-GPU path");
            std::vector<float> v(n);
            ck(cudaMemcpyAsync(v.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost, stream_), "attention trace copy");
            ck(cudaStreamSynchronize(stream_), "attention trace sync");
            std::ofstream file(attention_trace, std::ios::binary | (first ? std::ios::trunc : std::ios::app));
            char name[32] = {}; std::strncpy(name, label, sizeof(name)-1);
            const uint32_t count = uint32_t(n);
            if (!file.write(name, sizeof(name)) || !file.write(reinterpret_cast<const char*>(&count), sizeof(count)) ||
                !file.write(reinterpret_cast<const char*>(v.data()), n * sizeof(float)))
                throw std::runtime_error("ds41: attention trace write failed");
        };
        // STRATA_DS41_ENC_SKIP (1): above the last KV-source layer nothing reads a layer's rows beyond its window, so
        // layer l needs only the pass's last window + (layers - 1 - l) (window - 1) positions (the last token's logits and
        // the next pass's windows stay exact); the earlier tokens are dropped, in steps of 256 or more
        static const bool enc_skip = !(std::getenv("STRATA_DS41_ENC_SKIP") && std::string(std::getenv("STRATA_DS41_ENC_SKIP")) == "0");
        const int skip_from = [&] { int s = 0; for (int l = 0; l < c.layers; ++l) if (c.kv_src[size_t(l)] == l) s = l + 1; return s; }();
        const int end = pos + T;
        const auto loop0 = std::chrono::steady_clock::now();
        before_loop_s_ += std::chrono::duration<double>(loop0 - step0).count();
        if (halves_ && T > pftile_ && !sess_ && !hook && !dsp_.on && !remote_) {
            // PREFILL_HALVES: each layer's trunk runs its tiles (the attention-local buffers hold pftile_ rows), then its
            // FFN runs once over the rows the layer still needs - one stream of the layer's experts for the whole pass.
            const int A = pftile_;
            for (int l = 0; l < c.layers; ++l) {
                const std::string p = "blk." + std::to_string(l) + ".";
                if (dtiming_ && (T <= 64 || pf_timing_)) dmark(0);
                int row0 = 0;
                const int hs = halves_skip_from_ >= 0 ? halves_skip_from_ : skip_from;
                if (hs < c.layers && !all_logits && l >= hs) {
                    // the rows this layer still feeds: its last `need` positions, plus one window so their ring rows are
                    // written (256-token granularity, so the first tile is partial)
                    const int need = c.window + (c.layers - 1 - l) * (c.window - 1);
                    // 1024 rows of slack keep the truncated boundary's windows exact (KL 0.0007, vs 0.029 at one window)
                    const int slack = std::getenv("STRATA_DS41_HALVES_SLACK") ? std::atoi(std::getenv("STRATA_DS41_HALVES_SLACK")) : 1024;
                    row0 = std::max(0, ((end - need - slack - pos) / 256) * 256);   // (rows of this pass)
                    row0 = std::max(row0, T - A);   // the FFN's shifted view holds at most one tile
                    skipped_ += size_t(row0);
                }
                for (int t0 = row0; t0 < T; t0 += A) {
                    const int n = std::min(A, T - t0);
                    const size_t hb = size_t(n) * 4 * c.hidden * 4;
                    // the tile's rows of the pass-wide residual, swapped in as the trunk's h_
                    ck(cudaMemcpyAsync(ht_.p, h_.f(size_t(t0) * 4 * c.hidden), hb, cudaMemcpyDeviceToDevice, stream_), "halves h in");
                    std::swap(h_, ht_);
                    std::swap(h2_, h2t_);
                    std::swap(y_, yt_);
                    std::swap(xn_, xnt_);
                    tokens_ = n;
                    pos0_ = pos + t0;
                    rope_off_ = t0;
                    engram_off_ = t0;
                    if (l > 0 && coeff_ffn_full_.p)   // the previous layer's FFN coefficients for the attention pre-mix
                        ck(cudaMemcpyAsync(coeff_ffn_.p, coeff_ffn_full_.f(size_t(t0) * 24), size_t(n) * 24 * 4, cudaMemcpyDeviceToDevice, stream_), "halves coeff in");
                    if (const int e = c.engram_index(l); e >= 0) {
                        if (e > 0 && engram_later_.joinable()) {
                            engram_later_.join();
                            if (engram_failed_) throw std::runtime_error("ds41: Engram row read failed");
                            for (size_t j = 1; j < c.engram_layers.size(); ++j) upload_engram(int(j));
                        }
                        engram(l, e);
                    }
                    mixes(p + "hc_attn_", coeff_attn_);
                    if (l > 0 && fuse_ && share_q8() && hc_norm_ok(w(p + "attn_norm.weight"))) {
                        k::glm_hc_pre_norm_rows_q8(h_.f(), coeff_ffn_.f(), w(p + "attn_norm.weight"), x_.f(), xn_.f(), q8x_.p,
                                                   c.hidden, n, c.rms_eps, stream_);
                    } else {
                        if (l > 0) k::ds41_hc_pre(h_.f(), coeff_ffn_.f(), x_.f(), c.hidden, stream_, n);
                        else ck(cudaMemcpy2DAsync(x_.p, size_t(c.hidden) * 4, h_.p, size_t(4) * c.hidden * 4, size_t(c.hidden) * 4,
                                                  size_t(n), cudaMemcpyDeviceToDevice, stream_), "stream 0");
                        norm_xn(x_.f(), w(p + "attn_norm.weight"));
                    }
                    sel_base_ = t0;   // (the indexer selection is indexed by the pass's rows)
                    attention(l);
                    sel_base_ = 0;
                    k::ds41_hc_post(y_.f(), h_.f(), coeff_attn_.f(4), coeff_attn_.f(8), h2_.f(), c.hidden, stream_, n);
                    std::swap(h_, h2_);
                    mixes(p + "hc_ffn_", coeff_ffn_);
                    if (coeff_ffn_full_.p)
                        ck(cudaMemcpyAsync(coeff_ffn_full_.f(size_t(t0) * 24), coeff_ffn_.p, size_t(n) * 24 * 4,
                                           cudaMemcpyDeviceToDevice, stream_), "halves coeff out");
                    if (fuse_ && share_q8() && hc_norm_ok(w(p + "ffn_norm.weight"))) {
                        k::glm_hc_pre_norm_rows_q8(h_.f(), coeff_attn_.f(), w(p + "ffn_norm.weight"), x_.f(), xn_.f(), q8x_.p,
                                                   c.hidden, n, c.rms_eps, stream_);
                    } else {
                        k::ds41_hc_pre(h_.f(), coeff_attn_.f(), x_.f(), c.hidden, stream_, n);
                        norm_xn(x_.f(), w(p + "ffn_norm.weight"));
                    }
                    // the tile's FFN input and post-attention residual go to the pass-wide buffers
                    ck(cudaMemcpyAsync(xnt_.f(size_t(t0) * c.hidden), xn_.p, size_t(n) * c.hidden * 4, cudaMemcpyDeviceToDevice, stream_), "halves xn");
                    ck(cudaMemcpyAsync(ht_.f(size_t(t0) * 4 * c.hidden), h_.p, hb, cudaMemcpyDeviceToDevice, stream_), "halves h out");
                    std::swap(h_, ht_);
                    std::swap(h2_, h2t_);
                    std::swap(y_, yt_);
                    std::swap(xn_, xnt_);
                }
                if (dtiming_ && (T <= 64 || pf_timing_)) dmark(6);   // (dtime 0 = the layer's trunk, 6 = its FFN)
                tokens_ = T - row0;
                if (row0) {   // the rows the layer keeps move to the front (the dropped rows are dead)
                    const size_t rows = size_t(T - row0), hb = rows * 4 * c.hidden * 4;
                    ck(cudaMemcpyAsync(ht_.p, h_.f(size_t(row0) * 4 * c.hidden), hb, cudaMemcpyDeviceToDevice, stream_), "halves h down");
                    ck(cudaMemcpyAsync(h_.p, ht_.p, hb, cudaMemcpyDeviceToDevice, stream_), "halves h down 2");
                    ck(cudaMemcpyAsync(xnt_.p, xn_.f(size_t(row0) * c.hidden), rows * c.hidden * 4, cudaMemcpyDeviceToDevice, stream_), "halves xn down");
                    ck(cudaMemcpyAsync(xn_.p, xnt_.p, rows * c.hidden * 4, cudaMemcpyDeviceToDevice, stream_), "halves xn down 2");
                    if (coeff_ffn_full_.p) {
                        ck(cudaMemcpyAsync(coeff_ffn_.p, coeff_ffn_full_.f(size_t(row0) * 24), rows * 24 * 4, cudaMemcpyDeviceToDevice, stream_), "halves coeff down");
                    }
                }
                ffn(l);
                if (dtiming_ && (T <= 64 || pf_timing_)) dmark(8);
                // the post's coefficients: the pass-wide buffer, or the tile one when the rows were moved to the front
                const float* cpost = coeff_ffn_full_.p && row0 == 0 ? coeff_ffn_full_.f() : coeff_ffn_.f();
                if (tail_fused_) {   // FUSE: the sums and the residual add inside the post-mix
                    k::ds41_hc_post_sum(moe_.f(), tail_peer_ ? peer_sum_.f() : nullptr, tail_gsum_ ? g_sum_.f() : nullptr, y_.f(), h_.f(),
                                        cpost + 4, cpost + 8, h2_.f(), c.hidden, stream_, tokens_);
                    tail_fused_ = false;
                } else
                    k::ds41_hc_post(y_.f(), h_.f(), cpost + 4, cpost + 8, h2_.f(), c.hidden, stream_, tokens_);
                std::swap(h_, h2_);
                if (row0) {   // and the new residual back to its rows
                    const size_t rows = size_t(T - row0), hb = rows * 4 * c.hidden * 4;
                    ck(cudaMemcpyAsync(ht_.p, h_.p, hb, cudaMemcpyDeviceToDevice, stream_), "halves h up");
                    ck(cudaMemcpyAsync(h_.f(size_t(row0) * 4 * c.hidden), ht_.p, hb, cudaMemcpyDeviceToDevice, stream_), "halves h up 2");
                }
            }
            tokens_ = T;
            pos0_ = pos;
            rope_off_ = 0;
            engram_off_ = 0;
        } else
        for (int l = 0; l < c.layers; ++l) {
            if (enc_skip && !all_logits && l >= skip_from) {
                const int need = c.window + (c.layers - 1 - l) * (c.window - 1);
                const int drop = (end - need) - pos0_;
                if (drop >= 256) narrow(drop, have_pre);
                skipped_ += skip_;
            }
            const std::string p = "blk." + std::to_string(l) + ".";
            if (gpu_timing_ && producer_.joinable()) mark(0);
            if (dtiming_ && (T <= 64 || pf_timing_)) dmark(0);
            if (dsp_.on && !sess_) capture_features(l);
            if (const int e = c.engram_index(l); e >= 0) {
                if (e > 0 && engram_later_.joinable()) {   // the background rows
                    const auto w0 = std::chrono::steady_clock::now();
                    engram_later_.join();
                    if (engram_failed_) throw std::runtime_error("ds41: Engram row read failed");
                    for (size_t j = 1; j < c.engram_layers.size(); ++j) upload_engram(int(j));
                    prof_[4] += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                }
                engram(l, e);
                if (hook && observe) observe("engram_out", l, h_.f(), size_t(4) * c.hidden);
            }
            // STRATA_DS41_GRAPH_PROBE=1: one decode step's dense sequence of a plain layer, launched directly and as a CUDA
            // graph, 100 times each; prints both and exits (the repeats corrupt the state)
#if !defined(__HIP_PLATFORM_AMD__)
            if (T == 1 && have_pre && std::getenv("STRATA_DS41_GRAPH_PROBE") && c.kv_src[size_t(l)] != l && c.topk_src[size_t(l)] != l &&
                c.engram_index(l) < 0) {
                auto seq = [&] {
                    mixes(p + "hc_attn_", coeff_attn_);
                    k::ds41_hc_pre(h_.f(), coeff_ffn_.f(), x_.f(), c.hidden, stream_, T);
                    k::glm_rms_norm_rows(x_.f(), w(p + "attn_norm.weight"), xn_.f(), c.hidden, T, c.rms_eps, stream_);
                    attention(l);
                    k::ds41_hc_post(y_.f(), h_.f(), coeff_attn_.f(4), coeff_attn_.f(8), h2_.f(), c.hidden, stream_, T);
                    mixes(p + "hc_ffn_", coeff_ffn_);
                    k::ds41_hc_pre(h2_.f(), coeff_attn_.f(), x_.f(), c.hidden, stream_, T);
                    k::glm_rms_norm_rows(x_.f(), w(p + "ffn_norm.weight"), xn_.f(), c.hidden, T, c.rms_eps, stream_);
                    mat(p + "ffn_gate_inp.weight", xn_.f(), logits_.f(), T);
                };
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                seq();
                ck(cudaStreamSynchronize(stream_), "probe");
                cudaEventRecord(a, stream_);
                for (int r = 0; r < 100; ++r) seq();
                cudaEventRecord(b, stream_);
                ck(cudaStreamSynchronize(stream_), "probe");
                float direct = 0;
                cudaEventElapsedTime(&direct, a, b);
                cudaGraph_t graph;
                cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed), "capture");
                seq();
                ck(cudaStreamEndCapture(stream_, &graph), "capture end");
                size_t nodes = 0;
                cudaGraphGetNodes(graph, nullptr, &nodes);
                ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                cudaGraphLaunch(exec, stream_);
                ck(cudaStreamSynchronize(stream_), "probe");
                cudaEventRecord(a, stream_);
                for (int r = 0; r < 100; ++r) cudaGraphLaunch(exec, stream_);
                cudaEventRecord(b, stream_);
                ck(cudaStreamSynchronize(stream_), "probe");
                float graphed = 0;
                cudaEventElapsedTime(&graphed, a, b);
                std::cerr << "DS41 graph_probe layer=" << l << " nodes=" << nodes << " direct_us=" << direct * 10 << " graph_us=" << graphed * 10 << "\n";
                std::exit(0);
            }
#endif
            // attention sublayer: its input collapses with the previous layer's FFN pre-mix (layer 0: stream 0)
            mixes(p + "hc_attn_", coeff_attn_);
            const bool verify_exact = trunk_exact();
            if (have_pre && fuse_ && share_q8() && !hook && !verify_exact && hc_norm_ok(w(p + "attn_norm.weight"))) {   // pre-mix, norm and Q8 in one launch
                k::glm_hc_pre_norm_rows_q8(h_.f(), coeff_ffn_.f(), w(p + "attn_norm.weight"), x_.f(), xn_.f(), q8x_.p, c.hidden, T, c.rms_eps, stream_);
                q8x_ready_ = true;
            } else {
            if (have_pre) k::ds41_hc_pre(h_.f(), coeff_ffn_.f(), x_.f(), c.hidden, stream_, T);
            else ck(cudaMemcpy2DAsync(x_.p, size_t(c.hidden) * 4, h_.p, size_t(4) * c.hidden * 4, size_t(c.hidden) * 4, size_t(T),
                                      cudaMemcpyDeviceToDevice, stream_), "stream 0");
            if (l == 10) trace_attention("attn_pre", x_.f(), size_t(c.hidden));
            if (hook && observe) observe("hc_attn_pre", l, x_.f(), size_t(c.hidden));
            norm_xn(x_.f(), w(p + "attn_norm.weight"));
            }
            if (hook && observe) observe("attn_norm", l, xn_.f(), size_t(c.hidden));
            if (l == 10) trace_attention("attn_norm", xn_.f(), size_t(c.hidden));
            { const auto a0 = std::chrono::steady_clock::now(); attention(l); attn_host_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count(); }
            if (l == 10 && attention_trace && *attention_trace) {
                if (c.q_rank <= 0 || c.heads <= 0 || c.head_dim <= 0 || c.out_groups <= 0 || c.out_rank <= 0 ||
                    size_t(c.q_rank) > qa_.bytes / sizeof(float) || size_t(c.q_rank) > qr_.bytes / sizeof(float) ||
                    size_t(c.heads) * c.head_dim > q_.bytes / sizeof(float) ||
                    size_t(c.heads) * c.head_dim > att_.bytes / sizeof(float) || size_t(c.head_dim) > kvn_.bytes / sizeof(float) ||
                    size_t(c.out_groups) * c.out_rank > oa_.bytes / sizeof(float))
                    throw std::runtime_error("ds41: attention trace scratch is too small");
                trace_attention("query_a", qa_.f(), size_t(c.q_rank));
                trace_attention("query_norm", qr_.f(), size_t(c.q_rank));
                trace_attention("query", q_.f(), size_t(c.heads) * c.head_dim);
                trace_attention("kv", kvn_.f(), size_t(c.head_dim));
                trace_attention("attn_heads", att_.f(), size_t(c.heads) * c.head_dim);
                trace_attention("output_a", oa_.f(), size_t(c.out_groups) * c.out_rank);
            }
            if (l == 10) trace_attention("attn_out", y_.f(), size_t(c.hidden));
            if (dtiming_ && (T <= 64 || pf_timing_)) dmark(5);
            k::ds41_hc_post(y_.f(), h_.f(), coeff_attn_.f(4), coeff_attn_.f(8), h2_.f(), c.hidden, stream_, T);
            std::swap(h_, h2_);
            if (l == 10) trace_attention("attn_post", h_.f(), size_t(4) * c.hidden);
            if (defer_pending_) defer_apply(false);   // the previous layer's deferred experts (STRATA_DS41_DEFER_RANK)
            if (hook && observe) observe("hc_attn_post", l, h_.f(), size_t(4) * c.hidden);
            // FFN sublayer: its input collapses with this layer's attention pre-mix; its own pre-mix is the next
            // layer's attention input
            mixes(p + "hc_ffn_", coeff_ffn_);
            have_pre = true;
            if (fuse_ && share_q8() && !hook && !verify_exact && hc_norm_ok(w(p + "ffn_norm.weight"))) {
                k::glm_hc_pre_norm_rows_q8(h_.f(), coeff_attn_.f(), w(p + "ffn_norm.weight"), x_.f(), xn_.f(), q8x_.p, c.hidden, T, c.rms_eps, stream_);
                q8x_ready_ = true;
            } else {
            k::ds41_hc_pre(h_.f(), coeff_attn_.f(), x_.f(), c.hidden, stream_, T);
            if (l == 10) trace_attention("ffn_pre", x_.f(), size_t(c.hidden));
            if (hook && observe) observe("hc_ffn_pre", l, x_.f(), size_t(c.hidden));
            norm_xn(x_.f(), w(p + "ffn_norm.weight"));
            }
            if (hook && observe) observe("ffn_norm", l, xn_.f(), size_t(c.hidden));
            if (l == 10) trace_attention("ffn_norm", xn_.f(), size_t(c.hidden));
            if (dtiming_ && (T <= 64 || pf_timing_)) dmark(6);
            { const auto f0 = std::chrono::steady_clock::now(); ffn(l); ffn_host_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count(); }
            if (dtiming_ && (T <= 64 || pf_timing_)) dmark(8);
            if (tail_fused_) {   // FUSE: the experts' sums and the residual add inside the post-mix
                k::ds41_hc_post_sum(moe_.f(), tail_peer_ ? peer_sum_.f() : nullptr, tail_gsum_ ? g_sum_.f() : nullptr, y_.f(), h_.f(),
                                    coeff_ffn_.f(4), coeff_ffn_.f(8), h2_.f(), c.hidden, stream_, T);
                tail_fused_ = false;
            } else
            k::ds41_hc_post(y_.f(), h_.f(), coeff_ffn_.f(4), coeff_ffn_.f(8), h2_.f(), c.hidden, stream_, T);
            std::swap(h_, h2_);
            if (l == 9) trace_attention("layer9_out", h_.f(), size_t(4) * c.hidden, true);
            if (hook && observe) observe("l_out", l, h_.f(), size_t(4) * c.hidden);
            if (std::vector<float> r; hook && reference && reference("l_out", l, r) && r.size() == size_t(4) * c.hidden)
                ck(cudaMemcpy(h_.p, r.data(), r.size() * 4, cudaMemcpyHostToDevice), "force l_out");
        }
        if (defer_pending_) defer_apply(true);
        loop_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - loop0).count();
        if (dtiming_ && !dmarks_.empty()) {   // GPU ms from each mark to the next, by the earlier mark's phase
            ck(cudaStreamSynchronize(stream_), "timing");
            for (size_t j = 0; j + 1 < dmarks_.size(); ++j) {
                float ms = 0;
                cudaEventElapsedTime(&ms, dmarks_[j], dmarks_[j + 1]);
                dtime_[dmark_id_[j]] += ms;
            }
            dmark_id_.clear();
            dmark_used_ = 0;
            dmarks_.clear();
            if (pf_timing_ && T > 64) {   // a prompt pass: totals, then cleared (decode passes keep the per-pass averages)
                static const char* const pf[10] = {"pre", "q", "kv_index", "attn", "out", "ffn_pre", "router", "experts", "tail", "lane_wait"};
                double tot = 0;
                std::cerr << "DS41 chunk_gpu_ms T=" << T;
                for (int q = 0; q < 10; ++q) {
                    std::cerr << " " << pf[q] << "=" << dtime_[q];
                    tot += dtime_[q];
                }
                std::cerr << " sum=" << tot << "\n";
                if (!dma_ev_.empty()) {
                    double dma_ms = 0, wait_ms = 0;
                    for (const auto& [a, b] : dma_ev_) {
                        float ms = 0;
                        if (cudaEventElapsedTime(&ms, a, b) == cudaSuccess) dma_ms += ms;
                    }
                    for (const auto& [a, b] : wait_ev_) {
                        float ms = 0;
                        if (cudaEventElapsedTime(&ms, a, b) == cudaSuccess) wait_ms += ms;
                    }
                    std::cerr << "DS41 chunk_dma GB=" << dma_bytes_ / 1e9 << " copy_s=" << dma_ms / 1e3 << " GB/s="
                              << (dma_ms > 0 ? dma_bytes_ / (dma_ms * 1e-3) / 1e9 : 0.0) << " items=" << dma_ev_.size()
                              << " gpu_wait=" << wait_ms / 1e3 << "\n";
                    dma_ev_.clear();
                    wait_ev_.clear();
                    {
                        std::lock_guard<std::mutex> lock(dma_mu_);
                        dma_pool_used_ = 0;
                    }
                    dma_bytes_ = 0;
                }
                std::fill(std::begin(dtime_), std::end(dtime_), 0.0);
            } else {
                dtokens_ += 1;
            }
        }
        if (gpu_timing_ && !marks_.empty()) {   // dense (layer start to the routed experts) and routed GPU time per pass, ms
            ck(cudaStreamSynchronize(stream_), "timing");
            double dense = 0, moe = 0;
            for (size_t j = 0; j + 2 < marks_.size(); j += 3) {
                float a = 0, b = 0;
                cudaEventElapsedTime(&a, marks_[j], marks_[j + 1]);
                cudaEventElapsedTime(&b, marks_[j + 1], marks_[j + 2]);
                dense += a; moe += b;
            }
            std::cerr << "DS41 gpu_timing dense_ms=" << dense << " routed_ms=" << moe << " layers=" << marks_.size() / 3 << "\n";
            for (auto e : marks_) cudaEventDestroy(e);
            marks_.clear();
        }
        stream_end();
        if (skip_ && hook) throw std::logic_error("ds41: enc-skip on a one-token pass");
        // head: the last FFN pre-mix collapses the streams (every token's, or the last one's)
        const auto h0 = std::chrono::steady_clock::now();
        const int first = all_logits ? 0 : tokens_ - 1, n = tokens_ - first;
        if (logits_.bytes < size_t(n) * c.vocab * 4) logits_ = Dev(size_t(n) * c.vocab * 4);
        k::ds41_hc_pre(h_.f(size_t(first) * 4 * c.hidden),
                       (coeff_ffn_full_.p ? coeff_ffn_full_ : coeff_ffn_).f(size_t(first) * 24), x_.f(), c.hidden, stream_, n);
        k::glm_rms_norm_rows(x_.f(), w("output_norm.weight"), xn_.f(), c.hidden, n, c.rms_eps, stream_);
        mat("output.weight", xn_.f(), logits_.f(), n);
        if (argmax_only) {
            const size_t ids_bytes = checked_size_mul(size_t(n), sizeof(int), "verification argmax IDs");
            if (verify_argmax_ids_dev_.bytes < ids_bytes) verify_argmax_ids_dev_ = Dev(ids_bytes);
            if (!verify_argmax_bad_dev_.p) verify_argmax_bad_dev_ = Dev(sizeof(int));
            verify_argmax_host_.resize(size_t(n));
            ck(cudaMemsetAsync(verify_argmax_bad_dev_.p, 0, sizeof(int), stream_), "verification argmax status reset");
            k::ds41_argmax_rows(logits_.f(), n, c.vocab, c.vocab, verify_argmax_ids_dev_.i(), verify_argmax_bad_dev_.i(), stream_);
            ck(cudaMemcpyAsync(verify_argmax_host_.data(), verify_argmax_ids_dev_.p, ids_bytes, cudaMemcpyDeviceToHost, stream_),
               "verification argmax tokens");
            ck(cudaMemcpyAsync(&verify_argmax_bad_host_, verify_argmax_bad_dev_.p, sizeof(int), cudaMemcpyDeviceToHost, stream_),
               "verification argmax status");
        } else {
            logits_host_.resize(size_t(n) * c.vocab);
            ck(cudaMemcpyAsync(logits_host_.data(), logits_.p, logits_host_.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
        }
        int attn_bounds_error = 0;
        if (attn_error_.p) ck(cudaMemcpyAsync(&attn_bounds_error, attn_error_.p, sizeof(attn_bounds_error), cudaMemcpyDeviceToHost, stream_), "attention bounds result");
        ck(cudaStreamSynchronize(stream_), "sync");
        if (attn_bounds_error) throw std::runtime_error("ds41: gfx12 WMMA attention rejected invalid row metadata or selection indices (code " +
                                                        std::to_string(attn_bounds_error) + ")");
        if (argmax_only) {
            if (verify_argmax_bad_host_) throw std::runtime_error("ds41: speculative verification produced a NaN logit or invalid argmax");
            for (int token : verify_argmax_host_)
                if (token < 0 || token >= c.vocab) throw std::runtime_error("ds41: speculative target token outside vocabulary");
        }
        prof_[6] += std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
        if (hook && observe) observe("result_output", -1, logits_.f(), size_t(c.vocab));
        if (routes_)
            routes_.write(reinterpret_cast<const char*>(route_log_.data()), std::streamsize(route_log_.size() * 4));
        if (!spec_save_) { const auto a0 = std::chrono::steady_clock::now(); adapt(T); prof_[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count(); }
        return logits_host_;
    }

private:
    static constexpr int kTables = 5;   // rotation tables per position
    int row_width() const { return cfg_.head_dim + cfg_.index_dim; }
    // the indexer keys of a KV source's cache (an FP16 cache's pointer cast to float*)
    const float* cache_keys(int src) const {
        return kv_half_ ? reinterpret_cast<const float*>(reinterpret_cast<const uint16_t*>(caches_[size_t(src)].p) + cfg_.head_dim)
                        : caches_[size_t(src)].f(cfg_.head_dim);
    }
    Dev ig_q_, ig_k_, ig_c_;
    void index_gemm_scores(int src, int t0, int n, int ratio, float w_scale) {
        const auto& c = cfg_;
        const int H = c.index_heads, D = c.index_dim, R = (pos0_ + t0 + n) / ratio;
        if (R < 1) return;
        const size_t qn = size_t(n) * H * D;
        if (ig_q_.bytes < qn * 2) ig_q_ = Dev(qn * 2);
        k::glm_f16(iq_.f(size_t(t0) * H * D), ig_q_.h(), int64_t(qn), stream_);
        const size_t budget = size_t(256) << 20;
        const int RT = int(std::max<size_t>(256, std::min<size_t>(size_t(R), budget / (size_t(n) * H * 4))));
        if (ig_c_.bytes < size_t(RT) * n * H * 4) ig_c_ = Dev(size_t(RT) * n * H * 4);
        if (kv_kind_ == 2 && ig_k_.bytes < size_t(RT) * D * 2) ig_k_ = Dev(size_t(RT) * D * 2);
        const float one = 1, zero = 0;
        for (int row0 = 0; row0 < R; row0 += RT) {
            const int m = std::min(RT, R - row0);
            const void* A;
            int lda;
            if (kv_kind_ == 1) {
                A = reinterpret_cast<const uint16_t*>(caches_[size_t(src)].p) + size_t(row0) * row_width() + c.head_dim;
                lda = row_width();
            } else {
                k::ds41_keys_q8_f16(caches_[size_t(src)].b(), kv_row_bytes(), c.head_dim, D, row0, m, ig_k_.h(), stream_);
                A = ig_k_.p;
                lda = D;
            }
            if (cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, m, n * H, D, &one, A, CUDA_R_16F, lda, ig_q_.p, CUDA_R_16F, D, &zero, ig_c_.p,
                             CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("ds41: index GEMM");
            k::ds41_index_reduce(ig_c_.f(), m, n, H, iw_.f(size_t(t0) * H), w_scale, pos0_ + t0, ratio, row0, scores_.f(), score_stride_, stream_);
        }
    }

    // FP16 caches: tokens [t0, t0 + n) of the pass's selected rows of `src`, as floats in kv_sel_ (n <= kGatherTokens)
    const float* gather_sel(int src, int ts, int t0, int n, int ratio) {
        const int s0 = t0 + sel_base_;   // (merged pass: the selection is indexed by the pass's rows; the positions are not)
        if (kv_kind_ == 2) {
            k::ds41_gather_sel_q8(caches_[size_t(src)].b(), kv_row_bytes(), row_width(), sel_[size_t(ts)].i() + size_t(s0) * cfg_.index_top_k,
                                  cfg_.index_top_k, n, pos0_ + t0, ratio, kv_sel_.f(), stream_);
            return kv_sel_.f();
        }
        k::ds41_gather_sel_h(reinterpret_cast<const uint16_t*>(caches_[size_t(src)].p), row_width(), sel_[size_t(ts)].i() + size_t(s0) * cfg_.index_top_k,
                             cfg_.index_top_k, n, pos0_ + t0, ratio, kv_sel_.f(), stream_);
        return kv_sel_.f();
    }

    // a tensor's bytes in a read-only MAP_SHARED mapping of its shard (one per shard)
    const uint8_t* shared(const core::ArtifactTensor& t, bool whole = true) {
        auto it = shared_maps_.find(t.file);
        if (it == shared_maps_.end()) {
            const int fd = ::open(t.file->path().c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
            void* p = mmap(nullptr, t.file->file_size(), PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (p == MAP_FAILED) throw std::runtime_error("ds41: cannot map " + t.file->path());
            it = shared_maps_.emplace(t.file, std::make_pair(p, size_t(t.file->file_size()))).first;
        }
        // experts should stay page-cached: ask for the uncached ones now (readahead in the background); a tensor
        // shared with the remote node is read only where the caller populates it (no readahead into its rows)
        const uintptr_t a = reinterpret_cast<uintptr_t>(static_cast<const uint8_t*>(it->second.first) + t.file->data_start() + t.tensor->offset);
        const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE)), a0 = a / pg * pg;
        madvise(reinterpret_cast<void*>(a0), size_t(a + t.bytes - a0), whole ? MADV_WILLNEED : MADV_RANDOM);
        if (whole) populate_.push_back({reinterpret_cast<void*>(a0), size_t(a + t.bytes - a0)});
        return static_cast<const uint8_t*>(it->second.first) + t.file->data_start() + t.tensor->offset;
    }

    // a host-made F32 matrix [out][in] as weight `name`
    bool borrow_weight(const std::string& key, size_t* bytes) {
        if (!borrow_) return false;
        const auto it = borrow_->weights_.find(key);
        if (it == borrow_->weights_.end()) return false;
        Weight W;
        W.type = it->second.type;
        W.in = it->second.in;
        W.out = it->second.out;
        W.row_bytes = it->second.row_bytes;
        W.data = Dev::view(it->second.data);
        *bytes = W.data.bytes;
        weights_.emplace(key, std::move(W));
        return true;
    }
    size_t upload_f32(const std::string& name, const std::vector<float>& v, int in, int out) {
        if (size_t b; borrow_weight(name, &b)) return b;
        Weight W;
        W.type = GGML_TYPE_F32;
        W.in = in;
        W.out = out;
        W.row_bytes = size_t(in) * 4;
        W.data = Dev(v.size() * 4);
        ck(cudaMemcpy(W.data.p, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload");
        weights_.emplace(name, std::move(W));
        return v.size() * 4;
    }
    size_t upload(const std::string& name, const std::string& as = {}) {
        if (size_t b; borrow_weight(as.empty() ? name : as, &b)) return b;
        const auto& t = art_.at(name);
        Weight W;
        W.type = int(t.tensor->type);
        W.in = int(t.tensor->shape[0]);
        W.out = 1;
        for (size_t i = 1; i < t.tensor->shape.size(); ++i) W.out *= int(t.tensor->shape[i]);
        W.row_bytes = ggml_row_size(ggml_type(W.type), W.in);
        // STRATA_DS41_DENSE_Q4K=1 (lossy, opt-in): Q6_K and Q5_K trunk matrices requantized to Q4_K at load (about
        // -30 % of their VRAM, for the expert tiers)
        static const bool dense_q4k = strata_env_is_one("STRATA_DS41_DENSE_Q4K");
        // STRATA_DS41_DENSE_Q5K=1: only the Q6_K matrices, to Q5_K
        static const bool dense_q5k = strata_env_is_one("STRATA_DS41_DENSE_Q5K");
        if (dense_q5k && !dense_q4k && W.type == GGML_TYPE_Q6_K && W.in % 256 == 0 && W.out > 0) {
            const int64_t n = int64_t(W.in) * W.out;
            const auto f = to_float(W.type, t.data(), n);
            const size_t rb = ggml_row_size(GGML_TYPE_Q5_K, W.in), qb = rb * size_t(W.out);
            std::vector<uint8_t> q(qb);
            const int nt = 8;
            std::vector<std::thread> th;
            for (int k = 0; k < nt; ++k)
                th.emplace_back([&, k] {
                    const int64_t r0 = int64_t(W.out) * k / nt, r1 = int64_t(W.out) * (k + 1) / nt;
                    if (r1 > r0) ggml_quantize_chunk(GGML_TYPE_Q5_K, f.data() + r0 * W.in, q.data() + size_t(r0) * rb, 0, r1 - r0, W.in, nullptr);
                });
            for (auto& x : th) x.join();
            W.type = GGML_TYPE_Q5_K;
            W.row_bytes = rb;
            W.data = Dev(qb);
            ck(cudaMemcpy(W.data.p, q.data(), qb, cudaMemcpyHostToDevice), "upload Q5_K");
            t.file->discard_tensor_pages(*t.tensor, t.bytes);
            weights_.emplace(as.empty() ? name : as, std::move(W));
            return qb;
        }
        if (dense_q4k && (W.type == GGML_TYPE_Q6_K || W.type == GGML_TYPE_Q5_K) && W.in % 256 == 0 && W.out > 0) {
            const int64_t n = int64_t(W.in) * W.out;
            const auto f = to_float(W.type, t.data(), n);
            const size_t rb = ggml_row_size(GGML_TYPE_Q4_K, W.in), qb = rb * size_t(W.out);
            std::vector<uint8_t> q(qb);
            const int nt = 8;
            std::vector<std::thread> th;
            for (int k = 0; k < nt; ++k)
                th.emplace_back([&, k] {
                    const int64_t r0 = int64_t(W.out) * k / nt, r1 = int64_t(W.out) * (k + 1) / nt;
                    if (r1 > r0) ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data() + r0 * W.in, q.data() + size_t(r0) * rb, 0, r1 - r0, W.in, nullptr);
                });
            for (auto& x : th) x.join();
            W.type = GGML_TYPE_Q4_K;
            W.row_bytes = rb;
            W.data = Dev(qb);
            ck(cudaMemcpy(W.data.p, q.data(), qb, cudaMemcpyHostToDevice), "upload Q4_K");
            t.file->discard_tensor_pages(*t.tensor, t.bytes);
            weights_.emplace(as.empty() ? name : as, std::move(W));
            return qb;
        }
        W.data = Dev(t.bytes);
        ck(cudaMemcpy(W.data.p, t.data(), t.bytes, cudaMemcpyHostToDevice), "upload");
        // the GPU copy is the only one used: its host pages leave the page cache to the experts
        t.file->discard_tensor_pages(*t.tensor, t.bytes);
        weights_.emplace(as.empty() ? name : as, std::move(W));
        return t.bytes;
    }
    const Weight& weight(const std::string& name) const {
        const auto it = weights_.find(name);
        if (it == weights_.end()) throw std::runtime_error("ds41: weight not on the GPU: " + name);
        return it->second;
    }
    const float* w(const std::string& name) const { return weight(name).data.f(); }

    // y[t] = W x[t] for nt tokens (x[t] at t W.in, y[t] at t rows) over rows [row0, row0 + rows) of W (rows 0: all)
    // STRATA_DS41_DENSE_MMQ=1: a quantized trunk matrix times a pass of 9..kDenseRows tokens (batched sessions) by
    // llama.cpp's MMQ, int8 tensor cores: the weights are read once for every token (MMVQ takes 8 columns per launch,
    // the GEMM path dequantizes to FP16). x: nt rows of `in` floats, ldx apart; y rows ldy apart. false: not covered
    // (type, shape, or K not a multiple of 256: MMQ reads 256-value K chunks).
    static constexpr int kDenseRows = 256;
    Dev dm_xq_, dm_bounds_, dm_iota_;
    int* dm_hb_ = nullptr;
    int dm_nt_ = -1;
    std::set<int> dm_off_;   // types said once to take the other path
    bool dense_mmq(int type, const uint8_t* w, const float* x, int ldx, float* y, int ldy, int in, int rows, int nt) {
        namespace mq = strata::prefill::mmq;
        static const bool on = std::getenv("STRATA_DS41_DENSE_MMQ") && std::string(std::getenv("STRATA_DS41_DENSE_MMQ")) == "1";
        // (passes from gemm_min_ tokens up keep the GEMM path unless they are sessions: a prompt chunk's GEMMs win)
        if (!on || nt <= 8 || nt > kDenseRows || (nt >= gemm_min_ && !sess_) || in % 256 || !mq::built()) return false;
        if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || !mq::fits(type, rows)) {
            if (dm_off_.insert(type).second) std::cerr << "DS41 dense_mmq off for type " << type << "\n";
            return false;
        }
        if (!mmq_) mmq_ = std::make_unique<mq::Context>();
        if (dm_xq_.bytes < mq::q8_bytes(kDenseRows, in)) dm_xq_ = Dev(mq::q8_bytes(kDenseRows, std::max(in, 8192)));
        if (!dm_iota_.p) {
            dm_bounds_ = Dev(256);
            dm_iota_ = Dev((size_t(kDenseRows) + 128) * 4);
            mq::iota(dm_iota_.i(), kDenseRows + 128, stream_);
        }
        if (dm_nt_ != nt) {   // bounds {0, nt} (set_bounds is HIP-only); a pass's calls share nt, passes end synchronized
            if (!dm_hb_) dm_hb_ = reinterpret_cast<int*>(pinned_keep(16));
            dm_hb_[0] = 0;
            dm_hb_[1] = nt;
            ck(cudaMemcpyAsync(dm_bounds_.p, dm_hb_, 8, cudaMemcpyHostToDevice, stream_), "dense bounds");
            dm_nt_ = nt;
        }
        mq::quantize(x, nullptr, dm_xq_.p, type, in, ldx, nt, stream_);
        mq::Product pr;
        pr.w = w; pr.type = type; pr.w_rows = rows; pr.w_cols = in; pr.expert_bytes = mq::matrix_bytes(type, rows, in);
        pr.n = 1; pr.xq = dm_xq_.p; pr.bounds = dm_bounds_.i(); pr.ids = dm_iota_.i(); pr.total_rows = nt; pr.max_rows = nt;
        pr.dst = y; pr.ld_dst = ldy; pr.dst_rows = nt;
        mmq_->run(pr, stream_);
        return true;
    }

    void mat(const std::string& name, const float* x, float* y, int nt = 1, int row0 = 0, int rows = 0) {
        const Weight& W = weight(name);
        if (!rows) rows = W.out;
        const uint8_t* base = static_cast<const uint8_t*>(W.data.p) + size_t(row0) * W.row_bytes;
        if (dense_mmq(W.type, base, x, W.in, y, rows, W.in, rows, nt)) return;
        if (nt >= gemm_min_) {   // a chunk: GEMMs (cuBLAS FP32, BF16, or the native blocks dequantized to FP16)
            if (W.type == GGML_TYPE_F32) {
                const float one = 1, zero = 0;
                if (cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, nt, W.in, &one, reinterpret_cast<const float*>(base), W.in, x,
                                W.in, &zero, y, rows) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("ds41: cublasSgemm " + name);
            } else if (W.type == GGML_TYPE_BF16) {
                k::ds41_to_bf16(x, xh_.h(), int64_t(nt) * W.in, stream_);
                gemm_->bf16(xh_.h(), reinterpret_cast<const uint16_t*>(base), y, nt, rows, W.in);
            } else {
                k::glm_f16(x, xh_.h(), int64_t(nt) * W.in, stream_);
                gemm_->native(xh_.h(), W.type, base, y, nt, rows, W.in);
            }
            return;
        }
        if (W.type == GGML_TYPE_F32) {
            k::ds41_f32_gemv(reinterpret_cast<const float*>(base), x, y, rows, W.in, stream_, nt);
        } else if (W.type == GGML_TYPE_BF16) {
            // the batched BF16 GEMV is another kernel (other sums): it stays per column in exact verification unless
            // STRATA_DS41_VERIFY_FAST_BF16=1 as well
            static const bool exact = strata_env_is_one("STRATA_DS41_VERIFY_EXACT") &&
                                      !(strata_env_is_one("STRATA_DS41_VERIFY_FAST_TRUNK") && strata_env_is_one("STRATA_DS41_VERIFY_FAST_BF16"));
            if (nt == 1) k::exl3_bf16_gemv(reinterpret_cast<const uint16_t*>(base), x, y, W.in, rows, stream_);
            else if (exact)   // a short pass rounds as the one-token passes do
                for (int t = 0; t < nt; ++t)
                    k::exl3_bf16_gemv(reinterpret_cast<const uint16_t*>(base), x + size_t(t) * W.in, y + size_t(t) * rows, W.in, rows, stream_);
            else k::ds41_bf16_gemv(reinterpret_cast<const uint16_t*>(base), x, y, rows, W.in, stream_, nt);
        } else {
            if (!k::native_mmvq_supported(W.type)) throw std::runtime_error("ds41: unsupported trunk type in " + name);
            quantize(x, q8_.p, W.in, nt);
            mmvq(W.type, base, q8_.p, y, W.in, rows, nt);
        }
    }
    // STRATA_DS41_SHARE_Q8=1: y = W x for the weights of a sublayer that all read xn_, from one quantization of it
    // (q8x_ holds it after quantize_xn); the other weight types go through mat
    bool share_q8() const {
        static const bool on = std::getenv("STRATA_DS41_SHARE_Q8") && std::string(std::getenv("STRATA_DS41_SHARE_Q8")) == "1";
        return on && tokens_ < gemm_min_;
    }
    void quantize_xn() {
        if (share_q8() && !q8x_ready_) quantize(xn_.f(), q8x_.p, cfg_.hidden, tokens_);
        q8x_ready_ = false;
    }
    // the fused pre-mix/norm/Q8 kernel's preconditions (else the three launches)
    bool hc_norm_ok(const float* norm_w) const {
        const auto aligned = [](const void* q) { return (reinterpret_cast<uintptr_t>(q) & 15) == 0; };
        return cfg_.hidden % 32 == 0 && aligned(h_.p) && aligned(x_.p) && aligned(xn_.p) && aligned(norm_w) && aligned(q8x_.p);
    }
    // the norm into xn_, and with FUSE and SHARE_Q8 its Q8_1 blocks into q8x_ in the same launch
    void norm_xn(const float* x, const float* w) {
        if (fuse_ && share_q8() && hc_norm_ok(w)) {
            k::glm_rms_norm_rows_q8(x, w, xn_.f(), q8x_.p, cfg_.hidden, tokens_, cfg_.rms_eps, stream_);
            q8x_ready_ = true;
            return;
        }
        k::glm_rms_norm_rows(x, w, xn_.f(), cfg_.hidden, tokens_, cfg_.rms_eps, stream_);
        q8x_ready_ = false;
    }
    void mat_xn(const std::string& name, float* y) {
        const Weight& W = weight(name);
        if (!share_q8() || W.type == GGML_TYPE_F32 || W.type == GGML_TYPE_BF16 || W.in != cfg_.hidden ||
            dense_mmq(W.type, static_cast<const uint8_t*>(W.data.p), xn_.f(), W.in, y, W.out, W.in, W.out, tokens_)) {
            if (!share_q8() || W.type == GGML_TYPE_F32 || W.type == GGML_TYPE_BF16 || W.in != cfg_.hidden) mat(name, xn_.f(), y, tokens_);
            return;
        }
        mmvq(W.type, static_cast<const uint8_t*>(W.data.p), q8x_.p, y, W.in, W.out, tokens_);
    }

    // native_quantize_q8_1 over nt columns, at most 8 per launch
    void quantize(const float* x, void* q8, int in, int nt) {
        if (fuse_) { k::native_quantize_q8_1(x, q8, in * nt, 1, stream_); return; }   // contiguous columns: one vector, the same blocks
        const size_t col = k::native_q8_1_bytes(in, 1);
        for (int j = 0; j < nt; j += 8)
            k::native_quantize_q8_1(x + size_t(j) * in, static_cast<uint8_t*>(q8) + size_t(j) * col, in, std::min(8, nt - j), stream_);
    }
    // native_mmvq over nt columns, at most 8 per launch
    void mmvq(int type, const uint8_t* w, const void* q8, float* y, int in, int out, int nt) {
        const size_t col = k::native_q8_1_bytes(in, 1);
        // STRATA_DS41_VERIFY_EXACT=1: one column per launch, so a short pass rounds as the one-token passes do
        static const bool exact = trunk_exact();
        if (exact) {
            for (int j = 0; j < nt; ++j)
                k::native_mmvq(type, w, static_cast<const uint8_t*>(q8) + size_t(j) * col, y + size_t(j) * out, in, out, 1, stream_);
            return;
        }
        for (int j = 0; j < nt; j += 8)
            k::native_mmvq(type, w, static_cast<const uint8_t*>(q8) + size_t(j) * col, y + size_t(j) * out, in, out,
                           std::min(8, nt - j), stream_);
    }

    // the hyper-connection mixes of every token's streams into coeff (pre, post, comb per token)
    void mixes(const std::string& p, Dev& coeff) {
        const auto& c = cfg_;
        // STRATA_DS41_FUSE_MIXES=1: one launch when fn is F32 (separate from FUSE: its GEMV is a different kernel, so the
        // compiler's contraction of its sums is not guaranteed to match f32_gemv's)
        static const bool fuse_mixes = std::getenv("STRATA_DS41_FUSE_MIXES") && std::string(std::getenv("STRATA_DS41_FUSE_MIXES")) == "1";
        // STRATA_DS41_MIX_TILE=n: n tokens per call, so flat_ / mixes_ hold n rows instead of the pass's (a 8192-token
        // pass saves 0.64 GiB at n = 1024)
        const int step = mix_tile_ > 0 ? mix_tile_ : tokens_;
        for (int t0 = 0; t0 < tokens_; t0 += step) {
            const int n = std::min(step, tokens_ - t0);
            const size_t hoff = size_t(t0) * 4 * c.hidden, coff = size_t(t0) * 24;
            if (fuse_mixes) {
                const Weight& W = weight(p + "fn.weight");
                if (W.type == GGML_TYPE_F32 && W.in == 4 * c.hidden && W.out == 24 &&
                    k::ds41_hc_mixes(h_.f(hoff), reinterpret_cast<const float*>(W.data.p), w(p + "scale.weight"), w(p + "base.weight"), flat_.f(),
                                     mixes_.f(), coeff.f(coff), c.hidden, c.sinkhorn, c.hc_eps, c.rms_eps, stream_, n))
                    continue;
            }
            k::glm_rms_norm_rows(h_.f(hoff), nullptr, flat_.f(), 4 * c.hidden, n, c.rms_eps, stream_);
            mat(p + "fn.weight", flat_.f(), mixes_.f(), n);
            k::ds41_hc_coeff(mixes_.f(), w(p + "scale.weight"), w(p + "base.weight"), coeff.f(coff), c.sinkhorn, c.hc_eps, stream_, n);
        }
    }

    // the Engram rows of token i (position pos) for the e-th Engram layer, into engram_host_
    void gather_engram(int e, int i, int pos, const std::vector<int32_t>& hist) {
        const auto& c = cfg_;
        const int cols = c.engram_cols();
        std::vector<int64_t> rows(static_cast<size_t>(cols));
        engram_hash(e, pos, [&](int q) { return hist[size_t(q)]; }, rows.data());
        const size_t per_token = size_t(cols) * c.engram_dim;
        float* v = engram_host_.data() + (size_t(e) * chunk_ + size_t(i)) * per_token;
        for (int j = 0; j < cols; ++j) engram_reqs_.push_back({rows[size_t(j)], v + size_t(j) * c.engram_dim, e});
    }

    // the rows of the e-th Engram table for position pos, tokens by tok_at(position)
    template <class Tok> void engram_hash(int e, int pos, const Tok& tok_at, int64_t* rows) const { engram_rows_for(cfg_, e, pos, tok_at, rows); }
    template <class Tok> static void engram_rows_for(const Config& c, int e, int pos, const Tok& tok_at, int64_t* rows) {
        const int cols = c.engram_cols(), n = c.ngram;
        // the n-gram hash of PR #30817's llama_model_deepseek41::engram_hash
        const uint64_t* mult = c.multipliers.data() + size_t(e) * n;
        const uint64_t* primes = c.primes.data() + size_t(e) * cols;
        const uint64_t* offs = c.offsets.data() + size_t(e) * cols;
        const int64_t pad = c.engram_pad_mapped ? c.engram_pad : c.token_map[size_t(c.engram_pad)];
        auto cid = [&](int64_t t) -> int64_t {
            return t < 0 || t >= int64_t(c.token_map.size()) ? -1 : c.token_map[size_t(t)];
        };
        std::vector<uint64_t> toks(static_cast<size_t>(n));
        int64_t id = cid(tok_at(pos));
        bool blocked = id == -1;
        toks[0] = uint64_t(blocked ? pad : id);
        for (int s = 1; s < n; ++s) {
            id = pos - s < 0 ? -1 : cid(tok_at(pos - s));
            blocked = blocked || id == -1;
            toks[size_t(s)] = uint64_t(blocked ? pad : id);
        }
        uint64_t rolling = toks[0] * mult[0];
        for (int o = 1; o < n; ++o) {
            rolling ^= toks[size_t(o)] * mult[o];
            for (int h = 0; h < c.engram_heads; ++h) {
                const int col = (o - 1) * c.engram_heads + h;
                rows[size_t(col)] = int64_t(rolling % primes[col] + offs[col]);
            }
        }
    }

    // STRATA_DS41_ENGRAM_PREFETCH=1 (with DSpark): the later Engram tables' file rows of a drafted position are read
    // into the page cache as soon as its token is known (64 threads), ahead of the verify pass that reads them
    std::vector<std::thread> pf_threads_;
    std::deque<std::pair<int, int64_t>> pf_queue_;   // (table, row)
    std::mutex pf_m_;
    std::condition_variable pf_cv_;
    bool pf_stop_ = false;
    void engram_prefetch(int pos, const std::vector<int>& toks_from_pos, int from, int upto) {   // toks: positions pos.., history before
        static const bool on = std::getenv("STRATA_DS41_ENGRAM_PREFETCH") && std::string(std::getenv("STRATA_DS41_ENGRAM_PREFETCH")) == "1";
        const auto& c = cfg_;
        if (!on || c.engram_layers.size() < 2 || engram_remote_) return;
        if (pf_threads_.empty())
            for (int t = 0, n = std::getenv("STRATA_DS41_ENGRAM_PREFETCH_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_ENGRAM_PREFETCH_THREADS"))) : 64; t < n; ++t)
                pf_threads_.emplace_back([this] {
                    unpin_helper();
                    alignas(4096) static thread_local uint8_t buf[3 * 4096];
                    for (;;) {
                        std::pair<int, int64_t> job;
                        {
                            std::unique_lock<std::mutex> lock(pf_m_);
                            pf_cv_.wait(lock, [&] { return pf_stop_ || !pf_queue_.empty(); });
                            if (pf_stop_) return;
                            job = pf_queue_.front();
                            pf_queue_.pop_front();
                        }
                        const auto& t = *engram_tables_[size_t(job.first)];
                        const size_t row_bytes = ggml_row_size(ggml_type(int(t.tensor->type)), cfg_.engram_dim);
                        const uint64_t off = t.file->data_start() + t.tensor->offset + uint64_t(job.second) * row_bytes;
                        const uint64_t page = off & ~uint64_t(4095);
                        const size_t len = (off + row_bytes - page + 4095) & ~size_t(4095);
                        (void)!pread(engram_fds_[size_t(job.first)], buf, len, off_t(page));
                    }
                });
        const int cols = c.engram_cols();
        std::vector<int64_t> rows(static_cast<size_t>(cols));
        auto tok_at = [&](int q) { return q < pos ? history_[size_t(q)] : toks_from_pos[size_t(q - pos)]; };
        {
            std::lock_guard<std::mutex> lock(pf_m_);
            for (int q = from; q < upto; ++q)
                for (size_t e = 1; e < c.engram_layers.size(); ++e) {   // (table 0 too measured slower: the reads compete)
                    engram_hash(int(e), q, tok_at, rows.data());
                    for (int64_t r : rows)
                        if (engram_ram_.empty() || engram_slot(int(e), r) < 0) pf_queue_.push_back({int(e), r});
                }
        }
        pf_cv_.notify_all();
    }

    // The queued Engram rows, read with O_DIRECT preads (the 110 GB table must not fill the page cache the experts
    // live in) by STRATA_DS41_ENGRAM_THREADS (16) threads, each row's 4 KiB page(s), then dequantized.
    // STRATA_DS41_ENGRAM_RAM_GIB=G: up to G GiB of the Engram tables read into this process's memory, table by table
    // in layer order: the first Engram layer's table (read synchronously before layer 1) whole, the later ones' leading
    // rows (read in the background, under the layers before them). The rows past it are still read from the file.
    // STRATA_DS41_ENGRAM_ROWS=path (with ENGRAM_RAM_GIB): the rows kept in memory are the most referenced ones of a
    // profile (`--engram-profile`, ranked per table by count), merged across the tables by count per byte up to the
    // budget, instead of each table's leading rows; engram_slot_[table][row] is a row's slot in the copy, or -1.
    // Each table is read once sequentially (64 MiB pieces, 16 threads) and its kept rows copied out in row order.
    std::vector<std::vector<int32_t>> engram_slot_;   // per distinct table
    std::vector<int> engram_table_of_;                // Engram layer -> its distinct table
    bool engram_ranked_ = false;
    int64_t engram_slot(int table, int64_t row) const {
        if (engram_ranked_) {
            const auto& sl = engram_slot_[size_t(engram_table_of_[size_t(table)])];
            return sl.empty() ? -1 : sl[size_t(row)];
        }
        return row < engram_ram_rows_[size_t(table)] ? row : -1;
    }
    bool load_engram_ranked(double gib, const std::string& path) {
        const auto t0 = std::chrono::steady_clock::now();
        std::ifstream f(path, std::ios::binary);
        char magic[8] = {};
        uint32_t nt = 0;
        if (!f.read(magic, 8) || std::string(magic, 8) != "DS41ROWS" || !f.read(reinterpret_cast<char*>(&nt), 4)) {
            std::cerr << "DS41 engram rows: cannot read " << path << "\n";
            return false;
        }
        // the profile's tables are the distinct tables in Engram-layer order
        std::vector<int> table_of(engram_tables_.size(), -1);
        std::vector<int> firsts;
        for (size_t e = 0; e < engram_tables_.size(); ++e) {
            for (size_t j = 0; j < e; ++j) if (engram_tables_[j] == engram_tables_[e]) { table_of[e] = table_of[j]; break; }
            if (table_of[e] < 0) { table_of[e] = int(firsts.size()); firsts.push_back(int(e)); }
        }
        if (nt != firsts.size()) { std::cerr << "DS41 engram rows: " << nt << " tables in the profile, " << firsts.size() << " here\n"; return false; }
        struct Ranked { std::vector<int64_t> row; std::vector<uint32_t> count; size_t at = 0; int64_t rows = 0; size_t row_bytes = 0; };
        std::vector<Ranked> ranked(firsts.size());
        for (size_t k = 0; k < firsts.size(); ++k) {
            const auto& t = *engram_tables_[size_t(firsts[k])];
            uint64_t rows = 0, n = 0;
            if (!f.read(reinterpret_cast<char*>(&rows), 8) || !f.read(reinterpret_cast<char*>(&n), 8)) return false;
            ranked[k].row_bytes = ggml_row_size(ggml_type(int(t.tensor->type)), cfg_.engram_dim);
            ranked[k].rows = int64_t(t.bytes / ranked[k].row_bytes);
            if (int64_t(rows) != ranked[k].rows) { std::cerr << "DS41 engram rows: table " << k << " has " << ranked[k].rows << " rows, the profile " << rows << "\n"; return false; }
            ranked[k].row.resize(n);
            ranked[k].count.resize(n);
            std::vector<char> rec(size_t(n) * 12);
            if (!f.read(rec.data(), std::streamsize(rec.size()))) return false;
            for (uint64_t i = 0; i < n; ++i) {
                std::memcpy(&ranked[k].row[i], rec.data() + i * 12, 8);
                std::memcpy(&ranked[k].count[i], rec.data() + i * 12 + 8, 4);
            }
        }
        // the budget, by count per byte across the tables
        std::vector<std::vector<int32_t>> slots(firsts.size());
        for (size_t k = 0; k < firsts.size(); ++k) slots[k].assign(size_t(ranked[k].rows), -1);
        double left = gib * double(1ull << 30);
        std::vector<int64_t> kept(firsts.size(), 0);
        size_t referenced = 0;
        for (;;) {
            int best = -1;
            double bv = -1;
            for (size_t k = 0; k < ranked.size(); ++k) {
                if (ranked[k].at >= ranked[k].row.size()) continue;
                const double v = double(ranked[k].count[ranked[k].at]) / double(ranked[k].row_bytes);
                if (v > bv) { bv = v; best = int(k); }
            }
            if (best < 0 || left < double(ranked[size_t(best)].row_bytes)) break;
            auto& r = ranked[size_t(best)];
            const int64_t row = r.row[r.at++];
            ++referenced;
            if (row < 0 || row >= r.rows || slots[size_t(best)][size_t(row)] != -1) continue;
            slots[size_t(best)][size_t(row)] = -2;   // kept; its slot by row order below
            left -= double(r.row_bytes);
            ++kept[size_t(best)];
        }
        for (size_t k = 0; k < firsts.size(); ++k) {
            int32_t next = 0;
            for (auto& v : slots[k]) if (v == -2) v = next++;
        }
        // the copies: one sequential read of each table
        std::vector<std::thread> workers;
        std::atomic<bool> failed{false};
        size_t bytes = 0;
        std::vector<uint8_t*> ram(firsts.size(), nullptr);
        for (size_t k = 0; k < firsts.size(); ++k) {
            const auto& t = *engram_tables_[size_t(firsts[k])];
            const size_t row = ranked[k].row_bytes, n = size_t(kept[k]) * row;
            void* m = mmap(nullptr, std::max<size_t>(n, 4096), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) throw std::runtime_error("ds41: cannot allocate the Engram copy");
            madvise(m, n, MADV_HUGEPAGE);
            ram_maps_.push_back({m, std::max<size_t>(n, 4096)});
            ram[k] = static_cast<uint8_t*>(m);
            bytes += n;
            if (!n) continue;
            const int fd = ::open(t.file->path().c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
            const uint64_t base = t.file->data_start() + t.tensor->offset;
            const size_t piece = (size_t(64) << 20) / row * row, total = size_t(ranked[k].rows) * row;
            const int nt_ = 16;
            const int32_t* slot = slots[k].data();
            uint8_t* dst = ram[k];
            for (int w = 0; w < nt_; ++w)
                workers.emplace_back([&, fd, base, piece, total, row, slot, dst, w] {
                    unpin_helper();
                    std::vector<uint8_t> buf(piece);
                    for (size_t o = size_t(w) * piece; o < total && !failed; o += size_t(nt_) * piece) {
                        const size_t len = std::min(piece, total - o);
                        for (size_t got = 0; got < len;) {
                            const ssize_t r = pread(fd, buf.data() + got, len - got, off_t(base + o + got));
                            if (r <= 0) { failed = true; break; }
                            got += size_t(r);
                        }
                        if (failed) break;
                        for (size_t i = 0, r0 = o / row; i < len / row; ++i) {
                            const int32_t sl = slot[r0 + i];
                            if (sl >= 0) std::memcpy(dst + size_t(sl) * row, buf.data() + i * row, row);
                        }
                        posix_fadvise(fd, off_t(base + o), off_t(len), POSIX_FADV_DONTNEED);
                    }
                });
            for (auto& w : workers) w.join();
            workers.clear();
            ::close(fd);
        }
        if (failed) throw std::runtime_error("ds41: Engram table read failed");
        engram_ram_.assign(engram_tables_.size(), nullptr);
        engram_ram_rows_.assign(engram_tables_.size(), 0);
        for (size_t e = 0; e < engram_tables_.size(); ++e) engram_ram_[e] = ram[size_t(table_of[e])];
        engram_slot_ = std::move(slots);
        engram_table_of_ = table_of;
        engram_ranked_ = true;
        std::cerr << "DS41 engram in memory GiB=" << bytes / double(1ull << 30) << " ranked_rows=" << referenced;
        for (size_t k = 0; k < firsts.size(); ++k) std::cerr << " table" << k << "=" << kept[k] << "/" << ranked[k].rows;
        std::cerr << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
        return true;
    }
    void load_engram_ram(double gib) {
        const auto t0 = std::chrono::steady_clock::now();
        double total = 0;
        for (const auto* t : engram_tables_) total += double(t->bytes);
        double left = gib * double(1ull << 30);
        const double share = std::min(1.0, left / std::max(1.0, total));
        engram_ram_.assign(engram_tables_.size(), nullptr);
        engram_ram_rows_.assign(engram_tables_.size(), 0);
        if (left <= 0) return;
        if (const char* rows = std::getenv("STRATA_DS41_ENGRAM_ROWS"); rows && load_engram_ranked(gib, rows)) return;
        std::vector<std::thread> workers;
        std::atomic<bool> failed{false};
        size_t bytes = 0;
        for (size_t e = 0; e < engram_tables_.size(); ++e) {
            bool dup = false;   // a shared table (#30817) is read once
            for (size_t j = 0; j < e; ++j) if (engram_tables_[j] == engram_tables_[e]) { engram_ram_[e] = engram_ram_[j]; engram_ram_rows_[e] = engram_ram_rows_[j]; dup = true; }
            if (dup) continue;
            const auto& t = *engram_tables_[e];
            const size_t row = ggml_row_size(ggml_type(t.tensor->type), cfg_.engram_dim);
            const int64_t rows = int64_t(t.bytes / row), keep = std::min<int64_t>(rows, int64_t(std::max(0.0, left) / double(row)));
            left -= double(keep) * double(row);
            const size_t n = size_t(keep) * row;
            void* m = mmap(nullptr, std::max<size_t>(n, 4096), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) throw std::runtime_error("ds41: cannot allocate the Engram copy");
            madvise(m, n, MADV_HUGEPAGE);
            ram_maps_.push_back({m, std::max<size_t>(n, 4096)});
            engram_ram_[e] = static_cast<uint8_t*>(m);
            engram_ram_rows_[e] = keep;
            bytes += n;
            const int fd = ::open(t.file->path().c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
            const uint64_t base = t.file->data_start() + t.tensor->offset;
            const size_t piece = size_t(64) << 20;
            const int nt = 16;
            for (int w = 0; w < nt; ++w)
                workers.emplace_back([&, fd, base, n, m, w] {
                    unpin_helper();
                    for (size_t o = size_t(w) * piece; o < n && !failed; o += size_t(nt) * piece) {
                        const size_t len = std::min(piece, n - o);
                        for (size_t got = 0; got < len;) {
                            const ssize_t r = pread(fd, static_cast<uint8_t*>(m) + o + got, len - got, off_t(base + o + got));
                            if (r <= 0) { failed = true; break; }
                            got += size_t(r);
                        }
                        posix_fadvise(fd, off_t(base + o), off_t(len), POSIX_FADV_DONTNEED);
                    }
                });
            for (auto& w : workers) w.join();
            workers.clear();
            ::close(fd);
        }
        if (failed) throw std::runtime_error("ds41: Engram table read failed");
        std::cerr << "DS41 engram in memory GiB=" << bytes / double(1ull << 30) << " share=" << share << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }

    void upload_engram(int e) {
        const size_t per = size_t(chunk_) * cfg_.engram_cols() * cfg_.engram_dim;
        ck(cudaMemcpyAsync(engram_rows_.f(size_t(e) * per), engram_host_.data() + size_t(e) * per, per * 4, cudaMemcpyHostToDevice, stream_),
           "engram rows");
    }
    void read_engram_rows() {
        read_rows(engram_reqs_);
        if (engram_failed_) throw std::runtime_error("ds41: Engram row read failed");
    }
    template <class Reqs> void read_rows(Reqs& reqs) {   // (EngramReq is declared below)
        engram_failed_ = false;
        if (reqs.empty()) return;
        const int dim = cfg_.engram_dim;
        // rows in memory now, on this thread; threads only for the rows read from the files
        std::remove_reference_t<Reqs> misses;
        if (!engram_ram_.empty()) {
            for (const auto& r : reqs) {
                const auto& t = *engram_tables_[size_t(r.table)];
                if (const int64_t sl = engram_slot(r.table, r.row); sl >= 0) {
                    const int type = int(t.tensor->type);
                    const size_t row_bytes = ggml_row_size(ggml_type(type), dim);
                    ggml_get_type_traits(ggml_type(type))->to_float(engram_ram_[size_t(r.table)] + size_t(sl) * row_bytes, r.dst, dim);
                } else {
                    misses.push_back(r);
                }
            }
            reqs.clear();
            if (misses.empty()) return;
        }
        auto& engram_reqs_ = engram_ram_.empty() ? reqs : misses;
        if (engram_remote_) {   // one batched request for all the pages
            std::vector<strata::ds41::engram_wire::Entry> entries;
            std::vector<uint64_t> in_page;
            for (const auto& r : engram_reqs_) {
                const auto& t = *engram_tables_[size_t(r.table)];
                const size_t row_bytes = ggml_row_size(ggml_type(int(t.tensor->type)), dim);
                const uint64_t off = t.file->data_start() + t.tensor->offset + uint64_t(r.row) * row_bytes;
                const uint64_t page = off & ~uint64_t(4095);
                const std::string name = t.file->path();
                const size_t at = name.rfind("-of-");
                const uint32_t shard = at == std::string::npos || at < 5 ? 1u : uint32_t(std::atoi(name.c_str() + at - 5));
                entries.push_back({shard, uint32_t((off + row_bytes - page + 4095) & ~size_t(4095)), page});
                in_page.push_back(off - page);
            }
            std::vector<uint8_t> bytes;
            try { engram_remote_->read(entries, bytes); } catch (const std::exception& e) { std::cerr << e.what() << "\n"; engram_failed_ = true; engram_reqs_.clear(); reqs.clear(); return; }
            size_t at = 0;
            for (size_t q = 0; q < engram_reqs_.size(); ++q) {
                const auto& t = *engram_tables_[size_t(engram_reqs_[q].table)];
                ggml_get_type_traits(ggml_type(int(t.tensor->type)))->to_float(bytes.data() + at + in_page[q], engram_reqs_[q].dst, dim);
                at += entries[q].len;
            }
            engram_reqs_.clear();
            reqs.clear();
            return;
        }
        const int nt = std::min<int>(engram_threads_, int(engram_reqs_.size()));
        std::vector<std::thread> workers;
        std::atomic<bool> failed{false};
        for (int w = 0; w < nt; ++w)
            workers.emplace_back([&, w] {
                unpin_helper();
                alignas(4096) static thread_local uint8_t buf[3 * 4096];
                for (size_t q = size_t(w); q < engram_reqs_.size(); q += size_t(nt)) {
                    const auto& t = *engram_tables_[size_t(engram_reqs_[q].table)];
                    const int type = int(t.tensor->type);
                    const size_t row_bytes = ggml_row_size(ggml_type(type), dim);
                    if (const int64_t sl = engram_ram_.empty() ? -1 : engram_slot(engram_reqs_[q].table, engram_reqs_[q].row); sl >= 0) {   // in memory
                        ggml_get_type_traits(ggml_type(type))->to_float(engram_ram_[size_t(engram_reqs_[q].table)] + size_t(sl) * row_bytes,
                                                                        engram_reqs_[q].dst, dim);
                        continue;
                    }
                    const uint64_t off = t.file->data_start() + t.tensor->offset + uint64_t(engram_reqs_[q].row) * row_bytes;
                    const uint64_t page = off & ~uint64_t(4095);
                    const size_t len = (off + row_bytes - page + 4095) & ~size_t(4095);
                    if (pread(engram_fds_[size_t(engram_reqs_[q].table)], buf, len, off_t(page)) != ssize_t(len)) { failed = true; continue; }
                    ggml_get_type_traits(ggml_type(type))->to_float(buf + (off - page), engram_reqs_[q].dst, dim);
                }
            });
        for (auto& w : workers) w.join();
        engram_reqs_.clear();
        reqs.clear();
        if (failed) engram_failed_ = true;
    }

    // layer l's Engram (the e-th): rows already on the GPU
    void engram(int l, int e) {
        const auto& c = cfg_;
        const std::string p = "blk." + std::to_string(l) + ".";
        const size_t per = size_t(c.engram_cols()) * c.engram_dim;   // floats per token of this layer's rows
        const float* rows = engram_rows_.f(size_t(e) * chunk_ * per + size_t(engram_off_) * per);
        const int step = engram_tile_ > 0 ? engram_tile_ : tokens_;
        for (int t0 = 0; t0 < tokens_; t0 += step) {   // ENGRAM_TILE: the projection and the gate n tokens at a time
            const int n = std::min(step, tokens_ - t0);
            mat(p + "engram_kv.weight", rows + size_t(t0) * per, engram_kv_.f(), n);
            if (n == 1 && observe && t0 + n == tokens_) observe("engram_kv", l, engram_kv_.f(), size_t(5) * c.hidden);
            k::ds41_engram(h_.f(size_t(t0) * 4 * c.hidden), engram_kv_.f(), w(p + "engram_gate.weight"), c.hidden, c.rms_eps, stream_, n);
        }
    }

    const float* table(int i, int which) const { return cs_.f((size_t(rope_off_ + i) * kTables + size_t(which)) * 64); }

    // drop the pass's first d tokens: the per-token state that outlives a layer moves down by d
    void shift_down(void* base, size_t per, int d, int keep) {
        uint8_t* b = static_cast<uint8_t*>(base);
        for (int j = 0; j < keep; j += d)   // copies of at most d tokens never overlap, and run in order
            ck(cudaMemcpyAsync(b + size_t(j) * per, b + size_t(j + d) * per, size_t(std::min(d, keep - j)) * per, cudaMemcpyDeviceToDevice,
                               stream_), "narrow");
    }
    void narrow(int d, bool have_pre) {
        const auto& c = cfg_;
        const int keep = tokens_ - d;
        shift_down(h_.p, size_t(4) * c.hidden * 4, d, keep);
        if (have_pre) shift_down(coeff_ffn_.p, 24 * 4, d, keep);
        shift_down(cs_.p, size_t(kTables) * 64 * 4, d, keep);
        const size_t per_e = size_t(c.engram_cols()) * c.engram_dim;
        for (size_t e = 0; e < c.engram_layers.size(); ++e) shift_down(engram_rows_.f(e * chunk_ * per_e), per_e * 4, d, keep);
        for (int l = 0; l < c.layers; ++l) {
            if (c.topk_src[size_t(l)] != l) continue;
            shift_down(sel_[size_t(l)].p, size_t(c.index_top_k) * 4, d, keep);
            if (sel_on_gpu_[size_t(l)]) shift_down(sel_cnt_[size_t(l)].p, 4, d, keep);
            auto& n = sel_count_[size_t(l)];
            std::copy(n.begin() + d, n.begin() + d + keep, n.begin());
        }
        shift_down(cand_dev_.p, size_t(cand_stride_), d, keep);
        std::move(candidates_.begin() + d, candidates_.begin() + d + keep, candidates_.begin());
        pos0_ += d;
        tokens_ = keep;
        skip_ += d;
    }

    void attention(int l) {
        const auto& c = cfg_;
        const int T = tokens_;
        const std::string p = "blk." + std::to_string(l) + ".";
        const int ratio = c.ratio[size_t(l)];
        const int fwd = ratio ? 2 : 0, back = ratio ? 3 : 1;
        const int nope = c.head_dim - c.rope_dim, pairs = c.rope_dim / 2, step = kTables * 64;
        const bool hook = T == 1 && observe && !sess_;
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(1);
        quantize_xn();
        mat_xn(p + "attn_q_a.weight", qa_.f());
        k::glm_rms_norm_rows(qa_.f(), w(p + "attn_q_a_norm.weight"), qr_.f(), c.q_rank, T, c.rms_eps, stream_);
        if (hook) observe("qr_norm", l, qr_.f(), size_t(c.q_rank));
        mat(p + "attn_q_b.weight", qr_.f(), q_.f(), T);
        k::ds41_rope(q_.f(), T * c.heads, c.head_dim, nope, pairs, table(0, fwd), stream_, c.heads, step);
        if (hook) observe("q", l, q_.f(), size_t(c.heads) * c.head_dim);
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(2);
        // the window keys: normalized, rotated, into the ring (slot = position % ring)
        mat_xn(p + "attn_kv.weight", kv_.f());
        k::glm_rms_norm_rows(kv_.f(), w(p + "attn_kv_a_norm.weight"), kvn_.f(), c.head_dim, T, c.rms_eps, stream_);
        k::ds41_rope(kvn_.f(), T, c.head_dim, nope, pairs, table(0, fwd), stream_, 1, step);
        if (sess_) k::ds41_scatter_rows(kvn_.f(), T, c.head_dim, tpos_dev_.i() + T, bwin_[size_t(l)].f(), size_t(c.head_dim), stream_);
        else if (T < 16)   // (16+ tokens: each attention batch writes its own rows just before it reads them)
        for (int i = 0; i < T;) {   // contiguous runs of ring slots
            const int slot = (pos0_ + i) % ring_, run = std::min(T - i, ring_ - slot);
            ck(cudaMemcpyAsync(windows_[size_t(l)].f(size_t(slot) * c.head_dim), kvn_.f(size_t(i) * c.head_dim),
                               size_t(run) * c.head_dim * 4, cudaMemcpyDeviceToDevice, stream_), "window");
            i += run;
        }
        if (hook) observe("kv", l, kvn_.f(), size_t(c.head_dim));
        const float* comp_rows = nullptr;
        const int src = ratio ? c.kv_src[size_t(l)] : -1, ts = ratio ? c.topk_src[size_t(l)] : -1;
        if (ratio) {
            if (src == l) { if (sess_) compress_sessions(l); else compress(l); }
            comp_rows = sess_ ? (kv_half_ ? nullptr : bcache_[size_t(src)].f()) : caches_[size_t(src)].f();
            if (ts == l) { if (sess_) index_sessions(l); else index(l); sel_on_gpu_[size_t(l)] = gpu_sel_; }
        }
        const float* sinks = w(p + "attn_sinks.weight");
        const float scale = 1.0f / std::sqrt(float(c.head_dim));
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(3);
        if (sess_) {   // sessions: one launch pair, each token over its own ring and cache
            int* info = attn_info_;
            int bound = 1;
            for (int i = 0; i < T; ++i) {
                const int pos = tpos_host_[i];
                if (pos < 0 || pos == INT32_MAX) throw std::runtime_error("ds41: session attention position bounds");
                const int rows = std::min(pos + 1, c.window);
                const int nsel = ratio && (pos + 1) / ratio > 0 ? std::min(c.index_top_k, (pos + 1) / ratio) : 0;   // the bound (count on the GPU)
                const int first = (pos - rows + 1) % ring_b_;
                if (ring_b_ < 1 || first < 0 || first >= ring_b_ || rows < 1 || rows > ring_b_ || nsel < 0 ||
                    nsel > c.index_top_k || rows > 4096 - nsel)
                    throw std::runtime_error("ds41: session attention window bounds");
                info[4 * i] = first;
                info[4 * i + 1] = rows;
                info[4 * i + 2] = nsel;
                info[4 * i + 3] = 0;
                bound = std::max(bound, rows + nsel);
            }
            if (bound < 1 || bound > 4096) throw std::runtime_error("ds41: session attention bound bounds");
            ck(cudaMemcpyAsync(attn_info_dev_.p, info, size_t(T) * 16, cudaMemcpyHostToDevice, stream_), "attention info");
            const size_t rw = size_t(row_width());
            const int* sel = ratio ? sel_[size_t(ts)].i() : attn_info_dev_.i();
            int sel_stride = c.index_top_k;
            size_t css = ratio ? rows_b(src) * rw : 0;
            if (kv_half_ && ratio) {   // FP16 / Q8_0: each token's selected rows as floats, [T][top_k][rw]
                if (T > kGatherTokens) throw std::runtime_error("ds41: sessions wider than the gather buffer");
                if (kv_kind_ == 1)
                    k::ds41_gather_sel_h(reinterpret_cast<const uint16_t*>(bcache_[size_t(src)].p), int(rw), sel, c.index_top_k, T, 0, ratio,
                                         kv_sel_.f(), stream_, tpos_dev_.i(), rows_b(src) * rw);
                else
                    k::ds41_gather_sel_q8(bcache_[size_t(src)].b(), kv_row_bytes(), int(rw), sel, c.index_top_k, T, 0, ratio, kv_sel_.f(), stream_,
                                          tpos_dev_.i(), rows_b(src) * kv_row_bytes());
                comp_rows = kv_sel_.f();
                sel = sel_iota_.i();   // row j of each token's block
                sel_stride = 0;
                css = size_t(c.index_top_k) * rw;
            }
            k::ds41_attention_multi(q_.f(), T, c.heads, c.head_dim, bwin_[size_t(l)].f(), ring_b_, attn_info_dev_.i(), bound, comp_rows,
                                    comp_rows ? row_width() : 0, sel, sel_stride, ratio ? sel_cnt_[size_t(ts)].i() : nullptr, sinks, scale, attn_scores_.f(),
                                    att_.f(), stream_, attn_scores_.bytes, size_t(ring_b_) * c.head_dim, css);
        } else if (T >= 16) {   // one launch for the pass: per-token {first slot, window rows, selected rows}
            if (ring_ <= 0 || c.window <= 0 || c.window > ring_ || c.index_top_k < 1 || c.index_top_k > 4096 || sel_base_ < 0)
                throw std::runtime_error("ds41: batched attention geometry bounds");
            if (ratio && (src < 0 || size_t(src) >= caches_.size()))
                throw std::runtime_error("ds41: batched attention cache layer bounds");
            if (ratio && (ts < 0 || size_t(ts) >= sel_count_.size() || size_t(ts) >= sel_.size() ||
                          size_t(ts) >= sel_on_gpu_.size()))
                throw std::runtime_error("ds41: batched attention selection layer bounds");
            if (ratio && (size_t(sel_base_) > sel_count_[size_t(ts)].size() ||
                          size_t(T) > sel_count_[size_t(ts)].size() - size_t(sel_base_)))
                throw std::runtime_error("ds41: batched attention selection count span bounds");
            int* info = attn_info_;
            const size_t host_info_bytes = checked_size_mul(size_t(T), size_t(16), "attention host info span");
            if (!attn_info_ || host_info_bytes > attn_info_host_bytes_)
                throw std::runtime_error("ds41: batched attention host info capacity");
            require_device_range(attn_info_dev_, 0, host_info_bytes, "attention info upload");
            for (int i = 0; i < T; ++i) {
                const int64_t pos64 = int64_t(pos0_) + i;
                if (pos64 < 0 || pos64 >= INT32_MAX) throw std::runtime_error("ds41: batched attention position bounds");
                const int pos = int(pos64), rows = int(std::min<int64_t>(pos64 + 1, c.window));
                const int64_t first64 = (pos64 - rows + 1) % ring_;
                if (rows < 1 || rows > ring_ || first64 < 0 || first64 >= ring_)
                    throw std::runtime_error("ds41: batched attention window bounds");
                int nsel = 0;
                if (ratio && !sel_on_gpu_[size_t(ts)] && (pos64 + 1) / ratio > 0)
                    nsel = sel_count_[size_t(ts)][size_t(sel_base_) + size_t(i)];
                if (nsel < 0 || nsel > c.index_top_k || rows > 4096 - nsel)
                    throw std::runtime_error("ds41: batched attention selected-row bounds");
                info[4 * i] = int(first64);
                info[4 * i + 1] = rows;
                info[4 * i + 2] = nsel;
                info[4 * i + 3] = 0;
            }
            ck(cudaMemcpyAsync(attn_info_dev_.p, info, size_t(T) * 16, cudaMemcpyHostToDevice, stream_), "attention info");
            if (ratio && sel_on_gpu_[size_t(ts)])
                k::ds41_patch_info(attn_info_dev_.i(), sel_cnt_[size_t(ts)].i() + sel_base_, T, pos0_, ratio, stream_);
            // STRATA_DS41_ATTN_SUB (256): tokens per launch, so no single kernel holds the GPU for long
            const size_t qw = size_t(c.heads) * c.head_dim;
            const bool half_cache = kv_half_ && ratio;   // FP16 cache: the kernel reads the halves directly
            const int stride = attn_sub_;
            for (int t0 = 0; t0 < T; t0 += stride) {
                const int n = std::min(stride, T - t0);
                for (int i = 0; i < n;) {   // this batch's window rows, contiguous runs of ring slots
                    const int slot = (pos0_ + t0 + i) % ring_, run = std::min(n - i, ring_ - slot);
                    ck(cudaMemcpyAsync(windows_[size_t(l)].f(size_t(slot) * c.head_dim), kvn_.f(size_t(t0 + i) * c.head_dim),
                                       size_t(run) * c.head_dim * 4, cudaMemcpyDeviceToDevice, stream_), "window");
                    i += run;
                }
                const size_t q_offset = checked_size_mul(checked_size_mul(size_t(t0), qw, "attention q offset"), sizeof(float), "attention q offset");
                const size_t io_offset = checked_size_mul(size_t(t0), size_t(16), "attention info offset");
                const size_t tensor_bytes = checked_size_mul(checked_size_mul(size_t(n), qw, "attention tensor span"), sizeof(float), "attention tensor span");
                const int* info_batch = static_cast<const int*>(device_range_ptr(attn_info_dev_, io_offset, size_t(n) * 16, "attention info"));
                const float* q_batch = static_cast<const float*>(device_range_ptr(q_, q_offset, tensor_bytes, "attention q"));
                float* out_batch = static_cast<float*>(device_range_ptr(att_, q_offset, tensor_bytes, "attention output"));
                (void)device_range_ptr(windows_[size_t(l)], 0,
                                       checked_size_mul(checked_size_mul(size_t(ring_), size_t(c.head_dim), "attention ring span"), sizeof(float),
                                                        "attention ring span"),
                                       "attention ring");
                const int* sel_batch = nullptr;
                size_t comp_rows = 0;
                if (ratio) {
                    comp_rows = rows_b(src);
                    if (comp_rows > size_t(INT32_MAX)) throw std::runtime_error("ds41: attention cache row count exceeds int32");
                    const size_t cache_row_bytes = checked_size_mul(size_t(row_width()), half_cache ? sizeof(uint16_t) : sizeof(float), "attention cache row");
                    const size_t cache_bytes = checked_size_mul(comp_rows, cache_row_bytes, "attention cache span");
                    require_device_range(caches_[size_t(src)], 0, cache_bytes, "attention compressed cache");
                    const size_t sel_row = size_t(sel_base_) + size_t(t0);
                    const size_t sel_offset = checked_size_mul(checked_size_mul(sel_row, size_t(c.index_top_k), "attention selection offset"), sizeof(int),
                                                               "attention selection offset");
                    const size_t sel_bytes = checked_size_mul(checked_size_mul(size_t(n), size_t(c.index_top_k), "attention selection span"), sizeof(int),
                                                              "attention selection span");
                    sel_batch = static_cast<const int*>(device_range_ptr(sel_[size_t(ts)], sel_offset, sel_bytes, "attention selections"));
                }
                static const char* const wmma_env = std::getenv("STRATA_DS41_ATTN_WMMA");
                static const bool wmma_requested = wmma_env && (std::string(wmma_env) == "1" || std::string(wmma_env) == "2");
                if (wmma_requested && ratio && !half_cache)
                    throw std::runtime_error("ds41: gfx12 WMMA attention requires the FP16 KV cache");
                k::ds41_attention_batched(q_batch, n, c.heads, c.head_dim, windows_[size_t(l)].f(), ring_, info_batch,
                                          ratio ? static_cast<const float*>(caches_[size_t(src)].p) : nullptr, row_width(), sel_batch,
                                          c.index_top_k, sinks, scale, out_batch, stream_, half_cache, int(comp_rows),
                                          attn_bounds_check_ ? attn_error_.i() : nullptr);
            }
            // The WMMA diagnostic kernel returns early for invalid row metadata. Check its
            // status before the next layer can consume a partially unwritten attention output.
            if (attn_bounds_check_) {
                int bounds_error = 0;
                ck(cudaMemcpyAsync(&bounds_error, attn_error_.p, sizeof(bounds_error), cudaMemcpyDeviceToHost, stream_),
                   "attention bounds result");
                ck(cudaStreamSynchronize(stream_), "attention bounds sync");
                if (bounds_error)
                    throw std::runtime_error("ds41: gfx12 WMMA attention rejected invalid row metadata or selection indices (code " +
                                             std::to_string(bounds_error) + ")");
            }
        } else if (static bool multi = std::getenv("STRATA_DS41_ATTN_MULTI") && std::string(std::getenv("STRATA_DS41_ATTN_MULTI")) == "1";
                   multi && T >= 2 && T <= 8) {   // a few tokens (a verify pass): one launch pair over all of them
            int* info = attn_info_;
            int bound = 1;
            const bool dev = ratio && sel_on_gpu_[size_t(ts)];
            for (int i = 0; i < T; ++i) {
                if (pos0_ < 0 || pos0_ > INT32_MAX - i - 1) throw std::runtime_error("ds41: multi attention position bounds");
                const int pos = pos0_ + i, rows = std::min(pos + 1, c.window);
                int nsel = 0;
                if (ratio && (pos + 1) / ratio > 0) nsel = dev ? std::min(c.index_top_k, (pos + 1) / ratio) : sel_count_[size_t(ts)][size_t(i)];
                const int first = (pos - rows + 1) % ring_;
                if (ring_ < 1 || first < 0 || first >= ring_ || rows < 1 || rows > ring_ || nsel < 0 ||
                    nsel > c.index_top_k || rows > 4096 - nsel)
                    throw std::runtime_error("ds41: multi attention window bounds");
                info[4 * i] = first;
                info[4 * i + 1] = rows;
                info[4 * i + 2] = nsel;
                info[4 * i + 3] = 0;
                bound = std::max(bound, rows + nsel);
            }
            if (bound < 1 || bound > 4096) throw std::runtime_error("ds41: multi attention bound bounds");
            // the same rows as the last upload: skip it (only with the selected counts on the device: the host's
            // selection, and so info[2], varies per layer)
            const long key = (long(pos0_) * 64 + T) * 8 + ratio * 2 + (dev ? 1 : 0);
            if (!fuse_ || !dev || key != attn_info_key_) {
                ck(cudaMemcpyAsync(attn_info_dev_.p, info, size_t(T) * 16, cudaMemcpyHostToDevice, stream_), "attention info");
                attn_info_key_ = key;
            }
            const bool gathered = kv_half_ && ratio;
            const float* comp_multi = gathered ? gather_sel(src, ts, 0, T, ratio) : comp_rows;   // null: no compressed rows
            k::ds41_attention_multi(q_.f(), T, c.heads, c.head_dim, windows_[size_t(l)].f(), ring_, attn_info_dev_.i(), bound,
                                    comp_multi, comp_multi ? row_width() : 0,
                                    gathered ? sel_iota_.i() : ratio ? sel_[size_t(ts)].i() : attn_info_dev_.i(), c.index_top_k,
                                    dev ? sel_cnt_[size_t(ts)].i() : nullptr, sinks, scale, attn_scores_.f(), att_.f(), stream_, attn_scores_.bytes);
        } else
        for (int i = 0; i < T; ++i) {
            const int pos = pos0_ + i, rows = std::min(pos + 1, c.window), first = (pos - rows + 1) % ring_;
            int nsel = 0;
            const int* sel = nullptr;
            const int* nsel_dev = nullptr;   // a selection made on the GPU: its count stays there (nsel the bound)
            if (ratio && (pos + 1) / ratio > 0) {
                const bool dev = sel_on_gpu_[size_t(ts)];
                nsel = dev ? std::min(c.index_top_k, (pos + 1) / ratio) : sel_count_[size_t(ts)][size_t(i)];
                if (dev) nsel_dev = sel_cnt_[size_t(ts)].i() + i;
                sel = sel_[size_t(ts)].i() + size_t(i) * c.index_top_k;
            }
            const float* comp_i = comp_rows;
            if (kv_half_ && ratio && sel) {   // FP16: this token's selected rows as floats
                comp_i = gather_sel(src, ts, i, 1, ratio);
                sel = sel_iota_.i();
            }
            k::ds41_attention(q_.f(size_t(i) * c.heads * c.head_dim), c.heads, c.head_dim, windows_[size_t(l)].f(), first, ring_,
                              rows, comp_i, row_width(), sel, nsel, sinks, scale, attn_scores_.f(),
                              att_.f(size_t(i) * c.heads * c.head_dim), stream_, nsel_dev);
        }
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(4);
        k::ds41_rope(att_.f(), T * c.heads, c.head_dim, nope, pairs, table(0, back), stream_, c.heads, step);
        if (hook) observe("attn_derope", l, att_.f(), size_t(c.heads) * c.head_dim);
        // grouped output projection: group g's out_rank rows read heads [g heads/groups, (g+1) heads/groups)
        const int group_in = c.heads / c.out_groups * c.head_dim, width = c.heads * c.head_dim;
        const Weight& A = weight(p + "attn_output_a.weight");
        if (A.in != group_in) throw std::runtime_error("ds41: attn_output_a width");
        if (A.type == GGML_TYPE_F32 || A.type == GGML_TYPE_BF16) throw std::runtime_error("ds41: float attn_output_a");
        if (T > 8 && dense_mmq(A.type, static_cast<const uint8_t*>(A.data.p), att_.f(), width, oa_.f(), c.out_groups * c.out_rank, group_in,
                               c.out_rank, T)) {   // group 0 by MMQ: the others too (same type and shape)
            for (int g = 1; g < c.out_groups; ++g)
                dense_mmq(A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes, att_.f(size_t(g) * group_in), width,
                          oa_.f(size_t(g) * c.out_rank), c.out_groups * c.out_rank, group_in, c.out_rank, T);
            for (int t0 = 0; t0 < T; t0 += (outa_tile_ > 0 ? outa_tile_ : T)) {
                const int n = std::min(outa_tile_ > 0 ? outa_tile_ : T, T - t0);
                mat(p + "attn_output_b.weight", oa_.f(size_t(t0) * c.out_groups * c.out_rank), y_.f(size_t(t0) * c.hidden), n);
            }
            return;
        }
        if (T >= gemm_min_) {
            const int xt = outa_tile_ > 0 ? outa_tile_ : T;   // OUTA_TILE: rows per conversion + projection call
            for (int t0 = 0; t0 < T; t0 += xt) {
                const int n = std::min(xt, T - t0);
                k::glm_f16(att_.f(size_t(t0) * width), xh_.h(), int64_t(n) * width, stream_);
                uint16_t* xg = xh_.h() + size_t(n) * width;   // one group's columns, packed
                for (int g = 0; g < c.out_groups; ++g) {
                    ck(cudaMemcpy2DAsync(xg, size_t(group_in) * 2, xh_.h() + size_t(g) * group_in, size_t(width) * 2, size_t(group_in) * 2,
                                         size_t(n), cudaMemcpyDeviceToDevice, stream_), "group pack");
                    gemm_->native(xg, A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes,
                                  oa_.f(size_t(t0) * c.out_groups * c.out_rank + size_t(g) * c.out_rank), n, c.out_rank, group_in,
                                  int64_t(c.out_groups) * c.out_rank);
                }
            }
            for (int t0 = 0; t0 < T; t0 += xt) {
                const int n = std::min(xt, T - t0);
                mat(p + "attn_output_b.weight", oa_.f(size_t(t0) * c.out_groups * c.out_rank), y_.f(size_t(t0) * c.hidden), n);
            }
            if (hook) observe("attn_out", l, y_.f(), size_t(c.hidden));
            return;
        }
        static const bool regroup = std::getenv("STRATA_DS41_OUT_REGROUP") && std::string(std::getenv("STRATA_DS41_OUT_REGROUP")) == "1";
        if (regroup && T >= 2 && T <= 8 && chunk_ >= c.out_groups * T) {   // group-major: one regroup, one quantize, a GEMV per group over its T columns
            float* attg = att_.f(size_t(T) * width);   // att_ holds chunk_ tokens: the pass's second half is free when T <= chunk_ / 2
            k::ds41_swap_blocks(att_.f(), attg, T, c.out_groups, group_in, stream_);
            const size_t colg = k::native_q8_1_bytes(group_in, 1);
            quantize(attg, q8_.p, group_in, T * c.out_groups);
            static const bool exact = trunk_exact();
            if (!(fuse_ && !exact && k::native_mmvq_groups(A.type, A.data.p, q8_.p, oa_part_.f(), group_in, c.out_rank, T, c.out_groups, stream_)))
                for (int g = 0; g < c.out_groups; ++g)
                    mmvq(A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes,
                         static_cast<const uint8_t*>(q8_.p) + size_t(g) * T * colg, oa_part_.f(size_t(g) * T * c.out_rank), group_in, c.out_rank, T);
            k::ds41_swap_blocks(oa_part_.f(), oa_.f(), c.out_groups, T, c.out_rank, stream_);
            mat(p + "attn_output_b.weight", oa_.f(), y_.f(), T);
            return;
        }
        quantize(att_.f(), q8_.p, width, T);
        const size_t col_all = k::native_q8_1_bytes(width, 1), col_group = k::native_q8_1_bytes(group_in, 1);
        if (T == 1 && col_all == col_group * size_t(c.out_groups)) {   // one token: each group's input and output are contiguous
            for (int g = 0; g < c.out_groups; ++g)
                mmvq(A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes,
                     static_cast<const uint8_t*>(q8_.p) + size_t(g) * col_group, oa_.f(size_t(g) * c.out_rank), group_in, c.out_rank, 1);
            mat(p + "attn_output_b.weight", oa_.f(), y_.f(), T);
            if (hook) observe("attn_out", l, y_.f(), size_t(c.hidden));
            return;
        }
        for (int g = 0; g < c.out_groups; ++g) {
            ck(cudaMemcpy2DAsync(q8g_.p, col_group, static_cast<const uint8_t*>(q8_.p) + size_t(g) * col_group, col_all, col_group,
                                 size_t(T), cudaMemcpyDeviceToDevice, stream_), "group pack");
            mmvq(A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes, q8g_.p, oa_part_.f(),
                 group_in, c.out_rank, T);
            ck(cudaMemcpy2DAsync(oa_.f(size_t(g) * c.out_rank), size_t(c.out_groups) * c.out_rank * 4, oa_part_.p,
                                 size_t(c.out_rank) * 4, size_t(c.out_rank) * 4, size_t(T), cudaMemcpyDeviceToDevice, stream_),
               "group unpack");
        }
        mat(p + "attn_output_b.weight", oa_.f(), y_.f(), T);
        if (hook) observe("attn_out", l, y_.f(), size_t(c.hidden));
    }

    size_t rows_b(int l) const { return size_t(context_ / cfg_.ratio[size_t(l)] + 1); }   // a session's cache rows (layer l)
    // session 0's indexer keys of a KV source (FP16: the pointer cast to float*)
    const float* session_keys(int src) const {
        return kv_kind_ == 1 ? reinterpret_cast<const float*>(reinterpret_cast<const uint16_t*>(bcache_[size_t(src)].p) + cfg_.head_dim)
                             : bcache_[size_t(src)].f(cfg_.head_dim);
    }
    // sessions: the compressor's rows of the pass's odd (ratio 2) or all (ratio 1) positions into each session's cache
    void compress_sessions(int l) {
        const auto& c = cfg_;
        const int T = tokens_, ratio = c.ratio[size_t(l)];
        if (ratio > 2) throw std::runtime_error("ds41: only compress ratios 1 and 2");
        const std::string p = "blk." + std::to_string(l) + ".";
        mat(p + "attn_compressor_kv.weight", xn_.f(), comp_v_.f(), T);
        if (ratio > 1) mat(p + "attn_compressor_gate.weight", xn_.f(), comp_s_.f(), T);
        const size_t hd = size_t(c.head_dim), rw = size_t(row_width()), id = size_t(c.index_dim);
        float* merged = att_.f();   // free until this layer's attention
        float* normed = att_.f(size_t(T) * hd);
        float* rows = att_.f(2 * size_t(T) * hd);
        float* keys = rows + size_t(T) * rw;
        float* knormed = keys + size_t(T) * id;
        k::ds41_compress_sessions(comp_v_.f(), comp_s_.f(), ratio > 1 ? bstash_v_[size_t(l)].f() : nullptr, ratio > 1 ? bstash_s_[size_t(l)].f() : nullptr,
                                  c.head_dim, tpos_dev_.i(), ratio, T, merged, stream_);
        k::glm_rms_norm_rows(merged, w(p + "attn_compressor_norm.weight"), normed, c.head_dim, T, c.rms_eps, stream_);
        ck(cudaMemcpy2DAsync(rows, rw * 4, normed, hd * 4, hd * 4, size_t(T), cudaMemcpyDeviceToDevice, stream_), "rows");
        mat(p + "indexer.attn_k.weight", normed, keys, T);   // from the RoPE-free latent
        k::glm_rms_norm_rows(keys, w(p + "indexer.k_norm.weight"), knormed, c.index_dim, T, c.rms_eps, stream_);
        ck(cudaMemcpy2DAsync(rows + hd, rw * 4, knormed, id * 4, id * 4, size_t(T), cudaMemcpyDeviceToDevice, stream_), "index rows");
        const float* cs = table(0, ratio == 1 ? 2 : 4);   // a ratio-2 row stands for its group's first position
        k::ds41_rope(rows + hd, T, int(rw), c.index_dim - c.rope_dim, c.rope_dim / 2, cs, stream_, 1, kTables * 64);
        k::ds41_rope(rows, T, int(rw), c.head_dim - c.rope_dim, c.rope_dim / 2, cs, stream_, 1, kTables * 64);
        const int* dst_row = tpos_dev_.i() + (ratio == 1 ? 2 : 3) * T;
        if (kv_kind_ == 0) {
            k::ds41_scatter_rows(rows, T, int(rw), dst_row, bcache_[size_t(l)].f(), rw, stream_);
            return;
        }
        uint8_t* packed = reinterpret_cast<uint8_t*>(knormed + size_t(T) * id);   // the rows as stored, then scattered
        if (kv_kind_ == 1) k::glm_f16(rows, reinterpret_cast<uint16_t*>(packed), int64_t(T) * int64_t(rw), stream_);
        else k::ds41_quant_rows_q8(rows, T, int(rw), packed, kv_row_bytes(), stream_);
        k::ds41_scatter_bytes(packed, T, kv_row_bytes(), dst_row, bcache_[size_t(l)].p, kv_row_bytes(), stream_);
    }
    // sessions: each token's top-k over its own session's visible rows, on the GPU
    void index_sessions(int l) {
        const auto& c = cfg_;
        const int T = tokens_, ratio = c.ratio[size_t(l)];
        const std::string p = "blk." + std::to_string(l) + ".";
        if (T > score_tile_) throw std::runtime_error("ds41: sessions wider than the score tile");
        mat(p + "indexer.attn_q_b.weight", qr_.f(), iq_.f(), T);
        k::ds41_rope(iq_.f(), T * c.index_heads, c.index_dim, c.index_dim - c.rope_dim, c.rope_dim / 2, table(0, 2), stream_,
                     c.index_heads, kTables * 64);
        mat(p + "indexer.proj.weight", xn_.f(), iw_.f(), T);
        const int src = c.kv_src[size_t(l)];
        const float w_scale = 1.0f / std::sqrt(float(c.index_dim * c.index_heads));
        int widest = 0;
        for (int i = 0; i < T; ++i) widest = std::max(widest, (tpos_host_[i] + 1) / ratio);
        const auto i0 = std::chrono::steady_clock::now();
        if (kv_kind_ == 2)
            k::ds41_index_score_q8(iq_.f(), c.index_heads, c.index_dim, bcache_[size_t(src)].b(), kv_row_bytes(), c.head_dim, 0, ratio, T, widest,
                                   iw_.f(), w_scale, scores_.f(), score_stride_, stream_, tpos_dev_.i(), rows_b(src) * kv_row_bytes());
        else
            k::ds41_index_score_sessions(iq_.f(), c.index_heads, c.index_dim, session_keys(src), row_width(), rows_b(src) * size_t(row_width()),
                                         tpos_dev_.i(), widest, ratio, T, iw_.f(), w_scale, scores_.f(), score_stride_, stream_, kv_kind_ == 1);
        const int mode = c.candidate_source == l ? 1 : c.candidate_source >= 0 && c.candidate_source < l ? 2 : 0;
        k::ds41_index_topk(scores_.f(), score_stride_, T, 0, ratio, c.index_top_k, sel_[size_t(l)].i(), sel_cnt_[size_t(l)].i(), mode,
                           static_cast<uint8_t*>(cand_dev_.p), cand_stride_, c.candidate_block, c.candidate_blocks, stream_, tpos_dev_.i());
        prof_[5] += std::chrono::duration<double>(std::chrono::steady_clock::now() - i0).count();
        gpu_sel_ = true;
    }

    // the KV-source layer's compressor: one compressed row per `ratio` positions, in position order
    void compress(int l) {
        const auto& c = cfg_;
        const int T = tokens_;
        const std::string p = "blk." + std::to_string(l) + ".";
        const int ratio = c.ratio[size_t(l)];
        if (ratio > 2) throw std::runtime_error("ds41: only compress ratios 1 and 2");
        if (fuse_) { mat_xn(p + "attn_compressor_kv.weight", comp_v_.f()); if (ratio > 1) mat_xn(p + "attn_compressor_gate.weight", comp_s_.f()); }
        else {
        mat(p + "attn_compressor_kv.weight", xn_.f(), comp_v_.f(), T);
        if (ratio > 1) mat(p + "attn_compressor_gate.weight", xn_.f(), comp_s_.f(), T);
        }
        const size_t hd = size_t(c.head_dim);
        if (T >= 16) {   // the pass's rows at once: rows i_first, i_first + ratio, ... (each group's last position)
            const int i_first = ratio == 1 ? 0 : (pos0_ % 2 ? 0 : 1);
            const int rows = (T - i_first + ratio - 1) / ratio, g0 = (pos0_ + i_first) / ratio;
            const size_t rw = size_t(row_width());
            float* merged = att_.f();   // free until this layer's attention: [rows][hd] x 4 scratch
            float* normed = att_.f(size_t(T) * hd);
            float* keys = att_.f(2 * size_t(T) * hd);
            float* knormed = att_.f(3 * size_t(T) * hd);
            const float* v = comp_v_.f();
            if (ratio > 1) {
                int r = 0, j0 = 0;
                if (pos0_ % 2) {   // the first position closes the group the previous pass stashed
                    k::ds41_compress2(stash_v_[size_t(l)].f(), stash_s_[size_t(l)].f(), comp_v_.f(), comp_s_.f(), merged, c.head_dim, stream_);
                    r = 1;
                    j0 = 1;
                }
                k::ds41_compress2_rows(comp_v_.f(size_t(j0) * hd), comp_s_.f(size_t(j0) * hd), merged + size_t(r) * hd, rows - r, c.head_dim, stream_);
                v = merged;
                if ((pos0_ + T - 1) % 2 == 0) {   // the last position opens a group: stash it
                    ck(cudaMemcpyAsync(stash_v_[size_t(l)].p, comp_v_.f(size_t(T - 1) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "stash");
                    ck(cudaMemcpyAsync(stash_s_[size_t(l)].p, comp_s_.f(size_t(T - 1) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "stash");
                }
            }
            if (rows < 1) return;
            float* dst = kv_half_ ? kv_rows_.f() : caches_[size_t(l)].f(size_t(g0) * rw);   // FP16: built here, then stored
            k::glm_rms_norm_rows(v, w(p + "attn_compressor_norm.weight"), normed, c.head_dim, rows, c.rms_eps, stream_);
            ck(cudaMemcpy2DAsync(dst, rw * 4, normed, hd * 4, hd * 4, size_t(rows), cudaMemcpyDeviceToDevice, stream_), "rows");
            mat(p + "indexer.attn_k.weight", normed, keys, rows);   // from the RoPE-free latent
            k::glm_rms_norm_rows(keys, w(p + "indexer.k_norm.weight"), knormed, c.index_dim, rows, c.rms_eps, stream_);
            ck(cudaMemcpy2DAsync(dst + hd, rw * 4, knormed, size_t(c.index_dim) * 4, size_t(c.index_dim) * 4, size_t(rows),
                                 cudaMemcpyDeviceToDevice, stream_), "index rows");
            const float* cs = table(i_first, ratio == 1 ? 2 : 4);
            const int step = ratio * kTables * 64;
            k::ds41_rope(dst + hd, rows, int(rw), c.index_dim - c.rope_dim, c.rope_dim / 2, cs, stream_, 1, step);
            k::ds41_rope(dst, rows, int(rw), c.head_dim - c.rope_dim, c.rope_dim / 2, cs, stream_, 1, step);
            if (kv_kind_ == 1) k::glm_f16(dst, reinterpret_cast<uint16_t*>(caches_[size_t(l)].p) + size_t(g0) * rw, int64_t(rows) * int64_t(rw), stream_);
            if (kv_kind_ == 2) k::ds41_quant_rows_q8(dst, rows, int(rw), caches_[size_t(l)].b(size_t(g0) * kv_row_bytes()), kv_row_bytes(), stream_);
            return;
        }
        for (int i = 0; i < T; ++i) {
            const int pos = pos0_ + i;
            const float* v = comp_v_.f(size_t(i) * hd);
            if (spec_save_ && ratio > 1 && i < 8) {   // a verify pass: each position's value and score, for rewind
                ck(cudaMemcpyAsync(spec_v_[size_t(l)].f(size_t(i) * hd), v, hd * 4, cudaMemcpyDeviceToDevice, stream_), "spec stash");
                ck(cudaMemcpyAsync(spec_s_[size_t(l)].f(size_t(i) * hd), comp_s_.f(size_t(i) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "spec stash");
            }
            if (ratio > 1 && pos % ratio == 0) {   // the first position of a group: keep its value and score
                ck(cudaMemcpyAsync(stash_v_[size_t(l)].p, v, hd * 4, cudaMemcpyDeviceToDevice, stream_), "stash");
                ck(cudaMemcpyAsync(stash_s_[size_t(l)].p, comp_s_.f(size_t(i) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "stash");
                continue;
            }
            if (ratio > 1) {
                k::ds41_compress2(stash_v_[size_t(l)].f(), stash_s_[size_t(l)].f(), v, comp_s_.f(size_t(i) * hd), comp_.f(), c.head_dim, stream_);
                v = comp_.f();
            }
            const int g = pos / ratio, first = pos - (ratio - 1);   // a row stands for the first position of its group
            float* row = kv_half_ ? kv_rows_.f(size_t(i) * row_width()) : caches_[size_t(l)].f(size_t(g) * row_width());
            k::glm_rms_norm_rows(v, w(p + "attn_compressor_norm.weight"), row, c.head_dim, 1, c.rms_eps, stream_);
            if (T == 1 && observe) observe("comp_latent", l, row, hd);   // as dumped: normalized, before RoPE
            mat(p + "indexer.attn_k.weight", row, ik_.f());   // from the RoPE-free latent
            k::glm_rms_norm_rows(ik_.f(), w(p + "indexer.k_norm.weight"), row + c.head_dim, c.index_dim, 1, c.rms_eps, stream_);
            const float* cs = table(i, first == pos ? 2 : 4);
            k::ds41_rope(row + c.head_dim, 1, c.index_dim, c.index_dim - c.rope_dim, c.rope_dim / 2, cs, stream_);
            k::ds41_rope(row, 1, c.head_dim, c.head_dim - c.rope_dim, c.rope_dim / 2, cs, stream_);
            if (kv_kind_ == 1) k::glm_f16(row, reinterpret_cast<uint16_t*>(caches_[size_t(l)].p) + size_t(g) * row_width(), row_width(), stream_);
            if (kv_kind_ == 2) k::ds41_quant_rows_q8(row, 1, row_width(), caches_[size_t(l)].b(size_t(g) * kv_row_bytes()), kv_row_bytes(), stream_);
        }
    }

    // the host's selection for token i of layer l from its scores s (visible rows; masked in place for mode 2):
    // candidate blocks at the source layer (the newest row's block pinned), only those blocks later; ascending indices
    long topk_checked_ = 0, topk_differ_ = 0;
    std::vector<int> host_select(int l, float* s, int visible, int i) {
        const auto& c = cfg_;
        if (visible <= 0) return {};
        const int bs = c.candidate_block;
        if (c.candidate_source == l) {
            const int nb = (visible + bs - 1) / bs;
            std::vector<float> best(size_t(nb), -INFINITY);
            for (int t = 0; t < visible; ++t) best[size_t(t / bs)] = std::max(best[size_t(t / bs)], s[t]);
            best[size_t((visible - 1) / bs)] += 1e30f;
            candidates_[size_t(i)] = top_k(best.data(), nb, c.candidate_blocks);
        } else if (c.candidate_source >= 0 && c.candidate_source < l) {
            std::vector<char> keep(size_t((visible + bs - 1) / bs), 0);
            for (int b : candidates_[size_t(i)]) if (b < int(keep.size())) keep[size_t(b)] = 1;
            for (int t = 0; t < visible; ++t) if (!keep[size_t(t / bs)]) s[t] = -INFINITY;
        }
        auto picked = top_k(s, visible, c.index_top_k);
        // rows the mask removed are not attended to
        picked.erase(std::remove_if(picked.begin(), picked.end(), [&](int t) { return std::isinf(s[t]) && s[t] < 0; }), picked.end());
        std::sort(picked.begin(), picked.end());
        return picked;
    }

    // the lightning indexer of an index-source layer: each token's top-k over its visible compressed rows
    void index(int l) {
        const auto& c = cfg_;
        const int T = tokens_, ratio = c.ratio[size_t(l)];
        const std::string p = "blk." + std::to_string(l) + ".";
        mat(p + "indexer.attn_q_b.weight", qr_.f(), iq_.f(), T);
        k::ds41_rope(iq_.f(), T * c.index_heads, c.index_dim, c.index_dim - c.rope_dim, c.rope_dim / 2, table(0, 2), stream_,
                     c.index_heads, kTables * 64);
        if (fuse_) mat_xn(p + "indexer.proj.weight", iw_.f()); else mat(p + "indexer.proj.weight", xn_.f(), iw_.f(), T);
        const int src = c.kv_src[size_t(l)];
        const float w_scale = 1.0f / std::sqrt(float(c.index_dim * c.index_heads));
        // STRATA_DS41_GPU_TOPK (1): passes of 16+ tokens score and select on the GPU (no host round trip);
        // STRATA_DS41_TOPK_CHECK=1 also runs the host selection and counts differences
        static const bool gpu_topk = !(std::getenv("STRATA_DS41_GPU_TOPK") && std::string(std::getenv("STRATA_DS41_GPU_TOPK")) == "0");
        static const bool topk_check = std::getenv("STRATA_DS41_TOPK_CHECK") != nullptr;
        gpu_sel_ = false;
        // STRATA_DS41_GPU_TOPK_MIN (16): the fewest tokens selected on the GPU (1: decode too)
        static const int gpu_topk_min = std::getenv("STRATA_DS41_GPU_TOPK_MIN") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_GPU_TOPK_MIN"))) : 16;
        if ((gpu_topk && T >= gpu_topk_min) || T > score_tile_) {   // (a pass wider than the score tile: always here)
            const auto i0 = std::chrono::steady_clock::now();
            const int mode = c.candidate_source == l ? 1 : c.candidate_source >= 0 && c.candidate_source < l ? 2 : 0;
            // STRATA_DS41_INDEX_GEMM=1 (FP16 or Q8 caches): the scores as an FP16 tensor-core GEMM of the keys and the
            // token-heads' queries per tile of rows (products into a 256 MiB buffer), then relu and the head weights
            static const bool index_gemm = std::getenv("STRATA_DS41_INDEX_GEMM") && std::string(std::getenv("STRATA_DS41_INDEX_GEMM")) == "1";
            static const bool index_fast = std::getenv("STRATA_DS41_INDEX_FAST") && std::string(std::getenv("STRATA_DS41_INDEX_FAST")) == "1";
            // STRATA_DS41_INDEX_GEMM_MIN=n (FP16 / Q8 caches): tiles of at least n tokens (prompt chunks) score by the GEMM,
            // narrower ones (decode) as above - the GEMM's keys are read once per tile, the per-row kernels per token
            static const int gemm_min = std::getenv("STRATA_DS41_INDEX_GEMM_MIN") ? std::atoi(std::getenv("STRATA_DS41_INDEX_GEMM_MIN")) : 0;
            for (int t0 = 0; t0 < T; t0 += score_tile_) {   // score_tile_ tokens at a time through the score buffer
                const int n = std::min(score_tile_, T - t0);
                if (gemm_min > 0 && n >= gemm_min && kv_kind_ != 0) {
                    index_gemm_scores(src, t0, n, ratio, w_scale);
                } else if (index_fast &&
                    k::ds41_index_score_fast(kv_kind_, iq_.f(size_t(t0) * c.index_heads * c.index_dim), c.index_heads, c.index_dim, caches_[size_t(src)].p,
                                             kv_row_bytes(), c.head_dim, pos0_ + t0, ratio, n, (pos0_ + t0 + n) / ratio, 0,
                                             iw_.f(size_t(t0) * c.index_heads), w_scale, scores_.f(), score_stride_, stream_)) {
                } else if (index_gemm && kv_kind_ != 0) {
                    index_gemm_scores(src, t0, n, ratio, w_scale);
                } else if (kv_kind_ == 2)
                    k::ds41_index_score_q8(iq_.f(size_t(t0) * c.index_heads * c.index_dim), c.index_heads, c.index_dim, caches_[size_t(src)].b(),
                                           kv_row_bytes(), c.head_dim, pos0_ + t0, ratio, n, 0, iw_.f(size_t(t0) * c.index_heads), w_scale,
                                           scores_.f(), score_stride_, stream_);
                else
                    k::ds41_index_score_batched(iq_.f(size_t(t0) * c.index_heads * c.index_dim), c.index_heads, c.index_dim, cache_keys(src),
                                                row_width(), pos0_ + t0, ratio, n, iw_.f(size_t(t0) * c.index_heads), w_scale, scores_.f(),
                                                score_stride_, stream_, kv_kind_ == 1);
                k::ds41_index_topk(scores_.f(), score_stride_, n, pos0_ + t0, ratio, c.index_top_k,
                                   sel_[size_t(l)].i() + size_t(sel_base_ + t0) * c.index_top_k,
                                   sel_cnt_[size_t(l)].i() + sel_base_ + t0, mode, static_cast<uint8_t*>(cand_dev_.p) + size_t(t0) * cand_stride_, cand_stride_,
                                   c.candidate_block, c.candidate_blocks, stream_);
                // STRATA_DS41_TOPK_CHECK=1: the host's selection from these same GPU scores (so only the selection is
                // compared), tile by tile
                if (topk_check) {
                    const int wide = (pos0_ + t0 + n) / ratio;
                    std::vector<float> all(size_t(n) * std::max(wide, 1));
                    std::vector<int> gp(size_t(n) * c.index_top_k), gc(static_cast<size_t>(n));
                    if (wide > 0)
                        ck(cudaMemcpy2DAsync(all.data(), size_t(wide) * 4, scores_.p, size_t(score_stride_) * 4, size_t(wide) * 4, size_t(n),
                                             cudaMemcpyDeviceToHost, stream_), "check scores");
                    ck(cudaMemcpyAsync(gp.data(), sel_[size_t(l)].i() + size_t(sel_base_ + t0) * c.index_top_k, gp.size() * 4, cudaMemcpyDeviceToHost, stream_), "check sel");
                    ck(cudaMemcpyAsync(gc.data(), sel_cnt_[size_t(l)].i() + sel_base_ + t0, gc.size() * 4, cudaMemcpyDeviceToHost, stream_), "check count");
                    ck(cudaStreamSynchronize(stream_), "check sync");
                    int bad = 0;
                    for (int i = 0; i < n; ++i) {
                        const int visible = (pos0_ + t0 + i + 1) / ratio;
                        const auto picked = host_select(l, all.data() + size_t(i) * std::max(wide, 1), visible, t0 + i);
                        bool same = gc[size_t(i)] == int(picked.size());
                        for (size_t j = 0; same && j < picked.size(); ++j) same = gp[size_t(i) * c.index_top_k + j] == picked[j];
                        bad += !same;
                    }
                    topk_checked_ += n;
                    topk_differ_ += bad;
                    std::cerr << "DS41 topk_check layer=" << l << " pos=" << pos0_ + t0 << " tokens=" << n << " differ=" << bad << "\n";
                }
            }
            prof_[5] += std::chrono::duration<double>(std::chrono::steady_clock::now() - i0).count();
            gpu_sel_ = true;
            return;
        }
        int widest = 0;
        for (int i = 0; i < T; ++i) {
            const int visible = (pos0_ + i + 1) / ratio;
            widest = std::max(widest, visible);
            if (kv_kind_ == 2)
                k::ds41_index_score_q8(iq_.f(size_t(i) * c.index_heads * c.index_dim), c.index_heads, c.index_dim, caches_[size_t(src)].b(),
                                       kv_row_bytes(), c.head_dim, 0, 1, 1, visible, iw_.f(size_t(i) * c.index_heads), w_scale,
                                       scores_.f(size_t(i) * score_stride_), score_stride_, stream_);
            else
                k::ds41_index_score(iq_.f(size_t(i) * c.index_heads * c.index_dim), c.index_heads, c.index_dim, cache_keys(src),
                                    row_width(), visible, iw_.f(size_t(i) * c.index_heads), w_scale, scores_.f(size_t(i) * score_stride_), stream_,
                                    kv_kind_ == 1);
        }
        const auto i0 = std::chrono::steady_clock::now();
        struct Done { double& t; std::chrono::steady_clock::time_point s; ~Done() { t += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count(); } } done{prof_[5], i0};
        if (widest == 0) { std::fill(sel_count_[size_t(l)].begin(), sel_count_[size_t(l)].end(), 0); return; }
        std::vector<float> all(size_t(T) * widest);
        ck(cudaMemcpy2DAsync(all.data(), size_t(widest) * 4, scores_.p, size_t(score_stride_) * 4, size_t(widest) * 4, size_t(T),
                             cudaMemcpyDeviceToHost, stream_), "scores");
        ck(cudaStreamSynchronize(stream_), "scores sync");
        std::vector<int> picks(size_t(T) * c.index_top_k, 0);
        const int bs = c.candidate_block;
        (void)bs;
        for (int i = 0; i < T; ++i) {
            const int visible = (pos0_ + i + 1) / ratio;
            if (visible == 0) { sel_count_[size_t(l)][size_t(sel_base_ + i)] = 0; continue; }
            const auto picked = host_select(l, all.data() + size_t(i) * widest, visible, i);
            sel_count_[size_t(l)][size_t(sel_base_ + i)] = int(picked.size());
            std::copy(picked.begin(), picked.end(), picks.begin() + size_t(i) * c.index_top_k);
        }
        ck(cudaMemcpyAsync(sel_[size_t(l)].i() + size_t(sel_base_) * c.index_top_k, picks.data(), picks.size() * 4,
                           cudaMemcpyHostToDevice, stream_), "sel");
        ck(cudaStreamSynchronize(stream_), "sel sync");   // picks leaves scope
    }

    // The GPU expert tier: slots in one VRAM arena. STRATA_DS41_EXPERT_PROFILE=routes.i32 (STRATA_DS41_ROUTES of an
    // earlier run) fills it with the most-used experts at load; without one it starts empty. Then it adapts: every
    // expert's score decays by STRATA_DS41_TIER_DECAY (0.995) per token and gains 1 per use, and after each pass up to
    // STRATA_DS41_TIER_PROMOTE (2) missed experts replace lower-scored residents of the same size (0: static tier).
    // Size: STRATA_DS41_GPU_EXPERT_MIB, default free VRAM less STRATA_DS41_GPU_RESERVE_MIB (1536).
    struct Slot { uint8_t* p; size_t bytes; int owner; bool pending; cudaEvent_t done; };
    // a tier slot: this node's rows of one expert, packed [gate fl_ rows | up fl_ rows | down hidden rows of fl_ columns]
    size_t slice_gate(int l) const { return experts_[size_t(l)].f.gu_row * size_t(fl_); }
    size_t slice_down_row(int l) const { return ggml_row_size(ggml_type(experts_[size_t(l)].f.d_type), fl_); }
    size_t expert_bytes(int l) const { return 2 * slice_gate(l) + slice_down_row(l) * size_t(cfg_.hidden); }
    // copies expert e's slice of layer l to dst (host)
    void slice_to_host(int l, int e, uint8_t* dst) const {   // (RAM experts only)
        const auto& L = experts_[size_t(l)];
        const size_t g = slice_gate(l), dr = slice_down_row(l);
        if (!ram_pos_.empty()) e = ram_pos_[size_t(l)][size_t(e)];
        std::memcpy(dst, L.gate + size_t(e) * L.gate_stride, g);
        std::memcpy(dst + g, L.up + size_t(e) * L.gate_stride, g);
        const uint8_t* down = L.down + size_t(e) * L.down_stride;
        if (dr == L.f.d_row) std::memcpy(dst + 2 * g, down, dr * size_t(cfg_.hidden));
        else for (int r = 0; r < cfg_.hidden; ++r) std::memcpy(dst + 2 * g + size_t(r) * dr, down + size_t(r) * L.f.d_row, dr);
    }
    void load_tier() {
        const auto& c = cfg_;
        slot_.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
        if (score_.size() != size_t(c.layers) * c.experts) score_.assign(size_t(c.layers) * c.experts, 0.0f);
        if (std::getenv("STRATA_DS41_NO_TIER")) return;
        auto env = [](const char* name, double fallback) { const char* v = std::getenv(name); return v ? std::stod(v) : fallback; };
        promote_ = int(env("STRATA_DS41_TIER_PROMOTE", 2));
        decay_ = float(env("STRATA_DS41_TIER_DECAY", 0.995));
        gx_q8_ = Dev(k::native_q8_1_bytes(c.hidden, 1));
        g_gate_ = Dev(size_t(c.ff) * 4);
        g_up_ = Dev(size_t(c.ff) * 4);
        g_h_ = Dev(size_t(c.ff) * 4);
        g_h_q8_ = Dev(k::native_q8_1_bytes(c.ff, 1));
        g_out_ = Dev(size_t(c.hidden) * 4);
        std::vector<double> count(size_t(c.layers) * c.experts, 0);
        if (!std::getenv("STRATA_DS41_EXPERT_PROFILE"))   // after a prompt: its own routing (the decayed scores)
            for (size_t i = 0; i < count.size(); ++i) count[i] = score_[i];
        if (const char* profile = std::getenv("STRATA_DS41_EXPERT_PROFILE")) {
            std::ifstream g(profile, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the expert profile " + std::string(profile));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int k = 0; k < c.top_k; ++k) {
                        const int e = v[size_t(l) * c.top_k + k];
                        if (e >= 0 && e < c.experts) count[size_t(l) * c.experts + e] += 1;
                    }
            // after a streamed prompt (the tier lent, no promotions meanwhile): the prompt's own routing (score_) joins
            // the profile's, both as shares, the prompt's weighted by STRATA_DS41_TIER_PROMPT_WEIGHT (10) - what the
            // promotions of a chunked prompt would have done
            double ps = 0, ss = 0;
            for (size_t i = 0; i < count.size(); ++i) { ps += count[i]; ss += score_[i]; }
            const double wp = env("STRATA_DS41_TIER_PROMPT_WEIGHT", 10.0);
            if (ps > 0 && ss > 0 && wp > 0)
                for (size_t i = 0; i < count.size(); ++i) count[i] = count[i] / ps + wp * double(score_[i]) / ss;
        }
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "memory info");
        // STRATA_DS41_DESKTOP_MIB (3072): VRAM left to the desktop (compositor, browser) and the decoder's scratch
        const size_t reserve = std::max<size_t>(size_t(env("STRATA_DS41_GPU_RESERVE_MIB", env("STRATA_DS41_DESKTOP_MIB", 3072))) << 20, vram_reserve + (128 << 20));
        size_t budget = free_b > reserve ? free_b - reserve : 0;
#if defined(__HIP_PLATFORM_AMD__)
        // cudaMemGetInfo can overstate usable memory on a desktop AMD card.
        // Apply the physical floor to the expert arena too, including ordinary
        // prefill: unlike Dev allocations, this arena is allocated directly.
        const size_t physical_free = physical_vram_free();
        const size_t keep = physical_vram_keep();
        const size_t required_keep = std::max(reserve, keep);
        budget = std::min(budget, physical_free > required_keep ? physical_free - required_keep : size_t(0));
#endif
        if (std::getenv("STRATA_DS41_GPU_EXPERT_MIB")) budget = std::min(budget, size_t(env("STRATA_DS41_GPU_EXPERT_MIB", 0)) << 20);
        // one arena: many small allocations fragment and overshoot the reported free memory
        while (budget > (size_t(256) << 20)) {
#if defined(__HIP_PLATFORM_AMD__)
            const size_t physical_free_now = physical_vram_free();
            budget = std::min(budget, physical_free_now > required_keep ? physical_free_now - required_keep : size_t(0));
            if (budget <= (size_t(256) << 20)) break;
#endif
            void* p = nullptr;
            if (cudaMalloc(&p, budget) == cudaSuccess) { arena_.p = p; arena_.bytes = budget; break; }
            (void)cudaGetLastError();
            budget -= size_t(256) << 20;
        }
        if (!arena_.p) return;
        std::vector<int> order(count.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return count[size_t(a)] > count[size_t(b)]; });
        size_t used = 0;
        double hits = 0, all = 0;
        for (double x : count) all += x;
        for (int i : order) {   // the profile's experts, most used first
            if (count[size_t(i)] <= 0) break;
            const int l = i / c.experts, e = i % c.experts;
            const size_t b = expert_bytes(l);
            if (used + b > budget) continue;
            uint8_t* r = static_cast<uint8_t*>(arena_.p) + used;
            const auto& L = experts_[size_t(l)];
            const size_t g = slice_gate(l), dr = slice_down_row(l);
            ck(cudaMemcpy(r, L.gate + size_t(e) * L.gate_stride, g, cudaMemcpyHostToDevice), "tier gate");
            ck(cudaMemcpy(r + g, L.up + size_t(e) * L.gate_stride, g, cudaMemcpyHostToDevice), "tier up");
            ck(cudaMemcpy2DAsync(r + 2 * g, dr, L.down + size_t(e) * L.down_stride, L.f.d_row, dr, size_t(c.hidden), cudaMemcpyHostToDevice, stream_), "tier down");
            ck(cudaStreamSynchronize(stream_), "tier down");
            slot_[size_t(l)][size_t(e)] = int(slots_.size());
            slots_.push_back({r, b, i, false, nullptr});
            score_[size_t(i)] = float(count[size_t(i)] / std::max(1.0, all / double(c.layers * c.top_k)));
            used += b;
            hits += count[size_t(i)];
        }
        // the rest as empty slots, sized like the layers' experts in proportion
        if (promote_ > 0) {
            for (int l = 0; budget - used >= expert_bytes(l); l = (l + 1) % c.layers) {
                slots_.push_back({static_cast<uint8_t*>(arena_.p) + used, expert_bytes(l), -1, false, nullptr});
                used += expert_bytes(l);
            }
            size_t largest = 0;
            for (int l = 0; l < c.layers; ++l) largest = std::max(largest, expert_bytes(l));
            for (int i = 0; i < promote_; ++i) {
                void* h = nullptr;
                ck(cudaHostAlloc(&h, largest, cudaHostAllocDefault), "tier staging");
                staging_.push_back(static_cast<uint8_t*>(h));
            }
            copy_ = make_stream(cudaStreamNonBlocking);
            for (auto& sl : slots_) ck(cudaEventCreateWithFlags(&sl.done, cudaEventDisableTiming), "tier event");
            recorded_ = std::make_unique<std::atomic<bool>[]>(slots_.size());
            for (size_t i = 0; i < slots_.size(); ++i) recorded_[i].store(true);
            uploader_ = std::thread([this] { unpin_helper(); upload_loop(); });
        }
        if (std::getenv("STRATA_DS41_TIER_DEBUG")) {
            int res = 0;
            for (const auto& r : slot_) for (int x : r) res += x >= 0;
            std::cerr << "DS41 tier_debug resident=" << res << " tier_lent=" << tier_lent_ << " producer=" << producer_.joinable() << "\n";
        }
        std::cerr << "DS41 gpu_tier slots=" << slots_.size() << " MiB=" << used / double(1 << 20) << " profile_hit="
                  << (all > 0 ? hits / all : 0) << " promote=" << promote_ << " decay=" << decay_ << "\n";
    }

    // the promotions' copies, off the decode thread: rows to pinned staging, then an async upload on copy_
    struct Job { int slot, layer, expert; };
    void upload_loop() {
        const auto& c = cfg_;
        size_t next = 0;
        std::vector<cudaEvent_t> staged(staging_.size(), nullptr);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lock(jobs_mutex_);
                jobs_cv_.wait(lock, [&] { return stop_uploads_ || !jobs_.empty(); });
                if (jobs_.empty()) return;
                j = jobs_.front();
                jobs_.pop_front();
            }
            const size_t k = next++ % staging_.size();
            if (staged[k]) cudaEventSynchronize(staged[k]);   // the staging buffer's previous upload
            Slot& sl = slots_[size_t(j.slot)];
            uint8_t* st = staging_[k];
            slice_to_host(j.layer, j.expert, st);
            cudaMemcpyAsync(sl.p, st, sl.bytes, cudaMemcpyHostToDevice, copy_);
            cudaEventRecord(sl.done, copy_);
            staged[k] = sl.done;
            recorded_[size_t(j.slot)].store(true);
        }
    }

    // after a pass: scores decay (once per token), finished uploads join the tier, and up to promote_ of the missed
    // experts replace lower-scored residents of the same size
    void adapt(int tokens) {
        const auto& c = cfg_;
        const float decay = std::pow(decay_, float(tokens));
        for (auto& v : score_) v *= decay;
        for (int i : used_) score_[size_t(i)] += 1.0f;
        if (promote_ <= 0 || slots_.empty()) { used_.clear(); missed_.clear(); return; }
        for (size_t s = 0; s < slots_.size(); ++s) {
            Slot& sl = slots_[s];
            if (sl.pending && recorded_[s].load() && cudaEventQuery(sl.done) == cudaSuccess) {
                sl.pending = false;
                slot_[size_t(sl.owner / c.experts)][size_t(sl.owner % c.experts)] = int(s);
                residency_changed();
            }
        }
        std::sort(missed_.begin(), missed_.end());
        missed_.erase(std::unique(missed_.begin(), missed_.end()), missed_.end());
        std::stable_sort(missed_.begin(), missed_.end(), [&](int a, int b) { return score_[size_t(a)] > score_[size_t(b)]; });
        int done = 0;
        for (int cand : missed_) {
            if (done >= promote_ * tokens) break;   // a pass of several tokens promotes as many tokens would
            const int l = cand / c.experts, e = cand % c.experts;
            if (slot_[size_t(l)][size_t(e)] >= 0) continue;
            bool in_flight = false;
            for (const auto& sl : slots_) in_flight |= sl.pending && sl.owner == cand;
            if (in_flight) continue;
            const size_t b = expert_bytes(l);
            int victim = -1;
            float low = score_[size_t(cand)];
            for (size_t s = 0; s < slots_.size(); ++s) {
                const Slot& sl = slots_[s];
                if (sl.bytes != b || sl.pending) continue;
                const float v = sl.owner < 0 ? -1.0f : score_[size_t(sl.owner)];
                if (v < low) { low = v; victim = int(s); }
            }
            if (victim < 0) continue;
            Slot& sl = slots_[size_t(victim)];
            if (sl.owner >= 0) slot_[size_t(sl.owner / c.experts)][size_t(sl.owner % c.experts)] = -1;
            sl.owner = cand;
            sl.pending = true;
            recorded_[size_t(victim)].store(false);
            {
                std::lock_guard<std::mutex> lock(jobs_mutex_);
                jobs_.push_back({victim, l, e});
            }
            jobs_cv_.notify_one();
            ++done;
            ++promotions_;
        }
        used_.clear();
        missed_.clear();
    }

    // token i's resident experts of `ids` on the GPU: g_sum_[i] = sum w_k E_k(xn_[i]) (queued on the stream)
    // one expert at base (the tier layout) for token i, weight w, into g_sum_
    void expert_chain(int l, const uint8_t* base, int i, float w) {
        const auto& c = cfg_;
        const auto& L = experts_[size_t(l)];
        const size_t g = slice_gate(l);
        const int F = fl_;
        k::native_quantize_q8_1(xn_.f(size_t(i) * c.hidden), gx_q8_.p, c.hidden, 1, stream_);
        k::native_mmvq(L.f.gu_type, base, gx_q8_.p, g_gate_.f(), c.hidden, F, 1, stream_);
        k::native_mmvq(L.f.gu_type, base + g, gx_q8_.p, g_up_.f(), c.hidden, F, 1, stream_);
        k::glm_swiglu(g_gate_.f(), g_up_.f(), g_h_.f(), F, c.clamp[size_t(l)], stream_);
        k::native_quantize_q8_1(g_h_.f(), g_h_q8_.p, F, 1, stream_);
        k::native_mmvq(L.f.d_type, base + 2 * g, g_h_q8_.p, g_out_.f(), F, c.hidden, 1, stream_);
        k::ds41_axpy(g_out_.f(), w, g_sum_.f(size_t(i) * c.hidden), c.hidden, stream_);
    }
    // ---- STRATA_DS41_REMOTE_OWN=ADDR:PORT (opt-in, STD2 in include/strata/net/expert_tp.hpp): a worker
    // (strata-ds41-worker) owns whole experts: the most-used experts of the profile (STRATA_DS41_CANON_PROFILE, or
    // STRATA_DS41_REMOTE_OWN_PROFILE) after the canonical tier's, up to STRATA_DS41_REMOTE_OWN_MIB (8192) MiB. This node
    // sends them at setup (pread, never through its mapping) and drops them from its page cache; per layer it sends the
    // pass's owned routes with the activations right after the router, and adds their raw outputs in the lane's route
    // order with its own (finish_with_external), so the sums are the single node's.
    void connect_own(const char* addr) {
        const auto& c = cfg_;
        namespace net = strata::net;
        if (!remote_addr_.empty() || peer_ || ram_copy_ || stage_cpu_expert_ || fl_ != c.ff || !ram_pos_.empty() || !packed_.empty() ||
            (chunk_ >= stream_min_ && !(std::getenv("STRATA_DS41_OWN_STREAM") && *std::getenv("STRATA_DS41_OWN_STREAM"))) || c.top_k < 1 || c.top_k > 16)
            throw std::invalid_argument("ds41: STRATA_DS41_REMOTE_OWN needs plain CPU routing with lane prompt passes (no remote TP, peer, "
                                        "RAM copy, staging, packed copy or streamed prompt chunks)");
        const char* profile = std::getenv("STRATA_DS41_REMOTE_OWN_PROFILE");
        if (!profile || !*profile) profile = std::getenv("STRATA_DS41_CANON_PROFILE");
        if (!profile || !*profile) throw std::invalid_argument("ds41: STRATA_DS41_REMOTE_OWN needs a route profile (STRATA_DS41_REMOTE_OWN_PROFILE)");
        const double mib = strata_env_value_double("STRATA_DS41_REMOTE_OWN_MIB", 8192.0);
        if (!(mib >= 1 && mib <= 262144)) throw std::invalid_argument("ds41: STRATA_DS41_REMOTE_OWN_MIB must be 1 through 262144");
        // the owned set: hits per byte, after the canonical tier's experts
        std::vector<double> count(size_t(c.layers) * c.experts, 0);
        {
            std::ifstream g(profile, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the ownership profile " + std::string(profile));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int k = 0; k < c.top_k; ++k) {
                        const int e = v[size_t(l) * c.top_k + k];
                        if (e >= 0 && e < c.experts) count[size_t(l) * c.experts + e] += 1;
                    }
        }
        // STRATA_DS41_REMOTE_OWN_SKIP_MIB=S: the band starts after the S hottest MiB of the profile (the same set whatever
        // the canonical tier's final size, so the worker's kept rows stay valid); unset: after the canonical tier
        const double skip_mib = strata_env_value_double("STRATA_DS41_REMOTE_OWN_SKIP_MIB", -1.0);
        std::vector<std::tuple<double, int, int>> order;
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (skip_mib >= 0 || canon_off_.empty() || canon_off_[size_t(l) * c.experts + e] < 0)
                    order.emplace_back(count[size_t(l) * c.experts + e] / double(expert_bytes(l)), l, e);
        std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        if (skip_mib >= 0) {
            double skipped = 0;
            size_t k = 0;
            while (k < order.size() && skipped + double(expert_bytes(std::get<1>(order[k]))) <= skip_mib * double(1 << 20))
                skipped += double(expert_bytes(std::get<1>(order[k++])));
            order.erase(order.begin(), order.begin() + std::ptrdiff_t(k));
        }
        own_.assign(size_t(c.layers) * c.experts, 0);
        size_t used = 0, picked = 0;
        // STRATA_DS41_REMOTE_OWN_LOAD=f (0 < f < 1): walking the experts hottest first, the worker takes one only while
        // its share of the profile's expected bytes (count x expert bytes) stays at most f, so it also holds colder
        // experts (RAM) instead of only the hottest band (load); unset: the hottest band up to the MiB cap
        const double load_f = strata_env_value_double("STRATA_DS41_REMOTE_OWN_LOAD", 0.0);
        if (!(load_f >= 0 && load_f < 1)) throw std::invalid_argument("ds41: STRATA_DS41_REMOTE_OWN_LOAD must be in [0, 1)");
        double load_all = 0, load_own = 0;
        for (const auto& [score, l, e] : order) {
            const double load = count[size_t(l) * c.experts + e] * double(expert_bytes(l));
            load_all += load;
            if (double(used + expert_bytes(l)) > mib * double(1 << 20)) continue;
            if (load_f > 0 && load_own + load > load_f * load_all) continue;
            own_[size_t(l) * c.experts + e] = 1;
            used += expert_bytes(l);
            load_own += load;
            ++picked;
            (void)score;
        }
        std::cerr << "DS41 remote_own load_share=" << (load_all > 0 ? load_own / load_all : 0.0) << " (of the non-canonical profile bytes)\n";
        auto r = std::make_unique<Remote>();
        r->hidden = c.hidden;
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 0;
        cfg.world = 2;
        cfg.root_addr = addr;
        if (!std::getenv("UCOMM_TIMEOUT_MS")) cfg.timeout_ms = 0;
        std::cerr << "DS41 remote_own waiting for the worker at " << addr << " (" << picked << " experts, "
                  << double(used) / double(1ull << 30) << " GiB)\n";
        Remote::ok(ucomm_init(&cfg, &r->comm), "connect");
        const int most = std::max(chunk_, 64);
        r->request.resize(sizeof(net::TpRequest) + size_t(most) * c.top_k * sizeof(net::TpJob) + size_t(most) * c.hidden * 4);
        r->reply.resize(size_t(most) * c.top_k * c.hidden);
        for (const auto& [ptr, bytes] : {std::pair<void*, size_t>{r->request.data(), r->request.size()}, {r->reply.data(), r->reply.size() * 4}}) {
            ucomm_mr_t* region = nullptr;
            if (ucomm_mr_reg(r->comm, ptr, bytes, &region) == UCOMM_OK) r->regions.push_back(region);
        }
        ext_vals_.resize(size_t(most) * c.top_k * c.hidden);
        // setup: TpSetup, a TpLayer per layer (experts = owned count, pad = expert count), the owned ids
        std::vector<uint8_t> setup(sizeof(net::TpSetup));
        std::vector<int32_t> ids;
        uint64_t h = 14695981039346656037ull;
        auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull; };
        for (int l = 0; l < c.layers; ++l) {
            const auto& F = experts_[size_t(l)].f;
            int owned = 0;
            for (int e = 0; e < c.experts; ++e) if (own_[size_t(l) * c.experts + e]) { ids.push_back(e); ++owned; }
            const net::TpLayer spec{l, int32_t(F.gu_type), int32_t(F.d_type), c.ff, 0, owned, c.clamp[size_t(l)], c.experts};
            const auto* b = reinterpret_cast<const uint8_t*>(&spec);
            setup.insert(setup.end(), b, b + sizeof spec);
            for (const char* part : {"gate", "up", "down"}) {
                const auto& t = art_.at("blk." + std::to_string(l) + ".ffn_" + part + "_exps.weight");
                const int64_t geometry[3] = {l, int64_t(t.tensor->type), int64_t(t.bytes)};
                mix(geometry, sizeof geometry);
                mix(t.file->path().c_str(), t.file->path().size());
                const int64_t where[2] = {int64_t(t.file->file_size()), int64_t(t.tensor->offset)};
                mix(where, sizeof where);
            }
        }
        mix(ids.data(), ids.size() * 4);
        const auto* ib = reinterpret_cast<const uint8_t*>(ids.data());
        setup.insert(setup.end(), ib, ib + ids.size() * 4);
        // the owned experts' profile counts (float, the ids' order): the worker fills its GPU tier from them at setup
        std::vector<float> counts;
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (own_[size_t(l) * c.experts + e]) counts.push_back(float(count[size_t(l) * c.experts + e]));
        const auto* cb = reinterpret_cast<const uint8_t*>(counts.data());
        setup.insert(setup.end(), cb, cb + counts.size() * 4);
        const net::TpSetup head{net::kTpMagicDs41Own, uint32_t(c.layers), uint32_t(c.hidden), net::kReplyRaw, net::kWeightsStreamed, 1, h ? h : 1};
        std::memcpy(setup.data(), &head, sizeof head);
        if (setup.size() > (size_t(1) << 20)) throw std::runtime_error("ds41 remote_own: the setup exceeds the worker's 1 MiB message");
        Remote::ok(ucomm_send(r->comm, 1, net::kTagSetup, setup.data(), setup.size()), "setup");
        net::TpAccept accept{};
        Remote::ok(ucomm_recv(r->comm, 1, net::kTagAccept, &accept, sizeof accept, nullptr), "accept");
        if (!accept.ok)
            throw std::runtime_error("ds41 remote_own: the worker refused: " + std::string(accept.reason, strnlen(accept.reason, sizeof accept.reason)));
        const auto t0 = std::chrono::steady_clock::now();
        double sent = 0;
        std::map<std::string, int> fds;
        auto fd_of = [&](const core::ArtifactTensor& t) {
            auto it = fds.find(t.file->path());
            if (it == fds.end()) {
                const int fd = ::open(t.file->path().c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("ds41 remote_own: cannot open " + t.file->path());
                it = fds.emplace(t.file->path(), fd).first;
            }
            return it->second;
        };
        auto read_at = [](int fd, uint8_t* dst, size_t n, uint64_t off) {
            for (size_t got = 0; got < n;) {
                const ssize_t k = pread(fd, dst + got, n - got, off_t(off + got));
                if (k <= 0) throw std::runtime_error("ds41 remote_own: read failed");
                got += size_t(k);
            }
        };
        {
            size_t largest = 0;
            for (int l = 0; l < c.layers; ++l) largest = std::max(largest, expert_bytes(l));
            std::array<std::vector<uint8_t>, 2> staging;
            std::array<ucomm_req_t*, 2> inflight{};
            std::array<ucomm_mr_t*, 2> registered{};
            for (int k = 0; k < 2; ++k) {
                staging[size_t(k)].resize(largest);
                if (ucomm_mr_reg(r->comm, staging[size_t(k)].data(), largest, &registered[size_t(k)]) != UCOMM_OK) registered[size_t(k)] = nullptr;
            }
            int which = 0;
            for (int l = 0; l < c.layers; ++l) {
                const std::string p = "blk." + std::to_string(l) + ".";
                const auto &G = art_.at(p + "ffn_gate_exps.weight"), &U = art_.at(p + "ffn_up_exps.weight"), &D = art_.at(p + "ffn_down_exps.weight");
                const auto& F = experts_[size_t(l)].f;
                const uint64_t go = G.file->data_start() + G.tensor->offset, uo = U.file->data_start() + U.tensor->offset;
                const uint64_t dof = D.file->data_start() + D.tensor->offset;
                const size_t gstride = F.gu_row * size_t(c.ff), dstride = F.d_row * size_t(c.hidden), bytes = 2 * gstride + dstride;
                if (bytes != expert_bytes(l)) throw std::runtime_error("ds41 remote_own: expert blob geometry");
                for (int e = 0; e < c.experts; ++e) {
                    if (!own_[size_t(l) * c.experts + e]) continue;
                    if (!accept.have) {
                        if (inflight[size_t(which)]) Remote::ok(ucomm_wait(inflight[size_t(which)], nullptr), "weights");
                        inflight[size_t(which)] = nullptr;
                        uint8_t* blob = staging[size_t(which)].data();
                        read_at(fd_of(G), blob, gstride, go + size_t(e) * gstride);
                        read_at(fd_of(U), blob + gstride, gstride, uo + size_t(e) * gstride);
                        read_at(fd_of(D), blob + 2 * gstride, dstride, dof + size_t(e) * dstride);
                        Remote::ok(ucomm_isend(r->comm, 1, net::kTagWeights, blob, bytes, &inflight[size_t(which)]), "weights");
                        which ^= 1;
                        sent += double(bytes);
                    }
                    // the owned expert leaves this node's page cache (this node never reads it again)
                    posix_fadvise(fd_of(G), off_t(go + size_t(e) * gstride), off_t(gstride), POSIX_FADV_DONTNEED);
                    posix_fadvise(fd_of(U), off_t(uo + size_t(e) * gstride), off_t(gstride), POSIX_FADV_DONTNEED);
                    posix_fadvise(fd_of(D), off_t(dof + size_t(e) * dstride), off_t(dstride), POSIX_FADV_DONTNEED);
                }
            }
            for (auto* q : inflight) if (q) Remote::ok(ucomm_wait(q, nullptr), "weights");
            for (auto* m : registered) if (m) ucomm_mr_dereg(m);
        }
        for (const auto& [path, fd] : fds) ::close(fd);
        net::TpReady ready{};
        Remote::ok(ucomm_recv(r->comm, 1, net::kTagReady, &ready, sizeof ready, nullptr), "ready");
        std::cerr << "DS41 remote_own backend=" << ucomm_backend_name(r->comm) << " experts=" << picked << " GiB="
                  << double(used) / double(1ull << 30) << " weights=" << (accept.have ? "kept" : "sent") << " sent_GiB="
                  << sent / double(1ull << 30) << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " worker_gpu_slots=" << ready.slots << "\n";
        own_remote_ = std::move(r);
        if (const char* v = std::getenv("STRATA_DS41_OWN_PROMOTE"); v && std::atoi(v) > 0) {
            const char* addr = std::getenv("STRATA_DS41_OWN_FETCH");
            if (!addr || !*addr) throw std::invalid_argument("ds41: STRATA_DS41_OWN_PROMOTE needs STRATA_DS41_OWN_FETCH=HOST:PORT");
            promote_start(std::atoi(v), addr);
        }
        if (const char* v = std::getenv("STRATA_DS41_CANON_ADAPT"); v && std::atoi(v) > 0) adapt_start(std::atoi(v), std::getenv("STRATA_DS41_OWN_FETCH"));
        // STRATA_DS41_OWN_STREAM=HOST:PORT (the worker's --stream-port): a streamed prompt pass receives the worker-owned experts of each item over this link,
        // straight into the pinned host slots (the kept ones are copied from the local shards as before)
        if (const char* v = std::getenv("STRATA_DS41_OWN_STREAM"); v && *v) {
            ucomm_config scfg;
            ucomm_config_default(&scfg);
            scfg.rank = 0;
            scfg.world = 2;
            scfg.root_addr = v;
            if (!std::getenv("UCOMM_TIMEOUT_MS")) scfg.timeout_ms = 0;
            Remote::ok(ucomm_init(&scfg, &ost_comm_), "stream connect");
            std::cerr << "DS41 own_stream connected " << v << "\n";
        }
    }
    void canon_enqueue(int l, int T, int K, const std::vector<int32_t>& entries, const std::vector<int>& gstart,
                       const std::vector<std::pair<int, int>>& groups) {
        const auto& c = cfg_;
        const auto& f = experts_[size_t(l)].f;
        const int n = int(entries.size()), G = int(gstart.size()) - 1, E = canon_tmax_ * c.top_k;
        const size_t g = slice_gate(l);
        const size_t xq_bytes = size_t(c.hidden / 256) * 292, hq_bytes = size_t(c.ff / 256) * 292;
        if (T < 1 || T > canon_tmax_ || n < 1 || n > E || G < 1 || G > n || gstart.front() != 0 || gstart.back() != n || groups.size() != size_t(G) ||
            canon_xq_.bytes < size_t(T) * xq_bytes || canon_hq_.bytes < size_t(n) * hq_bytes ||
            canon_gu_.bytes < size_t(2) * n * c.ff * sizeof(float) || canon_out_.bytes < size_t(n) * c.hidden * sizeof(float))
            throw std::runtime_error("ds41: canonical tier pass geometry out of bounds");
        // meta: [gu ptrs G][down ptrs G][gstart G+1][eact n][identity n]
        uint64_t* gp = reinterpret_cast<uint64_t*>(canon_meta_host_);
        uint64_t* dp = gp + G;
        int* gs = reinterpret_cast<int*>(dp + G);
        int* ea = gs + G + 1;
        int* id = ea + n;
        const size_t meta_bytes = size_t(2) * G * sizeof(uint64_t) + size_t(G + 1 + 2 * n) * sizeof(int);
        if (meta_bytes > canon_meta_.bytes) throw std::runtime_error("ds41: canonical tier metadata exceeds its buffer");
        const size_t eb = expert_bytes(l);
        std::vector<std::pair<uint8_t*, int>> copies;   // streamed experts: (ring slot, expert)
        int Gh = G;                                      // groups [0, Gh) are held, [Gh, G) streamed (in that order)
        for (int j = 0; j < G; ++j) if (groups[size_t(j)].second >= 0) { Gh = j; break; }
        for (int j = Gh; j < G; ++j) if (groups[size_t(j)].second < 0) throw std::runtime_error("ds41: canonical tier group order");
        for (int j = 0; j < G; ++j) {
            const int i0 = entries[size_t(gstart[size_t(j)])];
            const int e = int(canon_original_expert(l, i0)), slot = groups[size_t(j)].second;
            if (e != groups[size_t(j)].first) throw std::runtime_error("ds41: canonical tier group order mismatch");
            if (slot < 0) {
                const int64_t off = canon_off_[size_t(l) * c.experts + e];
                if (off < 0 || size_t(off) + eb > canon_arena_.bytes) throw std::runtime_error("ds41: canonical tier slot out of bounds");
                gp[j] = uint64_t(reinterpret_cast<uintptr_t>(canon_arena_.b(size_t(off))));
            } else {
                // a streamed expert: gate | up | down straight from the mapped tensors into its ring slot
                if (slot >= canon_stream_ || size_t(slot + 1) * canon_ring_slot_ > canon_ring_.bytes || eb > canon_ring_slot_)
                    throw std::runtime_error("ds41: canonical stream slot out of bounds");
                uint8_t* dst = canon_ring_.b(size_t(slot) * canon_ring_slot_);
                const auto& LE = experts_[size_t(l)];
                if (slice_down_row(l) != LE.f.d_row) throw std::runtime_error("ds41: canonical stream needs whole down rows");
                copies.push_back({dst, e});
                gp[j] = uint64_t(reinterpret_cast<uintptr_t>(dst));
            }
            dp[j] = gp[j] + 2 * g;
        }
        for (int j = 0; j <= G; ++j) gs[j] = gstart[size_t(j)];
        for (int j = 0; j < n; ++j) {
            if (entries[size_t(j)] < 0 || entries[size_t(j)] >= T * K) throw std::runtime_error("ds41: canonical tier entry out of bounds");
            ea[j] = entries[size_t(j)] / K;
            id[j] = j;
        }
        uint8_t* md = canon_meta_.b();
        ck(cudaMemcpyAsync(md, canon_meta_host_, meta_bytes, cudaMemcpyHostToDevice, stream_), "canonical tier metadata");
        const auto* d_gp = reinterpret_cast<const uint8_t* const*>(md);
        const auto* d_dp = d_gp + G;
        const int* d_gs = reinterpret_cast<const int*>(md + size_t(2) * G * sizeof(uint64_t));
        const int* d_ea = d_gs + G + 1;
        const int* d_id = d_ea + n;
        // the streamed experts' copies on their own stream (the ring slots' last readers, the previous layer's
        // kernels, finished at that layer's results sync), overlapping the held experts' chain below
        if (!copies.empty()) {
            if (!canon_copy_s_) {
                canon_copy_s_ = make_stream(cudaStreamNonBlocking);
                ck(cudaEventCreateWithFlags(&canon_copy_ev_, cudaEventDisableTiming), "canonical stream event");
                // STRATA_DS41_CANON_COPY_STREAMS=2: the streamed experts alternate over two copy streams (DMA engines)
                if (strata_env_value_double("STRATA_DS41_CANON_COPY_STREAMS", 1.0) == 2.0) {
                    canon_copy_s2_ = make_stream(cudaStreamNonBlocking);
                    ck(cudaEventCreateWithFlags(&canon_copy_ev2_, cudaEventDisableTiming), "canonical stream event 2");
                }
            }
            const auto& LE = experts_[size_t(l)];
            const size_t dn = LE.f.d_row * size_t(c.hidden);
            if (canon_pin_) {   // pinned level: one copy of the whole blob per expert
                for (size_t j = 0; j < copies.size(); ++j) {
                    const auto& [dst, e] = copies[j];
                    const int64_t po = canon_pin_off_[size_t(l) * c.experts + e];
                    if (po < 0) throw std::runtime_error("ds41: streamed expert is not in the pinned level");
                    cudaStream_t cs = canon_copy_s2_ && (j & 1) ? canon_copy_s2_ : canon_copy_s_;
                    ck(cudaMemcpyAsync(dst, canon_pin_ + po, eb, cudaMemcpyHostToDevice, cs), "canonical pinned stream");
                }
            } else if (canon_stage_threads_ > 0) {
                if (!canon_stager_) canon_stager_ = std::make_unique<CanonStager>(canon_stage_threads_, canon_copy_s_);
                canon_jobs_.clear();
                for (const auto& [dst, e] : copies) {
                    const size_t slot = size_t(dst - canon_ring_.b()) / canon_ring_slot_;
                    if (slot >= size_t(canon_stream_) || 2 * g + dn > canon_ring_slot_) throw std::runtime_error("ds41: canonical pinned slot out of bounds");
                    uint8_t* pin = canon_pinned_ + slot * canon_ring_slot_;
                    canon_jobs_.push_back({LE.gate + size_t(e) * LE.gate_stride, pin, dst, g});
                    canon_jobs_.push_back({LE.up + size_t(e) * LE.gate_stride, pin + g, dst + g, g});
                    canon_jobs_.push_back({LE.down + size_t(e) * LE.down_stride, pin + 2 * g, dst + 2 * g, dn});
                }
                canon_stager_->run(canon_jobs_);
            } else
                for (size_t j = 0; j < copies.size(); ++j) {
                    const auto& [dst, e] = copies[j];
                    cudaStream_t cs = canon_copy_s2_ && (j & 1) ? canon_copy_s2_ : canon_copy_s_;
                    ck(cudaMemcpyAsync(dst, LE.gate + size_t(e) * LE.gate_stride, g, cudaMemcpyHostToDevice, cs), "canonical stream gate");
                    ck(cudaMemcpyAsync(dst + g, LE.up + size_t(e) * LE.gate_stride, g, cudaMemcpyHostToDevice, cs), "canonical stream up");
                    ck(cudaMemcpyAsync(dst + 2 * g, LE.down + size_t(e) * LE.down_stride, dn, cudaMemcpyHostToDevice, cs), "canonical stream down");
                }
            ck(cudaEventRecord(canon_copy_ev_, canon_copy_s_), "canonical stream copies done");
            if (canon_copy_s2_) ck(cudaEventRecord(canon_copy_ev2_, canon_copy_s2_), "canonical stream copies done 2");
        }
        k::quantize_q8_K(xn_.f(), canon_xq_.b(), int64_t(T) * c.hidden, stream_);
        // groups [g0, g1) with entries [gstart[g0], gstart[g1]): gate/up, SwiGLU, hidden Q8_K, down
        auto chain = [&](int g0, int g1) {
            if (g0 >= g1) return;
            const int e0 = gstart[size_t(g0)], e1 = gstart[size_t(g1)], ne = e1 - e0;
            k::ds41_canon_dot(f.gu_type, d_gp + g0, d_gs + g0, d_ea, g1 - g0, 2, canon_xq_.b(), canon_xq_.bytes, T, canon_gu_.f(),
                              canon_gu_.bytes, n, c.hidden, c.ff, g, stream_);
            k::ds41_canon_swiglu(canon_gu_.f(size_t(e0) * c.ff), canon_gu_.f(size_t(n + e0) * c.ff), canon_h_.f(size_t(e0) * c.ff),
                                 ne * c.ff, c.clamp[size_t(l)], stream_);
            k::quantize_q8_K(canon_h_.f(size_t(e0) * c.ff), canon_hq_.b(size_t(e0) * hq_bytes), int64_t(ne) * c.ff, stream_);
            k::ds41_canon_dot(f.d_type, d_dp + g0, d_gs + g0, d_id, g1 - g0, 1, canon_hq_.b(), canon_hq_.bytes, n, canon_out_.f(),
                              canon_out_.bytes, n, c.ff, c.hidden, 0, stream_);
        };
        chain(0, Gh);
        if (!copies.empty()) {
            ck(cudaStreamWaitEvent(stream_, canon_copy_ev_, 0), "canonical stream wait");
            if (canon_copy_s2_) ck(cudaStreamWaitEvent(stream_, canon_copy_ev2_, 0), "canonical stream wait 2");
        }
        chain(Gh, G);
        ck(cudaMemcpyAsync(canon_res_, canon_out_.p, size_t(n) * c.hidden * sizeof(float), cudaMemcpyDeviceToHost, stream_),
           "canonical tier results");
    }
    // the expert of route entry i of the current layer (the route ids the pass started with)
    int canon_original_expert(int l, int i) const {
        if (i < 0 || size_t(i) >= canon_ids_now_.size()) throw std::runtime_error("ds41: canonical tier route index out of bounds");
        (void)l;
        return canon_ids_now_[size_t(i)];
    }
    // STRATA_DS41_CANON_CHECK=1: entry i's GPU output against the CPU rows (after the results sync)
    void canon_check(int l, int K, int i) {
        const auto& c = cfg_;
        const auto& f = experts_[size_t(l)].f;
        const int e = canon_ids_now_.at(size_t(i));
        std::vector<uint8_t> w(expert_bytes(l)), xq(f.act_bytes), hq(f.h_bytes);
        slice_to_host(l, e, w.data());
        std::vector<float> hidden(size_t(c.ff)), ref(size_t(c.hidden));
        cpu::native_quant_act(f, pinned_[0] + size_t(i / K) * c.hidden, xq.data());
        const void* xp[] = {xq.data()}; float* hp[] = {hidden.data()};
        cpu::native_gu_rows(f, w.data(), xp, 1, hp, 0, c.ff);
        cpu::native_quant_h(f, hidden.data(), hq.data());
        const void* qp[] = {hq.data()}; float* rp[] = {ref.data()};
        cpu::native_down_rows(f, w.data(), qp, 1, rp, 0, c.hidden);
        size_t diff = 0; double sq = 0, norm = 0;
        for (int j = 0; j < c.hidden; ++j) {
            const float got = canon_res_[size_t(j)];
            if (!std::isfinite(got)) throw std::runtime_error("ds41: nonfinite canonical tier output");
            diff += std::memcmp(&got, &ref[size_t(j)], sizeof(float)) != 0;
            sq += double(got - ref[size_t(j)]) * double(got - ref[size_t(j)]); norm += double(ref[size_t(j)]) * ref[size_t(j)];
        }
        std::cerr << "DS41 canon_check layer=" << l << " expert=" << e << " differing=" << diff << "/" << c.hidden
                  << " relative_l2=" << std::sqrt(sq / (norm + 1e-300)) << "\n";
        canon_checked_[size_t(l)] = 1;
    }
    std::vector<int32_t> canon_ids_now_;
    // STRATA_DS41_REMOTE_OWN: whole experts owned by a remote worker (STD2), per layer x expert
    std::unique_ptr<Remote> own_remote_;
    ucomm_t* ost_comm_ = nullptr;            // STRATA_DS41_OWN_STREAM: the worker's owned experts for streamed prompt passes
    std::vector<ucomm_mr_t*> ost_mrs_;       // the pinned host slots registered on it
    std::thread own_fetcher_;
    int own_staged_ = 0;                     // streamed items whose worker-owned experts have arrived (like staged_ for the kept ones)
    double own_fetch_s_ = 0, own_fetch_gb_ = 0;   // the stream's bytes and wall time per streamed pass
    std::vector<uint8_t> own_;
    // STRATA_DS41_OWN_PROMOTE=N (with STRATA_DS41_OWN_FETCH=HOST:PORT, a strata-ds41-cpu-worker --fetch-port): up to N experts the worker owns
    // that the routes keep asking for are fetched over ucomm into this node's mapped model files (the hole of an expert the model
    // file holds no data for) and then run on this node's lane; the values are the same, only the node changes
    struct PromoGeo { int fd[3]; uint64_t base[3]; size_t gstride, dstride, bytes; };
    std::vector<PromoGeo> prom_geo_;
    std::unique_ptr<std::atomic<uint32_t>[]> own_score_;
    std::unique_ptr<std::atomic<uint8_t>[]> prom_state_;   // 0 on the worker, 1 fetched or fetching
    std::vector<uint8_t> prom_candidates_;                 // owned at setup
    std::mutex prom_m_;
    std::vector<int> prom_ready_;
    std::atomic<bool> prom_has_ready_{false}, prom_stop_{false};
    std::thread prom_thread_;
    ucomm_t* prom_comm_ = nullptr;
    int prom_budget_ = 0;
    std::atomic<int> prom_done_{0};
    double prom_bytes_ = 0;
    void promote_start(int budget, const std::string& addr) {
        const auto& c = cfg_;
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 0;
        cfg.world = 2;
        cfg.root_addr = addr.c_str();
        if (!std::getenv("UCOMM_TIMEOUT_MS")) cfg.timeout_ms = 0;
        Remote::ok(ucomm_init(&cfg, &prom_comm_), "promote connect");
        const size_t n = size_t(c.layers) * size_t(c.experts);
        own_score_ = std::make_unique<std::atomic<uint32_t>[]>(n);
        prom_state_ = std::make_unique<std::atomic<uint8_t>[]>(n);
        for (size_t i = 0; i < n; ++i) { own_score_[i] = 0; prom_state_[i] = 0; }
        prom_candidates_ = own_;
        prom_budget_ = budget;
        prom_geo_.resize(size_t(c.layers));
        for (int l = 0; l < c.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const core::ArtifactTensor* T[3] = {&art_.at(p + "ffn_gate_exps.weight"), &art_.at(p + "ffn_up_exps.weight"), &art_.at(p + "ffn_down_exps.weight")};
            const auto& F = experts_[size_t(l)].f;
            PromoGeo& g = prom_geo_[size_t(l)];
            g.gstride = F.gu_row * size_t(c.ff);
            g.dstride = F.d_row * size_t(c.hidden);
            g.bytes = 2 * g.gstride + g.dstride;
            for (int k = 0; k < 3; ++k) {
                g.fd[k] = ::open(T[k]->file->path().c_str(), O_WRONLY);
                if (g.fd[k] < 0) throw std::runtime_error("ds41 promote: cannot open " + T[k]->file->path() + " for writing");
                g.base[k] = T[k]->file->data_start() + T[k]->tensor->offset;
            }
        }
        prom_thread_ = std::thread([this] { promote_loop(); });
    }
    void promote_loop() {
        unpin_helper();
        const auto& c = cfg_;
        size_t largest = 0;
        for (const auto& g : prom_geo_) largest = std::max(largest, g.bytes);
        std::vector<uint8_t> blob(largest);
        ucomm_mr_t* region = nullptr;
        if (ucomm_mr_reg(prom_comm_, blob.data(), blob.size(), &region) != UCOMM_OK) region = nullptr;
        const size_t n = size_t(c.layers) * size_t(c.experts);
        constexpr uint32_t kMinScore = 3;
        while (!prom_stop_ && prom_done_ < prom_budget_) {
            uint32_t best = kMinScore - 1;
            size_t pick = n;
            for (size_t i = 0; i < n; ++i) {
                if (!prom_candidates_[i] || prom_state_[i].load(std::memory_order_relaxed)) continue;
                const uint32_t v = own_score_[i].load(std::memory_order_relaxed);
                if (v > best) { best = v; pick = i; }
            }
            if (pick == n) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            prom_state_[pick] = 1;
            const int32_t req[3] = {int32_t(pick / size_t(c.experts)), int32_t(pick % size_t(c.experts)), 0};
            const auto& g = prom_geo_[size_t(req[0])];
            ucomm_req_t* rx = nullptr;
            Remote::ok(ucomm_irecv(prom_comm_, 1, 51, blob.data(), g.bytes, &rx), "promote receive");
            Remote::ok(ucomm_send(prom_comm_, 1, 50, req, sizeof req), "promote request");
            size_t got = 0;
            Remote::ok(ucomm_wait(rx, &got), "promote wait");
            if (got != g.bytes) throw std::runtime_error("ds41 promote: short blob");
            const size_t e = size_t(req[1]);
            const struct { const uint8_t* src; size_t len; uint64_t at; int fd; } parts[3] = {
                {blob.data(), g.gstride, g.base[0] + e * g.gstride, g.fd[0]},
                {blob.data() + g.gstride, g.gstride, g.base[1] + e * g.gstride, g.fd[1]},
                {blob.data() + 2 * g.gstride, g.dstride, g.base[2] + e * g.dstride, g.fd[2]}};
            for (const auto& part : parts)
                for (size_t done = 0; done < part.len;) {
                    const ssize_t k = pwrite(part.fd, part.src + done, part.len - done, off_t(part.at + done));
                    if (k <= 0) throw std::runtime_error("ds41 promote: write failed");
                    done += size_t(k);
                }
            prom_bytes_ += double(g.bytes);
            { std::lock_guard<std::mutex> lock(prom_m_); prom_ready_.push_back(int(pick)); }
            prom_has_ready_.store(true, std::memory_order_release);
            ++prom_done_;
        }
        if (region) ucomm_mr_dereg(region);
    }
    void promote_apply() {   // on the decode thread: experts whose rows are in place now run on this node
        if (!prom_has_ready_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(prom_m_);
        for (int i : prom_ready_) own_[size_t(i)] = 0;
        prom_applied_ += prom_ready_.size();
        prom_ready_.clear();
        prom_has_ready_.store(false, std::memory_order_relaxed);
    }
    size_t prom_applied_ = 0;
    uint64_t prom_local_routes_ = 0, prom_remote_routes_ = 0;   // decode-pass routes of the setup-owned experts: run here after promotion / still sent
    void promote_stop() {
        prom_stop_ = true;
        if (prom_thread_.joinable()) prom_thread_.join();
        if (prom_comm_) {
            const int32_t end[3] = {-1, -1, 0};
            (void)ucomm_send(prom_comm_, 1, 50, end, sizeof end);
            ucomm_finalize(prom_comm_);
            prom_comm_ = nullptr;
        }
        for (auto& g : prom_geo_) for (int k = 0; k < 3; ++k) if (g.fd[k] >= 0) { ::close(g.fd[k]); g.fd[k] = -1; }
        prom_geo_.clear();
    }
    // STRATA_DS41_CANON_ADAPT=N (opt-in, with the canonical tier; STRATA_DS41_OWN_FETCH=HOST:PORT for the worker's experts): the tier follows
    // the routes. A thread brings the most-used experts that are not resident (rows from the worker over ucomm, or from the mapped model) into
    // pinned buffers; the decode thread copies one over the least-used same-size resident when it has been used 2+ times more, between
    // passes on the compute stream. The canonical kernels' values do not depend on which expert sits in the tier, so tokens do not change.
    struct AdaptRes { int l, e; int64_t off; size_t bytes; };
    struct AdaptBuf { uint8_t* host = nullptr; cudaEvent_t ev = nullptr; std::atomic<int> state{0}; int l = -1, e = -1; int64_t off = -1; };   // 0 free, 1 filling, 2 ready, 3 copying
    std::vector<AdaptRes> ad_res_;
    std::unique_ptr<std::atomic<uint32_t>[]> ad_score_, ad_rejected_;
    std::unique_ptr<std::atomic<uint8_t>[]> ad_state_;   // 1: resident or being brought
    std::vector<uint8_t> ad_owned_;
    std::vector<std::unique_ptr<AdaptBuf>> ad_bufs_;
    std::thread ad_thread_;
    std::atomic<bool> ad_stop_{false};
    ucomm_t* ad_comm_ = nullptr;
    int ad_budget_ = 0, ad_swaps_ = 0;
    size_t adapt_max_bytes_ = 0, ad_chunk_ = 512 << 10;
    uint32_t ad_owned_weight_ = 1;   // STRATA_DS41_ADAPT_OWNED_WEIGHT
    int ad_gap_ms_ = 5;
    cudaStream_t ad_copy_ = nullptr;   // the swaps' uploads run here, after the compute stream's earlier work, beside it
    cudaEvent_t ad_sync_ = nullptr;
    std::atomic<int> ad_fetched_{0};
    void adapt_start(int budget, const char* fetch_addr) {
        const auto& c = cfg_;
        if (canon_off_.empty()) throw std::invalid_argument("ds41: STRATA_DS41_CANON_ADAPT needs the canonical tier (STRATA_DS41_CANON_MIB)");
        if (prom_thread_.joinable()) throw std::invalid_argument("ds41: STRATA_DS41_CANON_ADAPT and STRATA_DS41_OWN_PROMOTE are alternatives");
        const size_t n = size_t(c.layers) * size_t(c.experts);
        ad_score_ = std::make_unique<std::atomic<uint32_t>[]>(n);
        ad_rejected_ = std::make_unique<std::atomic<uint32_t>[]>(n);
        ad_state_ = std::make_unique<std::atomic<uint8_t>[]>(n);
        for (size_t i = 0; i < n; ++i) { ad_score_[i] = 0; ad_rejected_[i] = 0; ad_state_[i] = 0; }
        ad_owned_ = own_.empty() ? std::vector<uint8_t>(n, 0) : own_;
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (canon_off_[size_t(l) * c.experts + e] >= 0) {
                    ad_res_.push_back({l, e, canon_off_[size_t(l) * c.experts + e], (expert_bytes(l) + 255) / 256 * 256});
                    ad_state_[size_t(l) * c.experts + e] = 1;
                }
        size_t max_b = 0;
        for (int l = 0; l < c.layers; ++l) max_b = std::max(max_b, expert_bytes(l));
        adapt_max_bytes_ = max_b;
        ck(cudaStreamCreateWithFlags(&ad_copy_, cudaStreamNonBlocking), "adapt stream");
        ck(cudaEventCreateWithFlags(&ad_sync_, cudaEventDisableTiming), "adapt sync event");
        for (int i = 0; i < 6; ++i) {
            auto b = std::make_unique<AdaptBuf>();
            b->host = reinterpret_cast<uint8_t*>(pinned_keep(max_b));
            ck(cudaEventCreateWithFlags(&b->ev, cudaEventDisableTiming), "adapt event");
            ad_bufs_.push_back(std::move(b));
        }
        if (fetch_addr && *fetch_addr) {
            ucomm_config cfg;
            ucomm_config_default(&cfg);
            cfg.rank = 0;
            cfg.world = 2;
            cfg.root_addr = fetch_addr;
            if (!std::getenv("UCOMM_TIMEOUT_MS")) cfg.timeout_ms = 0;
            Remote::ok(ucomm_init(&cfg, &ad_comm_), "adapt connect");
            // STRATA_DS41_CANON_PIN_OWNED_MIB: the hottest experts the worker owns (profile order, not in the tier) are fetched once into the pinned
            // level; from then on the GPU streams and runs them instead of the worker's lane
            if (canon_pin_ && canon_pin_cap_ > canon_pin_used_) {
                const auto f0 = std::chrono::steady_clock::now();
                uint8_t* tmp = reinterpret_cast<uint8_t*>(pinned_keep(max_b));
                ucomm_mr_t* mr = nullptr;
                (void)ucomm_mr_reg(ad_comm_, tmp, max_b, &mr);
                int got_n = 0;
                for (const auto& [l, e] : canon_order_) {
                    const size_t idx = size_t(l) * c.experts + e;
                    const size_t b = (expert_bytes(l) + 255) / 256 * 256;
                    if (!ad_owned_[idx] || canon_off_[idx] >= 0 || canon_pin_off_[idx] >= 0) continue;
                    if (canon_pin_used_ + b > canon_pin_cap_) break;
                    const int32_t req[3] = {l, e, 0};
                    ucomm_req_t* r = nullptr;
                    Remote::ok(ucomm_irecv(ad_comm_, 1, 51, tmp, expert_bytes(l), &r), "pin receive");
                    Remote::ok(ucomm_send(ad_comm_, 1, 50, req, sizeof req), "pin request");
                    size_t got = 0;
                    Remote::ok(ucomm_wait(r, &got), "pin wait");
                    if (got != expert_bytes(l)) throw std::runtime_error("ds41 pin: short blob");
                    std::memcpy(canon_pin_ + canon_pin_used_, tmp, got);
                    canon_pin_off_[idx] = int64_t(canon_pin_used_);
                    canon_pin_used_ += b;
                    ++got_n;
                }
                if (mr) ucomm_mr_dereg(mr);
                std::cerr << "DS41 canon_pin_owned experts=" << got_n << " MiB=" << (canon_pin_used_ >> 20) << " s="
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count() << "\n";
            }
        }
        if (const char* v = std::getenv("STRATA_DS41_ADAPT_CHUNK_KIB")) ad_chunk_ = size_t(std::max(0, std::atoi(v))) << 10;
        if (const char* v = std::getenv("STRATA_DS41_ADAPT_GAP_MS")) ad_gap_ms_ = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("STRATA_DS41_ADAPT_OWNED_WEIGHT")) ad_owned_weight_ = uint32_t(std::max(1, std::atoi(v)));
        ad_budget_ = budget;
        ad_thread_ = std::thread([this] { adapt_loop(); });
    }
    void adapt_loop() {
        unpin_helper();
        const auto& c = cfg_;
        const size_t n = size_t(c.layers) * size_t(c.experts);
        constexpr uint32_t kMin = 3;
        std::vector<ucomm_mr_t*> regions;
        if (ad_comm_)
            for (auto& b : ad_bufs_) {
                ucomm_mr_t* r = nullptr;
                if (ucomm_mr_reg(ad_comm_, b->host, adapt_max_bytes_, &r) == UCOMM_OK) regions.push_back(r);
            }
        while (!ad_stop_) {
            AdaptBuf* buf = nullptr;
            for (auto& b : ad_bufs_) if (b->state.load(std::memory_order_acquire) == 0) { buf = b.get(); break; }
            uint32_t best = kMin - 1;
            size_t pick = n;
            if (buf)
                for (size_t i = 0; i < n; ++i) {
                    if (ad_state_[i].load(std::memory_order_relaxed)) continue;
                    const uint32_t raw = ad_score_[i].load(std::memory_order_relaxed);
                    const uint32_t v = raw * (ad_owned_[i] ? ad_owned_weight_ : 1u);   // a worker-owned expert on the GPU saves the slower lane
                    if (v > best && raw >= ad_rejected_[i].load(std::memory_order_relaxed) + 4) { best = v; pick = i; }
                }
            if (!buf || pick == n) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            ad_state_[pick] = 1;
            buf->state.store(1, std::memory_order_relaxed);
            const int l = int(pick / size_t(c.experts)), e = int(pick % size_t(c.experts));
            if (ad_owned_[pick]) {
                if (!ad_comm_) { ad_state_[pick] = 0; ad_rejected_[pick] = ~0u / 2; buf->state.store(0); continue; }
                // the blob arrives in pieces of ad_chunk_ bytes so a reply of the worker's lane never waits behind a whole 7 MB transfer
                const int32_t req[3] = {l, e, int32_t(ad_chunk_)};
                const size_t total = expert_bytes(l), piece = ad_chunk_ ? ad_chunk_ : total;
                std::vector<ucomm_req_t*> rx;
                for (size_t off = 0; off < total; off += piece) {
                    ucomm_req_t* r = nullptr;
                    Remote::ok(ucomm_irecv(ad_comm_, 1, 51, buf->host + off, std::min(piece, total - off), &r), "adapt receive");
                    rx.push_back(r);
                }
                Remote::ok(ucomm_send(ad_comm_, 1, 50, req, sizeof req), "adapt request");
                size_t got_total = 0;
                for (auto* r : rx) {
                    size_t got = 0;
                    Remote::ok(ucomm_wait(r, &got), "adapt wait");
                    got_total += got;
                }
                if (got_total != total) throw std::runtime_error("ds41 adapt: short blob");
                if (ad_gap_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ad_gap_ms_));
            } else {
                slice_to_host(l, e, buf->host);
            }
            buf->l = l;
            buf->e = e;
            ++ad_fetched_;
            buf->state.store(2, std::memory_order_release);
        }
        for (auto* r : regions) ucomm_mr_dereg(r);
    }
    // on the decode thread: finished uploads become usable at the start of a pass (poll), new swaps are issued after the pass's own GPU work is
    // queued, so their DMA overlaps the lane/worker wait instead of delaying the pass's activations
    void adapt_apply(bool issue) {
        if (!ad_thread_.joinable()) return;
        const auto& c = cfg_;
        for (auto& b : ad_bufs_) {
            const int st = b->state.load(std::memory_order_acquire);
            if (st == 3) {   // the upload is done: the expert is usable from this pass on
                if (cudaEventQuery(b->ev) == cudaSuccess) {
                    canon_off_[size_t(b->l) * c.experts + b->e] = b->off;
                    b->state.store(0, std::memory_order_release);
                } else {
                    (void)cudaGetLastError();
                }
                continue;
            }
            if (st != 2 || !issue) continue;
            const size_t cand = size_t(b->l) * c.experts + b->e, need = (expert_bytes(b->l) + 255) / 256 * 256;
            if (ad_swaps_ >= ad_budget_) { ad_state_[cand] = 0; ad_rejected_[cand] = ~0u / 2; b->state.store(0, std::memory_order_release); continue; }
            int victim = -1;
            uint32_t low = ~0u;
            for (size_t i = 0; i < ad_res_.size(); ++i) {
                if (ad_res_[i].bytes != need) continue;
                const size_t ri = size_t(ad_res_[i].l) * c.experts + ad_res_[i].e;
                const uint32_t v = ad_score_[ri].load(std::memory_order_relaxed) * (ad_owned_[ri] ? ad_owned_weight_ : 1u);
                if (v < low) { low = v; victim = int(i); }
            }
            const uint32_t cs = ad_score_[cand].load(std::memory_order_relaxed) * (ad_owned_[cand] ? ad_owned_weight_ : 1u);
            if (victim < 0 || cs < low + 2) {   // not worth a slot yet
                ad_state_[cand] = 0;
                ad_rejected_[cand] = ad_score_[cand].load(std::memory_order_relaxed);
                b->state.store(0, std::memory_order_release);
                continue;
            }
            AdaptRes& v = ad_res_[size_t(victim)];
            const size_t old = size_t(v.l) * c.experts + v.e;
            canon_off_[old] = -1;   // no pass from here on reads the slot; the ones already queued finish before the copy starts
            ad_state_[old] = 0;
            ad_rejected_[old] = ad_score_[old].load(std::memory_order_relaxed);
            ck(cudaEventRecord(ad_sync_, stream_), "adapt sync record");
            ck(cudaStreamWaitEvent(ad_copy_, ad_sync_, 0), "adapt sync wait");
            ck(cudaMemcpyAsync(canon_arena_.b(size_t(v.off)), b->host, expert_bytes(b->l), cudaMemcpyHostToDevice, ad_copy_), "adapt upload");
            ck(cudaEventRecord(b->ev, ad_copy_), "adapt event record");
            b->off = v.off;
            v.l = b->l;
            v.e = b->e;
            b->state.store(3, std::memory_order_release);
            ++ad_swaps_;
        }
    }
    void adapt_count(int l, const int32_t* ids, size_t n) {
        if (!ad_score_) return;
        for (size_t i = 0; i < n; ++i)
            if (ids[i] >= 0 && ids[i] < cfg_.experts) ad_score_[size_t(l) * cfg_.experts + size_t(ids[i])].fetch_add(1, std::memory_order_relaxed);
    }
public:
    /// Dual decoders (STRATA_DS41_DUAL) share one static tier: seed the adaptive tier's scores from a route trace (STRATA_DS41_ROUTES format, STRATA_DS41_CANON_TRAFFIC),
    /// let the running adaptation fetch and swap in the hottest experts (the worker's included) until nothing more is worth a slot or max_swaps, then stop it for good.
    void adapt_settle(const char* traffic_path, int max_swaps) {
        const auto& c = cfg_;
        if (!ad_thread_.joinable() || !ad_score_) throw std::invalid_argument("ds41: adapt_settle needs STRATA_DS41_CANON_ADAPT and STRATA_DS41_OWN_FETCH");
        {
            std::ifstream g(traffic_path, std::ios::binary);
            if (!g) throw std::runtime_error("ds41: cannot read the traffic trace " + std::string(traffic_path));
            std::vector<int32_t> v(size_t(c.layers) * c.top_k);
            while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
                for (int l = 0; l < c.layers; ++l)
                    for (int k = 0; k < c.top_k; ++k) {
                        const int e = v[size_t(l) * c.top_k + k];
                        if (e >= 0 && e < c.experts) ad_score_[size_t(l) * c.experts + size_t(e)].fetch_add(1, std::memory_order_relaxed);
                    }
        }
        const auto t0 = std::chrono::steady_clock::now();
        int quiet = 0, last = ad_swaps_;
        while (ad_swaps_ < max_swaps && quiet < 400) {   // 400 x 5 ms without a swap: settled
            adapt_apply(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (ad_swaps_ != last) { last = ad_swaps_; quiet = 0; } else ++quiet;
        }
        for (int i = 0; i < 100; ++i) { adapt_apply(false); std::this_thread::sleep_for(std::chrono::milliseconds(2)); }   // the last uploads
        ck(cudaStreamSynchronize(stream_), "tier settle");
        adapt_stop();
        std::cerr << "DS41 canon_settle swaps=" << ad_swaps_ << " fetched=" << ad_fetched_.load() << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }
private:
    void adapt_stop() {
        ad_stop_ = true;
        if (ad_thread_.joinable()) ad_thread_.join();
        if (ad_copy_) { (void)cudaStreamSynchronize(ad_copy_); (void)cudaStreamDestroy(ad_copy_); ad_copy_ = nullptr; }
        if (ad_comm_) {
            const int32_t end[3] = {-1, -1, 0};
            (void)ucomm_send(ad_comm_, 1, 50, end, sizeof end);
            ucomm_finalize(ad_comm_);
            ad_comm_ = nullptr;
        }
    }
    std::vector<float> ext_vals_;     // canonical + remote raw route outputs, merged by finish_with_external
    std::vector<int32_t> ext_entries_;
    uint64_t own_routes_ = 0;
    bool own_pending_merge_ = false;
    std::chrono::steady_clock::time_point own_merge_t0_;
public:
    // STRATA_DS41_REMOTE_OWN, decode / verify passes: ms of canonical GPU sync, worker reply wait, lane finish + merge;
    // [3] the layer calls counted
    double own_ms_[4] = {0, 0, 0, 0}, own_latency_ms_ = 0;
    double own_prep_ms_ = 0, own_send_ms_ = 0;
    long own_starts_ = 0;   // own_latency_: from sending a layer's request to its reply in hand
    std::chrono::steady_clock::time_point own_start_t_;
    std::string promote_summary() const {
        if (ad_budget_ > 0) return "DS41 canon_adapt swaps=" + std::to_string(ad_swaps_) + " fetched=" + std::to_string(ad_fetched_.load()) + "\n";
        if (prom_budget_ <= 0) return {};
        return "DS41 own_promote applied=" + std::to_string(prom_applied_) + " fetched=" + std::to_string(prom_done_.load()) + " GiB=" +
               std::to_string(prom_bytes_ / double(1ull << 30)) + " decode routes of setup-owned experts: local " + std::to_string(prom_local_routes_) +
               " remote " + std::to_string(prom_remote_routes_) + "\n";
    }
private:
    int canon_stream_ = 0;          // STRATA_DS41_CANON_STREAM: experts streamed per layer pass
    cudaStream_t canon_copy_s_ = nullptr;
    cudaEvent_t canon_copy_ev_ = nullptr;
    cudaStream_t canon_copy_s2_ = nullptr;
    cudaEvent_t canon_copy_ev2_ = nullptr;
    int canon_stage_threads_ = 0;   // STRATA_DS41_CANON_STREAM_THREADS (0: pageable copies)
    uint8_t* canon_pinned_ = nullptr;
    // STRATA_DS41_CANON_PIN_MIB: the next-hottest experts after the tier, resident in host-pinned memory (same blob layout as the arena);
    // the streamed experts of a pass come from here by one async copy each
    uint8_t* canon_pin_ = nullptr;
    std::vector<int64_t> canon_pin_off_;   // [layer][expert] byte offset into canon_pin_, -1: not pinned
    std::vector<std::pair<int, int>> canon_order_;   // (layer, expert) by profile score per byte, hottest first
    size_t canon_pin_used_ = 0, canon_pin_cap_ = 0;   // pinned level: bytes filled / allocated (STRATA_DS41_CANON_PIN_MIB + _OWNED_MIB)
    std::vector<CanonStager::Job> canon_jobs_;
    std::unique_ptr<CanonStager> canon_stager_;
    size_t canon_ring_slot_ = 0;
    Dev canon_ring_;
    void staged_cpu_order_chain(int l, const uint8_t* base, size_t capacity, int i, float w, float* result) {
        const auto& c = cfg_;
        const auto& f = experts_[size_t(l)].f;
        const size_t g = slice_gate(l), d = f.d_row * size_t(c.hidden);
        const size_t xq_bytes = size_t(c.hidden / 256) * 292, hq_bytes = size_t(c.ff / 256) * 292;
        if (!base || g > capacity / 2 || d > capacity - 2*g || !stage_cpu_xq_.p || !stage_cpu_hq_.p ||
            stage_cpu_xq_.bytes < xq_bytes || stage_cpu_hq_.bytes < hq_bytes || i < 0 || i >= 8)
            throw std::runtime_error("ds41: CPU-order staged expert buffers or route are out of bounds");
        k::quantize_q8_K(xn_.f(size_t(i) * c.hidden), stage_cpu_xq_.b(), c.hidden, stream_);
        if (strata_env_is_one("STRATA_DS41_STAGE_PAIR_DOT")) {
            const size_t row_bytes = size_t(c.hidden / 256) * size_t(f.gu_type == 19 ? 50 : 66);
            if ((f.gu_type != 16 && f.gu_type != 19) || g != size_t(c.ff) * row_bytes)
                throw std::runtime_error("ds41: paired staging gate stride is invalid");
            k::ds41_cpu_order_iq_dot_pair(f.gu_type, base, 2*g, stage_cpu_xq_.b(), stage_cpu_xq_.bytes,
                                          g_gate_.f(), g_gate_.bytes, g_up_.f(), g_up_.bytes, c.hidden, c.ff, stream_);
        } else {
            k::ds41_cpu_order_iq_dot(f.gu_type, base, g, stage_cpu_xq_.b(), stage_cpu_xq_.bytes,
                                     g_gate_.f(), g_gate_.bytes, c.hidden, c.ff, stream_);
            k::ds41_cpu_order_iq_dot(f.gu_type, base + g, g, stage_cpu_xq_.b(), stage_cpu_xq_.bytes,
                                     g_up_.f(), g_up_.bytes, c.hidden, c.ff, stream_);
        }
        if (stage_cpu_host_act_) {
            if (!stage_cpu_gateup_ || !stage_cpu_hq_host_ || stage_cpu_hidden_.size() != size_t(c.ff) || f.h_bytes != hq_bytes)
                throw std::runtime_error("ds41: CPU-order host activation buffers are invalid");
            const bool sample = stage_cpu_timing_ && stage_cpu_transfers_ < 1000;
            const auto wait_start = std::chrono::steady_clock::now();
            ck(cudaMemcpyAsync(stage_cpu_gateup_, g_gate_.p, size_t(c.ff) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "staged gate readback");
            ck(cudaMemcpyAsync(stage_cpu_gateup_ + c.ff, g_up_.p, size_t(c.ff) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "staged up readback");
            ck(cudaStreamSynchronize(stream_), "staged activation sync");
            const auto act_start = std::chrono::steady_clock::now();
            if (sample) stage_timing_[2] += std::chrono::duration<double>(act_start - wait_start).count();
            const float limit = c.clamp[size_t(l)];
            for (int j = 0; j < c.ff; ++j) {
                const float gate = std::fmin(stage_cpu_gateup_[j], limit);
                const float up = std::fmax(-limit, std::fmin(stage_cpu_gateup_[c.ff + j], limit));
                stage_cpu_hidden_[size_t(j)] = (gate / (1.f + std::exp(-gate))) * up;
            }
            cpu::native_quant_h(f, stage_cpu_hidden_.data(), stage_cpu_hq_host_);
            if (sample) stage_timing_[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - act_start).count();
            ck(cudaMemcpyAsync(stage_cpu_hq_.p, stage_cpu_hq_host_, hq_bytes, cudaMemcpyHostToDevice, stream_), "staged hidden Q8_K upload");
        } else {
            k::ds41_cpu_order_swiglu(g_gate_.f(), g_up_.f(), g_h_.f(), c.ff, c.clamp[size_t(l)], stream_);
            k::quantize_q8_K(g_h_.f(), stage_cpu_hq_.b(), c.ff, stream_);
        }
        k::ds41_cpu_order_iq_dot(f.d_type, base + 2*g, d, stage_cpu_hq_.b(), stage_cpu_hq_.bytes,
                                 g_out_.f(), g_out_.bytes, c.ff, c.hidden, stream_);
        if (strata_env_is_one("STRATA_DS41_STAGE_CPU_ORDER_CHECK") && !stage_cpu_order_checked_[size_t(l)]) {
            if (f.act_bytes != xq_bytes || f.h_bytes != hq_bytes)
                throw std::runtime_error("ds41: CPU-order diagnostic activation geometry mismatch");
            std::vector<uint8_t> xq(f.act_bytes), hq(f.h_bytes);
            std::vector<uint8_t> gpu_xq(f.act_bytes), gpu_hq(f.h_bytes);
            std::vector<float> hidden(size_t(c.ff)), reference(size_t(c.hidden)), got(size_t(c.hidden));
            cpu::native_quant_act(f, pinned_[0] + size_t(i) * c.hidden, xq.data());
            const void* xp[] = {xq.data()}; float* hp[] = {hidden.data()};
            cpu::native_gu_rows(f, stage_cpu_host_, xp, 1, hp, 0, c.ff);
            cpu::native_quant_h(f, hidden.data(), hq.data());
            const void* qp[] = {hq.data()}; float* rp[] = {reference.data()};
            cpu::native_down_rows(f, stage_cpu_host_, qp, 1, rp, 0, c.hidden);
            ck(cudaMemcpyAsync(gpu_xq.data(), stage_cpu_xq_.p, f.act_bytes, cudaMemcpyDeviceToHost, stream_), "CPU-order input Q8_K check");
            ck(cudaMemcpyAsync(gpu_hq.data(), stage_cpu_hq_.p, f.h_bytes, cudaMemcpyDeviceToHost, stream_), "CPU-order hidden Q8_K check");
            ck(cudaMemcpyAsync(got.data(), g_out_.p, size_t(c.hidden) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "CPU-order expert check copy");
            ck(cudaStreamSynchronize(stream_), "CPU-order expert check sync");
            size_t different = 0; double squared = 0, norm = 0, max_abs = 0;
            for (int j = 0; j < c.hidden; ++j) {
                if (!std::isfinite(reference[size_t(j)]) || !std::isfinite(got[size_t(j)]))
                    throw std::runtime_error("ds41: nonfinite CPU-order expert diagnostic");
                different += std::memcmp(&reference[size_t(j)], &got[size_t(j)], sizeof(float)) != 0;
                const double delta = double(reference[size_t(j)]) - double(got[size_t(j)]);
                squared += delta * delta; norm += double(reference[size_t(j)]) * reference[size_t(j)];
                max_abs = std::max(max_abs, std::abs(delta));
            }
            std::cerr << "DS41 cpu_order_check layer=" << l << " differing=" << different << "/" << c.hidden
                      << " max_abs=" << max_abs << " relative_l2=" << std::sqrt(squared / (norm + 1e-300));
            size_t xq_diff = 0, hq_diff = 0;
            for (size_t j = 0; j < xq.size(); ++j) xq_diff += xq[j] != gpu_xq[j];
            for (size_t j = 0; j < hq.size(); ++j) hq_diff += hq[j] != gpu_hq[j];
            std::cerr << " input_q8_bytes=" << xq_diff << " hidden_q8_bytes=" << hq_diff << "\n";
            stage_cpu_diag_out_ = std::move(got);
            stage_cpu_order_checked_[size_t(l)] = 1;
        }
        if (result)
            ck(cudaMemcpyAsync(result, g_out_.p, size_t(c.hidden) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "CPU-order staged result");
        else
            k::ds41_axpy(g_out_.f(), w, g_sum_.f(size_t(i) * c.hidden), c.hidden, stream_);
    }
    void staged_cpu_order_batch(int l, const uint8_t* base, size_t capacity, const std::vector<int>& routes, int K, float* results) {
        const auto& c = cfg_;
        const auto& f = experts_[size_t(l)].f;
        const size_t g = slice_gate(l), d = f.d_row * size_t(c.hidden);
        const size_t xq = size_t(c.hidden / 256) * 292, hq = size_t(c.ff / 256) * 292;
        if (!stage_cpu_batch_act_ || !stage_batch_gateup_ || !stage_batch_hq_ || !results || !base ||
            routes.empty() || routes.size() > 8 || K < 1 || g > capacity / 2 || d > capacity - 2*g ||
            stage_cpu_xq_.bytes < xq || stage_cpu_hq_.bytes < hq || f.h_bytes != hq ||
            stage_cpu_hidden_.size() != size_t(c.ff))
            throw std::runtime_error("ds41: batched host activation buffers or dimensions invalid");
        const size_t row_bytes = size_t(c.hidden / 256) * size_t(f.gu_type == 19 ? 50 : 66);
        if ((f.gu_type != 16 && f.gu_type != 19) || g != size_t(c.ff) * row_bytes)
            throw std::runtime_error("ds41: batched host activation gate stride invalid");
        ++stage_batch_groups_;
        stage_batch_routes_ += routes.size();
        for (size_t j = 0; j < routes.size(); ++j) {
            const int i = routes[j] / K;
            if (routes[j] < 0 || i >= 8) throw std::runtime_error("ds41: batched host activation route invalid");
            k::quantize_q8_K(xn_.f(size_t(i) * c.hidden), stage_cpu_xq_.b(), c.hidden, stream_);
            k::ds41_cpu_order_iq_dot_pair(f.gu_type, base, 2*g, stage_cpu_xq_.b(), stage_cpu_xq_.bytes,
                                          g_gate_.f(), g_gate_.bytes, g_up_.f(), g_up_.bytes, c.hidden, c.ff, stream_);
            float* dest = stage_batch_gateup_ + j * size_t(2) * c.ff;
            ck(cudaMemcpyAsync(dest, g_gate_.p, size_t(c.ff) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "batch staged gate readback");
            ck(cudaMemcpyAsync(dest + c.ff, g_up_.p, size_t(c.ff) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "batch staged up readback");
        }
        ck(cudaStreamSynchronize(stream_), "batch staged activation sync");
        const float limit = c.clamp[size_t(l)];
        for (size_t j = 0; j < routes.size(); ++j) {
            const float* src = stage_batch_gateup_ + j * size_t(2) * c.ff;
            for (int k = 0; k < c.ff; ++k) {
                const float gate = std::fmin(src[k], limit);
                const float up = std::fmax(-limit, std::fmin(src[c.ff + k], limit));
                stage_cpu_hidden_[size_t(k)] = (gate / (1.f + std::exp(-gate))) * up;
            }
            uint8_t* q = stage_batch_hq_ + j * hq;
            cpu::native_quant_h(f, stage_cpu_hidden_.data(), q);
            ck(cudaMemcpyAsync(stage_cpu_hq_.p, q, hq, cudaMemcpyHostToDevice, stream_), "batch staged hidden upload");
            k::ds41_cpu_order_iq_dot(f.d_type, base + 2*g, d, stage_cpu_hq_.b(), stage_cpu_hq_.bytes,
                                     g_out_.f(), g_out_.bytes, c.ff, c.hidden, stream_);
            ck(cudaMemcpyAsync(results + j * size_t(c.hidden), g_out_.p, size_t(c.hidden) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_), "batch staged result");
        }
    }
    std::vector<Dev> dstream_;   // STRATA_DS41_DECODE_STREAM: GPU 0's slots for the CPU's experts
    // One bounded staging slot: mapped expert pages themselves are never registered.
    bool stage_cpu_expert_ = false, stage_cpu_copy_pending_ = false, stage_cpu_order_ = false;
    bool stage_cpu_collect_ = false;
    bool stage_cpu_host_act_ = false;
    bool stage_cpu_batch_act_ = false;
    size_t stage_batch_groups_ = 0, stage_batch_routes_ = 0;
    bool stage_cpu_timing_ = false;
    size_t stage_timed_groups_ = 0;
    double stage_timing_[5] = {};
    int stage_cpu_quota_ = 1;
    struct StageCacheEntry { int layer = -1, expert = -1; uint64_t age = 0; };
    int stage_cache_slots_requested_ = 0;
    uint64_t stage_cache_clock_ = 0, stage_cache_hits_ = 0, stage_cache_misses_ = 0;
    uint64_t stage_cache_skipped_misses_ = 0;
    std::vector<uint32_t> stage_cache_frequency_;
    Dev stage_cache_device_;
    std::vector<StageCacheEntry> stage_cache_;
    float* stage_cpu_results_ = nullptr;
    float* stage_cpu_gateup_ = nullptr;
    uint8_t* stage_cpu_hq_host_ = nullptr;
    float* stage_batch_gateup_ = nullptr;
    uint8_t* stage_batch_hq_ = nullptr;
    std::vector<float> stage_cpu_hidden_;
    std::vector<uint8_t> stage_cpu_order_checked_;
    // STRATA_DS41_CANON_MIB: the canonical tier (arena offset per layer x expert, -1: not held) and its scratch
    std::vector<int64_t> canon_off_;
    Dev canon_arena_, canon_xq_, canon_gu_, canon_h_, canon_hq_, canon_out_, canon_meta_;
    uint8_t* canon_meta_host_ = nullptr;
    float* canon_res_ = nullptr;
    int canon_tmax_ = 8;   // widest pass the canonical tier takes (scratch sized for it)
    std::vector<uint8_t> canon_checked_;
    std::vector<float> stage_cpu_diag_out_;
    Dev stage_cpu_xq_, stage_cpu_hq_;
    size_t stage_cpu_capacity_ = 0, stage_cpu_transfers_ = 0;
    uint8_t* stage_cpu_host_ = nullptr;
    uint8_t* stage_cpu_check_host_ = nullptr;
    Dev stage_cpu_device_;
    cudaEvent_t stage_cpu_copy_done_ = nullptr;
    std::vector<uint8_t> stage_cpu_checked_;
    GroupedRunner grouped0_;     // STRATA_DS41_DECODE_GROUPED: GPU 0's
    GroupedRunner grown0_;       // STRATA_DS41_GROWN_GROUPED: GPU 0's grown experts
    // STRATA_DS41_DEFER_RANK: the deferred routes (lane ids / weights), their input and output (pinned), on the device
    std::vector<int> def_ids_;
    std::vector<float> def_wts_;
    float* def_in_ = nullptr;
    float* def_out_ = nullptr;
    Dev def_dev_, def_coeff_;
    bool defer_pending_ = false;
    int defer_tokens_ = 0;
    // the pending deferred sum into the streams: after the next layer's attention post-mix (its comb), or after the
    // last layer (last: no further mix)
    // STRATA_DS41_DEFER_REMOTE=1 (lossy, opt-in): the worker / canonical-tier / lane merge of a layer runs as this job after the next
    // layer's attention instead of before it
    std::function<void()> defer_job_;
    std::vector<int32_t> dr_ids_;
    std::vector<float> dr_wts_;
    void defer_apply(bool last) {
        if (defer_job_) { auto job = std::move(defer_job_); defer_job_ = nullptr; job(); }
        else lane_->finish();
        defer_pending_ = false;
        ck(cudaMemcpyAsync(def_dev_.p, def_out_, size_t(defer_tokens_) * cfg_.hidden * 4, cudaMemcpyHostToDevice, stream_), "deferred experts");
        k::ds41_defer_add(def_dev_.f(), h_.f(), def_coeff_.f(4), last ? nullptr : coeff_attn_.f(8), cfg_.hidden, stream_, defer_tokens_);
    }
    size_t max_stride_ = 0;
    void gpu_experts(int l, int i, const int* ids, const float* wts, int n) {
        const auto& c = cfg_;
        const auto& L = experts_[size_t(l)];
        const size_t g = slice_gate(l);
        const int F = fl_;
        float* sum = g_sum_.f(size_t(i) * c.hidden);
        k::native_quantize_q8_1(xn_.f(size_t(i) * c.hidden), gx_q8_.p, c.hidden, 1, stream_);
        for (int j = 0; j < n; ++j) {
            const uint8_t* base = slots_[size_t(slot_[size_t(l)][size_t(ids[j])])].p;
            k::native_mmvq(L.f.gu_type, base, gx_q8_.p, g_gate_.f(), c.hidden, F, 1, stream_);
            k::native_mmvq(L.f.gu_type, base + g, gx_q8_.p, g_up_.f(), c.hidden, F, 1, stream_);
            k::glm_swiglu(g_gate_.f(), g_up_.f(), g_h_.f(), F, c.clamp[size_t(l)], stream_);
            k::native_quantize_q8_1(g_h_.f(), g_h_q8_.p, F, 1, stream_);
            k::native_mmvq(L.f.d_type, base + 2 * g, g_h_q8_.p, g_out_.f(), F, c.hidden, 1, stream_);
            k::ds41_axpy(g_out_.f(), wts[j], sum, c.hidden, stream_);
        }
    }

    // STRATA_DS41_STREAM_MIN (1024): chunks of at least this many tokens compute every routed expert on the GPU, the
    // non-resident ones in batches: copied by STRATA_DS41_STAGE_THREADS (8) threads from the mapped GGUF into one of
    // two pinned host buffers, uploaded to one of two VRAM buffers (STRATA_DS41_STAGE_MIB, 256, each), computed there;
    // the host copy of one batch runs under the upload and compute of the previous ones. (Registering the mapping
    // for direct DMA would pin all of it: the 113 GB of experts leave too little of 128 GB for the rest.)
    // one expert (rows at base: [gate | up | down]) over its n tokens (idx, w on the device): g_sum_[idx] += w E(xn_[idx])
    void expert_columns(int l, const uint8_t* base, const int* idx, const float* w, int n) {
        const auto& c = cfg_;
        const auto& L = experts_[size_t(l)];
        const size_t g = slice_gate(l);
        const int F = fl_;
        for (int j = 0; j < n; j += 8) {
            const int m = std::min(8, n - j);
            k::ds41_gather(xn_.f(), idx + j, m, c.hidden, s_x_.f(), stream_);
            k::native_quantize_q8_1(s_x_.f(), s_xq_.p, c.hidden, m, stream_);
            k::native_mmvq(L.f.gu_type, base, s_xq_.p, s_gate_.f(), c.hidden, F, m, stream_);
            k::native_mmvq(L.f.gu_type, base + g, s_xq_.p, s_up_.f(), c.hidden, F, m, stream_);
            k::glm_swiglu(s_gate_.f(), s_up_.f(), s_h_.f(), m * F, c.clamp[size_t(l)], stream_);
            k::native_quantize_q8_1(s_h_.f(), s_hq_.p, F, m, stream_);
            k::native_mmvq(L.f.d_type, base + 2 * g, s_hq_.p, s_out_.f(), F, c.hidden, m, stream_);
            k::ds41_scatter_axpy(s_out_.f(), idx + j, w + j, m, c.hidden, g_sum_.f(), stream_);
        }
    }

    // ---- pipelined expert streaming (passes of at least stream_min_ tokens) ----
    // A producer thread copies every layer's experts, in index order, in items of up to slot_bytes_: into one of
    // hslots_ pinned host buffers (STRATA_DS41_STAGE_THREADS threads, non-temporal stores), then by DMA on copy_s_
    // into one of dslots_ VRAM slots, running ahead of the compute across layer boundaries; the FFN of each layer
    // consumes its items in order (MMQ over the item's experts, those without tokens as empty row ranges).
    struct StreamItem { int layer, first, n; size_t stride; };
    size_t stream_stride(int l) const {
        const auto& L = experts_[size_t(l)];
        const size_t g = L.f.gu_row * size_t(cfg_.ff), d = L.f.d_row * size_t(cfg_.hidden);
        // whole blocks of both formats (MMQ steps experts in blocks of its type), 256-byte aligned
        const size_t unit = std::lcm(std::lcm(size_t(ggml_type_size(ggml_type(L.f.gu_type))), size_t(ggml_type_size(ggml_type(L.f.d_type)))), size_t(256));
        return (2 * g + d + unit - 1) / unit * unit;
    }
    static void copy_nt(uint8_t* dst, const uint8_t* src, size_t n) {
        static const bool nt = __builtin_cpu_supports("avx2") && std::getenv("STRATA_DS41_NT") != nullptr;   // opt-in: measured slower here
        if (nt) copy_nt_avx2(dst, src, n);
        else std::memcpy(dst, src, n);
    }

    // STRATA_DS41_PREFILL_HYBRID=m (opt-in; one GPU, the experts read in place): in a streamed pass the producer copies
    // only the experts the layer's routing uses, and those with at most m tokens (m >= 1; at most
    // STRATA_DS41_PREFILL_HYBRID_MAX, 0.35, of the layer's used experts) run on the CPU lane meanwhile, not copied.
    // m = 0: unused experts skipped only. The producer waits for each layer's routing before its copies.
    int hybrid_m() const {
        static const int m = std::getenv("STRATA_DS41_PREFILL_HYBRID") ? std::atoi(std::getenv("STRATA_DS41_PREFILL_HYBRID")) : -1;
        return (m >= 0 && !peer_ && !remote_ && packed_.empty() && fl_ == cfg_.ff) ? m : -1;
    }
    void stream_begin() {
        const auto& c = cfg_;
        stream_end();
        items_.clear();
        need_.assign(size_t(c.layers), std::vector<char>(size_t(c.experts), 1));
        need_ready_.assign(size_t(c.layers), hybrid_m() >= 0 ? 0 : 1);
        item0_.assign(size_t(c.layers) + 1, 0);
        for (int l = 0; l < c.layers; ++l) {
            item0_[size_t(l)] = int(items_.size());
            const size_t b = stream_stride(l), per = std::max<size_t>(1, slot_bytes_ / b);
            const int S = s0_.empty() ? n_stream_[size_t(l)] : s0_[size_t(l)];   // positions of order_[l] streamed here
            for (int e = 0; e < S; e += int(per)) items_.push_back({l, e, int(std::min<size_t>(per, size_t(S - e))), b});
        }
        item0_[size_t(c.layers)] = int(items_.size());
        produced_ = 0;
        consumed_ = 0;
        staged_ = 0;
        own_staged_ = 0;
        dma_issued_ = 0;
        fetched_ = 0;
        rdma_issued_ = 0;
        stop_stream_ = false;
        pin_experts();
        unpinned_.clear();
        for (int idx = 0; idx < int(items_.size()); ++idx)
            if (!pinned_layer_[size_t(items_[size_t(idx)].layer)]) unpinned_.push_back(idx);
        if (!unpinned_.empty() && hring_.empty()) alloc_hring();
        if (ost_comm_ && ost_mrs_.empty())
            for (uint8_t* h : hring_) {
                ucomm_mr_t* mr = nullptr;
                if (ucomm_mr_reg(ost_comm_, h, slot_bytes_, &mr) == UCOMM_OK) ost_mrs_.push_back(mr);
            }
        if (ost_comm_ && !hring_.empty() && ost_mrs_.size() != hring_.size()) std::cerr << "DS41 own_stream: only " << ost_mrs_.size() << " of " << hring_.size() << " host slots registered\n";
        if (std::getenv("STRATA_DS41_COPY_PROBE")) {   // the staging copy alone, before the pass: GB/s
            for (int l = 0; l < c.layers; ++l) {
                if (pinned_layer_[size_t(l)]) continue;
                const auto& L = experts_[size_t(l)];
                const Src& src = stream_src_[size_t(l)];
                const size_t g = L.f.gu_row * size_t(c.ff);
                const int per = int(slot_bytes_ / (2 * g + L.f.d_row * size_t(c.hidden)));
                for (int rep = 0; rep < 2; ++rep) {
                    const auto a = std::chrono::steady_clock::now();
                    std::vector<std::thread> w;
                    const int nt = stage_threads_;
                    for (int t = 0; t < nt; ++t)
                        w.emplace_back([&, t] { unpin_helper(); for (int j = t; j < per; j += nt) copy_nt(hring_[0] + size_t(j) * 3 * g, src.gate + size_t(j) * g, 3 * g); });
                    for (auto& x : w) x.join();
                    std::cerr << "DS41 copy_probe layer=" << l << " rep=" << rep << " GB/s="
                              << double(per) * 3 * g / 1e9 / std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() << "\n";
                }
                break;
            }
        }
        if (peer_frac_ > 0) peer_->stream_begin();
        producer_ = std::thread([this] { unpin_helper(); produce(); });
        stager_ = std::thread([this] { unpin_helper(); stage(); });
        if (ost_comm_ && !unpinned_.empty())
            own_fetcher_ = std::thread([this] {
                unpin_helper();
                try {
                    own_fetch();
                } catch (const std::exception& e) {   // a failed transfer cannot be repaired mid-pass; the other stream threads would wait for it forever
                    std::cerr << "ds41: the worker's expert stream failed: " << e.what() << "\n";
                    std::_Exit(1);
                }
            });
        if (remote_ && !remote_gpu_) fetcher_ = std::thread([this] { unpin_helper(); fetch(); });
    }
    void stream_end() {
        if (peer_) peer_->stream_end();
        if (!producer_.joinable()) return;
        { std::lock_guard<std::mutex> lock(sm_); stop_stream_ = true; }
        scv_.notify_all();
        producer_.join();
        if (stager_.joinable()) stager_.join();
        if (fetcher_.joinable()) fetcher_.join();
        if (own_fetcher_.joinable()) {
            own_fetcher_.join();
            std::cerr << "DS41 own_fetch GB=" << own_fetch_gb_ << " s=" << own_fetch_s_ << " GB/s=" << (own_fetch_s_ > 0 ? own_fetch_gb_ / own_fetch_s_ : 0.0) << "\n";
            own_fetch_gb_ = own_fetch_s_ = 0;
        }
    }
    // The worker-owned experts of each unpinned item, in the order of the items: one request (layer, ids) on the stream link, one message per expert received
    // straight into its position of the item's pinned host slot.
    void own_fetch() {
        const auto& c = cfg_;
        const int N = int(unpinned_.size());
        const auto t_start = std::chrono::steady_clock::now();
        struct Wall { double& s; std::chrono::steady_clock::time_point t0; ~Wall() { s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } } wall{own_fetch_s_, t_start};
        // STRATA_DS41_OWN_STREAM_DEPTH (2): items whose requests are out at once (the worker answers them in order, so the link never waits for a request)
        static const int depth = std::getenv("STRATA_DS41_OWN_STREAM_DEPTH") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_OWN_STREAM_DEPTH"))) : 2;
        struct Flight { int u; std::vector<ucomm_req_t*> rx; double bytes; };
        std::deque<Flight> flying;
        int posted = 0, done = 0;
        auto slot_free = [&](int u) {   // the host slot's previous item (u - hslots_) has been DMAed
            if (u < hslots_) return true;
            std::lock_guard<std::mutex> lock(sm_);
            return stop_stream_ || dma_issued_ >= u - hslots_ + 1;
        };
        auto post = [&](int u) {
            const StreamItem& it = items_[size_t(unpinned_[size_t(u)])];
            const int hs = u % hslots_;
            if (u >= hslots_) cudaEventSynchronize(ev_dma_[size_t(hs)]);
            const size_t eb = expert_bytes(it.layer);
            std::vector<int32_t> want;
            std::vector<int> pos;
            for (int j = 0; j < it.n; ++j) {
                const int e = order_[size_t(it.layer)][size_t(it.first + j)];
                if (!own_.empty() && own_[size_t(it.layer) * c.experts + size_t(e)]) { want.push_back(e); pos.push_back(j); }
            }
            Flight f{u, {}, double(want.size()) * double(eb) / 1e9};
            if (!want.empty()) {
                for (size_t i = 0; i < want.size(); ++i) {
                    ucomm_req_t* r = nullptr;
                    Remote::ok(ucomm_irecv(ost_comm_, 1, 62, hring_[size_t(hs)] + size_t(pos[i]) * it.stride, eb, &r), "stream receive");
                    f.rx.push_back(r);
                }
                const int32_t hdr[4] = {it.layer, int32_t(want.size()), 0, 0};
                Remote::ok(ucomm_send(ost_comm_, 1, 60, hdr, sizeof hdr), "stream request");
                Remote::ok(ucomm_send(ost_comm_, 1, 61, want.data(), want.size() * 4), "stream ids");
            }
            flying.push_back(std::move(f));
        };
        while (done < N) {
            while (posted < N && int(flying.size()) < depth && slot_free(posted)) post(posted++);
            if (flying.empty()) {   // nothing out: wait for the slot of the next item
                std::unique_lock<std::mutex> lock(sm_);
                scv_.wait(lock, [&] { return stop_stream_ || posted < hslots_ || dma_issued_ >= posted - hslots_ + 1; });
                if (stop_stream_) return;
                continue;
            }
            Flight f = std::move(flying.front());
            flying.pop_front();
            for (auto* r : f.rx) Remote::ok(ucomm_wait(r, nullptr), "stream wait");
            own_fetch_gb_ += f.bytes;
            {
                std::lock_guard<std::mutex> lock(sm_);
                own_staged_ = f.u + 1;
                if (stop_stream_) return;
            }
            scv_.notify_all();
            ++done;
        }
    }
    // STRATA_DS41_PIN_GB (60): register this many GiB of the shared mapping's experts for DMA, whole layers spread
    // evenly over the depth (0: none; STRATA_DS41_PIN_EXPERTS=1: all). Read-only, ~0.06 s per GiB, once, at the first
    // streamed prompt; the pages stay pinned for the decoder's life. Their layers stream straight from the page cache
    // (strided DMA, no host copy), the others through the pinned host slots, so the two overlap.
    void pin_experts() {
        if (pinned_tried_) return;
        pinned_tried_ = true;
        const auto& c = cfg_;
        pinned_layer_.assign(size_t(c.layers), 0);
        gu_pin_.assign(size_t(c.layers), 0);
        down_pin_.assign(size_t(c.layers), 0);
        if (fl_ != c.ff) { pin_split(); return; }
        double total = 0;
        for (int l = 0; l < c.layers; ++l) total += double(expert_bytes(l)) * c.experts;
        const char* all = std::getenv("STRATA_DS41_PIN_EXPERTS");
        const char* gb = std::getenv("STRATA_DS41_PIN_GB");
        const double budget = all && std::string(all) == "1" ? total : (gb ? std::atof(gb) : 60.0) * double(1ull << 30);
        const double f = std::min(1.0, budget / total);
        if (f <= 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        for (int l = 0; l < c.layers; ++l) {
            if (std::floor((l + 1) * f + 1e-9) == std::floor(l * f + 1e-9)) continue;   // spread: every 1/f-th layer
            pinned_layer_[size_t(l)] = 1;
            for (const char* n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const auto& t = art_.at("blk." + std::to_string(l) + "." + n);
                const uintptr_t a = reinterpret_cast<uintptr_t>(shared(t));
                ranges.push_back({a / page * page, (a + t.bytes + page - 1) / page * page});
            }
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;   // registrations must not share a page
        for (const auto& r : ranges)
            if (!merged.empty() && r.first <= merged.back().second) merged.back().second = std::max(merged.back().second, r.second);
            else merged.push_back(r);
        size_t bytes = 0;
        for (const auto& [a, b] : merged) {
            if (cudaHostRegister(reinterpret_cast<void*>(a), b - a, 0x08 /* read-only */) != cudaSuccess) {
                (void)cudaGetLastError();
                for (const auto& r : registered_) cudaHostUnregister(r);
                registered_.clear();
                pinned_layer_.assign(size_t(c.layers), 0);
                std::cerr << "DS41 expert pinning failed; streaming through host copies\n";
                return;
            }
            registered_.push_back(reinterpret_cast<void*>(a));
            bytes += b - a;
        }
        gu_pin_ = down_pin_ = pinned_layer_;
        std::cerr << "DS41 experts pinned GiB=" << bytes / double(1ull << 30) << " layers="
                  << std::count(pinned_layer_.begin(), pinned_layer_.end(), 1) << "/" << c.layers << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }

    // Two nodes: the same budget on whole down tensors (all of their columns are read here); the gate/up tensors
    // are only partly here, so registering them whole would read the worker's rows into this node's page cache.
    void pin_split() {
        const auto& c = cfg_;
        const char* gb = std::getenv("STRATA_DS41_PIN_GB");
        double budget = (gb ? std::atof(gb) : 60.0) * double(1ull << 30);
        if (budget <= 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
        bool failed = false;
        auto reg = [&](const uint8_t* p, size_t n) {
            if (failed) return false;
            const uintptr_t a = reinterpret_cast<uintptr_t>(p) / page * page, b = (reinterpret_cast<uintptr_t>(p) + n + page - 1) / page * page;
            if (cudaHostRegister(reinterpret_cast<void*>(a), b - a, 0x08 /* read-only */) != cudaSuccess) {
                (void)cudaGetLastError();
                failed = true;
                return false;
            }
            registered_.push_back(reinterpret_cast<void*>(a));
            return true;
        };
        auto spread = [](int l, double f) { return std::floor((l + 1) * f + 1e-9) != std::floor(l * f + 1e-9); };
        double down_total = 0, gu_total = 0, bytes = 0;
        for (int l = 0; l < c.layers; ++l) {
            down_total += double(experts_[size_t(l)].down_stride) * c.experts;
            gu_total += 2.0 * double(experts_[size_t(l)].f.gu_row) * fl_ * c.experts;
        }
        const double fd = std::min(1.0, budget / down_total);
        for (int l = 0; l < c.layers && !failed; ++l)
            if (spread(l, fd)) {
                const auto& L = experts_[size_t(l)];
                if (reg(stream_src_[size_t(l)].down, L.down_stride * size_t(c.experts))) { down_pin_[size_t(l)] = 1; bytes += double(L.down_stride) * c.experts; }
            }
        // STRATA_DS41_PIN_GATE_UP=1: also this node's gate/up rows per expert. Off by default: ~12k registrations
        // slow every later host copy in HIP (decode after a 4K prompt fell from 6 to 3 tok/s)
        const bool gate_up = std::getenv("STRATA_DS41_PIN_GATE_UP") && std::string(std::getenv("STRATA_DS41_PIN_GATE_UP")) == "1";
        const double fg = gate_up ? std::min(1.0, std::max(0.0, budget - bytes) / gu_total) : 0.0;
        for (int l = 0; l < c.layers && !failed && fg > 0; ++l)
            if (spread(l, fg)) {
                const auto& L = experts_[size_t(l)];
                const size_t gl = L.f.gu_row * size_t(fl_);
                bool all = true;
                for (int e = 0; e < c.experts && all; ++e)
                    all = reg(stream_src_[size_t(l)].gate + size_t(e) * L.gate_stride, gl) && reg(stream_src_[size_t(l)].up + size_t(e) * L.gate_stride, gl);
                if (all) { gu_pin_[size_t(l)] = 1; bytes += 2.0 * double(gl) * c.experts; }
            }
        for (int l = 0; l < c.layers; ++l) pinned_layer_[size_t(l)] = gu_pin_[size_t(l)] && down_pin_[size_t(l)];
        std::cerr << "DS41 experts pinned GiB=" << bytes / double(1ull << 30) << " down_layers="
                  << std::count(down_pin_.begin(), down_pin_.end(), 1) << " gate_up_layers=" << std::count(gu_pin_.begin(), gu_pin_.end(), 1)
                  << "/" << c.layers << (failed ? " (registration failed; the rest is copied)" : "") << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }

    // The stager copies the unpinned layers' items, in order, into the pinned host slots (STRATA_DS41_STAGE_HOST_SLOTS,
    // 6) ahead of the producer, which only issues DMAs: pinned items straight from the mapping, the others from their
    // host slot. So the host copies and the pinned items' DMAs overlap.
    void stage() {
        const auto& c = cfg_;
        for (int u = 0; u < int(unpinned_.size()); ++u) {
            const StreamItem& it = items_[size_t(unpinned_[size_t(u)])];
            const int hs = u % hslots_;
            if (u >= hslots_) {   // the slot's previous item uploaded
                std::unique_lock<std::mutex> lock(sm_);
                scv_.wait(lock, [&] { return stop_stream_ || dma_issued_ >= u - hslots_ + 1; });
                if (stop_stream_) return;
                lock.unlock();
                cudaEventSynchronize(ev_dma_[size_t(hs)]);
            }
            const auto c0 = std::chrono::steady_clock::now();
            const auto& L = experts_[size_t(it.layer)];
            const Src& src = stream_src_[size_t(it.layer)];
            const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
            uint8_t* host = hring_[size_t(hs)];
            std::vector<std::thread> workers;
            const int nt = std::max(1, std::min(stage_threads_, it.n));
            for (int t = 0; t < nt; ++t)
                workers.emplace_back([&, t] {
                    unpin_helper();
                    const size_t gl = L.f.gu_row * size_t(fl_);   // this node's gate/up rows (all of them alone)
                    const bool gu = !gu_pin_[size_t(it.layer)], dn = !down_pin_[size_t(it.layer)];   // the parts not DMAed from the mapping
                    // STRATA_DS41_STAGE_SKIP=1 (a timing probe): the host slots are not filled, so a node whose shards hold holes (the reversed layout)
                    // never reads them (a shared mapping materializes a hole's pages: tmpfs fills up); the DMAs and the MMQ still run on whatever the slots hold
                    static const bool skip = strata_env_is_one("STRATA_DS41_STAGE_SKIP");
                    for (int j = t; j < it.n && !skip; j += nt) {
                        const int e = order_[size_t(it.layer)][size_t(it.first + j)];
                        if (ost_comm_ && !own_.empty() && own_[size_t(it.layer) * c.experts + size_t(e)]) continue;   // arrives from the worker (own_fetch)
                        if (gu) {
                            copy_nt(host + size_t(j) * it.stride, src.gate + size_t(e) * g, gl);
                            copy_nt(host + size_t(j) * it.stride + g, src.up + size_t(e) * g, gl);
                        }
                        if (dn) copy_nt(host + size_t(j) * it.stride + 2 * g, src.down + size_t(e) * d, d);
                    }
                });
            for (auto& w : workers) w.join();
            prof_[7] += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
            {
                std::lock_guard<std::mutex> lock(sm_);
                staged_ = u + 1;
                if (stop_stream_) return;
            }
            scv_.notify_all();
        }
    }

    void produce() {
        const auto& c = cfg_;
        int u = 0;   // unpinned items so far
        for (int idx = 0; idx < int(items_.size()); ++idx) {
            const StreamItem& it = items_[size_t(idx)];
            const int ds = idx % dslots_;
            const bool pinned = pinned_layer_[size_t(it.layer)];
            {
                std::unique_lock<std::mutex> lock(sm_);
                scv_.wait(lock, [&] {   // staged (unpinned), and the VRAM slot's previous item's compute enqueued
                    return stop_stream_ || ((pinned || (staged_ >= u + 1 && (!ost_comm_ || own_staged_ >= u + 1))) && (idx < dslots_ || consumed_ >= idx - dslots_ + 1));
                });
                if (stop_stream_) return;
            }
            if (idx >= dslots_) cudaStreamWaitEvent(copy_s_, ev_free_[size_t(ds)], 0);
            cudaEvent_t dma_a = nullptr, dma_b = nullptr;
            if (pf_timing_) {   // the item's copies on the copy stream, for the pass's DMA busy time
                dma_a = dma_event();
                ck(cudaEventRecord(dma_a, copy_s_), "dma start");
            }
            uint8_t* dst = static_cast<uint8_t*>(dring_[size_t(ds)].p);
            double item_dma_bytes = double(it.n) * double(it.stride);
            if (!pinned) {   // the staged parts: the whole slot alone; on two nodes this node's gate/up rows and an unpinned down
                const int hs = u % hslots_;
                const auto& L = experts_[size_t(it.layer)];
                const size_t g = L.f.gu_row * size_t(c.ff), gl = L.f.gu_row * size_t(fl_), d = L.f.d_row * size_t(c.hidden);
                const uint8_t* h = hring_[size_t(hs)];
                if (gl == g && !gu_pin_[size_t(it.layer)] && !down_pin_[size_t(it.layer)]) {
                    cudaMemcpyAsync(dst, h, size_t(it.n) * it.stride, cudaMemcpyHostToDevice, copy_s_);
                } else {
                    if (!gu_pin_[size_t(it.layer)]) {
                        cudaMemcpy2DAsync(dst, it.stride, h, it.stride, gl, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                        cudaMemcpy2DAsync(dst + g, it.stride, h + g, it.stride, gl, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                    }
                    if (!down_pin_[size_t(it.layer)])
                        cudaMemcpy2DAsync(dst + 2 * g, it.stride, h + 2 * g, it.stride, d, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                }
                cudaEventRecord(ev_dma_[size_t(hs)], copy_s_);
                ++u;
            }
            if (expert_debug_ == 2) {   // EXPERT_DEBUG=2: the products timed alone (stale slots)
            } else if (!packed_.empty()) {   // the packed pinned copy: one DMA
                cudaMemcpyAsync(dst, packed_[size_t(it.layer)] + size_t(it.first) * it.stride, size_t(it.n) * it.stride, cudaMemcpyHostToDevice, copy_s_);
            } else {   // zero copy: strided DMAs from the pinned mapping into the slot
                const auto& L = experts_[size_t(it.layer)];
                const Src& src = stream_src_[size_t(it.layer)];
                const size_t g = L.f.gu_row * size_t(c.ff), gl = L.f.gu_row * size_t(fl_), d = L.f.d_row * size_t(c.hidden);
                if (pinned && hybrid_m() >= 0 && gu_pin_[size_t(it.layer)] && down_pin_[size_t(it.layer)] && gl == g) {
                    {   // PREFILL_HYBRID: the layer's routing first, then runs of the experts it needs here
                        std::unique_lock<std::mutex> lock(sm_);
                        scv_.wait(lock, [&] { return stop_stream_ || need_ready_[size_t(it.layer)]; });
                        if (stop_stream_) return;
                    }
                    const auto& need = need_[size_t(it.layer)];
                    item_dma_bytes = 0;
                    for (int j0 = 0; j0 < it.n;) {
                        if (!need[size_t(it.first + j0)]) { ++j0; continue; }
                        int j1 = j0;
                        while (j1 < it.n && need[size_t(it.first + j1)]) ++j1;
                        const size_t e0 = size_t(it.first + j0), n = size_t(j1 - j0);
                        item_dma_bytes += double(n) * double(2 * g + d);
                        uint8_t* d0 = dst + size_t(j0) * it.stride;
                        cudaMemcpy2DAsync(d0, it.stride, src.gate + e0 * g, g, g, n, cudaMemcpyHostToDevice, copy_s_);
                        cudaMemcpy2DAsync(d0 + g, it.stride, src.up + e0 * g, g, g, n, cudaMemcpyHostToDevice, copy_s_);
                        cudaMemcpy2DAsync(d0 + 2 * g, it.stride, src.down + e0 * d, d, d, n, cudaMemcpyHostToDevice, copy_s_);
                        j0 = j1;
                    }
                } else if (gu_pin_[size_t(it.layer)] && gl == g) {
                    cudaMemcpy2DAsync(dst, it.stride, src.gate + size_t(it.first) * g, g, gl, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                    cudaMemcpy2DAsync(dst + g, it.stride, src.up + size_t(it.first) * g, g, gl, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                    if (down_pin_[size_t(it.layer)])
                        cudaMemcpy2DAsync(dst + 2 * g, it.stride, src.down + size_t(it.first) * d, d, d, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                } else if (gu_pin_[size_t(it.layer)]) {   // two nodes: each expert's rows are their own registration
                    for (int j = 0; j < it.n; ++j) {
                        const size_t e = size_t(it.first + j);
                        ck(cudaMemcpyAsync(dst + size_t(j) * it.stride, src.gate + e * g, gl, cudaMemcpyHostToDevice, copy_s_), "pinned gate");
                        ck(cudaMemcpyAsync(dst + size_t(j) * it.stride + g, src.up + e * g, gl, cudaMemcpyHostToDevice, copy_s_), "pinned up");
                    }
                }
                if (down_pin_[size_t(it.layer)] && !(gu_pin_[size_t(it.layer)] && gl == g))   // (the branches above copied it)
                    cudaMemcpy2DAsync(dst + 2 * g, it.stride, src.down + size_t(it.first) * d, d, d, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
            }
            if (expert_debug_ != 2 && remote_ && !remote_gpu_) {   // the worker's rows: gate/up rows [fl_, ff) and each down row's columns from fl_ on
                {
                    std::unique_lock<std::mutex> lock(sm_);
                    scv_.wait(lock, [&] { return stop_stream_ || fetched_ >= idx + 1; });
                    if (stop_stream_) return;
                }
                const auto& rg = remote_->geo[size_t(it.layer)];
                const auto& L = experts_[size_t(it.layer)];
                const size_t g = L.f.gu_row * size_t(c.ff), gl = L.f.gu_row * size_t(fl_), rows = rg.rows * rg.gu_row;
                const uint8_t* rb = rbuf_[size_t(idx % int(rbuf_.size()))];
                // one upload of the item's blobs to VRAM scratch, then device copies into the slot's layout (the down
                // columns are ~100-byte pieces of every row: too narrow for the copy engine)
                uint8_t* sc = static_cast<uint8_t*>(rscratch_.p);
                ck(cudaMemcpyAsync(sc, rb, size_t(it.n) * rg.stride, cudaMemcpyHostToDevice, copy_s_), "fetched upload");
                ck(cudaEventRecord(ev_rdma_[size_t(idx % int(rbuf_.size()))], copy_s_), "fetched event");
                ck(cudaMemcpy2DAsync(dst + gl, it.stride, sc, rg.stride, rows, size_t(it.n), cudaMemcpyDeviceToDevice, copy_s_), "fetched gate");
                ck(cudaMemcpy2DAsync(dst + g + gl, it.stride, sc + rows, rg.stride, rows, size_t(it.n), cudaMemcpyDeviceToDevice, copy_s_), "fetched up");
                k::ds41_copy_rows(dst + 2 * g + rg.d_keep, L.f.d_row, it.stride, sc + 2 * rows, rg.d_rest, rg.stride, int(rg.d_rest), c.hidden,
                                  it.n, copy_s_);
            }
            cudaEventRecord(ev_ready_[size_t(ds)], copy_s_);
            if (pf_timing_) {
                dma_b = dma_event();
                ck(cudaEventRecord(dma_b, copy_s_), "dma end");
                dma_ev_.push_back({dma_a, dma_b});
                dma_bytes_ += item_dma_bytes;
            }
            {
                std::lock_guard<std::mutex> lock(sm_);
                produced_ = idx + 1;
                dma_issued_ = u;
                rdma_issued_ = idx + 1;
            }
            scv_.notify_all();
        }
    }

    // The worker's rows of each item, in order, into registered pinned buffers (rbuf_), one fetch ahead per buffer.
    void fetch() {
        namespace net = strata::net;
        const int R = int(rbuf_.size());
        for (int idx = 0; idx < int(items_.size()); ++idx) {
            const StreamItem& it = items_[size_t(idx)];
            const int k = idx % R;
            if (idx >= R) {   // the buffer's previous item uploaded
                std::unique_lock<std::mutex> lock(sm_);
                scv_.wait(lock, [&] { return stop_stream_ || rdma_issued_ >= idx - R + 1; });
                if (stop_stream_) return;
                lock.unlock();
                cudaEventSynchronize(ev_rdma_[size_t(k)]);
            }
            const auto f0 = std::chrono::steady_clock::now();
            const auto& rg = remote_->geo[size_t(it.layer)];
            const net::TpRequest ask{it.layer, it.first, -it.n, 0};
            ucomm_req_t* got = nullptr;
            Remote::ok(ucomm_irecv(remote_->comm, 1, net::kTagFetch, rbuf_[size_t(k)], size_t(it.n) * rg.stride, &got), "fetch");
            Remote::ok(ucomm_send(remote_->comm, 1, net::kTagRequest, &ask, sizeof ask), "fetch request");
            Remote::ok(ucomm_wait(got, nullptr), "fetch");
            fetch_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
            {
                std::lock_guard<std::mutex> lock(sm_);
                fetched_ = idx + 1;
                if (stop_stream_) return;
            }
            scv_.notify_all();
        }
    }

    // MMQ over experts resident on this GPU (res0 area of layer l, positions [j0, j1) of it), rows from the pass's tables
    // STRATA_DS41_REBALANCE=N (with STRATA_DS41_RAM_PINNED): after the prompt, up to N swaps, per layer the most-routed
    // experts in RAM with the least-routed resident ones on either GPU (the prompt's routes, the profile's order breaking
    // ties): the GPU expert goes to the host, the RAM one into its slot, the first into the second's packed position
    void rebalance(int cap = -1) {   // cap >= 0: at most that many swaps (the decode-time rebalance), else STRATA_DS41_REBALANCE
        const auto& c = cfg_;
        static const int env_budget = std::getenv("STRATA_DS41_REBALANCE") ? std::atoi(std::getenv("STRATA_DS41_REBALANCE")) : 0;
        const int budget = cap >= 0 ? std::min(cap, env_budget) : env_budget;
        if (budget <= 0 || packed_.empty() || rt_count_.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        // per layer, the swaps a layer gets: proportional to how unbalanced it is, at most the budget in all
        struct Cand { float gain; int l, ram, gpu; };
        std::vector<Cand> cands;
        for (int l = 0; l < c.layers; ++l) {
            const float* cnt = rt_count_.data() + size_t(l) * c.experts;
            std::vector<int> ram, gpu;
            for (int e = 0; e < c.experts; ++e) {
                const char w0 = where_[size_t(l)][size_t(e)];
                if (w0 == 3 || w0 == 4) continue;   // grown (GPU 0 / the peer) and in RAM: not swapped
                (w0 ? gpu : ram).push_back(e);
            }
            std::sort(ram.begin(), ram.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
            std::sort(gpu.begin(), gpu.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
            for (size_t j = 0; j < std::min(ram.size(), gpu.size()); ++j) {
                if (cnt[ram[j]] <= cnt[gpu[j]] + 1) break;
                cands.push_back({cnt[ram[j]] - cnt[gpu[j]], l, ram[j], gpu[j]});
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
        if (int(cands.size()) > budget) cands.resize(size_t(budget));
        // the swaps in batches of kBatch: every GPU expert out to a pinned buffer and the RAM expert in over two streams
        // per GPU (both GPUs at once), then the outgoing experts into the RAM positions by threads (a swap had been
        // two blocking copies and a memcpy, ~3 ms)
        constexpr int kBatch = 32;
        uint8_t* tmpp = nullptr;
        const int nb = int(std::min<size_t>(cands.size(), kBatch));
        ck(cudaHostAlloc(reinterpret_cast<void**>(&tmpp), std::max(nb, 1) * max_stride_, cudaHostAllocPortable), "rebalance buffer");
        struct Free { uint8_t* p; ~Free() { cudaFreeHost(p); } } free_tmp{tmpp};
        cudaStream_t st[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        for (int d = 0; d < (peer_ ? 2 : 1); ++d) {
            PeerGpu::Guard g(d == 0 ? 0 : peer_->dev);
            for (auto& x : st[d]) ck(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking), "rebalance stream");
        }
        auto sync_all = [&] {
            for (int d = 0; d < (peer_ ? 2 : 1); ++d) {
                PeerGpu::Guard g(d == 0 ? 0 : peer_->dev);
                for (auto x : st[d]) ck(cudaStreamSynchronize(x), "rebalance copies");
            }
        };
        int n0 = 0, n1 = 0;
        for (size_t b0 = 0; b0 < cands.size(); b0 += kBatch) {
            const size_t b1 = std::min(cands.size(), b0 + kBatch);
            for (size_t k = b0; k < b1; ++k) {
                const Cand& x = cands[k];
                const int l = x.l, A = x.gpu, B = x.ram;
                const size_t stride = stream_stride(l);
                uint8_t* hostB = packed_[size_t(l)] + size_t(ram_pos_[size_t(l)][size_t(B)]) * stride;
                uint8_t* tmp = tmpp + (k - b0) * max_stride_;
                const bool on0 = where_[size_t(l)][size_t(A)] == 1;
                uint8_t* dev = on0 ? slots_[size_t(slot_[size_t(l)][size_t(A)])].p
                                   : peer_->base[size_t(l)] + size_t(peer_->pos[size_t(l)][size_t(A)]) * stride;
                PeerGpu::Guard g(on0 ? 0 : peer_->dev);
                cudaStream_t s1 = st[on0 ? 0 : 1][k & 1];
                ck(cudaMemcpyAsync(tmp, dev, stride, cudaMemcpyDeviceToHost, s1), "rebalance out");
                ck(cudaMemcpyAsync(dev, hostB, stride, cudaMemcpyHostToDevice, s1), "rebalance in");
            }
            sync_all();
            std::vector<std::thread> th;
            const int nth = std::min<int>(8, int(b1 - b0));
            for (int t = 0; t < nth; ++t)
                th.emplace_back([&, t] {
                    for (size_t k = b0 + size_t(t); k < b1; k += size_t(nth)) {
                        const Cand& x = cands[k];
                        const size_t stride = stream_stride(x.l);
                        std::memcpy(packed_[size_t(x.l)] + size_t(ram_pos_[size_t(x.l)][size_t(x.ram)]) * stride, tmpp + (k - b0) * max_stride_, stride);
                    }
                });
            for (auto& t : th) t.join();
            for (size_t k = b0; k < b1; ++k) {
                const Cand& x = cands[k];
                const int l = x.l, A = x.gpu, B = x.ram;
                const int pB = ram_pos_[size_t(l)][size_t(B)];
                if (where_[size_t(l)][size_t(A)] == 1) {   // GPU 0: its slot
                    const int si = slot_[size_t(l)][size_t(A)];
                    slot_[size_t(l)][size_t(B)] = si;
                    slot_[size_t(l)][size_t(A)] = -1;
                    slots_[size_t(si)].owner = l * c.experts + B;
                    auto& ord = order_[size_t(l)];
                    for (size_t q = size_t(n_stream_[size_t(l)]); q < ord.size(); ++q) if (ord[q] == A) { ord[q] = B; break; }
                    ++n0;
                } else {   // the peer GPU: its arena position
                    const int q = peer_->pos[size_t(l)][size_t(A)];
                    peer_->pos[size_t(l)][size_t(B)] = q;
                    peer_->pos[size_t(l)][size_t(A)] = -1;
                    ++n1;
                }
                std::swap(where_[size_t(l)][size_t(A)], where_[size_t(l)][size_t(B)]);
                ram_pos_[size_t(l)][size_t(A)] = pB;
                ram_pos_[size_t(l)][size_t(B)] = -1;
                order_[size_t(l)][size_t(pB)] = A;
                if (peer_ && !peer_->spos.empty() && pB >= s0_[size_t(l)]) {   // the peer streams that position
                    peer_->spos[size_t(l)][size_t(A)] = pB - s0_[size_t(l)];
                    peer_->spos[size_t(l)][size_t(B)] = -1;
                }
            }
        }
        for (int d = 0; d < (peer_ ? 2 : 1); ++d) {
            PeerGpu::Guard g(d == 0 ? 0 : peer_->dev);
            for (auto x : st[d]) cudaStreamDestroy(x);
        }
        std::fill(rt_count_.begin(), rt_count_.end(), 0.0f);
        residency_changed();
        std::cerr << "DS41 rebalance swaps gpu0=" << n0 << " gpu1=" << n1 << " candidates=" << cands.size() << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    }

    // STRATA_DS41_REBALANCE_DECODE=N[,M] (with REBALANCE): rebalance again on the decode routes counted since the
    // prompt's rebalance, after N generated tokens (and every M after; 0 = once); STRATA_DS41_REBALANCE_DECODE_MAX (500)
    // swaps at most. Between passes only (the stream synchronized, no peer job in flight).
    int decoded_ = 0, next_rebalance_ = -1, rebalance_every_ = 0;
public:
    void decoded(int tokens) {
        static const int first = [] {
            const char* v = std::getenv("STRATA_DS41_REBALANCE_DECODE");
            return v ? std::max(0, std::atoi(v)) : 0;
        }();
        if (first <= 0) return;
        if (next_rebalance_ < 0) {
            next_rebalance_ = first;
            if (const char* v = std::getenv("STRATA_DS41_REBALANCE_DECODE"); v && std::strchr(v, ',')) rebalance_every_ = std::max(0, std::atoi(std::strchr(v, ',') + 1));
        }
        decoded_ += tokens;
        if (decoded_ < next_rebalance_) return;
        static const int cap = std::getenv("STRATA_DS41_REBALANCE_DECODE_MAX") ? std::atoi(std::getenv("STRATA_DS41_REBALANCE_DECODE_MAX")) : 500;
        ck(cudaStreamSynchronize(stream_), "decode rebalance");
        if (peer_) { PeerGpu::Guard g(peer_->dev); ck(cudaStreamSynchronize(peer_->stream), "decode rebalance"); }
        rebalance(cap);
        next_rebalance_ = rebalance_every_ > 0 ? decoded_ + rebalance_every_ : INT32_MAX;
    }
private:

    // STRATA_DS41_GROW=1 (with STRATA_DS41_DECODE_GROUPED): after the prompt, the chunk buffers shrink to 64 tokens and
    // the VRAM they free (less STRATA_DS41_GPU_RESERVE_MIB) takes copies of the prompt's most-routed RAM experts
    // (where_ 3: computed on GPU 0 in decode, still streamed from RAM in prompt passes)
    Dev grow_arena_;
    bool grown_ = false;
    // the peer's part of REFILL: its free VRAM (less GPU_RESERVE_MIB) takes copies of the most-routed RAM experts left
    bool peer_grown_ = false;
    void grow_peer() {
        const auto& c = cfg_;
        if (!peer_ || peer_grown_ || packed_.empty() || rt_count_.empty()) return;
        peer_grown_ = true;
        size_t keep = std::max<size_t>(size_t(std::getenv("STRATA_DS41_GPU_RESERVE_MIB") ? std::atoi(std::getenv("STRATA_DS41_GPU_RESERVE_MIB")) : 1024) << 20, vram_reserve + (128 << 20));
#if defined(__HIP_PLATFORM_AMD__)
        keep = std::max(keep, physical_vram_keep());
#endif
        const size_t f = peer_->available_bytes();
        if (f <= keep + max_stride_) return;
        const size_t n = (f - keep) / max_stride_;
        std::vector<std::pair<float, int>> cand;
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (where_[size_t(l)][size_t(e)] == 0 && rt_count_[size_t(l) * c.experts + size_t(e)] > 0)
                    cand.push_back({-rt_count_[size_t(l) * c.experts + size_t(e)], l * c.experts + e});
        std::sort(cand.begin(), cand.end());
        if (cand.size() > n) cand.resize(n);
        if (cand.empty()) return;
        {
            PeerGpu::Guard g(peer_->dev);
            peer_->garena = Dev(cand.size() * max_stride_);
        }
        peer_->gpos.assign(size_t(c.layers), std::vector<int>(size_t(c.experts), -1));
        peer_->gbase.clear();
        for (size_t j = 0; j < cand.size(); ++j) {
            const int l = cand[j].second / c.experts, e = cand[j].second % c.experts;
            const size_t stride = stream_stride(l);
            uint8_t* dst = peer_->garena.b(j * max_stride_);
            {
                PeerGpu::Guard g(peer_->dev);
                ck(cudaMemcpy(dst, packed_[size_t(l)] + size_t(ram_pos_[size_t(l)][size_t(e)]) * stride, stride, cudaMemcpyHostToDevice), "grow peer");
            }
            peer_->gpos[size_t(l)][size_t(e)] = int(j);
            peer_->gbase.push_back(dst);
            where_[size_t(l)][size_t(e)] = 4;
        }
        residency_changed();
        std::cerr << "DS41 grow_peer experts=" << cand.size() << " GiB=" << cand.size() * max_stride_ / double(1ull << 30) << "\n";
    }

    void grow() {
        const auto& c = cfg_;
        if (grown_ || packed_.empty() || rt_count_.empty()) return;
        grown_ = true;
        if (chunk_ > 128) resize(128);   // (128: OUT_REGROUP needs out_groups x T columns for verify passes of up to 8)
        ck(cudaStreamSynchronize(stream_), "grow");
        size_t f = 0, t = 0;
        ck(cudaMemGetInfo(&f, &t), "memory info");
        const size_t keep = std::max<size_t>(size_t(std::getenv("STRATA_DS41_GPU_RESERVE_MIB") ? std::atoi(std::getenv("STRATA_DS41_GPU_RESERVE_MIB")) : 1024) << 20, vram_reserve + (128 << 20));
        if (f <= keep + max_stride_) return;
        const size_t n = (f - keep) / max_stride_;
        std::vector<std::pair<float, int>> cand;   // (-routes, l * experts + e)
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e)
                if (where_[size_t(l)][size_t(e)] == 0 && rt_count_[size_t(l) * c.experts + size_t(e)] > 0)
                    cand.push_back({-rt_count_[size_t(l) * c.experts + size_t(e)], l * c.experts + e});
        std::sort(cand.begin(), cand.end());
        if (cand.size() > n) cand.resize(n);
        if (cand.empty()) return;
        grow_arena_ = Dev(cand.size() * max_stride_);
        for (size_t j = 0; j < cand.size(); ++j) {
            const int l = cand[j].second / c.experts, e = cand[j].second % c.experts;
            const size_t stride = stream_stride(l);
            uint8_t* dst = grow_arena_.b(j * max_stride_);
            ck(cudaMemcpyAsync(dst, packed_[size_t(l)] + size_t(ram_pos_[size_t(l)][size_t(e)]) * stride, stride, cudaMemcpyHostToDevice, stream_), "grow");
            slot_[size_t(l)][size_t(e)] = int(slots_.size());
            slots_.push_back({dst, stride, l * c.experts + e, false, nullptr});
            where_[size_t(l)][size_t(e)] = 3;
        }
        ck(cudaStreamSynchronize(stream_), "grow");
        residency_changed();
        std::cerr << "DS41 grow experts=" << cand.size() << " GiB=" << cand.size() * max_stride_ / double(1ull << 30) << "\n";
    }

    // STRATA_DS41_DECODE_MMQ=1 (with STRATA_DS41_RAM_COPY's residency): a short pass's routes to GPU 0's resident
    // experts as one MMQ product per layer (the resident positions in order, those without tokens as empty row
    // ranges) instead of a GEMV chain per route; adds into g_sum_
    Dev dq_xq_, dq_gu_, dq_hf_, dq_hq_, dq_dout_, dq_idx_, dq_w_, dq_bound_, dq_iota_;
    std::vector<std::vector<int>> dq_qpos_;   // [layer] expert -> its index among GPU0's resident experts (cleared by rebalance / grow)
    void residency_changed() {   // the per-layer expert orders cached by resident_mmq and the peer's prefill
        dq_qpos_.clear();
        if (peer_) peer_->order_of.clear();
    }
    std::vector<std::vector<int>> dq_tok_;
    std::vector<std::vector<float>> dq_wt_;
    std::vector<int> dq_touched_;
    int* dq_host_ = nullptr;
    static constexpr int kDqRows = 512;   // routes of a pass through resident_mmq (T x top_k; batched sessions: 32 x 6)
    bool resident_mmq(int l, int T, int K, const int* ids, const float* wts) {
        const auto& c = cfg_;
        namespace mq = strata::prefill::mmq;
        if (res0_base_.empty() || !mq::built()) return false;
        if (!mmq_) mmq_ = std::make_unique<mq::Context>();   // (a short prompt never made one)
        const auto& L = experts_[size_t(l)];
        if (!mq::fits(L.f.gu_type, 2 * c.ff) || !mq::fits(L.f.d_type, c.hidden) || T * K > kDqRows) return false;
        const auto& ord = order_[size_t(l)];
        const int S = n_stream_[size_t(l)], R = int(ord.size()) - S;
        if (R <= 0) return false;
        if (!dq_host_) {
            dq_xq_ = Dev(mq::q8_bytes(kDqRows, c.hidden));
            dq_gu_ = Dev(size_t(kDqRows) * 2 * c.ff * 4);
            dq_hf_ = Dev(size_t(kDqRows) * c.ff * 4);
            dq_hq_ = Dev(mq::q8_bytes(kDqRows, c.ff));
            dq_dout_ = Dev(size_t(kDqRows) * c.hidden * 4);
            dq_idx_ = Dev(size_t(kDqRows) * 4);
            dq_w_ = Dev(size_t(kDqRows) * 4);
            dq_bound_ = Dev((size_t(c.experts) + 8) * 4);
            dq_iota_ = Dev((size_t(kDqRows) + 128) * 4);
            mq::iota(dq_iota_.i(), kDqRows + 128, stream_);
            dq_host_ = reinterpret_cast<int*>(pinned_keep((size_t(kDqRows) * 2 + size_t(c.experts) + 8) * 4));
        }
        if (dq_qpos_.size() != size_t(c.layers)) dq_qpos_.assign(size_t(c.layers), {});
        auto& qpos = dq_qpos_[size_t(l)];   // per layer, once (the residency of a layer is fixed between rebalances: see rebalance)
        if (qpos.size() != size_t(c.experts)) {
            qpos.assign(size_t(c.experts), -1);
            for (int q = 0; q < R; ++q) qpos[size_t(ord[size_t(S + q)])] = q;
        }
        if (dq_tok_.size() < size_t(R)) { dq_tok_.resize(size_t(R)); dq_wt_.resize(size_t(R)); }
        for (int q : dq_touched_) { dq_tok_[size_t(q)].clear(); dq_wt_[size_t(q)].clear(); }
        dq_touched_.clear();
        auto& tok = dq_tok_;
        auto& wt = dq_wt_;
        int rows = 0, mx = 0;
        for (int i = 0; i < T * K; ++i) {
            const int e = ids[i];
            if (e < 0 || where_[size_t(l)][size_t(e)] != 1) continue;
            const int q = qpos[size_t(e)];
            if (q < 0) continue;
            if (tok[size_t(q)].empty()) dq_touched_.push_back(q);
            tok[size_t(q)].push_back(i / K);
            wt[size_t(q)].push_back(wts[i]);
            ++rows;
        }
        if (!rows) return false;
        int* hidx = dq_host_;
        float* hw = reinterpret_cast<float*>(dq_host_ + kDqRows);
        int* hb = dq_host_ + 2 * kDqRows;
        int at = 0;
        for (int q = 0; q < R; ++q) {
            hb[q] = at;
            for (size_t j = 0; j < tok[size_t(q)].size(); ++j) { hidx[at] = tok[size_t(q)][j]; hw[at] = wt[size_t(q)][j]; ++at; }
            mx = std::max(mx, int(tok[size_t(q)].size()));
        }
        hb[R] = at;
        static const bool compact = std::getenv("STRATA_DS41_MMQ_COMPACT") && std::string(std::getenv("STRATA_DS41_MMQ_COMPACT")) == "1";
        ck(cudaMemcpyAsync(dq_idx_.p, hidx, size_t(rows) * 4, cudaMemcpyHostToDevice, stream_), "dq rows");
        ck(cudaMemcpyAsync(dq_w_.p, hw, size_t(rows) * 4, cudaMemcpyHostToDevice, stream_), "dq weights");
        ck(cudaMemcpyAsync(dq_bound_.p, hb, size_t(R + 1) * 4, cudaMemcpyHostToDevice, stream_), "dq bounds");
        const size_t g = L.f.gu_row * size_t(c.ff), stride = res0_stride_[size_t(l)];
        // the products: one over every resident expert, or (MMQ_COMPACT) one per run of experts with rows. A run's rows
        // are the contiguous block [hb[q0], hb[q1]); its activations are quantized on their own (the MMQ layout is
        // blocked by K over all rows), its bounds relative to it (uploaded after the absolute ones)
        std::vector<std::pair<int, int>> runs;
        if (compact) {
            for (int q = 0; q < R;) {
                if (hb[q + 1] == hb[q]) { ++q; continue; }
                int q1 = q;
                while (q1 < R && hb[q1 + 1] > hb[q1]) ++q1;
                runs.push_back({q, q1});
                q = q1;
            }
            int* rb = hb + R + 1;
            int at2 = 0;
            for (auto [q0, q1] : runs) for (int q = q0; q <= q1; ++q) rb[at2++] = hb[q] - hb[q0];
            if (R + 1 + at2 <= c.experts + 8) ck(cudaMemcpyAsync(dq_bound_.i() + R + 1, rb, size_t(at2) * 4, cudaMemcpyHostToDevice, stream_), "dq run bounds");
            else runs.clear();
        }
        const bool whole = runs.empty();
        if (whole) runs.push_back({0, R});
        auto run_products = [&](bool down) {
            int boff = R + 1;
            for (auto [q0, q1] : runs) {
                const int r0 = whole ? 0 : hb[q0], nrows = whole ? rows : hb[q1] - hb[q0];
                int rmx = mx;
                if (!whole) { rmx = 0; for (int q = q0; q < q1; ++q) rmx = std::max(rmx, hb[q + 1] - hb[q]); }
                mq::Product pr;
                pr.type = down ? L.f.d_type : L.f.gu_type;
                pr.w = res0_base_[size_t(l)] + size_t(q0) * stride + (down ? 2 * g : 0);
                pr.w_rows = down ? c.hidden : 2 * c.ff;
                pr.w_cols = down ? c.ff : c.hidden;
                pr.expert_bytes = stride;
                pr.n = q1 - q0;
                if (down) mq::quantize(dq_hf_.f(size_t(r0) * c.ff), nullptr, dq_hq_.p, L.f.d_type, c.ff, c.ff, nrows, stream_);
                else mq::quantize(xn_.f(), dq_idx_.i() + r0, dq_xq_.p, L.f.gu_type, c.hidden, c.hidden, nrows, stream_);
                pr.xq = down ? dq_hq_.p : dq_xq_.p;
                pr.bounds = whole ? dq_bound_.i() : dq_bound_.i() + boff;
                pr.ids = dq_iota_.i();
                pr.total_rows = nrows;
                pr.max_rows = rmx;
                pr.dst = down ? dq_dout_.f(size_t(r0) * c.hidden) : dq_gu_.f(size_t(r0) * 2 * c.ff);
                pr.ld_dst = down ? c.hidden : 2 * c.ff; pr.dst_rows = nrows;
                mmq_->run(pr, stream_);
                boff += q1 - q0 + 1;
            }
        };
        run_products(false);
        mq::swiglu(dq_gu_.f(), dq_hf_.f(), rows, c.ff, false, stream_, c.clamp[size_t(l)]);
        run_products(true);
        k::ds41_scatter_axpy(dq_dout_.f(), dq_idx_.i(), dq_w_.f(), rows, c.hidden, g_sum_.f(), stream_);
        return true;
    }

    void resident_product(int l, int j0, int j1, int row_first, int rows, int bound0, int max_rows, int) {
        const auto& c = cfg_;
        namespace mq = strata::prefill::mmq;
        const auto& L = experts_[size_t(l)];
        const size_t g = L.f.gu_row * size_t(c.ff), stride = res0_stride_[size_t(l)];
        const uint8_t* stage = res0_base_[size_t(l)] + size_t(j0) * stride;
        const int32_t* rtok = s_idx_.i() + row_first;
        const float* rw = s_w_.f(row_first);
        mq::quantize(xn_.f(), rtok, m_xq_.p, L.f.gu_type, c.hidden, c.hidden, rows, stream_);
        mq::Product gu;
        gu.w = stage; gu.type = L.f.gu_type; gu.w_rows = 2 * c.ff; gu.w_cols = c.hidden; gu.expert_bytes = stride;
        gu.n = j1 - j0; gu.xq = m_xq_.p; gu.bounds = s_bound_.i() + bound0; gu.ids = s_iota_.i(); gu.total_rows = rows;
        gu.max_rows = max_rows; gu.dst = s_gu_.f(); gu.ld_dst = 2 * c.ff; gu.dst_rows = rows;
        mmq_->run(gu, stream_);
        mq::swiglu(s_gu_.f(), s_hf_.f(), rows, c.ff, false, stream_, c.clamp[size_t(l)]);
        mq::quantize(s_hf_.f(), nullptr, m_hq_.p, L.f.d_type, c.ff, c.ff, rows, stream_);
        mq::Product dn = gu;
        dn.w = stage + 2 * g; dn.type = L.f.d_type; dn.w_rows = c.hidden; dn.w_cols = c.ff; dn.xq = m_hq_.p;
        dn.dst = s_dout_.f(); dn.ld_dst = c.hidden;
        mmq_->run(dn, stream_);
        k::ds41_scatter_axpy(s_dout_.f(), rtok, rw, rows, c.hidden, g_sum_.f(), stream_);
    }

    void stream_experts(int l, const std::vector<int>& ids, const std::vector<float>& wts) {
        const auto& c = cfg_;
        namespace mq = strata::prefill::mmq;
        // STRATA_DS41_EXPERT_DEBUG=1: the streamed products skipped (timing the DMA alone); =2: the DMAs skipped (timing
        // the products on stale slots). Both give wrong answers; they only split the expert segment of a prompt pass.
        const int expert_debug = expert_debug_;
        const int T = tokens_, K = c.top_k;
        if (l < 0 || l >= c.layers || T <= 0 || K <= 0 || size_t(T) > SIZE_MAX / size_t(K))
            throw std::runtime_error("ds41: streamed expert input geometry bounds");
        const size_t route_capacity = size_t(T) * size_t(K);
        if (route_capacity > size_t(INT32_MAX) || ids.size() != route_capacity || wts.size() != ids.size() ||
            size_t(l) >= experts_.size() || size_t(l) >= order_.size() ||
            (!where_.empty() && (size_t(l) >= where_.size() || where_[size_t(l)].size() != size_t(c.experts))) ||
            (peer_frac_ > 0 && (size_t(l) >= ram_pos_.size() || ram_pos_[size_t(l)].size() != size_t(c.experts))) ||
            (spec_save_ && spec_used_.size() != size_t(T)) ||
            (!rt_count_.empty() && rt_count_.size() < size_t(c.layers) * size_t(c.experts)) ||
            !s_hidx_ || !s_hw_ || !s_hbound_ || s_idx_.bytes < route_capacity * sizeof(int32_t) ||
            s_w_.bytes < route_capacity * sizeof(float) || s_bound_.bytes < sizeof(int32_t))
            throw std::runtime_error("ds41: streamed expert route or buffer capacity bounds");
        if (rcap_ <= 0 || rcap_ > 8192)
            throw std::runtime_error("ds41: streamed expert row cap bounds");
        const auto& L = experts_[size_t(l)];
        const size_t g = L.f.gu_row * size_t(c.ff);
        if (!use_mmq_ || !mq::fits(L.f.gu_type, 2 * c.ff) || !mq::fits(L.f.d_type, c.hidden))
            throw std::runtime_error("ds41: streamed experts need MMQ for these formats");
        // the routes grouped by expert, in token order; rows of the layer laid out by expert
        std::vector<std::vector<int>> tok(size_t(c.experts));
        std::vector<std::vector<float>> wt(size_t(c.experts));
        for (int i = 0; i < T; ++i)
            for (int k = 0; k < K; ++k) {
                const int e = ids[size_t(i) * K + k];
                if (e < 0 || e >= c.experts) throw std::runtime_error("ds41: routed expert out of bounds");
                if (spec_save_) spec_used_[size_t(i)].push_back(l * c.experts + e);
                else {
                    used_.push_back(l * c.experts + e);
                    if (!rt_count_.empty()) rt_count_[size_t(l) * c.experts + size_t(e)] += 1;
                }
                if (!where_.empty() && where_[size_t(l)][size_t(e)] == 2) continue;   // the peer GPU's
                if (peer_frac_ > 0 && ram_pos_[size_t(l)][size_t(e)] >= s0_[size_t(l)]) continue;   // streamed by the peer
                tok[size_t(e)].push_back(i);
                wt[size_t(e)].push_back(wts[size_t(i) * K + k]);
            }
        // PREFILL_HYBRID: the experts with the fewest tokens to the CPU lane (their GPU rows empty), the others' copies
        // released to the producer; the lane runs while the GPU streams
        const int hm = hybrid_m();
        bool lane_on = false;
        float* host_y = pinned_[1 + (l & 1)];
        if (hm >= 0) {
            static const double hmax = std::getenv("STRATA_DS41_PREFILL_HYBRID_MAX") ? std::atof(std::getenv("STRATA_DS41_PREFILL_HYBRID_MAX")) : 0.35;
            std::vector<int> used_e;
            for (int e = 0; e < c.experts; ++e) if (!tok[size_t(e)].empty()) used_e.push_back(e);
            std::stable_sort(used_e.begin(), used_e.end(), [&](int a, int b) { return tok[size_t(a)].size() < tok[size_t(b)].size(); });
            std::vector<char> cpu(size_t(c.experts), 0);
            const int cap_cpu = int(hmax * double(used_e.size()));
            for (int j = 0; j < int(used_e.size()) && j < cap_cpu && int(tok[size_t(used_e[size_t(j)])].size()) <= hm; ++j)
                cpu[size_t(used_e[size_t(j)])] = 1;
            std::vector<int32_t> cpu_ids(size_t(T) * K, -1);
            for (int e = 0; e < c.experts; ++e)
                if (cpu[size_t(e)])
                    for (int i : tok[size_t(e)]) {
                        for (int k = 0; k < K; ++k) if (ids[size_t(i) * K + k] == e) cpu_ids[size_t(i) * K + k] = e;
                    }
            {
                std::lock_guard<std::mutex> lock(sm_);
                auto& need = need_[size_t(l)];
                double skipped = 0;
                for (int e = 0; e < c.experts; ++e) {
                    need[size_t(e)] = !tok[size_t(e)].empty() && !cpu[size_t(e)];
                    if (!need[size_t(e)]) skipped += double(expert_bytes(l));
                }
                hybrid_skipped_gb_ += skipped / 1e9;
                need_ready_[size_t(l)] = 1;
            }
            scv_.notify_all();
            for (int e = 0; e < c.experts; ++e) if (cpu[size_t(e)]) { tok[size_t(e)].clear(); wt[size_t(e)].clear(); }
            if (std::any_of(cpu_ids.begin(), cpu_ids.end(), [](int32_t x) { return x >= 0; })) {
                hybrid_ids_ = std::move(cpu_ids);
                hybrid_w_ = wts;
                ck(cudaMemcpyAsync(pinned_[0], xn_.p, size_t(T) * c.hidden * 4, cudaMemcpyDeviceToHost, stream_), "hybrid input");
                ck(cudaStreamSynchronize(stream_), "hybrid input");
                hybrid_t0_ = std::chrono::steady_clock::now();
                lane_->begin(experts_[size_t(l)], pinned_[0], T, hybrid_ids_.data(), hybrid_w_.data(), K, host_y);
                lane_on = true;
            }
        }
        // rows laid out by position: order_[l] lists the streamed experts (positions [0, S)), then those resident on
        // this GPU (the experts on the peer GPU are not here)
        const auto& ord = order_[size_t(l)];
        if (ord.size() > size_t(INT_MAX) || ord.size() > size_t(c.experts) || size_t(l + 1) >= item0_.size() ||
            item0_.size() != size_t(c.layers) + 1 || dslots_ <= 0 || dring_.size() < size_t(dslots_) ||
            ev_ready_.size() < size_t(dslots_) || ev_free_.size() < size_t(dslots_))
            throw std::runtime_error("ds41: streamed expert order bounds");
        const int P = int(ord.size()), S = n_stream_[size_t(l)];
        if (P > c.experts || S < 0 || S > P)
            throw std::runtime_error("ds41: streamed expert order bounds");
        std::vector<int> row0(size_t(P) + 1, 0);
        int64_t cumulative_rows = 0;
        for (int q = 0; q < P; ++q) {
            const int e = ord[size_t(q)];
            if (e < 0 || e >= c.experts || tok[size_t(e)].size() != wt[size_t(e)].size() || cumulative_rows < 0 ||
                uint64_t(cumulative_rows) > route_capacity ||
                tok[size_t(e)].size() > route_capacity - size_t(cumulative_rows))
                throw std::runtime_error("ds41: streamed expert row table bounds");
            cumulative_rows += int64_t(tok[size_t(e)].size());
            if (cumulative_rows > INT32_MAX) throw std::runtime_error("ds41: streamed expert row table overflow");
            row0[size_t(q) + 1] = int(cumulative_rows);
            std::copy(tok[size_t(e)].begin(), tok[size_t(e)].end(), s_hidx_ + row0[size_t(q)]);
            std::copy(wt[size_t(e)].begin(), wt[size_t(e)].end(), s_hw_ + row0[size_t(q)]);
            tier_all_ += double(tok[size_t(e)].size());
        }
        // bounds of every product: per item, sub-products within the row cap; relative to the product's first row.
        // item -1: positions [j0, j1) resident on this GPU (one area per layer at res0_stride_)
        struct Product { int item, j0, j1, row_first, rows, bound0, max_rows; };
        std::vector<Product> prods;
        int nb = 0;
        const size_t bound_capacity = s_bound_.bytes / sizeof(int32_t);
        auto append_bound = [&](int32_t value) {
            if (nb < 0 || size_t(nb) >= bound_capacity)
                throw std::runtime_error("ds41: streamed expert bounds buffer capacity");
            s_hbound_[nb++] = value;
        };
        auto group = [&](int item, int base, int count) {   // positions base + [0, count)
            if (base < 0 || count < 0 || base > P || count > P - base || (item < -1) ||
                (item >= 0 && size_t(item) >= items_.size()))
                throw std::runtime_error("ds41: streamed expert product group bounds");
            for (int j0 = 0; j0 < count;) {
                const int e0 = ord[size_t(base + j0)];
                if (e0 < 0 || e0 >= c.experts) throw std::runtime_error("ds41: streamed expert group index bounds");
                const int expert_rows = int(tok[size_t(e0)].size());
                if (expert_rows > rcap_) {
                    for (int offset = 0; offset < expert_rows; offset += rcap_) {
                        const int n = std::min(rcap_, expert_rows - offset);
                        prods.push_back(Product{item, j0, j0 + 1, row0[size_t(base + j0)] + offset, n, nb, n});
                        append_bound(0);
                        append_bound(n);
                    }
                    ++j0;
                    continue;
                }
                int j1 = j0, rows = 0, mx = 0;
                while (j1 < count && (j1 == j0 || rows + int(tok[size_t(ord[size_t(base + j1)])].size()) <= rcap_)) {
                    const int n = int(tok[size_t(ord[size_t(base + j1)])].size());
                    rows += n;
                    mx = std::max(mx, n);
                    ++j1;
                }
                Product pr{item, j0, j1, row0[size_t(base + j0)], rows, nb, mx};
                for (int j = j0; j <= j1; ++j) append_bound(row0[size_t(base + j)] - pr.row_first);
                prods.push_back(pr);
                j0 = j1;
            }
        };
        if (item0_[size_t(l)] < 0 || item0_[size_t(l) + 1] < item0_[size_t(l)] ||
            size_t(item0_[size_t(l) + 1]) > items_.size())
            throw std::runtime_error("ds41: streamed expert item range bounds");
        for (int idx = item0_[size_t(l)]; idx < item0_[size_t(l) + 1]; ++idx) group(idx, items_[size_t(idx)].first, items_[size_t(idx)].n);
        if (P > S) group(-1, S, P - S);
        const int rows = row0[size_t(P)];
        if (rows < 0 || size_t(rows) > route_capacity || size_t(rows) * sizeof(int32_t) > s_idx_.bytes ||
            size_t(rows) * sizeof(float) > s_w_.bytes || !s_hidx_ || !s_hw_ || !s_hbound_ ||
            size_t(nb) > bound_capacity)
            throw std::runtime_error("ds41: streamed expert route or bounds buffer capacity");
        for (const Product& pr : prods) {
            const int nbound = pr.j1 - pr.j0 + 1;
            if (pr.j0 < 0 || pr.j1 <= pr.j0 || nbound < 2 || pr.bound0 < 0 || pr.rows < 0 || pr.rows > rcap_ ||
                pr.max_rows < 0 || pr.max_rows > pr.rows || pr.row_first < 0 || pr.row_first > rows ||
                pr.rows > rows - pr.row_first || pr.bound0 > nb || nbound > nb - pr.bound0)
                throw std::runtime_error("ds41: streamed expert product metadata bounds");
            const int32_t* b = s_hbound_ + pr.bound0;
            if (b[0] != 0 || b[nbound - 1] != pr.rows)
                throw std::runtime_error("ds41: streamed expert product range endpoints");
            int32_t largest = 0;
            for (int j = 1; j < nbound; ++j) {
                if (b[j] < b[j - 1] || b[j] > pr.rows)
                    throw std::runtime_error("ds41: streamed expert product range bounds");
                largest = std::max(largest, b[j] - b[j - 1]);
            }
            if (largest > pr.max_rows)
                throw std::runtime_error("ds41: streamed expert product row cap bounds");
            if (pr.item >= 0) {
                const StreamItem& it = items_[size_t(pr.item)];
                const size_t stage_slot = size_t(pr.item) % size_t(dslots_);
                if (stage_slot >= dring_.size() || it.layer != l || pr.j1 > it.n ||
                    it.stride < 2 * g + L.f.d_row * size_t(c.hidden) ||
                    size_t(it.n) > SIZE_MAX / it.stride || size_t(it.n) * it.stride > dring_[stage_slot].bytes)
                    throw std::runtime_error("ds41: streamed expert stage span bounds");
            } else if (pr.j1 > P - S) {
                throw std::runtime_error("ds41: resident expert product span bounds");
            }
        }
        ck(cudaMemcpyAsync(s_idx_.p, s_hidx_, size_t(rows) * 4, cudaMemcpyHostToDevice, stream_), "expert tokens");
        ck(cudaMemcpyAsync(s_w_.p, s_hw_, size_t(rows) * 4, cudaMemcpyHostToDevice, stream_), "expert weights");
        ck(cudaMemcpyAsync(s_bound_.p, s_hbound_, size_t(nb) * 4, cudaMemcpyHostToDevice, stream_), "expert bounds");
        ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * c.hidden * 4, stream_), "expert sum");
        int waited = -1;
        auto finish = [&](int idx) {   // the item's slot is free once its products ran
            ck(cudaEventRecord(ev_free_[size_t(idx % dslots_)], stream_), "slot free");
            { std::lock_guard<std::mutex> lock(sm_); consumed_ = idx + 1; }
            scv_.notify_all();
        };
        for (size_t q = 0; q < prods.size(); ++q) {
            const Product& pr = prods[q];
            if (pr.item < 0) {   // resident on this GPU: no wait
                if (waited >= 0) { finish(waited); waited = -1; }
                if (pr.rows == 0) continue;
                resident_product(l, pr.j0 + (S - S), pr.j1, pr.row_first, pr.rows, pr.bound0, pr.max_rows, S);
                continue;
            }
            const StreamItem& it = items_[size_t(pr.item)];
            if (pr.item != waited) {
                if (waited >= 0) finish(waited);
                const auto w0 = std::chrono::steady_clock::now();
                {
                    std::unique_lock<std::mutex> lock(sm_);
                    scv_.wait(lock, [&] { return produced_ >= pr.item + 1; });
                }
                prof_[1] += 0;   // (waits are part of the expert time)
                stream_wait_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                cudaEvent_t ev_a = nullptr;
                if (pf_timing_) {   // the compute stream's wait for this item's copy
                    ev_a = dma_event();
                    ck(cudaEventRecord(ev_a, stream_), "wait start");
                }
                ck(cudaStreamWaitEvent(stream_, ev_ready_[size_t(pr.item % dslots_)], 0), "slot ready");
                if (pf_timing_) {
                    cudaEvent_t ev_b = dma_event();
                    ck(cudaEventRecord(ev_b, stream_), "wait end");
                    wait_ev_.push_back({ev_a, ev_b});
                }
                waited = pr.item;
            }
            if (pr.rows == 0) continue;
            const uint8_t* stage = static_cast<const uint8_t*>(dring_[size_t(pr.item % dslots_)].p) + size_t(pr.j0) * it.stride;
            const int32_t* rtok = s_idx_.i() + pr.row_first;
            const float* rw = s_w_.f(pr.row_first);
            if (expert_debug == 1) continue;   // EXPERT_DEBUG=1: the DMA's wait kept, the products skipped
            mq::quantize(xn_.f(), rtok, m_xq_.p, L.f.gu_type, c.hidden, c.hidden, pr.rows, stream_);
            if (remote_ && remote_gpu_) {   // this node's rows: gate and up rows [0, fl_), h zero-padded to the full down rows
                const int F = fl_;
                mq::Product ga;
                ga.w = stage; ga.type = L.f.gu_type; ga.w_rows = F; ga.w_cols = c.hidden; ga.expert_bytes = it.stride;
                ga.n = pr.j1 - pr.j0; ga.xq = m_xq_.p; ga.bounds = s_bound_.i() + pr.bound0; ga.ids = s_iota_.i(); ga.total_rows = pr.rows;
                ga.max_rows = pr.max_rows; ga.dst = s_gu_.f(); ga.ld_dst = 2 * F; ga.dst_rows = pr.rows;
                mmq_->run(ga, stream_);
                mq::Product ua = ga;
                ua.w = stage + g; ua.dst = s_gu_.f() + F;
                mmq_->run(ua, stream_);
                mq::swiglu(s_gu_.f(), s_dout_.f(), pr.rows, F, false, stream_, c.clamp[size_t(l)]);
                ck(cudaMemcpy2DAsync(s_hf_.p, size_t(c.ff) * 4, s_dout_.p, size_t(F) * 4, size_t(F) * 4, size_t(pr.rows), cudaMemcpyDeviceToDevice, stream_), "padded h");
                mq::quantize(s_hf_.f(), nullptr, m_hq_.p, L.f.d_type, c.ff, c.ff, pr.rows, stream_);
                mq::Product dn = ga;
                dn.w = stage + 2 * g; dn.type = L.f.d_type; dn.w_rows = c.hidden; dn.w_cols = c.ff; dn.xq = m_hq_.p;
                dn.dst = s_dout_.f(); dn.ld_dst = c.hidden;
                mmq_->run(dn, stream_);
                k::ds41_scatter_axpy(s_dout_.f(), rtok, rw, pr.rows, c.hidden, g_sum_.f(), stream_);
                continue;
            }
            mq::Product gu;
            gu.w = stage; gu.type = L.f.gu_type; gu.w_rows = 2 * c.ff; gu.w_cols = c.hidden; gu.expert_bytes = it.stride;
            gu.n = pr.j1 - pr.j0; gu.xq = m_xq_.p; gu.bounds = s_bound_.i() + pr.bound0; gu.ids = s_iota_.i(); gu.total_rows = pr.rows;
            gu.max_rows = pr.max_rows; gu.dst = s_gu_.f(); gu.ld_dst = 2 * c.ff; gu.dst_rows = pr.rows;
            mmq_->run(gu, stream_);
            mq::swiglu(s_gu_.f(), s_hf_.f(), pr.rows, c.ff, false, stream_, c.clamp[size_t(l)]);
            mq::quantize(s_hf_.f(), nullptr, m_hq_.p, L.f.d_type, c.ff, c.ff, pr.rows, stream_);
            mq::Product dn = gu;
            dn.w = stage + 2 * g; dn.type = L.f.d_type; dn.w_rows = c.hidden; dn.w_cols = c.ff; dn.xq = m_hq_.p;
            dn.dst = s_dout_.f(); dn.ld_dst = c.hidden;
            mmq_->run(dn, stream_);
            k::ds41_scatter_axpy(s_dout_.f(), rtok, rw, pr.rows, c.hidden, g_sum_.f(), stream_);
        }
        if (waited >= 0) finish(waited);
        if (lane_on) {   // PREFILL_HYBRID: the CPU's experts join the routed sum
            lane_->finish();
            hybrid_cpu_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - hybrid_t0_).count();
            if (hybrid_dev_.bytes < size_t(T) * c.hidden * 4) hybrid_dev_ = Dev(size_t(T) * c.hidden * 4);   // (s_dout_ holds rcap_ rows only)
            ck(cudaMemcpyAsync(hybrid_dev_.p, host_y, size_t(T) * c.hidden * 4, cudaMemcpyHostToDevice, stream_), "hybrid sum");
            k::ds41_add(g_sum_.f(), hybrid_dev_.f(), g_sum_.f(), int(size_t(T) * c.hidden), stream_);
        }
        // the row tables are rewritten by the next layer only after its router sync, which follows these uploads
    }

    void shared_expert(int l) {   // y_ = the shared expert of xn_ (q8x_ holds it)
        const auto& c = cfg_;
        const int T = tokens_;
        const std::string p = "blk." + std::to_string(l) + ".";
        const int step = int(std::min<size_t>(sh_rows_, shexp_tile_ > 0 ? size_t(shexp_tile_) : size_t(T)));
        for (int t0 = 0; t0 < T; t0 += step) {   // SHEXP_TILE: n tokens at a time
            const int n = std::min(step, T - t0);
            const float* x = xn_.f(size_t(t0) * c.hidden);
            mat(p + "ffn_gate_shexp.weight", x, sh_gate_.f(), n);
            mat(p + "ffn_up_shexp.weight", x, sh_up_.f(), n);
            k::glm_swiglu(sh_gate_.f(), sh_up_.f(), sh_h_.f(), n * c.ff, c.shared_clamp[size_t(l)], stream_);
            mat(p + "ffn_down_shexp.weight", sh_h_.f(), y_.f(size_t(t0) * c.hidden), n);
        }
    }
    // GPU_ROUTE: the tier's slot pointers to the device when they changed (stream-ordered; the staging is reused only
    // after its last upload finished)
    void sync_slot_table() {
        const auto& c = cfg_;
        const size_t n = size_t(c.layers) * c.experts;
        if (!slot_up_) {
            slot_up_ = reinterpret_cast<unsigned long long*>(pinned_keep(n * 8));
            slot_dev_ = Dev(n * 8);
            ck(cudaEventCreateWithFlags(&slot_ev_, cudaEventDisableTiming), "slot event");
            std::vector<float> b(n);
            for (int l = 0; l < c.layers; ++l) std::copy(bias_[size_t(l)].begin(), bias_[size_t(l)].end(), b.begin() + size_t(l) * c.experts);
            bias_dev_ = Dev(n * 4);
            ck(cudaMemcpy(bias_dev_.p, b.data(), n * 4, cudaMemcpyHostToDevice), "router bias");
            slot_last_.assign(n, ~0ull);
        }
        std::vector<unsigned long long> now(n, 0ull);
        for (int l = 0; l < c.layers; ++l)
            for (int e = 0; e < c.experts; ++e) {
                const int s = slot_[size_t(l)][size_t(e)];
                if (s >= 0) now[size_t(l) * c.experts + size_t(e)] = reinterpret_cast<unsigned long long>(slots_[size_t(s)].p);
            }
        if (now == slot_last_) return;
        ck(cudaEventSynchronize(slot_ev_), "slot staging");
        std::copy(now.begin(), now.end(), slot_up_);
        ck(cudaMemcpyAsync(slot_dev_.p, slot_up_, n * 8, cudaMemcpyHostToDevice, stream_), "slot table");
        ck(cudaEventRecord(slot_ev_, stream_), "slot event");
        slot_last_ = std::move(now);
    }

    void ffn(int l) {
        const auto& c = cfg_;
        const int T = tokens_, K = c.top_k;
        const size_t H = size_t(c.hidden);
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool hook = T == 1 && observe;
        // router on the host: sqrt(softplus(logits)), top-k on score + bias, weights from the unbiased score
        const auto f0 = std::chrono::steady_clock::now();
        if (fuse_) { quantize_xn(); mat_xn(p + "ffn_gate_inp.weight", logits_.f()); }
        else mat(p + "ffn_gate_inp.weight", xn_.f(), logits_.f(), T);
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(7);
        // STRATA_DS41_GPU_ROUTE=1 (opt-in; decode / verify passes on one GPU with DECODE_GROUPED, no GROW): top-k and the
        // GPU-held experts run on the GPU at once; the host takes the ids / weights afterwards for its lane
        static const bool gpu_route_on = std::getenv("STRATA_DS41_GPU_ROUTE") && std::string(std::getenv("STRATA_DS41_GPU_ROUTE")) == "1";
        // Exact verification uses the same per-expert execution path as its greedy control.
        static const bool grouped_on = !strata_env_is_one("STRATA_DS41_VERIFY_EXACT") && std::getenv("STRATA_DS41_DECODE_GROUPED") && std::string(std::getenv("STRATA_DS41_DECODE_GROUPED")) == "1" &&
                                       k::native_expert_supported(experts_[0].f.gu_type, experts_[0].f.d_type, c.hidden, c.ff);
        static const bool route_grow = std::getenv("STRATA_DS41_GROW") && std::string(std::getenv("STRATA_DS41_GROW")) == "1";
        static const bool route_subst = std::getenv("STRATA_DS41_GPU_SUBST") != nullptr;
        static const int route_topk = std::getenv("STRATA_DS41_CPU_TOPK") ? std::atoi(std::getenv("STRATA_DS41_CPU_TOPK")) : -1;
        static const bool topk_original_weight = strata_env_is_one("STRATA_DS41_TOPK_ORIGINAL_WEIGHT");
        // Fraction of the original normalized route weight, not a renormalized
        // top-k. GPU-held routes retain their original weights.
        static const float cpu_trim = cpu_trim_threshold();
        const bool dev_route = gpu_route_on && grouped_on && !route_grow && !route_subst && T <= 16 && !hook && !peer_ && !remote_ &&
                               !producer_.joinable() && fl_ == c.ff && !slot_.empty() && !slots_.empty();
        std::vector<float> logits(dev_route ? 0 : size_t(T) * c.experts);
        if (!dev_route) ck(cudaMemcpyAsync(logits.data(), logits_.p, logits.size() * 4, cudaMemcpyDeviceToHost, stream_), "router");
        float* host_y = pinned_[1 + (l & 1)];
        if (!producer_.joinable() || (remote_ && remote_gpu_) || peer_)   // the CPU lane's, the worker's or the peer GPU's input
            ck(cudaMemcpyAsync(pinned_[0], xn_.p, size_t(T) * H * 4, cudaMemcpyDeviceToHost, stream_), "expert input");
        const bool shared_first = fuse_ && !hook;   // the shared expert runs under the host's routing: the host waits
        if (dev_route) {
            if (l == 0 || !route_ev_) sync_slot_table();
            const int cap = T * K;
            if (!route_ev_) {
                ck(cudaEventCreateWithFlags(&route_ev_, cudaEventDisableTiming), "route event");
                route_table_ = Dev(size_t(16) * K * 32 + 64);
                route_ids_ = Dev(size_t(16) * K * 4);
                route_w_ = Dev(size_t(16) * K * 4);
                route_host_ = reinterpret_cast<int*>(pinned_keep(size_t(16) * K * 8));
            }
            const bool lossy_topk = route_topk >= 0 && route_topk < K;
            k::ds41_route_table(logits_.f(), bias_dev_.f() + size_t(l) * c.experts, reinterpret_cast<const unsigned long long*>(slot_dev_.p) + size_t(l) * c.experts,
                                T, c.experts, K, c.normalize, c.expert_scale, lossy_topk ? route_topk : -1, cap, route_ids_.i(), route_w_.f(), route_table_.p, stream_, topk_original_weight);
            ck(cudaMemcpyAsync(route_host_, route_ids_.p, size_t(cap) * 4, cudaMemcpyDeviceToHost, stream_), "route ids");
            ck(cudaMemcpyAsync(route_host_ + cap, route_w_.p, size_t(cap) * 4, cudaMemcpyDeviceToHost, stream_), "route weights");
            ck(cudaEventRecord(route_ev_, stream_), "route event");
            ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum");
            const auto& LE = experts_[size_t(l)];
            grouped0_.run_device(LE.f.gu_type, LE.f.d_type, c.hidden, c.ff, c.clamp[size_t(l)], route_table_.b(), cap, xn_.f(), T, g_sum_.f(), stream_);
            if (shared_first) shared_expert(l);
            if (dtiming_) {
                const auto route_wait0 = std::chrono::steady_clock::now();
                ck(cudaEventSynchronize(route_ev_), "route sync");
                route_wait_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - route_wait0).count();
            } else {
                ck(cudaEventSynchronize(route_ev_), "route sync");
            }
        } else {
        if (shared_first) {                        // for the copies (an event), not for it
            ck(cudaEventRecord(router_ev_, stream_), "router event");
            shared_expert(l);
            ck(cudaEventSynchronize(router_ev_), "router sync");
        } else
        ck(cudaStreamSynchronize(stream_), "router sync");
        }
        if (peer_late_pending_) {   // the previous layer's wait for the peer on GPU 0's stream (its events are complete)
            float ms = 0;
            if (cudaEventElapsedTime(&ms, peer_late_ev_[0], peer_late_ev_[1]) == cudaSuccess) peer_late_ms_ += ms;
            else (void)cudaGetLastError();
            peer_late_pending_ = false;
        }
        auto& ids = host_route_ids_;
        auto& wts = host_route_weights_;
        ids.resize(size_t(T) * K);
        wts.resize(size_t(T) * K);
        const auto& bias = bias_[size_t(l)];
        // Lossy, opt-in, decode / verify passes (T <= 16) only, for a CPU lane that is the bottleneck:
        //  STRATA_DS41_GPU_SUBST=r (0..1): a pick held by neither GPU becomes the best-selected GPU-resident expert
        //    outside the picks when that one's affinity (probs, before the bias) is at least r times the pick's (its own
        //    weight)
        //  STRATA_DS41_CPU_TOPK=n (0..top_k-1): picks of rank >= n held by neither GPU are dropped (cut_), and the
        //    weights renormalised over the picks kept
        static const float gpu_subst = std::getenv("STRATA_DS41_GPU_SUBST") ? float(std::atof(std::getenv("STRATA_DS41_GPU_SUBST"))) : 0.0f;
        // STRATA_DS41_GPU_SUBST_RANK (3): only picks of this rank and lower are substituted
        static const int subst_rank = std::getenv("STRATA_DS41_GPU_SUBST_RANK") ? std::atoi(std::getenv("STRATA_DS41_GPU_SUBST_RANK")) : 3;
        static const int cpu_topk = std::getenv("STRATA_DS41_CPU_TOPK") ? std::atoi(std::getenv("STRATA_DS41_CPU_TOPK")) : -1;
        const bool lossy = T <= 16 && !hook && !slot_.empty() && (gpu_subst > 0.0f || cpu_trim > 0 || (cpu_topk >= 0 && cpu_topk < K));
        cut_.assign(lossy && (cpu_topk >= 0 || cpu_trim > 0) ? size_t(T) * K : 0, 0);
        auto held = [&](int e) {
            return slot_[size_t(l)][size_t(e)] >= 0 || (!where_.empty() && (where_[size_t(l)][size_t(e)] == 2 || where_[size_t(l)][size_t(e)] == 4));
        };
        // Lossy, opt-in, passes of up to 16 tokens: an expert that is cheap to run wins the selection within a margin
        // (the GLM decoder's route affinity).  STRATA_DS41_ROUTE_AFFINITY=d: one held on the GPU (the canonical tier or
        // the resident tier); STRATA_DS41_WINDOW_AFFINITY=d: one an earlier token of this pass already picked (its
        // weights are read once for both).  Weights stay the model's (probs of the picks, normalised).
        static const float route_affinity = strata_env_value_double("STRATA_DS41_ROUTE_AFFINITY", 0.0);
        static const float window_affinity = strata_env_value_double("STRATA_DS41_WINDOW_AFFINITY", 0.0);
        if (!(route_affinity >= 0.f && route_affinity <= 1.f && window_affinity >= 0.f && window_affinity <= 1.f))
            throw std::invalid_argument("ds41: route / window affinity must be in [0, 1]");
        const bool affine = T <= 16 && !hook && (route_affinity > 0.f || window_affinity > 0.f);
        auto route = [&](int i0, int i1) {
        std::vector<float> probs(size_t(c.experts)), selection(size_t(c.experts));
        std::vector<uint8_t> in_window(affine && window_affinity > 0.f ? size_t(c.experts) : 0, 0);
        for (int i = i0; i < i1; ++i) {
            const float* lg = logits.data() + size_t(i) * c.experts;
            for (int e = 0; e < c.experts; ++e) {
                probs[size_t(e)] = std::sqrt(softplus(lg[e]));
                selection[size_t(e)] = probs[size_t(e)] + bias[size_t(e)];
            }
            if (affine)
                for (int e = 0; e < c.experts; ++e) {
                    const bool resident = (!canon_off_.empty() && canon_off_[size_t(l) * c.experts + e] >= 0) ||
                                          (!slot_.empty() && slot_[size_t(l)][size_t(e)] >= 0);
                    if (resident) selection[size_t(e)] += route_affinity;
                    else if (!in_window.empty() && in_window[size_t(e)]) selection[size_t(e)] += window_affinity;
                }
            auto pick = top_k(selection.data(), c.experts, K);
            if (affine) {
                const auto plain = [&] {
                    std::vector<float> sel(size_t(c.experts));
                    for (int e = 0; e < c.experts; ++e) sel[size_t(e)] = probs[size_t(e)] + bias[size_t(e)];
                    return top_k(sel.data(), c.experts, K);
                }();
                for (int k = 0; k < K; ++k)
                    if (std::find(plain.begin(), plain.end(), pick[size_t(k)]) == plain.end()) ++affinity_swaps_;
                for (int x : pick) if (!in_window.empty()) in_window[size_t(x)] = 1;
            }
            if (lossy && gpu_subst > 0.0f)
                for (int k = subst_rank; k < K; ++k) {
                    const int e = pick[size_t(k)];
                    if (held(e)) continue;
                    int best = -1;
                    for (int x = 0; x < c.experts; ++x)
                        if (held(x) && std::find(pick.begin(), pick.end(), x) == pick.end() && (best < 0 || selection[size_t(x)] > selection[size_t(best)])) best = x;
                    if (best >= 0 && probs[size_t(best)] >= gpu_subst * probs[size_t(e)]) { pick[size_t(k)] = best; ++substituted_; }
                }
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                ids[size_t(i) * K + k] = pick[size_t(k)];
                wts[size_t(i) * K + k] = probs[size_t(pick[size_t(k)])];
                if (!cut_.empty() && cpu_topk >= 0 && k >= cpu_topk && !held(pick[size_t(k)])) {
                    cut_[size_t(i) * K + k] = 1; ++dropped_;
                    if (topk_original_weight) sum += wts[size_t(i) * K + k];
                    continue;
                }
                sum += wts[size_t(i) * K + k];
            }
            for (int k = 0; k < K; ++k) {
                float& v = wts[size_t(i) * K + k];
                if (!cut_.empty() && cut_[size_t(i) * K + k]) { v = 0.0f; continue; }
                if (c.normalize && K > 1) v /= sum + 1e-20f;
                v *= c.expert_scale;
            }
            if (lossy && cpu_trim > 0 && c.normalize && c.expert_scale > 0)
                for (int k = 2; k < K; ++k) {
                    const size_t j = size_t(i) * K + k;
                    if (!cut_[j] && !held(ids[j]) && wts[j] < cpu_trim * c.expert_scale) {
                        cut_[j] = 1; wts[j] = 0.0f; ++dropped_;
                    }
                }
        }
        };
        if (dev_route) {   // GPU_ROUTE: the GPU's picks and weights (a dropped pick: weight 0 and not resident)
            std::copy(route_host_, route_host_ + size_t(T) * K, ids.begin());
            std::copy(reinterpret_cast<const float*>(route_host_ + size_t(T) * K), reinterpret_cast<const float*>(route_host_ + size_t(2) * T * K), wts.begin());
            if (!cut_.empty())
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] < 0 || ids[i] >= c.experts) throw std::runtime_error("ds41: routed expert out of bounds");
                    const int rank = int(i % size_t(K));
                    const bool resident = held(ids[i]);
                    const bool topk_cut = cpu_topk >= 0 && rank >= cpu_topk && !resident && wts[i] == 0.0f;
                    const bool trim_cut = cpu_trim > 0 && c.normalize && c.expert_scale > 0 && rank >= 2 && !resident &&
                                          wts[i] > 0 && wts[i] < cpu_trim * c.expert_scale;
                    cut_[i] = topk_cut || trim_cut;
                    if (trim_cut) wts[i] = 0.0f;
                }
            for (size_t i = 0; i < cut_.size(); ++i) dropped_ += cut_[i];
        } else if (T >= 256) {   // long passes: tokens split over helper threads
            const int nt = 8;
            std::vector<std::thread> workers;
            for (int t = 0; t < nt; ++t)
                workers.emplace_back([&, t] { unpin_helper(); route(int(int64_t(T) * t / nt), int(int64_t(T) * (t + 1) / nt)); });
            for (auto& x : workers) x.join();
        } else {
            route(0, T);
        }
        if (std::vector<float> ri, rw; hook && reference && reference("ffn_moe_topk", l, ri) && reference("ffn_moe_weights_scaled", l, rw) &&
            ri.size() == size_t(K) && rw.size() == size_t(K)) {
            for (int k = 0; k < K; ++k) { ids[size_t(k)] = int(ri[size_t(k)]); wts[size_t(k)] = rw[size_t(k)]; }
        }
        if (hook) {
            observe_host("ffn_moe_logits", l, logits.data(), size_t(c.experts));
            std::vector<float> fid(ids.begin(), ids.end());
            observe_host("ffn_moe_topk", l, fid.data(), fid.size());
            observe_host("ffn_moe_weights_scaled", l, wts.data(), wts.size());
        }
        for (int i = 0; i < T; ++i)
            std::copy(ids.begin() + size_t(i) * K, ids.begin() + size_t(i + 1) * K,
                      route_log_.begin() + (size_t(i + skip_) * c.layers + size_t(l)) * K);
        const auto e0 = std::chrono::steady_clock::now();
        prof_[1] += std::chrono::duration<double>(e0 - f0).count();
        // the worker's rows: every pass with a GPU worker, decode passes alone with fetched rows
        const bool remote = remote_ && (!producer_.joinable() || remote_gpu_);
        if (remote) remote_->start(l, T, K, ids.data(), wts.data(), pinned_[0]);
        // the GPU: the shared expert, then the resident routed experts; the CPU lane takes the others meanwhile
        if (!shared_first) { quantize_xn(); shared_expert(l); }
        if (producer_.joinable()) {   // a streamed pass: every routed expert on the GPU, the others streamed in
            if (gpu_timing_) mark(1);
            const auto p0 = std::chrono::steady_clock::now();
            bool peer = false;
            if (peer_) {   // (the worker thread is idle in a streamed pass: the result through host_out as always)
                const bool th = peer_->to_host;
                peer_->to_host = true;
                peer = peer_->prefill(l, T, K, ids.data(), wts.data(), pinned_[0]);
                peer_->to_host = th;
            }
            const auto p1 = std::chrono::steady_clock::now();
            stream_experts(l, ids, wts);
            const auto p2 = std::chrono::steady_clock::now();
            peer_prep_s_ += std::chrono::duration<double>(p1 - p0).count();
            stream_enqueue_s_ += std::chrono::duration<double>(p2 - p1).count();
            if (peer) {   // the peer GPU's partial sums join the routed sum
                const auto w0 = std::chrono::steady_clock::now();
                peer_->collect();
                peer_wait_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                if (peer_sum_.bytes < size_t(T) * H * 4) peer_sum_ = Dev(size_t(T) * H * 4);
                ck(cudaMemcpyAsync(peer_sum_.p, peer_->host_out, size_t(T) * H * 4, cudaMemcpyHostToDevice, stream_), "peer sums");
                k::ds41_add(g_sum_.f(), peer_sum_.f(), g_sum_.f(), int(size_t(T) * H), stream_);
            }
            if (remote) {   // the worker's partial sums join the routed sum
                remote_->finish_raw(T);
                ck(cudaMemcpyAsync(remote_sum_.p, remote_->reply.data(), size_t(T) * H * 4, cudaMemcpyHostToDevice, stream_), "worker sums");
                k::ds41_add(g_sum_.f(), remote_sum_.f(), g_sum_.f(), int(size_t(T) * H), stream_);
                // a pinned reply buffer is received into again only after the next layer's router sync
                if (!remote_->reply_pinned) ck(cudaStreamSynchronize(stream_), "worker sums");
            }
            if (gpu_timing_) mark(2);
            prof_[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
            if (hook) observe("ffn_moe_out", l, g_sum_.f(), H);
            k::ds41_add(g_sum_.f(), y_.f(), y_.f(), int(size_t(T) * H), stream_);
            return;
        }
        auto& cpu_ids = host_cpu_route_ids_;
        cpu_ids = ids;
        auto& gids = host_gpu_route_ids_;
        auto& gn = host_gpu_route_counts_;
        auto& gws = host_gpu_route_weights_;
        if (dev_route) {
            gids.clear();
            gn.clear();
            gws.clear();
        } else {
            gids.resize(size_t(T) * 16);
            gn.assign(size_t(T), 0);
            gws.resize(size_t(T) * 16);
        }
        for (int i = 0; i < T; ++i) {
            int n = 0;
            for (int k = 0; k < K; ++k) {
                const int e = ids[size_t(i) * K + k];
                if (e < 0 || e >= c.experts) throw std::runtime_error("ds41: routed expert out of bounds");
                const bool on_gpu = !slot_.empty() && slot_[size_t(l)][size_t(e)] >= 0;
                if (spec_save_) spec_used_[size_t(i)].push_back(l * c.experts + e);
                else {
                    used_.push_back(l * c.experts + e);
                    if (!rt_count_.empty()) rt_count_[size_t(l) * c.experts + size_t(e)] += 1;
                }
                if (!where_.empty() && (where_[size_t(l)][size_t(e)] == 2 || where_[size_t(l)][size_t(e)] == 4)) {   // the peer GPU's (4: grown there)
                    cpu_ids[size_t(i) * K + k] = -1;
                    tier_hits_ += 1;
                    tier_all_ += 1;
                    continue;
                }
                if (!cut_.empty() && cut_[size_t(i) * K + k]) { cpu_ids[size_t(i) * K + k] = -1; continue; }   // CPU_TOPK
                if (!on_gpu) { (spec_save_ ? spec_missed_[size_t(i)] : missed_).push_back(l * c.experts + e); continue; }
                if (!dev_route) {
                    gids[size_t(i) * 16 + size_t(n)] = e;
                    gws[size_t(i) * 16 + size_t(n)] = wts[size_t(i) * K + k];
                }
                ++n;
                cpu_ids[size_t(i) * K + k] = -1;
            }
            if (!dev_route) gn[size_t(i)] = n;
            tier_hits_ += n;
            tier_all_ += K;
            if (static int dbg = 0; T == 1 && std::getenv("STRATA_DS41_TIER_DEBUG") && dbg < 80) {
                ++dbg;
                int res = 0;
                for (int x : slot_[size_t(l)]) res += x >= 0;
                std::cerr << "DS41 tier_debug l=" << l << " hits=" << n << "/" << K << " resident_in_layer=" << res << " ids=";
                for (int k = 0; k < K; ++k) std::cerr << ids[size_t(i) * K + k] << (k + 1 < K ? "," : "\n");
            }
        }
        // STRATA_DS41_DECODE_STREAM=a,b (with STRATA_DS41_RAM_PINNED): up to a / b of the CPU's experts (the most routes
        // first, each GPU's own socket part first) are copied to GPU 0 / the peer GPU and computed there meanwhile
        std::vector<std::pair<int, std::vector<int>>> to_gpu0;   // (position, routes)
        {
            static const std::pair<int, int> quota = [] {
                const char* v = std::getenv("STRATA_DS41_DECODE_STREAM");
                if (!v) return std::pair<int, int>(0, 0);
                int a = 0, b = 0;
                std::sscanf(v, "%d,%d", &a, &b);
                return std::pair<int, int>(std::max(0, a), std::max(0, b));
            }();
            if ((quota.first || quota.second) && !packed_.empty()) {
                std::map<int, std::vector<int>> by;   // expert -> its CPU routes
                for (int i = 0; i < T * K; ++i) if (cpu_ids[size_t(i)] >= 0) by[ids[size_t(i)]].push_back(i);
                std::vector<std::pair<int, int>> order;   // (-routes, expert)
                for (const auto& [e, r] : by) order.push_back({-int(r.size()), e});
                std::sort(order.begin(), order.end());
                const int s0 = s0_[size_t(l)];
                int n0 = 0, n1 = 0;
                std::vector<char> taken(order.size(), 0);
                const size_t stride = stream_stride(l);
                for (int pass = 0; pass < 2; ++pass)
                    for (size_t j = 0; j < order.size(); ++j) {
                        if (taken[j]) continue;
                        const int e = order[j].second, q = ram_pos_[size_t(l)][size_t(e)];
                        const bool own0 = q < s0;
                        const bool can0 = n0 < quota.first && (pass == 1 || own0), can1 = peer_ && n1 < quota.second && (pass == 1 || !own0);
                        if (!can0 && !can1) continue;
                        taken[j] = 1;
                        if (can0 && (own0 || !can1)) {
                            to_gpu0.push_back({q, by[e]});
                            ++n0;
                        } else {
                            for (int i : by[e]) peer_->extra.push_back({packed_[size_t(l)] + size_t(q) * stride, i / K, wts[size_t(i)]});
                            ++n1;
                        }
                        for (int i : by[e]) cpu_ids[size_t(i)] = -1;
                    }
            }
        }
        std::vector<int32_t> stage_original_ids;
        if (stage_cpu_collect_) stage_original_ids = cpu_ids;
        std::vector<int32_t> stage_diag_ids;
        if (stage_cpu_order_ && T == 1 && strata_env_is_one("STRATA_DS41_STAGE_CPU_ORDER_CHECK"))
            stage_diag_ids = cpu_ids;
        int staged_cpu_expert = -1;
        std::vector<int> staged_cpu_routes;
        std::vector<std::pair<int, std::vector<int>>> staged_cpu_groups;
        if (stage_cpu_expert_ && T <= 8) {
            if (T < 1 || K < 1 || K > 16 || l < 0 || l >= c.layers || peer_ || remote_ || producer_.joinable() ||
                !packed_.empty() || !ram_pos_.empty() || size_t(T) * size_t(K) != cpu_ids.size() ||
                ids.size() != cpu_ids.size() || wts.size() != cpu_ids.size())
                throw std::runtime_error("ds41: staged CPU expert route geometry or execution mode is invalid");
            std::map<int, std::vector<int>> candidates;
            for (int i = 0; i < T * K; ++i) {
                const int e = cpu_ids[size_t(i)];
                if (e < 0) continue;
                if (e >= c.experts || ids[size_t(i)] != e || !std::isfinite(wts[size_t(i)]) || wts[size_t(i)] < 0)
                    throw std::runtime_error("ds41: staged CPU expert route is invalid");
                candidates[e].push_back(i);
            }
            // Prefer experts reused across verification rows; ties use lower ID.
            for (auto& [e, routes] : candidates) staged_cpu_groups.emplace_back(e, std::move(routes));
            if (!stage_cache_frequency_.empty()) {
                for (const auto& g : staged_cpu_groups) {
                    const size_t idx = size_t(l) * size_t(c.experts) + size_t(g.first);
                    if (idx >= stage_cache_frequency_.size()) throw std::runtime_error("ds41: staging frequency index out of bounds");
                    auto& n = stage_cache_frequency_[idx];
                    if (n != UINT32_MAX) ++n;
                }
            }
            auto cached = [&](int expert) {
                return std::any_of(stage_cache_.begin(), stage_cache_.end(),
                                   [&](const StageCacheEntry& entry) { return entry.layer == l && entry.expert == expert; });
            };
            std::stable_sort(staged_cpu_groups.begin(), staged_cpu_groups.end(), [&](const auto& a, const auto& b) {
                const bool ca = cached(a.first), cb = cached(b.first);
                if (ca != cb) return ca;
                if (a.second.size() != b.second.size()) return a.second.size() > b.second.size();
                if (!stage_cache_frequency_.empty())
                    return stage_cache_frequency_[size_t(l) * size_t(c.experts) + size_t(a.first)] >
                           stage_cache_frequency_[size_t(l) * size_t(c.experts) + size_t(b.first)];
                return false;
            });
            if (staged_cpu_groups.size() > size_t(stage_cpu_quota_)) staged_cpu_groups.resize(size_t(stage_cpu_quota_));
            // Extra experts may run only when already cached; the first still
            // supplies misses so the bounded cache can adapt.
            if (strata_env_is_one("STRATA_DS41_STAGE_EXTRA_CACHED") && staged_cpu_groups.size() > 1)
                staged_cpu_groups.erase(std::remove_if(staged_cpu_groups.begin() + 1, staged_cpu_groups.end(),
                    [&](const auto& g) { return !cached(g.first); }), staged_cpu_groups.end());
            // After warming the bounded cache, let the CPU handle three out of
            // four uncached candidates; periodically insert one to adapt.
            static const bool throttle_cache_misses = strata_env_is_one("STRATA_DS41_STAGE_CACHE_THROTTLE");
            if (throttle_cache_misses && !staged_cpu_groups.empty() && !stage_cache_.empty() &&
                !cached(staged_cpu_groups.front().first) &&
                std::none_of(stage_cache_.begin(), stage_cache_.end(), [](const StageCacheEntry& e) { return e.layer < 0; }) &&
                (++stage_cache_skipped_misses_ % 4) != 0)
                staged_cpu_groups.clear();
            if (!staged_cpu_groups.empty()) staged_cpu_expert = staged_cpu_groups.front().first;
            for (const auto& group : staged_cpu_groups)
                staged_cpu_routes.insert(staged_cpu_routes.end(), group.second.begin(), group.second.end());
            if (staged_cpu_routes.size() > size_t(T) * size_t(stage_cpu_quota_))
                throw std::runtime_error("ds41: staged CPU route count exceeds quota");
            for (int i : staged_cpu_routes) cpu_ids[size_t(i)] = -1;
        }
        // the canonical tier's routes leave the lane (they return through finish_with_external)
        std::vector<int32_t> canon_original, canon_entries;
        std::vector<int> canon_gstart;
        std::vector<std::pair<int, int>> canon_groups;   // (expert, ring slot or -1 for the static arena)
        if (prom_thread_.joinable()) promote_apply();
        if (ad_thread_.joinable()) { adapt_apply(false); adapt_count(l, cpu_ids.data(), cpu_ids.size()); }
        if (!canon_off_.empty() && T <= canon_tmax_ && size_t(T) * size_t(K) == cpu_ids.size()) {
            std::map<int, std::vector<int>> held, rest;
            for (int i = 0; i < T * K; ++i) {
                const int e = cpu_ids[size_t(i)];
                if (e < 0) continue;
                if (e >= c.experts) throw std::runtime_error("ds41: canonical tier route out of bounds");
                ++canon_cpu_routes_;
                if (!own_.empty() && own_[size_t(l) * c.experts + e] && canon_off_[size_t(l) * c.experts + e] < 0 &&
                    (canon_pin_off_.empty() || canon_pin_off_[size_t(l) * c.experts + e] < 0)) continue;   // the remote owner's (STRATA_DS41_REMOTE_OWN), unless pinned here
                (canon_off_[size_t(l) * c.experts + e] >= 0 ? held : rest)[e].push_back(i);
            }
            // STRATA_DS41_CANON_CAP=n: at most n resident experts run on the GPU in a pass (worker-owned ones first, then the widest);
            // the others stay on the lanes - the values are the same - when the GPU would be the slower path
            static const int canon_cap = std::getenv("STRATA_DS41_CANON_CAP") ? std::max(0, std::atoi(std::getenv("STRATA_DS41_CANON_CAP"))) : 0;
            if (canon_cap > 0 && held.size() > size_t(canon_cap)) {
                std::vector<std::pair<int, std::vector<int>>> h(held.begin(), held.end());
                std::stable_sort(h.begin(), h.end(), [&](const auto& a, const auto& b) {
                    const bool oa = !own_.empty() && own_[size_t(l) * c.experts + a.first], ob = !own_.empty() && own_[size_t(l) * c.experts + b.first];
                    return oa != ob ? oa : a.second.size() > b.second.size();
                });
                held.clear();
                for (size_t i = 0; i < size_t(canon_cap); ++i) held.emplace(h[i].first, std::move(h[i].second));
            }
            // STRATA_DS41_CANON_STREAM=q: the q widest remaining CPU experts (most routes; lower id first) are copied
            // to the GPU this pass and run there with the same arithmetic
            std::vector<std::pair<int, std::vector<int>>> streamed;
            if (canon_stream_ > 0 && !rest.empty()) {
                std::vector<std::pair<int, std::vector<int>>> r;
                for (const auto& kv : rest)
                    if (canon_pin_off_.empty() || canon_pin_off_[size_t(l) * c.experts + kv.first] >= 0) r.push_back(kv);   // pinned level: only its experts stream
                // the worker's experts first (the worker's lane is the slower path), then the widest
                const bool own_first = !own_.empty() && !canon_pin_off_.empty();
                std::stable_sort(r.begin(), r.end(), [&](const auto& a, const auto& b) {
                    if (own_first) {
                        const bool oa = own_[size_t(l) * c.experts + a.first], ob = own_[size_t(l) * c.experts + b.first];
                        if (oa != ob) return oa;
                    }
                    return a.second.size() > b.second.size();
                });
                if (r.size() > size_t(canon_stream_)) r.resize(size_t(canon_stream_));
                streamed = std::move(r);
            }
            if (!held.empty() || !streamed.empty()) {
                canon_original = cpu_ids;
                canon_ids_now_ = cpu_ids;
                auto add = [&](int e, const std::vector<int>& routes, int slot) {
                    canon_gstart.push_back(int(canon_entries.size()));
                    canon_groups.push_back({e, slot});
                    for (int i : routes) { canon_entries.push_back(i); cpu_ids[size_t(i)] = -1; }
                };
                for (const auto& [e, routes] : held) add(e, routes, -1);
                for (size_t j = 0; j < streamed.size(); ++j) { add(streamed[j].first, streamed[j].second, int(j)); canon_streamed_ += streamed[j].second.size(); }
                canon_gstart.push_back(int(canon_entries.size()));
                canon_routes_ += canon_entries.size();
            }
        }
        // STRATA_DS41_REMOTE_OWN: the worker's routes leave the lane and are sent now, with the pass's activations
        std::vector<int32_t> own_ids;
        int own_n = 0;
        if (own_remote_ && size_t(T) * size_t(K) == cpu_ids.size()) {
            if (prom_thread_.joinable()) promote_apply();
            own_ids.assign(cpu_ids.size(), -1);
            for (size_t i = 0; i < cpu_ids.size(); ++i) {
                const int e = cpu_ids[i];
                if (prom_state_ && T <= 8 && e >= 0 && e < c.experts && prom_candidates_[size_t(l) * c.experts + e])
                    ++(own_[size_t(l) * c.experts + e] ? prom_remote_routes_ : prom_local_routes_);
                if (e >= 0 && e < c.experts && own_[size_t(l) * c.experts + e]) {
                    own_ids[i] = e;
                    ++own_n;
                    static const bool decode_only = std::getenv("STRATA_DS41_OWN_PROMOTE_DECODE_ONLY") && std::string(std::getenv("STRATA_DS41_OWN_PROMOTE_DECODE_ONLY")) == "1";
                    if (own_score_ && (T <= 8 || !decode_only)) own_score_[size_t(l) * c.experts + e].fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (own_n) {
                if (canon_original.empty()) canon_original = cpu_ids;
                for (size_t i = 0; i < cpu_ids.size(); ++i) if (own_ids[i] >= 0) cpu_ids[i] = -1;
                own_start_t_ = std::chrono::steady_clock::now();
                own_prep_ms_ += std::chrono::duration<double, std::milli>(own_start_t_ - e0).count();   // router result in hand -> the worker's request goes out
                own_remote_->start(l, T, K, own_ids.data(), wts.data(), pinned_[0], own_n);
                own_send_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - own_start_t_).count();
                ++own_starts_;
                own_routes_ += size_t(own_n);
            }
        }
        if (!ram_pos_.empty())   // packed copy: the lane addresses positions
            for (int& e : cpu_ids) if (e >= 0) e = ram_pos_[size_t(l)][size_t(e)];
        // STRATA_DS41_LANE_ASYNC=1: the CPU lane's other threads start while this one enqueues the GPU experts
        const bool lane_async = staged_cpu_expert >= 0 || !canon_entries.empty() || own_n > 0 || (std::getenv("STRATA_DS41_LANE_ASYNC") && std::string(std::getenv("STRATA_DS41_LANE_ASYNC")) == "1");
        static const bool decode_mmq = std::getenv("STRATA_DS41_DECODE_MMQ") && std::string(std::getenv("STRATA_DS41_DECODE_MMQ")) == "1";
        static const bool decode_grouped = !strata_env_is_one("STRATA_DS41_VERIFY_EXACT") && std::getenv("STRATA_DS41_DECODE_GROUPED") && std::string(std::getenv("STRATA_DS41_DECODE_GROUPED")) == "1" &&
                                           k::native_expert_supported(experts_[0].f.gu_type, experts_[0].f.d_type, c.hidden, c.ff);
        // STRATA_DS41_PEER_MMQ (follows DECODE_MMQ): the peer GPU's routes by MMQ too (0: its GEMV chains)
        static const bool peer_mmq = decode_mmq && !(std::getenv("STRATA_DS41_PEER_MMQ") && std::string(std::getenv("STRATA_DS41_PEER_MMQ")) == "0");
        const int peer_mode = decode_grouped ? 2 : peer_ && peer_mmq && peer_->mmq && T > 1 ? 1 : 0;
        // the peer's worker thread first: its work starts while this thread prepares the lane and GPU 0's experts
        struct Posted {   // the job reads ids / wts until it is enqueued: an unwind waits for that first
            PeerGpu* p = nullptr;
            ~Posted() { if (p) { try { p->wait_enqueued(); } catch (...) {} } }
        } posted;
        if (peer_ && peer_->threaded) {
            if (peer_sum_.bytes < size_t(T) * H * 4) {   // (resize ran before the peer existed)
                ck(cudaStreamSynchronize(stream_), "peer sum");   // its last reader (the previous layer's add) done
                peer_sum_ = Dev(size_t(T) * H * 4);
                if (!peer_late_ev_[0])
                    for (auto& e : peer_late_ev_) ck(cudaEventCreate(&e), "peer late event");
            }
            peer_->post({l, T, K, peer_mode, ids.data(), wts.data(), pinned_[0], peer_sum_.p});
            posted.p = peer_.get();
        }
        // STRATA_DS41_PEER_FIRST=1: the peer GPU's work enqueued before the lane starts and GPU 0's experts (it is the
        // longest of the three: it had started last); the same calls, so the same results
        static const bool peer_first = std::getenv("STRATA_DS41_PEER_FIRST") && std::string(std::getenv("STRATA_DS41_PEER_FIRST")) == "1";
        bool peer = false, peer_done = false;
        if (peer_first && peer_ && !peer_->threaded) {
            peer = peer_mode == 2 ? peer_->grouped(l, T, K, ids.data(), wts.data(), pinned_[0])
                 : peer_mode == 1 ? peer_->prefill(l, T, K, ids.data(), wts.data(), pinned_[0])
                                  : peer_->decode(l, T, K, ids.data(), wts.data(), pinned_[0]);
            peer_done = true;
        }
        // STRATA_DS41_DEFER_RANK=R (1..top_k-1; lossy, opt-in): the CPU lane's routes of router rank >= R (the token's picks
        // are in score order) are computed after the others, while the GPU runs the next layer's attention, and added to
        // the streams before that layer's FFN (defer_apply) - KTransformers' / dsx's expert deferral. Passes of up to 16
        // tokens (decode, verify); prompts stay exact
        // STRATA_DS41_DEFER_ALL=1: every CPU-lane route deferred (rank >= 0), so the lane overlaps the GPU's next
        // attention completely (lossy like DEFER_RANK, more so)
        static const bool defer_all = std::getenv("STRATA_DS41_DEFER_ALL") && std::string(std::getenv("STRATA_DS41_DEFER_ALL")) == "1";
        static const int defer_rank = defer_all ? 0 : std::getenv("STRATA_DS41_DEFER_RANK") ? std::atoi(std::getenv("STRATA_DS41_DEFER_RANK")) : 0;
        bool defer_now = false;
        if ((defer_rank > 0 || defer_all) && defer_rank < K && T <= 16 && !hook && !remote) {
            def_ids_.assign(cpu_ids.size(), -1);
            def_wts_ = wts;
            for (size_t i = 0; i < cpu_ids.size(); ++i)
                if (cpu_ids[i] >= 0 && int(i % size_t(K)) >= defer_rank) { def_ids_[i] = cpu_ids[i]; cpu_ids[i] = -1; defer_now = true; }
        }
        static const bool defer_remote_on = strata_env_is_one("STRATA_DS41_DEFER_REMOTE");
        const bool defer_remote = defer_remote_on && own_n > 0 && T <= 16 && !hook && !remote && lane_async && !defer_now;
        if (defer_remote) {
            if (!def_in_) {
                def_in_ = static_cast<float*>(pinned_keep(size_t(16) * H * 4));
                def_out_ = static_cast<float*>(pinned_keep(size_t(16) * H * 4));
                def_dev_ = Dev(size_t(16) * H * 4);
                def_coeff_ = Dev(size_t(16) * 24 * 4);
            }
            std::memcpy(def_in_, pinned_[0], size_t(T) * H * 4);
            dr_ids_ = cpu_ids;
            dr_wts_ = wts;
            lane_->begin(experts_[size_t(l)], def_in_, T, dr_ids_.data(), dr_wts_.data(), K, def_out_);
        } else
        if (lane_async) lane_->begin(experts_[size_t(l)], pinned_[0], T, cpu_ids.data(), wts.data(), K, host_y);
        if (!canon_entries.empty()) canon_enqueue(l, T, K, canon_entries, canon_gstart, canon_groups);
        if (ad_thread_.joinable()) adapt_apply(true);
        bool any_gpu = false;
        bool batched = false;
        if (dev_route) {   // GPU_ROUTE: the GPU-held experts already ran into g_sum_
            any_gpu = true;
            batched = true;
        } else if (decode_grouped && fl_ == c.ff && std::any_of(gn.begin(), gn.end(), [](int n) { return n > 0; })) {
            ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum");
            any_gpu = true;
            std::map<int, std::vector<std::pair<int, float>>> by;
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < gn[size_t(i)]; ++j) by[gids[size_t(i) * 16 + size_t(j)]].push_back({i, gws[size_t(i) * 16 + size_t(j)]});
            std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
            for (auto& [e, r] : by) groups.push_back({slots_[size_t(slot_[size_t(l)][size_t(e)])].p, std::move(r)});
            const auto& L = experts_[size_t(l)];
            batched = grouped0_.run(L.f.gu_type, L.f.d_type, c.hidden, c.ff, c.clamp[size_t(l)], groups, xn_.f(), T, g_sum_.f(), stream_);
        }
        if (!batched && decode_mmq && std::any_of(gn.begin(), gn.end(), [](int n) { return n > 0; })) {
            ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum");
            any_gpu = true;
            batched = resident_mmq(l, T, K, ids.data(), wts.data());
            static const bool grown_grouped = std::getenv("STRATA_DS41_GROWN_GROUPED") && std::string(std::getenv("STRATA_DS41_GROWN_GROUPED")) == "1";
            const auto& LE = experts_[size_t(l)];
            if (batched && grown_ && grown_grouped && fl_ == c.ff && k::native_expert_supported(LE.f.gu_type, LE.f.d_type, c.hidden, c.ff)) {
                // STRATA_DS41_GROWN_GROUPED=1: GROW's experts in one grouped launch
                std::map<int, std::vector<std::pair<int, float>>> by;
                for (int i = 0; i < T; ++i)
                    for (int j = 0; j < gn[size_t(i)]; ++j) {
                        const int e = gids[size_t(i) * 16 + size_t(j)];
                        if (where_[size_t(l)][size_t(e)] == 3) by[e].push_back({i, gws[size_t(i) * 16 + size_t(j)]});
                    }
                std::vector<std::pair<const uint8_t*, std::vector<std::pair<int, float>>>> groups;
                for (auto& [e, r] : by) groups.push_back({slots_[size_t(slot_[size_t(l)][size_t(e)])].p, std::move(r)});
                grown0_.run(LE.f.gu_type, LE.f.d_type, c.hidden, c.ff, c.clamp[size_t(l)], groups, xn_.f(), T, g_sum_.f(), stream_);
            } else if (batched && grown_)   // GROW's experts (where_ 3, outside the resident arena): a GEMV chain per route
                for (int i = 0; i < T; ++i) {
                    int gi[16], m = 0;
                    float gw[16];
                    for (int j = 0; j < gn[size_t(i)]; ++j) {
                        const int e = gids[size_t(i) * 16 + size_t(j)];
                        if (where_[size_t(l)][size_t(e)] == 3) { gi[m] = e; gw[m] = gws[size_t(i) * 16 + size_t(j)]; ++m; }
                    }
                    if (m) gpu_experts(l, i, gi, gw, m);
                }
        }
        for (int i = 0; i < T && !batched; ++i) {
            const int n = gn[size_t(i)];
            if (n && !any_gpu) { ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum"); any_gpu = true; }
            if (n) gpu_experts(l, i, gids.data() + size_t(i) * 16, gws.data() + size_t(i) * 16, n);
        }
        if (!to_gpu0.empty()) {   // the CPU's experts copied here: a GEMV chain per route
            if (!any_gpu) { ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum"); any_gpu = true; }
            const size_t stride = stream_stride(l);
            while (dstream_.size() < to_gpu0.size()) dstream_.emplace_back(max_stride_ ? max_stride_ : stride);
            for (size_t j = 0; j < to_gpu0.size(); ++j) {
                ck(cudaMemcpyAsync(dstream_[j].p, packed_[size_t(l)] + size_t(to_gpu0[j].first) * stride, stride, cudaMemcpyHostToDevice, stream_), "streamed expert");
                for (int i : to_gpu0[j].second) expert_chain(l, dstream_[j].b(), i / K, wts[size_t(i)]);
            }
        }
        if (staged_cpu_expert >= 0) {
            const size_t bytes = expert_bytes(l);
            if (!stage_cpu_host_ || !stage_cpu_check_host_ || !stage_cpu_device_.p || !stage_cpu_copy_done_ ||
                size_t(l) >= stage_cpu_checked_.size() || bytes == 0 || bytes > stage_cpu_capacity_ || bytes > stage_cpu_device_.bytes ||
                staged_cpu_groups.empty() || staged_cpu_groups.size() > size_t(stage_cpu_quota_) ||
                staged_cpu_routes.size() > size_t(8 * stage_cpu_quota_) || (stage_cpu_collect_ && !stage_cpu_results_))
                throw std::runtime_error("ds41: staged CPU expert scratch or source bounds are invalid");
            if (!stage_cpu_collect_ && !any_gpu) { ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "stage tier sum"); any_gpu = true; }
            size_t result_index = 0;
            for (const auto& group : staged_cpu_groups) {
                const int expert = group.first;
                if (expert < 0 || expert >= c.experts || group.second.empty())
                    throw std::runtime_error("ds41: staged CPU expert index is invalid");
                for (int i : group.second)
                    if (i < 0 || i >= T * K || ids[size_t(i)] != expert || i / K >= T)
                        throw std::runtime_error("ds41: staged CPU expert route bounds are invalid");
                const bool stage_sample = stage_cpu_timing_ && stage_cpu_transfers_ < 1000;
                uint8_t* device_expert = stage_cpu_device_.b();
                bool cache_hit = false;
                if (!stage_cache_.empty()) {
                    if (!stage_cpu_collect_ || stage_cache_.size() < size_t(stage_cpu_quota_) || !stage_cache_device_.p ||
                        stage_cache_device_.bytes / stage_cpu_capacity_ != stage_cache_.size())
                        throw std::runtime_error("ds41: staging cache execution mode or capacity invalid");
                    size_t slot = stage_cache_.size();
                    for (size_t j = 0; j < stage_cache_.size(); ++j)
                        if (stage_cache_[j].layer == l && stage_cache_[j].expert == expert) { slot = j; cache_hit = true; break; }
                    if (!cache_hit) {
                        const bool frequency_evict = strata_env_is_one("STRATA_DS41_STAGE_CACHE_LFU") && !stage_cache_frequency_.empty();
                        auto priority = [&](const StageCacheEntry& entry) {
                            if (entry.layer < 0) return uint32_t(0);
                            if (entry.layer >= c.layers || entry.expert < 0 || entry.expert >= c.experts)
                                throw std::runtime_error("ds41: staging cache entry out of bounds");
                            return stage_cache_frequency_[size_t(entry.layer) * size_t(c.experts) + size_t(entry.expert)];
                        };
                        slot = size_t(std::min_element(stage_cache_.begin(), stage_cache_.end(),
                                      [&](const auto& a, const auto& b) {
                                          if (frequency_evict) {
                                              if ((a.layer < 0) != (b.layer < 0)) return a.layer < 0;
                                              const auto fa = priority(a), fb = priority(b);
                                              if (fa != fb) return fa < fb;
                                          }
                                          return a.age < b.age;
                                      }) - stage_cache_.begin());
                        ++stage_cache_misses_;
                    } else ++stage_cache_hits_;
                    if (slot >= stage_cache_.size() || slot * stage_cpu_capacity_ > stage_cache_device_.bytes - bytes)
                        throw std::runtime_error("ds41: staging cache slot out of bounds");
                    device_expert = stage_cache_device_.b(slot * stage_cpu_capacity_);
                    stage_cache_[slot] = {l, expert, ++stage_cache_clock_};
                }
                if (stage_sample) ++stage_timed_groups_;
                if (!cache_hit) {
                    // H2D consumes the host slot before it can be overwritten;
                    // each prior cache use has completed at collection sync.
                    const auto reuse_start = std::chrono::steady_clock::now();
                    if (stage_cpu_copy_pending_) ck(cudaEventSynchronize(stage_cpu_copy_done_), "CPU expert stage reuse");
                    const auto pack_start = std::chrono::steady_clock::now();
                    if (stage_sample) stage_timing_[0] += std::chrono::duration<double>(pack_start - reuse_start).count();
                    slice_to_host(l, expert, stage_cpu_host_);
                    if (stage_sample) stage_timing_[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - pack_start).count();
                    ck(cudaMemcpyAsync(device_expert, stage_cpu_host_, bytes, cudaMemcpyHostToDevice, stream_), "CPU expert stage H2D");
                    ck(cudaEventRecord(stage_cpu_copy_done_, stream_), "CPU expert stage copy event");
                    stage_cpu_copy_pending_ = true;
                    if (!stage_cpu_checked_[size_t(l)]) {
                        ck(cudaMemcpyAsync(stage_cpu_check_host_, device_expert, bytes, cudaMemcpyDeviceToHost, stream_), "CPU expert stage D2H check");
                        ck(cudaStreamSynchronize(stream_), "CPU expert stage check sync");
                        if (std::memcmp(stage_cpu_host_, stage_cpu_check_host_, bytes) != 0)
                            throw std::runtime_error("ds41: staged CPU expert copy mismatch");
                        stage_cpu_checked_[size_t(l)] = 1;
                    }
                }
                if (stage_cpu_batch_act_ && group.second.size() > 1) {
                    if (result_index + group.second.size() > staged_cpu_routes.size())
                        throw std::runtime_error("ds41: batched staged result bounds invalid");
                    for (size_t j = 0; j < group.second.size(); ++j)
                        if (staged_cpu_routes[result_index + j] != group.second[j])
                            throw std::runtime_error("ds41: batched staged result ordering invalid");
                    staged_cpu_order_batch(l, device_expert, bytes, group.second, K, stage_cpu_results_ + result_index * H);
                    result_index += group.second.size();
                } else for (int i : group.second) {
                    if (result_index >= staged_cpu_routes.size() || staged_cpu_routes[result_index] != i)
                        throw std::runtime_error("ds41: staged CPU result ordering is invalid");
                    if (stage_cpu_order_) staged_cpu_order_chain(l, device_expert, bytes, i / K, wts[size_t(i)],
                                                                  stage_cpu_collect_ ? stage_cpu_results_ + result_index * H : nullptr);
                    else expert_chain(l, device_expert, i / K, wts[size_t(i)]);
                    ++result_index;
                }
                ++stage_cpu_transfers_;
            }
            if (result_index != staged_cpu_routes.size())
                throw std::runtime_error("ds41: incomplete staged CPU results");
        }
        if (!peer_done)
            peer = peer_ && !peer_->threaded &&
                   (peer_mode == 2 ? peer_->grouped(l, T, K, ids.data(), wts.data(), pinned_[0])
                    : peer_mode == 1 ? peer_->prefill(l, T, K, ids.data(), wts.data(), pinned_[0])
                                     : peer_->decode(l, T, K, ids.data(), wts.data(), pinned_[0]));
        if (dtiming_ && (T <= 64 || pf_timing_)) dmark(9);   // "experts": the GPU's experts enqueued (and run); "lane_wait": the CPU's and the peer's after
        const auto x1 = std::chrono::steady_clock::now();
        if (!canon_entries.empty() && own_n == 0) {
            ck(cudaStreamSynchronize(stream_), "canonical tier results sync");
            if (strata_env_is_one("STRATA_DS41_CANON_CHECK") && !canon_checked_[size_t(l)]) canon_check(l, K, canon_entries[0]);
            lane_->finish_with_external(canon_original.data(), canon_entries.data(), canon_res_, int(canon_entries.size()), K);
        } else if (own_n > 0 && defer_remote) {
            ext_entries_.clear();
            for (size_t i = 0; i < own_ids.size(); ++i) if (own_ids[i] >= 0) ext_entries_.push_back(int32_t(i));
            if (int(ext_entries_.size()) != own_n || size_t(own_n) * size_t(H) > own_remote_->reply.size())
                throw std::runtime_error("ds41: remote_own merge geometry");
            defer_job_ = [this, l, K, own_n, orig = canon_original, ent = canon_entries, ext = ext_entries_] {
                if (!ent.empty()) {
                    ck(cudaStreamSynchronize(stream_), "deferred canonical tier results sync");
                    if (strata_env_is_one("STRATA_DS41_CANON_CHECK") && !canon_checked_[size_t(l)]) canon_check(l, K, ent[0]);
                }
                own_remote_->finish_raw(0);
                lane_->finish_with_external2(orig.data(), ent.data(), canon_res_, int(ent.size()), ext.data(), own_remote_->reply.data(), own_n, K);
            };
        } else if (own_n > 0) {   // the remote worker's routes (and the canonical tier's): one merge in the lane's order
            const auto o0 = std::chrono::steady_clock::now();
            if (!canon_entries.empty()) {
                ck(cudaStreamSynchronize(stream_), "canonical tier results sync");
                if (strata_env_is_one("STRATA_DS41_CANON_CHECK") && !canon_checked_[size_t(l)]) canon_check(l, K, canon_entries[0]);
            }
            const auto o1 = std::chrono::steady_clock::now();
            own_remote_->finish_raw(T);
            const auto o2 = std::chrono::steady_clock::now();
            if (T <= 8 || sess_) {   // decode / verify passes (and session passes): GPU sync, the worker's reply, then the merge (below)
                own_ms_[0] += std::chrono::duration<double, std::milli>(o1 - o0).count();
                own_ms_[1] += std::chrono::duration<double, std::milli>(o2 - o1).count();
                own_latency_ms_ += std::chrono::duration<double, std::milli>(o2 - own_start_t_).count();
                own_pending_merge_ = true;
                own_merge_t0_ = o2;
            }
            ext_entries_.clear();
            for (size_t i = 0; i < own_ids.size(); ++i) if (own_ids[i] >= 0) ext_entries_.push_back(int32_t(i));
            if (int(ext_entries_.size()) != own_n || size_t(own_n) * size_t(H) > own_remote_->reply.size())
                throw std::runtime_error("ds41: remote_own merge geometry");
            lane_->finish_with_external2(canon_original.data(), canon_entries.data(), canon_res_, int(canon_entries.size()),
                                         ext_entries_.data(), own_remote_->reply.data(), own_n, K);
            if (own_pending_merge_) {
                own_ms_[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - own_merge_t0_).count();
                own_ms_[3] += 1;
                own_pending_merge_ = false;
            }
        } else if (stage_cpu_collect_ && staged_cpu_expert >= 0) {
            const bool sample = stage_cpu_timing_ && stage_cpu_transfers_ <= 1000;
            const auto wait_start = std::chrono::steady_clock::now();
            ck(cudaStreamSynchronize(stream_), "CPU-order staged results sync");
            lane_->finish_with_external(stage_original_ids.data(), staged_cpu_routes.data(), stage_cpu_results_, int(staged_cpu_routes.size()), K);
            if (sample) stage_timing_[4] += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
        } else if (lane_async) lane_->finish();
        else lane_->run(experts_[size_t(l)], pinned_[0], T, cpu_ids.data(), wts.data(), K, host_y);
        if (!stage_cpu_collect_ && !stage_diag_ids.empty() && staged_cpu_expert >= 0 && staged_cpu_routes.size() == 1 &&
            stage_cpu_diag_out_.size() == size_t(H)) {
            std::vector<float> full(size_t(H), 0.f);
            lane_->run(experts_[size_t(l)], pinned_[0], T, stage_diag_ids.data(), wts.data(), K, full.data());
            double sq = 0, norm = 0, max_abs = 0;
            const float weight = wts[size_t(staged_cpu_routes[0])];
            for (int j = 0; j < H; ++j) {
                const double delta = double(full[size_t(j)]) - (double(host_y[j]) + double(weight * stage_cpu_diag_out_[size_t(j)]));
                sq += delta * delta; norm += double(full[size_t(j)]) * full[size_t(j)];
                max_abs = std::max(max_abs, std::abs(delta));
            }
            std::cerr << "DS41 cpu_order_merge layer=" << l << " max_abs=" << max_abs
                      << " relative_l2=" << std::sqrt(sq / (norm + 1e-300)) << "\n";
        }
        const auto x2 = std::chrono::steady_clock::now();
        if (defer_remote) {   // the merge job above runs in defer_apply; this layer's post coefficients kept for its add
            ck(cudaMemcpyAsync(def_coeff_.p, coeff_ffn_.p, size_t(T) * 24 * 4, cudaMemcpyDeviceToDevice, stream_), "defer coeff");
            defer_pending_ = true;
            defer_tokens_ = T;
        }
        if (defer_now) {   // the deferred routes on the lane now; this layer's post coefficients kept for their add
            if (!def_in_) {
                def_in_ = static_cast<float*>(pinned_keep(size_t(16) * H * 4));
                def_out_ = static_cast<float*>(pinned_keep(size_t(16) * H * 4));
                def_dev_ = Dev(size_t(16) * H * 4);
                def_coeff_ = Dev(size_t(16) * 24 * 4);
            }
            std::memcpy(def_in_, pinned_[0], size_t(T) * H * 4);
            ck(cudaMemcpyAsync(def_coeff_.p, coeff_ffn_.p, size_t(T) * 24 * 4, cudaMemcpyDeviceToDevice, stream_), "defer coeff");
            lane_->begin(experts_[size_t(l)], def_in_, T, def_ids_.data(), def_wts_.data(), K, def_out_);
            defer_pending_ = true;
            defer_tokens_ = T;
        }
        if (remote) remote_->finish(host_y, T);
        if (peer) {
            peer_->collect();
            for (size_t i = 0; i < size_t(T) * H; ++i) host_y[i] += peer_->host_out[i];
        }
        bool peer_dev = false;   // the peer's sums on GPU 0, behind its event (the thread)
        if (peer_ && peer_->threaded) {
            peer_->wait_enqueued();
            posted.p = nullptr;
            if (peer_->busy) {
                peer_dev = true;
                if (dtiming_) ck(cudaEventRecord(peer_late_ev_[0], stream_), "peer late");
                ck(cudaStreamWaitEvent(stream_, peer_->done, 0), "peer done");
                if (dtiming_) { ck(cudaEventRecord(peer_late_ev_[1], stream_), "peer late"); peer_late_pending_ = true; }
            }
        }
        const auto x3 = std::chrono::steady_clock::now();
        xt_[0] += std::chrono::duration<double>(x1 - e0).count();   // routing bookkeeping + GPU enqueue (+ the lane when synchronous)
        xt_[1] += std::chrono::duration<double>(x2 - x1).count();   // the lane (its wait when async)
        xt_[2] += std::chrono::duration<double>(x3 - x2).count();   // the peer's collect and merge
        prof_[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
        if (defer_remote) ck(cudaMemsetAsync(moe_.p, 0, size_t(T) * H * 4, stream_), "experts (deferred)");
        else
        ck(cudaMemcpyAsync(moe_.p, host_y, size_t(T) * H * 4, cudaMemcpyHostToDevice, stream_), "experts");
        if (fuse_ && !hook) {   // the adds inside the post-mix (forward's tail)
            tail_fused_ = true;
            tail_peer_ = peer_dev;
            tail_gsum_ = any_gpu;
            return;
        }
        if (peer_dev) k::ds41_add(moe_.f(), peer_sum_.f(), moe_.f(), int(size_t(T) * H), stream_);   // lane + peer, as the host add
        if (any_gpu) k::ds41_add(moe_.f(), g_sum_.f(), moe_.f(), int(size_t(T) * H), stream_);
        if (hook) {
            observe("ffn_shexp", l, y_.f(), H);
            observe("ffn_moe_out", l, moe_.f(), H);
        }
        k::ds41_add(moe_.f(), y_.f(), y_.f(), int(size_t(T) * H), stream_);
        if (const char* path = std::getenv("STRATA_DS41_ORDER_TRACE"); path && *path) {
            if (T != 1 || H != 5120 || c.layers != 40 || fuse_ || hook || peer_ || remote_ || producer_.joinable())
                throw std::runtime_error("ds41: order trace requires one token and the bounded single-GPU path");
            std::vector<float> diagnostic(size_t(H) * 2);
            ck(cudaMemcpyAsync(diagnostic.data(), moe_.p, size_t(H) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "order trace moe");
            ck(cudaMemcpyAsync(diagnostic.data() + H, y_.p, size_t(H) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "order trace total");
            ck(cudaStreamSynchronize(stream_), "order trace sync");
            std::ofstream file(path, std::ios::binary | (l == 0 ? std::ios::trunc : std::ios::app));
            if (!file.write(reinterpret_cast<const char*>(pinned_[0]), size_t(H) * sizeof(float)) ||
                !file.write(reinterpret_cast<const char*>(diagnostic.data()), size_t(H) * 2 * sizeof(float)))
                throw std::runtime_error("ds41: order trace write failed");
        }
        // no sync: the next layer's router sync orders this upload before host_y is written again (two layers on)
    }

public:
    /// STRATA_DS41_ROUTES=path: every token's expert ids, int32 [layers][top_k] per token.
    void log_routes(const std::string& path) { routes_.open(path, std::ios::binary); }
    /// Seconds: [0] passes, [1] GPU work up to each layer's router (synchronous), [2] CPU experts, [3] adapt,
    /// [4] Engram gather, [5] indexer host, [6] head, [7] streamed experts' host copies.
    double prof_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double tier_hits_ = 0, tier_all_ = 0, promotions_ = 0;
    uint64_t canon_routes_ = 0, canon_cpu_routes_ = 0, canon_streamed_ = 0;   // STRATA_DS41_CANON_MIB: routes on the GPU / CPU routes seen / streamed
    uint64_t own_routes() const { return own_routes_; }
    void reset_own_routes() { own_routes_ = 0; }
    void reset_lane_stats() { if (lane_) lane_->async_reset(); }
    // the calling host thread onto this decoder's lane core 0 (two decoders in a process: each host thread is its lane's thread 0)
    void pin_host_thread() {
        if (!lane_ || lane_->caller_cpu() < 0) return;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(lane_->caller_cpu(), &one);
        pthread_setaffinity_np(pthread_self(), sizeof one, &one);
    }

private:
    const Decoder* borrow_ = nullptr;   // STRATA_DS41_DUAL: the first decoder, whose GPU weights and canonical tier this one views
    core::ModelArtifact art_;
    std::map<const GgufFile*, std::pair<void*, size_t>> shared_maps_;
    std::vector<std::pair<void*, size_t>> populate_;
    struct Src { const uint8_t *gate, *up, *down; };
    std::vector<Src> stream_src_;   // the experts in the shared mapping (the streamed prompt's copies)
    Config cfg_;
    int context_;
    bool tier_lent_ = false;
    Dev attn_info_dev_, attn_error_;
    size_t attn_info_host_bytes_ = 0;
    bool attn_bounds_check_ = std::getenv("STRATA_DS41_ATTN_BOUNDS_CHECK") &&
                              std::string(std::getenv("STRATA_DS41_ATTN_BOUNDS_CHECK")) == "1";
    int* attn_info_ = nullptr;   // pinned: per-token attention rows of a pass
    int rcap_min_ = 8192;
    int engram_tile_ = 0;   // STRATA_DS41_ENGRAM_TILE: Engram projection rows per call (0: the whole pass)
    int mix_tile_ = 0;      // STRATA_DS41_MIX_TILE: hyper-connection mix rows per call (0: the whole pass)
    int shexp_tile_ = 0;    // STRATA_DS41_SHEXP_TILE: shared-expert rows per call (0: the whole pass)
    int outa_tile_ = 0;     // STRATA_DS41_OUTA_TILE: output-projection rows per call (0: the whole pass)
    size_t sh_rows_ = 0;    // the shared expert's intermediate rows (SHEXP_TILE, capped by the tile)
    bool halves_ = false;   // STRATA_DS41_PREFILL_HALVES: trunk per tile, FFN once per layer
    int pftile_ = 4096;     // STRATA_DS41_PREFILL_TILE: the trunk's rows per call in halves mode
    int halves_skip_from_ = -1;   // STRATA_DS41_HALVES_SKIP_FROM=l: the merged pass truncates rows from layer l on (-1: skip_from)
    int sel_base_ = 0;            // the indexer selection's first row in the pass (0 outside the merged pass)
    int rope_off_ = 0;      // halves mode: the trunk tile's first token (rope tables, engram rows, cache rows)
    int engram_off_ = 0;    // halves mode: the trunk tile's first token in engram_rows_
    Dev ht_, h2t_, yt_, xnt_, coeff_ffn_full_;
    int attn_sub_ = [] { const char* v = std::getenv("STRATA_DS41_ATTN_SUB"); return v ? std::max(16, std::atoi(v)) : 256; }();
    std::vector<std::unique_ptr<void, void (*)(void*)>> pinned_bufs_;
    int chunk_ = 256, ring_ = 0, pos0_ = 0, tokens_ = 1, score_stride_ = 0, score_tile_ = 1;
    // batched sessions (sessions_init): per-layer [S][...] rings, caches, ratio-2 stashes; each session's tokens;
    // per pass the positions, window slots and cache rows [4][S] (pinned, and on the device)
    int sessions_ = 0, ring_b_ = 0;
    bool sess_ = false;   // a pass of sessions
    std::vector<Dev> bwin_, bcache_, bstash_v_, bstash_s_;
    std::vector<std::vector<int32_t>> shist_;
    Dev tpos_dev_;
    int* tpos_host_ = nullptr;
    // STRATA_DS41_KV=f16: the compressed caches in FP16 (half the bytes; the indexer reads them so, attention reads
    // each pass's selected rows gathered back to floats in kv_sel_ through the identity selection sel_iota_)
    // (=q8: Q8_0 rows, an FP16 scale per 32 values)
    const int kv_kind_ = [] { const char* v = std::getenv("STRATA_DS41_KV"); return !v ? 0 : std::string(v) == "f16" ? 1 : std::string(v) == "q8" ? 2 : 0; }();
    const bool kv_half_ = kv_kind_ != 0;   // not float storage
    size_t kv_row_bytes() const {
        const size_t rw = size_t(cfg_.head_dim + cfg_.index_dim);
        return kv_kind_ == 2 ? rw / 32 * 34 : rw * (kv_kind_ == 1 ? 2 : 4);
    }
    Dev kv_rows_, kv_sel_, sel_iota_;
    static constexpr int kGatherTokens = 256;
    cudaStream_t stream_ = nullptr;
    std::map<std::string, Weight> weights_;
    std::vector<std::vector<float>> bias_;
    // STRATA_DS41_GPU_ROUTE: the router's bias and the tier's slot pointers on the device ([layers][experts]), the
    // route table / ids / weights, their pinned host copies and the event the host waits on
    Dev bias_dev_, slot_dev_, route_table_, route_ids_, route_w_;
    unsigned long long* slot_up_ = nullptr;   // pinned staging of slot_dev_
    std::vector<unsigned long long> slot_last_;
    int* route_host_ = nullptr;               // pinned: ids [16 * K] then weights
    cudaEvent_t route_ev_ = nullptr, slot_ev_ = nullptr;
    std::vector<LaneLayer> experts_;
    std::unique_ptr<ExpertLane> lane_;
    std::vector<char> cut_;               // CPU_TOPK: routes dropped this pass
    uint64_t substituted_ = 0, dropped_ = 0, affinity_swaps_ = 0;   // GPU_SUBST / CPU_TOPK counts
    Dev attn_scores_, h_, h2_, x_, xn_, flat_, mixes_, coeff_attn_, coeff_ffn_, qa_, qr_, q_, kv_, kvn_, att_, oa_, oa_part_, y_, q8_, q8g_, cs_;
    Dev comp_v_, comp_s_, comp_, ik_, iq_, iw_, logits_, sh_gate_, sh_up_, sh_h_, moe_, scores_, engram_rows_, engram_kv_;
    std::vector<Dev> windows_, caches_, stash_v_, stash_s_, sel_;
    std::vector<std::vector<int>> sel_count_, candidates_;
    std::vector<const core::ArtifactTensor*> engram_tables_;   // per Engram layer (the same table when shared)
    std::vector<int> engram_fds_, owned_fds_;
    std::unique_ptr<EngramRemote> engram_remote_;
    struct EngramReq { int64_t row; float* dst; int table; };
    std::vector<EngramReq> engram_reqs_, engram_reqs_bg_;
    std::thread engram_later_;
    std::vector<uint8_t*> engram_ram_;      // STRATA_DS41_ENGRAM_RAM_GIB: the leading rows of each table in memory
    std::vector<int64_t> engram_ram_rows_;
    std::atomic<bool> engram_failed_{false};
    int engram_threads_ = 16;
    std::vector<int32_t> history_, route_log_;
    // Reused by the layer-serial MoE path instead of rebuilding route buffers
    // and the CPU/grouped-GPU routing scratch on every FFN call.
    std::vector<int> host_route_ids_;
    std::vector<float> host_route_weights_;
    std::vector<int> host_cpu_route_ids_, host_gpu_route_ids_, host_gpu_route_counts_;
    std::vector<float> host_gpu_route_weights_;
    std::vector<float> logits_host_;
    Dev verify_argmax_ids_dev_, verify_argmax_bad_dev_;
    std::vector<int> verify_argmax_host_;
    int verify_argmax_bad_host_ = 0;
    float* pinned_[3] = {nullptr, nullptr, nullptr};   // the CPU lane's input (and the embeddings) and its two outputs
    std::ofstream routes_;
    std::vector<float> engram_host_, rope_host_;   // the uploads' sources outlive a pass
    // the GPU expert tier
    Dev arena_;
    std::vector<Slot> slots_;
    std::vector<std::vector<int>> slot_;   // [layer][expert] -> slot, or -1
    std::vector<float> score_;
    std::vector<int> used_, missed_;
    std::vector<float> rt_count_;   // STRATA_DS41_REBALANCE: routes per expert since the last rebalance
    std::vector<uint8_t*> staging_;
    cudaStream_t copy_ = nullptr;
    std::unique_ptr<std::atomic<bool>[]> recorded_;
    std::deque<Job> jobs_;
    std::mutex jobs_mutex_;
    std::condition_variable jobs_cv_;
    bool stop_uploads_ = false;
    std::thread uploader_;
    int promote_ = 0;
    float decay_ = 0.995f;
    Dev gx_q8_, g_gate_, g_up_, g_h_, g_h_q8_, g_out_, g_sum_;
    // trunk GEMMs (long chunks)
    int gemm_min_ = 32;
    cublasHandle_t blas_ = nullptr;
    std::unique_ptr<strata::prefill::Gemm> gemm_;
    Dev xh_;   // FP16 / BF16 activations of a chunk
    Dev q8x_;  // xn_ quantized once for the GEMVs of a sublayer that all read it (STRATA_DS41_SHARE_Q8)
    // STRATA_DS41_FUSE=1: fewer launches per layer, the same arithmetic (bit-identical): one quantize over all of a
    // call's columns, the grouped attention output in one launch, norm + Q8 in one kernel (q8x_ready_: q8x_ holds the
    // current xn_), the router and shared expert before the router sync, the FFN tail's adds inside hc_post, the
    // attention info uploaded once per (ratio, positions) of a pass
    const bool fuse_ = std::getenv("STRATA_DS41_FUSE") && std::string(std::getenv("STRATA_DS41_FUSE")) == "1";
    bool q8x_ready_ = false;
    bool tail_fused_ = false, tail_peer_ = false, tail_gsum_ = false;
    long attn_info_key_ = -1;
    // streamed experts (long chunks)
    int stream_min_ = 1024;
    int stage_threads_ = 8;
    // pipelined streaming
    std::vector<StreamItem> items_;
    std::vector<int> item0_;
    std::vector<Dev> dring_;
    std::vector<uint8_t*> hring_;
    std::vector<cudaEvent_t> ev_ready_, ev_free_, ev_dma_;
    int dslots_ = 0, hslots_ = 3;
    size_t slot_bytes_ = 0;
    std::thread producer_;
    std::mutex sm_;
    std::condition_variable scv_;
    int produced_ = 0, consumed_ = 0, staged_ = 0, dma_issued_ = 0;
    // STRATA_DS41_PREFILL_HYBRID=m: per layer, the experts the producer copies (need_[l][e]; need_ready_[l] once routed)
    std::vector<std::vector<char>> need_;
    std::vector<char> need_ready_;
    double hybrid_cpu_s_ = 0, hybrid_skipped_gb_ = 0;
    std::vector<int32_t> hybrid_ids_;
    Dev hybrid_dev_;
    std::vector<float> hybrid_w_;
    std::chrono::steady_clock::time_point hybrid_t0_;
    std::thread stager_, fetcher_;
    // the worker's rows of streamed items: registered pinned buffers, their uploads' events
    std::vector<uint8_t*> rbuf_;
    std::vector<ucomm_mr_t*> rbuf_regions_;
    std::vector<cudaEvent_t> ev_rdma_;
    Dev rscratch_;   // VRAM: one item's worker blobs, before they are placed (copy_s_ orders its reuse)
    int fetched_ = 0, rdma_issued_ = 0;
public:
    double fetch_s_ = 0;
private:
    std::vector<int> unpinned_;   // the pass's items of unpinned layers, in order
    bool stop_stream_ = false;
    bool pinned_tried_ = false;
    // the GPU indexer: per index-source layer, counts [chunk]; candidate flags [chunk][cand_stride_]
    std::vector<Dev> sel_cnt_;
    std::vector<char> sel_on_gpu_;
    Dev cand_dev_;
    int cand_stride_ = 1;
    bool gpu_sel_ = false;
    int skip_ = 0;   // tokens dropped by enc-skip in this pass
    // two nodes: this node's FFN rows [0, fl_) of each routed expert (fl_ = ff alone), the full formats for the
    // streamed prompt, the worker
    int fl_ = 0;
    std::vector<cpu::NativeFmt> full_;
    std::string remote_addr_;
    std::unique_ptr<Remote> remote_;
    bool remote_gpu_ = false;   // the worker runs its rows of streamed passes (else they are fetched back)
    // STRATA_DS41_RAM_COPY residency
    bool ram_copy_ = false;
    std::vector<std::vector<char>> where_;   // [layer][expert]: 0 host memory, 1 GPU0, 2 the peer GPU (empty: no residency)
    std::vector<std::vector<int>> order_;    // [layer] positions -> expert: streamed experts first, then GPU0-resident
    std::vector<int> n_stream_;              // [layer] streamed positions
    std::vector<uint8_t*> res0_base_;
    std::vector<size_t> res0_stride_;
    std::unique_ptr<PeerGpu> peer_;
    Dev peer_sum_;
    cudaEvent_t peer_late_ev_[2] = {nullptr, nullptr};   // PEER_THREAD + DECODE_TIMING: GPU 0's wait for the peer per layer
    cudaEvent_t router_ev_ = nullptr;                     // FUSE: the router's copies done (the shared expert queued behind)
    bool peer_late_pending_ = false;
    double peer_late_ms_ = 0;
    std::vector<std::pair<void*, size_t>> ram_maps_;
    std::vector<uint8_t> embd_copy_;   // STRATA_DS41_RAM_COPY: token_embd in memory
    bool packed_on_ = false;           // STRATA_DS41_RAM_PINNED
    std::vector<uint8_t*> packed_;     // [layer] pinned RAM experts by position (positions [0, n_stream_))
    std::vector<std::vector<int>> ram_pos_;   // [layer][expert] -> position in packed_, or -1
    std::vector<int> s0_;              // [layer] positions this GPU streams; [s0_, n_stream_) the peer's
    double peer_frac_ = 0;
    unsigned long numa_[2] = {0, 0};   // mbind masks of GPU0's / the peer's socket
    bool packed_mapped_ = false;
    bool engram_host_pinned_ = false;       // packed_ is mmap'ed and registered (else cudaHostAlloc)
    Dev remote_sum_;            // its partial sums of a streamed pass, uploaded
    std::vector<void*> registered_;
    std::vector<char> pinned_layer_, gu_pin_, down_pin_;   // pinned_layer_: both parts (no host copy)
public:
    double stream_wait_s_ = 0;
    long skipped_ = 0;
    double peer_prep_s_ = 0, peer_wait_s_ = 0, stream_enqueue_s_ = 0, attn_host_s_ = 0, ffn_host_s_ = 0, route_wait_s_ = 0, before_loop_s_ = 0, loop_s_ = 0;   // streamed passes: host time per part
    // STRATA_DS41_GPU_TIMING: per-layer GPU events of streamed passes
    // ---- DSpark speculative decoding (STRATA_DS41_DSPARK=draft.gguf, llama.cpp's dflash arch, DSV4 stages) ----
    // The drafter: 3 DSV4.1 blocks (lagged hyper-connections, window attention over its own ring, MoE with MXFP4
    // experts), the target's token embeddings and lm head, and a Markov bias chained over the block. Its ring holds the
    // keys of committed positions, projected from the target's layer-input means at dsp_.targets (fc, norm, attn_kv).
    // A step drafts [anchor, MASK...] at positions pos.., the target verifies [anchor, drafts] in one pass, the
    // accepted prefix and the target's next token commit, and the decoder rewinds past the rest.
    static constexpr int kFeatRing = 256;   // target feature positions kept (>= window + a pass)
    static constexpr int kInj = 128;        // positions per injection call
    struct Dspark {
        std::unique_ptr<GgufFile> file;
        int layers = 0, block = 5, draft_limit = 5, ring = 0, experts = 0, top_k = 0, sinkhorn = 20, mask = 128799, markov = 0;
        float scale = 1.5f, clamp = 10, clamp_sh = 10, eps = 1e-20f, hc_eps = 1e-6f;
        bool normalize = true;
        std::vector<int> targets;
        std::vector<std::vector<float>> bias;
        struct Ex { const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr; size_t gu_row = 0, d_row = 0; Dev data; };
        std::vector<Ex> ex;
        // With DSPARK_STREAM_EXPERTS the immutable expert tensors stay mapped on the host. One guarded
        // GPU buffer is reused in stream order for the currently selected expert.
        bool stream_experts = false, batch_stage = false, pin_experts = false, cache_experts = false;
        Dev expert_stage;
        uint8_t* expert_host_stage = nullptr;
        size_t expert_host_stage_bytes = 0, expert_stage_stride = 0, expert_stage_groups = 0;
        struct CacheSlot { int key = -1; uint64_t age = 0; };
        std::vector<CacheSlot> expert_cache;
        size_t expert_cache_slots = 0;
        uint64_t expert_cache_clock = 0, expert_cache_hits = 0, expert_cache_misses = 0;
        struct HostRegistrations {
            std::vector<void*> ranges;
            ~HostRegistrations() { for (void* p : ranges) cudaHostUnregister(p); }
        } expert_registrations;
        std::vector<Dev> rings;
        std::vector<uint16_t> w1;   // markov_w1 rows, host
        std::vector<float> conf;    // conf_proj: [hidden + markov] (sigmoid of its dot with [pre-norm hidden; w1 row])
        float* hx = nullptr;        // pinned: the block's pre-norm hidden rows
        size_t hx_bytes = 0;
        int* info = nullptr;        // pinned: the block's attention info (ds41_attention_multi)
        size_t info_bytes = 0;
        Dev info_dev;
        std::vector<Dev> bias_dev;  // STRATA_DS41_DSPARK_GPU_ROUTE: router bias per layer, routes, pair -> token
        Dev rids, rw, ptok;
        Dev w1_dev, conf_dev, dtok, dconf, head_error;   // device Markov head and its bounds/non-finite status
        int* htok = nullptr;
        size_t htok_bytes = 0;
        Dev fring, feat, fc, g, h, h2, x, xn, flat, mix, ca, cf, qa, qr, q, kv, kvn, att, oa, y, rl, sg, su, sh, eg, eu, eh, ed, moe, logits, cs, rowf, bvec;
        float* host = nullptr;      // pinned scratch
        size_t host_bytes = 0;
        bool on = false;
        long drafted = 0, accepted = 0, steps = 0;
        double draft_s = 0, verify_s = 0, inject_s = 0, draft_layers_s = 0, draft_route_s = 0, draft_head_s = 0, lookup_s = 0;
        std::array<std::deque<int>, 5> cache_sim;
        std::array<uint64_t, 5> cache_hits{};
        uint64_t cache_accesses = 0;
        // STRATA_DS41_DSPARK_SCHED: verify ms per pass width T (running means, weight 0.2; 0 = not seen), the draft's ms
        double verify_ms[16] = {}, draft_ms = 0;
        long sched_cut = 0;   // tokens the schedule left unverified
        long lookup_steps = 0, lookup_drafted = 0, lookup_accepted = 0;
    } dsp_;
    std::vector<Dev> spec_v_, spec_s_;
    std::array<std::vector<int>, 8> spec_used_, spec_missed_;
    Dev dq8_;   // the drafter's quantized activations (STRATA_DS41_DSPARK_Q8)   // verify passes: each position's compressor value/score (ratio-2 KV sources)
    bool spec_save_ = false;

    // a BF16 / F32 drafter weight times n tokens (x rows ldx apart), y rows ldy apart; BF16 reads each row once per 8 tokens
    void dmat(const std::string& name, const float* x, int ldx, int n, float* y, int ldy, int row0 = 0, int rows = 0) {
        const Weight& W = weight(name);
        if (!rows) rows = W.out;
        if (!x || !y || n < 1 || n > kInj || W.in < 1 || ldx < W.in || row0 < 0 || rows < 1 ||
            row0 > W.out || rows > W.out - row0 || ldy < rows || !W.data.p ||
            W.row_bytes == 0 || size_t(W.out) > W.data.bytes / W.row_bytes)
            throw std::runtime_error("ds41: drafter matrix bounds: " + name);
        const uint8_t* base = static_cast<const uint8_t*>(W.data.p) + size_t(row0) * W.row_bytes;
        for (int t0 = 0; t0 < n; t0 += 8) {
            const int m = std::min(8, n - t0);
            int idx[8];
            for (int t = 0; t < m; ++t) idx[t] = t0 + t;
            if (W.type == GGML_TYPE_BF16)
                k::ds41_bf16_gemv_mt(reinterpret_cast<const uint16_t*>(base), x, ldx, idx, m, y + size_t(t0) * ldy, ldy, rows, W.in, stream_);
            else if (W.type == GGML_TYPE_F32 && ldx == W.in && ldy == rows)
                k::ds41_f32_gemv(reinterpret_cast<const float*>(base), x + size_t(t0) * ldx, y + size_t(t0) * ldy, rows, W.in, stream_, m);
            else if (k::native_mmvq_supported(W.type) && ldx == W.in && ldy == rows) {   // STRATA_DS41_DSPARK_Q8
                if (dq8_.bytes < k::native_q8_1_bytes(W.in, 1) * 8) dq8_ = Dev(k::native_q8_1_bytes(W.in, 1) * 8);
                quantize(x + size_t(t0) * ldx, dq8_.p, W.in, m);
                mmvq(W.type, base, dq8_.p, y + size_t(t0) * ldy, W.in, rows, m);
            }
            else
                throw std::runtime_error("ds41: drafter weight type " + name);
        }
    }

    void load_dspark(const std::string& path) {
        const auto& c = cfg_;
        auto& d = dsp_;
        const auto t0 = std::chrono::steady_clock::now();
        d.file = std::make_unique<GgufFile>(path);
        const GgufFile& f = *d.file;
        auto num = [&](const std::string& k, double fb) { const auto* v = f.get("dflash." + k); return v ? (v->is_num() ? v->num() : v->items.empty() ? fb : v->items[0].num()) : fb; };
        d.layers = int(num("block_count", 3));
        d.block = int(num("block_size", 5));
        d.draft_limit = strata_env_is_one("STRATA_DS41_DSPARK_EXTEND") ? 7 : d.block;
        d.experts = int(num("expert_count", 128));
        d.top_k = int(num("expert_used_count", 3));
        d.scale = float(num("expert_weights_scale", 1.5));
        d.clamp = float(num("swiglu_clamp_exp", 10));
        d.clamp_sh = float(num("swiglu_clamp_shexp", 10));
        d.eps = float(num("attention.layer_norm_rms_epsilon", 1e-20));
        d.sinkhorn = int(num("hyper_connection.sinkhorn_iterations", 20));
        d.hc_eps = float(num("hyper_connection.epsilon", 1e-6));
        if (const char* m = std::getenv("STRATA_DS41_DSPARK_MASK")) d.mask = std::atoi(m);
        d.stream_experts = strata_env_is_one("STRATA_DS41_DSPARK_STREAM_EXPERTS");
        d.batch_stage = strata_env_is_one("STRATA_DS41_DSPARK_BATCH_STAGE");
        d.pin_experts = strata_env_is_one("STRATA_DS41_DSPARK_PIN_EXPERTS");
        d.cache_experts = strata_env_is_one("STRATA_DS41_DSPARK_CACHE_EXPERTS");
        if (d.cache_experts) {
            d.expert_cache_slots = size_t(dspark_cache_slots());
            d.expert_cache.resize(d.expert_cache_slots);
        }
        if (d.layers < 1 || d.layers > c.layers || d.block < 1 || d.block > 7 || d.experts < 1 || d.experts > 1024 ||
            d.top_k < 1 || d.top_k > std::min(d.experts, 16) || d.mask < 0 || d.mask >= c.vocab)
            throw std::runtime_error("ds41: invalid DSpark block, layer, expert, top-k, or mask bounds");
        if (d.layers > INT32_MAX / d.experts) throw std::runtime_error("ds41: DSpark expert key overflow");
        if (d.stream_experts && strata_env_is_one("STRATA_DS41_DSPARK_GPU_ROUTE"))
            throw std::runtime_error("ds41: streamed DSpark experts require host routing");
        if (d.batch_stage && !d.stream_experts) throw std::runtime_error("ds41: DSpark batch staging requires streamed experts");
        if (d.pin_experts && !d.batch_stage) throw std::runtime_error("ds41: DSpark expert pinning requires batch staging");
        if (d.cache_experts && (!d.batch_stage || d.pin_experts || size_t(d.draft_limit) * size_t(d.top_k) > d.expert_cache_slots))
            throw std::runtime_error("ds41: DSpark expert cache requires bounded unpinned batch staging");
#if defined(__HIP_PLATFORM_AMD__)
        // Registered file-backed mappings have coincided with GPU permission
        // faults and sustained reclaim pressure on AMD. Fail closed until the
        // faulting path is isolated and validated; use bounded host staging.
        if (d.pin_experts)
            throw std::runtime_error("ds41: DSpark mapped expert pinning is disabled on AMD; use batch staging without pinning");
#endif
        for (int64_t t : f.int_array("dflash.target_layers")) {
            if (t < 0 || t >= c.layers) throw std::runtime_error("ds41: DSpark target layer out of bounds");
            d.targets.push_back(int(t));
        }
        if (d.targets.empty() || d.targets.size() > size_t(c.layers)) throw std::runtime_error("ds41: invalid DSpark target layers");
        if (c.hidden <= 0 || c.hidden > INT32_MAX / 4 || c.ff <= 0 || c.q_rank <= 0 || c.head_dim <= 0 ||
            c.heads <= 0 || c.out_groups <= 0 || c.out_rank <= 0 || c.window <= 0 || c.window > INT32_MAX - 16 ||
            c.heads % c.out_groups != 0 || int64_t(c.heads) * c.head_dim > INT32_MAX ||
            int64_t(c.out_groups) * c.out_rank > INT32_MAX ||
            d.targets.size() > size_t(INT32_MAX / c.hidden) || c.rope_dim < 0 || c.rope_dim > c.head_dim ||
            c.rope_dim > 128 || c.rope_dim % 2 != 0)
            throw std::runtime_error("ds41: invalid target geometry for DSpark");
        if (int(num("embedding_length", 0)) != c.hidden || int(num("attention.head_count", 0)) != c.heads ||
            int(num("attention.key_length", 0)) != c.head_dim || int(num("attention.q_lora_rank", 0)) != c.q_rank ||
            int(num("attention.output_group_count", 0)) != c.out_groups || int(num("attention.output_lora_rank", 0)) != c.out_rank ||
            int(num("expert_feed_forward_length", 0)) != c.ff || int(num("attention.sliding_window", 0)) != c.window)
            throw std::runtime_error("ds41: the DSpark draft's geometry differs from the target's");
        std::map<std::string, const TensorInfo*> ts;
        for (const auto& t : f.tensors()) ts[t.name] = &t;
        auto checked = [&](const std::string& name) -> const TensorInfo& {
            const auto it = ts.find(name);
            if (it == ts.end()) throw std::runtime_error("ds41: DSpark tensor missing: " + name);
            const auto& t = *it->second;
            if (f.data_start() > f.file_size()) throw std::runtime_error("ds41: DSpark data start outside file");
            const uint64_t bytes = tensor_payload_bytes(t), payload = f.file_size() - f.data_start();
            if (!bytes || t.offset > payload || bytes > payload - t.offset)
                throw std::runtime_error("ds41: DSpark tensor outside file: " + name);
            return t;
        };
        const auto& fc = checked("fc.weight");
        const auto& w1 = checked("markov_w1.weight");
        const auto& w2 = checked("markov_w2.weight");
        if (fc.shape != std::vector<uint64_t>{uint64_t(d.targets.size()) * uint64_t(c.hidden), uint64_t(c.hidden)} ||
            w1.type != GGML_TYPE_BF16 || w1.shape.size() != 2 || w1.shape[0] == 0 || w1.shape[0] > uint64_t(INT32_MAX) ||
            w1.shape[1] != uint64_t(c.vocab) || w2.shape != w1.shape)
            throw std::runtime_error("ds41: invalid DSpark feature or Markov tensor shape");
        // Validate every matrix and raw float vector consumed by the draft
        // kernels before uploading weights or launching the first draft.
        auto matrix = [&](const std::string& name, int in, int out) {
            const auto& t = checked(name);
            if (in <= 0 || out <= 0 || t.shape.empty() || t.shape[0] != uint64_t(in))
                throw std::runtime_error("ds41: DSpark matrix input shape: " + name);
            uint64_t rows = 1;
            for (size_t i = 1; i < t.shape.size(); ++i) {
                if (!t.shape[i] || t.shape[i] > uint64_t(out) / rows)
                    throw std::runtime_error("ds41: DSpark matrix output shape: " + name);
                rows *= t.shape[i];
            }
            if (rows != uint64_t(out) ||
                (t.type != GGML_TYPE_F32 && t.type != GGML_TYPE_BF16 && !k::native_mmvq_supported(int(t.type))) ||
                (t.type == GGML_TYPE_BF16 && in % 64))
                throw std::runtime_error("ds41: DSpark matrix shape or type: " + name);
        };
        auto float_vector = [&](const std::string& name, size_t count) {
            const auto& t = checked(name);
            if (t.type != GGML_TYPE_F32 || tensor_payload_bytes(t) < count * sizeof(float))
                throw std::runtime_error("ds41: DSpark float vector shape or type: " + name);
        };
        matrix("fc.weight", int(d.targets.size()) * c.hidden, c.hidden);
        matrix("markov_w2.weight", int(w1.shape[0]), c.vocab);
        float_vector("enc.output_norm.weight", size_t(c.hidden));
        float_vector("output_norm.weight", size_t(c.hidden));
        for (int l = 0; l < d.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            matrix(p + "attn_q_a.weight", c.hidden, c.q_rank);
            matrix(p + "attn_q_b.weight", c.q_rank, c.heads * c.head_dim);
            matrix(p + "attn_kv.weight", c.hidden, c.head_dim);
            matrix(p + "attn_output_a.weight", c.heads / c.out_groups * c.head_dim, c.out_groups * c.out_rank);
            matrix(p + "attn_output_b.weight", c.out_groups * c.out_rank, c.hidden);
            matrix(p + "ffn_gate_inp.weight", c.hidden, d.experts);
            matrix(p + "ffn_gate_shexp.weight", c.hidden, c.ff);
            matrix(p + "ffn_up_shexp.weight", c.hidden, c.ff);
            matrix(p + "ffn_down_shexp.weight", c.ff, c.hidden);
            for (const char* h : {"hc_attn_", "hc_ffn_"}) {
                matrix(p + h + "fn.weight", 4 * c.hidden, 24);
                float_vector(p + h + "base.weight", 24);
                float_vector(p + h + "scale.weight", 3);
            }
            float_vector(p + "attn_sinks.weight", size_t(c.heads));
            float_vector(p + "attn_norm.weight", size_t(c.hidden));
            float_vector(p + "attn_q_a_norm.weight", size_t(c.q_rank));
            float_vector(p + "attn_kv_a_norm.weight", size_t(c.head_dim));
            float_vector(p + "ffn_norm.weight", size_t(c.hidden));
        }
        size_t bytes = 0;
        auto put = [&](const std::string& name) {
            const TensorInfo& t = checked(name);
            if (t.shape.empty() || t.shape[0] == 0 || t.shape[0] > uint64_t(INT32_MAX))
                throw std::runtime_error("ds41: DSpark tensor dimension: " + name);
            Weight W;
            W.type = int(t.type);
            W.in = int(t.shape[0]);
            W.out = 1;
            for (size_t i = 1; i < t.shape.size(); ++i) {
                if (!t.shape[i] || t.shape[i] > uint64_t(INT32_MAX / W.out))
                    throw std::runtime_error("ds41: DSpark tensor dimension: " + name);
                W.out *= int(t.shape[i]);
            }
            W.row_bytes = ggml_row_size(ggml_type(W.type), W.in);
            // STRATA_DS41_DSPARK_Q8=1: the BF16 matrices (but the grouped output projection) as Q8_0, read by native MMVQ
            static const bool q8 = std::getenv("STRATA_DS41_DSPARK_Q8") && std::string(std::getenv("STRATA_DS41_DSPARK_Q8")) == "1";
            if (q8 && W.type == GGML_TYPE_BF16 && W.out >= 256 && W.in % 32 == 0 && name.find("attn_output_a") == std::string::npos) {
                const auto v = to_float(GGML_TYPE_BF16, f.tensor_data(t), int64_t(W.in) * W.out);
                std::vector<uint8_t> qd(ggml_row_size(GGML_TYPE_Q8_0, W.in) * size_t(W.out));
                ggml_quantize_chunk(GGML_TYPE_Q8_0, v.data(), qd.data(), 0, W.out, W.in, nullptr);
                W.type = GGML_TYPE_Q8_0;
                W.row_bytes = ggml_row_size(GGML_TYPE_Q8_0, W.in);
                W.data = Dev(qd.size());
                ck(cudaMemcpy(W.data.p, qd.data(), qd.size(), cudaMemcpyHostToDevice), "dspark upload");
                bytes += qd.size();
                weights_.emplace("dspark." + name, std::move(W));
                return;
            }
            const size_t n = W.row_bytes * size_t(W.out);
            W.data = Dev(n);
            ck(cudaMemcpy(W.data.p, f.tensor_data(t), n, cudaMemcpyHostToDevice), "dspark upload");
            bytes += n;
            weights_.emplace("dspark." + name, std::move(W));
        };
        for (const char* n : {"fc.weight", "enc.output_norm.weight", "output_norm.weight", "markov_w2.weight"}) put(n);
        for (int l = 0; l < d.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            for (const char* n : {"attn_sinks.weight", "attn_kv_a_norm.weight", "attn_q_a_norm.weight", "attn_kv.weight", "attn_output_a.weight",
                                  "attn_output_b.weight", "attn_q_a.weight", "attn_q_b.weight", "attn_norm.weight", "ffn_gate_inp.weight",
                                  "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight", "ffn_norm.weight", "hc_attn_base.weight",
                                  "hc_attn_fn.weight", "hc_attn_scale.weight", "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight"})
                put(p + n);
            const auto& b = checked(p + "exp_probs_b.bias");
            if (b.shape != std::vector<uint64_t>{uint64_t(d.experts)}) throw std::runtime_error("ds41: DSpark router bias shape");
            d.bias.push_back(to_float(int(b.type), f.tensor_data(b), d.experts));
            Dspark::Ex e;
            const auto &G = checked(p + "ffn_gate_exps.weight"), &U = checked(p + "ffn_up_exps.weight"), &D = checked(p + "ffn_down_exps.weight");
            if (G.type != 39 || U.type != 39 || D.type != 39) throw std::runtime_error("ds41: DSpark experts must be MXFP4");
            const std::vector<uint64_t> gu_shape{uint64_t(c.hidden), uint64_t(c.ff), uint64_t(d.experts)};
            const std::vector<uint64_t> down_shape{uint64_t(c.ff), uint64_t(c.hidden), uint64_t(d.experts)};
            if (G.shape != gu_shape || U.shape != gu_shape || D.shape != down_shape)
                throw std::runtime_error("ds41: DSpark expert tensor shape");
            e.gu_row = ggml_row_size(GGML_TYPE_MXFP4, c.hidden);
            e.d_row = ggml_row_size(GGML_TYPE_MXFP4, c.ff);
            if (e.gu_row > SIZE_MAX / size_t(c.ff) || e.d_row > SIZE_MAX / size_t(c.hidden))
                throw std::runtime_error("ds41: DSpark expert row size overflow");
            const size_t ge = e.gu_row * size_t(c.ff), de = e.d_row * size_t(c.hidden);
            if (ge > SIZE_MAX / size_t(d.experts) || de > SIZE_MAX / size_t(d.experts))
                throw std::runtime_error("ds41: DSpark expert tensor size overflow");
            const size_t gb = ge * size_t(d.experts), db = de * size_t(d.experts);
            if (gb > tensor_payload_bytes(G) || gb > tensor_payload_bytes(U) || db > tensor_payload_bytes(D))
                throw std::runtime_error("ds41: DSpark expert tensor capacity bounds");
            if (d.stream_experts) {
                e.gate = f.tensor_data(G); e.up = f.tensor_data(U); e.down = f.tensor_data(D);
                if (!d.expert_stage.p) {
                    if (ge > (SIZE_MAX - de) / 2) throw std::runtime_error("ds41: DSpark expert stage size overflow");
                    d.expert_stage_stride = 2 * ge + de;
                    d.expert_stage_groups = d.batch_stage ? size_t(d.draft_limit) * size_t(d.top_k) : 1;
                    const size_t slots = d.cache_experts ? d.expert_cache_slots : d.expert_stage_groups;
                    if (!slots || d.expert_stage_stride > SIZE_MAX / slots ||
                        (d.batch_stage && d.expert_stage_stride > SIZE_MAX / d.expert_stage_groups))
                        throw std::runtime_error("ds41: DSpark expert staging size overflow");
                    d.expert_stage = Dev(d.expert_stage_stride * slots);
                    if (d.batch_stage && !d.pin_experts) {
                        d.expert_host_stage_bytes = d.expert_stage_stride * d.expert_stage_groups;
                        d.expert_host_stage = reinterpret_cast<uint8_t*>(pinned_keep(d.expert_host_stage_bytes));
                    }
                    bytes += d.expert_stage.bytes;
                }
            } else {
                if (gb > (SIZE_MAX - db) / 2) throw std::runtime_error("ds41: DSpark expert device size overflow");
                e.data = Dev(2 * gb + db);
                ck(cudaMemcpy(e.data.b(), f.tensor_data(G), gb, cudaMemcpyHostToDevice), "dspark experts");
                ck(cudaMemcpy(e.data.b(gb), f.tensor_data(U), gb, cudaMemcpyHostToDevice), "dspark experts");
                ck(cudaMemcpy(e.data.b(2 * gb), f.tensor_data(D), db, cudaMemcpyHostToDevice), "dspark experts");
                e.gate = e.data.b(); e.up = e.data.b(gb); e.down = e.data.b(2 * gb);
                bytes += 2 * gb + db;
            }
            d.ex.push_back(std::move(e));
        }
        if (d.pin_experts) {
            const auto tpin = std::chrono::steady_clock::now();
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) throw std::runtime_error("ds41: cannot determine page size for DSpark expert pinning");
            const uintptr_t page = uintptr_t(page_size);
            std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
            for (const auto& e : d.ex) {
                const size_t gb = e.gu_row * size_t(c.ff) * size_t(d.experts);
                const size_t db = e.d_row * size_t(c.hidden) * size_t(d.experts);
                for (const auto& item : {std::pair<const uint8_t*, size_t>{e.gate, gb}, {e.up, gb}, {e.down, db}}) {
                    const uintptr_t a = reinterpret_cast<uintptr_t>(item.first);
                    if (!page || a > UINTPTR_MAX - item.second || a + item.second > UINTPTR_MAX - (page - 1))
                        throw std::runtime_error("ds41: DSpark expert pin range overflow");
                    ranges.push_back({a / page * page, (a + item.second + page - 1) / page * page});
                }
            }
            std::sort(ranges.begin(), ranges.end());
            std::vector<std::pair<uintptr_t, uintptr_t>> merged;
            for (const auto& r : ranges) {
                if (!merged.empty() && r.first <= merged.back().second) merged.back().second = std::max(merged.back().second, r.second);
                else merged.push_back(r);
            }
            size_t pinned_bytes = 0;
            // Register in bounded pieces so sustained reclaim pressure can stop
            // pinning before systemd-oomd kills the process. MemAvailable alone
            // includes reclaimable cache and does not reflect this pressure.
#if defined(__linux__)
            auto pressure = [](const char* field) {
                std::ifstream in("/proc/pressure/memory");
                std::string line;
                while (std::getline(in, line)) {
                    if (line.compare(0, 5, "full ") != 0) continue;
                    const std::string key = std::string(field) + "=";
                    const size_t p = line.find(key);
                    if (p != std::string::npos) return std::stod(line.substr(p + key.size()));
                }
                throw std::runtime_error("ds41: cannot read memory pressure for DSpark expert pinning");
            };
            if (pressure("avg60") > 10.0)
                throw std::runtime_error("ds41: memory pressure too high to pin DSpark experts");
#endif
            const size_t piece = std::max<size_t>(page, ((size_t(64) << 20) / page) * page);
            size_t pieces = 0;
            for (const auto& [a, b] : merged) {
                const size_t bytes = b - a;
                const size_t count = bytes / piece + (bytes % piece != 0);
                if (count > SIZE_MAX - pieces) throw std::runtime_error("ds41: DSpark pin count overflow");
                pieces += count;
            }
            d.expert_registrations.ranges.reserve(pieces);
            for (const auto& [a, b] : merged) {
                for (uintptr_t p = a; p < b;) {
#if defined(__linux__)
                    if (pressure("avg10") > 10.0)
                        throw std::runtime_error("ds41: memory pressure rose while pinning DSpark experts");
#endif
                    const size_t n = std::min<size_t>(b - p, piece);
                    ck(cudaHostRegister(reinterpret_cast<void*>(p), n, 0x08 /* read-only */), "dspark expert pin");
                    d.expert_registrations.ranges.push_back(reinterpret_cast<void*>(p));
                    pinned_bytes += n;
                    p += n;
                }
            }
            std::cerr << "DS41 dspark pinned_GiB=" << double(pinned_bytes) / double(1ull << 30) << " s="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - tpin).count() << "\n";
        }
        {   // markov_w1: rows by token, kept on the host (one row per drafted position)
            const auto& t = w1;
            d.markov = int(t.shape[0]);
            d.w1.resize(size_t(t.shape[0]) * t.shape[1]);
            std::memcpy(d.w1.data(), f.tensor_data(t), d.w1.size() * 2);
        }
        if (std::getenv("STRATA_DS41_DSPARK_DEVICE_HEAD") && std::string(std::getenv("STRATA_DS41_DSPARK_DEVICE_HEAD")) == "1") {
            d.w1_dev = Dev(d.w1.size() * 2);
            ck(cudaMemcpy(d.w1_dev.p, d.w1.data(), d.w1.size() * 2, cudaMemcpyHostToDevice), "markov w1");
            bytes += d.w1.size() * 2;
        }
        if (const auto it = ts.find("conf_proj.weight"); it != ts.end()) {
            const auto& t = checked("conf_proj.weight");
            d.conf = to_float(int(t.type), f.tensor_data(t), int64_t(t.shape[0]));
            if (int(d.conf.size()) != c.hidden + d.markov) d.conf.clear();
        }
        const size_t H = size_t(c.hidden), B = 8, nt = std::max<size_t>(B, kInj);
        d.ring = c.window + 2 * B;
        for (int l = 0; l < d.layers; ++l) d.rings.emplace_back(size_t(d.ring) * c.head_dim * 4);
        d.fring = Dev(size_t(kFeatRing) * d.targets.size() * H * 4);
        d.feat = Dev(nt * d.targets.size() * H * 4);
        d.fc = Dev(nt * H * 4);
        d.g = Dev(nt * H * 4);
        d.kv = Dev(nt * c.head_dim * 4);
        d.kvn = Dev(nt * c.head_dim * 4);
        d.cs = Dev(nt * 2 * 64 * 4);
        d.h = Dev(B * 4 * H * 4); d.h2 = Dev(B * 4 * H * 4);
        d.x = Dev(B * H * 4); d.xn = Dev(B * H * 4); d.flat = Dev(B * 4 * H * 4);
        d.mix = Dev(B * 24 * 4); d.ca = Dev(B * 24 * 4); d.cf = Dev(B * 24 * 4);
        d.qa = Dev(B * c.q_rank * 4); d.qr = Dev(B * c.q_rank * 4);
        d.q = Dev(B * size_t(c.heads) * c.head_dim * 4); d.att = Dev(B * size_t(c.heads) * c.head_dim * 4);
        d.oa = Dev(B * size_t(c.out_groups) * c.out_rank * 4); d.y = Dev(B * H * 4);
        d.rl = Dev(B * d.experts * 4);
        d.sg = Dev(B * c.ff * 4); d.su = Dev(B * c.ff * 4); d.sh = Dev(B * c.ff * 4);
        const size_t P = B * size_t(d.top_k);
        d.eg = Dev(P * c.ff * 4); d.eu = Dev(P * c.ff * 4); d.eh = Dev(P * c.ff * 4); d.ed = Dev(P * H * 4);
        d.moe = Dev(B * H * 4);
        d.logits = Dev(B * size_t(c.vocab) * 4);
        d.rowf = Dev(size_t(d.markov) * 4);
        d.bvec = Dev(size_t(c.vocab) * 4);
        const size_t host_floats = std::max({size_t(c.vocab), nt * 2 * 64 + 1024, B * H, B * size_t(d.experts), size_t(d.markov)});
        if (host_floats > SIZE_MAX / sizeof(float)) throw std::runtime_error("ds41: DSpark host scratch size overflow");
        d.host_bytes = host_floats * sizeof(float);
        d.host = pinned_keep(d.host_bytes);
        d.hx_bytes = B * H * sizeof(float);
        d.hx = pinned_keep(d.hx_bytes);
        d.info_bytes = B * 16;
        d.info = reinterpret_cast<int*>(pinned_keep(d.info_bytes));
        d.dtok = Dev((B + 1) * 4);
        d.dconf = Dev(B * 4);
        if (d.w1_dev.p) d.head_error = Dev(sizeof(int));
        d.htok_bytes = (2 * B + 2) * sizeof(int);
        d.htok = reinterpret_cast<int*>(pinned_keep(d.htok_bytes));
        if (!d.conf.empty()) {
            d.conf_dev = Dev(d.conf.size() * 4);
            ck(cudaMemcpy(d.conf_dev.p, d.conf.data(), d.conf.size() * 4, cudaMemcpyHostToDevice), "conf proj");
        }
        d.info_dev = Dev(B * 16);
        for (int l = 0; l < d.layers; ++l) {
            d.bias_dev.emplace_back(size_t(d.experts) * 4);
            ck(cudaMemcpy(d.bias_dev.back().p, d.bias[size_t(l)].data(), size_t(d.experts) * 4, cudaMemcpyHostToDevice), "dspark bias");
        }
        d.rids = Dev(P * 4);
        d.rw = Dev(P * 4);
        d.ptok = Dev(P * 4);
        {
            std::vector<int> pt(P);
            for (size_t i = 0; i < P; ++i) pt[i] = int(i) / d.top_k;
            ck(cudaMemcpy(d.ptok.p, pt.data(), P * 4, cudaMemcpyHostToDevice), "dspark pairs");
        }
        spec_v_.resize(size_t(c.layers));
        spec_s_.resize(size_t(c.layers));
        for (int l = 0; l < c.layers; ++l)
            if (c.kv_src[size_t(l)] == l && c.ratio[size_t(l)] > 1) {
                spec_v_[size_t(l)] = Dev(B * c.head_dim * 4);
                spec_s_[size_t(l)] = Dev(B * c.head_dim * 4);
            }
        validate_dspark_scratch(d.draft_limit);
        d.on = true;
        if (std::getenv("STRATA_DS41_DSPARK_TEST")) dspark_selftest(ts);
        std::cerr << "DS41 dspark layers=" << d.layers << " block=" << d.block << " experts=" << d.experts << " top_k=" << d.top_k
                  << " targets=" << d.targets.size() << " gpu_MiB=" << bytes / double(1 << 20) << " s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << (d.stream_experts ? " streamed_experts=1" : "")
                  << (d.draft_limit != d.block ? " draft_limit=" + std::to_string(d.draft_limit) : "") << "\n";
    }

    // STRATA_DS41_DSPARK_TEST=1: the few-token GEMVs against host references (an MXFP4 expert, a BF16 projection)
    void dspark_selftest(const std::map<std::string, const TensorInfo*>& ts) {
        const auto& c = cfg_;
        auto& d = dsp_;
        const int n = 3, H = c.hidden;
        std::vector<float> x(size_t(n) * H);
        uint32_t r = 12345;
        for (auto& v : x) { r = r * 1664525u + 1013904223u; v = float(int(r >> 9) % 2001 - 1000) / 1000.0f; }
        Dev xd(x.size() * 4), yd(size_t(n) * c.ff * 4 > size_t(n) * c.q_rank * 4 ? size_t(n) * c.ff * 4 : size_t(n) * c.q_rank * 4), y2(size_t(n) * c.q_rank * 4);
        ck(cudaMemcpy(xd.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "test");
        const int rows[3] = {0, 1, 2};
        {   // MXFP4: layer 0, expert 5's gate rows
            const auto& E = d.ex[0];
            const int e = std::min(5, d.experts - 1);
            const size_t ge = E.gu_row * size_t(c.ff);
            const uint8_t* gate = E.gate + ge * size_t(e);
            if (d.stream_experts) {
                ck(cudaMemcpyAsync(d.expert_stage.p, gate, ge, cudaMemcpyHostToDevice, stream_), "dspark test stage");
                gate = d.expert_stage.b();
            }
            k::ds41_mxfp4_gemv_mt(gate, E.gu_row, xd.f(), H, rows, n, yd.f(), c.ff, c.ff, H, stream_);
            std::vector<float> y(size_t(n) * c.ff);
            ck(cudaStreamSynchronize(stream_), "dspark test gate");
            ck(cudaMemcpy(y.data(), yd.p, y.size() * 4, cudaMemcpyDeviceToHost), "test");
            const auto& G = *ts.at("blk.0.ffn_gate_exps.weight");
            double worst = 0;
            for (int rr : {0, std::min(1, c.ff - 1), std::min(777, c.ff - 1), c.ff - 1}) {
                const auto w = to_float(39, d.file->tensor_data(G) + (size_t(e) * c.ff + size_t(rr)) * E.gu_row, H);
                for (int t = 0; t < n; ++t) {
                    double ref = 0;
                    for (int j = 0; j < H; ++j) ref += double(w[size_t(j)]) * x[size_t(t) * H + size_t(j)];
                    worst = std::max(worst, std::abs(ref - y[size_t(t) * c.ff + size_t(rr)]) / (std::abs(ref) + 1e-3));
                }
            }
            std::cerr << "DS41 dspark_test mxfp4 worst_rel=" << worst << "\n";
        }
        {   // The actual q_a type, batched against one token at a time.
            const std::string name = "dspark.blk.0.attn_q_a.weight";
            dmat(name, xd.f(), H, n, yd.f(), c.q_rank);
            for (int i = 0; i < n; ++i) dmat(name, xd.f(size_t(i) * H), H, 1, y2.f(size_t(i) * c.q_rank), c.q_rank);
            std::vector<float> a(size_t(n) * c.q_rank), b(a.size());
            ck(cudaStreamSynchronize(stream_), "dspark test q_a");
            ck(cudaMemcpy(a.data(), yd.p, a.size() * 4, cudaMemcpyDeviceToHost), "test");
            ck(cudaMemcpy(b.data(), y2.p, b.size() * 4, cudaMemcpyDeviceToHost), "test");
            double worst = 0;
            for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, double(std::abs(a[i] - b[i]) / (std::abs(b[i]) + 1e-3f)));
            std::cerr << "DS41 dspark_test q_a worst_rel=" << worst << "\n";
        }
    }

    // at layer l's start: the stream means of its rows (positions pos0_ + i) into the feature ring
    void capture_features(int l) {
        const auto& c = cfg_;
        const auto& d = dsp_;
        for (size_t k = 0; k < d.targets.size(); ++k) {
            if (d.targets[k] != l) continue;
            const size_t H = size_t(c.hidden), ld = d.targets.size() * H;
            for (int i = std::max(0, tokens_ - kFeatRing); i < tokens_;) {
                const int slot = (pos0_ + i) % kFeatRing, run = std::min(tokens_ - i, kFeatRing - slot);
                k::ds41_stream_mean(h_.f(size_t(i) * 4 * H), d.fring.f(size_t(slot) * ld + k * H), ld, c.hidden, run, stream_);
                i += run;
            }
        }
    }

    // rope tables {forward, back} of n positions from p0, uploaded to dsp_.cs
    void dspark_tables(int p0, int n) {
        const auto& c = cfg_;
        float* cs = dsp_.host;
        for (int i = 0; i < n; ++i) {
            rope_table(c.rope_plain, c.rope_dim, p0 + i, 1.0f, cs + size_t(i) * 128);
            rope_table(c.rope_plain, c.rope_dim, p0 + i, -1.0f, cs + size_t(i) * 128 + 64);
        }
        ck(cudaMemcpyAsync(dsp_.cs.p, cs, size_t(n) * 128 * 4, cudaMemcpyHostToDevice, stream_), "dspark rope");
    }

    // the drafter's keys of committed positions [p0, p0 + n) from the feature ring
    void dspark_inject(int p0, int n) {
        const auto& c = cfg_;
        auto& d = dsp_;
        if (!d.on || p0 < 0 || n < 0 || p0 > context_ || n > context_ - p0)
            throw std::runtime_error("ds41: DSpark injection position bounds");
        if (n == 0) return;
        auto checked_bytes = [](size_t a, size_t b) {
            if (a && b > SIZE_MAX / a) throw std::runtime_error("ds41: DSpark injection size overflow");
            return a * b;
        };
        if (c.hidden <= 0 || c.head_dim <= 0 || c.window <= 0 || d.layers < 1 || d.layers > c.layers || d.ring < 1 ||
            d.rings.size() != size_t(d.layers) || d.targets.empty() || d.targets.size() > size_t(c.layers))
            throw std::runtime_error("ds41: DSpark injection geometry bounds");
        for (int target : d.targets)
            if (target < 0 || target >= c.layers) throw std::runtime_error("ds41: DSpark injection target bounds");
        const size_t H = size_t(c.hidden);
        if (d.targets.size() > SIZE_MAX / H) throw std::runtime_error("ds41: DSpark injection feature width overflow");
        const size_t F = d.targets.size() * H;
        if (F > size_t(INT32_MAX)) throw std::runtime_error("ds41: DSpark injection leading dimension overflow");
        const size_t feature_ring_bytes = checked_bytes(checked_bytes(size_t(kFeatRing), F), sizeof(float));
        const size_t feature_batch_bytes = checked_bytes(checked_bytes(size_t(kInj), F), sizeof(float));
        const size_t hidden_batch_bytes = checked_bytes(checked_bytes(size_t(kInj), H), sizeof(float));
        const size_t kv_batch_bytes = checked_bytes(checked_bytes(size_t(kInj), size_t(c.head_dim)), sizeof(float));
        const size_t rope_batch_bytes = checked_bytes(checked_bytes(size_t(kInj), size_t(128)), sizeof(float));
        if (!d.fring.p || d.fring.bytes < feature_ring_bytes || !d.feat.p || d.feat.bytes < feature_batch_bytes ||
            !d.fc.p || d.fc.bytes < hidden_batch_bytes || !d.g.p || d.g.bytes < hidden_batch_bytes ||
            !d.kv.p || d.kv.bytes < kv_batch_bytes || !d.kvn.p || d.kvn.bytes < kv_batch_bytes ||
            !d.cs.p || d.cs.bytes < rope_batch_bytes || !d.host || d.host_bytes < rope_batch_bytes)
            throw std::runtime_error("ds41: DSpark injection scratch capacity bounds");
        if (c.window > INT32_MAX - 16 || d.ring < c.window + 16)
            throw std::runtime_error("ds41: DSpark injection ring geometry bounds");
        const size_t ring_bytes = checked_bytes(checked_bytes(size_t(d.ring), size_t(c.head_dim)), sizeof(float));
        for (const auto& ring : d.rings)
            if (!ring.p || ring.bytes < ring_bytes) throw std::runtime_error("ds41: DSpark injection ring capacity bounds");
        const auto t0 = std::chrono::steady_clock::now();
        const int nope = c.head_dim - c.rope_dim, pairs = c.rope_dim / 2;
        for (int a = 0; a < n; a += kInj) {
            const int m = std::min(kInj, n - a), q0 = p0 + a;
            for (int i = 0; i < m;) {   // gather the ring rows (contiguous runs)
                const int slot = (q0 + i) % kFeatRing, run = std::min(m - i, kFeatRing - slot);
                if (slot < 0 || slot >= kFeatRing || run < 1 || run > m - i || slot + run > kFeatRing)
                    throw std::runtime_error("ds41: DSpark feature ring copy bounds");
                ck(cudaMemcpyAsync(d.feat.f(size_t(i) * F), d.fring.f(size_t(slot) * F), size_t(run) * F * 4, cudaMemcpyDeviceToDevice, stream_), "features");
                i += run;
            }
            dmat("dspark.fc.weight", d.feat.f(), int(F), m, d.fc.f(), c.hidden);
            k::glm_rms_norm_rows(d.fc.f(), w("dspark.enc.output_norm.weight"), d.g.f(), c.hidden, m, d.eps, stream_);
            ck(cudaStreamSynchronize(stream_), "dspark tables");   // the pinned table buffer is reused
            dspark_tables(q0, m);
            for (int l = 0; l < d.layers; ++l) {
                const std::string p = "dspark.blk." + std::to_string(l) + ".";
                dmat(p + "attn_kv.weight", d.g.f(), c.hidden, m, d.kv.f(), c.head_dim);
                k::glm_rms_norm_rows(d.kv.f(), w(p + "attn_kv_a_norm.weight"), d.kvn.f(), c.head_dim, m, d.eps, stream_);
                k::ds41_rope(d.kvn.f(), m, c.head_dim, nope, pairs, d.cs.f(), stream_, 1, 128);
                for (int i = 0; i < m;) {
                    const int slot = (q0 + i) % d.ring, run = std::min(m - i, d.ring - slot);
                    if (slot < 0 || slot >= d.ring || run < 1 || run > m - i || slot + run > d.ring)
                        throw std::runtime_error("ds41: DSpark key ring copy bounds");
                    ck(cudaMemcpyAsync(d.rings[size_t(l)].f(size_t(slot) * c.head_dim), d.kvn.f(size_t(i) * c.head_dim),
                                       size_t(run) * c.head_dim * 4, cudaMemcpyDeviceToDevice, stream_), "dspark ring");
                    i += run;
                }
            }
        }
        d.inject_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    void dspark_mixes(const std::string& p, Dev& coeff, int B) {
        const auto& c = cfg_;
        auto& d = dsp_;
        k::glm_rms_norm_rows(d.h.f(), nullptr, d.flat.f(), 4 * c.hidden, B, d.eps, stream_);
        dmat(p + "fn.weight", d.flat.f(), 4 * c.hidden, B, d.mix.f(), 24);
        k::ds41_hc_coeff(d.mix.f(), w(p + "scale.weight"), w(p + "base.weight"), coeff.f(), d.sinkhorn, d.hc_eps, stream_, B);
    }

    void validate_dspark_scratch(int B) const {
        const auto& c = cfg_;
        const auto& d = dsp_;
        if (c.window > INT32_MAX - 16 || B < 1 || B > 7 || d.draft_limit < B || d.layers < 1 || d.layers > c.layers || d.targets.empty() ||
            d.targets.size() > size_t(c.layers) || d.ex.size() != size_t(d.layers) || d.rings.size() != size_t(d.layers) ||
            c.window + B > 4096 || d.ring < c.window + B)
            throw std::runtime_error("ds41: DSpark pre-run geometry bounds");
        auto product = [](std::initializer_list<size_t> dims) {
            size_t n = sizeof(float);
            for (size_t d : dims) {
                if (d && n > SIZE_MAX / d) throw std::runtime_error("ds41: DSpark scratch size overflow");
                n *= d;
            }
            return n;
        };
        auto require = [](const Dev& b, size_t need, const char* name) {
            if (!b.p || b.bytes < need) throw std::runtime_error(std::string("ds41: DSpark pre-run device capacity: ") + name);
        };
        auto require_host = [](const void* p, size_t have, size_t need, const char* name) {
            if (!p || have < need) throw std::runtime_error(std::string("ds41: DSpark pre-run host capacity: ") + name);
        };
        const size_t H = size_t(c.hidden), F = d.targets.size();
        const size_t P = size_t(B) * size_t(d.top_k), width = size_t(c.heads) * size_t(c.head_dim);
        const size_t host_floats = std::max({size_t(c.vocab), size_t(kInj) * 128 + 1024, size_t(B) * H,
                                             size_t(B) * size_t(d.experts), size_t(d.markov)});
        if (host_floats > SIZE_MAX / sizeof(float)) throw std::runtime_error("ds41: DSpark host capacity overflow");
        require_host(d.host, d.host_bytes, host_floats * sizeof(float), "general scratch");
        require_host(d.hx, d.hx_bytes, product({size_t(B), H}), "confidence hidden rows");
        require_host(d.info, d.info_bytes, size_t(B) * 16, "attention info");
        require_host(d.htok, d.htok_bytes, size_t(2 * B + 2) * sizeof(int), "device head tokens");
        require(d.fring, product({size_t(kFeatRing), F, H}), "feature ring");
        require(d.feat, product({size_t(kInj), F, H}), "feature batch");
        require(d.fc, product({size_t(kInj), H}), "feature projection");
        require(d.g, product({size_t(kInj), H}), "normalized features");
        require(d.kv, product({size_t(kInj), size_t(c.head_dim)}), "injection keys");
        require(d.kvn, product({size_t(kInj), size_t(c.head_dim)}), "normalized injection keys");
        require(d.cs, product({size_t(kInj), 128}), "rope tables");
        require(d.info_dev, size_t(B) * 16, "device attention info");
        require(d.h, product({size_t(B), 4, H}), "stream state");
        require(d.h2, product({size_t(B), 4, H}), "stream state ping-pong");
        require(d.x, product({size_t(B), H}), "hidden rows");
        require(d.xn, product({size_t(B), H}), "normalized hidden rows");
        require(d.flat, product({size_t(B), 4, H}), "flattened streams");
        require(d.mix, product({size_t(B), 24}), "mix projections");
        require(d.ca, product({size_t(B), 24}), "attention coefficients");
        require(d.cf, product({size_t(B), 24}), "FFN coefficients");
        require(d.qa, product({size_t(B), size_t(c.q_rank)}), "query low rank");
        require(d.qr, product({size_t(B), size_t(c.q_rank)}), "normalized query low rank");
        require(d.q, product({size_t(B), width}), "queries");
        require(d.att, product({size_t(B), width}), "attention output");
        require(d.oa, product({size_t(B), size_t(c.out_groups), size_t(c.out_rank)}), "output projection rows");
        require(d.y, product({size_t(B), H}), "hidden projection");
        require(d.rl, product({size_t(B), size_t(d.experts)}), "router logits");
        require(d.sg, product({size_t(B), size_t(c.ff)}), "shared gate");
        require(d.su, product({size_t(B), size_t(c.ff)}), "shared up");
        require(d.sh, product({size_t(B), size_t(c.ff)}), "shared activation");
        require(d.eg, product({P, size_t(c.ff)}), "expert gate");
        require(d.eu, product({P, size_t(c.ff)}), "expert up");
        require(d.eh, product({P, size_t(c.ff)}), "expert activation");
        require(d.ed, product({P, H}), "expert down");
        require(d.moe, product({size_t(B), H}), "MoE output");
        require(d.logits, product({size_t(B), size_t(c.vocab)}), "draft logits");
        require(d.rowf, product({size_t(d.markov)}), "Markov row");
        require(d.bvec, product({size_t(c.vocab)}), "draft bias logits");
        require(d.dtok, size_t(B + 1) * sizeof(int), "device tokens");
        require(d.dconf, size_t(B) * sizeof(float), "confidence values");
        if (!d.conf.empty()) require(d.conf_dev, product({d.conf.size()}), "confidence weights");
        if (d.w1_dev.p) {
            require(d.w1_dev, d.w1.size() * sizeof(uint16_t), "device Markov table");
            require(d.head_error, sizeof(int), "device-head error flag");
        }
        require(d.rids, P * sizeof(int), "expert routes");
        require(d.rw, P * sizeof(float), "expert route weights");
        require(d.ptok, P * sizeof(int), "expert token indices");
        for (const auto& ring : d.rings) require(ring, product({size_t(d.ring), size_t(c.head_dim)}), "key ring");
        if (d.stream_experts) {
            for (const auto& e : d.ex)
                if (!e.gate || !e.up || !e.down) throw std::runtime_error("ds41: DSpark pre-run expert pointer bounds");
            const size_t ge = d.ex.front().gu_row * size_t(c.ff), de = d.ex.front().d_row * H;
            if (ge > (SIZE_MAX - de) / 2) throw std::runtime_error("ds41: DSpark pre-run stage size overflow");
            const size_t stride = 2 * ge + de;
            const size_t groups = d.batch_stage ? size_t(d.draft_limit) * size_t(d.top_k) : 1;
            const size_t slots = d.cache_experts ? d.expert_cache_slots : groups;
            if (!stride || !slots || stride > SIZE_MAX / slots || d.expert_stage_stride != stride ||
                d.expert_stage_groups != groups)
                throw std::runtime_error("ds41: DSpark pre-run expert stage geometry bounds");
            require(d.expert_stage, stride * slots, "expert device stage");
            if (d.batch_stage && !d.pin_experts) {
                if (stride > SIZE_MAX / groups) throw std::runtime_error("ds41: DSpark pre-run host stage size overflow");
                require_host(d.expert_host_stage, d.expert_host_stage_bytes, stride * groups, "expert host stage");
            }
        }
    }

    // drafts n tokens: [anchor, MASK x (n - 1)] at positions pos.., slot i predicting position pos + i + 1
    // DSPARK_SCHED: of the k <= kmax drafted tokens, how many to verify. Expected tokens of verifying j of them:
    // 1 + sum_{i<=j} prod_{t<=i} c_t (the bonus token always comes); cost: the draft's ms + the verify ms of a pass of
    // j + 1 tokens - measured widths as measured, others from the line through the measured ones (until two widths are
    // known: 15 % of the narrowest per extra token). The first steps verify everything, so the widths get measured.
    int sched_prefix(const float* conf, int kmax) const {
        const auto& d = dsp_;
        if (kmax <= 0 || d.steps < 3) return kmax;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int n = 0, t0 = 0;
        for (int t = 1; t < 16; ++t)
            if (d.verify_ms[t] > 0) { sx += t; sy += d.verify_ms[t]; sxx += double(t) * t; sxy += t * d.verify_ms[t]; ++n; if (!t0) t0 = t; }
        if (!n) return kmax;
        double slope = 0.15 * d.verify_ms[t0], icpt = d.verify_ms[t0] - slope * t0;
        if (n >= 2) {
            slope = (n * sxy - sx * sy) / std::max(1e-9, n * sxx - sx * sx);
            icpt = (sy - slope * sx) / n;
        }
        auto vms = [&](int t) { return t < 16 && d.verify_ms[t] > 0 ? d.verify_ms[t] : std::max(1e-3, icpt + slope * t); };
        int best = 0;
        double best_rate = 1.0 / (d.draft_ms + vms(1)), expect = 1.0, run = 1.0;
        for (int j = 1; j <= kmax; ++j) {
            run *= double(conf[j - 1]);
            expect += run;
            const double rate = expect / (d.draft_ms + vms(j + 1));
            if (rate > best_rate) { best_rate = rate; best = j; }
        }
        return best;
    }

    std::vector<int> dspark_draft(int anchor, int pos, int n) {
        const auto& c = cfg_;
        auto& d = dsp_;
        if (!d.on || n < 1 || n > d.draft_limit || n > 7 || anchor < 0 || anchor >= c.vocab ||
            pos < 0 || pos > context_ - n || pos > INT32_MAX - n || d.ring < c.window + n || d.ex.size() != size_t(d.layers))
            throw std::runtime_error("ds41: DSpark draft bounds");
        validate_dspark_scratch(n);
        const auto t0 = std::chrono::steady_clock::now();
        const int B = n;
        const size_t H = size_t(c.hidden);
        const int nope = c.head_dim - c.rope_dim, pairs = c.rope_dim / 2, width = c.heads * c.head_dim;
        const int group_in = c.heads / c.out_groups * c.head_dim;
        std::vector<int> known;
        known.reserve(size_t(B) + 1);
        known.push_back(anchor);   // positions pos..: their tokens as the draft decides them
        engram_prefetch(pos, known, pos, pos + 1);
        ck(cudaStreamSynchronize(stream_), "dspark");
        {   // embeddings
            const auto& t = art_.at("token_embd.weight");
            const size_t row = ggml_row_size(ggml_type(t.tensor->type), c.hidden);
            const uint8_t* table = embd_copy_.empty() ? t.data() : embd_copy_.data();
            float* e = d.host;
            for (int i = 0; i < B; ++i) {
                const auto v = to_float(int(t.tensor->type), table + size_t(i == 0 ? anchor : d.mask) * row, c.hidden);
                std::copy(v.begin(), v.end(), e + size_t(i) * H);
            }
            ck(cudaMemcpyAsync(d.x.p, e, size_t(B) * H * 4, cudaMemcpyHostToDevice, stream_), "dspark embd");
            k::ds41_broadcast_streams(d.x.f(), d.h.f(), c.hidden, stream_, B);
            ck(cudaStreamSynchronize(stream_), "dspark embd");
        }
        dspark_tables(pos, B);
        const float scale = 1.0f / std::sqrt(float(c.head_dim));
        int rows_idx[8];
        for (int i = 0; i < 8; ++i) rows_idx[i] = i;
        for (int l = 0; l < d.layers; ++l) {
            const std::string p = "dspark.blk." + std::to_string(l) + ".";
            dspark_mixes(p + "hc_attn_", d.ca, B);
            if (l > 0) k::ds41_hc_pre(d.h.f(), d.cf.f(), d.x.f(), c.hidden, stream_, B);
            else ck(cudaMemcpy2DAsync(d.x.p, H * 4, d.h.p, 4 * H * 4, H * 4, size_t(B), cudaMemcpyDeviceToDevice, stream_), "stream 0");
            k::glm_rms_norm_rows(d.x.f(), w(p + "attn_norm.weight"), d.xn.f(), c.hidden, B, d.eps, stream_);
            // attention: q, the block's keys into the ring, each token over its window up to the block's end (non-causal)
            dmat(p + "attn_q_a.weight", d.xn.f(), c.hidden, B, d.qa.f(), c.q_rank);
            k::glm_rms_norm_rows(d.qa.f(), w(p + "attn_q_a_norm.weight"), d.qr.f(), c.q_rank, B, d.eps, stream_);
            dmat(p + "attn_q_b.weight", d.qr.f(), c.q_rank, B, d.q.f(), width);
            k::ds41_rope(d.q.f(), B * c.heads, c.head_dim, nope, pairs, d.cs.f(), stream_, c.heads, 128);
            dmat(p + "attn_kv.weight", d.xn.f(), c.hidden, B, d.kv.f(), c.head_dim);
            k::glm_rms_norm_rows(d.kv.f(), w(p + "attn_kv_a_norm.weight"), d.kvn.f(), c.head_dim, B, d.eps, stream_);
            k::ds41_rope(d.kvn.f(), B, c.head_dim, nope, pairs, d.cs.f(), stream_, 1, 128);
            if (fuse_) k::ds41_ring_rows(d.kvn.f(), B, c.head_dim, pos, d.ring, d.rings[size_t(l)].f(), stream_);
            else
            for (int i = 0; i < B; ++i)
                ck(cudaMemcpyAsync(d.rings[size_t(l)].f(size_t((pos + i) % d.ring) * c.head_dim), d.kvn.f(size_t(i) * c.head_dim),
                                   size_t(c.head_dim) * 4, cudaMemcpyDeviceToDevice, stream_), "dspark ring");
            const int last = pos + B - 1;
            int bound = 1;
            if (l == 0) {   // the same rows in every layer
                for (int i = 0; i < B; ++i) {
                    const int first = std::max(0, pos + i - (c.window - 1));
                    const int first_slot = first % d.ring;
                    const int rows = last - first + 1;
                    if (first_slot < 0 || first_slot >= d.ring || rows < 1 || rows > d.ring || rows > 4096)
                        throw std::runtime_error("ds41: DSpark attention window bounds");
                    d.info[4 * i] = first_slot;
                    d.info[4 * i + 1] = rows;
                    d.info[4 * i + 2] = 0;
                    d.info[4 * i + 3] = 0;
                }
                ck(cudaMemcpyAsync(d.info_dev.p, d.info, size_t(B) * 16, cudaMemcpyHostToDevice, stream_), "dspark info");
            }
            bound = 1;
            for (int i = 0; i < B; ++i) bound = std::max(bound, d.info[4 * i + 1]);
            if (bound < 1 || bound > d.ring || bound > 4096)
                throw std::runtime_error("ds41: DSpark attention bound exceeds ring capacity");
            k::ds41_attention_multi(d.q.f(), B, c.heads, c.head_dim, d.rings[size_t(l)].f(), d.ring, d.info_dev.i(), bound, nullptr, 0,
                                    d.info_dev.i(), 0, nullptr, w(p + "attn_sinks.weight"), scale, attn_scores_.f(), d.att.f(), stream_,
                                    attn_scores_.bytes);
            k::ds41_rope(d.att.f(), B * c.heads, c.head_dim, nope, pairs, d.cs.f(64), stream_, c.heads, 128);
            {   // grouped output projection: group g's out_rank rows read its heads' columns
                const Weight& A = weight(p + "attn_output_a.weight");
                if (A.in != group_in || A.out != c.out_groups * c.out_rank)
                    throw std::runtime_error("ds41: DSpark grouped output projection shape");
                if (A.type == GGML_TYPE_BF16 && fuse_)
                    k::ds41_bf16_gemv_mt_groups(reinterpret_cast<const uint16_t*>(A.data.p), size_t(c.out_rank) * A.row_bytes / 2, d.att.f(), width,
                                                group_in, rows_idx, B, d.oa.f(), c.out_groups * c.out_rank, c.out_rank, c.out_rank, group_in,
                                                c.out_groups, stream_);
                else if (A.type == GGML_TYPE_BF16)
                for (int g = 0; g < c.out_groups; ++g)
                    k::ds41_bf16_gemv_mt(reinterpret_cast<const uint16_t*>(A.data.b(size_t(g) * c.out_rank * A.row_bytes)), d.att.f(size_t(g) * group_in),
                                         width, rows_idx, B, d.oa.f(size_t(g) * c.out_rank), c.out_groups * c.out_rank, c.out_rank, group_in, stream_);
                else if (k::native_mmvq_supported(A.type)) {
                    const size_t col_all = k::native_q8_1_bytes(width, 1), col_group = k::native_q8_1_bytes(group_in, 1);
                    if (col_all != col_group * size_t(c.out_groups) || q8_.bytes < col_all * size_t(B) ||
                        q8g_.bytes < col_group * size_t(B) || oa_part_.bytes < size_t(B) * c.out_rank * sizeof(float))
                        throw std::runtime_error("ds41: DSpark grouped output projection scratch");
                    quantize(d.att.f(), q8_.p, width, B);
                    for (int g = 0; g < c.out_groups; ++g) {
                        ck(cudaMemcpy2DAsync(q8g_.p, col_group, q8_.b(size_t(g) * col_group), col_all, col_group,
                                             size_t(B), cudaMemcpyDeviceToDevice, stream_), "dspark group pack");
                        mmvq(A.type, A.data.b(size_t(g) * c.out_rank * A.row_bytes), q8g_.p, oa_part_.f(), group_in, c.out_rank, B);
                        ck(cudaMemcpy2DAsync(d.oa.f(size_t(g) * c.out_rank), size_t(c.out_groups) * c.out_rank * 4,
                                             oa_part_.p, size_t(c.out_rank) * 4, size_t(c.out_rank) * 4, size_t(B),
                                             cudaMemcpyDeviceToDevice, stream_), "dspark group unpack");
                    }
                } else throw std::runtime_error("ds41: unsupported DSpark grouped output projection type");
            }
            dmat(p + "attn_output_b.weight", d.oa.f(), c.out_groups * c.out_rank, B, d.y.f(), c.hidden);
            k::ds41_hc_post(d.y.f(), d.h.f(), d.ca.f(4), d.ca.f(8), d.h2.f(), c.hidden, stream_, B);
            std::swap(d.h, d.h2);
            // FFN
            dspark_mixes(p + "hc_ffn_", d.cf, B);
            k::ds41_hc_pre(d.h.f(), d.ca.f(), d.x.f(), c.hidden, stream_, B);
            k::glm_rms_norm_rows(d.x.f(), w(p + "ffn_norm.weight"), d.xn.f(), c.hidden, B, d.eps, stream_);
            dmat(p + "ffn_gate_inp.weight", d.xn.f(), c.hidden, B, d.rl.f(), d.experts);
            // STRATA_DS41_DSPARK_GPU_ROUTE=1: routes on the device, every (token, pick) pair one row of each launch
            static const bool gpu_route = std::getenv("STRATA_DS41_DSPARK_GPU_ROUTE") && std::string(std::getenv("STRATA_DS41_DSPARK_GPU_ROUTE")) == "1";
            if (gpu_route) {
                const auto& E = d.ex[size_t(l)];
                const int P = B * d.top_k;
                k::ds41_moe_route(d.rl.f(), d.bias_dev[size_t(l)].f(), B, d.experts, d.top_k, d.scale, d.normalize, d.rids.i(), d.rw.f(), stream_);
                dmat(p + "ffn_gate_shexp.weight", d.xn.f(), c.hidden, B, d.sg.f(), c.ff);
                dmat(p + "ffn_up_shexp.weight", d.xn.f(), c.hidden, B, d.su.f(), c.ff);
                k::glm_swiglu(d.sg.f(), d.su.f(), d.sh.f(), B * c.ff, d.clamp_sh, stream_);
                dmat(p + "ffn_down_shexp.weight", d.sh.f(), c.ff, B, d.y.f(), c.hidden);
                const size_t ge = E.gu_row * size_t(c.ff), de = E.d_row * H;
                k::ds41_mxfp4_moe(E.gate, ge, E.gu_row, d.rids.i(), P, d.xn.f(), c.hidden, d.top_k, d.eg.f(), c.ff, c.ff, c.hidden, stream_);
                k::ds41_mxfp4_moe(E.up, ge, E.gu_row, d.rids.i(), P, d.xn.f(), c.hidden, d.top_k, d.eu.f(), c.ff, c.ff, c.hidden, stream_);
                k::glm_swiglu(d.eg.f(), d.eu.f(), d.eh.f(), P * c.ff, d.clamp, stream_);
                k::ds41_mxfp4_moe(E.down, de, E.d_row, d.rids.i(), P, d.eh.f(), c.ff, 1, d.ed.f(), c.hidden, c.hidden, c.ff, stream_);
                k::ds41_scatter_axpy(d.ed.f(), d.ptok.i(), d.rw.f(), P, c.hidden, d.y.f(), stream_);
                k::ds41_hc_post(d.y.f(), d.h.f(), d.cf.f(4), d.cf.f(8), d.h2.f(), c.hidden, stream_, B);
                std::swap(d.h, d.h2);
                continue;
            }
            float* rl = d.host;
            ck(cudaMemcpyAsync(rl, d.rl.p, size_t(B) * d.experts * 4, cudaMemcpyDeviceToHost, stream_), "dspark router");
            dmat(p + "ffn_gate_shexp.weight", d.xn.f(), c.hidden, B, d.sg.f(), c.ff);
            dmat(p + "ffn_up_shexp.weight", d.xn.f(), c.hidden, B, d.su.f(), c.ff);
            k::glm_swiglu(d.sg.f(), d.su.f(), d.sh.f(), B * c.ff, d.clamp_sh, stream_);
            dmat(p + "ffn_down_shexp.weight", d.sh.f(), c.ff, B, d.y.f(), c.hidden);
            { const auto r0 = std::chrono::steady_clock::now(); ck(cudaStreamSynchronize(stream_), "dspark router");
              d.draft_route_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count(); }
            // routes: sqrt(softplus) + bias picks, normalized unbiased weights x scale; pairs grouped by expert
            std::vector<std::pair<int, int>> pr;   // (expert, token)
            std::vector<float> pw;
            std::map<std::pair<int, int>, float> wt;
            for (int t = 0; t < B; ++t) {
                std::vector<float> prob(size_t(d.experts)), sel(size_t(d.experts));
                for (int e = 0; e < d.experts; ++e) {
                    prob[size_t(e)] = std::sqrt(softplus(rl[size_t(t) * d.experts + e]));
                    sel[size_t(e)] = prob[size_t(e)] + d.bias[size_t(l)][size_t(e)];
                }
                const auto pick = top_k(sel.data(), d.experts, d.top_k);
                float sum = 0;
                for (int e : pick) sum += prob[size_t(e)];
                for (int e : pick) {
                    float v = prob[size_t(e)];
                    if (d.normalize && d.top_k > 1) v /= sum + 1e-20f;
                    wt[{e, t}] = v * d.scale;
                    pr.push_back({e, t});
                }
            }
            std::sort(pr.begin(), pr.end());
            const auto& E = d.ex[size_t(l)];
            if (d.stream_experts) {
                const size_t ge = E.gu_row * size_t(c.ff), de = E.d_row * H;
                if (d.expert_stage_stride != 2 * ge + de || d.expert_stage.bytes < d.expert_stage_stride)
                    throw std::runtime_error("ds41: DSpark expert stage too small");
                std::vector<size_t> group_slots;
                if (d.batch_stage && strata_env_is_one("STRATA_DS41_DSPARK_CACHE_SIM")) {
                    constexpr std::array<size_t, 5> capacities = {16, 24, 45, 90, 100};
                    for (size_t j0 = 0; j0 < pr.size();) {
                        size_t j1 = j0 + 1;
                        while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                        const int e = pr[j0].first;
                        if (e < 0 || e >= d.experts) throw std::runtime_error("ds41: DSpark cache simulation route bounds");
                        const int key = l * d.experts + e;
                        ++d.cache_accesses;
                        for (size_t cidx = 0; cidx < d.cache_sim.size(); ++cidx) {
                            auto& cache = d.cache_sim[cidx];
                            auto it = std::find(cache.begin(), cache.end(), key);
                            if (it != cache.end()) { ++d.cache_hits[cidx]; cache.erase(it); }
                            else if (cache.size() == capacities[cidx]) cache.pop_front();
                            cache.push_back(key);
                        }
                        j0 = j1;
                    }
                }
                if (d.cache_experts) {
                    // The router sync above completes the preceding layer's
                    // reads before any slot can be reused. Keep all slots used
                    // by this layer locked until its copies are enqueued.
                    std::vector<uint8_t> locked(d.expert_cache_slots, 0);
                    std::vector<int> misses;
                    std::vector<size_t> miss_slots;
                    group_slots.reserve(pr.size());
                    for (size_t j0 = 0; j0 < pr.size();) {
                        size_t j1 = j0 + 1;
                        while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                        const int e = pr[j0].first;
                        if (e < 0 || e >= d.experts || group_slots.size() >= d.expert_stage_groups)
                            throw std::runtime_error("ds41: DSpark cached expert route bounds");
                        const int key = l * d.experts + e;
                        size_t slot = d.expert_cache_slots;
                        for (size_t s = 0; s < d.expert_cache_slots; ++s)
                            if (d.expert_cache[s].key == key) { slot = s; break; }
                        if (slot != d.expert_cache_slots) ++d.expert_cache_hits;
                        else {
                            for (size_t s = 0; s < d.expert_cache_slots; ++s)
                                if (!locked[s] && (slot == d.expert_cache_slots || d.expert_cache[s].age < d.expert_cache[slot].age)) slot = s;
                            if (slot == d.expert_cache_slots) throw std::runtime_error("ds41: DSpark expert cache has no free slot");
                            d.expert_cache[slot].key = key;
                            misses.push_back(e);
                            miss_slots.push_back(slot);
                            ++d.expert_cache_misses;
                        }
                        d.expert_cache[slot].age = ++d.expert_cache_clock;
                        locked[slot] = true;
                        group_slots.push_back(slot);
                        j0 = j1;
                    }
                    if (!d.expert_host_stage || d.expert_host_stage_bytes < d.expert_stage_stride ||
                        misses.size() > d.expert_stage_groups || misses.size() > d.expert_host_stage_bytes / d.expert_stage_stride ||
                        d.expert_stage.bytes / d.expert_stage_stride < d.expert_cache_slots)
                        throw std::runtime_error("ds41: DSpark expert cache staging bounds");
                    auto copy_misses = [&](size_t worker, size_t workers) {
                        for (size_t i = worker; i < misses.size(); i += workers) {
                            const size_t e = size_t(misses[i]);
                            uint8_t* host = d.expert_host_stage + i * d.expert_stage_stride;
                            std::memcpy(host, E.gate + e * ge, ge);
                            std::memcpy(host + ge, E.up + e * ge, ge);
                            std::memcpy(host + 2 * ge, E.down + e * de, de);
                        }
                    };
                    const size_t workers = std::min<size_t>(strata_env_is_one("STRATA_DS41_DSPARK_PARALLEL_STAGE") ? 4 : 1, misses.size());
                    if (workers) {
                        std::vector<std::jthread> threads;
                        threads.reserve(workers - 1);
                        for (size_t i = 1; i < workers; ++i) threads.emplace_back(copy_misses, i, workers);
                        copy_misses(0, workers);
                    }
                    if (std::getenv("STRATA_DS41_DSPARK_TEST"))
                        for (size_t i = 0; i < misses.size(); ++i) {
                            const size_t e = size_t(misses[i]);
                            const uint8_t* host = d.expert_host_stage + i * d.expert_stage_stride;
                            if (std::memcmp(host, E.gate + e * ge, ge) || std::memcmp(host + ge, E.up + e * ge, ge) ||
                                std::memcmp(host + 2 * ge, E.down + e * de, de))
                                throw std::runtime_error("ds41: cached expert staging mismatch");
                        }
                    for (size_t i = 0; i < misses.size(); ++i)
                        ck(cudaMemcpyAsync(d.expert_stage.b(miss_slots[i] * d.expert_stage_stride),
                                           d.expert_host_stage + i * d.expert_stage_stride, d.expert_stage_stride,
                                           cudaMemcpyHostToDevice, stream_), "dspark cached expert upload");
                    if (std::getenv("STRATA_DS41_DSPARK_TEST")) {
                        ck(cudaStreamSynchronize(stream_), "dspark cached expert test");
                        size_t group = 0;
                        for (size_t j0 = 0; j0 < pr.size();) {
                            size_t j1 = j0 + 1;
                            while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                            const size_t e = size_t(pr[j0].first), slot = group_slots[group++];
                            if (d.expert_cache[slot].key != l * d.experts + int(e))
                                throw std::runtime_error("ds41: cached expert key mismatch");
                            const uint8_t* source[] = {E.gate + e * ge, E.up + e * ge, E.down + e * de};
                            const size_t offsets[] = {0, ge, 2 * ge}, lengths[] = {ge, ge, de};
                            for (int part = 0; part < 3; ++part) {
                                if (lengths[part] < 32) throw std::runtime_error("ds41: cached expert test bounds");
                                for (size_t at : {size_t(0), lengths[part] - 32}) {
                                    uint8_t sample[32];
                                    ck(cudaMemcpy(sample, d.expert_stage.b(slot * d.expert_stage_stride + offsets[part] + at),
                                                  sizeof sample, cudaMemcpyDeviceToHost), "dspark cached expert sample");
                                    if (std::memcmp(sample, source[part] + at, sizeof sample))
                                        throw std::runtime_error("ds41: cached expert device mismatch");
                                }
                            }
                            j0 = j1;
                        }
                    }
                } else if (d.batch_stage) {
                    static const bool parallel_stage = strata_env_is_one("STRATA_DS41_DSPARK_PARALLEL_STAGE");
                    size_t groups = 0;
                    std::vector<int> stage_experts;
                    if (parallel_stage && !d.pin_experts) stage_experts.reserve(pr.size());
                    for (size_t j0 = 0; j0 < pr.size();) {
                        size_t j1 = j0 + 1;
                        while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                        const int e = pr[j0].first;
                        if (e < 0 || e >= d.experts || groups >= d.expert_stage_groups)
                            throw std::runtime_error("ds41: DSpark expert batch bounds");
                        if (d.pin_experts) {
                            uint8_t* stage = d.expert_stage.b(groups * d.expert_stage_stride);
                            ck(cudaMemcpyAsync(stage, E.gate + size_t(e) * ge, ge, cudaMemcpyHostToDevice, stream_), "dspark pinned gate");
                            ck(cudaMemcpyAsync(stage + ge, E.up + size_t(e) * ge, ge, cudaMemcpyHostToDevice, stream_), "dspark pinned up");
                            ck(cudaMemcpyAsync(stage + 2 * ge, E.down + size_t(e) * de, de, cudaMemcpyHostToDevice, stream_), "dspark pinned down");
                        } else if (parallel_stage) {
                            stage_experts.push_back(e);
                        } else {
                            uint8_t* host = d.expert_host_stage + groups * d.expert_stage_stride;
                            std::memcpy(host, E.gate + size_t(e) * ge, ge);
                            std::memcpy(host + ge, E.up + size_t(e) * ge, ge);
                            std::memcpy(host + 2 * ge, E.down + size_t(e) * de, de);
                        }
                        ++groups;
                        j0 = j1;
                    }
                    if (!d.pin_experts && parallel_stage) {
                        if (!d.expert_host_stage || d.expert_host_stage_bytes < d.expert_stage_stride ||
                            groups > d.expert_host_stage_bytes / d.expert_stage_stride ||
                            groups > d.expert_stage.bytes / d.expert_stage_stride)
                            throw std::runtime_error("ds41: DSpark host staging bounds");
                        const size_t workers = std::min<size_t>(4, groups);
                        auto copy_groups = [&](size_t worker) {
                            for (size_t i = worker; i < groups; i += workers) {
                                const size_t e = size_t(stage_experts[i]);
                                uint8_t* host = d.expert_host_stage + i * d.expert_stage_stride;
                                std::memcpy(host, E.gate + e * ge, ge);
                                std::memcpy(host + ge, E.up + e * ge, ge);
                                std::memcpy(host + 2 * ge, E.down + e * de, de);
                            }
                        };
                        if (workers) {
                            std::vector<std::jthread> threads;
                            threads.reserve(workers - 1);
                            for (size_t i = 1; i < workers; ++i) threads.emplace_back(copy_groups, i);
                            copy_groups(0);
                        } // join before the async copy reads the staging buffer
                        if (std::getenv("STRATA_DS41_DSPARK_TEST"))
                            for (size_t i = 0; i < groups; ++i) {
                                const size_t e = size_t(stage_experts[i]);
                                const uint8_t* host = d.expert_host_stage + i * d.expert_stage_stride;
                                if (std::memcmp(host, E.gate + e * ge, ge) || std::memcmp(host + ge, E.up + e * ge, ge) ||
                                    std::memcmp(host + 2 * ge, E.down + e * de, de))
                                    throw std::runtime_error("ds41: parallel expert staging mismatch");
                            }
                    }
                    if (!d.pin_experts)
                        ck(cudaMemcpyAsync(d.expert_stage.p, d.expert_host_stage, groups * d.expert_stage_stride,
                                           cudaMemcpyHostToDevice, stream_), "dspark expert batch stage");
                }
                size_t group = 0;
                for (size_t j0 = 0; j0 < pr.size();) {
                    size_t j1 = j0;
                    while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                    const int e = pr[j0].first, m = int(j1 - j0);
                    if (e < 0 || e >= d.experts || m < 1 || m > 8) throw std::runtime_error("ds41: DSpark expert route bounds");
                    int toks[8], rows[8];
                    for (int j = 0; j < m; ++j) {
                        const int t = pr[j0 + size_t(j)].second;
                        if (t < 0 || t >= B) throw std::runtime_error("ds41: DSpark expert token bounds");
                        toks[j] = t; rows[j] = int(j0) + j;
                    }
                    if (d.cache_experts && group >= group_slots.size()) throw std::runtime_error("ds41: DSpark cache group bounds");
                    const size_t slot = d.cache_experts ? group_slots[group] : (d.batch_stage ? group : 0);
                    if (slot >= d.expert_stage.bytes / d.expert_stage_stride) throw std::runtime_error("ds41: DSpark cache slot bounds");
                    uint8_t* stage = d.expert_stage.b(slot * d.expert_stage_stride);
                    if (!d.batch_stage) {
                        // One stream orders each copy after the previous expert's kernels, allowing safe reuse.
                        ck(cudaMemcpyAsync(stage, E.gate + size_t(e) * ge, ge, cudaMemcpyHostToDevice, stream_), "dspark gate stage");
                        ck(cudaMemcpyAsync(stage + ge, E.up + size_t(e) * ge, ge, cudaMemcpyHostToDevice, stream_), "dspark up stage");
                        ck(cudaMemcpyAsync(stage + 2 * ge, E.down + size_t(e) * de, de, cudaMemcpyHostToDevice, stream_), "dspark down stage");
                    }
                    k::ds41_mxfp4_gemv_mt(stage, E.gu_row, d.xn.f(), c.hidden, toks, m, d.eg.f(j0 * c.ff), c.ff, c.ff, c.hidden, stream_);
                    k::ds41_mxfp4_gemv_mt(stage + ge, E.gu_row, d.xn.f(), c.hidden, toks, m, d.eu.f(j0 * c.ff), c.ff, c.ff, c.hidden, stream_);
                    k::glm_swiglu(d.eg.f(j0 * c.ff), d.eu.f(j0 * c.ff), d.eh.f(j0 * c.ff), m * c.ff, d.clamp, stream_);
                    k::ds41_mxfp4_gemv_mt(stage + 2 * ge, E.d_row, d.eh.f(), c.ff, rows, m, d.ed.f(j0 * H), c.hidden, c.hidden, c.ff, stream_);
                    ++group;
                    j0 = j1;
                }
            } else {
            for (size_t j0 = 0; j0 < pr.size();) {
                size_t j1 = j0;
                while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                const int e = pr[j0].first, m = int(j1 - j0);
                if (e < 0 || e >= d.experts || m < 1 || m > 8) throw std::runtime_error("ds41: DSpark expert route bounds");
                int toks[8], prs[8];
                for (int j = 0; j < m; ++j) {
                    const int t = pr[j0 + size_t(j)].second;
                    if (t < 0 || t >= B) throw std::runtime_error("ds41: DSpark expert token bounds");
                    toks[j] = t; prs[j] = int(j0) + j;
                }
                const size_t ge = E.gu_row * size_t(c.ff) * size_t(e);
                k::ds41_mxfp4_gemv_mt(E.gate + ge, E.gu_row, d.xn.f(), c.hidden, toks, m, d.eg.f(j0 * c.ff), c.ff, c.ff, c.hidden, stream_);
                k::ds41_mxfp4_gemv_mt(E.up + ge, E.gu_row, d.xn.f(), c.hidden, toks, m, d.eu.f(j0 * c.ff), c.ff, c.ff, c.hidden, stream_);
                (void)prs;
                j0 = j1;
            }
            const int P = int(pr.size());
            k::glm_swiglu(d.eg.f(), d.eu.f(), d.eh.f(), P * c.ff, d.clamp, stream_);
            for (size_t j0 = 0; j0 < pr.size();) {
                size_t j1 = j0;
                while (j1 < pr.size() && pr[j1].first == pr[j0].first) ++j1;
                const int e = pr[j0].first, m = int(j1 - j0);
                if (e < 0 || e >= d.experts || m < 1 || m > 8) throw std::runtime_error("ds41: DSpark expert route bounds");
                int rows[8];
                for (int j = 0; j < m; ++j) rows[j] = int(j0) + j;
                k::ds41_mxfp4_gemv_mt(E.down + E.d_row * H * size_t(e), E.d_row, d.eh.f(), c.ff, rows, m, d.ed.f(j0 * H), c.hidden, c.hidden,
                                      c.ff, stream_);
                j0 = j1;
            }
            }
            const int P = int(pr.size());
            for (int j = 0; j < P; ++j) k::ds41_axpy(d.ed.f(size_t(j) * H), wt[pr[size_t(j)]], d.y.f(size_t(pr[size_t(j)].second) * H), c.hidden, stream_);
            k::ds41_hc_post(d.y.f(), d.h.f(), d.cf.f(4), d.cf.f(8), d.h2.f(), c.hidden, stream_, B);
            std::swap(d.h, d.h2);
        }
        // head: the last FFN mix collapses the streams; the target's lm head; the Markov bias, chained greedily
        ck(cudaStreamSynchronize(stream_), "dspark layers");
        const auto th = std::chrono::steady_clock::now();
        d.draft_layers_s += std::chrono::duration<double>(th - t0).count();
        k::ds41_hc_pre(d.h.f(), d.cf.f(), d.x.f(), c.hidden, stream_, B);
        k::glm_rms_norm_rows(d.x.f(), w("dspark.output_norm.weight"), d.xn.f(), c.hidden, B, d.eps, stream_);
        mat("output.weight", d.xn.f(), d.logits.f(), B);
        // STRATA_DS41_DSPARK_PMIN (0): the draft stops at the first position whose confidence is below it
        static const float pmin = std::getenv("STRATA_DS41_DSPARK_PMIN") ? float(std::atof(std::getenv("STRATA_DS41_DSPARK_PMIN"))) : 0.0f;
        // STRATA_DS41_DSPARK_SCHED=1 (device head): the drafted prefix verified is the one with the most expected tokens
        // per ms - the confidences' running products as the expected accepted tokens (DSpark's schedule), the measured
        // verify ms per pass width as the cost (on MoE offload a wider pass reads more experts: EVICT)
        static const bool sched = std::getenv("STRATA_DS41_DSPARK_SCHED") && std::string(std::getenv("STRATA_DS41_DSPARK_SCHED")) == "1";
        const bool use_conf = (pmin > 0 || sched) && !d.conf.empty();
        std::vector<int> out;
        out.reserve(size_t(B));
        if (d.w1_dev.p) {   // the chain on the device: one download of the tokens (and confidences) at the end
            ck(cudaMemsetAsync(d.head_error.p, 0, sizeof(int), stream_), "device-head error clear");
            d.htok[0] = anchor;
            ck(cudaMemcpyAsync(d.dtok.p, d.htok, 4, cudaMemcpyHostToDevice, stream_), "anchor");
            for (int i = 0; i < B; ++i) {
                k::ds41_markov_row(d.w1_dev.h(), d.markov, c.vocab, d.dtok.i(), i, d.rowf.f(), d.head_error.i(), stream_);
                if (use_conf) k::ds41_confidence(d.conf_dev.f(), d.x.f(size_t(i) * H), c.hidden, d.rowf.f(), d.markov, d.dconf.f(), i, stream_);
                dmat("dspark.markov_w2.weight", d.rowf.f(), d.markov, 1, d.bvec.f(), c.vocab);
                k::ds41_add_argmax_into(d.logits.f(size_t(i) * c.vocab), d.bvec.f(), c.vocab, d.dtok.i(), i, d.head_error.i(), stream_);
            }
            float* hconf = reinterpret_cast<float*>(d.htok + B + 1);
            int* hbad = d.htok + 2 * B + 1;
            ck(cudaMemcpyAsync(d.htok + 1, d.dtok.i() + 1, size_t(B) * 4, cudaMemcpyDeviceToHost, stream_), "drafts");
            if (use_conf) ck(cudaMemcpyAsync(hconf, d.dconf.p, size_t(B) * 4, cudaMemcpyDeviceToHost, stream_), "confidences");
            ck(cudaMemcpyAsync(hbad, d.head_error.p, sizeof(int), cudaMemcpyDeviceToHost, stream_), "device-head status");
            ck(cudaStreamSynchronize(stream_), "draft");
            if (*hbad) throw std::runtime_error("ds41: device draft head encountered an invalid token or non-finite logits");
            for (int i = 0; i < B; ++i)
                if (d.htok[i + 1] < 0 || d.htok[i + 1] >= c.vocab)
                    throw std::runtime_error("ds41: device draft token outside vocabulary");
            int keep = B;
            for (int i = 0; i < B; ++i)
                if (use_conf && pmin > 0 && hconf[i] < pmin) { keep = i; break; }
            if (sched && use_conf) keep = sched_prefix(hconf, keep);
            for (int i = 0; i < keep; ++i) {
                out.push_back(d.htok[1 + i]);
                known.push_back(d.htok[1 + i]);
                engram_prefetch(pos, known, pos + int(known.size()) - 1, pos + int(known.size()));
            }
            d.draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            d.draft_head_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - th).count();
            return out;
        }
        if (use_conf) ck(cudaMemcpyAsync(d.hx, d.x.p, size_t(B) * H * 4, cudaMemcpyDeviceToHost, stream_), "dspark hidden");
        int prev = anchor;
        float* row = d.host;
        for (int i = 0; i < B; ++i) {
            const uint16_t* w1 = d.w1.data() + size_t(prev) * d.markov;
            for (int j = 0; j < d.markov; ++j) { const uint32_t u = uint32_t(w1[j]) << 16; std::memcpy(row + j, &u, 4); }
            if (use_conf) {
                if (i == 0) ck(cudaStreamSynchronize(stream_), "dspark hidden");
                double z = 0;
                for (int j = 0; j < c.hidden; ++j) z += double(d.conf[size_t(j)]) * d.hx[size_t(i) * H + size_t(j)];
                for (int j = 0; j < d.markov; ++j) z += double(d.conf[size_t(c.hidden + j)]) * row[j];
                if (1.0 / (1.0 + std::exp(-z)) < pmin) break;
            }
            ck(cudaMemcpyAsync(d.rowf.p, row, size_t(d.markov) * 4, cudaMemcpyHostToDevice, stream_), "markov row");
            dmat("dspark.markov_w2.weight", d.rowf.f(), d.markov, 1, d.bvec.f(), c.vocab);
            k::ds41_add(d.logits.f(size_t(i) * c.vocab), d.bvec.f(), d.bvec.f(), c.vocab, stream_);
            ck(cudaMemcpyAsync(row, d.bvec.p, size_t(c.vocab) * 4, cudaMemcpyDeviceToHost, stream_), "draft logits");
            ck(cudaStreamSynchronize(stream_), "draft");
            prev = int(std::max_element(row, row + c.vocab) - row);
            out.push_back(prev);
            known.push_back(prev);
            engram_prefetch(pos, known, pos + int(known.size()) - 1, pos + int(known.size()));
        }
        d.draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        d.draft_head_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - th).count();
        return out;
    }

    // STRATA_DS41_MARGINS=1: each generated token's top-2 logit gap (to tell a near tie from a wrong state)
    static void log_margin(const float* lg, int V, int index) {
        static const bool on = std::getenv("STRATA_DS41_MARGINS") != nullptr;
        if (!on) return;
        int a = 0, b = -1;
        for (int i = 1; i < V; ++i)
            if (lg[i] > lg[a]) { b = a; a = i; }
            else if (b < 0 || lg[i] > lg[b]) b = i;
        std::cerr << "MARGIN i=" << index << " top=" << a << " second=" << b << " gap=" << lg[a] - lg[b] << "\n";
    }

    // greedy decoding with DSpark drafts: `tok` (the prompt's next token) at `pos`, until `steps` tokens
    // STRATA_DS41_DSPARK_N (the block size): tokens drafted per step
    std::vector<int> spec_generate(int tok, int pos, int steps) {
        auto& d = dsp_;
        if (tok < 0 || tok >= cfg_.vocab || pos < 0 || pos > context_ || steps < 0 || steps > context_ - pos ||
            (!sess_ && size_t(pos) != history_.size()))
            throw std::runtime_error("ds41: speculative decode position, token, or output bounds");
        if (steps == 0) return {};
        // Verify passes and the drafter's attention scratch can hold up to eight positions.
        if (chunk_ < 8) throw std::runtime_error("ds41: speculative decoding requires decode chunk >= 8");
        const int m = std::min({pos, cfg_.window, kFeatRing});
        if (d.on) dspark_inject(pos - m, m);   // (lookup drafts alone: no drafter to feed)
        if (spec_v_.empty()) {   // lookup drafts alone: the verify pass's ratio-2 stash for rewind (load_dspark's, 8 positions)
            spec_v_.resize(size_t(cfg_.layers));
            spec_s_.resize(size_t(cfg_.layers));
            for (int l = 0; l < cfg_.layers; ++l)
                if (cfg_.kv_src[size_t(l)] == l && cfg_.ratio[size_t(l)] > 1) {
                    spec_v_[size_t(l)] = Dev(size_t(8) * cfg_.head_dim * 4);
                    spec_s_[size_t(l)] = Dev(size_t(8) * cfg_.head_dim * 4);
                }
        }
        const char* nv = std::getenv("STRATA_DS41_DSPARK_N");
        const int nd = std::max(1, std::min(d.draft_limit, nv ? std::atoi(nv) : d.draft_limit));
        const int V = cfg_.vocab;
        const bool device_argmax = strata_env_is_one("STRATA_DS41_VERIFY_ARGMAX_GPU") && !std::getenv("STRATA_DS41_MARGINS");
        std::vector<int> out{tok};
        // STRATA_DS41_LOOKUP=1: draft from the token history first (prompt-lookup decoding): the longest suffix of
        // the last STRATA_DS41_LOOKUP_MAX (4) .. STRATA_DS41_LOOKUP_MIN (2) tokens that occurred before, its most recent
        // occurrence's next tokens (up to STRATA_DS41_LOOKUP_N, 7). DSpark when nothing matches. The target verifies
        // either draft the same way, so the greedy output does not depend on it; a hit costs no drafter pass, and its
        // n-grams' Engram rows were read before (page cache / RAM)
        static const bool lookup = std::getenv("STRATA_DS41_LOOKUP") && std::string(std::getenv("STRATA_DS41_LOOKUP")) == "1";
        static const int lk_max = std::getenv("STRATA_DS41_LOOKUP_MAX") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_LOOKUP_MAX"))) : 4;
        static const int lk_min = std::getenv("STRATA_DS41_LOOKUP_MIN") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_LOOKUP_MIN"))) : 2;
        static const int lk_n = std::getenv("STRATA_DS41_LOOKUP_N") ? std::max(1, std::min(7, std::atoi(std::getenv("STRATA_DS41_LOOKUP_N")))) : 7;
        // STRATA_DS41_LOOKUP_ADAPT (1): each source's accepted tokens per step as a running mean (weight 0.3); a lookup
        // draft is cut to that mean + 2 tokens, and taken only if its mean plus the drafter's time saved (as tokens,
        // STRATA_DS41_LOOKUP_BONUS 0.5) reaches DSpark's mean
        static const bool adapt = !(std::getenv("STRATA_DS41_LOOKUP_ADAPT") && std::string(std::getenv("STRATA_DS41_LOOKUP_ADAPT")) == "0");
        static const float bonus = std::getenv("STRATA_DS41_LOOKUP_BONUS") ? float(std::atof(std::getenv("STRATA_DS41_LOOKUP_BONUS"))) : 0.5f;
        const bool auto_draft = strata_env_is_one("STRATA_DS41_DSPARK_AUTO");
        const bool auto_backoff = auto_draft && strata_env_is_one("STRATA_DS41_DSPARK_AUTO_BACKOFF");
        int auto_since_probe = 0, auto_skipped = 0;
        int auto_interval = 15;
        std::array<uint64_t, 8> accept_hist{};
        const bool accept_hist_on = strata_env_is_one("STRATA_DS41_DSPARK_ACCEPT_HIST");
        float ema_lk = 3.0f, ema_ds = auto_draft ? 1.0f : 2.5f;
        auto history_draft = [&](int cap) {
            if (cap <= 0 || pos < 0 || size_t(pos) > history_.size() || pos == INT32_MAX) return std::vector<int>{};
            std::vector<int> seq(size_t(pos) + 1);   // positions < pos, then tok at pos
            std::copy_n(history_.begin(), pos, seq.begin());
            seq[size_t(pos)] = tok;
            const int L = pos + 1;
            const int max_k = std::min(lk_max, L - 1);
            for (int k = max_k; k >= lk_min; --k) {
                for (int j = L - k - 1; j >= 0; --j) {   // the most recent earlier occurrence of the last k tokens
                    if (!std::equal(seq.begin() + j, seq.begin() + j + k, seq.end() - k)) continue;
                    const int start = j + k;
                    const int take = std::min(cap, L - start);
                    std::vector<int> d(static_cast<size_t>(take));
                    std::copy_n(seq.begin() + start, take, d.begin());
                    if (!d.empty()) return d;
                }
            }
            return std::vector<int>{};
        };
        while (int(out.size()) < steps) {
            const int cap = std::min(nd, steps - int(out.size()));
            std::vector<int> drafts;
            bool from_history = false;
            bool skipped_draft = false;
            // without a DSpark drafter (STRATA_DS41_LOOKUP=1 alone, e.g. a GPU without room for it) a step tries the history
            // while its running mean of accepted tokens is at least STRATA_DS41_LOOKUP_GATE (1.0) - a rejected draft still
            // widens the pass, which the CPU lane pays for; otherwise a plain one-token pass (the mean recovers +0.25 each)
            static const float lk_gate = std::getenv("STRATA_DS41_LOOKUP_GATE") ? float(std::atof(std::getenv("STRATA_DS41_LOOKUP_GATE"))) : 1.0f;
            if (lookup && (d.on ? (!adapt || ema_lk + bonus >= ema_ds) : (!adapt || ema_lk >= lk_gate))) {
                const auto l0 = std::chrono::steady_clock::now();
                const int lcap = adapt ? std::max(1, std::min(lk_n, int(ema_lk + 2.0f))) : lk_n;
                drafts = history_draft(std::min(lcap, steps - int(out.size())));
                from_history = !drafts.empty();
                d.lookup_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count();
                if (from_history) {
                    std::vector<int> toks{tok};
                    toks.insert(toks.end(), drafts.begin(), drafts.end());
                    engram_prefetch(pos, toks, pos + 1, pos + 1 + int(drafts.size()));
                }
            }
            if (!from_history && d.on) {
                if (auto_draft && ema_ds < 0.75f && auto_since_probe < auto_interval) {
                    skipped_draft = true;
                    ++auto_since_probe;
                    ++auto_skipped;
                } else {
                    auto_since_probe = 0;
                    const double ds0 = d.draft_s;
                    drafts = dspark_draft(tok, pos, cap);
                    const double dms = (d.draft_s - ds0) * 1e3;
                    d.draft_ms = d.draft_ms > 0 ? 0.8 * d.draft_ms + 0.2 * dms : dms;
                    d.sched_cut += cap - int(drafts.size());
                }
            }
            const int n = int(drafts.size());   // the confidence cutoff may draft fewer
            std::vector<int> v{tok};
            v.insert(v.end(), drafts.begin(), drafts.end());
            const auto v0 = std::chrono::steady_clock::now();
            for (auto& v : spec_used_) v.clear();
            for (auto& v : spec_missed_) v.clear();
            spec_save_ = true;
            if (device_argmax) (void)forward_argmax(v.data(), n + 1, pos);
            else (void)forward(v.data(), n + 1, pos, true);
            spec_save_ = false;
            auto argmax = [&](int r) {
                if (device_argmax) return verify_argmax_host_[size_t(r)];
                const float* row = logits_host_.data() + size_t(r) * V;
                return int(std::max_element(row, row + V) - row);
            };
            int a = 0, next = argmax(0);
            if (!device_argmax) log_margin(logits_host_.data(), V, int(out.size()));
            while (a < n && next == drafts[size_t(a)]) {
                out.push_back(next);
                ++a;
                next = argmax(a);
                if (!device_argmax) log_margin(logits_host_.data() + size_t(a) * V, V, int(out.size()));
            }
            rewind(pos + a + 1);
            for (int i = 0; i <= a; ++i) {
                used_.insert(used_.end(), spec_used_[size_t(i)].begin(), spec_used_[size_t(i)].end());
                missed_.insert(missed_.end(), spec_missed_[size_t(i)].begin(), spec_missed_[size_t(i)].end());
                if (!rt_count_.empty())
                    for (int expert : spec_used_[size_t(i)]) rt_count_[size_t(expert)] += 1;
            }
            { const auto a0 = std::chrono::steady_clock::now(); this->adapt(a + 1); prof_[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count(); }
            const double vsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count();
            d.verify_s += vsec;
            if (n + 1 < 16) d.verify_ms[n + 1] = d.verify_ms[n + 1] > 0 ? 0.8 * d.verify_ms[n + 1] + 0.2 * vsec * 1e3 : vsec * 1e3;
            if (d.on) dspark_inject(pos, a + 1);
            d.drafted += n;
            d.accepted += a;
            d.steps += 1;
            if (accept_hist_on) {
                if (a < 0 || size_t(a) >= accept_hist.size()) throw std::logic_error("ds41: draft acceptance histogram bounds");
                ++accept_hist[size_t(a)];
            }
            decoded(a + 1);
            if (from_history) { d.lookup_steps += 1; d.lookup_drafted += n; d.lookup_accepted += a; ema_lk = 0.7f * ema_lk + 0.3f * float(a); }
            else {
                if (d.on && !skipped_draft) {
                    ema_ds = 0.7f * ema_ds + 0.3f * float(a);
                    if (auto_backoff) auto_interval = a == 0 ? std::min(60, auto_interval * 2) : 15;
                }
                if (lookup && adapt) ema_lk = std::min(ema_lk + 0.25f, 4.0f);   // a lookup chance again after DSpark / plain steps
            }
            out.push_back(next);
            tok = next;
            pos += a + 1;
        }
        out.resize(size_t(steps));
        std::cerr << "DS41 dspark steps=" << d.steps << " drafted=" << d.drafted << " accepted=" << d.accepted << " accept_rate="
                  << double(d.accepted) / std::max<long>(d.drafted, 1) << " tokens_per_step=" << double(d.accepted + d.steps) / std::max<long>(d.steps, 1)
                  << " draft_s=" << d.draft_s << " (layers " << d.draft_layers_s << " of which router waits " << d.draft_route_s << ", head "
                  << d.draft_head_s << ") verify_s=" << d.verify_s << " inject_s=" << d.inject_s << " cut=" << d.sched_cut << " verify_ms_by_width=";
        for (int t = 1; t < 9; ++t) std::cerr << (t > 1 ? "," : "") << d.verify_ms[t];
        if (accept_hist_on) {
            std::cerr << " accept_hist=";
            for (size_t i = 0; i < accept_hist.size(); ++i) std::cerr << (i ? "," : "") << accept_hist[i];
        }
        if (lookup)
            std::cerr << " lookup_steps=" << d.lookup_steps << " lookup_drafted=" << d.lookup_drafted << " lookup_accepted=" << d.lookup_accepted
                      << " lookup_s=" << d.lookup_s;
        if (auto_draft) std::cerr << " auto_skipped=" << auto_skipped;
        std::cerr << "\n";
        if (d.cache_accesses)
            std::cerr << "DS41 dspark_cache_sim accesses=" << d.cache_accesses << " hits_16=" << d.cache_hits[0]
                      << " hits_24=" << d.cache_hits[1] << " hits_45=" << d.cache_hits[2] << " hits_90=" << d.cache_hits[3]
                      << " hits_100=" << d.cache_hits[4] << "\n";
        if (d.cache_experts)
            std::cerr << "DS41 dspark_cache hits=" << d.expert_cache_hits << " misses=" << d.expert_cache_misses << "\n";
        return out;
    }

    // forget positions >= pos (after a verify pass covering them): the token history, and the open compressor group
    void rewind(int pos) {
        const auto& c = cfg_;
        if (pos < 0 || size_t(pos) > history_.size()) throw std::logic_error("ds41: rewind outside history");
        if (pos == int(history_.size())) return;
        history_.resize(size_t(pos));
        const int last = pos - 1, i = last - pos0_;
        for (int l = 0; l < c.layers; ++l) {
            if (spec_v_.empty() || spec_v_[size_t(l)].bytes == 0) continue;
            if (last % 2 != 0 || i < 0 || i >= tokens_) continue;   // the kept positions close their groups (or precede the pass)
            const size_t hd = size_t(c.head_dim);
            ck(cudaMemcpyAsync(stash_v_[size_t(l)].p, spec_v_[size_t(l)].f(size_t(i) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "unstash");
            ck(cudaMemcpyAsync(stash_s_[size_t(l)].p, spec_s_[size_t(l)].f(size_t(i) * hd), hd * 4, cudaMemcpyDeviceToDevice, stream_), "unstash");
        }
    }

    bool gpu_timing_ = std::getenv("STRATA_DS41_GPU_TIMING") != nullptr;
    std::vector<cudaEvent_t> marks_;
    // STRATA_DS41_DECODE_TIMING=1: GPU time of one-token passes by phase (events from a reused pool)
    bool dtiming_ = std::getenv("STRATA_DS41_DECODE_TIMING") != nullptr;
    // STRATA_DS41_PREFILL_TIMING=1 (with DECODE_TIMING): the same phase events on prompt passes of any width, printed per
    // pass as totals and cleared, so the pass does not land in the decode averages
    bool pf_timing_ = std::getenv("STRATA_DS41_PREFILL_TIMING") != nullptr;
    // STRATA_DS41_EXPERT_DEBUG=1 / =2: a streamed prompt pass with the expert products / the expert DMAs skipped (see
    // stream_experts); timing only
    int expert_debug_ = std::getenv("STRATA_DS41_EXPERT_DEBUG") ? std::atoi(std::getenv("STRATA_DS41_EXPERT_DEBUG")) : 0;
    // PREFILL_TIMING: the streamed prompt passes' DMA busy time and bytes (events on the copy stream, per item) and the
    // compute stream's wait for each item
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> dma_ev_, wait_ev_;
    std::vector<cudaEvent_t> dma_pool_;   // (the producer thread and the pass both take from it)
    std::mutex dma_mu_;
    size_t dma_pool_used_ = 0;
    double dma_bytes_ = 0;
    cudaEvent_t dma_event() {
        std::lock_guard<std::mutex> lock(dma_mu_);
        if (dma_pool_used_ == dma_pool_.size()) {
            cudaEvent_t e;
            ck(cudaEventCreate(&e), "dma event");
            dma_pool_.push_back(e);
        }
        return dma_pool_[dma_pool_used_++];
    }
    std::vector<cudaEvent_t> dpool_, dmarks_;
    size_t dmark_used_ = 0;
    // phases: 0 layer start (Engram, mixes, norm), 1 q, 2 kv / compressor / indexer, 3 attention, 4 output projection,
    // 5 hc post + FFN mixes + norm, 6 router GEMV, 7 experts (router sync to the FFN's end), 8 hc post to the next layer
    double dtime_[10] = {}, dtokens_ = 0;
    std::vector<int> dmark_id_;
    double xt_[3] = {};   // decode FFN host time: enqueue, lane, peer
    double peer_gpu_ms() const { return peer_ ? peer_->gpu_ms : 0.0; }
    double peer_late_ms() const { return peer_late_ms_; }
    const double* peer_split_ms() const { static const double zero[5] = {}; return peer_ ? peer_->split_ms : zero; }
    void dmark(int id) {
        dmark_id_.push_back(id);
        if (dmark_used_ == dpool_.size()) {
            cudaEvent_t e;
            ck(cudaEventCreate(&e), "timing event");
            dpool_.push_back(e);
        }
        cudaEvent_t e = dpool_[dmark_used_++];
        ck(cudaEventRecord(e, stream_), "timing event");
        dmarks_.push_back(e);
    }
    void mark(int) {
        cudaEvent_t e;
        ck(cudaEventCreate(&e), "timing event");
        ck(cudaEventRecord(e, stream_), "timing event");
        marks_.push_back(e);
    }   // token-layers enc-skip dropped (all passes)
private:
    uint8_t* hstage_[2] = {nullptr, nullptr};
    cudaEvent_t stage_uploaded_[2] = {nullptr, nullptr};
    size_t stage_bytes_ = 0;
    Dev stage_[2], s_x_, s_xq_, s_gate_, s_up_, s_h_, s_hq_, s_out_, s_idx_, s_w_;
    cudaEvent_t stage_copied_[2] = {nullptr, nullptr}, stage_free_[2] = {nullptr, nullptr};
    bool stage_used_[2] = {false, false};
    unsigned stage_next_ = 0;
    cudaStream_t copy_s_ = nullptr;
    int* s_hidx_ = nullptr;
    float* s_hw_ = nullptr;
    int32_t* s_hbound_ = nullptr;
    bool use_mmq_ = false;
    int rcap_ = 8192;
    std::unique_ptr<strata::prefill::mmq::Context> mmq_;
    Dev s_bound_, m_xq_, s_gu_, s_hf_, m_hq_, s_dout_, s_iota_;
};

namespace {

// a tools/ds41f_llama_dump.cpp tensor: int32 type, int64 ne[4], f32 or i32 elements
bool read_dump(const std::filesystem::path& file, std::vector<float>& v, std::vector<int64_t>& ne) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    int32_t type = 0;
    f.read(reinterpret_cast<char*>(&type), 4);
    ne.assign(4, 0);
    f.read(reinterpret_cast<char*>(ne.data()), 32);
    const int64_t n = ne[0] * ne[1] * ne[2] * ne[3];
    v.resize(size_t(n));
    if (type == 26) {
        std::vector<int32_t> iv(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(iv.data()), n * 4);
        for (int64_t i = 0; i < n; ++i) v[size_t(i)] = float(iv[size_t(i)]);
    } else {
        f.read(reinterpret_cast<char*>(v.data()), n * 4);
    }
    return bool(f);
}

std::vector<int> parse_ids(std::string text) {
    if (text.starts_with("@")) {
        std::ifstream f(text.substr(1));
        if (!f) throw std::runtime_error("ds41: cannot read " + text.substr(1));
        text.assign(std::istreambuf_iterator<char>(f), {});
    }
    for (char& ch : text) if (ch == ',' || ch == '\n') ch = ' ';
    std::vector<int> ids;
    std::stringstream ss(text);
    for (int v; ss >> v;) ids.push_back(v);
    return ids;
}

}  // namespace

bool is_ds41(const std::string& path) {
    try {
        GgufFile f(path);
        const auto* a = f.get("general.architecture");
        return a && a->type == MetaType::STRING && a->s == "deepseek41";
    } catch (...) {
        return false;
    }
}

// STRATA_DS41_SESSIONS: every session's prompt, then `steps` greedy tokens for all of them in passes of one token per
// session. STRATA_DS41_SESSION_CHECK=n: afterwards the first n sessions again, alone (prompt, then plain decoding), and
// their tokens against the batched ones.
std::unique_ptr<std::barrier<>> g_decode_barrier;
int run_sessions(Decoder& d, const std::vector<int>& prompt, int S, int L, int steps) {
    const size_t V = size_t(d.config().vocab);
    auto argmax = [](const float* v, size_t n) { return int(std::max_element(v, v + n) - v); };
    auto offset = [&](int s) { return size_t(s) * size_t(L) % (prompt.size() - size_t(L) + 1); };
    auto prefill = [&](int s) {   // session s's prompt on the single-sequence state; its first token
        const int* toks = prompt.data() + offset(s);
        std::vector<float> last;
        for (int i = 0; i < L; i += d.chunk()) {
            const int n = std::min(L - i, d.chunk());
            const auto& out = d.forward(toks + i, n, i, false);
            last.assign(out.end() - std::ptrdiff_t(V), out.end());
        }
        return argmax(last.data(), V);
    };
    if (g_decode_barrier) d.pin_host_thread();
    d.sessions_init(S);
    const auto p0 = std::chrono::steady_clock::now();
    std::vector<int> next(size_t(S), 0);
    for (int s = 0; s < S; ++s) {
        next[size_t(s)] = prefill(s);
        d.park(s);
    }
    const auto p1 = std::chrono::steady_clock::now();
    std::fill(std::begin(d.prof_), std::end(d.prof_), 0.0);
    d.tier_hits_ = d.tier_all_ = 0;
    d.canon_routes_ = d.canon_cpu_routes_ = d.canon_streamed_ = 0;
    d.reset_own_routes();
    d.reset_lane_stats();
    d.attn_host_s_ = d.ffn_host_s_ = 0;
    d.own_prep_ms_ = d.own_send_ms_ = 0; d.own_starts_ = 0;
    d.prefill_done();
    std::fill(std::begin(d.xt_), std::end(d.xt_), 0.0);
    std::fill(std::begin(d.dtime_), std::end(d.dtime_), 0.0);
    d.dtokens_ = 0;
    const double peer_gpu0 = d.peer_gpu_ms(), peer_late0 = d.peer_late_ms();
    if (g_decode_barrier) g_decode_barrier->arrive_and_wait();   // dual: both halves start the timed decode together
    const auto t2 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> out(static_cast<size_t>(S));
    // STRATA_DS41_SESSION_TF=1 (with SESSION_CHECK=n): keep the first n sessions' logits for the teacher-forced check below
    const bool tf = strata_env_is_one("STRATA_DS41_SESSION_TF");
    const int tf_n = !tf || !std::getenv("STRATA_DS41_SESSION_CHECK") ? 0 : std::min(S, std::atoi(std::getenv("STRATA_DS41_SESSION_CHECK")));
    std::vector<std::vector<std::vector<float>>> blog(static_cast<size_t>(tf_n));
    for (int k = 0; k < steps; ++k) {
        for (int s = 0; s < S; ++s) out[size_t(s)].push_back(next[size_t(s)]);
        if (k + 1 == steps) break;
        const auto& lg = d.forward_sessions(next.data(), S);
        for (int s = 0; s < tf_n; ++s) blog[size_t(s)].emplace_back(lg.begin() + std::ptrdiff_t(size_t(s) * V), lg.begin() + std::ptrdiff_t(size_t(s + 1) * V));
        for (int s = 0; s < S; ++s) next[size_t(s)] = argmax(lg.data() + size_t(s) * V, V);
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(p1 - p0).count(), decode_s = std::chrono::duration<double>(t3 - t2).count();
    if (d.dtokens_ > 0) {   // (the prompts' passes are wider than the timed ones)
        static const char* names[10] = {"pre", "q", "kv_index", "attn", "out", "ffn_pre", "router", "experts", "tail", "lane_wait"};
        std::cerr << "DS41 decode_gpu_ms_per_pass";
        for (int q = 0; q < 10; ++q) std::cerr << " " << names[q] << "=" << d.dtime_[q] / d.dtokens_;
        std::cerr << " passes=" << d.dtokens_ << " peer_gpu=" << (d.peer_gpu_ms() - peer_gpu0) / d.dtokens_
                  << " peer_late=" << (d.peer_late_ms() - peer_late0) / d.dtokens_ << "\n";
    }
    d.miss_line(double(std::max(1, steps - 1)));   // lane_async busy ms per pass
    if (d.own_starts_ > 0) std::cerr << "DS41 worker_request per layer call ms: prep " << d.own_prep_ms_ / d.own_starts_ << " send " << d.own_send_ms_ / d.own_starts_ << "\n";
    std::cerr << "DS41 host_ms_per_pass attention=" << d.attn_host_s_ * 1e3 / std::max(1, steps - 1) << " ffn=" << d.ffn_host_s_ * 1e3 / std::max(1, steps - 1) << "\n";
    if (steps > 1) {   // routes per layer per pass: seen by the CPU path / on the GPU tier / sent to the worker (the rest: the B550 lane)
        const double lp = double(steps - 1) * d.config().layers;
        std::cerr << "DS41 routes_per_layer total=" << double(d.canon_cpu_routes_) / lp << " canon_gpu=" << double(d.canon_routes_) / lp
                  << " worker=" << double(d.own_routes()) / lp << " b550_lane=" << (double(d.canon_cpu_routes_) - double(d.canon_routes_) - double(d.own_routes())) / lp << "\n";
    }
    if (d.own_ms_[3] > 0)
        std::cerr << "DS41 remote_own_decode layer_calls=" << d.own_ms_[3] << " (per layer call ms gpu_sync " << d.own_ms_[0] / d.own_ms_[3] << " / worker_wait "
                  << d.own_ms_[1] / d.own_ms_[3] << " / merge+lane " << d.own_ms_[2] / d.own_ms_[3] << ") request_to_reply_ms=" << d.own_latency_ms_ / d.own_ms_[3] << "\n";
    std::cerr << "DS41 profile_s step=" << d.prof_[0] << " gpu_to_router=" << d.prof_[1] << " cpu_experts=" << d.prof_[2]
              << " gpu_tier_hit=" << (d.tier_all_ > 0 ? d.tier_hits_ / d.tier_all_ : 0) << " engram=" << d.prof_[4] << " index_host=" << d.prof_[5]
              << " head=" << d.prof_[6] << " (ffn: enqueue " << d.xt_[0] << " lane " << d.xt_[1] << " peer " << d.xt_[2] << ")\n";
    for (int s = 0; s < std::min(S, 4); ++s) {
        std::cout << "tokens[" << s << "]:";
        for (int t : out[size_t(s)]) std::cout << ' ' << t;
        std::cout << "\n";
    }
    const double passes = std::max(1, steps - 1);
    std::cerr << "DS41 sessions=" << S << " prompt=" << L << " prefill_s=" << prefill_s << " generated=" << steps << " pass_ms="
              << decode_s * 1e3 / passes << " decode_tok_s=" << S * passes / std::max(decode_s, 1e-9)
              << " per_session_tok_s=" << passes / std::max(decode_s, 1e-9)
              << " canon_hit=" << (d.canon_cpu_routes_ ? double(d.canon_routes_) / double(d.canon_cpu_routes_) : 0.0) << "\n";
    const int check = std::getenv("STRATA_DS41_SESSION_CHECK") ? std::min(S, std::atoi(std::getenv("STRATA_DS41_SESSION_CHECK"))) : 0;
    if (tf_n > 0) {   // the alone run is fed the batched run's tokens: top-1 agreement and KL(alone || batched) of the same positions
        auto lse = [&](const float* v) { double m = *std::max_element(v, v + V), a = 0; for (size_t i = 0; i < V; ++i) a += std::exp(double(v[i]) - m); return m + std::log(a); };
        double kl_sum = 0, kl_max = 0; long agree = 0, total = 0;
        for (int s = 0; s < tf_n; ++s) {
            d.reset_sequence();
            int tok = prefill(s);
            for (int k = 0; k + 1 < steps; ++k) {
                const auto& lg = d.step(out[size_t(s)][size_t(k)], L + k);
                const auto& bl = blog[size_t(s)][size_t(k)];
                const double la = lse(lg.data()), lb = lse(bl.data());
                double kl = 0;
                for (size_t i = 0; i < V; ++i) { const double pa = std::exp(double(lg[i]) - la); if (pa > 1e-12) kl += pa * ((double(lg[i]) - la) - (double(bl[i]) - lb)); }
                kl_sum += kl; kl_max = std::max(kl_max, kl);
                agree += argmax(lg.data(), V) == argmax(bl.data(), V);
                ++total;
            }
            (void)tok;
        }
        std::cerr << "DS41 session_tf positions=" << total << " top1_agree=" << double(agree) / std::max<long>(total, 1) << " kl_mean=" << kl_sum / std::max<long>(total, 1)
                  << " kl_max=" << kl_max << "\n";
        return 0;
    }
    for (int s = 0; s < check; ++s) {
        d.reset_sequence();
        int tok = prefill(s), same = 0, prefix = -1;
        for (int k = 0; k < steps; ++k) {
            const bool eq = tok == out[size_t(s)][size_t(k)];
            same += eq;
            if (!eq && prefix < 0) prefix = k;
            if (k + 1 == steps) break;
            const auto& lg = d.step(tok, L + k);
            tok = argmax(lg.data(), V);
        }
        std::cerr << "DS41 session_check s=" << s << " same=" << same << "/" << steps << " first_diff=" << prefix << "\n";
    }
    return 0;
}

int engram_profile_run(const std::string& model, const std::string& streams, const std::string& out) {
    core::ModelArtifact art(model);
    const Config c = Config::read(*art.metadata(), art.descriptor(), [&](const std::string& n) { try { art.at(n); return true; } catch (...) { return false; } },
                                  [&](const std::string& n) { return tensor_ints(art.at(n)); });
    if (c.engram_layers.empty()) throw std::runtime_error("ds41: no Engram layers");
    // the distinct tables in Engram-layer order
    std::vector<std::string> names;
    std::vector<int> table_of;
    for (size_t e = 0; e < c.engram_layers.size(); ++e) {
        const std::string n = c.engram_per_layer ? "blk." + std::to_string(c.engram_layers[e]) + ".engram_embd.weight" : std::string("per_layer_token_embd.weight");
        int k = -1;
        for (size_t j = 0; j < names.size(); ++j) if (names[j] == n) k = int(j);
        if (k < 0) { k = int(names.size()); names.push_back(n); }
        table_of.push_back(k);
    }
    std::vector<int64_t> rows_of(names.size());
    std::vector<std::vector<uint32_t>> count(names.size());
    for (size_t k = 0; k < names.size(); ++k) {
        const auto& t = art.at(names[k]);
        rows_of[k] = int64_t(t.bytes / ggml_row_size(ggml_type(int(t.tensor->type)), c.engram_dim));
        count[k].assign(size_t(rows_of[k]), 0);
    }
    const int cols = c.engram_cols();
    std::vector<int64_t> rows(static_cast<size_t>(cols));
    size_t tokens = 0;
    std::stringstream ss(streams);
    for (std::string item; std::getline(ss, item, ';');) {
        if (item.empty()) continue;
        const auto ids = parse_ids(item);
        auto tok_at = [&](int q) { return ids[size_t(q)]; };
        for (int pos = 0; pos < int(ids.size()); ++pos)
            for (size_t e = 0; e < c.engram_layers.size(); ++e) {
                Decoder::engram_rows_for(c, int(e), pos, tok_at, rows.data());
                auto& cnt = count[size_t(table_of[e])];
                for (int j = 0; j < cols; ++j)
                    if (rows[size_t(j)] >= 0 && rows[size_t(j)] < rows_of[size_t(table_of[e])] && cnt[size_t(rows[size_t(j)])] != UINT32_MAX) ++cnt[size_t(rows[size_t(j)])];
            }
        tokens += ids.size();
        std::cerr << "DS41 engram profile: " << item << " " << ids.size() << " tokens\n";
    }
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("ds41: cannot write " + out);
    f.write("DS41ROWS", 8);
    const uint32_t nt = uint32_t(names.size());
    f.write(reinterpret_cast<const char*>(&nt), 4);
    for (size_t k = 0; k < names.size(); ++k) {
        std::vector<std::pair<uint32_t, int64_t>> ranked;
        for (int64_t r = 0; r < rows_of[k]; ++r) if (count[k][size_t(r)]) ranked.push_back({count[k][size_t(r)], r});
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        const uint64_t rows64 = uint64_t(rows_of[k]), n = ranked.size();
        f.write(reinterpret_cast<const char*>(&rows64), 8);
        f.write(reinterpret_cast<const char*>(&n), 8);
        uint64_t refs = 0, top = 0;
        for (size_t i = 0; i < ranked.size(); ++i) {
            f.write(reinterpret_cast<const char*>(&ranked[i].second), 8);
            f.write(reinterpret_cast<const char*>(&ranked[i].first), 4);
            refs += ranked[i].first;
            if (i < ranked.size() / 2) top += ranked[i].first;
        }
        std::cerr << "DS41 engram profile table " << k << " (" << names[k] << "): rows=" << rows64 << " referenced=" << n << " references=" << refs
                  << " top_half_share=" << (refs ? double(top) / double(refs) : 0.0) << "\n";
    }
    std::cerr << "DS41 engram profile: " << tokens << " tokens -> " << out << "\n";
    return 0;
}

int run(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: strata-glm-decode DS41-MODEL.gguf <ids|@file> [steps] [--threads=N] [--context=N] "
                     "[--check=DUMPDIR] [--dump-logits=PATH]\n";
        return 2;
    }
    int steps = 0, threads = int(std::max(1u, std::thread::hardware_concurrency() / 2)), context = 0;
    std::string check_dir, dump_logits, save_all, engram_profile;
    bool force = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.starts_with("--engram-profile=")) engram_profile = a.substr(17);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--context=")) context = std::stoi(a.substr(10));
        else if (a.starts_with("--check=")) check_dir = a.substr(8);
        else if (a == "--check-force") force = true;   // each layer from the reference's input and expert picks
        else if (a.starts_with("--dump-logits=")) dump_logits = a.substr(14);
        else if (a.starts_with("--save-all-logits=")) save_all = a.substr(18);   // every prompt position's logits, f32
        else if (a.starts_with("--remote-tp=")) setenv("STRATA_DS41_REMOTE_TP", a.substr(12).c_str(), 1);   // HOST:PORT here
        else if (a.starts_with("--remote-tp-share=")) setenv("STRATA_DS41_REMOTE_SHARE", a.substr(18).c_str(), 1);   // k/9
        else if (!a.starts_with("--")) steps = std::stoi(a);
        else throw std::invalid_argument("ds41: unknown option " + a);
    }
    // --engram-profile=OUT: no model weights; the Engram rows every position of the token streams (argv[2]: @file, or
    // several @file separated by ';') would read, counted per table and written ranked (STRATA_DS41_ENGRAM_ROWS)
    if (!engram_profile.empty()) return engram_profile_run(argv[1], argv[2], engram_profile);
    (void)cpu_trim_threshold();   // reject invalid tuning before allocating model resources
    if (strata_env_is_one("STRATA_DS41_DSPARK_CACHE_EXPERTS")) {
        const int slots = dspark_cache_slots();
        if (const char* path = std::getenv("STRATA_DS41_DSPARK"); path && *path) {
            // Check cache capacity against draft metadata before loading target
            // weights or allocating any device staging buffers.
            GgufFile draft(path);
            auto value = [&](const char* key, double fallback) {
                const auto* v = draft.get(key);
                return v ? (v->is_num() ? v->num() : v->items.empty() ? fallback : v->items[0].num()) : fallback;
            };
            const double block = value("dflash.block_size", 5), top = value("dflash.expert_used_count", 3);
            if (!std::isfinite(block) || !std::isfinite(top) || block < 1 || block > 7 || top < 1 || top > 16 ||
                slots < (strata_env_is_one("STRATA_DS41_DSPARK_EXTEND") ? 7 : int(block)) * int(top))
                throw std::invalid_argument("ds41: DSpark cache cannot hold one draft batch");
        }
    }
#if defined(__HIP_PLATFORM_AMD__)
    const char* draft_path = std::getenv("STRATA_DS41_DSPARK");
    if (draft_path && *draft_path && strata_env_is_one("STRATA_DS41_DSPARK_STREAM_EXPERTS")) {
        if (strata_env_is_one("STRATA_DS41_DSPARK_PIN_EXPERTS"))
            throw std::runtime_error("ds41: DSpark mapped expert pinning is disabled on AMD; use batch staging without pinning");
        const char* hybrid = std::getenv("STRATA_DS41_PREFILL_HYBRID");
        if (strata_env_is_one("STRATA_DS41_RAM_COPY") || strata_env_is_one("STRATA_DS41_DENSE_MMQ") ||
            strata_env_is_one("STRATA_DS41_DECODE_MMQ") || (hybrid && std::atoi(hybrid) > 0))
            throw std::runtime_error("ds41: streamed DSpark cannot use unbounded or unvalidated AMD memory paths");
        const char* pin_gb = std::getenv("STRATA_DS41_PIN_GB");
        char* end = nullptr;
        const double pin = pin_gb ? std::strtod(pin_gb, &end) : 60.0;
        if (!pin_gb || end == pin_gb || *end || !std::isfinite(pin) || pin != 0.0 ||
            strata_env_is_one("STRATA_DS41_PIN_EXPERTS"))
            throw std::runtime_error("ds41: streamed DSpark on AMD requires STRATA_DS41_PIN_GB=0 and no mapped expert pinning");
        const char* populate = std::getenv("STRATA_DS41_POPULATE");
        if (!populate || std::string(populate) != "0")
            throw std::runtime_error("ds41: streamed DSpark on AMD requires STRATA_DS41_POPULATE=0 to avoid bulk page faults");
        const char* chunk_env = std::getenv("STRATA_DS41_PREFILL_CHUNK");
        const int safe_chunk = chunk_env ? std::atoi(chunk_env) : 0;
        if (!strata_env_is_one("STRATA_DS41_DSPARK_BATCH_STAGE") || safe_chunk < 8 || safe_chunk > 64 ||
            strata_env_is_one("STRATA_DS41_GROW"))
            throw std::runtime_error("ds41: streamed DSpark on AMD requires batch staging, prefill chunk 8..64 and GROW off");
#if defined(__linux__)
        std::ifstream pressure("/proc/pressure/memory"), meminfo("/proc/meminfo");
        std::string line;
        double avg10 = -1, avg60 = -1;
        unsigned long long available_kib = 0;
        while (std::getline(pressure, line)) if (line.compare(0, 5, "full ") == 0) {
            std::istringstream fields(line.substr(5));
            std::string field;
            while (fields >> field) {
                if (field.rfind("avg10=", 0) == 0) avg10 = std::stod(field.substr(6));
                if (field.rfind("avg60=", 0) == 0) avg60 = std::stod(field.substr(6));
            }
        }
        while (std::getline(meminfo, line)) if (line.rfind("MemAvailable:", 0) == 0)
            std::istringstream(line.substr(13)) >> available_kib;
        if (avg10 < 0 || avg60 < 0 || avg10 > 5 || avg60 > 5 || available_kib < (16ull << 20))
            throw std::runtime_error("ds41: insufficient host memory headroom or excessive memory pressure for streamed DSpark");
#endif
    }
#endif
    // several prompts separated by ';' (each @file or text) run one after another from an empty sequence (their
    // logits appended to --save-all-logits in turn); the generation continues the last one
    std::vector<std::vector<int>> prompts;
    {
        std::stringstream pss(argv[2]);
        for (std::string item; std::getline(pss, item, ';');)
            if (!item.empty()) prompts.push_back(parse_ids(item));
    }
    if (prompts.empty() || prompts.back().empty()) throw std::invalid_argument("ds41: empty prompt");
    const auto& prompt = prompts.back();
    // STRATA_DS41_SESSIONS=S: S sessions decoded together, session s's prompt the STRATA_DS41_SESSION_LEN (prompt / S)
    // tokens from s x that length (wrapping inside the prompt)
    const int sessions = std::getenv("STRATA_DS41_SESSIONS") ? std::max(0, std::atoi(std::getenv("STRATA_DS41_SESSIONS"))) : 0;
    const int slen = !sessions ? 0 : std::getenv("STRATA_DS41_SESSION_LEN") ? std::atoi(std::getenv("STRATA_DS41_SESSION_LEN"))
                                                                           : int(prompt.size()) / sessions;
    if (sessions && (slen < 1 || slen > int(prompt.size()))) throw std::invalid_argument("ds41: session length");
    const int plen = sessions ? slen : int(prompt.size());
    if (!context) context = plen + steps + 1;
    const auto t0 = std::chrono::steady_clock::now();
    // a long prompt runs as one layer-major pass (up to STRATA_DS41_PROMPT_BATCH, 4096 tokens): every expert crosses
    // PCIe once per pass; shorter prompts take chunks of STRATA_DS41_PREFILL_CHUNK (256) with the CPU lane
    int chunk = 0;
    {
        const char* v = std::getenv("STRATA_DS41_PROMPT_BATCH");
        const int batch = v ? std::max(1, std::atoi(v)) : 4096;
        const int stream_min = std::getenv("STRATA_DS41_STREAM_MIN") ? std::atoi(std::getenv("STRATA_DS41_STREAM_MIN")) : 1024;
        if (plen >= stream_min && !std::getenv("STRATA_DS41_PREFILL_CHUNK") && check_dir.empty())
            chunk = std::min(plen, batch);
    }
    // STRATA_DS41_SYNC_SPIN=1: the host spins on the GPU's syncs (the router's every layer) instead of the runtime's
    // default yield / block, which wakes later; costs a core while it waits
    if (std::getenv("STRATA_DS41_SYNC_SPIN") && std::string(std::getenv("STRATA_DS41_SYNC_SPIN")) == "1")
        ck(cudaSetDeviceFlags(cudaDeviceScheduleSpin), "spin sync");
    Decoder d(argv[1], context, threads, chunk);
    if (const char* r = std::getenv("STRATA_DS41_ROUTES")) d.log_routes(r);
    // STRATA_DS41_DUAL=1 (with STRATA_DS41_SESSIONS=S): a second decoder in this process views the first one's GPU weights and canonical tier
    // and runs S/2 of the sessions on its own host thread, CUDA stream, lane and worker connection (every port +10): while one half waits
    // for the worker, the other half's trunk runs on the GPU, and the worker serves both. Needs a static tier (no CANON_ADAPT) and the worker
    // started with --root2/--engram-port2.
    // STRATA_DS41_CANON_SETTLE=trace (with CANON_ADAPT + OWN_FETCH): the adaptive tier is filled once from that route trace and then frozen (a static, workload-fitted
    // tier: what two decoders can share, and the same tier for a single decoder to compare with)
    if (const char* tr = std::getenv("STRATA_DS41_CANON_SETTLE"); tr && *tr) {
        d.adapt_settle(tr, 3000);
        unsetenv("STRATA_DS41_CANON_ADAPT");
    }
    std::unique_ptr<Decoder> d2;
    if (strata_env_is_one("STRATA_DS41_DUAL") && std::getenv("STRATA_DS41_SESSIONS")) {
        auto bump = [](const char* name) {   // HOST:PORT -> HOST:PORT+10
            if (const char* v = std::getenv(name); v && *v) {
                std::string a = v;
                const auto colon = a.rfind(':');
                if (colon != std::string::npos) setenv(name, (a.substr(0, colon + 1) + std::to_string(std::atoi(a.c_str() + colon + 1) + 10)).c_str(), 1);
            }
        };
        bump("STRATA_DS41_REMOTE_OWN");
        bump("STRATA_DS41_ENGRAM_REMOTE");
        bump("STRATA_DS41_OWN_FETCH");
        setenv("STRATA_DS41_LANE_CPU_OFFSET", std::to_string(threads).c_str(), 1);   // the second lane on the next cores
        {   // the first lane pinned this thread to its core 0; the second lane enumerates the cores this thread may use
            cpu_set_t all;
            CPU_ZERO(&all);
            for (long i = 0; i < sysconf(_SC_NPROCESSORS_CONF) && i < CPU_SETSIZE; ++i) CPU_SET(int(i), &all);
            pthread_setaffinity_np(pthread_self(), sizeof all, &all);
        }
        d2 = std::make_unique<Decoder>(argv[1], context, threads, chunk, &d);
        std::cerr << "DS41 dual: second decoder ready\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "DS41 load_s=" << std::chrono::duration<double>(t1 - t0).count() << " threads=" << threads << "\n";
    if (sessions && d2) {
        const int s1 = (sessions + 1) / 2, s2 = sessions - s1;
        std::vector<int> prompt2(prompt);
        std::rotate(prompt2.begin(), prompt2.begin() + std::ptrdiff_t(size_t(s1) * size_t(slen) % prompt.size()), prompt2.end());
        g_decode_barrier = std::make_unique<std::barrier<>>(2);
        int rc2 = 0;
        std::thread t2([&] { try { rc2 = run_sessions(*d2, prompt2, s2, slen, steps); } catch (const std::exception& e) { std::cerr << "ds41 (second decoder): " << e.what() << "\n"; rc2 = 1; } });
        int rc1 = 1;
        try { rc1 = run_sessions(d, prompt, s1, slen, steps); } catch (const std::exception& e) { std::cerr << "ds41: " << e.what() << "\n"; }
        t2.join();
        return rc1 | rc2;
    }
    if (sessions) return run_sessions(d, prompt, sessions, slen, steps);
    if (std::getenv("STRATA_DS41_GPU_BENCH")) { d.gpu_bench(); return 0; }
    // --check: every observed tensor of every prompt position against the dump's s<pos>/<name>-<layer>.bin
    int position = 0;
    double worst = 0, kl_sum = 0;
    int kl_n = 0, top1 = 0;
    std::string worst_at;
    std::vector<float> host;
    auto compare = [&](const std::string& name, int layer, const float* got, size_t n) {
        const std::filesystem::path f = std::filesystem::path(check_dir) / ("s" + std::to_string(position)) /
                                        (layer >= 0 ? name + "-" + std::to_string(layer) + ".bin" : name + ".bin");
        std::vector<float> ref;
        std::vector<int64_t> ne;
        if (!read_dump(f, ref, ne)) return;
        if (ref.size() != n) {
            std::cerr << "CHECK " << f.string() << " size " << ref.size() << " vs " << n << "\n";
            return;
        }
        const std::string at = name + (layer >= 0 ? "-" + std::to_string(layer) : "");
        if (name == "ffn_moe_topk") {   // ids: the same set, in any order
            std::vector<float> a(got, got + n), b = ref;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a != b) {
                std::cerr << "CHECK pos=" << position << " " << at << " differ: got";
                for (size_t i = 0; i < n; ++i) std::cerr << ' ' << got[i];
                std::cerr << " ref";
                for (size_t i = 0; i < n; ++i) std::cerr << ' ' << ref[i];
                std::cerr << "\n";
            }
            return;
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < n; ++i) { num += double(got[i] - ref[i]) * (got[i] - ref[i]); den += double(ref[i]) * ref[i]; }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        if (rel > worst) { worst = rel; worst_at = f.string(); }
        if (name == "result_output") {   // the reference's logits: argmax and KL(ref || ours) of this position
            auto softmax = [](const float* v, size_t n) {
                std::vector<double> p(n);
                const double mx = *std::max_element(v, v + n);
                double sum = 0;
                for (size_t i = 0; i < n; ++i) { p[i] = std::exp(double(v[i]) - mx); sum += p[i]; }
                for (auto& x : p) x /= sum;
                return p;
            };
            const auto pr = softmax(ref.data(), n), pg = softmax(got, n);
            double kl = 0;
            for (size_t i = 0; i < n; ++i) if (pr[i] > 0) kl += pr[i] * std::log(pr[i] / std::max(pg[i], 1e-300));
            const size_t ar = size_t(std::max_element(ref.begin(), ref.end()) - ref.begin());
            const size_t ag = size_t(std::max_element(got, got + n) - got);
            std::cerr << "LOGITS pos=" << position << " argmax ours=" << ag << " ref=" << ar << " KL=" << kl << " rel=" << rel << "\n";
            kl_sum += kl;
            top1 += ag == ar;
            ++kl_n;
            return;
        }
        if (layer <= 2 || layer % 8 == 0 || layer == d.config().layers - 1 || rel > 0.05)
            std::cerr << "CHECK pos=" << position << " " << at << " rel=" << rel << "\n";
    };
    if (!check_dir.empty() && force)
        d.reference = [&](const std::string& name, int layer, std::vector<float>& out) {
            std::vector<int64_t> ne;
            return read_dump(std::filesystem::path(check_dir) / ("s" + std::to_string(position)) /
                                 (name + "-" + std::to_string(layer) + ".bin"), out, ne);
        };
    if (!check_dir.empty()) {
        d.observe = [&](const std::string& name, int layer, const float* dev, size_t n) {
            host.resize(n);
            ck(cudaDeviceSynchronize(), "check sync");
            ck(cudaMemcpy(host.data(), dev, n * 4, cudaMemcpyDeviceToHost), "check copy");
            compare(name, layer, host.data(), n);
        };
        d.observe_host = compare;
    }
    std::vector<float> last;
    std::ofstream all_logits;
    if (!save_all.empty()) all_logits.open(save_all, std::ios::binary);
    // the prompt in chunks (one token at a time under --check, whose hooks compare single-token passes)
    const int pass = check_dir.empty() ? d.chunk() : 1;
    const size_t V = size_t(d.config().vocab);
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
    const auto& pr = prompts[pi];
    if (pi) { d.reset_sequence(); std::cerr << "DS41 prompt " << pi + 1 << " of " << prompts.size() << " tokens=" << pr.size() << "\n"; }
    for (size_t i = 0; i < pr.size(); i += size_t(pass)) {
        position = int(i);
        const int n = int(std::min(pr.size() - i, size_t(pass)));
        const double copy0 = d.prof_[7];
        const auto c0 = std::chrono::steady_clock::now();
        const auto& out = d.forward(pr.data() + i, n, int(i), all_logits.is_open());
        if (std::getenv("STRATA_DS41_CHUNK_LOG"))
            std::cerr << "DS41 chunk pos=" << i << " tokens=" << n << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count()
                      << " stage_copy_s=" << d.prof_[7] - copy0 << " gpu_to_router_s=" << d.prof_[1] << " experts_s=" << d.prof_[2]
                      << " engram_s=" << d.prof_[4] << " index_s=" << d.prof_[5] << " stream_wait_s=" << d.stream_wait_s_
                      << " enc_skip_token_layers=" << d.skipped_ << " fetch_s=" << d.fetch_s_ << " peer_prep_s=" << d.peer_prep_s_
                      << " peer_wait_s=" << d.peer_wait_s_ << " stream_enqueue_s=" << d.stream_enqueue_s_ << " attention_host_s=" << d.attn_host_s_
                      << " ffn_host_s=" << d.ffn_host_s_ << " before_layers_s=" << d.before_loop_s_ << " layers_s=" << d.loop_s_ << "\n";
        if (all_logits) all_logits.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
        last.assign(out.end() - std::ptrdiff_t(V), out.end());
    }
    }
    const auto t2 = std::chrono::steady_clock::now();
    // the prompt's own profile, then the counters restart so the final profile is decode alone
    std::cerr << "DS41 profile_prompt_s step=" << d.prof_[0] << " gpu_to_router=" << d.prof_[1] << " cpu_experts=" << d.prof_[2]
              << " engram=" << d.prof_[4] << " index_host=" << d.prof_[5] << " head=" << d.prof_[6] << "\n";    d.hybrid_line();
    std::fill(std::begin(d.prof_), std::end(d.prof_), 0.0);
    d.tier_hits_ = d.tier_all_ = 0;
    d.canon_routes_ = d.canon_cpu_routes_ = d.canon_streamed_ = 0;
    std::fill(std::begin(d.own_ms_), std::end(d.own_ms_), 0.0);
    d.prefill_done();
    d.miss_test_drop();
    const auto t2b = std::chrono::steady_clock::now();
    std::fill(std::begin(d.dtime_), std::end(d.dtime_), 0.0);
    d.dtokens_ = 0;
    std::cerr << "DS41 prefill_done_s=" << std::chrono::duration<double>(t2b - t2).count() << "\n";
    if (!check_dir.empty())
        std::cerr << "CHECK worst rel=" << worst << " at " << worst_at << " mean_KL=" << kl_sum / std::max(kl_n, 1)
                  << " top1=" << top1 << "/" << kl_n << "\n";
    if (!dump_logits.empty()) {
        std::ofstream f(dump_logits, std::ios::binary);
        f.write(reinterpret_cast<const char*>(last.data()), std::streamsize(last.size() * 4));
    }
    std::vector<int> out;
    int pos = int(prompt.size());
    const double attn_host0 = d.attn_host_s_, ffn_host0 = d.ffn_host_s_, route_wait0 = d.route_wait_s_, loop0 = d.loop_s_;
    std::fill(std::begin(d.xt_), std::end(d.xt_), 0.0);
    const double peer_gpu0 = d.peer_gpu_ms(), peer_late0 = d.peer_late_ms();
    // STRATA_DS41_DSPARK_OFF=1: the drafter loaded (the same residency) but plain decoding, to compare against
    const bool lookup_only = std::getenv("STRATA_DS41_LOOKUP") && std::string(std::getenv("STRATA_DS41_LOOKUP")) == "1";
    if ((d.dsp_.on || lookup_only) && steps > 0 && !std::getenv("STRATA_DS41_DSPARK_OFF")) out = d.spec_generate(int(std::max_element(last.begin(), last.end()) - last.begin()), pos, steps);
    else
    for (int s = 0; s < steps; ++s) {
        const int tok = int(std::max_element(last.begin(), last.end()) - last.begin());
        Decoder::log_margin(last.data(), int(last.size()), s);
        out.push_back(tok);
        if (s + 1 == steps) break;
        last = d.step(tok, pos++);
        d.decoded(1);
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t2 - t1).count();
    const double decode_s = std::chrono::duration<double>(t3 - t2b).count();
    if (d.dtokens_ > 0) {
        static const char* names[10] = {"pre", "q", "kv_index", "attn", "out", "ffn_pre", "router", "experts", "tail", "lane_wait"};
        std::cerr << "DS41 decode_gpu_ms_per_pass";
        for (int q = 0; q < 10; ++q) std::cerr << " " << names[q] << "=" << d.dtime_[q] / d.dtokens_;
        std::cerr << " passes=" << d.dtokens_ << " host_ms_per_pass attention=" << (d.attn_host_s_ - attn_host0) * 1e3 / d.dtokens_
                  << " ffn=" << (d.ffn_host_s_ - ffn_host0) * 1e3 / d.dtokens_
                  << " route_sync=" << (d.route_wait_s_ - route_wait0) * 1e3 / d.dtokens_
                  << " layers=" << (d.loop_s_ - loop0) * 1e3 / d.dtokens_
                  << " (ffn: enqueue " << d.xt_[0] * 1e3 / d.dtokens_ << " lane " << d.xt_[1] * 1e3 / d.dtokens_ << " peer " << d.xt_[2] * 1e3 / d.dtokens_
                  << " peer_gpu " << (d.peer_gpu_ms() - peer_gpu0) / d.dtokens_ << " peer_late " << (d.peer_late_ms() - peer_late0) / d.dtokens_ << ")";
        const double* ps = d.peer_split_ms();   // (reset at prefill_done: decode's calls)
        std::cerr << " peer_split_ms_per_pass upload=" << ps[0] / d.dtokens_ << " gu=" << ps[1] / d.dtokens_ << " mid=" << ps[2] / d.dtokens_
                  << " down=" << ps[3] / d.dtokens_ << " tail=" << ps[4] / d.dtokens_ << "\n";
    }
    std::cerr << "DS41 profile_s step=" << d.prof_[0] << " gpu_to_router=" << d.prof_[1] << " cpu_experts=" << d.prof_[2]
              << " gpu_tier_hit=" << (d.tier_all_ > 0 ? d.tier_hits_ / d.tier_all_ : 0)
              << " canon_hit=" << (d.canon_cpu_routes_ ? double(d.canon_routes_) / double(d.canon_cpu_routes_) : 0.0)
              << " canon_streamed=" << (d.canon_cpu_routes_ ? double(d.canon_streamed_) / double(d.canon_cpu_routes_) : 0.0) << " promotions=" << d.promotions_ << " adapt=" << d.prof_[3] << " engram=" << d.prof_[4]
              << " index_host=" << d.prof_[5] << " head=" << d.prof_[6] << " stage_copy=" << d.prof_[7] << " stream_wait=" << d.stream_wait_s_ << "\n";
    d.miss_line(double(out.size()));
    std::cerr << d.promote_summary();
    if (d.own_ms_[3] > 0)
        std::cerr << "DS41 remote_own_decode layer_calls=" << d.own_ms_[3] << " total_s gpu_sync=" << d.own_ms_[0] / 1e3
                  << " worker_wait=" << d.own_ms_[1] / 1e3 << " merge=" << d.own_ms_[2] / 1e3 << " (per layer call ms "
                  << d.own_ms_[0] / d.own_ms_[3] << " / " << d.own_ms_[1] / d.own_ms_[3] << " / " << d.own_ms_[2] / d.own_ms_[3] << ") request_to_reply_ms=" << d.own_latency_ms_ / d.own_ms_[3] << "\n";
    std::cout << "tokens:";
    for (int t : out) std::cout << ' ' << t;
    std::cout << "\n";
    std::cerr << "DS41 prompt=" << prompt.size() << " prompt_tok_s=" << prompt.size() / std::max(prefill_s, 1e-9)
              << " generated=" << out.size() << " decode_tok_s="
              << (out.size() > 1 ? double(out.size() - 1) / std::max(decode_s, 1e-9) : 0.0) << "\n";
    return 0;
}

}  // namespace strata::ds41
