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
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
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
#include <unistd.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
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

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    explicit Dev(size_t n) : bytes(n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
    Dev& operator=(Dev&& o) noexcept { std::swap(p, o.p); std::swap(bytes, o.bytes); return *this; }
    ~Dev() { if (p) cudaFree(p); }
    float* f(size_t at = 0) const { return static_cast<float*>(p) + at; }
    int* i() const { return static_cast<int*>(p); }
    uint16_t* h() const { return static_cast<uint16_t*>(p); }
};

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

class Decoder {
public:
    Decoder(const std::string& path, int context, int threads, int chunk = 0)
        : art_(path), cfg_(Config::read(*art_.metadata(), art_.descriptor())), context_(context) {
        if (chunk > 0) chunk_ = chunk;
        const auto& c = cfg_;
        if (context_ < 1) throw std::invalid_argument("ds41: context must be positive");
        // STRATA_DS41_PREFILL_CHUNK: tokens per forward pass of the prompt (default 256)
        if (const char* v = std::getenv("STRATA_DS41_PREFILL_CHUNK"); v && chunk <= 0) chunk_ = std::max(1, std::atoi(v));
        ring_ = c.window + chunk_;
        stream_ = make_stream(0);   // the default (blocking) flags
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
                for (const char* n : {"engram_kv.weight", "engram_gate.weight"}) bytes += upload(p + n);
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
            lane.f.swiglu_limit = c.clamp[size_t(l)];
            lane.f.iq1s_rows = true;
            // a read-only shared mapping of the shard: first touches are about twice as fast to copy as the
            // loader's private mapping (11.8 vs 20.4 GB/s on TR16), for the streamed prompt experts and the lane
            static const bool lane_shared = !(std::getenv("STRATA_DS41_LANE_SHARED") && std::string(std::getenv("STRATA_DS41_LANE_SHARED")) == "0");
            lane.gate = lane_shared ? shared(g) : g.data();
            lane.up = lane_shared ? shared(u) : u.data();
            lane.down = lane_shared ? shared(d) : d.data();
            stream_src_.push_back({shared(g), shared(u), shared(d)});
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
                caches_[size_t(l)] = Dev(size_t(context_ / c.ratio[size_t(l)] + 1) * row_width() * 4);
                if (c.ratio[size_t(l)] > 1) { stash_v_[size_t(l)] = Dev(size_t(c.head_dim) * 4); stash_s_[size_t(l)] = Dev(size_t(c.head_dim) * 4); }
            }
        score_stride_ = context_ + 1;
        attn_scores_ = Dev(k::ds41_attention_scratch(c.heads));
        comp_ = Dev(size_t(c.head_dim) * 4);
        ik_ = Dev(size_t(c.index_dim) * 4);
        if (!c.engram_layers.empty()) {
            const auto& t = art_.at("per_layer_token_embd.weight");
            engram_table_ = &t;
            engram_fd_ = ::open(t.file->path().c_str(), O_RDONLY | O_DIRECT);
            if (engram_fd_ < 0) throw std::runtime_error("ds41: cannot open the Engram table with O_DIRECT: " + t.file->path());
            if (const char* v = std::getenv("STRATA_DS41_ENGRAM_THREADS")) engram_threads_ = std::max(1, std::atoi(v));
            // the loader's mapping may have pulled table pages in: drop them from the page cache
            posix_fadvise(engram_fd_, off_t(t.file->data_start() + t.tensor->offset), off_t(t.bytes), POSIX_FADV_DONTNEED);
        }
        (void)process_cpus();   // before the lane pins this thread
        lane_ = std::make_unique<ExpertLane>(c.hidden, c.ff, threads, true);
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
            stage_bytes_ = size_t(v ? std::atoi(v) : 256) << 20;
            if (const char* t = std::getenv("STRATA_DS41_STAGE_THREADS")) stage_threads_ = std::max(1, std::atoi(t));
            for (int i = 0; i < 2; ++i) {
                ck(cudaEventCreateWithFlags(&stage_uploaded_[i], cudaEventDisableTiming), "stage event");
                ck(cudaEventCreateWithFlags(&stage_copied_[i], cudaEventDisableTiming), "stage event");
                ck(cudaEventCreateWithFlags(&stage_free_[i], cudaEventDisableTiming), "stage event");
            }
            copy_s_ = make_stream(cudaStreamNonBlocking);
            use_mmq_ = strata::prefill::mmq::built() && !(std::getenv("STRATA_DS41_MMQ") && std::string(std::getenv("STRATA_DS41_MMQ")) == "0");
            if (use_mmq_) mmq_ = std::make_unique<strata::prefill::mmq::Context>();
            if (const char* r = std::getenv("STRATA_DS41_MMQ_ROWS")) rcap_min_ = std::max(256, std::atoi(r));
        }
        (void)H;
        score_.assign(size_t(c.layers) * c.experts, 0.0f);
        resize(chunk_);
        // a long first prompt borrows the tier's VRAM; prefill_done() shrinks the buffers and loads the tier
        tier_lent_ = chunk_ >= stream_min_;
        if (!tier_lent_) load_tier();
        else pin_experts();   // at load, not in the first prompt's time
    }

    /// After a long prompt: the chunk buffers shrink to STRATA_DS41_DECODE_CHUNK (64) and the GPU expert tier takes
    /// the VRAM, seeded from the profile file or else from the prompt's own routing.
    void prefill_done() {
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
        // the window ring: slot = position % ring; re-laid when its size changes
        const int ring = c.window + tokens;
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
        for (Dev* d : {&h_, &h2_, &x_, &xn_, &flat_, &mixes_, &coeff_attn_, &coeff_ffn_, &qa_, &qr_, &q_, &kv_, &kvn_, &att_, &oa_, &oa_part_, &y_,
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
        x_ = Dev(T * H * 4);
        xn_ = Dev(T * H * 4);
        flat_ = Dev(T * 4 * H * 4);
        mixes_ = Dev(T * 24 * 4);
        coeff_attn_ = Dev(T * 24 * 4);
        coeff_ffn_ = Dev(T * 24 * 4);
        qa_ = Dev(T * size_t(c.q_rank) * 4);
        qr_ = Dev(T * size_t(c.q_rank) * 4);
        q_ = Dev(T * size_t(c.heads) * c.head_dim * 4);
        kv_ = Dev(T * size_t(c.head_dim) * 4);
        kvn_ = Dev(T * size_t(c.head_dim) * 4);
        att_ = Dev(T * size_t(c.heads) * c.head_dim * 4);
        oa_ = Dev(T * size_t(c.out_groups) * c.out_rank * 4);
        oa_part_ = Dev(T * size_t(c.out_rank) * 4);
        y_ = Dev(T * H * 4);
        q8_ = Dev(k::native_q8_1_bytes(std::max(c.heads * c.head_dim, 5 * c.hidden), 1) * T);
        q8g_ = Dev(k::native_q8_1_bytes(c.heads / c.out_groups * c.head_dim, 1) * T);
        cs_ = Dev(T * kTables * 64 * 4);
        rope_host_.assign(T * kTables * 64, 0.0f);
        comp_v_ = Dev(T * size_t(c.head_dim) * 4);
        comp_s_ = Dev(T * size_t(c.head_dim) * 4);
        iq_ = Dev(T * size_t(c.index_heads) * c.index_dim * 4);
        iw_ = Dev(T * size_t(c.index_heads) * 4);
        logits_ = Dev(std::max(T * size_t(c.experts), size_t(c.vocab)) * 4);   // the router's, or one position's logits (more on request)
        sh_gate_ = Dev(T * size_t(c.ff) * 4);
        sh_up_ = Dev(T * size_t(c.ff) * 4);
        sh_h_ = Dev(T * size_t(c.ff) * 4);
        moe_ = Dev(T * H * 4);
        scores_ = Dev(T * size_t(score_stride_) * 4);
        sel_.resize(size_t(c.layers));
        sel_count_.assign(size_t(c.layers), std::vector<int>(T, 0));
        for (int l = 0; l < c.layers; ++l)
            if (c.topk_src[size_t(l)] == l) sel_[size_t(l)] = Dev(T * size_t(c.index_top_k) * 4);
        sel_cnt_.resize(size_t(c.layers));
        sel_on_gpu_.assign(size_t(c.layers), 0);
        for (int l = 0; l < c.layers; ++l)
            if (c.topk_src[size_t(l)] == l) sel_cnt_[size_t(l)] = Dev(T * 4);
        cand_stride_ = c.candidate_block > 0 ? (score_stride_ + c.candidate_block - 1) / c.candidate_block : 1;
        cand_dev_ = Dev(T * size_t(cand_stride_));
        candidates_.assign(T, {});
        if (!c.engram_layers.empty()) {
            engram_rows_ = Dev(c.engram_layers.size() * T * c.engram_cols() * c.engram_dim * 4);
            engram_host_.assign(c.engram_layers.size() * T * c.engram_cols() * c.engram_dim, 0.0f);
            engram_kv_ = Dev(T * 5 * H * 4);
        }
        if (tokens >= gemm_min_) xh_ = Dev(T * size_t(c.heads) * c.head_dim * 2 + T * size_t(c.heads / c.out_groups) * c.head_dim * 2);
        g_sum_ = Dev(T * H * 4);
        attn_info_dev_ = Dev(T * 16);
        attn_info_ = reinterpret_cast<int*>(pinned(T * 16));
        pinned_[0] = pinned(T * H * 4);
        pinned_[1] = pinned(T * H * 4);
        pinned_[2] = pinned(T * H * 4);
        dring_.clear();
        if (tokens >= stream_min_) {   // the streamed experts' rings: VRAM slots from what is left, pinned host slots
            slot_bytes_ = stage_bytes_;
            if (hring_.empty()) {
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
            s_x_ = Dev(size_t(8) * H * 4);
            s_xq_ = Dev(k::native_q8_1_bytes(c.hidden, 8));
            s_gate_ = Dev(size_t(8) * c.ff * 4);
            s_up_ = Dev(size_t(8) * c.ff * 4);
            s_h_ = Dev(size_t(8) * c.ff * 4);
            s_hq_ = Dev(k::native_q8_1_bytes(c.ff, 8));
            s_out_ = Dev(size_t(8) * H * 4);
            s_idx_ = Dev(T * size_t(c.top_k) * 4);
            s_w_ = Dev(T * size_t(c.top_k) * 4);
            s_bound_ = Dev((size_t(c.experts) * 2 + 8) * 4);
            s_hidx_ = reinterpret_cast<int*>(pinned(T * size_t(c.top_k) * 8 + (size_t(c.experts) * 2 + 8) * 4));
            s_hw_ = reinterpret_cast<float*>(s_hidx_ + T * size_t(c.top_k));
            s_hbound_ = reinterpret_cast<int32_t*>(s_hw_ + T * size_t(c.top_k));
            rcap_ = std::max(rcap_min_, int(T));   // one expert's rows always fit a batch
            if (use_mmq_) {
                namespace mq = strata::prefill::mmq;
                m_xq_ = Dev(mq::q8_bytes(rcap_, c.hidden));
                s_gu_ = Dev(size_t(rcap_) * 2 * c.ff * 4);
                s_hf_ = Dev(size_t(rcap_) * c.ff * 4);
                m_hq_ = Dev(mq::q8_bytes(rcap_, c.ff));
                s_dout_ = Dev(size_t(rcap_) * H * 4);
                s_iota_ = Dev((size_t(rcap_) + 128) * 4);
                mq::iota(s_iota_.i(), rcap_ + 128, stream_);
            }
            size_t free_b = 0, total_b = 0;
            ck(cudaMemGetInfo(&free_b, &total_b), "memory info");
            const char* dv = std::getenv("STRATA_DS41_STAGE_SLOTS");
            const char* dm = std::getenv("STRATA_DS41_DESKTOP_MIB");
            const size_t keep = size_t(dm ? std::max(0, std::atoi(dm)) : 3072) << 20;
            const size_t fit = free_b > keep ? (free_b - keep) / slot_bytes_ : 0;
            dslots_ = int(std::min<size_t>(dv ? size_t(std::max(2, std::atoi(dv))) : 8, fit));
            if (dslots_ < 2) throw std::runtime_error("ds41: no VRAM for the streamed experts' slots (shorter STRATA_DS41_PROMPT_BATCH)");
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

    // pinned host memory owned by the decoder (released by resize)
    float* pinned(size_t bytes) {
        void* h = nullptr;
        ck(cudaHostAlloc(&h, std::max<size_t>(bytes, 16), cudaHostAllocDefault), "pinned");
        pinned_bufs_.emplace_back(h, [](void* q) { cudaFreeHost(q); });
        return static_cast<float*>(h);
    }
    ~Decoder() {
        stream_end();
        for (void* r : registered_) cudaHostUnregister(r);
        for (const auto& [file, m] : shared_maps_) munmap(m.first, m.second);
        if (engram_fd_ >= 0) ::close(engram_fd_);
        if (uploader_.joinable()) {
            { std::lock_guard<std::mutex> lock(jobs_mutex_); stop_uploads_ = true; }
            jobs_cv_.notify_all();
            uploader_.join();
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    const Config& config() const { return cfg_; }
    int chunk() const { return chunk_; }

    /// Comparison hooks for --check (single-token passes only): (name, layer, device pointer, floats) / a host pointer.
    std::function<void(const std::string&, int, const float*, size_t)> observe, observe_host;
    /// Teacher forcing for --check-force: fills a host vector with the reference tensor (name, layer); false if none.
    std::function<bool(const std::string&, int, std::vector<float>&)> reference;

    /// One token at position pos (positions must arrive in order from 0). Returns the logits on the host.
    const std::vector<float>& step(int token, int pos) { return forward(&token, 1, pos, false); }

    /// `count` tokens (at most chunk()) at positions pos .. pos + count - 1. Returns the last token's logits, or with
    /// all_logits every token's ([count][vocab]).
    const std::vector<float>& forward(const int* tokens, int count, int pos, bool all_logits) {
        const auto& c = cfg_;
        const int T = count;
        const auto step0 = std::chrono::steady_clock::now();
        struct Done { double& t; std::chrono::steady_clock::time_point s; ~Done() { t += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count(); } } done{prof_[0], step0};
        if (T < 1 || T > chunk_) throw std::invalid_argument("ds41: chunk size");
        if (pos != int(history_.size())) throw std::logic_error("ds41: positions must be consecutive from 0");
        if (pos + T > context_) throw std::runtime_error("ds41: context exhausted");
        pos0_ = pos;
        tokens_ = T;
        const bool hook = T == 1;
        // embeddings: Q6_K rows dequantized on the host
        {
            const auto& t = art_.at("token_embd.weight");
            const size_t row = ggml_row_size(ggml_type(t.tensor->type), c.hidden);
            for (int i = 0; i < T; ++i) {
                const auto e = to_float(int(t.tensor->type), t.data() + size_t(tokens[i]) * row, c.hidden);
                std::copy(e.begin(), e.end(), pinned_[1] + size_t(i) * c.hidden);
            }
            ck(cudaMemcpyAsync(x_.p, pinned_[1], size_t(T) * c.hidden * 4, cudaMemcpyHostToDevice, stream_), "embed");
            k::ds41_broadcast_streams(x_.f(), h_.f(), c.hidden, stream_, T);
        }
        // the rotation tables of each position: [0] plain forward, [1] plain back, [2] compressed forward, [3] back,
        // [4] compressed forward at pos - 1 (a ratio-2 group's row stands for its first position)
        for (int i = 0; i < T; ++i) {
            const int p = pos + i;
            float* cs = rope_host_.data() + size_t(i) * kTables * 64;
            rope_table(c.rope_plain, c.rope_dim, p, 1.0f, cs);
            rope_table(c.rope_plain, c.rope_dim, p, -1.0f, cs + 64);
            rope_table(c.rope_compressed, c.rope_dim, p, 1.0f, cs + 128);
            rope_table(c.rope_compressed, c.rope_dim, p, -1.0f, cs + 192);
            rope_table(c.rope_compressed, c.rope_dim, std::max(p - 1, 0), 1.0f, cs + 256);
        }
        ck(cudaMemcpyAsync(cs_.p, rope_host_.data(), size_t(T) * kTables * 64 * 4, cudaMemcpyHostToDevice, stream_), "rope");
        history_.insert(history_.end(), tokens, tokens + T);
        if (T >= stream_min_) stream_begin();   // the producer starts on layer 0's experts while the trunk runs
        // every Engram layer's rows, [engram][token][cols x dim]
        {
            const auto g0 = std::chrono::steady_clock::now();
            for (size_t e = 0; e < c.engram_layers.size(); ++e)
                for (int i = 0; i < T; ++i) gather_engram(int(e), i, pos + i);
            read_engram_rows();
            prof_[4] += std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
            if (!c.engram_layers.empty())
                ck(cudaMemcpyAsync(engram_rows_.p, engram_host_.data(), engram_host_.size() * 4, cudaMemcpyHostToDevice, stream_),
                   "engram rows");
        }
        if (hook && observe) observe("hc_init", -1, h_.f(), size_t(4) * c.hidden);
        route_log_.assign(size_t(T) * c.layers * c.top_k, -1);
        bool have_pre = false;
        skip_ = 0;
        // STRATA_DS41_ENC_SKIP (1): above the last KV-source layer nothing reads a layer's rows beyond its window, so
        // layer l needs only the pass's last window + (layers - 1 - l) (window - 1) positions (the last token's logits and
        // the next pass's windows stay exact); the earlier tokens are dropped, in steps of 256 or more
        static const bool enc_skip = !(std::getenv("STRATA_DS41_ENC_SKIP") && std::string(std::getenv("STRATA_DS41_ENC_SKIP")) == "0");
        const int skip_from = [&] { int s = 0; for (int l = 0; l < c.layers; ++l) if (c.kv_src[size_t(l)] == l) s = l + 1; return s; }();
        const int end = pos + T;
        for (int l = 0; l < c.layers; ++l) {
            if (enc_skip && !all_logits && l >= skip_from) {
                const int need = c.window + (c.layers - 1 - l) * (c.window - 1);
                const int drop = (end - need) - pos0_;
                if (drop >= 256) narrow(drop, have_pre);
                skipped_ += skip_;
            }
            const std::string p = "blk." + std::to_string(l) + ".";
            if (gpu_timing_ && producer_.joinable()) mark(0);
            if (const int e = c.engram_index(l); e >= 0) {
                engram(l, e);
                if (hook && observe) observe("engram_out", l, h_.f(), size_t(4) * c.hidden);
            }
            // attention sublayer: its input collapses with the previous layer's FFN pre-mix (layer 0: stream 0)
            mixes(p + "hc_attn_", coeff_attn_);
            if (have_pre) k::ds41_hc_pre(h_.f(), coeff_ffn_.f(), x_.f(), c.hidden, stream_, T);
            else ck(cudaMemcpy2DAsync(x_.p, size_t(c.hidden) * 4, h_.p, size_t(4) * c.hidden * 4, size_t(c.hidden) * 4, size_t(T),
                                      cudaMemcpyDeviceToDevice, stream_), "stream 0");
            if (hook && observe) observe("hc_attn_pre", l, x_.f(), size_t(c.hidden));
            k::glm_rms_norm_rows(x_.f(), w(p + "attn_norm.weight"), xn_.f(), c.hidden, T, c.rms_eps, stream_);
            if (hook && observe) observe("attn_norm", l, xn_.f(), size_t(c.hidden));
            attention(l);
            k::ds41_hc_post(y_.f(), h_.f(), coeff_attn_.f(4), coeff_attn_.f(8), h2_.f(), c.hidden, stream_, T);
            std::swap(h_, h2_);
            if (hook && observe) observe("hc_attn_post", l, h_.f(), size_t(4) * c.hidden);
            // FFN sublayer: its input collapses with this layer's attention pre-mix; its own pre-mix is the next
            // layer's attention input
            mixes(p + "hc_ffn_", coeff_ffn_);
            have_pre = true;
            k::ds41_hc_pre(h_.f(), coeff_attn_.f(), x_.f(), c.hidden, stream_, T);
            if (hook && observe) observe("hc_ffn_pre", l, x_.f(), size_t(c.hidden));
            k::glm_rms_norm_rows(x_.f(), w(p + "ffn_norm.weight"), xn_.f(), c.hidden, T, c.rms_eps, stream_);
            if (hook && observe) observe("ffn_norm", l, xn_.f(), size_t(c.hidden));
            ffn(l);
            k::ds41_hc_post(y_.f(), h_.f(), coeff_ffn_.f(4), coeff_ffn_.f(8), h2_.f(), c.hidden, stream_, T);
            std::swap(h_, h2_);
            if (hook && observe) observe("l_out", l, h_.f(), size_t(4) * c.hidden);
            if (std::vector<float> r; hook && reference && reference("l_out", l, r) && r.size() == size_t(4) * c.hidden)
                ck(cudaMemcpy(h_.p, r.data(), r.size() * 4, cudaMemcpyHostToDevice), "force l_out");
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
        k::ds41_hc_pre(h_.f(size_t(first) * 4 * c.hidden), coeff_ffn_.f(size_t(first) * 24), x_.f(), c.hidden, stream_, n);
        k::glm_rms_norm_rows(x_.f(), w("output_norm.weight"), xn_.f(), c.hidden, n, c.rms_eps, stream_);
        mat("output.weight", xn_.f(), logits_.f(), n);
        logits_host_.resize(size_t(n) * c.vocab);
        ck(cudaMemcpyAsync(logits_host_.data(), logits_.p, logits_host_.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
        ck(cudaStreamSynchronize(stream_), "sync");
        prof_[6] += std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
        if (hook && observe) observe("result_output", -1, logits_.f(), size_t(c.vocab));
        if (routes_)
            routes_.write(reinterpret_cast<const char*>(route_log_.data()), std::streamsize(route_log_.size() * 4));
        { const auto a0 = std::chrono::steady_clock::now(); adapt(T); prof_[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count(); }
        return logits_host_;
    }

private:
    static constexpr int kTables = 5;   // rotation tables per position
    int row_width() const { return cfg_.head_dim + cfg_.index_dim; }

    // a tensor's bytes in a read-only MAP_SHARED mapping of its shard (one per shard)
    const uint8_t* shared(const core::ArtifactTensor& t) {
        auto it = shared_maps_.find(t.file);
        if (it == shared_maps_.end()) {
            const int fd = ::open(t.file->path().c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ds41: cannot open " + t.file->path());
            void* p = mmap(nullptr, t.file->file_size(), PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (p == MAP_FAILED) throw std::runtime_error("ds41: cannot map " + t.file->path());
            it = shared_maps_.emplace(t.file, std::make_pair(p, size_t(t.file->file_size()))).first;
        }
        // experts should stay page-cached: ask for the uncached ones now (readahead in the background)
        const uintptr_t a = reinterpret_cast<uintptr_t>(static_cast<const uint8_t*>(it->second.first) + t.file->data_start() + t.tensor->offset);
        const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE)), a0 = a / pg * pg;
        madvise(reinterpret_cast<void*>(a0), size_t(a + t.bytes - a0), MADV_WILLNEED);
        populate_.push_back({reinterpret_cast<void*>(a0), size_t(a + t.bytes - a0)});
        return static_cast<const uint8_t*>(it->second.first) + t.file->data_start() + t.tensor->offset;
    }

    size_t upload(const std::string& name) {
        const auto& t = art_.at(name);
        Weight W;
        W.type = int(t.tensor->type);
        W.in = int(t.tensor->shape[0]);
        W.out = 1;
        for (size_t i = 1; i < t.tensor->shape.size(); ++i) W.out *= int(t.tensor->shape[i]);
        W.row_bytes = ggml_row_size(ggml_type(W.type), W.in);
        W.data = Dev(t.bytes);
        ck(cudaMemcpy(W.data.p, t.data(), t.bytes, cudaMemcpyHostToDevice), "upload");
        // the GPU copy is the only one used: its host pages leave the page cache to the experts
        t.file->discard_tensor_pages(*t.tensor, t.bytes);
        weights_.emplace(name, std::move(W));
        return t.bytes;
    }
    const Weight& weight(const std::string& name) const {
        const auto it = weights_.find(name);
        if (it == weights_.end()) throw std::runtime_error("ds41: weight not on the GPU: " + name);
        return it->second;
    }
    const float* w(const std::string& name) const { return weight(name).data.f(); }

    // y[t] = W x[t] for nt tokens (x[t] at t W.in, y[t] at t rows) over rows [row0, row0 + rows) of W (rows 0: all)
    void mat(const std::string& name, const float* x, float* y, int nt = 1, int row0 = 0, int rows = 0) {
        const Weight& W = weight(name);
        if (!rows) rows = W.out;
        const uint8_t* base = static_cast<const uint8_t*>(W.data.p) + size_t(row0) * W.row_bytes;
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
            if (nt == 1) k::exl3_bf16_gemv(reinterpret_cast<const uint16_t*>(base), x, y, W.in, rows, stream_);
            else k::ds41_bf16_gemv(reinterpret_cast<const uint16_t*>(base), x, y, rows, W.in, stream_, nt);
        } else {
            if (!k::native_mmvq_supported(W.type)) throw std::runtime_error("ds41: unsupported trunk type in " + name);
            quantize(x, q8_.p, W.in, nt);
            mmvq(W.type, base, q8_.p, y, W.in, rows, nt);
        }
    }
    // native_quantize_q8_1 over nt columns, at most 8 per launch
    void quantize(const float* x, void* q8, int in, int nt) {
        const size_t col = k::native_q8_1_bytes(in, 1);
        for (int j = 0; j < nt; j += 8)
            k::native_quantize_q8_1(x + size_t(j) * in, static_cast<uint8_t*>(q8) + size_t(j) * col, in, std::min(8, nt - j), stream_);
    }
    // native_mmvq over nt columns, at most 8 per launch
    void mmvq(int type, const uint8_t* w, const void* q8, float* y, int in, int out, int nt) {
        const size_t col = k::native_q8_1_bytes(in, 1);
        for (int j = 0; j < nt; j += 8)
            k::native_mmvq(type, w, static_cast<const uint8_t*>(q8) + size_t(j) * col, y + size_t(j) * out, in, out,
                           std::min(8, nt - j), stream_);
    }

    // the hyper-connection mixes of every token's streams into coeff (pre, post, comb per token)
    void mixes(const std::string& p, Dev& coeff) {
        const auto& c = cfg_;
        k::glm_rms_norm_rows(h_.f(), nullptr, flat_.f(), 4 * c.hidden, tokens_, c.rms_eps, stream_);
        mat(p + "fn.weight", flat_.f(), mixes_.f(), tokens_);
        k::ds41_hc_coeff(mixes_.f(), w(p + "scale.weight"), w(p + "base.weight"), coeff.f(), c.sinkhorn, c.hc_eps, stream_, tokens_);
    }

    // the Engram rows of token i (position pos) for the e-th Engram layer, into engram_host_
    void gather_engram(int e, int i, int pos) {
        const auto& c = cfg_;
        const int cols = c.engram_cols(), n = c.ngram;
        // the n-gram hash of PR #30817's llama_model_deepseek41::engram_hash
        const uint64_t* mult = c.multipliers.data() + size_t(e) * n;
        const uint64_t* primes = c.primes.data() + size_t(e) * cols;
        const uint64_t* offs = c.offsets.data() + size_t(e) * cols;
        const int64_t pad = c.token_map[size_t(c.engram_pad)];
        auto cid = [&](int64_t t) -> int64_t {
            return t < 0 || t >= int64_t(c.token_map.size()) ? -1 : c.token_map[size_t(t)];
        };
        std::vector<uint64_t> toks(static_cast<size_t>(n));
        int64_t id = cid(history_[size_t(pos)]);
        bool blocked = id == -1;
        toks[0] = uint64_t(blocked ? pad : id);
        for (int s = 1; s < n; ++s) {
            id = pos - s < 0 ? -1 : cid(history_[size_t(pos - s)]);
            blocked = blocked || id == -1;
            toks[size_t(s)] = uint64_t(blocked ? pad : id);
        }
        std::vector<int64_t> rows(static_cast<size_t>(cols));
        uint64_t rolling = toks[0] * mult[0];
        for (int o = 1; o < n; ++o) {
            rolling ^= toks[size_t(o)] * mult[o];
            for (int h = 0; h < c.engram_heads; ++h) {
                const int col = (o - 1) * c.engram_heads + h;
                rows[size_t(col)] = int64_t(rolling % primes[col] + offs[col]);
            }
        }
        const size_t per_token = size_t(cols) * c.engram_dim;
        float* v = engram_host_.data() + (size_t(e) * chunk_ + size_t(i)) * per_token;
        for (int j = 0; j < cols; ++j) engram_reqs_.push_back({rows[size_t(j)], v + size_t(j) * c.engram_dim});
    }

    // The queued Engram rows, read with O_DIRECT preads (the 110 GB table must not fill the page cache the experts
    // live in) by STRATA_DS41_ENGRAM_THREADS (16) threads, each row's 4 KiB page(s), then dequantized.
    void read_engram_rows() {
        if (engram_reqs_.empty()) return;
        const auto& t = *engram_table_;
        const int type = int(t.tensor->type), dim = cfg_.engram_dim;
        const size_t row_bytes = ggml_row_size(ggml_type(type), dim);
        const uint64_t base = t.file->data_start() + t.tensor->offset;
        const int nt = std::min<int>(engram_threads_, int(engram_reqs_.size()));
        std::vector<std::thread> workers;
        std::atomic<bool> failed{false};
        for (int w = 0; w < nt; ++w)
            workers.emplace_back([&, w] {
                unpin_helper();
                alignas(4096) static thread_local uint8_t buf[3 * 4096];
                for (size_t q = size_t(w); q < engram_reqs_.size(); q += size_t(nt)) {
                    const uint64_t off = base + uint64_t(engram_reqs_[q].row) * row_bytes;
                    const uint64_t page = off & ~uint64_t(4095);
                    const size_t len = (off + row_bytes - page + 4095) & ~size_t(4095);
                    if (pread(engram_fd_, buf, len, off_t(page)) != ssize_t(len)) { failed = true; continue; }
                    ggml_get_type_traits(ggml_type(type))->to_float(buf + (off - page), engram_reqs_[q].dst, dim);
                }
            });
        for (auto& w : workers) w.join();
        engram_reqs_.clear();
        if (failed) throw std::runtime_error("ds41: Engram row read failed");
    }

    // layer l's Engram (the e-th): rows already on the GPU
    void engram(int l, int e) {
        const auto& c = cfg_;
        const std::string p = "blk." + std::to_string(l) + ".";
        mat(p + "engram_kv.weight", engram_rows_.f(size_t(e) * chunk_ * c.engram_cols() * c.engram_dim), engram_kv_.f(), tokens_);
        if (tokens_ == 1 && observe) observe("engram_kv", l, engram_kv_.f(), size_t(5) * c.hidden);
        k::ds41_engram(h_.f(), engram_kv_.f(), w(p + "engram_gate.weight"), c.hidden, c.rms_eps, stream_, tokens_);
    }

    const float* table(int i, int which) const { return cs_.f((size_t(i) * kTables + size_t(which)) * 64); }

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
        const bool hook = T == 1 && observe;
        mat(p + "attn_q_a.weight", xn_.f(), qa_.f(), T);
        k::glm_rms_norm_rows(qa_.f(), w(p + "attn_q_a_norm.weight"), qr_.f(), c.q_rank, T, c.rms_eps, stream_);
        if (hook) observe("qr_norm", l, qr_.f(), size_t(c.q_rank));
        mat(p + "attn_q_b.weight", qr_.f(), q_.f(), T);
        k::ds41_rope(q_.f(), T * c.heads, c.head_dim, nope, pairs, table(0, fwd), stream_, c.heads, step);
        if (hook) observe("q", l, q_.f(), size_t(c.heads) * c.head_dim);
        // the window keys: normalized, rotated, into the ring (slot = position % ring)
        mat(p + "attn_kv.weight", xn_.f(), kv_.f(), T);
        k::glm_rms_norm_rows(kv_.f(), w(p + "attn_kv_a_norm.weight"), kvn_.f(), c.head_dim, T, c.rms_eps, stream_);
        k::ds41_rope(kvn_.f(), T, c.head_dim, nope, pairs, table(0, fwd), stream_, 1, step);
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
            if (src == l) compress(l);
            comp_rows = caches_[size_t(src)].f();
            if (ts == l) { index(l); sel_on_gpu_[size_t(l)] = gpu_sel_; }
        }
        const float* sinks = w(p + "attn_sinks.weight");
        const float scale = 1.0f / std::sqrt(float(c.head_dim));
        if (T >= 16) {   // one launch for the pass: per-token {first slot, window rows, selected rows}
            int* info = attn_info_;
            for (int i = 0; i < T; ++i) {
                const int pos = pos0_ + i, rows = std::min(pos + 1, c.window);
                info[4 * i] = (pos - rows + 1) % ring_;
                info[4 * i + 1] = rows;
                info[4 * i + 2] = ratio && !sel_on_gpu_[size_t(ts)] && (pos + 1) / ratio > 0 ? sel_count_[size_t(ts)][size_t(i)] : 0;
                info[4 * i + 3] = 0;
            }
            ck(cudaMemcpyAsync(attn_info_dev_.p, info, size_t(T) * 16, cudaMemcpyHostToDevice, stream_), "attention info");
            if (ratio && sel_on_gpu_[size_t(ts)]) k::ds41_patch_info(attn_info_dev_.i(), sel_cnt_[size_t(ts)].i(), T, pos0_, ratio, stream_);
            // STRATA_DS41_ATTN_SUB (256): tokens per launch, so no single kernel holds the GPU for long
            static const int sub = [] { const char* v = std::getenv("STRATA_DS41_ATTN_SUB"); return v ? std::max(16, std::atoi(v)) : 256; }();
            const size_t qw = size_t(c.heads) * c.head_dim;
            for (int t0 = 0; t0 < T; t0 += sub)
                k::ds41_attention_batched(q_.f(size_t(t0) * qw), std::min(sub, T - t0), c.heads, c.head_dim, windows_[size_t(l)].f(), ring_,
                                          attn_info_dev_.i() + 4 * t0, comp_rows, row_width(),
                                          (ratio ? sel_[size_t(ts)].i() : attn_info_dev_.i()) + (ratio ? size_t(t0) * c.index_top_k : 0),
                                          c.index_top_k, sinks, scale, att_.f(size_t(t0) * qw), stream_);
        } else
        for (int i = 0; i < T; ++i) {
            const int pos = pos0_ + i, rows = std::min(pos + 1, c.window), first = (pos - rows + 1) % ring_;
            int nsel = 0;
            const int* sel = nullptr;
            if (ratio && (pos + 1) / ratio > 0) {
                nsel = sel_count_[size_t(ts)][size_t(i)];
                sel = sel_[size_t(ts)].i() + size_t(i) * c.index_top_k;
            }
            k::ds41_attention(q_.f(size_t(i) * c.heads * c.head_dim), c.heads, c.head_dim, windows_[size_t(l)].f(), first, ring_,
                              rows, comp_rows, row_width(), sel, nsel, sinks, scale, attn_scores_.f(),
                              att_.f(size_t(i) * c.heads * c.head_dim), stream_);
        }
        k::ds41_rope(att_.f(), T * c.heads, c.head_dim, nope, pairs, table(0, back), stream_, c.heads, step);
        if (hook) observe("attn_derope", l, att_.f(), size_t(c.heads) * c.head_dim);
        // grouped output projection: group g's out_rank rows read heads [g heads/groups, (g+1) heads/groups)
        const int group_in = c.heads / c.out_groups * c.head_dim, width = c.heads * c.head_dim;
        const Weight& A = weight(p + "attn_output_a.weight");
        if (A.in != group_in) throw std::runtime_error("ds41: attn_output_a width");
        if (A.type == GGML_TYPE_F32 || A.type == GGML_TYPE_BF16) throw std::runtime_error("ds41: float attn_output_a");
        if (T >= gemm_min_) {
            k::glm_f16(att_.f(), xh_.h(), int64_t(T) * width, stream_);
            uint16_t* xg = xh_.h() + size_t(T) * width;   // one group's columns, packed
            for (int g = 0; g < c.out_groups; ++g) {
                ck(cudaMemcpy2DAsync(xg, size_t(group_in) * 2, xh_.h() + size_t(g) * group_in, size_t(width) * 2, size_t(group_in) * 2,
                                     size_t(T), cudaMemcpyDeviceToDevice, stream_), "group pack");
                gemm_->native(xg, A.type, static_cast<const uint8_t*>(A.data.p) + size_t(g) * c.out_rank * A.row_bytes,
                              oa_.f(size_t(g) * c.out_rank), T, c.out_rank, group_in, int64_t(c.out_groups) * c.out_rank);
            }
            mat(p + "attn_output_b.weight", oa_.f(), y_.f(), T);
            if (hook) observe("attn_out", l, y_.f(), size_t(c.hidden));
            return;
        }
        quantize(att_.f(), q8_.p, width, T);
        const size_t col_all = k::native_q8_1_bytes(width, 1), col_group = k::native_q8_1_bytes(group_in, 1);
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

    // the KV-source layer's compressor: one compressed row per `ratio` positions, in position order
    void compress(int l) {
        const auto& c = cfg_;
        const int T = tokens_;
        const std::string p = "blk." + std::to_string(l) + ".";
        const int ratio = c.ratio[size_t(l)];
        if (ratio > 2) throw std::runtime_error("ds41: only compress ratios 1 and 2");
        mat(p + "attn_compressor_kv.weight", xn_.f(), comp_v_.f(), T);
        if (ratio > 1) mat(p + "attn_compressor_gate.weight", xn_.f(), comp_s_.f(), T);
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
            float* dst = caches_[size_t(l)].f(size_t(g0) * rw);
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
            return;
        }
        for (int i = 0; i < T; ++i) {
            const int pos = pos0_ + i;
            const float* v = comp_v_.f(size_t(i) * hd);
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
            float* row = caches_[size_t(l)].f(size_t(g) * row_width());
            k::glm_rms_norm_rows(v, w(p + "attn_compressor_norm.weight"), row, c.head_dim, 1, c.rms_eps, stream_);
            if (T == 1 && observe) observe("comp_latent", l, row, hd);   // as dumped: normalized, before RoPE
            mat(p + "indexer.attn_k.weight", row, ik_.f());   // from the RoPE-free latent
            k::glm_rms_norm_rows(ik_.f(), w(p + "indexer.k_norm.weight"), row + c.head_dim, c.index_dim, 1, c.rms_eps, stream_);
            const float* cs = table(i, first == pos ? 2 : 4);
            k::ds41_rope(row + c.head_dim, 1, c.index_dim, c.index_dim - c.rope_dim, c.rope_dim / 2, cs, stream_);
            k::ds41_rope(row, 1, c.head_dim, c.head_dim - c.rope_dim, c.rope_dim / 2, cs, stream_);
        }
    }

    // the lightning indexer of an index-source layer: each token's top-k over its visible compressed rows
    void index(int l) {
        const auto& c = cfg_;
        const int T = tokens_, ratio = c.ratio[size_t(l)];
        const std::string p = "blk." + std::to_string(l) + ".";
        mat(p + "indexer.attn_q_b.weight", qr_.f(), iq_.f(), T);
        k::ds41_rope(iq_.f(), T * c.index_heads, c.index_dim, c.index_dim - c.rope_dim, c.rope_dim / 2, table(0, 2), stream_,
                     c.index_heads, kTables * 64);
        mat(p + "indexer.proj.weight", xn_.f(), iw_.f(), T);
        const int src = c.kv_src[size_t(l)];
        const float w_scale = 1.0f / std::sqrt(float(c.index_dim * c.index_heads));
        // STRATA_DS41_GPU_TOPK (1): passes of 16+ tokens score and select on the GPU (no host round trip);
        // STRATA_DS41_TOPK_CHECK=1 also runs the host selection and counts differences
        static const bool gpu_topk = !(std::getenv("STRATA_DS41_GPU_TOPK") && std::string(std::getenv("STRATA_DS41_GPU_TOPK")) == "0");
        static const bool topk_check = std::getenv("STRATA_DS41_TOPK_CHECK") != nullptr;
        gpu_sel_ = false;
        std::vector<int> gpu_picks, gpu_counts;
        if (gpu_topk && T >= 16) {
            const auto i0 = std::chrono::steady_clock::now();
            k::ds41_index_score_batched(iq_.f(), c.index_heads, c.index_dim, caches_[size_t(src)].f(c.head_dim), row_width(), pos0_, ratio,
                                        T, iw_.f(), w_scale, scores_.f(), score_stride_, stream_);
            const int mode = c.candidate_source == l ? 1 : c.candidate_source >= 0 && c.candidate_source < l ? 2 : 0;
            k::ds41_index_topk(scores_.f(), score_stride_, T, pos0_, ratio, c.index_top_k, sel_[size_t(l)].i(), sel_cnt_[size_t(l)].i(),
                               mode, static_cast<uint8_t*>(cand_dev_.p), cand_stride_, c.candidate_block, c.candidate_blocks, stream_);
            prof_[5] += std::chrono::duration<double>(std::chrono::steady_clock::now() - i0).count();
            if (!topk_check) { gpu_sel_ = true; return; }
            gpu_picks.resize(size_t(T) * c.index_top_k);
            gpu_counts.resize(size_t(T));
            ck(cudaMemcpyAsync(gpu_picks.data(), sel_[size_t(l)].p, gpu_picks.size() * 4, cudaMemcpyDeviceToHost, stream_), "check sel");
            ck(cudaMemcpyAsync(gpu_counts.data(), sel_cnt_[size_t(l)].p, gpu_counts.size() * 4, cudaMemcpyDeviceToHost, stream_), "check count");
            ck(cudaStreamSynchronize(stream_), "check sync");
        }
        int widest = 0;
        for (int i = 0; i < T; ++i) {
            const int visible = (pos0_ + i + 1) / ratio;
            widest = std::max(widest, visible);
            k::ds41_index_score(iq_.f(size_t(i) * c.index_heads * c.index_dim), c.index_heads, c.index_dim,
                                caches_[size_t(src)].f(c.head_dim), row_width(), visible, iw_.f(size_t(i) * c.index_heads),
                                w_scale, scores_.f(size_t(i) * score_stride_), stream_);
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
        for (int i = 0; i < T; ++i) {
            const int visible = (pos0_ + i + 1) / ratio;
            float* s = all.data() + size_t(i) * widest;
            if (visible == 0) { sel_count_[size_t(l)][size_t(i)] = 0; continue; }
            // candidate blocks: the source layer keeps its top blocks (pinning the block of the newest row); later
            // index sources score only inside them
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
            picked.erase(std::remove_if(picked.begin(), picked.end(), [&](int t) { return std::isinf(s[t]) && s[t] < 0; }),
                         picked.end());
            std::sort(picked.begin(), picked.end());
            sel_count_[size_t(l)][size_t(i)] = int(picked.size());
            std::copy(picked.begin(), picked.end(), picks.begin() + size_t(i) * c.index_top_k);
        }
        if (!gpu_counts.empty()) {
            int bad = 0;
            for (int i = 0; i < T; ++i) {
                const int n = sel_count_[size_t(l)][size_t(i)];
                bool same = gpu_counts[size_t(i)] == n;
                for (int j = 0; same && j < n; ++j) same = gpu_picks[size_t(i) * c.index_top_k + j] == picks[size_t(i) * c.index_top_k + j];
                bad += !same;
            }
            std::cerr << "DS41 topk_check layer=" << l << " tokens=" << T << " differ=" << bad << "\n";
        }
        ck(cudaMemcpyAsync(sel_[size_t(l)].p, picks.data(), picks.size() * 4, cudaMemcpyHostToDevice, stream_), "sel");
        ck(cudaStreamSynchronize(stream_), "sel sync");   // picks leaves scope
    }

    // The GPU expert tier: slots in one VRAM arena. STRATA_DS41_EXPERT_PROFILE=routes.i32 (STRATA_DS41_ROUTES of an
    // earlier run) fills it with the most-used experts at load; without one it starts empty. Then it adapts: every
    // expert's score decays by STRATA_DS41_TIER_DECAY (0.995) per token and gains 1 per use, and after each pass up to
    // STRATA_DS41_TIER_PROMOTE (2) missed experts replace lower-scored residents of the same size (0: static tier).
    // Size: STRATA_DS41_GPU_EXPERT_MIB, default free VRAM less STRATA_DS41_GPU_RESERVE_MIB (1536).
    struct Slot { uint8_t* p; size_t bytes; int owner; bool pending; cudaEvent_t done; };
    size_t expert_bytes(int l) const {
        const auto& L = experts_[size_t(l)];
        return 2 * L.f.gu_row * size_t(cfg_.ff) + L.f.d_row * size_t(cfg_.hidden);
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
        }
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "memory info");
        // STRATA_DS41_DESKTOP_MIB (3072): VRAM left to the desktop (compositor, browser) and the decoder's scratch
        const size_t reserve = size_t(env("STRATA_DS41_GPU_RESERVE_MIB", env("STRATA_DS41_DESKTOP_MIB", 3072))) << 20;
        size_t budget = free_b > reserve ? free_b - reserve : 0;
        if (std::getenv("STRATA_DS41_GPU_EXPERT_MIB")) budget = std::min(budget, size_t(env("STRATA_DS41_GPU_EXPERT_MIB", 0)) << 20);
        // one arena: many small allocations fragment and overshoot the reported free memory
        while (budget > (size_t(256) << 20)) {
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
            const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
            ck(cudaMemcpy(r, L.gate + size_t(e) * g, g, cudaMemcpyHostToDevice), "tier gate");
            ck(cudaMemcpy(r + g, L.up + size_t(e) * g, g, cudaMemcpyHostToDevice), "tier up");
            ck(cudaMemcpy(r + 2 * g, L.down + size_t(e) * d, d, cudaMemcpyHostToDevice), "tier down");
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
            const auto& L = experts_[size_t(j.layer)];
            const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
            uint8_t* st = staging_[k];
            std::memcpy(st, L.gate + size_t(j.expert) * g, g);
            std::memcpy(st + g, L.up + size_t(j.expert) * g, g);
            std::memcpy(st + 2 * g, L.down + size_t(j.expert) * d, d);
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
    void gpu_experts(int l, int i, const int* ids, const float* wts, int n) {
        const auto& c = cfg_;
        const auto& L = experts_[size_t(l)];
        const size_t g = L.f.gu_row * size_t(c.ff);
        float* sum = g_sum_.f(size_t(i) * c.hidden);
        k::native_quantize_q8_1(xn_.f(size_t(i) * c.hidden), gx_q8_.p, c.hidden, 1, stream_);
        for (int j = 0; j < n; ++j) {
            const uint8_t* base = slots_[size_t(slot_[size_t(l)][size_t(ids[j])])].p;
            k::native_mmvq(L.f.gu_type, base, gx_q8_.p, g_gate_.f(), c.hidden, c.ff, 1, stream_);
            k::native_mmvq(L.f.gu_type, base + g, gx_q8_.p, g_up_.f(), c.hidden, c.ff, 1, stream_);
            k::glm_swiglu(g_gate_.f(), g_up_.f(), g_h_.f(), c.ff, c.clamp[size_t(l)], stream_);
            k::native_quantize_q8_1(g_h_.f(), g_h_q8_.p, c.ff, 1, stream_);
            k::native_mmvq(L.f.d_type, base + 2 * g, g_h_q8_.p, g_out_.f(), c.ff, c.hidden, 1, stream_);
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
        const size_t g = L.f.gu_row * size_t(c.ff);
        for (int j = 0; j < n; j += 8) {
            const int m = std::min(8, n - j);
            k::ds41_gather(xn_.f(), idx + j, m, c.hidden, s_x_.f(), stream_);
            k::native_quantize_q8_1(s_x_.f(), s_xq_.p, c.hidden, m, stream_);
            k::native_mmvq(L.f.gu_type, base, s_xq_.p, s_gate_.f(), c.hidden, c.ff, m, stream_);
            k::native_mmvq(L.f.gu_type, base + g, s_xq_.p, s_up_.f(), c.hidden, c.ff, m, stream_);
            k::glm_swiglu(s_gate_.f(), s_up_.f(), s_h_.f(), m * c.ff, c.clamp[size_t(l)], stream_);
            k::native_quantize_q8_1(s_h_.f(), s_hq_.p, c.ff, m, stream_);
            k::native_mmvq(L.f.d_type, base + 2 * g, s_hq_.p, s_out_.f(), c.ff, c.hidden, m, stream_);
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

    void stream_begin() {
        const auto& c = cfg_;
        stream_end();
        items_.clear();
        item0_.assign(size_t(c.layers) + 1, 0);
        for (int l = 0; l < c.layers; ++l) {
            item0_[size_t(l)] = int(items_.size());
            const size_t b = stream_stride(l), per = std::max<size_t>(1, slot_bytes_ / b);
            for (int e = 0; e < c.experts; e += int(per)) items_.push_back({l, e, int(std::min<size_t>(per, size_t(c.experts - e))), b});
        }
        item0_[size_t(c.layers)] = int(items_.size());
        produced_ = 0;
        consumed_ = 0;
        staged_ = 0;
        dma_issued_ = 0;
        stop_stream_ = false;
        pin_experts();
        unpinned_.clear();
        for (int idx = 0; idx < int(items_.size()); ++idx)
            if (!pinned_layer_[size_t(items_[size_t(idx)].layer)]) unpinned_.push_back(idx);
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
        producer_ = std::thread([this] { unpin_helper(); produce(); });
        stager_ = std::thread([this] { unpin_helper(); stage(); });
    }
    void stream_end() {
        if (!producer_.joinable()) return;
        { std::lock_guard<std::mutex> lock(sm_); stop_stream_ = true; }
        scv_.notify_all();
        producer_.join();
        if (stager_.joinable()) stager_.join();
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
        std::cerr << "DS41 experts pinned GiB=" << bytes / double(1ull << 30) << " layers="
                  << std::count(pinned_layer_.begin(), pinned_layer_.end(), 1) << "/" << c.layers << " s="
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
                    for (int j = t; j < it.n; j += nt) {
                        const int e = it.first + j;
                        copy_nt(host + size_t(j) * it.stride, src.gate + size_t(e) * g, g);
                        copy_nt(host + size_t(j) * it.stride + g, src.up + size_t(e) * g, g);
                        copy_nt(host + size_t(j) * it.stride + 2 * g, src.down + size_t(e) * d, d);
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
                    return stop_stream_ || ((pinned || staged_ >= u + 1) && (idx < dslots_ || consumed_ >= idx - dslots_ + 1));
                });
                if (stop_stream_) return;
            }
            if (idx >= dslots_) cudaStreamWaitEvent(copy_s_, ev_free_[size_t(ds)], 0);
            uint8_t* dst = static_cast<uint8_t*>(dring_[size_t(ds)].p);
            if (pinned) {   // zero copy: three strided DMAs from the pinned mapping into the slot
                const auto& L = experts_[size_t(it.layer)];
                const Src& src = stream_src_[size_t(it.layer)];
                const size_t g = L.f.gu_row * size_t(c.ff), d = L.f.d_row * size_t(c.hidden);
                cudaMemcpy2DAsync(dst, it.stride, src.gate + size_t(it.first) * g, g, g, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                cudaMemcpy2DAsync(dst + g, it.stride, src.up + size_t(it.first) * g, g, g, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
                cudaMemcpy2DAsync(dst + 2 * g, it.stride, src.down + size_t(it.first) * d, d, d, size_t(it.n), cudaMemcpyHostToDevice, copy_s_);
            } else {
                const int hs = u % hslots_;
                cudaMemcpyAsync(dst, hring_[size_t(hs)], size_t(it.n) * it.stride, cudaMemcpyHostToDevice, copy_s_);
                cudaEventRecord(ev_dma_[size_t(hs)], copy_s_);
                ++u;
            }
            cudaEventRecord(ev_ready_[size_t(ds)], copy_s_);
            {
                std::lock_guard<std::mutex> lock(sm_);
                produced_ = idx + 1;
                dma_issued_ = u;
            }
            scv_.notify_all();
        }
    }

    void stream_experts(int l, const std::vector<int>& ids, const std::vector<float>& wts) {
        const auto& c = cfg_;
        namespace mq = strata::prefill::mmq;
        const int T = tokens_, K = c.top_k;
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
                tok[size_t(e)].push_back(i);
                wt[size_t(e)].push_back(wts[size_t(i) * K + k]);
                used_.push_back(l * c.experts + e);
            }
        std::vector<int> row0(size_t(c.experts) + 1, 0);
        for (int e = 0; e < c.experts; ++e) {
            row0[size_t(e) + 1] = row0[size_t(e)] + int(tok[size_t(e)].size());
            std::copy(tok[size_t(e)].begin(), tok[size_t(e)].end(), s_hidx_ + row0[size_t(e)]);
            std::copy(wt[size_t(e)].begin(), wt[size_t(e)].end(), s_hw_ + row0[size_t(e)]);
            tier_all_ += double(tok[size_t(e)].size());
        }
        // bounds of every product: per item, sub-products within the row cap; relative to the product's first row
        struct Product { int item, j0, j1, row_first, rows, bound0, max_rows; };
        std::vector<Product> prods;
        int nb = 0;
        for (int idx = item0_[size_t(l)]; idx < item0_[size_t(l) + 1]; ++idx) {
            const StreamItem& it = items_[size_t(idx)];
            for (int j0 = 0; j0 < it.n;) {
                int j1 = j0, rows = 0, mx = 0;
                while (j1 < it.n && (j1 == j0 || rows + int(tok[size_t(it.first + j1)].size()) <= rcap_)) {
                    const int n = int(tok[size_t(it.first + j1)].size());
                    rows += n;
                    mx = std::max(mx, n);
                    ++j1;
                }
                Product pr{idx, j0, j1, row0[size_t(it.first + j0)], rows, nb, mx};
                for (int j = j0; j <= j1; ++j) s_hbound_[nb++] = row0[size_t(it.first + j)] - pr.row_first;
                prods.push_back(pr);
                j0 = j1;
            }
        }
        const int rows = row0[size_t(c.experts)];
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
                ck(cudaStreamWaitEvent(stream_, ev_ready_[size_t(pr.item % dslots_)], 0), "slot ready");
                waited = pr.item;
            }
            if (pr.rows == 0) continue;
            const uint8_t* stage = static_cast<const uint8_t*>(dring_[size_t(pr.item % dslots_)].p) + size_t(pr.j0) * it.stride;
            const int32_t* rtok = s_idx_.i() + pr.row_first;
            const float* rw = s_w_.f(pr.row_first);
            mq::quantize(xn_.f(), rtok, m_xq_.p, L.f.gu_type, c.hidden, c.hidden, pr.rows, stream_);
            mq::Product gu;
            gu.w = stage; gu.type = L.f.gu_type; gu.w_rows = 2 * c.ff; gu.w_cols = c.hidden; gu.expert_bytes = it.stride;
            gu.n = pr.j1 - pr.j0; gu.xq = m_xq_.p; gu.bounds = s_bound_.i() + pr.bound0; gu.ids = s_iota_.i(); gu.total_rows = pr.rows;
            gu.max_rows = pr.max_rows; gu.dst = s_gu_.f(); gu.ld_dst = 2 * c.ff;
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
        // the row tables are rewritten by the next layer only after its router sync, which follows these uploads
    }

    void ffn(int l) {
        const auto& c = cfg_;
        const int T = tokens_, K = c.top_k;
        const size_t H = size_t(c.hidden);
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool hook = T == 1 && observe;
        // router on the host: sqrt(softplus(logits)), top-k on score + bias, weights from the unbiased score
        const auto f0 = std::chrono::steady_clock::now();
        mat(p + "ffn_gate_inp.weight", xn_.f(), logits_.f(), T);
        std::vector<float> logits(size_t(T) * c.experts);
        ck(cudaMemcpyAsync(logits.data(), logits_.p, logits.size() * 4, cudaMemcpyDeviceToHost, stream_), "router");
        float* host_y = pinned_[1 + (l & 1)];
        if (!producer_.joinable())   // the CPU lane's input (a streamed pass computes every expert on the GPU)
            ck(cudaMemcpyAsync(pinned_[0], xn_.p, size_t(T) * H * 4, cudaMemcpyDeviceToHost, stream_), "expert input");
        ck(cudaStreamSynchronize(stream_), "router sync");
        std::vector<int> ids(size_t(T) * K);
        std::vector<float> wts(size_t(T) * K);
        const auto& bias = bias_[size_t(l)];
        auto route = [&](int i0, int i1) {
        std::vector<float> probs(size_t(c.experts)), selection(size_t(c.experts));
        for (int i = i0; i < i1; ++i) {
            const float* lg = logits.data() + size_t(i) * c.experts;
            for (int e = 0; e < c.experts; ++e) {
                probs[size_t(e)] = std::sqrt(softplus(lg[e]));
                selection[size_t(e)] = probs[size_t(e)] + bias[size_t(e)];
            }
            const auto pick = top_k(selection.data(), c.experts, K);
            float sum = 0;
            for (int k = 0; k < K; ++k) { ids[size_t(i) * K + k] = pick[size_t(k)]; wts[size_t(i) * K + k] = probs[size_t(pick[size_t(k)])]; sum += wts[size_t(i) * K + k]; }
            for (int k = 0; k < K; ++k) {
                float& v = wts[size_t(i) * K + k];
                if (c.normalize && K > 1) v /= sum + 1e-20f;
                v *= c.expert_scale;
            }
        }
        };
        if (T >= 256) {   // long passes: tokens split over helper threads
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
        // the GPU: the shared expert, then the resident routed experts; the CPU lane takes the others meanwhile
        mat(p + "ffn_gate_shexp.weight", xn_.f(), sh_gate_.f(), T);
        mat(p + "ffn_up_shexp.weight", xn_.f(), sh_up_.f(), T);
        k::glm_swiglu(sh_gate_.f(), sh_up_.f(), sh_h_.f(), T * c.ff, c.shared_clamp[size_t(l)], stream_);
        mat(p + "ffn_down_shexp.weight", sh_h_.f(), y_.f(), T);
        if (producer_.joinable()) {   // a streamed pass: every routed expert on the GPU, the others streamed in
            if (gpu_timing_) mark(1);
            stream_experts(l, ids, wts);
            if (gpu_timing_) mark(2);
            prof_[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
            if (hook) observe("ffn_moe_out", l, g_sum_.f(), H);
            k::ds41_add(g_sum_.f(), y_.f(), y_.f(), int(size_t(T) * H), stream_);
            return;
        }
        std::vector<int> cpu_ids(ids);
        bool any_gpu = false;
        for (int i = 0; i < T; ++i) {
            int gid[16];
            float gw[16];
            int n = 0;
            for (int k = 0; k < K; ++k) {
                const int e = ids[size_t(i) * K + k];
                const bool on_gpu = !slot_.empty() && slot_[size_t(l)][size_t(e)] >= 0;
                used_.push_back(l * c.experts + e);
                if (!on_gpu) { missed_.push_back(l * c.experts + e); continue; }
                gid[n] = e;
                gw[n++] = wts[size_t(i) * K + k];
                cpu_ids[size_t(i) * K + k] = -1;
            }
            if (n && !any_gpu) { ck(cudaMemsetAsync(g_sum_.p, 0, size_t(T) * H * 4, stream_), "tier sum"); any_gpu = true; }
            if (n) gpu_experts(l, i, gid, gw, n);
            tier_hits_ += n;
            tier_all_ += K;
        }
        lane_->run(experts_[size_t(l)], pinned_[0], T, cpu_ids.data(), wts.data(), K, host_y);
        prof_[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
        ck(cudaMemcpyAsync(moe_.p, host_y, size_t(T) * H * 4, cudaMemcpyHostToDevice, stream_), "experts");
        if (any_gpu) k::ds41_add(moe_.f(), g_sum_.f(), moe_.f(), int(size_t(T) * H), stream_);
        if (hook) {
            observe("ffn_shexp", l, y_.f(), H);
            observe("ffn_moe_out", l, moe_.f(), H);
        }
        k::ds41_add(moe_.f(), y_.f(), y_.f(), int(size_t(T) * H), stream_);
        // no sync: the next layer's router sync orders this upload before host_y is written again (two layers on)
    }

public:
    /// STRATA_DS41_ROUTES=path: every token's expert ids, int32 [layers][top_k] per token.
    void log_routes(const std::string& path) { routes_.open(path, std::ios::binary); }
    /// Seconds: [0] passes, [1] GPU work up to each layer's router (synchronous), [2] CPU experts, [3] adapt,
    /// [4] Engram gather, [5] indexer host, [6] head, [7] streamed experts' host copies.
    double prof_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double tier_hits_ = 0, tier_all_ = 0, promotions_ = 0;

private:
    core::ModelArtifact art_;
    std::map<const GgufFile*, std::pair<void*, size_t>> shared_maps_;
    std::vector<std::pair<void*, size_t>> populate_;
    struct Src { const uint8_t *gate, *up, *down; };
    std::vector<Src> stream_src_;   // the experts in the shared mapping (the streamed prompt's copies)
    Config cfg_;
    int context_;
    bool tier_lent_ = false;
    Dev attn_info_dev_;
    int* attn_info_ = nullptr;   // pinned: per-token attention rows of a pass
    int rcap_min_ = 8192;
    std::vector<std::unique_ptr<void, void (*)(void*)>> pinned_bufs_;
    int chunk_ = 256, ring_ = 0, pos0_ = 0, tokens_ = 1, score_stride_ = 0;
    cudaStream_t stream_ = nullptr;
    std::map<std::string, Weight> weights_;
    std::vector<std::vector<float>> bias_;
    std::vector<LaneLayer> experts_;
    std::unique_ptr<ExpertLane> lane_;
    Dev attn_scores_, h_, h2_, x_, xn_, flat_, mixes_, coeff_attn_, coeff_ffn_, qa_, qr_, q_, kv_, kvn_, att_, oa_, oa_part_, y_, q8_, q8g_, cs_;
    Dev comp_v_, comp_s_, comp_, ik_, iq_, iw_, logits_, sh_gate_, sh_up_, sh_h_, moe_, scores_, engram_rows_, engram_kv_;
    std::vector<Dev> windows_, caches_, stash_v_, stash_s_, sel_;
    std::vector<std::vector<int>> sel_count_, candidates_;
    const core::ArtifactTensor* engram_table_ = nullptr;
    struct EngramReq { int64_t row; float* dst; };
    std::vector<EngramReq> engram_reqs_;
    int engram_fd_ = -1, engram_threads_ = 16;
    std::vector<int32_t> history_, route_log_;
    std::vector<float> logits_host_;
    float* pinned_[3] = {nullptr, nullptr, nullptr};   // the CPU lane's input (and the embeddings) and its two outputs
    std::ofstream routes_;
    std::vector<float> engram_host_, rope_host_;   // the uploads' sources outlive a pass
    // the GPU expert tier
    Dev arena_;
    std::vector<Slot> slots_;
    std::vector<std::vector<int>> slot_;   // [layer][expert] -> slot, or -1
    std::vector<float> score_;
    std::vector<int> used_, missed_;
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
    std::thread stager_;
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
    std::vector<void*> registered_;
    std::vector<char> pinned_layer_;
public:
    double stream_wait_s_ = 0;
    long skipped_ = 0;
    // STRATA_DS41_GPU_TIMING: per-layer GPU events of streamed passes
    bool gpu_timing_ = std::getenv("STRATA_DS41_GPU_TIMING") != nullptr;
    std::vector<cudaEvent_t> marks_;
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

int run(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: strata-glm-decode DS41-MODEL.gguf <ids|@file> [steps] [--threads=N] [--context=N] "
                     "[--check=DUMPDIR] [--dump-logits=PATH]\n";
        return 2;
    }
    int steps = 0, threads = int(std::max(1u, std::thread::hardware_concurrency() / 2)), context = 0;
    std::string check_dir, dump_logits, save_all;
    bool force = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--context=")) context = std::stoi(a.substr(10));
        else if (a.starts_with("--check=")) check_dir = a.substr(8);
        else if (a == "--check-force") force = true;   // each layer from the reference's input and expert picks
        else if (a.starts_with("--dump-logits=")) dump_logits = a.substr(14);
        else if (a.starts_with("--save-all-logits=")) save_all = a.substr(18);   // every prompt position's logits, f32
        else if (!a.starts_with("--")) steps = std::stoi(a);
        else throw std::invalid_argument("ds41: unknown option " + a);
    }
    const auto prompt = parse_ids(argv[2]);
    if (prompt.empty()) throw std::invalid_argument("ds41: empty prompt");
    if (!context) context = int(prompt.size()) + steps + 1;
    const auto t0 = std::chrono::steady_clock::now();
    // a long prompt runs as one layer-major pass (up to STRATA_DS41_PROMPT_BATCH, 4096 tokens): every expert crosses
    // PCIe once per pass; shorter prompts take chunks of STRATA_DS41_PREFILL_CHUNK (256) with the CPU lane
    int chunk = 0;
    {
        const char* v = std::getenv("STRATA_DS41_PROMPT_BATCH");
        const int batch = v ? std::max(1, std::atoi(v)) : 4096;
        const int stream_min = std::getenv("STRATA_DS41_STREAM_MIN") ? std::atoi(std::getenv("STRATA_DS41_STREAM_MIN")) : 1024;
        if (int(prompt.size()) >= stream_min && !std::getenv("STRATA_DS41_PREFILL_CHUNK") && check_dir.empty())
            chunk = std::min(int(prompt.size()), batch);
    }
    Decoder d(argv[1], context, threads, chunk);
    if (const char* r = std::getenv("STRATA_DS41_ROUTES")) d.log_routes(r);
    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "DS41 load_s=" << std::chrono::duration<double>(t1 - t0).count() << " threads=" << threads << "\n";
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
    for (size_t i = 0; i < prompt.size(); i += size_t(pass)) {
        position = int(i);
        const int n = int(std::min(prompt.size() - i, size_t(pass)));
        const double copy0 = d.prof_[7];
        const auto c0 = std::chrono::steady_clock::now();
        const auto& out = d.forward(prompt.data() + i, n, int(i), all_logits.is_open());
        if (std::getenv("STRATA_DS41_CHUNK_LOG"))
            std::cerr << "DS41 chunk pos=" << i << " tokens=" << n << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count()
                      << " stage_copy_s=" << d.prof_[7] - copy0 << " gpu_to_router_s=" << d.prof_[1] << " experts_s=" << d.prof_[2]
                      << " engram_s=" << d.prof_[4] << " index_s=" << d.prof_[5] << " stream_wait_s=" << d.stream_wait_s_
                      << " enc_skip_token_layers=" << d.skipped_ << "\n";
        if (all_logits) all_logits.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
        last.assign(out.end() - std::ptrdiff_t(V), out.end());
    }
    const auto t2 = std::chrono::steady_clock::now();
    d.prefill_done();
    const auto t2b = std::chrono::steady_clock::now();
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
    for (int s = 0; s < steps; ++s) {
        const int tok = int(std::max_element(last.begin(), last.end()) - last.begin());
        out.push_back(tok);
        if (s + 1 == steps) break;
        last = d.step(tok, pos++);
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t2 - t1).count();
    const double decode_s = std::chrono::duration<double>(t3 - t2b).count();
    std::cerr << "DS41 profile_s step=" << d.prof_[0] << " gpu_to_router=" << d.prof_[1] << " cpu_experts=" << d.prof_[2]
              << " gpu_tier_hit=" << (d.tier_all_ > 0 ? d.tier_hits_ / d.tier_all_ : 0) << " promotions=" << d.promotions_ << " adapt=" << d.prof_[3] << " engram=" << d.prof_[4]
              << " index_host=" << d.prof_[5] << " head=" << d.prof_[6] << " stage_copy=" << d.prof_[7] << " stream_wait=" << d.stream_wait_s_ << "\n";
    std::cout << "tokens:";
    for (int t : out) std::cout << ' ' << t;
    std::cout << "\n";
    std::cerr << "DS41 prompt=" << prompt.size() << " prompt_tok_s=" << prompt.size() / std::max(prefill_s, 1e-9)
              << " generated=" << out.size() << " decode_tok_s="
              << (out.size() > 1 ? double(out.size() - 1) / std::max(decode_s, 1e-9) : 0.0) << "\n";
    return 0;
}

}  // namespace strata::ds41
