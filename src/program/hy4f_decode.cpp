// src/program/hy4f_decode.cpp - strata-hy4f: Hy4-Flash (GGUF arch hy_v4_flash, Tencent Hy4 "512x71B" A12B) decode on
// one GPU + the CPU, one token at a time.
//
// Reference: llama.cpp PR #32517 (src/models/hy-v4-flash.cpp). Per block: Engram (4 layers: 3-gram hashed rows of a
// 62.7 GiB table read from the mmapped GGUF), iHC pre-mix (4 streams, sigmoid pre/post, no Sinkhorn), gated MLA with
// sinks (absorbed: 576-wide latent+rope keys, 512-wide latent values; SWA-512 or full layers), sandwich norms, and
// either the dense FFN (layer 0) or the latent MoE: router on the full width (sigmoid + bias, top 16 of 512, normed,
// x3.998), routed_down 3072 -> 1536, 16 routed experts at 1536 (clamped SwiGLU 10) on the CPU lane (ds41::ExpertLane,
// NVFP4 rows bit-equal to ggml-cpu), routed_up 1536 -> 3072, plus the shared expert on the GPU.
//
// Everything that is not a routed expert or the Engram table lives on the GPU (~4.2 GB). Dense GEMVs are q8_1
// activations into iq_mmvq (NVFP4 / Q8_0, llama.cpp's CUDA arithmetic); attention is ds41_attention over 640-wide
// rows (576 padded). The DSA indexer (kpool 4, top-k 2048) selects every token below 2048 positions, so up to that
// context the attention here is exact without it; longer contexts are refused until the indexer is ported.
//
//     strata-hy4f MODEL.gguf --ids 1,2,3 [--n 32] [--layers L] [--dump DIR] [--threads 15] [--ctx 2048]
//
// --layers L runs only the first L blocks (no head) and with --dump writes DIR/l_out-<il>.bin per block: the four
// streams of every prompt position ([n][4][pos], the ds41f_llama_dump .bin format: int32 type, int64 ne[4], f32).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/hy4f.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace cpu = strata::kernels::cpu;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int kF32 = 0, kQ8_0 = 8, kNvfp4 = 40;
constexpr int kAttnSplit = 8;
constexpr int kRow = 640;   // MLA attention row: latent 512 | rope 64 | zero pad (ds41_attention wants dim % 128)

float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m | 0x400), int(e) - 25);
    return s ? -v : v;
}

void dequant_q8_0(const uint8_t* src, int64_t n, float* dst) {
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t d16;
        std::memcpy(&d16, src + b * 34, 2);
        const float d = half_to_float(d16);
        for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * float(int8_t(src[b * 34 + 2 + i]));
    }
}

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
};

// one weight on the GPU: type, GGUF shape (ne0 = inner), device bytes
struct W {
    int type = -1;
    int64_t in = 0, out = 0;   // ne0, product of the rest
    void* d = nullptr;
    explicit operator bool() const { return d != nullptr; }
};

struct Layer {
    bool swa = false, dense = false, engram = false;
    int heads = 0, engram_idx = -1;
    float rope_base = 0;
    W attn_norm, sinks, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, wo, gate, post_attn_norm;
    W hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    W ffn_norm, post_ffn_norm;
    W ffn_gate, ffn_up, ffn_down;                                   // dense
    W gate_inp, routed_down, routed_up, sh_gate, sh_up, sh_down;     // MoE
    W eng_kv, eng_kv_b, eng_gate, eng_gamma, eng_hc_scale;
    bool indexer = false;
    W ix_qb, ix_k, ix_knorm, ix_proj, ix_ape, ix_gate, ix_val, ix_cnorm;
    float *vg = nullptr, *pooled = nullptr;   // per position [value | gate], per complete group the pooled key
    std::vector<float> bias, lookup_norm;
    W bias_d;                            // host: router bias, Engram row norm
    strata::ds41::LaneLayer lane;
    float* kv = nullptr;   // ring [ring][kRow]
    int ring = 0;
};


// The GPU expert tier: whole routed experts ([gate | up | down] NVFP4 blobs) in a VRAM arena, computed by
// native_expert_grouped while the CPU lane computes the rest of the layer's routes. Adaptive: a route that misses
// raises its expert's decayed count; misses take free slots, then replace the lowest-count resident (a per-token
// budget of uploads through a pinned staging ring on a copy stream ordered after the decode stream).
struct Tier {
    int experts = 0, slots = 0, used = 0;
    size_t blob = 0, gate_bytes = 0, down_bytes = 0;
    uint8_t* arena = nullptr;
    std::vector<int> slot_of, owner;
    std::vector<char> ready;
    std::vector<cudaEvent_t> slot_ev;
    std::vector<float> score;
    cudaStream_t copy = nullptr;
    static constexpr int kStage = 4;
    uint8_t* stage[kStage] = {};
    cudaEvent_t stage_ev[kStage] = {};
    int next_stage = 0;
    k::NativeExpertLayout L;
    void* table_d = nullptr;
    uint8_t* table_h[2] = {};
    cudaEvent_t table_ev[2] = {};
    unsigned calls = 0;
    void* scratch = nullptr;
    float* out = nullptr;
    void* xq = nullptr;
    float* sum = nullptr;
    long hits = 0, routes = 0, promotions = 0;
    unsigned long long* d_slots = nullptr;   // device [layers][experts]: the resident blob's address, 0 = not on the GPU
    unsigned long long* addr_h = nullptr;    // pinned, per slot: its blob's address (the source of the publish copies)
    std::vector<int> pending;                // slots uploading; published to d_slots once their copy is done
    // pending slots whose upload finished: their address into d_slots (stream-ordered before the next routes)
    void publish(cudaStream_t s) {
        for (size_t i = 0; i < pending.size();) {
            const int sl = pending[i];
            if (cudaEventQuery(slot_ev[size_t(sl)]) != cudaSuccess) { ++i; continue; }
            ready[size_t(sl)] = 1;
            ck(cudaMemcpyAsync(d_slots + owner[size_t(sl)], addr_h + sl, 8, cudaMemcpyHostToDevice, s), "tier publish");
            pending[i] = pending.back();
            pending.pop_back();
        }
    }
    int budget = 0, left = 0;

    bool on() const { return slots > 0; }
    bool is_ready(int s) {
        if (!ready[size_t(s)] && cudaEventQuery(slot_ev[size_t(s)]) == cudaSuccess) ready[size_t(s)] = 1;
        return ready[size_t(s)] != 0;
    }
    // group the resident routes: blob address and weight; out rows e, summed into `sum` (n floats)
    void run(const std::vector<std::pair<const uint8_t*, float>>& g, const float* x, int n, cudaStream_t s) {
        const int G = int(g.size());
        const int sb = int(calls++ & 1u);
        ck(cudaEventSynchronize(table_ev[sb]), "tier table staging");
        uint8_t* h = table_h[sb];
        auto* ptr = reinterpret_cast<unsigned long long*>(h);
        auto* start = reinterpret_cast<int32_t*>(ptr + 16);
        int32_t* ng = start + 17;
        int32_t* dst = ng + 1;
        int32_t* tok = dst + 16;
        float* w = reinterpret_cast<float*>(tok + 16);
        for (int i = 0; i < G; ++i) {
            ptr[i] = reinterpret_cast<unsigned long long>(g[size_t(i)].first);
            start[i] = i;
            dst[i] = i;
            tok[i] = 0;
            w[i] = g[size_t(i)].second;
        }
        start[G] = G;
        *ng = G;
        const size_t bytes = size_t(reinterpret_cast<uint8_t*>(w + 16) - h);
        auto* tb = (uint8_t*) table_d + size_t(sb) * 512;
        ck(cudaMemcpyAsync(tb, h, bytes, cudaMemcpyHostToDevice, s), "tier table");
        ck(cudaEventRecord(table_ev[sb], s), "tier table event");
        k::quantize_q8_1_rows(x, 1, n, xq, s);
        const size_t o_start = 16 * 8, o_ng = o_start + 17 * 4, o_dst = o_ng + 4, o_tok = o_dst + 16 * 4, o_w = o_tok + 16 * 4;
        k::native_expert_grouped(L, reinterpret_cast<const unsigned long long*>(tb), reinterpret_cast<const int32_t*>(tb + o_start),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), reinterpret_cast<const int32_t*>(tb + o_dst),
                                 reinterpret_cast<const int32_t*>(tb + o_tok), 16, 16, xq, scratch, out, s);
        ck(cudaMemsetAsync(sum, 0, size_t(n) * 4, s), "tier sum");
        k::ds41_scatter_axpy(out, reinterpret_cast<const int32_t*>(tb + o_tok), reinterpret_cast<const float*>(tb + o_w), G, n, sum, s);
    }
    // upload expert `key` (gate/up/down rows at the given host addresses) into a slot; false: not worth a slot
    bool promote(int key, const uint8_t* gate, const uint8_t* up, const uint8_t* down, cudaStream_t s, cudaEvent_t after, bool free_only) {
        int slot = -1;
        if (used < slots) slot = used++;
        else {
            if (free_only) return false;
            float best = INFINITY;
            for (int i = 0; i < slots; ++i)
                if (ready[size_t(i)] && score[size_t(owner[size_t(i)])] < best) { best = score[size_t(owner[size_t(i)])]; slot = i; }
            if (slot < 0 || best >= score[size_t(key)]) return false;
            slot_of[size_t(owner[size_t(slot)])] = -1;
            // unpublish before the slot is overwritten: later routes stop using it; earlier work is ordered before `after`
            ck(cudaMemsetAsync(d_slots + owner[size_t(slot)], 0, 8, s), "tier unpublish");
        }
        ck(cudaEventRecord(after, s), "tier order event");
        const int b = next_stage++ % kStage;
        ck(cudaEventSynchronize(stage_ev[b]), "tier staging");
        std::memcpy(stage[b], gate, gate_bytes);
        std::memcpy(stage[b] + L.up_off, up, gate_bytes);
        std::memcpy(stage[b] + L.down_off, down, down_bytes);
        const float tail[4] = {1.0f, 1.0f, 1.0f, 0.0f};   // GGUF NVFP4 has no per-tensor scale
        std::memcpy(stage[b] + L.tail_off, tail, sizeof tail);
        ck(cudaStreamWaitEvent(copy, after, 0), "tier order");
        ck(cudaMemcpyAsync(arena + size_t(slot) * blob, stage[b], blob, cudaMemcpyHostToDevice, copy), "tier upload");
        ck(cudaEventRecord(stage_ev[b], copy), "tier stage event");
        ck(cudaEventRecord(slot_ev[size_t(slot)], copy), "tier slot event");
        owner[size_t(slot)] = key;
        slot_of[size_t(key)] = slot;
        ready[size_t(slot)] = 0;
        pending.push_back(slot);
        ++promotions;
        return true;
    }
};

std::string join_ids(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

class Hy4 {
public:
    Hy4(const std::string& path, int threads, int ctx, int layers) : g_(path), ctx_(ctx) {
        read_config();
        n_run_ = layers > 0 ? std::min(layers, n_layer_) : n_layer_;
        ck(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "stream");
        ck(cudaHostAlloc((void**) &ring_, ring_cap_, cudaHostAllocDefault), "h2d ring");
        load(threads);
    }

    // one token at position pos; returns the logits (when the head runs) on the host
    const std::vector<float>& step(int tok, int pos);
    // the four streams after block il for the last step (host copy), filled when dumping
    std::vector<std::vector<float>> dumps;   // [il] -> [4 * n] of the last step
    bool want_dump = false;
    std::map<std::string, std::vector<float>> dbg;   // name-<il> -> values of every step, appended
    int n_run() const { return n_run_; }
    int n_layer() const { return n_layer_; }
    int vocab() const { return vocab_; }
    double expert_ms = 0, gpu_ms = 0;
    Tier tier;
    // STRATA_HY4F_PROF=1: per-phase host times (ms, summed over steps); attention is synchronized to be measured
    bool prof = false;
    bool fused = true;   // STRATA_HY4F_UNFUSED=1: the separate kernels (A/B)
    bool graphs = true;  // STRATA_HY4F_GRAPH=0: eager only
    long graph_launches = 0, graph_captures = 0;
    double prof_attn = 0, prof_route = 0, prof_experts = 0, prof_post = 0;
    using clk = std::chrono::steady_clock;
    static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
    void tier_setup(double gib, double reserve_gib, int budget);

private:
    strata::GgufFile g_;
    int n_layer_ = 0, n_embd_ = 0, vocab_ = 0, hc_ = 0, n_expert_ = 0, top_k_ = 0, n_latent_ = 0, n_ff_exp_ = 0;
    int q_rank_ = 0, kv_rank_ = 0, rope_dim_ = 0, k_head_ = 0, v_head_ = 0, window_ = 0, dense_ff_ = 0;
    float eps_ = 0, hc_eps_ = 0, hc_mag_ = 0, w_scale_ = 1, clamp_ = 0;
    bool w_norm_ = true;
    float rope_full_ = 0, rope_swa_ = 0;
    int ix_top_ = 0, ix_heads_ = 0, ix_dim_ = 0, kpool_ = 0;
    float *ivg_ = nullptr;
    float *iq_ = nullptr, *ik_ = nullptr, *iw_ = nullptr, *iscore_ = nullptr, *cs2_ = nullptr;
    int* isel_ = nullptr;
    std::vector<float> h_iscore_, h_cs2_;
    std::vector<int> h_isel_;
    long sparse_calls = 0;
    void indexer(Layer& l, int pos, int& nsel);
    // the step block: ints (position, attention windows, slots, Engram flag) then floats (rope tables full | swa | pool,
    // the embedding row, every Engram module's rows); one upload per step, read by the kernels (graph-capturable)
    enum { SV_POS = 0, SV_INFO_FULL = 4, SV_INFO_SWA = 8, SV_SLOT_FULL = 12, SV_SLOT_SWA = 13, SV_ANY = 14, SV_INTS = 32 };
    uint8_t *step_d_ = nullptr, *step_h_ = nullptr;
    size_t step_bytes_ = 0;
    int* sv_ = nullptr;           // device ints
    float* emb_d_ = nullptr;      // device embedding row
    float* eng_rows_d_ = nullptr; // device Engram rows [module][cols * dim]
    float* logits_h_ = nullptr;   // pinned
    void prepare_step(int tok, int pos);
    bool engram_rows(int e, int tok, int pos, float* out);   // false: the token has no valid n-gram
    // segments between the CPU lane's host waits: eager, or captured once and replayed as HIP graphs
    bool graph_step_ = false;
    std::vector<cudaGraphExec_t> execs_;
    int seg_ = 0;
    template <class F> void segment(F&& f);
    void moe_route(Layer& l);
    void moe_side(Layer& l);
    void moe_host(Layer& l);
    void moe_tail(Layer& l);
    // Engram
    std::vector<int> e_layers_;
    int ngram_ = 0, e_heads_ = 0, e_dim_ = 0, e_cols_ = 0;
    std::vector<uint64_t> e_mult_, e_primes_, e_offs_;
    std::vector<int64_t> e_map_;
    int64_t e_pad_ = 0;
    const uint8_t* e_table_ = nullptr;
    size_t e_row_bytes_ = 0;
    // globals
    const strata::TensorInfo* tok_embd_ = nullptr;
    W out_norm_, output_, hc_head_fn_, hc_head_base_, hc_head_scale_;
    std::vector<Layer> L_;
    int n_run_ = 0, ctx_ = 0;
    std::unique_ptr<strata::ds41::ExpertLane> lane_;
    cudaStream_t s_ = nullptr;
    std::vector<int> history_;
    std::vector<float> logits_;
    // device scratch
    float *h_ = nullptr, *x_ = nullptr, *cur_ = nullptr, *flat_ = nullptr, *mix_ = nullptr, *coef_ = nullptr;
    float *qr_ = nullptr, *q_ = nullptr, *kvpe_ = nullptr, *kvn_ = nullptr, *qa_ = nullptr, *q640_ = nullptr, *att_ = nullptr;
    float *av_ = nullptr, *gt_ = nullptr, *o_ = nullptr, *ff_g_ = nullptr, *ff_u_ = nullptr, *ff_a_ = nullptr, *logit_e_ = nullptr;
    float *rin_ = nullptr, *rout_ = nullptr, *sh_ = nullptr, *eng_r_ = nullptr, *eng_kv_ = nullptr, *cs_ = nullptr, *scores_ = nullptr;
    float* logits_d_ = nullptr;
    size_t scores_bytes_ = 0;
    void* q8_ = nullptr;
    void* q8b_ = nullptr;
    // host staging
    float *h_logit_e_ = nullptr, *h_rin_ = nullptr, *h_rout_ = nullptr;   // pinned
    int* h_route_ = nullptr;   // pinned, after h_rin_: hy4_route's {cpu ids, weights, ids}
    void* rtable_ = nullptr;   // device: hy4_route's grouped table
    void* rxq_ = nullptr;      // device: q8_1 of the latent for the GPU tier
    std::vector<float> h_cs_;
    cudaEvent_t ev_ = nullptr, ev_promo_ = nullptr;

    const strata::TensorInfo& T(const std::string& n) const {
        const auto* t = g_.find(n);
        if (!t) throw std::runtime_error("hy4f: missing tensor " + n);
        return *t;
    }
    bool has(const std::string& n) const { return g_.find(n) != nullptr; }
    const strata::MetaValue& M(const std::string& k) const {
        const auto* v = g_.get("hy_v4_flash." + k);
        if (!v) throw std::runtime_error("hy4f: missing hy_v4_flash." + k);
        return *v;
    }
    std::vector<int64_t> ints(const std::string& k) const { return g_.int_array("hy_v4_flash." + k); }
    template <class Tv> std::vector<Tv> items(const std::string& k) const {
        std::vector<Tv> out;
        for (const auto& it : M(k).items) out.push_back(Tv(it.is_num() ? it.num() : 0));
        return out;
    }

    void read_config();
    W upload(const std::string& name);
    void load(int threads);
    void gemv(const W& w, const float* x, float* y);              // quantizes x (w.in floats) first
    void gemv_q8(const W& w, const void* q8, float* y);           // x already quantized
    void rms(const float* x, const W* w, float* y, int n, void* q8 = nullptr);
    void hc_pre(const W& fn, const W& scale, const W& base, bool post);
    void hc_mix(const W& fn, const W& scale, const W& base, bool post, const W& norm);   // + norm -> cur_, q8_
    void post(const W& norm, const float* add);
    void attention(Layer& l, int pos);
    void ffn(Layer& l);
    void engram(Layer& l, int tok, int pos);
    void dbg_host(const std::string& name, int il, const float* v, size_t n) {
        if (!want_dump) return;
        auto& d = dbg[name + "-" + std::to_string(il)];
        d.insert(d.end(), v, v + n);
    }
    void dbg_dev(const std::string& name, int il, const float* v, size_t n) {
        if (!want_dump) return;
        std::vector<float> h(n);
        d2h(h.data(), v, n * 4);
        dbg_host(name, il, h.data(), n);
    }
    int cur_layer_ = 0;
    // Every host<->device copy goes through pinned memory: no pageable / file-mapped source reaches the driver (a
    // 2026-11-19 TR16 hang followed amdgpu svm_range work and a corrupted page freelist; pageable copies from the
    // mmapped GGUF were in flight). h2d: a pinned ring on the decode stream (synchronized when it wraps); d2h: synchronous.
    uint8_t* ring_ = nullptr;
    size_t ring_cap_ = size_t(64) << 20, ring_off_ = 0;
    uint8_t* back_ = nullptr;
    size_t back_cap_ = 0;
    void h2d(void* dst, const void* src, size_t n) {
        const auto* p = (const uint8_t*) src;
        auto* d = (uint8_t*) dst;
        while (n) {
            const size_t c = std::min(n, ring_cap_);
            if (ring_off_ + c > ring_cap_) { ck(cudaStreamSynchronize(s_), "h2d ring"); ring_off_ = 0; }
            std::memcpy(ring_ + ring_off_, p, c);
            ck(cudaMemcpyAsync(d, ring_ + ring_off_, c, cudaMemcpyHostToDevice, s_), "h2d");
            ring_off_ = (ring_off_ + c + 255) & ~size_t(255);
            p += c; d += c; n -= c;
        }
    }
    void d2h(void* dst, const void* src, size_t n) {
        if (n > back_cap_) {
            if (back_) ck(cudaFreeHost(back_), "d2h free");
            back_cap_ = std::max(n, size_t(1) << 20);
            ck(cudaHostAlloc((void**) &back_, back_cap_, cudaHostAllocDefault), "d2h buffer");
        }
        ck(cudaMemcpyAsync(back_, src, n, cudaMemcpyDeviceToHost, s_), "d2h");
        ck(cudaStreamSynchronize(s_), "d2h sync");
        std::memcpy(dst, back_, n);
    }
    float* pinned(size_t floats) {
        void* p = nullptr;
        ck(cudaHostAlloc(&p, floats * 4, cudaHostAllocDefault), "pinned");
        std::memset(p, 0, floats * 4);
        return (float*) p;
    }
    float* alloc(size_t floats) {
        void* p = nullptr;
        ck(cudaMalloc(&p, std::max<size_t>(floats, 4) * sizeof(float)), "alloc");
        ck(cudaMemset(p, 0, std::max<size_t>(floats, 4) * sizeof(float)), "memset");
        return (float*) p;
    }
};

void Hy4::read_config() {
    if (const auto* a = g_.get("general.architecture"); !a || a->s != "hy_v4_flash")
        throw std::runtime_error("hy4f: not a hy_v4_flash GGUF");
    const int blocks = int(M("block_count").u);
    const int nextn = g_.get("hy_v4_flash.nextn_predict_layers") ? int(M("nextn_predict_layers").u) : 0;
    n_layer_ = blocks - nextn;
    n_embd_ = int(M("embedding_length").u);
    vocab_ = int(M("vocab_size").u);
    hc_ = int(M("hyper_connection.count").u);
    hc_eps_ = float(M("hyper_connection.epsilon").num());
    hc_mag_ = float(M("hyper_connection.magnitude").num());
    n_expert_ = int(M("expert_count").u);
    top_k_ = int(M("expert_used_count").u);
    n_latent_ = int(M("expert_latent_length").u);
    n_ff_exp_ = int(M("expert_feed_forward_length").u);
    dense_ff_ = int(M("feed_forward_length").u);
    q_rank_ = int(M("attention.q_lora_rank").u);
    kv_rank_ = int(M("attention.kv_lora_rank").u);
    rope_dim_ = int(M("rope.dimension_count").u);
    k_head_ = int(M("attention.key_length_mla").u) - rope_dim_;   // the nope part (128)
    v_head_ = int(M("attention.value_length_mla").u);
    window_ = int(M("attention.sliding_window").u);
    eps_ = float(M("attention.layer_norm_rms_epsilon").num());
    w_scale_ = float(M("expert_weights_scale").num());
    w_norm_ = M("expert_weights_norm").u != 0;
    if (M("expert_gating_func").u != 2) throw std::runtime_error("hy4f: expected sigmoid gating");
    rope_full_ = float(M("rope.freq_base").num());
    rope_swa_ = float(M("rope.freq_base_swa").num());
    const auto clamp = items<float>("swiglu_clamp_exp");
    clamp_ = clamp.size() > 1 ? clamp[1] : 0.0f;
    const auto heads = items<int>("attention.head_count");
    const auto swa = items<int>("attention.sliding_window_pattern");
    if (int(heads.size()) < n_layer_ || int(swa.size()) < n_layer_) throw std::runtime_error("hy4f: per-layer arrays too short");
    if (kv_rank_ != 512 || rope_dim_ != 64 || n_embd_ % 256) throw std::runtime_error("hy4f: unexpected MLA geometry");
    L_.resize(size_t(n_layer_));
    const int dense_lead = int(M("leading_dense_block_count").u);
    for (int i = 0; i < n_layer_; ++i) {
        auto& l = L_[size_t(i)];
        l.heads = heads[size_t(i)];
        l.swa = swa[size_t(i)] != 0;
        l.dense = i < dense_lead;
        l.rope_base = l.swa ? rope_swa_ : rope_full_;
        l.ring = l.swa ? window_ : ctx_;
        if (l.heads % 8) throw std::runtime_error("hy4f: head count must be a multiple of 8");
    }
    if (g_.get("hy_v4_flash.attention.indexer.top_k")) {
        ix_top_ = int(M("attention.indexer.top_k").u);
        ix_heads_ = int(M("attention.indexer.head_count").u);
        ix_dim_ = int(M("attention.indexer.key_length").u);
        kpool_ = int(M("attention.indexer.kpool").u);
        if (!g_.get("hy_v4_flash.attention.indexer.kpool_select_tail") || !M("attention.indexer.kpool_select_tail").u)
            throw std::runtime_error("hy4f: only the select-tail indexer is implemented");
        const auto types = items<int>("attention.indexer.types");
        for (int i = 0; i < n_layer_ && i < int(types.size()); ++i) L_[size_t(i)].indexer = types[size_t(i)] != 0;
        for (const auto& l : L_)
            if (l.indexer && l.swa) throw std::runtime_error("hy4f: a window layer owns an indexer");
        for (const auto& l : L_)
            if (!l.indexer && !l.swa && ctx_ > ix_top_) throw std::runtime_error("hy4f: a full layer without an indexer");
        if (ix_dim_ != 128 || kpool_ < 2 || ix_top_ % kpool_) throw std::runtime_error("hy4f: unexpected indexer geometry");
    } else if (ctx_ > 4096) throw std::runtime_error("hy4f: no indexer: context limited to 4096 rows");
    // Engram (optional: a cut model from tools/hy4f_truncate_gguf.py has none)
    if (!g_.get("hy_v4_flash.ple.layers")) return;
    for (int64_t x : ints("ple.layers")) e_layers_.push_back(int(x));
    ngram_ = int(M("ple.ngram_size").u);
    e_heads_ = int(M("ple.heads_per_ngram").u);
    e_dim_ = int(M("embedding_length_per_layer_input").u);
    e_cols_ = (ngram_ - 1) * e_heads_;
    for (int64_t x : ints("ple.layer_multipliers")) e_mult_.push_back(uint64_t(x));
    for (int64_t x : ints("ple.head_offsets")) e_offs_.push_back(uint64_t(x));
    for (int64_t x : ints("ple.head_vocab_sizes")) e_primes_.push_back(uint64_t(x));
    e_map_ = ints("engram.token_map");
    const int64_t pad_id = int64_t(M("engram.pad_token_id").u);
    if (pad_id < 0 || pad_id >= int64_t(e_map_.size())) throw std::runtime_error("hy4f: Engram pad id out of the map");
    e_pad_ = e_map_[size_t(pad_id)];
    const size_t ne = e_layers_.size();
    if (e_mult_.size() != ne * size_t(ngram_) || e_primes_.size() != ne * size_t(e_cols_) || e_offs_.size() != e_primes_.size())
        throw std::runtime_error("hy4f: Engram metadata is inconsistent");
    for (size_t e = 0; e < ne; ++e) {
        if (e_layers_[e] >= n_layer_) throw std::runtime_error("hy4f: Engram layer out of range");
        L_[size_t(e_layers_[e])].engram = true;
        L_[size_t(e_layers_[e])].engram_idx = int(e);
    }
    const auto& tbl = T("per_layer_token_embd.weight");
    if (tbl.type != kQ8_0 || int64_t(tbl.shape[0]) != e_dim_) throw std::runtime_error("hy4f: Engram table must be Q8_0 rows");
    e_table_ = g_.tensor_data(tbl);
    e_row_bytes_ = size_t(e_dim_ / 32 * 34);
}

W Hy4::upload(const std::string& name) {
    const auto& t = T(name);
    W w;
    w.type = int(t.type);
    w.in = int64_t(t.shape[0]);
    w.out = int64_t(t.elements() / t.shape[0]);
    size_t bytes = 0;
    if (w.type == kF32) bytes = size_t(t.elements()) * 4;
    else if (w.type == kQ8_0) bytes = size_t(t.elements() / 32) * 34;
    else if (w.type == kNvfp4) bytes = size_t(t.elements() / 64) * 36;
    else throw std::runtime_error("hy4f: " + name + " has unsupported type " + t.type_name());
    ck(cudaMalloc(&w.d, std::max<size_t>(bytes, 16)), "weight alloc");
    h2d(w.d, g_.tensor_data(t), bytes);
    ck(cudaStreamSynchronize(s_), "weight upload");
    return w;
}

void Hy4::load(int threads) {
    const auto t0 = std::chrono::steady_clock::now();
    size_t dev_bytes = 0;
    auto up = [&](const std::string& n) {
        W w = upload(n);
        const auto& t = T(n);
        dev_bytes += w.type == kF32 ? size_t(t.elements()) * 4 : w.type == kQ8_0 ? size_t(t.elements() / 32) * 34 : size_t(t.elements() / 64) * 36;
        return w;
    };
    tok_embd_ = &T("token_embd.weight");
    if (tok_embd_->type != kQ8_0) throw std::runtime_error("hy4f: token_embd must be Q8_0");
    lane_ = std::make_unique<strata::ds41::ExpertLane>(n_latent_, n_ff_exp_, threads + 1, true);   // begin(): threads 1.. compute
    for (int i = 0; i < n_run_; ++i) {
        auto& l = L_[size_t(i)];
        const std::string p = "blk." + std::to_string(i) + ".";
        l.attn_norm = up(p + "attn_norm.weight");
        l.sinks = up(p + "attn_sinks.weight");
        l.q_a = up(p + "attn_q_a.weight");
        l.q_a_norm = up(p + "attn_q_a_norm.weight");
        l.q_b = up(p + "attn_q_b.weight");
        l.kv_a = up(p + "attn_kv_a_mqa.weight");
        l.kv_a_norm = up(p + "attn_kv_a_norm.weight");
        l.k_b = up(p + "attn_k_b.weight");
        l.v_b = up(p + "attn_v_b.weight");
        l.wo = up(p + "attn_output.weight");
        l.gate = up(p + "attn_gate.weight");
        l.post_attn_norm = up(p + "post_attention_norm.weight");
        l.hc_attn_fn = up(p + "hc_attn_fn.weight");
        l.hc_attn_base = up(p + "hc_attn_base.weight");
        l.hc_attn_scale = up(p + "hc_attn_scale.weight");
        l.hc_ffn_fn = up(p + "hc_ffn_fn.weight");
        l.hc_ffn_base = up(p + "hc_ffn_base.weight");
        l.hc_ffn_scale = up(p + "hc_ffn_scale.weight");
        l.ffn_norm = up(p + "ffn_norm.weight");
        l.post_ffn_norm = up(p + "post_ffw_norm.weight");
        if (l.k_b.type != kQ8_0 || l.v_b.type != kQ8_0) throw std::runtime_error("hy4f: k_b / v_b must be Q8_0");
        if (l.q_b.out != int64_t(l.heads) * (k_head_ + rope_dim_)) throw std::runtime_error("hy4f: q_b width");
        if (l.dense) {
            l.ffn_gate = up(p + "ffn_gate.weight");
            l.ffn_up = up(p + "ffn_up.weight");
            l.ffn_down = up(p + "ffn_down.weight");
        } else {
            l.gate_inp = up(p + "ffn_gate_inp.weight");
            if (l.gate_inp.type != kF32) throw std::runtime_error("hy4f: router must be F32");
            l.routed_down = up(p + "ffn_routed_down.weight");
            l.routed_up = up(p + "ffn_routed_up.weight");
            l.sh_gate = up(p + "ffn_gate_shexp.weight");
            l.sh_up = up(p + "ffn_up_shexp.weight");
            l.sh_down = up(p + "ffn_down_shexp.weight");
            l.bias_d = up(p + "exp_probs_b.bias");
            const auto& b = T(p + "exp_probs_b.bias");
            l.bias.assign((const float*) g_.tensor_data(b), (const float*) g_.tensor_data(b) + n_expert_);
            const auto& gt = T(p + "ffn_gate_exps.weight");
            const auto& ut = T(p + "ffn_up_exps.weight");
            const auto& dt = T(p + "ffn_down_exps.weight");
            std::string err;
            if (!cpu::native_fmt(int(gt.type), int(dt.type), n_latent_, n_ff_exp_, l.lane.f, err))
                throw std::runtime_error("hy4f: " + p + " experts: " + err);
            l.lane.f.swiglu_limit = clamp_;
            l.lane.f.tail_off = 0;   // GGUF NVFP4 has no per-expert global-scale tail (native_fmt assumes Strata's packs)
            l.lane.gate = g_.tensor_data(gt);
            l.lane.up = g_.tensor_data(ut);
            l.lane.down = g_.tensor_data(dt);
            l.lane.gate_stride = l.lane.f.gu_row * size_t(n_ff_exp_);
            l.lane.down_stride = l.lane.f.d_row * size_t(n_latent_);
            if (l.lane.gate_stride * size_t(n_expert_) > size_t(gt.elements()) * 36 / 64 + 1)
                throw std::runtime_error("hy4f: expert stride exceeds the tensor");
        }
        if (l.engram) {
            l.eng_kv = up(p + "engram_kv.weight");
            l.eng_kv_b = up(p + "engram_kv_b.weight");
            l.eng_gate = up(p + "engram_gate.weight");
            l.eng_gamma = up(p + "engram_pre_gamma.weight");
            l.eng_hc_scale = up(p + "engram_hc_scale.weight");
            const auto& ln = T(p + "engram_lookup_norm.weight");
            l.lookup_norm.assign((const float*) g_.tensor_data(ln), (const float*) g_.tensor_data(ln) + ln.elements());
        }
        l.kv = alloc(size_t(l.ring) * kRow);
        if (l.indexer) {
            l.ix_qb = up(p + "indexer.attn_q_b.weight");
            l.ix_k = up(p + "indexer.attn_k.weight");
            l.ix_knorm = up(p + "indexer.k_norm.weight");
            l.ix_proj = up(p + "indexer.proj.weight");
            l.ix_ape = up(p + "indexer_compressor_ape.weight");
            l.ix_gate = up(p + "indexer_compressor_gate.weight");
            l.ix_val = up(p + "indexer_compressor_value.weight");
            l.ix_cnorm = up(p + "indexer_compressor_norm.weight");
            l.vg = alloc(size_t(ctx_) * 2 * ix_dim_);
            l.pooled = alloc(size_t(ctx_ / kpool_ + 1) * ix_dim_);
        }
    }
    if (n_run_ == n_layer_) {
        out_norm_ = up("output_norm.weight");
        output_ = up("output.weight");
        hc_head_fn_ = up("output_hc_fn.weight");
        hc_head_base_ = up("output_hc_base.weight");
        hc_head_scale_ = up("output_hc_scale.weight");
        logits_d_ = alloc(size_t(vocab_));
    }
    int max_heads = 0;
    for (const auto& l : L_) max_heads = std::max(max_heads, l.heads);
    const size_t n = size_t(n_embd_);
    h_ = alloc(n * hc_);
    x_ = alloc(n);
    cur_ = alloc(n);
    flat_ = alloc(n * hc_);
    mix_ = alloc(64);
    coef_ = alloc(64);
    qr_ = alloc(size_t(q_rank_));
    q_ = alloc(size_t(max_heads) * (k_head_ + rope_dim_));
    kvpe_ = alloc(size_t(kv_rank_ + rope_dim_));
    kvn_ = alloc(size_t(kv_rank_));
    qa_ = alloc(size_t(max_heads) * kv_rank_);
    q640_ = alloc(size_t(max_heads) * kRow);
    att_ = alloc(size_t(max_heads) * kRow);
    av_ = alloc(size_t(max_heads) * v_head_);
    gt_ = alloc(size_t(max_heads) * v_head_);
    o_ = alloc(n);
    const size_t ffw = size_t(std::max(dense_ff_, n_ff_exp_));
    ff_g_ = alloc(ffw);
    ff_u_ = alloc(ffw);
    ff_a_ = alloc(ffw);
    logit_e_ = alloc(size_t(n_expert_));
    rin_ = alloc(size_t(n_latent_ + 3 * top_k_));   // one download: the latent, then hy4_route's ints
    ck(cudaMalloc(&rtable_, k::hy4_route_table_bytes(top_k_)), "route table");
    ck(cudaMalloc(&rxq_, size_t(n_latent_) / 32 * 36), "route xq");
    rout_ = alloc(size_t(n_latent_));
    sh_ = alloc(n);
    eng_r_ = alloc(size_t(std::max(1, e_cols_ * e_dim_)));
    eng_kv_ = alloc(n * (hc_ + 1));
    {   // the step block
        const size_t ints = SV_INTS * 4, cs = size_t(rope_dim_) * 3 * 4, emb = size_t(n_embd_) * 4,
                     eng = size_t(std::max<size_t>(1, e_layers_.size())) * size_t(std::max(1, e_cols_ * e_dim_)) * 4;
        step_bytes_ = ints + cs + emb + eng;
        ck(cudaMalloc((void**) &step_d_, step_bytes_), "step block");
        ck(cudaMemset(step_d_, 0, step_bytes_), "step block");
        ck(cudaHostAlloc((void**) &step_h_, step_bytes_, cudaHostAllocDefault), "step block staging");
        std::memset(step_h_, 0, step_bytes_);
        sv_ = (int*) step_d_;
        cs_ = (float*) (step_d_ + ints);
        emb_d_ = (float*) (step_d_ + ints + cs);
        eng_rows_d_ = (float*) (step_d_ + ints + cs + emb);
        logits_h_ = pinned(size_t(vocab_));
    }
    if (ix_dim_) {
        iq_ = alloc(size_t(ix_heads_) * ix_dim_);
        ik_ = alloc(size_t(ix_dim_));
        iw_ = alloc(size_t(ix_heads_));
        iscore_ = alloc(size_t(ctx_ / kpool_ + 1));
        ivg_ = alloc(size_t(2 * ix_dim_));
        cs2_ = alloc(size_t(rope_dim_));
        ck(cudaMalloc((void**) &isel_, size_t(ctx_ + kpool_) * 4), "isel");
    }
    ck(cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ev_promo_, cudaEventDisableTiming), "event");
    // + the split-output partials (STRATA_DS41_ATTN_SPLIT, set in run(): the unsplit kernel stages 8 x rows floats in LDS,
    // past 64 KB near 2048 rows)
    scores_bytes_ = k::ds41_attention_scratch(max_heads) + size_t(kAttnSplit) * max_heads * kRow * 4;
    scores_ = alloc(scores_bytes_ / 4 + 1);
    const size_t q8 = std::max({n * hc_, ffw, size_t(max_heads) * v_head_, size_t(q_rank_), size_t(e_cols_ * e_dim_)}) / 32 * 36;
    ck(cudaMalloc(&q8_, q8), "q8");
    ck(cudaMalloc(&q8b_, q8), "q8b");
    h_logit_e_ = pinned(size_t(n_expert_));
    h_rin_ = pinned(size_t(n_latent_ + 3 * top_k_));
    h_route_ = reinterpret_cast<int*>(h_rin_ + n_latent_);
    h_rout_ = pinned(size_t(n_latent_));
    dumps.assign(size_t(n_run_), {});
    if (fused) {   // warm-up: every fused kernel once on scratch, then a sync (see run())
        const auto& l0 = L_[0];
        k::hy4_hc_mix(h_, l0.hc_attn_fn.d, 2 * hc_, (const float*) l0.hc_attn_scale.d, (const float*) l0.hc_attn_base.d, coef_, hc_mag_,
                      hc_eps_, eps_, (const float*) l0.attn_norm.d, x_, cur_, q8_, n_embd_, hc_, s_);
        k::hy4_norm_hc_post(o_, nullptr, (const float*) l0.attn_norm.d, coef_ + hc_, flat_, n_embd_, hc_, eps_, s_);
        k::hy4_mla_finish(q_, l0.heads, k_head_ + rope_dim_, k_head_, kvpe_, (const float*) l0.kv_a_norm.d, cs_, q640_, att_, sv_ + SV_SLOT_SWA,
                          eps_, s_);
        k::hy4_mul_sigmoid_q8(av_, gt_, q8b_, 32, s_);
        ck(cudaStreamSynchronize(s_), "warm-up");
        ck(cudaMemsetAsync(cur_, 0, size_t(n_embd_) * 4, s_), "reset");
        ck(cudaMemsetAsync(coef_, 0, 64 * 4, s_), "reset");
        ck(cudaStreamSynchronize(s_), "warm-up");
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "hy4f: %d of %d blocks, %.2f GiB on the GPU, %d lane threads, ctx %d, loaded in %.1f s\n", n_run_, n_layer_,
                 double(dev_bytes) / double(1u << 30), threads, ctx_, s);
}


void Hy4::tier_setup(double gib, double reserve_gib, int budget) {
    if (gib == 0) return;
    Tier& t = tier;
    t.L = k::native_expert_layout(kNvfp4, kNvfp4, n_latent_, n_ff_exp_);
    t.L.swiglu_limit = clamp_;   // L.tail_off: the GPU kernels always apply the {s_gate, s_up, s_down} tail; slots carry 1.0s
    if (!k::native_expert_supported(kNvfp4, kNvfp4, n_latent_, n_ff_exp_)) throw std::runtime_error("hy4f: no GPU kernels for NVFP4 experts");
    const auto* any = &L_[size_t(n_layer_ - 1)].lane;
    t.gate_bytes = any->f.gu_row * size_t(n_ff_exp_);
    t.down_bytes = any->f.d_row * size_t(n_latent_);
    t.blob = t.L.bytes;
    if (t.L.up_off != t.gate_bytes || t.L.down_off != 2 * t.gate_bytes || t.L.tail_off != 2 * t.gate_bytes + t.down_bytes ||
        t.blob != t.L.tail_off + 16)
        throw std::runtime_error("hy4f: GPU expert blob layout does not match the GGUF rows");
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "mem info");
    // default (gib < 0): free VRAM - reserve, at most 4 GiB (the card is shared: trunk ~4.2 GB + 4 = ~8 GB in all)
    const double want = gib > 0 ? gib * double(1u << 30)
                                : std::min(double(free_b) - reserve_gib * double(1u << 30), 4.0 * double(1u << 30));
    t.slots = want > 0 ? int(want / double(t.blob)) : 0;
    if (t.slots <= 0) { t.slots = 0; return; }
    ck(cudaMalloc(&t.arena, size_t(t.slots) * t.blob), "tier arena");
    t.experts = n_expert_;
    t.slot_of.assign(size_t(n_layer_) * n_expert_, -1);
    t.score.assign(size_t(n_layer_) * n_expert_, 0.0f);
    t.owner.assign(size_t(t.slots), -1);
    t.ready.assign(size_t(t.slots), 0);
    t.slot_ev.resize(size_t(t.slots));
    for (auto& e : t.slot_ev) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "slot event");
    ck(cudaStreamCreateWithFlags(&t.copy, cudaStreamNonBlocking), "copy stream");
    for (int b = 0; b < Tier::kStage; ++b) {
        ck(cudaHostAlloc((void**) &t.stage[b], t.blob, cudaHostAllocPortable), "tier staging");
        ck(cudaEventCreateWithFlags(&t.stage_ev[b], cudaEventDisableTiming), "stage event");
    }
    ck(cudaMalloc(&t.table_d, 1024), "tier tables");
    for (int b = 0; b < 2; ++b) {
        ck(cudaHostAlloc((void**) &t.table_h[b], 512, cudaHostAllocPortable), "tier table staging");
        ck(cudaEventCreateWithFlags(&t.table_ev[b], cudaEventDisableTiming), "table event");
    }
    ck(cudaMalloc(&t.scratch, k::native_expert_scratch_bytes(16, n_ff_exp_)), "tier scratch");
    ck(cudaMalloc((void**) &t.out, size_t(16) * n_latent_ * 4), "tier out");
    ck(cudaMalloc(&t.xq, size_t(n_latent_) / 32 * 36), "tier xq");
    ck(cudaMalloc((void**) &t.sum, size_t(n_latent_) * 4), "tier sum");
    t.budget = budget;
    ck(cudaMalloc((void**) &t.d_slots, size_t(n_layer_) * n_expert_ * 8), "tier slot table");
    ck(cudaMemset(t.d_slots, 0, size_t(n_layer_) * n_expert_ * 8), "tier slot table");
    ck(cudaHostAlloc((void**) &t.addr_h, size_t(t.slots) * 8, cudaHostAllocDefault), "tier addresses");
    for (int i = 0; i < t.slots; ++i) t.addr_h[i] = reinterpret_cast<unsigned long long>(t.arena + size_t(i) * t.blob);
    std::fprintf(stderr, "hy4f: GPU expert tier %d slots (%.2f GiB, %.1f%% of the routed experts), %d uploads/token\n", t.slots,
                 double(t.slots) * double(t.blob) / double(1u << 30), 100.0 * t.slots / double(n_expert_ * (n_layer_ - 1)), budget);
}

void Hy4::gemv_q8(const W& w, const void* q8, float* y) {
    k::iq_mmvq(w.type, w.d, q8, y, int(w.in), int(w.out), 1, s_);
}

void Hy4::gemv(const W& w, const float* x, float* y) {
    if (w.type == kF32) {
        k::ds41_f32_gemv((const float*) w.d, x, y, int(w.out), int(w.in), s_);
        return;
    }
    k::quantize_q8_1_rows(x, 1, w.in, q8b_, s_);
    k::iq_mmvq(w.type, w.d, q8b_, y, int(w.in), int(w.out), 1, s_);
}

void Hy4::rms(const float* x, const W* w, float* y, int n, void* q8) {
    if (q8) k::glm_rms_norm_rows_q8(x, w ? (const float*) w->d : nullptr, y, q8, n, 1, eps_, s_);
    else k::glm_rms_norm_rows(x, w ? (const float*) w->d : nullptr, y, n, 1, eps_, s_);
}

// coef_ = pre[hc] (| post[hc]), x_ = sum_s pre_s h_s
void Hy4::hc_mix(const W& fn, const W& scale, const W& base, bool post, const W& norm) {
    if (fused && fn.type == kNvfp4) {
        k::hy4_hc_mix(h_, fn.d, post ? 2 * hc_ : hc_, (const float*) scale.d, (const float*) base.d, coef_, hc_mag_, hc_eps_, eps_,
                      (const float*) norm.d, x_, cur_, q8_, n_embd_, hc_, s_);
        return;
    }
    hc_pre(fn, scale, base, post);
    rms(x_, &norm, cur_, n_embd_, q8_);
}

void Hy4::post(const W& norm, const float* add) {   // h_s += rms(o_ + add) * norm * post_s
    if (fused) {
        k::hy4_norm_hc_post(o_, add, (const float*) norm.d, coef_ + hc_, h_, n_embd_, hc_, eps_, s_);
        return;
    }
    if (add) k::ds41_add(o_, add, o_, n_embd_, s_);
    rms(o_, &norm, o_, n_embd_);
    k::hy4_hc_post(o_, coef_ + hc_, h_, n_embd_, hc_, s_);
}

void Hy4::hc_pre(const W& fn, const W& scale, const W& base, bool post) {
    const int n = n_embd_ * hc_;
    rms(h_, nullptr, flat_, n, q8_);
    gemv_q8(fn, q8_, mix_);
    k::hy4_hc_coeff(mix_, (const float*) scale.d, (const float*) base.d, coef_, hc_, post, hc_mag_, hc_eps_, s_);
    k::hy4_hc_pre(h_, coef_, x_, n_embd_, hc_, s_);
}

bool Hy4::engram_rows(int e, int tok, int pos, float* rows) {
    const Layer& l = L_[size_t(e_layers_[size_t(e)])];
    const int64_t dead = -1;
    auto cid = [&](int t) -> int64_t { return t < 0 || t >= int(e_map_.size()) ? dead : e_map_[size_t(t)]; };
    std::vector<uint64_t> toks(static_cast<size_t>(ngram_));
    int64_t c = cid(tok);
    const bool is_dead = c == dead;
    toks[0] = uint64_t(is_dead ? e_pad_ : c);
    for (int s = 1; s < ngram_; ++s) {
        c = pos - s < 0 ? dead : cid(history_[size_t(pos - s)]);
        toks[size_t(s)] = uint64_t(c == dead ? e_pad_ : c);
    }
    const uint64_t* mult = e_mult_.data() + size_t(e) * ngram_;
    const uint64_t* primes = e_primes_.data() + size_t(e) * e_cols_;
    const uint64_t* offs = e_offs_.data() + size_t(e) * e_cols_;
    uint64_t rolling = toks[0] * mult[0];
    for (int o = 1; o < ngram_; ++o) {
        rolling ^= toks[size_t(o)] * mult[o];
        for (int hh = 0; hh < e_heads_; ++hh) {
            const int col = (o - 1) * e_heads_ + hh;
            const uint64_t id = rolling % primes[col] + offs[col];
            float* r = rows + size_t(col) * e_dim_;
            dequant_q8_0(e_table_ + id * e_row_bytes_, e_dim_, r);
            // ggml_rms_norm: float sum, then 1 / sqrt(mean + eps); FLT_EPSILON as the reference's unset eps
            float sum = 0;
            for (int j = 0; j < e_dim_; ++j) sum += r[j] * r[j];
            const float scale = 1.0f / std::sqrt(sum / float(e_dim_) + 1.1920929e-07f);
            const float valid = is_dead ? 0.0f : 1.0f;
            for (int j = 0; j < e_dim_; ++j) r[j] = r[j] * scale * l.lookup_norm[size_t(col) * e_dim_ + j] * valid;
        }
    }
    return !is_dead;
}

void Hy4::engram(Layer& l, int, int) {
    const float* rows = eng_rows_d_ + size_t(l.engram_idx) * e_cols_ * e_dim_;
    gemv(l.eng_kv, rows, eng_kv_);
    k::ds41_add(eng_kv_, (const float*) l.eng_kv_b.d, eng_kv_, n_embd_ * (hc_ + 1), s_);
    k::hy4_engram(h_, eng_kv_, (const float*) l.eng_gate.d, (const float*) l.eng_gamma.d, (const float*) l.eng_hc_scale.d, n_embd_, hc_,
                  1.1920929e-07f, true, s_, sv_ + SV_ANY);
}

void Hy4::prepare_step(int tok, int pos) {
    int* iv = (int*) step_h_;
    float* cs = (float*) (step_h_ + SV_INTS * 4);
    float* emb = cs + rope_dim_ * 3;
    float* eng = emb + n_embd_;
    iv[SV_POS] = pos;
    const int full_ring = ctx_, swa_ring = window_;
    const int rf = std::min(pos + 1, full_ring), rs = std::min(pos + 1, swa_ring);
    iv[SV_INFO_FULL + 0] = ((pos - rf + 1) % full_ring + full_ring) % full_ring;
    iv[SV_INFO_FULL + 1] = rf;
    iv[SV_INFO_FULL + 2] = 0;
    iv[SV_INFO_SWA + 0] = ((pos - rs + 1) % swa_ring + swa_ring) % swa_ring;
    iv[SV_INFO_SWA + 1] = rs;
    iv[SV_INFO_SWA + 2] = 0;
    iv[SV_SLOT_FULL] = pos % full_ring;
    iv[SV_SLOT_SWA] = pos % swa_ring;
    // rope tables: this position at the full and window bases, and the pool's first member (full base)
    const int pool_pos = kpool_ ? pos / kpool_ * kpool_ : 0;
    for (int t = 0; t < 3; ++t) {
        const float base = t == 1 ? rope_swa_ : rope_full_;
        const float theta_scale = std::pow(base, -2.0f / float(rope_dim_));
        float theta = float(t == 2 ? pool_pos : pos);
        for (int i = 0; i < rope_dim_; i += 2) {
            cs[t * rope_dim_ + i] = std::cos(theta);
            cs[t * rope_dim_ + i + 1] = std::sin(theta);
            theta *= theta_scale;
        }
    }
    dequant_q8_0(g_.tensor_data(*tok_embd_) + size_t(tok) * size_t(n_embd_ / 32 * 34), n_embd_, emb);
    bool any = true;
    for (size_t e = 0; e < e_layers_.size(); ++e)
        if (e_layers_[e] < n_run_) any = engram_rows(int(e), tok, pos, eng + e * size_t(e_cols_ * e_dim_));
    iv[SV_ANY] = any ? 1 : 0;
    ck(cudaMemcpyAsync(step_d_, step_h_, step_bytes_, cudaMemcpyHostToDevice, s_), "step block");
}

void Hy4::attention(Layer& l, int pos) {
    const int H = l.heads, qk = k_head_ + rope_dim_;
    // cur_ normalized, its q8_1 in q8_
    gemv_q8(l.q_a, q8_, qr_);
    gemv_q8(l.kv_a, q8_, kvpe_);
    gemv_q8(l.gate, q8_, gt_);
    rms(qr_, &l.q_a_norm, qr_, q_rank_, q8b_);
    gemv_q8(l.q_b, q8b_, q_);
    if (l.indexer) {   // the inputs are live now: qr's q8_1 (q8b_) and cur's (q8_)
        gemv_q8(l.ix_qb, q8b_, iq_);
        gemv_q8(l.ix_k, q8_, ik_);
        gemv_q8(l.ix_proj, q8_, iw_);
    }
    const float* cs = cs_ + (l.swa ? rope_dim_ : 0);
    const int slot = pos % l.ring;
    if (fused) {
        // absorb straight into the 640-wide rows; then rope tails, the normed latent and the cache row in one launch
        k::hy4_q8_0_heads(l.k_b.d, q_, qk, q640_, kRow, kv_rank_, k_head_, H, s_);
        k::hy4_mla_finish(q_, H, qk, k_head_, kvpe_, (const float*) l.kv_a_norm.d, cs, q640_, l.kv, sv_ + (l.swa ? SV_SLOT_SWA : SV_SLOT_FULL),
                          eps_, s_);
    } else {
        rms(kvpe_, &l.kv_a_norm, kvn_, kv_rank_);
        // rope (NORM: adjacent pairs) on q_pe (after the 128 nope values of each head) and k_pe; tables from step()
        k::ds41_rope(q_, H, qk, k_head_, rope_dim_ / 2, cs, s_);
        k::ds41_rope(kvpe_, 1, kv_rank_ + rope_dim_, kv_rank_, rope_dim_ / 2, cs, s_);
        // absorb: qa[h] = W_kb[h] q_nope[h]  (W_kb[h]: [512][128])
        k::hy4_q8_0_heads(l.k_b.d, q_, qk, qa_, kv_rank_, kv_rank_, k_head_, H, s_);
        k::hy4_mla_pack_q(qa_, q_ + k_head_, qk, q640_, H, s_);
        k::hy4_mla_pack_k(kvn_, kvpe_ + kv_rank_, l.kv + size_t(slot) * kRow, s_);
    }
    const int rows = std::min(pos + 1, l.ring);
    const int first = ((pos - rows + 1) % l.ring + l.ring) % l.ring;
    const float scale = 1.0f / std::sqrt(float(qk));
    int nsel = 0;
    if (l.indexer) indexer(l, pos, nsel);
    if (nsel > 0)   // the selected rows only (the tail and the top pools); positions are the full layer's slots
        k::ds41_attention(q640_, H, kRow, l.kv, 0, 1, 0, l.kv, kRow, isel_, nsel, (const float*) l.sinks.d, scale, scores_, att_, s_);
    else {   // the window from the step block (graph-capturable): {first slot, rows, 0}
        (void) first;
        (void) rows;
        const int bound = l.swa ? l.ring : std::min(l.ring, ix_top_ ? ix_top_ : 4096);
        k::ds41_attention_multi(q640_, 1, H, kRow, l.kv, l.ring, sv_ + (l.swa ? SV_INFO_SWA : SV_INFO_FULL), bound, nullptr, 0, nullptr, 0,
                                nullptr, (const float*) l.sinks.d, scale, scores_, att_, s_, scores_bytes_);
    }
    // un-absorb: av[h] = W_vb[h] att[h][0..512)  (W_vb[h]: [256][512])
    k::hy4_q8_0_heads(l.v_b.d, att_, kRow, av_, v_head_, v_head_, kv_rank_, H, s_);
    if (fused) {
        k::hy4_mul_sigmoid_q8(av_, gt_, q8b_, H * v_head_, s_);
        gemv_q8(l.wo, q8b_, o_);
    } else {
        k::hy4_mul_sigmoid(av_, gt_, av_, H * v_head_, s_);
        gemv(l.wo, av_, o_);
    }   // the post norm is in post()
}


// DSA kpool indexer of a full layer at `pos` (llama.cpp build_kpool_select, select-tail): this token's [value | gate]
// row, the pooled key of a group it completes, and - when more complete groups precede its own than the top-k keeps -
// the selection: the top (top_k / kpool - 1) groups by score plus the token's own group (the tail). nsel = 0: every
// row is selected (exact full attention).
void Hy4::indexer(Layer& l, int pos, int& nsel) {
    const int D = ix_dim_, half = D - rope_dim_;
    rms(ik_, &l.ix_knorm, ik_, D);
    gemv(l.ix_val, ik_, ivg_);
    gemv(l.ix_gate, ik_, ivg_ + D);
    k::hy4_copy_row(ivg_, l.vg, sv_ + SV_POS, 2 * D, s_);
    // the pooled key of a group this token completes, rotated at its first member's position (the third rope table)
    k::hy4_kpool_step(l.vg, (const float*) l.ix_ape.d, (const float*) l.ix_cnorm.d, cs_ + 2 * rope_dim_, l.pooled, sv_ + SV_POS, kpool_, D,
                      rope_dim_ / 2, eps_, s_);
    const int visible = pos / kpool_;                  // complete groups before the token's own
    const int top = ix_top_ / kpool_ - 1;
    nsel = 0;
    if (visible <= top) return;   // every row selected (never reached in a graph step: pos < 2048)
    k::ds41_rope(iq_, ix_heads_, D, half, rope_dim_ / 2, cs_, s_);
    k::ds41_index_score(iq_, ix_heads_, D, l.pooled, D, visible, iw_, 1.0f / std::sqrt(float(D * ix_heads_)), iscore_, s_);
    h_iscore_.resize(size_t(visible));
    d2h(h_iscore_.data(), iscore_, size_t(visible) * 4);
    std::vector<int> order(static_cast<size_t>(visible));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
        return h_iscore_[size_t(a)] > h_iscore_[size_t(b)] || (h_iscore_[size_t(a)] == h_iscore_[size_t(b)] && a < b);
    });
    h_isel_.clear();
    for (int i = 0; i < top; ++i)
        for (int m = 0; m < kpool_; ++m) h_isel_.push_back(order[size_t(i)] * kpool_ + m);
    for (int p = pos - pos % kpool_; p <= pos; ++p) h_isel_.push_back(p);
    nsel = int(h_isel_.size());
    h2d(isel_, h_isel_.data(), h_isel_.size() * 4);
    ++sparse_calls;
}

void Hy4::ffn(Layer& l) {   // the dense FFN (leading blocks); cur_ normalized, its q8_1 in q8_
    gemv_q8(l.ffn_gate, q8_, ff_g_);
    gemv_q8(l.ffn_up, q8_, ff_u_);
    if (fused) {
        k::native_swiglu_quantize_q8_1(ff_g_, ff_u_, q8b_, dense_ff_, 1, s_);
        gemv_q8(l.ffn_down, q8b_, o_);
    } else {
        k::glm_swiglu(ff_g_, ff_u_, ff_a_, dense_ff_, 0.0f, s_);
        gemv(l.ffn_down, ff_a_, o_);
    }
    post(l.post_ffn_norm, nullptr);
}

// MoE, in four parts around the CPU lane's host wait. route (device): router logits, routed_down, hy4_route, one download.
void Hy4::moe_route(Layer& l) {
    k::ds41_f32_gemv((const float*) l.gate_inp.d, cur_, logit_e_, n_expert_, n_embd_, s_);
    gemv_q8(l.routed_down, q8_, rin_);
    const unsigned long long* slots = tier.on() ? tier.d_slots + size_t(cur_layer_) * n_expert_ : nullptr;
    int* route_d = reinterpret_cast<int*>(rin_ + n_latent_);
    k::hy4_route(logit_e_, (const float*) l.bias_d.d, slots, n_expert_, top_k_, w_scale_, w_norm_, rtable_, route_d, s_);
    ck(cudaMemcpyAsync(h_rin_, rin_, size_t(n_latent_ + 3 * top_k_) * 4, cudaMemcpyDeviceToHost, s_), "latent + routes down");
}

// side (device, while the lane runs): the GPU tier's routes from hy4_route's table, and the shared expert
void Hy4::moe_side(Layer& l) {
    if (tier.on()) {
        auto* tb = (const uint8_t*) rtable_;
        const size_t K = size_t(top_k_), o_start = K * 8, o_ng = o_start + (K + 1) * 4, o_dst = o_ng + 4, o_tok = o_dst + K * 4,
                     o_w = o_tok + K * 4;
        k::quantize_q8_1_rows(rin_, 1, n_latent_, rxq_, s_);
        k::native_expert_grouped(tier.L, reinterpret_cast<const unsigned long long*>(tb), reinterpret_cast<const int32_t*>(tb + o_start),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), reinterpret_cast<const int32_t*>(tb + o_dst),
                                 reinterpret_cast<const int32_t*>(tb + o_tok), top_k_, top_k_, rxq_, tier.scratch, tier.out, s_);
        ck(cudaMemsetAsync(tier.sum, 0, size_t(n_latent_) * 4, s_), "tier sum");
        k::ds41_scatter_axpy_dev(tier.out, reinterpret_cast<const int32_t*>(tb + o_tok), reinterpret_cast<const float*>(tb + o_w),
                                 reinterpret_cast<const int32_t*>(tb + o_ng), n_latent_, tier.sum, s_);
    }
    gemv_q8(l.sh_gate, q8_, ff_g_);
    gemv_q8(l.sh_up, q8_, ff_u_);
    if (fused) {
        k::native_swiglu_quantize_q8_1(ff_g_, ff_u_, q8b_, n_ff_exp_, 1, s_);
        gemv_q8(l.sh_down, q8b_, sh_);
    } else {
        k::glm_swiglu(ff_g_, ff_u_, ff_a_, n_ff_exp_, 0.0f, s_);
        gemv(l.sh_down, ff_a_, sh_);
    }
}

// host: wait for the routes (ev_, recorded after moe_route), the CPU lane, tier bookkeeping and uploads
void Hy4::moe_host(Layer& l) {
    const auto tr = clk::now();
    ck(cudaEventSynchronize(ev_), "routed wait");
    const auto t0 = clk::now();
    if (prof) prof_route += ms(tr, t0);
    const int32_t* cpu_ids = h_route_;
    const float* w = reinterpret_cast<const float*>(h_route_ + top_k_);
    const int32_t* ids = h_route_ + 2 * top_k_;
    std::vector<int> misses;
    if (tier.on()) {
        for (int i = 0; i < top_k_; ++i) {
            const int key = cur_layer_ * n_expert_ + ids[i];
            tier.score[size_t(key)] += 1.0f;
            if (cpu_ids[i] < 0) ++tier.hits;
            else if (tier.slot_of[size_t(key)] < 0) misses.push_back(key);
        }
        tier.routes += top_k_;
    }
    lane_->begin(l.lane, h_rin_, 1, cpu_ids, w, top_k_, h_rout_);
    if (!misses.empty()) {   // uploads while the lane computes
        std::sort(misses.begin(), misses.end(), [&](int a, int b) { return tier.score[size_t(a)] > tier.score[size_t(b)]; });
        int fills = 0;
        for (int key : misses) {
            const bool free_slot = tier.used < tier.slots;
            if (!free_slot && tier.left <= 0) break;
            if (free_slot && fills >= 4) break;
            const int e = key % n_expert_;
            if (tier.promote(key, l.lane.gate + size_t(e) * l.lane.gate_stride, l.lane.up + size_t(e) * l.lane.gate_stride,
                             l.lane.down + size_t(e) * l.lane.down_stride, s_, ev_promo_, false)) {
                if (free_slot) ++fills;
                else --tier.left;
            }
        }
    }
    lane_->finish();
    const auto te = clk::now();
    if (prof) prof_experts += ms(t0, te);
    expert_ms += ms(t0, te);
    dbg_dev("ffn_moe_logits", cur_layer_, logit_e_, size_t(n_expert_));
}

// tail (device): the lane's sum up, + the tier's, routed_up, the post norm with the shared expert, the residual
void Hy4::moe_tail(Layer& l) {
    ck(cudaMemcpyAsync(rout_, h_rout_, size_t(n_latent_) * 4, cudaMemcpyHostToDevice, s_), "latent up");
    if (tier.on()) k::ds41_add(rout_, tier.sum, rout_, n_latent_, s_);
    gemv(l.routed_up, rout_, o_);
    post(l.post_ffn_norm, sh_);
}

template <class F> void Hy4::segment(F&& f) {
    const size_t id = size_t(seg_++);
    if (!graph_step_) { f(); return; }
    if (id >= execs_.size()) execs_.resize(id + 1, nullptr);
    if (!execs_[id]) {
        cudaGraph_t g = nullptr;
        ck(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal), "capture begin");
        f();
        ck(cudaStreamEndCapture(s_, &g), "capture end");
        ck(cudaGraphInstantiate(&execs_[id], g, 0), "graph instantiate");
        ck(cudaGraphDestroy(g), "graph destroy");
        ++graph_captures;
    }
    ck(cudaGraphLaunch(execs_[id], s_), "graph launch");
    ++graph_launches;
}

const std::vector<float>& Hy4::step(int tok, int pos) {
    if (pos >= ctx_) throw std::runtime_error("hy4f: context full");
    if (int(history_.size()) != pos) history_.resize(size_t(pos));
    history_.push_back(tok);
    const auto t0 = std::chrono::steady_clock::now();
    if (tier.on()) {
        for (auto& x : tier.score) x *= 0.98f;
        tier.left = tier.budget;
    }
    // graphs: fused kernels only, no dumps / profiling syncs, and no indexer selection (its top-k is on the host)
    graph_step_ = graphs && fused && !want_dump && !prof && (ix_top_ == 0 || pos < ix_top_);
    seg_ = 0;
    prepare_step(tok, pos);
    const int n = n_embd_;
    auto dump = [&](int i) {
        if (!want_dump) return;
        dumps[size_t(i)].resize(size_t(n) * hc_);
        d2h(dumps[size_t(i)].data(), h_, dumps[size_t(i)].size() * 4);
    };
    auto block = [&](int i) {   // up to the FFN input
        auto& l = L_[size_t(i)];
        cur_layer_ = i;
        if (l.engram) engram(l, tok, pos);
        hc_mix(l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, true, l.attn_norm);
        dbg_dev("attn_norm", i, cur_, size_t(n));
        auto ta = clk::now();
        attention(l, pos);
        if (prof) { ck(cudaStreamSynchronize(s_), "prof"); prof_attn += ms(ta, clk::now()); }
        post(l.post_attn_norm, nullptr);
        hc_mix(l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, true, l.ffn_norm);
    };
    int prev = -1, i = 0;
    while (true) {
        int j = i;   // the next MoE block (the end of this segment), or n_run_
        while (j < n_run_ && L_[size_t(j)].dense) ++j;
        const bool last = j >= n_run_;
        if (tier.on()) tier.publish(s_);   // before the segment's routes (outside any graph)
        segment([&] {
            if (prev >= 0) {
                moe_tail(L_[size_t(prev)]);
                dump(prev);
            } else {
                k::ds41_broadcast_streams(emb_d_, h_, n, s_);
            }
            for (int b = i; b < std::min(j + 1, n_run_); ++b) {
                block(b);
                if (L_[size_t(b)].dense) {
                    ffn(L_[size_t(b)]);
                    dump(b);
                } else {
                    moe_route(L_[size_t(b)]);
                }
            }
            if (last && n_run_ == n_layer_) {
                hc_mix(hc_head_fn_, hc_head_scale_, hc_head_base_, false, out_norm_);
                gemv_q8(output_, q8_, logits_d_);
                ck(cudaMemcpyAsync(logits_h_, logits_d_, size_t(vocab_) * 4, cudaMemcpyDeviceToHost, s_), "logits");
            }
        });
        if (last) break;
        ck(cudaEventRecord(ev_, s_), "route event");
        cur_layer_ = j;
        segment([&] { moe_side(L_[size_t(j)]); });
        moe_host(L_[size_t(j)]);
        prev = j;
        i = j + 1;
    }
    ck(cudaStreamSynchronize(s_), "step");
    if (n_run_ == n_layer_) logits_.assign(logits_h_, logits_h_ + vocab_);
    gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return logits_;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back(std::stoi(item));
    return out;
}

void write_bin(const std::string& file, const std::vector<float>& v, const int64_t ne[4]) {
    std::ofstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + file);
    const int32_t type = 0;
    f.write((const char*) &type, 4);
    for (int i = 0; i < 4; ++i) f.write((const char*) &ne[i], 8);
    f.write((const char*) v.data(), std::streamsize(v.size() * 4));
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata-hy4f MODEL.gguf --ids 1,2,3 [--n 32] [--layers L] [--dump DIR] [--threads 15] [--ctx 2048]\n");
        return 2;
    }
    std::string model = argv[1], dump;
    std::vector<int> ids;
    int n_gen = 16, layers = 0, threads = 15, ctx = 2048, budget = 8;
    double tier_gib = -1, reserve_gib = 3.0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--ids") ids = parse_ids(next());
        else if (a == "--n") n_gen = std::stoi(next());
        else if (a == "--layers") layers = std::stoi(next());
        else if (a == "--dump") dump = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--tier-gib") tier_gib = std::stod(next());          // 0: no GPU tier; -1: free VRAM - reserve
        else if (a == "--tier-reserve-gib") reserve_gib = std::stod(next());
        else if (a == "--tier-uploads") budget = std::stoi(next());          // replacements per token
        else throw std::runtime_error("unknown argument " + a);
    }
    setenv("STRATA_DS41_ATTN_SPLIT", std::to_string(kAttnSplit).c_str(), 0);
    // a fused kernel's first launch once produced nothing (position 0 all zeros; not reproduced in 6 runs since):
    // load every code object up front, and warm the kernels up in load()
    setenv("HIP_ENABLE_DEFERRED_LOADING", "0", 0);
    if (ids.empty()) throw std::runtime_error("--ids is required (token ids, e.g. from llama-tokenize)");
    Hy4 m(model, threads, ctx, layers);
    m.tier_setup(tier_gib, reserve_gib, budget);
    m.graphs = !(std::getenv("STRATA_HY4F_GRAPH") && std::string(std::getenv("STRATA_HY4F_GRAPH")) == "0");
    m.fused = !(std::getenv("STRATA_HY4F_UNFUSED") && std::string(std::getenv("STRATA_HY4F_UNFUSED")) == "1");
    m.prof = std::getenv("STRATA_HY4F_PROF") && std::string(std::getenv("STRATA_HY4F_PROF")) == "1";
    const bool partial = m.n_run() < m.n_layer();
    m.want_dump = !dump.empty();
    std::vector<std::vector<float>> all(static_cast<size_t>(m.n_run()));
    std::vector<float> all_logits;
    auto t0 = std::chrono::steady_clock::now();
    int next = -1;
    for (size_t p = 0; p < ids.size(); ++p) {
        const auto& lg = m.step(ids[p], int(p));
        if (m.want_dump)
            for (int i = 0; i < m.n_run(); ++i) all[size_t(i)].insert(all[size_t(i)].end(), m.dumps[size_t(i)].begin(), m.dumps[size_t(i)].end());
        if (!lg.empty()) {
            if (m.want_dump) all_logits.insert(all_logits.end(), lg.begin(), lg.end());
            next = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
        }
    }
    const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "hy4f: prompt %zu tokens in %.2f s (%.1f ms/token, lane %.1f ms/token)\n", ids.size(), prompt_s,
                 prompt_s * 1e3 / double(ids.size()), m.expert_ms / double(ids.size()));
    if (!dump.empty()) {
        std::filesystem::create_directories(dump);
        for (int i = 0; i < m.n_run(); ++i) {
            const int64_t ne[4] = {3072, 4, int64_t(ids.size()), 1};
            write_bin(dump + "/l_out-" + std::to_string(i) + ".bin", all[size_t(i)], ne);
        }
        for (const auto& [name, v] : m.dbg) {
            const int64_t ne[4] = {int64_t(v.size() / ids.size()), int64_t(ids.size()), 1, 1};
            write_bin(dump + "/" + name + ".bin", v, ne);
        }
        if (!all_logits.empty()) {
            const int64_t ne[4] = {m.vocab(), int64_t(ids.size()), 1, 1};
            write_bin(dump + "/result_output.bin", all_logits, ne);
        }
    }
    if (partial || next < 0) return 0;
    std::vector<int> out;
    m.expert_ms = 0;
    m.tier.hits = m.tier.routes = m.tier.promotions = 0;
    m.prof_attn = m.prof_route = m.prof_experts = m.prof_post = 0;
    m.gpu_ms = 0;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_gen; ++i) {
        out.push_back(next);
        const auto& lg = m.step(next, int(ids.size() + size_t(i)));
        next = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s\n", join_ids(out).c_str());
    std::fprintf(stderr, "hy4f: decode %d tokens in %.2f s: %.2f tok/s (experts %.1f ms/token; tier hit %.1f%%, %ld uploads)\n", n_gen,
                 gen_s, double(n_gen) / gen_s, m.expert_ms / std::max(1, n_gen),
                 m.tier.routes ? 100.0 * double(m.tier.hits) / double(m.tier.routes) : 0.0, m.tier.promotions);
    std::fprintf(stderr, "hy4f: graphs %ld captured, %ld launches\n", m.graph_captures, m.graph_launches);
    if (m.prof)
        std::fprintf(stderr, "hy4f prof (ms/token): attention %.2f, to router %.2f, experts %.2f, after experts %.2f, other %.2f\n",
                     m.prof_attn / n_gen, m.prof_route / n_gen, m.prof_experts / n_gen, m.prof_post / n_gen,
                     gen_s * 1e3 / n_gen - (m.prof_attn + m.prof_route + m.prof_experts + m.prof_post) / n_gen);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-hy4f: %s\n", e.what());
        return 1;
    }
}
