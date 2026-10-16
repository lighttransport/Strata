// src/muse/gpu_trunk.cu - the Muse Spark block trunk on one GPU (strata/muse/gpu_trunk.hpp).  One stream; the host
// waits only where it needs the device's results.  With TrunkConfig::graphs each layer's two halves and the head are
// captured once and replayed: positions are read from device memory, and every host transfer goes through a fixed
// pinned buffer, so a captured graph is valid for every token.
#include "strata/muse/gpu_trunk.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/muse.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::muse {
namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("muse gpu trunk: ") + what + ": " + cudaGetErrorString(e));
}

struct DevTensor {
    void* p = nullptr;
    int type = 0;
    int64_t ne0 = 0, ne1 = 1;
};
}  // namespace

struct GpuTrunk::Impl {
    TrunkConfig c;
    cudaStream_t s = nullptr;
    struct Layer {
        bool swa = true;
        DevTensor attn_norm, attn_q, attn_k, attn_v, attn_gate, attn_output, attn_q_norm, attn_k_norm, attn_alpha,
            attn_beta, ffn_norm, post_ffw_norm, ffn_alpha, ffn_beta, router, router_bias, latent_down, latent_up,
            pre_exp_norm;
        uint16_t *kc = nullptr, *vc = nullptr;
        int ring = 0;
    };
    std::vector<Layer> layers;
    DevTensor out_norm, out;
    float *x, *h, *q, *kk, *v, *g, *a, *o, *z, *logits, *w, *y, *f, *t_attn, *t_ffn_inp, *head_logits;
    int32_t *ids, *pos;
    float* attn_part = nullptr;   // split-K partials
    void* q8;
    size_t bytes = 0;
    std::vector<void*> allocs;
    // pinned host staging: the stream in, the router's results out
    float *hx = nullptr, *hy = nullptr, *hz = nullptr, *hw = nullptr;
    int32_t *hids = nullptr, *hpos = nullptr;
    std::vector<void*> pinned;
    std::vector<cudaGraphExec_t> g_attn, g_finish;
    cudaGraphExec_t g_head = nullptr;

    template <class T> T* host_buf(size_t n) {
        void* p = nullptr;
        ck(cudaMallocHost(&p, n * sizeof(T)), "cudaMallocHost");
        pinned.push_back(p);
        return (T*) p;
    }
    // enqueue directly, or capture the work once into `exec` and replay it
    void run(cudaGraphExec_t& exec, const std::function<void()>& enqueue) {
        if (!c.graphs) { enqueue(); return; }
        if (!exec) {
            cudaGraph_t graph = nullptr;
            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(s, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                throw;
            }
            ck(cudaStreamEndCapture(s, &graph), "end capture");
            const cudaError_t e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            ck(e, "instantiate");
        }
        ck(cudaGraphLaunch(exec, s), "graph launch");
    }

    void* alloc(size_t n) {
        void* p = nullptr;
        ck(cudaMalloc(&p, n), "cudaMalloc");
        allocs.push_back(p);
        bytes += n;
        return p;
    }
    float* fbuf(size_t n) { return (float*) alloc(n * sizeof(float)); }
    DevTensor upload(const HostTensor& t) {
        DevTensor d;
        d.type = t.type;
        d.ne0 = t.ne0;
        d.ne1 = t.ne1;
        d.p = alloc(t.bytes);
        ck(cudaMemcpy(d.p, t.data, t.bytes, cudaMemcpyHostToDevice), "upload");
        return d;
    }
    // y = W x through Q8_1 (quantized weights) or the F32 GEMV
    void proj(const DevTensor& w, const float* xin, float* yout) {
        if (w.type == 0) {
            k::muse_gemv_f32((const float*) w.p, xin, yout, (int) w.ne0, (int) w.ne1, s);
            return;
        }
        k::native_quantize_q8_1(xin, q8, (int) w.ne0, 1, s);
        mmvq(w, yout);
    }
    // y = W q8, the activation quantized by the previous native_quantize_q8_1 (shared by projections of one input)
    void mmvq(const DevTensor& w, float* yout) {
        if (w.type == 0) throw std::logic_error("muse gpu trunk: an F32 weight takes no Q8_1 input");
        k::native_mmvq(w.type, w.p, q8, yout, (int) w.ne0, (int) w.ne1, 1, s);
    }
    const float* fp(const DevTensor& t) const { return (const float*) t.p; }
};

GpuTrunk::GpuTrunk(const TrunkConfig& c, const std::vector<TrunkLayerHost>& layers, const HostTensor* output_norm,
                   const HostTensor* output)
    : impl_(new Impl) {
    Impl& I = *impl_;
    I.c = c;
    if (c.head_dim != 64) throw std::invalid_argument("muse gpu trunk: head_dim must be 64");
    ck(cudaStreamCreate(&I.s), "cudaStreamCreate");
    const int H = c.hidden, QD = c.q_heads * c.head_dim, KD = c.kv_heads * c.head_dim;
    for (const TrunkLayerHost& L : layers) {
        Impl::Layer d;
        d.swa = L.swa;
        d.attn_norm = I.upload(L.attn_norm); d.attn_q = I.upload(L.attn_q); d.attn_k = I.upload(L.attn_k);
        d.attn_v = I.upload(L.attn_v); d.attn_gate = I.upload(L.attn_gate); d.attn_output = I.upload(L.attn_output);
        d.attn_q_norm = I.upload(L.attn_q_norm); d.attn_k_norm = I.upload(L.attn_k_norm);
        d.attn_alpha = I.upload(L.attn_alpha); d.attn_beta = I.upload(L.attn_beta); d.ffn_norm = I.upload(L.ffn_norm);
        d.post_ffw_norm = I.upload(L.post_ffw_norm); d.ffn_alpha = I.upload(L.ffn_alpha); d.ffn_beta = I.upload(L.ffn_beta);
        d.router = I.upload(L.router); d.router_bias = I.upload(L.router_bias); d.latent_down = I.upload(L.latent_down);
        d.latent_up = I.upload(L.latent_up); d.pre_exp_norm = I.upload(L.pre_exp_norm);
        d.ring = L.swa ? c.window : 0;
        const size_t slots = (size_t) (L.swa ? c.window : c.context);
        d.kc = (uint16_t*) I.alloc(slots * KD * 2);
        d.vc = (uint16_t*) I.alloc(slots * KD * 2);
        I.layers.push_back(d);
    }
    if (output_norm && output) {
        I.out_norm = I.upload(*output_norm);
        I.out = I.upload(*output);
    }
    I.x = I.fbuf(H); I.h = I.fbuf(H); I.q = I.fbuf(QD); I.kk = I.fbuf(KD); I.v = I.fbuf(KD); I.g = I.fbuf(QD);
    I.a = I.fbuf(QD); I.o = I.fbuf(H); I.z = I.fbuf(c.latent); I.logits = I.fbuf(c.experts); I.w = I.fbuf(c.top_k);
    I.y = I.fbuf(c.latent); I.f = I.fbuf(H); I.t_attn = I.fbuf(H); I.t_ffn_inp = I.fbuf(H);
    I.head_logits = output ? I.fbuf((size_t) output->ne1) : nullptr;
    if (const size_t n = k::muse_attention_scratch_floats(c.q_heads, c.head_dim, c.attention_splits)) I.attn_part = I.fbuf(n);
    I.ids = (int32_t*) I.alloc(sizeof(int32_t) * c.top_k);
    I.pos = (int32_t*) I.alloc(sizeof(int32_t));
    I.q8 = I.alloc(k::native_q8_1_bytes(std::max(H, std::max(QD, c.latent)), 1));
    I.hx = I.host_buf<float>((size_t) H); I.hy = I.host_buf<float>((size_t) c.latent);
    I.hz = I.host_buf<float>((size_t) c.latent); I.hw = I.host_buf<float>((size_t) c.top_k);
    I.hids = I.host_buf<int32_t>((size_t) c.top_k); I.hpos = I.host_buf<int32_t>(1);
    I.g_attn.assign(layers.size(), nullptr);
    I.g_finish.assign(layers.size(), nullptr);
    z_.resize((size_t) c.latent);
    w_.resize((size_t) c.top_k);
    ids_.resize((size_t) c.top_k);
}

GpuTrunk::~GpuTrunk() {
    if (!impl_) return;
    if (impl_->s) cudaStreamSynchronize(impl_->s);
    for (auto* v : {&impl_->g_attn, &impl_->g_finish})
        for (cudaGraphExec_t e : *v)
            if (e) cudaGraphExecDestroy(e);
    if (impl_->g_head) cudaGraphExecDestroy(impl_->g_head);
    for (void* p : impl_->pinned) cudaFreeHost(p);
    for (void* p : impl_->allocs) cudaFree(p);
    if (impl_->s) cudaStreamDestroy(impl_->s);
}

size_t GpuTrunk::device_bytes() const { return impl_->bytes; }

void GpuTrunk::begin(const float* x, int pos) {
    Impl& I = *impl_;
    // the staging is free: the previous head() waited for the stream
    std::memcpy(I.hx, x, sizeof(float) * I.c.hidden);
    I.hpos[0] = pos;
    ck(cudaMemcpyAsync(I.x, I.hx, sizeof(float) * I.c.hidden, cudaMemcpyHostToDevice, I.s), "begin x");
    ck(cudaMemcpyAsync(I.pos, I.hpos, sizeof(int32_t), cudaMemcpyHostToDevice, I.s), "begin pos");
}

void GpuTrunk::attention_and_route(int layer) {
    Impl& I = *impl_;
    const TrunkConfig& c = I.c;
    const Impl::Layer& L = I.layers.at((size_t) layer);
    const int H = c.hidden, KD = c.kv_heads * c.head_dim;
    void* s = I.s;
    I.run(I.g_attn.at((size_t) layer), [&] {
        k::muse_rms_norm(I.x, nullptr, I.fp(L.attn_norm), I.h, 1, H, c.rms_eps, s);
        k::native_quantize_q8_1(I.h, I.q8, H, 1, s);   // one Q8_1 copy of h for q, k, v and the gate
        I.mmvq(L.attn_q, I.q);
        I.mmvq(L.attn_k, I.kk);
        I.mmvq(L.attn_v, I.v);
        I.mmvq(L.attn_gate, I.g);
        k::muse_rms_norm(I.q, nullptr, I.fp(L.attn_q_norm), I.q, c.q_heads, c.head_dim, c.rms_eps, s);
        k::muse_rms_norm(I.kk, nullptr, I.fp(L.attn_k_norm), I.kk, c.kv_heads, c.head_dim, c.rms_eps, s);
        if (L.swa) {
            k::muse_rope(I.q, c.q_heads, c.head_dim, I.pos, c.rope_base, s);
            k::muse_rope(I.kk, c.kv_heads, c.head_dim, I.pos, c.rope_base, s);
        }
        k::muse_kv_append(I.kk, I.v, L.kc, L.vc, KD, I.pos, L.ring, s);
        k::muse_attention(I.q, L.kc, L.vc, I.a, c.q_heads, c.kv_heads, c.head_dim, I.pos, L.ring,
                          1.f / std::sqrt((float) c.head_dim), s, c.attention_splits, I.attn_part);
        k::muse_head_norm_gate(I.a, I.g, c.q_heads, c.head_dim, c.rms_eps, s);
        I.proj(L.attn_output, I.a, I.o);
        k::muse_rms_norm(I.o, nullptr, nullptr, I.o, 1, H, c.post_eps, s);
        ck(cudaMemcpyAsync(I.t_attn, I.o, sizeof(float) * H, cudaMemcpyDeviceToDevice, I.s), "trace");
        k::muse_residual(I.x, I.o, I.fp(L.attn_alpha), I.fp(L.attn_beta), H, s);
        ck(cudaMemcpyAsync(I.t_ffn_inp, I.x, sizeof(float) * H, cudaMemcpyDeviceToDevice, I.s), "trace");
        k::muse_rms_norm(I.x, nullptr, I.fp(L.ffn_norm), I.h, 1, H, c.rms_eps, s);
        I.proj(L.latent_down, I.h, I.z);
        k::muse_rms_norm(I.z, nullptr, I.fp(L.pre_exp_norm), I.z, 1, c.latent, c.rms_eps, s);
        I.proj(L.router, I.h, I.logits);
        k::glm_router(I.logits, I.fp(L.router_bias), I.ids, I.w, c.experts, c.top_k, c.expert_weights_scale, s);
        ck(cudaMemcpyAsync(I.hz, I.z, sizeof(float) * c.latent, cudaMemcpyDeviceToHost, I.s), "z");
        ck(cudaMemcpyAsync(I.hids, I.ids, sizeof(int32_t) * c.top_k, cudaMemcpyDeviceToHost, I.s), "ids");
        ck(cudaMemcpyAsync(I.hw, I.w, sizeof(float) * c.top_k, cudaMemcpyDeviceToHost, I.s), "weights");
    });
    ck(cudaStreamSynchronize(I.s), "attention sync");
    std::memcpy(z_.data(), I.hz, sizeof(float) * c.latent);
    std::memcpy(ids_.data(), I.hids, sizeof(int32_t) * c.top_k);
    std::memcpy(w_.data(), I.hw, sizeof(float) * c.top_k);
}

void GpuTrunk::finish_moe(int layer, const float* y) {
    Impl& I = *impl_;
    const TrunkConfig& c = I.c;
    const Impl::Layer& L = I.layers.at((size_t) layer);
    void* s = I.s;
    // the staging is free: attention_and_route waited for the stream
    std::memcpy(I.hy, y, sizeof(float) * c.latent);
    I.run(I.g_finish.at((size_t) layer), [&] {
        ck(cudaMemcpyAsync(I.y, I.hy, sizeof(float) * c.latent, cudaMemcpyHostToDevice, I.s), "y");
        I.proj(L.latent_up, I.y, I.f);
        k::muse_rms_norm(I.f, I.fp(L.post_ffw_norm), nullptr, I.f, 1, c.hidden, c.post_eps, s);
        k::muse_residual(I.x, I.f, I.fp(L.ffn_alpha), I.fp(L.ffn_beta), c.hidden, s);
    });
}

void GpuTrunk::head(std::vector<float>& logits) {
    Impl& I = *impl_;
    if (!I.out.p) throw std::logic_error("muse gpu trunk: built without the head");
    I.run(I.g_head, [&] {
        k::muse_rms_norm(I.x, nullptr, I.fp(I.out_norm), I.h, 1, I.c.hidden, I.c.rms_eps, I.s);
        I.proj(I.out, I.h, I.head_logits);
        k::muse_logit_cap(I.head_logits, (int) I.out.ne1, I.c.logit_scale, I.c.softcap, I.s);
    });
    logits.resize((size_t) I.out.ne1);
    ck(cudaMemcpyAsync(logits.data(), I.head_logits, sizeof(float) * logits.size(), cudaMemcpyDeviceToHost, I.s), "logits");
    ck(cudaStreamSynchronize(I.s), "head sync");
}

void GpuTrunk::trace(TrunkTrace which, std::vector<float>& out) const {
    const Impl& I = *impl_;
    const float* src = which == TrunkTrace::AttnPostNorm ? I.t_attn
                     : which == TrunkTrace::FfnInp      ? I.t_ffn_inp
                     : which == TrunkTrace::FfnPostNorm ? I.f
                                                        : I.x;
    out.resize((size_t) I.c.hidden);
    ck(cudaMemcpyAsync(out.data(), src, sizeof(float) * out.size(), cudaMemcpyDeviceToHost, I.s), "trace");
    ck(cudaStreamSynchronize(I.s), "trace sync");
}

}  // namespace strata::muse
