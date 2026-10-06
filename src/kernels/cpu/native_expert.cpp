// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// Uses ggml activation quantizers and optional shared AVX dot kernels.
// Shared kernels can change float reduction order relative to ggml-cpu.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/q23_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }
bool use_q23_avx2() {
    const char* value=std::getenv("STRATA_Q23_AVX2");
    return value&&std::strcmp(value,"1")==0;
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    if (gu_type < 0 || gu_type >= GGML_TYPE_COUNT || d_type < 0 || d_type >= GGML_TYPE_COUNT ||
        n_embd <= 0 || n_ff <= 0 || n_ff > kNativeFF) {
        err = "native experts: unsupported type or geometry";
        return false;
    }
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.lossless = false;
    f.q23_layout = 0;
    f.observer = nullptr;
    f.observer_layer = -1;
    f.fuse_h_quant = false;
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}
void native_quant_h_rows(const NativeFmt& f,const float* h,void* dst,int first,int last) {
    const int block=ggml_blck_size(ggml_type(f.d_act));
    if(first<0||last<first||last>f.n_ff||first%block||last%block)throw std::invalid_argument("unaligned hidden quantization chunk");
    traits(f.d_act)->from_float(h+first,static_cast<uint8_t*>(dst)+ggml_row_size(ggml_type(f.d_act),first),last-first);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, const uint8_t* separate_up) {
    if (f.q23_layout && (f.gu_type==10 || f.gu_type==11)) {
        thread_local float up[8][kNativeFF];float* ptr[8];for(int t=0;t<nt;++t)ptr[t]=up[t];
        q23_packed_rows(f.gu_type,blob,int(f.n_embd),act,nt,ff,r0,r1,f.q23_layout==2);
        q23_packed_rows(f.gu_type,separate_up?separate_up:blob+f.up_off,int(f.n_embd),act,nt,ptr,r0,r1,f.q23_layout==2);
        for(int t=0;t<nt;++t)for(int r=r0;r<r1;++r) {float g=ff[t][r],u=up[t][r];if(f.swiglu_limit>0){g=std::fmin(g,f.swiglu_limit);u=std::fmax(-f.swiglu_limit,std::fmin(u,f.swiglu_limit));}ff[t][r]=g/(1.f+std::exp(-g))*u;}
        return;
    }
    if ((f.gu_type==10 || f.gu_type==11) && use_q23_avx2()) {
        q23_gu_rows(f.gu_type,blob,separate_up?separate_up:blob+f.up_off,f.gu_row,int(f.n_embd),act,nt,ff,r0,r1,f.swiglu_limit);
        return;
    }
    if (f.lossless) {
        iq256_gu_rows_clamped(f.gu_type + 100, blob, f.gu_row, f.up_off, (int)f.n_embd,
                             act, nt, ff, r0, r1, f.swiglu_limit, separate_up);
        return;
    }
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = std::getenv("STRATA_NO_IQ256") == nullptr;
    if (f.swiglu_limit > 0 && avx2 && iq256_supported(f.gu_type)) {
        iq256_gu_rows_clamped(f.gu_type, blob, f.gu_row, f.up_off, (int)f.n_embd,
                             act, nt, ff, r0, r1, f.swiglu_limit, separate_up);
        return;
    }
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    if (!separate_up && f.swiglu_limit == 0 && kq && f.gu_type == 12 && nt >= 2) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (!separate_up && f.swiglu_limit == 0 && nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = (separate_up ? separate_up : blob + f.up_off) + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            if (f.swiglu_limit > 0) {
                g = std::fmin(g, f.swiglu_limit);
                u = std::fmax(-f.swiglu_limit, std::fmin(u, f.swiglu_limit));
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1, const uint8_t* separate_down) {
    if (f.q23_layout && (f.d_type==10 || f.d_type==11)) {
        q23_packed_rows(f.d_type,separate_down?separate_down:blob+f.down_off,int(f.n_ff),hq,nt,out,r0,r1,f.q23_layout==2);return;
    }
    if ((f.d_type==10 || f.d_type==11) && use_q23_avx2()) {
        q23_rows(f.d_type,separate_down?separate_down:blob+f.down_off,f.d_row,int(f.n_ff),hq,nt,out,r0,r1);
        return;
    }
    if (f.lossless && (f.d_type == 21 || f.d_type == 22)) {
        iq256_rows(f.d_type + 100, separate_down ? separate_down : blob + f.down_off,
                   f.d_row, (int)f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    if (!separate_down && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (nt >= 2 && f.d_type == 20 && iq4nl_mt) {
        iq4nl256_down_rows((separate_down ? separate_down : blob + f.down_off), f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (f.swiglu_limit > 0 && std::getenv("STRATA_NO_IQ256") == nullptr && iq256_supported(f.d_type)) {
        iq256_rows(f.d_type, (separate_down ? separate_down : blob + f.down_off), f.d_row, (int)f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = (separate_down ? separate_down : blob + f.down_off) + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
