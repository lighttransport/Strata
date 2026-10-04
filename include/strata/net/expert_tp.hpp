// Cross-node tensor parallelism for GLM routed experts (decode), over ucomm (tools/ucomm).
//
// Every CPU-computed routed expert is split by its FFN rows: the decoder keeps gate/up rows [0, split) and the
// matching first `split` columns of down; the remote worker computes rows [split, n_ff) and the remaining down
// columns. Down is linear in its input columns, so the two partial outputs add up to the whole expert. The worker
// returns one router-weighted sum per token, so a layer costs one request (the quantized activation and the
// expert list) and one reply, independent of top_k. The reply is fp16 with a per-token scale by default (half the
// bytes of fp32: 8 KiB instead of 16 KiB per token at hidden 4096; fp32 stays selectable).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::net {

inline constexpr uint32_t kTpMagic = 0x32505453; // "STP2"
inline constexpr uint32_t kTagSetup = 1, kTagReady = 2, kTagRequest = 10, kTagReply = 11;

enum : uint32_t { kReplyF32 = 0, kReplyF16 = 1 };
struct TpSetup {
    uint32_t magic, layers, hidden, reply; // reply: kReplyF32, or kReplyF16 (per token: f32 scale, hidden f16 values)
};
inline size_t tp_reply_bytes(uint32_t reply, int nt, int hidden) {
    return reply == kReplyF16 ? (size_t)nt * (4 + 2 * (size_t)hidden) : (size_t)nt * hidden * 4;
}
struct TpLayer { // one MoE layer's expert geometry, as the decoder sees it
    int32_t layer, gu_type, d_type, n_ff, split, experts;
    float swiglu_limit;
    int32_t pad;
};
struct TpReady {
    uint32_t ok, slots;   // slots: distinct dummy experts per layer the worker holds
    double fill_seconds;
};
struct TpJob {
    int32_t expert, token;
    float weight;
};
struct TpRequest { // followed by njobs TpJob, then nt activations of act_bytes each. layer < 0: shut down
    int32_t layer, nt, njobs, act_bytes;
};

} // namespace strata::net
