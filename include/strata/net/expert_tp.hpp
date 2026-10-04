// Cross-node tensor parallelism for GLM routed experts (decode), over ucomm (tools/ucomm).
//
// Every CPU-computed routed expert is split by its FFN rows: the decoder keeps gate/up rows [0, split) and the
// matching first `split` columns of down; the remote worker computes rows [split, n_ff) and the remaining down
// columns. Down is linear in its input columns, so the two partial outputs add up to the whole expert. The worker
// returns one router-weighted sum per token (hidden floats), so a layer costs one request (the quantized
// activation and the expert list) and one reply, independent of top_k.
#pragma once

#include <cstdint>

namespace strata::net {

inline constexpr uint32_t kTpMagic = 0x31505453; // "STP1"
inline constexpr uint32_t kTagSetup = 1, kTagReady = 2, kTagRequest = 10, kTagReply = 11;

struct TpSetup {
    uint32_t magic, layers, hidden, pool_mib;
};
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
