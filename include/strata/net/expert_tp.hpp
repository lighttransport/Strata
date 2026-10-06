// Cross-node tensor parallelism for GLM routed experts (decode), over ucomm (tools/ucomm).
//
// Every CPU-computed routed expert is split by its FFN rows: the decoder keeps gate/up rows [0, split) and the
// matching first `split` columns of down; the remote worker computes rows [split, n_ff) and the remaining down
// columns. Down is linear in its input columns, so the two partial outputs add up to the whole expert. The worker
// returns one router-weighted sum per token, so a layer costs one request (the quantized activation and the
// expert list) and one reply, independent of top_k. The reply is fp16 with a per-token scale by default (half the
// bytes of fp32: 8 KiB instead of 16 KiB per token at hidden 4096; fp32 stays selectable).
//
// Weights: with TpSetup::weights = kWeightsStreamed the decoder sends the worker's rows of every expert over the
// network, so the worker host needs no model file or persistent disk. After the setup the worker answers with a
// TpAccept: whether its rows fit in its memory (its RAM, or the free space of --weights-dir such as /dev/shm), and
// whether it already holds rows with the setup's fingerprint (a --weights-dir file kept from an earlier run). If
// they fit and are not held, the decoder sends one kTagWeights message per (layer, expert) in ascending layer
// order: gate rows [split, n_ff), up rows [split, n_ff), then for each of the hidden down rows the bytes of
// columns [split, n_ff). kWeightsDummy keeps synthetic weights (speed measurements only; the output is
// meaningless).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::net {

inline constexpr uint32_t kTpMagic = 0x34505453; // "STP4"
inline constexpr uint32_t kTagSetup = 1, kTagReady = 2, kTagWeights = 3, kTagAccept = 4, kTagRequest = 10, kTagReply = 11;

enum : uint32_t { kReplyF32 = 0, kReplyF16 = 1 };
enum : uint32_t { kWeightsDummy = 0, kWeightsStreamed = 1 };
struct TpSetup {
    uint32_t magic, layers, hidden, reply; // reply: kReplyF32, or kReplyF16 (per token: f32 scale, hidden f16 values)
    uint32_t weights, pad;                 // kWeightsDummy or kWeightsStreamed
    uint64_t fingerprint;                  // identifies the streamed rows (model bytes and split); 0: never reuse
};
struct TpAccept { // worker -> decoder, after the setup
    uint32_t ok, have;                     // ok = 0: refused (reason); have = 1: rows already held, nothing is sent
    uint64_t needed_bytes, available_bytes;
    char reason[240];
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
    uint32_t ok, slots;   // slots: distinct experts per layer the worker holds (all of them with streamed weights)
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
