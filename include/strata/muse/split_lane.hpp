// include/strata/muse/split_lane.hpp - Muse Spark's routed experts split by rows between two nodes over ucomm.
//
// Node 0 (the one running the trunk) holds ff rows [0, cut) of every expert, node 1 the rows [cut, ff)
// (strata/muse/row_split.hpp). Per MoE layer the primary sends node 1 the layer's latents and routes, computes its own
// partial expert outputs meanwhile, receives node 1's partials, adds them (local + remote, the same order every time)
// and applies ffn_post_exp_norm and the router weights (ExpertLane::combine). The result is deterministic; it differs
// from a single node's full rows only in summation order.
//
// Wire format (ucomm tags kTagRequest / kTagReply; both ranks little endian):
//   request: int32 {op, layer, nt, top_k} (one message; op 1: run, 0: stop), then float z[nt][latent] followed by
//            int32 ids[nt * top_k] (a second message, sized from the first)
//   reply:   float partial[nt * top_k][latent]
#pragma once

#include "strata/muse/expert_lane.hpp"

#include "ucomm.h"

#include <cstdint>
#include <vector>

namespace strata::muse {

inline constexpr uint32_t kTagRequest = 0x4d530001u, kTagReply = 0x4d530002u;

class SplitLane {
public:
    /// local: the lane over this node's share; c, peer: the ucomm context and the node holding the rest.
    SplitLane(ExpertLane& local, ucomm_t* c, int peer, int latent, float post_eps);
    /// One MoE layer: y[t] as ExpertLane::run, from both nodes' partials. local_layer: this node's share's tensors.
    void run(int layer, const LaneLayer& local_layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k,
             float* y);
    /// Tells the worker to leave its loop.
    void stop();
    double remote_wait_ms() const { return wait_ms_; }   // time spent waiting for node 1 after the local share was done

private:
    ExpertLane& local_;
    ucomm_t* c_;
    int peer_, latent_;
    float post_eps_;
    int32_t head_[4] = {0, 0, 0, 0};
    std::vector<uint8_t> request_;
    std::vector<float> reply_;
    double wait_ms_ = 0;
};

/// Node 1's loop: answers requests with the partials of its share until a stop request. layers[l]: its share of
/// layer l. Returns 0 after a stop, nonzero on a transport error.
int serve_expert_share(ucomm_t* c, int peer, ExpertLane& lane, const std::vector<LaneLayer>& layers, int latent);
/// One request whose 16-byte header has already been received (op 1): receives the payload, runs this node's share
/// and replies. For a node that serves while it also waits for other messages. Returns 0, or nonzero on an error.
int serve_expert_request(ucomm_t* c, int peer, ExpertLane& lane, const std::vector<LaneLayer>& layers, int latent,
                         const int32_t head[4], std::vector<uint8_t>& scratch);

}  // namespace strata::muse
