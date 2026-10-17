// src/muse/split_lane.cpp - strata/muse/split_lane.hpp
#include "strata/muse/split_lane.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::muse {

namespace {
void ok(ucomm_status s, const char* what) {
    if (s != UCOMM_OK) throw std::runtime_error(std::string("muse split lane: ") + what + ": " + ucomm_strerror(s));
}
}  // namespace

SplitLane::SplitLane(ExpertLane& local, ucomm_t* c, int peer, int latent, float post_eps)
    : local_(local), c_(c), peer_(peer), latent_(latent), post_eps_(post_eps) {}

void SplitLane::run(int layer, const LaneLayer& local_layer, const float* z, int nt, const int32_t* ids, const float* w,
                    int top_k, float* y) {
    const size_t zf = (size_t) nt * latent_, entries = (size_t) nt * top_k;
    head_[0] = 1; head_[1] = layer; head_[2] = nt; head_[3] = top_k;
    request_.resize(zf * 4 + entries * 4);
    std::memcpy(request_.data(), z, zf * 4);
    std::memcpy(request_.data() + zf * 4, ids, entries * 4);
    ucomm_req_t *sent_head = nullptr, *sent = nullptr;
    ok(ucomm_isend(c_, peer_, kTagRequest, head_, sizeof head_, &sent_head), "send");
    ok(ucomm_isend(c_, peer_, kTagRequest, request_.data(), request_.size(), &sent), "send");
    reply_.resize(entries * latent_);
    ucomm_req_t* got = nullptr;
    ok(ucomm_irecv(c_, peer_, kTagReply, reply_.data(), reply_.size() * 4, &got), "receive");
    local_.run_raw(local_layer, z, nt, ids, top_k);   // the local share while node 1 computes its own
    const auto t0 = std::chrono::steady_clock::now();
    size_t n = 0;
    ok(ucomm_wait(sent_head, nullptr), "send wait");
    ok(ucomm_wait(sent, nullptr), "send wait");
    ok(ucomm_wait(got, &n), "receive wait");
    wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (n != reply_.size() * 4) throw std::runtime_error("muse split lane: reply size");
    float* raw = local_.raw();
    for (size_t i = 0; i < reply_.size(); ++i) raw[i] += reply_[i];
    ExpertLane::combine(raw, local_layer.post_exp, post_eps_, latent_, nt, w, top_k, y);
}

void SplitLane::stop() {
    const int32_t head[4] = {0, 0, 0, 0};
    ok(ucomm_send(c_, peer_, kTagRequest, head, sizeof head), "stop");
}

int serve_expert_request(ucomm_t* c, int peer, ExpertLane& lane, const std::vector<LaneLayer>& layers, int latent,
                         const int32_t head[4], std::vector<uint8_t>& request) {
    const int layer = head[1], nt = head[2], top_k = head[3];
    if (layer < 0 || layer >= (int) layers.size() || nt < 1 || nt > 65536 || top_k < 1 || top_k > 64) return 1;
    const size_t zf = (size_t) nt * latent, entries = (size_t) nt * top_k;
    request.resize(zf * 4 + entries * 4);
    size_t n = 0;
    if (ucomm_recv(c, peer, kTagRequest, request.data(), request.size(), &n) != UCOMM_OK || n != request.size()) return 1;
    const float* z = reinterpret_cast<const float*>(request.data());
    const int32_t* ids = reinterpret_cast<const int32_t*>(request.data() + zf * 4);
    lane.run_raw(layers[(size_t) layer], z, nt, ids, top_k);
    return ucomm_send(c, peer, kTagReply, lane.raw(), entries * latent * 4) != UCOMM_OK;
}

int serve_expert_share(ucomm_t* c, int peer, ExpertLane& lane, const std::vector<LaneLayer>& layers, int latent) {
    std::vector<uint8_t> request;
    for (;;) {
        // a fixed header first, then the payload whose size it gives
        int32_t head[4];
        size_t n = 0;
        if (ucomm_recv(c, peer, kTagRequest, head, sizeof head, &n) != UCOMM_OK || n != sizeof head) return 1;
        if (head[0] == 0) return 0;
        if (serve_expert_request(c, peer, lane, layers, latent, head, request)) return 1;
    }
}

}  // namespace strata::muse
