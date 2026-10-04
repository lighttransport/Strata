// Remote half of GLM expert tensor parallelism (include/strata/net/expert_tp.hpp). CPU only, no model file:
// the expert weights are dummy data with the decoder's per-layer formats, so this measures the parallel decode's
// speed, not its output. A (layer, expert) pair maps onto one of `slots` distinct dummy experts per layer; with the
// default pool the working set per token is far larger than any cache, so each token streams its expert bytes from
// DRAM exactly as real weights would.
//   strata-glm-tp-worker --root HOST:PORT [--threads=N] [--pool-gib=G] [--backend=auto|ib|udp] [--null-compute]
//   offline: --save-setup=FILE while serving, then --setup-file=FILE --selftest=N to time the expert math alone
#include "ggml.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/net/expert_tp.hpp"
#include "ucomm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {
namespace cpu = strata::kernels::cpu;
namespace net = strata::net;

void ok(ucomm_status s, const char *what) {
    if (s != UCOMM_OK) throw std::runtime_error(std::string(what) + ": " + ucomm_strerror(s));
}

struct Layer {
    cpu::NativeFmt f;          // the remote half: n_ff = rows here, d_row = this half's down row
    size_t gate_bytes = 0, slot_bytes = 0;
    uint8_t *base = nullptr;   // slots * slot_bytes: [gate rows | up rows | down rows] per slot
    int experts = 0;
};

} // namespace

int main(int argc, char **argv) {
    std::string root, backend = "auto";
    int threads = 0;
    double pool_gib = 24;
    std::string save_setup, setup_file;
    int selftest = 0;
    bool null_compute = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a.starts_with("--root=")) root = a.substr(7);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--pool-gib=")) pool_gib = std::stod(a.substr(11));
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else if (a.starts_with("--save-setup=")) save_setup = a.substr(13);
        else if (a.starts_with("--setup-file=")) setup_file = a.substr(13);
        else if (a.starts_with("--selftest=")) selftest = std::stoi(a.substr(11));
        else if (a == "--null-compute") null_compute = true; // reply zeros at once: measures communication only
        else {
            std::fprintf(stderr, "usage: %s --root HOST:PORT [--threads=N] [--pool-gib=24] [--backend=auto|ib|udp]\n", argv[0]);
            return 2;
        }
    }
    if (root.empty() && setup_file.empty()) {
        std::fprintf(stderr, "--root is required (the decoder's --remote-tp address)\n");
        return 2;
    }
    try {
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 1;
        cfg.world = 2;
        cfg.root_addr = root.c_str();
        cfg.timeout_ms = 0; // the decoder may load for minutes before it connects or asks
        cfg.backend = backend == "ib" ? UCOMM_BACKEND_IB : backend == "udp" ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
        ucomm_t *c = nullptr;
        std::vector<uint8_t> msg(1 << 20);
        size_t n = 0;
        if (setup_file.empty()) {
            ok(ucomm_init(&cfg, &c), "ucomm_init");
            std::fprintf(stderr, "TP_WORKER connected backend=%s\n", ucomm_backend_name(c));
            ok(ucomm_recv(c, 0, net::kTagSetup, msg.data(), msg.size(), &n), "setup");
            if (!save_setup.empty())
                if (FILE *fp = std::fopen(save_setup.c_str(), "wb")) { std::fwrite(msg.data(), 1, n, fp); std::fclose(fp); }
        } else { // offline: a setup saved by --save-setup, for --selftest without a decoder
            FILE *fp = std::fopen(setup_file.c_str(), "rb");
            if (!fp) throw std::runtime_error("cannot open " + setup_file);
            n = std::fread(msg.data(), 1, msg.size(), fp);
            std::fclose(fp);
        }
        net::TpSetup setup;
        std::memcpy(&setup, msg.data(), sizeof setup);
        if (setup.magic != net::kTpMagic || n != sizeof setup + setup.layers * sizeof(net::TpLayer))
            throw std::runtime_error("bad setup message");
        const int H = (int)setup.hidden;
        std::map<int, Layer> layers;
        size_t per_set = 0, max_experts = 0;
        for (uint32_t i = 0; i < setup.layers; ++i) {
            net::TpLayer s;
            std::memcpy(&s, msg.data() + sizeof setup + i * sizeof s, sizeof s);
            Layer L;
            std::string error;
            if (!cpu::native_fmt(s.gu_type, s.d_type, H, s.n_ff, L.f, error)) throw std::runtime_error(error);
            const int rows = s.n_ff - s.split;
            L.f.n_ff = rows;
            L.f.d_row = ggml_row_size((ggml_type)s.d_type, rows);
            L.f.h_bytes = ggml_row_size((ggml_type)L.f.d_act, rows);
            L.f.up_off = L.f.gu_row * (size_t)rows;
            L.f.down_off = 2 * L.f.up_off;
            L.f.bytes = L.f.down_off + L.f.d_row * (size_t)H;
            L.f.swiglu_limit = s.swiglu_limit;
            L.slot_bytes = (L.f.bytes + 4095) & ~(size_t)4095;
            L.experts = s.experts;
            per_set += L.slot_bytes;
            max_experts = std::max(max_experts, (size_t)s.experts);
            layers[s.layer] = L;
        }
        size_t slots = (size_t)(pool_gib * (1ull << 30)) / per_set;
        slots = std::max<size_t>(1, std::min(slots, max_experts));
        const size_t total = slots * per_set;
        auto *pool_mem = (uint8_t *)std::aligned_alloc(4096, total);
        if (!pool_mem) throw std::runtime_error("cannot allocate the dummy expert pool");
        // Dummy weights are real quantized data: Gaussian rows quantized by ggml into each layer's formats (random
        // bytes would decode to Inf/NaN/denormal block scales and send the float math down slow paths). One template
        // per distinct (type, rows, width) is quantized, then copied into every slot, which also commits the pages.
        const auto fill_start = std::chrono::steady_clock::now();
        const int cores = std::max(1u, std::thread::hardware_concurrency());
        auto parallel = [&](size_t count, auto &&body) {
            std::vector<std::thread> th;
            for (int t = 0; t < cores; ++t)
                th.emplace_back([&, t] { for (size_t i = count * t / cores; i < count * (t + 1) / cores; ++i) body(i); });
            for (auto &t : th) t.join();
        };
        std::map<std::tuple<int, int, int>, std::vector<uint8_t>> templates;
        auto quantized = [&](int type, int rows, int width) -> const std::vector<uint8_t> & {
            auto &q = templates[{type, rows, width}];
            if (!q.empty()) return q;
            const size_t row_bytes = ggml_row_size((ggml_type)type, width);
            q.resize(row_bytes * rows);
            std::vector<float> imatrix(width, 1.f);
            const bool need_imatrix = ggml_quantize_requires_imatrix((ggml_type)type);
            ggml_quantize_init((ggml_type)type);
            parallel((size_t)rows, [&](size_t r) {
                std::vector<float> row(width);
                uint64_t x = 0x9e3779b97f4a7c15ull ^ (r * 0x100000001b3ull + (uint64_t)type);
                for (int i = 0; i < width; i += 2) { // Box-Muller, std 0.02 like trained FFN weights
                    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return ((x >> 11) + 0.5) / 9007199254740992.0; };
                    const double u = next(), v = next(), m = 0.02 * std::sqrt(-2 * std::log(u));
                    row[i] = (float)(m * std::cos(6.283185307179586 * v));
                    if (i + 1 < width) row[i + 1] = (float)(m * std::sin(6.283185307179586 * v));
                }
                ggml_quantize_chunk((ggml_type)type, row.data(), q.data() + r * row_bytes, 0, 1, width,
                                    need_imatrix ? imatrix.data() : nullptr);
            });
            return q;
        };
        size_t off = 0;
        for (auto &[l, L] : layers) {
            L.base = pool_mem + off;
            off += slots * L.slot_bytes;
            const int rows = (int)L.f.n_ff;
            const auto &gu = quantized(L.f.gu_type, rows, H);
            const auto &down = quantized(L.f.d_type, H, rows);
            parallel(slots, [&](size_t e) {
                uint8_t *blob = L.base + e * L.slot_bytes;
                std::memcpy(blob, gu.data(), gu.size());
                std::memcpy(blob + L.f.up_off, gu.data(), gu.size());
                std::memcpy(blob + L.f.down_off, down.data(), down.size());
            });
        }
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_start).count();
        std::fprintf(stderr, "TP_WORKER layers=%u slots=%zu pool_GiB=%.2f fill_s=%.1f\n", setup.layers, slots,
                     total / double(1ull << 30), fill_s);

        cpu::ExpertPool pool(threads > 0 ? threads - 1 : 0, true, true);
        std::fprintf(stderr, "TP_WORKER pool threads=%d\n", pool.workers() + (pool.host_works() ? 1 : 0));
        if (selftest > 0) { // decode-shaped requests (one token, top-8, every layer) without the network
            std::vector<float> x(H), outs8((size_t)8 * H);
            for (int i = 0; i < H; ++i) x[i] = std::sin(0.1f * i);
            std::vector<uint8_t> act(cpu::kNativeActBytes);
            double ms = 0;
            int64_t count = 0;
            uint64_t rng = 12345;
            for (int it = 0; it < selftest; ++it)
                for (auto &[l, L] : layers) {
                    cpu::native_quant_act(L.f, x.data(), act.data());
                    std::vector<cpu::ExpertJobMulti> js(8);
                    for (int j = 0; j < 8; ++j) {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        const uint8_t *blob = L.base + (size_t)(rng % slots) * L.slot_bytes;
                        js[j].blob = blob;
                        js[j].native_up = blob + L.f.up_off;
                        js[j].native_down = blob + L.f.down_off;
                        js[j].nt = 1;
                        js[j].nact[0] = act.data();
                        js[j].out[0] = outs8.data() + (size_t)j * H;
                    }
                    const auto t0 = std::chrono::steady_clock::now();
                    pool.run_split_multi_native(L.f, js.data(), 8);
                    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    ++count;
                }
            const double mb = [&] { double b = 0; for (auto &[l, L] : layers) b += 8.0 * L.f.bytes; return b / layers.size() / 1e6; }();
            std::fprintf(stderr, "TP_WORKER selftest requests=%lld ms_per_request=%.3f MB_per_request=%.1f GB/s=%.1f\n",
                         (long long)count, ms / count, mb, mb / (ms / count));
            if (!c) return 0;
        }
        net::TpReady ready{1, (uint32_t)slots, fill_s};
        ok(ucomm_send(c, 0, net::kTagReady, &ready, sizeof ready), "ready");

        std::vector<float> outs, reply;
        std::vector<uint8_t> wire;
        // fp16 reply: each token's values divided by a scale that maps its largest magnitude to 60000
        auto send_reply = [&](int nt) {
            if (setup.reply != net::kReplyF16) {
                ok(ucomm_send(c, 0, net::kTagReply, reply.data(), reply.size() * sizeof(float)), "reply");
                return;
            }
            wire.resize(net::tp_reply_bytes(setup.reply, nt, H));
            std::vector<float> scaled(H);
            for (int t = 0; t < nt; ++t) {
                const float *v = reply.data() + (size_t)t * H;
                float peak = 0;
                for (int i = 0; i < H; ++i) peak = std::max(peak, std::fabs(v[i]));
                const float scale = peak > 0 ? peak / 60000.f : 1.f;
                for (int i = 0; i < H; ++i) scaled[i] = v[i] / scale;
                uint8_t *dst = wire.data() + (size_t)t * (4 + 2 * (size_t)H);
                std::memcpy(dst, &scale, 4);
                ggml_fp32_to_fp16_row(scaled.data(), (ggml_fp16_t *)(dst + 4), H);
            }
            ok(ucomm_send(c, 0, net::kTagReply, wire.data(), wire.size()), "reply");
        };
        std::vector<cpu::ExpertJobMulti> jobs;
        int64_t requests = 0;
        double busy_ms = 0;
        for (;;) {
            ok(ucomm_recv(c, 0, net::kTagRequest, msg.data(), msg.size(), &n), "request");
            net::TpRequest rq;
            std::memcpy(&rq, msg.data(), sizeof rq);
            if (rq.layer < 0) break;
            if (null_compute) {
                reply.assign((size_t)rq.nt * H, 0.f);
                ++requests;
                send_reply(rq.nt);
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            auto &L = layers.at(rq.layer);
            const auto *list = (const net::TpJob *)(msg.data() + sizeof rq);
            const uint8_t *acts = msg.data() + sizeof rq + rq.njobs * sizeof(net::TpJob);
            outs.assign((size_t)rq.njobs * H, 0.f);
            jobs.clear();
            std::map<int, size_t> group;
            for (int j = 0; j < rq.njobs; ++j) {
                auto [it, inserted] = group.emplace(list[j].expert, jobs.size());
                if (inserted) {
                    jobs.emplace_back();
                    const uint8_t *blob = L.base + (size_t)(list[j].expert % (int)slots) * L.slot_bytes;
                    jobs.back().blob = blob;
                    jobs.back().native_up = blob + L.f.up_off;
                    jobs.back().native_down = blob + L.f.down_off;
                }
                auto &job = jobs[it->second];
                const int k = job.nt++;
                job.nact[k] = acts + (size_t)list[j].token * rq.act_bytes;
                job.out[k] = outs.data() + (size_t)j * H;
            }
            pool.run_split_multi_native(L.f, jobs.data(), (int)jobs.size());
            reply.assign((size_t)rq.nt * H, 0.f);
            for (int j = 0; j < rq.njobs; ++j) {
                float *dst = reply.data() + (size_t)list[j].token * H;
                const float *src = outs.data() + (size_t)j * H;
                for (int i = 0; i < H; ++i) {
                    const float v = list[j].weight * src[i];
                    dst[i] += std::isfinite(v) ? v : 0.f; // dummy weights can decode to Inf/NaN scales
                }
            }
            busy_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++requests;
            send_reply(rq.nt);
        }
        std::fprintf(stderr, "TP_WORKER done requests=%lld compute_ms=%.1f (%.3f ms/request)\n", (long long)requests,
                     busy_ms, requests ? busy_ms / requests : 0.0);
        ucomm_finalize(c);
        std::free(pool_mem);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "TP_WORKER error: %s\n", e.what());
        return 1;
    }
    return 0;
}
