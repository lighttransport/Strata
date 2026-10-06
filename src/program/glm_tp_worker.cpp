// Remote half of GLM expert tensor parallelism (include/strata/net/expert_tp.hpp). CPU only, no model file.
// Normally the decoder streams this worker's rows of every expert at setup (--remote-tp-weights=send, the
// default), and the output is the model's. With --remote-tp-weights=dummy the expert weights are synthetic data
// with the decoder's per-layer formats, which measures the parallel decode's speed, not its output: a (layer,
// expert) pair then maps onto one of `slots` distinct dummy experts per layer; with the default pool the working
// set per token is far larger than any cache, so each token streams its expert bytes from DRAM as real weights
// would.
//
// Streamed rows live in this process's anonymous memory, or with --weights-dir=DIR in a file there: on a tmpfs such
// as /dev/shm they stay in RAM after the worker exits, and the next worker given the same rows (same model bytes
// and split, the decoder's fingerprint) maps them instead of receiving them again. Either way the worker first
// checks that the rows fit (MemAvailable minus --reserve-gib, and DIR's free space) and refuses the setup if not.
//   strata-glm-tp-worker --root HOST:PORT [--threads=N] [--weights-dir=/dev/shm] [--reserve-gib=2]
//                        [--pool-gib=G] [--backend=auto|ib|udp] [--null-compute]
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
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

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

constexpr double kGiB = double(1ull << 30);

size_t mem_available() {
    std::ifstream info("/proc/meminfo");
    std::string key;
    size_t kib = 0;
    while (info >> key >> kib) {
        if (key == "MemAvailable:") return kib * 1024;
        info.ignore(256, '\n');
    }
    return (size_t)sysconf(_SC_AVPHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
}

// The expert rows: anonymous memory, or a shared mapping of a --weights-dir file (kept for the next worker).
struct Storage {
    uint8_t *data = nullptr;
    size_t bytes = 0;
    std::string path; // the complete file; rows are received into path + ".partial" and renamed when complete
    ~Storage() {
        if (data) munmap(data, bytes);
    }
    void map_anonymous() {
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot allocate the expert rows");
        data = (uint8_t *)p;
    }
    void map_file(const std::string &file, bool create) {
        const int fd = open(file.c_str(), create ? O_RDWR | O_CREAT | O_TRUNC : O_RDWR, 0600);
        if (fd < 0) throw std::runtime_error("cannot open " + file);
        if (create && ftruncate(fd, (off_t)bytes) != 0) {
            close(fd);
            throw std::runtime_error("cannot size " + file);
        }
        // Kept rows are mapped populated: faulting 4 KiB pages in on first use from every pool thread at once
        // stalled the first decode for tens of seconds.
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | (create ? 0 : MAP_POPULATE), fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + file);
        data = (uint8_t *)p;
    }
    void complete() { // the received rows become reusable
        if (path.empty()) return;
        if (msync(data, bytes, MS_SYNC) != 0 || std::rename((path + ".partial").c_str(), path.c_str()) != 0)
            throw std::runtime_error("cannot finish " + path);
    }
};

uint64_t fnv(const void *p, size_t n, uint64_t h = 14695981039346656037ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t *)p)[i]) * 1099511628211ull;
    return h;
}

} // namespace

int main(int argc, char **argv) {
    std::string root, backend = "auto";
    int threads = 0;
    double pool_gib = 24;
    std::string save_setup, setup_file, weights_dir;
    double reserve_gib = 2;
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
        else if (a.starts_with("--weights-dir=")) weights_dir = a.substr(14);
        else if (a.starts_with("--reserve-gib=")) reserve_gib = std::stod(a.substr(14));
        else if (a == "--null-compute") null_compute = true; // reply zeros at once: measures communication only
        else {
            std::fprintf(stderr, "usage: %s --root HOST:PORT [--threads=N] [--weights-dir=/dev/shm] [--reserve-gib=2] "
                                 "[--pool-gib=24] [--backend=auto|ib|udp]\n", argv[0]);
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
        const bool streamed = setup.weights == net::kWeightsStreamed;
        if (!streamed && setup.weights != net::kWeightsDummy) throw std::runtime_error("bad setup weights mode");
        if (streamed && !setup_file.empty()) throw std::runtime_error("a saved setup has no weights; use the decoder");
        size_t slots = (size_t)(pool_gib * (1ull << 30)) / per_set;
        slots = streamed ? max_experts : std::max<size_t>(1, std::min(slots, max_experts));
        const size_t total = slots * per_set;
        // Fit check before any rows move, and reuse of rows a previous worker kept in --weights-dir.
        Storage store;
        store.bytes = total;
        net::TpAccept accept{};
        accept.ok = 1;
        accept.needed_bytes = total;
        if (streamed && !weights_dir.empty() && setup.fingerprint) {
            // The file name covers the decoder's fingerprint and this worker's layout of the rows.
            uint64_t key = fnv(&setup.fingerprint, sizeof setup.fingerprint);
            key = fnv(&setup.hidden, sizeof setup.hidden, key);
            key = fnv(msg.data() + sizeof setup, setup.layers * sizeof(net::TpLayer), key);
            char name[64];
            std::snprintf(name, sizeof name, "strata-tp-%016llx.weights", (unsigned long long)key);
            store.path = weights_dir + "/" + name;
            struct stat st{};
            if (stat(store.path.c_str(), &st) == 0 && (size_t)st.st_size == total) {
                store.map_file(store.path, false);
                accept.have = 1;
            }
        }
        if (!accept.have) {
            size_t available = mem_available();
            std::string where = "MemAvailable";
            if (!store.path.empty()) {
                // Rows kept for other setups would hold this memory (tmpfs pages are RAM): one setup per directory.
                for (const auto &entry : std::filesystem::directory_iterator(weights_dir)) {
                    const auto file = entry.path().filename().string();
                    if (file.starts_with("strata-tp-") && (file.ends_with(".weights") || file.ends_with(".weights.partial"))) {
                        std::fprintf(stderr, "TP_WORKER removing %s (other rows)\n", entry.path().c_str());
                        std::filesystem::remove(entry.path());
                    }
                }
                available = mem_available();
                struct statvfs fs{};
                if (statvfs(weights_dir.c_str(), &fs) != 0) throw std::runtime_error("cannot inspect " + weights_dir);
                const size_t free_bytes = (size_t)fs.f_bavail * fs.f_frsize;
                if (free_bytes < available) { available = free_bytes; where = weights_dir + " free space"; }
            }
            const size_t reserve = (size_t)(reserve_gib * kGiB);
            accept.available_bytes = available > reserve ? available - reserve : 0;
            if (total > accept.available_bytes) {
                accept.ok = 0;
                std::snprintf(accept.reason, sizeof accept.reason,
                              "worker rows need %.2f GiB; %s minus --reserve-gib=%.1f leaves %.2f GiB", total / kGiB,
                              where.c_str(), reserve_gib, accept.available_bytes / kGiB);
            }
        }
        if (c) ok(ucomm_send(c, 0, net::kTagAccept, &accept, sizeof accept), "accept");
        if (!accept.ok) throw std::runtime_error(accept.reason);
        if (!accept.have) {
            if (store.path.empty()) store.map_anonymous();
            else store.map_file(store.path + ".partial", true);
        }
        uint8_t *pool_mem = store.data;
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
            if (streamed) { // the decoder sends each expert's blob in layer order, unless the rows were kept
                if (accept.have) continue;
                // One registration per layer instead of one per message (IB); a layer stays far below the usual
                // locked-memory limit, the whole pool may not. Without it the transfer still works, more slowly.
                ucomm_mr_t *region = nullptr;
                if (ucomm_mr_reg(c, L.base, (size_t)L.experts * L.slot_bytes, &region) != UCOMM_OK) region = nullptr;
                for (int e = 0; e < L.experts; ++e) {
                    size_t got = 0;
                    ok(ucomm_recv(c, 0, net::kTagWeights, L.base + (size_t)e * L.slot_bytes, L.slot_bytes, &got), "weights");
                    if (got != L.f.bytes) throw std::runtime_error("weights message size mismatch at layer " + std::to_string(l));
                }
                if (region) ucomm_mr_dereg(region);
                continue;
            }
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
        if (streamed && !accept.have) store.complete();
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_start).count();
        std::fprintf(stderr, "TP_WORKER layers=%u slots=%zu pool_GiB=%.2f weights=%s storage=%s fill_s=%.1f\n", setup.layers,
                     slots, total / kGiB, !streamed ? "dummy" : accept.have ? "kept" : "streamed",
                     store.path.empty() ? "memory" : store.path.c_str(), fill_s);

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
        // Message buffers are registered once (IB): ucomm otherwise registers every rendezvous buffer per message,
        // a firmware command on ConnectX-3 that cost ~0.2 ms per layer. Sized for the largest token count.
        reply.reserve((size_t)cpu::MAXT * H);
        wire.reserve(net::tp_reply_bytes(net::kReplyF16, cpu::MAXT, H));
        std::vector<ucomm_mr_t *> regions;
        for (const auto &[p, bytes] : {std::pair<void *, size_t>{msg.data(), msg.size()},
                                       {reply.data(), reply.capacity() * sizeof(float)}, {wire.data(), wire.capacity()}}) {
            ucomm_mr_t *region = nullptr;
            if (ucomm_mr_reg(c, p, bytes, &region) == UCOMM_OK) regions.push_back(region);
        }
        std::vector<cpu::ExpertJobMulti> jobs;
        int64_t requests = 0;
        double busy_ms = 0, idle_ms = 0, reply_ms = 0;
        auto since = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        };
        for (;;) {
            const auto waiting = std::chrono::steady_clock::now();
            ok(ucomm_recv(c, 0, net::kTagRequest, msg.data(), msg.size(), &n), "request");
            idle_ms += since(waiting);
            net::TpRequest rq;
            std::memcpy(&rq, msg.data(), sizeof rq);
            if (rq.layer < 0) break;
            if (rq.nt < 1 || rq.nt > cpu::MAXT) throw std::runtime_error("request token count out of range");
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
                    if (list[j].expert < 0 || list[j].expert >= L.experts || list[j].token < 0 || list[j].token >= rq.nt)
                    throw std::runtime_error("request names an invalid expert or token");
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
                    dst[i] += streamed || std::isfinite(v) ? v : 0.f; // dummy weights can decode to Inf/NaN scales
                }
            }
            busy_ms += since(t0);
            ++requests;
            const auto replying = std::chrono::steady_clock::now();
            send_reply(rq.nt);
            reply_ms += since(replying);
        }
        const double per = requests ? 1.0 / requests : 0.0;
        std::fprintf(stderr, "TP_WORKER done requests=%lld compute_ms=%.1f (%.3f ms/request) idle_ms_per_request=%.3f "
                             "reply_ms_per_request=%.3f\n", (long long)requests, busy_ms, busy_ms * per, idle_ms * per, reply_ms * per);
        for (auto *region : regions) ucomm_mr_dereg(region);
        ucomm_finalize(c);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "TP_WORKER error: %s\n", e.what());
        return 1;
    }
    return 0;
}
