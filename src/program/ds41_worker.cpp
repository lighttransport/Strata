// strata-ds41-worker: the remote rows of DeepSeek-V4.1's routed experts on a node with a GPU (two-node runs,
// include/strata/net/expert_tp.hpp, docs/DS41F_PLAN.md section 8c). The protocol and the stored rows are those of
// strata-glm-tp-worker's DS4.1 mode (the decoder streams rows [fl, ff) of every expert at setup, kept in
// --weights-dir by fingerprint); the work moves to the GPU:
//   decode requests (fewer than STRATA_DS41_WORKER_MMQ_MIN, 64, tokens): the experts resident in a VRAM tier of this
//     node's blobs on the GPU (MMVQ), the others on the CPU lane meanwhile; the tier adapts like the decoder's (scores
//     decay per token, up to STRATA_DS41_WORKER_PROMOTE (4) missed experts replace lower-scored ones per request),
//     and is refilled from the scores after each prompt pass;
//   prompt passes: every route on the GPU with MMQ, the layer's blobs uploaded while the previous layer computes.
// TpReady.ok = 2 tells the decoder it may send prompt passes instead of fetching the rows.
//   strata-ds41-worker --root HOST:PORT [--threads=N] [--weights-dir=/dev/shm | --rows=pinned|anon|hugetlb] [--reserve-gib=16]
//                      [--gpu-reserve-mib=1024] [--backend=auto|ib|udp]
#include "ggml.h"
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/net/expert_tp.hpp"
#include "strata/net/tp_storage.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ucomm.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/statvfs.h>

namespace {
namespace cpu = strata::kernels::cpu;
namespace k = strata::kernels;
namespace mq = strata::prefill::mmq;
namespace net = strata::net;

void ok(ucomm_status s, const char* what) {
    if (s != UCOMM_OK) throw std::runtime_error(std::string(what) + ": " + ucomm_strerror(s));
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 worker ") + what + ": " + cudaGetErrorString(e));
}
struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    explicit Dev(size_t n) : bytes(n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
    Dev& operator=(Dev&& o) noexcept { std::swap(p, o.p); std::swap(bytes, o.bytes); return *this; }
    ~Dev() { if (p) cudaFree(p); }
    float* f(size_t at = 0) const { return static_cast<float*>(p) + at; }
    int* i() const { return static_cast<int*>(p); }
    uint8_t* b(size_t at = 0) const { return static_cast<uint8_t*>(p) + at; }
};
struct Layer {
    cpu::NativeFmt f;   // this node's rows: n_ff = rows, packed down rows
    size_t slot_bytes = 0, pstride = 0;   // a stored blob's slot; a blob in VRAM for MMQ (whole blocks of both types)
    uint8_t* base = nullptr;
    int experts = 0;
    float clamp = 0;
};
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
} // namespace

int main(int argc, char** argv) {
    std::string root, backend = "auto", weights_dir, rows_mode = "pinned";
    int threads = 0;
    // --reserve-gib (16): RAM left after the rows; the 62 GB B550 OOM-killed a worker at 42 GB of rows (2026-10-21)
    double reserve_gib = 16, gpu_reserve_mib = 1024;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a.starts_with("--root=")) root = a.substr(7);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else if (a.starts_with("--weights-dir=")) weights_dir = a.substr(14);
        else if (a.starts_with("--reserve-gib=")) reserve_gib = std::stod(a.substr(14));
        else if (a.starts_with("--gpu-reserve-mib=")) gpu_reserve_mib = std::stod(a.substr(18));
        else if (a.starts_with("--rows=")) rows_mode = a.substr(7);   // pinned (default without --weights-dir), anon or hugetlb
        else {
            std::fprintf(stderr, "usage: %s --root HOST:PORT [--threads=N] [--weights-dir=/dev/shm | --rows=pinned|anon|hugetlb] [--reserve-gib=16] "
                                 "[--gpu-reserve-mib=1024] [--backend=auto|ib|udp]\n", argv[0]);
            return 2;
        }
    }
    if (root.empty()) {
        std::fprintf(stderr, "--root is required (the decoder's --remote-tp address)\n");
        return 2;
    }
    if (rows_mode != "pinned" && rows_mode != "anon" && rows_mode != "hugetlb") {
        std::fprintf(stderr, "--rows must be pinned, anon or hugetlb\n");
        return 2;
    }
    if (rows_mode == "hugetlb" && !weights_dir.empty()) {
        std::fprintf(stderr, "--rows=hugetlb keeps the rows in this process; it cannot be combined with --weights-dir\n");
        return 2;
    }
    try {
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 1;
        cfg.world = 2;
        cfg.root_addr = root.c_str();
        cfg.timeout_ms = 0;
        cfg.backend = backend == "ib" ? UCOMM_BACKEND_IB : backend == "udp" ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
        ucomm_t* c = nullptr;
        ok(ucomm_init(&cfg, &c), "ucomm_init");
        std::fprintf(stderr, "DS41_WORKER connected backend=%s\n", ucomm_backend_name(c));
        std::vector<uint8_t> setup_msg(1 << 20);
        size_t n = 0;
        ok(ucomm_recv(c, 0, net::kTagSetup, setup_msg.data(), setup_msg.size(), &n), "setup");
        net::TpSetup setup;
        std::memcpy(&setup, setup_msg.data(), sizeof setup);
        if (setup.magic != net::kTpMagicDs41 || setup.weights != net::kWeightsStreamed ||
            n != sizeof setup + setup.layers * sizeof(net::TpLayer))
            throw std::runtime_error("bad setup message (a DS4.1 decoder with streamed weights is expected)");
        const int H = int(setup.hidden);
        std::vector<Layer> layers(setup.layers);
        size_t total = 0, widest = 0;
        int rows = 0;
        for (uint32_t i = 0; i < setup.layers; ++i) {
            net::TpLayer s;
            std::memcpy(&s, setup_msg.data() + sizeof setup + i * sizeof s, sizeof s);
            if (s.layer != int(i)) throw std::runtime_error("setup layers out of order");
            Layer& L = layers[i];
            std::string error;
            if (!cpu::native_fmt(s.gu_type, s.d_type, H, s.n_ff, L.f, error)) throw std::runtime_error(error);
            rows = s.n_ff - s.split;
            L.f.n_ff = rows;
            L.f.d_row = ggml_row_size(ggml_type(s.d_type), rows);
            L.f.h_bytes = ggml_row_size(ggml_type(L.f.d_act), rows);
            L.f.up_off = L.f.gu_row * size_t(rows);
            L.f.down_off = 2 * L.f.up_off;
            L.f.bytes = L.f.down_off + L.f.d_row * size_t(H);
            L.f.swiglu_limit = L.clamp = s.swiglu_limit;
            L.f.iq1s_rows = true;
            L.slot_bytes = net::tp_blob_stride(L.f.bytes);
            const size_t unit = std::lcm(std::lcm(size_t(ggml_type_size(ggml_type(s.gu_type))), size_t(ggml_type_size(ggml_type(s.d_type)))), size_t(256));
            L.pstride = (L.f.bytes + unit - 1) / unit * unit;
            L.experts = s.experts;
            total += size_t(L.experts) * L.slot_bytes;
            widest = std::max(widest, L.f.bytes);
        }
        // rows: kept in --weights-dir (same fingerprint) or received now
        net::Storage store;
        store.bytes = total;
        net::TpAccept accept{};
        accept.ok = 1;
        accept.needed_bytes = total;
        if (!weights_dir.empty() && setup.fingerprint) {
            uint64_t key = net::fnv(&setup.fingerprint, sizeof setup.fingerprint);
            key = net::fnv(&setup.hidden, sizeof setup.hidden, key);
            key = net::fnv(setup_msg.data() + sizeof setup, setup.layers * sizeof(net::TpLayer), key);
            char name[64];
            std::snprintf(name, sizeof name, "strata-tp-%016llx.weights", (unsigned long long)key);
            store.path = weights_dir + "/" + name;
            struct stat st{};
            if (stat(store.path.c_str(), &st) == 0 && size_t(st.st_size) == total) {
                store.map_file(store.path, false);
                accept.have = 1;
            }
        }
        if (!accept.have) {
            size_t available = net::mem_available();
            std::string where = "MemAvailable";
            if (!store.path.empty()) {
                for (const auto& entry : std::filesystem::directory_iterator(weights_dir)) {
                    const auto file = entry.path().filename().string();
                    if (file.starts_with("strata-tp-") && (file.ends_with(".weights") || file.ends_with(".weights.partial"))) {
                        std::fprintf(stderr, "DS41_WORKER removing %s (other rows)\n", entry.path().c_str());
                        std::filesystem::remove(entry.path());
                    }
                }
                available = net::mem_available();
                struct statvfs fs{};
                if (statvfs(weights_dir.c_str(), &fs) != 0) throw std::runtime_error("cannot inspect " + weights_dir);
                const size_t free_bytes = size_t(fs.f_bavail) * fs.f_frsize;
                if (free_bytes < available) { available = free_bytes; where = weights_dir + " free space"; }
            }
            const size_t reserve = size_t(reserve_gib * net::kGiB);
            if (rows_mode == "hugetlb") {
                // the pool is set aside already: the rows need its free pages, and --reserve-gib of other RAM stays free
                accept.available_bytes = net::hugetlb_available();
                if (available < reserve) {
                    accept.ok = 0;
                    std::snprintf(accept.reason, sizeof accept.reason, "MemAvailable %.2f GiB (outside the hugetlb pool) is below --reserve-gib=%.1f",
                                  available / net::kGiB, reserve_gib);
                } else if (total > accept.available_bytes) {
                    accept.ok = 0;
                    std::snprintf(accept.reason, sizeof accept.reason, "worker rows need %.2f GiB; the 2 MiB hugetlb pool has %.2f GiB free (vm.nr_hugepages)",
                                  total / net::kGiB, accept.available_bytes / net::kGiB);
                }
            } else {
                accept.available_bytes = available > reserve ? available - reserve : 0;
                if (total > accept.available_bytes) {
                    accept.ok = 0;
                    std::snprintf(accept.reason, sizeof accept.reason, "worker rows need %.2f GiB; %s minus --reserve-gib=%.1f leaves %.2f GiB",
                                  total / net::kGiB, where.c_str(), reserve_gib, accept.available_bytes / net::kGiB);
                }
            }
        }
        // Without --weights-dir the rows live in this process: by default page-locked host memory (cudaHostAlloc),
        // one buffer per layer, all allocated before the decoder is told to send. A shortfall then refuses the setup
        // instead of the OOM killer ending the transfer at 40+ GB, and the GPU and IB read the buffers without a
        // later 40 GB registration. --rows=anon keeps lazily faulted anonymous memory (the earlier behaviour).
        // --rows=hugetlb puts them on the 2 MiB hugetlb pool, registered with cudaHostRegister after the fill: a 60 GiB
        // cudaHostAlloc on the 62.7 GiB B550 left 0.35 GiB free (2026-10-25); the pool's pages skip most struct-page
        // memory (vm.hugetlb_optimize_vmemmap) and the reclaim lists.
        const bool pinned_rows = store.path.empty() && rows_mode == "pinned";
        const bool huge_rows = store.path.empty() && rows_mode == "hugetlb";
        if (accept.ok && huge_rows && !store.map_hugetlb()) {   // mapped before the decoder sends: a shortfall refuses the setup
            accept.ok = 0;
            std::snprintf(accept.reason, sizeof accept.reason, "hugetlb rows: cannot map %.2f GiB of 2 MiB pages (%s; pool has %.2f GiB free)",
                          total / net::kGiB, std::strerror(errno), net::hugetlb_available() / net::kGiB);
        }
        std::vector<uint8_t*> pinned_layers;
        auto free_pinned = [&] { for (auto* p : pinned_layers) cudaFreeHost(p); pinned_layers.clear(); };
        if (accept.ok && pinned_rows) {
            const auto a0 = std::chrono::steady_clock::now();
            size_t got = 0;
            for (const auto& L : layers) {
                void* p = nullptr;
                if (cudaHostAlloc(&p, size_t(L.experts) * L.slot_bytes, cudaHostAllocPortable) != cudaSuccess) {
                    (void)cudaGetLastError();
                    accept.ok = 0;
                    std::snprintf(accept.reason, sizeof accept.reason,
                                  "page-locked rows: allocation failed after %.2f of %.2f GiB (MemAvailable %.2f GiB)", got / net::kGiB,
                                  total / net::kGiB, net::mem_available() / net::kGiB);
                    free_pinned();
                    break;
                }
                pinned_layers.push_back(static_cast<uint8_t*>(p));
                got += size_t(L.experts) * L.slot_bytes;
            }
            if (accept.ok)
                std::fprintf(stderr, "DS41_WORKER rows page-locked GiB=%.2f in %zu buffers (%.1f s)\n", got / net::kGiB, pinned_layers.size(),
                             since(a0) / 1e3);
        }
        ok(ucomm_send(c, 0, net::kTagAccept, &accept, sizeof accept), "accept");
        if (!accept.ok) throw std::runtime_error(accept.reason);
        const auto fill0 = std::chrono::steady_clock::now();
        if (!accept.have && !pinned_rows && !huge_rows) {
            if (store.path.empty()) store.map_anonymous();
            else store.map_file(store.path + ".partial", true);
        }
        size_t off = 0;
        for (size_t li = 0; li < layers.size(); ++li) {
            auto& L = layers[li];
            L.base = pinned_rows ? pinned_layers[li] : store.data + off;
            off += size_t(L.experts) * L.slot_bytes;
            if (accept.have) continue;
            ucomm_mr_t* region = nullptr;
            if (ucomm_mr_reg(c, L.base, size_t(L.experts) * L.slot_bytes, &region) != UCOMM_OK) region = nullptr;
            for (int e = 0; e < L.experts; ++e) {
                size_t got = 0;
                ok(ucomm_recv(c, 0, net::kTagWeights, L.base + size_t(e) * L.slot_bytes, L.slot_bytes, &got), "weights");
                if (got != L.f.bytes) throw std::runtime_error("weights message size mismatch");
            }
            if (region) ucomm_mr_dereg(region);
        }
        if (!accept.have) store.complete();
        const double fill_ms = since(fill0);

        // GPU: the stored rows registered for DMA (tier uploads and prompt passes read them directly)
        cudaStream_t stream = nullptr, copy = nullptr;
        ck(cudaStreamCreate(&stream), "stream");
        ck(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "copy stream");
        const auto r0 = std::chrono::steady_clock::now();
        const bool registered = pinned_rows || cudaHostRegister(store.data, store.bytes, cudaHostRegisterPortable) == cudaSuccess;
        if (!registered) (void)cudaGetLastError();
        std::fprintf(stderr, "DS41_WORKER rows GiB=%.2f %s storage=%s fill_s=%.1f registered=%d (%.1f s)\n", total / net::kGiB,
                     accept.have ? "kept" : "received", pinned_rows ? "page-locked" : huge_rows ? "hugetlb 2 MiB" : store.path.empty() ? "anonymous" : store.path.c_str(),
                     fill_ms / 1e3, int(registered), since(r0) / 1e3);
        const int K_max = 16, maxT = 4096;
        // prompt-pass buffers: two layers of blobs (double-buffered), the routes' rows
        size_t layer_bytes = 0;
        for (const auto& L : layers) layer_bytes = std::max(layer_bytes, size_t(L.experts) * L.pstride);
        const bool use_mmq = mq::built();
        const int rcap = 8192;
        Dev lbuf[2] = {Dev(layer_bytes), Dev(layer_bytes)};
        Dev z_dev(size_t(maxT) * H * 4), out_dev(size_t(maxT) * H * 4);
        Dev xq(mq::q8_bytes(rcap, H)), gu_dev(size_t(rcap) * 2 * rows * 4), hf_dev(size_t(rcap) * rows * 4), hq(mq::q8_bytes(rcap, rows));
        Dev dout(size_t(rcap) * H * 4), idx_dev(size_t(maxT) * K_max * 4), w_dev(size_t(maxT) * K_max * 4), bound_dev(size_t(4096) * 4),
            iota_dev((size_t(rcap) + 128) * 4);
        if (use_mmq) mq::iota(iota_dev.i(), rcap + 128, stream);
        std::unique_ptr<mq::Context> mmq = use_mmq ? std::make_unique<mq::Context>() : nullptr;
        cudaEvent_t lready[2], lfree[2];
        for (int i = 0; i < 2; ++i) {
            ck(cudaEventCreateWithFlags(&lready[i], cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&lfree[i], cudaEventDisableTiming), "event");
        }
        int lbuf_layer[2] = {-1, -1};
        // decode buffers and the tier: one arena of equal slots (the widest blob)
        Dev xq8(k::native_q8_1_bytes(H, 1)), g_gate(size_t(rows) * 4), g_up(size_t(rows) * 4), g_h(size_t(rows) * 4),
            g_hq(k::native_q8_1_bytes(rows, 1)), g_out(size_t(H) * 4);
        const size_t slot = (widest + 255) / 256 * 256;
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "memory info");
        const size_t keep = size_t(gpu_reserve_mib) << 20;
        size_t nslots = free_b > keep ? (free_b - keep) / slot : 0;
        if (const char* v = std::getenv("STRATA_DS41_WORKER_TIER_SLOTS")) nslots = std::min(nslots, size_t(std::atoll(v)));
        Dev arena;
        while (nslots > 0) {
            void* p = nullptr;
            if (cudaMalloc(&p, nslots * slot) == cudaSuccess) { arena.p = p; arena.bytes = nslots * slot; break; }
            (void)cudaGetLastError();
            nslots = nslots * 15 / 16;
        }
        const int L_n = int(layers.size()), E = layers.empty() ? 0 : layers[0].experts;
        std::vector<int> slot_of(size_t(L_n) * E, -1), owner(nslots, -1);
        std::vector<char> pending(nslots, 0);
        std::vector<cudaEvent_t> done(nslots, nullptr);
        for (auto& e : done) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        std::vector<float> score(size_t(L_n) * E, 0.0f);
        const float decay = 0.995f;
        const int promote = std::getenv("STRATA_DS41_WORKER_PROMOTE") ? std::atoi(std::getenv("STRATA_DS41_WORKER_PROMOTE")) : 4;
        const int mmq_min = std::getenv("STRATA_DS41_WORKER_MMQ_MIN") ? std::atoi(std::getenv("STRATA_DS41_WORKER_MMQ_MIN")) : 64;
        bool reseed = false;
        auto slot_ptr = [&](int s) { return arena.b(size_t(s) * slot); };
        auto upload_blob = [&](int s, int l, int e) {   // async, on `copy`
            const Layer& L = layers[size_t(l)];
            ck(cudaMemcpyAsync(slot_ptr(s), L.base + size_t(e) * L.slot_bytes, L.f.bytes, cudaMemcpyHostToDevice, copy), "tier upload");
            ck(cudaEventRecord(done[size_t(s)], copy), "tier event");
        };
        // fill the tier with the highest scores (after a prompt pass)
        auto refill = [&] {
            std::vector<int> order(score.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[size_t(a)] > score[size_t(b)]; });
            std::fill(slot_of.begin(), slot_of.end(), -1);
            std::fill(owner.begin(), owner.end(), -1);
            size_t s = 0;
            for (int i : order) {
                if (s >= nslots || score[size_t(i)] <= 0) break;
                owner[s] = i;
                slot_of[size_t(i)] = int(s);
                upload_blob(int(s), i / E, i % E);
                ++s;
            }
            ck(cudaStreamSynchronize(copy), "tier fill");
            std::fill(pending.begin(), pending.end(), 0);
            std::fprintf(stderr, "DS41_WORKER tier refilled slots=%zu/%zu GiB=%.2f\n", s, nslots, double(s) * slot / net::kGiB);
        };
        std::fprintf(stderr, "DS41_WORKER gpu tier slots=%zu GiB=%.2f mmq=%d layer_buffers_MiB=%.0f\n", nslots, double(nslots) * slot / net::kGiB,
                     int(use_mmq), 2.0 * layer_bytes / (1 << 20));

        auto lane = std::make_unique<strata::ds41::ExpertLane>(H, rows, threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency() / 2)), true);
        // request and reply buffers, registered (IB) and pinned (the reply's device copy)
        const size_t msg_bytes = sizeof(net::TpRequest) + size_t(maxT) * K_max * sizeof(net::TpJob) + size_t(maxT) * H * 4;
        uint8_t* msg = nullptr;
        float* reply = nullptr;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&msg), msg_bytes, cudaHostAllocDefault), "request buffer");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&reply), size_t(maxT) * H * 4, cudaHostAllocDefault), "reply buffer");
        std::vector<ucomm_mr_t*> regions;
        for (const auto& [p, bytes] : {std::pair<void*, size_t>{msg, msg_bytes}, {reply, size_t(maxT) * H * 4}}) {
            ucomm_mr_t* region = nullptr;
            if (ucomm_mr_reg(c, p, bytes, &region) == UCOMM_OK) regions.push_back(region);
        }
        net::TpReady ready{use_mmq ? 2u : 1u, uint32_t(nslots), fill_ms / 1e3};
        ok(ucomm_send(c, 0, net::kTagReady, &ready, sizeof ready), "ready");

        std::vector<int32_t> ids, cpu_ids;
        std::vector<float> wts, cpu_y, gpu_y;
        std::vector<int> hidx, hbound;
        std::vector<float> hw;
        int64_t requests = 0, prompt_layers = 0;
        double busy_ms = 0, prompt_ms = 0, fetch_ms = 0, hits = 0, routes = 0;
        for (;;) {
            ok(ucomm_recv(c, 0, net::kTagRequest, msg, msg_bytes, &n), "request");
            net::TpRequest rq;
            std::memcpy(&rq, msg, sizeof rq);
            if (rq.layer < 0) break;
            if (rq.layer >= L_n) throw std::runtime_error("request names an invalid layer");
            Layer& L = layers[size_t(rq.layer)];
            if (rq.njobs < 0) {   // fetch (a decoder without GPU prompt passes)
                const auto f0 = std::chrono::steady_clock::now();
                const int first = rq.nt, count = -rq.njobs;
                if (first < 0 || first + count > L.experts) throw std::runtime_error("fetch names invalid experts");
                ok(ucomm_send(c, 0, net::kTagFetch, L.base + size_t(first) * L.slot_bytes, size_t(count) * L.slot_bytes), "fetch");
                fetch_ms += since(f0);
                continue;
            }
            const int nt = rq.nt;
            if (nt < 1 || nt > maxT || rq.njobs % nt || rq.act_bytes != 4 * H || rq.njobs / nt > K_max) throw std::runtime_error("bad DS4.1 request");
            const int K = rq.njobs / nt;
            const auto* list = reinterpret_cast<const net::TpJob*>(msg + sizeof rq);
            const float* z = reinterpret_cast<const float*>(msg + sizeof rq + size_t(rq.njobs) * sizeof(net::TpJob));
            ids.resize(size_t(rq.njobs));
            wts.resize(size_t(rq.njobs));
            for (int j = 0; j < rq.njobs; ++j) {
                if (list[j].expert < -1 || list[j].expert >= L.experts || list[j].token != j / K) throw std::runtime_error("invalid route");
                ids[size_t(j)] = list[j].expert;
                wts[size_t(j)] = list[j].weight;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const float dl = std::pow(decay, float(nt) / float(L_n));   // per layer request: a token decays once over all layers
            for (int e = 0; e < E; ++e) score[size_t(rq.layer) * E + e] *= dl;
            for (int j = 0; j < rq.njobs; ++j) if (ids[size_t(j)] >= 0) score[size_t(rq.layer) * E + ids[size_t(j)]] += 1.0f;
            if (use_mmq && nt >= mmq_min) {
                // ---- a prompt pass: every route of the layer on the GPU (MMQ over this node's rows) ----
                const int b = rq.layer % 2;
                auto load = [&](int l, int buf) {   // the layer's blobs into a VRAM buffer, at MMQ's stride
                    const Layer& M = layers[size_t(l)];
                    if (lbuf_layer[buf] >= 0) ck(cudaStreamWaitEvent(copy, lfree[buf], 0), "buffer free");
                    ck(cudaMemcpy2DAsync(lbuf[buf].p, M.pstride, M.base, M.slot_bytes, M.f.bytes, size_t(M.experts), cudaMemcpyHostToDevice, copy), "layer upload");
                    ck(cudaEventRecord(lready[buf], copy), "layer event");
                    lbuf_layer[buf] = l;
                };
                if (lbuf_layer[b] != rq.layer) load(rq.layer, b);
                ck(cudaStreamWaitEvent(stream, lready[b], 0), "layer ready");
                if (rq.layer + 1 < L_n && lbuf_layer[1 - b] != rq.layer + 1) load(rq.layer + 1, 1 - b);   // the next layer meanwhile
                // routes grouped by expert
                std::vector<std::vector<int>> tok(static_cast<size_t>(E));
                std::vector<std::vector<float>> wt(static_cast<size_t>(E));
                for (int j = 0; j < rq.njobs; ++j) {
                    if (ids[size_t(j)] < 0) continue;
                    tok[size_t(ids[size_t(j)])].push_back(j / K);
                    wt[size_t(ids[size_t(j)])].push_back(wts[size_t(j)]);
                }
                std::vector<int> row0(static_cast<size_t>(E) + 1, 0);
                hidx.clear();
                hw.clear();
                for (int e = 0; e < E; ++e) {
                    row0[size_t(e) + 1] = row0[size_t(e)] + int(tok[size_t(e)].size());
                    hidx.insert(hidx.end(), tok[size_t(e)].begin(), tok[size_t(e)].end());
                    hw.insert(hw.end(), wt[size_t(e)].begin(), wt[size_t(e)].end());
                }
                struct Prod { int e0, e1, row_first, rows, bound0, max_rows; };
                std::vector<Prod> prods;
                hbound.clear();
                for (int e0 = 0; e0 < E;) {
                    int e1 = e0, r = 0, mx = 0;
                    while (e1 < E && (e1 == e0 || r + int(tok[size_t(e1)].size()) <= rcap)) {
                        r += int(tok[size_t(e1)].size());
                        mx = std::max(mx, int(tok[size_t(e1)].size()));
                        ++e1;
                    }
                    Prod p{e0, e1, row0[size_t(e0)], r, int(hbound.size()), mx};
                    for (int e = e0; e <= e1; ++e) hbound.push_back(row0[size_t(e)] - p.row_first);
                    prods.push_back(p);
                    e0 = e1;
                }
                if (hbound.size() * 4 > bound_dev.bytes) bound_dev = Dev(hbound.size() * 4);
                const int total_rows = row0[size_t(E)];
                ck(cudaMemcpyAsync(z_dev.p, z, size_t(nt) * H * 4, cudaMemcpyHostToDevice, stream), "z");
                ck(cudaMemcpyAsync(idx_dev.p, hidx.data(), size_t(total_rows) * 4, cudaMemcpyHostToDevice, stream), "rows");
                ck(cudaMemcpyAsync(w_dev.p, hw.data(), size_t(total_rows) * 4, cudaMemcpyHostToDevice, stream), "weights");
                ck(cudaMemcpyAsync(bound_dev.p, hbound.data(), hbound.size() * 4, cudaMemcpyHostToDevice, stream), "bounds");
                ck(cudaMemsetAsync(out_dev.p, 0, size_t(nt) * H * 4, stream), "sum");
                const uint8_t* lb = lbuf[b].b();
                for (const Prod& p : prods) {
                    if (p.rows == 0) continue;
                    const int32_t* rtok = idx_dev.i() + p.row_first;
                    const float* rw = w_dev.f(p.row_first);
                    mq::quantize(z_dev.f(), rtok, xq.p, L.f.gu_type, H, H, p.rows, stream);
                    mq::Product gu;
                    gu.w = lb + size_t(p.e0) * L.pstride; gu.type = L.f.gu_type; gu.w_rows = 2 * rows; gu.w_cols = H; gu.expert_bytes = L.pstride;
                    gu.n = p.e1 - p.e0; gu.xq = xq.p; gu.bounds = bound_dev.i() + p.bound0; gu.ids = iota_dev.i(); gu.total_rows = p.rows;
                    gu.max_rows = p.max_rows; gu.dst = gu_dev.f(); gu.ld_dst = 2 * rows;
                    mmq->run(gu, stream);
                    mq::swiglu(gu_dev.f(), hf_dev.f(), p.rows, rows, false, stream, L.clamp);
                    mq::quantize(hf_dev.f(), nullptr, hq.p, L.f.d_type, rows, rows, p.rows, stream);
                    mq::Product dn = gu;
                    dn.w = lb + size_t(p.e0) * L.pstride + L.f.down_off; dn.type = L.f.d_type; dn.w_rows = H; dn.w_cols = rows; dn.xq = hq.p;
                    dn.dst = dout.f(); dn.ld_dst = H;
                    mmq->run(dn, stream);
                    k::ds41_scatter_axpy(dout.f(), rtok, rw, p.rows, H, out_dev.f(), stream);
                }
                ck(cudaEventRecord(lfree[b], stream), "buffer used");
                ck(cudaMemcpyAsync(reply, out_dev.p, size_t(nt) * H * 4, cudaMemcpyDeviceToHost, stream), "reply");
                ck(cudaStreamSynchronize(stream), "prompt layer");
                prompt_ms += since(t0);
                ++prompt_layers;
                reseed = true;
            } else {
                // ---- decode: resident experts on the GPU, the others on the CPU lane meanwhile ----
                if (reseed && nslots) { refill(); reseed = false; }
                for (size_t s = 0; s < nslots; ++s)   // finished promotions join the tier
                    if (pending[s] && cudaEventQuery(done[s]) == cudaSuccess) { pending[s] = 0; slot_of[size_t(owner[s])] = int(s); }
                cpu_ids = ids;
                bool any_gpu = false;
                ck(cudaMemcpyAsync(z_dev.p, z, size_t(nt) * H * 4, cudaMemcpyHostToDevice, stream), "z");
                for (int t = 0; t < nt; ++t) {
                    bool quantized = false;
                    for (int kk = 0; kk < K; ++kk) {
                        const int j = t * K + kk, e = ids[size_t(j)];
                        if (e < 0) continue;
                        ++routes;
                        const int s = slot_of[size_t(rq.layer) * E + e];
                        if (s < 0) continue;
                        ++hits;
                        if (!any_gpu) { ck(cudaMemsetAsync(out_dev.p, 0, size_t(nt) * H * 4, stream), "sum"); any_gpu = true; }
                        if (!quantized) { k::native_quantize_q8_1(z_dev.f(size_t(t) * H), xq8.p, H, 1, stream); quantized = true; }
                        const uint8_t* base = slot_ptr(s);
                        k::native_mmvq(L.f.gu_type, base, xq8.p, g_gate.f(), H, rows, 1, stream);
                        k::native_mmvq(L.f.gu_type, base + L.f.up_off, xq8.p, g_up.f(), H, rows, 1, stream);
                        k::glm_swiglu(g_gate.f(), g_up.f(), g_h.f(), rows, L.clamp, stream);
                        k::native_quantize_q8_1(g_h.f(), g_hq.p, rows, 1, stream);
                        k::native_mmvq(L.f.d_type, base + L.f.down_off, g_hq.p, g_out.f(), rows, H, 1, stream);
                        k::ds41_axpy(g_out.f(), wts[size_t(j)], out_dev.f(size_t(t) * H), H, stream);
                        cpu_ids[size_t(j)] = -1;
                    }
                }
                if (any_gpu) ck(cudaMemcpyAsync(reply, out_dev.p, size_t(nt) * H * 4, cudaMemcpyDeviceToHost, stream), "gpu sum");
                cpu_y.resize(size_t(nt) * H);
                strata::ds41::LaneLayer view;
                view.f = L.f;
                view.gate = L.base;
                view.up = L.base + L.f.up_off;
                view.down = L.base + L.f.down_off;
                view.gate_stride = view.down_stride = L.slot_bytes;
                lane->run(view, z, nt, cpu_ids.data(), wts.data(), K, cpu_y.data());
                if (any_gpu) {
                    ck(cudaStreamSynchronize(stream), "decode");
                    for (size_t i = 0; i < size_t(nt) * H; ++i) reply[i] += cpu_y[i];
                } else {
                    std::memcpy(reply, cpu_y.data(), size_t(nt) * H * 4);
                }
                // promotions: missed experts with higher scores replace the lowest-scored slots
                int done_n = 0;
                for (int j = 0; j < rq.njobs && done_n < promote && nslots; ++j) {
                    const int e = ids[size_t(j)];
                    if (e < 0) continue;
                    const int cand = rq.layer * E + e;
                    if (slot_of[size_t(cand)] >= 0) continue;
                    bool in_flight = false;
                    for (size_t s = 0; s < nslots && !in_flight; ++s) in_flight = pending[s] && owner[s] == cand;
                    if (in_flight) continue;
                    int victim = -1;
                    float low = score[size_t(cand)];
                    for (size_t s = 0; s < nslots; ++s) {
                        if (pending[s]) continue;
                        const float v = owner[s] < 0 ? -1.0f : score[size_t(owner[s])];
                        if (v < low) { low = v; victim = int(s); }
                    }
                    if (victim < 0) continue;
                    if (owner[size_t(victim)] >= 0) slot_of[size_t(owner[size_t(victim)])] = -1;
                    owner[size_t(victim)] = cand;
                    pending[size_t(victim)] = 1;
                    upload_blob(victim, rq.layer, e);
                    ++done_n;
                }
                busy_ms += since(t0);
            }
            ++requests;
            ok(ucomm_send(c, 0, net::kTagReply, reply, size_t(nt) * H * 4), "reply");
        }
        std::fprintf(stderr, "DS41_WORKER done requests=%lld decode_ms=%.1f prompt_layers=%lld prompt_ms=%.1f tier_hit=%.3f fetch_s=%.2f\n",
                     (long long)requests, busy_ms, (long long)prompt_layers, prompt_ms, routes > 0 ? hits / routes : 0.0, fetch_ms / 1e3);
        for (auto* region : regions) ucomm_mr_dereg(region);
        if (registered && !pinned_rows) cudaHostUnregister(store.data);
        free_pinned();
        ucomm_finalize(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DS41_WORKER error: %s\n", e.what());
        return 1;
    }
    return 0;
}
