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
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
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
#include <map>
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
    std::vector<int> local;   // STD2: global expert id -> stored blob (-1: not owned)
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
    int stream_q = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a.starts_with("--root=")) root = a.substr(7);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else if (a.starts_with("--weights-dir=")) weights_dir = a.substr(14);
        else if (a.starts_with("--reserve-gib=")) reserve_gib = std::stod(a.substr(14));
        else if (a.starts_with("--gpu-reserve-mib=")) gpu_reserve_mib = std::stod(a.substr(18));
        else if (a.starts_with("--stream=")) stream_q = std::stoi(a.substr(9));   // STD2: experts DMAed to the GPU per request
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
        // STD1: rows [split, ff) of every expert; STD2 (whole-expert ownership): whole blobs of the owned experts
        const bool own = setup.magic == net::kTpMagicDs41Own;
        if ((setup.magic != net::kTpMagicDs41 && !own) || setup.weights != net::kWeightsStreamed || setup.layers > 4096 ||
            n < sizeof setup + setup.layers * sizeof(net::TpLayer))
            throw std::runtime_error("bad setup message (a DS4.1 decoder with streamed weights is expected)");
        size_t owned_ids = 0;
        for (uint32_t i = 0; i < setup.layers; ++i) {
            net::TpLayer s;
            std::memcpy(&s, setup_msg.data() + sizeof setup + i * sizeof s, sizeof s);
            if (own && (s.experts < 0 || s.split != 0 || s.pad < s.experts || s.pad > 65536))
                throw std::runtime_error("bad STD2 layer: owned count, split or expert count");
            owned_ids += own ? size_t(s.experts) : 0;
        }
        const bool own_counts = own && setup.pad == 1;   // STD2: the owned experts' profile counts follow the ids
        if (n != sizeof setup + setup.layers * sizeof(net::TpLayer) + owned_ids * sizeof(int32_t) * (own_counts ? 2 : 1) ||
            (own && setup.reply != net::kReplyRaw))
            throw std::runtime_error("bad setup message size or reply kind");
        const int32_t* owned_list = reinterpret_cast<const int32_t*>(setup_msg.data() + sizeof setup + setup.layers * sizeof(net::TpLayer));
        const float* owned_counts = own_counts ? reinterpret_cast<const float*>(owned_list + owned_ids) : nullptr;
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
            if (own) {
                L.local.assign(size_t(s.pad), -1);
                for (int j = 0; j < s.experts; ++j) {
                    const int e = *owned_list++;
                    if (e < 0 || e >= s.pad || L.local[size_t(e)] >= 0 || (j && e <= owned_list[-2]))
                        throw std::runtime_error("STD2 owned expert ids must be ascending and in range");
                    L.local[size_t(e)] = j;
                }
            }
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
            key = net::fnv(setup_msg.data() + sizeof setup, n - sizeof setup, key);   // the layers (and STD2's owned ids)
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
        const bool use_mmq = !own && mq::built();   // STD2: no prompt MMQ (its prompt passes run on the CPU lane)
        const int rcap = 8192;
        Dev lbuf[2] = {Dev(use_mmq ? layer_bytes : 0), Dev(use_mmq ? layer_bytes : 0)};
        // STD2 uses z_dev for at most 64 tokens and none of the STD1 prompt / decode buffers (VRAM for its tier)
        const size_t zt = own ? 64 : size_t(maxT), rc = own ? 1 : size_t(rcap);
        Dev z_dev(zt * H * 4), out_dev(zt * H * 4);
        Dev xq(mq::q8_bytes(int(rc), H)), gu_dev(rc * 2 * rows * 4), hf_dev(rc * rows * 4), hq(mq::q8_bytes(int(rc), rows));
        Dev dout(rc * H * 4), idx_dev(size_t(maxT) * K_max * 4), w_dev(size_t(maxT) * K_max * 4), bound_dev(size_t(4096) * 4),
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
        // STD2: a tier of whole owned experts run with the CPU lane's arithmetic (the canonical kernels), so where an
        // expert runs never changes a value; STRATA_DS41_WORKER_TIER_SLOTS=0 keeps it off
        // STD2 --stream=q: a ring of q slots the widest non-resident experts of a request are DMAed into (from the
        // registered rows) and run with the same kernels, beside the CPU lane
        if (stream_q < 0 || stream_q > 16) throw std::runtime_error("--stream must be 0 through 16");
        if (!own || !registered) stream_q = 0;
        Dev ring(size_t(stream_q) * slot);
        if (stream_q && nslots > size_t(stream_q)) nslots -= size_t(stream_q);
        cudaEvent_t ring_ready = nullptr;
        ck(cudaEventCreateWithFlags(&ring_ready, cudaEventDisableTiming), "ring event");
        // slots: uniform (STD1), or STD2's per-layer pools of that layer's blob size, each layer's share of the VRAM
        // following its share of the profile's expected bytes (count x blob), at most its owned count
        std::vector<size_t> slot_off;
        std::vector<int> slot_layer;
        size_t arena_bytes = 0;
        auto plan = [&](size_t budget) {
            slot_off.clear(); slot_layer.clear(); arena_bytes = 0;
            if (!own) {
                const size_t n2 = budget / slot;
                for (size_t i = 0; i < n2; ++i) { slot_off.push_back(i * slot); slot_layer.push_back(-1); }
                arena_bytes = n2 * slot;
                return;
            }
            std::vector<double> w(layers.size(), 0.0);
            double wt = 0;
            for (size_t l = 0; l < layers.size(); ++l) {
                const Layer& M = layers[l];
                double c = 0;
                if (owned_counts) {
                    size_t at = 0;
                    for (size_t m = 0; m < l; ++m) at += size_t(layers[m].experts);
                    for (int j = 0; j < M.experts; ++j) c += owned_counts[at + size_t(j)];
                } else c = M.experts;
                w[l] = c * double(M.f.bytes);
                wt += w[l];
            }
            for (size_t l = 0; l < layers.size(); ++l) {
                const Layer& M = layers[l];
                const size_t ls = (M.f.bytes + 255) / 256 * 256;
                size_t k2 = wt > 0 ? size_t(double(budget) * w[l] / wt / double(ls)) : 0;
                k2 = std::min(k2, size_t(M.experts));
                for (size_t i = 0; i < k2; ++i) { slot_off.push_back(arena_bytes); slot_layer.push_back(int(l)); arena_bytes += ls; }
            }
        };
        {
            size_t budget = nslots * slot;
            if (const char* v = std::getenv("STRATA_DS41_WORKER_TIER_SLOTS")) budget = std::min(budget, size_t(std::atoll(v)) * slot);
            plan(budget);
        }
        Dev arena;
        while (arena_bytes > 0) {
            void* p = nullptr;
            if (cudaMalloc(&p, arena_bytes) == cudaSuccess) { arena.p = p; arena.bytes = arena_bytes; break; }
            (void)cudaGetLastError();
            plan(arena_bytes * 15 / 16);
        }
        nslots = slot_off.size();
        const int L_n = int(layers.size()), E = layers.empty() ? 0 : own ? int(layers[0].local.size()) : layers[0].experts;
        if (own) for (const auto& M : layers) if (int(M.local.size()) != E) throw std::runtime_error("STD2 layers differ in expert count");
        std::vector<int> slot_of(size_t(L_n) * E, -1), owner(nslots, -1);
        std::vector<char> pending(nslots, 0);
        std::vector<cudaEvent_t> done(nslots, nullptr);
        for (auto& e : done) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        std::vector<float> score(size_t(L_n) * E, 0.0f);
        const float decay = 0.995f;
        const int promote = std::getenv("STRATA_DS41_WORKER_PROMOTE") ? std::atoi(std::getenv("STRATA_DS41_WORKER_PROMOTE")) : 4;
        const int mmq_min = std::getenv("STRATA_DS41_WORKER_MMQ_MIN") ? std::atoi(std::getenv("STRATA_DS41_WORKER_MMQ_MIN")) : 64;
        bool reseed = false;
        auto slot_ptr = [&](int s) { return arena.b(slot_off[size_t(s)]); };
        auto upload_blob = [&](int s, int l, int e) {   // async, on `copy` (e: the expert id; STD2 maps it to its blob)
            const Layer& L = layers[size_t(l)];
            if (own) e = L.local.at(size_t(e));
            if (e < 0) throw std::runtime_error("tier upload of an expert this worker does not own");
            ck(cudaMemcpyAsync(slot_ptr(s), L.base + size_t(e) * L.slot_bytes, L.f.bytes, cudaMemcpyHostToDevice, copy), "tier upload");
            ck(cudaEventRecord(done[size_t(s)], copy), "tier event");
        };
        // fill the tier with the highest scores (after a prompt pass)
        auto refill = [&] {
            std::vector<int> order(score.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[size_t(a)] > score[size_t(b)]; });
            ck(cudaStreamSynchronize(copy), "tier promotions");   // every pending upload done
            std::fill(pending.begin(), pending.end(), 0);
            // the wanted experts: the top of each pool's layer (one pool when uniform)
            std::vector<size_t> cap(size_t(L_n) + 1, 0);
            for (size_t s2 = 0; s2 < nslots; ++s2) ++cap[size_t(slot_layer[s2] + 1)];
            std::vector<size_t> taken(size_t(L_n) + 1, 0);
            std::vector<char> want(score.size(), 0);
            for (int i : order) {
                if (score[size_t(i)] <= 0) break;
                const size_t pl = size_t(own ? i / E + 1 : 0);
                if (taken[pl] < cap[pl]) { want[size_t(i)] = 1; ++taken[pl]; }
            }
            // residents still wanted keep their slots; the others are freed, then the newcomers move in
            std::vector<std::vector<int>> free_of(size_t(L_n) + 1);
            for (size_t s2 = nslots; s2-- > 0;) {
                const int o = owner[s2];
                if (o >= 0 && want[size_t(o)] && slot_of[size_t(o)] == int(s2)) continue;
                if (o >= 0 && slot_of[size_t(o)] == int(s2)) slot_of[size_t(o)] = -1;
                owner[s2] = -1;
                free_of[size_t(slot_layer[s2] + 1)].push_back(int(s2));
            }
            size_t s = 0;
            for (int i : order) {
                if (score[size_t(i)] <= 0) break;
                if (!want[size_t(i)] || slot_of[size_t(i)] >= 0) continue;
                auto& pool = free_of[size_t(own ? i / E + 1 : 0)];
                if (pool.empty()) continue;
                const int sl = pool.back();
                pool.pop_back();
                owner[size_t(sl)] = i;
                slot_of[size_t(i)] = sl;
                upload_blob(sl, i / E, i % E);
                ++s;
            }
            ck(cudaStreamSynchronize(copy), "tier fill");
            std::fill(pending.begin(), pending.end(), 0);
            std::fprintf(stderr, "DS41_WORKER tier refilled uploaded=%zu slots=%zu GiB=%.2f\n", s, nslots, double(arena_bytes) / net::kGiB);
        };
        std::fprintf(stderr, "DS41_WORKER gpu tier slots=%zu GiB=%.2f mmq=%d layer_buffers_MiB=%.0f\n", nslots, double(arena_bytes) / net::kGiB,
                     int(use_mmq), 2.0 * layer_bytes / (1 << 20));

        auto lane = std::make_unique<strata::ds41::ExpertLane>(H, rows, threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency() / 2)), true);
        // STD2 GPU tier scratch: passes of up to 64 tokens (decode and prompt chunks) x K_max routes
        const int CT = 64, CE = CT * K_max;
        Dev c_xq, c_gu, c_h, c_hq, c_out, c_meta;
        float* c_res = nullptr;
        uint8_t* c_meta_host = nullptr;
        std::vector<float> cpu_rows;
        if (own) {
            if (H % 256 || rows % 256 || H > 5120 || rows > 5120) throw std::runtime_error("STD2 geometry outside the canonical kernels");
            c_xq = Dev(size_t(CT) * (H / 256) * 292);
            c_gu = Dev(size_t(2) * CE * rows * 4);
            c_h = Dev(size_t(CE) * rows * 4);
            c_hq = Dev(size_t(CE) * (rows / 256) * 292);
            c_out = Dev(size_t(CE) * H * 4);
            c_meta = Dev(size_t(2) * CE * 8 + size_t(3 * CE + 2) * 4);
            ck(cudaHostAlloc(reinterpret_cast<void**>(&c_res), size_t(CE) * H * 4, cudaHostAllocDefault), "canonical results");
            ck(cudaHostAlloc(reinterpret_cast<void**>(&c_meta_host), c_meta.bytes, cudaHostAllocDefault), "canonical metadata");
            cpu_rows.resize(size_t(maxT) * H);
        }
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
        if (owned_counts && nslots) {   // STD2: the tier starts from the decoder's profile
            // the counts are stored layer by layer in id order
            size_t at = 0;
            for (int l = 0; l < L_n; ++l) {
                for (int e = 0; e < E; ++e) {
                    const int j = layers[size_t(l)].local[size_t(e)];
                    if (j >= 0) score[size_t(l) * E + e] = owned_counts[at + size_t(j)];
                }
                at += size_t(layers[size_t(l)].experts);
            }
            refill();
        }
        net::TpReady ready{use_mmq ? 2u : 1u, uint32_t(nslots), fill_ms / 1e3};
        ok(ucomm_send(c, 0, net::kTagReady, &ready, sizeof ready), "ready");

        std::vector<int32_t> ids, cpu_ids;
        std::vector<float> wts, cpu_y, gpu_y;
        std::vector<int> hidx, hbound;
        std::vector<float> hw;
        int64_t requests = 0, prompt_layers = 0;
        double dec_ms[4] = {0, 0, 0, 0}, dec_routes[2] = {0, 0}, dec_pre[2] = {0, 0}, enq_ms[5] = {0, 0, 0, 0, 0};   // STD2 decode requests: total, lane, GPU wait, count
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
            if (own) {   // STD2: the owned routes' raw outputs, in route order
                const auto t0 = std::chrono::steady_clock::now();
                ids.assign(size_t(rq.njobs), -1);
                std::vector<int> glob(size_t(rq.njobs), -1);
                int count = 0;
                for (int j = 0; j < rq.njobs; ++j) {
                    const int e = list[j].expert;
                    if (e < -1 || e >= int(L.local.size()) || list[j].token != j / K) throw std::runtime_error("invalid STD2 route");
                    if (e < 0) continue;
                    if (L.local[size_t(e)] < 0) throw std::runtime_error("STD2 route names an expert this worker does not own");
                    ids[size_t(j)] = L.local[size_t(e)];
                    glob[size_t(j)] = e;
                    ++count;
                }
                if (size_t(count) * H > size_t(maxT) * H) throw std::runtime_error("STD2 reply exceeds its buffer");
                const float dl = std::pow(decay, float(nt) / float(L_n));
                for (int e = 0; e < E; ++e) score[size_t(rq.layer) * E + e] *= dl;
                for (int j = 0; j < rq.njobs; ++j) if (glob[size_t(j)] >= 0) score[size_t(rq.layer) * E + glob[size_t(j)]] += 1.0f;
                // tier-held routes on the GPU (passes of up to 8 tokens), the others on the CPU lane meanwhile
                cpu_ids = ids;
                std::vector<int> gpos(size_t(rq.njobs), -1);
                int ngpu = 0;
                const auto tm = std::chrono::steady_clock::now();   // routes mapped, scores updated
                if (nt > 8) reseed = true;   // a prompt pass: the tier refills from the scores before decode
                else if (reseed && nslots) { refill(); reseed = false; }
                // STRATA_DS41_WORKER_GPU_MIN_NT (1): passes of fewer tokens stay on the CPU lane (values are the same)
                static const int gpu_min_nt = std::getenv("STRATA_DS41_WORKER_GPU_MIN_NT") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_WORKER_GPU_MIN_NT"))) : 1;
                if ((nslots || stream_q) && nt <= CT && nt >= gpu_min_nt) {
                    for (size_t s2 = 0; s2 < nslots; ++s2)
                        if (pending[s2] && cudaEventQuery(done[s2]) == cudaSuccess) { pending[s2] = 0; slot_of[size_t(owner[s2])] = int(s2); }
                    std::map<int, std::vector<int>> held, rest;
                    for (int j = 0; j < rq.njobs; ++j) {
                        if (glob[size_t(j)] < 0) continue;
                        const bool res = nslots && slot_of[size_t(rq.layer) * E + glob[size_t(j)]] >= 0;
                        (res ? held : rest)[glob[size_t(j)]].push_back(j);
                    }
                    std::vector<std::pair<int, std::vector<int>>> streamed(rest.begin(), rest.end());
                    std::stable_sort(streamed.begin(), streamed.end(), [](const auto& x, const auto& y) { return x.second.size() > y.second.size(); });
                    if (streamed.size() > size_t(stream_q)) streamed.resize(size_t(stream_q));
                    const int Gh = int(held.size()), G = Gh + int(streamed.size());
                    if (G) {
                        uint64_t* gp = reinterpret_cast<uint64_t*>(c_meta_host);
                        uint64_t* dp = gp + G;
                        int* gs = reinterpret_cast<int*>(dp + G);
                        int* ea = gs + G + 1;
                        int g = 0;
                        auto add = [&](const uint8_t* base, const std::vector<int>& js) {
                            gs[g] = ngpu;
                            gp[g] = uint64_t(reinterpret_cast<uintptr_t>(base));
                            dp[g] = gp[g] + L.f.down_off;
                            for (int j : js) { gpos[size_t(j)] = ngpu; ea[ngpu] = j / K; cpu_ids[size_t(j)] = -1; ++ngpu; }
                            ++g;
                        };
                        for (const auto& [e, js] : held) add(slot_ptr(slot_of[size_t(rq.layer) * E + e]), js);
                        for (size_t r2 = 0; r2 < streamed.size(); ++r2) {   // the DMA from the registered rows, on `copy`
                            uint8_t* dst = ring.b(r2 * slot);
                            ck(cudaMemcpyAsync(dst, L.base + size_t(L.local[size_t(streamed[r2].first)]) * L.slot_bytes, L.f.bytes,
                                               cudaMemcpyHostToDevice, copy), "stream upload");
                            add(dst, streamed[r2].second);
                        }
                        if (!streamed.empty()) ck(cudaEventRecord(ring_ready, copy), "ring event");
                        gs[G] = ngpu;
                        int* id = ea + ngpu;
                        for (int j = 0; j < ngpu; ++j) id[j] = j;
                        if (ngpu > CE) throw std::runtime_error("STD2 GPU entries exceed the scratch");
                        const size_t meta_bytes = size_t(2) * G * 8 + size_t(G + 1 + 2 * ngpu) * 4;
                        const auto q0 = std::chrono::steady_clock::now();
                        ck(cudaMemcpyAsync(c_meta.p, c_meta_host, meta_bytes, cudaMemcpyHostToDevice, stream), "canonical metadata");
                        const auto q1 = std::chrono::steady_clock::now();
                        const auto* d_gp = reinterpret_cast<const uint8_t* const*>(c_meta.p);
                        const auto* d_dp = d_gp + G;
                        const int* d_gs = reinterpret_cast<const int*>(c_meta.b(size_t(2) * G * 8));
                        const int* d_ea = d_gs + G + 1;
                        const int* d_id = d_ea + ngpu;
                        ck(cudaMemcpyAsync(z_dev.p, z, size_t(nt) * H * 4, cudaMemcpyHostToDevice, stream), "z");
                        k::quantize_q8_K(z_dev.f(), c_xq.b(), int64_t(nt) * H, stream);
                        const auto q2 = std::chrono::steady_clock::now();
                        auto chain = [&](int g0, int g1) {
                            if (g0 >= g1) return;
                            const int e0 = gs[g0], ne = gs[g1] - gs[g0];
                            k::ds41_canon_dot(L.f.gu_type, d_gp + g0, d_gs + g0, d_ea, g1 - g0, 2, c_xq.b(), c_xq.bytes, nt, c_gu.f(), c_gu.bytes,
                                              ngpu, H, rows, L.f.up_off, stream);
                            k::ds41_canon_swiglu(c_gu.f(size_t(e0) * rows), c_gu.f(size_t(ngpu + e0) * rows), c_h.f(size_t(e0) * rows), ne * rows,
                                                 L.clamp, stream);
                            k::quantize_q8_K(c_h.f(size_t(e0) * rows), c_hq.b(size_t(e0) * (rows / 256) * 292), int64_t(ne) * rows, stream);
                            k::ds41_canon_dot(L.f.d_type, d_dp + g0, d_gs + g0, d_id, g1 - g0, 1, c_hq.b(), c_hq.bytes, ngpu, c_out.f(),
                                              c_out.bytes, ngpu, rows, H, 0, stream);
                        };
                        chain(0, Gh);
                        if (G > Gh) { ck(cudaStreamWaitEvent(stream, ring_ready, 0), "ring wait"); chain(Gh, G); }
                        const auto q3 = std::chrono::steady_clock::now();
                        ck(cudaMemcpyAsync(c_res, c_out.p, size_t(ngpu) * H * 4, cudaMemcpyDeviceToHost, stream), "canonical results");
                        const auto q4 = std::chrono::steady_clock::now();
                        if (nt <= 8) {
                            enq_ms[0] += std::chrono::duration<double, std::milli>(q1 - q0).count();
                            enq_ms[1] += std::chrono::duration<double, std::milli>(q2 - q1).count();
                            enq_ms[2] += std::chrono::duration<double, std::milli>(q3 - q2).count();
                            enq_ms[3] += std::chrono::duration<double, std::milli>(q4 - q3).count();
                            enq_ms[4] += 1;
                        }
                    }
                }
                strata::ds41::LaneLayer view;
                view.f = L.f;
                view.gate = L.base;
                view.up = L.base + L.f.up_off;
                view.down = L.base + L.f.down_off;
                view.gate_stride = view.down_stride = L.slot_bytes;
                const auto c0 = std::chrono::steady_clock::now();
                lane->run_raw(view, z, nt, cpu_ids.data(), K, cpu_rows.data());
                const auto c1 = std::chrono::steady_clock::now();
                if (ngpu) ck(cudaStreamSynchronize(stream), "canonical tier");
                if (nt <= 8) {   // decode / verify requests: the split
                    dec_ms[0] += since(t0);                                                     // whole request (so far)
                    dec_ms[1] += std::chrono::duration<double, std::milli>(c1 - c0).count();    // CPU lane
                    dec_ms[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c1).count();   // GPU wait after the lane
                    dec_ms[3] += 1;
                    dec_pre[0] += std::chrono::duration<double, std::milli>(tm - t0).count();
                    dec_pre[1] += std::chrono::duration<double, std::milli>(c0 - tm).count();
                    dec_routes[0] += count; dec_routes[1] += ngpu;
                }
                size_t r = 0, ci = 0;
                for (int j = 0; j < rq.njobs; ++j) {
                    if (ids[size_t(j)] < 0) continue;
                    const float* src = gpos[size_t(j)] >= 0 ? c_res + size_t(gpos[size_t(j)]) * H : cpu_rows.data() + (ci++) * size_t(H);
                    std::memcpy(reply + (r++) * size_t(H), src, size_t(H) * 4);
                }
                hits += ngpu;
                routes += count;
                // promotions: missed experts with higher scores replace the lowest-scored slots
                int done_n = 0;
                for (int j = 0; j < rq.njobs && done_n < promote && nslots; ++j) {
                    const int e = glob[size_t(j)];
                    if (e < 0) continue;
                    const int cand = rq.layer * E + e;
                    if (slot_of[size_t(cand)] >= 0) continue;
                    bool in_flight = false;
                    for (size_t s2 = 0; s2 < nslots && !in_flight; ++s2) in_flight = pending[s2] && owner[s2] == cand;
                    if (in_flight) continue;
                    int victim = -1;
                    float low = score[size_t(cand)];
                    for (size_t s2 = 0; s2 < nslots; ++s2) {
                        if (pending[s2] || (slot_layer[s2] >= 0 && slot_layer[s2] != rq.layer)) continue;
                        const float v = owner[s2] < 0 ? -1.0f : score[size_t(owner[s2])];
                        if (v < low) { low = v; victim = int(s2); }
                    }
                    if (victim < 0) continue;
                    if (owner[size_t(victim)] >= 0) slot_of[size_t(owner[size_t(victim)])] = -1;
                    owner[size_t(victim)] = cand;
                    pending[size_t(victim)] = 1;
                    upload_blob(victim, rq.layer, e);
                    ++done_n;
                }
                busy_ms += since(t0);
                ++requests;
                ok(ucomm_send(c, 0, net::kTagReply, reply, size_t(count) * H * 4), "reply");
                continue;
            }
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
                    gu.max_rows = p.max_rows; gu.dst = gu_dev.f(); gu.ld_dst = 2 * rows; gu.dst_rows = p.rows;
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
        if (dec_ms[3] > 0)
            std::fprintf(stderr, "DS41_WORKER decode_requests=%.0f ms_per_request total=%.3f cpu_lane=%.3f gpu_wait_after_lane=%.3f routes_per_request=%.2f gpu_share=%.3f\n",
                         dec_ms[3], dec_ms[0] / dec_ms[3], dec_ms[1] / dec_ms[3], dec_ms[2] / dec_ms[3], dec_routes[0] / dec_ms[3],
                         dec_routes[0] > 0 ? dec_routes[1] / dec_routes[0] : 0.0);
        if (dec_ms[3] > 0)
            std::fprintf(stderr, "DS41_WORKER decode_pre_ms map_scores=%.3f gpu_enqueue=%.3f (meta %.3f z+quant %.3f chains %.3f d2h %.3f)\n",
                         dec_pre[0] / dec_ms[3], dec_pre[1] / dec_ms[3], enq_ms[0] / std::max(1.0, enq_ms[4]), enq_ms[1] / std::max(1.0, enq_ms[4]),
                         enq_ms[2] / std::max(1.0, enq_ms[4]), enq_ms[3] / std::max(1.0, enq_ms[4]));
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
