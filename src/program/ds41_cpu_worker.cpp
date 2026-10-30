// strata-ds41-cpu-worker: the whole-expert worker (STD2, include/strata/net/expert_tp.hpp) on a node without a GPU, for a
// decoder that runs on another node's GPU. It owns the experts the decoder names in its setup and reads their rows from
// its own copy of the model, so no expert crosses the network at setup; each decode or prompt request runs on the CPU lane
// (the canonical kernels' arithmetic, bit-identical to the decoder's own lane).
//   strata-ds41-cpu-worker --root HOST:PORT --model FIRST_SHARD.gguf [--threads=N] [--backend=auto|ib|udp]
//                          [--engram-port=P] [--engram-threads=N]
// --engram-port: a second ucomm context (rank 1 of HOST:P) serves row reads of the model's files, for a decoder that holds
// none of the Engram tables (strata-glm-decode with STRATA_DS41_ENGRAM_REMOTE=HOST:P).
#include "ggml.h"
#include "strata/core/model.hpp"
#include "strata/ds41/engram_wire.hpp"
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/net/expert_tp.hpp"
#include "ucomm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

namespace {
namespace cpu = strata::kernels::cpu;
namespace net = strata::net;
namespace ew = strata::ds41::engram_wire;

void ok(ucomm_status s, const char* what) {
    if (s != UCOMM_OK) throw std::runtime_error(std::string(what) + ": " + ucomm_strerror(s));
}
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// the logical CPUs of a NUMA node (/sys/devices/system/node/nodeN/cpulist), empty when the node does not exist
std::vector<int> node_cpus(int node) {
    std::vector<int> cpus;
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string list;
    if (!f || !std::getline(f, list)) return cpus;
    size_t at = 0;
    while (at < list.size()) {
        const size_t comma = std::min(list.find(',', at), list.size());
        const std::string part = list.substr(at, comma - at);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) cpus.push_back(c);
        at = comma + 1;
    }
    return cpus;
}
std::vector<int> g_helper_cpus;   // --helper-cpus: where the Engram, fetch and ucomm helper threads run, away from the lane's pinned cores
void pin_helper() {
    if (g_helper_cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : g_helper_cpus) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}
void pin_to(const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

struct Layer {
    cpu::NativeFmt f;
    size_t slot_bytes = 0;
    uint8_t* base = nullptr;
    int experts = 0;
    float clamp = 0;
    std::vector<int> local;   // global expert id -> stored blob (-1: not owned)
};

// the Engram row reads: shard number -> descriptor of this node's file, the pages read with buffered preads
class EngramServer {
  public:
    EngramServer(const std::string& first_shard, const std::string& root, int backend, int threads) : threads_(std::max(1, threads)) {
        const std::string name = std::filesystem::path(first_shard).filename().string();
        const size_t at = name.rfind("-00001-of-");
        if (at == std::string::npos) throw std::runtime_error("--engram-port needs a split model (NAME-00001-of-0000N.gguf)");
        const std::string stem = name.substr(0, at), tail = name.substr(at + 10);
        const int count = std::atoi(tail.c_str());
        for (int i = 1; i <= count; ++i) {
            char suffix[40];
            std::snprintf(suffix, sizeof suffix, "-%05d-of-%05d.gguf", i, count);
            const auto path = std::filesystem::path(first_shard).parent_path() / (stem + suffix);
            fds_.push_back(::open(path.c_str(), O_RDONLY));
        }
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 1;
        cfg.world = 2;
        cfg.root_addr = root.c_str();
        cfg.timeout_ms = 0;
        cfg.backend = backend == 1 ? UCOMM_BACKEND_IB : backend == 2 ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
        ok(ucomm_init(&cfg, &c_), "engram ucomm_init");
        thread_ = std::thread([this] { serve(); });
    }
    ~EngramServer() {
        if (thread_.joinable()) thread_.join();
        for (int fd : fds_) if (fd >= 0) ::close(fd);
    }
    void join() { if (thread_.joinable()) thread_.join(); }

  private:
    void serve() {
        pin_helper();
        std::vector<uint8_t> request(ew::kMaxRequestBytes), reply;
        double reads = 0, ms = 0;
        for (;;) {
            size_t n = 0;
            ok(ucomm_recv(c_, 0, ew::kTagRequest, request.data(), request.size(), &n), "engram request");
            if (n < sizeof(ew::Header)) throw std::runtime_error("engram: short request");
            ew::Header h;
            std::memcpy(&h, request.data(), sizeof h);
            if (h.count == 0xFFFFFFFFu) break;
            if (n != sizeof h + size_t(h.count) * sizeof(ew::Entry)) throw std::runtime_error("engram: bad request size");
            const auto* entries = reinterpret_cast<const ew::Entry*>(request.data() + sizeof h);
            size_t total = 0;
            for (uint32_t i = 0; i < h.count; ++i) {
                if (entries[i].shard < 1 || size_t(entries[i].shard) > fds_.size() || fds_[entries[i].shard - 1] < 0 || entries[i].len > ew::kMaxRowBytes)
                    throw std::runtime_error("engram: bad entry");
                total += entries[i].len;
            }
            reply.resize(total);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<size_t> at(h.count);
            for (uint32_t i = 0, o = 0; i < h.count; ++i) { at[i] = o; o += entries[i].len; }
            std::atomic<bool> failed{false};
            const int nt = std::min<int>(threads_, int(h.count));
            std::vector<std::thread> pool;
            for (int w = 0; w < nt; ++w)
                pool.emplace_back([&, w] {
                    pin_helper();
                    for (uint32_t i = uint32_t(w); i < h.count; i += uint32_t(nt))
                        if (pread(fds_[entries[i].shard - 1], reply.data() + at[i], entries[i].len, off_t(entries[i].offset)) != ssize_t(entries[i].len)) failed = true;
                });
            for (auto& t : pool) t.join();
            if (failed) throw std::runtime_error("engram: read failed");
            reads += h.count;
            ms += since(t0);
            ok(ucomm_send(c_, 0, ew::kTagReply, reply.data(), reply.size()), "engram reply");
        }
        std::fprintf(stderr, "DS41_CPU_WORKER engram rows=%.0f read_ms=%.1f\n", reads, ms);
        ucomm_finalize(c_);
        c_ = nullptr;
    }
    int threads_;
    std::vector<int> fds_;
    ucomm_t* c_ = nullptr;
    std::thread thread_;
};
}  // namespace

int main(int argc, char** argv) {
    std::string root, backend = "auto", model;
    int threads = 0, engram_port = 0, engram_threads = 16, fetch_port = 0, engram_port2 = 0, fetch_port2 = 0, stream_port = 0, stream_cpu = -1;
    std::string helper_cpus, root2;   // --root2/--engram-port2/--fetch-port2: a second decoder served at the same time (one lane, requests in arrival order)
    bool numa_place = false, persist = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a.starts_with("--root=")) root = a.substr(7);
        else if (a.starts_with("--model=")) model = a.substr(8);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10));
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else if (a.starts_with("--engram-port=")) engram_port = std::stoi(a.substr(14));
        else if (a.starts_with("--engram-threads=")) engram_threads = std::stoi(a.substr(17));
        else if (a.starts_with("--fetch-port=")) fetch_port = std::stoi(a.substr(13));
        else if (a.starts_with("--stream-cpu=")) stream_cpu = std::stoi(a.substr(13));   // the stream thread spin-polls: its own CPU
        else if (a.starts_with("--stream-port=")) stream_port = std::stoi(a.substr(14));   // a context serving lists of owned experts to the decoder's streamed prefill
        else if (a.starts_with("--root2=")) root2 = a.substr(8);
        else if (a.starts_with("--engram-port2=")) engram_port2 = std::stoi(a.substr(15));
        else if (a.starts_with("--fetch-port2=")) fetch_port2 = std::stoi(a.substr(14));
        else if (a == "--numa-place") numa_place = true;
        else if (a.starts_with("--helper-cpus=")) helper_cpus = a.substr(14);   // e.g. 14,15,30,31
        else if (a == "--persist") persist = true;   // serve one decoder session after another with the same rows (the setup must repeat)
        else {
            std::fprintf(stderr, "usage: %s --root HOST:PORT --model FIRST_SHARD.gguf [--threads=N] [--backend=auto|ib|udp] [--engram-port=P] [--engram-threads=N] [--fetch-port=P]\n", argv[0]);
            return 2;
        }
    }
    for (size_t at = 0; at < helper_cpus.size();) {
        const size_t comma = std::min(helper_cpus.find(',', at), helper_cpus.size());
        g_helper_cpus.push_back(std::atoi(helper_cpus.substr(at, comma - at).c_str()));
        at = comma + 1;
    }
    if (root.empty() || model.empty()) {
        std::fprintf(stderr, "--root and --model are required\n");
        return 2;
    }
    try {
        strata::core::ModelArtifact art{std::filesystem::path(model)};
        // rows, geometry and the lane outlive a decoder session (--persist)
        std::vector<Layer> layers;
        std::vector<uint8_t> loaded_setup;
        net::TpSetup setup{};
        size_t total = 0;
        int H = 0, rows = 0;
        double fill_ms = 0;
        bool loaded = false;
        std::unique_ptr<strata::ds41::ExpertLane> lane;
        std::mutex setup_mu, lane_make_mu;
        // the lane: a ticket lock, so two decoders' requests take turns in arrival order
        std::atomic<unsigned> lane_next{0}, lane_serving{0};   // the rows load once; the lane runs one request at a time
        auto serve_once = [&](const std::string& root, int engram_port, int fetch_port) -> bool {
        pin_helper();
        // the decoder opens its Engram context before the main one: the same order here
        std::unique_ptr<EngramServer> engram;
        if (engram_port > 0) {
            const std::string host = root.substr(0, root.rfind(':'));
            engram = std::make_unique<EngramServer>(model, host + ":" + std::to_string(engram_port), backend == "ib" ? 1 : backend == "udp" ? 2 : 0, engram_threads);
        }
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 1;
        cfg.world = 2;
        cfg.root_addr = root.c_str();
        cfg.timeout_ms = 0;
        cfg.backend = backend == "ib" ? UCOMM_BACKEND_IB : backend == "udp" ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
        ucomm_t* c = nullptr;
        ok(ucomm_init(&cfg, &c), "ucomm_init");
        std::fprintf(stderr, "DS41_CPU_WORKER connected backend=%s\n", ucomm_backend_name(c));
        std::vector<uint8_t> setup_msg(1 << 20);
        size_t n = 0;
        ok(ucomm_recv(c, 0, net::kTagSetup, setup_msg.data(), setup_msg.size(), &n), "setup");
        std::unique_lock<std::mutex> setup_lock(setup_mu);   // the first connection reads the rows; the second finds them loaded
        std::memcpy(&setup, setup_msg.data(), sizeof setup);
        if (loaded) {
            if (n != loaded_setup.size() || !std::equal(loaded_setup.begin(), loaded_setup.end(), setup_msg.begin()))
                throw std::runtime_error("the decoder's setup differs from the one the rows were read for");
        } else {
        if (setup.magic != net::kTpMagicDs41Own || setup.weights != net::kWeightsStreamed || setup.reply != net::kReplyRaw || setup.layers > 4096 ||
            n < sizeof setup + setup.layers * sizeof(net::TpLayer))
            throw std::runtime_error("bad setup message (a DS4.1 whole-expert decoder is expected)");
        size_t owned_ids = 0;
        for (uint32_t i = 0; i < setup.layers; ++i) {
            net::TpLayer s;
            std::memcpy(&s, setup_msg.data() + sizeof setup + i * sizeof s, sizeof s);
            if (s.experts < 0 || s.split != 0 || s.pad < s.experts || s.pad > 65536) throw std::runtime_error("bad layer: owned count, split or expert count");
            owned_ids += size_t(s.experts);
        }
        const size_t counts_bytes = setup.pad == 1 ? owned_ids * sizeof(float) : 0;
        if (n != sizeof setup + setup.layers * sizeof(net::TpLayer) + owned_ids * sizeof(int32_t) + counts_bytes) throw std::runtime_error("bad setup message size");
        const int32_t* owned_list = reinterpret_cast<const int32_t*>(setup_msg.data() + sizeof setup + setup.layers * sizeof(net::TpLayer));
        H = int(setup.hidden);
        layers.assign(setup.layers, Layer());
        total = 0;
        rows = 0;
        for (uint32_t i = 0; i < setup.layers; ++i) {
            net::TpLayer s;
            std::memcpy(&s, setup_msg.data() + sizeof setup + i * sizeof s, sizeof s);
            if (s.layer != int(i)) throw std::runtime_error("setup layers out of order");
            Layer& L = layers[i];
            std::string error;
            if (!cpu::native_fmt(s.gu_type, s.d_type, H, s.n_ff, L.f, error)) throw std::runtime_error(error);
            rows = s.n_ff;
            L.f.n_ff = rows;
            L.f.d_row = ggml_row_size(ggml_type(s.d_type), rows);
            L.f.h_bytes = ggml_row_size(ggml_type(L.f.d_act), rows);
            L.f.up_off = L.f.gu_row * size_t(rows);
            L.f.down_off = 2 * L.f.up_off;
            L.f.bytes = L.f.down_off + L.f.d_row * size_t(H);
            L.f.swiglu_limit = L.clamp = s.swiglu_limit;
            L.f.iq1s_rows = true;
            L.slot_bytes = net::tp_blob_stride(L.f.bytes);
            L.experts = s.experts;
            L.local.assign(size_t(s.pad), -1);
            for (int j = 0; j < s.experts; ++j) {
                const int e = *owned_list++;
                if (e < 0 || e >= s.pad || L.local[size_t(e)] >= 0 || (j && e <= owned_list[-2])) throw std::runtime_error("owned expert ids must be ascending and in range");
                L.local[size_t(e)] = j;
            }
            total += size_t(L.experts) * L.slot_bytes;
        }
        // the rows come from this node's model: the geometry must agree with the decoder's setup before anything is read
        const auto fill0 = std::chrono::steady_clock::now();
        for (uint32_t l = 0; l < setup.layers; ++l) {
            Layer& L = layers[l];
            const std::string p = "blk." + std::to_string(l) + ".ffn_";
            const auto &G = art.tensors().at(p + "gate_exps.weight"), &U = art.tensors().at(p + "up_exps.weight"), &D = art.tensors().at(p + "down_exps.weight");
            const size_t gstride = L.f.gu_row * size_t(rows), dstride = L.f.d_row * size_t(H);
            if (int(G.tensor->type) != int(L.f.gu_type) || int(U.tensor->type) != int(L.f.gu_type) || int(D.tensor->type) != int(L.f.d_type) ||
                G.bytes < size_t(L.local.size()) * gstride || U.bytes < size_t(L.local.size()) * gstride || D.bytes < size_t(L.local.size()) * dstride ||
                2 * gstride + dstride != L.f.bytes)
                throw std::runtime_error("model layer " + std::to_string(l) + " disagrees with the decoder's setup");
            void* mem = mmap(nullptr, std::max<size_t>(1, size_t(L.experts) * L.slot_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem == MAP_FAILED) throw std::runtime_error("cannot map the rows");
            // STRATA_DS41_WORKER_THP=1: transparent huge pages for the rows (THP is in madvise mode here): the HCA then needs 512x fewer translation entries when
            // the rows are registered and read in scattered order (the stream service), and the lane's TLB misses fall
            if (std::getenv("STRATA_DS41_WORKER_THP") && std::string(std::getenv("STRATA_DS41_WORKER_THP")) == "1")
                madvise(mem, std::max<size_t>(1, size_t(L.experts) * L.slot_bytes), MADV_HUGEPAGE);
            L.base = static_cast<uint8_t*>(mem);
            const uint8_t *g = G.data(), *u = U.data(), *d = D.data();
            // --numa-place: blob j is first touched by a thread of node j % 2, so each expert sits whole in one node's memory
            // (STRATA_DS41_LANE_NUMA=1 then gives every expert to the threads of that node)
            const std::vector<int> cpus0 = numa_place ? node_cpus(0) : std::vector<int>(), cpus1 = numa_place ? node_cpus(1) : std::vector<int>();
            const int parts = numa_place && !cpus0.empty() && !cpus1.empty() ? 2 : 1;
            auto fill = [&](int part) {
                if (parts == 2) pin_to(part == 0 ? cpus0 : cpus1);
                for (size_t e = 0; e < L.local.size(); ++e) {
                    const int j = L.local[e];
                    if (j < 0 || (parts == 2 && (j & 1) != part)) continue;
                    uint8_t* blob = L.base + size_t(j) * L.slot_bytes;
                    std::memcpy(blob, g + e * gstride, gstride);
                    std::memcpy(blob + gstride, u + e * gstride, gstride);
                    std::memcpy(blob + 2 * gstride, d + e * dstride, dstride);
                }
            };
            if (parts == 2) {
                std::thread other([&] { fill(1); });
                fill(0);
                other.join();
                cpu_set_t all;
                CPU_ZERO(&all);
                for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
                pthread_setaffinity_np(pthread_self(), sizeof all, &all);
            } else {
                fill(0);
            }
        }
        fill_ms = since(fill0);
        std::fprintf(stderr, "DS41_CPU_WORKER rows GiB=%.2f read from the local model in %.1f s\n", total / 1073741824.0, fill_ms / 1e3);
        loaded = true;
        loaded_setup.assign(setup_msg.begin(), setup_msg.begin() + std::ptrdiff_t(n));
        }
        setup_lock.unlock();
        net::TpAccept accept{};
        accept.ok = 1;
        accept.have = 1;   // nothing is sent: the decoder skips the weights
        accept.needed_bytes = accept.available_bytes = total;
        ok(ucomm_send(c, 0, net::kTagAccept, &accept, sizeof accept), "accept");

        const int K_max = 16, maxT = 1024;
        { std::lock_guard<std::mutex> g(lane_make_mu);
        if (!lane) lane = std::make_unique<strata::ds41::ExpertLane>(H, rows, threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency() / 2)), true);
        }
        if (!root2.empty()) pin_helper();   // (the lane's constructor pinned this thread to its core 0: waiting for requests belongs on a helper core)
        const size_t msg_bytes = sizeof(net::TpRequest) + size_t(maxT) * K_max * sizeof(net::TpJob) + size_t(maxT) * H * 4;
        std::vector<uint8_t> msg(msg_bytes);
        std::vector<float> reply(size_t(maxT) * K_max * H), cpu_rows(size_t(maxT) * K_max * H);
        std::vector<ucomm_mr_t*> regions;
        for (const auto& [p, bytes] : {std::pair<void*, size_t>{msg.data(), msg.size()}, {reply.data(), reply.size() * 4}}) {
            ucomm_mr_t* region = nullptr;
            if (ucomm_mr_reg(c, p, bytes, &region) == UCOMM_OK) regions.push_back(region);
        }
        net::TpReady ready{1u, 0u, fill_ms / 1e3};
        ok(ucomm_send(c, 0, net::kTagReady, &ready, sizeof ready), "ready");
        // --fetch-port: a third context serves whole owned experts to a decoder that promotes them (STRATA_DS41_OWN_PROMOTE)
        ucomm_t* fetch_c = nullptr;
        std::thread fetch_thread;
        if (fetch_port > 0) {
            const std::string host = root.substr(0, root.rfind(':'));
            const std::string addr = host + ":" + std::to_string(fetch_port);
            ucomm_config fcfg = cfg;
            fcfg.root_addr = addr.c_str();
            ok(ucomm_init(&fcfg, &fetch_c), "fetch ucomm_init");
            fetch_thread = std::thread([&] {
                pin_helper();
                std::vector<uint8_t> blob;
                double served = 0;
                for (;;) {
                    int32_t req[3] = {0, 0, 0};
                    size_t got = 0;
                    ok(ucomm_recv(fetch_c, 0, 50, req, sizeof req, &got), "fetch request");
                    if (req[0] < 0) break;
                    if (got != sizeof req) req[2] = 0;
                    if (req[0] >= int(layers.size()) || req[1] < 0 || req[1] >= int(layers[size_t(req[0])].local.size()) || layers[size_t(req[0])].local[size_t(req[1])] < 0)
                        throw std::runtime_error("fetch names an expert this worker does not own");
                    const Layer& L = layers[size_t(req[0])];
                    const uint8_t* src = L.base + size_t(L.local[size_t(req[1])]) * L.slot_bytes;
                    const size_t piece = req[2] > 0 ? size_t(req[2]) : L.f.bytes;
                    for (size_t off = 0; off < L.f.bytes; off += piece)
                        ok(ucomm_send(fetch_c, 0, 51, src + off, std::min(piece, L.f.bytes - off)), "fetch reply");
                    served += 1;
                }
                std::fprintf(stderr, "DS41_CPU_WORKER fetch experts=%.0f\n", served);
                ucomm_finalize(fetch_c);
                fetch_c = nullptr;
            });
        }

        // --stream-port: a context that serves lists of whole owned experts (the decoder's streamed prefill: tags 60 header {layer, n}, 61 the n expert ids,
        // 62 one message of L.f.bytes per expert, in the order asked); layer < 0 ends it
        ucomm_t* stream_c = nullptr;
        std::thread stream_thread;
        if (stream_port > 0) {
            const std::string host = root.substr(0, root.rfind(':'));
            const std::string addr = host + ":" + std::to_string(stream_port);
            ucomm_config scfg = cfg;
            scfg.root_addr = addr.c_str();
            ok(ucomm_init(&scfg, &stream_c), "stream ucomm_init");
            // The experts are copied into a small registered staging ring (STRATA_DS41_STREAM_RING blobs, 24) by a helper thread and sent from there: the HCA's
            // address translation thrashes when it reads scattered experts out of the registered 74 GB of rows (30 GB region: 1.19 GB/s against 3.8 from a small
            // one, bench stream_pattern), and an unregistered send registers and releases the buffer per message.
            struct StreamState {   // owned by both threads (the block ends before they do)
                struct Task { const uint8_t* src; size_t len; uint64_t idx; };
                size_t ring_slot = 0, ring_n = 0;
                void* ring_mem = nullptr;
                ucomm_mr_t* ring_mr = nullptr;
                std::mutex tq_mu;
                std::condition_variable tq_cv;
                std::deque<Task> tq;
                std::unique_ptr<std::atomic<uint64_t>[]> ready;   // per ring slot: 1 + the sequence number of the blob copied into it
                std::atomic<uint64_t> ring_tail{0};                // blobs sent
                uint64_t next_idx = 0;                             // sequence number of the next task (sender thread)
                std::atomic<bool> copier_stop{false};
                std::vector<std::thread> copiers;
            };
            auto st = std::make_shared<StreamState>();
            for (auto& L : layers) st->ring_slot = std::max(st->ring_slot, L.f.bytes);
            st->ring_slot = (st->ring_slot + 4095) / 4096 * 4096;
            st->ring_n = std::getenv("STRATA_DS41_STREAM_RING") ? size_t(std::max(2, std::atoi(std::getenv("STRATA_DS41_STREAM_RING")))) : 24;
            st->ring_mem = mmap(nullptr, st->ring_slot * st->ring_n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (st->ring_mem == MAP_FAILED) throw std::runtime_error("cannot map the stream ring");
            ok(ucomm_mr_reg(stream_c, st->ring_mem, st->ring_slot * st->ring_n, &st->ring_mr), "stream ring registration");
            std::fprintf(stderr, "DS41_CPU_WORKER stream: staging ring %zu x %.2f MB registered\n", st->ring_n, st->ring_slot / 1e6);
            st->ready = std::make_unique<std::atomic<uint64_t>[]>(st->ring_n);
            for (size_t i = 0; i < st->ring_n; ++i) st->ready[i] = 0;
            const int ncopy = std::getenv("STRATA_DS41_STREAM_COPIERS") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_STREAM_COPIERS"))) : 4;
            for (int ci = 0; ci < ncopy; ++ci) st->copiers.emplace_back([st] {
                pin_helper();
                for (;;) {
                    StreamState::Task t;
                    {
                        std::unique_lock<std::mutex> lock(st->tq_mu);
                        st->tq_cv.wait(lock, [&] { return st->copier_stop.load() || !st->tq.empty(); });
                        if (st->tq.empty()) return;
                        t = st->tq.front();
                        st->tq.pop_front();
                    }
                    while (t.idx - st->ring_tail.load(std::memory_order_acquire) >= st->ring_n) std::this_thread::yield();   // the slot's previous blob is sent
                    std::memcpy(static_cast<uint8_t*>(st->ring_mem) + (t.idx % st->ring_n) * st->ring_slot, t.src, t.len);
                    st->ready[t.idx % st->ring_n].store(t.idx + 1, std::memory_order_release);
                }
            });
            stream_thread = std::thread([&, st] {
                if (stream_cpu >= 0) {
                    cpu_set_t one;
                    CPU_ZERO(&one);
                    CPU_SET(stream_cpu, &one);
                    pthread_setaffinity_np(pthread_self(), sizeof one, &one);
                } else {
                    pin_helper();
                }
                std::vector<int32_t> want;
                double served = 0, bytes = 0, busy = 0, idle = 0, gap = 0, ids_wait = 0;
                auto last_end = std::chrono::steady_clock::now();
                for (;;) {
                    int32_t hdr[4] = {0, 0, 0, 0};
                    size_t got = 0;
                    const auto r0 = std::chrono::steady_clock::now();
                    ok(ucomm_recv(stream_c, 0, 60, hdr, sizeof hdr, &got), "stream request");
                    idle += since(r0);
                    if (hdr[0] < 0) break;
                    if (hdr[0] >= int(layers.size()) || hdr[1] < 0 || hdr[1] > 4096) throw std::runtime_error("stream request out of range");
                    const Layer& L = layers[size_t(hdr[0])];
                    want.resize(size_t(hdr[1]));
                    const auto i0 = std::chrono::steady_clock::now();
                    if (!want.empty()) ok(ucomm_recv(stream_c, 0, 61, want.data(), want.size() * 4, &got), "stream ids");
                    ids_wait += since(i0);
                    gap += since(r0) - 0.0;
                    const auto t0 = std::chrono::steady_clock::now();
                    const uint64_t first_idx = st->next_idx;
                    {
                        std::lock_guard<std::mutex> lock(st->tq_mu);
                        for (int e : want) {
                            if (e < 0 || e >= int(L.local.size()) || L.local[size_t(e)] < 0) throw std::runtime_error("stream names an expert this worker does not own");
                            st->tq.push_back({L.base + size_t(L.local[size_t(e)]) * L.slot_bytes, L.f.bytes, st->next_idx++});
                        }
                    }
                    st->tq_cv.notify_all();
                    for (size_t i = 0; i < want.size(); ++i) {
                        const uint64_t idx = first_idx + i;
                        while (st->ready[idx % st->ring_n].load(std::memory_order_acquire) != idx + 1) std::this_thread::yield();
                        ok(ucomm_send(stream_c, 0, 62, static_cast<uint8_t*>(st->ring_mem) + (idx % st->ring_n) * st->ring_slot, L.f.bytes), "stream expert");
                        st->ring_tail.store(idx + 1, std::memory_order_release);
                    }
                    busy += since(t0);
                    served += double(want.size());
                    bytes += double(want.size()) * double(L.f.bytes);
                }
                (void) gap; (void) last_end;
                std::fprintf(stderr, "DS41_CPU_WORKER stream experts=%.0f GB=%.2f send_s=%.2f GB/s=%.2f wait_for_request_s=%.2f ids_wait_s=%.2f\n", served, bytes / 1e9, busy / 1e3, busy > 0 ? bytes / 1e6 / busy : 0.0, idle / 1e3, ids_wait / 1e3);
                st->copier_stop = true;
                st->tq_cv.notify_all();
                for (auto& th : st->copiers) th.join();
                ucomm_mr_dereg(st->ring_mr);
                ucomm_finalize(stream_c);
                stream_c = nullptr;
                munmap(st->ring_mem, st->ring_slot * st->ring_n);
            });
        }
        std::vector<int32_t> ids;
        int64_t requests = 0;
        double busy_ms = 0, lane_ms = 0;
        struct Bin { double n = 0, lane = 0, total = 0; };
        Bin bins[6];   // decode / verify requests (nt <= 8) by owned-route count: 0-3, 4-7, 8-11, 12-15, 16-19, 20+
        double wait_ms = 0;   // time between a reply and the next request (the decoder's side of the layer)
        auto last_reply = std::chrono::steady_clock::now();
        for (;;) {
            ok(ucomm_recv(c, 0, net::kTagRequest, msg.data(), msg_bytes, &n), "request");
            const auto got_request = std::chrono::steady_clock::now();
            if (requests) wait_ms += std::chrono::duration<double, std::milli>(got_request - last_reply).count();
            net::TpRequest rq;
            std::memcpy(&rq, msg.data(), sizeof rq);
            if (rq.layer < 0) break;
            if (rq.layer >= int(setup.layers)) throw std::runtime_error("request names an invalid layer");
            const Layer& L = layers[size_t(rq.layer)];
            const int nt = rq.nt;
            if (rq.njobs < 0 || nt < 1 || nt > maxT || rq.njobs % nt || rq.act_bytes != 4 * H || rq.njobs / nt > K_max) throw std::runtime_error("bad DS4.1 request");
            const int K = rq.njobs / nt;
            const auto* list = reinterpret_cast<const net::TpJob*>(msg.data() + sizeof rq);
            const float* z = reinterpret_cast<const float*>(msg.data() + sizeof rq + size_t(rq.njobs) * sizeof(net::TpJob));
            const auto t0 = std::chrono::steady_clock::now();
            ids.assign(size_t(rq.njobs), -1);
            int count = 0;
            for (int j = 0; j < rq.njobs; ++j) {
                const int e = list[j].expert;
                if (e < -1 || e >= int(L.local.size()) || list[j].token != j / K) throw std::runtime_error("invalid route");
                if (e < 0) continue;
                if (L.local[size_t(e)] < 0) throw std::runtime_error("route names an expert this worker does not own");
                ids[size_t(j)] = L.local[size_t(e)];
                ++count;
            }
            strata::ds41::LaneLayer view;
            view.f = L.f;
            view.gate = L.base;
            view.up = L.base + L.f.up_off;
            view.down = L.base + L.f.down_off;
            view.gate_stride = view.down_stride = L.slot_bytes;
            const auto c0 = std::chrono::steady_clock::now();
            if (root2.empty()) lane->run_raw(view, z, nt, ids.data(), K, cpu_rows.data());
            else {   // two decoders share the lane: one request at a time, the caller computing as the lane's thread 0 on that thread's core
                const unsigned ticket = lane_next.fetch_add(1, std::memory_order_acq_rel);
                while (lane_serving.load(std::memory_order_acquire) != ticket) std::this_thread::yield();
                struct Release { std::atomic<unsigned>& s; ~Release() { s.fetch_add(1, std::memory_order_release); } } release{lane_serving};
                cpu_set_t prev;
                pthread_getaffinity_np(pthread_self(), sizeof prev, &prev);
                if (lane->caller_cpu() >= 0) {
                    cpu_set_t one;
                    CPU_ZERO(&one);
                    CPU_SET(lane->caller_cpu(), &one);
                    pthread_setaffinity_np(pthread_self(), sizeof one, &one);
                }
                lane->run_raw(view, z, nt, ids.data(), K, cpu_rows.data());
                pthread_setaffinity_np(pthread_self(), sizeof prev, &prev);
            }
            lane_ms += since(c0);
            std::memcpy(reply.data(), cpu_rows.data(), size_t(count) * H * 4);
            busy_ms += since(t0);
            ++requests;
            ok(ucomm_send(c, 0, net::kTagReply, reply.data(), size_t(count) * H * 4), "reply");
            last_reply = std::chrono::steady_clock::now();
            if (nt <= 8) {
                Bin& b = bins[std::min(5, count / 4)];
                b.n += 1;
                b.lane += since(c0);
                b.total += std::chrono::duration<double, std::milli>(last_reply - got_request).count();
            }
        }
        std::fprintf(stderr, "DS41_CPU_WORKER done requests=%lld busy_ms=%.1f lane_ms=%.1f between_requests_ms=%.1f\n", (long long)requests, busy_ms, lane_ms, wait_ms);
        for (int i = 0; i < 6; ++i)
            if (bins[i].n) std::fprintf(stderr, "DS41_CPU_WORKER nt<=8 routes %d-%d: n=%.0f lane_ms=%.3f request_to_reply_ms=%.3f\n", i * 4, i * 4 + 3, bins[i].n, bins[i].lane / bins[i].n, bins[i].total / bins[i].n);
        for (auto* r : regions) ucomm_mr_dereg(r);
        if (fetch_thread.joinable()) fetch_thread.join();
        if (stream_thread.joinable()) stream_thread.join();
        ucomm_finalize(c);
        if (engram) engram->join();
        if (!persist || !root2.empty()) return false;
        lane.reset();   // its threads would yield-spin through the wait for the next decoder
        return true;
        };
        auto serve_loop = [&](const std::string& r, int ep, int fp) { while (serve_once(r, ep, fp)) {} };
        if (root2.empty()) { serve_loop(root, engram_port, fetch_port); return 0; }
        std::thread second([&] {
            try { serve_loop(root2, engram_port2, fetch_port2); }
            catch (const std::exception& e) { std::fprintf(stderr, "DS41_CPU_WORKER error (second decoder): %s\n", e.what()); }
        });
        serve_loop(root, engram_port, fetch_port);
        second.join();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DS41_CPU_WORKER error: %s\n", e.what());
        return 1;
    }
}
