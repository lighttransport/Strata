// Experimental GLM decode and streamed GPU prefill. Decode uses CPU routed experts.
// Greedy decoding with verified speculation; no graph capture.
#include "ggml.h"
#include "strata/core/expert_cache.hpp"
#include "strata/core/model.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#ifdef __linux__
#include <sched.h>
#include <sys/resource.h>
#endif
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace {
namespace k = strata::kernels;
namespace cpu = k::cpu;
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("GLM: cuBLAS failure " + std::to_string(e));
}
struct Device {
    void *p = nullptr;
    size_t bytes;
    bool owned = true;
    static inline size_t live = 0, peak = 0, limit = SIZE_MAX;
    Device(void *view, size_t bytes) : p(view), bytes(bytes), owned(false) {}
    explicit Device(size_t n, bool zero = false) : bytes(n) {
        if (live > limit || n > limit - live)
            throw std::runtime_error("GLM: GPU allocation budget exceeded");
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Keep additional physical headroom for the display and library allocations.
        constexpr size_t reserve = size_t(2) * 1024 * 1024 * 1024;
        if (free_bytes < reserve || n > free_bytes - reserve)
            throw std::runtime_error("GLM: GPU allocation would consume display headroom");
        check(cudaMalloc(&p, n));
        live += n;
        peak = std::max(peak, live);
        if (zero) {
            auto status = cudaMemset(p, 0, n);
            if (status != cudaSuccess) {
                cudaFree(p); p = nullptr; live -= n;
                check(status);
            }
        }
    }
    ~Device() {
        if (owned) {
            cudaFree(p);
            live -= bytes;
        }
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    float *f(size_t offset = 0) const { return (float *)p + offset; }
    void put(const void *x, size_t n) {
        if (n > bytes)
            throw std::runtime_error("GLM: upload exceeds buffer");
        check(cudaMemcpy(p, x, n, cudaMemcpyHostToDevice));
    }
    void put_async(const void *x, size_t n, cudaStream_t stream) {
        if (n > bytes) throw std::runtime_error("GLM: async upload exceeds buffer");
        check(cudaMemcpyAsync(p, x, n, cudaMemcpyHostToDevice, stream));
    }
    std::vector<float> floats(size_t n) const {
        if (n > bytes / sizeof(float))
            throw std::runtime_error("GLM: read exceeds buffer");
        std::vector<float> x(n);
        check(cudaMemcpy(x.data(), p, n * 4, cudaMemcpyDeviceToHost));
        return x;
    }
};
constexpr size_t MiB = 1024 * 1024;
namespace mmq = strata::prefill::mmq;
struct Pinned {
    void *p = nullptr;
    explicit Pinned(size_t bytes) { check(cudaMallocHost(&p, bytes)); }
    ~Pinned() { cudaFreeHost(p); }
};
struct GpuLocality {
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t previous;
    bool changed = false;
#endif
    GpuLocality() {
#ifdef __linux__
        int dev;
        check(cudaGetDevice(&dev));
        char pci[32];
        check(cudaDeviceGetPCIBusId(pci, sizeof(pci), dev));
        std::ifstream node_file(std::string("/sys/bus/pci/devices/") + pci + "/numa_node");
        int node = -1;
        node_file >> node;
        std::ifstream list_file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
        std::string list;
        list_file >> list;
        sched_getaffinity(0, sizeof(previous), &previous);
        std::istringstream parts(list);
        std::string part;
        while (std::getline(parts, part, ',')) {
            auto dash = part.find('-');
            int a = std::stoi(part), b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
            for (int i = a; i <= b; ++i)
                if (i < CPU_SETSIZE && CPU_ISSET(i, &previous))
                    cpus.push_back(i);
        }
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : cpus)
                CPU_SET(c, &set);
            changed = sched_setaffinity(0, sizeof(set), &set) == 0;
        }
#endif
    }
    ~GpuLocality() {
#ifdef __linux__
        if (changed)
            sched_setaffinity(0, sizeof(previous), &previous);
#endif
    }
};
class ExpertStager {
    std::mutex mutex;
    std::condition_variable work, finished;
    std::function<void(int)> job;
    std::vector<std::thread> workers;
    int generation = 0, pending = 0;
    bool stop = false;

  public:
    explicit ExpertStager(const std::vector<int> &cpus) {
        for (int id = 0; id < 2; ++id)
            workers.emplace_back([this, id, cpus] {
#ifdef __linux__
                if (!cpus.empty()) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(cpus[id % cpus.size()], &set);
                    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                }
#endif
                int seen = 0;
                for (;;) {
                    std::unique_lock lock(mutex);
                    work.wait(lock, [&] { return stop || generation != seen; });
                    if (stop)
                        return;
                    seen = generation;
                    auto fn = job;
                    lock.unlock();
                    fn(id);
                    lock.lock();
                    if (!--pending)
                        finished.notify_one();
                }
            });
    }
    ~ExpertStager() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        work.notify_all();
        for (auto &w : workers)
            w.join();
    }
    void run(std::function<void(int)> fn) {
        std::unique_lock lock(mutex);
        job = std::move(fn);
        pending = 2;
        ++generation;
        work.notify_all();
        finished.wait(lock, [&] { return !pending; });
    }
};
struct GpuPrefill {
    static constexpr size_t slot_bytes = 208 * MiB;
    int chunk;
    size_t cursor = 0;
    std::unique_ptr<Device> arena, dq, blas_workspace, mmq_workspace, slots[2];
    std::unique_ptr<Pinned> pinned[2], counts;
    std::unique_ptr<ExpertStager> stager;
    strata::prefill::Gemm gemm;
    std::unique_ptr<mmq::Context> context;
    cudaStream_t copy = nullptr;
    cudaEvent_t ready[2] = {}, done[2] = {};
    uint64_t transferred = 0, groups = 0;
    double stage_ms = 0;
    static size_t arena_bytes(int width, size_t context) {
        const size_t tile = std::min(width, 64), pools = std::max<size_t>(1, context / 4);
        const size_t mla = tile * (2052 * (512 + 64) + 2 * 64 * 512 + 33 * pools) * 4
                         + (size_t)width * 176 * 1024 + MiB;
        const size_t required = std::max({64 * MiB, (size_t)width * 512 * 1024, mla});
        return (required + 64 * MiB - 1) / (64 * MiB) * (64 * MiB);
    }
    explicit GpuPrefill(int width, void *stream, bool compact = false, size_t capacity = 8192) : chunk(width) {
        if (compact) {
            arena = std::make_unique<Device>(32 * MiB);
            counts = std::make_unique<Pinned>(289 * 4);
            return;
        }
        GpuLocality locality;
        arena = std::make_unique<Device>(arena_bytes(width, capacity));
        dq = std::make_unique<Device>(128 * MiB);
        blas_workspace = std::make_unique<Device>(32 * MiB);
        mmq_workspace = std::make_unique<Device>(128 * MiB);
        for (int i = 0; i < 2; ++i) {
            slots[i] = std::make_unique<Device>(slot_bytes);
            pinned[i] = std::make_unique<Pinned>(slot_bytes);
        }
        counts = std::make_unique<Pinned>(289 * 4);
        stager = std::make_unique<ExpertStager>(locality.cpus);
        std::string error;
        if (!gemm.init_external(stream, (uint16_t *)dq->p, dq->bytes / 2, blas_workspace->p,
                                blas_workspace->bytes, error))
            throw std::runtime_error(error);
        context = std::make_unique<mmq::Context>(mmq_workspace->p, mmq_workspace->bytes);
        check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            check(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
            check(cudaEventRecord(ready[i], copy));
            check(cudaEventRecord(done[i], (cudaStream_t)stream));
        }
        std::cerr << "GPU prefill chunk=" << chunk << " staging_workers=2 numa_cpus=";
        for (size_t i = 0; i < std::min<size_t>(2, locality.cpus.size()); ++i)
            std::cerr << (i ? "," : "") << locality.cpus[i];
        std::cerr << '\n';
    }
    ~GpuPrefill() {
        if (copy)
            cudaStreamSynchronize(copy);
        for (int i = 0; i < 2; ++i) {
            if (ready[i])
                cudaEventDestroy(ready[i]);
            if (done[i])
                cudaEventDestroy(done[i]);
        }
        if (copy)
            cudaStreamDestroy(copy);
    }
    std::unique_ptr<Device> allocate(size_t bytes) {
        size_t offset = (cursor + 255) / 256 * 256;
        if (offset > arena->bytes || bytes > arena->bytes - offset)
            throw std::runtime_error("GLM: prefill scratch exhausted");
        cursor = offset + bytes;
        return std::make_unique<Device>((char *)arena->p + offset, bytes);
    }
    void upload(const strata::core::ArtifactTensor &G, const strata::core::ArtifactTensor &U,
                const strata::core::ArtifactTensor &D, int start, int n, int slot, void *stream,
                const int *selected = nullptr, bool blobs = false) {
        size_t gh = G.bytes / 288, db = D.bytes / 288, bytes = n * (2 * gh + db);
        if (bytes + 16384 > slot_bytes)
            throw std::runtime_error("GLM: native expert group exceeds ring slot");
        check(cudaEventSynchronize(ready[slot]));
        auto begin = std::chrono::steady_clock::now();
        stager->run([&](int worker) {
            for (int j = worker; j < n; j += 2) {
                auto *p = (uint8_t *)pinned[slot]->p;
                const int expert = selected ? selected[j] : start + j;
                const size_t gu_offset = blobs ? j * (2 * gh + db) : j * 2 * gh;
                const size_t down_offset = blobs ? gu_offset + 2 * gh : n * 2 * gh + j * db;
                std::memcpy(p + gu_offset, G.data() + expert * gh, gh);
                std::memcpy(p + gu_offset + gh, U.data() + expert * gh, gh);
                std::memcpy(p + down_offset, D.data() + expert * db, db);
            }
        });
        std::memset((char *)pinned[slot]->p + bytes, 0, 16384);
        stage_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        check(cudaEventSynchronize(done[slot]));
        check(cudaMemcpyAsync(slots[slot]->p, pinned[slot]->p, bytes + 16384, cudaMemcpyHostToDevice, copy));
        check(cudaEventRecord(ready[slot], copy));
        check(cudaStreamWaitEvent((cudaStream_t)stream, ready[slot], 0));
        transferred += bytes;
        ++groups;
    }
};

struct LayerState {
    std::unique_ptr<Device> recurrent, conv_q, conv_k, conv_v, cache, keys, gates, pooled;
};
class Decoder {
    strata::core::ModelArtifact artifact;
    const strata::core::ModelDescriptor &m;
    cudaStream_t stream = nullptr;
    cudaEvent_t moe_ready = nullptr;
    cublasHandle_t blas = nullptr;
    cpu::ExpertPool pool;
    size_t capacity, budget, resident = 0;
    int position = 0;
    int batch_tokens = 1;
    bool fast = false, profile = false, gpu_decode_experts = false;
    std::unique_ptr<GpuPrefill> gpu;
    std::map<std::string, std::unique_ptr<Device>> phase;
    std::function<bool()> cancelled;
    bool cache_frozen = false;
    std::ofstream routing_trace;
    struct PreparedExperts {
        cpu::NativeFmt format;
        std::unique_ptr<uint8_t[]> gate, up, down;
    };
    std::map<int, PreparedExperts> prepared_experts;
    std::vector<int> prepack_cpus;
    long long host_affinity = -1;
    std::unique_ptr<Pinned> host_moe;
    std::unique_ptr<Pinned> host_hit_metadata, host_hit_results;
    std::vector<int> host_selected;
    std::vector<float> host_results, host_sum;
    std::vector<std::vector<uint8_t>> host_quant;
    std::vector<cpu::ExpertJobMulti> host_jobs;
    bool capture_hidden = false;
    int mtp_position = 0, prefill_width_saved = 0;
    std::vector<float> target_hidden, mtp_hidden;
    std::unique_ptr<Device> mtp_experts;
    bool mtp_ready = false, mtp_cpu_experts = false;
    k::NativeExpertLayout mtp_layout;
    size_t mtp_expert_bytes = 0;
    uint64_t cache_hits = 0, cache_entries = 0;
    std::vector<std::unique_ptr<strata::core::ExpertCache>> expert_cache;
    size_t decode_cache_budget = 0;
    int decode_cache_slots_limit = 0;
    std::vector<std::vector<uint64_t>> prefill_routes;
    std::vector<std::map<int, std::unique_ptr<Device>>> decode_resident;
    std::vector<LayerState> states;
    std::unique_ptr<Device> verify_history;
    std::map<Device *, size_t> history_offsets;
    size_t history_stride = 0;
    int history_slots = 0, history_position = 0, history_valid = 0;
    bool capturing_history = false;
    void capture_state(Device *buffer, int token) {
        if (!capturing_history || token >= history_slots) return;
        auto it = history_offsets.find(buffer);
        if (it == history_offsets.end()) return;
        check(cudaMemcpyAsync((char *)verify_history->p + token * history_stride + it->second,
                              buffer->p, buffer->bytes, cudaMemcpyDeviceToDevice, stream));
    }
    std::map<std::string, std::unique_ptr<Device>> scratch;
    struct Weight {
        std::unique_ptr<Device> data;
        std::list<std::string>::iterator order;
    };
    std::map<std::string, Weight> weights;
    std::list<std::string> lru;
    void check_stop() {
        if (cancelled && cancelled())
            throw std::runtime_error("cancelled");
    }
    void reset_phase() {
        if (gpu) {
            phase.clear();
            gpu->cursor = 0;
        }
    }
    Device &buf(const std::string &name, size_t floats) {
        bool main = name == "streams" || name == "collapsed" || name == "x" || name == "y" ||
                    name == "hc_coeff" || name == "output";
        if (gpu && !main) {
            auto &p = phase[name];
            if (!p || p->bytes < floats * 4)
                p = gpu->allocate(floats * 4);
            return *p;
        }
        auto &p = scratch[name];
        if (!p || p->bytes < floats * 4)
            p = std::make_unique<Device>(floats * 4);
        return *p;
    }
    static std::vector<float> dequant(const strata::core::ArtifactTensor &tensor) {
        const auto &t = *tensor.tensor;
        const auto *traits = ggml_get_type_traits((ggml_type)t.type);
        if (!traits || !traits->to_float)
            throw std::runtime_error("GLM: no dequantizer for " + t.name);
        size_t rows = 1;
        for (size_t i = 1; i < t.shape.size(); ++i)
            rows *= t.shape[i];
        std::vector<float> out(rows * t.shape[0]);
        const size_t row = tensor.bytes / rows;
        for (size_t i = 0; i < rows; ++i)
            traits->to_float(tensor.data() + i * row, out.data() + i * t.shape[0], t.shape[0]);
        return out;
    }
    Device &weight(const std::string &name, bool floating = false) {
        const std::string key = name + (floating ? ":fp32" : ":raw");
        auto it = weights.find(key);
        if (it != weights.end()) {
            lru.splice(lru.begin(), lru, it->second.order);
            return *it->second.data;
        }
        if (gpu && !name.starts_with("blk.45."))
            throw std::runtime_error("GLM: fixed GPU weight missing: " + key);
        const auto &t = artifact.at(name);
        std::vector<float> values;
        if (floating && t.tensor->type != 0)
            values = dequant(t);
        const size_t bytes = values.empty() ? t.bytes : values.size() * 4;
        if (bytes > budget)
            throw std::runtime_error("GLM: weight exceeds dense cache budget: " + name);
        while (resident + bytes > budget && !lru.empty()) {
            // Previous kernels may still hold a pointer to an evicted weight.
            check(cudaStreamSynchronize(stream));
            const auto old = lru.back();
            resident -= weights.at(old).data->bytes;
            weights.erase(old);
            lru.pop_back();
        }
        auto data = std::make_unique<Device>(bytes);
        data->put(values.empty() ? (const void *)t.data() : values.data(), bytes);
        resident += bytes;
        lru.push_front(key);
        auto [inserted, ok] = weights.emplace(key, Weight{std::move(data), lru.begin()});
        (void)ok;
        return *inserted->second.data;
    }
    void mat(const std::string &name, const float *x, float *y, bool floating = false, int nt = 0) {
        if (!nt)
            nt = batch_tokens;
        const auto &t = *artifact.at(name).tensor;
        if (t.shape.size() != 2)
            throw std::runtime_error("GLM: matrix must have two dimensions: " + name);
        const int in = (int)t.shape[0], out = (int)t.shape[1];
        auto &W = weight(name, floating);
        const float one = 1, zero = 0;
        if (fast && name != "output.weight") {
            if (t.type == 0 || floating) {
                // Keep cuBLAS projection shapes identical across 2048/4096 chunk boundaries.
                for (int t0 = 0; t0 < nt; t0 += 2048)
                    check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, std::min(2048, nt - t0), in, &one,
                                      W.f(), in, x + (size_t)t0 * in, in, &zero, y + (size_t)t0 * out, out));
            } else {
                auto &half = buf("mat_f16", (size_t)gpu->chunk * 16384 / 2);
                k::glm_f16(x, (uint16_t *)half.p, (int64_t)nt * in, stream);
                for (int t0 = 0; t0 < nt; t0 += 2048)
                    gpu->gemm.native((uint16_t *)half.p + (size_t)t0 * in, t.type, W.p, y + (size_t)t0 * out,
                                     std::min(2048, nt - t0), out, in);
            }
            return;
        }
        if (t.type == 0 || floating) {
            for (int t = 0; t < nt; ++t)
                check(cublasSgemv(blas, CUBLAS_OP_T, in, out, &one, W.f(), in, x + t * in, 1, &zero,
                                  y + t * out, 1));
            return;
        }
        auto &q = buf("q8", k::native_q8_1_bytes(in, nt) / 4);
        k::native_quantize_q8_1(x, q.p, in, nt, stream);
        if (!k::native_mmvq_supported(t.type))
            throw std::runtime_error("GLM: unsupported dense quantization: " + name);
        k::native_mmvq(t.type, W.p, q.p, y, in, out, nt, stream);
    }
    void norm(const std::string &name, const float *x, float *y, int width) {
        auto &W = weight(name);
        k::glm_rms_norm(x, W.f(), y, width, batch_tokens, m.rms_epsilon, stream);
    }
    void hc_read(const std::string &p, const std::string &kind, const float *r, float *x, float *c) {
        auto &normalized = buf("hc_normalized", m.hidden * 4 * batch_tokens);
        auto &projected = buf("hc_projected", 24 * batch_tokens);
        k::glm_rms_norm(r, nullptr, normalized.f(), m.hidden * 4, batch_tokens, m.rms_epsilon, stream);
        mat(p + "hc_" + kind + "_fn.weight", normalized.f(), projected.f(), true);
        // Each upload can evict previous cache entries, so acquire scalar tensors first,
        // then enqueue immediately. These small tensors together fit every legal budget.
        auto &base = weight(p + "hc_" + kind + "_base.weight");
        auto &scale = weight(p + "hc_" + kind + "_scale.weight");
        if (fast) {
            k::glm_mhc_read_batch(r, projected.f(), base.f(), scale.f(), c, x, m.hidden,
                                  m.sinkhorn_iterations, m.hc_epsilon, batch_tokens, stream);
            return;
        }
        for (int t = 0; t < batch_tokens; ++t)
            k::glm_mhc_read(r + t * 4 * m.hidden, projected.f(t * 24), base.f(), scale.f(), c + t * 24,
                            x + t * m.hidden, m.hidden, m.sinkhorn_iterations, m.hc_epsilon, stream);
    }
    void ffn(const std::string &p, const float *x, float *out, const std::string &suffix, int ff,
             float limit) {
        auto &gate = buf("ff_gate", ff * batch_tokens);
        auto &up = buf("ff_up", ff * batch_tokens);
        auto &hidden = buf("ff_hidden", ff * batch_tokens);
        mat(p + "ffn_gate" + suffix + ".weight", x, gate.f());
        mat(p + "ffn_up" + suffix + ".weight", x, up.f());
        k::glm_swiglu(gate.f(), up.f(), hidden.f(), ff * batch_tokens, limit, stream);
        mat(p + "ffn_down" + suffix + ".weight", hidden.f(), out);
    }
    void moe_gpu(const std::string &p, const float *x, float *out,
                 const strata::core::LayerDescriptor &layer) {
        const int nt = batch_tokens, H = m.hidden, F = layer.intermediate, E = m.experts, K = m.top_k;
        auto &logits = buf("router_logits", nt * E);
        auto &ids = buf("router_ids", nt * K);
        auto &rw = buf("router_weights", nt * K);
        auto &bounds = buf("router_bounds", E + 1);
        auto &dest = buf("router_dest", nt * K + 128);
        auto &source = buf("router_source", nt * K + 128);
        auto &cursor = buf("router_cursor", E);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        k::glm_route_batch(logits.f(), bias.f(), (int *)ids.p, rw.f(), E, K, m.expert_scale, nt, stream);
        k::glm_group_routes((int *)ids.p, (int *)bounds.p, (int *)dest.p, (int *)source.p, (int *)cursor.p, E,
                            K, nt, stream);
        check(cudaMemcpyAsync(gpu->counts->p, bounds.p, (E + 1) * 4, cudaMemcpyDeviceToHost, stream));
        // Shared FFN can run while the tiny host routing table is being read.
        ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
        check(cudaStreamSynchronize(stream));
        const int *hb = (int *)gpu->counts->p;
        if (decode_cache_budget) {
            auto &frequency = prefill_routes.at(std::stoi(p.substr(4)));
            for (int e = 0; e < E; ++e) frequency[e] += hb[e + 1] - hb[e];
        }
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        size_t gh = G.bytes / E, db = D.bytes / E;
        auto &result = buf("moe_results", (size_t)nt * K * H);
        const int max_rows = std::min(4096, nt * K);
        auto &gu = buf("moe_gu", (size_t)max_rows * 2 * F);
        auto &hidden = buf("moe_hidden", (size_t)max_rows * F);
        auto &qx = buf("moe_qx", (mmq::q8_bytes(max_rows, H) + 3) / 4);
        auto &qh = buf("moe_qh", (mmq::q8_bytes(max_rows, F) + 3) / 4);
        auto &local = buf("moe_local_bounds", 17);
        auto &identity = buf("moe_identity", max_rows + 128);
        mmq::iota((int *)identity.p, max_rows + 128, stream);
        for (int first = 0; first < E; first += 16) {
            check_stop();
            int n = std::min(16, E - first);
            int begin = hb[first], end = hb[first + n];
            if (begin == end)
                continue;
            int slot = gpu->groups % 2;
            gpu->upload(G, U, D, first, n, slot, stream);
            for (int offset = begin; offset < end; offset += max_rows) {
                check_stop();
                int rows = std::min(max_rows, end - offset), most = 0;
                for (int e = first; e < first + n; ++e)
                    most = std::max(
                        most, std::max(0, std::min(offset + rows, hb[e + 1]) - std::max(offset, hb[e])));
                k::glm_group_bounds((int *)bounds.p, (int *)local.p, first, n, offset, rows, stream);
                mmq::quantize(x, (int *)source.p + offset, qx.p, G.tensor->type, H, H, rows, stream);
                mmq::Product gate{gpu->slots[slot]->p, (int)G.tensor->type, 2 * F, H,    2 * gh, n,    qx.p,
                                  (int *)local.p,      (int *)identity.p,   rows,  most, gu.f(), 2 * F};
                gpu->context->run(gate, stream);
                mmq::swiglu(gu.f(), hidden.f(), rows, F, false, stream, layer.swiglu_limit);
                mmq::quantize(hidden.f(), nullptr, qh.p, D.tensor->type, F, F, rows, stream);
                mmq::Product down{(char *)gpu->slots[slot]->p + n * 2 * gh,
                                  (int)D.tensor->type,
                                  H,
                                  F,
                                  db,
                                  n,
                                  qh.p,
                                  (int *)local.p,
                                  (int *)dest.p + offset,
                                  rows,
                                  most,
                                  result.f(),
                                  H};
                gpu->context->run(down, stream);
            }
            check(cudaEventRecord(gpu->done[slot], stream));
        }
        k::glm_route_sum(result.f(), rw.f(), out, H, K, nt, stream);
    }

    void moe_streamed(const std::string &p, const float *x, float *out,
                      const strata::core::LayerDescriptor &layer) {
        const int nt = batch_tokens, H = m.hidden, E = m.experts, K = m.top_k;
        auto &logits = buf("decode_router_logits", E * nt);
        auto &ids = buf("decode_router_ids", K * nt);
        auto &weights = buf("decode_router_weights", K * nt);
        auto &bounds = buf("decode_router_bounds", E + 1);
        auto &dest = buf("decode_router_dest", K * nt + 128);
        auto &source = buf("decode_router_source", K * nt + 128);
        auto &cursor = buf("decode_router_cursor", E);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        k::glm_route_batch(logits.f(), bias.f(), (int *)ids.p, weights.f(), E, K, m.expert_scale, nt, stream);
        k::glm_group_routes((int *)ids.p, (int *)bounds.p, (int *)dest.p, (int *)source.p, (int *)cursor.p, E,
                            K, nt, stream);
        check(cudaMemcpyAsync(gpu->counts->p, bounds.p, (E + 1) * 4, cudaMemcpyDeviceToHost, stream));
        ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
        check(cudaStreamSynchronize(stream));
        const auto *hb = (const int *)gpu->counts->p;
        std::vector<int> active;
        for (int e = 0; e < E; ++e)
            if (hb[e + 1] != hb[e])
                active.push_back(e);
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        const auto layout = k::native_expert_layout(G.tensor->type, D.tensor->type, H, layer.intermediate);
        auto clamped = layout;
        clamped.swiglu_limit = layer.swiglu_limit;
        const size_t blob_bytes = (G.bytes + U.bytes + D.bytes) / E;
        auto &qa = buf("decode_q8", (k::native_q8_1_bytes(H, nt) + 3) / 4);
        auto &result = buf("decode_experts", (size_t)K * nt * H);
        auto &scratch = buf("decode_expert_scratch",
                            (k::native_expert_scratch_bytes(K * nt, layer.intermediate) + 3) / 4);
        auto &pointers = buf("decode_expert_pointers", 32);
        auto &starts = buf("decode_expert_starts", 17);
        auto &groups = buf("decode_expert_groups", 1);
        k::native_quantize_q8_1(x, qa.p, H, nt, stream);
        for (size_t first = 0; first < active.size(); first += 16) {
            check_stop();
            const int n = std::min<size_t>(16, active.size() - first), slot = gpu->groups % 2;
            const int begin = hb[active[first]], end = hb[active[first + n - 1] + 1];
            gpu->upload(G, U, D, 0, n, slot, stream, active.data() + first, true);
            std::vector<unsigned long long> addresses(n);
            std::vector<int> local(n + 1);
            for (int j = 0; j < n; ++j) {
                addresses[j] = (unsigned long long)gpu->slots[slot]->p + j * blob_bytes;
                local[j] = hb[active[first + j]] - begin;
            }
            local[n] = end - begin;
            pointers.put(addresses.data(), n * 8);
            starts.put(local.data(), (n + 1) * 4);
            groups.put(&n, 4);
            k::native_expert_grouped(clamped, (const unsigned long long *)pointers.p, (const int *)starts.p,
                                     (const int *)groups.p, (const int *)dest.p + begin,
                                     (const int *)source.p + begin, n, end - begin, qa.p, scratch.p,
                                     result.f(), stream);
            check(cudaEventRecord(gpu->done[slot], stream));
        }
        k::glm_route_sum(result.f(), weights.f(), out, H, K, nt, stream);
    }
    void moe_mtp(const std::string &p, const float *x, float *out,
                 const strata::core::LayerDescriptor &layer) {
        const int H = m.hidden, K = m.top_k;
        auto &logits = buf("mtp_router", m.experts), &ids = buf("mtp_ids", K);
        auto &routing = buf("mtp_routing", K);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        k::glm_router(logits.f(), bias.f(), (int *)ids.p, routing.f(), m.experts, K, m.expert_scale, stream);
        auto *selected = (int *)host_moe->p;
        check(cudaMemcpyAsync(selected, ids.p, K * 4, cudaMemcpyDeviceToHost, stream));
        auto *activation = (float *)(selected + cpu::MAXT * K) + cpu::MAXT * K;
        if (mtp_cpu_experts)
            check(cudaMemcpyAsync(activation, x, H * 4, cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecord(moe_ready, stream));
        ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
        check(cudaEventSynchronize(moe_ready));
        if (mtp_cpu_experts) {
            const auto &G = artifact.at(p + "ffn_gate_exps.weight");
            const auto &U = artifact.at(p + "ffn_up_exps.weight");
            const auto &D = artifact.at(p + "ffn_down_exps.weight");
            cpu::NativeFmt format;
            std::string error;
            if (!cpu::native_fmt(G.tensor->type, D.tensor->type, H, layer.intermediate, format, error))
                throw std::runtime_error(error);
            format.swiglu_limit = layer.swiglu_limit;
            host_quant.resize(1);
            host_quant[0].resize(format.act_bytes);
            cpu::native_quant_act(format, activation, host_quant[0].data());
            host_results.resize(K * H);
            host_jobs.assign(K, cpu::ExpertJobMulti{});
            for (int j = 0; j < K; ++j) {
                auto &job = host_jobs[j];
                job.blob = G.data() + selected[j] * (G.bytes / m.experts);
                job.native_up = U.data() + selected[j] * (U.bytes / m.experts);
                job.native_down = D.data() + selected[j] * (D.bytes / m.experts);
                job.nt = 1;
                job.nact[0] = host_quant[0].data();
                job.out[0] = host_results.data() + j * H;
            }
            pool.run_split_multi_native(format, host_jobs.data(), K);
            auto &result = buf("mtp_expert_results", K * H);
            result.put(host_results.data(), K * H * 4);
            k::glm_route_sum(result.f(), routing.f(), out, H, K, 1, stream);
            return;
        }
        std::vector<unsigned long long> addresses(K);
        std::vector<int> starts(K + 1), dest(K), source(K, 0);
        for (int j = 0; j < K; ++j) {
            addresses[j] = (unsigned long long)mtp_experts->p + selected[j] * mtp_expert_bytes;
            starts[j] = dest[j] = j;
        }
        starts[K] = K;
        auto &ptr = buf("mtp_pointers", K * 2), &bounds = buf("mtp_bounds", K + 1);
        auto &dst = buf("mtp_dest", K), &src = buf("mtp_source", K), &groups = buf("mtp_groups", 1);
        auto &qa = buf("mtp_q8", (k::native_q8_1_bytes(H, 1) + 3) / 4);
        auto &scratch = buf("mtp_expert_scratch", (k::native_expert_scratch_bytes(K, layer.intermediate) + 3) / 4);
        auto &result = buf("mtp_expert_results", K * H);
        ptr.put(addresses.data(), K * 8); bounds.put(starts.data(), (K + 1) * 4);
        dst.put(dest.data(), K * 4); src.put(source.data(), K * 4); groups.put(&K, 4);
        k::native_quantize_q8_1(x, qa.p, H, 1, stream);
        k::native_expert_grouped(mtp_layout, (const unsigned long long *)ptr.p, (const int *)bounds.p,
                                 (const int *)groups.p, (const int *)dst.p, (const int *)src.p, K, K,
                                 qa.p, scratch.p, result.f(), stream);
        k::glm_route_sum(result.f(), routing.f(), out, H, K, 1, stream);
    }
    void moe(const std::string &p, int l, const float *x, float *out,
             const strata::core::LayerDescriptor &layer) {
        if (l == (int)m.layers.size() && mtp_ready) {
            moe_mtp(p, x, out, layer);
            return;
        }
        if (!fast && gpu_decode_experts) {
            moe_streamed(p, x, out, layer);
            return;
        }
        if (fast) {
            moe_gpu(p, x, out, layer);
            return;
        }
        const int nt = batch_tokens;
        auto &logits = buf("router_logits", m.experts * nt);
        auto &ids = buf("router_ids", m.top_k * nt);
        auto &rw = buf("router_weights", m.top_k * nt);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        for (int t = 0; t < nt; ++t)
            k::glm_router(logits.f(t * m.experts), bias.f(), (int *)ids.p + t * m.top_k, rw.f(t * m.top_k),
                          m.experts, m.top_k, m.expert_scale, stream);
        auto *host_ids = (int *)host_moe->p;
        auto *routing = (float *)(host_ids + cpu::MAXT * m.top_k);
        auto *activation = routing + cpu::MAXT * m.top_k;
        check(cudaMemcpyAsync(host_ids, ids.p, m.top_k * nt * 4, cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(routing, rw.p, m.top_k * nt * 4, cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(activation, x, m.hidden * nt * 4, cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecord(moe_ready, stream));
        // Shared FFN overlaps the routing readback and CPU routed experts.
        ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
        check(cudaEventSynchronize(moe_ready));
        host_selected.assign(host_ids, host_ids + m.top_k * nt);
        auto &selected = host_selected;
        if (routing_trace.is_open()) {
            for (int t = 0; t < nt; ++t) {
                routing_trace << position + t << ',' << l;
                for (int j = 0; j < m.top_k; ++j) routing_trace << ',' << selected[t * m.top_k + j];
                routing_trace << '\n';
            }
            if (!routing_trace) throw std::runtime_error("GLM: failed writing routing trace");
        }
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        cpu::NativeFmt f;
        std::string error;
        if (!cpu::native_fmt(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate, f, error))
            throw std::runtime_error(error);
        f.swiglu_limit = layer.swiglu_limit;
        auto prepared = prepared_experts.find(l);
        if (prepared != prepared_experts.end()) f = prepared->second.format;
        host_quant.resize(nt);
        for (auto &q : host_quant) q.resize(f.act_bytes);
        auto &quant = host_quant;
        for (int t = 0; t < nt; ++t)
            cpu::native_quant_act(f, activation + t * m.hidden, quant[t].data());
        std::map<int, size_t> groups;
        std::vector<std::vector<uint8_t>> blobs;
        host_results.resize(selected.size() * m.hidden);
        auto &results = host_results;
        host_jobs.clear();
        auto &jobs = host_jobs;
        auto *cache = expert_cache[l].get();
        std::map<int, std::vector<int>> hits;
        for (size_t j = 0; j < selected.size(); ++j) {
            const int e = selected[j], t = j / m.top_k;
            ++cache_entries;
            if (decode_resident[l].count(e) || (cache && cache->slot_of(0, e) >= 0)) {
                hits[e].push_back(j);
                ++cache_hits;
                continue;
            }
            auto [where, inserted] = groups.emplace(e, jobs.size());
            if (inserted && !cache) {
                jobs.emplace_back();
                auto &job = jobs.back();
                const auto *gate = prepared == prepared_experts.end() ? G.data() : prepared->second.gate.get();
                const auto *up = prepared == prepared_experts.end() ? U.data() : prepared->second.up.get();
                const auto *down = prepared == prepared_experts.end() || !prepared->second.down ? D.data() : prepared->second.down.get();
                job.blob = gate + (size_t)e * f.up_off;
                job.native_up = up + (size_t)e * f.up_off;
                job.native_down = down + (size_t)e * (f.bytes - f.down_off);
            } else if (inserted) {
                blobs.emplace_back(f.bytes);
                auto &blob = blobs.back();
                jobs.emplace_back();
                std::memcpy(blob.data(), G.data() + (size_t)e * f.up_off, f.up_off);
                std::memcpy(blob.data() + f.up_off, U.data() + (size_t)e * f.up_off, f.up_off);
                std::memcpy(blob.data() + f.down_off, D.data() + (size_t)e * (f.bytes - f.down_off),
                            f.bytes - f.down_off);
                jobs.back().blob = blob.data();
            }
            auto &job = jobs[where->second];
            const int slot = job.nt++;
            job.nact[slot] = quant[t].data();
            job.out[slot] = results.data() + j * m.hidden;
        }
        std::vector<unsigned long long> pointers;
        std::vector<int> starts = {0}, dest, tok;
        for (const auto &[e, entries] : hits) {
            const auto fixed = decode_resident[l].find(e);
            pointers.push_back((unsigned long long)(fixed != decode_resident[l].end()
                ? fixed->second->p : cache->device_slot(cache->slot_of(0, e))));
            for (int j : entries) {
                dest.push_back(j);
                tok.push_back(j / m.top_k);
            }
            starts.push_back(dest.size());
        }
        Device *dp = nullptr, *ds = nullptr, *dd = nullptr, *dt = nullptr, *dn = nullptr, *qa = nullptr,
               *scr = nullptr, *gpu_out = nullptr;
        if (!hits.empty()) {
            const size_t maximum = cpu::MAXT * m.top_k;
            if (selected.size() > maximum || pointers.size() > maximum)
                throw std::runtime_error("GLM: cache routing exceeds pinned buffer geometry");
            if (!host_hit_metadata) {
                auto metadata = std::make_unique<Pinned>(maximum * 8 + (3 * maximum + 2) * 4);
                auto results_buffer = std::make_unique<Pinned>(maximum * m.hidden * 4);
                host_hit_metadata = std::move(metadata);
                host_hit_results = std::move(results_buffer);
            }
            auto *hp = (unsigned long long *)host_hit_metadata->p;
            auto *hs = (int *)(hp + maximum);
            auto *hd = hs + maximum + 1;
            auto *ht = hd + maximum;
            auto *hn = ht + maximum;
            std::copy(pointers.begin(), pointers.end(), hp);
            std::copy(starts.begin(), starts.end(), hs);
            std::copy(dest.begin(), dest.end(), hd);
            std::copy(tok.begin(), tok.end(), ht);
            *hn = pointers.size();
            dp = &buf("hit_pointers", pointers.size() * 2);
            ds = &buf("hit_starts", starts.size());
            dd = &buf("hit_dest", dest.size());
            dt = &buf("hit_tok", tok.size());
            dn = &buf("hit_groups", 1);
            qa = &buf("hit_q8", k::native_q8_1_bytes(m.hidden, nt) / 4);
            scr = &buf("hit_scratch", k::native_expert_scratch_bytes(dest.size(), f.n_ff) / 4);
            gpu_out = &buf("hit_output", results.size());
            dp->put_async(hp, pointers.size() * 8, stream);
            ds->put_async(hs, starts.size() * 4, stream);
            dd->put_async(hd, dest.size() * 4, stream);
            dt->put_async(ht, tok.size() * 4, stream);
            dn->put_async(hn, 4, stream);
            check(cudaMemsetAsync(gpu_out->p, 0, results.size() * 4, stream));
            k::native_quantize_q8_1(x, qa->p, m.hidden, nt, stream);
        }
        // Resident experts can overlap CPU misses.
        if (!hits.empty()) {
            auto layout = k::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            layout.swiglu_limit = f.swiglu_limit;
            k::native_expert_grouped(layout, (const unsigned long long *)dp->p, (const int *)ds->p,
                                     (const int *)dn->p, (const int *)dd->p, (const int *)dt->p,
                                     pointers.size(), dest.size(), qa->p, scr->p, gpu_out->f(), stream);
            check(cudaMemcpyAsync(host_hit_results->p, gpu_out->p, results.size() * 4,
                                  cudaMemcpyDeviceToHost, stream));
        }
        pool.run_split_multi_native(f, jobs.data(), jobs.size());
        check(cudaStreamSynchronize(stream));
        if (!hits.empty()) {
            const auto *resident = (const float *)host_hit_results->p;
            for (int j : dest)
                std::copy_n(resident + j * m.hidden, m.hidden, results.data() + j * m.hidden);
        }
        if (cache && !cache_frozen) {
            for (int e : selected)
                if (cache->slot_of(0, e) < 0) {
                    const int slot = cache->admit(0, e);
                    if (slot >= 0 &&
                        !cache->fill_slot(slot, blobs.at(groups.at(e)).data(), stream, error, f.bytes))
                        throw std::runtime_error(error);
                }
            // The miss staging buffers must outlive their asynchronous uploads.
            check(cudaStreamSynchronize(stream));
        }
        host_sum.assign(m.hidden * nt, 0.f);
        auto &sum = host_sum;
        for (size_t j = 0; j < selected.size(); ++j)
            for (int i = 0; i < m.hidden; ++i)
                sum[(j / m.top_k) * m.hidden + i] += routing[j] * results[j * m.hidden + i];
        auto &routed = buf("moe_sum", m.hidden * nt);
        routed.put(sum.data(), sum.size() * 4);
        const float one = 1;
        check(cublasSaxpy(blas, m.hidden * nt, &one, routed.f(), 1, out, 1));
    }
    void kda(const std::string &p, int l, const float *x, float *out) {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads;
        auto &state = states[l];
        auto &q = buf("linear_q", n * batch_tokens);
        auto &key = buf("linear_k", n * batch_tokens);
        auto &v = buf("linear_v", n * batch_tokens);
        auto &tmp = buf("linear_raw", n * batch_tokens);
        const std::array<std::string, 3> names = {"q", "k", "v"};
        const std::array<float *, 3> dest = {q.f(), key.f(), v.f()};
        const std::array<Device *, 3> hist = {state.conv_q.get(), state.conv_k.get(), state.conv_v.get()};
        for (int i = 0; i < 3; ++i) {
            mat(p + "attn_" + names[i] + ".weight", x, tmp.f());
            auto &W = weight(p + "ssm_conv1d_" + names[i] + ".weight");
            if (fast)
                k::glm_conv_batch(tmp.f(), W.f(), hist[i]->f(), dest[i], n, m.conv_kernel, batch_tokens,
                                  stream);
            else
                for (int t = 0; t < batch_tokens; ++t) {
                    k::glm_conv(tmp.f(t * n), W.f(), hist[i]->f(), dest[i] + t * n, n, m.conv_kernel, stream);
                    capture_state(hist[i], t);
                }
        }
        auto &low = buf("linear_low", dim * batch_tokens);
        auto &decay = buf("linear_decay", n * batch_tokens);
        auto &beta = buf("linear_beta", heads * batch_tokens);
        mat(p + "ssm_f_a.weight", x, low.f());
        mat(p + "ssm_f_b.weight", low.f(), tmp.f());
        auto &dt = weight(p + "ssm_dt.bias");
        auto &a = weight(p + "ssm_a");
        if (fast)
            k::glm_kda_gate_batch(tmp.f(), dt.f(), a.f(), decay.f(), heads, dim, m.gate_lower_bound,
                                  batch_tokens, stream);
        else
            for (int t = 0; t < batch_tokens; ++t)
                k::glm_kda_gate(tmp.f(t * n), dt.f(), a.f(), decay.f(t * n), heads, dim, m.gate_lower_bound,
                                stream);
        mat(p + "ssm_beta.weight", x, beta.f());
        auto &y = buf("linear_y", n * batch_tokens);
        if (gpu) {
            const int chunk = capturing_history ? 1 : 64;
            for (int t = 0; t < batch_tokens; t += chunk) {
                check_stop();
                int count = std::min(chunk, batch_tokens - t);
                k::glm_kda_chunk(state.recurrent->f(), q.f(t * n), key.f(t * n), v.f(t * n), decay.f(t * n),
                                 beta.f(t * heads), y.f(t * n), heads, dim, count, stream);
                capture_state(state.recurrent.get(), t);
            }
        } else
            for (int t = 0; t < batch_tokens; ++t)
                k::glm_kda_step(state.recurrent->f(), q.f(t * n), key.f(t * n), v.f(t * n), decay.f(t * n),
                                beta.f(t * heads), y.f(t * n), heads, dim, stream);
        mat(p + "ssm_g_a.weight", x, low.f());
        mat(p + "ssm_g_b.weight", low.f(), tmp.f());
        auto &W = weight(p + "ssm_norm.weight");
        if (fast)
            k::glm_kda_output_batch(y.f(), tmp.f(), W.f(), q.f(), heads, dim, m.rms_epsilon, batch_tokens,
                                    stream);
        else
            for (int t = 0; t < batch_tokens; ++t)
                k::glm_kda_output(y.f(t * n), tmp.f(t * n), W.f(), q.f(t * n), heads, dim, m.rms_epsilon,
                                  stream);
        mat(p + "attn_output.weight", q.f(), out);
    }
    void head_mat(const std::string &name, const float *x, float *y) {
        const auto &t = *artifact.at(name).tensor;
        auto &W = weight(name, true);
        const int in = t.shape[0], out = t.shape[1], heads = t.shape[2];
        const float one = 1, zero = 0;
        check(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, 1, in, &one, W.f(), in,
                                        (long long)in * out, x, in, in, &zero, y, out, out, heads));
    }
    void head_batch(const std::string &name, const float *x, float *y, int tokens, int ld_x, int ld_y) {
        const auto &t = *artifact.at(name).tensor;
        auto &W = weight(name, true);
        int in = t.shape[0], out = t.shape[1], heads = t.shape[2];
        const float one = 1, zero = 0;
        check(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one, W.f(), in,
                                        (long long)in * out, x, ld_x, in, &zero, y, ld_y, out, heads));
    }
    void mla_gpu(const std::string &p, int l, const float *x, float *out) {
        auto &state = states[l];
        int B = batch_tokens, heads = m.attention_heads, latent = m.kv_rank, dim = 256, ih = m.index_heads,
            idim = m.index_dim, pool_size = m.index_pool;
        const int tile = fast ? 64 : 1;
        constexpr int stride = 2052;
        auto &low = buf("mla_low", m.q_rank * B);
        auto &qr = buf("mla_qr", m.q_rank * B);
        auto &q = buf("mla_q", heads * dim * B);
        mat(p + "attn_q_a.weight", x, low.f());
        norm(p + "attn_q_a_norm.weight", low.f(), qr.f(), m.q_rank);
        mat(p + "attn_q_b.weight", qr.f(), q.f());
        auto &raw = buf("mla_raw", latent * B);
        mat(p + "attn_kv_a_mqa.weight", x, raw.f());
        norm(p + "attn_kv_a_norm.weight", raw.f(), state.cache->f((size_t)position * latent), latent);
        auto &iq = buf("index_q", ih * idim * B);
        auto &iw = buf("index_w", ih * B);
        auto &ik = buf("index_k", idim * B);
        auto &ikn = buf("index_kn", idim * B);
        auto &gate = buf("index_gate", idim * B);
        mat(p + "indexer.attn_q_b.weight", qr.f(), iq.f());
        mat(p + "indexer.proj.weight", x, iw.f());
        mat(p + "indexer.attn_k.weight", x, ik.f());
        mat(p + "indexer_compressor_gate.weight", x, gate.f());
        auto &knw = weight(p + "indexer.k_norm.weight");
        auto &knb = weight(p + "indexer.k_norm.bias");
        k::glm_layer_norm_batch(ik.f(), knw.f(), knb.f(), ikn.f(), idim, B, 1e-6, stream);
        auto &ape = weight(p + "indexer_compressor_ape.weight");
        if (capturing_history) {
            for (int t = 0; t < B; ++t) {
                k::glm_index_prepare(ikn.f(t * idim), gate.f(t * idim), ape.f(), state.keys->f(),
                                     state.gates->f(), state.pooled->f(), position + t, 1, pool_size, idim, stream);
                capture_state(state.keys.get(), t);
                capture_state(state.gates.get(), t);
            }
        } else
            k::glm_index_prepare(ikn.f(), gate.f(), ape.f(), state.keys->f(), state.gates->f(), state.pooled->f(),
                                 position, B, pool_size, idim, stream);
        // Fixed pool leading dimension avoids batch-dependent SGEMM reductions.
        int pools = capacity / pool_size, ntile = std::min(tile, B);
        auto &dots = buf("index_dots", (size_t)ntile * ih * std::max(1, pools));
        auto &scores = buf("index_scores", (size_t)ntile * std::max(1, pools));
        auto &ids = buf("index_ids", ntile * stride);
        auto &counts = buf("index_counts", ntile);
        auto &aq = buf("mla_absorbed_q", ntile * heads * latent);
        auto &av = buf("mla_absorbed_v", ntile * heads * latent);
        auto &gathered = buf("mla_gathered", (size_t)ntile * stride * latent);
        auto &attn = buf("mla_scores", ntile * heads * stride);
        auto &value = buf("mla_value", heads * dim * B);
        const float one = 1, zero = 0;
        for (int t = 0; t < B; t += tile) {
            check_stop();
            int count = std::min(tile, B - t);
            if (pools)
                check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, pools, count * ih, idim, &one,
                                  state.pooled->f(), idim, iq.f(t * ih * idim), idim, &zero, dots.f(),
                                  pools));
            k::glm_index_reduce(dots.f(), iw.f(t * ih), scores.f(), ih, pools, count, position + t, pool_size,
                                idim, stream);
            k::glm_index_select_batch(scores.f(), (int *)ids.p, (int *)counts.p, pools, position + t, count,
                                      pool_size, m.index_top_k, stride, stream);
            head_batch(p + "attn_k_b.weight", q.f(t * heads * dim), aq.f(), count, heads * dim,
                       heads * latent);
            k::glm_mla_gather(state.cache->f(), (int *)ids.p, (int *)counts.p, gathered.f(), count, stride,
                              latent, stream);
            check(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, stride, heads, latent, &one,
                                            gathered.f(), latent, (long long)stride * latent, aq.f(), latent,
                                            (long long)heads * latent, &zero, attn.f(), stride,
                                            (long long)heads * stride, count));
            k::glm_mla_softmax(attn.f(), (int *)counts.p, heads, stride, count, 1.f / std::sqrt((float)dim),
                               stream);
            check(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, latent, heads, stride, &one,
                                            gathered.f(), latent, (long long)stride * latent, attn.f(),
                                            stride, (long long)heads * stride, &zero, av.f(), latent,
                                            (long long)heads * latent, count));
            head_batch(p + "attn_v_b.weight", av.f(), value.f(t * heads * dim), count, heads * latent,
                       heads * dim);
        }
        mat(p + "attn_output.weight", value.f(), out);
    }

    void mla(const std::string &p, int l, const float *x, float *out) {
        if (gpu) {
            mla_gpu(p, l, x, out);
            return;
        }
        auto &state = states[l];
        const int heads = m.attention_heads, latent = m.kv_rank, idim = m.index_dim;
        // The local artifact is NoPE with 256-dimensional Q/K and V heads.
        const int dim = artifact.at(p + "attn_k_b.weight").tensor->shape[0];
        auto &low = buf("mla_low", m.q_rank * batch_tokens);
        auto &qr = buf("mla_qr", m.q_rank * batch_tokens);
        auto &q = buf("mla_q", heads * dim * batch_tokens);
        mat(p + "attn_q_a.weight", x, low.f());
        norm(p + "attn_q_a_norm.weight", low.f(), qr.f(), m.q_rank);
        mat(p + "attn_q_b.weight", qr.f(), q.f());
        auto &raw = buf("mla_raw", latent * batch_tokens);
        mat(p + "attn_kv_a_mqa.weight", x, raw.f());
        norm(p + "attn_kv_a_norm.weight", raw.f(), state.cache->f((size_t)position * latent), latent);
        auto &iq = buf("index_q", m.index_heads * idim * batch_tokens);
        auto &iw = buf("index_w", m.index_heads * batch_tokens);
        auto &ik = buf("index_k", idim * batch_tokens);
        auto &gate = buf("index_gate", idim * batch_tokens);
        mat(p + "indexer.attn_q_b.weight", qr.f(), iq.f());
        mat(p + "indexer.proj.weight", x, iw.f());
        mat(p + "indexer.attn_k.weight", x, ik.f());
        mat(p + "indexer_compressor_gate.weight", x, gate.f());
        auto &scores = buf("index_scores", std::max<int64_t>(1, (position + batch_tokens) / m.index_pool));
        auto &ids = buf("index_ids", m.index_top_k + m.index_pool - 1);
        auto &count = buf("index_count", 1);
        auto &aq = buf("mla_absorbed_query", heads * latent);
        auto &av = buf("mla_absorbed_value", heads * latent);
        auto &value = buf("mla_value", heads * dim * batch_tokens);
        for (int t = 0; t < batch_tokens; ++t) {
            const int tokens = position + t + 1, pools = tokens / m.index_pool;
            const size_t tail = (position + t) % m.index_pool;
            auto &knw = weight(p + "indexer.k_norm.weight");
            auto &knb = weight(p + "indexer.k_norm.bias");
            k::glm_layer_norm(ik.f(t * idim), knw.f(), knb.f(), state.keys->f(tail * idim), idim, 1e-6,
                              stream);
            check(cudaMemcpyAsync(state.gates->f(tail * idim), gate.f(t * idim), idim * 4,
                                  cudaMemcpyDeviceToDevice, stream));
            if (tokens % m.index_pool == 0) {
                auto &ape = weight(p + "indexer_compressor_ape.weight");
                k::glm_index_pool(state.keys->f(), state.gates->f(), ape.f(),
                                  state.pooled->f((size_t)(pools - 1) * idim), m.index_pool, idim, stream);
            }
            k::glm_index_score(iq.f(t * m.index_heads * idim), iw.f(t * m.index_heads), state.pooled->f(),
                               scores.f(), m.index_heads, idim, pools, stream);
            k::glm_index_select(scores.f(), (int *)ids.p, (int *)count.p, tokens, m.index_pool, m.index_top_k,
                                stream);
            head_mat(p + "attn_k_b.weight", q.f(t * heads * dim), aq.f());
            k::glm_mla(aq.f(), state.cache->f(), (int *)ids.p, (int *)count.p, av.f(), heads, latent,
                       1.f / std::sqrt((float)dim), stream);
            head_mat(p + "attn_v_b.weight", av.f(), value.f(t * heads * dim));
        }
        mat(p + "attn_output.weight", value.f(), out);
    }

  public:
    struct Snapshot {
        int position, mtp_position;
        std::vector<std::array<std::vector<float>, 8>> layers;
    };
    Snapshot snapshot() const {
        check(cudaStreamSynchronize(stream));
        Snapshot result{position, mtp_position, {}};
        result.layers.resize(states.size());
        for (size_t l = 0; l < states.size(); ++l) {
            const auto &s = states[l];
            const std::array<const Device *, 8> buffers = {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(),
                                                           s.conv_v.get(),    s.cache.get(),  s.keys.get(),
                                                           s.gates.get(),     s.pooled.get()};
            for (size_t b = 0; b < buffers.size(); ++b)
                if (buffers[b])
                    result.layers[l][b] = buffers[b]->floats(buffers[b]->bytes / 4);
        }
        return result;
    }
    void restore(const Snapshot &snapshot) {
        check(cudaStreamSynchronize(stream));
        if (snapshot.layers.size() != states.size() || snapshot.position < 0 ||
            snapshot.position > (int)capacity)
            throw std::invalid_argument("GLM: incompatible state snapshot");
        for (size_t l = 0; l < states.size(); ++l) {
            auto &s = states[l];
            const std::array<Device *, 8> buffers = {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(),
                                                     s.conv_v.get(),    s.cache.get(),  s.keys.get(),
                                                     s.gates.get(),     s.pooled.get()};
            for (size_t b = 0; b < buffers.size(); ++b) {
                const auto &data = snapshot.layers[l][b];
                if (data.size() * 4 != (buffers[b] ? buffers[b]->bytes : 0))
                    throw std::invalid_argument("GLM: incompatible snapshot geometry");
                if (buffers[b])
                    buffers[b]->put(data.data(), data.size() * 4);
            }
        }
        position = snapshot.position;
        mtp_position = snapshot.mtp_position;
    }
    Decoder(const std::string &path, size_t ctx, size_t cache_bytes, int threads, size_t expert_bytes = 0, bool pin_cpu = false)
        : artifact(path), m(artifact.descriptor()), pool(threads, pin_cpu, true), capacity(ctx),
          budget(cache_bytes) {
        if (m.architecture != "glm5next" || ctx < 1 || ctx > (size_t)m.context || budget < 64 * 1024 * 1024)
            throw std::invalid_argument("GLM: invalid architecture, context, or cache budget");
        validate();
        prepack_cpus = cpu::physical_cores(false);
        if (pin_cpu) {
            const auto cores = cpu::physical_cores(false);
            if (!cores.empty()) host_affinity = cpu::pin_current_thread(cores.front());
        }
        host_moe = std::make_unique<Pinned>(cpu::MAXT * (2 * m.top_k + m.hidden) * 4);
        check(cudaStreamCreate(&stream));
        check(cudaEventCreateWithFlags(&moe_ready, cudaEventDisableTiming));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream));
        states.resize(m.layers.size());
        decode_resident.resize(m.layers.size());
        expert_cache.resize(m.layers.size());
        for (size_t l = 0; l < states.size(); ++l) {
            auto &s = states[l];
            if (m.layers[l].mixer == strata::core::MixerKind::Kda) {
                s.recurrent =
                    std::make_unique<Device>(m.linear_heads * m.linear_dim * m.linear_dim * 4, true);
                const size_t conv = m.linear_heads * m.linear_dim * (m.conv_kernel - 1) * 4;
                s.conv_q = std::make_unique<Device>(conv, true);
                s.conv_k = std::make_unique<Device>(conv, true);
                s.conv_v = std::make_unique<Device>(conv, true);
            } else {
                s.cache = std::make_unique<Device>(ctx * m.kv_rank * 4, true);
                s.keys = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
                s.gates = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
                s.pooled =
                    std::make_unique<Device>(std::max<size_t>(1, ctx / m.index_pool) * m.index_dim * 4, true);
            }
        }
        if (expert_bytes) {
            size_t moe_layers = 0;
            for (const auto &layer : m.layers)
                moe_layers += layer.ffn == strata::core::FfnKind::Moe;
            if (!moe_layers)
                throw std::runtime_error("GLM: expert cache requested for a model without MoE layers");
            const size_t per_layer = expert_bytes / moe_layers;
            for (size_t l = 0; l < m.layers.size(); ++l)
                if (m.layers[l].ffn == strata::core::FfnKind::Moe) {
                    const auto p = "blk." + std::to_string(l) + ".";
                    cpu::NativeFmt f;
                    std::string error;
                    if (!cpu::native_fmt(artifact.at(p + "ffn_gate_exps.weight").tensor->type,
                                         artifact.at(p + "ffn_down_exps.weight").tensor->type, m.hidden,
                                         m.layers[l].intermediate, f, error))
                        throw std::runtime_error(error);
                    const size_t slots = std::min<size_t>(m.experts, per_layer / f.bytes);
                    if (!slots)
                        throw std::runtime_error(
                            "GLM: expert-cache budget needs at least one native blob per MoE layer");
                    expert_cache[l] = std::make_unique<strata::core::ExpertCache>();
                    if (!expert_cache[l]->open(slots, 1, m.experts, f.bytes, error))
                        throw std::runtime_error(error);
                }
        }
    }
    ~Decoder() {
        cpu::restore_thread_affinity(host_affinity);
        if (stream)
            cudaStreamSynchronize(stream);
        gpu.reset();
        if (moe_ready) cudaEventDestroy(moe_ready);
        if (blas)
            cublasDestroy(blas);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void validate() const {
        if (m.rope_dim != 0 || m.expert_groups != 1 || m.selected_groups != 1 || m.gating_function != 2 ||
            !m.normalize_expert_weights || m.shared_experts != 1 || m.conv_kernel < 2)
            throw std::runtime_error("GLM: this decoder requires NoPE MLA, normalized sigmoid routing, one "
                                     "expert group and one shared expert");
        const uint64_t H = m.hidden, E = m.experts, N = m.linear_heads * m.linear_dim;
        auto fp32 = [&](const std::string &name) {
            if (artifact.at(name).tensor->type != GGML_TYPE_F32)
                throw std::runtime_error("GLM: expected FP32 scalar/norm tensor: " + name);
        };
        artifact.shape("token_embd.weight", {H, (uint64_t)m.vocab});
        artifact.shape("output.weight", {H, (uint64_t)m.vocab});
        artifact.shape("output_norm.weight", {H});
        fp32("output_norm.weight");
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const auto &layer = m.layers[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const uint64_t ff = layer.intermediate;
            for (const char *part : {"attn", "ffn"}) {
                artifact.shape(p + part + "_norm.weight", {H});
                fp32(p + part + "_norm.weight");
                artifact.shape(p + "hc_" + part + "_fn.weight", {4 * H, 24});
                artifact.shape(p + "hc_" + part + "_base.weight", {24});
                artifact.shape(p + "hc_" + part + "_scale.weight", {3});
                fp32(p + "hc_" + part + "_base.weight");
                fp32(p + "hc_" + part + "_scale.weight");
            }
            if (layer.ffn == strata::core::FfnKind::Moe) {
                artifact.shape(p + "ffn_gate_exps.weight", {H, ff, E});
                artifact.shape(p + "ffn_up_exps.weight", {H, ff, E});
                artifact.shape(p + "ffn_down_exps.weight", {ff, H, E});
                if (artifact.at(p + "ffn_gate_exps.weight").tensor->type !=
                    artifact.at(p + "ffn_up_exps.weight").tensor->type)
                    throw std::runtime_error("GLM: gate/up types differ");
                artifact.shape(p + "ffn_gate_inp.weight", {H, E});
                artifact.shape(p + "exp_probs_b.bias", {E});
                fp32(p + "exp_probs_b.bias");
            }
            const std::string suffix = layer.ffn == strata::core::FfnKind::Moe ? "_shexp" : "";
            const uint64_t shared = layer.ffn == strata::core::FfnKind::Moe ? m.shared_intermediate : ff;
            artifact.shape(p + "ffn_gate" + suffix + ".weight", {H, shared});
            artifact.shape(p + "ffn_up" + suffix + ".weight", {H, shared});
            artifact.shape(p + "ffn_down" + suffix + ".weight", {shared, H});
            if (layer.mixer == strata::core::MixerKind::Kda) {
                for (const char *part : {"q", "k", "v"}) {
                    artifact.shape(p + "attn_" + part + ".weight", {H, N});
                    artifact.shape(p + "ssm_conv1d_" + part + ".weight", {(uint64_t)m.conv_kernel, 1, N});
                    fp32(p + "ssm_conv1d_" + part + ".weight");
                }
                artifact.shape(p + "attn_output.weight", {N, H});
                artifact.shape(p + "ssm_dt.bias", {N});
                artifact.shape(p + "ssm_a", {(uint64_t)m.linear_heads});
                artifact.shape(p + "ssm_beta.weight", {H, (uint64_t)m.linear_heads});
                artifact.shape(p + "ssm_norm.weight", {(uint64_t)m.linear_dim});
                for (const char *name : {"ssm_dt.bias", "ssm_a", "ssm_norm.weight"})
                    fp32(p + name);
                for (const char *part : {"f", "g"}) {
                    artifact.shape(p + "ssm_" + part + "_a.weight", {H, (uint64_t)m.linear_dim});
                    artifact.shape(p + "ssm_" + part + "_b.weight", {(uint64_t)m.linear_dim, N});
                }
            } else {
                const uint64_t q = m.q_rank, v = m.kv_rank, heads = m.attention_heads, ih = m.index_heads,
                               d = m.index_dim;
                artifact.shape(p + "attn_q_a.weight", {H, q});
                artifact.shape(p + "attn_q_a_norm.weight", {q});
                artifact.shape(p + "attn_q_b.weight", {q, heads * 256});
                artifact.shape(p + "attn_kv_a_mqa.weight", {H, v});
                artifact.shape(p + "attn_kv_a_norm.weight", {v});
                artifact.shape(p + "attn_k_b.weight", {256, v, heads});
                artifact.shape(p + "attn_v_b.weight", {v, 256, heads});
                artifact.shape(p + "attn_output.weight", {heads * 256, H});
                artifact.shape(p + "indexer.attn_q_b.weight", {q, ih * d});
                artifact.shape(p + "indexer.attn_k.weight", {H, d});
                artifact.shape(p + "indexer.proj.weight", {H, ih});
                artifact.shape(p + "indexer.k_norm.weight", {d});
                artifact.shape(p + "indexer.k_norm.bias", {d});
                artifact.shape(p + "indexer_compressor_gate.weight", {H, d});
                artifact.shape(p + "indexer_compressor_ape.weight", {d, (uint64_t)m.index_pool});
                for (const char *name :
                     {"attn_q_a_norm.weight", "attn_kv_a_norm.weight", "indexer.k_norm.weight",
                      "indexer.k_norm.bias", "indexer_compressor_ape.weight"})
                    fp32(p + name);
            }
        }
    }
    std::vector<float> batch(const std::vector<int> &tokens, bool prefill = true, bool all_logits = false) {
        fast = gpu && prefill;
        reset_phase();
        const int nt = tokens.size();
        if (nt < 1 || nt > (fast ? gpu->chunk : cpu::MAXT) || position + nt > (int)capacity)
            throw std::out_of_range("GLM: invalid batch size or context capacity");
        for (int token : tokens)
            if (token < 0 || token >= m.vocab)
                throw std::out_of_range("GLM: token or context out of range");
        batch_tokens = nt;
        const auto &embedding = artifact.at("token_embd.weight");
        const auto *traits = ggml_get_type_traits((ggml_type)embedding.tensor->type);
        if (embedding.tensor->type != GGML_TYPE_F32 && (!traits || !traits->to_float))
            throw std::runtime_error("GLM: unsupported embedding quantization");
        std::vector<float> emb(m.hidden * 4 * nt);
        for (int t = 0; t < nt; ++t) {
            auto *row = emb.data() + t * m.hidden * 4;
            const auto *data = embedding.data() + (size_t)tokens[t] * (embedding.bytes / m.vocab);
            if (embedding.tensor->type == GGML_TYPE_F32)
                std::memcpy(row, data, m.hidden * sizeof(float));
            else
                traits->to_float(data, row, m.hidden);
            for (int i = 1; i < 4; ++i)
                std::copy_n(row, m.hidden, row + i * m.hidden);
        }
        auto &r = buf("streams", m.hidden * 4 * nt);
        r.put(emb.data(), emb.size() * 4);
        auto &collapsed = buf("collapsed", m.hidden * nt);
        auto &x = buf("x", m.hidden * nt);
        auto &y = buf("y", m.hidden * nt);
        auto &c = buf("hc_coeff", 24 * nt);
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto &layer = m.layers[l];
            check_stop();
            auto layer_start = std::chrono::steady_clock::now();
            reset_phase();
            hc_read(p, "attn", r.f(), collapsed.f(), c.f());
            norm(p + "attn_norm.weight", collapsed.f(), x.f(), m.hidden);
            reset_phase();
            if (layer.mixer == strata::core::MixerKind::Kda)
                kda(p, l, x.f(), y.f());
            else
                mla(p, l, x.f(), y.f());
            if (fast)
                k::glm_mhc_write_batch(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
            else
                for (int t = 0; t < nt; ++t)
                    k::glm_mhc_write(r.f(t * 4 * m.hidden), c.f(t * 24), y.f(t * m.hidden),
                                     r.f(t * 4 * m.hidden), m.hidden, stream);
            if (profile)
                check(cudaStreamSynchronize(stream));
            auto mixer_end = std::chrono::steady_clock::now();
            reset_phase();
            hc_read(p, "ffn", r.f(), collapsed.f(), c.f());
            norm(p + "ffn_norm.weight", collapsed.f(), x.f(), m.hidden);
            reset_phase();
            if (layer.ffn == strata::core::FfnKind::Dense)
                ffn(p, x.f(), y.f(), "", layer.intermediate, layer.swiglu_limit);
            else
                moe(p, l, x.f(), y.f(), layer);
            if (fast)
                k::glm_mhc_write_batch(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
            else
                for (int t = 0; t < nt; ++t)
                    k::glm_mhc_write(r.f(t * 4 * m.hidden), c.f(t * 24), y.f(t * m.hidden),
                                     r.f(t * 4 * m.hidden), m.hidden, stream);
            if (profile) {
                check(cudaStreamSynchronize(stream));
                auto end = std::chrono::steady_clock::now();
                std::cerr << "LAYER " << l << " mixer_ms="
                          << std::chrono::duration<double, std::milli>(mixer_end - layer_start).count()
                          << " ffn_ms=" << std::chrono::duration<double, std::milli>(end - mixer_end).count()
                          << '\n';
            }
        }
        reset_phase();
        for (int t = 0; t < nt; ++t)
            k::glm_hyper_head(r.f(t * 4 * m.hidden), collapsed.f(t * m.hidden), m.hidden, stream);
        norm("output_norm.weight", collapsed.f(), x.f(), m.hidden);
        if (capture_hidden) {
            target_hidden.resize((position + nt) * m.hidden);
            check(cudaMemcpy(target_hidden.data() + position * m.hidden, x.p, (size_t)nt * m.hidden * 4,
                             cudaMemcpyDeviceToHost));
        }
        const int head_tokens = all_logits ? nt : 1;
        auto &logits = buf("output", (size_t)m.vocab * head_tokens);
        mat("output.weight", x.f(all_logits ? 0 : (nt - 1) * m.hidden), logits.f(), false, head_tokens);
        check(cudaStreamSynchronize(stream));
        position += nt;
        batch_tokens = 1;
        auto result = logits.floats((size_t)m.vocab * head_tokens);
        for (float v : result)
            if (!std::isfinite(v))
                throw std::runtime_error("GLM: non-finite logits");
        return result;
    }
    std::vector<float> step(int token) { return batch({token}, false); }
    void enable_verify_history(int slots) {
        if (!gpu || verify_history) return;
        size_t bytes = 0;
        history_offsets.clear();
        for (size_t l = 0; l < m.layers.size(); ++l) {
            auto &s = states[l];
            for (auto *b : {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(), s.conv_v.get(), s.keys.get(), s.gates.get()})
                if (b) { history_offsets.emplace(b, bytes); bytes += b->bytes; }
        }
        slots = std::min(slots, cpu::MAXT - 1);
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t reserve = size_t(2) * 1024 * MiB + 64 * MiB;
        const size_t physical = free_bytes > reserve ? free_bytes - reserve : 0;
        const size_t owned = Device::live <= Device::limit ? Device::limit - Device::live : 0;
        slots = std::min<size_t>(std::max(0, slots), std::min(physical, owned) / bytes);
        if (!slots) { history_offsets.clear(); return; }
        verify_history = std::make_unique<Device>(bytes * slots);
        history_stride = bytes; history_slots = slots;
        std::cerr << "VERIFY_HISTORY slots=" << slots << " MiB=" << bytes * slots / double(MiB) << '\n';
    }
    void clear_verify_history() {
        check(cudaStreamSynchronize(stream));
        verify_history.reset(); history_offsets.clear(); history_slots = history_valid = 0;
    }
    int verify_history_slots() const { return history_slots; }
    bool restore_verified(int count) {
        if (!verify_history || count < 1 || count > history_valid) return false;
        for (auto [buffer, offset] : history_offsets)
            check(cudaMemcpyAsync(buffer->p, (char *)verify_history->p + (count - 1) * history_stride + offset,
                                  buffer->bytes, cudaMemcpyDeviceToDevice, stream));
        position = history_position + count;
        return true;
    }
    std::vector<float> verify(const std::vector<int> &tokens) {
        history_position = position;
        history_valid = 0;
        capturing_history = bool(verify_history);
        try {
            auto result = batch(tokens, false, true);
            history_valid = std::min<size_t>(history_slots, tokens.size());
            capturing_history = false;
            return result;
        } catch (...) { capturing_history = false; throw; }
    }
    int vocabulary() const { return m.vocab; }
    void prepare_cpu(size_t bytes) {
        if (!bytes || !prepared_experts.empty()) return;
        if (gpu_decode_experts) throw std::invalid_argument("CPU packing requires CPU target experts");
        for (const auto &cache : expert_cache)
            if (cache) throw std::invalid_argument("CPU packing requires expert-cache-mib=0");
        size_t required = 0;
        for (size_t l = 3; l < m.layers.size(); ++l) {
            const auto p = "blk." + std::to_string(l) + ".";
            for (const auto *part : {"gate", "up", "down"}) {
                const auto &tensor = artifact.at(p + "ffn_" + part + "_exps.weight");
                const auto &shape = tensor.tensor->shape;
                required += cpu::iq256_prepared_row_bytes(tensor.tensor->type, shape[0]) * shape[1] * shape[2];
            }
        }
        if (required > bytes) throw std::runtime_error("CPU lossless packing exceeds host budget: needs " + std::to_string((required + MiB - 1) / MiB) + " MiB");
#ifdef __linux__
        std::ifstream host_info("/proc/meminfo");
        std::string host_key, host_rest; size_t host_total = 0, host_value;
        while (host_info >> host_key >> host_value) {
            std::getline(host_info, host_rest);
            if (host_key == "MemTotal:") { host_total = host_value * 1024; break; }
        }
        // Converted anonymous weights coexist with streamed file pages and the desktop.
        // MemAvailable alone does not protect against sustained reclaim and oomd.
        const size_t host_reserve = size_t(64) * 1024 * MiB;
        if (!host_total || host_total < host_reserve || required > host_total - host_reserve)
            throw std::runtime_error("CPU packing requires 64 GiB of host capacity beyond packed weights");
#endif
        const auto start = std::chrono::steady_clock::now();
        try {
        for (size_t l = 3; l < m.layers.size(); ++l) {
            check_stop();
            const auto p = "blk." + std::to_string(l) + ".";
            const auto &G = artifact.at(p + "ffn_gate_exps.weight"), &D = artifact.at(p + "ffn_down_exps.weight");
            auto &entry = prepared_experts[l]; auto &f = entry.format;
            std::string error;
            if (!cpu::native_fmt(G.tensor->type, D.tensor->type, m.hidden, m.layers[l].intermediate, f, error)) throw std::runtime_error(error);
            if (f.gu_type != 21 && f.gu_type != 22) throw std::invalid_argument("unsupported lossless gate format");
            f.lossless = true; f.swiglu_limit = m.layers[l].swiglu_limit;
            f.gu_row = cpu::iq256_prepared_row_bytes(f.gu_type, f.n_embd);
            const size_t down_row = cpu::iq256_prepared_row_bytes(f.d_type, f.n_ff);
            if (down_row) f.d_row = down_row;
            f.up_off = f.gu_row * f.n_ff; f.down_off = 2 * f.up_off;
            f.bytes = f.down_off + f.d_row * f.n_embd;
            for (const auto *part : {"gate", "up", "down"}) {
                const auto &tensor = artifact.at(p + "ffn_" + part + "_exps.weight");
                const auto &shape = tensor.tensor->shape;
                const size_t row = cpu::iq256_prepared_row_bytes(tensor.tensor->type, shape[0]);
                if (!row) continue;
                const size_t expert = row * shape[1], allocation = expert * shape[2];
#ifdef __linux__
                std::ifstream info("/proc/meminfo"); std::string key, rest; size_t available = 0, value;
                while (info >> key >> value) { std::getline(info, rest); if (key == "MemAvailable:") { available = value * 1024; break; } }
                std::ifstream pressure("/proc/pressure/memory");
                std::string pressure_kind, pressure_avg;
                while (pressure >> pressure_kind >> pressure_avg) {
                    std::getline(pressure, rest);
                    if (pressure_avg.starts_with("avg10=") && std::stod(pressure_avg.substr(6)) > 5.0)
                        throw std::runtime_error("CPU packing stopped due to host memory pressure");
                }
                if (available < allocation + 32 * 1024 * MiB) throw std::runtime_error("CPU packing requires 32 GiB of remaining host memory");
#endif
                auto &dst = std::string(part) == "gate" ? entry.gate : std::string(part) == "up" ? entry.up : entry.down;
                dst.reset(new uint8_t[allocation]);
                std::atomic<int> next{0}; std::vector<std::thread> workers;
                const size_t nworkers = std::min<size_t>(16, std::max<size_t>(1, prepack_cpus.size()));
                for (size_t w = 0; w < nworkers; ++w) workers.emplace_back([&, w] {
                    if (!prepack_cpus.empty()) cpu::pin_current_thread(prepack_cpus[w]);
                    for (;;) {
                        const int e = next.fetch_add(1); if (e >= (int)shape[2]) break;
                        cpu::iq256_prepare_rows(tensor.tensor->type, tensor.data() + e * (tensor.bytes / shape[2]),
                                                dst.get() + e * expert, shape[0], shape[1]);
                    }
                });
                for (auto &worker : workers) worker.join();
            }
            std::cerr << "CPU_PACK layer=" << l << " ready\n";
        }
        } catch (...) {
            prepared_experts.clear();
            throw;
        }
        std::cerr << "CPU_PACK bytes=" << required << " ms=" << std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() << '\n';
    }
    void enable_mtp_capture() { capture_hidden = true; }
    void set_mtp_cpu_experts(bool enabled) { mtp_cpu_experts = enabled; }
    int token_position() const { return position; }
    std::vector<float> mtp_step(int token, const float *previous, bool head = true) {
        check_stop();
        if (!mtp_ready || token < 0 || token >= m.vocab || mtp_position >= (int)capacity)
            throw std::out_of_range("GLM: invalid MTP token or position");
        struct Restore {
            int &pos, &nt; bool &fast;
            int old_pos, old_nt; bool old_fast;
            ~Restore() { pos = old_pos; nt = old_nt; fast = old_fast; }
        } restore{position, batch_tokens, fast, position, batch_tokens, fast};
        position = mtp_position; batch_tokens = 1; fast = false;
        reset_phase();
        const int H = m.hidden;
        const std::string p = "blk.45.";
        const auto &embedding = artifact.at("token_embd.weight");
        const size_t row_bytes = embedding.bytes / m.vocab;
        std::vector<float> row(H);
        ggml_get_type_traits((ggml_type)embedding.tensor->type)->to_float(
            embedding.data() + token * row_bytes, row.data(), H);
        auto &emb = buf("mtp_embedding", H), &prev = buf("mtp_previous", H);
        auto &joined = buf("mtp_joined", 2 * H), &cur = buf("mtp_residual", H);
        auto &x = buf("mtp_x", H), &y = buf("mtp_y", H), &hidden = buf("mtp_hidden", H);
        emb.put(row.data(), H * 4); prev.put(previous, H * 4);
        norm(p + "nextn.enorm.weight", emb.f(), joined.f(), H);
        norm(p + "nextn.hnorm.weight", prev.f(), joined.f(H), H);
        mat(p + "nextn.eh_proj.weight", joined.f(), cur.f());
        norm(p + "attn_norm.weight", cur.f(), x.f(), H);
        mla_gpu(p, 45, x.f(), y.f());
        const float one = 1;
        check(cublasSaxpy(blas, H, &one, y.f(), 1, cur.f(), 1));
        norm(p + "ffn_norm.weight", cur.f(), x.f(), H);
        moe_mtp(p, x.f(), y.f(), m.draft_layers.at(0));
        check(cublasSaxpy(blas, H, &one, y.f(), 1, cur.f(), 1));
        norm(p + "nextn.shared_head_norm.weight", cur.f(), hidden.f(), H);
        ++mtp_position;
        if (!head) { check(cudaStreamSynchronize(stream)); return {}; }
        mtp_hidden = hidden.floats(H);
        auto &output = buf("mtp_logits", m.vocab);
        mat("output.weight", hidden.f(), output.f());
        return output.floats(m.vocab);
    }
    std::vector<float> mtp_propose(int token, bool first) {
        const float *previous = first ? target_hidden.data() + (position - 1) * m.hidden : mtp_hidden.data();
        return mtp_step(token, previous);
    }
    void sync_mtp(const std::vector<int> &tokens, int start) {
        std::vector<float> zero(m.hidden);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const int prev = start + i - 1;
            const float *hidden = prev < 0 ? zero.data() : target_hidden.data() + prev * m.hidden;
            mtp_step(tokens[i], hidden, false);
        }
    }
    void set_decode_cache_budget(size_t bytes) {
        decode_cache_budget = bytes;
        if (bytes) prefill_routes.assign(m.layers.size(), std::vector<uint64_t>(m.experts));
    }
    void set_decode_cache_slots(int slots) { decode_cache_slots_limit = slots; }
    void prepare_decode_cache() {
        if (!decode_cache_budget) return;
        if (!prepared_experts.empty() || mtp_experts || gpu_decode_experts)
            throw std::invalid_argument("GLM: prefix-trained decode cache requires unpacked CPU target experts");
        compact_decode();
        struct Candidate { uint64_t count; int layer, expert; size_t bytes; };
        std::vector<Candidate> candidates;
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            const auto p = "blk." + std::to_string(l) + ".";
            const size_t bytes = (artifact.at(p + "ffn_gate_exps.weight").bytes +
                artifact.at(p + "ffn_up_exps.weight").bytes +
                artifact.at(p + "ffn_down_exps.weight").bytes) / m.experts;
            for (int e = 0; e < m.experts; ++e)
                if (prefill_routes[l][e]) candidates.push_back({prefill_routes[l][e], (int)l, e, bytes});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            if (a.count != b.count) return a.count > b.count;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });
        size_t used = 0, slots = 0;
        uint64_t fingerprint = 14695981039346656037ULL;
        constexpr size_t scratch_reserve = 768 * MiB;
        for (const auto &c : candidates) {
            if (decode_cache_slots_limit && slots >= (size_t)decode_cache_slots_limit) break;
            if (c.bytes > decode_cache_budget - used) continue;
            if (Device::live > Device::limit || Device::limit - Device::live < scratch_reserve ||
                c.bytes > Device::limit - Device::live - scratch_reserve) break;
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total));
            if (free < 2 * 1024 * MiB + scratch_reserve ||
                c.bytes > free - 2 * 1024 * MiB - scratch_reserve) break;
            auto entry = std::make_unique<Device>(c.bytes);
            const auto p = "blk." + std::to_string(c.layer) + ".";
            size_t offset = 0;
            for (const char *role : {"gate", "up", "down"}) {
                const auto &t = artifact.at(p + "ffn_" + role + "_exps.weight");
                const size_t bytes = t.bytes / m.experts;
                if (offset > entry->bytes || bytes > entry->bytes - offset)
                    throw std::runtime_error("GLM: decode cache expert geometry mismatch");
                check(cudaMemcpy((char *)entry->p + offset, t.data() + c.expert * bytes,
                                 bytes, cudaMemcpyHostToDevice));
                offset += bytes;
            }
            decode_resident[c.layer].emplace(c.expert, std::move(entry));
            fingerprint ^= (uint64_t)c.layer * m.experts + c.expert;
            fingerprint *= 1099511628211ULL;
            used += c.bytes; ++slots;
        }
        if (decode_cache_slots_limit && slots != (size_t)decode_cache_slots_limit)
            throw std::runtime_error("GLM: GPU headroom cannot reproduce requested decode cache slots");
        std::cerr << "DECODE_CACHE prefix_trained=1 slots=" << slots
                  << " MiB=" << used / double(MiB) << " fingerprint=" << fingerprint << '\n';
    }
    void compact_decode() {
        if (!gpu || prefill_width_saved || gpu_decode_experts) return;
        check(cudaStreamSynchronize(stream));
        if (gpu->copy) check(cudaStreamSynchronize(gpu->copy));
        prefill_width_saved = gpu->chunk;
        phase.clear(); scratch.clear(); gpu.reset();
        gpu = std::make_unique<GpuPrefill>(cpu::MAXT, stream, true);
        fast = false;
    }
    void prepare_mtp(const std::vector<int> &prompt) {
        if (!gpu || gpu_decode_experts || m.draft_layers.size() != 1 ||
            target_hidden.size() < prompt.size() * m.hidden)
            throw std::invalid_argument("GLM: MTP requires captured GPU prefill and CPU target experts");
        compact_decode();
        const auto &layer = m.draft_layers.at(0);
        if (layer.mixer != strata::core::MixerKind::SparseMla)
            throw std::invalid_argument("GLM: MTP requires a NoPE MLA draft block");
        for (const auto &[name, tensor] : artifact.tensors())
            if (name.starts_with("blk.45.") && !name.ends_with("_exps.weight"))
                weight(name, name.ends_with("attn_k_b.weight") || name.ends_with("attn_v_b.weight"));
        const auto &G = artifact.at("blk.45.ffn_gate_exps.weight");
        const auto &U = artifact.at("blk.45.ffn_up_exps.weight");
        const auto &D = artifact.at("blk.45.ffn_down_exps.weight");
        mtp_layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate);
        mtp_layout.swiglu_limit = layer.swiglu_limit;
        mtp_expert_bytes = (G.bytes + U.bytes + D.bytes) / m.experts;
        if (!mtp_cpu_experts) {
            mtp_experts = std::make_unique<Device>(mtp_expert_bytes * m.experts);
            const size_t gu = G.bytes / m.experts, down = D.bytes / m.experts;
            std::vector<uint8_t> packed(mtp_expert_bytes);
            for (int e = 0; e < m.experts; ++e) {
                check_stop();
                std::memcpy(packed.data(), G.data() + e * gu, gu);
                std::memcpy(packed.data() + gu, U.data() + e * gu, gu);
                std::memcpy(packed.data() + 2 * gu, D.data() + e * down, down);
                check(cudaMemcpy((char *)mtp_experts->p + e * mtp_expert_bytes, packed.data(), packed.size(),
                                 cudaMemcpyHostToDevice));
            }
        }
        states.emplace_back();
        auto &state = states.back();
        state.cache = std::make_unique<Device>(capacity * m.kv_rank * 4, true);
        state.keys = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
        state.gates = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
        state.pooled = std::make_unique<Device>(std::max<size_t>(1, capacity / m.index_pool) * m.index_dim * 4, true);
        mtp_position = 0;
        mtp_ready = true;
        sync_mtp(prompt, 0);
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        std::cerr << "MTP free_MiB=" << free_bytes / double(MiB) << '\n';
        std::cerr << "MTP expert_backend=" << (mtp_cpu_experts ? "cpu" : "gpu") << '\n';
        std::cerr << "MTP resident_MiB=" << (mtp_experts ? mtp_experts->bytes : 0) / double(MiB)
                  << " allocated_MiB=" << Device::live / double(MiB) << " primed_tokens=" << prompt.size() << '\n';
    }
    struct Checkpoint {
        int position = 0, mtp_position = 0;
        size_t draft_begin = 0;
        std::unique_ptr<Device> storage;
        std::vector<std::pair<Device *, size_t>> buffers;
    };
    Checkpoint checkpoint() {
        Checkpoint cp;
        size_t bytes = 0;
        cp.draft_begin = SIZE_MAX;
        for (size_t l = 0; l < states.size(); ++l) {
            auto &state = states[l];
            if (l == m.layers.size()) cp.draft_begin = cp.buffers.size();
            for (Device *buffer : {state.recurrent.get(), state.conv_q.get(), state.conv_k.get(),
                                   state.conv_v.get(), state.keys.get(), state.gates.get()})
                if (buffer) {
                    cp.buffers.emplace_back(buffer, bytes);
                    bytes += buffer->bytes;
                }
        }
        cp.storage = std::make_unique<Device>(bytes);
        std::cerr << "CHECKPOINT MiB=" << bytes / double(MiB) << '\n';
        return cp;
    }
    void save(Checkpoint &cp) {
        cp.position = position;
        cp.mtp_position = mtp_position;
        for (auto [buffer, offset] : cp.buffers)
            check(cudaMemcpyAsync((char *)cp.storage->p + offset, buffer->p, buffer->bytes,
                                  cudaMemcpyDeviceToDevice, stream));
    }
    void restore(const Checkpoint &cp) {
        for (auto [buffer, offset] : cp.buffers)
            check(cudaMemcpyAsync(buffer->p, (char *)cp.storage->p + offset, buffer->bytes,
                                  cudaMemcpyDeviceToDevice, stream));
        // Latents and pooled keys are append-only; position masks discarded future entries.
        position = cp.position;
        mtp_position = cp.mtp_position;
    }
    void restore_mtp(const Checkpoint &cp) {
        for (size_t i = cp.draft_begin; i < cp.buffers.size(); ++i) {
            auto [buffer, offset] = cp.buffers[i];
            check(cudaMemcpyAsync(buffer->p, (char *)cp.storage->p + offset, buffer->bytes,
                                  cudaMemcpyDeviceToDevice, stream));
        }
        mtp_position = cp.mtp_position;
    }

    int enable_gpu(int requested, size_t total_budget, bool reserve_checkpoint = false) {
        if (gpu)
            return gpu->chunk;
        if (m.layers.size() != 45 || m.hidden != 4096 || m.experts != 288 || m.top_k != 8 ||
            m.linear_dim != 128 || m.linear_heads != 64 || m.attention_heads != 64 || m.kv_rank != 512 ||
            m.q_rank != 1536 || m.index_pool != 4 || m.index_top_k != 2048 || m.index_heads != 32 ||
            m.index_dim != 128)
            throw std::invalid_argument("GLM: GPU prefill requires the inspected GLM-5.3-Flash geometry");
        for (size_t l = 3; l < m.layers.size(); ++l) {
            auto p = "blk." + std::to_string(l) + ".";
            if (!mmq::supported(artifact.at(p + "ffn_gate_exps.weight").tensor->type) ||
                !mmq::supported(artifact.at(p + "ffn_down_exps.weight").tensor->type))
                throw std::invalid_argument("GLM: unsupported prefill expert quantization");
        }
        if (total_budget <= 1024 * MiB || total_budget > 12288 * MiB)
            throw std::invalid_argument("GLM: GPU budget must be at most 12288 MiB");
        for (const auto &cache : expert_cache)
            if (cache)
                throw std::invalid_argument("GLM: GPU prefill currently requires expert-cache-mib=0");
        size_t fixed = 0;
        auto floating = [](const std::string &n) {
            return n.ends_with("hc_attn_fn.weight") || n.ends_with("hc_ffn_fn.weight") ||
                   n.ends_with("attn_k_b.weight") || n.ends_with("attn_v_b.weight");
        };
        auto eligible = [](const std::string &n) {
            return n != "token_embd.weight" && !n.starts_with("blk.45.") && !n.ends_with("_exps.weight");
        };
        for (const auto &[n, t] : artifact.tensors())
            if (eligible(n)) {
                size_t bytes = t.bytes;
                if (floating(n)) {
                    bytes = 4;
                    for (auto d : t.tensor->shape)
                        bytes *= d;
                }
                fixed += bytes;
            }
        size_t free, total;
        check(cudaMemGetInfo(&free, &total));
        Device::limit =
            std::min(total_budget - 1024 * MiB, Device::live + (free > 1024 * MiB ? free - 1024 * MiB : 0));
        auto needed = [&](int b) {
            return Device::live + fixed + GpuPrefill::arena_bytes(b, capacity) +
                   (128 + 32 + 128 + 416) * MiB + (size_t)b * m.hidden * 7 * 4 + MiB +
                   (reserve_checkpoint ? 152 * MiB : 0);
        };
        int width = requested ? requested : 4096;
        if (!requested && needed(width) > Device::limit)
            width = 2048;
        if (width < 1 || width > 4096 || needed(width) > Device::limit)
            throw std::runtime_error(
                "GLM: prefill cannot fit GPU budget; reduce context or use legacy prefill");
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        budget = SIZE_MAX;
        for (const auto &[n, t] : artifact.tensors())
            if (eligible(n))
                weight(n, floating(n));
        gpu = std::make_unique<GpuPrefill>(width, stream, false, capacity);
        // Allocate persistent residual and output buffers before carving phase scratch.
        for (auto name : {"streams", "collapsed", "x", "y", "hc_coeff", "output"}) {
            size_t count = std::string(name) == "streams"    ? width * m.hidden * 4
                           : std::string(name) == "hc_coeff" ? width * 24
                           : std::string(name) == "output"   ? m.vocab
                                                             : width * m.hidden;
            buf(name, count);
        }
        std::cerr << "GPU allocated_MiB=" << Device::live / (double)MiB
                  << " reserved_runtime_MiB=1024 budget_MiB=" << total_budget / MiB << '\n';
        return width;
    }
    void warm_weights() const {
        volatile uint8_t checksum = 0;
        for (const auto &[name, t] : artifact.tensors())
            if (!name.starts_with("blk.45."))
                for (size_t i = 0; i < t.bytes; i += 4096)
                    checksum = checksum ^ t.data()[i];
        std::cerr << "main weights prefaulted checksum=" << (int)checksum << '\n';
    }
    void set_gpu_decode_experts(bool value) {
        if (value && !gpu)
            throw std::invalid_argument("GPU decode experts require GPU prefill allocation");
        gpu_decode_experts = value;
    }
    void reset_decode_stats() {
        if (gpu) {
            gpu->transferred = 0;
            gpu->groups = 0;
            gpu->stage_ms = 0;
        }
        pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = 0;
        pool.ms_native_local_prepare = 0;
        pool.native_local_queries = 0;
        pool.multi_bytes = 0;
    }
    void set_profile(bool enabled) { profile = enabled; }
    void set_routing_trace(const std::string &path) {
        if (path.empty()) return;
        routing_trace.open(path);
        if (!routing_trace) throw std::runtime_error("GLM: cannot open routing trace");
    }
    void set_cancel(std::function<bool()> fn) { cancelled = std::move(fn); }
    void report_gpu() const {
        std::cerr << "CPU_EXPERT gu_ms=" << pool.ms_multi_gu << " quant_ms=" << pool.ms_multi_q
                  << " down_ms=" << pool.ms_multi_down << " bytes=" << pool.multi_bytes << '\n';
        if (pool.native_local_queries)
            std::cerr << "NATIVE_NUMA_TIMING prepare_ms=" << pool.ms_native_local_prepare
                      << " queries=" << pool.native_local_queries << '\n';
        if (gpu)
            std::cerr << "GPU peak_allocated_MiB=" << Device::peak / (double)MiB
                      << " native_expert_bytes=" << gpu->transferred << " expert_groups=" << gpu->groups
                      << " staging_ms=" << gpu->stage_ms << '\n';
    }
    void freeze_cache() { cache_frozen = true; }
    void reset() {
        check(cudaStreamSynchronize(stream));
        mtp_ready = false;
        for (auto &cache : decode_resident) cache.clear();
        for (auto &counts : prefill_routes) std::fill(counts.begin(), counts.end(), 0);
        clear_verify_history();
        if (gpu) {
            if (gpu->copy) check(cudaStreamSynchronize(gpu->copy));
            gpu->transferred = gpu->groups = 0;
            gpu->stage_ms = 0;
        }
        if (prefill_width_saved) {
            phase.clear(); scratch.clear(); gpu.reset(); mtp_experts.reset();
            for (auto it = weights.begin(); it != weights.end();) {
                if (it->first.starts_with("blk.45.")) {
                    resident -= it->second.data->bytes;
                    lru.erase(it->second.order);
                    it = weights.erase(it);
                } else ++it;
            }
            states.resize(m.layers.size());
            gpu = std::make_unique<GpuPrefill>(prefill_width_saved, stream, false, capacity);
            prefill_width_saved = 0;
        }
        reset_phase();
        for (auto &s : states)
            for (auto *b : {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(), s.conv_v.get(), s.cache.get(),
                            s.keys.get(), s.gates.get(), s.pooled.get()})
                if (b)
                    check(cudaMemsetAsync(b->p, 0, b->bytes, stream));
        position = 0;
        mtp_position = 0;
        target_hidden.clear();
        batch_tokens = 1;
        cache_frozen = false;
        cache_hits = cache_entries = 0;
    }
    void report_cache() const {
        bool enabled = false;
        for (const auto &cache : decode_resident) enabled |= !cache.empty();
        for (const auto &c : expert_cache)
            enabled |= bool(c);
        if (enabled)
            std::cerr << "resident expert entries=" << cache_hits << "/" << cache_entries << '\n';
    }
};
} // namespace

static std::vector<int> token_ids(const std::string &text) {
    std::vector<int> ids;
    std::istringstream input(text);
    std::string part;
    while (std::getline(input, part, ',')) {
        size_t consumed = 0;
        const int id = std::stoi(part, &consumed);
        if (consumed != part.size())
            throw std::invalid_argument("invalid token ID");
        ids.push_back(id);
    }
    return ids;
}

// Existing Python API engine protocol. Every request starts with empty model state;
// mmap handles, dense weights and optional resident experts survive between requests.
static int prefill_width(const std::string &value) {
    if (value == "auto")
        return 0;
    size_t used = 0;
    int n = std::stoi(value, &used);
    if (used != value.size() || n < 1 || n > 4096)
        throw std::invalid_argument("GLM: invalid prefill batch");
    return n;
}
static int greedy(const std::vector<float> &logits, size_t offset = 0, size_t count = 0) {
    if (!count)
        count = logits.size();
    return std::max_element(logits.begin() + offset, logits.begin() + offset + count) - logits.begin() -
           offset;
}
struct LookupStats {
    int generated = 0, rounds = 0, proposed = 0, accepted = 0, replayed = 0;
    double milliseconds = 0;
    bool stopped = false;
};
static LookupStats generate_lookup(Decoder &decoder, std::vector<float> &logits,
                                   const std::vector<int> &prompt, int max_new, const std::vector<int> &stops,
                                   int depth, const std::function<void(int)> &emit) {
    strata::spec::SuffixDrafter lookup(3, 32, prompt.size() + max_new);
    lookup.append(prompt.data(), prompt.size());
    decoder.compact_decode();
    auto checkpoint = decoder.checkpoint();
    decoder.enable_verify_history(depth);
    LookupStats stats;
    const auto start = std::chrono::steady_clock::now();
    auto output = [&](int token) {
        emit(token);
        lookup.append(token);
        ++stats.generated;
        stats.stopped = std::find(stops.begin(), stops.end(), token) != stops.end();
    };
    while (stats.generated < max_new && !stats.stopped) {
        const int anchor = greedy(logits);
        output(anchor);
        if (stats.stopped || stats.generated == max_new)
            break;
        int32_t drafts[7];
        const int k = lookup.propose(std::min(depth, max_new - stats.generated - 1), drafts);
        if (!k) {
            logits = decoder.step(anchor);
            continue;
        }
        std::vector<int> window{anchor};
        window.insert(window.end(), drafts, drafts + k);
        decoder.save(checkpoint);
        auto all = decoder.verify(window);
        ++stats.rounds;
        stats.proposed += k;
        int accepted = 0;
        while (accepted < k && drafts[accepted] == greedy(all, (size_t)accepted * decoder.vocabulary(),
                                                          decoder.vocabulary())) {
            output(drafts[accepted]);
            ++accepted;
            if (stats.stopped)
                break;
        }
        stats.accepted += accepted;
        if (accepted != k || stats.stopped) {
            window.resize(accepted + 1 - int(stats.stopped));
            if (decoder.restore_verified(window.size())) {
                const size_t offset = (window.size() - 1) * decoder.vocabulary();
                logits.assign(all.begin() + offset, all.begin() + offset + decoder.vocabulary());
            } else {
                decoder.restore(checkpoint);
                if (!window.empty()) {
                    logits = decoder.batch(window, false);
                    stats.replayed += window.size();
                }
            }
        } else {
            logits.assign(all.end() - decoder.vocabulary(), all.end());
        }
    }
    stats.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "SPECULATIVE source=lookup generated=" << stats.generated << " rounds=" << stats.rounds
              << " proposed=" << stats.proposed << " accepted=" << stats.accepted
              << " replayed=" << stats.replayed << " ms=" << stats.milliseconds
              << " tok_s=" << std::max(0, stats.generated - 1) * 1000 / stats.milliseconds << '\n';
    return stats;
}
static LookupStats generate_mtp(Decoder &decoder, std::vector<float> &logits, int max_new,
                               const std::vector<int> &stops, int depth,
                               const std::function<void(int)> &emit) {
    auto checkpoint = decoder.checkpoint();
    decoder.enable_verify_history(depth);
    // A rejection after accepting n drafts restores n+1 target positions.
    // Limiting proposals to the retained slots keeps every rejection prefix
    // restorable instead of paying for another CPU expert pass.
    const int requested_depth = depth;
    if (decoder.verify_history_slots())
        depth = std::min(depth, decoder.verify_history_slots());
    std::cerr << "MTP_DEPTH requested=" << requested_depth << " effective=" << depth << '\n';
    LookupStats stats;
    double draft_ms = 0, verify_ms = 0, resync_ms = 0;
    auto elapsed_ms = [](auto begin) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    };
    const auto start = std::chrono::steady_clock::now();
    auto output = [&](int token) {
        emit(token); ++stats.generated;
        stats.stopped = std::find(stops.begin(), stops.end(), token) != stops.end();
    };
    while (stats.generated < max_new && !stats.stopped) {
        const int anchor = greedy(logits);
        output(anchor);
        if (stats.stopped || stats.generated == max_new) break;
        const int k = std::min(depth, max_new - stats.generated - 1);
        if (!k) {
            const int pos = decoder.token_position();
            logits = decoder.step(anchor);
            decoder.sync_mtp({anchor}, pos);
            continue;
        }
        decoder.save(checkpoint);
        const int pos = decoder.token_position();
        std::vector<int> window{anchor};
        const auto draft_start = std::chrono::steady_clock::now();
        for (int i = 0; i < k; ++i)
            window.push_back(greedy(decoder.mtp_propose(window.back(), i == 0)));
        draft_ms += elapsed_ms(draft_start);
        const auto verify_start = std::chrono::steady_clock::now();
        auto all = decoder.verify(window);
        verify_ms += elapsed_ms(verify_start);
        ++stats.rounds; stats.proposed += k;
        int accepted = 0;
        while (accepted < k && window[accepted + 1] ==
               greedy(all, (size_t)accepted * decoder.vocabulary(), decoder.vocabulary())) {
            output(window[++accepted]);
            if (stats.stopped) break;
        }
        stats.accepted += accepted;
        if (accepted != k || stats.stopped) {
            window.resize(accepted + 1 - int(stats.stopped));
            if (decoder.restore_verified(window.size())) {
                decoder.restore_mtp(checkpoint);
                const size_t offset = (window.size() - 1) * decoder.vocabulary();
                logits.assign(all.begin() + offset, all.begin() + offset + decoder.vocabulary());
            } else {
                decoder.restore(checkpoint);
                if (!window.empty()) {
                    logits = decoder.batch(window, false);
                    stats.replayed += window.size();
                }
            }
        } else {
            decoder.restore_mtp(checkpoint);
            logits.assign(all.end() - decoder.vocabulary(), all.end());
        }
        const auto resync_start = std::chrono::steady_clock::now();
        decoder.sync_mtp(window, pos);
        resync_ms += elapsed_ms(resync_start);
    }
    stats.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cerr << "SPECULATIVE source=mtp generated=" << stats.generated << " rounds=" << stats.rounds
              << " proposed=" << stats.proposed << " accepted=" << stats.accepted
              << " replayed=" << stats.replayed << " ms=" << stats.milliseconds
              << " tok_s=" << std::max(0, stats.generated - 1) * 1000 / stats.milliseconds << '\n';
    std::cerr << "MTP_TIMING draft_ms=" << draft_ms << " verify_ms=" << verify_ms
              << " resync_ms=" << resync_ms << '\n';
    return stats;
}
static size_t gpu_budget_bytes(const std::string &value) {
    size_t used = 0;
    const auto mib = std::stoull(value, &used);
    if (used != value.size() || mib <= 1024 || mib > 12288)
        throw std::invalid_argument("GPU budget must be greater than 1024 and at most 12288 MiB");
    return mib * MiB;
}
static void preflight_gpu_budget(size_t budget) {
    if (budget <= 1024 * MiB || budget > 12288 * MiB)
        throw std::invalid_argument("GPU budget must be greater than 1024 and at most 12288 MiB");
    Device::limit = budget - 1024 * MiB;
}
static int serve(int argc, char **argv) {
    if (argc < 6)
        throw std::invalid_argument(
            "usage: strata-glm-decode --serve <shard> <context> <dense-MiB> <threads> "
            "[expert-MiB=0] [prefill-batch=8] [eos-id=-1] [gpu-budget-MiB=12288] [lookup-depth=0] "
            "[gpu-decode-experts=0]");
    const int ctx = std::stoi(argv[3]), dense = std::stoi(argv[4]), threads = std::stoi(argv[5]);
    const int experts = argc > 6 ? std::stoi(argv[6]) : 0;
    int width = argc > 7 ? prefill_width(argv[7]) : 8;
    const size_t gpu_budget = argc > 9 ? gpu_budget_bytes(argv[9]) : 12288 * MiB;
    const auto stops = argc > 8 ? token_ids(argv[8]) : std::vector<int>{};
    const int lookup_depth = argc > 10 ? std::stoi(argv[10]) : 0;
    const bool stream_decode = argc > 11 && std::stoi(argv[11]) != 0;
    const bool use_mtp = argc > 12 && std::string(argv[12]) == "mtp";
    const int draft_depth = argc > 13 ? std::stoi(argv[13]) : 3;
    const bool pin_cpu = argc > 14 && std::string(argv[14]) == "auto";
    const size_t cpu_prepack_mib = argc > 15 ? std::stoull(argv[15]) : 0;
    if (cpu_prepack_mib > 131072 || (cpu_prepack_mib && (stream_decode || experts))) throw std::invalid_argument("invalid serve CPU packing settings");
    if (draft_depth < 1 || draft_depth > 7 || (use_mtp && (lookup_depth || stream_decode || experts)))
        throw std::invalid_argument("invalid serve MTP settings");
    if (ctx < 1 || dense < 64 || threads < 1 || experts < 0 || lookup_depth < 0 || lookup_depth > 7)
        throw std::invalid_argument("invalid serve settings");
    if (width == 0 || width > cpu::MAXT || stream_decode || use_mtp)
        preflight_gpu_budget(gpu_budget);
    if ((width == 0 || width > cpu::MAXT || stream_decode || use_mtp) && experts)
        throw std::invalid_argument("GPU prefill/decode requires expert-cache-MiB=0");
    Decoder decoder(argv[2], ctx, (size_t)dense * 1024 * 1024, threads, (size_t)experts * 1024 * 1024, pin_cpu);
    if (width == 0 || width > cpu::MAXT || stream_decode || use_mtp)
        width = decoder.enable_gpu(width, gpu_budget, lookup_depth > 0);
    decoder.set_gpu_decode_experts(stream_decode);
    if (use_mtp) decoder.enable_mtp_capture();
    std::cout << "INFO engine=glm-experimental sampling=greedy\nREADY " << ctx << " stop\n" << std::flush;
    struct Request {
        std::string line;
        std::shared_ptr<std::atomic<bool>> cancel;
    };
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Request> requests;
    std::shared_ptr<std::atomic<bool>> active;
    bool ended = false;
    std::jthread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::lock_guard lock(mutex);
            if (line == "STOP") {
                if (active)
                    *active = true;
                else if (!requests.empty())
                    *requests.front().cancel = true;
                continue;
            }
            const bool quit = line == "QUIT";
            requests.push_back({std::move(line), std::make_shared<std::atomic<bool>>(false)});
            ready.notify_one();
            if (quit)
                break;
        }
        std::lock_guard lock(mutex);
        ended = true;
        ready.notify_one();
    });
    using Clock = std::chrono::steady_clock;
    auto millis = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex);
            ready.wait(lock, [&] { return ended || !requests.empty(); });
            if (requests.empty())
                break;
            request = std::move(requests.front());
            requests.pop_front();
            active = request.cancel;
        }
        if (request.line == "QUIT")
            break;
        try {
            std::istringstream command(request.line);
            std::string op, ids, extra;
            int count;
            if (!(command >> op >> count >> ids) || op != "GEN" || command >> extra)
                throw std::invalid_argument(
                    "expected GEN <max_new> <comma-separated IDs>; only greedy text is supported");
            const auto tokens = token_ids(ids);
            if (tokens.empty() || count < 1 || tokens.size() + (size_t)count > (size_t)ctx)
                throw std::invalid_argument("request exceeds context or has invalid token counts");
            decoder.reset();
            decoder.set_cancel([&request] { return bool(*request.cancel); });
            auto start = Clock::now();
            std::vector<float> logits;
            for (size_t t = 0; t < tokens.size(); t += width) {
                if (*request.cancel)
                    break;
                const size_t end = std::min(tokens.size(), t + width);
                logits = decoder.batch(std::vector<int>(tokens.begin() + t, tokens.begin() + end));
                std::cout << "PP " << end << ' ' << tokens.size() << '\n' << std::flush;
            }
            decoder.freeze_cache();
            if (!*request.cancel) decoder.prepare_cpu(cpu_prepack_mib * MiB);
            if (use_mtp && !*request.cancel) decoder.prepare_mtp(tokens);
            const auto prefilled = Clock::now();
            decoder.reset_decode_stats();
            int generated = 0;
            std::string finish = "length";
            if (use_mtp && !*request.cancel) {
                auto result = generate_mtp(decoder, logits, count, stops, draft_depth, [&](int token) {
                    if (*request.cancel) throw std::runtime_error("cancelled");
                    std::cout << "T " << token << '\n' << std::flush;
                });
                generated = result.generated;
                if (result.stopped) finish = "stop";
            } else if (lookup_depth && !*request.cancel) {
                auto result =
                    generate_lookup(decoder, logits, tokens, count, stops, lookup_depth, [&](int token) {
                        if (*request.cancel)
                            throw std::runtime_error("cancelled");
                        std::cout << "T " << token << '\n' << std::flush;
                    });
                generated = result.generated;
                if (result.stopped)
                    finish = "stop";
            } else
                for (; generated < count; ++generated) {
                    if (*request.cancel) {
                        finish = "stop";
                        break;
                    }
                    const int next = std::max_element(logits.begin(), logits.end()) - logits.begin();
                    std::cout << "T " << next << '\n' << std::flush;
                    if (std::find(stops.begin(), stops.end(), next) != stops.end()) {
                        ++generated;
                        finish = "stop";
                        break;
                    }
                    if (generated + 1 < count)
                        logits = decoder.step(next);
                }
            std::cout << "DONE " << generated << ' ' << tokens.size() << ' ' << millis(start, prefilled)
                      << ' ' << millis(prefilled, Clock::now()) << ' ' << finish << " 0 0 0\n"
                      << std::flush;
        } catch (const std::exception &e) {
            decoder.set_cancel({});
            decoder.reset();
            if (request.cancel->load())
                std::cout << "DONE 0 0 0 0 stop 0 0 0\n" << std::flush;
            else
                std::cout << "ERR " << e.what() << '\n' << std::flush;
        }
        decoder.set_cancel({});
        decoder.report_gpu();
        std::lock_guard lock(mutex);
        active.reset();
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: strata-glm-decode <shard.gguf> <comma-separated token IDs> [steps=1] "
                     "[dense-cache-MiB=4096] [threads=6] [--prefill-batch=1..4096|auto] "
                     "[--gpu-budget-mib=12288] [--check-prefill] "
                     "[--profile-decode] "
                     "[--expert-cache-mib=N] [--routing-trace=path] [--check-replay] [--dump-logits=path] [--stop-ids=IDs]\n";
        return 2;
    }
    try {
        if (std::string(argv[1]) == "--serve")
            return serve(argc, argv);
        std::string ids_text = argv[2];
        if (ids_text.starts_with("@")) {
            std::ifstream f(ids_text.substr(1));
            if (!f)
                throw std::runtime_error("cannot read token file");
            ids_text.assign(std::istreambuf_iterator<char>(f), {});
        }
        auto tokens = token_ids(ids_text);
        const int steps = argc > 3 ? std::stoi(argv[3]) : 1;
        const int mib = argc > 4 ? std::stoi(argv[4]) : 4096;
        const int threads = argc > 5 ? std::stoi(argv[5]) : 6;
        bool replay = false;
        bool check_prefill = false;
        int expert_mib = 0, decode_cache_mib = 0, decode_cache_slots = 0;
        int prefill_batch = 8;
        size_t gpu_budget = 12288 * MiB;
        bool force_gpu = false, warm_weights = false, profile = false;
        bool profile_decode = false;
        int repetitions = 1, context = 0, reference_batch = 1, lookup_depth = 0;
        bool check_verify = false, stream_decode = false, pin_cpu = false, use_mtp = false;
        bool mtp_cpu_experts = false;
        int draft_depth = 3, decode_repetitions = 1;
        size_t cpu_prepack_mib = 0;
        std::string dump, routing_trace_path;
        std::vector<int> stops;
        for (int i = 6; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--check-replay")
                replay = true;
            else if (flag == "--check-prefill")
                check_prefill = true;
            else if (flag.starts_with("--expert-cache-mib="))
                expert_mib = std::stoi(flag.substr(19));
            else if (flag.starts_with("--decode-cache-mib=")) {
                decode_cache_mib = std::stoi(flag.substr(19)); force_gpu = true;
            }
            else if (flag.starts_with("--decode-cache-slots=")) decode_cache_slots = std::stoi(flag.substr(21));
            else if (flag.starts_with("--prefill-batch="))
                prefill_batch = prefill_width(flag.substr(16));
            else if (flag.starts_with("--gpu-budget-mib="))
                gpu_budget = gpu_budget_bytes(flag.substr(17));
            else if (flag == "--decode-experts=gpu") {
                stream_decode = true;
                force_gpu = true;
            } else if (flag == "--decode-experts=cpu")
                stream_decode = false;
            else if (flag == "--speculative=mtp") { use_mtp = true; force_gpu = true; }
            else if (flag == "--mtp-experts=cpu") mtp_cpu_experts = true;
            else if (flag == "--mtp-experts=gpu") mtp_cpu_experts = false;
            else if (flag.starts_with("--draft-depth=")) draft_depth = std::stoi(flag.substr(14));
            else if (flag.starts_with("--decode-bench=")) decode_repetitions = std::stoi(flag.substr(15));
            else if (flag == "--speculative=lookup")
                lookup_depth = 3;
            else if (flag == "--speculative=none") { lookup_depth = 0; use_mtp = false; }
            else if (flag == "--check-verify")
                check_verify = true;
            else if (flag.starts_with("--lookup-depth="))
                lookup_depth = std::stoi(flag.substr(15));
            else if (flag.starts_with("--cpu-prepack-mib=")) cpu_prepack_mib = std::stoull(flag.substr(18));
            else if (flag == "--cpu-affinity=auto") pin_cpu = true;
            else if (flag == "--cpu-affinity=none") pin_cpu = false;
            else if (flag.starts_with("--routing-trace=")) routing_trace_path = flag.substr(16);
            else if (flag == "--profile")
                profile = true;
            else if (flag == "--profile-decode")
                profile_decode = true;
            else if (flag == "--gpu-prefill")
                force_gpu = true;
            else if (flag == "--warm-weights")
                warm_weights = true;
            else if (flag.starts_with("--bench="))
                repetitions = std::stoi(flag.substr(8));
            else if (flag.starts_with("--reference-batch="))
                reference_batch = prefill_width(flag.substr(18));
            else if (flag.starts_with("--context="))
                context = std::stoi(flag.substr(10));
            else if (flag.starts_with("--dump-logits="))
                dump = flag.substr(14);
            else if (flag.starts_with("--stop-ids="))
                stops = token_ids(flag.substr(11));
            else
                throw std::invalid_argument("unknown option: " + flag);
        }
        if (tokens.empty() || steps < 1 || mib < 64 || threads < 1 || prefill_batch < 0 ||
            prefill_batch > 4096 || repetitions < 1 || context < 0 ||
            (context && tokens.size() + steps > (size_t)context) || expert_mib < 0 || (replay && steps < 2) ||
            (check_prefill && (prefill_batch == 1 || expert_mib != 0)))
            throw std::invalid_argument("invalid arguments");
        if (check_verify && (context ? (size_t)context : tokens.size() + steps) < tokens.size() + 8)
            throw std::invalid_argument("--check-verify needs eight available context positions");
        if (cpu_prepack_mib > 131072 || (cpu_prepack_mib && (stream_decode || expert_mib))) throw std::invalid_argument("invalid CPU packing budget or backend");
        if (decode_repetitions < 1 || decode_repetitions > 10 || draft_depth < 1 || draft_depth > 7 || (use_mtp && (lookup_depth || replay || stream_decode)))
            throw std::invalid_argument("MTP needs depth 1..7, CPU experts, and no lookup or replay");
        if (mtp_cpu_experts && !use_mtp)
            throw std::invalid_argument("--mtp-experts=cpu requires --speculative=mtp");
        if (lookup_depth < 0 || lookup_depth > 7 || (lookup_depth && replay))
            throw std::invalid_argument("lookup depth must be 1..7; replay and lookup cannot be combined");
        if (reference_batch < 1 || reference_batch > (prefill_batch ? prefill_batch : 4096))
            throw std::invalid_argument("invalid reference batch");
        if (prefill_batch == 0 || prefill_batch > cpu::MAXT || force_gpu)
            preflight_gpu_budget(gpu_budget);
        if ((prefill_batch == 0 || prefill_batch > cpu::MAXT || force_gpu) && expert_mib)
            throw std::invalid_argument("GPU prefill/decode requires expert-cache-mib=0");
        if (decode_cache_mib < 0 || decode_cache_mib > 4096 ||
            (decode_cache_mib && ((use_mtp && !mtp_cpu_experts) || lookup_depth || stream_decode || expert_mib || cpu_prepack_mib)))
            throw std::invalid_argument("decode-cache-mib requires CPU target experts, single or CPU-draft MTP, no prepacking, and 0..4096 MiB");
        if (decode_cache_slots < 0 || decode_cache_slots > 12960 || (decode_cache_slots && !decode_cache_mib))
            throw std::invalid_argument("decode-cache-slots needs a cache budget and 0..12960 slots");
        Decoder decoder(argv[1], context ? context : tokens.size() + steps, (size_t)mib * 1024 * 1024,
                        threads, (size_t)expert_mib * 1024 * 1024, pin_cpu);
        if (prefill_batch == 0 || prefill_batch > cpu::MAXT || force_gpu)
            prefill_batch = decoder.enable_gpu(prefill_batch, gpu_budget, lookup_depth > 0 || check_verify);
        decoder.set_gpu_decode_experts(stream_decode);
        decoder.set_profile(profile);
        decoder.set_routing_trace(routing_trace_path);
        decoder.set_decode_cache_budget((size_t)decode_cache_mib * MiB);
        decoder.set_decode_cache_slots(decode_cache_slots);
        decoder.set_mtp_cpu_experts(mtp_cpu_experts);
        if (use_mtp) decoder.enable_mtp_capture();
        if (warm_weights)
            decoder.warm_weights();
        std::vector<float> logits;
        auto prefill = [&](int width) {
            std::vector<float> last;
            for (size_t t = 0; t < tokens.size(); t += width) {
                const size_t end = std::min(tokens.size(), t + width);
                last = decoder.batch(std::vector<int>(tokens.begin() + t, tokens.begin() + end));
            }
            return last;
        };
        if (check_prefill) {
            auto initial = decoder.snapshot();
            auto expected = prefill(reference_batch);
            auto expected_state = decoder.snapshot();
            decoder.restore(initial);
            logits = prefill(prefill_batch);
            auto actual_state = decoder.snapshot();
            auto close = [&](const std::vector<float> &a, const std::vector<float> &b) {
                if (a.size() != b.size())
                    return false;
                if (!(force_gpu || prefill_batch > cpu::MAXT))
                    return a == b;
                double error = 0, norm = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (!std::isfinite(a[i]))
                        return false;
                    error += (double)(a[i] - b[i]) * (a[i] - b[i]);
                    norm += (double)b[i] * b[i];
                }
                const double relative = std::sqrt(error / (norm + 1e-30));
                if (relative > 0.02)
                    std::cerr << "parity relative_L2=" << relative << " size=" << a.size() << "\n";
                return relative <= 0.02;
            };
            bool same = close(logits, expected) && actual_state.position == expected_state.position;
            for (size_t l = 0; l < actual_state.layers.size(); ++l)
                for (size_t i = 0; i < 8; ++i)
                    same &= close(actual_state.layers[l][i], expected_state.layers[l][i]);
            same &= std::max_element(logits.begin(), logits.end()) - logits.begin() ==
                    std::max_element(expected.begin(), expected.end()) - expected.begin();
            std::cerr << "parity greedy actual="
                      << (std::max_element(logits.begin(), logits.end()) - logits.begin()) << " reference="
                      << (std::max_element(expected.begin(), expected.end()) - expected.begin()) << "\n";
            if (!same)
                throw std::runtime_error(
                    "GLM: batched prefill state or logits differ from sequential execution");
            std::cerr << "batched prefill state and logits passed parity\n";
        } else {
            using Clock = std::chrono::steady_clock;
            if (repetitions > 1) {
                decoder.reset();
                prefill(prefill_batch);
            }
            for (int trial = 0; trial < repetitions; ++trial) {
                decoder.reset();
                auto start = Clock::now();
#ifdef __linux__
                rusage faults_before{}, faults_after{};
                getrusage(RUSAGE_SELF, &faults_before);
#endif
                logits = prefill(prefill_batch);
                double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                std::cerr << "PREFILL trial=" << trial << " tokens=" << tokens.size()
                          << " batch=" << prefill_batch << " ms=" << ms
                          << " tok_s=" << tokens.size() * 1000 / ms << '\n';
#ifdef __linux__
                getrusage(RUSAGE_SELF, &faults_after);
                std::cerr << "FAULTS trial=" << trial
                          << " major=" << faults_after.ru_majflt - faults_before.ru_majflt
                          << " minor=" << faults_after.ru_minflt - faults_before.ru_minflt << '\n';
#endif
                decoder.report_gpu();
            }
        }
        decoder.prepare_decode_cache();
        decoder.freeze_cache();
        if (check_verify) {
            auto cp = decoder.checkpoint();
            decoder.save(cp);
            std::vector<int> window{greedy(logits), 11, 9647, 0, 1, 2, 3, 4};
            std::vector<float> expected;
            for (int token : window) {
                auto row = decoder.step(token);
                expected.insert(expected.end(), row.begin(), row.end());
            }
            decoder.restore(cp);
            const auto actual = decoder.verify(window);
            if (actual != expected)
                throw std::runtime_error("GLM: verify logits differ from sequential decode");
            decoder.restore(cp);
            const auto replayed = decoder.step(window[0]);
            if (!std::equal(replayed.begin(), replayed.end(), expected.begin()))
                throw std::runtime_error("GLM: device checkpoint rollback changed decode logits");
            decoder.restore(cp);
            decoder.enable_verify_history(cpu::MAXT - 1);
            if (decoder.verify_history_slots()) {
                decoder.restore(cp);
                auto retained = decoder.verify(window);
                if (retained != expected)
                    throw std::runtime_error("GLM: retained verification logits differ from sequential decode");
                for (int n = 1; n <= decoder.verify_history_slots(); ++n) {
                    decoder.restore(cp);
                    decoder.batch(std::vector<int>(window.begin(), window.begin() + n), false);
                    auto expected_next = decoder.step(17);
                    if (!decoder.restore_verified(n) || decoder.step(17) != expected_next)
                        throw std::runtime_error("GLM: retained verification prefix differs from replay");
                }
                std::cerr << "VERIFY retained prefixes and next-token logits identical\n";
            }
            decoder.clear_verify_history();
            decoder.restore(cp);
            std::cerr << "VERIFY sequential logits and device rollback identical\n";
        }
        if (cpu_prepack_mib > 131072) throw std::invalid_argument("CPU packing budget must be at most 131072 MiB");
        decoder.prepare_cpu(cpu_prepack_mib * MiB);
        if (use_mtp) {
            const auto start = std::chrono::steady_clock::now();
            decoder.prepare_mtp(tokens);
            std::cerr << "MTP_PRIME ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() << '\n';
        }
        std::unique_ptr<Decoder::Snapshot> decode_start_state;
        const auto decode_start_logits = logits;
        if (decode_repetitions > 1) decode_start_state = std::make_unique<Decoder::Snapshot>(decoder.snapshot());
        for (int trial = 0; trial < decode_repetitions; ++trial) {
            if (trial) { decoder.restore(*decode_start_state); logits = decode_start_logits; }
            std::cerr << "DECODE_TRIAL index=" << trial << '\n';
        if (profile_decode) check(cudaProfilerStart());
        decoder.reset_decode_stats();
        if (use_mtp) {
            generate_mtp(decoder, logits, steps, stops, draft_depth, [](int token) {
                std::cout << token << '\n' << std::flush;
            });
        } else if (lookup_depth) {
            generate_lookup(decoder, logits, tokens, steps, stops, lookup_depth, [](int token) {
                std::cout << token << '\n' << std::flush;
            });
        } else {
            const auto decode_start = std::chrono::steady_clock::now();
            int decode_steps = 0;
            for (int i = 0; i < steps; ++i) {
                const int next = std::max_element(logits.begin(), logits.end()) - logits.begin();
                std::cout << next << '\n' << std::flush;
                if (std::find(stops.begin(), stops.end(), next) != stops.end())
                    break;
                if (i + 1 < steps) {
                    const auto step_start = std::chrono::steady_clock::now();
                    if (replay) {
                        auto saved = decoder.snapshot();
                        auto expected = decoder.step(next);
                        decoder.restore(saved);
                        logits = decoder.step(next);
                        if (logits != expected)
                            throw std::runtime_error("GLM: snapshot replay logits differ");
                        std::cerr << "snapshot replay logits identical\n";
                    } else
                        logits = decoder.step(next);
                    ++decode_steps;
                    const double step_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - step_start)
                                               .count();
                    std::cerr << "DECODE_STEP index=" << decode_steps << " ms=" << step_ms << '\n';
                }
            }
            const double decode_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start)
                    .count();
            if (decode_steps)
                std::cerr << "DECODE steps=" << decode_steps << " ms=" << decode_ms
                          << " tok_s=" << decode_steps * 1000 / decode_ms << '\n';
        }
        decoder.report_gpu();
        if (profile_decode) check(cudaProfilerStop());
        }
        if (!dump.empty()) {
            std::ofstream output(dump, std::ios::binary);
            output.write((const char *)logits.data(), logits.size() * sizeof(float));
            if (!output)
                throw std::runtime_error("GLM: failed to write logits: " + dump);
        }
        decoder.report_cache();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
