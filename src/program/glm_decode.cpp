#include "strata/kernels/cpu/exl3/pool.hpp"
#include "strata/kernels/cpu/exl3/dense_q8.hpp"
#include "strata/kernels/exl3.hpp"
// Experimental GLM decode and streamed GPU prefill. CPU experts overlap resident GPU experts.
// Greedy decoding with verified speculation and optional recurrent CUDA graphs.
#include "ggml.h"
#include "strata/core/expert_cache.hpp"
#include "strata/core/exact_expert_cache.hpp"
#include "strata/core/model.hpp"
#include "strata/core/device.hpp"
#include "strata/core/runtime_memory.hpp"
#include "strata/artifact/expert_calibration.hpp"
#include "strata/artifact/expert_block_screen.hpp"
#include "strata/artifact/json.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/glm_mailbox.hpp"
#include "strata/kernels/glm_q23.hpp"
#include "strata/kernels/canon_expert.hpp"
#include "strata/kernels/glm_q8.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/partition.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/net/expert_tp.hpp"
#include "ucomm.h"
#include <algorithm>
#include <iomanip>
#include <immintrin.h>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#ifdef __linux__
#include <sched.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace {
namespace k = strata::kernels;
namespace cpu = k::cpu;
// rocBLAS does not implement HIPBLAS_PEDANTIC_MATH. Its default FP32 SGEMM
// path is the supported HIP mode; CUDA keeps its existing pedantic setting.
#ifdef STRATA_USE_HIP
constexpr auto glm_blas_math_mode = CUBLAS_DEFAULT_MATH;
#else
constexpr auto glm_blas_math_mode = CUBLAS_PEDANTIC_MATH;
#endif
bool cached_lookup_enabled() {
    const char *value = std::getenv("STRATA_GLM_CACHE_LOOKUP");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) != "1") throw std::invalid_argument("GLM: cache lookup must be 0 or 1");
    return true;
}
int mtp_lookup_depth() {
    static const int depth = [] {
        const char *value = std::getenv("STRATA_GLM_MTP_LOOKUP_DEPTH");
        if (!value) return 0;
        size_t used = 0;
        const int n = std::stoi(value, &used);
        if (value[used] || n < 0 || n > 7)
            throw std::invalid_argument("GLM: MTP lookup depth must be 0..7");
        return n;
    }();
    return depth;
}
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("GLM: cuBLAS failure " + std::to_string(e));
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int device) {
        check(cudaGetDevice(&previous));
        if (previous != device) check(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
#ifdef __linux__
cpu_set_t allocation_cpus;
#endif
void capture_allocation() {
#ifdef __linux__
    CPU_ZERO(&allocation_cpus);
    if (sched_getaffinity(0, sizeof(allocation_cpus), &allocation_cpus))
        throw std::runtime_error("GLM: cannot read CPU allocation");
#endif
}
struct DeviceAccount {
    size_t live = 0, peak = 0, limit = SIZE_MAX;
    std::map<std::string, size_t> by_tag;   // live bytes per allocation category, for GPU_LIVE
    std::mutex tag_mutex;
};
struct Device {
    using Account = DeviceAccount;
    // Category of allocations made on this thread; set through DeviceTag.
    static inline thread_local const char *tag = "other";
    const char *category = tag;
    static inline std::array<Account, 64> accounts;
    static Account &account() {
        int device;
        check(cudaGetDevice(&device));
        return accounts.at(device);
    }
    static size_t &live() { return account().live; }
    static size_t &peak() { return account().peak; }
    static size_t &limit() { return account().limit; }
    static size_t budget_reserve() {
        // CUDA allocations outside Device (driver, graphs, library internals)
        // count against a capacity simulation too. Keep 1 GiB by default; the
        // opt-in 512 MiB reserve must be checked against actual process usage.
        static const size_t mib = [] {
            const char *value = std::getenv("STRATA_GLM_BUDGET_RESERVE_MIB");
            if (!value) return size_t(1024);
            size_t used = 0;
            const size_t n = std::stoull(value, &used);
            if (value[used] || n < 512 || n > 2048)
                throw std::invalid_argument("GLM: budget reserve must be 512..2048 MiB");
            return n;
        }();
        return mib * 1024 * 1024;
    }
    static size_t physical_reserve() {
        // STRATA_GLM_GPU_RESERVE_MIB: free device memory to leave for other clients, instead of the defaults below.
        static const long configured = [] {
            const char *value = std::getenv("STRATA_GLM_GPU_RESERVE_MIB");
            const long mib = value ? std::atol(value) : -1;
            if (value && (mib < 128 || mib > 8192)) throw std::invalid_argument("GLM: GPU reserve must be 128..8192 MiB");
            return mib;
        }();
        if (configured >= 0) return size_t(configured) * 1024 * 1024;
        int device;
        check(cudaGetDevice(&device));
        int timeout = 0; // CUDA 13 dropped cudaDeviceProp::kernelExecTimeoutEnabled; the attribute works in 12 and 13
        check(cudaDeviceGetAttribute(&timeout, cudaDevAttrKernelExecTimeout, device));
        return size_t(timeout ? 2048 : 512) * 1024 * 1024;
    }
    void *p = nullptr;
    size_t bytes;
    int owner = 0;
    bool owned = true;
    Device(void *view, size_t bytes) : p(view), bytes(bytes), owned(false) { check(cudaGetDevice(&owner)); }
    explicit Device(size_t n, bool zero = false) : bytes(n) {
        check(cudaGetDevice(&owner));
        auto &a = accounts.at(owner);
        if (a.live > a.limit || n > a.limit - a.live)
            throw std::runtime_error("GLM: GPU allocation budget exceeded: device=" + std::to_string(owner) +
                " request_MiB=" + std::to_string(n / double(1024 * 1024)) +
                " live_MiB=" + std::to_string(a.live / double(1024 * 1024)) +
                " limit_MiB=" + std::to_string(a.limit / double(1024 * 1024)));
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Headless compute cards can use their capacity; display GPUs retain 2 GiB.
        const size_t reserve = physical_reserve();
        if (free_bytes < reserve || n > free_bytes - reserve)
            throw std::runtime_error("GLM: GPU allocation would consume runtime headroom: device=" +
                std::to_string(owner) + " request_MiB=" + std::to_string(n / double(1024 * 1024)) +
                " free_MiB=" + std::to_string(free_bytes / double(1024 * 1024)));
        check(cudaMalloc(&p, n));
        a.live += n;
        a.peak = std::max(a.peak, a.live);
        { std::lock_guard<std::mutex> lock(a.tag_mutex); a.by_tag[category] += n; }
        if (zero) {
            auto status = cudaMemset(p, 0, n);
            if (status != cudaSuccess) {
                cudaFree(p); p = nullptr; a.live -= n;
                check(status);
            }
        }
    }
    ~Device() {
        if (owned) {
            DeviceScope scope(owner);
            cudaFree(p);
            auto &a = accounts.at(owner);
            a.live -= bytes;
            std::lock_guard<std::mutex> lock(a.tag_mutex);
            a.by_tag[category] -= bytes;
        }
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    float *f(size_t offset = 0) const { return (float *)p + offset; }
    void put(const void *x, size_t n) {
        DeviceScope scope(owner);
        if (n > bytes)
            throw std::runtime_error("GLM: upload exceeds buffer");
        check(cudaMemcpy(p, x, n, cudaMemcpyHostToDevice));
    }
    void put_async(const void *x, size_t n, cudaStream_t stream) {
        DeviceScope scope(owner);
        if (n > bytes) throw std::runtime_error("GLM: async upload exceeds buffer");
        check(cudaMemcpyAsync(p, x, n, cudaMemcpyHostToDevice, stream));
    }
    std::vector<float> floats(size_t n) const {
        DeviceScope scope(owner);
        if (n > bytes / sizeof(float))
            throw std::runtime_error("GLM: read exceeds buffer");
        std::vector<float> x(n);
        check(cudaMemcpy(x.data(), p, n * 4, cudaMemcpyDeviceToHost));
        return x;
    }
};
struct DeviceTag {
    const char *previous = Device::tag;
    explicit DeviceTag(const char *tag) { Device::tag = tag; }
    ~DeviceTag() { Device::tag = previous; }
};
constexpr size_t MiB = 1024 * 1024;
size_t prefill_expert_stride(const strata::core::ArtifactTensor &g,
                            const strata::core::ArtifactTensor &u,
                            const strata::core::ArtifactTensor &d) {
    // MMQ expresses expert strides in quantization blocks, whose byte sizes
    // differ between gate/up and down. Padding must be a multiple of both.
    const size_t align = std::lcm(size_t(16), std::lcm(ggml_type_size((ggml_type)g.tensor->type),
                                                    ggml_type_size((ggml_type)d.tensor->type)));
    const size_t bytes = (g.bytes + u.bytes + d.bytes) / g.tensor->shape[2];
    return (bytes + align - 1) / align * align;
}
namespace mmq = strata::prefill::mmq;
struct Pinned {
    void *p = nullptr;
    explicit Pinned(size_t bytes) { check(cudaMallocHost(&p, bytes)); }
    ~Pinned() { cudaFreeHost(p); }
};
// GPU-driven decode step (STRATA_GLM_STEP_PIPELINE=1). One slot per layer; the generation advances once per
// step, so a slot never needs clearing and a late flag from an earlier step cannot satisfy a wait.
struct StepMailbox {
    void *host = nullptr;
    k::GlmMailboxView view, device_view;
    std::unique_ptr<Device> generation;
    unsigned step = 0;
    StepMailbox(int slots, int max_tokens, int top_k, int hidden, bool rows = false, bool wanted = false) {
        const size_t bytes = k::glm_mailbox_bytes(slots, max_tokens, top_k, hidden, rows, wanted);
        check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(host, 0, bytes);
        void *mapped = nullptr;
        check(cudaHostGetDevicePointer(&mapped, host, 0));
        view = k::glm_mailbox_view(host, slots, max_tokens, top_k, hidden, rows, wanted);
        device_view = k::glm_mailbox_view(mapped, slots, max_tokens, top_k, hidden, rows, wanted);
        generation = std::make_unique<Device>(256, true);
    }
    ~StepMailbox() { cudaFreeHost(host); }
    StepMailbox(const StepMailbox &) = delete;
    StepMailbox &operator=(const StepMailbox &) = delete;
    unsigned *device_generation() const { return (unsigned *)generation->p; }
    void begin(cudaStream_t stream) {
        if (!++step) ++step;
        __atomic_store_n(view.control + 32, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(view.control + 16, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(view.control, step, __ATOMIC_SEQ_CST);
        k::glm_mailbox_begin(device_view, device_generation(), stream);
    }
    // Spins for the GPU's publication of this step; a GPU-side timeout or a stalled step throws.
    void wait_published(int slot, const std::atomic<bool> &launch_failed) const {
        const auto *flag = view.flags + (size_t)slot * 32;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned spins = 0; __atomic_load_n(flag, __ATOMIC_ACQUIRE) != step; ++spins) {
            _mm_pause();
            if ((spins & 0xffff) == 0xffff) {
                if (launch_failed) throw std::runtime_error("GLM: step enqueue failed");
                if (const unsigned error = __atomic_load_n(view.control + 16, __ATOMIC_ACQUIRE))
                    throw std::runtime_error("GLM: step pipeline GPU wait timed out at layer " + std::to_string(error - 1));
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(60))
                    throw std::runtime_error("GLM: step pipeline layer " + std::to_string(slot) + " was never published");
            }
        }
    }
    bool published(int slot) const { return __atomic_load_n(view.flags + (size_t)slot * 32, __ATOMIC_ACQUIRE) == step; }
    // Spins until the GPU has copied this step's resident rows into the mailbox.
    void wait_rows(int slot, const std::atomic<bool> &launch_failed) const {
        const auto *flag = view.flags + (size_t)slot * 32 + 8;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned spins = 0; __atomic_load_n(flag, __ATOMIC_ACQUIRE) != step; ++spins) {
            _mm_pause();
            if ((spins & 0xffff) == 0xffff) {
                if (launch_failed) throw std::runtime_error("GLM: step enqueue failed");
                if (__atomic_load_n(view.control + 32, __ATOMIC_ACQUIRE)) throw std::runtime_error("GLM: step pipeline aborted");
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(60))
                    throw std::runtime_error("GLM: resident rows of layer slot " + std::to_string(slot) + " never arrived");
            }
        }
    }
    void complete(int slot) const { __atomic_store_n(view.flags + (size_t)slot * 32 + 16, step, __ATOMIC_RELEASE); }
    void abort() const { __atomic_store_n(view.control + 32, 1u, __ATOMIC_SEQ_CST); }
    unsigned gpu_error() const { return __atomic_load_n(view.control + 16, __ATOMIC_ACQUIRE); }
    int *ids(int slot) const { return view.ids + (size_t)slot * view.max_tokens * view.top_k; }
    float *weights(int slot) const { return view.weights + (size_t)slot * view.max_tokens * view.top_k; }
    float *act(int slot) const { return view.act + (size_t)slot * view.max_tokens * view.hidden; }
    float *sum(int slot) const { return view.sum + (size_t)slot * view.max_tokens * view.hidden; }
    float *rows(int slot) const { return view.rows + (size_t)slot * view.max_tokens * view.top_k * view.hidden; }
    const int *wanted(int slot) const { return view.wanted ? view.wanted + (size_t)slot * view.max_tokens * view.top_k : nullptr; }
};
bool direct_upload_enabled() {
    const char *value = std::getenv("STRATA_GLM_DIRECT_WEIGHT_UPLOAD");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) != "1")
        throw std::invalid_argument("GLM: direct weight upload must be 0 or 1");
    return true;
}
// Registration owns no model bytes. It must outlive every GPU copy and be
// released before the artifact unmaps its original, byte-identical COW pages.
struct RegisteredWeights {
    int device;
    std::vector<void *> ranges;
    size_t bytes = 0;
    RegisteredWeights(const strata::core::ModelArtifact &artifact, int primary, const std::vector<int> &devices)
        : device(primary) {
#if defined(__linux__) && !defined(STRATA_USE_HIP)
        DeviceScope scope(primary);
        for (int id : devices) {
            int supported = 0;
            check(cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterReadOnlySupported, id));
            if (!supported) throw std::runtime_error("GLM: device cannot register read-only model pages");
        }
        const auto begin = std::chrono::steady_clock::now();
        const size_t page = sysconf(_SC_PAGESIZE);
        std::vector<std::pair<uintptr_t, uintptr_t>> spans, merged;
        for (const auto &[name, tensor] : artifact.tensors()) {
            if (name.starts_with("blk.45.") || !name.ends_with("_exps.weight")) continue;
            const auto start = reinterpret_cast<uintptr_t>(tensor.data());
            spans.emplace_back(start / page * page, (start + tensor.bytes + page - 1) / page * page);
        }
        std::sort(spans.begin(), spans.end());
        for (auto [first, last] : spans) {
            if (!merged.empty() && first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, last);
            else merged.emplace_back(first, last);
        }
        ranges.reserve(merged.size());
        try {
            for (auto [first, last] : merged) {
                auto status = cudaHostRegister((void *)first, last - first,
                    cudaHostRegisterPortable | cudaHostRegisterReadOnly);
                if (status != cudaSuccess) {
                    cudaGetLastError();
                    throw std::runtime_error(std::string("GLM: model page registration failed: ") +
                        cudaGetErrorString(status));
                }
                ranges.push_back((void *)first); bytes += last - first;
            }
        } catch (...) {
            for (auto p : ranges) cudaHostUnregister(p);
            throw;
        }
        std::cerr << "HOST_WEIGHT_UPLOAD registered_bytes=" << bytes << " ranges=" << ranges.size()
                  << " ms=" << std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - begin).count() << '\n';
#else
        throw std::invalid_argument("GLM: direct read-only weight upload requires Linux CUDA");
#endif
    }
    ~RegisteredWeights() {
        // Cleanup also runs while unwinding a CUDA error; never throw here.
        int previous = device;
        cudaGetDevice(&previous); cudaSetDevice(device);
        for (auto p : ranges) cudaHostUnregister(p);
        cudaSetDevice(previous);
    }
};
struct GpuLocality {
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t previous;
    bool changed = false;
    int previous_policy = MPOL_DEFAULT;
    std::array<unsigned long, 16> previous_nodes{};
    bool policy_changed = false;
#endif
    GpuLocality() {
#ifdef __linux__
        int dev;
        check(cudaGetDevice(&dev));
        char pci[32];
        check(cudaDeviceGetPCIBusId(pci, sizeof(pci), dev));
        for (char *c = pci; *c; ++c) *c = (char)std::tolower((unsigned char)*c);
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
                if (i < CPU_SETSIZE && CPU_ISSET(i, &allocation_cpus))
                    cpus.push_back(i);
        }
        // Consumer PCIe devices often report numa_node=-1. The decoder's
        // coordinator is already pinned to one core here; inheriting that mask
        // serializes all staging workers. Fall back to the original allowed
        // CPUs, never outside the caller's allocation/cpuset.
        if (cpus.empty())
            for (int i = 0; i < CPU_SETSIZE; ++i)
                if (CPU_ISSET(i, &allocation_cpus)) cpus.push_back(i);
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : cpus)
                CPU_SET(c, &set);
            changed = sched_setaffinity(0, sizeof(set), &set) == 0;
        }
        // Pinned buffers stay near their PCIe root even when model mappings
        // use an interleaved process policy. Restore the caller's policy.
        if (node >= 0 && node < 1024 &&
            syscall(SYS_get_mempolicy, &previous_policy, previous_nodes.data(), 1024, nullptr, 0) == 0) {
            std::array<unsigned long, 16> nodes{};
            nodes[node / 64] |= 1UL << (node % 64);
            policy_changed = syscall(SYS_set_mempolicy, MPOL_BIND, nodes.data(), 1024) == 0;
        }
#endif
    }
    ~GpuLocality() {
#ifdef __linux__
        if (changed)
            sched_setaffinity(0, sizeof(previous), &previous);
        if (policy_changed)
            syscall(SYS_set_mempolicy, previous_policy,
                    previous_policy == MPOL_DEFAULT ? nullptr : previous_nodes.data(), 1024);
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
        int count = 2;
        if (const char *value = std::getenv("STRATA_GLM_STAGE_WORKERS")) {
            size_t used = 0;
            count = std::stoi(value, &used);
            if (value[used] || count < 1 || count > 8)
                throw std::invalid_argument("GLM: staging workers must be 1..8");
        }
        for (int id = 0; id < count; ++id)
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
        pending = workers.size();
        ++generation;
        work.notify_all();
        finished.wait(lock, [&] { return !pending; });
    }
    int size() const { return workers.size(); }
};
// Remote TP: one expert tensor whose owned NUMA rows hold only the local part (gate/up rows [0, keep), or down
// columns [0, keep) of every row). copy() assembles a whole expert from them and the file's remaining bytes.
struct TrimmedSource {
    const cpu::NumaTensor *local = nullptr;
    size_t row = 0, local_row = 0, rows = 0, local_rows = 0; // full and local row bytes; rows per expert
    bool down = false;
    void copy(const uint8_t *file, int expert, uint8_t *dest) const {
        const uint8_t *source = file + (size_t)expert * rows * row;
        if (!down) {
            local->copy(expert, dest);
            std::memcpy(dest + local_rows * row, source + local_rows * row, (rows - local_rows) * row);
            return;
        }
        thread_local std::vector<uint8_t> part;
        part.resize(rows * local_row);
        local->copy(expert, part.data());
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(dest + r * row, part.data() + r * local_row, local_row);
            std::memcpy(dest + r * row + local_row, source + r * row + local_row, row - local_row);
        }
    }
};
struct GpuPrefill {
    // One 16-expert group per ring slot: 208 MiB covers GLM-5.3-Flash's GGUFs; larger experts (a Q4_K_M REAP-50
    // GGUF) raise it when the model loads.
    static inline size_t slot_bytes = 208 * MiB;
    int chunk;
    size_t cursor = 0, peak_cursor = 0;
    std::unique_ptr<Device> arena, dq, blas_workspace, mmq_workspace, slots[2];
    std::unique_ptr<Pinned> pinned[2], counts;
    std::unique_ptr<ExpertStager> stager;
    int owner = 0;
    std::unique_ptr<strata::prefill::Gemm> gemm;
    std::unique_ptr<mmq::Context> context;
    cudaStream_t copy = nullptr;
    cudaEvent_t ready[2] = {}, done[2] = {};
    uint64_t transferred = 0, groups = 0;
    uint64_t tensor_rows_actual = 0, tensor_rows_padded = 0, tensor_dequant_values = 0;
    double stage_ms = 0;
    bool defer_device_wait = false;
    const bool direct_upload = direct_upload_enabled();
    const bool staged_prefetch = [] {
        const char *v = std::getenv("STRATA_GLM_STAGE_PREFETCH");
        if (!v || std::string(v) == "0") return false;
        if (std::string(v) == "1") return true;
        throw std::invalid_argument("GLM: staged prefetch must be 0 or 1");
    }();
    const bool reuse_prefetch_slots = [] {
        const char *v = std::getenv("STRATA_GLM_PREFETCH_REUSE_DEVICE");
        if (!v || std::string(v) == "0") return false;
        if (std::string(v) == "1") return true;
        throw std::invalid_argument("GLM: prefetch device reuse must be 0 or 1");
    }();
    // Groups uploaded during the preceding mixer: (layer, group) -> pool index.
    static inline size_t prefetch_bytes = 0;
    static int prefetch_groups() {
        const char *value = std::getenv("STRATA_GLM_PREFETCH_GROUPS");
        if (!value) return 0;
        const char *staged = std::getenv("STRATA_GLM_STAGE_PREFETCH");
        const int maximum = staged && std::string(staged) == "1" ? 18 : 9;
        for (int groups = 1; groups <= maximum; ++groups)
            if (std::string(value) == std::to_string(groups)) return groups;
        throw std::invalid_argument("GLM: prefetch groups must be 1..9, or 1..18 with staged prefetch");
    }
    std::vector<std::unique_ptr<Device>> pool;
    std::vector<cudaEvent_t> pool_ready, pool_done;
    std::map<std::pair<int, int>, int> fetched;
    int take(int layer, int group) {
        auto it = fetched.find({layer, group});
        if (it == fetched.end()) return -1;
        const int index = it->second;
        fetched.erase(it);
        return index;
    }
    void prefetch(const strata::core::ArtifactTensor &G, const strata::core::ArtifactTensor &U,
                  const strata::core::ArtifactTensor &D, int layer, int group, int index, size_t stride) {
        const size_t gh = G.bytes / G.tensor->shape[2], db = D.bytes / D.tensor->shape[2], bytes = 16 * stride;
        if (bytes + 16384 > pool[index]->bytes) throw std::runtime_error("GLM: native expert group exceeds prefetch slot");
        auto *p = (char *)pool[index]->p;
        check(cudaStreamWaitEvent(copy, pool_done[index], 0));
        if (staged_prefetch) {
            // Reuse the regular two pinned buffers. The foreground ring uploader
            // runs only after the prefetch future joins; the shared ready events
            // protect both users from overwriting a host buffer during DMA.
            const int host_slot = index % 2;
            check(cudaEventSynchronize(ready[host_slot]));
            auto *host = static_cast<uint8_t *>(pinned[host_slot]->p);
            const auto begin = std::chrono::steady_clock::now();
            stager->run([&](int worker) {
                for (int j = worker; j < 16; j += stager->size()) {
                    const int expert = group * 16 + j;
                    auto *dest = host + j * stride;
                    copy_part(G, expert, dest, gh);
                    copy_part(U, expert, dest + gh, gh);
                    copy_part(D, expert, dest + 2 * gh, db);
                    if (stride > 2 * gh + db)
                        std::memset(dest + 2 * gh + db, 0, stride - 2 * gh - db);
                }
            });
            std::memset(host + bytes, 0, 16384);
            stage_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            check(cudaMemcpyAsync(p, host, bytes + 16384, cudaMemcpyHostToDevice, copy));
            check(cudaEventRecord(pool_ready[index], copy));
            check(cudaEventRecord(ready[host_slot], copy));
            transferred += bytes;
            fetched[{layer, group}] = index;
            return;
        }
        for (int j = 0; j < 16; ++j) {
            const int expert = group * 16 + j;
            check(cudaMemcpyAsync(p + j * stride, G.data() + expert * gh, gh, cudaMemcpyHostToDevice, copy));
            check(cudaMemcpyAsync(p + j * stride + gh, U.data() + expert * gh, gh, cudaMemcpyHostToDevice, copy));
            check(cudaMemcpyAsync(p + j * stride + 2 * gh, D.data() + expert * db, db, cudaMemcpyHostToDevice, copy));
            if (stride > 2 * gh + db)
                check(cudaMemsetAsync(p + j * stride + 2 * gh + db, 0, stride - 2 * gh - db, copy));
        }
        check(cudaMemsetAsync(p + bytes, 0, 16384, copy));
        check(cudaEventRecord(pool_ready[index], copy));
        transferred += bytes;
        fetched[{layer, group}] = index;
    }
    static size_t arena_bytes(int width, size_t context) {
        const size_t tile = std::min(width, 64), pools = std::max<size_t>(1, context / 4);
        const size_t mla = tile * (2052 * (512 + 64) + 2 * 64 * 512 + 33 * pools) * 4
                         + (size_t)width * 176 * 1024 + MiB;
        // The per-token MLA buffers (q, value, indexer queries) outgrow the allowance above once the context passes
        // 8192 tokens and the tile's pooled scores fill the arena; give them their own room.
        // FP16 MLA keeps query and score panels beside the FP32 panels. At
        // short contexts the old 512 KiB/token floor left no room for them.
        const size_t headroom = context > 8192 ? (size_t)width * 256 * 1024 + 64 * MiB
                              : std::getenv("STRATA_GLM_MLA_F16") ? 64 * MiB : 0;
        const size_t required = std::max({64 * MiB, (size_t)width * 512 * 1024, mla + headroom});
        size_t bytes = (required + 64 * MiB - 1) / (64 * MiB) * (64 * MiB);
        if (const char *v = std::getenv("STRATA_GLM_PREFILL_SCRATCH_CAP_MIB")) {
            size_t used = 0;
            const auto cap = std::stoull(v, &used);
            if (v[used] || cap < 64 || cap > 16384)
                throw std::invalid_argument("GLM: prefill scratch cap must be 64..16384 MiB");
            bytes = std::min(bytes, size_t(cap) * MiB);
        }
        return bytes;
    }
    explicit GpuPrefill(int width, void *stream, bool compact = false, size_t capacity = 8192,
                        bool experts_only = false, bool tensor_experts = false, bool tensor_batches = false, bool native_exl = false) : chunk(width) {
        check(cudaGetDevice(&owner));
        DeviceTag tag("prefill");
        if (compact) {
            arena = std::make_unique<Device>(32 * MiB);
            counts = std::make_unique<Pinned>(289 * 4);
            return;
        }
        if (staged_prefetch && (direct_upload || !prefetch_bytes))
            throw std::invalid_argument("GLM: staged prefetch requires GGUF expert groups and no direct upload");
        if (staged_prefetch && prefetch_bytes > slot_bytes)
            throw std::invalid_argument("GLM: staged prefetch group exceeds the pinned ring buffer");
        if (reuse_prefetch_slots && (!staged_prefetch || prefetch_groups() < 2))
            throw std::invalid_argument("GLM: prefetch device reuse requires at least two staged groups");
        GpuLocality locality;
        arena = std::make_unique<Device>(experts_only ? (size_t)width * 320 * 1024 + 192 * MiB
                                                     : arena_bytes(width, capacity));
        if(native_exl) {
            std::cerr<<"EXL3 GPU prefill chunk="<<chunk<<" scratch_MiB="<<arena->bytes/double(MiB)<<'\n';
            return;
        }
        if (!experts_only || tensor_experts) {
            dq = std::make_unique<Device>((tensor_batches ? 512 : experts_only ? 64 : 128) * MiB);
            blas_workspace = std::make_unique<Device>(32 * MiB);
        }
        mmq_workspace = std::make_unique<Device>(128 * MiB);
        for (int i = 0; i < 2; ++i) {
            if (!reuse_prefetch_slots) slots[i] = std::make_unique<Device>(slot_bytes);
            // Direct uploads read registered model pages; no staging buffer is ever touched.
            if (!direct_upload) pinned[i] = std::make_unique<Pinned>(slot_bytes);
        }
        counts = std::make_unique<Pinned>(289 * 4);
        stager = std::make_unique<ExpertStager>(locality.cpus);
        std::string error;
        if (!experts_only || tensor_experts) {
            gemm = std::make_unique<strata::prefill::Gemm>();
            if (!gemm->init_external(stream, (uint16_t *)dq->p, dq->bytes / 2, blas_workspace->p,
                                    blas_workspace->bytes, error))
                throw std::runtime_error(error);
        }
        context = std::make_unique<mmq::Context>(mmq_workspace->p, mmq_workspace->bytes);
        check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            check(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
            check(cudaEventRecord(ready[i], copy));
            check(cudaEventRecord(done[i], (cudaStream_t)stream));
        }
        if ((direct_upload || staged_prefetch) && prefetch_bytes)
            for (int i = 0; i < prefetch_groups(); ++i) {
                pool.push_back(std::make_unique<Device>(prefetch_bytes));
                pool_ready.emplace_back(); pool_done.emplace_back();
                check(cudaEventCreateWithFlags(&pool_ready[i], cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&pool_done[i], cudaEventDisableTiming));
                check(cudaEventRecord(pool_ready[i], copy));
                check(cudaEventRecord(pool_done[i], (cudaStream_t)stream));
            }
        if (reuse_prefetch_slots) {
            // Groups are consumed in ascending order. Once the first two
            // prefetched groups finish, their buffers can hold later ring
            // uploads. Both uses share the same completion event, including
            // when the next layer starts prefetching into those buffers.
            for (int i = 0; i < 2; ++i) {
                slots[i] = std::make_unique<Device>(pool[i]->p, pool[i]->bytes);
                check(cudaEventDestroy(done[i]));
                done[i] = pool_done[i]; // owned and destroyed by pool_done
            }
        }
        if (staged_prefetch)
            std::cerr << "PREFETCH_STAGED groups=" << pool.size()
                      << " device_MiB=" << pool.size() * prefetch_bytes / double(MiB)
                      << " extra_host_MiB=0 shared_host_buffers=2 reused_device_slots="
                      << (reuse_prefetch_slots ? 2 : 0) << '\n';
        std::cerr << "GPU prefill chunk=" << chunk << " staging_workers=" << stager->size() << " numa_cpus=";
        for (size_t i = 0; i < std::min<size_t>(stager->size(), locality.cpus.size()); ++i)
            std::cerr << (i ? "," : "") << locality.cpus[i];
        std::cerr << '\n';
    }
    ~GpuPrefill() {
        DeviceScope scope(owner);
        if (copy)
            cudaStreamSynchronize(copy);
        context.reset();
        gemm.reset();
        for (int i = 0; i < 2; ++i) {
            if (ready[i])
                cudaEventDestroy(ready[i]);
            if (done[i] && !reuse_prefetch_slots)
                cudaEventDestroy(done[i]);
        }
        for (auto event : pool_ready) cudaEventDestroy(event);
        for (auto event : pool_done) cudaEventDestroy(event);
        if (copy)
            cudaStreamDestroy(copy);
    }
    std::unique_ptr<Device> allocate(size_t bytes) {
        size_t offset = (cursor + 255) / 256 * 256;
        if (offset > arena->bytes || bytes > arena->bytes - offset)
            throw std::runtime_error("GLM: prefill scratch exhausted");
        cursor = offset + bytes;
        peak_cursor = std::max(peak_cursor, cursor);
        return std::make_unique<Device>((char *)arena->p + offset, bytes);
    }
    const std::map<const strata::TensorInfo*,const cpu::NumaTensor*>* numa_sources=nullptr;
    // Remote TP: tensors whose owned rows hold only this node's part of each expert. An upload assembles the whole
    // expert from those rows and the worker's part read from the file, so only that part's pages are cached.
    const std::map<const strata::TensorInfo*,TrimmedSource>* trimmed_sources=nullptr;
    // Lazily mapped experts (STRATA_GLM_MAPPED_LAZY): staging is often the first read of an expert's pages, so each
    // half of its rows is read under the policy of the node whose workers will use it, as the decode pool splits them.
    bool place_by_node=false;
    void copy_part(const strata::core::ArtifactTensor& t,int expert,void* dest,size_t bytes)const {
        if(place_by_node) {
#ifdef __linux__
            const uint8_t* src=t.data()+size_t(expert)*bytes;const size_t half=bytes/2;
            for(int node=0;node<2;++node) {
                unsigned long mask=1ul<<node;
                syscall(SYS_set_mempolicy,MPOL_PREFERRED,&mask,8*sizeof(mask));
                std::memcpy(static_cast<uint8_t*>(dest)+node*half,src+node*half,node?bytes-half:half);
            }
            syscall(SYS_set_mempolicy,MPOL_DEFAULT,nullptr,0);
            return;
#endif
        }
        if(numa_sources) {
            auto found=numa_sources->find(t.tensor);
            if(found!=numa_sources->end()){found->second->copy(expert,dest);return;}
        }
        if(trimmed_sources) {
            auto found=trimmed_sources->find(t.tensor);
            if(found!=trimmed_sources->end()){found->second.copy(t.data(),expert,static_cast<uint8_t*>(dest));return;}
        }
        std::memcpy(dest,t.data()+size_t(expert)*bytes,bytes);
    }
    void upload(const strata::core::ArtifactTensor &G, const strata::core::ArtifactTensor &U,
                const strata::core::ArtifactTensor &D, int start, int n, int slot, void *stream,
                const int *selected = nullptr, bool blobs = false, size_t expert_stride = 0,
                const int *route_bounds = nullptr) {
        // route_bounds (per-expert prefix counts of this batch's routes): experts without a route are not
        // staged, so their pages are not read. Only for groups that no later batch reuses.
        size_t gh = G.bytes / G.tensor->shape[2], db = D.bytes / D.tensor->shape[2];
        const size_t stride = expert_stride ? expert_stride : 2 * gh + db;
        const size_t bytes = n * stride;
        if (!slots[slot] || bytes + 16384 > slots[slot]->bytes)
            throw std::runtime_error("GLM: native expert group exceeds ring slot");
        if (direct_upload) {
            // Model pages remain registered and immutable for the decoder's
            // lifetime, so CPU staging and pinned-buffer reuse waits disappear.
            check(cudaStreamWaitEvent(copy, done[slot], 0));
            for (int j = 0; j < n; ++j) {
                const int expert = selected ? selected[j] : start + j;
                const size_t gu = blobs ? j * stride : j * 2 * gh;
                const size_t down = blobs ? gu + 2 * gh : n * 2 * gh + j * db;
                check(cudaMemcpyAsync((char *)slots[slot]->p + gu,
                    G.data() + expert * gh, gh, cudaMemcpyHostToDevice, copy));
                check(cudaMemcpyAsync((char *)slots[slot]->p + gu + gh,
                    U.data() + expert * gh, gh, cudaMemcpyHostToDevice, copy));
                check(cudaMemcpyAsync((char *)slots[slot]->p + down,
                    D.data() + expert * db, db, cudaMemcpyHostToDevice, copy));
                if (blobs && stride > 2 * gh + db)
                    check(cudaMemsetAsync((char *)slots[slot]->p + gu + 2 * gh + db,
                        0, stride - 2 * gh - db, copy));
            }
            check(cudaMemsetAsync((char *)slots[slot]->p + bytes, 0, 16384, copy));
            check(cudaEventRecord(ready[slot], copy));
            check(cudaStreamWaitEvent((cudaStream_t)stream, ready[slot], 0));
            transferred += bytes; ++groups;
            return;
        }
        check(cudaEventSynchronize(ready[slot]));
        auto begin = std::chrono::steady_clock::now();
        stager->run([&](int worker) {
            for (int j = worker; j < n; j += stager->size()) {
                auto *p = (uint8_t *)pinned[slot]->p;
                const int expert = selected ? selected[j] : start + j;
                if (route_bounds && route_bounds[expert + 1] == route_bounds[expert]) continue;
                const size_t gu_offset = blobs ? j * stride : j * 2 * gh;
                const size_t down_offset = blobs ? gu_offset + 2 * gh : n * 2 * gh + j * db;
                copy_part(G,expert,p+gu_offset,gh);
                copy_part(U,expert,p+gu_offset+gh,gh);
                copy_part(D,expert,p+down_offset,db);
                if (blobs && stride > 2 * gh + db)
                    std::memset(p + gu_offset + 2 * gh + db, 0, stride - 2 * gh - db);
            }
        });
        std::memset((char *)pinned[slot]->p + bytes, 0, 16384);
        stage_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (defer_device_wait) check(cudaStreamWaitEvent(copy, done[slot], 0));
        else check(cudaEventSynchronize(done[slot]));
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
// Cache complete original MMQ groups: residency must not change the product
// shape, activation quantizer, stream-k partition, or floating reduction order.
// 16-expert prefill groups in the loaded model (18 for GLM-5.3-Flash's 288 experts, 9 for a REAP-50 GGUF).
static int prefill_expert_groups = 18;
struct PrefillGroupCache {
    struct Layer {
        bool selected = false;
        std::vector<int> admitted;
        std::map<int, std::unique_ptr<Device>> entries;
        std::vector<std::unique_ptr<Device>> spare;
    };
    std::map<int, Layer> layers;
    size_t per_layer = 0;
    std::map<int, size_t> planned_groups;
    uint64_t hit_bytes = 0, hits = 0;
    bool preallocate = false;
    bool early_ring_release = false;
    struct Repair {
        std::vector<std::future<void>> copies;
        std::function<void()> verify;
    };
    std::map<std::pair<int, int>, Repair> repairs;
    size_t waited_groups = 0, checked_groups = 0;
    double repair_wait_ms = 0;
    std::exception_ptr repair_failure;
    void finish_group(int layer, int group) {
        auto it = repairs.find({layer, group});
        if (it == repairs.end()) {
            if (repair_failure) std::rethrow_exception(repair_failure);
            return;
        }
        auto repair = std::move(it->second);
        repairs.erase(it);
        const auto begin = std::chrono::steady_clock::now();
        std::exception_ptr failure;
        for (auto &copy : repair.copies) {
            try { copy.get(); }
            catch (...) { if (!failure) failure = std::current_exception(); }
        }
        ++waited_groups;
        repair_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        if (!failure && repair.verify) {
            try { repair.verify(); ++checked_groups; }
            catch (...) { failure = std::current_exception(); }
        }
        if (failure && !repair_failure) repair_failure = failure;
        if (repair_failure) std::rethrow_exception(repair_failure);
    }
    void finish_repairs() {
        std::exception_ptr failure;
        while (!repairs.empty()) {
            const auto key = repairs.begin()->first;
            try { finish_group(key.first, key.second); }
            catch (...) { if (!failure) failure = std::current_exception(); }
        }
        if (failure) std::rethrow_exception(failure);
        if (repair_failure) std::rethrow_exception(repair_failure);
    }
    ~PrefillGroupCache() {
        try { finish_repairs(); }
        catch (const std::exception &e) { std::cerr << "PREFILL_REPAIR cleanup_error=" << e.what() << '\n'; }
        catch (...) {}
    }
    void reserve(int layer_id, size_t bytes, size_t slots) {
        auto &layer = layers[layer_id];
        while (layer.entries.size() + layer.spare.size() < slots)
            layer.spare.push_back(std::make_unique<Device>(bytes));
    }
    void release_spare() {
        for (auto &[id, layer] : layers) layer.spare.clear();
    }
    void reset() {
        finish_repairs();
        waited_groups = checked_groups = 0; repair_wait_ms = 0;
        for (auto &[id, layer] : layers) layer.selected = false;
        hit_bytes = hits = 0;
    }
    void *expert(int layer_id, int expert_id, size_t expert_bytes) const {
        if (repair_failure) std::rethrow_exception(repair_failure);
        auto layer = layers.find(layer_id);
        if (layer == layers.end()) return nullptr;
        if (repairs.count({layer_id, expert_id / 16}))
            throw std::runtime_error("GLM: decode attempted to use unrepaired prefill weights");
        auto group = layer->second.entries.find(expert_id / 16);
        if (group == layer->second.entries.end()) return nullptr;
        return (char *)group->second->p + (expert_id % 16) * expert_bytes;
    }
    void select(int layer_id, const int *bounds, size_t group_bytes, int participant, int participants, int primary_groups) {
        auto &layer = layers[layer_id];
        if (layer.selected) return;
        layer.admitted.clear(); layer.selected = true;
        std::vector<int> groups;
        for (int group = 0; group < prefill_expert_groups; ++group)
            if (strata::prefill::owns_expert_group(group, participant, participants, primary_groups) &&
                bounds[group * 16 + 16] != bounds[group * 16]) groups.push_back(group);
        std::sort(groups.begin(), groups.end(), [&](int a, int b) {
            const int ca = bounds[a * 16 + 16] - bounds[a * 16];
            const int cb = bounds[b * 16 + 16] - bounds[b * 16];
            return ca != cb ? ca > cb : a < b;
        });
        const size_t cap = planned_groups.count(layer_id) ? planned_groups.at(layer_id)
                                                        : per_layer / (group_bytes + 16384);
        groups.resize(std::min(groups.size(), cap));
        layer.admitted = std::move(groups);
        for (auto it = layer.entries.begin(); it != layer.entries.end();)
            if (std::find(layer.admitted.begin(), layer.admitted.end(), it->first) == layer.admitted.end()) {
                finish_group(layer_id, it->first);
                if (preallocate) layer.spare.push_back(std::move(it->second));
                it = layer.entries.erase(it);
            } else ++it;
    }
    // A group already uploaded into prefetch slot `index`: weights() after its upload.
    void *adopt(int layer_id, int first, GpuPrefill &gpu, int index, size_t stride, cudaStream_t stream) {
        auto &layer = layers[layer_id];
        const int group = first / 16;
        const size_t bytes = 16 * stride;
        if (layer.entries.count(group)) throw std::runtime_error("GLM: prefetched a resident prefill group");
        check(cudaStreamWaitEvent(stream, gpu.pool_ready[index], 0));
        if (std::find(layer.admitted.begin(), layer.admitted.end(), group) == layer.admitted.end())
            return gpu.pool[index]->p;
        std::unique_ptr<Device> entry;
        if (preallocate && !layer.spare.empty()) {
            entry = std::move(layer.spare.back()); layer.spare.pop_back();
            if (entry->bytes != bytes + 16384) throw std::runtime_error("GLM: reserved group geometry changed");
        } else entry = std::make_unique<Device>(bytes + 16384);
        check(cudaMemcpyAsync(entry->p, gpu.pool[index]->p, bytes + 16384, cudaMemcpyDeviceToDevice, stream));
        check(cudaEventRecord(gpu.pool_done[index], stream));
        void *p = entry->p;
        layer.entries.emplace(group, std::move(entry));
        return p;
    }
    void *weights(int layer_id, int first, int n, GpuPrefill &gpu, int slot,
                  const strata::core::ArtifactTensor &g, const strata::core::ArtifactTensor &u,
                  const strata::core::ArtifactTensor &d, cudaStream_t stream, const int *route_bounds = nullptr) {
        auto &layer = layers[layer_id];
        const int group = first / 16;
        const size_t stride = prefill_expert_stride(g, u, d), bytes = n * stride;
        if (auto it = layer.entries.find(group); it != layer.entries.end()) {
            finish_group(layer_id, group);
            ++hits; hit_bytes += bytes;
            return it->second->p;
        }
        // Both MMQ prefill and native decode address these byte-identical blobs.
        const bool retained = std::find(layer.admitted.begin(), layer.admitted.end(), group) != layer.admitted.end();
        if (retained && route_bounds) throw std::runtime_error("GLM: a retained prefill group needs every expert staged");
        gpu.upload(g, u, d, first, n, slot, stream, nullptr, true, stride, route_bounds);
        if (!retained)
            return gpu.slots[slot]->p;
        std::unique_ptr<Device> entry;
        if (preallocate && !layer.spare.empty()) {
            entry = std::move(layer.spare.back()); layer.spare.pop_back();
            if (entry->bytes != bytes + 16384) throw std::runtime_error("GLM: reserved group geometry changed");
        } else entry = std::make_unique<Device>(bytes + 16384);
        check(cudaMemcpyAsync(entry->p, gpu.slots[slot]->p, bytes + 16384, cudaMemcpyDeviceToDevice, stream));
        if (early_ring_release) check(cudaEventRecord(gpu.done[slot], stream));
        void *p = entry->p;
        layer.entries.emplace(group, std::move(entry));
        return p;
    }
};
// Host-memory residency for a host smaller than the routed experts (STRATA_GLM_PREFER_RESIDENT_MIB with lazy mapped
// rows). It tracks which experts' pages this process has read and keeps their total under a budget; the router is
// biased to select only those, and experts it would have selected are read in the background and join later.
// Pages are released explicitly, so the kernel's reclaim has no reason to take resident ones.
struct RamTier {
    enum : uint8_t { Absent = 0, Resident = 1, Loading = 2 };
    struct Range { const uint8_t *data; size_t bytes; int fd; uint64_t offset; int node; };
    int layers = 0, experts = 0;
    size_t budget = 0, resident_bytes = 0, loads = 0, evictions = 0;
    std::vector<uint8_t> state;
    std::vector<float> score;
    std::vector<float> window;                              // unbiased routes since the last rebalance
    std::vector<size_t> expert_bytes;                       // per layer; 0 for layers without routed experts
    std::function<std::vector<Range>(int, int)> ranges;     // the page-aligned parts of one expert
    std::mutex mutex;
    std::condition_variable wake, idle;
    std::deque<int> queue;
    std::vector<int> finished;
    int active = 0;
    bool stopping = false;
    std::vector<std::thread> threads;
    RamTier(int layers, int experts, size_t budget) : layers(layers), experts(experts), budget(budget),
        state((size_t)layers * experts), score((size_t)layers * experts), window((size_t)layers * experts), expert_bytes(layers) {
        for (int i = 0; i < 4; ++i) threads.emplace_back([this] { work(); });
    }
    ~RamTier() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        for (auto &t : threads) t.join();
    }
    static void read_range(const Range &r) {
#ifdef __linux__
        // Page-cache pages are allocated under the reading thread's policy: put each half on its node.
        unsigned long mask = 1ul << r.node;
        syscall(SYS_set_mempolicy, MPOL_PREFERRED, &mask, 8 * sizeof(mask));
        const uintptr_t first = (uintptr_t)r.data / 4096 * 4096, last = ((uintptr_t)r.data + r.bytes + 4095) / 4096 * 4096;
#ifdef MADV_POPULATE_READ
        if (!madvise((void *)first, last - first, MADV_POPULATE_READ)) return;
#endif
        volatile uint8_t sink = 0;
        for (uintptr_t page = first; page < last; page += 4096) sink = sink + *(const uint8_t *)page;
#else
        (void)r;
#endif
    }
    static void drop_range(const Range &r) {
#ifdef __linux__
        // Only pages that lie wholly inside the expert: its neighbours keep the pages they share with it.
        const uintptr_t begin = (uintptr_t)r.data, first = (begin + 4095) / 4096 * 4096, last = (begin + r.bytes) / 4096 * 4096;
        if (last <= first) return;
        madvise((void *)first, last - first, MADV_DONTNEED);
        posix_fadvise(r.fd, r.offset + (first - begin), last - first, POSIX_FADV_DONTNEED);
#else
        (void)r;
#endif
    }
    void work() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return stopping || !queue.empty(); });
            if (stopping) return;
            const int job = queue.front();
            queue.pop_front();
            ++active;
            lock.unlock();
            // Reads are indices; drops (the syscalls that release an evicted expert's pages) are ~index.
            if (job >= 0) for (const auto &r : ranges(job / experts, job % experts)) read_range(r);
            else for (const auto &r : ranges(~job / experts, ~job % experts)) drop_range(r);
            lock.lock();
            --active;
            if (job >= 0) finished.push_back(job);
            if (queue.empty() && !active) idle.notify_all();
        }
    }
    // Starts reading an absent expert; false when it is already resident or on its way.
    bool request(int layer, int expert) {
        const int index = layer * experts + expert;
        if (state[index] != Absent) return false;
        state[index] = Loading;
        { std::lock_guard<std::mutex> lock(mutex); queue.push_back(index); }
        wake.notify_one();
        return true;
    }
    // Releases a resident expert now (its pages are dropped on a worker thread).
    void evict(int index) {
        if (state[index] != Resident) return;
        state[index] = Absent;
        resident_bytes -= expert_bytes[index / experts];
        { std::lock_guard<std::mutex> lock(mutex); queue.push_back(~index); }
        wake.notify_one();
        ++evictions;
    }
    size_t pending() {
        std::lock_guard<std::mutex> lock(mutex);
        size_t reads = active;
        for (int job : queue) reads += job >= 0;
        return reads;
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [&] { return queue.empty() && !active; });
    }
    // Finished reads become resident; then the coldest residents leave until the budget holds. `keep(index)`
    // protects experts the caller is about to use. Returns whether residency changed.
    bool settle(const std::function<bool(int)> &keep) {
        std::vector<int> done;
        { std::lock_guard<std::mutex> lock(mutex); done.swap(finished); }
        for (int index : done) {
            state[index] = Resident;
            resident_bytes += expert_bytes[index / experts];
            ++loads;
        }
        bool changed = !done.empty();
        if (resident_bytes > budget) {
            std::vector<int> order;
            for (int index = 0; index < (int)state.size(); ++index)
                if (state[index] == Resident && !keep(index)) order.push_back(index);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return score[a] != score[b] ? score[a] < score[b] : a < b; });
            for (int index : order) {
                if (resident_bytes <= budget) break;
                state[index] = Absent;
                resident_bytes -= expert_bytes[index / experts];
                { std::lock_guard<std::mutex> lock(mutex); queue.push_back(~index); }
                wake.notify_one();
                ++evictions;
                changed = true;
            }
        }
        return changed;
    }
};
// Small persistent decode workspace, independent of the large prefill arena.
// One per participating GPU, with pinned metadata and results for CPU overlap.
struct ResidentExecutor {
    struct Graph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        const float *input = nullptr;
        const int *ids = nullptr;
        ~Graph() {
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
        }
    };
    static constexpr size_t entries = cpu::MAXT * 8;
    std::unique_ptr<Device> metadata, activation, quantized, scratch, output;
    std::unique_ptr<Pinned> host_metadata, host_activation, host_output;
    // Resident geometry is fixed; expert addresses and route rows are data in
    // pinned metadata, so replacements do not invalidate these graphs.
    using GraphKey = std::tuple<int, int, float, size_t>;
    std::map<GraphKey, std::unique_ptr<Graph>> graphs;
    std::map<std::tuple<int, bool, int>, std::unique_ptr<Graph>> device_graphs;
    std::unique_ptr<Device> lookup, route_ids;
    std::unique_ptr<Pinned> host_lookup;
    std::unique_ptr<Pinned> host_reduction;
    bool lookup_preloaded = false;
    struct ReaderEvent {
        cudaEvent_t event = nullptr;
        ReaderEvent() { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        ~ReaderEvent() { if (event) cudaEventDestroy(event); }
    };
    std::map<int, std::unique_ptr<ReaderEvent>> readers;
    std::map<int, std::unique_ptr<Pinned>> reductions;
    cudaEvent_t reader_done(int layer) {
        auto &entry = readers[layer];
        if (!entry) entry = std::make_unique<ReaderEvent>();
        return entry->event;
    }
    void record_reader(int layer, cudaStream_t stream) {
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(stream, &status));
        check(cudaEventRecordWithFlags(reader_done(layer), stream,
              status == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
    }
    Pinned &reduction_for(int layer) {
        auto &entry = reductions[layer];
        if (!entry) entry = std::make_unique<Pinned>(entries * 4096 * sizeof(float) + entries * sizeof(int));
        return *entry;
    }
    ResidentExecutor() {
        DeviceTag tag("resident_exec");
        const size_t meta_bytes = entries * 8 + (3 * entries + 2) * 4;
        metadata = std::make_unique<Device>(meta_bytes);
        activation = std::make_unique<Device>(cpu::MAXT * 4096 * 4);
        quantized = std::make_unique<Device>(k::native_q8_1_bytes(4096, cpu::MAXT));
        auto canonical = k::native_expert_layout(10, 11, 4096, 2048);
        scratch = std::make_unique<Device>(std::max(k::native_expert_scratch_bytes(entries, 2048),
                                                    k::glm_q23_scratch_bytes(canonical, cpu::MAXT, entries)));
        output = std::make_unique<Device>(entries * 4096 * 4);
        host_metadata = std::make_unique<Pinned>(meta_bytes);
        host_activation = std::make_unique<Pinned>(cpu::MAXT * 4096 * 4);
        host_output = std::make_unique<Pinned>(output->bytes);
        lookup = std::make_unique<Device>(64 * 288 * sizeof(unsigned long long));
        host_lookup = std::make_unique<Pinned>(lookup->bytes);
        std::memset(host_lookup->p, 0, lookup->bytes);
        route_ids = std::make_unique<Device>(entries * sizeof(int));
        host_reduction = std::make_unique<Pinned>(entries * 4096 * sizeof(float) + entries * sizeof(int));
        check(cudaMemset(scratch->p, 0, scratch->bytes));
    }
    void update_lookup(int layer, int expert, void *address, cudaStream_t stream) {
        const size_t offset = (size_t(layer) * 288 + expert) * sizeof(unsigned long long);
        auto *cell = (unsigned long long *)((char *)host_lookup->p + offset);
        *cell = (unsigned long long)address;
        // Readers finish before slot replacement updates pinned lookup cells.
        (void)stream;
    }
    // Canonical layers (canon_expert.hpp) take the Q2_K/Q3_K kernels whose rows equal the CPU pool's bit for bit.
    void expert_rows(const k::NativeExpertLayout &layout, bool canon, const unsigned long long *ptr, const int *starts,
                     const int *count, const int *dest, const int *tokens, size_t groups, size_t rows, const float *x,
                     int nt, cudaStream_t stream) {
        if (canon) {
            k::glm_q23_expert_grouped(layout, ptr, starts, count, dest, tokens, groups, rows, x, nt, scratch->p,
                                      output->f(), stream);
            return;
        }
        k::native_quantize_q8_1(x, quantized->p, 4096, nt, stream);
        k::native_expert_grouped(layout, ptr, starts, count, dest, tokens, groups, rows, quantized->p, scratch->p,
                                 output->f(), stream, 0, nt);
    }
    void run_device(const k::NativeExpertLayout &layout, const float *x, const int *ids,
                    int layer, cudaStream_t stream, bool use_graphs, bool readback = true, int nt = 1,
                    bool canon = false) {
        if (nt < 1 || nt > cpu::MAXT) throw std::runtime_error("GLM: invalid resident batch");
        auto *ptr = (unsigned long long *)metadata->p;
        auto *starts = (int *)(ptr + entries);
        auto *dest = starts + entries + 1, *tokens = dest + entries, *count = tokens + entries;
        auto enqueue = [&] {
            const size_t offset = (size_t)layer * 288 * sizeof(unsigned long long);
            const void *source = (char *)host_lookup->p + offset;
            if (!lookup_preloaded)
                check(cudaMemcpyAsync((char *)lookup->p + offset, source,
                                      288 * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
            if (nt == 1)
                k::glm_resident_routes(ids, (const unsigned long long *)lookup->p + layer * 288,
                                       ptr, starts, dest, tokens, count, stream);
            else k::glm_resident_routes_batch(ids, (const unsigned long long *)lookup->p + layer * 288,
                                              ptr, starts, dest, tokens, count, nt, stream);
            expert_rows(layout, canon, ptr, starts, count, dest, tokens, nt * 8, nt * 8, x, nt, stream);
            if (readback)
                check(cudaMemcpyAsync(host_output->p, output->p, (size_t)nt * 8 * 4096 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        };
        if (!use_graphs) { enqueue(); return; }
        auto &graph = device_graphs[{layer, readback, nt + (canon ? 100 : 0)}];
        if (graph && (graph->input != x || graph->ids != ids)) graph.reset();
        if (!graph) {
            auto capture = std::make_unique<Graph>();
            capture->input = x; capture->ids = ids;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                throw;
            }
            check(cudaStreamEndCapture(stream, &capture->graph));
            check(cudaGraphInstantiate(&capture->executable, capture->graph, nullptr, nullptr, 0));
            graph = std::move(capture);
        }
        check(cudaGraphLaunch(graph->executable, stream));
    }
    void run(const k::NativeExpertLayout &layout, const std::map<int, std::vector<int>> &hits,
             const std::function<void *(int)> &address, const float *x, int nt, cudaStream_t stream,
             bool use_graphs = false, bool canon = false) {
        auto *ptr = (unsigned long long *)host_metadata->p;
        auto *starts = (int *)(ptr + entries);
        auto *dest = starts + entries + 1;
        auto *tokens = dest + entries;
        auto *count = tokens + entries;
        size_t groups = 0, rows = 0;
        starts[0] = 0;
        for (const auto &[expert, indices] : hits) {
            ptr[groups] = (unsigned long long)address(expert);
            for (int j : indices) {
                dest[rows] = j; tokens[rows++] = j / 8;
            }
            starts[++groups] = rows;
        }
        if (!groups || groups > entries || rows > entries || nt > cpu::MAXT)
            throw std::runtime_error("GLM: resident decode workspace geometry exceeded");
        *count = groups;
        std::memcpy(host_activation->p, x, (size_t)nt * 4096 * 4);
        const auto offset = [&](const void *p) { return (const char *)p - (const char *)host_metadata->p; };
        auto *base = (char *)metadata->p;
        auto enqueue = [&] {
            metadata->put_async(host_metadata->p, metadata->bytes, stream);
            activation->put_async(host_activation->p, (size_t)nt * 4096 * 4, stream);
            expert_rows(layout, canon, (const unsigned long long *)base,
                (const int *)(base + offset(starts)), (const int *)(base + offset(count)),
                (const int *)(base + offset(dest)), (const int *)(base + offset(tokens)),
                groups, rows, activation->f(), nt, stream);
            check(cudaMemcpyAsync(host_output->p, output->p, (size_t)nt * 8 * 4096 * 4,
                                  cudaMemcpyDeviceToHost, stream));
        };
        if (!use_graphs || nt != 1 || rows != groups) {
            enqueue(); return;
        }
        auto &graph = graphs[{layout.gu_type + (canon ? 1000 : 0), layout.d_type, layout.swiglu_limit, groups}];
        if (!graph) {
            auto captured = std::make_unique<Graph>();
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                throw;
            }
            check(cudaStreamEndCapture(stream, &captured->graph));
            check(cudaGraphInstantiate(&captured->executable, captured->graph, nullptr, nullptr, 0));
            graph = std::move(captured);
        }
        check(cudaGraphLaunch(graph->executable, stream));
    }
    void collect(const std::map<int, std::vector<int>> &hits, std::vector<float> &results) const {
        for (const auto &[expert, indices] : hits)
            for (int j : indices)
                std::copy_n((const float *)host_output->p + j * 4096, 4096, results.data() + j * 4096);
    }
};
// One persistent host submission thread per secondary GPU. Exceptions travel
// through the future; destruction drains its task before releasing buffers.
class GpuWorker {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    bool queued = false, stopping = false;
    std::thread thread;
  public:
    explicit GpuWorker(int device, bool allow_queue = false) : queued(allow_queue), thread([this, device] {
        check(cudaSetDevice(device));
        GpuLocality locality;
        for (;;) {
            std::unique_lock lock(mutex);
            ready.wait(lock, [&] { return stopping || !tasks.empty(); });
            if (tasks.empty() && stopping) return;
            auto fn = std::move(tasks.front()); tasks.pop_front();
            lock.unlock(); fn();
        }
    }) {}
    std::future<void> submit(std::function<void()> fn) {
        auto promise = std::make_shared<std::promise<void>>();
        auto result = promise->get_future();
        std::lock_guard lock(mutex);
        if ((!queued && !tasks.empty()) || stopping) throw std::runtime_error("GLM: GPU worker is unavailable");
        tasks.push_back([fn = std::move(fn), promise] {
            try { fn(); promise->set_value(); }
            catch (...) { promise->set_exception(std::current_exception()); }
        });
        ready.notify_one(); return result;
    }
    ~GpuWorker() {
        { std::lock_guard lock(mutex); stopping = true; }
        ready.notify_one(); thread.join();
    }
};
// Background copies target reserved cache slots. A slot is removed from
// routing before a copy starts, and becomes visible at the next token boundary.
struct ExpertCopyWorker {
    std::unique_ptr<GpuWorker> worker;
    std::unique_ptr<Pinned> staging;
    cudaStream_t stream = nullptr;
    cudaEvent_t completed = nullptr;
    int device;
    const bool direct_upload = direct_upload_enabled();
    explicit ExpertCopyWorker(int id) : device(id) {
        DeviceScope scope(id);
        GpuLocality locality;
        staging = std::make_unique<Pinned>(16 * MiB);
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&completed, cudaEventBlockingSync | cudaEventDisableTiming));
        worker = std::make_unique<GpuWorker>(id, true);
    }
    ~ExpertCopyWorker() {
        worker.reset();
        DeviceScope scope(device);
        if (stream) cudaStreamSynchronize(stream);
        if (completed) cudaEventDestroy(completed);
        if (stream) cudaStreamDestroy(stream);
    }
    std::future<void> copy(void *dst, const std::array<const uint8_t *, 3> &source,
                           const std::array<size_t, 3> &bytes) {
        return worker->submit([this, dst, source, bytes] {
            size_t offset = 0;
            for (int i = 0; i < 3; ++i) {
                if (offset + bytes[i] > 16 * MiB) throw std::runtime_error("GLM: expert copy staging too small");
                if (direct_upload)
                    check(cudaMemcpyAsync((char *)dst + offset, source[i], bytes[i], cudaMemcpyHostToDevice, stream));
                else std::memcpy((char *)staging->p + offset, source[i], bytes[i]);
                offset += bytes[i];
            }
            if (!direct_upload) check(cudaMemcpyAsync(dst, staging->p, offset, cudaMemcpyHostToDevice, stream));
            check(cudaEventRecord(completed, stream));
            check(cudaEventSynchronize(completed));
        });
    }
};
struct DecodeGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool warmed = false;
    int *moe_ids = nullptr;
    float *moe_weights = nullptr;
    size_t moe_cursor = 0;
    ~DecodeGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
struct SecondaryPrefill {
    int device;
    bool tensor_experts = false;
    bool tensor_batches = false;
    int saved_width = 0;
    bool peer = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t finished = nullptr;
    // Split MLA tiles: a handle configured like the primary one and the FP32 absorb weights.
    cublasHandle_t blas = nullptr;
    std::map<std::string, std::unique_ptr<Device>> mla_weights;
    // Results travel back on their own stream while this device keeps computing.
    cudaStream_t back = nullptr;
    cudaEvent_t produced = nullptr;
    // Decode tensor parallelism: this device owns the upper half of every KDA head.
    struct TensorParallel {
        std::map<std::string, std::unique_ptr<Device>> weights;
        std::vector<std::unique_ptr<Device>> recurrent, conv[3];
        std::unique_ptr<Device> x, tmp, proj[3], low, decay, beta, y, q8;
        cudaEvent_t x_ready = nullptr, y_ready = nullptr;
        std::map<int, std::unique_ptr<DecodeGraph>> graphs;
        ~TensorParallel() {
            if (x_ready) cudaEventDestroy(x_ready);
            if (y_ready) cudaEventDestroy(y_ready);
        }
    };
    std::unique_ptr<TensorParallel> tp;
    std::unique_ptr<GpuPrefill> gpu;
    std::unique_ptr<GpuWorker> worker;
    std::unique_ptr<Pinned> bounce;
    PrefillGroupCache cache;
    std::unique_ptr<ResidentExecutor> decode;
    std::map<std::string, std::unique_ptr<Device>> buffers;
    Device &buf(const std::string &name, size_t floats) {
        auto &b = buffers[name];
        if (!b) b = gpu->allocate(floats * 4);
        if (b->bytes < floats * 4) throw std::runtime_error("GLM: secondary workspace too small");
        return *b;
    }
    SecondaryPrefill(int id, int primary, int width, size_t budget, bool tensor = false, bool tensor_batch = false)
        : device(id), tensor_experts(tensor), tensor_batches(tensor_batch) {
        DeviceScope scope(id);
        Device::limit() = std::min(Device::limit(), budget - Device::budget_reserve());
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
        check(cudaStreamCreateWithFlags(&back, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
        gpu = std::make_unique<GpuPrefill>(width, stream, false, 8192, true, tensor_experts, tensor_batches);
        allocate_workspace(width);
        int forward = 0, backward = 0;
        check(cudaDeviceCanAccessPeer(&forward, primary, id));
        check(cudaDeviceCanAccessPeer(&backward, id, primary));
        peer = forward && backward && std::getenv("STRATA_GLM_DISABLE_PEER") == nullptr;
        if (peer) {
            auto enable = [](int other) {
                auto status = cudaDeviceEnablePeerAccess(other, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else check(status);
            };
            enable(primary);
            { DeviceScope main(primary); enable(id); }
        } else {
            GpuLocality locality;
            bounce = std::make_unique<Pinned>((size_t)width * 8 * 4096 * 4);
        }
        check(cudaStreamSynchronize(stream));
        worker = std::make_unique<GpuWorker>(id);
        std::cerr << "GPU_SECONDARY device=" << id << " peer=" << peer
                  << " workspace_MiB=" << Device::live() / double(MiB) << '\n';
    }
    void allocate_workspace(int width) {
        buf("x", (size_t)width * 4096);
        buf("bounds", 289); buf("dest", width * 8 + 128); buf("source", width * 8 + 128);
        buf("ids", width * 8 + 128); buf("rw", width * 8 + 128);
        buf("result", (size_t)width * 8 * 4096);
        buf("packed", (size_t)width * 8 * 4096);
        buf("gu", 4096 * 4096); buf("hidden", 4096 * 2048);
        buf("qx", (mmq::q8_bytes(4096, 4096) + 3) / 4);
        buf("qh", (mmq::q8_bytes(4096, 2048) + 3) / 4);
        buf("local", 17); buf("identity", 4096 + 128);
        if (tensor_experts) {
            buf("tensor_x", 4096 * 4096 / 2);
            buf("tensor_h", 4096 * 2048 / 2);
            buf("tensor_down", 4096 * 4096);
        }
        mmq::iota((int *)buf("identity", 4096 + 128).p, 4096 + 128, stream);
    }
    void compact_decode() {
        DeviceScope scope(device);
        check(cudaStreamSynchronize(stream));
        if (saved_width) return;
        saved_width = gpu->chunk;
        buffers.clear(); gpu.reset(); bounce.reset();
        gpu = std::make_unique<GpuPrefill>(cpu::MAXT, stream, true);
    }
    void restore_prefill() {
        if (!saved_width) return;
        DeviceScope scope(device);
        buffers.clear(); gpu.reset();
        gpu = std::make_unique<GpuPrefill>(saved_width, stream, false, 8192, true, tensor_experts, tensor_batches);
        allocate_workspace(saved_width);
        if (!peer) {
            GpuLocality locality;
            bounce = std::make_unique<Pinned>((size_t)saved_width * 8 * 4096 * 4);
        }
        check(cudaStreamSynchronize(stream));
        saved_width = 0;
    }
    ~SecondaryPrefill() {
        worker.reset();
        DeviceScope scope(device);
        if (stream) cudaStreamSynchronize(stream);
        if (blas) cublasDestroy(blas);
        mla_weights.clear();
        tp.reset();
        if (back) { cudaStreamSynchronize(back); cudaStreamDestroy(back); }
        if (produced) cudaEventDestroy(produced);
        gpu.reset();
        if (finished) cudaEventDestroy(finished);
        if (stream) cudaStreamDestroy(stream);
    }
};
// Rank 0 of cross-node expert tensor parallelism (include/strata/net/expert_tp.hpp): the remote worker computes
// FFN rows [split, n_ff) of every CPU-routed expert and returns one router-weighted sum per token.
struct RemoteTp {
    ucomm_t *comm = nullptr;
    int hidden = 0;
    std::map<int, int> split; // layer -> rows kept here
    std::vector<uint8_t> request;
    std::vector<float> reply;
    std::vector<uint8_t> wire;
    uint32_t reply_format = strata::net::kReplyF16;
    ucomm_req_t *sending = nullptr, *receiving = nullptr;
    double wait_ms = 0, send_wait_ms = 0;
    int64_t calls = 0;
    std::vector<ucomm_mr_t *> regions; // request and reply buffers, registered once (IB)
    std::set<int> trimmed; // layers whose owned NUMA rows hold only the local rows, in their own contiguous layout
    bool check = std::getenv("STRATA_GLM_REMOTE_TP_CHECK") && std::string(std::getenv("STRATA_GLM_REMOTE_TP_CHECK")) == "1";
    std::map<int, double> check_worst; // layer -> largest |local + remote - whole| / max |whole|
    int64_t check_calls = 0;
    static void ok(ucomm_status s, const char *what) {
        if (s != UCOMM_OK) throw std::runtime_error(std::string("GLM remote TP ") + what + ": " + ucomm_strerror(s));
    }
    // Sends one layer's remote work: the routed (expert, token, weight) list and each token's quantized activation.
    void request_layer(int layer, int nt, const std::vector<strata::net::TpJob> &list,
                       const std::vector<std::vector<uint8_t>> &quant, size_t act_bytes) {
        const strata::net::TpRequest head{layer, nt, (int32_t)list.size(), (int32_t)act_bytes};
        const size_t list_bytes = list.size() * sizeof(strata::net::TpJob);
        request.resize(sizeof head + list_bytes + (size_t)nt * act_bytes);
        std::memcpy(request.data(), &head, sizeof head);
        std::memcpy(request.data() + sizeof head, list.data(), list_bytes);
        for (int t = 0; t < nt; ++t)
            std::memcpy(request.data() + sizeof head + list_bytes + (size_t)t * act_bytes, quant[t].data(), act_bytes);
        start(nt);
    }
    void start(int nt) {
        reply.resize((size_t)nt * hidden);
        wire.resize(strata::net::tp_reply_bytes(reply_format, nt, hidden));
        ok(ucomm_irecv(comm, 1, strata::net::kTagReply, wire.data(), wire.size(), &receiving), "receive");
        ok(ucomm_isend(comm, 1, strata::net::kTagRequest, request.data(), request.size(), &sending), "send");
    }
    void finish(float *sum) {
        const auto t0 = std::chrono::steady_clock::now();
        ok(ucomm_wait(sending, nullptr), "send");
        send_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ok(ucomm_wait(receiving, nullptr), "reply");
        sending = receiving = nullptr;
        wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++calls;
        const int nt = (int)(reply.size() / hidden);
        if (reply_format == strata::net::kReplyF16) {
            for (int t = 0; t < nt; ++t) {
                const uint8_t *src = wire.data() + (size_t)t * (4 + 2 * (size_t)hidden);
                float scale;
                std::memcpy(&scale, src, 4);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)(src + 4), reply.data() + (size_t)t * hidden, hidden);
                for (int i = 0; i < hidden; ++i) sum[(size_t)t * hidden + i] += scale * reply[(size_t)t * hidden + i];
            }
        } else {
            std::memcpy(reply.data(), wire.data(), reply.size() * sizeof(float));
            for (size_t i = 0; i < reply.size(); ++i) sum[i] += reply[i];
        }
    }
    ~RemoteTp() {
        if (!comm) return;
        strata::net::TpRequest quit{-1, 0, 0, 0};
        ucomm_send(comm, 1, strata::net::kTagRequest, &quit, sizeof quit);
        std::cerr << "REMOTE_TP backend=" << ucomm_backend_name(comm) << " calls=" << calls << " wait_ms=" << wait_ms
                  << " wait_ms_per_call=" << (calls ? wait_ms / calls : 0.0)
                  << " send_wait_ms_per_call=" << (calls ? send_wait_ms / calls : 0.0) << '\n';
        for (auto *region : regions) ucomm_mr_dereg(region);
        if (check) {
            double worst = 0;
            for (const auto &[l, rel] : check_worst) worst = std::max(worst, rel);
            std::cerr << "REMOTE_TP_CHECK calls=" << check_calls << " worst_rel=" << worst;
            for (const auto &[l, rel] : check_worst) std::cerr << ' ' << l << ':' << rel;
            std::cerr << '\n';
        }
        ucomm_finalize(comm);
    }
};
class Decoder {
    strata::core::ModelArtifact artifact;
    const strata::core::ModelDescriptor &m;
    std::unique_ptr<RegisteredWeights> registered_weights;
    cudaStream_t stream = nullptr;
    cudaEvent_t moe_ready = nullptr, mla_ready = nullptr;
    cublasHandle_t blas = nullptr;
    cpu::ExpertPool pool;
    std::unique_ptr<RemoteTp> remote_tp;
    size_t capacity, budget, resident = 0;
    int position = 0;
    int batch_tokens = 1;
    bool split_projections = [] { const char *v = std::getenv("STRATA_GLM_SPLIT_PROJECTIONS"); return v && std::string(v) == "1"; }();
    std::map<std::string, std::unique_ptr<Device>> window_projections;
    std::set<std::string> prepared_projections;
    int projection_first = 0;
    bool fast = false, profile = false, gpu_decode_experts = false;
    std::unique_ptr<GpuPrefill> gpu;
    std::unique_ptr<GpuWorker> prefetch_launcher;
    std::future<void> prefetch_pending;
    int primary_device = 0;
    bool secondary_enabled = true;
    bool decode_prefill_cache = false;
    bool tensor_prefill_experts = false;
    bool tensor_batched_experts = false;
    bool tensor_bucket_experts = std::getenv("STRATA_GLM_F16_BUCKETS") != nullptr;
    int tensor_bucket_step = [] {
        const char *value = std::getenv("STRATA_GLM_F16_BUCKET_STEP");
        if (!value) return 0;
        for (int step : {0, 32, 64, 128})
            if (std::string(value) == std::to_string(step)) return step;
        throw std::invalid_argument("GLM: F16 bucket step must be 0,32,64,128");
    }();
    bool stable_routes = std::getenv("STRATA_GLM_STABLE_ROUTES") != nullptr || tensor_bucket_experts;
    bool preallocate_groups = std::getenv("STRATA_GLM_PREFILL_PREALLOC") != nullptr;
    bool defer_copy_wait = std::getenv("STRATA_GLM_DEFER_COPY_WAIT") != nullptr;
    bool restore_prefill_groups = std::getenv("STRATA_GLM_RESTORE_PREFILL_CACHE") != nullptr;
    bool check_restored_groups = std::getenv("STRATA_GLM_CHECK_PREFILL_RESTORE") != nullptr;
    bool defer_prefill_repair = [] {
        const char *value = std::getenv("STRATA_GLM_DEFER_PREFILL_REPAIR");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: deferred prefill repair must be 0 or 1");
    }();
    bool prepare_kda_inputs = [] {
        const char *value = std::getenv("STRATA_GLM_KDA_PREPARE");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: KDA preparation must be 0 or 1");
    }();
    bool dequant_once = std::getenv("STRATA_GLM_DEQUANT_ONCE") != nullptr;
    // Opt-in requantization of matrices otherwise kept as FP32 for MLA/mHC.
    // The original GGUF and its tensor metadata remain unchanged.
    const int floating_q8_repack = [] {
        const char *value = std::getenv("STRATA_GLM_FLOATING_Q8");
        if (!value || std::string(value) == "0") return 0;
        if (std::string(value) == "mla") return 1;
        if (std::string(value) == "all") return 2;
        throw std::invalid_argument("GLM: floating Q8 must be 0, mla or all");
    }();
    const int dense_q4_repack = [] {
        const char *value = std::getenv("STRATA_GLM_DENSE_Q4");
        if (!value || std::string(value) == "0") return 0;
        if (std::string(value) == "mixers") return 1;
        if (std::string(value) == "all") return 2;
        if (std::string(value) == "q5") return 3;
        if (std::string(value) == "outputs") return 4;
        if (std::string(value) == "q3mixers") return 5;
        throw std::invalid_argument("GLM: dense Q4 must be 0, mixers, all, q5, outputs or q3mixers");
    }();
    bool early_route_sync = std::getenv("STRATA_GLM_EARLY_ROUTE_SYNC") != nullptr;
    bool async_peer_return = std::getenv("STRATA_GLM_ASYNC_PEER_RETURN") != nullptr;
    // "2": each finished group (or MLA tile) returns on the secondary's back stream at once.
    bool incremental_return = async_peer_return && std::string(std::getenv("STRATA_GLM_ASYNC_PEER_RETURN")) == "2";
    // The secondary returns per-token weighted partial sums (nt x H) instead of every routed row (nt x K x H).
    bool partial_return = std::getenv("STRATA_GLM_PARTIAL_RETURN") != nullptr;
    // Send the secondary its activations as FP16 (what the batched expert gather rounds to anyway).
    bool half_activations = std::getenv("STRATA_GLM_HALF_X") != nullptr;
    // Percentage of MLA query tiles computed by the secondary device ("1" = half).
    int split_mla_percent = [] {
        const char *value = std::getenv("STRATA_GLM_MLA_SPLIT");
        if (!value || std::string(value) == "1") return 50;
        const int percent = std::atoi(value);
        if (percent < 10 || percent > 90) throw std::invalid_argument("GLM: MLA split must be 1 or 10..90");
        return percent;
    }();
    bool async_decode_fill = std::getenv("STRATA_GLM_ASYNC_DECODE_FILL") != nullptr;
    bool split_mla = std::getenv("STRATA_GLM_MLA_SPLIT") != nullptr;
    bool tensor_mla = std::getenv("STRATA_GLM_MLA_F16") != nullptr;
    bool decode_tp = std::getenv("STRATA_GLM_DECODE_TP") != nullptr;
    // Prefill: the upper half of every KDA head's projections, convolution, recurrence and output on GPU1.
    bool kda_split = std::getenv("STRATA_GLM_KDA_SPLIT") != nullptr;
    // Decode: one q8_1 quantization of a mixer input shared by every projection reading it.
    bool quant_once = std::getenv("STRATA_GLM_QUANT_ONCE") != nullptr;
    const bool norm_q8_fused = [] {
        const char *v = std::getenv("STRATA_GLM_NORM_Q8_FUSED");
        if (!v || std::string(v) == "0") return false;
        if (std::string(v) == "1") return true;
        throw std::invalid_argument("GLM: norm Q8 fused must be 0 or 1");
    }();
    // Persistent: reset_phase() releases its arena between normalization and the consuming mixer/FFN.
    std::unique_ptr<Device> norm_quant;
    const float *shared_quant_src = nullptr;
    int shared_quant_in = 0,shared_quant_nt=0;
    Device *shared_quant = nullptr;
    void share_quant(const float *x, int in) {
        if (!quant_once || fast || batch_tokens > cpu::MAXT) return;
        if (shared_quant && shared_quant_src == x && shared_quant_in == in && shared_quant_nt==batch_tokens) return;
        shared_quant = &buf("q8_shared", k::native_q8_1_bytes(in, batch_tokens) / 4);
        k::native_quantize_q8_1(x, shared_quant->p, in, batch_tokens, stream);
        shared_quant_src = x; shared_quant_in = in;shared_quant_nt=batch_tokens;
    }
    void end_share_quant() { shared_quant_src = nullptr; shared_quant = nullptr; }
    bool tp_stale = true;
    std::vector<int> route_tail;
    int primary_prefill_groups = [] {
        const char *value = std::getenv("STRATA_GLM_PRIMARY_GROUPS");
        if (!value) return 9;
        for (int groups = 1; groups < 18; ++groups)
            if (std::string(value) == std::to_string(groups)) return groups;
        throw std::invalid_argument("GLM: primary prefill groups must be 1..17");
    }();
    int kda_row_parts = [] {
        const char *value = std::getenv("STRATA_GLM_KDA_ROW_PARTS");
        const int parts = value ? std::atoi(value) : 1;
        if (parts != 1 && parts != 4 && parts != 8)
            throw std::invalid_argument("GLM: KDA row parts must be 1, 4 or 8");
        return parts;
    }();
    // Prefill-only precision/performance experiment. Decode retains its chosen
    // recurrence and rollback-history implementation.
    const int prefill_kda_row_parts = [] {
        const char *value = std::getenv("STRATA_GLM_PREFILL_KDA_ROW_PARTS");
        if (!value) return 0; // inherit the existing geometry
        for (int parts : {1, 4, 8})
            if (std::string(value) == std::to_string(parts)) return parts;
        throw std::invalid_argument("GLM: prefill KDA row parts must be 1, 4 or 8");
    }();
    bool split_hc_projection = std::getenv("STRATA_GLM_HC_SPLIT") != nullptr;
    bool device_resident_experts = std::getenv("STRATA_GLM_DEVICE_EXPERTS") != nullptr;
    bool active_index_pools = std::getenv("STRATA_GLM_ACTIVE_POOLS") != nullptr;
    int kda_columns = [] {
        const char* value = std::getenv("STRATA_GLM_KDA_COLUMNS");
        const int columns = value ? std::atoi(value) : 128;
        if (columns != 32 && columns != 64 && columns != 128)
            throw std::invalid_argument("GLM: KDA columns must be 32, 64 or 128");
        return columns;
    }();
    bool device_lookup_ready = false;
    bool device_reduction = std::getenv("STRATA_GLM_DEVICE_REDUCTION") != nullptr;
    bool batched_resident = std::getenv("STRATA_GLM_BATCHED_RESIDENT") != nullptr;
    bool async_moe = std::getenv("STRATA_GLM_ASYNC_MOE") != nullptr;
    int decode_admit_min = [] {
        const char *value = std::getenv("STRATA_GLM_ADMIT_MIN");
        if (!value) return 3;
        for (int n = 1; n <= 64; ++n)
            if (std::string(value) == std::to_string(n)) return n;
        throw std::invalid_argument("GLM: decode admission minimum must be 1..64");
    }();
    bool layer_graphs = std::getenv("STRATA_GLM_LAYER_GRAPHS") != nullptr;
    // GPU-driven decode step: every layer is enqueued before the CPU experts run, and each MoE layer hands
    // off through a mapped mailbox instead of a host event wait, upload and relaunch.
    bool step_pipeline = [] {
        const char *value = std::getenv("STRATA_GLM_STEP_PIPELINE");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: step pipeline must be 0 or 1");
    }();
    std::map<int, std::unique_ptr<Device>> q8_fp32_scratch;
    std::array<size_t, 3> q8_scratch_bytes{};
    static int q8_scratch_role(const std::string &name) {
        return name.ends_with("attn_k_b.weight") ? 0 : name.ends_with("attn_v_b.weight") ? 1 : 2;
    }
    // One pool dispatch per MoE layer (STRATA_GLM_LAYER_FLOW=1): see ExpertPool::run_layer_native.
    bool layer_flow = std::getenv("STRATA_GLM_LAYER_FLOW") && std::string(std::getenv("STRATA_GLM_LAYER_FLOW")) == "1";
    // llama.cpp's multi-column GEMV layout for verify windows: faster, equal to single-token rows only to rounding.
    const bool multi_upstream = [] {
        const char *value = std::getenv("STRATA_GLM_MULTI_UPSTREAM");
        const bool on = value && std::string(value) == "1";
        if (on) k::native_mmvq_set_multi_exact(false);
        if (const char *rows = std::getenv("STRATA_GLM_MULTI_ROWS")) k::native_mmvq_set_multi_rows(std::atoi(rows));
        return on;
    }();
    bool pipelining = false, pipeline_residents = false, check_cpu_residency = false;
    // Decode reads the Q8_0 MLA absorb and mHC projection weights directly instead of their FP32 copies.
    bool q8_decode = [] {
        const char *value = std::getenv("STRATA_GLM_Q8_DECODE");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: Q8 decode must be 0 or 1");
    }();
    std::unique_ptr<StepMailbox> mailbox;
    // Split verify (STRATA_GLM_SPLIT_VERIFY=1): a pipelined verify window runs as two token groups interleaved per
    // layer, so the GPU computes one group's attention while the CPU computes the other group's experts. Group B
    // uses the mailbox slots after group A's, its own main buffers and its own resident-expert executor.
    const bool split_verify = std::getenv("STRATA_GLM_SPLIT_VERIFY") && std::string(std::getenv("STRATA_GLM_SPLIT_VERIFY")) == "1";
    const int merge_expert_groups = [] {
        const char *value = std::getenv("STRATA_GLM_MOE_MERGE_GROUPS");
        if (!value || std::string(value) == "0") return 0;
        if (std::string(value) == "1") return 1;
        if (std::string(value) == "early") return 2;
        throw std::invalid_argument("GLM: expert group merge must be 0, 1 or early");
    }();
    uint64_t merged_expert_layers = 0;
    int buffer_group = 0;          // main-buffer set the launcher enqueues into (1: group B)
    long split_windows = 0;
    int history_token_offset = 0;  // first token of the enqueued group within its verify window
    int mailbox_group_stride() const { return (int)m.layers.size() + 1; }
    // STRATA_GLM_DEFER_EXPERTS=1 (lossy, one-token pipelined decode): a MoE layer's CPU experts are added to the
    // hyper-connection streams after the next layer's mixer instead of before it, so that mixer runs on the GPU
    // while the CPU computes them. The GPU's resident and shared experts are not deferred. The added term is the
    // one the layer's write would have carried through the next write (glm_mailbox_wait_add_deferred); the only
    // change is that the next mixer does not see it.
    // =2 checks the arithmetic: the late add runs right after the layer's own write, which equals no deferral.
    const int defer_experts = [] { const char *v = std::getenv("STRATA_GLM_DEFER_EXPERTS"); return v ? std::atoi(v) : 0; }();
    // STRATA_GLM_DEFER_ROUTES=<n> (with DEFER_EXPERTS): partial deferral. Only the n lowest-weight cold routes of a
    // layer run late; the CPU writes their sum to the layer's group-B mailbox slot (slot + stride) in a second pass
    // while the GPU runs the next mixer, and the other cold routes are combined in time as without deferral.
    const int defer_routes = [] {
        const char *v = std::getenv("STRATA_GLM_DEFER_ROUTES");
        const int n = v ? std::atoi(v) : 0;
        if (n < 0 || n > 8) throw std::invalid_argument("GLM: STRATA_GLM_DEFER_ROUTES must be 0..8");
        return n;
    }();
    // STRATA_GLM_DEFER_MAX_SHARE=<x>: a cold route is deferred only when its weight is below x times the layer's
    // routed weight sum (default 1 = no cap), so layers whose smallest cold route matters stay in time.
    const float defer_max_share = [] {
        const char *v = std::getenv("STRATA_GLM_DEFER_MAX_SHARE");
        const float x = v ? std::strtof(v, nullptr) : 1.f;
        if (!(x > 0.f && x <= 1.f)) throw std::invalid_argument("GLM: STRATA_GLM_DEFER_MAX_SHARE must be in (0, 1]");
        return x;
    }();
    // STRATA_GLM_DEFER_MIN_LAYER=<l>: layers below l defer nothing (their late pass is empty), since early
    // layers are the most sensitive to delayed expert output.
    const int defer_min_layer = [] { const char *v = std::getenv("STRATA_GLM_DEFER_MIN_LAYER"); return v ? std::atoi(v) : 0; }();
    int deferred_target(int l) const { return defer_routes ? l + mailbox_group_stride() : l; }
    bool defer_active = false;
    const bool fuse_route_publish = [] { const char *v = std::getenv("STRATA_GLM_FUSE_ROUTE"); return v && std::string(v) == "1"; }();
    // STRATA_GLM_TIER_LOOKUP_ONCE=1: stable one-token pipeline residency is uploaded once before the layers,
    // replacing each layer's small H2D transfer. Cache modes that change residency inside a step retain theirs.
    const bool tier_lookup_once = [] {
        const char *v = std::getenv("STRATA_GLM_TIER_LOOKUP_ONCE");
        if (!v || std::string(v) == "0") return false;
        if (std::string(v) == "1") return true;
        throw std::invalid_argument("GLM: tier lookup once must be 0 or 1");
    }();
    const bool mla_one_kernels = [] { const char *v = std::getenv("STRATA_GLM_MLA_KERNELS"); return v && std::string(v) == "1"; }();
    std::unique_ptr<Device> mla_one_scratch;
    std::unique_ptr<Device> defer_coefficients;  // the deferred layer's 24 hyper-connection coefficients
    // Per-step host timeline (STRATA_GLM_STEP_TRACE=1): routed CPU work versus everything between layers.
    bool step_trace = std::getenv("STRATA_GLM_STEP_TRACE") != nullptr;
    struct StepTrace {
        long steps = 0, layers = 0;
        double head_ms = 0, cpu_ms = 0, gap_ms = 0, tail_ms = 0, enqueue_ms = 0;
        double sync_ms = 0, plan_ms = 0, copy_ms = 0;  // tail parts: stream sync, tier bookkeeping, logits readback
        std::chrono::steady_clock::time_point start, last_done, seen;
        bool any = false;
        // STRATA_GLM_STEP_TRACE=2: the same split per mailbox slot (layer, and layer + stride for group B).
        std::array<double, 128> slot_wait{}, slot_cpu{};
    } trace;
    const bool step_trace_slots = step_trace && std::string(std::getenv("STRATA_GLM_STEP_TRACE")) == "2";
    void trace_seen() { if (step_trace) trace.seen = std::chrono::steady_clock::now(); }
    void trace_done(int slot = -1) {
        if (!step_trace) return;
        const auto now = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const double wait = ms(trace.any ? trace.last_done : trace.start, trace.seen);
        (trace.any ? trace.gap_ms : trace.head_ms) += wait;
        trace.cpu_ms += ms(trace.seen, now);
        if (slot >= 0 && slot < (int)trace.slot_wait.size()) { trace.slot_wait[slot] += wait; trace.slot_cpu[slot] += ms(trace.seen, now); }
        trace.last_done = now; trace.any = true; ++trace.layers;
    }
    struct MoePrefix { int *ids; float *weights; size_t cursor; };
    std::map<int, MoePrefix> moe_prefixes;
    int prepared_moe_layer = -1;
    std::unique_ptr<ResidentExecutor> cached_decode, cached_decode_b;
    std::unique_ptr<SecondaryPrefill> secondary;
    PrefillGroupCache prefill_cache;
    std::map<std::string, std::unique_ptr<Device>> phase;
    bool decode_graphs = false;
    bool mla_graphs = [] { const char *v = std::getenv("STRATA_GLM_MLA_GRAPHS"); return v && std::string(v) == "1"; }();
    std::unique_ptr<Device> mla_positions;
    bool graph_mla() const {
        return mla_graphs && decode_graphs && gpu && !fast && row_sequences.empty() && !secondary && !active_index_pools &&
               !std::getenv("STRATA_GLM_CHECK_MLA");
    }
    bool verify_graphs=std::getenv("STRATA_GLM_VERIFY_GRAPHS")!=nullptr;
    int graph_key(int layer)const {return layer+10000*(batch_tokens-1)+(capturing_history?100000:0)+(pipelining?200000:0)+(pipeline_residents?400000:0)+(defer_active?(defer_routes?1600000:800000):0);}
    std::map<int, std::unique_ptr<DecodeGraph>> mixer_graphs;
    template<class F> void decode_graph(int layer, F &&enqueue, int key = -1) {
        if (!decode_graphs || !gpu || fast || !row_sequences.empty() || ((batch_tokens!=1 || capturing_history) &&
            !(verify_graphs && !artifact.is_exl3() && batch_tokens<=4))) {
            enqueue(); return;
        }
        auto &entry = mixer_graphs[key >= 0 ? key : graph_key(layer)];
        if (!entry) entry = std::make_unique<DecodeGraph>();
        if (!entry->warmed) {
            enqueue(); entry->warmed = true; return;
        }
        if (!entry->executable) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try {
                enqueue();
            } catch (const std::exception &e) {
                std::cerr << "DECODE_GRAPH_CAPTURE key=" << layer << " error=" << e.what() << '\n';
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                cudaGetLastError();
                throw;
            }
            check(cudaStreamEndCapture(stream, &entry->graph));
            check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr, nullptr, 0));
        }
        check(cudaGraphLaunch(entry->executable, stream));
    }
    std::function<bool()> cancelled;
    bool cache_frozen = false;
    std::ofstream routing_trace;
    uint64_t routing_event = 0;
    int routing_request = 0, routing_round = -1;
    const char *routing_phase = "target";
    bool routing_events = std::getenv("STRATA_GLM_ROUTING_EVENTS") != nullptr;
    void trace_routes(int layer, int position, int nt, int k, const int *ids) {
        if (!routing_trace.is_open()) return;
        if (routing_events) {
            routing_trace << "{\"request\":" << routing_request << ",\"round\":" << routing_round
                << ",\"phase\":\"" << routing_phase << "\",\"event\":" << routing_event++
                << ",\"position\":" << position << ",\"layer\":" << layer << ",\"routes\":[";
            for (int t = 0; t < nt; ++t) {
                if (t) routing_trace << ',';
                routing_trace << '[';
                for (int j = 0; j < k; ++j) { if (j) routing_trace << ','; routing_trace << ids[t*k+j]; }
                routing_trace << ']';
            }
            routing_trace << "],\"resident\":[";
            bool comma = false;
            if (layer >= 0 && size_t(layer) < decode_resident.size()) {
                for (const auto &[expert, entry] : decode_resident[layer]) {
                    if (comma) routing_trace << ',';
                    routing_trace << expert; comma = true;
                }
            }
            routing_trace << "]}\n";
        } else for (int t = 0; t < nt; ++t) {
            routing_trace << position+t << ',' << layer;
            for (int j = 0; j < k; ++j) routing_trace << ',' << ids[t*k+j];
            routing_trace << '\n';
        }
        if (!routing_trace) throw std::runtime_error("GLM: failed writing routing trace");
    }
    // STRATA_GLM_ROUTER_SCORE_TRACE=path (non-pipelined CPU-expert decode): per token and MoE layer, binary
    // records {int32 position, layer, experts; float logits[experts]}; path + ".bias" holds each layer's bias once.
    std::ofstream router_score_trace, router_bias_trace;
    std::unique_ptr<Pinned> host_router_logits;
    std::set<int> router_bias_written;
    struct PreparedExperts {
        cpu::NativeFmt format;
        std::unique_ptr<uint8_t[]> gate, up, down;
    };
    std::map<int, PreparedExperts> prepared_experts;
    struct NumaExperts {std::unique_ptr<cpu::NumaTensor> gate,up,down;};
    std::map<int,NumaExperts> numa_experts;
    std::map<const strata::TensorInfo*,const cpu::NumaTensor*> numa_sources;
    std::set<const strata::TensorInfo*> numa_local_only; // remote TP: owned rows hold part of each expert
    std::map<const strata::TensorInfo*,TrimmedSource> trimmed_sources; // the same tensors, for whole-expert uploads
    std::vector<uint8_t> trim_buffer;                    // remote TP: the local rows of one tensor, before the copy to nodes
    size_t numa_weight_bytes=0;
    int q23_layout=0;
    bool packed_huge_pages=false;
    std::unique_ptr<strata::artifact::ExpertCalibration> expert_observer;
    std::unique_ptr<strata::artifact::ExpertBlockScreen> block_screen;
    std::unique_ptr<Pinned> packed_cpu_sum;
    cudaEvent_t packed_sum_uploaded=nullptr;
    std::vector<int> prepack_cpus;
    cpu::ThreadAffinity host_affinity;   // invalid (nothing to restore) until the host thread is pinned
    std::unique_ptr<Pinned> host_moe;
    std::vector<int> host_selected;
    std::unique_ptr<strata::cpu::exl3::Pool> exl_pool;
    std::map<std::string,strata::cpu::exl3::Pool::Matrix> exl_matrices;
    const strata::cpu::exl3::Pool::Matrix& exl_matrix(int layer,int expert,const std::string& projection) {
        auto key=strata::artifact::GlmExl3::prefix(layer,expert,projection);
        auto it=exl_matrices.find(key);if(it==exl_matrices.end())it=exl_matrices.emplace(key,exl_pool->load(artifact.exl3(),layer,expert,projection)).first;
        return it->second;
    }
    void exl_gpu_linear(const strata::cpu::exl3::Pool::Matrix& matrix,const float* x,float* y,int nt) {
        const int in=matrix.in,out=matrix.out;
        auto& packed=buf("exl_packed",size_t(in)*out/16);
        for(const auto& rows:matrix.rows)if(!rows.packed.empty())
            check(cudaMemcpyAsync((uint8_t*)packed.p+rows.first*in/4,rows.packed.data(),rows.packed.size(),cudaMemcpyHostToDevice,stream));
        auto& scales_in=buf("exl_suh",in/2);auto& scales_out=buf("exl_svh",out/2);
        scales_in.put_async(matrix.suh.data(),in*2,stream);scales_out.put_async(matrix.svh.data(),out*2,stream);
        auto& w=buf("exl_reconstructed",size_t(in)*out);auto& h=buf("exl_input",size_t(in)*nt);
        k::exl3_reconstruct((const uint8_t*)packed.p,w.f(),in,out,stream);
        k::exl3_hadamard(x,h.f(),(const uint16_t*)scales_in.p,in,nt,true,stream);
        const float one=1,zero=0;
        check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,out,nt,in,&one,w.f(),in,h.f(),in,&zero,y,out));
        k::exl3_hadamard(y,y,(const uint16_t*)scales_out.p,out,nt,false,stream);
        check(cudaGetLastError());
    }
    void moe_exl3_gpu(const std::string& p,int layer_id,const float* x,float* out,const strata::core::LayerDescriptor& layer) {
        const int nt=batch_tokens;
        auto& logits=buf("router_logits",m.experts*nt);auto& ids=buf("router_ids",m.top_k*nt);auto& rw=buf("router_weights",m.top_k*nt);
        mat(p+"ffn_gate_inp.weight",x,logits.f());auto& bias=weight(p+"exp_probs_b.bias");
        for(int t=0;t<nt;++t)k::glm_router(logits.f(t*m.experts),bias.f(),(int*)ids.p+t*m.top_k,rw.f(t*m.top_k),m.experts,m.top_k,m.expert_scale,stream);
        std::vector<int> selected(nt*m.top_k);std::vector<float> routing(nt*m.top_k),activation(nt*m.hidden);
        check(cudaMemcpyAsync(selected.data(),ids.p,selected.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(routing.data(),rw.p,routing.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(activation.data(),x,activation.size()*4,cudaMemcpyDeviceToHost,stream));
        ffn(p,x,out,"_shexp",m.shared_intermediate,layer.shared_swiglu_limit);check(cudaStreamSynchronize(stream));
        std::map<int,std::vector<int>> groups;for(int j=0;j<int(selected.size());++j) {
            if(selected[j]<0||selected[j]>=m.experts)throw std::runtime_error("EXL3: invalid route");
            groups[selected[j]].push_back(j);
        }
        std::vector<float> sum(nt*m.hidden,0.f),input,result;
        for(const auto& [expert,indices]:groups) {
            check_stop();int count=indices.size();input.resize(count*m.hidden);result.resize(count*m.hidden);
            for(int j=0;j<count;++j)std::copy_n(activation.data()+(indices[j]/m.top_k)*m.hidden,m.hidden,input.data()+j*m.hidden);
            auto& dx=buf("exl_group_x",count*m.hidden);auto& gate=buf("exl_gate",count*layer.intermediate);
            auto& up=buf("exl_up",count*layer.intermediate);auto& dy=buf("exl_group_y",count*m.hidden);dx.put(input.data(),input.size()*4);
            exl_gpu_linear(exl_matrix(layer_id,expert,"gate_proj"),dx.f(),gate.f(),count);
            exl_gpu_linear(exl_matrix(layer_id,expert,"up_proj"),dx.f(),up.f(),count);
            k::glm_swiglu(gate.f(),up.f(),gate.f(),count*layer.intermediate,layer.swiglu_limit,stream);
            exl_gpu_linear(exl_matrix(layer_id,expert,"down_proj"),gate.f(),dy.f(),count);
            check(cudaMemcpyAsync(result.data(),dy.p,result.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            for(int j=0;j<count;++j)for(int i=0;i<m.hidden;++i)sum[(indices[j]/m.top_k)*m.hidden+i]+=routing[indices[j]]*result[j*m.hidden+i];
        }
        auto& routed=buf("moe_sum",sum.size());routed.put(sum.data(),sum.size()*4);const float one=1;
        check(cublasSaxpy(blas,sum.size(),&one,routed.f(),1,out,1));
    }
    struct NativeGpuExpert {
        std::unique_ptr<Device> data;
        std::array<size_t,3> packed,suh,svh,in,out;
        std::list<std::pair<int,int>>::iterator order;
    };
    size_t native_expert_gpu_limit=[] {
        const char* v=std::getenv("STRATA_EXL3_GPU_EXPERT_CACHE_MIB");
        size_t mib=v?std::stoull(v):0;
        if(mib>4096)throw std::invalid_argument("EXL3 GPU expert cache must be 0..12288 MiB");
        return mib*MiB;
    }();
    size_t native_expert_gpu_bytes=0,native_expert_gpu_hits=0,native_expert_gpu_misses=0,native_expert_gpu_copied=0;
    std::map<std::pair<int,int>,NativeGpuExpert> native_gpu_experts;
    std::list<std::pair<int,int>> native_gpu_expert_lru;
    std::unique_ptr<Pinned> native_expert_upload;
    size_t native_expert_upload_bytes=0;
    const strata::cpu::exl3::Pool::Matrix& native_matrix(int layer,int expert,const std::string& projection) {
        const auto key=strata::artifact::GlmExl3::prefix(layer,expert,projection);
        auto it=exl_matrices.find(key);
        if(it==exl_matrices.end())it=exl_matrices.emplace(key,exl_pool->load(artifact.exl3(),layer,expert,projection)).first;
        return it->second;
    }
    NativeGpuExpert& native_gpu_expert(int layer,int expert) {
        const auto key=std::make_pair(layer,expert);auto found=native_gpu_experts.find(key);
        if(found!=native_gpu_experts.end()) {
            ++native_expert_gpu_hits;native_gpu_expert_lru.splice(native_gpu_expert_lru.begin(),native_gpu_expert_lru,found->second.order);
            return found->second;
        }
        ++native_expert_gpu_misses;NativeGpuExpert entry;size_t bytes=0;
        std::array<const strata::cpu::exl3::Pool::Matrix*,3> matrices;
        const char* projections[]={"gate_proj","up_proj","down_proj"};
        for(unsigned i=0;i<3;++i) {
            const auto& m=native_matrix(layer,expert,projections[i]);matrices[i]=&m;entry.in[i]=m.in;entry.out[i]=m.out;
            entry.packed[i]=bytes;bytes+=m.in*m.out/4;entry.suh[i]=bytes;bytes+=m.in*2;entry.svh[i]=bytes;bytes+=m.out*2;
        }
        if(bytes>native_expert_gpu_limit)throw std::runtime_error("EXL3 GPU cache cannot hold one expert");
        // CUDA/runtime allocations and display use can grow after startup. Keep
        // extra room beyond Device's physical guard; the cache yields first.
        auto fits=[&] {
            size_t free,total;check(cudaMemGetInfo(&free,&total));
            return native_expert_gpu_bytes+bytes<=native_expert_gpu_limit &&
                free>=bytes+Device::physical_reserve()+256*MiB &&
                Device::limit()>=Device::live()+bytes+256*MiB;
        };
        auto evict=[&](std::list<std::pair<int,int>>::iterator victim) {
            check(cudaStreamSynchronize(stream));const auto old=*victim;
            native_expert_gpu_bytes-=native_gpu_experts.at(old).data->bytes;
            native_gpu_experts.erase(old);native_gpu_expert_lru.erase(victim);
        };
        while(!fits()) {
            if(native_gpu_expert_lru.empty())throw std::runtime_error("EXL3 GPU expert cache: insufficient runtime headroom for one expert");
            evict(std::prev(native_gpu_expert_lru.end()));
        }
        if(!native_expert_upload || native_expert_upload_bytes<bytes) {native_expert_upload=std::make_unique<Pinned>(bytes);native_expert_upload_bytes=bytes;}
        auto* dst=static_cast<uint8_t*>(native_expert_upload->p);
        for(unsigned i=0;i<3;++i) {
            const auto& m=*matrices[i];
            exl_pool->run([&](size_t rank) {
                const auto& r=m.rows[rank];if(!r.packed.empty())
                    std::memcpy(dst+entry.packed[i]+(r.first/16)*(m.in/16)*64,r.packed.data(),r.packed.size());
            });
            std::memcpy(dst+entry.suh[i],m.suh.data(),m.in*2);std::memcpy(dst+entry.svh[i],m.svh.data(),m.out*2);
        }
        entry.data=std::make_unique<Device>(bytes);entry.data->put(dst,bytes);
        native_expert_gpu_copied+=bytes;native_expert_gpu_bytes+=bytes;
        native_gpu_expert_lru.push_front(key);entry.order=native_gpu_expert_lru.begin();
        return native_gpu_experts.emplace(key,std::move(entry)).first->second;
    }
    void native_gpu_projection(const NativeGpuExpert& e,unsigned projection,const float* x,float* y) {
        const auto* data=static_cast<const uint8_t*>(e.data->p);const int in=e.in[projection],out=e.out[projection];
        auto& h=buf("native_expert_h",std::max(in,out));
        k::exl3_hadamard(x,h.f(),reinterpret_cast<const uint16_t*>(data+e.suh[projection]),in,1,true,stream);
        k::exl3_packed_gemv(data+e.packed[projection],h.f(),y,in,out,stream);
        k::exl3_hadamard(y,y,reinterpret_cast<const uint16_t*>(data+e.svh[projection]),out,1,false,stream);
    }
    void moe_exl3_gpu_decode(int layer,const float* x,float* out,const strata::core::LayerDescriptor& desc) {
        const auto* ids=static_cast<const int*>(host_moe->p);const float* routing=reinterpret_cast<const float*>(ids+cpu::MAXT*m.top_k);
        auto& gate=buf("exl_decode_gate",desc.intermediate);auto& up=buf("exl_decode_up",desc.intermediate);
        auto& hidden=buf("exl_decode_hidden",desc.intermediate);auto& result=buf("exl_decode_result",m.hidden);
        for(int j=0;j<m.top_k;++j) {
            check_stop();int expert=ids[j];if(expert<0||expert>=m.experts)throw std::runtime_error("EXL3: invalid GPU route");
            auto& e=native_gpu_expert(layer,expert);
            native_gpu_projection(e,0,x,gate.f());native_gpu_projection(e,1,x,up.f());
            k::glm_swiglu(gate.f(),up.f(),hidden.f(),desc.intermediate,desc.swiglu_limit,stream);
            native_gpu_projection(e,2,hidden.f(),result.f());
            check(cublasSaxpy(blas,m.hidden,routing+j,result.f(),1,out,1));
        }
        check(cudaGetLastError());
    }
    void moe_exl3(const std::string& p,int l,const float* x,float* out,const strata::core::LayerDescriptor& layer) {
        const int nt=batch_tokens;
        if(nt>cpu::MAXT)throw std::runtime_error("EXL3: CPU routed batch exceeds MAXT");
        enqueue_moe_prefix(p,l,x,out,layer);prepared_moe_layer=-1;
        check(cudaEventSynchronize(moe_ready));
        if(nt==1 && native_dense_q8 && native_expert_gpu_limit) {moe_exl3_gpu_decode(l,x,out,layer);return;}
        auto* ids=(int*)host_moe->p;auto* routing=(float*)(ids+cpu::MAXT*m.top_k);auto* activation=routing+cpu::MAXT*m.top_k;
        std::vector<float> gate(layer.intermediate),up(layer.intermediate),result(m.hidden),sum(nt*m.hidden,0.f);
        auto matrix=[&](int expert,const std::string& projection)->const strata::cpu::exl3::Pool::Matrix& {
            auto key=strata::artifact::GlmExl3::prefix(l,expert,projection);
            auto it=exl_matrices.find(key);if(it==exl_matrices.end())it=exl_matrices.emplace(key,exl_pool->load(artifact.exl3(),l,expert,projection)).first;
            return it->second;
        };
        for(int t=0;t<nt;++t)for(int j=0;j<m.top_k;++j) {
            check_stop();int e=ids[t*m.top_k+j];if(e<0||e>=m.experts)throw std::runtime_error("EXL3: invalid route");
            exl_pool->apply(matrix(e,"gate_proj"),activation+t*m.hidden,gate.data());
            exl_pool->apply(matrix(e,"up_proj"),activation+t*m.hidden,up.data());
            for(size_t i=0;i<gate.size();++i) {float g=std::min(gate[i],layer.swiglu_limit);float u=std::clamp(up[i],-layer.swiglu_limit,layer.swiglu_limit);gate[i]=g/(1+std::exp(-g))*u;}
            exl_pool->apply(matrix(e,"down_proj"),gate.data(),result.data());
            for(int i=0;i<m.hidden;++i)sum[t*m.hidden+i]+=routing[t*m.top_k+j]*result[i];
        }
        auto& routed=buf("moe_sum",sum.size());routed.put(sum.data(),sum.size()*4);const float one=1;
        check(cublasSaxpy(blas,sum.size(),&one,routed.f(),1,out,1));
    }
    std::vector<float> host_results, host_sum;
    std::vector<std::vector<uint8_t>> host_quant;
    std::vector<cpu::ExpertJobMulti> host_jobs;
    bool capture_hidden = false;
    int mtp_position = 0, prefill_width_saved = 0;
    std::vector<float> target_hidden, mtp_hidden;
    std::unique_ptr<Device> mtp_experts;
    // Device addresses of the GPU draft experts, so a draft step builds its expert groups without the host.
    std::unique_ptr<Device> mtp_lookup;
    bool mtp_ready = false, mtp_cpu_experts = false;
    // Batched draft-layer priming during prefill (STRATA_GLM_MTP_BATCHED): positions primed so far and the
    // last target hidden row of the previous batch, kept on the device.
    bool mtp_batched = std::getenv("STRATA_GLM_MTP_BATCHED") != nullptr;
    bool mtp_primed = false;
    std::unique_ptr<Device> mtp_prev_hidden;
    k::NativeExpertLayout mtp_layout;
    size_t mtp_expert_bytes = 0;
    uint64_t cache_hits = 0, cache_entries = 0, pruned_routes = 0;
    std::vector<std::unique_ptr<strata::core::ExpertCache>> expert_cache;
    size_t decode_cache_budget = 0;
    bool decode_cache_auto = false, decode_cache_extend = false;
    int decode_cache_window = 256;
    // Fixed-size tiers historically rank the whole prompt. Opt in to the
    // same recent-route window as the automatic tier, without changing
    // routing or expert arithmetic. Long headers otherwise dominate the
    // initial tier even when generation follows a different topic.
    const bool fixed_cache_recent = [] {
        const char *value = std::getenv("STRATA_GLM_DECODE_CACHE_RECENT");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: fixed decode cache recent must be 0 or 1");
    }();
    std::vector<std::vector<int>> prefill_recent_ids;
    std::array<std::unique_ptr<Device>, 2> decode_cache_storage;
    struct PromotedGroup {
        int layer, group;
        size_t stride;
        std::unique_ptr<Device> owner;
        uint16_t dirty = 0;
    };
    std::array<std::vector<PromotedGroup>, 2> decode_group_storage;
    std::array<std::map<uintptr_t, size_t>, 2> promoted_addresses;
    std::vector<std::map<int, std::unique_ptr<Device>>> remote_decode_resident;
    bool decode_cache_adapt = false;
    struct Victim {
        uint64_t score;
        int layer, expert, participant;
        bool operator>(const Victim &other) const {
            return std::tie(score, layer, expert, participant) >
                   std::tie(other.score, other.layer, other.expert, other.participant);
        }
    };
    struct Copy {
        int layer, expert, participant;
        std::unique_ptr<Device> entry;
        std::future<void> ready;
    };
    std::map<size_t, std::priority_queue<Victim, std::vector<Victim>, std::greater<Victim>>> victims;
    std::vector<std::vector<uint64_t>> decode_seen;
    std::array<std::unique_ptr<ExpertCopyWorker>, 2> expert_copies;
    std::array<std::vector<std::unique_ptr<Copy>>, 2> pending_copies;
    uint64_t adaptive_copies = 0, adaptive_bytes = 0;
    double adaptive_wait_ms = 0;
    void mark_promoted_slot(int participant, void *address, size_t bytes) {
        if (!restore_prefill_groups) return;
        auto &addresses = promoted_addresses[participant];
        const auto ptr = reinterpret_cast<uintptr_t>(address);
        auto it = addresses.upper_bound(ptr);
        if (it == addresses.begin()) return;
        --it;
        auto &group = decode_group_storage[participant].at(it->second);
        const size_t offset = ptr - it->first;
        if (offset >= 16 * group.stride) return; // Separate decode-only slab.
        if (offset % group.stride || bytes > group.stride)
            throw std::runtime_error("GLM: promoted slot geometry changed");
        group.dirty |= uint16_t(1u << (offset / group.stride));
    }
    void verify_prefill_group(int device, int layer, int group, size_t stride, Device *owner) {
        DeviceScope scope(device);
        const auto p = "blk." + std::to_string(layer) + ".";
        const auto &g = artifact.at(p + "ffn_gate_exps.weight");
        const auto &u = artifact.at(p + "ffn_up_exps.weight");
        const auto &d = artifact.at(p + "ffn_down_exps.weight");
        const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
        std::vector<uint8_t> actual(owner->bytes);
        check(cudaMemcpy(actual.data(), owner->p, actual.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 16; ++i) {
            const size_t expert = group * 16 + i;
            size_t offset = i * stride;
            const std::array<const uint8_t *, 3> source{g.data() + expert * sizes[0],
                u.data() + expert * sizes[1], d.data() + expert * sizes[2]};
            for (int role = 0; role < 3; ++role) {
                if (std::memcmp(actual.data() + offset, source[role], sizes[role]))
                    throw std::runtime_error("GLM: restored prefill bytes differ from model");
                offset += sizes[role];
            }
            const size_t end = (i + 1) * stride;
            if (std::any_of(actual.begin() + offset, actual.begin() + end,
                            [](uint8_t x) { return x != 0; }))
                throw std::runtime_error("GLM: restored prefill padding changed");
        }
        if (std::any_of(actual.end() - 16384, actual.end(), [](uint8_t x) { return x != 0; }))
            throw std::runtime_error("GLM: restored prefill guard changed");
    }
    void finish_prefill_repairs(bool report = false) {
        std::exception_ptr failure;
        for (int participant = 0; participant < (secondary ? 2 : 1); ++participant) {
            const int device = participant ? secondary->device : primary_device;
            auto &cache = participant ? secondary->cache : prefill_cache;
            try { cache.finish_repairs(); }
            catch (...) { if (!failure) failure = std::current_exception(); }
            if (report && cache.waited_groups)
                std::cerr << "PREFILL_REPAIR device=" << device << " groups=" << cache.waited_groups
                          << " checked_groups=" << cache.checked_groups << " wait_ms=" << cache.repair_wait_ms << '\n';
        }
        if (failure) std::rethrow_exception(failure);
    }
    void defer_promoted_repairs() {
        const auto begin = std::chrono::steady_clock::now();
        size_t retained = 0, restored = 0, bytes = 0;
        for (int participant = 0; participant < (secondary ? 2 : 1); ++participant) {
            const int device = participant ? secondary->device : primary_device;
            DeviceScope scope(device);
            auto &cache = participant ? secondary->cache : prefill_cache;
            for (auto &group : decode_group_storage[participant]) {
                const auto p = "blk." + std::to_string(group.layer) + ".";
                const auto &g = artifact.at(p + "ffn_gate_exps.weight");
                const auto &u = artifact.at(p + "ffn_up_exps.weight");
                const auto &d = artifact.at(p + "ffn_down_exps.weight");
                const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
                if (group.stride != prefill_expert_stride(g, u, d) ||
                    group.owner->bytes != 16 * group.stride + 16384)
                    throw std::runtime_error("GLM: restored group geometry changed");
                auto &entries = cache.layers[group.layer].entries;
                if (entries.count(group.group) || cache.repairs.count({group.layer, group.group}))
                    throw std::runtime_error("GLM: duplicate deferred prefill group");
                auto &repair = cache.repairs[{group.layer, group.group}];
                repair.copies.reserve(16);
                if (check_restored_groups)
                    repair.verify = [this, device, layer = group.layer, id = group.group,
                                     stride = group.stride, owner = group.owner.get()] {
                        verify_prefill_group(device, layer, id, stride, owner);
                    };
                // The cache owns each target until its futures are drained.
                auto &owner = entries.emplace(group.group, std::move(group.owner)).first->second;
                for (int i = 0; i < 16; ++i) {
                    if (!(group.dirty & (1u << i))) continue;
                    const int expert = group.group * 16 + i;
                    const std::array<const uint8_t *, 3> source{g.data() + expert * sizes[0],
                        u.data() + expert * sizes[1], d.data() + expert * sizes[2]};
                    repair.copies.push_back(expert_copies[participant]->copy(
                        (char *)owner->p + i * group.stride, source, sizes));
                    bytes += sizes[0] + sizes[1] + sizes[2]; ++restored;
                }
                ++retained;
            }
            decode_group_storage[participant].clear();
            promoted_addresses[participant].clear();
        }
        if (retained)
            std::cerr << "PREFILL_RESTORE groups=" << retained << " repaired_slots=" << restored
                      << " checked_groups=0 bytes=" << bytes << " ms=" << std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin).count() << " deferred=1\n";
    }
    void restore_promoted_groups() {
        if (defer_prefill_repair) { defer_promoted_repairs(); return; }
        const auto begin = std::chrono::steady_clock::now();
        size_t restored = 0, retained = 0, bytes = 0;
        std::vector<std::future<void>> repairs;
        // Queue both devices before waiting; each device worker serializes its
        // pinned staging buffer and signals only after the H2D read completes.
        for (int participant = 0; participant < (secondary ? 2 : 1); ++participant) {
            DeviceScope scope(participant ? secondary->device : primary_device);
            for (auto &group : decode_group_storage[participant]) {
                const auto p = "blk." + std::to_string(group.layer) + ".";
                const auto &g = artifact.at(p + "ffn_gate_exps.weight");
                const auto &u = artifact.at(p + "ffn_up_exps.weight");
                const auto &d = artifact.at(p + "ffn_down_exps.weight");
                const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
                if (group.stride != prefill_expert_stride(g, u, d) ||
                    group.owner->bytes != 16 * group.stride + 16384)
                    throw std::runtime_error("GLM: restored group geometry changed");
                for (int i = 0; i < 16; ++i) {
                    if (!(group.dirty & (1u << i))) continue;
                    const int expert = group.group * 16 + i;
                    const std::array<const uint8_t *, 3> source{g.data() + expert * sizes[0],
                        u.data() + expert * sizes[1], d.data() + expert * sizes[2]};
                    // Decode views and all pending readers/copies are gone.
                    // Restore original bytes; padding was never overwritten.
                    repairs.push_back(expert_copies[participant]->copy(
                        (char *)group.owner->p + i * group.stride, source, sizes));
                    bytes += sizes[0] + sizes[1] + sizes[2]; ++restored;
                }
            }
        }
        std::exception_ptr failure;
        for (auto &repair : repairs) {
            try { repair.get(); }
            catch (...) { if (!failure) failure = std::current_exception(); }
        }
        if (failure) std::rethrow_exception(failure);
        for (int participant = 0; participant < (secondary ? 2 : 1); ++participant) {
            DeviceScope scope(participant ? secondary->device : primary_device);
            auto &cache = participant ? secondary->cache : prefill_cache;
            for (auto &group : decode_group_storage[participant]) {
                if (check_restored_groups) {
                    verify_prefill_group(participant ? secondary->device : primary_device,
                                         group.layer, group.group, group.stride, group.owner.get());
                }
                auto &entries = cache.layers[group.layer].entries;
                if (entries.count(group.group))
                    throw std::runtime_error("GLM: duplicate restored prefill group");
                entries.emplace(group.group, std::move(group.owner));
                ++retained;
            }
            decode_group_storage[participant].clear();
            promoted_addresses[participant].clear();
        }
        if (retained)
            std::cerr << "PREFILL_RESTORE groups=" << retained << " repaired_slots=" << restored
                      << " checked_groups=" << (check_restored_groups ? retained : 0)
                      << " bytes=" << bytes << " ms=" << std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin).count() << '\n';
    }
    uint64_t expert_score(int layer, int expert) const {
        return decode_seen[layer][expert] + prefill_routes[layer][expert] / 32;
    }
    void finish_copy(int participant) {
        for (auto &copy : pending_copies[participant]) {
            const auto start = std::chrono::steady_clock::now();
            copy->ready.get();
            adaptive_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            const size_t bytes = copy->entry->bytes;
            void *address = copy->entry->p;
            auto &destination = participant ? remote_decode_resident : decode_resident;
            destination[copy->layer].emplace(copy->expert, std::move(copy->entry));
            if (device_lookup_ready) {
                const int device = participant ? secondary->device : primary_device;
                DeviceScope scope(device);
                auto &executor = participant ? secondary->decode : cached_decode;
                executor->update_lookup(copy->layer, copy->expert, address,
                                        participant ? secondary->stream : stream);
            }
            victims[bytes].push({expert_score(copy->layer, copy->expert),
                                 copy->layer, copy->expert, participant});
        }
        pending_copies[participant].clear();
    }
    void adapt_expert(int layer, int expert, const strata::core::ArtifactTensor &g,
                      const strata::core::ArtifactTensor &u, const strata::core::ArtifactTensor &d) {
        if (decode_seen[layer][expert] < (uint64_t)decode_admit_min ||
            (pending_copies[0].size() >= 8 && (!secondary || pending_copies[1].size() >= 8))) return;
        const size_t bytes = (g.bytes + u.bytes + d.bytes) / m.experts;
        auto &heap = victims[bytes];
        while (!heap.empty()) {
            const auto victim = heap.top();
            auto &cache = victim.participant ? remote_decode_resident : decode_resident;
            auto found = cache[victim.layer].find(victim.expert);
            if (found == cache[victim.layer].end()) { heap.pop(); continue; }
            const auto score = expert_score(victim.layer, victim.expert);
            if (score != victim.score) {
                heap.pop(); heap.push({score, victim.layer, victim.expert, victim.participant}); continue;
            }
            if (score >= expert_score(layer, expert)) return;
            // Bound queued admissions; busy cards defer work without waiting here.
            if (pending_copies[victim.participant].size() >= 8) return;
            // Only a selected victim needs its outstanding reader to finish.
            // This also protects the pinned lookup row copied by that product.
            if (async_moe && direct_experts() && device_reduction) {
                DeviceScope scope(victim.participant ? secondary->device : primary_device);
                auto &executor = victim.participant ? secondary->decode : cached_decode;
                check(cudaEventSynchronize(executor->reader_done(victim.layer)));
            }
            heap.pop();
            auto copy = std::make_unique<Copy>();
            copy->layer = layer; copy->expert = expert; copy->participant = victim.participant;
            copy->entry = std::move(found->second); cache[victim.layer].erase(found);
            if (device_lookup_ready) {
                DeviceScope scope(victim.participant ? secondary->device : primary_device);
                auto &executor = victim.participant ? secondary->decode : cached_decode;
                executor->update_lookup(victim.layer, victim.expert, nullptr,
                                        victim.participant ? secondary->stream : stream);
            }
            const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
            const std::array<const uint8_t *, 3> source{g.data() + expert * sizes[0],
                u.data() + expert * sizes[1], d.data() + expert * sizes[2]};
            mark_promoted_slot(victim.participant, copy->entry->p, copy->entry->bytes);
            copy->ready = expert_copies[victim.participant]->copy(copy->entry->p, source, sizes);
            pending_copies[victim.participant].push_back(std::move(copy));
            ++adaptive_copies; adaptive_bytes += bytes;
            return;
        }
    }

    int decode_cache_slots_limit = 0;
    std::vector<std::vector<uint64_t>> prefill_routes;
    std::vector<std::map<int, std::unique_ptr<Device>>> decode_resident;
    std::vector<LayerState> states;
    struct SequenceState {
        std::vector<LayerState> layers;
        int position = 0, draft_position = 0;
        bool draft_primed = false;
        std::vector<float> hidden, draft_hidden;
        std::unique_ptr<Device> previous_hidden;
    };
    std::vector<SequenceState> sequences;
    int current_sequence = 0;
    std::vector<int> row_sequences, row_positions;
    int sequence_row_offset = 0;
    const bool batch_split = [] { const char *v = std::getenv("STRATA_GLM_BATCH_SPLIT"); return v && std::string(v) == "1"; }();
    int row_position(int row) const { return row_positions[row + sequence_row_offset]; }
    LayerState &row_state(int row, int layer) {
        const int slot = row_sequences[row + sequence_row_offset];
        return slot == current_sequence ? states[layer] : sequences[slot].layers[layer];
    }

    std::unique_ptr<Device> verify_history;
    std::map<Device *, size_t> history_offsets;
    size_t history_stride = 0;
    size_t history_base_bytes = 0, history_output_offset = 0;
    struct ReplayHistory { size_t base, inputs; };
    std::map<Device *, ReplayHistory> replay_history;
    const bool compact_history = [] {
        const char *value = std::getenv("STRATA_GLM_COMPACT_VERIFY_HISTORY");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: compact verify history must be 0 or 1");
    }();
    int history_slots = 0, history_position = 0, history_valid = 0;
    bool capturing_history = false;
    void capture_state(Device *buffer, int token) {
        token += history_token_offset;
        if (!capturing_history || token >= history_slots) return;
        auto it = history_offsets.find(buffer);
        if (it == history_offsets.end()) return;
        check(cudaMemcpyAsync((char *)verify_history->p + history_base_bytes + token * history_stride + it->second,
                              buffer->p, buffer->bytes, cudaMemcpyDeviceToDevice, stream));
    }
    std::map<std::string, std::unique_ptr<Device>> scratch;
    struct Weight {
        std::unique_ptr<Device> data;
        std::list<std::string>::iterator order;
    };
    std::map<std::string, Weight> weights;
    std::list<std::string> lru;
#ifdef STRATA_USE_HIP
    std::unique_ptr<Pinned> hip_weight_upload;
    void upload_hip_file(Device &destination, size_t offset, const uint8_t *source, size_t bytes) {
        DeviceScope scope(destination.owner);
        if (offset > destination.bytes || bytes > destination.bytes - offset)
            throw std::runtime_error("GLM: file upload exceeds buffer");
        // CPU reads charge file pages to this process's memory cgroup.
        // Give HIP bounded pinned staging instead of pageable GGUF mappings.
        constexpr size_t chunk = 64 * MiB;
        if (!hip_weight_upload) hip_weight_upload = std::make_unique<Pinned>(chunk);
        for (size_t at = 0; at < bytes; at += chunk) {
            const size_t count = std::min(chunk, bytes - at);
            std::memcpy(hip_weight_upload->p, source + at, count);
            check(cudaMemcpy(static_cast<uint8_t *>(destination.p) + offset + at,
                             hip_weight_upload->p, count, cudaMemcpyHostToDevice));
        }
    }
#endif
    size_t native_pinned_bytes = 0;
    std::set<std::string> native_pinned_weights;
    bool native_dense_q8 = std::getenv("STRATA_EXL3_DENSE_QUANT") != nullptr || std::getenv("STRATA_EXL3_DENSE_Q8") != nullptr;
    int native_dense_bits=[] {const char* v=std::getenv("STRATA_EXL3_DENSE_BITS");if(v && std::string(v)!="6" && std::string(v)!="8")throw std::invalid_argument("EXL3 dense bits must be 6 or 8");return v?std::stoi(v):8;}();
    bool native_cpu_dense = std::getenv("STRATA_EXL3_CPU_DENSE") != nullptr;
    struct NativeDenseWeight {std::unique_ptr<Device> data;int bits;};
    std::map<std::string,NativeDenseWeight> native_q8_weights;
    size_t native_q8_bytes = 0;
    void prepare_native_dense_q8() {
        if(!artifact.is_exl3() || !native_dense_q8 || !native_q8_weights.empty())return;
        size_t required=0;
        for(const auto& [name,t]:artifact.tensors())
            if(name!="token_embd.weight" && t.tensor->type==GGML_TYPE_BF16 && t.tensor->shape.size()==2) {
                if(t.tensor->shape[0]%32)throw std::runtime_error("EXL3 Q8: invalid row width: "+name);
                required+=t.bytes>=8*MiB?t.tensor->elements()/32*((native_dense_bits==6 && name!="output.weight")?26:34):t.bytes;
            }
        if(required>budget)throw std::runtime_error("EXL3 Q8: resident fixed matrices exceed dense cache budget; need MiB="+std::to_string((required+MiB-1)/MiB));
        const auto start=std::chrono::steady_clock::now();size_t quantized=0;
        for(const auto& [name,t]:artifact.tensors()) {
            if(name=="token_embd.weight" || t.tensor->type!=GGML_TYPE_BF16 || t.tensor->shape.size()!=2)continue;
            check_stop();const size_t in=t.tensor->shape[0],out=t.tensor->shape[1];
            if(t.bytes<8*MiB) {
                auto data=std::make_unique<Device>(t.bytes);data->put(t.data(),data->bytes);
                native_q8_bytes+=data->bytes;native_q8_weights.emplace(name,NativeDenseWeight{std::move(data),16});continue;
            }
            ++quantized;
            if(native_dense_bits==6 && name!="output.weight") {
                std::vector<strata::cpu::exl3::DenseQ6Block> packed(in*out/32);
                const auto* source=reinterpret_cast<const uint16_t*>(t.data());
                exl_pool->run([&](size_t rank){for(size_t row=out*rank/exl_pool->size();row<out*(rank+1)/exl_pool->size();++row)strata::cpu::exl3::dense_q6_row(source+row*in,packed.data()+row*(in/32),in);});
                auto data=std::make_unique<Device>(packed.size()*sizeof(packed[0]));data->put(packed.data(),data->bytes);
                native_q8_bytes+=data->bytes;native_q8_weights.emplace(name,NativeDenseWeight{std::move(data),6});continue;
            }
            std::vector<strata::cpu::exl3::DenseQ8Block> packed(in*out/32);
            const auto* source=reinterpret_cast<const uint16_t*>(t.data());
            exl_pool->run([&](size_t rank) {
                for(size_t row=out*rank/exl_pool->size();row<out*(rank+1)/exl_pool->size();++row)
                    strata::cpu::exl3::dense_q8_row(source+row*in,packed.data()+row*(in/32),in);
            });
            auto data=std::make_unique<Device>(packed.size()*sizeof(packed[0]));
            data->put(packed.data(),data->bytes);native_q8_bytes+=data->bytes;
            native_q8_weights.emplace(name,NativeDenseWeight{std::move(data),8});
        }
        resident+=native_q8_bytes;
        std::cerr<<"EXL3_DENSE_QUANT bits="<<native_dense_bits<<" tensors="<<native_q8_weights.size()<<" quantized="<<quantized<<" preserved_bf16="<<native_q8_weights.size()-quantized<<" resident_MiB="<<native_q8_bytes/double(MiB)
                 <<" prepare_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    }
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
                    name == "hc_coeff" || name == "output" || name == "moe_ids" || name == "moe_weights";
        if (gpu && !main) {
            auto &p = phase[name];
            if (!p || p->bytes < floats * 4) {
                try { p = gpu->allocate(floats * 4); }
                catch (const std::runtime_error &) {
                    throw std::runtime_error("GLM: prefill scratch exhausted by " + name + " (" + std::to_string(floats * 4 / MiB) +
                                             " MiB, arena " + std::to_string(gpu->arena->bytes / MiB) + " MiB, position " +
                                             std::to_string(position) + ", width " + std::to_string(batch_tokens) + ")");
                }
            }
            return *p;
        }
        // Verification can grow a one-token buffer after layer graphs captured
        // its address. Reserve the whole verification window before first replay.
        if (gpu && main && batch_tokens <= cpu::MAXT && (batched_resident || !artifact.is_exl3())) {
            const size_t row = name == "streams" ? 4 * m.hidden :
                               name == "hc_coeff" ? 24 : name == "output" ? m.vocab :
                               name == "moe_ids" || name == "moe_weights" ? m.top_k : m.hidden;
            floats = std::max(floats, row * cpu::MAXT);
        }
        auto &p = scratch[buffer_group ? name + "#b" : name];
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
    // REAP-pruned GGUFs spell hc_* without ".weight" and the indexer compressor as indexer.kpool_*; weights are
    // cached under the original GLM-5.3-Flash names whichever spelling a caller uses.
    static std::string canonical_tensor_name(const std::string &name) {
        if (name.find(".hc_") != std::string::npos && !name.ends_with(".weight")) return name + ".weight";
        if (const auto k = name.find("indexer.kpool_"); k != std::string::npos)
            return name.substr(0, k) + "indexer_compressor_" + name.substr(k + 14) + ".weight";
        return name;
    }
    bool floating_q8_weight(const std::string &name) const {
        if (!q8_decode) return false;
        const std::string n = canonical_tensor_name(name);
        const bool mla = n.ends_with("attn_k_b.weight") || n.ends_with("attn_v_b.weight");
        const bool hc = n.ends_with("hc_attn_fn.weight") || n.ends_with("hc_ffn_fn.weight");
        if (!mla && !hc) return false;
        const auto &t = *artifact.at(name).tensor;
        return t.type == GGML_TYPE_Q8_0 || (floating_q8_repack >= (mla ? 1 : 2) && t.shape[0] % 32 == 0);
    }
    size_t floating_q8_bytes(const std::string &name) const {
        const auto &t = *artifact.at(name).tensor;
        size_t rows = 1;
        for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i];
        return rows * ggml_row_size(GGML_TYPE_Q8_0, t.shape[0]);
    }
    int dense_weight_type(const std::string &name) const {
        const auto &t = *artifact.at(name).tensor;
        if (!dense_q4_repack || t.shape.size() != 2 ||
            (t.type != GGML_TYPE_Q5_K && t.type != GGML_TYPE_Q6_K &&
             !(dense_q4_repack == 5 && t.type == GGML_TYPE_Q4_K)) ||
            !name.starts_with("blk.") || name.starts_with("blk.45.") || name.ends_with("_exps.weight") ||
            t.shape[0] % 256 || t.shape[0] * t.shape[1] < 4 * 1024 * 1024) return t.type;
        const int layer = std::stoi(name.substr(4, name.find('.', 4) - 4));
        if ((dense_q4_repack == 3 && t.type != GGML_TYPE_Q5_K) ||
            (dense_q4_repack == 4 && !name.ends_with("attn_output.weight"))) return t.type;
        const bool mixer = name.find(".attn_") != std::string::npos;
        const bool shared = name.ends_with("_shexp.weight");
        return layer >= 3 && (mixer || (dense_q4_repack == 2 && shared))
            ? int(dense_q4_repack == 5 ? GGML_TYPE_Q3_K : GGML_TYPE_Q4_K) : int(t.type);
    }
    size_t dense_weight_bytes(const std::string &name) const {
        const auto &t = artifact.at(name);
        return dense_weight_type(name) == int(t.tensor->type) ? t.bytes
             : (size_t)t.tensor->shape[1] * ggml_row_size((ggml_type)dense_weight_type(name), t.tensor->shape[0]);
    }
    Device &weight(const std::string &name, bool floating = false) {
        const std::string key = canonical_tensor_name(name) + (floating ? ":fp32" : ":raw");
        auto it = weights.find(key);
        if (it != weights.end()) {
            lru.splice(lru.begin(), lru, it->second.order);
            return *it->second.data;
        }
        if (floating && gpu && floating_q8_weight(name)) {
            // Q8 decode keeps only the Q8_0 copy resident; FP32 users (prefill GEMMs) get a scratch expansion.
            const auto raw_key = canonical_tensor_name(name) + ":raw";
            if (!weights.count(raw_key) && name.starts_with("blk.45.")) weight(name, false);
            auto raw = weights.find(raw_key);
            if (raw != weights.end()) {
                size_t values = 1;
                for (auto d : artifact.at(name).tensor->shape) values *= d;
                auto &scratch = q8_fp32_scratch[q8_scratch_role(name)];
                if (!scratch || scratch->bytes < values * 4) {
                    check(cudaStreamSynchronize(stream));   // earlier GEMMs may still read the smaller buffer
                    scratch = std::make_unique<Device>(values * 4);
                }
                k::glm_q8_dequant(raw->second.data->p, scratch->f(), (long long)values, stream);
                return *scratch;
            }
        }
        if (gpu && !artifact.is_exl3() && !name.starts_with("blk.45."))
            throw std::runtime_error("GLM: fixed GPU weight missing: " + key);
        const auto &t = artifact.at(name);
        std::vector<float> values;
        std::vector<uint8_t> repacked;
        const int repack_type = floating_q8_weight(name) ? GGML_TYPE_Q8_0 : dense_weight_type(name);
        if (!floating && repack_type != int(t.tensor->type)) {
            const auto &source = *t.tensor;
            const size_t width = source.shape[0], row_bytes = ggml_row_size((ggml_type)repack_type, width);
            const size_t rows = (repack_type == GGML_TYPE_Q8_0 ? floating_q8_bytes(name) : dense_weight_bytes(name)) / row_bytes;
            const auto *traits = ggml_get_type_traits((ggml_type)source.type);
            if (source.type != GGML_TYPE_F32 && (!traits || !traits->to_float))
                throw std::runtime_error("GLM: no floating Q8 source dequantizer: " + name);
            repacked.resize(rows * row_bytes);
            std::atomic<bool> size_failed{false};
            auto convert = [&](size_t first, size_t last) {
                struct Affinity {
#ifdef __linux__
                    cpu_set_t previous; bool changed = false;
                    explicit Affinity(bool enabled) {
                        if (enabled && !sched_getaffinity(0, sizeof(previous), &previous))
                            changed = !sched_setaffinity(0, sizeof(allocation_cpus), &allocation_cpus);
                    }
                    ~Affinity() { if (changed) sched_setaffinity(0, sizeof(previous), &previous); }
#else
                    explicit Affinity(bool) {}
#endif
                } affinity(repack_type == GGML_TYPE_Q4_K);
                std::vector<float> row(width);
                for (size_t r = first; r < last; ++r) {
                    const auto *src = t.data() + r * (t.bytes / rows);
                    if (source.type == GGML_TYPE_F32) std::memcpy(row.data(), src, width * sizeof(float));
                    else traits->to_float(src, row.data(), width);
                    if (ggml_quantize_chunk((ggml_type)repack_type, row.data(), repacked.data() + r * row_bytes,
                                           0, 1, width, nullptr) != row_bytes) size_failed = true;
                }
            };
            // Large K-quant matrices are converted only at startup. Disjoint
            // row ranges keep quantization deterministic and bound host scratch.
            const size_t workers = repack_type == GGML_TYPE_Q4_K ? std::min<size_t>(8, std::max(1u, std::thread::hardware_concurrency())) : 1;
            std::vector<std::thread> converters;
            for (size_t i = 1; i < workers; ++i)
                converters.emplace_back([&, i] { convert(rows * i / workers, rows * (i + 1) / workers); });
            convert(0, rows / workers);
            for (auto &worker : converters) worker.join();
            if (size_failed) throw std::runtime_error("GLM: dense repack row size mismatch: " + name);
            std::cerr << (repack_type == GGML_TYPE_Q8_0 ? "FLOATING_Q8" :
                         repack_type == GGML_TYPE_Q3_K ? "DENSE_Q3" : "DENSE_Q4")
                      << " tensor=" << name << " source_type=" << source.type
                      << " MiB=" << repacked.size() / double(MiB) << '\n';
        }
        if (floating && t.tensor->type != 0)
            values = dequant(t);
        if (name.ends_with(".ssm_a") && t.tensor->type == GGML_TYPE_F32) {
            // The KDA kernels take A as stored by the GLM-5.3-Flash GGUF (negative). Some converters (the REAP-50
            // GGUF) store it positive; every other shared tensor of those files is bit-identical.
            values.assign((const float *)t.data(), (const float *)t.data() + t.bytes / 4);
            if (std::all_of(values.begin(), values.end(), [](float v) { return v > 0; }))
                for (auto &v : values) v = -v;
        }
        const size_t bytes = !repacked.empty() ? repacked.size() : values.empty() ? t.bytes : values.size() * 4;
        if (bytes > budget)
            throw std::runtime_error("GLM: weight exceeds dense cache budget: " + name);
        while (resident + bytes > budget && !lru.empty()) {
            // Previous kernels may still hold a pointer to an evicted weight.
            check(cudaStreamSynchronize(stream));
            auto victim = std::prev(lru.end());
            while (native_pinned_weights.count(*victim)) {
                if (victim == lru.begin()) throw std::runtime_error("GLM: no evictable dense weight");
                --victim;
            }
            const auto old = *victim;
            resident -= weights.at(old).data->bytes;
            weights.erase(old);
            lru.erase(victim);
        }
        DeviceTag tag(name.starts_with("blk.45.") ? "dense_mtp" : floating ? "dense_fp32" : "dense");
        auto data = std::make_unique<Device>(bytes);
#ifdef STRATA_USE_HIP
        if (values.empty() && repacked.empty() && !artifact.is_exl3()) {
            upload_hip_file(*data, 0, t.data(), bytes);
        } else
#endif
        data->put(!repacked.empty() ? (const void *)repacked.data() : values.empty() ? (const void *)t.data() : values.data(), bytes);
        resident += bytes;
        // A cyclic layer trace defeats plain LRU. Retain a bounded fixed subset,
        // leaving half the cache for streamed matrices and simultaneous operands.
        if (artifact.is_exl3() && !std::getenv("STRATA_EXL3_LRU") &&
            native_pinned_bytes + bytes <= std::min((budget-native_q8_bytes)/2,
                budget-native_q8_bytes>artifact.at("output.weight").bytes?budget-native_q8_bytes-artifact.at("output.weight").bytes:0)) {
            native_pinned_weights.insert(key); native_pinned_bytes += bytes;
        }
        lru.push_front(key);
        auto [inserted, ok] = weights.emplace(key, Weight{std::move(data), lru.begin()});
        (void)ok;
        return *inserted->second.data;
    }
    void release_gpu_dense_host_pages(bool draft = false) {
        static const bool enabled = [] {
            const char *value = std::getenv("STRATA_GLM_RELEASE_GPU_DENSE_HOST");
            if (!value || std::string(value) == "0") return false;
            if (std::string(value) == "1") return true;
            throw std::invalid_argument("GLM: release GPU dense host must be 0 or 1");
        }();
        if (!enabled) return;
        if (!gpu || artifact.is_exl3() || direct_upload_enabled())
            throw std::invalid_argument("GLM: GPU dense host release requires ordinary GGUF GPU prefill");
        // Fixed GPU weights have persistent owned copies. Their clean source
        // pages otherwise compete with CPU experts in a memory-limited cgroup.
        // Embeddings are still read on the CPU and routed experts are not
        // fixed; neither is released here. The mapping remains valid and any
        // later source read can fault the same bytes back in.
        check(cudaStreamSynchronize(stream));
        const auto start = std::chrono::steady_clock::now();
        size_t bytes = 0, tensors = 0;
        for (const auto &[name, tensor] : artifact.tensors()) {
            if (name == "token_embd.weight" || name.ends_with("_exps.weight") ||
                name.starts_with("blk.45.") != draft) continue;
            const auto key = canonical_tensor_name(name);
            if (!weights.count(key + ":raw") && !weights.count(key + ":fp32")) continue;
            tensor.file->discard_tensor_pages(*tensor.tensor, tensor.bytes);
            bytes += tensor.bytes;
            ++tensors;
        }
        std::cerr << "GPU_DENSE_HOST_RELEASE phase=" << (draft ? "draft" : "main")
                  << " tensors=" << tensors << " source_MiB_advised=" << bytes / double(MiB)
                  << " ms=" << std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start).count() << '\n';
    }
    void mat(const std::string &name, const float *x, float *y, bool floating = false, int nt = 0) {
        if (!nt)
            nt = batch_tokens;
        const auto &t = *artifact.at(name).tensor;
        const int weight_type = dense_weight_type(name);
        if (prepared_projections.contains(name)) {
            const size_t width = t.shape[1];
            check(cudaMemcpyAsync(y, window_projections.at(name)->f((size_t)projection_first * width),
                                  (size_t)nt * width * 4, cudaMemcpyDeviceToDevice, stream));
            return;
        }

        if (t.shape.size() != 2)
            throw std::runtime_error("GLM: matrix must have two dimensions: " + name);
        const int in = (int)t.shape[0], out = (int)t.shape[1];
        if(artifact.is_exl3() && t.type==GGML_TYPE_BF16) {
            if(native_dense_q8) {
                const auto it=native_q8_weights.find(name);
                if(it==native_q8_weights.end())throw std::runtime_error("EXL3 Q8: fixed matrix not prepared: "+name);
                const auto& W=*it->second.data;
                if(it->second.bits==16) {
                    if(nt==1)k::exl3_bf16_gemv((const uint16_t*)W.p,x,y,in,out,stream);
                    else {
                        const float one=1,zero=0;const int panel_rows=512;
                        auto& panel=buf("bf16_dense_panel",size_t(panel_rows)*in);
                        for(int row=0;row<out;row+=panel_rows) {
                            int rows=std::min(panel_rows,out-row);
                            k::exl3_bf16_to_float((const uint16_t*)W.p+size_t(row)*in,panel.f(),rows*in,stream);
                            check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,nt,in,&one,panel.f(),in,x,in,&zero,y+row,out));
                        }
                    }
                    check(cudaGetLastError());return;
                }
                const bool q6=it->second.bits==6;
                if(nt==1) {
                    if(q6)k::exl3_dense_q6_gemv(W.p,x,y,in,out,stream);else k::exl3_dense_q8_gemv(W.p,x,y,in,out,stream);
                } else {
                    const float one=1,zero=0;const int panel_rows=512;
                    auto& panel=buf("q8_dense_panel",size_t(panel_rows)*in);
                    for(int row=0;row<out;row+=panel_rows) {
                        int rows=std::min(panel_rows,out-row);
                        const auto* packed=(const uint8_t*)W.p+size_t(row)*(in/32)*(q6?26:34);
                        if(q6)k::exl3_dense_q6_to_float(packed,panel.f(),rows*in,stream);else k::exl3_dense_q8_to_float(packed,panel.f(),rows*in,stream);
                        check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,nt,in,&one,panel.f(),in,x,in,&zero,y+row,out));
                    }
                }
                check(cudaGetLastError());return;
            }
            if(nt==1 && !fast && native_cpu_dense) {
                std::vector<float> host_x(in),host_y(out);
                check(cudaMemcpyAsync(host_x.data(),x,size_t(in)*4,cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                const auto* rows=reinterpret_cast<const uint16_t*>(artifact.at(name).data());
                exl_pool->run([&](size_t rank) {
                    size_t first=size_t(out)*rank/exl_pool->size(),last=size_t(out)*(rank+1)/exl_pool->size();
                    for(size_t row=first;row<last;++row)
                        host_y[row]=strata::cpu::exl3::bf16_dot(rows+row*in,host_x.data(),in);
                });
                check(cudaMemcpyAsync(y,host_y.data(),size_t(out)*4,cudaMemcpyHostToDevice,stream));
                check(cudaStreamSynchronize(stream));
                return;
            }
            auto& W=weight(name,false);const float one=1,zero=0;
            if(nt==1)k::exl3_bf16_gemv((const uint16_t*)W.p,x,y,in,out,stream);
            else {
                const int panel_rows=512;auto& panel=buf("bf16_dense_panel",size_t(panel_rows)*in);
                for(int row=0;row<out;row+=panel_rows) {
                    int rows=std::min(panel_rows,out-row);
                    k::exl3_bf16_to_float((const uint16_t*)W.p+size_t(row)*in,panel.f(),rows*in,stream);
                    check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,nt,in,&one,panel.f(),in,x,in,&zero,y+row,out));
                }
            }
            check(cudaGetLastError());return;
        }
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
                if (dequant_once) {
                    gpu->gemm->native_chunked((uint16_t *)half.p, weight_type, W.p, y, nt, out, in, 2048);
                    return;
                }
                for (int t0 = 0; t0 < nt; t0 += 2048)
                    gpu->gemm->native((uint16_t *)half.p + (size_t)t0 * in, weight_type, W.p, y + (size_t)t0 * out,
                                     std::min(2048, nt - t0), out, in);
            }
            return;
        }
        if (t.type == 0 || floating) {
            if (q8_decode && nt <= 8 && in % 4 == 0 && ((uintptr_t)W.p & 15) == 0 && ((uintptr_t)x & 15) == 0) {
                k::glm_f32_rows_gemv(W.f(), x, y, out, in, nt, stream);
                return;
            }
            for (int t = 0; t < nt; ++t)
                check(cublasSgemv(blas, CUBLAS_OP_T, in, out, &one, W.f(), in, x + t * in, 1, &zero,
                                  y + t * out, 1));
            return;
        }
        if (!k::native_mmvq_supported(weight_type))
            throw std::runtime_error("GLM: unsupported dense quantization: " + name);
        if (shared_quant && x == shared_quant_src && in == shared_quant_in && nt == shared_quant_nt) {
            k::native_mmvq(weight_type, W.p, shared_quant->p, y, in, out, nt, stream);
            return;
        }
        auto &q = buf("q8", k::native_q8_1_bytes(in, nt) / 4);
        k::native_quantize_q8_1(x, q.p, in, nt, stream);
        k::native_mmvq(weight_type, W.p, q.p, y, in, out, nt, stream);
    }
    void norm(const std::string &name, const float *x, float *y, int width) {
        auto &W = weight(name);
        if (norm_q8_fused && q8_decode && quant_once && !fast && batch_tokens == 1 &&
            width == m.hidden && norm_quant) {
            k::glm_rms_norm_rows_q8(x, W.f(), y, norm_quant->p, width, 1, m.rms_epsilon, stream);
            shared_quant = norm_quant.get(); shared_quant_src = y;
            shared_quant_in = width; shared_quant_nt = 1;
            return;
        }
        if (norm_q8_fused && width == m.hidden) end_share_quant();
        // The 1024-thread rows kernel sums in a different order, so it is part of the opt-in decode kernels and
        // used only outside prefill: every decode and verify width then rounds alike.
        if (q8_decode && !fast) k::glm_rms_norm_rows(x, W.f(), y, width, batch_tokens, m.rms_epsilon, stream);
        else k::glm_rms_norm(x, W.f(), y, width, batch_tokens, m.rms_epsilon, stream);
    }
    void hc_read(const std::string &p, const std::string &kind, const float *r, float *x, float *c) {
        auto &normalized = buf("hc_normalized", m.hidden * 4 * batch_tokens);
        auto &projected = buf("hc_projected", 24 * batch_tokens);
        // STRATA_GLM_HC_FUSED=1 (opt-in, decode of one token with the Q8 projection): norm, projection, Sinkhorn
        // coefficients and the read in two launches instead of five; the norm scale is applied after the projection,
        // so rounding differs from the unfused path.
        static const bool hc_fused = [] { const char *v = std::getenv("STRATA_GLM_HC_FUSED"); return v && v[0] == '1'; }();
        if (hc_fused && !fast && batch_tokens == 1 && floating_q8_weight(p + "hc_" + kind + "_fn.weight")) {
            auto &fb = weight(p + "hc_" + kind + "_base.weight");
            auto &fs = weight(p + "hc_" + kind + "_scale.weight");
            auto &qw = weight(p + "hc_" + kind + "_fn.weight");
            auto &partial = buf("hc_fused_partial", 25 * 32);
            k::glm_hc_read_fused(qw.p, r, fb.f(), fs.f(), c, x, partial.f(), m.hidden, m.sinkhorn_iterations,
                                 m.hc_epsilon, m.rms_epsilon, stream);
            return;
        }
        // STRATA_GLM_HC_NORM_ROWS=1 (opt-in): the mHC norm over 4 x hidden values takes the 1024-thread rows kernel
        // under the same rule as norm() (decode kernels only, never prefill); one 256-thread block per row is
        // latency-bound, most of all on HIP (about 18 us per launch on the RX 9070 XT).
        static const bool hc_rows = [] { const char *v = std::getenv("STRATA_GLM_HC_NORM_ROWS"); return v && v[0] == '1'; }();
        if (hc_rows && q8_decode && !fast)
            k::glm_rms_norm_rows(r, nullptr, normalized.f(), m.hidden * 4, batch_tokens, m.rms_epsilon, stream);
        else k::glm_rms_norm(r, nullptr, normalized.f(), m.hidden * 4, batch_tokens, m.rms_epsilon, stream);
        const auto &hc_name = p + "hc_" + kind + "_fn.weight";
        if (!fast && floating_q8_weight(hc_name)) {
            auto &partial = buf("hc_q8_partial", 24 * cpu::MAXT * 32);
            k::glm_q8_rows_gemv(weight(hc_name).p, normalized.f(), projected.f(), partial.f(), 24, m.hidden * 4,
                                batch_tokens, stream);
        } else if (split_hc_projection && !fast && batch_tokens == 1) {
            auto &w = weight(p + "hc_" + kind + "_fn.weight", true);
            auto &scratch = buf("hc_partial", 24 * 32);
            k::glm_hc_project(normalized.f(), w.f(), projected.f(), scratch.f(), m.hidden * 4, stream);
        } else {
            mat(p + "hc_" + kind + "_fn.weight", normalized.f(), projected.f(), true);
        }
        // Each upload can evict previous cache entries, so acquire scalar tensors first,
        // then enqueue immediately. These small tensors together fit every legal budget.
        auto &base = weight(p + "hc_" + kind + "_base.weight");
        auto &scale = weight(p + "hc_" + kind + "_scale.weight");
        if (fast) {
            k::glm_mhc_read_batch(r, projected.f(), base.f(), scale.f(), c, x, m.hidden,
                                  m.sinkhorn_iterations, m.hc_epsilon, batch_tokens, stream);
            return;
        }
        static const bool serial_mhc = std::getenv("STRATA_GLM_SERIAL_MHC") != nullptr;
        if (batch_tokens > 1 && !serial_mhc) {
            k::glm_mhc_read_tokens(r, projected.f(), base.f(), scale.f(), c, x, m.hidden, m.sinkhorn_iterations,
                                   m.hc_epsilon, batch_tokens, stream);
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
        share_quant(x, m.hidden);
        mat(p + "ffn_gate" + suffix + ".weight", x, gate.f());
        mat(p + "ffn_up" + suffix + ".weight", x, up.f());
        end_share_quant();
        k::glm_swiglu(gate.f(), up.f(), hidden.f(), ff * batch_tokens, limit, stream);
        mat(p + "ffn_down" + suffix + ".weight", hidden.f(), out);
    }
    void expert_groups(const std::string &p, const strata::core::LayerDescriptor &layer,
                       GpuPrefill &target, PrefillGroupCache &cache, const float *x,
                       const int *bounds, const int *dest, const int *source, const int *hb,
                       float *result, cudaStream_t work_stream, int participant, int participants,
                       const std::function<Device &(const std::string &, size_t)> &workspace,
                       const std::function<void(int)> &group_done = {}, const uint16_t *x_half = nullptr) {
        const int nt = batch_tokens, H = m.hidden, F = layer.intermediate, E = m.experts;
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        const size_t gh = G.bytes / E, stride = prefill_expert_stride(G, U, D);
        const int layer_id = std::stoi(p.substr(4));
        std::array<int, 289> recent_bounds{};
        const int *admission = hb;
        if (decode_cache_extend && layer_id < (int)prefill_routes.size()) {
            for (int e = 0; e < E; ++e)
                recent_bounds[e + 1] = recent_bounds[e] + prefill_routes[layer_id][e];
            admission = recent_bounds.data();
        }
        cache.select(layer_id, admission, 16 * stride, participant, participants, primary_prefill_groups);
        const int max_rows = std::min(4096, nt * (int)m.top_k);
        auto &gu = workspace("gu", (size_t)max_rows * 2 * F);
        auto &hidden = workspace("hidden", (size_t)max_rows * F);
        auto &qx = workspace("qx", (mmq::q8_bytes(max_rows, H) + 3) / 4);
        auto &qh = workspace("qh", (mmq::q8_bytes(max_rows, F) + 3) / 4);
        auto &local = workspace("local", 17);
        auto &identity = workspace("identity", max_rows + 128);
        mmq::iota((int *)identity.p, max_rows + 128, work_stream);
        for (int first = 0; first < E; first += 16) {
            if (!strata::prefill::owns_expert_group(first / 16, participant, participants, primary_prefill_groups)) continue;
            check_stop();
            const int n = std::min(16, E - first), begin = hb[first], end = hb[first + n];
            if (begin == end) continue;
            const int slot = target.groups % 2;
            const int fetched = n == 16 ? target.take(layer_id, first / 16) : -1;
            void *weights = fetched >= 0 ? cache.adopt(layer_id, first, target, fetched, stride, work_stream)
                                         : cache.weights(layer_id, first, n, target, slot, G, U, D, work_stream,
                                                         lazy_experts ? hb : nullptr);
            const bool pooled = fetched >= 0 && weights == target.pool[fetched]->p;
            const bool release_after_products = pooled ||
                (fetched < 0 && (!cache.early_ring_release || weights == target.slots[slot]->p));
            const cudaEvent_t released = pooled ? target.pool_done[fetched] : target.done[slot];
            if (tensor_prefill_experts) {
                auto &tx = workspace("tensor_x", ((size_t)max_rows * H + 1) / 2);
                auto &th = workspace("tensor_h", ((size_t)max_rows * F + 1) / 2);
                auto &td = workspace("tensor_down", (size_t)max_rows * H);
                if (tensor_batched_experts) {
                    target.tensor_rows_actual += end - begin;
                    std::vector<unsigned> masks{0};
                    if (tensor_bucket_experts) {
                        std::map<int, unsigned> buckets;
                        for (int e = 0; e < n; ++e) {
                            const int rows = hb[first + e + 1] - hb[first + e];
                            if (!rows) continue;
                            const int bucket = tensor_bucket_step ? (rows + tensor_bucket_step - 1) / tensor_bucket_step
                                : std::max(5, int(std::bit_width(unsigned(rows - 1))));
                            buckets[bucket] |= 1u << e;
                        }
                        masks.clear();
                        for (const auto &[bucket, mask] : buckets) masks.push_back(mask);
                    }
                    auto *dense = (uint16_t *)target.dq->p;
                    for (unsigned mask : masks) {
                        const int batches = mask ? std::popcount(mask) : n;
                        int most = 0;
                        for (int e = 0; e < n; ++e)
                            if (!mask || (mask & (1u << e))) most = std::max(most, hb[first + e + 1] - hb[first + e]);
                        const int tile = tensor_bucket_experts ? std::max(1, max_rows / batches)
                                                               : std::min(256, std::max(1, max_rows / batches));
                        for (int offset = 0; offset < most; offset += tile) {
                            const int rows = std::min(tile, most - offset), all_rows = rows * batches;
                            target.tensor_rows_padded += all_rows;
                            target.tensor_dequant_values += (uint64_t)batches * 3 * F * H;
                            if (x_half)
                                k::glm_gather_expert_f16_from_half(x_half, (uint16_t *)tx.p, bounds, source,
                                                                   first, batches, offset, rows, H, work_stream, mask);
                            else k::glm_gather_expert_f16(x, (uint16_t *)tx.p, bounds, source,
                                                    first, batches, offset, rows, H, work_stream, mask);
                            int packed = 0;
                            for (int e = 0; e < n; ++e) if (!mask || (mask & (1u << e))) {
                                k::dequant_f16(G.tensor->type, (char *)weights + e * stride, 0, 2 * F, H,
                                               dense + (size_t)packed * 2 * F * H, work_stream);
                                ++packed;
                            }
                            target.gemm->f16_batched((uint16_t *)tx.p, dense, gu.f(), rows, 2 * F, H, batches);
                            mmq::swiglu(gu.f(), hidden.f(), all_rows, F, false, work_stream, layer.swiglu_limit);
                            k::glm_f16(hidden.f(), (uint16_t *)th.p, (int64_t)all_rows * F, work_stream);
                            packed = 0;
                            for (int e = 0; e < n; ++e) if (!mask || (mask & (1u << e))) {
                                k::dequant_f16(D.tensor->type, (char *)weights + e * stride + 2 * gh, 0, H, F,
                                               dense + (size_t)packed * H * F, work_stream);
                                ++packed;
                            }
                            target.gemm->f16_batched((uint16_t *)th.p, dense, td.f(), rows, H, F, batches);
                            k::glm_scatter_expert_rows(td.f(), result, bounds, dest,
                                                      first, batches, offset, rows, H, work_stream, mask);
                        }
                    }
                    if (release_after_products) check(cudaEventRecord(released, work_stream));
                    if (group_done) group_done(first);
                    continue;
                }
                for (int e = first; e < first + n; ++e) {
                    const auto *w = (const char *)weights + (e - first) * stride;
                    for (int offset = hb[e]; offset < hb[e + 1]; offset += max_rows) {
                        const int rows = std::min(max_rows, hb[e + 1] - offset);
                        k::glm_gather_f16(x, (uint16_t *)tx.p, source, offset, rows, H, work_stream);
                        target.gemm->native((uint16_t *)tx.p, G.tensor->type, w, gu.f(), rows, 2 * F, H);
                        mmq::swiglu(gu.f(), hidden.f(), rows, F, false, work_stream, layer.swiglu_limit);
                        k::glm_f16(hidden.f(), (uint16_t *)th.p, (int64_t)rows * F, work_stream);
                        target.gemm->native((uint16_t *)th.p, D.tensor->type, w + 2 * gh, td.f(), rows, H, F);
                        k::glm_scatter_rows(td.f(), result, dest, offset, rows, H, work_stream);
                    }
                }
                if (release_after_products) check(cudaEventRecord(released, work_stream));
                if (group_done) group_done(first);
                continue;
            }
            for (int offset = begin; offset < end; offset += max_rows) {
                check_stop();
                const int rows = std::min(max_rows, end - offset);
                int most = 0;
                for (int e = first; e < first + n; ++e)
                    most = std::max(most, std::max(0, std::min(offset + rows, hb[e + 1]) - std::max(offset, hb[e])));
                k::glm_group_bounds(bounds, (int *)local.p, first, n, offset, rows, work_stream);
                mmq::quantize(x, source + offset, qx.p, G.tensor->type, H, H, rows, work_stream);
                mmq::Product gate{weights, (int)G.tensor->type, 2 * F, H, stride, n, qx.p,
                                  (int *)local.p, (int *)identity.p, rows, most, gu.f(), 2 * F};
                target.context->run(gate, work_stream);
                mmq::swiglu(gu.f(), hidden.f(), rows, F, false, work_stream, layer.swiglu_limit);
                mmq::quantize(hidden.f(), nullptr, qh.p, D.tensor->type, F, F, rows, work_stream);
                mmq::Product down{(char *)weights + 2 * gh, (int)D.tensor->type, H, F, stride, n,
                                  qh.p, (int *)local.p, dest + offset, rows, most, result, H};
                target.context->run(down, work_stream);
            }
            if (release_after_products) check(cudaEventRecord(released, work_stream));
            if (group_done) group_done(first);
        }
    }
    // Queue this layer's leading non-resident groups on the copy streams before its mixer.
    void finish_prefetch() {
        if (prefetch_pending.valid()) prefetch_pending.get();
    }
    void prefetch_groups_sync(int layer_id) {
        if (!fast || !gpu || gpu->pool.empty() || m.experts != 288) return;
        const auto p = "blk." + std::to_string(layer_id) + ".";
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        const size_t stride = prefill_expert_stride(G, U, D);
        const bool split = secondary && secondary_enabled;
        for (int participant = 0; participant < (split ? 2 : 1); ++participant) {
            DeviceScope scope(participant ? secondary->device : primary_device);
            auto &target = participant ? *secondary->gpu : *gpu;
            auto &cache = participant ? secondary->cache : prefill_cache;
            target.fetched.clear();
            const auto found = cache.layers.find(layer_id);
            int index = 0;
            for (int group = 0; group < prefill_expert_groups && index < (int)target.pool.size(); ++group) {
                if (!strata::prefill::owns_expert_group(group, participant, split ? 2 : 1, primary_prefill_groups)) continue;
                if (found != cache.layers.end() && found->second.entries.count(group)) continue;
                target.prefetch(G, U, D, layer_id, group, index++, stride);
            }
        }
    }
    void prefetch_groups(int layer_id) {
        finish_prefetch();
        if (!fast || !gpu || gpu->pool.empty() || m.experts != 288) return;
        if (!gpu->staged_prefetch) { prefetch_groups_sync(layer_id); return; }
        if (!prefetch_launcher) prefetch_launcher = std::make_unique<GpuWorker>(primary_device);
        // Enqueue the mixer on the main thread while the staging worker fills
        // the extra pool and queues its H2D copies. Join before consuming the map.
        prefetch_pending = prefetch_launcher->submit([this, layer_id] { prefetch_groups_sync(layer_id); });
    }
    void moe_gpu(const std::string &p, const float *x, float *out,
                 const strata::core::LayerDescriptor &layer) {
        const int nt = batch_tokens, H = m.hidden, E = m.experts, K = m.top_k;
        auto &logits = buf("router_logits", nt * E);
        auto &ids = buf("router_ids", nt * K), &rw = buf("router_weights", nt * K);
        auto &bounds = buf("router_bounds", E + 1), &dest = buf("router_dest", nt * K + 128);
        auto &source = buf("router_source", nt * K + 128);
        if(artifact.has_expert_pack()) {
            check(cudaMemsetAsync(static_cast<int*>(dest.p)+nt*K,0,128*sizeof(int),stream));
            check(cudaMemsetAsync(static_cast<int*>(source.p)+nt*K,0,128*sizeof(int),stream));
        }
        auto &cursor = buf("router_cursor", stable_routes ? E * ((nt * K + 255) / 256 + 1) : E);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        if (std::getenv("STRATA_GLM_CHECK_ROUTER")) {
            check(cudaStreamSynchronize(stream));
            const auto values = logits.floats((size_t)nt * E);
            for (size_t i = 0; i < values.size(); ++i)
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("GLM: non-finite router logit layer=" + p +
                        " token=" + std::to_string(position + i / E) +
                        " expert=" + std::to_string(i % E));
        }
        const float *prefill_bonus = route_bonus_row(std::stoi(p.substr(4)));
        if (prefill_bonus && ram_tier) {
            // The memory tier learns the prompt's unbiased routes; the batch itself runs on the biased ones.
            auto &want = buf("router_wanted", nt * K), &want_w = buf("router_wanted_weights", nt * K);
            k::glm_route_batch(logits.f(), bias.f(), (int *)want.p, want_w.f(), E, K, m.expert_scale, nt, stream);
            prefill_wanted.resize((size_t)nt * K);
            check(cudaMemcpyAsync(prefill_wanted.data(), want.p, prefill_wanted.size() * 4, cudaMemcpyDeviceToHost, stream));
        } else prefill_wanted.clear();
        k::glm_route_batch(logits.f(), bias.f(), (int *)ids.p, rw.f(), E, K, m.expert_scale, nt, stream, prefill_bonus);
        if (stable_routes)
            k::glm_group_routes_stable((int *)ids.p, (int *)bounds.p, (int *)dest.p, (int *)source.p,
                                      (int *)cursor.p, E, K, nt, stream);
        else k::glm_group_routes((int *)ids.p, (int *)bounds.p, (int *)dest.p, (int *)source.p,
                                 (int *)cursor.p, E, K, nt, stream);
        check(cudaMemcpyAsync(gpu->counts->p, bounds.p, (E + 1) * 4, cudaMemcpyDeviceToHost, stream));
        const bool recent_routes = (decode_cache_budget || decode_cache_auto) &&
                                   (decode_cache_auto || fixed_cache_recent) && decode_cache_window;
        if (early_route_sync) {
            // The routing table reaches the host before the shared expert is queued; both devices start on it.
            if (recent_routes) {
                const int recent = std::min(nt, decode_cache_window);
                route_tail.resize(recent * K);
                check(cudaMemcpyAsync(route_tail.data(), (int *)ids.p + (nt - recent) * K,
                                      route_tail.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
            }
            check(cudaEventRecord(moe_ready, stream));
            ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
            check(cudaEventSynchronize(moe_ready));
        } else {
            ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
            check(cudaStreamSynchronize(stream));
        }
        finish_prefetch();
        const int *hb = (int *)gpu->counts->p;
        if ((decode_cache_budget || decode_cache_auto) && std::stoi(p.substr(4)) < (int)prefill_routes.size()) {
            const int layer_id = std::stoi(p.substr(4));
            auto &frequency = prefill_routes.at(layer_id);
            if (recent_routes) {
                const int recent = std::min(nt, decode_cache_window);
                std::vector<int> tail(recent * K);
                if (early_route_sync) tail = route_tail;
                else check(cudaMemcpy(tail.data(), (int *)ids.p + (nt - recent) * K,
                                 tail.size() * sizeof(int), cudaMemcpyDeviceToHost));
                auto &history = prefill_recent_ids.at(layer_id);
                const size_t keep = std::min(history.size(), size_t(decode_cache_window - recent) * K);
                history.erase(history.begin(), history.end() - keep);
                history.insert(history.end(), tail.begin(), tail.end());
                std::fill(frequency.begin(), frequency.end(), 0);
                for (int expert : history) ++frequency[expert];
            } else {
                for (int e = 0; e < E; ++e) frequency[e] += hb[e + 1] - hb[e];
            }
        }
        if (ram_tier) {
            // The prompt's routes keep their experts in memory through the batch and warm for the decode.
            const int layer_id = std::stoi(p.substr(4));
            if (ram_tier->expert_bytes[layer_id]) {
                for (int e = 0; e < E; ++e)
                    if (hb[e + 1] != hb[e]) ram_used[(size_t)layer_id * E + e] = 1;
                for (int e : prefill_wanted) {   // synchronized with the routing table above
                    ram_tier->score[(size_t)layer_id * E + e] += 1.f;
                    ram_tier->window[(size_t)layer_id * E + e] += 1.f;
                }
            }
        }
        auto &result = buf("moe_results", (size_t)nt * K * H);
        const bool split = secondary && secondary_enabled;
        const bool partial = split && partial_return && secondary->peer;
        const bool half_x = split && half_activations && secondary->peer && tensor_prefill_experts && tensor_batched_experts;
        uint16_t *x_half = nullptr;
        if (half_x) {
            x_half = (uint16_t *)buf("moe_x_half", ((size_t)nt * H + 1) / 2).p;
            k::glm_f16(x, x_half, (int64_t)nt * H, stream);
            // Queued after the routing sync: the secondary must wait for this conversion explicitly.
            check(cudaEventRecord(mla_ready, stream));
        }
        const bool returned = split && async_peer_return && secondary->peer && !partial;
        float *received_rows = returned ? buf("moe_peer_results", (size_t)nt * K * H).f() : nullptr;
        float *received_partial = partial ? buf("moe_partial", (size_t)nt * H).f() : nullptr;
        std::future<void> pending;
        if (split) {
            // The tiny routing table remains immutable until pending.get().
            pending = secondary->worker->submit([&, nt] {
                auto &other = *secondary;
                auto upload = [&](void *dst, const void *src, size_t bytes) {
                    if (other.peer) {
                        check(cudaMemcpyPeerAsync(dst, other.device, src, primary_device, bytes, other.stream));
                    } else {
                        { DeviceScope scope(primary_device);
                          check(cudaMemcpy(other.bounce->p, src, bytes, cudaMemcpyDeviceToHost)); }
                        check(cudaMemcpyAsync(dst, other.bounce->p, bytes, cudaMemcpyHostToDevice, other.stream));
                        check(cudaStreamSynchronize(other.stream));
                    }
                };
                auto &ox = other.buf("x", (size_t)nt * H);
                const uint16_t *ox_half = half_x ? (const uint16_t *)ox.p : nullptr;
                if (half_x) check(cudaStreamWaitEvent(other.stream, mla_ready, 0));
                auto &ob = other.buf("bounds", E + 1);
                auto &od = other.buf("dest", nt * K + 128), &os = other.buf("source", nt * K + 128);
                auto &orr = other.buf("result", (size_t)nt * K * H);
                auto &packed = other.buf("packed", (size_t)nt * K * H);
                if (half_x) upload(ox.p, x_half, (size_t)nt * H * 2);
                else upload(ox.p, x, (size_t)nt * H * 4);
                upload(ob.p, bounds.p, (E + 1) * 4);
                upload(od.p, dest.p, nt * K * 4); upload(os.p, source.p, nt * K * 4);
                if (partial) {
                    upload(other.buf("ids", nt * K).p, ids.p, nt * K * 4);
                    upload(other.buf("rw", nt * K).p, rw.p, nt * K * 4);
                }
                const bool stepwise = returned && incremental_return;
                auto give_back = [&](int first) {
                    const int begin = hb[first], rows = hb[first + 16] - begin;
                    if (!rows) return;
                    k::glm_copy_route_rows(orr.f(), packed.f(), (int *)od.p, begin, rows, H, true, other.stream);
                    check(cudaEventRecord(other.produced, other.stream));
                    check(cudaStreamWaitEvent(other.back, other.produced, 0));
                    check(cudaMemcpyPeerAsync(received_rows + (size_t)begin * H, primary_device,
                                              packed.f((size_t)begin * H), other.device, (size_t)rows * H * 4,
                                              other.back));
                };
                expert_groups(p, layer, *other.gpu, other.cache, ox.f(), (int *)ob.p, (int *)od.p,
                              (int *)os.p, hb, orr.f(), other.stream, 1, 2,
                              [&](const std::string &name, size_t n) -> Device & { return other.buf(name, n); },
                              stepwise ? std::function<void(int)>(give_back) : std::function<void(int)>(), ox_half);
                if (stepwise) {
                    check(cudaEventRecord(other.finished, other.back));
                    return;
                }
                if (partial) {
                    k::glm_route_sum_owned(orr.f(), other.buf("rw", nt * K).f(), (int *)other.buf("ids", nt * K).p,
                                           nullptr, packed.f(), H, K, nt, primary_prefill_groups, 1, other.stream);
                    check(cudaEventRecord(other.produced, other.stream));
                    check(cudaStreamWaitEvent(other.back, other.produced, 0));
                    check(cudaMemcpyPeerAsync(received_partial, primary_device, packed.p, other.device,
                                              (size_t)nt * H * 4, other.back));
                    check(cudaEventRecord(other.finished, other.back));
                    return;
                }
                for (int first = 0; first < E; first += 16) {
                    if (!strata::prefill::owns_expert_group(first / 16, 1, 2, primary_prefill_groups)) continue;
                    k::glm_copy_route_rows(orr.f(), packed.f(), (int *)od.p, hb[first],
                                            hb[first + 16] - hb[first], H, true, other.stream);
                    // Returned on the secondary stream while the primary still computes its own groups.
                    if (returned && hb[first + 16] != hb[first])
                        check(cudaMemcpyPeerAsync(received_rows + (size_t)hb[first] * H, primary_device,
                                                  packed.f((size_t)hb[first] * H), other.device,
                                                  (size_t)(hb[first + 16] - hb[first]) * H * 4, other.stream));
                }
                check(cudaEventRecord(other.finished, other.stream));
            });
        }
        try {
            expert_groups(p, layer, *gpu, prefill_cache, x, (int *)bounds.p, (int *)dest.p,
                          (int *)source.p, hb, result.f(), stream, 0, split ? 2 : 1,
                          [&](const std::string &name, size_t n) -> Device & { return buf("moe_" + name, n); });
        } catch (...) {
            if (pending.valid()) pending.wait();
            throw;
        }
        if (partial) {
            pending.get();
            check(cudaStreamWaitEvent(stream, secondary->finished, 0));
            k::glm_route_sum_owned(result.f(), rw.f(), (int *)ids.p, received_partial, out, H, K, nt,
                                   primary_prefill_groups, 0, stream);
            return;
        }
        if (split) {
            pending.get();
            auto &other = *secondary;
            auto &received = buf("moe_peer_results", (size_t)nt * K * H);
            check(cudaStreamWaitEvent(stream, other.finished, 0));
            for (int first = 0; first < E; first += 16) {
                if (!strata::prefill::owns_expert_group(first / 16, 1, 2, primary_prefill_groups)) continue;
                const int begin = hb[first], rows = hb[first + 16] - begin;
                if (!rows) continue;
                const size_t bytes = (size_t)rows * H * 4;
                const float *packed = other.buf("packed", (size_t)nt * K * H).f((size_t)begin * H);
                float *dst = received.f((size_t)begin * H);
                if (returned) {
                } else if (other.peer) {
                    check(cudaMemcpyPeerAsync(dst, primary_device, packed, other.device, bytes, stream));
                } else {
                    { DeviceScope scope(other.device);
                      check(cudaEventSynchronize(other.finished));
                      check(cudaMemcpy(other.bounce->p, packed, bytes, cudaMemcpyDeviceToHost)); }
                    check(cudaMemcpyAsync(dst, other.bounce->p, bytes, cudaMemcpyHostToDevice, stream));
                    check(cudaStreamSynchronize(stream));
                }
                k::glm_copy_route_rows(received.f(), result.f(), (int *)dest.p, begin, rows, H, false, stream);
            }
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
        if (!mtp_cpu_experts && mtp_lookup) {
            // GPU draft experts: groups come from the device lookup (one group per route, as the host built
            // them), so the step needs no host synchronization.
            ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
            auto &ptr = buf("mtp_pointers", K * 2), &bounds = buf("mtp_bounds", K + 1);
            auto &dst = buf("mtp_dest", K), &src = buf("mtp_source", K), &groups = buf("mtp_groups", 1);
            auto &qa = buf("mtp_q8", (k::native_q8_1_bytes(H, 1) + 3) / 4);
            auto &scratch = buf("mtp_expert_scratch", (k::native_expert_scratch_bytes(K, layer.intermediate) + 3) / 4);
            auto &result = buf("mtp_expert_results", K * H);
            k::glm_resident_routes((const int *)ids.p, (const unsigned long long *)mtp_lookup->p,
                                   (unsigned long long *)ptr.p, (int *)bounds.p, (int *)dst.p, (int *)src.p,
                                   (int *)groups.p, stream);
            k::native_quantize_q8_1(x, qa.p, H, 1, stream);
            k::native_expert_grouped(mtp_layout, (const unsigned long long *)ptr.p, (const int *)bounds.p,
                                     (const int *)groups.p, (const int *)dst.p, (const int *)src.p, K, K,
                                     qa.p, scratch.p, result.f(), stream);
            k::glm_route_sum(result.f(), routing.f(), out, H, K, 1, stream);
            return;
        }
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
    bool direct_experts() const {
        return device_resident_experts && device_lookup_ready && !fast && decode_prefill_cache &&
               (batch_tokens == 1 || (batched_resident && batch_tokens <= cpu::MAXT)) &&
               (!capture_hidden || batched_resident) && (!secondary || !secondary_enabled || secondary->peer);
    }
    void enqueue_moe_prefix(const std::string &p, int l, const float *x, float *out,
                             const strata::core::LayerDescriptor &layer) {
        const int nt = batch_tokens;
        auto &logits = buf("router_logits", m.experts * nt);
        auto &ids = buf("router_ids", m.top_k * nt);
        auto &rw = buf("router_weights", m.top_k * nt);
        share_quant(x, m.hidden);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        const float *bonus = route_bonus_row(l);
        if (nt > 1 && m.experts <= 512) k::glm_router_tokens(logits.f(), bias.f(), (int *)ids.p, rw.f(), m.experts, m.top_k, m.expert_scale, nt, stream, bonus);
        else k::glm_router(logits.f(), bias.f(), (int *)ids.p, rw.f(), m.experts, m.top_k, m.expert_scale, stream, bonus);
        auto *host_ids = (int *)host_moe->p;
        auto *routing = (float *)(host_ids + cpu::MAXT * m.top_k);
        auto *activation = routing + cpu::MAXT * m.top_k;
        check(cudaMemcpyAsync(host_ids, ids.p, m.top_k * nt * sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(routing, rw.p, m.top_k * nt * sizeof(float), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(activation, x, m.hidden * nt * sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (host_router_logits)
            check(cudaMemcpyAsync(host_router_logits->p, logits.p, (size_t)m.experts * nt * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(stream, &status));
        check(cudaEventRecordWithFlags(moe_ready, stream,
              status == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
        ffn(p, x, out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
        if (direct_experts()) {
            const auto &G = artifact.at(p + "ffn_gate_exps.weight");
            const auto &D = artifact.at(p + "ffn_down_exps.weight");
            auto layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate);
            layout.swiglu_limit = layer.swiglu_limit;
            const bool combined = layer_graphs && nt == 1 && !capturing_history && layer.mixer == strata::core::MixerKind::Kda;
            cached_decode->run_device(layout, x, (int *)ids.p, l, stream,
                                      decode_graphs && !combined, !device_reduction, nt);
            if (async_moe && device_reduction) cached_decode->record_reader(l, stream);
        }
        moe_prefixes[l] = {(int *)ids.p, rw.f(), gpu ? gpu->cursor : 0};
    }
    void moe(const std::string &p, int l, const float *x, float *out,
             const strata::core::LayerDescriptor &layer) {
        if(artifact.is_exl3()) {if(fast)moe_exl3_gpu(p,l,x,out,layer);else moe_exl3(p,l,x,out,layer);return;}
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
        if (prepared_moe_layer != l) enqueue_moe_prefix(p, l, x, out, layer);
        prepared_moe_layer = -1;
        const auto prefix = moe_prefixes.at(l);
        auto *host_ids = (int *)host_moe->p;
        auto *routing = (float *)(host_ids + cpu::MAXT * m.top_k);
        auto *activation = routing + cpu::MAXT * m.top_k;
        const bool direct = direct_experts();
        const bool asynchronous = async_moe && direct && device_reduction;
        // The secondary copies its output before reusing it. The primary waits
        // for that copy before reduction and before reusing this phase arena.
        Device *async_remote = asynchronous ? &buf("async_remote_results", (size_t)nt * 8 * m.hidden) : nullptr;
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        auto layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate);
        layout.swiglu_limit = layer.swiglu_limit;
        std::future<void> pending;
        struct WaitPending {
            std::future<void> &future;
            ~WaitPending() { if (future.valid()) future.wait(); }
        } wait_pending{pending};
        // Complete the pinned route readback before the worker reads it.
        // Its row remains owned by this layer until the worker future is joined.
        check(cudaEventSynchronize(moe_ready));
        if (direct) {
            if (secondary && secondary_enabled) {
                pending = secondary->worker->submit([&, layout] {
                    auto &other = *secondary;
                    check(cudaStreamWaitEvent(other.stream, moe_ready, 0));
                    check(cudaMemcpyPeerAsync(other.decode->activation->p, other.device, x, primary_device,
                                              (size_t)nt * m.hidden * sizeof(float), other.stream));
                    check(cudaMemcpyAsync(other.decode->route_ids->p, host_ids,
                                          (size_t)nt * m.top_k * sizeof(int), cudaMemcpyHostToDevice, other.stream));
                    other.decode->run_device(layout, other.decode->activation->f(), (int *)other.decode->route_ids->p,
                                              l, other.stream, decode_graphs, !device_reduction, nt);
                    if (asynchronous) {
                        other.decode->record_reader(l, other.stream);
                        check(cudaMemcpyPeerAsync(async_remote->p, primary_device, other.decode->output->p,
                                                  other.device, (size_t)nt * 8 * m.hidden * sizeof(float), other.stream));
                        check(cudaEventRecord(other.finished, other.stream));
                    } else check(cudaStreamSynchronize(other.stream));
                });
            }
        }
        check(cudaEventSynchronize(moe_ready));
        trace_seen();
        host_selected.assign(host_ids, host_ids + m.top_k * nt);
        auto &selected = host_selected;
        if(expert_observer)expert_observer->input(l,position,nt,m.hidden,m.top_k,selected.data(),activation);
        if(block_screen)block_screen->input(l,position,nt,m.hidden,m.top_k,selected.data(),routing,activation);
        ram_touch(l, selected.data(), nullptr, nt * m.top_k);
        trace_routes(l, position, nt, m.top_k, selected.data());
        if (host_router_logits) {
            if (router_bias_written.insert(l).second) {
                const auto bias = weight(p + "exp_probs_b.bias").floats(m.experts);
                const int header[2] = {l, m.experts};
                router_bias_trace.write((const char *)header, sizeof(header));
                router_bias_trace.write((const char *)bias.data(), bias.size() * sizeof(float));
                router_bias_trace.flush();
            }
            for (int t = 0; t < nt; ++t) {
                const int header[3] = {position + t, l, m.experts};
                router_score_trace.write((const char *)header, sizeof(header));
                router_score_trace.write((const char *)host_router_logits->p + (size_t)t * m.experts * sizeof(float),
                                         (size_t)m.experts * sizeof(float));
            }
            if (!router_score_trace || !router_bias_trace) throw std::runtime_error("GLM: failed writing router score trace");
        }
        cpu::NativeFmt f;
        std::string error;
        if (!cpu::native_fmt(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate, f, error))
            throw std::runtime_error(error);
        f.swiglu_limit = layer.swiglu_limit;
        f.q23_layout=q23_layout;
        f.observer=expert_observer&&expert_observer->collects_moments()?expert_observer.get():nullptr;f.observer_layer=l;
        f.fuse_h_quant=artifact.has_expert_pack()&&(!std::getenv("STRATA_GLM_PACK_FUSE_QUANT")||std::string(std::getenv("STRATA_GLM_PACK_FUSE_QUANT"))!="0");
        auto prepared = prepared_experts.find(l);
        if (prepared != prepared_experts.end()) f = prepared->second.format;
        f.canon = canon_types(G.tensor->type, D.tensor->type);
        f.gate_skip = f.canon ? gate_skip[l] : 0.f;
        if(block_screen){f.hidden_transform=block_screen.get();f.observer_layer=l;f.fuse_h_quant=false;}
        layout.gate_skip = f.gate_skip;
        if (f.canon) cpu::canon_gate_stats_layer(l);
        if (f.canon && (prepared != prepared_experts.end() || remote_tp || direct || expert_cache[l]))
            throw std::runtime_error("GLM: canonical experts need the plain CPU expert path or the pipelined tier");
        host_quant.resize(nt);
        for (auto &q : host_quant) q.resize(f.act_bytes);
        auto &quant = host_quant;
        if (!asynchronous)
            for (int t = 0; t < nt; ++t)
                cpu::native_quant_act(f, activation + t * m.hidden, quant[t].data());
        std::map<int, size_t> groups;
        std::vector<std::vector<uint8_t>> blobs;
        host_results.resize(selected.size() * m.hidden);
        auto &results = host_results;
        host_jobs.clear();
        auto &jobs = host_jobs;
        auto *cache = expert_cache[l].get();
        std::map<int, std::vector<int>> hits, remote_hits;
        auto primary_address = [&](int e) -> void * {
            if (auto it = decode_resident[l].find(e); it != decode_resident[l].end()) return it->second->p;
            if (decode_prefill_cache)
                if (auto *p = prefill_cache.expert(l, e, prefill_expert_stride(G, U, D))) return p;
            return cache && cache->slot_of(0, e) >= 0 ? cache->device_slot(cache->slot_of(0, e)) : nullptr;
        };
        auto secondary_address = [&](int e) -> void * {
            if (auto it = remote_decode_resident[l].find(e); it != remote_decode_resident[l].end()) return it->second->p;
            return decode_prefill_cache && secondary && secondary_enabled
                ? secondary->cache.expert(l, e, prefill_expert_stride(G, U, D)) : nullptr;
        };
        for (size_t j = 0; j < selected.size(); ++j) {
            const int e = selected[j], t = j / m.top_k;
            if (decode_cache_adapt) ++decode_seen[l][e];
            ++cache_entries;
            if (primary_address(e)) {
                hits[e].push_back(j);
                ++cache_hits;
                continue;
            }
            if (secondary_address(e)) {
                remote_hits[e].push_back(j);
                ++cache_hits;
                continue;
            }
            auto [where, inserted] = groups.emplace(e, jobs.size());
            if (inserted && !cache) {
                jobs.emplace_back();
                auto &job = jobs.back();
                job.expert_id=e;
                const auto *gate = prepared == prepared_experts.end() ? G.data() : prepared->second.gate.get();
                const auto *up = prepared == prepared_experts.end() ? U.data() : prepared->second.up.get();
                const auto *down = prepared == prepared_experts.end() || !prepared->second.down ? D.data() : prepared->second.down.get();
                job.blob = gate + (size_t)e * f.up_off;
                job.native_up = up + (size_t)e * f.up_off;
                job.native_down = down + (size_t)e * (f.bytes - f.down_off);
                if(auto owned=numa_experts.find(l);owned!=numa_experts.end())for(int node=0;node<2;++node)
                    job.numa[node]={owned->second.gate->shard(e,node),owned->second.up->shard(e,node),owned->second.down->shard(e,node)};
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
        // Cross-node TP: send the remote rows' work before the local pool starts, collect after the host sum.
        cpu::NativeFmt local_f = f;
        const bool remote = remote_tp && !jobs.empty() && !direct && !cache && prepared == prepared_experts.end() &&
                            remote_tp->split.count(l);
        if (remote) {
            local_f = remote_local_fmt(l, f);
            if (asynchronous)   // the request carries the activations, so they are quantized first
                for (int t = 0; t < nt; ++t)
                    cpu::native_quant_act(f, activation + t * m.hidden, quant[t].data());
            std::vector<strata::net::TpJob> list;
            for (size_t j = 0; j < selected.size(); ++j)
                if (groups.count(selected[j])) list.push_back({selected[j], (int32_t)(j / m.top_k), routing[j]});
            remote_tp->request_layer(l, nt, list, quant, f.act_bytes);
        } else if (remote_tp && remote_tp->trimmed.count(l) && !jobs.empty())
            throw std::logic_error("GLM remote TP: this node holds only part of each expert, but the layer ran locally");
        if (!direct && !remote_hits.empty()) {
            pending = secondary->worker->submit([&] {
                auto &other = *secondary;
                other.decode->run(layout, remote_hits, secondary_address, activation, nt, other.stream, decode_graphs);
                check(cudaStreamSynchronize(other.stream));
            });
        }
        try {
            if (!direct && !hits.empty()) {
                if (!cached_decode) cached_decode = std::make_unique<ResidentExecutor>();
                cached_decode->run(layout, hits, primary_address, activation, nt, stream, decode_graphs, f.canon);
            }
            // Quantize on the host only when at least one CPU expert needs it.
            if (asynchronous && !jobs.empty() && !remote)
                for (int t = 0; t < nt; ++t)
                    cpu::native_quant_act(f, activation + t * m.hidden, quant[t].data());
            // Both GPUs execute resident experts while CPU workers handle misses.
            pool.run_split_multi_native(remote ? local_f : f, jobs.data(), jobs.size());
            if (!(direct && device_reduction) && (!artifact.has_expert_pack()||!hits.empty()||!remote_hits.empty())) check(cudaStreamSynchronize(stream));
        } catch (...) {
            if (pending.valid()) pending.wait();
            throw;
        }
        if (pending.valid()) pending.get();
        if (direct && device_reduction) {
            auto &cpu = buf("device_cpu_results", (size_t)nt * 8 * m.hidden);
            auto &remote = asynchronous ? *async_remote : buf("device_remote_results", (size_t)nt * 8 * m.hidden);
            auto &owners = buf("device_result_owners", nt * 8);
            auto *host = (float *)(asynchronous ? cached_decode->reduction_for(l).p : cached_decode->host_reduction->p);
            if (asynchronous && secondary && secondary_enabled)
                check(cudaStreamWaitEvent(stream, secondary->finished, 0));
            auto *ownership = (int *)(host + ResidentExecutor::entries * 4096);
            std::memcpy(host, results.data(), (size_t)nt * 8 * m.hidden * sizeof(float));
            std::fill_n(ownership, nt * 8, 0);
            for (const auto &[expert, indices] : hits) for (int j : indices) ownership[j] = 1;
            for (const auto &[expert, indices] : remote_hits) for (int j : indices) ownership[j] = 2;
            cpu.put_async(host, (size_t)nt * 8 * m.hidden * sizeof(float), stream);
            owners.put_async(ownership, (size_t)nt * 8 * sizeof(int), stream);
            if (!asynchronous && !remote_hits.empty())
                check(cudaMemcpyPeerAsync(remote.p, primary_device, secondary->decode->output->p,
                                          secondary->device, (size_t)nt * 8 * m.hidden * sizeof(float), stream));
            if (nt == 1)
                k::glm_moe_reduce(cpu.f(), cached_decode->output->f(), remote.f(), (int *)owners.p, prefix.weights, out, m.hidden, stream);
            else k::glm_moe_reduce_batch(cpu.f(), cached_decode->output->f(), remote.f(), (int *)owners.p,
                                         prefix.weights, out, m.hidden, nt, stream);
            // Slot replacement can overwrite weights from any layer. Finish
            // the resident products before admitting a background copy.
            if (decode_cache_adapt && !asynchronous) check(cudaStreamSynchronize(stream));
        } else {
            if (!hits.empty()) cached_decode->collect(hits, results);
            if (!remote_hits.empty()) secondary->decode->collect(remote_hits, results);
        }
        if (decode_cache_adapt) {
            std::vector<int> candidates;
            for (const auto &[expert, job] : groups) candidates.push_back(expert);
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return decode_seen[l][a] != decode_seen[l][b] ? decode_seen[l][a] > decode_seen[l][b] : a < b;
            });
            for (int expert : candidates) adapt_expert(l, expert, G, U, D);
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
        if (direct && device_reduction) return;
        if(artifact.has_expert_pack()&&!remote) {
            if(!packed_cpu_sum){packed_cpu_sum=std::make_unique<Pinned>(cpu::MAXT*m.hidden*4);check(cudaEventCreateWithFlags(&packed_sum_uploaded,cudaEventDisableTiming));}
            check(cudaEventSynchronize(packed_sum_uploaded));
            auto* sum=static_cast<float*>(packed_cpu_sum->p);pool.reduce_routed(results.data(),routing,sum,nt,m.top_k,m.hidden,f.canon);
            auto& routed=buf("moe_sum",m.hidden*nt);routed.put_async(sum,size_t(nt)*m.hidden*4,stream);check(cudaEventRecord(packed_sum_uploaded,stream));
            const float one=1;check(cublasSaxpy(blas,m.hidden*nt,&one,routed.f(),1,out,1));return;
        }
        host_sum.assign(m.hidden * nt, 0.f);
        auto &sum = host_sum;
        for (size_t j = 0; j < selected.size(); ++j)
            for (int i = 0; i < m.hidden; ++i)
                sum[(j / m.top_k) * m.hidden + i] += routing[j] * results[j * m.hidden + i];
        if (remote) remote_tp->finish(sum.data());
        auto &routed = buf("moe_sum", m.hidden * nt);
        routed.put(sum.data(), sum.size() * 4);
        const float one = 1;
        check(cublasSaxpy(blas, m.hidden * nt, &one, routed.f(), 1, out, 1));
    }
    // The step pipeline covers the plain CPU-expert decode path; every other expert placement keeps moe().
    void prepare_step_mailbox() {
        if (mailbox) return;
        if (m.experts > 512) throw std::runtime_error("GLM: step pipeline expert count");
        mailbox = std::make_unique<StepMailbox>(mailbox_group_stride() * (split_verify || defer_routes > 0 ? 2 : 1),
                                               cpu::MAXT, m.top_k, m.hidden, canon_experts, bool(ram_tier) || tier_learn_unbiased);
        if (tier_learn_unbiased)
            std::cerr << "DECODE_CACHE_LEARN source=unbiased_router\n";
    }
    bool pipeline_ready(int nt) const {
        if (!step_pipeline || !gpu || fast || artifact.is_exl3() || nt < 1 || nt > cpu::MAXT || profile ||
            gpu_decode_experts || tp_active() || (secondary && secondary_enabled) || decode_cache_adapt ||
            device_resident_experts || decode_prefill_cache)
            return false;
        for (const auto &cache : expert_cache)
            if (cache) return false;
        return true;
    }
    // GPU half of a pipelined MoE layer: route, publish, shared expert, then wait for the CPU's routed sum.
    // The pipelined MoE layer in two halves: publish (router, mailbox publication, resident experts, shared
    // expert) and combine (wait for the CPU sum and add it). Split verify enqueues the other group in between.
    struct MoePending {
        int layer, slot, nt; int *ids; float *rw, *out; ResidentExecutor *executor; bool canon;
        const float *x = nullptr;
    };
    MoePending enqueue_moe_publish(const std::string &p, int l, const float *x, float *out,
                                   const strata::core::LayerDescriptor &layer, int slot, ResidentExecutor *executor) {
        const int nt = batch_tokens;
        auto &logits = buf("router_logits", m.experts * nt);
        auto &ids = buf("moe_ids", m.top_k * nt);
        auto &rw = buf("moe_weights", m.top_k * nt);
        share_quant(x, m.hidden);
        mat(p + "ffn_gate_inp.weight", x, logits.f());
        auto &bias = weight(p + "exp_probs_b.bias");
        const float *bonus = route_bonus_row(l);
        const bool wants_unbiased = tier_learn_unbiased || (bonus && ram_tier);
        if (fuse_route_publish && nt == 1 && m.experts <= 512 && !wants_unbiased) {
            // Router and publication in one launch; the pending record below is unchanged.
            k::glm_router_publish(logits.f(), bias.f(), (int *)ids.p, rw.f(), m.experts, m.top_k, m.expert_scale, bonus,
                                  mailbox->device_view, slot, x, mailbox->device_generation(), stream);
            MoePending pending{l, slot, nt, (int *)ids.p, rw.f(), out, executor, canon_layer(l), x};
            enqueue_moe_residents(pending, p, layer);
            return pending;
        }
        if (nt > 1 && m.experts <= 512) k::glm_router_tokens(logits.f(), bias.f(), (int *)ids.p, rw.f(), m.experts, m.top_k, m.expert_scale, nt, stream, bonus);
        else k::glm_router(logits.f(), bias.f(), (int *)ids.p, rw.f(), m.experts, m.top_k, m.expert_scale, stream, bonus);
        const int *wanted = tier_learn_unbiased ? (int *)ids.p : nullptr;
        if (bonus && (ram_tier || tier_learn_unbiased)) {
            // Unbiased selection for read-ahead or GPU tier learning; its weights are not executed.
            auto &want = buf("moe_wanted", m.top_k * nt), &scratch = buf("moe_wanted_weights", m.top_k * nt);
            if (nt > 1 && m.experts <= 512) k::glm_router_tokens(logits.f(), bias.f(), (int *)want.p, scratch.f(), m.experts, m.top_k, m.expert_scale, nt, stream);
            else k::glm_router(logits.f(), bias.f(), (int *)want.p, scratch.f(), m.experts, m.top_k, m.expert_scale, stream);
            wanted = (int *)want.p;
        }
        k::glm_mailbox_publish(mailbox->device_view, slot, (int *)ids.p, rw.f(), x, nt, mailbox->device_generation(), stream, wanted);
        MoePending pending{l, slot, nt, (int *)ids.p, rw.f(), out, executor, canon_layer(l), x};
        enqueue_moe_residents(pending, p, layer);
        return pending;
    }
    // Resident experts run while the CPU computes the others; then the shared expert.
    void enqueue_moe_residents(const MoePending &pending, const std::string &p, const strata::core::LayerDescriptor &layer) {
        if (pipeline_residents) {
            const auto &G = artifact.at(p + "ffn_gate_exps.weight");
            const auto &D = artifact.at(p + "ffn_down_exps.weight");
            auto layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate);
            layout.swiglu_limit = layer.swiglu_limit;
            layout.gate_skip = pending.canon ? gate_skip[pending.layer] : 0.f;
            pending.executor->run_device(layout, pending.x, pending.ids, pending.layer, stream, false, false, pending.nt, pending.canon);
            // Canonical, strict: the CPU adds every route, so the resident rows go to the mailbox.
            if (pending.canon && canon_strict())
                k::glm_mailbox_post_rows(mailbox->device_view, pending.slot, pending.ids, pending.rw,
                                         (const unsigned long long *)pending.executor->lookup->p + (size_t)pending.layer * 288,
                                         pending.executor->output->f(), pending.nt, mailbox->device_generation(), stream);
        }
        ffn(p, pending.x, pending.out, "_shexp", m.shared_intermediate, layer.shared_swiglu_limit);
    }
    void enqueue_moe_combine(const MoePending &pending) {
        if (defer_active && !defer_routes) {
            // The CPU's sum follows after the next mixer; only device-combined resident rows are added here.
            if (pipeline_residents && !(pending.canon && canon_strict()))
                k::glm_mailbox_add_resident(mailbox->device_view, pending.out, pending.nt, pending.ids, pending.rw,
                                            (const unsigned long long *)pending.executor->lookup->p + (size_t)pending.layer * 288,
                                            pending.executor->output->f(), stream);
            return;
        }
        if (pending.canon && (canon_strict() || !pipeline_residents))
            k::glm_mailbox_wait_add(mailbox->device_view, pending.slot, pending.out, pending.nt,
                                    mailbox->device_generation(), stream);
        else if (pipeline_residents)
            k::glm_mailbox_wait_add_resident(mailbox->device_view, pending.slot, pending.out, pending.nt,
                                             mailbox->device_generation(), pending.ids, pending.rw,
                                             (const unsigned long long *)pending.executor->lookup->p + (size_t)pending.layer * 288,
                                             pending.executor->output->f(), stream);
        else k::glm_mailbox_wait_add(mailbox->device_view, pending.slot, pending.out, pending.nt,
                                     mailbox->device_generation(), stream);
    }
    void enqueue_moe_pipelined(const std::string &p, int l, const float *x, float *out,
                               const strata::core::LayerDescriptor &layer) {
        enqueue_moe_combine(enqueue_moe_publish(p, l, x, out, layer, l, cached_decode.get()));
    }
    // CPU half: the same jobs and fixed-order reduction as moe()'s CPU path, read from and written to the mailbox.
    // route_mask (bit j = this call computes route j) and out_slot (sum target, -1 = slot) serve partial deferral:
    // the pass with out_slot >= 0 is the late one and performs none of the layer's bookkeeping side effects.
    void cpu_moe_mailbox(int l, int slot, int nt, int position, int second_slot = -1, int second_nt = 0,
                         uint64_t route_mask = ~uint64_t{0}, int out_slot = -1) {
        const int H = m.hidden, K = m.top_k;
        const int sum_slot = out_slot < 0 ? slot : out_slot;
        const bool primary = out_slot < 0;
        if (!primary && (exact_cache || remote_tp || second_slot >= 0 || !pipeline_residents))
            throw std::runtime_error("GLM: partial deferral needs plain pipelined decode with GPU residents");
        const int first_nt = nt;
        const auto &layer = m.layers[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        const int *selected = mailbox->ids(slot);
        const float *routing = mailbox->weights(slot), *activation = mailbox->act(slot);
        std::vector<int> merged_ids;
        std::vector<float> merged_weights, merged_activations;
        if (second_slot >= 0) {
            if (second_nt < 1 || nt + second_nt > cpu::MAXT || !canon_layer(l) || !canon_strict())
                throw std::runtime_error("GLM: paired expert service requires canonical strict rows and width <= 8");
            merged_ids.assign(selected, selected + nt * K);
            merged_ids.insert(merged_ids.end(), mailbox->ids(second_slot), mailbox->ids(second_slot) + second_nt * K);
            merged_weights.assign(routing, routing + nt * K);
            merged_weights.insert(merged_weights.end(), mailbox->weights(second_slot), mailbox->weights(second_slot) + second_nt * K);
            merged_activations.assign(activation, activation + nt * H);
            merged_activations.insert(merged_activations.end(), mailbox->act(second_slot), mailbox->act(second_slot) + second_nt * H);
            nt += second_nt;
            selected = merged_ids.data(); routing = merged_weights.data(); activation = merged_activations.data();
            ++merged_expert_layers;
        }
        for (int j = 0; j < nt * K; ++j)
            if (selected[j] < 0 || selected[j] >= m.experts) throw std::runtime_error("GLM: invalid published expert");
        if (primary) {
            if (expert_observer) expert_observer->input(l, position, nt, H, K, selected, activation);
            if (block_screen) block_screen->input(l, position, nt, H, K, selected, routing, activation);
            if (!cache_probe_routes.empty())
                for (int j = 0; j < nt * K; ++j) ++cache_probe_routes.at(l).at(selected[j]);
            ram_touch(l, selected, mailbox->wanted(slot), nt * K);
            trace_routes(l, position, nt, K, selected);
        }
        const auto &G = artifact.at(p + "ffn_gate_exps.weight");
        const auto &U = artifact.at(p + "ffn_up_exps.weight");
        const auto &D = artifact.at(p + "ffn_down_exps.weight");
        cpu::NativeFmt f;
        std::string error;
        if (!cpu::native_fmt(G.tensor->type, D.tensor->type, H, layer.intermediate, f, error))
            throw std::runtime_error(error);
        f.swiglu_limit = layer.swiglu_limit;
        f.q23_layout = q23_layout;
        f.observer = expert_observer && expert_observer->collects_moments() ? expert_observer.get() : nullptr;
        f.observer_layer = l;
        f.fuse_h_quant = artifact.has_expert_pack() && (!std::getenv("STRATA_GLM_PACK_FUSE_QUANT") ||
                                                        std::string(std::getenv("STRATA_GLM_PACK_FUSE_QUANT")) != "0");
        auto prepared = prepared_experts.find(l);
        if (prepared != prepared_experts.end()) f = prepared->second.format;
        f.canon = canon_types(G.tensor->type, D.tensor->type);
        f.gate_skip = f.canon ? gate_skip[l] : 0.f;
        if(block_screen){f.hidden_transform=block_screen.get();f.observer_layer=l;f.fuse_h_quant=false;}
        if (f.canon && primary) cpu::canon_gate_stats_layer(l);
        if (f.canon && (prepared != prepared_experts.end() || remote_tp))
            throw std::runtime_error("GLM: canonical experts cannot be combined with prepared experts or remote TP");
        host_quant.resize(nt);
        for (int t = 0; t < nt; ++t) {
            host_quant[t].resize(f.act_bytes);
            cpu::native_quant_act(f, activation + (size_t)t * H, host_quant[t].data());
        }
        // Canonical rows go straight into the mailbox: the GPU reads the ones its part of the chain needs.
        if (!f.canon) host_results.resize((size_t)nt * K * H);
        float *const results = f.canon ? mailbox->rows(slot) : host_results.data();
        host_jobs.clear();
        std::array<int, 512> job_of;
        job_of.fill(-1);
        const auto owned = numa_experts.find(l);
        const auto *resident = pipeline_residents ? (const unsigned long long *)cached_decode->host_lookup->p + (size_t)l * 288 : nullptr;
        std::array<float, cpu::MAXT * 8> cpu_weights;
        if (nt * K > (int)cpu_weights.size()) throw std::runtime_error("GLM: step pipeline route count");
        // Canonical sum with resident routes: their rows arrive in the mailbox from the GPU, and this thread adds
        // all routes after the pool has written the others. Without resident routes the pool's last writers do.
        bool gpu_rows = false;
        if (f.canon && resident && canon_strict())
            for (int j = 0; j < nt * K; ++j) gpu_rows = gpu_rows || (routing[j] != 0.f && resident[selected[j]]);
        for (int j = 0; j < nt * K; ++j) {
            const int e = selected[j], t = j / K;
            if (primary) ++cache_entries;
            cpu_weights[j] = routing[j];
            if (routing[j] == 0.f) {
                // Pruned by STRATA_GLM_ROUTE_MIN_SHARE; contributes nothing.
                if (primary) ++pruned_routes;
                if (!f.canon) std::fill_n(results + (size_t)j * H, H, 0.f);
                continue;
            }
            if ((resident && resident[e]) || !(route_mask >> j & 1u)) {
                // The GPU adds this route (resident) or the other pass does (masked out); this sum sees an exact zero.
                if (primary && resident && resident[e]) ++cache_hits;
                cpu_weights[j] = 0.f;
                if (!f.canon) std::fill_n(results + (size_t)j * H, H, 0.f);
                continue;
            }
            if (gpu_rows) cpu_weights[j] = 0.f;
            if (job_of[e] < 0) {
                job_of[e] = (int)host_jobs.size();
                auto &job = host_jobs.emplace_back();
                job.expert_id = e;
                const auto *gate = prepared == prepared_experts.end() ? G.data() : prepared->second.gate.get();
                const auto *up = prepared == prepared_experts.end() ? U.data() : prepared->second.up.get();
                const auto *down = prepared == prepared_experts.end() || !prepared->second.down ? D.data() : prepared->second.down.get();
                job.blob = gate + (size_t)e * f.up_off;
                job.native_up = up + (size_t)e * f.up_off;
                job.native_down = down + (size_t)e * (f.bytes - f.down_off);
                if (owned != numa_experts.end())
                    for (int node = 0; node < 2; ++node)
                        job.numa[node] = {owned->second.gate->shard(e, node), owned->second.up->shard(e, node),
                                          owned->second.down->shard(e, node)};
            }
            auto &job = host_jobs[job_of[e]];
            const int slot = job.nt++;
            job.nact[slot] = host_quant[t].data();
            job.out[slot] = results + (size_t)j * H;
        }
        std::vector<std::shared_ptr<strata::core::ExactExpertCache::Entry>> exact_leases;
        if (exact_cache) {
            std::vector<std::pair<int,strata::core::ExactExpertCache::Source>> requests;
            for (const auto& job : host_jobs) {
                strata::core::ExactExpertCache::Source source;
                int i=0;
                for (const auto* tensor : {&G,&U,&D}) {
                    const size_t bytes=tensor->bytes/m.experts;
                    source[i++]={tensor->file->path(),tensor->file->data_start()+tensor->tensor->offset+job.expert_id*bytes,bytes};
                }
                requests.emplace_back(l*m.experts+job.expert_id,std::move(source));
            }
            exact_leases=exact_cache->acquire(requests);
            for (size_t i=0;i<host_jobs.size();++i) {
                host_jobs[i].blob=exact_leases[i]->data(0);
                host_jobs[i].native_up=exact_leases[i]->data(1);
                host_jobs[i].native_down=exact_leases[i]->data(2);
            }
            if (l+1 == (int)m.layers.size() && (position%32==0)) report_exact_cache(position);
        }
        // Cross-node TP: the worker's rows of the same routes run while this node computes its own rows; its
        // router-weighted partial sums are added after the local sum.
        cpu::NativeFmt run_f = f;
        bool remote = false;
        if (remote_tp && remote_tp->split.count(l)) {
            std::vector<strata::net::TpJob> list;
            for (int j = 0; j < nt * K; ++j)
                if (cpu_weights[j] != 0.f) list.push_back({selected[j], (int32_t)(j / K), cpu_weights[j]});
            if (!list.empty()) {
                run_f = remote_local_fmt(l, f);
                remote_tp->request_layer(l, nt, list, host_quant, f.act_bytes);
                remote = true;
            }
        }
        const bool early_groups = second_slot >= 0 && merge_expert_groups == 2;
        if (early_groups) {
            // A's resident rows precede B's publication on the GPU stream.
            // B's resident work can overlap the CPU pass; its final canonical
            // reduction happens below, after those rows have arrived.
            if (gpu_rows) mailbox->wait_rows(slot, launch_failed);
            bool second_gpu_rows = false;
            for (int j = 0; j < nt * K; ++j) {
                const bool remote_row = j >= first_nt * K && resident && resident[selected[j]];
                cpu_weights[j] = remote_row ? 0.f : routing[j];
                second_gpu_rows |= remote_row && routing[j] != 0.f;
            }
            struct Completion { StepMailbox *mailbox; int slots[2]; } completion{mailbox.get(), {slot, second_slot}};
            cpu::ExpertPool::LayerGroups groups;
            groups.split = first_nt; groups.sum[0] = mailbox->sum(sum_slot); groups.sum[1] = mailbox->sum(second_slot);
            groups.context = &completion;
            groups.complete = [](void *context, int group) noexcept {
                auto &ready = *static_cast<Completion *>(context);
                if (!group) ready.mailbox->complete(ready.slots[0]);
            };
            pool.run_layer_native(run_f, host_jobs.data(), host_jobs.size(), results, cpu_weights.data(), groups, nt, K);
            if (second_gpu_rows) {
                mailbox->wait_rows(second_slot, launch_failed);
                for (int j = first_nt * K; j < nt * K; ++j)
                    if (routing[j] != 0.f && resident[selected[j]])
                        std::copy_n(mailbox->rows(second_slot) + (size_t)(j - first_nt * K) * H,
                                    H, results + (size_t)j * H);
                for (int t = first_nt; t < nt; ++t)
                    cpu::canon_route_sum(results + (size_t)t * K * H, routing + (size_t)t * K,
                                        mailbox->sum(second_slot) + (size_t)(t - first_nt) * H, K, H, 0, H);
            }
            mailbox->complete(second_slot);
        } else if (layer_flow)
            pool.run_layer_native(run_f, host_jobs.data(), host_jobs.size(), results, cpu_weights.data(),
                                  mailbox->sum(sum_slot), nt, K);
        else {
            if (!host_jobs.empty()) pool.run_split_multi_native(run_f, host_jobs.data(), host_jobs.size());
            pool.reduce_routed(results, cpu_weights.data(), mailbox->sum(sum_slot), nt, K, H, f.canon);
        }
        if (gpu_rows && !early_groups) {
            mailbox->wait_rows(slot, launch_failed);
            if (second_slot >= 0) {
                mailbox->wait_rows(second_slot, launch_failed);
                // Missing routes of both groups were computed into the first
                // slot. Only resident rows of group B come from its GPU slot.
                for (int j = first_nt * K; j < nt * K; ++j)
                    if (routing[j] != 0.f && resident[selected[j]])
                        std::copy_n(mailbox->rows(second_slot) + (size_t)(j - first_nt * K) * H,
                                    H, results + (size_t)j * H);
            }
            if (std::getenv("STRATA_GLM_CHECK_RESIDENT_ROWS")) {
                std::vector<float> reference(H);
                for (int j = 0; j < nt * K; ++j) {
                    const int e = selected[j];
                    if (!resident[e] || routing[j] == 0.f) continue;
                    cpu::ExpertJobMulti job{};
                    job.expert_id = e; job.nt = 1;
                    job.blob = G.data() + (size_t)e * f.up_off;
                    job.native_up = U.data() + (size_t)e * f.up_off;
                    job.native_down = D.data() + (size_t)e * (f.bytes - f.down_off);
                    job.nact[0] = host_quant[j / K].data(); job.out[0] = reference.data();
                    pool.run_split_multi_native(f, &job, 1);
                    if (std::memcmp(reference.data(), results + (size_t)j * H, H * sizeof(float))) {
                        size_t different = 0; float max_error = 0;
                        for (int r = 0; r < H; ++r) {
                            different += reference[r] != results[(size_t)j * H + r];
                            max_error = std::max(max_error, std::abs(reference[r] - results[(size_t)j * H + r]));
                        }
                        std::cerr << "RESIDENT_PARITY layer=" << l << " expert=" << e
                                  << " rows=" << different << " max_abs=" << max_error << '\n';
                        if (const char *path = std::getenv("STRATA_GLM_RESIDENT_FIXTURE")) {
                            std::ofstream out(path, std::ios::binary);
                            const int32_t header[]{G.tensor->type, D.tensor->type, H, layer.intermediate};
                            out.write((const char *)header, sizeof(header));
                            out.write((const char *)&f.swiglu_limit, sizeof(float));
                            out.write((const char *)job.blob, f.up_off);
                            out.write((const char *)job.native_up, f.up_off);
                            out.write((const char *)job.native_down, f.bytes - f.down_off);
                            out.write((const char *)(activation + (size_t)(j / K) * H), H * sizeof(float));
                            if (!out) throw std::runtime_error("GLM: failed writing resident parity fixture");
                        }
                        throw std::runtime_error("GLM: resident CPU/GPU rows differ at layer " +
                            std::to_string(l) + " expert " + std::to_string(e));
                    }
                }
            }
            for (int t = 0; t < nt; ++t)
                cpu::canon_route_sum(results + (size_t)t * K * H, routing + (size_t)t * K, mailbox->sum(sum_slot) + (size_t)t * H, K, H, 0, H);
        }
        if (second_slot >= 0 && !early_groups)
            std::copy_n(mailbox->sum(sum_slot) + (size_t)first_nt * H, (size_t)second_nt * H, mailbox->sum(second_slot));
        if (remote) remote_tp->finish(mailbox->sum(sum_slot));
        if (remote && remote_tp->check) remote_tp_check(l, f, G, U, D, nt, K, cpu_weights.data(), mailbox->sum(sum_slot));
        if (resident && primary) {
            const int *learned = tier_learn_unbiased ? mailbox->wanted(slot) : selected;
            if (tier_learn_unbiased) {
                if (second_slot >= 0 || !learned)
                    throw std::runtime_error("GLM: unbiased tier learning needs an unmerged mailbox");
                for (int j = 0; j < nt * K; ++j)
                    if (learned[j] < 0 || learned[j] >= m.experts)
                        throw std::runtime_error("GLM: invalid unbiased cache-learning route");
            }
            tier_select(l, learned, nt * K, resident);
        }
    }
    // Adaptive GPU tier for the step pipeline (STRATA_GLM_TIER_ADAPT=1 with --decode-cache-mib): recently
    // routed CPU experts replace cold resident ones. A replacement is detached from the lookup while the GPU
    // waits on the current layer; the service thread copies its bytes into pinned staging while it waits for
    // later layers (otherwise idle time); the upload into the freed slot starts after the step completes, on
    // its own stream; and a finished upload joins the lookup at a later step start.
    bool tier_adapt = std::getenv("STRATA_GLM_TIER_ADAPT") != nullptr;
    // Opt-in: cache popularity follows the router before residency bonuses.
    // Actual execution still uses the configured affinity and original weights.
    const bool tier_learn_unbiased = [] {
        const char *value = std::getenv("STRATA_GLM_TIER_LEARN_UNBIASED");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: unbiased tier learning must be 0 or 1");
    }();
    const int cache_probe_steps = [] {
        const char *value = std::getenv("STRATA_GLM_CACHE_PROBE");
        if (!value) return 0;
        size_t end = 0; const int steps = std::stoi(value, &end);
        if (value[end] || (steps != 0 && (steps < 8 || steps > 64)))
            throw std::invalid_argument("GLM: cache probe must be 0 or 8..64 tokens");
        return steps;
    }();
    std::vector<std::vector<uint64_t>> cache_probe_routes;
    bool sequence_speculating = false;
    const bool tier_owned_reserve = [] {
        const char *v = std::getenv("STRATA_GLM_TIER_OWNED_RESERVE"); return v && std::string(v) == "1";
    }();
    // STRATA_GLM_CANON=1: Q2_K/Q3_K routed experts use the canonical arithmetic of canon_expert.hpp on the CPU and
    // on the GPU, and the routed sum is the canonical chain, so logits do not depend on which experts are resident.
    const bool canon_experts = [] { const char *v = std::getenv("STRATA_GLM_CANON"); return v && std::string(v) == "1"; }();
    // STRATA_GLM_ROUTE_AFFINITY (opt-in, lossy): GPU-resident experts win selection ties within this margin, in
    // probability units. The per-expert bonuses are rebuilt at each step start with the lookup and uploaded once;
    // the pipelined router reads its layer's row.
    float route_affinity = [] {
        const char *v = std::getenv("STRATA_GLM_ROUTE_AFFINITY");
        const float x = v ? std::strtof(v, nullptr) : 0.f;
        if (!(x >= 0.f && x <= 2.f)) throw std::invalid_argument("STRATA_GLM_ROUTE_AFFINITY must be in [0, 2]");
        return x;
    }();
    // Optional per-layer overrides keep sensitive layers on their original routes.
    // Unlisted layers inherit the scalar affinity; this never changes the default.
    const std::vector<float> route_affinity_profile = [] {
        std::vector<float> values(64, -1.f);
        const char *path = std::getenv("STRATA_GLM_ROUTE_AFFINITY_PROFILE");
        if (!path) return values;
        const auto layer_profile = strata::artifact::read_json(path);
        for (const auto &[layer, value] : layer_profile.at("layers").object) {
            size_t end = 0; const auto index = std::stoul(layer, &end); const float affinity = value.real();
            if (layer[end] || index >= values.size() || !(affinity >= 0.f && affinity <= 2.f))
                throw std::invalid_argument("GLM: invalid affinity profile layer/value");
            values[index] = affinity;
        }
        return values;
    }();
    const bool profile_has_affinity = std::any_of(route_affinity_profile.begin(), route_affinity_profile.end(),
                                                 [](float value) { return value > 0.f; });
    bool has_route_affinity() const { return route_affinity > 0.f || profile_has_affinity; }
    std::unique_ptr<Pinned> host_route_bonus;
    std::unique_ptr<Device> route_bonus;
    bool route_bonus_active = false, route_bonus_dirty = true, route_bias_enabled = true;
    const float *route_bonus_row(int l) const {
        return route_bonus_active && l >= 0 && l < (int)m.layers.size() ? route_bonus->f((size_t)l * m.experts) : nullptr;
    }
    // STRATA_GLM_PREFER_RESIDENT_MIB=N (opt-in, lossy; needs STRATA_GLM_MAPPED_LAZY=1): for a host smaller than
    // the routed experts. About N MiB of experts stay in memory; routing selects among those (a bonus larger than
    // any score difference), and an expert the unbiased router asks for is read in the background and joins at a
    // later step, displacing the coldest one. Decode never waits for the disk. Which experts are in memory depends
    // on read completion times, so outputs are not reproducible run to run.
    std::unique_ptr<strata::core::ExactExpertCache> exact_cache;
    void prepare_exact_cache() {
        const char* value = std::getenv("STRATA_GLM_EXACT_CACHE_MIB");
        if (!value) return;
        size_t parsed = 0, mib = std::stoull(value, &parsed);
        if (parsed != std::strlen(value) || !mib || mib > 131072)
            throw std::invalid_argument("GLM: exact cache MiB must be 1..131072");
        if (!pipeline_ready(1) || tier_adapt || decode_cache_adapt || ram_tier || has_route_affinity() ||
            !numa_experts.empty() || !prepared_experts.empty() || q23_layout || capture_hidden ||
            artifact.has_expert_pack() || secondary || direct_upload_enabled() || block_screen ||
            std::getenv("STRATA_GLM_ROUTE_MIN_SHARE"))
            throw std::invalid_argument("GLM: exact cache prototype needs plain GGUF, single CPU/GPU pipeline, fixed GPU cache, unbiased routes and no speculation");
        exact_cache.reset();
        for (size_t l=0; l<m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            for (const char* role : {"gate", "up", "down"}) {
                const auto& t=artifact.at("blk."+std::to_string(l)+".ffn_"+role+"_exps.weight");
                if (!t.file || t.native || t.bytes % m.experts)
                    throw std::invalid_argument("GLM: exact cache requires native GGUF tensor ranges");
                t.file->discard_tensor_pages(*t.tensor,t.bytes);
            }
        }
        exact_cache=std::make_unique<strata::core::ExactExpertCache>(mib*MiB);
        std::cerr << "EXACT_CACHE budget_MiB=" << mib << " readers=4 routing=unchanged gpu=fixed cpu=lru direct=1\n";
    }
    void report_exact_cache(int position) {
        if (!exact_cache) return;
        const auto& s=exact_cache->stats;
        std::cerr << "EXACT_CACHE_STATS position=" << position << " hits=" << s.hits << " misses=" << s.misses
                  << " evictions=" << s.evictions << " read_bytes=" << s.read_bytes
                  << " resident_bytes=" << exact_cache->resident_bytes() << " wait_ms=" << s.wait_ms << '\n';
    }
    std::unique_ptr<RamTier> ram_tier;
    std::vector<int> prefill_wanted;
    bool lazy_experts = false;                     // STRATA_GLM_MAPPED_LAZY: prefill stages only routed experts
    std::vector<uint8_t> ram_used;                 // experts routed in the current step: not evicted at its end
    std::map<const void *, int> ram_files;         // one descriptor per mapped GGUF, for page-cache advice
    // Selection bonus of in-memory experts. The default exceeds any score difference, so routing never leaves the
    // memory set. STRATA_GLM_RAM_MARGIN=0.10 (hybrid): in-memory experts win near ties only; an expert outside the
    // set that still wins is read from the file on first use (the decode waits for it) and joins the set.
    const float kResidentBonus = tier_setting("STRATA_GLM_RAM_MARGIN", 4.f);
    size_t ram_misses = 0, ram_miss_bytes = 0;
    const float kRamDecay = tier_setting("STRATA_GLM_RAM_DECAY", 0.995f);
    const float kRamAdmit = tier_setting("STRATA_GLM_RAM_ADMIT", 8.f);
    const size_t kRamInflight = (size_t)tier_setting("STRATA_GLM_RAM_INFLIGHT", 32);
    // Called at each step start, after the resident lookup has been rebuilt and before the step is enqueued.
    void refresh_route_bonus() {
        const bool affinity = route_bias_enabled && has_route_affinity() && pipelining && pipeline_residents;
        const bool active = affinity || ram_tier;
        // Affinity follows the tier, which may change every step; memory residency sets the dirty flag itself.
        if (!active) { route_bonus_active = false; return; }
        if (route_bonus_active && !affinity && !route_bonus_dirty) return;
        const size_t count = m.layers.size() * (size_t)m.experts;
        if (!route_bonus) {
            host_route_bonus = std::make_unique<Pinned>(count * sizeof(float));
            route_bonus = std::make_unique<Device>(count * sizeof(float));
        }
        auto *bonus = (float *)host_route_bonus->p;
        std::fill_n(bonus, count, 0.f);
        if (ram_tier)
            for (size_t i = 0; i < count; ++i)
                if (ram_tier->state[i] == RamTier::Resident) bonus[i] = kResidentBonus;
        if (affinity)
            for (size_t l = 0; l < decode_resident.size(); ++l)
                for (const auto &[expert, slot] : decode_resident[l]) bonus[l * m.experts + expert] += route_affinity_profile[l] >= 0.f ? route_affinity_profile[l] : route_affinity;
        check(cudaStreamSynchronize(stream));   // the previous step's kernels have finished with the old bonuses
        route_bonus->put_async(bonus, count * sizeof(float), stream);
        route_bonus_active = true;
        route_bonus_dirty = affinity;
    }
    // Builds the memory tier over the lazily mapped experts and reads the initial selection.
    void prepare_ram_tier() {
        const char *value = std::getenv("STRATA_GLM_PREFER_RESIDENT_MIB");
        if (!value || ram_tier) return;
        const size_t mib = std::stoull(value);
        if (mib < 1024 || numa_experts.empty()) throw std::invalid_argument("GLM: prefer-resident needs lazy mapped expert rows and at least 1024 MiB");
        const int E = m.experts;
        ram_tier = std::make_unique<RamTier>((int)m.layers.size(), E, mib * MiB);
        ram_used.assign(m.layers.size() * (size_t)E, 0);
        auto parts = [this](int l) {
            const auto p = "blk." + std::to_string(l) + ".ffn_";
            return std::array<const strata::core::ArtifactTensor *, 3>{&artifact.at(p + "gate_exps.weight"),
                &artifact.at(p + "up_exps.weight"), &artifact.at(p + "down_exps.weight")};
        };
        for (const auto &[l, entry] : numa_experts) {
            size_t bytes = 0;
            for (const auto *t : parts(l)) {
                bytes += t->bytes / E;
                if (!ram_files.count(t->file)) {
#ifdef __linux__
                    const int fd = ::open(t->file->path().c_str(), O_RDONLY);
                    if (fd < 0) throw std::runtime_error("GLM: cannot open " + t->file->path());
                    ram_files[t->file] = fd;
#endif
                }
            }
            ram_tier->expert_bytes[l] = bytes;
        }
        ram_tier->ranges = [this, parts, E](int l, int e) {
            std::vector<RamTier::Range> out;
            for (const auto *t : parts(l)) {
                const size_t bytes = t->bytes / E, half = bytes / 2;
                const uint8_t *data = t->data() + (size_t)e * bytes;
                const uint64_t offset = t->file->data_start() + t->tensor->offset + (uint64_t)e * bytes;
                const int fd = ram_files.at(t->file);
                // Output rows split evenly between the nodes, as the node-owned jobs read them.
                out.push_back({data, half, fd, offset, 0});
                out.push_back({data + half, bytes - half, fd, offset + half, 1});
            }
            return out;
        };
        // Initial selection: the calibration prior's hottest experts, at least two selections' worth per layer.
        const auto prior = load_expert_prior();
        struct Item { double share; int layer, expert; };
        std::vector<Item> order;
        size_t selected = 0, count = 0;
        for (const auto &[l, entry] : numa_experts) {
            std::vector<Item> layer;
            for (int e = 0; e < E; ++e) layer.push_back({prior.empty() ? 1. / (e + 1) : prior[l][e], l, e});
            std::sort(layer.begin(), layer.end(), [](const Item &a, const Item &b) { return a.share != b.share ? a.share > b.share : a.expert < b.expert; });
            for (int i = 0; i < E; ++i) {
                if (i < 2 * m.top_k) {
                    ram_tier->request(l, layer[i].expert); selected += ram_tier->expert_bytes[l]; ++count;
                } else order.push_back(layer[i]);
            }
        }
        std::sort(order.begin(), order.end(), [](const Item &a, const Item &b) {
            return a.share != b.share ? a.share > b.share : a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
        });
        for (const auto &item : order) {
            if (selected + ram_tier->expert_bytes[item.layer] > ram_tier->budget) break;
            ram_tier->request(item.layer, item.expert); selected += ram_tier->expert_bytes[item.layer]; ++count;
        }
        const auto start = std::chrono::steady_clock::now();
        ram_tier->wait_idle();
        ram_tier->settle([](int) { return false; });
        route_bonus_dirty = true;
        std::cerr << "RAM_TIER budget_MiB=" << mib << " experts=" << count << " resident_MiB=" << ram_tier->resident_bytes / double(MiB)
                  << " read_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
    }
    // Routed experts stay warm; an expert the unbiased router wanted but could not have gains score too.
    void ram_touch(int l, const int *selected, const int *wanted, int routes) {
        if (!ram_tier || !ram_tier->expert_bytes[l]) return;
        // Scores count the unbiased router's choices: the routes taken favour residents and would only
        // confirm the current set.
        const size_t base = (size_t)l * m.experts;
        const int *unbiased = wanted ? wanted : selected;
        for (int j = 0; j < routes; ++j) {
            ram_tier->score[base + unbiased[j]] += 1.f;
            ram_tier->window[base + unbiased[j]] += 1.f;
            ram_used[base + selected[j]] = 1;
            // A selected expert outside the set is about to be read by the workers' page faults: it is resident
            // from now on, and the budget is restored at the end of the step.
            auto &state = ram_tier->state[base + selected[j]];
            if (state == RamTier::Absent) {
                state = RamTier::Resident;
                ram_tier->resident_bytes += ram_tier->expert_bytes[l];
                ++ram_misses; ram_miss_bytes += ram_tier->expert_bytes[l];
                route_bonus_dirty = true;
            }
        }
    }
    // Rebalance (STRATA_GLM_RAM_REBALANCE=N tokens; default 512 while the context is under 4096 tokens, then
    // 2048): the set becomes the experts with the most unbiased routes in the window (the long-term score breaks
    // ties), within the budget. Leaving experts are released first, then the missing ones read in the background.
    size_t ram_window_tokens = 0, ram_rebalances = 0, ram_last_rebalance = 0;
    const int kRamRebalance = (int)tier_setting("STRATA_GLM_RAM_REBALANCE", 0);
    const float kRamHistory = tier_setting("STRATA_GLM_RAM_HISTORY", 0.25f);
    // A rebalance needs this many tokens of evidence, and replaces at most this share of the budget: every replaced
    // expert is a disk read while decode runs.
    const int kRamMinWindow = (int)tier_setting("STRATA_GLM_RAM_MIN_WINDOW", 256);
    const float kRamChurn = tier_setting("STRATA_GLM_RAM_CHURN", 0.05f);
    void ram_rebalance() {
        auto &t = *ram_tier;
        std::vector<int> order;
        for (int index = 0; index < (int)t.state.size(); ++index)
            if (t.expert_bytes[index / t.experts]) order.push_back(index);
        auto value = [&](int i) { return t.window[i] + kRamHistory * t.score[i]; };
        std::sort(order.begin(), order.end(), [&](int a, int b) { return value(a) != value(b) ? value(a) > value(b) : a < b; });
        std::vector<uint8_t> target(t.state.size());
        size_t bytes = 0;
        // Every layer keeps at least two selections' worth, so routing always has a choice.
        std::vector<int> per_layer(t.layers);
        for (int index : order) {
            const int layer = index / t.experts;
            if (per_layer[layer] >= 2 * m.top_k) continue;
            target[index] = 1; bytes += t.expert_bytes[layer]; ++per_layer[layer];
        }
        for (int index : order) {
            if (target[index]) continue;
            if (bytes + t.expert_bytes[index / t.experts] > t.budget) continue;
            target[index] = 1; bytes += t.expert_bytes[index / t.experts];
        }
        // Joining in value order and leaving in reverse value order, up to the churn cap in bytes.
        const size_t cap = (size_t)(kRamChurn * t.budget);
        size_t moved = 0, leave = 0, join = 0;
        std::vector<int> joins, leaves;
        for (int index : order) if (target[index] && t.state[index] == RamTier::Absent) joins.push_back(index);
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            if (!target[*it] && t.state[*it] == RamTier::Resident && !ram_used[*it]) leaves.push_back(*it);
        size_t li = 0;
        for (int index : joins) {
            const size_t bytes_in = t.expert_bytes[index / t.experts];
            if (moved + bytes_in > cap) break;
            size_t freed = 0;
            while (freed < bytes_in && li < leaves.size()) {   // make room first, so the budget holds
                const int out = leaves[li++];
                freed += t.expert_bytes[out / t.experts]; t.evict(out); ++leave;
            }
            if (freed < bytes_in && t.resident_bytes + bytes_in > t.budget) break;
            if (t.request(index / t.experts, index % t.experts)) { ++join; moved += bytes_in; }
        }
        std::fill(t.window.begin(), t.window.end(), 0.f);
        ++ram_rebalances;
        route_bonus_dirty = true;
        std::cerr << "RAM_REBALANCE position=" << position << " leave=" << leave << " join=" << join << '\n';
    }
    // After each step: finished reads join, the coldest leave, and the most wanted absent experts start reading.
    void ram_plan() {
        if (!ram_tier) return;
        auto &t = *ram_tier;
        if (t.settle([&](int index) { return ram_used[index] != 0; })) route_bonus_dirty = true;
        const int interval = std::max(kRamMinWindow, kRamRebalance > 0 ? kRamRebalance : position < 4096 ? 512 : 2048);
        if ((size_t)position >= ram_last_rebalance + interval) {
            ram_rebalance();
            ram_last_rebalance = position;
        }
        std::fill(ram_used.begin(), ram_used.end(), 0);
        // Between rebalances only experts the unbiased router keeps asking for are read (admit threshold).
        const size_t pending = t.pending();
        if (pending < kRamInflight) {
            std::vector<int> wanted;
            for (int index = 0; index < (int)t.state.size(); ++index)
                if (t.state[index] == RamTier::Absent && t.score[index] >= kRamAdmit) wanted.push_back(index);
            const size_t n = std::min(wanted.size(), kRamInflight - pending);
            std::partial_sort(wanted.begin(), wanted.begin() + n, wanted.end(),
                              [&](int a, int b) { return t.score[a] != t.score[b] ? t.score[a] > t.score[b] : a < b; });
            for (size_t i = 0; i < n; ++i) t.request(wanted[i] / t.experts, wanted[i] % t.experts);
        }
        for (auto &v : t.score) v *= kRamDecay;
    }
    // STRATA_GLM_GATE_SKIP=thresholds.json (opt-in, lossy; canonical mode): {"layers":{"3":tau,...}} from
    // tools/glm_gate_thresholds.py. A routed expert's unit whose |silu(gate)| is below its layer's tau contributes
    // zero on both devices, and the CPU does not read that unit's up row.
    const std::vector<float> gate_skip = [] {
        std::vector<float> taus(64, 0.f);
        const char *path = std::getenv("STRATA_GLM_GATE_SKIP");
        if (!path || !*path) return taus;
        const auto json = strata::artifact::read_json(path);
        for (const auto &[layer, tau] : json.at("layers").object) {
            const float value = tau.real();
            if (std::stoul(layer) >= taus.size() || !(value >= 0.f && value < 1e3f)) throw std::invalid_argument("STRATA_GLM_GATE_SKIP: invalid entry");
            taus[std::stoul(layer)] = value;
        }
        return taus;
    }();
    // Strict combine: resident rows travel to the host and the CPU adds all routes in the canonical order, so logits
    // do not depend on placement. When routing itself follows residency (affinity, prefer-resident) that invariance
    // has nothing to protect, and the GPU adds its rows to the CPU's sum on the device instead: no row transfers.
    // STRATA_GLM_CANON_COMBINE=strict|device overrides.
    bool canon_strict() const {
        if (!route_bias_enabled && !ram_tier) return true; // unbiased quality reference
        static const char *mode = std::getenv("STRATA_GLM_CANON_COMBINE");
        if (mode && std::string(mode) == "strict") return true;
        if (mode && std::string(mode) == "device") return false;
        return !has_route_affinity() && !ram_tier;
    }
    bool canon_types(int gate, int down) const {
        return canon_experts && (gate == 10 || gate == 11) && (down == 10 || down == 11);
    }
    bool canon_layer(int l) const {
        const auto p = "blk." + std::to_string(l) + ".";
        return canon_types(artifact.at(p + "ffn_gate_exps.weight").tensor->type, artifact.at(p + "ffn_down_exps.weight").tensor->type);
    }
    std::vector<std::vector<float>> tier_score;
    // Expected routes per token of each expert when the tier was selected (prompt routes blended with the prior).
    // Scores start from it, so the first routed experts do not displace the selection before it has been used.
    std::vector<std::vector<float>> tier_rate;
    struct TierCopy {
        int layer = 0, expert = 0, staging = -1, victim_layer = -1, victim_expert = -1;
        std::unique_ptr<Device> entry;
        std::array<const uint8_t *, 3> source{};
        std::array<size_t, 3> sizes{};
        size_t bytes = 0;
        std::atomic<bool> staged{false};
        bool uploading = false;
        uint64_t selected_step = 0, uploaded_step = 0;
    };
    std::list<TierCopy> tier_copies;
    // Copies a promoted expert's bytes into pinned staging off the decode threads.
    struct TierStager {
        std::mutex mutex;
        std::condition_variable wake, idle;
        std::deque<std::pair<TierCopy *, uint8_t *>> jobs;
        bool busy = false, stopping = false;
        std::thread thread{[this] {
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                wake.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (stopping) return;
                auto [copy, dst] = jobs.front();
                jobs.pop_front();
                busy = true;
                lock.unlock();
                size_t offset = 0;
                for (int part = 0; part < 3; ++part) {
                    std::memcpy(dst + offset, copy->source[part], copy->sizes[part]);
                    offset += copy->sizes[part];
                }
                copy->staged.store(true, std::memory_order_release);
                lock.lock();
                busy = false;
                if (jobs.empty()) idle.notify_all();
            }
        }};
        void submit(TierCopy *copy, uint8_t *dst) {
            { std::lock_guard<std::mutex> lock(mutex); jobs.emplace_back(copy, dst); }
            wake.notify_one();
        }
        void wait_idle() {
            std::unique_lock<std::mutex> lock(mutex);
            idle.wait(lock, [&] { return jobs.empty() && !busy; });
        }
        ~TierStager() {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            wake.notify_one();
            thread.join();
        }
    };
    std::unique_ptr<Pinned> tier_staging;
    // Reverse member destruction joins the worker before releasing its target.
    std::unique_ptr<TierStager> tier_stager;
    std::vector<int> tier_free_staging;
    std::vector<cudaEvent_t> tier_events;
    cudaStream_t tier_stream = nullptr;
    uint64_t tier_swaps = 0;
    // Tier changes follow step counts, not completion times, so a run's residency sequence is reproducible: a
    // promotion selected in step s uploads after step s + 1 and joins the lookup at the start of step s + 3.
    uint64_t tier_step = 0;
    static constexpr size_t kTierStagingBytes = 16 * MiB;
    static float tier_setting(const char *name, float fallback) {
        const char *value = std::getenv(name);
        return value ? std::stof(value) : fallback;
    }
    const float kTierDecay = tier_setting("STRATA_GLM_TIER_DECAY", 0.98f);
    const float kTierAdmit = tier_setting("STRATA_GLM_TIER_ADMIT", 2.f);
    const float kTierMargin = tier_setting("STRATA_GLM_TIER_MARGIN", 1.5f);
    const size_t kTierInflight = (size_t)tier_setting("STRATA_GLM_TIER_INFLIGHT", 16);
    const float kTierSeed = tier_setting("STRATA_GLM_TIER_SEED", 50.f);   // 1 / (1 - decay): a steady rate's score
    // STRATA_GLM_TIER_DIRECT=1: the expert tensors' file pages are registered with the GPU once, and promotions
    // upload by DMA straight from the mapping. The staging thread's copy (a read and a write of every promoted
    // byte) no longer competes with the CPU experts for memory bandwidth.
    const bool tier_direct = [] { const char *v = std::getenv("STRATA_GLM_TIER_DIRECT"); return v && std::string(v) == "1"; }();
    struct HostRegistration {
        std::vector<uintptr_t> ranges;
        ~HostRegistration() { for (auto p : ranges) cudaHostUnregister((void *)p); }
    };
    std::unique_ptr<HostRegistration> tier_registration;
    void register_tier_sources() {
        const auto t0 = std::chrono::steady_clock::now();
        const uintptr_t page = 4096;
        std::vector<std::pair<uintptr_t, uintptr_t>> spans, merged;
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            for (const char *role : {"gate", "up", "down"}) {
                const auto &t = artifact.at("blk." + std::to_string(l) + ".ffn_" + role + "_exps.weight");
                const auto a = (uintptr_t)t.data();
                spans.push_back({a & ~(page - 1), (a + t.bytes + page - 1) & ~(page - 1)});
            }
        }
        std::sort(spans.begin(), spans.end());
        for (const auto &r : spans) {
            if (!merged.empty() && r.first <= merged.back().second) merged.back().second = std::max(merged.back().second, r.second);
            else merged.push_back(r);
        }
        auto reg = std::make_unique<HostRegistration>();
        size_t bytes = 0;
        for (const auto &r : merged) {
            const auto e = cudaHostRegister((void *)r.first, r.second - r.first, cudaHostRegisterReadOnly);
            if (e != cudaSuccess) throw std::runtime_error(std::string("GLM: tier source registration failed: ") + cudaGetErrorString(e));
            reg->ranges.push_back(r.first);
            bytes += r.second - r.first;
        }
        tier_registration = std::move(reg);
        std::cerr << "TIER_DIRECT registered_MiB=" << bytes / 1048576 << " ranges=" << merged.size() << " ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << '\n';
    }
    void prepare_tier_staging() {
        if (!tier_adapt || tier_staging) return;
        if (tier_direct && !tier_registration) register_tier_sources();
        // Pinned staging and CUDA objects are created here, before the step launches: allocating them while
        // the GPU waits on the service thread could synchronize with that wait.
        if (!tier_staging) {
            tier_staging = std::make_unique<Pinned>(kTierInflight * kTierStagingBytes);
            for (size_t i = 0; i < kTierInflight; ++i) {
                tier_free_staging.push_back((int)i);
                cudaEvent_t event;
                check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                tier_events.push_back(event);
            }
            check(cudaStreamCreateWithFlags(&tier_stream, cudaStreamNonBlocking));
            tier_stager = std::make_unique<TierStager>();
        }
    }
    void tier_begin_step() {
        if (!tier_adapt || !pipelining || sequence_speculating) return;
        if (tier_score.empty()) {
            tier_score.assign(m.layers.size(), std::vector<float>(m.experts));
            if (tier_rate.size() == tier_score.size())
                for (size_t l = 0; l < tier_score.size(); ++l)
                    for (int e = 0; e < m.experts; ++e) tier_score[l][e] = kTierSeed * tier_rate[l][e];
        }
        prepare_tier_staging();
        ++tier_step;
        for (auto &layer : tier_score) for (auto &v : layer) v *= kTierDecay;
        for (auto it = tier_copies.begin(); it != tier_copies.end();) {
            if (!it->uploading || tier_step < it->uploaded_step + 2) { ++it; continue; }
            check(cudaEventSynchronize(tier_events[it->staging]));
            if (std::getenv("STRATA_GLM_CHECK_TIER_UPLOADS")) {
                std::vector<uint8_t> actual(it->bytes);
                check(cudaMemcpy(actual.data(), it->entry->p, it->bytes, cudaMemcpyDeviceToHost));
                size_t offset = 0;
                for (int part = 0; part < 3; ++part) {
                    if (std::memcmp(actual.data() + offset, it->source[part], it->sizes[part]))
                        throw std::runtime_error("GLM: tier upload differs from source expert bytes");
                    offset += it->sizes[part];
                }
            }
            decode_resident[it->layer].emplace(it->expert, std::move(it->entry));
            tier_free_staging.push_back(it->staging);
            it = tier_copies.erase(it);
        }
    }
    // Called by the CPU service before it releases layer l, so the GPU's later lookup copies see the change.
    void tier_select(int l, const int *selected, int routes, const unsigned long long *) {
        if (!tier_adapt || !tier_staging) return;
        auto &score = tier_score[l];
        for (int j = 0; j < routes; ++j) score[selected[j]] += 1.f;
    }
    // After a step: pairs the hottest non-resident experts with the coldest resident ones, as many as staging
    // slots are free. A victim leaves decode_resident here, so the next step's lookup no longer lists it.
    void tier_plan() {
        if (!tier_adapt || !tier_staging || sequence_speculating || tier_free_staging.empty() || tier_score.empty()) return;
        struct Item { float score; int layer, expert; };
        std::vector<Item> wanted, cold;
        for (size_t l = 0; l < tier_score.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe || (canon_experts && !canon_layer((int)l))) continue;
            const auto &score = tier_score[l];
            for (int e = 0; e < m.experts; ++e)
                if (score[e] >= kTierAdmit && !decode_resident[l].count(e)) wanted.push_back({score[e], (int)l, e});
            for (const auto &[e, slot] : decode_resident[l]) cold.push_back({score[e], (int)l, e});
        }
        for (const auto &c : tier_copies)
            std::erase_if(wanted, [&](const Item &w) { return w.layer == c.layer && w.expert == c.expert; });
        const size_t n = std::min({wanted.size(), cold.size(), tier_free_staging.size()});
        if (!n) return;
        const auto hotter = [](const Item &a, const Item &b) {
            return a.score != b.score ? a.score > b.score : a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
        };
        const auto colder = [](const Item &a, const Item &b) {
            return a.score != b.score ? a.score < b.score : a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
        };
        std::partial_sort(wanted.begin(), wanted.begin() + n, wanted.end(), hotter);
        std::partial_sort(cold.begin(), cold.begin() + n, cold.end(), colder);
        for (size_t i = 0; i < n; ++i) {
            const auto &w = wanted[i], &v = cold[i];
            if (w.score <= kTierMargin * v.score) break;
            const auto p = "blk." + std::to_string(w.layer) + ".";
            const auto &g = artifact.at(p + "ffn_gate_exps.weight"), &u = artifact.at(p + "ffn_up_exps.weight"),
                       &d = artifact.at(p + "ffn_down_exps.weight");
            const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
            const size_t bytes = sizes[0] + sizes[1] + sizes[2];
            auto found = decode_resident[v.layer].find(v.expert);
            if (bytes > kTierStagingBytes || found->second->bytes != bytes) continue;
            auto &copy = tier_copies.emplace_back();
            copy.layer = w.layer; copy.expert = w.expert; copy.entry = std::move(found->second);
            copy.victim_layer = v.layer; copy.victim_expert = v.expert;
            copy.sizes = sizes; copy.bytes = bytes;
            copy.source = {g.data() + w.expert * sizes[0], u.data() + w.expert * sizes[1], d.data() + w.expert * sizes[2]};
            copy.staging = tier_free_staging.back();
            tier_free_staging.pop_back();
            copy.selected_step = tier_step;
            decode_resident[v.layer].erase(found);
            if (tier_direct) copy.staged.store(true, std::memory_order_release);
            else tier_stager->submit(&copy, (uint8_t *)tier_staging->p + (size_t)copy.staging * kTierStagingBytes);
        }
    }
    // After the step's final synchronization: every freed slot is idle, so staged copies may upload.
    void tier_submit() {
        for (auto &copy : tier_copies) {
            if (copy.uploading || copy.selected_step >= tier_step) continue;
            while (!copy.staged.load(std::memory_order_acquire)) std::this_thread::yield();
            copy.uploaded_step = tier_step;
            if (tier_direct) {
                size_t offset = 0;
                for (int part = 0; part < 3; ++part) {
                    check(cudaMemcpyAsync((uint8_t *)copy.entry->p + offset, copy.source[part], copy.sizes[part],
                                          cudaMemcpyHostToDevice, tier_stream));
                    offset += copy.sizes[part];
                }
            } else
                check(cudaMemcpyAsync(copy.entry->p, (uint8_t *)tier_staging->p + (size_t)copy.staging * kTierStagingBytes,
                                      copy.bytes, cudaMemcpyHostToDevice, tier_stream));
            check(cudaEventRecord(tier_events[copy.staging], tier_stream));
            copy.uploading = true;
            ++tier_swaps;
        }
    }
    void tier_drain() {
        if (tier_stager) tier_stager->wait_idle();
        if (tier_stream) check(cudaStreamSynchronize(tier_stream));
        // An upload that finished holds its new expert; a slot not yet uploaded still holds its victim.
        for (auto &copy : tier_copies) {
            if (copy.uploading) decode_resident[copy.layer].emplace(copy.expert, std::move(copy.entry));
            else decode_resident[copy.victim_layer].emplace(copy.victim_expert, std::move(copy.entry));
            tier_free_staging.push_back(copy.staging);
        }
        tier_copies.clear();
        tier_score.clear();
    }
    std::unique_ptr<GpuWorker> launcher;
    std::future<void> launch_pending;
    std::atomic<bool> launch_failed{false};
    // One CPU expert pass: a MoE layer's mailbox slot, the tokens it holds and the first one's position.
    struct ServiceItem { int layer, slot, tokens, position; };
    void service_pipeline(const std::vector<ServiceItem> &items) {
        for (size_t i = 0; i < items.size(); ++i) {
            const auto &item = items[i];
            check_stop();
            mailbox->wait_published(item.slot, launch_failed);
            if (merge_expert_groups && canon_layer(item.layer) && i + 1 < items.size() && items[i + 1].layer == item.layer) {
                const auto &other = items[++i];
                if (!canon_experts || !canon_strict() || ram_tier || remote_tp || !row_sequences.empty() || step_trace || defer_routes ||
                    other.position != item.position + item.tokens || other.slot != item.slot + mailbox_group_stride())
                    throw std::invalid_argument("GLM: expert group merge requires single-sequence canonical strict split verification without tracing or RAM routing");
                // Group B's mixer runs while the CPU waits here. Its experts
                // then share one weight read with group A whenever routes overlap.
                mailbox->wait_published(other.slot, launch_failed);
                cpu_moe_mailbox(item.layer, item.slot, item.tokens, item.position, other.slot, other.tokens);
                if (merge_expert_groups != 2) {
                    mailbox->complete(item.slot);
                    mailbox->complete(other.slot);
                }
                continue;
            }
            trace_seen();
            if (!(defer_active && defer_routes)) {
                cpu_moe_mailbox(item.layer, item.slot, item.tokens, item.position);
                mailbox->complete(item.slot);
            } else {
                // Partial deferral: the n lowest-weight cold routes form the late pass (group-B slot).
                const int K = m.top_k, late_slot = item.slot + mailbox_group_stride();
                const int *ids = mailbox->ids(item.slot);
                const float *w = mailbox->weights(item.slot);
                const auto *resident = (const unsigned long long *)cached_decode->host_lookup->p + (size_t)item.layer * 288;
                unsigned cold = 0, late = 0;
                float total = 0.f;
                for (int j = 0; j < item.tokens * K; ++j) {
                    total += w[j];
                    if (w[j] != 0.f && !resident[ids[j]]) cold |= 1u << j;
                }
                for (int n = 0; n < defer_routes && item.layer >= defer_min_layer; ++n) {
                    int pick = -1;
                    for (int j = 0; j < item.tokens * K; ++j)
                        if ((cold >> j & 1u) && !(late >> j & 1u) && (pick < 0 || w[j] < w[pick])) pick = j;
                    if (pick < 0 || w[pick] >= defer_max_share * total) break;
                    late |= 1u << pick;
                }
                if ((late & ~cold) || __builtin_popcount(late) > std::min(defer_routes, __builtin_popcount(cold)))
                    throw std::runtime_error("GLM: deferred route selection");
                cpu_moe_mailbox(item.layer, item.slot, item.tokens, item.position, -1, 0, ~late);
                mailbox->complete(item.slot);
                if (late) cpu_moe_mailbox(item.layer, item.slot, item.tokens, item.position, -1, 0, late, late_slot);
                else std::fill_n(mailbox->sum(late_slot), (size_t)item.tokens * m.hidden, 0.f);  // the late add is unconditional
                mailbox->complete(late_slot);
            }
            trace_done(item.slot);
        }
    }
    // The launcher thread enqueues the whole step while this thread runs each MoE layer's CPU experts.
    template <class F> void run_pipelined_step(F &enqueue, int split = 0) {
        // The service order is the publication order: per MoE layer, group A then (split verify) group B.
        std::vector<ServiceItem> layers;
        for (size_t l = 0; l < m.layers.size(); ++l)
            if (m.layers[l].ffn == strata::core::FfnKind::Moe) {
                if (!split) layers.push_back({(int)l, (int)l, batch_tokens, position});
                else {
                    layers.push_back({(int)l, (int)l, split, position});
                    layers.push_back({(int)l, (int)l + mailbox_group_stride(), batch_tokens - split, position + split});
                }
            }
        if (!launcher) launcher = std::make_unique<GpuWorker>(primary_device);
        launch_failed = false;
        launch_pending = launcher->submit([&] {
            try {
                enqueue();
            } catch (...) {
                launch_failed = true;
                mailbox->abort();
                throw;
            }
            if (step_trace)
                trace.enqueue_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trace.start).count();
        });
        try {
            service_pipeline(layers);
        } catch (...) {
            // Aborted waits drain the stream, so the launcher finishes before its captures go out of scope.
            mailbox->abort();
            launch_pending.wait();
            if (launch_failed) launch_pending.get();  // the launcher's own error explains the stop
            throw;
        }
        launch_pending.get();
    }
    // Releases an interrupted step: pending GPU waits return without their sums, the launcher finishes
    // enqueueing, and the stream drains before any buffer is reused.
    struct PipelineGuard {
        Decoder &decoder;
        ~PipelineGuard() {
            if (!decoder.pipelining) return;
            decoder.pipelining = false;
            decoder.mailbox->abort();
            if (decoder.launch_pending.valid()) decoder.launch_pending.wait();
            cudaStreamSynchronize(decoder.stream);
            cudaGetLastError();
        }
    };
    // Upper-half rows of a quantized or FP32 matrix, byte-identical to the primary copy.
    static std::unique_ptr<Device> upper_rows(const strata::core::ArtifactTensor &t, int rows, int first) {
        if (t.tensor->shape.size() != 2 || (int)t.tensor->shape[1] != rows || t.bytes % rows)
            throw std::invalid_argument("GLM: tensor parallel row split needs whole rows");
        const size_t row_bytes = t.bytes / rows;
        auto copy = std::make_unique<Device>((size_t)(rows - first) * row_bytes);
        copy->put(t.data() + (size_t)first * row_bytes, copy->bytes);
        return copy;
    }
    void setup_decode_tp() {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads, half = heads / 2;
        if (heads % 2 || m.conv_kernel < 2) throw std::invalid_argument("GLM: tensor parallel decode needs even heads");
        auto &other = *secondary;
        DeviceScope scope(other.device);
        other.tp = std::make_unique<SecondaryPrefill::TensorParallel>();
        auto &tp = *other.tp;
        // An event belongs to the device whose stream records it; the other device only waits on it.
        { DeviceScope primary(primary_device); check(cudaEventCreateWithFlags(&tp.x_ready, cudaEventDisableTiming)); }
        check(cudaEventCreateWithFlags(&tp.y_ready, cudaEventDisableTiming));
        tp.recurrent.resize(m.layers.size());
        for (auto &c : tp.conv) c.resize(m.layers.size());
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].mixer != strata::core::MixerKind::Kda) continue;
            const auto p = "blk." + std::to_string(l) + ".";
            for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_f_b.weight", "ssm_g_b.weight"})
                tp.weights.emplace(p + name, upper_rows(artifact.at(p + name), n, n / 2));
            tp.weights.emplace(p + "ssm_beta.weight", upper_rows(artifact.at(p + "ssm_beta.weight"), heads, half));
            for (const char *name : {"ssm_f_a.weight", "ssm_g_a.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight",
                                     "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"}) {
                const auto &t = artifact.at(p + name);
                auto copy = std::make_unique<Device>(t.bytes);
                copy->put(t.data(), t.bytes);
                tp.weights.emplace(p + name, std::move(copy));
            }
            tp.recurrent[l] = std::make_unique<Device>((size_t)half * dim * dim * 4);
            for (auto &c : tp.conv) c[l] = std::make_unique<Device>((size_t)(n / 2) * (m.conv_kernel - 1) * 4);
        }
        tp.x = std::make_unique<Device>((size_t)m.hidden * 4);
        tp.tmp = std::make_unique<Device>((size_t)n / 2 * 4);
        for (auto &b : tp.proj) b = std::make_unique<Device>((size_t)n / 2 * 4);
        tp.low = std::make_unique<Device>((size_t)m.linear_dim * 4);
        tp.decay = std::make_unique<Device>((size_t)n / 2 * 4);
        tp.beta = std::make_unique<Device>((size_t)half * 4);
        tp.y = std::make_unique<Device>((size_t)n / 2 * 4);
        tp.q8 = std::make_unique<Device>(k::native_q8_1_bytes(std::max((int)m.hidden, (int)m.linear_dim)));
        check(cudaStreamSynchronize(other.stream));
        std::cerr << "DECODE_TP device=" << other.device << " layers=" << tp.weights.size() / 14
                  << " MiB=" << Device::live() / double(MiB) << '\n';
    }
    bool tp_active() const {
        return decode_tp && secondary && secondary->tp && secondary_enabled && !fast && batch_tokens == 1 &&
               !capturing_history && !capture_hidden;
    }
    // Primary states carry the authoritative values after prefill or a rollback; refresh the secondary halves.
    void tp_sync_states() {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads;
        auto &other = *secondary;
        auto &tp = *other.tp;
        check(cudaStreamSynchronize(stream));
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].mixer != strata::core::MixerKind::Kda) continue;
            auto &state = states[l];
            const size_t state_half = (size_t)heads / 2 * dim * dim * 4, conv_half = (size_t)n / 2 * (m.conv_kernel - 1) * 4;
            check(cudaMemcpyPeerAsync(tp.recurrent[l]->p, other.device, (char *)state.recurrent->p + state_half,
                                      primary_device, state_half, stream));
            const std::array<Device *, 3> hist = {state.conv_q.get(), state.conv_k.get(), state.conv_v.get()};
            for (int i = 0; i < 3; ++i)
                check(cudaMemcpyPeerAsync(tp.conv[i][l]->p, other.device, (char *)hist[i]->p + conv_half,
                                          primary_device, conv_half, stream));
        }
        check(cudaStreamSynchronize(stream));
        tp_stale = false;
    }
    // The secondary half of one KDA mixer: x arrives from the primary, its half of q leaves for the primary.
    void tp_kda_remote(const std::string &p, int l, const float *x_primary, float *q_primary) {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads, half = heads / 2;
        auto &other = *secondary;
        auto &tp = *other.tp;
        cudaStream_t s = other.stream;
        auto weight_of = [&](const std::string &name) -> Device & { return *tp.weights.at(p + name); };
        auto project = [&](const std::string &name, const float *in, float *out, int width, int rows) {
            const auto &t = *artifact.at(p + name).tensor;
            if (!k::native_mmvq_supported(t.type)) throw std::runtime_error("GLM: tensor parallel needs quantized rows: " + name);
            k::native_quantize_q8_1(in, tp.q8->p, width, 1, s);
            k::native_mmvq(t.type, weight_of(name).p, tp.q8->p, out, width, rows, 1, s);
        };
        // Peer copies cannot be captured; with peer access and UVA a plain device copy crosses devices.
        check(cudaMemcpyAsync(tp.x->p, x_primary, (size_t)m.hidden * 4, cudaMemcpyDeviceToDevice, s));
        const std::array<std::string, 3> names = {"q", "k", "v"};
        for (int i = 0; i < 3; ++i) {
            project("attn_" + names[i] + ".weight", tp.x->f(), tp.tmp->f(), m.hidden, n / 2);
            k::glm_conv(tp.tmp->f(), weight_of("ssm_conv1d_" + names[i] + ".weight").f((size_t)n / 2 * m.conv_kernel),
                        tp.conv[i][l]->f(), tp.proj[i]->f(), n / 2, m.conv_kernel, s);
        }
        project("ssm_f_a.weight", tp.x->f(), tp.low->f(), m.hidden, dim);
        project("ssm_f_b.weight", tp.low->f(), tp.tmp->f(), dim, n / 2);
        k::glm_kda_gate(tp.tmp->f(), weight_of("ssm_dt.bias").f((size_t)n / 2), weight_of("ssm_a").f(half), tp.decay->f(),
                        half, dim, m.gate_lower_bound, s);
        project("ssm_beta.weight", tp.x->f(), tp.beta->f(), m.hidden, half);
        k::glm_kda_step(tp.recurrent[l]->f(), tp.proj[0]->f(), tp.proj[1]->f(), tp.proj[2]->f(), tp.decay->f(),
                        tp.beta->f(), tp.y->f(), half, dim, s);
        project("ssm_g_a.weight", tp.x->f(), tp.low->f(), m.hidden, dim);
        project("ssm_g_b.weight", tp.low->f(), tp.tmp->f(), dim, n / 2);
        k::glm_kda_output(tp.y->f(), tp.tmp->f(), weight_of("ssm_norm.weight").f(), tp.proj[0]->f(), half, dim,
                          m.rms_epsilon, s);
        check(cudaMemcpyAsync(q_primary + n / 2, tp.proj[0]->p, (size_t)n / 2 * 4, cudaMemcpyDeviceToDevice, s));
    }
    template<class F> void tp_graph(int layer, F &&enqueue) {
        auto &other = *secondary;
        if (!decode_graphs) { enqueue(); return; }
        auto &entry = other.tp->graphs[layer];
        if (!entry) entry = std::make_unique<DecodeGraph>();
        if (!entry->warmed) { enqueue(); entry->warmed = true; return; }
        if (!entry->executable) {
            check(cudaStreamBeginCapture(other.stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(other.stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                cudaGetLastError();
                throw;
            }
            check(cudaStreamEndCapture(other.stream, &entry->graph));
            check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr, nullptr, 0));
        }
        check(cudaGraphLaunch(entry->executable, other.stream));
    }
    bool kda_split_active() const {
        return kda_split && fast && secondary && secondary->tp && secondary_enabled && secondary->gpu &&
               secondary->gpu->gemm && !capturing_history;
    }
    // Secondary halves of the KDA states are authoritative after a split prefill; mirror them to the primary.
    void kda_split_sync_back() {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads;
        auto &other = *secondary;
        auto &tp = *other.tp;
        check(cudaStreamSynchronize(other.stream));
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (!tp.recurrent[l]) continue;
            auto &state = states[l];
            const size_t state_half = (size_t)heads / 2 * dim * dim * 4, conv_half = (size_t)n / 2 * (m.conv_kernel - 1) * 4;
            check(cudaMemcpyPeerAsync((char *)state.recurrent->p + state_half, primary_device, tp.recurrent[l]->p,
                                      other.device, state_half, stream));
            const std::array<Device *, 3> hist = {state.conv_q.get(), state.conv_k.get(), state.conv_v.get()};
            for (int i = 0; i < 3; ++i)
                check(cudaMemcpyPeerAsync((char *)hist[i]->p + conv_half, primary_device, tp.conv[i][l]->p,
                                          other.device, conv_half, stream));
        }
        check(cudaStreamSynchronize(stream));
        tp_stale = false;
    }
    // Upper-half KDA mixer of one prefill batch on the secondary: FP16 x in, FP32 upper-half output rows out.
    void kda_split_remote(const std::string &p, int l, const uint16_t *x_half_primary, float *y_upper_primary) {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads, half = heads / 2, nt = batch_tokens;
        auto &other = *secondary;
        auto &tp = *other.tp;
        cudaStream_t s = other.stream;
        auto weight_of = [&](const std::string &name) -> Device & { return *tp.weights.at(p + name); };
        const size_t room = (size_t)other.gpu->chunk * 8 * 4096;
        float *a = other.buf("packed", room).f(), *b = other.buf("result", room).f();
        auto *xh = (uint16_t *)a; a += ((size_t)nt * m.hidden + 1) / 2;
        float *tmp = a; a += (size_t)nt * n / 2;
        float *q = a; a += (size_t)nt * n / 2;
        float *key = a; a += (size_t)nt * n / 2;
        float *v = a; a += (size_t)nt * n / 2;
        float *decay = b; b += (size_t)nt * n / 2;
        float *y = b; b += (size_t)nt * n / 2;
        float *low = b; b += (size_t)nt * dim;
        float *beta = b; b += (size_t)nt * half;
        float *qi = b; b += (size_t)nt * half;
        auto *lowh = (uint16_t *)b; b += ((size_t)nt * dim + 1) / 2;
        if ((size_t)(a - other.buf("packed", room).f()) > room || (size_t)(b - other.buf("result", room).f()) > room)
            throw std::runtime_error("GLM: KDA split workspace exceeds the secondary buffers");
        auto project = [&](const std::string &name, const uint16_t *in, float *out, int width, int rows) {
            const auto &t = *artifact.at(p + name).tensor;
            other.gpu->gemm->native_chunked(in, t.type, weight_of(name).p, out, nt, rows, width, 2048);
        };
        check(cudaMemcpyPeerAsync(xh, other.device, x_half_primary, primary_device, (size_t)nt * m.hidden * 2, s));
        const std::array<std::string, 3> names = {"q", "k", "v"};
        const std::array<float *, 3> dest = {q, key, v};
        for (int i = 0; i < 3; ++i) {
            project("attn_" + names[i] + ".weight", xh, tmp, m.hidden, n / 2);
            k::glm_conv_batch(tmp, weight_of("ssm_conv1d_" + names[i] + ".weight").f((size_t)n / 2 * m.conv_kernel),
                              tp.conv[i][l]->f(), dest[i], n / 2, m.conv_kernel, nt, s);
        }
        project("ssm_f_a.weight", xh, low, m.hidden, dim);
        k::glm_f16(low, lowh, (int64_t)nt * dim, s);
        project("ssm_f_b.weight", lowh, tmp, dim, n / 2);
        k::glm_kda_gate_batch(tmp, weight_of("ssm_dt.bias").f((size_t)n / 2), weight_of("ssm_a").f(half), decay, half, dim,
                              m.gate_lower_bound, nt, s);
        project("ssm_beta.weight", xh, beta, m.hidden, half);
        const bool prepared = prepare_kda_inputs;
        if (prepared) k::glm_kda_prepare(q, key, decay, beta, qi, half, nt, s);
        for (int t = 0; t < nt; t += 64) {
            const int count = std::min(64, nt - t);
            k::glm_kda_chunk(tp.recurrent[l]->f(), q + (size_t)t * n / 2, key + (size_t)t * n / 2, v + (size_t)t * n / 2,
                             decay + (size_t)t * n / 2, beta + (size_t)t * half, y + (size_t)t * n / 2, half, dim, count,
                             s, kda_columns, (prefill_kda_row_parts ? prefill_kda_row_parts : kda_row_parts), prepared ? qi + (size_t)t * half : nullptr);
        }
        project("ssm_g_a.weight", xh, low, m.hidden, dim);
        k::glm_f16(low, lowh, (int64_t)nt * dim, s);
        project("ssm_g_b.weight", lowh, tmp, dim, n / 2);
        k::glm_kda_output_batch(y, tmp, weight_of("ssm_norm.weight").f(), q, half, dim, m.rms_epsilon, nt, s);
        check(cudaMemcpyPeerAsync(y_upper_primary, primary_device, q, other.device, (size_t)nt * n / 2 * 4, s));
        check(cudaEventRecord(other.finished, s));
    }
    void kda(const std::string &p, int l, const float *x, float *out) {
        const int dim = m.linear_dim, heads = m.linear_heads, n = dim * heads;
        const int row_parts = fast && prefill_kda_row_parts ? prefill_kda_row_parts : kda_row_parts;
        auto &state = states[l];
        auto &q = buf("linear_q", n * batch_tokens);
        auto &key = buf("linear_k", n * batch_tokens);
        auto &v = buf("linear_v", n * batch_tokens);
        auto &tmp = buf("linear_raw", n * batch_tokens);
        const std::array<std::string, 3> names = {"q", "k", "v"};
        const std::array<float *, 3> dest = {q.f(), key.f(), v.f()};
        const std::array<Device *, 3> hist = {state.conv_q.get(), state.conv_k.get(), state.conv_v.get()};
        share_quant(x, m.hidden);
        if (kda_split_active()) {
            const int half = heads / 2, nt = batch_tokens;
            auto &other = *secondary;
            auto &xh = buf("kda_x_half", ((size_t)nt * m.hidden + 1) / 2);
            k::glm_f16(x, (uint16_t *)xh.p, (int64_t)nt * m.hidden, stream);
            auto &upper = buf("kda_y_upper", (size_t)nt * n / 2);
            check(cudaEventRecord(mla_ready, stream));
            auto *xh_ptr = (const uint16_t *)xh.p;
            float *upper_ptr = upper.f();
            std::future<void> pending = other.worker->submit([=, this, &other] {
                check(cudaStreamWaitEvent(other.stream, mla_ready, 0));
                kda_split_remote(p, l, xh_ptr, upper_ptr);
            });
            try {
                auto project = [&](const std::string &name, const uint16_t *in, float *y, int width, int rows) {
                    const auto &t = *artifact.at(name).tensor;
                    gpu->gemm->native_chunked(in, t.type, weight(name).p, y, nt, rows, width, 2048);
                };
                for (int i = 0; i < 3; ++i) {
                    project(p + "attn_" + names[i] + ".weight", xh_ptr, tmp.f(), m.hidden, n / 2);
                    k::glm_conv_batch(tmp.f(), weight(p + "ssm_conv1d_" + names[i] + ".weight").f(), hist[i]->f(),
                                      dest[i], n / 2, m.conv_kernel, nt, stream);
                }
                auto &low = buf("linear_low", (size_t)dim * nt);
                auto &lowh = buf("linear_low_half", ((size_t)dim * nt + 1) / 2);
                auto &decay = buf("linear_decay", (size_t)n / 2 * nt);
                auto &beta = buf("linear_beta", (size_t)half * nt);
                project(p + "ssm_f_a.weight", xh_ptr, low.f(), m.hidden, dim);
                k::glm_f16(low.f(), (uint16_t *)lowh.p, (int64_t)nt * dim, stream);
                project(p + "ssm_f_b.weight", (const uint16_t *)lowh.p, tmp.f(), dim, n / 2);
                k::glm_kda_gate_batch(tmp.f(), weight(p + "ssm_dt.bias").f(), weight(p + "ssm_a").f(), decay.f(), half,
                                      dim, m.gate_lower_bound, nt, stream);
                project(p + "ssm_beta.weight", xh_ptr, beta.f(), m.hidden, half);
                auto &y = buf("linear_y", (size_t)n / 2 * nt);
                const bool prepared = prepare_kda_inputs;
                float *qi = prepared ? buf("linear_qi", (size_t)half * nt).f() : nullptr;
                if (prepared) k::glm_kda_prepare(q.f(), key.f(), decay.f(), beta.f(), qi, half, nt, stream);
                for (int t = 0; t < nt; t += 64) {
                    check_stop();
                    const int count = std::min(64, nt - t);
                    k::glm_kda_chunk(state.recurrent->f(), q.f((size_t)t * n / 2), key.f((size_t)t * n / 2),
                                     v.f((size_t)t * n / 2), decay.f((size_t)t * n / 2), beta.f((size_t)t * half),
                                     y.f((size_t)t * n / 2), half, dim, count, stream, kda_columns, row_parts,
                                     prepared ? qi + (size_t)t * half : nullptr);
                }
                project(p + "ssm_g_a.weight", xh_ptr, low.f(), m.hidden, dim);
                k::glm_f16(low.f(), (uint16_t *)lowh.p, (int64_t)nt * dim, stream);
                project(p + "ssm_g_b.weight", (const uint16_t *)lowh.p, tmp.f(), dim, n / 2);
                k::glm_kda_output_batch(y.f(), tmp.f(), weight(p + "ssm_norm.weight").f(), q.f(), half, dim,
                                        m.rms_epsilon, nt, stream);
            } catch (...) { if (pending.valid()) pending.wait(); throw; }
            pending.get();
            check(cudaStreamWaitEvent(stream, other.finished, 0));
            // Interleave both halves into [token][head][dim] rows for the output projection.
            auto &full = buf("kda_q_full", (size_t)n * nt);
            check(cudaMemcpy2DAsync(full.p, (size_t)n * 4, q.p, (size_t)n / 2 * 4, (size_t)n / 2 * 4, nt,
                                    cudaMemcpyDeviceToDevice, stream));
            check(cudaMemcpy2DAsync((char *)full.p + (size_t)n / 2 * 4, (size_t)n * 4, upper.p, (size_t)n / 2 * 4,
                                    (size_t)n / 2 * 4, nt, cudaMemcpyDeviceToDevice, stream));
            end_share_quant();
            mat(p + "attn_output.weight", full.f(), out);
            return;
        }
        if (tp_active()) {
            // Lower half of the heads here; the upper half runs on the secondary and lands in q.
            const int half = heads / 2;
            auto rows = [&](const std::string &name, const float *in, float *out, int count) {
                const auto &t = *artifact.at(name).tensor;
                const int in_width = (int)t.shape[0];
                if (!k::native_mmvq_supported(t.type)) throw std::runtime_error("GLM: tensor parallel needs quantized rows: " + name);
                auto &W = weight(name);
                if (shared_quant && in == shared_quant_src && in_width == shared_quant_in) {
                    k::native_mmvq(t.type, W.p, shared_quant->p, out, in_width, count, 1, stream);
                    return;
                }
                auto &q8 = buf("q8", k::native_q8_1_bytes(std::max(in_width, n), 1) / 4);
                k::native_quantize_q8_1(in, q8.p, in_width, 1, stream);
                k::native_mmvq(t.type, W.p, q8.p, out, in_width, count, 1, stream);
            };
            for (int i = 0; i < 3; ++i) {
                rows(p + "attn_" + names[i] + ".weight", x, tmp.f(), n / 2);
                k::glm_conv(tmp.f(), weight(p + "ssm_conv1d_" + names[i] + ".weight").f(), hist[i]->f(), dest[i],
                            n / 2, m.conv_kernel, stream);
            }
            auto &low = buf("linear_low", dim);
            auto &decay = buf("linear_decay", n / 2);
            auto &beta = buf("linear_beta", half);
            rows(p + "ssm_f_a.weight", x, low.f(), dim);
            rows(p + "ssm_f_b.weight", low.f(), tmp.f(), n / 2);
            k::glm_kda_gate(tmp.f(), weight(p + "ssm_dt.bias").f(), weight(p + "ssm_a").f(), decay.f(), half, dim,
                            m.gate_lower_bound, stream);
            rows(p + "ssm_beta.weight", x, beta.f(), half);
            auto &y = buf("linear_y", n / 2);
            k::glm_kda_step(state.recurrent->f(), q.f(), key.f(), v.f(), decay.f(), beta.f(), y.f(), half, dim, stream);
            rows(p + "ssm_g_a.weight", x, low.f(), dim);
            rows(p + "ssm_g_b.weight", low.f(), tmp.f(), n / 2);
            k::glm_kda_output(y.f(), tmp.f(), weight(p + "ssm_norm.weight").f(), q.f(), half, dim, m.rms_epsilon, stream);
            cudaStreamCaptureStatus status;
            check(cudaStreamIsCapturing(stream, &status));
            check(cudaStreamWaitEvent(stream, secondary->tp->y_ready,
                                      status == cudaStreamCaptureStatusActive ? cudaEventWaitExternal : 0));
            end_share_quant();
            mat(p + "attn_output.weight", q.f(), out);
            return;
        }
        for (int i = 0; i < 3; ++i) {
            mat(p + "attn_" + names[i] + ".weight", x, tmp.f());
            auto &W = weight(p + "ssm_conv1d_" + names[i] + ".weight");
            if (fast)
                k::glm_conv_batch(tmp.f(), W.f(), hist[i]->f(), dest[i], n, m.conv_kernel, batch_tokens,
                                  stream);
            else
                for (int t = 0; t < batch_tokens; ++t) {
                    Device *history = hist[i];
                    if (!row_sequences.empty()) {
                        auto &rs = row_state(t, l);
                        history = i == 0 ? rs.conv_q.get() : i == 1 ? rs.conv_k.get() : rs.conv_v.get();
                    }
                    k::glm_conv(tmp.f(t * n), W.f(), history->f(), dest[i] + t * n, n, m.conv_kernel, stream);
                    capture_state(history, t);
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
        if (!row_sequences.empty()) {
            for (int t = 0; t < batch_tokens; ++t)
                k::glm_kda_chunk(row_state(t, l).recurrent->f(), q.f(t * n), key.f(t * n), v.f(t * n),
                                 decay.f(t * n), beta.f(t * heads), y.f(t * n), heads, dim, 1, stream,
                                 kda_columns, row_parts);
        } else if (gpu) {
            const bool prepared = fast && prepare_kda_inputs;
            float *qi = nullptr;
            if (prepared) {
                if (row_parts != 1 && kda_columns != 128)
                    throw std::invalid_argument("GLM: parallel KDA preparation requires 128 columns");
                qi = buf("linear_qi", (size_t)heads * batch_tokens).f();
                k::glm_kda_prepare(q.f(), key.f(), decay.f(), beta.f(), qi, heads, batch_tokens, stream);
            }
            if (capturing_history && history_base_bytes) {
                if (prepared || row_parts != 1)
                    throw std::invalid_argument("GLM: compact history requires unprepared serial KDA");
                const auto record = replay_history.at(state.recurrent.get());
                auto *saved = (float *)((char *)verify_history->p + record.inputs);
                const int count = std::max(0, std::min(batch_tokens, history_slots - history_token_offset));
                for (auto [source, width] : std::array<std::pair<const float *, int>, 5>{{
                         {q.f(), n}, {key.f(), n}, {v.f(), n}, {decay.f(), n}, {beta.f(), heads}}}) {
                    if (count) check(cudaMemcpyAsync(saved + (size_t)history_token_offset * width,
                        source, (size_t)count * width * 4, cudaMemcpyDeviceToDevice, stream));
                    saved += (size_t)history_slots * width;
                }
                k::glm_kda_chunk(state.recurrent->f(), q.f(), key.f(), v.f(), decay.f(), beta.f(), y.f(),
                    heads, dim, batch_tokens, stream, kda_columns, row_parts);
            } else {
            // A verify window's rollback history comes straight from the recurrence kernel.
            float *snapshots = nullptr;
            if (capturing_history && !prepared && row_parts == 1 && batch_tokens <= 64)
                if (auto it = history_offsets.find(state.recurrent.get()); it != history_offsets.end())
                    snapshots = (float *)((char *)verify_history->p + it->second);
            if (snapshots && history_token_offset >= history_slots) {
                // A later split-verify group has no rollback slots left: the plain recurrence.
                k::glm_kda_chunk(state.recurrent->f(), q.f(), key.f(), v.f(), decay.f(), beta.f(), y.f(), heads, dim,
                                 batch_tokens, stream, kda_columns, row_parts);
            } else if (snapshots) {
                k::glm_kda_chunk(state.recurrent->f(), q.f(), key.f(), v.f(), decay.f(), beta.f(), y.f(), heads, dim,
                                 batch_tokens, stream, kda_columns, row_parts, nullptr,
                                 snapshots + (size_t)history_token_offset * (history_stride / 4),
                                 (long long)(history_stride / 4), history_slots - history_token_offset);
            } else {
            const int chunk = capturing_history ? 1 : 64;
            for (int t = 0; t < batch_tokens; t += chunk) {
                check_stop();
                int count = std::min(chunk, batch_tokens - t);
                k::glm_kda_chunk(state.recurrent->f(), q.f(t * n), key.f(t * n), v.f(t * n), decay.f(t * n),
                                 beta.f(t * heads), y.f(t * n), heads, dim, count, stream, kda_columns,
                                 row_parts, prepared ? qi + t * heads : nullptr);
                capture_state(state.recurrent.get(), t);
            }
            }
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
        end_share_quant();
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
        const int *device_pos = graph_mla() && l < (int)m.layers.size() ? (const int *)mla_positions->p + buffer_group : nullptr;
        int B = batch_tokens, heads = m.attention_heads, latent = m.kv_rank, dim = 256, ih = m.index_heads,
            idim = m.index_dim, pool_size = m.index_pool;
        const int tile = fast ? 64 : 1;
        constexpr int stride = 2052;
        const bool diagnose = std::getenv("STRATA_GLM_CHECK_MLA") && p == "blk.35.";
        auto inspect = [&](const char *stage, const float *data, size_t n, cudaStream_t work) {
            if (!diagnose) return;
            check(cudaStreamSynchronize(work));
            std::vector<float> values(n);
            check(cudaMemcpy(values.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
            float maximum = 0.f;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("GLM: non-finite MLA layer=" + p + " stage=" + stage +
                                             " index=" + std::to_string(i));
                maximum = std::max(maximum, std::abs(values[i]));
            }
            // Tensor-wide checks precede tile checks; only these need a range log.
            if (n > size_t(64) * heads * stride)
                std::cerr << "MLA_FINITE stage=" << stage << " values=" << n << " max_abs=" << maximum << '\n';
        };
        inspect("input", x, (size_t)B * m.hidden, stream);
        auto &low = buf("mla_low", m.q_rank * B);
        auto &qr = buf("mla_qr", m.q_rank * B);
        auto &q = buf("mla_q", heads * dim * B);
        share_quant(x, m.hidden);
        mat(p + "attn_q_a.weight", x, low.f());
        norm(p + "attn_q_a_norm.weight", low.f(), qr.f(), m.q_rank);
        mat(p + "attn_q_b.weight", qr.f(), q.f());
        auto &raw = buf("mla_raw", latent * B);
        mat(p + "attn_kv_a_mqa.weight", x, raw.f());
        if (!row_sequences.empty()) {
            auto &normalized = buf("mla_cache_rows", latent * B);
            norm(p + "attn_kv_a_norm.weight", raw.f(), normalized.f(), latent);
            for (int t = 0; t < B; ++t)
                check(cudaMemcpyAsync(row_state(t, l).cache->f((size_t)row_position(t) * latent),
                                      normalized.f((size_t)t * latent), (size_t)latent * 4,
                                      cudaMemcpyDeviceToDevice, stream));
        } else if (device_pos) {
            auto &normalized = buf("mla_cache_rows", latent * B);
            norm(p + "attn_kv_a_norm.weight", raw.f(), normalized.f(), latent);
            k::glm_cache_rows(normalized.f(), state.cache->f(), latent, B, device_pos, stream);
        } else norm(p + "attn_kv_a_norm.weight", raw.f(), state.cache->f((size_t)position * latent), latent);
        auto &iq = buf("index_q", ih * idim * B);
        auto &iw = buf("index_w", ih * B);
        auto &ik = buf("index_k", idim * B);
        auto &ikn = buf("index_kn", idim * B);
        auto &gate = buf("index_gate", idim * B);
        mat(p + "indexer.attn_q_b.weight", qr.f(), iq.f());
        mat(p + "indexer.proj.weight", x, iw.f());
        mat(p + "indexer.attn_k.weight", x, ik.f());
        mat(p + "indexer_compressor_gate.weight", x, gate.f());
        end_share_quant();
        inspect("query_low", low.f(), (size_t)B * m.q_rank, stream);
        inspect("query_norm", qr.f(), (size_t)B * m.q_rank, stream);
        inspect("query", q.f(), (size_t)B * heads * dim, stream);
        inspect("latent_raw", raw.f(), (size_t)B * latent, stream);
        inspect("latent_norm", state.cache->f((size_t)position * latent), (size_t)B * latent, stream);
        inspect("index_query", iq.f(), (size_t)B * ih * idim, stream);
        inspect("index_weight", iw.f(), (size_t)B * ih, stream);
        inspect("index_key", ik.f(), (size_t)B * idim, stream);
        inspect("index_gate", gate.f(), (size_t)B * idim, stream);
        auto &knw = weight(p + "indexer.k_norm.weight");
        auto &knb = weight(p + "indexer.k_norm.bias");
        k::glm_layer_norm_batch(ik.f(), knw.f(), knb.f(), ikn.f(), idim, B, 1e-6, stream);
        auto &ape = weight(p + "indexer_compressor_ape.weight");
        if (!row_sequences.empty()) {
            for (int t = 0; t < B; ++t) {
                auto &rs = row_state(t, l);
                k::glm_index_prepare(ikn.f(t * idim), gate.f(t * idim), ape.f(), rs.keys->f(),
                                     rs.gates->f(), rs.pooled->f(), row_position(t), 1, pool_size, idim, stream);
            }
        } else if (capturing_history) {
            for (int t = 0; t < B; ++t) {
                k::glm_index_prepare(ikn.f(t * idim), gate.f(t * idim), ape.f(), state.keys->f(),
                                     state.gates->f(), state.pooled->f(), position + t, 1, pool_size, idim, stream, device_pos, t);
                capture_state(state.keys.get(), t);
                capture_state(state.gates.get(), t);
            }
        } else
            k::glm_index_prepare(ikn.f(), gate.f(), ape.f(), state.keys->f(), state.gates->f(), state.pooled->f(),
                                 position, B, pool_size, idim, stream, device_pos);
        inspect("index_norm", ikn.f(), (size_t)B * idim, stream);
        inspect("pooled", state.pooled->f(), (size_t)(position + B) / pool_size * idim, stream);
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
        const auto &kb = *artifact.at(p + "attn_k_b.weight").tensor;
        const auto &vb = *artifact.at(p + "attn_v_b.weight").tensor;
        // One device's share of the query tiles; every product keeps its single-device shape.
        struct Tiles {
            cublasHandle_t blas; cudaStream_t stream;
            const float *pooled, *cache, *iq, *iw, *q, *wk, *wv;
            float *value, *dots, *scores, *aq, *av, *gathered, *attn;
            int *ids, *counts;
            int first, tokens;
            // Stepwise return of each finished tile to the primary device.
            cudaStream_t back = nullptr; cudaEvent_t produced = nullptr; float *home = nullptr; int home_device = 0, device = 0;
            // FP16 attention products (STRATA_GLM_MLA_F16): rounded queries and probabilities.
            strata::prefill::Gemm *gemm = nullptr;
            uint16_t *aq16 = nullptr, *attn16 = nullptr;
            const void *wk8 = nullptr, *wv8 = nullptr;  // Q8_0 decode weights (STRATA_GLM_Q8_DECODE)
            // Q8_0 verify windows: whole-window absorbed queries/values, so each projection reads its weights
            // once for every token (each token's products are those of a one-token call).
            float *aq_all = nullptr, *av_all = nullptr;
        };
        auto run_tiles = [&, B, heads, latent, dim, ih, idim, pool_size, pools, tile](const Tiles &d, bool stoppable) {
            const float one = 1, zero = 0;
            auto absorb = [&](const strata::TensorInfo &t, const float *W, const float *x, float *y, int tokens,
                              int ld_x, int ld_y) {
                const int in = t.shape[0], out = t.shape[1], count = t.shape[2];
                check(cublasSgemmStridedBatched(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one, W, in,
                                                (long long)in * out, x, ld_x, in, &zero, y, ld_y, out, count));
            };
            const bool hoisted = d.aq_all && d.av_all && d.wk8 && d.wv8 && !d.gemm && !d.back;
            if (hoisted)
                k::glm_q8_heads_gemv(d.wk8, d.q, d.aq_all, (int)kb.shape[2], (int)kb.shape[0], (int)kb.shape[1],
                                     d.tokens, heads * dim, heads * latent, d.stream);
            for (int t = 0; t < d.tokens; t += tile) {
                if (stoppable) check_stop();
                const int count = std::min(tile, d.tokens - t);
                const int row = d.first + t;
                const int at = row_sequences.empty() ? position + row : row_position(row);
                const float *pooled_rows = row_sequences.empty() ? d.pooled : row_state(row, l).pooled->f();
                const float *cache_rows = row_sequences.empty() ? d.cache : row_state(row, l).cache->f();
                float *aq_t = hoisted ? d.aq_all + (size_t)t * heads * latent : d.aq;
                float *av_t = hoisted ? d.av_all + (size_t)t * heads * latent : d.av;
                const int scored_pools = active_index_pools ? (at + count) / pool_size : pools;
                if (scored_pools)
                    check(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, scored_pools, count * ih, idim, &one,
                                      pooled_rows, idim, d.iq + (size_t)t * ih * idim, idim, &zero, d.dots, pools));
                k::glm_index_reduce(d.dots, d.iw + (size_t)t * ih, d.scores, ih, pools, count, at, pool_size,
                                    idim, d.stream, device_pos, d.first + t);
                k::glm_index_select_batch(d.scores, d.ids, d.counts, pools, at, count,
                                          pool_size, m.index_top_k, stride, d.stream, device_pos, d.first + t);
                if (hoisted) {}
                else if (d.wk8)
                    k::glm_q8_heads_gemv(d.wk8, d.q + (size_t)t * heads * dim, d.aq, (int)kb.shape[2], (int)kb.shape[0],
                                         (int)kb.shape[1], count, heads * dim, heads * latent, d.stream);
                else absorb(kb, d.wk, d.q + (size_t)t * heads * dim, d.aq, count, heads * dim, heads * latent);
                if (d.gemm) {
                    auto *keys = (uint16_t *)d.gathered;
                    k::glm_mla_gather_f16(cache_rows, d.ids, d.counts, keys, count, stride, latent, d.stream);
                    k::glm_f16(d.aq, d.aq16, (int64_t)count * heads * latent, d.stream);
                    d.gemm->f16_batched(d.aq16, keys, d.attn, heads, stride, latent, count);
                    inspect("attention_scores", d.attn, (size_t)count * heads * stride, d.stream);
                k::glm_mla_softmax(d.attn, d.counts, heads, stride, count, 1.f / std::sqrt((float)dim), d.stream);
                inspect("attention_probabilities", d.attn, (size_t)count * heads * stride, d.stream);
                    k::glm_f16(d.attn, d.attn16, (int64_t)count * heads * stride, d.stream);
                    d.gemm->f16_batched_nn(keys, d.attn16, d.av, latent, heads, stride, count);
                    inspect("absorbed_value", d.av, (size_t)count * heads * latent, d.stream);
                absorb(vb, d.wv, d.av, d.value + (size_t)t * heads * dim, count, heads * latent, heads * dim);
                inspect("head_value", d.value + (size_t)t * heads * dim, (size_t)count * heads * dim, d.stream);
                    if (d.back) {
                        check(cudaEventRecord(d.produced, d.stream));
                        check(cudaStreamWaitEvent(d.back, d.produced, 0));
                        check(cudaMemcpyPeerAsync(d.home + (size_t)t * heads * dim, d.home_device,
                                                  d.value + (size_t)t * heads * dim, d.device,
                                                  (size_t)count * heads * dim * 4, d.back));
                    }
                    continue;
                }
                k::glm_mla_gather(cache_rows, d.ids, d.counts, d.gathered, count, stride, latent, d.stream);
                // STRATA_GLM_MLA_KERNELS=1, one query: two decode kernels instead of the strided-batched SGEMMs.
                if (mla_one_kernels && count == 1 && !graph_mla() && !d.back) {
                    const size_t scratch = k::glm_mla_one_scratch_bytes(stride, latent, heads);
                    if (!mla_one_scratch || mla_one_scratch->bytes < scratch) mla_one_scratch = std::make_unique<Device>(scratch);
                    k::glm_mla_scores_one(d.gathered, aq_t, d.attn, stride, latent, heads, d.stream);
                    inspect("attention_scores", d.attn, (size_t)heads * stride, d.stream);
                    k::glm_mla_softmax(d.attn, d.counts, heads, stride, count, 1.f / std::sqrt((float)dim), d.stream);
                    k::glm_mla_values_one(d.gathered, d.attn, av_t, mla_one_scratch->f(), stride, latent, heads, d.stream);
                } else {
                    check(cublasSgemmStridedBatched(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, stride, heads, latent, &one,
                                                    d.gathered, latent, (long long)stride * latent, aq_t, latent,
                                                    (long long)heads * latent, &zero, d.attn, stride,
                                                    (long long)heads * stride, count));
                    inspect("attention_scores", d.attn, (size_t)count * heads * stride, d.stream);
                    k::glm_mla_softmax(d.attn, d.counts, heads, stride, count, 1.f / std::sqrt((float)dim), d.stream);
                    inspect("attention_probabilities", d.attn, (size_t)count * heads * stride, d.stream);
                    check(cublasSgemmStridedBatched(d.blas, CUBLAS_OP_N, CUBLAS_OP_N, latent, heads, stride, &one,
                                                    d.gathered, latent, (long long)stride * latent, d.attn,
                                                    stride, (long long)heads * stride, &zero, av_t, latent,
                                                    (long long)heads * latent, count));
                }
                inspect("absorbed_value", av_t, (size_t)count * heads * latent, d.stream);
                if (hoisted) continue;
                if (d.wv8)
                    k::glm_q8_heads_gemv(d.wv8, d.av, d.value + (size_t)t * heads * dim, (int)vb.shape[2], (int)vb.shape[0],
                                         (int)vb.shape[1], count, heads * latent, heads * dim, d.stream);
                else absorb(vb, d.wv, d.av, d.value + (size_t)t * heads * dim, count, heads * latent, heads * dim);
                inspect("head_value", d.value + (size_t)t * heads * dim, (size_t)count * heads * dim, d.stream);
                if (d.back) {
                    check(cudaEventRecord(d.produced, d.stream));
                    check(cudaStreamWaitEvent(d.back, d.produced, 0));
                    check(cudaMemcpyPeerAsync(d.home + (size_t)t * heads * dim, d.home_device,
                                              d.value + (size_t)t * heads * dim, d.device,
                                              (size_t)count * heads * dim * 4, d.back));
                }
            }
            if (hoisted)
                k::glm_q8_heads_gemv(d.wv8, d.av_all, d.value, (int)vb.shape[2], (int)vb.shape[0], (int)vb.shape[1],
                                     d.tokens, heads * latent, heads * dim, d.stream);
            (void)B;
        };
        const bool q8_mla = !fast && floating_q8_weight(p + "attn_k_b.weight") && floating_q8_weight(p + "attn_v_b.weight");
        Tiles local{blas, stream, state.pooled->f(), state.cache->f(), iq.f(), iw.f(), q.f(),
                    q8_mla ? nullptr : weight(p + "attn_k_b.weight", true).f(),
                    q8_mla ? nullptr : weight(p + "attn_v_b.weight", true).f(),
                    value.f(), dots.f(), scores.f(), aq.f(), av.f(), gathered.f(), attn.f(),
                    (int *)ids.p, (int *)counts.p, 0, B};
        if (q8_mla) {
            local.wk8 = weight(p + "attn_k_b.weight").p;
            local.wv8 = weight(p + "attn_v_b.weight").p;
            if (B > 1 && tile == 1) {
                local.aq_all = buf("mla_absorbed_q_window", (size_t)B * heads * latent).f();
                local.av_all = buf("mla_absorbed_v_window", (size_t)B * heads * latent).f();
            }
        }
        const bool halves = tensor_mla && fast && gpu && gpu->gemm;
        if (halves) {
            local.gemm = gpu->gemm.get();
            local.aq16 = (uint16_t *)buf("mla_absorbed_q16", ((size_t)ntile * heads * latent + 1) / 2).p;
            local.attn16 = (uint16_t *)buf("mla_scores16", ((size_t)ntile * heads * stride + 1) / 2).p;
        }
        std::future<void> pending;
        if (split_mla && fast && secondary && secondary_enabled && secondary->blas && B >= 2 * tile &&
            secondary->mla_weights.count(p + "attn_k_b.weight")) {
            auto &other = *secondary;
            const int tiles = (B + tile - 1) / tile;
            const int near = std::clamp(tiles - tiles * split_mla_percent / 100, 1, tiles - 1) * tile, far = B - near;
            const size_t rows = (size_t)position + B, np = std::max(1, pools);
            // The expert result buffers are idle during a mixer.
            const size_t room = (size_t)gpu->chunk * 8 * 4096;
            const size_t packed_need = (size_t)far * heads * dim + (size_t)ntile * stride * latent + rows * latent;
            const size_t result_need = (size_t)far * heads * dim + (size_t)ntile * heads * stride +
                (size_t)ntile * ih * np + (size_t)ntile * np + (size_t)far * ih * idim + (size_t)far * ih +
                3 * (size_t)ntile * heads * latent + (size_t)ntile * stride + ntile + np * idim + 4096 +
                (size_t)ntile * heads * stride;
            if (packed_need <= room && result_need <= room) {
                float *a = other.buf("packed", room).f(), *b = other.buf("result", room).f();
                Tiles remote{other.blas, other.stream, nullptr, nullptr, nullptr, nullptr, nullptr,
                             other.mla_weights.at(p + "attn_k_b.weight")->f(),
                             other.mla_weights.at(p + "attn_v_b.weight")->f(),
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, near, far};
                float *rq = a; a += (size_t)far * heads * dim;
                remote.gathered = a; a += (size_t)ntile * stride * latent;
                float *rcache = a;
                remote.value = b; b += (size_t)far * heads * dim;
                remote.attn = b; b += (size_t)ntile * heads * stride;
                remote.dots = b; b += (size_t)ntile * ih * np;
                remote.scores = b; b += (size_t)ntile * np;
                float *riq = b; b += (size_t)far * ih * idim;
                float *riw = b; b += (size_t)far * ih;
                remote.aq = b; b += (size_t)ntile * heads * latent;
                remote.av = b; b += (size_t)ntile * heads * latent;
                remote.ids = (int *)b; b += (size_t)ntile * stride;
                remote.counts = (int *)b; b += (ntile + 255) / 256 * 256;
                if (halves && other.gpu->gemm) {
                    remote.gemm = other.gpu->gemm.get();
                    remote.aq16 = (uint16_t *)b; b += (size_t)ntile * heads * latent;
                    remote.attn16 = (uint16_t *)b; b += (size_t)ntile * heads * stride;
                }
                float *rpooled = b;
                remote.q = rq; remote.cache = rcache; remote.iq = riq; remote.iw = riw; remote.pooled = rpooled;
                local.tokens = near;
                check(cudaEventRecord(mla_ready, stream));
                const float *sq = q.f((size_t)near * heads * dim), *siq = iq.f((size_t)near * ih * idim);
                const float *siw = iw.f((size_t)near * ih), *scache = state.cache->f(), *spooled = state.pooled->f();
                float *returned = value.f((size_t)near * heads * dim);
                if (incremental_return) {
                    remote.back = other.back; remote.produced = other.produced; remote.home = returned;
                    remote.home_device = primary_device; remote.device = other.device;
                }
                pending = other.worker->submit([=, this, &other, &run_tiles] {
                    auto fetch = [&](void *dst, const void *src, size_t floats) {
                        check(cudaMemcpyPeerAsync(dst, other.device, src, primary_device, floats * 4, other.stream));
                    };
                    check(cudaStreamWaitEvent(other.stream, mla_ready, 0));
                    fetch(rq, sq, (size_t)far * heads * dim);
                    fetch(riq, siq, (size_t)far * ih * idim);
                    fetch(riw, siw, (size_t)far * ih);
                    fetch(rcache, scache, rows * latent);
                    fetch(rpooled, spooled, np * idim);
                    run_tiles(remote, false);
                    if (remote.back) {
                        check(cudaEventRecord(other.finished, other.back));
                        return;
                    }
                    check(cudaMemcpyPeerAsync(returned, primary_device, remote.value, other.device,
                                              (size_t)far * heads * dim * 4, other.stream));
                    check(cudaEventRecord(other.finished, other.stream));
                });
            }
        }
        try { run_tiles(local, true); }
        catch (...) { if (pending.valid()) pending.wait(); throw; }
        if (pending.valid()) {
            pending.get();
            check(cudaStreamWaitEvent(stream, secondary->finished, 0));
        }
        inspect("value", value.f(), (size_t)B * heads * dim, stream);
        mat(p + "attn_output.weight", value.f(), out);
        inspect("output", out, (size_t)B * m.hidden, stream);
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

    void configure_prefill_schedule() {
        if (gpu && gpu->staged_prefetch && (secondary || artifact.is_exl3()))
            throw std::invalid_argument("GLM: staged prefetch currently requires one GGUF GPU");
        auto configure = [&](int device, GpuPrefill* prefill, PrefillGroupCache& cache, bool compact) {
            if (!prefill) return;
            prefill->numa_sources=&numa_sources;prefill->trimmed_sources=&trimmed_sources;
            prefill->place_by_node=lazy_experts && pool.numa_rows_available();
            prefill->defer_device_wait = defer_copy_wait;
            cache.preallocate = preallocate_groups;
            cache.early_ring_release = restore_prefill_groups;
            if (!preallocate_groups) { cache.release_spare(); return; }
            if (compact) return;
            DeviceScope scope(device);
            for (size_t l = 0; l < m.layers.size(); ++l) {
                if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                const auto p = "blk." + std::to_string(l) + ".";
                const size_t bytes = 16 * prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                    artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight")) + 16384;
                const size_t slots = cache.planned_groups.count(l) ? cache.planned_groups.at(l) : cache.per_layer / bytes;
                cache.reserve(l, bytes, slots);
            }
        };
        configure(primary_device, gpu.get(), prefill_cache, prefill_width_saved != 0);
        if (secondary)
            configure(secondary->device, secondary->gpu.get(), secondary->cache, secondary->saved_width != 0);
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
        tp_stale = true;
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
    void probe_decode_cache(const std::vector<float> &logits, const std::vector<int> &stops) {
        if (!cache_probe_steps) return;
        if (!gpu || !decode_cache_budget || decode_cache_auto || !step_pipeline || !canon_experts || capture_hidden ||
            ram_tier || secondary || remote_tp || merge_expert_groups || history_slots || !sequences.empty() ||
            gpu_decode_experts || profile || artifact.is_exl3() || block_screen)
            throw std::invalid_argument("GLM: unsupported cache probe setup gpu=" + std::to_string(bool(gpu)) +
                " budget=" + std::to_string(decode_cache_budget) + " auto=" + std::to_string(decode_cache_auto) +
                " pipeline=" + std::to_string(step_pipeline) + " canon=" + std::to_string(canon_experts) +
                " hidden=" + std::to_string(capture_hidden) + " ram=" + std::to_string(bool(ram_tier)) +
                " secondary=" + std::to_string(bool(secondary)) + " merge=" + std::to_string(merge_expert_groups) +
                " exl3=" + std::to_string(artifact.is_exl3()) + " pack=" + std::to_string(artifact.has_expert_pack()));
        int pseudo_tokens = 1024;
        if (const char *value = std::getenv("STRATA_GLM_CACHE_PROBE_WEIGHT")) {
            size_t end = 0; pseudo_tokens = std::stoi(value, &end);
            if (value[end] || pseudo_tokens < 256 || pseudo_tokens > 8192)
                throw std::invalid_argument("GLM: cache probe weight must be 256..8192 pseudo-tokens");
        }
        const int requested = std::min(cache_probe_steps, int(capacity) - position - 1);
        if (requested < 8) { std::cerr << "CACHE_PROBE tokens=0 skipped=context_headroom\n"; return; }
        for (const auto &layer : decode_resident)
            if (!layer.empty()) throw std::invalid_argument("GLM: cache probe must precede cache admission");
        const auto start = std::chrono::steady_clock::now();
        const auto saved = snapshot(); const bool old_graphs = decode_graphs;
        const char *old_phase = routing_phase; routing_phase = "cache_probe";
        set_decode_graphs(false);
        cache_probe_routes.assign(m.layers.size(), std::vector<uint64_t>(m.experts));
        auto current = logits; int steps = 0;
        const char *check_probe = std::getenv("STRATA_GLM_CHECK_CACHE_PROBE");
        const bool verify = check_probe && std::string(check_probe) == "1";
        std::vector<float> first_logits; int first_token = -1;
        try {
            for (; steps < requested; ++steps) {
                check_stop();
                const int token = int(std::max_element(current.begin(), current.end()) - current.begin());
                if (std::find(stops.begin(), stops.end(), token) != stops.end()) break;
                current = step(token);
                if (verify && !steps) { first_logits = current; first_token = token; }
            }
            restore(saved);
            if (verify && first_token >= 0) {
                auto learned = std::move(cache_probe_routes);
                const auto after = snapshot();
                if (after.position != saved.position || after.mtp_position != saved.mtp_position)
                    throw std::runtime_error("GLM: cache probe position restore failed");
                for (size_t l = 0; l < saved.layers.size(); ++l)
                    for (size_t b = 0; b < saved.layers[l].size(); ++b) {
                        const auto &a = saved.layers[l][b], &v = after.layers[l][b];
                        if (a.size() != v.size() || (!a.empty() && std::memcmp(a.data(), v.data(), a.size()*sizeof(float))))
                            throw std::runtime_error("GLM: cache probe state restore failed");
                    }
                if (step(first_token) != first_logits)
                    throw std::runtime_error("GLM: cache probe next logits differ after restore");
                restore(saved); cache_probe_routes = std::move(learned);
                std::cerr << "CACHE_PROBE_PARITY state=identical next_logits=identical\n";
            }
            set_decode_graphs(old_graphs); routing_phase = old_phase;
        } catch (...) {
            cache_probe_routes.clear(); restore(saved); set_decode_graphs(old_graphs); routing_phase = old_phase;
            throw;
        }
        if (steps >= 8) {
            // Weight probe counts explicitly, retaining the existing 256-token prior.
            for (auto &layer : cache_probe_routes)
                for (auto &count : layer) count = (count * pseudo_tokens + steps / 2) / steps;
            prefill_routes = std::move(cache_probe_routes);
        }
        cache_probe_routes.clear();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cerr << "CACHE_PROBE tokens=" << steps << " pseudo_tokens=" << pseudo_tokens << " ms=" << ms << " state=restored\n";
        std::cerr << "RESPONSE_PREPARE kind=cache_probe ms=" << ms << '\n';
    }
    Decoder(const std::string &path, size_t ctx, size_t cache_bytes, int threads, size_t expert_bytes = 0, bool pin_cpu = false,
            const std::string& expert_pack="",const std::string& pack_profile="",const std::string& cpu_backend="auto")
        : artifact(path), m(artifact.descriptor()), pool(artifact.is_exl3()?1:threads, artifact.is_exl3()?false:pin_cpu, true), capacity(ctx),
          budget(cache_bytes) {
        if (m.experts % 16 || m.experts > 288) throw std::invalid_argument("GLM: expert count must be a multiple of 16, at most 288");
        prefill_expert_groups = (int)m.experts / 16;
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            const auto p = "blk." + std::to_string(l) + ".";
            const size_t group = 16 * prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight")) + 16384;
            GpuPrefill::slot_bytes = std::max(GpuPrefill::slot_bytes, (group + MiB - 1) / MiB * MiB);
        }
        const char* calibration_dir=std::getenv("STRATA_GLM_CALIBRATION_DIR"),*activation_trace=std::getenv("STRATA_GLM_ACTIVATION_TRACE");
        if(calibration_dir||activation_trace) {
            if(artifact.is_exl3()||((calibration_dir&&*calibration_dir)&&!expert_pack.empty()))throw std::invalid_argument("moment calibration requires original GGUF; activation capture requires GGUF");
            expert_observer=std::make_unique<strata::artifact::ExpertCalibration>(calibration_dir?calibration_dir:"",activation_trace?activation_trace:"",(int)m.experts);
        }
        if(cpu_backend!="auto"&&cpu_backend!="native"&&cpu_backend!="packed-dot"&&cpu_backend!="packed-lut")throw std::invalid_argument("invalid CPU expert backend");
        if(expert_pack.empty()&&(!pack_profile.empty()||cpu_backend.starts_with("packed-")))throw std::invalid_argument("packed CPU backend/profile requires --expert-pack");
        if(!expert_pack.empty()) {
            // A validated Q23 assembly already runs through the native single-node path. Its explicit down-only
            // Q22 overlay needs no dual-node remap; the original-source and packed CPU backends retain theirs.
            const bool native_assembly=artifact.has_expert_pack()&&(cpu_backend=="auto"||cpu_backend=="native");
            if((!native_assembly&&(!pin_cpu||!pool.numa_rows_available()))||expert_bytes||direct_upload_enabled())throw std::invalid_argument("expert pack requires two pinned NUMA nodes and no legacy cache/direct upload");
            std::set<std::string> retained;
            if(!pack_profile.empty()) {
                const auto profile=strata::artifact::read_json(pack_profile);const auto& list=profile.at("retain");
                if(list.kind!=strata::artifact::Json::Array)throw std::invalid_argument("expert pack profile retain must be an array");
                for(const auto& name:list.array)if(!retained.insert(name.string()).second)throw std::invalid_argument("duplicate retained projection");
            }
            artifact.overlay_experts(expert_pack,retained);q23_layout=cpu_backend=="packed-dot"?1:cpu_backend=="packed-lut"?2:0;
            packed_huge_pages=std::getenv("STRATA_GLM_PACK_HUGE")&&std::string(std::getenv("STRATA_GLM_PACK_HUGE"))=="1";
            std::cerr<<"EXPERT_PACK source="<<expert_pack<<" backend="<<(cpu_backend=="auto"?"native":cpu_backend)<<" retained="<<retained.size()<<'\n';
        }
        // Opt-in for small GPUs: size staging from the effective artifact, after
        // overlays. The historical 208 MiB floor needlessly occupies two slots
        // when every group in this model is smaller. The projection bytes and
        // kernels are unchanged, including original-format exception layers.
        if (!artifact.is_exl3()) {
            const char *compact = std::getenv("STRATA_GLM_COMPACT_STAGING");
            if (compact && std::string(compact) != "0" && std::string(compact) != "1")
                throw std::invalid_argument("GLM: compact staging must be 0 or 1");
            if (compact && std::string(compact) == "1") {
                size_t largest = 0;
                for (size_t l = 0; l < m.layers.size(); ++l) {
                    if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                    const auto p = "blk." + std::to_string(l) + ".";
                    largest = std::max(largest, 16 * prefill_expert_stride(
                        artifact.at(p + "ffn_gate_exps.weight"), artifact.at(p + "ffn_up_exps.weight"),
                        artifact.at(p + "ffn_down_exps.weight")) + 16384);
                }
                GpuPrefill::slot_bytes = (largest + MiB - 1) / MiB * MiB;
                std::cerr << "STAGING compact=1 slot_MiB=" << GpuPrefill::slot_bytes / double(MiB) << '\n';
            }
        }
        check(cudaGetDevice(&primary_device));
        if (m.architecture != "glm5next" || ctx < 1 || ctx > (size_t)m.context || budget < 64 * 1024 * 1024)
            throw std::invalid_argument("GLM: invalid architecture, context, or cache budget");
        validate();
        if(artifact.is_exl3()) {
            if(ctx>65536)throw std::invalid_argument("EXL3: context above 64K requires host KV offload, which is deferred");
#ifdef __linux__
            const uint64_t ram=uint64_t(sysconf(_SC_PHYS_PAGES))*uint64_t(sysconf(_SC_PAGESIZE));
            const auto census=artifact.census();
            if(census.main_experts+census.main_fixed+6ULL*1024*MiB>ram)throw std::runtime_error("EXL3: main weights plus 6 GiB host reserve exceed system RAM");
#endif
            if(expert_bytes)throw std::invalid_argument("EXL3: GGUF expert cache is unsupported");
            exl_pool=std::make_unique<strata::cpu::exl3::Pool>(threads,pin_cpu);
        }
        if (!artifact.is_exl3() && GpuPrefill::prefetch_groups() && m.experts == 288)
            for (size_t l = 0; l < m.layers.size(); ++l) {
                if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                const auto p = "blk." + std::to_string(l) + ".";
                GpuPrefill::prefetch_bytes = std::max(GpuPrefill::prefetch_bytes,
                    16 * prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                        artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight")) + 16384);
            }
        prepack_cpus = cpu::physical_cores(false);
        if (pin_cpu) {
            const auto cores = cpu::physical_cores(false);
            if (!cores.empty()) host_affinity = cpu::pin_current_thread(cores.front());
        }
        host_moe = std::make_unique<Pinned>(cpu::MAXT * (2 * m.top_k + m.hidden) * 4);
        check(cudaStreamCreate(&stream));
        check(cudaEventCreateWithFlags(&moe_ready, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&mla_ready, cudaEventDisableTiming));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream));
        states.resize(m.layers.size());
        if (capture_hidden) ensure_draft_state();
        decode_resident.resize(m.layers.size());
        remote_decode_resident.resize(m.layers.size());
        expert_cache.resize(m.layers.size());
        DeviceTag state_tag("state");
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
        report_exact_cache(position);
        try { finish_prefetch(); } catch (...) {}
        prefetch_launcher.reset();
        try { finish_prefill_repairs(); } catch (...) {}
        // The last step can leave a promotion reading cold file pages into
        // pinned staging. Drain and join that worker while its sources,
        // destination and upload stream are still alive.
        try { tier_drain(); } catch (...) {}
        tier_stager.reset();
        if (tier_stream) {
            cudaStreamSynchronize(tier_stream);
            for (auto event : tier_events) cudaEventDestroy(event);
            tier_events.clear();
            cudaStreamDestroy(tier_stream);
            tier_stream = nullptr;
        }
        tier_staging.reset();
        for (int i = 0; i < 2; ++i) {
            try { finish_copy(i); } catch (...) {}
            expert_copies[i].reset();
        }
        mixer_graphs.clear();
        if (cached_decode) cached_decode->device_graphs.clear();
        secondary.reset();
        cpu::restore_thread_affinity(host_affinity);
        if (stream)
            cudaStreamSynchronize(stream);
        gpu.reset();
        if (moe_ready) cudaEventDestroy(moe_ready);
        if (mla_ready) cudaEventDestroy(mla_ready);
        if (packed_sum_uploaded) cudaEventDestroy(packed_sum_uploaded);
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
                if(!artifact.is_exl3()) {
                artifact.shape(p + "ffn_gate_exps.weight", {H, ff, E});
                artifact.shape(p + "ffn_up_exps.weight", {H, ff, E});
                artifact.shape(p + "ffn_down_exps.weight", {ff, H, E});
                if (artifact.at(p + "ffn_gate_exps.weight").tensor->type !=
                    artifact.at(p + "ffn_up_exps.weight").tensor->type)
                    throw std::runtime_error("GLM: gate/up types differ");
                }
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
        if (!prefill) finish_prefill_repairs();
        if (decode_cache_adapt) for (int i = 0; i < 2; ++i) finish_copy(i);
        fast = gpu && prefill;
        reset_phase();
        const int nt = tokens.size();
        if (nt < 1 || nt > (fast ? gpu->chunk : cpu::MAXT) || (row_sequences.empty() && position + nt > (int)capacity))
            throw std::out_of_range("GLM: invalid batch size or context capacity");
        for (int token : tokens)
            if (token < 0 || token >= m.vocab || (artifact.is_exl3()&&!artifact.exl3().token_ids[token]))
                throw std::out_of_range("GLM: token or context out of range");
        batch_tokens = nt;
        if (norm_q8_fused) {
            end_share_quant();
            if (q8_decode && quant_once && !fast && nt == 1 && m.hidden % 32 == 0 && !norm_quant)
                norm_quant = std::make_unique<Device>(k::native_q8_1_bytes(m.hidden, 1));
        }
        if (cached_decode) cached_decode->lookup_preloaded = false;
        if (cached_decode_b) cached_decode_b->lookup_preloaded = false;
        pipelining = false;
        if (pipeline_ready(nt)) prepare_step_mailbox();
        pipelining = pipeline_ready(nt);
        if (exact_cache && !pipelining) throw std::runtime_error("GLM: exact cache requires pipelined decode");
        PipelineGuard pipeline_guard{*this};
        pipeline_residents = false;
        if (pipelining) {
            tier_begin_step();
            if (!check_cpu_residency)
                for (const auto &layer : decode_resident) pipeline_residents = pipeline_residents || !layer.empty();
            if (pipeline_residents) {
                // Device experts are data: the lookup is copied into each layer's GPU pass at execution time.
                if (m.layers.size() > 64 || m.experts > 288) throw std::runtime_error("GLM: resident lookup geometry");
                if (!cached_decode) cached_decode = std::make_unique<ResidentExecutor>();
                auto *table = (unsigned long long *)cached_decode->host_lookup->p;
                std::memset(table, 0, cached_decode->lookup->bytes);
                for (size_t l = 0; l < decode_resident.size(); ++l)
                    for (const auto &[expert, slot] : decode_resident[l]) table[l * 288 + expert] = (unsigned long long)slot->p;
                cached_decode->lookup_preloaded = tier_lookup_once && nt == 1 && !fast && !capturing_history &&
                    row_sequences.empty() && !secondary && !exact_cache && !ram_tier && !decode_cache_adapt;
                if (cached_decode->lookup_preloaded)
                    cached_decode->lookup->put_async(table, cached_decode->lookup->bytes, stream);
                if (split_verify && nt >= 2) {
                    if (!cached_decode_b) cached_decode_b = std::make_unique<ResidentExecutor>();
                    std::memcpy(cached_decode_b->host_lookup->p, table, cached_decode->lookup->bytes);
                }
            }
        }
        refresh_route_bonus();
        if (step_trace && !fast) { trace.start = std::chrono::steady_clock::now(); trace.any = false; }
        if (device_resident_experts && !fast && (nt == 1 || batched_resident) && decode_prefill_cache && !device_lookup_ready) {
            if (m.layers.size() > 64 || m.experts > 288) throw std::runtime_error("GLM: resident lookup geometry");
            for (int participant = 0; participant < (secondary && secondary_enabled ? 2 : 1); ++participant) {
                DeviceScope scope(participant ? secondary->device : primary_device);
                auto &executor = participant ? secondary->decode : cached_decode;
                auto *table = (unsigned long long *)executor->host_lookup->p;
                std::memset(table, 0, executor->lookup->bytes);
                const auto &resident = participant ? remote_decode_resident : decode_resident;
                auto &cache = participant ? secondary->cache : prefill_cache;
                for (size_t l = 0; l < m.layers.size(); ++l) {
                    if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                    const auto p = "blk." + std::to_string(l) + ".";
                    const size_t stride = prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                        artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight"));
                    for (int e = 0; e < m.experts; ++e) {
                        auto it = resident[l].find(e);
                        table[l * 288 + e] = (unsigned long long)(it != resident[l].end() ? it->second->p : cache.expert(l, e, stride));
                    }
                }
                executor->lookup->put_async(table, executor->lookup->bytes, participant ? secondary->stream : stream);
            }
            device_lookup_ready = true;
        }
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
        // Split verify: group A is the first split tokens (in r's rows), group B the rest (in its own streams).
        const int split = (row_sequences.empty() || batch_split) && pipelining && split_verify && nt >= 2 && !fast && !tp_active() && !kda_split_active() ? nt / 2 : 0;
        if (split) {
            buffer_group = 1;
            buf("streams", m.hidden * 4 * nt).put(emb.data() + (size_t)split * 4 * m.hidden, (size_t)(nt - split) * 4 * m.hidden * 4);
            buffer_group = 0;
        }
        // put() completes the host-to-device copy. Do not retain up to 512 MiB
        // of idle host embeddings throughout a long GPU prefill.
        std::vector<float>().swap(emb);
        if (graph_mla()) {
            if (!mla_positions) mla_positions = std::make_unique<Device>(2 * sizeof(int));
            k::glm_decode_position((int *)mla_positions->p, position, stream);
            k::glm_decode_position((int *)mla_positions->p + 1, position + split, stream);
        }
        if (pipelining) mailbox->begin(stream);
        defer_active = defer_experts && pipelining && nt == 1 && !fast && !capturing_history && row_sequences.empty() &&
                       !tp_active() && !kda_split_active() &&
                       (!defer_routes || (pipeline_residents && !canon_strict() && !merge_expert_groups));
        if (defer_active && defer_routes && mailbox->view.slots < 2 * mailbox_group_stride())
            throw std::runtime_error("GLM: partial deferral needs the group-B mailbox slots");
        if (defer_active && !defer_coefficients) defer_coefficients = std::make_unique<Device>(24 * sizeof(float));
        int deferred_slot = -1;
        auto flush_deferred = [&](const float *later) {
            if (deferred_slot < 0) return;
            k::glm_mailbox_wait_add_deferred(mailbox->device_view, deferred_slot, r.f(), defer_coefficients->f(), later,
                                             mailbox->device_generation(), stream);
            deferred_slot = -1;
        };
        // Pipelined steps enqueue from the launcher thread: the stream's launch queue can fill while the
        // GPU waits for this thread's CPU experts.
        auto enqueue_layers = [&] {
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto &layer = m.layers[l];
            check_stop();
            auto layer_start = std::chrono::steady_clock::now();
            reset_phase();
            if (fast && !artifact.is_exl3() && layer.ffn == strata::core::FfnKind::Moe) prefetch_groups(l);
            auto enqueue_mixer = [&] {
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
                    k::glm_mhc_write_tokens(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
            };
            const bool combined = row_sequences.empty() && layer_graphs && decode_graphs && !fast &&
                ((nt==1 && !capturing_history) || (verify_graphs && !artifact.is_exl3() && nt<=4)) &&
                (layer.mixer == strata::core::MixerKind::Kda || graph_mla()) && layer.ffn == strata::core::FfnKind::Moe && !tp_active();
            if (combined && pipelining) {
                // Graph replays skip this lambda: the deferred slot is fixed per layer and tracked outside it.
                const int previous = defer_active ? deferred_slot : -1;
                decode_graph(l, [&] {
                    enqueue_mixer();
                    if (previous >= 0)
                        k::glm_mailbox_wait_add_deferred(mailbox->device_view, previous, r.f(), defer_coefficients->f(),
                                                         c.f(), mailbox->device_generation(), stream);
                    reset_phase();
                    hc_read(p, "ffn", r.f(), collapsed.f(), c.f());
                    norm(p + "ffn_norm.weight", collapsed.f(), x.f(), m.hidden);
                    reset_phase();
                    enqueue_moe_pipelined(p, l, x.f(), y.f(), layer);
                    k::glm_mhc_write_tokens(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
                    if (defer_active)
                        check(cudaMemcpyAsync(defer_coefficients->p, c.f(), 24 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    if (defer_active && defer_experts == 2)
                        k::glm_mailbox_wait_add_deferred(mailbox->device_view, deferred_target((int)l), r.f(), defer_coefficients->f(),
                                                         nullptr, mailbox->device_generation(), stream);
                });
                if (defer_active && defer_experts != 2) deferred_slot = deferred_target((int)l);
                continue;
            }
            if (combined) {
                auto enqueue_layer = [&] {
                    enqueue_mixer();
                    reset_phase();
                    hc_read(p, "ffn", r.f(), collapsed.f(), c.f());
                    norm(p + "ffn_norm.weight", collapsed.f(), x.f(), m.hidden);
                    reset_phase();
                    enqueue_moe_prefix(p, l, x.f(), y.f(), layer);
                };
                // A batched verify call replaces moe_prefixes with a different
                // arena layout. Replay still writes the graph's captured layout.
                if (auto it = mixer_graphs.find(graph_key(l)); it != mixer_graphs.end() && it->second->executable) {
                    const auto &graph = *it->second;
                    moe_prefixes[l] = {graph.moe_ids, graph.moe_weights, graph.moe_cursor};
                }
                decode_graph(l, enqueue_layer);
                const auto prefix = moe_prefixes.at(l);
                auto &graph = *mixer_graphs.at(graph_key(l));
                graph.moe_ids = prefix.ids; graph.moe_weights = prefix.weights; graph.moe_cursor = prefix.cursor;
            } else if (layer.mixer == strata::core::MixerKind::Kda && tp_active()) {
                if (tp_stale) tp_sync_states();
                auto &other = *secondary;
                auto &tp = *other.tp;
                decode_graph(1000 + l, [&] {
                    hc_read(p, "attn", r.f(), collapsed.f(), c.f());
                    norm(p + "attn_norm.weight", collapsed.f(), x.f(), m.hidden);
                    cudaStreamCaptureStatus status;
                    check(cudaStreamIsCapturing(stream, &status));
                    check(cudaEventRecordWithFlags(tp.x_ready, stream,
                          status == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
                });
                reset_phase();
                // First phase allocation of this layer: kda() below receives the same buffer.
                float *q_primary = buf("linear_q", (size_t)m.linear_dim * m.linear_heads).f();
                {
                    DeviceScope scope(other.device);
                    check(cudaStreamWaitEvent(other.stream, tp.x_ready, 0));
                    tp_graph(l, [&] { tp_kda_remote(p, l, x.f(), q_primary); });
                    check(cudaEventRecord(tp.y_ready, other.stream));
                }
                decode_graph(2000 + l, [&] {
                    kda(p, l, x.f(), y.f());
                    k::glm_mhc_write_tokens(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
                });
            } else if ((layer.mixer == strata::core::MixerKind::Kda || graph_mla())) decode_graph(l, enqueue_mixer);
            else enqueue_mixer();
            if (profile)
                check(cudaStreamSynchronize(stream));
            auto mixer_end = std::chrono::steady_clock::now();
            flush_deferred(c.f());  // after this layer's mixer write, as in the combined path
            reset_phase();
            if (combined) {
                prepared_moe_layer = l;
                gpu->cursor = moe_prefixes.at(l).cursor;
            } else {
                hc_read(p, "ffn", r.f(), collapsed.f(), c.f());
                norm(p + "ffn_norm.weight", collapsed.f(), x.f(), m.hidden);
                reset_phase();
            }
            if (layer.ffn == strata::core::FfnKind::Dense)
                ffn(p, x.f(), y.f(), "", layer.intermediate, layer.swiglu_limit);
            else if (pipelining)
                enqueue_moe_pipelined(p, l, x.f(), y.f(), layer);
            else {
                moe(p, l, x.f(), y.f(), layer);
                if (!fast) trace_done();
            }
            if (fast)
                k::glm_mhc_write_batch(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
            else
                k::glm_mhc_write_tokens(r.f(), c.f(), y.f(), r.f(), m.hidden, nt, stream);
            if (defer_active && pipelining && layer.ffn == strata::core::FfnKind::Moe) {
                check(cudaMemcpyAsync(defer_coefficients->p, c.f(), 24 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                deferred_slot = deferred_target((int)l);
                if (defer_experts == 2) flush_deferred(nullptr);
            }
            if (profile) {
                check(cudaStreamSynchronize(stream));
                auto end = std::chrono::steady_clock::now();
                std::cerr << "LAYER " << l << " mixer_ms="
                          << std::chrono::duration<double, std::milli>(mixer_end - layer_start).count()
                          << " ffn_ms=" << std::chrono::duration<double, std::milli>(end - mixer_end).count()
                          << '\n';
            }
        }
        flush_deferred(nullptr);
        if (kda_split_active()) kda_split_sync_back();
        else if (!tp_active()) tp_stale = true; // primary-only states advanced without the secondary halves
        reset_phase();
        for (int t = 0; t < nt; ++t)
            k::glm_hyper_head(r.f(t * 4 * m.hidden), collapsed.f(t * m.hidden), m.hidden, stream);
        norm("output_norm.weight", collapsed.f(), x.f(), m.hidden);
        };
        // Split verify, per layer: combine group A's previous MoE layer, enqueue A's layer up to its publication,
        // then the same for B. The GPU computes one group's attention while the CPU computes the other group's
        // experts; layer state still advances in token order (A's tokens before B's at every layer).
        auto enqueue_split = [&] {
            const int base = position;
            struct Group { int first, count; Device *r = nullptr; MoePending pending{}; bool waiting = false; };
            std::array<Group, 2> groups{Group{0, split}, Group{split, nt - split}};
            auto select = [&](int g) {
                buffer_group = g; batch_tokens = groups[g].count; position = base + groups[g].first;
                history_token_offset = groups[g].first;
                sequence_row_offset = groups[g].first;
                reset_phase();
            };
            try {
                // Group A runs ahead through the dense layers to its first publication, so the CPU starts
                // while group B computes its dense layers; after that the groups alternate per layer.
                std::vector<std::pair<size_t, int>> order;
                size_t first_moe = 0;
                while (first_moe < m.layers.size() && m.layers[first_moe].ffn != strata::core::FfnKind::Moe) ++first_moe;
                for (int g = 0; g < 2; ++g)
                    for (size_t l = 0; l <= first_moe && l < m.layers.size(); ++l) order.emplace_back(l, g);
                for (size_t l = first_moe + 1; l < m.layers.size(); ++l) { order.emplace_back(l, 0); order.emplace_back(l, 1); }
                for (const auto &[l, g] : order) {
                    const std::string p = "blk." + std::to_string(l) + ".";
                    const auto &layer = m.layers[l];
                    check_stop();
                    if (split_projections && g == 0 && l > first_moe) {
                        prepared_projections.clear();
                        auto &input = window_projections["input"];
                        if (!input) input = std::make_unique<Device>((size_t)cpu::MAXT * m.hidden * 4);
                        // Both preceding expert results are required for a shared projection. Publish A's
                        // next expert work before B's stateful mixer, retaining overlap within this layer.
                        for (int h = 0; h < 2; ++h) {
                            select(h);
                            auto &gr = buf("streams", m.hidden * 4 * nt), &gc = buf("hc_coeff", 24 * nt);
                            auto &gx = buf("x", m.hidden * nt), &gy = buf("y", m.hidden * nt);
                            auto &col = buf("collapsed", m.hidden * nt);
                            if (groups[h].waiting) {
                                enqueue_moe_combine(groups[h].pending);
                                k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                                groups[h].waiting = false;
                            }
                            hc_read(p, "attn", gr.f(), col.f(), gc.f());
                            norm(p + "attn_norm.weight", col.f(), gx.f(), m.hidden);
                            check(cudaMemcpyAsync(input->f((size_t)groups[h].first * m.hidden), gx.f(),
                                                  (size_t)batch_tokens * m.hidden * 4, cudaMemcpyDeviceToDevice, stream));
                        }
                        buffer_group = 0; batch_tokens = nt; position = base; reset_phase();
                        std::vector<std::string> names = layer.mixer == strata::core::MixerKind::Kda
                            ? std::vector<std::string>{"attn_q.weight", "attn_k.weight", "attn_v.weight"}
                            : std::vector<std::string>{"attn_q_a.weight", "attn_kv_a_mqa.weight",
                                                      "indexer.proj.weight", "indexer.attn_k.weight",
                                                      "indexer_compressor_gate.weight"};
                        share_quant(input->f(), m.hidden);
                        for (const auto &suffix : names) {
                            const auto name = p + suffix;
                            auto &output = window_projections[name];
                            const size_t rows = artifact.at(name).tensor->shape[1];
                            if (!output) output = std::make_unique<Device>((size_t)cpu::MAXT * rows * 4);
                            mat(name, input->f(), output->f());
                        }
                        end_share_quant();
                        for (const auto &suffix : names) prepared_projections.insert(p + suffix);
                    }
                    {
                        select(g);
                        projection_first = groups[g].first;
                        auto &gr = buf("streams", m.hidden * 4 * nt), &gcol = buf("collapsed", m.hidden * nt);
                        auto &gx = buf("x", m.hidden * nt), &gy = buf("y", m.hidden * nt), &gc = buf("hc_coeff", 24 * nt);
                        groups[g].r = &gr;
                        // A KDA MoE layer's segment (previous combine through this layer's publication) replays
                        // as one graph per group, width, first token and whether a combine precedes it. MLA
                        // layers can use device positions so the captured cache and pooling operations replay safely.
                        const bool graphed = row_sequences.empty() && (layer.mixer == strata::core::MixerKind::Kda || graph_mla()) &&
                                             layer.ffn == strata::core::FfnKind::Moe && layer_graphs && decode_graphs;
                        const bool combine = groups[g].waiting;
                        const MoePending previous = groups[g].pending;
                        auto segment = [&] {
                            reset_phase();
                            if (combine) {
                                enqueue_moe_combine(previous);
                                k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                            }
                            hc_read(p, "attn", gr.f(), gcol.f(), gc.f());
                            norm(p + "attn_norm.weight", gcol.f(), gx.f(), m.hidden);
                            reset_phase();
                            if (layer.mixer == strata::core::MixerKind::Kda) kda(p, l, gx.f(), gy.f());
                            else mla(p, l, gx.f(), gy.f());
                            k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                            reset_phase();
                            hc_read(p, "ffn", gr.f(), gcol.f(), gc.f());
                            norm(p + "ffn_norm.weight", gcol.f(), gx.f(), m.hidden);
                            reset_phase();
                            groups[g].pending = enqueue_moe_publish(p, (int)l, gx.f(), gy.f(), layer,
                                                                    (int)l + g * mailbox_group_stride(),
                                                                    g ? cached_decode_b.get() : cached_decode.get());
                        };
                        if (graphed) {
                            const int key = graph_key((int)l) + 1000000 * (1 + g + 2 * groups[g].first + 32 * combine + 64 * !prepared_projections.empty());
                            groups[g].pending = {};
                            decode_graph((int)l, segment, key);
                            if (!groups[g].pending.ids) {
                                // A replayed graph skipped the enqueue: the publication used the same buffers.
                                groups[g].pending = {(int)l, (int)l + g * mailbox_group_stride(), batch_tokens,
                                                     (int *)buf("moe_ids", m.top_k * nt).p, buf("moe_weights", m.top_k * nt).f(),
                                                     gy.f(), g ? cached_decode_b.get() : cached_decode.get(), canon_layer((int)l),
                                                     gx.f()};
                            }
                            groups[g].waiting = true;
                            continue;
                        }
                        if (groups[g].waiting) {
                            enqueue_moe_combine(groups[g].pending);
                            k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                            groups[g].waiting = false;
                        }
                        hc_read(p, "attn", gr.f(), gcol.f(), gc.f());
                        norm(p + "attn_norm.weight", gcol.f(), gx.f(), m.hidden);
                        reset_phase();
                        if (layer.mixer == strata::core::MixerKind::Kda) kda(p, l, gx.f(), gy.f());
                        else mla(p, l, gx.f(), gy.f());
                        k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                        reset_phase();
                        hc_read(p, "ffn", gr.f(), gcol.f(), gc.f());
                        norm(p + "ffn_norm.weight", gcol.f(), gx.f(), m.hidden);
                        reset_phase();
                        if (layer.ffn == strata::core::FfnKind::Dense) {
                            ffn(p, gx.f(), gy.f(), "", layer.intermediate, layer.swiglu_limit);
                            k::glm_mhc_write_tokens(gr.f(), gc.f(), gy.f(), gr.f(), m.hidden, batch_tokens, stream);
                        } else {
                            groups[g].pending = enqueue_moe_publish(p, (int)l, gx.f(), gy.f(), layer,
                                                                    (int)l + g * mailbox_group_stride(),
                                                                    g ? cached_decode_b.get() : cached_decode.get());
                            groups[g].waiting = true;
                        }
                    }
                }
                prepared_projections.clear();
                for (int g = 0; g < 2; ++g) {
                    select(g);
                    if (!groups[g].waiting) continue;
                    enqueue_moe_combine(groups[g].pending);
                    k::glm_mhc_write_tokens(buf("streams", m.hidden * 4 * nt).f(), buf("hc_coeff", 24 * nt).f(),
                                            buf("y", m.hidden * nt).f(), buf("streams", m.hidden * 4 * nt).f(),
                                            m.hidden, batch_tokens, stream);
                }
                // The window's rows in order, then one output norm over all of them.
                buffer_group = 0; batch_tokens = nt; position = base; history_token_offset = 0; sequence_row_offset = 0;
                reset_phase();
                for (int g = 0; g < 2; ++g)
                    for (int t = 0; t < groups[g].count; ++t)
                        k::glm_hyper_head(groups[g].r->f((size_t)t * 4 * m.hidden),
                                          collapsed.f((size_t)(groups[g].first + t) * m.hidden), m.hidden, stream);
                norm("output_norm.weight", collapsed.f(), x.f(), m.hidden);
            } catch (...) {
                prepared_projections.clear();
                buffer_group = 0; batch_tokens = nt; position = base; history_token_offset = 0; sequence_row_offset = 0;
                throw;
            }
        };
        if (split) { ++split_windows; run_pipelined_step(enqueue_split, split); }
        else if (pipelining) run_pipelined_step(enqueue_layers);
        else enqueue_layers();
        if (capture_hidden && !row_sequences.empty()) {
            for (int t = 0; t < nt; ++t) {
                auto &hidden = row_sequences[t] == current_sequence ? target_hidden : sequences[row_sequences[t]].hidden;
                hidden.resize(std::max(hidden.size(), (size_t)(row_positions[t] + 1) * m.hidden));
                check(cudaMemcpy(hidden.data() + (size_t)row_positions[t] * m.hidden, x.f((size_t)t * m.hidden),
                                 (size_t)m.hidden * 4, cudaMemcpyDeviceToHost));
            }
        } else if (capture_hidden) {
            target_hidden.resize((position + nt) * m.hidden);
            check(cudaMemcpy(target_hidden.data() + position * m.hidden, x.p, (size_t)nt * m.hidden * 4,
                             cudaMemcpyDeviceToHost));
            if (fast && mtp_batched) mtp_prime_batch(tokens, x.f());
        }
        const int head_tokens = all_logits ? nt : 1;
        auto &logits = buf("output", (size_t)m.vocab * head_tokens);
        // Evaluation can request every prefill row. Native decode projections
        // accept at most MAXT columns; keep their arithmetic for each small panel.
        for (int first = 0; first < head_tokens; first += cpu::MAXT)
            mat("output.weight", x.f((size_t)(all_logits ? first : nt - 1) * m.hidden),
                logits.f((size_t)first * m.vocab), false, std::min(cpu::MAXT, head_tokens - first));
        const bool trace_tail = step_trace && !fast && trace.any;
        auto tail_mark = std::chrono::steady_clock::now();
        auto tail_part = [&](double &part) {
            if (!trace_tail) return;
            const auto now = std::chrono::steady_clock::now();
            part += std::chrono::duration<double, std::milli>(now - tail_mark).count();
            tail_mark = now;
        };
        check(cudaStreamSynchronize(stream));
        tail_part(trace.sync_ms);
        if (pipelining) {
            pipelining = false;
            tier_submit();
            tier_plan();
            if (const unsigned error = mailbox->gpu_error())
                throw std::runtime_error("GLM: step pipeline GPU wait timed out at layer " + std::to_string(error - 1));
        }
        ram_plan();
        tail_part(trace.plan_ms);
        position += nt;
        batch_tokens = 1;
        auto result = logits.floats((size_t)m.vocab * head_tokens);
        for (float v : result)
            if (!std::isfinite(v))
                throw std::runtime_error("GLM: non-finite logits");
        tail_part(trace.copy_ms);
        if(artifact.is_exl3()) {
            const auto& ids=artifact.exl3().token_ids;
            for(int t=0;t<head_tokens;++t)for(size_t id=0;id<ids.size();++id)
                if(!ids[id])result[t*m.vocab+id]=-std::numeric_limits<float>::infinity();
        }
        if (step_trace && !fast && trace.any) {
            trace.tail_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trace.last_done).count();
            ++trace.steps;
        }
        return result;
    }
    void begin_sequence_window() {
        if (ram_tier || decode_cache_adapt) throw std::invalid_argument("GLM: batch MTP requires a stable RAM tier and no legacy adaptive cache");
        pipelining = true;
        try { tier_begin_step(); } catch (...) { pipelining = false; throw; }
        pipelining = false; sequence_speculating = true;
    }
    void end_sequence_window() {
        sequence_speculating = false;
        tier_submit(); tier_plan();
    }
    void enable_sequences(int count) {
        if (count < 1 || count > cpu::MAXT || secondary || artifact.is_exl3())
            throw std::invalid_argument("GLM: sequence batching needs 1..8 slots on one GPU with GGUF weights");
        if (!sequences.empty()) throw std::logic_error("GLM: sequence slots already allocated");
        sequences.resize(count);
        DeviceTag tag("sequence_state");
        for (int slot = 1; slot < count; ++slot) {
            auto &layers = sequences[slot].layers;
            layers.resize(states.size());
            for (size_t l = 0; l < states.size(); ++l) {
                auto copy_shape = [](const std::unique_ptr<Device> &src, std::unique_ptr<Device> &dst) {
                    if (src) dst = std::make_unique<Device>(src->bytes, true);
                };
                auto &src = states[l], &dst = layers[l];
                copy_shape(src.recurrent, dst.recurrent); copy_shape(src.conv_q, dst.conv_q);
                copy_shape(src.conv_k, dst.conv_k); copy_shape(src.conv_v, dst.conv_v);
                copy_shape(src.cache, dst.cache); copy_shape(src.keys, dst.keys);
                copy_shape(src.gates, dst.gates); copy_shape(src.pooled, dst.pooled);
            }
        }
    }
    void select_sequence(int slot) {
        if (slot < 0 || slot >= (int)sequences.size() || !row_sequences.empty())
            throw std::invalid_argument("GLM: invalid sequence slot");
        if (slot == current_sequence) return;
        check(cudaStreamSynchronize(stream));
        clear_verify_history();
        auto &old = sequences[current_sequence];
        old.position = position; old.draft_position = mtp_position; old.draft_primed = mtp_primed;
        old.hidden.swap(target_hidden); old.draft_hidden.swap(mtp_hidden); old.previous_hidden.swap(mtp_prev_hidden);
        auto &next = sequences[slot];
        mtp_position = next.draft_position; mtp_primed = next.draft_primed;
        next.hidden.swap(target_hidden); next.draft_hidden.swap(mtp_hidden); next.previous_hidden.swap(mtp_prev_hidden);
        states.swap(sequences[current_sequence].layers);
        states.swap(sequences[slot].layers);
        position = sequences[slot].position;
        current_sequence = slot;
        mixer_graphs.clear();
    }
    std::vector<float> step_sequences(const std::vector<int> &slots, const std::vector<int> &tokens, bool consecutive = false) {
        if (slots.empty() || slots.size() != tokens.size() || slots.size() > cpu::MAXT || capturing_history || !gpu)
            throw std::invalid_argument("GLM: invalid independent decode batch");
        std::map<int, int> offsets;
        for (int slot : slots) {
            if (slot < 0 || slot >= (int)sequences.size() || (!consecutive && offsets[slot]))
                throw std::invalid_argument("GLM: invalid or duplicate decode slot");
            const int at = (slot == current_sequence ? position : sequences[slot].position) + offsets[slot]++;
            if (at >= (int)capacity) throw std::out_of_range("GLM: sequence context exhausted");
        }
        const int saved_position = position;
        const bool saved_capture = capture_hidden;
        row_sequences = slots;
        offsets.clear();
        for (int slot : slots) row_positions.push_back((slot == current_sequence ? position : sequences[slot].position) + offsets[slot]++);
        try {
            auto logits = batch(tokens, false, true);
            position = saved_position;
            for (size_t row = 0; row < slots.size(); ++row) {
                if (slots[row] == current_sequence) position = row_positions[row] + 1;
                else sequences[slots[row]].position = row_positions[row] + 1;
            }
            row_sequences.clear(); row_positions.clear(); capture_hidden = saved_capture;
            return logits;
        } catch (...) {
            row_sequences.clear(); row_positions.clear(); capture_hidden = saved_capture; position = saved_position;
            throw;
        }
    }
    std::vector<float> step(int token) { return batch({token}, false); }
    void enable_verify_history(int slots) {
        if (!gpu || verify_history) return;
        slots = std::min(slots, cpu::MAXT - 1);
        const bool replay = compact_history && slots > 1;
        if (replay && (!sequences.empty() || secondary || kda_row_parts != 1 || m.linear_dim != 128))
            throw std::invalid_argument("GLM: compact history requires single-sequence, single-GPU serial KDA");
        size_t bytes = 0;
        history_base_bytes = 0;
        history_offsets.clear();
        replay_history.clear();
        for (size_t l = 0; l < m.layers.size(); ++l) {
            auto &s = states[l];
            for (auto *b : {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(), s.conv_v.get(), s.keys.get(), s.gates.get()})
                if (b) {
                    if (replay && b == s.recurrent.get()) {
                        replay_history.emplace(b, ReplayHistory{history_base_bytes, 0});
                        history_base_bytes += b->bytes;
                    } else { history_offsets.emplace(b, bytes); bytes += b->bytes; }
                }
        }
        const size_t n = (size_t)m.linear_dim * m.linear_heads;
        const size_t input_per_layer = (4 * n + m.linear_heads) * 4;
        const size_t per_slot = bytes + (replay ? replay_history.size() * input_per_layer + n * 4 : 0);
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t reserve = Device::physical_reserve() + 64 * MiB;
        const size_t physical = free_bytes > reserve ? free_bytes - reserve : 0;
        const size_t owned = Device::live() <= Device::limit() ? Device::limit() - Device::live() : 0;
        const size_t available = std::min(physical, owned);
        slots = std::min<size_t>(std::max(0, slots), available > history_base_bytes ?
            (available - history_base_bytes) / per_slot : 0);
        if (!slots) { history_offsets.clear(); replay_history.clear(); history_base_bytes = 0; return; }
        size_t offset = history_base_bytes + bytes * slots;
        for (auto &[buffer, record] : replay_history) {
            record.inputs = offset; offset += input_per_layer * slots;
        }
        history_output_offset = offset;
        const size_t allocation = history_base_bytes + per_slot * slots;
        DeviceTag tag("verify_history");
        verify_history = std::make_unique<Device>(allocation);
        history_stride = bytes; history_slots = slots;
        std::cerr << "VERIFY_HISTORY slots=" << slots << " MiB=" << allocation / double(MiB);
        if (replay) std::cerr << " compact_replay=1";
        std::cerr << '\n';
    }
    void clear_verify_history() {
        check(cudaStreamSynchronize(stream));
        // History graphs own pointers into this allocation; discard before free.
        for(auto it=mixer_graphs.begin();it!=mixer_graphs.end();) {
            if(it->first>=100000)it=mixer_graphs.erase(it);else ++it;
        }
        verify_history.reset(); history_offsets.clear(); history_slots = history_valid = 0;
        replay_history.clear(); history_base_bytes = history_output_offset = 0;
    }
    void invalidate_verify_history() { history_valid = 0; }
    int verify_history_slots() const { return history_slots; }
    bool restore_verified(int count) {
        if (!verify_history || count < 1 || count > history_valid) return false;
        for (auto [buffer, offset] : history_offsets)
            check(cudaMemcpyAsync(buffer->p, (char *)verify_history->p + history_base_bytes + (count - 1) * history_stride + offset,
                                  buffer->bytes, cudaMemcpyDeviceToDevice, stream));
        const int heads = m.linear_heads, dim = m.linear_dim, n = heads * dim;
        for (const auto &[buffer, record] : replay_history) {
            check(cudaMemcpyAsync(buffer->p, (char *)verify_history->p + record.base,
                                  buffer->bytes, cudaMemcpyDeviceToDevice, stream));
            auto *inputs = (float *)((char *)verify_history->p + record.inputs);
            const size_t panel = (size_t)history_slots * n;
            k::glm_kda_chunk(buffer->f(), inputs, inputs + panel, inputs + 2 * panel,
                inputs + 3 * panel, inputs + 4 * panel,
                (float *)((char *)verify_history->p + history_output_offset),
                heads, dim, count, stream, kda_columns, 1);
        }
        position = history_position + count;
        return true;
    }
    std::vector<float> verify(const std::vector<int> &tokens) {
        history_position = position;
        history_valid = 0;
        capturing_history = bool(verify_history);
        for (const auto &[buffer, record] : replay_history)
            check(cudaMemcpyAsync((char *)verify_history->p + record.base, buffer->p,
                                  buffer->bytes, cudaMemcpyDeviceToDevice, stream));
        try {
            auto result = batch(tokens, false, true);
            history_valid = std::min<size_t>(history_slots, tokens.size());
            capturing_history = false;
            return result;
        } catch (...) { capturing_history = false; throw; }
    }
    int vocabulary() const { return m.vocab; }
    void flush_expert_observer(){if(expert_observer)expert_observer->flush();if(block_screen)block_screen->report(std::cerr);}
    void set_block_screen_budget(int keep){if(!block_screen)throw std::invalid_argument("block sweep requires a screen profile");block_screen->uniform_budget(keep);}
    void set_block_screen_layer(int layer,int keep){if(!block_screen)throw std::invalid_argument("block sensitivity requires a screen profile");block_screen->layer_budget(layer,keep);}
    void enable_block_screen(const std::string& profile,const std::string& capture) {
        if(profile.empty())throw std::invalid_argument("block capture requires --expert-block-screen");
        const char* min_share=std::getenv("STRATA_GLM_ROUTE_MIN_SHARE");
        if(route_affinity!=0||ram_tier||q23_layout||remote_tp||direct_upload_enabled()||
           (min_share&&std::string(min_share)!="0")||
           std::any_of(gate_skip.begin(),gate_skip.end(),[](float v){return v!=0;})||
           !std::getenv("STRATA_GLM_CANON")||std::string(std::getenv("STRATA_GLM_CANON"))!="1")
            throw std::invalid_argument("block screen requires unbiased canonical Q23 rows without gate skipping or remote TP");
        block_screen=std::make_unique<strata::artifact::ExpertBlockScreen>(artifact,profile,capture);
        std::cerr<<"BLOCK_SCREEN mode=oracle diagnostic_only=1 profile="<<profile<<'\n';
    }
    void prepare_numa_weights() {
        const char* enabled=std::getenv("STRATA_GLM_Q2_NUMA_WEIGHTS");
        if(!numa_experts.empty())return;
        if(!artifact.has_expert_pack()&&(!enabled||std::string(enabled)=="0"))return;
        if(enabled&&std::string(enabled)!="1"&&std::string(enabled)!="0")throw std::invalid_argument("STRATA_GLM_Q2_NUMA_WEIGHTS must be 0 or 1");
        // An explicit 0 keeps a pack on its file mapping (reclaimable page cache instead of owned arenas);
        // fused hidden quantization needs owned rows, so it must be disabled too.
        if(enabled&&std::string(enabled)=="0") {
            const char* fuse=std::getenv("STRATA_GLM_PACK_FUSE_QUANT");
            if(!fuse||std::string(fuse)!="0")throw std::invalid_argument("a mapped expert pack needs STRATA_GLM_PACK_FUSE_QUANT=0");
            return;
        }
        // A single-node PC can stage only selected experts from their file
        // mapping without constructing the two-node row views. CPU jobs keep
        // their normal whole-row scheduler; the memory cgroup owns file cache.
        const bool mapped_lazy = !remote_tp && std::getenv("STRATA_GLM_MAPPED_OWNED") &&
            std::string(std::getenv("STRATA_GLM_MAPPED_OWNED")) == "1" &&
            std::getenv("STRATA_GLM_MAPPED_LAZY") && std::string(std::getenv("STRATA_GLM_MAPPED_LAZY")) == "1";
        if (!pool.numa_rows_available() && mapped_lazy) {
            if (artifact.is_exl3() || secondary || !prepared_experts.empty() || direct_upload_enabled() ||
                q23_layout || packed_huge_pages || std::getenv("STRATA_GLM_PREFER_RESIDENT_MIB"))
                throw std::invalid_argument("single-node lazy experts require native GGUF, one GPU, plain mapped rows and unbiased routing");
            for (const auto& cache : expert_cache) if (cache)
                throw std::invalid_argument("single-node lazy experts cannot use legacy expert cache");
            lazy_experts = true;
            if (gpu) gpu->place_by_node = false;
            artifact.discard_original_experts();
            size_t bytes = 0;
            for (size_t l=3;l<m.layers.size();++l) for (const auto* part : {"gate","up","down"})
                bytes += artifact.at("blk."+std::to_string(l)+".ffn_"+part+"_exps.weight").bytes;
            std::cerr << "MAPPED_EXPERTS numa_rows=0 lazy=1 bytes=" << bytes << '\n';
            return;
        }
        if(artifact.is_exl3() || !pool.numa_rows_available() || secondary || !prepared_experts.empty() || direct_upload_enabled())
            throw std::invalid_argument("Q2 NUMA weights require native GGUF, pinned workers on two nodes, single GPU, no TP/prepack/direct upload");
        for(const auto& cache:expert_cache)if(cache)throw std::invalid_argument("Q2 NUMA weights cannot use legacy expert cache");
        size_t required=0;
        for(size_t l=3;l<m.layers.size();++l)for(const auto* part:{"gate","up","down"})
            required+=artifact.at("blk."+std::to_string(l)+".ffn_"+part+"_exps.weight").bytes;
        size_t budget_mib=artifact.has_expert_pack()?114688:98304;
        if(const char* value=std::getenv("STRATA_GLM_Q2_NUMA_WEIGHT_MIB")) {
            size_t used=0;budget_mib=std::stoull(value,&used);
            if(used!=std::strlen(value)||budget_mib>131072)throw std::invalid_argument("invalid NUMA weight MiB budget");
        }
        if(required>budget_mib*MiB)throw std::runtime_error("Q2 NUMA weights exceed configured packed-byte budget");
        // STRATA_GLM_MAPPED_OWNED=1: node-owned scheduling over the file mapping (pages moved, not copied).
        // Remote TP copies its local rows instead: a mapping cannot hold part of each expert row.
        const bool mapped=!remote_tp&&std::getenv("STRATA_GLM_MAPPED_OWNED")&&std::string(std::getenv("STRATA_GLM_MAPPED_OWNED"))=="1";
        if(mapped&&(q23_layout||packed_huge_pages))throw std::invalid_argument("mapped owned rows need the plain row layout");
        // STRATA_GLM_MAPPED_LAZY=1 (with mapped owned rows): for hosts with less memory than the experts. Nothing
        // is read or moved at load; expert pages are read from the file on first use and the page cache keeps
        // the recently used ones. The payload checksum pass, which reads every byte, is skipped.
        const bool lazy=mapped&&std::getenv("STRATA_GLM_MAPPED_LAZY")&&std::string(std::getenv("STRATA_GLM_MAPPED_LAZY"))=="1";
#ifdef __linux__
        size_t total=size_t(sysconf(_SC_PHYS_PAGES))*size_t(sysconf(_SC_PAGESIZE));
        if(!lazy&&(total<16*1024*MiB || required>total-16*1024*MiB))throw std::runtime_error("Q2 NUMA weights need 16 GiB host capacity beyond packed weights");
#endif
        auto start=std::chrono::steady_clock::now();
        size_t unmoved=0;
        artifact.discard_original_experts();
        // File-cache pages must yield before anonymous weights grow. Merely
        // dropping this process's PTEs allowed Linux to swap the new arenas.
        if(!mapped)for(size_t l=3;l<m.layers.size();++l)for(const auto* part:{"gate","up","down"}) {
            const auto& t=artifact.at("blk."+std::to_string(l)+".ffn_"+part+"_exps.weight");
            t.file->discard_tensor_pages(*t.tensor,t.bytes);
        }
        try {
            for(size_t l=3;l<m.layers.size();++l) {
                check_stop();auto& entry=numa_experts[l];const auto prefix="blk."+std::to_string(l)+".ffn_";
                for(const auto* part:{"gate","up","down"}) {
                    const auto& t=artifact.at(prefix+part+"_exps.weight");const auto& shape=t.tensor->shape;
                    if(shape.size()!=3 || shape[2]!=size_t(m.experts) || shape[1]%2)throw std::invalid_argument("Q2 NUMA weights: unsupported tensor shape");
                    size_t row=t.bytes/(shape[1]*shape[2]);const auto* source=t.data();
                    const auto* expected=t.file->get("strata.expert_pack.hash."+t.tensor->name);
                    if(expected&&!lazy) {uint64_t hash=14695981039346656037ULL;for(size_t k=0;k<t.bytes;++k){hash^=source[k];hash*=1099511628211ULL;}if(hash!=expected->u)throw std::runtime_error("expert pack payload checksum mismatch: "+t.tensor->name);}
                    const int packed=q23_layout&&(t.tensor->type==10||t.tensor->type==11)?t.tensor->type:0;
                    std::unique_ptr<cpu::NumaTensor> owned;
                    const int keep=remote_tp&&remote_tp->split.count(int(l))?remote_tp->split.at(int(l)):0;
                    if(keep) {
                        // Remote TP: this node keeps gate/up rows [0, keep) and down columns [0, keep) of every
                        // expert, contiguous; the worker holds the rest. GPU uploads still read whole experts
                        // from the file, so these rows are not a numa_source.
                        const bool down=std::string(part)=="down";
                        const size_t local_row=down?ggml_row_size(ggml_type(t.tensor->type),keep):row,local_rows=down?shape[1]:size_t(keep);
                        const size_t local_bytes=local_row*local_rows*shape[2];
                        if(trim_buffer.size()<local_bytes)trim_buffer.resize(local_bytes);   // reused: faulted in once
                        uint8_t* local=trim_buffer.data();
                        std::vector<std::jthread> gather;
                        const size_t workers=std::max<size_t>(1,std::min<size_t>(16,std::thread::hardware_concurrency()));
                        for(size_t w=0;w<workers;++w)gather.emplace_back([&,w] {
                            for(size_t e=shape[2]*w/workers;e<shape[2]*(w+1)/workers;++e) {
                                if(!down)std::memcpy(local+e*local_rows*row,source+e*shape[1]*row,local_rows*row);
                                else for(size_t r=0;r<shape[1];++r)
                                    std::memcpy(local+(e*shape[1]+r)*local_row,source+(e*shape[1]+r)*row,local_row);
                            }
                        });
                        gather.clear();
                        owned=std::make_unique<cpu::NumaTensor>(local,local_row,local_rows,shape[2],pool.numa_cores(),packed_huge_pages,packed,down?keep:int(shape[0]));
                        numa_local_only.insert(t.tensor);numa_weight_bytes+=local_bytes;
                        trimmed_sources[t.tensor]={owned.get(),row,local_row,shape[1],local_rows,down};
                        t.file->discard_tensor_pages(*t.tensor,t.bytes);
                    } else if(mapped) {
                        owned=std::make_unique<cpu::NumaTensor>(cpu::NumaTensor::View{},source,row,shape[1],shape[2],lazy);
                        unmoved+=owned->unmoved_pages;numa_weight_bytes+=t.bytes;
                    } else {
                        owned=std::make_unique<cpu::NumaTensor>(source,row,shape[1],shape[2],pool.numa_cores(),packed_huge_pages,packed,shape[0]);
                        numa_sources[t.tensor]=owned.get();numa_weight_bytes+=t.bytes;
                        t.file->discard_tensor_pages(*t.tensor,t.bytes);
                    }
                    auto& dest=std::string(part)=="gate"?entry.gate:std::string(part)=="up"?entry.up:entry.down;
                    dest=std::move(owned);
                }
                if(remote_tp&&remote_tp->split.count(int(l)))remote_tp->trimmed.insert(int(l));
#ifdef __linux__
                std::ifstream status("/proc/self/status");std::string key,line;
                while(std::getline(status,line))if(line.starts_with("VmSwap:")) {
                    size_t kib=std::stoull(line.substr(7));
                    if(kib>1024)throw std::runtime_error("Q2 NUMA preparation stopped: owned weights are swapping");
                }
#endif
            }
        }catch(...) {numa_sources.clear();numa_local_only.clear();trimmed_sources.clear();numa_experts.clear();numa_weight_bytes=0;if(remote_tp)remote_tp->trimmed.clear();throw;}
        std::vector<uint8_t>().swap(trim_buffer);
        if(gpu){gpu->numa_sources=&numa_sources;gpu->trimmed_sources=&trimmed_sources;}
        std::cerr<<"Q2_NUMA_WEIGHTS mapped="<<mapped<<" lazy="<<lazy<<" unmoved_pages="<<unmoved<<" bytes="<<numa_weight_bytes<<" node0_MiB="<<numa_weight_bytes/double(2*MiB)
                 <<" node1_MiB="<<numa_weight_bytes/double(2*MiB)<<" prepare_ms="
                 <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
        lazy_experts=lazy;
        if(gpu)gpu->place_by_node=lazy;
        if(std::getenv("STRATA_GLM_PREFER_RESIDENT_MIB")) {
            if(!lazy)throw std::invalid_argument("STRATA_GLM_PREFER_RESIDENT_MIB needs STRATA_GLM_MAPPED_LAZY=1");
            prepare_ram_tier();
        }
    }
    void prepare_cpu(size_t bytes) {
        if(artifact.is_exl3()&&bytes)throw std::invalid_argument("EXL3: GGUF CPU prepack is unsupported");
        if (!bytes || !prepared_experts.empty()) return;
        if(!numa_experts.empty())throw std::invalid_argument("CPU format prepacking cannot coexist with NUMA packed rows");
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
    void enable_mtp_capture(bool enabled = true) {
        capture_hidden = enabled;
        if (!enabled) return;
        ensure_draft_state();
    }
    // The NextN draft block keeps its own latent cache and indexer state behind the main layers.
    void ensure_draft_state() {
        if (m.draft_layers.size() != 1 || states.size() > m.layers.size()) return;
        states.emplace_back();
        auto &state = states.back();
        state.cache = std::make_unique<Device>(capacity * m.kv_rank * 4, true);
        state.keys = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
        state.gates = std::make_unique<Device>(m.index_pool * m.index_dim * 4, true);
        state.pooled = std::make_unique<Device>(std::max<size_t>(1, capacity / m.index_pool) * m.index_dim * 4, true);
        for (auto *b : {state.cache.get(), state.keys.get(), state.gates.get(), state.pooled.get()})
            check(cudaMemsetAsync(b->p, 0, b->bytes, stream));
        check(cudaStreamSynchronize(stream));
    }
    // Prime the draft block for every position of a prefill batch at once: position i consumes token i+1 and
    // the target hidden row i. The batch's last position waits for the next token (first proposal at decode).
    void mtp_prime_batch(const std::vector<int> &tokens, const float *hidden) {
        const int nt = tokens.size(), H = m.hidden;
        if (nt < 1 || states.size() <= m.layers.size()) return;
        const int skip = position == 0 ? 1 : 0, count = nt - skip;
        if (count < 1) { mtp_remember_hidden(hidden, nt); return; }
        const std::string p = "blk.45.";
        const auto &layer = m.draft_layers.at(0);
        struct Restore {
            int &pos, &nt; bool &fast;
            int old_pos, old_nt; bool old_fast;
            ~Restore() { pos = old_pos; nt = old_nt; fast = old_fast; }
        } restore{position, batch_tokens, fast, position, batch_tokens, fast};
        position = mtp_position; batch_tokens = count;
        reset_phase();
        const auto &embedding = artifact.at("token_embd.weight");
        const auto *traits = ggml_get_type_traits((ggml_type)embedding.tensor->type);
        std::vector<float> rows((size_t)count * H);
        for (int t = 0; t < count; ++t) {
            const auto *data = embedding.data() + (size_t)tokens[t + skip] * (embedding.bytes / m.vocab);
            if (embedding.tensor->type == GGML_TYPE_F32) std::memcpy(rows.data() + (size_t)t * H, data, H * 4);
            else traits->to_float(data, rows.data() + (size_t)t * H, H);
        }
        // Persistent batch buffers are free after output_norm: the residual streams hold the draft input,
        // `collapsed` its residual and `y` the mixer/FFN outputs; `x` (the target hidden) stays intact.
        auto &xd = buf("streams", (size_t)count * H), &cur = buf("collapsed", (size_t)count * H);
        auto &y = buf("y", (size_t)count * H);
        auto &emb = buf("mtp_embedding", (size_t)count * H), &prev = buf("mtp_previous", (size_t)count * H);
        emb.put(rows.data(), rows.size() * 4);
        if (skip) check(cudaMemcpyAsync(prev.p, hidden, (size_t)count * H * 4, cudaMemcpyDeviceToDevice, stream));
        else {
            check(cudaMemcpyAsync(prev.p, mtp_prev_hidden->p, (size_t)H * 4, cudaMemcpyDeviceToDevice, stream));
            check(cudaMemcpyAsync(prev.f(H), hidden, (size_t)(count - 1) * H * 4, cudaMemcpyDeviceToDevice, stream));
        }
        auto &en = buf("mtp_enorm", (size_t)count * H), &hn = buf("mtp_hnorm", (size_t)count * H);
        auto &joined = buf("mtp_joined", (size_t)count * 2 * H);
        auto &x = xd;
        norm(p + "nextn.enorm.weight", emb.f(), en.f(), H);
        norm(p + "nextn.hnorm.weight", prev.f(), hn.f(), H);
        check(cudaMemcpy2DAsync(joined.p, (size_t)2 * H * 4, en.p, (size_t)H * 4, (size_t)H * 4, count,
                                cudaMemcpyDeviceToDevice, stream));
        check(cudaMemcpy2DAsync(joined.f(H), (size_t)2 * H * 4, hn.p, (size_t)H * 4, (size_t)H * 4, count,
                                cudaMemcpyDeviceToDevice, stream));
        mat(p + "nextn.eh_proj.weight", joined.f(), cur.f());
        reset_phase();
        norm(p + "attn_norm.weight", cur.f(), x.f(), H);
        mla_gpu(p, m.layers.size(), x.f(), y.f());
        const float one = 1;
        check(cublasSaxpy(blas, count * H, &one, y.f(), 1, cur.f(), 1));
        reset_phase();
        norm(p + "ffn_norm.weight", cur.f(), x.f(), H);
        moe_gpu(p, x.f(), y.f(), layer);
        check(cublasSaxpy(blas, count * H, &one, y.f(), 1, cur.f(), 1));
        check(cudaStreamSynchronize(stream));
        mtp_position += count;
        mtp_primed = true;
        mtp_remember_hidden(hidden, nt);
    }
    void mtp_remember_hidden(const float *hidden, int nt) {
        if (!mtp_prev_hidden) mtp_prev_hidden = std::make_unique<Device>((size_t)m.hidden * 4);
        check(cudaMemcpyAsync(mtp_prev_hidden->p, hidden + (size_t)(nt - 1) * m.hidden, (size_t)m.hidden * 4,
                              cudaMemcpyDeviceToDevice, stream));
    }
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
    void rebalance_decode_cache() {
        if (!decode_cache_adapt || !decode_cache_extend) return;
        const auto start = std::chrono::steady_clock::now();
        decltype(victims) prefix_victims;
        for (int participant = 0; participant < 2; ++participant) {
            const auto &cache = participant ? remote_decode_resident : decode_resident;
            for (size_t layer = 0; layer < cache.size(); ++layer)
                for (const auto &[expert, entry] : cache[layer])
                    prefix_victims[entry->bytes].push({prefill_routes[layer][expert],
                        (int)layer, expert, participant});
        }
        struct Candidate { uint64_t frequency; int layer, expert; };
        std::vector<Candidate> candidates;
        for (size_t layer = 0; layer < prefill_routes.size(); ++layer)
            for (int expert = 0; expert < m.experts; ++expert)
                if (prefill_routes[layer][expert] && !decode_resident[layer].count(expert) &&
                    !remote_decode_resident[layer].count(expert))
                    candidates.push_back({prefill_routes[layer][expert], (int)layer, expert});
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            if (a.frequency != b.frequency) return a.frequency > b.frequency;
            return std::tie(a.layer, a.expert) < std::tie(b.layer, b.expert);
        });
        size_t used = 0, copies = 0;
        const size_t limit = [] {
            const char *value = std::getenv("STRATA_GLM_REBALANCE_MIB");
            return (value ? std::stoull(value) : 8192) * MiB;
        }();
        for (const auto &candidate : candidates) {
            check_stop();
            const auto p = "blk." + std::to_string(candidate.layer) + ".";
            const auto &g = artifact.at(p + "ffn_gate_exps.weight");
            const auto &u = artifact.at(p + "ffn_up_exps.weight");
            const auto &d = artifact.at(p + "ffn_down_exps.weight");
            const size_t bytes = (g.bytes + u.bytes + d.bytes) / m.experts;
            if (bytes > limit - used) continue;
            auto &heap = prefix_victims[bytes];
            if (heap.empty() || heap.top().score >= candidate.frequency) continue;
            const auto victim = heap.top(); heap.pop();
            auto &cache = victim.participant ? remote_decode_resident : decode_resident;
            auto found = cache[victim.layer].find(victim.expert);
            auto copy = std::make_unique<Copy>();
            copy->layer = candidate.layer; copy->expert = candidate.expert; copy->participant = victim.participant;
            copy->entry = std::move(found->second); cache[victim.layer].erase(found);
            const std::array<size_t, 3> sizes{g.bytes / m.experts, u.bytes / m.experts, d.bytes / m.experts};
            const std::array<const uint8_t *, 3> source{g.data() + candidate.expert * sizes[0],
                u.data() + candidate.expert * sizes[1], d.data() + candidate.expert * sizes[2]};
            mark_promoted_slot(victim.participant, copy->entry->p, copy->entry->bytes);
            copy->ready = expert_copies[victim.participant]->copy(copy->entry->p, source, sizes);
            pending_copies[victim.participant].push_back(std::move(copy));
            used += bytes; ++copies;
        }
        for (int i = 0; i < 2; ++i) finish_copy(i);
        victims.clear();
        for (int participant = 0; participant < 2; ++participant) {
            const auto &cache = participant ? remote_decode_resident : decode_resident;
            for (size_t layer = 0; layer < cache.size(); ++layer)
                for (const auto &[expert, entry] : cache[layer])
                    victims[entry->bytes].push({expert_score(layer, expert), (int)layer, expert, participant});
        }
        adaptive_wait_ms = 0;
        std::cerr << "DECODE_CACHE_REBALANCE copies=" << copies << " MiB=" << used / double(MiB)
                  << " ms=" << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() << '\n';
    }
    void set_decode_cache_budget(size_t bytes) {
        if(artifact.is_exl3()&&bytes)throw std::invalid_argument("EXL3: GGUF decode cache is unsupported");
        decode_cache_budget = bytes;
        if (bytes) {
            prefill_routes.assign(m.layers.size(), std::vector<uint64_t>(m.experts));
            if (fixed_cache_recent) prefill_recent_ids.resize(m.layers.size());
        }
    }
    void set_decode_cache_auto(bool value, bool extend = false) {
        if(artifact.is_exl3()&&(value||extend))throw std::invalid_argument("EXL3: GGUF decode cache is unsupported");
        decode_cache_auto = value; decode_cache_extend = extend;
        if (value) {
            if (!gpu || gpu->chunk <= cpu::MAXT)
                throw std::invalid_argument("GLM: automatic decode cache requires GPU prompt prefill");
            set_decode_prefill_cache(true);
            prefill_routes.assign(m.layers.size(), std::vector<uint64_t>(m.experts));
            prefill_recent_ids.resize(m.layers.size());
        }
    }
    void set_decode_cache_window(int value) {
        if (value < 0 || value > 8192) throw std::invalid_argument("GLM: decode cache window must be 0..8192");
        decode_cache_window = value;
    }
    float set_affinity_for_quality(float value) {
        if (decode_graphs || !(value >= 0.f && value <= 2.f))
            throw std::invalid_argument("GLM: affinity quality changes require direct launches and affinity 0..2");
        const float previous = route_affinity; route_affinity = value; route_bonus_dirty = true;
        return previous;
    }
    void set_route_bias_enabled(bool value) {
        if (ram_tier) throw std::invalid_argument("GLM: route quality reference does not support the RAM routing tier");
        route_bias_enabled = value; route_bonus_dirty = true;
    }
    void set_check_cpu_residency(bool value) { check_cpu_residency = value; }
    bool set_tier_adaptation(bool value) {
        tier_drain();
        const bool previous = tier_adapt;
        tier_adapt = value;
        return previous;
    }
    void set_decode_cache_adapt(bool value) {
        if (value && !decode_cache_auto)
            throw std::invalid_argument("GLM: adaptive cache requires auto or extend decode cache");
        decode_cache_adapt = value;
        if (value) {
            decode_seen.assign(m.layers.size(), std::vector<uint64_t>(m.experts));
            expert_copies[0] = std::make_unique<ExpertCopyWorker>(primary_device);
            if (secondary) expert_copies[1] = std::make_unique<ExpertCopyWorker>(secondary->device);
        }
    }
    void set_verify_graphs(bool enabled) {verify_graphs=enabled;}
    void set_decode_graphs(bool enabled) {if(artifact.is_exl3()&&enabled)throw std::invalid_argument("EXL3: CUDA graphs are not qualified");decode_graphs = enabled; }
    // Connects to strata-glm-tp-worker (rank 1) and hands it every MoE layer's expert geometry. `share` is the
    // fraction of each expert's FFN rows computed here, rounded to whole quantization blocks.
    // STRATA_GLM_REMOTE_TP_CHECK=1: the same routes' whole experts again from the file mapping (unfused, unsplit),
    // compared with the local + remote sum; the largest difference relative to the layer's largest value is reported
    // at exit. Slow (the mapping's pages come back from disk); for validating the split, not for timing.
    void remote_tp_check(int l, const cpu::NativeFmt &f, const strata::core::ArtifactTensor &G, const strata::core::ArtifactTensor &U,
                         const strata::core::ArtifactTensor &D, int nt, int K, const float *weights, const float *sum) {
        const int H = m.hidden;
        cpu::NativeFmt full = f;
        full.fuse_h_quant = false;
        full.observer = nullptr;
        std::vector<float> results((size_t)nt * K * H, 0.f), expected((size_t)nt * H);
        std::vector<cpu::ExpertJobMulti> jobs = host_jobs;
        for (auto &job : jobs) {
            const int e = job.expert_id;
            job.blob = G.data() + (size_t)e * f.up_off;
            job.native_up = U.data() + (size_t)e * f.up_off;
            job.native_down = D.data() + (size_t)e * (f.bytes - f.down_off);
            for (auto &node : job.numa) node = {};
            for (int k = 0; k < job.nt; ++k) job.out[k] = results.data() + (job.out[k] - host_results.data());
        }
        if (!jobs.empty()) pool.run_split_multi_native(full, jobs.data(), jobs.size());
        pool.reduce_routed(results.data(), weights, expected.data(), nt, K, H);
        double peak = 0, diff = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            peak = std::max(peak, (double)std::fabs(expected[i]));
            diff = std::max(diff, (double)std::fabs(expected[i] - sum[i]));
        }
        const double rel = peak > 0 ? diff / peak : 0;
        auto &worst = remote_tp->check_worst[l];
        worst = std::max(worst, rel);
        ++remote_tp->check_calls;
    }
    // Remote TP: the format of this node's rows. Owned NUMA rows of a split layer hold only rows [0, keep), in their
    // own layout; otherwise the full tensors are read with their full strides, rows and down columns [0, keep).
    cpu::NativeFmt remote_local_fmt(int l, const cpu::NativeFmt &f) const {
        const int keep = remote_tp->split.at(l);
        cpu::NativeFmt local = f;
        if (remote_tp->trimmed.count(l)) {
            cpu::NativeFmt g;
            std::string error;
            if (!cpu::native_fmt(f.gu_type, f.d_type, f.n_embd, keep, g, error)) throw std::runtime_error(error);
            local.n_ff = g.n_ff; local.d_row = g.d_row; local.up_off = g.up_off; local.down_off = g.down_off;
            local.bytes = g.bytes; local.h_bytes = g.h_bytes;
        } else {
            local.n_ff = keep;
            local.h_bytes = ggml_row_size((ggml_type)f.d_act, keep);
        }
        return local;
    }
    void enable_remote_tp(const std::string &address, double share, bool reply_f16, bool send_weights) {
        if(artifact.is_exl3())throw std::invalid_argument("EXL3: remote tensor parallel is unsupported");
        auto tp = std::make_unique<RemoteTp>();
        tp->reply_format = reply_f16 ? strata::net::kReplyF16 : strata::net::kReplyF32;
        tp->hidden = (int)m.hidden;
        ucomm_config cfg;
        ucomm_config_default(&cfg);
        cfg.rank = 0;
        cfg.world = 2;
        cfg.root_addr = address.c_str();
        cfg.timeout_ms = 0; // the worker may still be filling its dummy weights
        RemoteTp::ok(ucomm_init(&cfg, &tp->comm), "connect");
        // Sized for the largest step and registered once: ucomm otherwise registers each rendezvous buffer per
        // message, a firmware command on ConnectX-3 that cost ~0.2 ms per layer. Resizes stay within capacity.
        tp->request.resize(sizeof(strata::net::TpRequest) + (size_t)cpu::MAXT * m.top_k * sizeof(strata::net::TpJob) +
                           (size_t)cpu::MAXT * cpu::kNativeActBytes);
        tp->wire.resize(strata::net::tp_reply_bytes(strata::net::kReplyF32, cpu::MAXT, m.hidden));
        for (auto *buffer : {&tp->request, &tp->wire}) {
            ucomm_mr_t *region = nullptr;
            if (ucomm_mr_reg(tp->comm, buffer->data(), buffer->size(), &region) == UCOMM_OK) tp->regions.push_back(region);
        }
        std::vector<uint8_t> setup(sizeof(strata::net::TpSetup));
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto &G = artifact.at(p + "ffn_gate_exps.weight");
            const auto &D = artifact.at(p + "ffn_down_exps.weight");
            const int ff = m.layers[l].intermediate;
            // Owned NUMA rows split the local rows across two nodes in 256-row chunks: keep whole 512-row steps.
            const int block = ff % 512 == 0 ? 512 : std::max<int>(256, (int)ggml_blck_size((ggml_type)D.tensor->type));
            int keep = (int)std::lround(share * ff / block) * block;
            keep = std::clamp(keep, block, ff - block);
            if (ff % block) throw std::invalid_argument("GLM remote TP: expert width is not whole blocks");
            tp->split[(int)l] = keep;
            strata::net::TpLayer spec{(int32_t)l, (int32_t)G.tensor->type, (int32_t)D.tensor->type, ff, keep,
                                      (int32_t)m.experts, m.layers[l].swiglu_limit, 0};
            const auto *b = (const uint8_t *)&spec;
            setup.insert(setup.end(), b, b + sizeof spec);
        }
        // The streamed rows' identity, so a worker keeping them in --weights-dir can skip the transfer: the pack's
        // own payload checksums when present, plus the geometry and 16 spread 4 KiB samples of every expert tensor.
        uint64_t fingerprint = 0;
        if (send_weights) {
            uint64_t h = 14695981039346656037ull;
            auto mix = [&](const void *p, size_t n) {
                for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t *)p)[i]) * 1099511628211ull;
            };
            for (const auto &[l, keep] : tp->split)
                for (const auto *part : {"gate", "up", "down"}) {
                    const auto &t = artifact.at("blk." + std::to_string(l) + ".ffn_" + part + "_exps.weight");
                    const int64_t geometry[4] = {l, keep, (int64_t)t.tensor->type, (int64_t)t.bytes};
                    mix(geometry, sizeof geometry);
                    if (const auto *sum = t.file->get("strata.expert_pack.hash." + t.tensor->name)) mix(&sum->u, sizeof sum->u);
                    const size_t sample = std::min<size_t>(4096, t.bytes);
                    for (int k = 0; k < 16; ++k) mix(t.data() + (t.bytes - sample) * k / 15, sample);
                }
            fingerprint = h ? h : 1;
        }
        strata::net::TpSetup head{strata::net::kTpMagic, (uint32_t)tp->split.size(), (uint32_t)m.hidden, tp->reply_format,
                                  send_weights ? strata::net::kWeightsStreamed : strata::net::kWeightsDummy, 0, fingerprint};
        std::memcpy(setup.data(), &head, sizeof head);
        RemoteTp::ok(ucomm_send(tp->comm, 1, strata::net::kTagSetup, setup.data(), setup.size()), "setup");
        strata::net::TpAccept accept{};
        RemoteTp::ok(ucomm_recv(tp->comm, 1, strata::net::kTagAccept, &accept, sizeof accept, nullptr), "accept");
        if (!accept.ok) {
            // The worker's rows scale with its share of each expert: the local share that would fit, if any.
            const auto &[l0, keep0] = *tp->split.begin();
            const double remote = 1.0 - double(keep0) / m.layers[l0].intermediate;
            const double fit = accept.needed_bytes ? remote * double(accept.available_bytes) / double(accept.needed_bytes) : 0.0;
            std::ostringstream message;
            message << "GLM remote TP: the worker refused the setup: " << std::string(accept.reason, strnlen(accept.reason, sizeof accept.reason));
            if (fit > 0) message << "; --remote-tp-share=" << std::fixed << std::setprecision(2) << 1.0 - fit << " or more would fit";
            throw std::runtime_error(message.str());
        }
        double sent_gib = 0, send_s = 0;
        if (send_weights && !accept.have) {
            // The worker's rows of every expert (expert_tp.hpp), double-buffered so gathering overlaps sending.
            const auto t0 = std::chrono::steady_clock::now();
            std::array<std::vector<uint8_t>, 2> staging;
            std::array<ucomm_req_t *, 2> inflight{};
            std::array<ucomm_mr_t *, 2> registered{};
            size_t largest = 0;   // the staging buffers are sized and registered once (IB), not per message
            for (const auto &[l, keep] : tp->split) {
                const auto &G = artifact.at("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
                const auto &D = artifact.at("blk." + std::to_string(l) + ".ffn_down_exps.weight");
                const size_t ff = m.layers[l].intermediate, rows = ff - keep;
                largest = std::max(largest, 2 * rows * ggml_row_size((ggml_type)G.tensor->type, m.hidden) +
                                                (size_t)m.hidden * (ggml_row_size((ggml_type)D.tensor->type, ff) -
                                                                    ggml_row_size((ggml_type)D.tensor->type, keep)));
            }
            for (int k = 0; k < 2; ++k) {
                staging[k].resize(largest);
                if (ucomm_mr_reg(tp->comm, staging[k].data(), largest, &registered[k]) != UCOMM_OK) registered[k] = nullptr;
            }
            int which = 0;
            for (const auto &[l, keep] : tp->split) {
                check_stop();
                const std::string p = "blk." + std::to_string(l) + ".";
                const auto &G = artifact.at(p + "ffn_gate_exps.weight"), &U = artifact.at(p + "ffn_up_exps.weight");
                const auto &D = artifact.at(p + "ffn_down_exps.weight");
                const size_t ff = m.layers[l].intermediate, rows = ff - keep, H = m.hidden;
                const size_t gu_row = ggml_row_size((ggml_type)G.tensor->type, H);
                const size_t d_full = ggml_row_size((ggml_type)D.tensor->type, ff);
                const size_t d_keep = ggml_row_size((ggml_type)D.tensor->type, keep), d_rest = d_full - d_keep;
                if (U.tensor->type != G.tensor->type || G.bytes != gu_row * ff * m.experts || D.bytes != d_full * H * m.experts)
                    throw std::runtime_error("GLM remote TP: unexpected expert tensor geometry at layer " + std::to_string(l));
                const size_t bytes = 2 * rows * gu_row + H * d_rest;
                for (int e = 0; e < m.experts; ++e) {
                    if (inflight[which]) RemoteTp::ok(ucomm_wait(inflight[which], nullptr), "weights");
                    inflight[which] = nullptr;
                    auto &blob = staging[which];   // sized for the largest layer: never reallocated
                    std::memcpy(blob.data(), G.data() + (e * ff + keep) * gu_row, rows * gu_row);
                    std::memcpy(blob.data() + rows * gu_row, U.data() + (e * ff + keep) * gu_row, rows * gu_row);
                    for (size_t r = 0; r < H; ++r)
                        std::memcpy(blob.data() + 2 * rows * gu_row + r * d_rest, D.data() + (e * H + r) * d_full + d_keep, d_rest);
                    RemoteTp::ok(ucomm_isend(tp->comm, 1, strata::net::kTagWeights, blob.data(), bytes, &inflight[which]), "weights");
                    which ^= 1;
                    sent_gib += bytes / double(1ull << 30);
                }
            }
            for (auto *r : inflight)
                if (r) RemoteTp::ok(ucomm_wait(r, nullptr), "weights");
            for (auto *r : registered)
                if (r) ucomm_mr_dereg(r);
            send_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        strata::net::TpReady ready{};
        RemoteTp::ok(ucomm_recv(tp->comm, 1, strata::net::kTagReady, &ready, sizeof ready, nullptr), "ready");
        std::cerr << "REMOTE_TP connected backend=" << ucomm_backend_name(tp->comm) << " layers=" << tp->split.size()
                  << " rows_here=" << tp->split.begin()->second << "/" << m.layers[tp->split.begin()->first].intermediate
                  << " worker_slots=" << ready.slots << " reply=" << (reply_f16 ? "f16" : "f32")
                  << " weights=" << (!send_weights ? "dummy" : accept.have ? "kept" : "streamed") << " worker_GiB="
                  << accept.needed_bytes / double(1ull << 30) << " sent_GiB=" << sent_gib << " send_s=" << send_s << '\n';
        remote_tp = std::move(tp);
    }
    bool uses_device_experts() const { return device_resident_experts; }
    void prepare_mtp_history(int depth) {
        if (batched_resident || tier_owned_reserve) enable_verify_history(std::max(depth, mtp_lookup_depth()));
    }
    void set_device_experts(bool enabled) { device_resident_experts = enabled; }
    void set_bench_mode(bool device, bool reduction, bool active_pools, bool split, int tasks, bool profiling,
                         bool combined_graphs = false, int columns = 128,
                         bool stable = false, bool buckets = false, bool reserve_groups = false,
                         bool defer_wait = false, bool restore_groups = false, int row_parts = 1) {
        device_resident_experts = device;
        device_reduction = reduction;
        active_index_pools = active_pools;
        split_hc_projection = split;
        pool.set_native_tasks_per_thread(tasks);
        profile = profiling;
        layer_graphs = combined_graphs;
        kda_columns = columns;
        stable_routes = stable || buckets;
        tensor_bucket_experts = buckets;
        preallocate_groups = reserve_groups;
        defer_copy_wait = defer_wait;
        restore_prefill_groups = restore_groups;
        kda_row_parts = row_parts;
        configure_prefill_schedule();
        device_lookup_ready = false;
    }
    void set_decode_cache_slots(int slots) { decode_cache_slots_limit = slots; }
    bool has_decode_cache() const { return decode_cache_budget || decode_cache_auto; }
    bool numa_relocated_after_prefill = false;
    void relocate_numa_after_prefill() {
        const char *value = std::getenv("STRATA_GLM_NUMA_AFTER_PREFILL");
        if (!value || std::string(value) == "0" || numa_relocated_after_prefill) return;
        if (std::string(value) != "1")
            throw std::invalid_argument("GLM: NUMA after prefill must be 0 or 1");
        const char *mapped = std::getenv("STRATA_GLM_MAPPED_OWNED");
        if (!gpu || !lazy_experts || !mapped || std::string(mapped) != "1" ||
            numa_experts.empty() || remote_tp || ram_tier)
            throw std::invalid_argument("GLM: NUMA after prefill needs lazy mapped experts with no RAM routing tier");
        finish_prefetch();
        check(cudaStreamSynchronize(stream));
        if (gpu->copy) check(cudaStreamSynchronize(gpu->copy));
        const auto start = std::chrono::steady_clock::now();
        size_t unmoved = 0;
        for (auto &[layer, tensors] : numa_experts) {
            check_stop();
            for (auto *tensor : {tensors.gate.get(), tensors.up.get(), tensors.down.get()}) {
                tensor->relocate_view_pages();
                unmoved += tensor->unmoved_pages;
            }
        }
        numa_relocated_after_prefill = true;
        std::cerr << "NUMA_AFTER_PREFILL bytes=" << numa_weight_bytes << " unmoved_pages=" << unmoved
                  << " ms=" << std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start).count() << '\n';
    }
    void prepare_decode_cache() {
        tier_drain();
        relocate_numa_after_prefill();
        // The prompt's unbiased routes decide the memory set before the first decode step.
        if (ram_tier && position >= kRamMinWindow) { ram_rebalance(); ram_last_rebalance = position; }
        finish_prefill_repairs(true);
        prepare_numa_weights();
        device_lookup_ready = false;
        if (!decode_cache_budget && !decode_cache_auto) { prepare_exact_cache(); return; }
        auto report_prefill = [](int device, const GpuPrefill* prefill) {
            if (prefill)
                std::cerr << "PREFILL_WORK device=" << device << " transferred_bytes=" << prefill->transferred
                          << " staging_ms=" << prefill->stage_ms << " actual_rows=" << prefill->tensor_rows_actual
                          << " padded_rows=" << prefill->tensor_rows_padded
                          << " dequant_values=" << prefill->tensor_dequant_values << '\n';
        };
        report_prefill(primary_device, gpu.get());
        if (secondary) report_prefill(secondary->device, secondary->gpu.get());
        prefill_cache.release_spare();
        if (secondary) secondary->cache.release_spare();
        if (!prepared_experts.empty() || mtp_experts || gpu_decode_experts)
            throw std::invalid_argument("GLM: prefix-trained decode cache requires unpacked CPU target experts");
        compact_decode();
        if (decode_cache_auto) {
            if (!decode_cache_extend) prefill_cache.layers.clear();
            if (secondary) {
                secondary->compact_decode();
                DeviceScope scope(secondary->device);
                if (!decode_cache_extend) secondary->cache.layers.clear();
            }
        }
        if (decode_cache_adapt && decode_cache_extend) {
            // Reuse the already-uploaded groups as independent mutable slots.
            // Group owners outlive the views. The optional restoration path
            // records each overwritten original slot before reusing its group.
            auto promote = [&](PrefillGroupCache &cache, int participant) {
                DeviceScope scope(participant ? secondary->device : primary_device);
                auto &destination = participant ? remote_decode_resident : decode_resident;
                size_t slots = 0, bytes = 0;
                for (auto &[layer, groups] : cache.layers) {
                    const auto p = "blk." + std::to_string(layer) + ".";
                    const auto &g = artifact.at(p + "ffn_gate_exps.weight");
                    const auto &u = artifact.at(p + "ffn_up_exps.weight");
                    const auto &d = artifact.at(p + "ffn_down_exps.weight");
                    const size_t blob = (g.bytes + u.bytes + d.bytes) / m.experts;
                    const size_t stride = prefill_expert_stride(g, u, d);
                    for (auto &[group, owner] : groups.entries) {
                        for (int i = 0; i < 16; ++i) {
                            const int expert = group * 16 + i;
                            destination[layer].emplace(expert,
                                std::make_unique<Device>((char *)owner->p + i * stride, blob));
                            victims[blob].push({expert_score(layer, expert), layer, expert, participant});
                            ++slots; bytes += blob;
                        }
                        if (restore_prefill_groups)
                            promoted_addresses[participant].emplace(reinterpret_cast<uintptr_t>(owner->p),
                                                                    decode_group_storage[participant].size());
                        decode_group_storage[participant].push_back({layer, group, stride, std::move(owner)});
                    }
                }
                cache.layers.clear();
                std::cerr << "DECODE_CACHE_PROMOTE device=" << (participant ? secondary->device : primary_device)
                          << " slots=" << slots << " MiB=" << bytes / double(MiB) << '\n';
            };
            promote(prefill_cache, 0);
            if (secondary) promote(secondary->cache, 1);
        }
        struct Candidate { uint64_t count; int layer, expert; size_t bytes; };
        std::vector<Candidate> candidates;
        // STRATA_GLM_EXPERT_PRIOR: calibration routing counts (coverage.json) blended with this prompt's routes
        // as prior pseudo-tokens, so short prompts still select the generally hottest experts.
        const auto prior = load_expert_prior();
        constexpr double prior_tokens = 256;
        if (fixed_cache_recent && !decode_cache_auto)
            std::cerr << "DECODE_CACHE_WINDOW mode=fixed tokens=" << decode_cache_window << '\n';
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            if (canon_experts) {
                // A layer left in another format would round differently on the GPU; it stays on the CPU.
                if (!canon_layer((int)l)) continue;
            }
            if (!prior.empty()) {
                const auto &share = prior.at(l);
                uint64_t routed = 0;
                for (uint64_t c : prefill_routes[l]) routed += c;
                const double tokens = routed / double(m.top_k);
                const auto p = "blk." + std::to_string(l) + ".";
                const size_t bytes = (artifact.at(p + "ffn_gate_exps.weight").bytes + artifact.at(p + "ffn_up_exps.weight").bytes +
                                      artifact.at(p + "ffn_down_exps.weight").bytes) / m.experts;
                for (int e = 0; e < m.experts; ++e) {
                    // Expected routes per token, scaled to an integer rank.
                    const double rate = (prefill_routes[l][e] + prior_tokens * m.top_k * share[e]) / (tokens + prior_tokens);
                    if (rate > 0) candidates.push_back({(uint64_t)std::llround(rate * 1e6), (int)l, e, bytes});
                }
                continue;
            }
            const auto p = "blk." + std::to_string(l) + ".";
            const size_t bytes = (artifact.at(p + "ffn_gate_exps.weight").bytes +
                artifact.at(p + "ffn_up_exps.weight").bytes +
                artifact.at(p + "ffn_down_exps.weight").bytes) / m.experts;
            const size_t stride = prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight"));
            for (int e = 0; e < m.experts; ++e) {
                if (decode_cache_extend && (decode_resident[l].count(e) || remote_decode_resident[l].count(e) ||
                    prefill_cache.expert(l, e, stride) || (secondary && secondary->cache.expert(l, e, stride)))) continue;
                if (prefill_routes[l][e]) candidates.push_back({prefill_routes[l][e], (int)l, e, bytes});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            if (a.count != b.count) return a.count > b.count;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });
        tier_rate.clear();
        if (!prior.empty()) {
            tier_rate.assign(m.layers.size(), std::vector<float>(m.experts));
            for (const auto &c : candidates) tier_rate[c.layer][c.expert] = c.count * 1e-6f;
        }
        size_t used = 0, slots = 0;
        std::array<size_t, 2> remaining{};
        std::array<uint64_t, 2> load{};
        std::array<size_t, 2> device_slots{}, device_bytes{};
        if (decode_cache_auto) {
            for (int i = 0; i < (secondary ? 2 : 1); ++i) {
                DeviceScope scope(i ? secondary->device : primary_device);
                // The compact arena and resident executor are already owned.
                // Keep space for graph/runtime growth and the small output
                // buffers rather than reserving a prefill-sized workspace.
                const size_t reserve = (i ? 64 : 256) * MiB;
                size_t free = 0, total = 0;
                check(cudaMemGetInfo(&free, &total));
                const size_t physical = Device::physical_reserve() + reserve;
                const size_t logical = Device::live() + reserve;
                remaining[i] = std::min(free > physical ? free - physical : 0,
                                       Device::limit() > logical ? Device::limit() - logical : 0);
                // Coarse slabs absorb small driver allocation changes between requests.
                remaining[i] = remaining[i] / (64 * MiB) * (64 * MiB);
                DeviceTag tag("tier");
                if (remaining[i] && !candidates.empty())
                    decode_cache_storage[i] = std::make_unique<Device>(remaining[i]);
            }
        }
        uint64_t fingerprint = 14695981039346656037ULL;
        // Left free for what is allocated after the tier: with MTP that includes the draft block's fixed weights,
        // the verify history and two rollback checkpoints.
        size_t scratch_reserve = (capture_hidden ? 1280 : 768) * MiB;
        if (capture_hidden && tier_owned_reserve) {
            // These allocations already count against live/physical memory. Do not reserve
            // them twice. Cap the deduction so checkpoint and workspace headroom remain.
            auto &account = Device::account();
            std::lock_guard lock(account.tag_mutex);
            const size_t owned = account.by_tag["dense_mtp"] + account.by_tag["verify_history"];
            if (sequences.empty() && account.by_tag["dense_mtp"] && account.by_tag["verify_history"])
                scratch_reserve = 512 * MiB; // one target checkpoint, draft anchor and later decode buffers
            else scratch_reserve -= std::min<size_t>(512 * MiB, owned);
            std::cerr << "DECODE_RUNTIME owned_MiB=" << owned / double(MiB)
                      << " reserve_MiB=" << scratch_reserve / double(MiB) << '\n';
        }
        // Opt-in measured headroom for small-memory deployments. Physical free
        // memory and the allocator limit remain enforced for every admission.
        if (const char *value = std::getenv("STRATA_GLM_TIER_RUNTIME_RESERVE_MIB")) {
            size_t end = 0;
            const auto mib = std::stoull(value, &end);
            const size_t minimum = capture_hidden ? 512 : 256;
            if (value[end] || mib < minimum || mib > 4096)
                throw std::invalid_argument("GLM: tier runtime reserve must be 256..4096 MiB (at least 512 with MTP)");
            scratch_reserve = mib * MiB;
            std::cerr << "DECODE_RUNTIME override_reserve_MiB=" << mib << '\n';
        }
        for (const auto &c : candidates) {
            check_stop();
            if (decode_cache_slots_limit && slots >= (size_t)decode_cache_slots_limit) break;
            int participant = 0;
            if (decode_cache_auto) {
                if (remaining[0] < c.bytes && remaining[1] < c.bytes) continue;
                if (remaining[0] < c.bytes ||
                    (remaining[1] >= c.bytes && load[1] < load[0])) participant = 1;
            } else if (c.bytes > decode_cache_budget - used) continue;
            DeviceScope scope(participant ? secondary->device : primary_device);
            if (!decode_cache_auto && (Device::live() > Device::limit() || Device::limit() - Device::live() < scratch_reserve ||
                c.bytes > Device::limit() - Device::live() - scratch_reserve)) break;
            std::unique_ptr<Device> entry;
            if (decode_cache_auto) {
                entry = std::make_unique<Device>((char *)decode_cache_storage[participant]->p +
                                                 device_bytes[participant], c.bytes);
            } else {
                size_t free = 0, total = 0;
                check(cudaMemGetInfo(&free, &total));
                const size_t physical = Device::physical_reserve();
                if (free < physical + scratch_reserve || c.bytes > free - physical - scratch_reserve) break;
                entry = std::make_unique<Device>(c.bytes);
            }
            const auto p = "blk." + std::to_string(c.layer) + ".";
            size_t offset = 0;
            for (const char *role : {"gate", "up", "down"}) {
                const auto &t = artifact.at(p + "ffn_" + role + "_exps.weight");
                const size_t bytes = t.bytes / m.experts;
                if (offset > entry->bytes || bytes > entry->bytes - offset)
                    throw std::runtime_error("GLM: decode cache expert geometry mismatch");
                // Registered model pages let both devices fill concurrently; the streams are joined below.
#ifdef STRATA_USE_HIP
                upload_hip_file(*entry, offset, t.data() + c.expert * bytes, bytes);
#else
                if (async_decode_fill)
                    check(cudaMemcpyAsync((char *)entry->p + offset, t.data() + c.expert * bytes, bytes,
                                          cudaMemcpyHostToDevice, participant ? secondary->stream : stream));
                else check(cudaMemcpy((char *)entry->p + offset, t.data() + c.expert * bytes,
                                 bytes, cudaMemcpyHostToDevice));
#endif
                offset += bytes;
            }
            if (decode_cache_adapt)
                victims[c.bytes].push({expert_score(c.layer, c.expert), c.layer, c.expert, participant});
            auto &destination = participant ? remote_decode_resident : decode_resident;
            destination[c.layer].emplace(c.expert, std::move(entry));
            if (decode_cache_auto) {
                remaining[participant] -= c.bytes;
                load[participant] += c.count;
            }
            fingerprint ^= (uint64_t)c.layer * m.experts + c.expert;
            fingerprint *= 1099511628211ULL;
            used += c.bytes; ++slots;
            ++device_slots[participant]; device_bytes[participant] += c.bytes;
        }
        if (async_decode_fill) {
            if (secondary) {
                DeviceScope scope(secondary->device);
                check(cudaStreamSynchronize(secondary->stream));
            }
            check(cudaStreamSynchronize(stream));
        }
        rebalance_decode_cache();
        if (decode_cache_slots_limit && slots != (size_t)decode_cache_slots_limit)
            throw std::runtime_error("GLM: GPU headroom cannot reproduce requested decode cache slots");
        if (routing_events && routing_trace.is_open()) {
            for (size_t l = 0; l < decode_resident.size(); ++l) {
                routing_trace << "{\"kind\":\"cache\",\"request\":" << routing_request << ",\"layer\":" << l << ",\"experts\":[";
                bool comma = false;
                for (const auto &[expert, entry] : decode_resident[l]) {
                    if (comma) routing_trace << ',';
                    routing_trace << expert; comma = true;
                }
                routing_trace << "]}\n";
            }
        }
        std::cerr << "DECODE_CACHE prefix_trained=1 slots=" << slots
                  << " MiB=" << used / double(MiB) << " fingerprint=" << fingerprint << '\n';
        if (decode_cache_auto)
            for (int i = 0; i < (secondary ? 2 : 1); ++i)
                std::cerr << "DECODE_CACHE_DEVICE device=" << (i ? secondary->device : primary_device)
                          << " slots=" << device_slots[i] << " MiB=" << device_bytes[i] / double(MiB)
                          << " prefix_routes=" << load[i] << '\n';
        prepare_exact_cache();
    }
    // Per-layer routing shares from a calibration coverage.json ({"projections":[{"name":"blk.L.gu","counts":[...]}]}).
    std::vector<std::vector<double>> load_expert_prior() const {
        const char *path = std::getenv("STRATA_GLM_EXPERT_PRIOR");
        if (!path || !*path) return {};
        const auto json = strata::artifact::read_json(path);
        std::vector<std::vector<double>> shares(m.layers.size());
        for (const auto &entry : json.at("projections").array) {
            const auto &name = entry.at("name").string();
            if (!name.starts_with("blk.") || !name.ends_with(".gu")) continue;
            const size_t layer = std::stoul(name.substr(4, name.size() - 7));
            const auto &counts = entry.at("counts").array;
            if (layer >= m.layers.size() || counts.size() != (size_t)m.experts)
                throw std::runtime_error("GLM: expert prior geometry: " + name);
            double total = 0;
            for (const auto &c : counts) total += (double)c.integer();
            if (total <= 0) throw std::runtime_error("GLM: empty expert prior: " + name);
            for (const auto &c : counts) shares[layer].push_back((double)c.integer() / total);
        }
        for (size_t l = 0; l < m.layers.size(); ++l)
            if (m.layers[l].ffn == strata::core::FfnKind::Moe && shares[l].empty())
                throw std::runtime_error("GLM: expert prior lacks layer " + std::to_string(l));
        return shares;
    }
    void compact_decode() {
        if (!gpu || prefill_width_saved || gpu_decode_experts) return;
        finish_prefetch();
        check(cudaStreamSynchronize(stream));
        if (gpu->copy) check(cudaStreamSynchronize(gpu->copy));
        mixer_graphs.clear();
        if (secondary && secondary->tp) secondary->tp->graphs.clear();
        prefill_width_saved = gpu->chunk;
        phase.clear(); scratch.clear(); gpu.reset();
        gpu = std::make_unique<GpuPrefill>(cpu::MAXT, stream, true);
        fast = false;
    }
    bool prepare_mtp_for_cache(const std::vector<int> &prompt, int depth) {
        if (!tier_owned_reserve || !capture_hidden || !mtp_cpu_experts ||
            (!decode_cache_budget && !decode_cache_auto)) return false;
        prepare_mtp(prompt);
        prepare_mtp_history(depth);
        return true;
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
                weight(name, (name.ends_with("attn_k_b.weight") || name.ends_with("attn_v_b.weight")) && !floating_q8_weight(name));
        const auto &G = artifact.at("blk.45.ffn_gate_exps.weight");
        const auto &U = artifact.at("blk.45.ffn_up_exps.weight");
        const auto &D = artifact.at("blk.45.ffn_down_exps.weight");
        mtp_layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, layer.intermediate);
        mtp_layout.swiglu_limit = layer.swiglu_limit;
        mtp_expert_bytes = (G.bytes + U.bytes + D.bytes) / m.experts;
        if (!mtp_cpu_experts) {
            DeviceTag tag("mtp_experts");
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
            std::vector<unsigned long long> lookup(m.experts);
            for (int e = 0; e < m.experts; ++e) lookup[e] = (unsigned long long)mtp_experts->p + e * mtp_expert_bytes;
            mtp_lookup = std::make_unique<Device>(lookup.size() * 8);
            mtp_lookup->put(lookup.data(), lookup.size() * 8);
        }
        ensure_draft_state();
        mtp_ready = true;
        if (mtp_primed) {
            if (mtp_position != (int)prompt.size() - 1)
                throw std::runtime_error("GLM: batched MTP priming does not cover the prompt");
            std::cerr << "MTP_PRIME batched=1 positions=" << mtp_position << '\n';
        } else {
            mtp_position = 0;
            sync_mtp(prompt, 0);
        }
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        std::cerr << "MTP free_MiB=" << free_bytes / double(MiB) << '\n';
        std::cerr << "MTP expert_backend=" << (mtp_cpu_experts ? "cpu" : "gpu") << '\n';
        std::cerr << "MTP resident_MiB=" << (mtp_experts ? mtp_experts->bytes : 0) / double(MiB)
                  << " allocated_MiB=" << Device::live() / double(MiB) << " primed_tokens=" << prompt.size() << '\n';
        release_gpu_dense_host_pages(true);
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
    // Draft-block buffers only: save() and restore_mtp() then cover just the MTP state.
    Checkpoint mtp_checkpoint() {
        Checkpoint cp;
        size_t bytes = 0;
        cp.draft_begin = 0;
        if (states.size() > m.layers.size()) {
            auto &state = states[m.layers.size()];
            for (Device *buffer : {state.recurrent.get(), state.conv_q.get(), state.conv_k.get(),
                                   state.conv_v.get(), state.keys.get(), state.gates.get()})
                if (buffer) {
                    cp.buffers.emplace_back(buffer, bytes);
                    bytes += buffer->bytes;
                }
        }
        cp.storage = std::make_unique<Device>(std::max<size_t>(bytes, 256));
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
        tp_stale = true;
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

    void check_sequences(const std::vector<int> &prompt) {
        if (sequences.empty() || prompt.empty() || route_affinity != 0)
            throw std::invalid_argument("GLM: sequence parity requires slots, tokens and unbiased routing");
        compact_decode();
        std::vector<int> ids, inputs;
        for (int slot = 0; slot < (int)sequences.size(); ++slot) {
            select_sequence(slot);
            for (int t = 0; t <= slot; ++t) step(prompt[(t + slot) % prompt.size()]);
            ids.push_back(slot); inputs.push_back(prompt[(2 * slot + 1) % prompt.size()]);
        }
        for (int width = (int)ids.size(); width >= 1; --width) {
            std::vector<int> selected(ids.begin(), ids.begin() + width), tokens(inputs.begin(), inputs.begin() + width);
            std::vector<Checkpoint> before;
            for (int slot : selected) { select_sequence(slot); before.push_back(checkpoint()); save(before.back()); }
            auto actual = step_sequences(selected, tokens);
            for (int row = 0; row < width; ++row) {
                select_sequence(selected[row]); restore(before[row]);
                auto expected = step(tokens[row]);
                if (std::memcmp(actual.data() + (size_t)row * m.vocab, expected.data(), m.vocab * sizeof(float)))
                    throw std::runtime_error("GLM: independent sequence logits differ at width " + std::to_string(width) +
                                             " row " + std::to_string(row));
            }
            std::reverse(ids.begin(), ids.end()); std::reverse(inputs.begin(), inputs.end());
            std::cerr << "SEQUENCE_PARITY width=" << width << " logits=bitwise_equal\n";
        }
        if (sequences.size() >= 2) {
            std::vector<int> slots{0,0,0,1,1,1}, tokens;
            std::vector<Checkpoint> before;
            for (int slot : {0,1}) { select_sequence(slot); before.push_back(checkpoint()); save(before.back()); }
            for (int t = 0; t < 6; ++t) tokens.push_back(prompt[t % prompt.size()]);
            auto actual = step_sequences(slots, tokens, true);
            for (int slot : {0,1}) {
                select_sequence(slot); restore(before[slot]);
                for (int t = 0; t < 3; ++t) {
                    auto expected = step(tokens[slot * 3 + t]);
                    if (std::memcmp(actual.data() + (size_t)(slot * 3 + t) * m.vocab, expected.data(), m.vocab * 4))
                        throw std::runtime_error("GLM: independent verify windows differ");
                }
            }
            std::cerr << "SEQUENCE_PARITY windows=2x3 logits=bitwise_equal\n";
        }
    }
    int enable_gpu(int requested, size_t total_budget, bool reserve_checkpoint = false) {
        if(artifact.is_exl3()) {
            if(gpu)return gpu->chunk;
            int width=requested?requested:256;
            const size_t reserve = Device::budget_reserve();
            if(width<1||width>256||total_budget<2048*MiB||total_budget<=reserve)throw std::invalid_argument("EXL3: GPU prefill batch must be 1..256 and its budget must exceed the runtime reserve");
            size_t free,total;check(cudaMemGetInfo(&free,&total));
            Device::limit()=std::min(total_budget-reserve,Device::live()+(free>reserve?free-reserve:0));
            check(cublasSetMathMode(blas,glm_blas_math_mode));
            tensor_mla=false;
            gpu=std::make_unique<GpuPrefill>(width,stream,false,capacity,false,false,false,true);
            prepare_native_dense_q8();
            if(native_expert_gpu_limit) {
                if(!native_dense_q8)throw std::invalid_argument("EXL3 GPU expert cache requires resident dense quantization");
                size_t free,total;check(cudaMemGetInfo(&free,&total));
                const size_t physical=free>Device::physical_reserve()?free-Device::physical_reserve():0;
                const size_t available=std::min(physical,Device::limit()>Device::live()?Device::limit()-Device::live():0);
                if(available<896*MiB || native_expert_gpu_limit>available-896*MiB)
                    throw std::runtime_error("EXL3 GPU expert cache plus 896 MiB runtime reserve exceed GPU budget: available_MiB="+std::to_string(available/MiB));
            }
            return width;
        }
        if (gpu)
            return gpu->chunk;
        if (floating_q8_repack && (!q8_decode || secondary || direct_upload_enabled()))
            throw std::invalid_argument("GLM: floating Q8 repacking requires single-GPU GGUF Q8 decode");
        if (dense_q4_repack && (secondary || direct_upload_enabled()))
            throw std::invalid_argument("GLM: dense Q4 repacking requires ordinary single-GPU GGUF prefill");
        if (m.layers.size() != 45 || m.hidden != 4096 || m.experts > 288 || m.experts % 16 || m.top_k != 8 ||
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
        if (total_budget <= std::max(1024 * MiB, Device::budget_reserve()))
            throw std::invalid_argument("GLM: GPU budget must exceed 1024 MiB and its runtime reserve");
        for (const auto &cache : expert_cache)
            if (cache)
                throw std::invalid_argument("GLM: GPU prefill currently requires expert-cache-mib=0");
        size_t fixed = 0;
        auto floating = [](const std::string &name) {
            const std::string n = canonical_tensor_name(name);
            return n.ends_with("hc_attn_fn.weight") || n.ends_with("hc_ffn_fn.weight") ||
                   n.ends_with("attn_k_b.weight") || n.ends_with("attn_v_b.weight");
        };
        auto eligible = [](const std::string &n) {
            return n != "token_embd.weight" && !n.starts_with("blk.45.") && !n.ends_with("_exps.weight");
        };
        for (const auto &[n, t] : artifact.tensors())
            if (eligible(n)) {
                size_t bytes = dense_weight_bytes(n);
                if (floating(n) && !floating_q8_weight(n)) {
                    bytes = 4;
                    for (auto d : t.tensor->shape)
                        bytes *= d;
                }
                if (floating_q8_weight(n)) {
                    bytes = floating_q8_bytes(n);
                    size_t values = 1;
                    for (auto d : t.tensor->shape) values *= d;
                    q8_scratch_bytes[q8_scratch_role(n)] = std::max(q8_scratch_bytes[q8_scratch_role(n)], values * 4);
                }
                fixed += bytes;
            }
        for (size_t bytes : q8_scratch_bytes) fixed += bytes;   // prefill's FP32 expansions of Q8-only weights
        size_t free, total;
        check(cudaMemGetInfo(&free, &total));
        const size_t runtime_reserve = Device::budget_reserve();
        Device::limit() = std::min(total_budget - runtime_reserve,
            Device::live() + (free > runtime_reserve ? free - runtime_reserve : 0));
        auto needed = [&](int b) {
            return Device::live() + fixed + GpuPrefill::arena_bytes(b, capacity) +
                   (128 + 32 + 128) * MiB + 2 * GpuPrefill::slot_bytes + (size_t)b * m.hidden * 7 * 4 + MiB +
                   (reserve_checkpoint ? 152 * MiB : 0);
        };
        int width = requested ? requested : 4096;
        if (!sequences.empty()) {
            const int original = width;
            while (width > cpu::MAXT && needed(width) > Device::limit()) width /= 2;
            if (width != original) std::cerr << "BATCH_PREFILL requested=" << original << " effective=" << width << '\n';
        } else if (!requested && needed(width) > Device::limit()) width = 2048;
        if (width < 1 || width > 16384 || needed(width) > Device::limit())
            throw std::runtime_error(
                "GLM: prefill cannot fit GPU budget: needed_MiB=" + std::to_string(needed(width) / double(MiB)) +
                " limit_MiB=" + std::to_string(Device::limit() / double(MiB)) +
                " fixed_MiB=" + std::to_string(fixed / double(MiB)) +
                " live_MiB=" + std::to_string(Device::live() / double(MiB)) +
                "; reduce context or use legacy prefill");
        check(cublasSetMathMode(blas, glm_blas_math_mode));
        budget = SIZE_MAX;
        for (const auto &[n, t] : artifact.tensors())
            if (eligible(n)) {
                const bool q8 = floating_q8_weight(n);
                weight(n, floating(n) && !q8);
                if (q8) {
                    // Persistent prefill expansion buffers, so a later decode cache cannot take their space.
                    size_t values = 1;
                    for (auto d : t.tensor->shape) values *= d;
                    auto &scratch = q8_fp32_scratch[q8_scratch_role(n)];
                    if (!scratch || scratch->bytes < values * 4) scratch = std::make_unique<Device>(values * 4);
                }
            }
        gpu = std::make_unique<GpuPrefill>(width, stream, false, capacity);
        // Allocate persistent residual and output buffers before carving phase scratch.
        for (auto name : {"streams", "collapsed", "x", "y", "hc_coeff", "output"}) {
            size_t count = std::string(name) == "streams"    ? width * m.hidden * 4
                           : std::string(name) == "hc_coeff" ? width * 24
                           : std::string(name) == "output"   ? m.vocab
                                                             : width * m.hidden;
            buf(name, count);
        }
        std::cerr << "GPU allocated_MiB=" << Device::live() / (double)MiB
                  << " reserved_runtime_MiB=" << runtime_reserve / MiB << " budget_MiB=" << total_budget / MiB << '\n';
        release_gpu_dense_host_pages();
        if (const char *early = std::getenv("STRATA_GLM_TIER_PREALLOCATE")) {
            if (std::string(early) != "0" && std::string(early) != "1")
                throw std::invalid_argument("GLM: tier preallocate must be 0 or 1");
            if (std::string(early) == "1") prepare_tier_staging();
        }
        if (const char *early = std::getenv("STRATA_GLM_MAILBOX_PREALLOCATE")) {
            if (std::string(early) != "0" && std::string(early) != "1")
                throw std::invalid_argument("GLM: mailbox preallocate must be 0 or 1");
            if (std::string(early) == "1") {
                if (!step_pipeline || gpu_decode_experts)
                    throw std::invalid_argument("GLM: mailbox preallocate requires CPU step pipeline");
                prepare_step_mailbox();
            }
        }
        return width;
    }
    void configure_prefill(const std::vector<int> &devices, size_t total_budget,
                           size_t cache_bytes, bool automatic, bool reserve_mtp, bool tensor_experts = false,
                           bool tensor_batches = false) {
        if ((floating_q8_repack || dense_q4_repack) && (devices.size() != 1 || artifact.is_exl3()))
            throw std::invalid_argument("GLM: dense repacking requires one GPU and GGUF weights");
        if(artifact.is_exl3()) {if(devices.size()!=1||cache_bytes||automatic||reserve_mtp||tensor_experts||tensor_batches)throw std::invalid_argument("EXL3: requires single GPU, CPU experts, no speculation/cache");return;}
        if (primary_prefill_groups != 9 && (devices.size() != 2 || m.experts != 288))
            throw std::invalid_argument("GLM: weighted prefill partition requires two GPUs and 288 experts");
        std::cerr << "PREFILL_PARTITION primary_groups=" << (devices.size() == 2 ? primary_prefill_groups : prefill_expert_groups)
                  << " secondary_groups=" << (devices.size() == 2 ? 18 - primary_prefill_groups : 0) << '\n';
        tensor_prefill_experts = tensor_experts;
        tensor_batched_experts = tensor_batches;
        if (gpu && tensor_batches) {
            gpu->dq.reset();
            gpu->dq = std::make_unique<Device>(512 * MiB);
            gpu->gemm->rebind((uint16_t *)gpu->dq->p, gpu->dq->bytes / 2,
                              gpu->blas_workspace->p, gpu->blas_workspace->bytes);
        }
        if (devices.size() > 1 && !gpu)
            throw std::invalid_argument("GLM: two GPUs require GPU prefill");
        if (devices.size() == 2)
            secondary = std::make_unique<SecondaryPrefill>(devices[1], primary_device, gpu->chunk, total_budget, tensor_experts, tensor_batches);
        if (secondary && split_mla && secondary->peer) {
            if (q8_decode) throw std::invalid_argument("GLM: Q8 decode does not support split MLA");
            // Resident before the cache allowance below is measured.
            check(cudaStreamSynchronize(stream));
            for (size_t l = 0; l < m.layers.size(); ++l) {
                if (m.layers[l].mixer == strata::core::MixerKind::Kda) continue;
                for (const char *role : {"attn_k_b.weight", "attn_v_b.weight"}) {
                    const auto name = "blk." + std::to_string(l) + "." + role;
                    auto &W = weight(name, true);
                    DeviceScope scope(secondary->device);
                    auto copy = std::make_unique<Device>(W.bytes);
                    check(cudaMemcpyPeer(copy->p, secondary->device, W.p, primary_device, W.bytes));
                    secondary->mla_weights.emplace(name, std::move(copy));
                }
            }
            DeviceScope scope(secondary->device);
            check(cublasCreate(&secondary->blas));
            check(cublasSetStream(secondary->blas, secondary->stream));
            check(cublasSetMathMode(secondary->blas, glm_blas_math_mode));
        }
        if (secondary && (decode_tp || kda_split) && secondary->peer) setup_decode_tp();
        auto allowance = [&](int device, size_t reserve, PrefillGroupCache &cache) {
            DeviceScope scope(device);
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total));
            const size_t accounted = Device::live() < Device::limit() ? Device::limit() - Device::live() : 0;
            const size_t guard = Device::physical_reserve();
            const size_t physical = free > guard ? free - guard : 0;
            const size_t spare = std::min(accounted, physical);
            const size_t available = spare > reserve ? spare - reserve : 0;
            if (!automatic && cache_bytes > available)
                throw std::invalid_argument("GLM: prefill expert cache exceeds device headroom");
            const size_t bytes = automatic ? available : cache_bytes;
            std::cerr << "PREFILL_CACHE device=" << device << " budget_MiB=" << bytes / double(MiB)
                      << " reserve_MiB=" << reserve / MiB << '\n';
            // Give every layer the same initial group count, then spend the
            // remainder on complete groups. Independent per-layer division
            // stranded several GiB with the model's mixed quantization sizes.
            std::vector<std::pair<size_t, int>> costs;
            size_t round = 0;
            for (size_t l = 0; l < m.layers.size(); ++l) {
                if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                const auto p = "blk." + std::to_string(l) + ".";
                const size_t cost = 16 * prefill_expert_stride(artifact.at(p + "ffn_gate_exps.weight"),
                    artifact.at(p + "ffn_up_exps.weight"), artifact.at(p + "ffn_down_exps.weight")) + 16384;
                costs.emplace_back(cost, l); round += cost;
            }
            const size_t maximum = secondary ? (device == primary_device ? primary_prefill_groups : 18 - primary_prefill_groups) : prefill_expert_groups;
            const size_t base = round ? std::min(maximum, bytes / round) : 0;
            size_t left = bytes - base * round;
            std::sort(costs.begin(), costs.end());
            for (auto [cost, l] : costs) {
                size_t groups = base;
                if (groups < maximum && cost <= left) { ++groups; left -= cost; }
                cache.planned_groups[l] = groups;
            }
            std::cerr << "PREFILL_CACHE_PLAN device=" << device << " base_groups=" << base
                      << " unused_MiB=" << left / double(MiB) << '\n';
            cache.planned_groups[(int)m.layers.size()] = 0; // the draft block is never admitted
            return bytes / 42;
        };
        prefill_cache.per_layer = allowance(primary_device, reserve_mtp ? 4096 * MiB : 768 * MiB, prefill_cache);
        // The first MMQ products can lazily allocate driver workspace. A short
        // prompt may encounter the largest mixed-quant group late in admission.
        if (secondary) secondary->cache.per_layer = allowance(secondary->device, 256 * MiB, secondary->cache);
        configure_prefill_schedule();
    }
    void set_decode_prefill_cache(bool enabled) {
        if(artifact.is_exl3()&&enabled)throw std::invalid_argument("EXL3: GGUF expert caches are unsupported");
        if (enabled && (!gpu || !prepared_experts.empty() || gpu_decode_experts))
            throw std::invalid_argument("GLM: decode prefill cache requires unpacked CPU experts and GPU prefill");
        decode_prefill_cache = enabled;
        if (enabled) {
            if (m.hidden != 4096 || m.top_k != 8 || std::any_of(m.layers.begin(), m.layers.end(), [](const auto &l) {
                    return l.ffn == strata::core::FfnKind::Moe && l.intermediate != 2048;
                }))
                throw std::invalid_argument("GLM: resident decode requires the inspected expert geometry");
            if (!cached_decode) cached_decode = std::make_unique<ResidentExecutor>();
            if (secondary) {
                DeviceScope scope(secondary->device);
                if (!secondary->decode) secondary->decode = std::make_unique<ResidentExecutor>();
            }
        }
    }
    void set_secondary_enabled(bool value) {
        if (secondary_enabled == value) return;
        secondary_enabled = value;
        device_lookup_ready = false;
        // Admission follows group ownership. A parity run changes ownership
        // between its one-GPU reference and two-GPU execution.
        prefill_cache.reset();
        if (secondary) secondary->cache.reset();
    }
    void warm_weights(bool lock = false, const std::vector<int> &upload_devices = {}) {
        if(artifact.is_exl3())throw std::invalid_argument("EXL3: GGUF weight locking is unsupported");
        if(lock && !numa_sources.empty())throw std::invalid_argument("Q2 NUMA weights cannot lock the original GGUF mappings");
        if (direct_upload_enabled() && !lock)
            throw std::invalid_argument("GLM: direct weight upload requires --lock-weights");
        const auto start = std::chrono::steady_clock::now();
        volatile uint8_t checksum = 0;
        size_t locked = 0;
#ifdef __linux__
        if (lock) {
            const size_t page = (size_t)sysconf(_SC_PAGESIZE);
            std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
            for (const auto &[name, t] : artifact.tensors())
                if (!name.starts_with("blk.45."))
                    ranges.emplace_back((uintptr_t)t.data() / page * page,
                                        ((uintptr_t)t.data() + t.bytes + page - 1) / page * page);
            std::sort(ranges.begin(), ranges.end());
            size_t required = 0;
            uintptr_t end = 0;
            for (const auto &[first, last] : ranges) {
                required += last > std::max(first, end) ? last - std::max(first, end) : 0;
                end = std::max(end, last);
            }
            rlimit limit{};
            if (getrlimit(RLIMIT_MEMLOCK, &limit) ||
                (limit.rlim_cur != RLIM_INFINITY && required > limit.rlim_cur))
                throw std::runtime_error("GLM: --lock-weights needs a larger locked-memory limit (ulimit -l)");
        }
#endif
        for (const auto &[name, t] : artifact.tensors())
            if (!name.starts_with("blk.45.") && !numa_sources.count(t.tensor) && !numa_local_only.count(t.tensor)) {
                if (lock) {
#ifdef __linux__
                    // A file-system client can invalidate clean file-backed
                    // pages even after mlock. Materialize byte-identical COW
                    // pages in this MAP_PRIVATE view first. The GGUF files and
                    // all tensor addresses/layouts remain unchanged.
                    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
                    const uintptr_t first = (uintptr_t)t.data() / page * page;
                    const size_t extent = ((uintptr_t)t.data() + t.bytes - first + page - 1) / page * page;
                    if (mprotect((void *)first, extent, PROT_READ | PROT_WRITE))
                        throw std::runtime_error(std::string("GLM: private weight mapping failed: ") + std::strerror(errno));
                    auto *native = const_cast<volatile uint8_t *>(t.data());
                    for (size_t i = 0; i < t.bytes; i += page) native[i] = native[i];
                    native[t.bytes - 1] = native[t.bytes - 1];
                    if (mprotect((void *)first, extent, PROT_READ))
                        throw std::runtime_error("GLM: cannot restore read-only weight mapping");
                    if (mlock(t.data(), t.bytes))
                        throw std::runtime_error(std::string("GLM: cannot lock mapped weights: ") + std::strerror(errno));
                    locked += t.bytes;
#else
                    throw std::invalid_argument("GLM: --lock-weights requires Linux");
#endif
                }
                for (size_t i = 0; i < t.bytes; i += 4096)
                    checksum = checksum ^ t.data()[i];
            }
        std::cerr << "main weights prefaulted checksum=" << (int)checksum << '\n';
        std::cerr << "HOST_WEIGHTS locked_bytes=" << locked << " ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
        if (direct_upload_enabled() && !registered_weights) {
            auto devices = upload_devices;
            if (devices.empty()) {
                devices.push_back(primary_device);
                if (secondary) devices.push_back(secondary->device);
            }
            registered_weights = std::make_unique<RegisteredWeights>(artifact, primary_device, devices);
        }
    }
    void set_gpu_decode_experts(bool value) {
        if(artifact.is_exl3()&&value)throw std::invalid_argument("EXL3: decode currently requires CPU routed experts");
        if (value && !gpu)
            throw std::invalid_argument("GPU decode experts require GPU prefill allocation");
        gpu_decode_experts = value;
    }
    void reset_decode_stats() {
        native_expert_gpu_hits=native_expert_gpu_misses=native_expert_gpu_copied=0;
        if (gpu) {
            gpu->transferred = 0;
            gpu->groups = 0;
            gpu->stage_ms = 0;
        }
        pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = pool.ms_layer_flow = 0;
        trace = StepTrace{};
        pool.ms_native_local_prepare = 0;
        pool.native_local_queries = 0;
        pool.multi_bytes = 0;
    }
    void set_native_dense_q8(bool enabled,bool cpu_reference=false) {
        if(!artifact.is_exl3())throw std::invalid_argument("EXL3 Q8 requires a native artifact");
        native_dense_q8=enabled;native_cpu_dense=cpu_reference;
        if(enabled && !native_q8_weights.empty()) {
            // A BF16 quality-reference pass may have cached duplicate dense matrices.
            // They are unused once the protected mixed Q8/BF16 resident set is active.
            check(cudaStreamSynchronize(stream));
            for(auto it=weights.begin();it!=weights.end();) {
                const auto name=it->first.substr(0,it->first.rfind(':'));
                if(!native_q8_weights.count(name)){++it;continue;}
                if(native_pinned_weights.erase(it->first))native_pinned_bytes-=it->second.data->bytes;
                resident-=it->second.data->bytes;lru.erase(it->second.order);it=weights.erase(it);
            }
        }
    }
    void set_profile(bool enabled) { profile = enabled; }
    void set_trace_phase(int round, const char *phase) { routing_round = round; routing_phase = phase; }
    void set_routing_trace(const std::string &path) {
        if (path.empty()) return;
        routing_trace.open(path);
        if (!routing_trace) throw std::runtime_error("GLM: cannot open routing trace");
    }
    void set_router_score_trace(const std::string &path) {
        if (path.empty()) return;
        if (step_pipeline) throw std::invalid_argument("GLM: router score trace needs STRATA_GLM_STEP_PIPELINE unset");
        router_score_trace.open(path, std::ios::binary);
        router_bias_trace.open(path + ".bias", std::ios::binary);
        if (!router_score_trace || !router_bias_trace) throw std::runtime_error("GLM: cannot open router score trace");
        host_router_logits = std::make_unique<Pinned>((size_t)cpu::MAXT * m.experts * sizeof(float));
    }
    void set_cancel(std::function<bool()> fn) { cancelled = std::move(fn); }
    void report_gpu() const {
        if (ram_tier)
            std::cerr << "RAM_TIER_STATE resident_MiB=" << ram_tier->resident_bytes / double(MiB) << " reads=" << ram_tier->loads
                      << " evictions=" << ram_tier->evictions << " rebalances=" << ram_rebalances << " misses=" << ram_misses
                      << " miss_MiB=" << ram_miss_bytes / double(MiB) << '\n';
#ifdef __linux__
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
            std::cerr << "HOST_RSS peak_MiB=" << usage.ru_maxrss / 1024.0 << '\n';
#endif
        if(artifact.is_exl3() && native_expert_gpu_limit)
            std::cerr<<"EXL3_GPU_CACHE hits="<<native_expert_gpu_hits<<" misses="<<native_expert_gpu_misses
                     <<" copied_MiB="<<native_expert_gpu_copied/double(MiB)<<" entries="<<native_gpu_experts.size()
                     <<" allocated_MiB="<<native_expert_gpu_bytes/double(MiB)<<'\n';
        std::cerr << "CPU_EXPERT gu_ms=" << pool.ms_multi_gu << " quant_ms=" << pool.ms_multi_q
                  << " down_ms=" << pool.ms_multi_down << " layer_flow_ms=" << pool.ms_layer_flow
                  << " bytes=" << pool.multi_bytes << '\n';
        if (step_trace && trace.steps)
            std::cerr << "STEP_TRACE steps=" << trace.steps << " moe_layers=" << trace.layers
                      << " pipelined=" << (mailbox ? 1 : 0) << " head_ms=" << trace.head_ms
                      << " cpu_ms=" << trace.cpu_ms << " between_ms=" << trace.gap_ms << " tail_ms=" << trace.tail_ms
                      << " tail_sync_ms=" << trace.sync_ms << " tail_plan_ms=" << trace.plan_ms << " tail_copy_ms=" << trace.copy_ms
                      << " enqueue_ms=" << trace.enqueue_ms << " tier_swaps=" << tier_swaps << " split_windows=" << split_windows << " per_step_ms="
                      << (trace.head_ms + trace.cpu_ms + trace.gap_ms + trace.tail_ms) / trace.steps << '\n';
        if (step_trace_slots && trace.steps)
            for (int group = 0; group < 2; ++group) {
                std::cerr << "STEP_TRACE_SLOTS group=" << group << " wait_cpu_us_per_step=";
                for (size_t l = 0; l < m.layers.size(); ++l) {
                    const size_t slot = l + group * mailbox_group_stride();
                    if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
                    std::cerr << l << (m.layers[l].mixer == strata::core::MixerKind::Kda ? "k:" : "m:")
                              << std::llround(trace.slot_wait[slot] * 1000 / trace.steps) << '/'
                              << std::llround(trace.slot_cpu[slot] * 1000 / trace.steps) << ' ';
                }
                std::cerr << '\n';
            }
        if (pool.native_local_queries)
            std::cerr << "NATIVE_NUMA_TIMING prepare_ms=" << pool.ms_native_local_prepare
                      << " queries=" << pool.native_local_queries << '\n';
        if (gpu)
            std::cerr << "GPU peak_allocated_MiB=" << Device::peak() / (double)MiB
                      << " native_expert_bytes=" << gpu->transferred << " expert_groups=" << gpu->groups
                      << " staging_ms=" << gpu->stage_ms << '\n';
        auto report = [](int device, const GpuPrefill *prefill, const PrefillGroupCache &cache) {
            DeviceScope scope(device);
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total));
            const auto &a = Device::accounts.at(device);
            if (prefill && prefill->arena)
                std::cerr << "PREFILL_SCRATCH device=" << device
                          << " peak_MiB=" << prefill->peak_cursor / double(MiB)
                          << " capacity_MiB=" << prefill->arena->bytes / double(MiB) << '\n';
            {
                auto &tags = Device::accounts.at(device);
                std::lock_guard<std::mutex> lock(tags.tag_mutex);
                std::cerr << "GPU_LIVE device=" << device;
                for (const auto &[tag, bytes] : tags.by_tag)
                    if (bytes) std::cerr << ' ' << tag << "_MiB=" << bytes / double(MiB);
                std::cerr << '\n';
            }
            std::cerr << "GPU_DEVICE device=" << device << " peak_allocated_MiB=" << a.peak / double(MiB)
                      << " live_MiB=" << a.live / double(MiB)
                      << " native_expert_bytes=" << (prefill ? prefill->transferred : 0)
                      << " staging_ms=" << (prefill ? prefill->stage_ms : 0)
                      << " cache_hits=" << cache.hits << " cache_hit_bytes=" << cache.hit_bytes
                      << " cuda_used_MiB=" << (total - free) / double(MiB)
                      << " free_MiB=" << free / double(MiB) << '\n';
        };
        report(primary_device, gpu.get(), prefill_cache);
        if (secondary) report(secondary->device, secondary->gpu.get(), secondary->cache);
    }
    void check_exact_cache(const std::vector<float>& logits) {
        if (std::getenv("STRATA_GLM_EXACT_CACHE_CHECK")) {
            if (!exact_cache) throw std::invalid_argument("exact cache check requires exact cache enabled");
            auto cp=checkpoint(); save(cp);
            const std::vector<int> window{int(std::max_element(logits.begin(),logits.end())-logits.begin()),11,9647,0,1,2,3,4};
            std::vector<std::vector<float>> expected;
            auto cache=std::move(exact_cache);
            for (int token : window) expected.push_back(step(token));
            restore(cp); exact_cache=std::move(cache);
            for (size_t i=0;i<window.size();++i)
                if (step(window[i]) != expected[i])
                    throw std::runtime_error("GLM: exact cache changed decode logits");
            restore(cp);
            std::cerr << "EXACT_CACHE_CHECK steps=8 all_logits=bitwise_equal matched_gpu_residency=1\n";
        }
    }
    void freeze_cache() { cache_frozen = true; }
    void reset() {
        ++routing_request; routing_round = -1; routing_phase = "target";
        report_exact_cache(position);
        exact_cache.reset();
        finish_prefetch();
        finish_prefill_repairs();
        if(block_screen)block_screen->begin_sequence();
        device_lookup_ready = false;
        tier_drain();
        ram_last_rebalance = 0;
        if (ram_tier) std::fill(ram_tier->window.begin(), ram_tier->window.end(), 0.f);
        for (int i = 0; i < 2; ++i) finish_copy(i);
        victims.clear();
        for (auto &counts : decode_seen) std::fill(counts.begin(), counts.end(), 0);
        adaptive_copies = adaptive_bytes = 0; adaptive_wait_ms = 0;
        check(cudaStreamSynchronize(stream));
        mixer_graphs.clear();
        if (cached_decode) { cached_decode->graphs.clear(); cached_decode->device_graphs.clear(); }
        if (secondary && secondary->tp) secondary->tp->graphs.clear();
        prefill_cache.reset();
        if (secondary) {
            DeviceScope scope(secondary->device);
            check(cudaStreamSynchronize(secondary->stream));
            if (secondary->decode) { secondary->decode->graphs.clear(); secondary->decode->device_graphs.clear(); }
            secondary->cache.reset();
            secondary->gpu->transferred = secondary->gpu->groups = 0;
            secondary->gpu->stage_ms = 0;
            secondary->gpu->tensor_rows_actual = secondary->gpu->tensor_rows_padded = secondary->gpu->tensor_dequant_values = 0;
        }
        mtp_ready = false;
        mtp_primed = false;
        mtp_position = 0;
        for (auto &cache : decode_resident) cache.clear();
        for (auto &cache : remote_decode_resident) cache.clear();
        for (auto &storage : decode_cache_storage) storage.reset();
        if (restore_prefill_groups) restore_promoted_groups();
        else {
            for (auto &storage : decode_group_storage) storage.clear();
            for (auto &addresses : promoted_addresses) addresses.clear();
        }
        if (secondary) secondary->restore_prefill();
        for (auto &counts : prefill_routes) std::fill(counts.begin(), counts.end(), 0);
        for (auto &ids : prefill_recent_ids) ids.clear();
        // The verify history allocation stays: the tier is sized around it, and a later request might not find
        // the device memory again. Its contents are invalid.
        history_valid = 0;
        if (gpu) {
            if (gpu->copy) check(cudaStreamSynchronize(gpu->copy));
            gpu->transferred = gpu->groups = 0;
            gpu->stage_ms = 0;
            gpu->tensor_rows_actual = gpu->tensor_rows_padded = gpu->tensor_dequant_values = 0;
        }
        if (prefill_width_saved) {
            phase.clear(); scratch.clear(); gpu.reset(); mtp_experts.reset(); mtp_lookup.reset();
            for (auto it = weights.begin(); it != weights.end();) {
                if (it->first.starts_with("blk.45.")) {
                    resident -= it->second.data->bytes;
                    lru.erase(it->second.order);
                    it = weights.erase(it);
                } else ++it;
            }
            states.resize(m.layers.size());
            gpu = std::make_unique<GpuPrefill>(prefill_width_saved, stream, false, capacity, false,
                                              tensor_prefill_experts, tensor_batched_experts);
            prefill_width_saved = 0;
            if (capture_hidden) ensure_draft_state();
        }
        configure_prefill_schedule();
        reset_phase();
        for (auto &s : states)
            for (auto *b : {s.recurrent.get(), s.conv_q.get(), s.conv_k.get(), s.conv_v.get(), s.cache.get(),
                            s.keys.get(), s.gates.get(), s.pooled.get()})
                if (b)
                    check(cudaMemsetAsync(b->p, 0, b->bytes, stream));
        position = 0;
        mtp_position = 0;
        tp_stale = true;
        if (secondary && secondary->tp) {
            DeviceScope scope(secondary->device);
            auto &tp = *secondary->tp;
            for (size_t l = 0; l < m.layers.size(); ++l) {
                if (!tp.recurrent[l]) continue;
                check(cudaMemsetAsync(tp.recurrent[l]->p, 0, tp.recurrent[l]->bytes, secondary->stream));
                for (auto &c : tp.conv) check(cudaMemsetAsync(c[l]->p, 0, c[l]->bytes, secondary->stream));
            }
            check(cudaStreamSynchronize(secondary->stream));
        }
        target_hidden.clear();
        batch_tokens = 1;
        cache_frozen = false;
        cache_hits = cache_entries = pruned_routes = 0;
        merged_expert_layers = 0;
    }
    void report_cache() const {
        if (decode_cache_adapt)
            std::cerr << "DECODE_ADAPT copies=" << adaptive_copies << " bytes=" << adaptive_bytes
                      << " boundary_wait_ms=" << adaptive_wait_ms << '\n';
        bool enabled = false;
        for (const auto &cache : decode_resident) enabled |= !cache.empty();
        for (const auto &c : expert_cache)
            enabled |= bool(c);
        if (enabled || decode_prefill_cache)
            std::cerr << "resident expert entries=" << cache_hits << "/" << cache_entries << '\n';
        if (pruned_routes)
            std::cerr << "ROUTE_PRUNE pruned=" << pruned_routes << " routes=" << cache_entries << '\n';
        if (merged_expert_layers)
            std::cerr << "EXPERT_GROUP_MERGE layers=" << merged_expert_layers << '\n';
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

// Teacher forcing uses the same CPU expert backend as decode, including across
// prompt boundaries. Full baseline logits are streamed, never held corpus-wide.
using strata::core::lock_small_runtime_mappings;

static void evaluate_corpus(Decoder& decoder,const std::string& corpus,const std::string& save_path,const std::string& reference_path,
                            int prefill_chunk = 0) {
    if(!save_path.empty()&&!reference_path.empty())throw std::invalid_argument("choose eval save or reference, not both");
    if(prefill_chunk && decoder.has_decode_cache())throw std::invalid_argument("prefill evaluation requires the decode expert cache disabled");
    std::ifstream input(corpus);if(!input)throw std::runtime_error("cannot open evaluation corpus");
    std::ofstream saved;std::ifstream reference;
    const uint32_t header[2]={0x31455647,uint32_t(decoder.vocabulary())};
    if(!save_path.empty()){if(std::filesystem::exists(save_path))throw std::runtime_error("refusing to overwrite baseline logits");saved.open(save_path,std::ios::binary);saved.write(reinterpret_cast<const char*>(header),sizeof header);if(!saved)throw std::runtime_error("cannot save evaluation logits");}
    if(!reference_path.empty()){reference.open(reference_path,std::ios::binary);uint32_t h[2];if(!reference.read(reinterpret_cast<char*>(h),sizeof h)||std::memcmp(h,header,sizeof h))throw std::runtime_error("invalid evaluation baseline");}
    const int vocab=decoder.vocabulary();std::vector<float> base(vocab);std::string line;uint64_t count=0,different_logits=0;double nll=0,kl=0;int sequence=0,top1=0;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        const auto ids=token_ids(line);if(ids.size()<2)throw std::runtime_error("evaluation sequence needs at least two tokens");
        for(int id:ids)if(id<0||id>=vocab)throw std::runtime_error("invalid evaluation token");
        const uint32_t length=ids.size();
        if(saved.is_open()){saved.write(reinterpret_cast<const char*>(&length),4);saved.write(reinterpret_cast<const char*>(ids.data()),ids.size()*4);}
        if(reference.is_open()){uint32_t len=0;reference.read(reinterpret_cast<char*>(&len),4);if(len!=length)throw std::runtime_error("evaluation baseline sequence mismatch");std::vector<int> expected(len);if(!reference.read(reinterpret_cast<char*>(expected.data()),len*4)||expected!=ids)throw std::runtime_error("evaluation baseline tokens differ");}
        decoder.reset();lock_small_runtime_mappings();
        // A configured GPU tier takes part in the evaluation (its selection has no prompt routes here, so it
        // starts from the calibration prior): modes whose output depends on residency are measured as they run.
        if(decoder.has_decode_cache())decoder.prepare_decode_cache();
        // STRATA_GLM_EVAL_WIDTH: tokens per decode call (default 4); 1 scores one-token decode paths.
        static const size_t eval_width = [] { const char *v = std::getenv("STRATA_GLM_EVAL_WIDTH"); return v ? std::stoul(v) : 4ul; }();
        if (eval_width < 1 || eval_width > (size_t)cpu::MAXT) throw std::invalid_argument("STRATA_GLM_EVAL_WIDTH must be 1..8");
        const size_t width = prefill_chunk ? prefill_chunk : eval_width;
        for(size_t start=0;start+1<ids.size();start+=width) {
            const size_t end=std::min(start+width,ids.size()-1);auto logits=decoder.batch(std::vector<int>(ids.begin()+start,ids.begin()+end),prefill_chunk != 0,true);
            if(start==0)lock_small_runtime_mappings();
            for(size_t t=0;t<end-start;++t) {
                const auto* row=logits.data()+t*vocab;const double maximum=*std::max_element(row,row+vocab);double sum=0;
                for(int k=0;k<vocab;++k){if(!std::isfinite(row[k]))throw std::runtime_error("nonfinite evaluation logits");sum+=std::exp(double(row[k])-maximum);}const double logz=maximum+std::log(sum);
                nll+=logz-row[ids[start+t+1]];++count;
                if(saved.is_open()){saved.write(reinterpret_cast<const char*>(row),vocab*4);if(!saved)throw std::runtime_error("evaluation logits write failed");}
                if(reference.is_open()) {
                    if(!reference.read(reinterpret_cast<char*>(base.data()),vocab*4))throw std::runtime_error("truncated evaluation logits");
                    const double bm=*std::max_element(base.begin(),base.end());double bs=0;for(float v:base){if(!std::isfinite(v))throw std::runtime_error("nonfinite baseline logits");bs+=std::exp(double(v)-bm);}const double bz=bm+std::log(bs);
                    for(int k=0;k<vocab;++k){kl+=std::exp(double(base[k])-bz)*(double(base[k])-bz-row[k]+logz);different_logits+=std::bit_cast<uint32_t>(base[k])!=std::bit_cast<uint32_t>(row[k]);}
                    top1+=(std::max_element(row,row+vocab)-row)==(std::max_element(base.begin(),base.end())-base.begin());
                }
            }
        }
        ++sequence;std::cout<<"{\"sequence\":"<<sequence<<",\"tokens\":"<<count<<",\"nll\":"<<std::setprecision(12)<<nll/count<<",\"perplexity\":"<<std::exp(nll/count);
        if(reference.is_open())std::cout<<",\"kl\":"<<kl/count<<",\"top1_agreement\":"<<double(top1)/count<<",\"different_logits\":"<<different_logits;
        std::cout<<"}\n"<<std::flush;
    }
    if(!count)throw std::runtime_error("empty evaluation corpus");
    if(reference.is_open()&&reference.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("trailing evaluation baseline records");
    if(saved.is_open()){saved.close();if(!saved)throw std::runtime_error("evaluation logits close failed");}
    decoder.flush_expert_observer();decoder.report_gpu();
}

// Existing Python API engine protocol. Every request starts with empty model state;
// mmap handles, dense weights and optional resident experts survive between requests.
static int prefill_width(const std::string &value) {
    if (value == "auto")
        return 0;
    size_t used = 0;
    int n = std::stoi(value, &used);
    if (used != value.size() || n < 1 || n > 16384)
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
                                   int depth, const std::function<void(int)> &emit, bool report = true) {
    const int minimum_match = [] {
        const char *value = std::getenv("STRATA_GLM_LOOKUP_MIN_MATCH");
        if (!value) return cached_lookup_enabled() ? 6 : 3;
        size_t used = 0; const int n = std::stoi(value, &used);
        if (value[used] || n < 3 || n > 32) throw std::invalid_argument("GLM: lookup minimum match must be 3..32");
        return n;
    }();
    strata::spec::SuffixDrafter lookup(minimum_match, 32, prompt.size() + max_new);
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
    if (report) std::cerr << "SPECULATIVE source=lookup generated=" << stats.generated << " rounds=" << stats.rounds
              << " proposed=" << stats.proposed << " accepted=" << stats.accepted
              << " replayed=" << stats.replayed << " ms=" << stats.milliseconds
              << " tok_s=" << std::max(0, stats.generated - 1) * 1000 / stats.milliseconds << '\n';
    return stats;
}
static LookupStats generate_mtp(Decoder &decoder, std::vector<float> &logits, int max_new,
                               const std::vector<int> &stops, int depth,
                               const std::function<void(int)> &emit,
                               const std::vector<int> *prompt = nullptr) {
    auto checkpoint = decoder.checkpoint();
    // The first proposal already advanced the draft block with exactly the anchor's resync inputs (the
    // target hidden of the last verified position), so the resync can start after the anchor.
    static const bool skip_anchor = std::getenv("STRATA_GLM_MTP_SKIP_ANCHOR") &&
                                    std::string(std::getenv("STRATA_GLM_MTP_SKIP_ANCHOR")) == "1";
    auto anchor_state = decoder.mtp_checkpoint();
    int lookup_depth = mtp_lookup_depth();
    static const bool lookup_chain = [] {
        const char *value = std::getenv("STRATA_GLM_MTP_LOOKUP_CHAIN");
        if (!value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("GLM: MTP lookup chain must be 0 or 1");
    }();
    static const float continuation_margin = [] {
        const char *value = std::getenv("STRATA_GLM_MTP_CONTINUATION_MARGIN");
        if (!value) return 0.f;
        size_t used = 0; const float margin = std::stof(value, &used);
        if (value[used] || !(margin >= 0.f && margin <= 100.f))
            throw std::invalid_argument("GLM: MTP continuation margin must be 0..100");
        return margin;
    }();
    decoder.enable_verify_history(std::max(depth, lookup_depth));
    std::unique_ptr<strata::spec::SuffixDrafter> lookup;
    if (lookup_depth) {
        if (!prompt) throw std::invalid_argument("GLM: MTP lookup requires the request's prompt");
        const int minimum_match = [] {
            const char *value = std::getenv("STRATA_GLM_MTP_LOOKUP_MIN_MATCH");
            if (!value) return 6;
            size_t used = 0; const int n = std::stoi(value, &used);
            if (value[used] || n < 3 || n > 32)
                throw std::invalid_argument("GLM: MTP lookup minimum match must be 3..32");
            return n;
        }();
        lookup = std::make_unique<strata::spec::SuffixDrafter>(minimum_match, 32, prompt->size() + max_new);
        lookup->append(prompt->data(), prompt->size());
    }
    // A rejection after accepting n drafts restores n+1 target positions.
    // Limiting proposals to the retained slots keeps every rejection prefix
    // restorable instead of paying for another CPU expert pass.
    const int requested_depth = depth;
    if (decoder.verify_history_slots()) {
        depth = std::min(depth, decoder.verify_history_slots());
        lookup_depth = std::min(lookup_depth, decoder.verify_history_slots());
    } else lookup_depth = 0;
    std::cerr << "MTP_DEPTH requested=" << requested_depth << " effective=" << depth << '\n';
    LookupStats stats;
    double draft_ms = 0, verify_ms = 0, resync_ms = 0;
    int lookup_rounds = 0, lookup_proposed = 0, lookup_accepted = 0;
    std::array<int, 7> mtp_position_proposed{}, mtp_position_accepted{};
    int margin_skipped = 0;
    struct MatchStats { int rounds=0, proposed=0, accepted=0; };
    std::array<MatchStats, 4> match_stats{};
    auto elapsed_ms = [](auto begin) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    };
    const auto start = std::chrono::steady_clock::now();
    auto output = [&](int token) {
        emit(token); ++stats.generated;
        if (lookup) lookup->append(token);
        stats.stopped = std::find(stops.begin(), stops.end(), token) != stops.end();
    };
    while (stats.generated < max_new && !stats.stopped) {
        const int anchor = greedy(logits);
        output(anchor);
        if (stats.stopped || stats.generated == max_new) break;
        int32_t lookup_tokens[7];
        const int from_lookup = lookup_depth ? lookup->propose(
            std::min(lookup_depth, max_new - stats.generated - 1), lookup_tokens) : 0;
        const bool use_lookup = from_lookup >= 2;
        int selected_match = use_lookup ? lookup->last_match() : 0;
        int k = use_lookup ? from_lookup : std::min(depth, max_new - stats.generated - 1);
        if (!k) {
            const int pos = decoder.token_position();
            logits = decoder.step(anchor);
            decoder.sync_mtp({anchor}, pos);
            continue;
        }
        decoder.save(checkpoint);
        const int pos = decoder.token_position();
        std::vector<int> window{anchor};
        int mtp_proposals = 0;
        int lookup_proposals = use_lookup ? k : 0;
        decoder.set_trace_phase(stats.rounds, "draft");
        const auto draft_start = std::chrono::steady_clock::now();
        if (use_lookup) window.insert(window.end(), lookup_tokens, lookup_tokens + k);
        else for (int i = 0; i < k; ++i) {
            const auto draft_logits = decoder.mtp_propose(window.back(), i == 0);
            const int token = greedy(draft_logits);
            if (i && continuation_margin > 0.f) {
                float runner_up = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < draft_logits.size(); ++j)
                    if ((int)j != token) runner_up = std::max(runner_up, draft_logits[j]);
                // Only the speculative window shrinks. The unused draft state
                // is restored below before resynchronizing verified tokens.
                if (draft_logits[token] - runner_up < continuation_margin) { ++margin_skipped; break; }
            }
            window.push_back(token);
            ++mtp_position_proposed[i]; ++mtp_proposals;
            if (i == 0 && skip_anchor) decoder.save(anchor_state);
        }
        if (!use_lookup) k = mtp_proposals;
        if (!use_lookup && lookup_chain && lookup_depth > k) {
            const int extra = lookup->propose_after(window.data() + 1, k,
                std::min(lookup_depth, max_new - stats.generated - 1) - k, lookup_tokens);
            if (extra >= 2) {
                window.insert(window.end(), lookup_tokens, lookup_tokens + extra);
                lookup_proposals = extra; k += extra; selected_match = lookup->last_match();
            }
        }
        draft_ms += elapsed_ms(draft_start);
        decoder.set_trace_phase(stats.rounds, "verify");
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
        for (int i = 0; i < std::min(accepted, mtp_proposals); ++i) ++mtp_position_accepted[i];
        if (lookup_proposals) {
            ++lookup_rounds; lookup_proposed += lookup_proposals;
            lookup_accepted += std::max(0, accepted - mtp_proposals);
            auto &bucket = match_stats[std::min(3, selected_match / 8)];
            ++bucket.rounds; bucket.proposed += lookup_proposals;
            bucket.accepted += std::max(0, accepted - mtp_proposals);
        }
        size_t resync_from = 0;
        auto restore_draft = [&] {
            if (!use_lookup && skip_anchor && !window.empty()) { decoder.restore_mtp(anchor_state); resync_from = 1; }
            else decoder.restore_mtp(checkpoint);
        };
        if (accepted != k || stats.stopped) {
            window.resize(accepted + 1 - int(stats.stopped));
            if (decoder.restore_verified(window.size())) {
                restore_draft();
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
            restore_draft();
            logits.assign(all.end() - decoder.vocabulary(), all.end());
        }
        decoder.set_trace_phase(stats.rounds - 1, "resync");
        const auto resync_start = std::chrono::steady_clock::now();
        decoder.sync_mtp(std::vector<int>(window.begin() + resync_from, window.end()), pos + (int)resync_from);
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
    for (size_t i = 0; i < mtp_position_proposed.size(); ++i)
        if (mtp_position_proposed[i]) std::cerr << "MTP_POSITION index=" << i
            << " proposed=" << mtp_position_proposed[i] << " accepted=" << mtp_position_accepted[i] << '\n';
    if (continuation_margin > 0.f) std::cerr << "MTP_MARGIN minimum=" << continuation_margin
                                          << " skipped=" << margin_skipped << '\n';
    if (lookup) std::cerr << "MTP_LOOKUP rounds=" << lookup_rounds << " proposed=" << lookup_proposed
                          << " accepted=" << lookup_accepted << " max_depth=" << lookup_depth << '\n';
    for (size_t i = 0; lookup && i < match_stats.size(); ++i)
        if (match_stats[i].rounds) std::cerr << "MTP_LOOKUP_MATCH bucket=" << i
            << " rounds=" << match_stats[i].rounds << " proposed=" << match_stats[i].proposed
            << " accepted=" << match_stats[i].accepted << '\n';
    return stats;
}
static size_t gpu_budget_bytes(const std::string &value) {
    size_t used = 0;
    const auto mib = std::stoull(value, &used);
    if (used != value.size() || mib <= 1024 || mib > SIZE_MAX / MiB)
        throw std::invalid_argument("GPU budget must exceed 1024 MiB and fit size_t");
    return mib * MiB;
}
static void preflight_gpu_budget(size_t budget) {
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    // CUDA's usable capacity can be slightly below the card's advertised GiB.
    const size_t nominal_capacity = (total + 1024 * MiB - 1) / (1024 * MiB) * (1024 * MiB);
    if (budget <= std::max(1024 * MiB, Device::budget_reserve()) || budget > nominal_capacity)
        throw std::invalid_argument("GPU budget exceeds the selected device capacity");
    Device::limit() = std::min(budget, total) - Device::budget_reserve();
}
static std::vector<int> configure_devices(const std::string &text, size_t budget) {
    std::vector<int> devices;
    std::istringstream input(text);
    std::string item;
    int count = 0;
    check(cudaGetDeviceCount(&count));
    while (std::getline(input, item, ',')) {
        size_t used = 0;
        const int device = std::stoi(item, &used);
        if (used != item.size() || device < 0 || device >= count ||
            std::find(devices.begin(), devices.end(), device) != devices.end())
            throw std::invalid_argument("GLM: invalid or duplicate GPU device");
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device));
#ifdef STRATA_USE_HIP
        const auto arch_problem = strata::core::gpu_arch_problem(device);
        if (!arch_problem.empty()) throw std::invalid_argument("GLM: " + arch_problem);
#else
        if (props.major < 7) throw std::invalid_argument("GLM: GPU needs compute capability 7.0");
#endif
        devices.push_back(device);
    }
    if (devices.empty() || devices.size() > 2 || text.back() == ',')
        throw std::invalid_argument("GLM: select one or two GPU devices");
    for (int device : devices) {
        check(cudaSetDevice(device));
        preflight_gpu_budget(budget);
    }
    check(cudaSetDevice(devices.front()));
    return devices;
}
static std::pair<size_t, bool> cache_setting(const std::string &value) {
    if (value == "auto") return {0, true};
    size_t used = 0;
    const auto mib = std::stoull(value, &used);
    if (used != value.size() || mib > SIZE_MAX / MiB)
        throw std::invalid_argument("GLM: invalid prefill cache budget");
    return {mib * MiB, false};
}
#include "glm_batch_serve.inc"

static int serve(int argc, char **argv) {
    if (argc < 6)
        throw std::invalid_argument(
            "usage: strata-glm-decode --serve <shard> <context> <dense-MiB> <threads> "
            "[expert-MiB=0] [prefill-batch=8] [eos-id=-1] [gpu-budget-MiB=12288] [lookup-depth=0] "
            "[gpu-decode-experts=0]");
    int parallel = 0;
    // Named flags follow the legacy positional serve arguments; existing invocations remain valid.
    if (argc >= 31 && std::string(argv[29]) == "--batch") {
        parallel = std::stoi(argv[30]);
        if (argc != 31 || parallel < 1 || parallel > cpu::MAXT)
            throw std::invalid_argument("GLM: --batch must be 1..8");
        argc = 29;
    }
    const int ctx = std::stoi(argv[3]), dense = std::stoi(argv[4]), threads = std::stoi(argv[5]);
    const int experts = argc > 6 ? std::stoi(argv[6]) : 0;
    int width = argc > 7 ? prefill_width(argv[7]) : 8;
    const size_t gpu_budget = argc > 9 ? gpu_budget_bytes(argv[9]) : 12288 * MiB;
    const auto stops = argc > 8 ? token_ids(argv[8]) : std::vector<int>{};
    const int lookup_depth = argc > 10 ? std::stoi(argv[10]) : 0;
    if(std::filesystem::is_directory(argv[2])&&lookup_depth)throw std::invalid_argument("EXL3: lookup speculation is not qualified");
    const bool stream_decode = argc > 11 && std::stoi(argv[11]) != 0;
    const bool use_mtp = argc > 12 && std::string(argv[12]) == "mtp";
    const int draft_depth = argc > 13 ? std::stoi(argv[13]) : 3;
    if(argc>25) {
        std::string pages=argv[25];if(pages!="4k"&&pages!="huge")throw std::invalid_argument("weight_pages must be 4k or huge");
        strata::cpu::exl3::huge_pages=pages=="huge";
    }
    const bool pin_cpu = argc > 14 && (std::string(argv[14]) == "auto" || std::string(argv[14]) == "numa");
    const size_t cpu_prepack_mib = argc > 15 ? std::stoull(argv[15]) : 0;
    const auto devices = configure_devices(argc > 16 ? argv[16] : "0", gpu_budget);
    const auto [prefill_cache_bytes, prefill_cache_auto] = cache_setting(argc > 17 ? argv[17] : "0");
    if (devices.size() > 1 || prefill_cache_bytes || prefill_cache_auto) {
        if (width != 0 && width <= cpu::MAXT) width = 4096;
    }
    if (cpu_prepack_mib > 131072 || (cpu_prepack_mib && (stream_decode || experts))) throw std::invalid_argument("invalid serve CPU packing settings");
    if (draft_depth < 1 || draft_depth > 7 || (use_mtp && (lookup_depth || stream_decode || experts)))
        throw std::invalid_argument("invalid serve MTP settings");
    if (ctx < 1 || dense < 64 || threads < 1 || experts < 0 || lookup_depth < 0 || lookup_depth > 7)
        throw std::invalid_argument("invalid serve settings");
    if (direct_upload_enabled() && (!(argc > 18 && std::string(argv[18]) == "1") || cpu_prepack_mib))
        throw std::invalid_argument("GLM: direct weight upload requires locked original serve weights");
    if (width == 0 || width > cpu::MAXT || stream_decode || use_mtp)
        preflight_gpu_budget(gpu_budget);
    if ((width == 0 || width > cpu::MAXT || stream_decode || use_mtp) && experts)
        throw std::invalid_argument("GPU prefill/decode requires expert-cache-MiB=0");
    Decoder decoder(argv[2], ctx, (size_t)dense * 1024 * 1024, threads, (size_t)experts * 1024 * 1024, pin_cpu,
                    argc>26?argv[26]:"",argc>27?argv[27]:"",argc>28?argv[28]:"auto");
    if (use_mtp) decoder.enable_mtp_capture();
    if (parallel) {
        if (devices.size() != 1 || lookup_depth || stream_decode || experts || cpu_prepack_mib ||
            std::filesystem::is_directory(argv[2]))
            throw std::invalid_argument("GLM: batching requires one GPU, GGUF, and CPU routed experts");
        decoder.enable_sequences(parallel);
    }
    if (width == 0 || width > cpu::MAXT || stream_decode || use_mtp || parallel)
        width = decoder.enable_gpu(width, gpu_budget, lookup_depth > 0);
    if (direct_upload_enabled()) decoder.warm_weights(true, devices);
    const bool tensor_batches = argc > 24 && std::string(argv[24]) == "f16-batched";
    const bool tensor_experts = tensor_batches || (argc > 24 && std::string(argv[24]) == "f16");
    if (argc > 24 && std::string(argv[24]) != "f16" && std::string(argv[24]) != "f16-batched" && std::string(argv[24]) != "mmq")
        throw std::invalid_argument("GLM: prefill experts must be mmq, f16 or f16-batched");
    // A decode cache keeps the draft experts on the CPU: no device reserve for them, only the draft state,
    // which must exist before the cache allowance is measured.
    const bool mtp_device_experts = use_mtp && !(argc > 20 && std::string(argv[20]) != "0");
    if (use_mtp) decoder.enable_mtp_capture();
    decoder.configure_prefill(devices, gpu_budget, prefill_cache_bytes, prefill_cache_auto, mtp_device_experts, tensor_experts, tensor_batches);
    decoder.set_gpu_decode_experts(stream_decode);
    const bool decode_prefill_cache = argc > 19 && std::string(argv[19]) == "1";
    if (decode_prefill_cache && (cpu_prepack_mib || experts))
        throw std::invalid_argument("GLM: decode prefill cache cannot be combined with CPU prepacking or legacy expert cache");
    decoder.set_decode_prefill_cache(decode_prefill_cache);
    const std::string decode_cache_setting = argc > 20 ? argv[20] : "0";
    const bool decode_cache_extend = decode_cache_setting == "extend";
    const auto [decode_cache_bytes, decode_cache_auto] = cache_setting(decode_cache_extend ? "auto" : decode_cache_setting);
    if (decode_cache_bytes > 12288 * MiB || ((decode_cache_bytes || decode_cache_auto) &&
        (stream_decode || lookup_depth || experts || cpu_prepack_mib)))
        throw std::invalid_argument("GLM: serve decode cache requires single-token unpacked CPU target experts");
    // With a decode cache the draft block's experts stay on the CPU; the cache owns the spare device memory.
    if (use_mtp && (decode_cache_bytes || decode_cache_auto)) decoder.set_mtp_cpu_experts(true);
    decoder.set_decode_cache_budget(decode_cache_bytes);
    decoder.set_decode_cache_auto(decode_cache_auto, decode_cache_extend);
    decoder.set_decode_graphs(argc > 21 && std::string(argv[21]) == "1");
    decoder.set_decode_cache_window(argc > 22 ? std::stoi(argv[22]) : 256);
    decoder.set_decode_cache_adapt(argc > 23 && std::string(argv[23]) == "1");
    // Cross-node expert TP for serving (configs pass these through "env"); same meaning as the CLI flags.
    if (const char *address = std::getenv("STRATA_GLM_REMOTE_TP"); address && *address) {
        const char *share = std::getenv("STRATA_GLM_REMOTE_TP_SHARE"), *reply = std::getenv("STRATA_GLM_REMOTE_TP_REPLY");
        const double fraction = share && *share ? std::stod(share) : 0.5;
        if (!(fraction > 0 && fraction < 1)) throw std::invalid_argument("STRATA_GLM_REMOTE_TP_SHARE must be between 0 and 1");
        if (reply && *reply && std::string(reply) != "f16" && std::string(reply) != "f32")
            throw std::invalid_argument("STRATA_GLM_REMOTE_TP_REPLY must be f16 or f32");
        decoder.enable_remote_tp(address, fraction, !reply || std::string(reply) != "f32", true);
    }
    decoder.prepare_numa_weights();
    if (argc > 18 && std::string(argv[18]) == "1" && !direct_upload_enabled()) decoder.warm_weights(true);
    if (use_mtp) decoder.enable_mtp_capture();
    if (parallel) return serve_sequences(decoder, parallel, ctx, width, stops, use_mtp, draft_depth);
    std::cout << "INFO engine=glm-experimental sampling=greedy\nREADY " << ctx
              << " stop prefill-includes-reset\n" << std::flush;
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
            if (request.line.starts_with("TUNE ") && std::getenv("STRATA_GLM_ALLOW_TUNE")) {
                std::istringstream settings(request.line.substr(5));
                int device, reduction, pools, split, tasks, profiling;
                if (!(settings >> device >> reduction >> pools >> split >> tasks >> profiling) ||
                    device < 0 || device > 1 || reduction < 0 || reduction > 1 || pools < 0 || pools > 1 ||
                    split < 0 || split > 1 || profiling < 0 || profiling > 1 || tasks < 1 || tasks > 16)
                    throw std::invalid_argument("TUNE expects device reduction active-pools split-HC tasks profile");
                std::vector<std::string> options;
                for (std::string option; settings >> option;) options.push_back(option);
                if (options.size() > 8)
                    throw std::invalid_argument("TUNE has too many optional settings");
                for (size_t i = 0; i < options.size(); ++i) {
                    const bool valid = i == 1 ? options[i] == "32" || options[i] == "64" || options[i] == "128"
                        : i == 7 ? options[i] == "1" || options[i] == "4" || options[i] == "8"
                                 : options[i] == "0" || options[i] == "1";
                    if (!valid) throw std::invalid_argument("TUNE optional settings: graphs KDA-columns stable buckets preallocate defer-copy restore-prefill KDA-row-parts");
                }
                auto enabled = [&](size_t i) { return options.size() > i && options[i] == "1"; };
                if (options.size() > 7 && options[7] != "1" && options[1] != "128")
                    throw std::invalid_argument("TUNE parallel KDA requires 128 columns");
                decoder.reset();
                std::cerr << "TUNE_RESET_COMPLETE\n";
                decoder.set_bench_mode(device, reduction, pools, split, tasks, profiling,
                                       enabled(0), options.size() > 1 ? std::stoi(options[1]) : 128,
                                       enabled(2), enabled(3), enabled(4), enabled(5), enabled(6),
                                       options.size() > 7 ? std::stoi(options[7]) : 1);
                std::cout << "TUNED\n" << std::flush;
                continue;
            }
            std::istringstream command(request.line);
            std::string op, ids, extra;
            int count;
            if (!(command >> op >> count >> ids) || op != "GEN" || command >> extra)
                throw std::invalid_argument(
                    "expected GEN <max_new> <comma-separated IDs>; only greedy text is supported");
            const auto tokens = token_ids(ids);
            if (tokens.empty() || count < 1 || tokens.size() + (size_t)count > (size_t)ctx)
                throw std::invalid_argument("request exceeds context or has invalid token counts");
            auto start = Clock::now();
            decoder.reset();
            std::cerr << "RESPONSE_PREPARE kind=request_reset ms=" << millis(start, Clock::now()) << '\n';
            decoder.set_cancel([&request] { return bool(*request.cancel); });
            std::vector<float> logits;
            const auto prompt_start = Clock::now();
            for (size_t t = 0; t < tokens.size(); t += width) {
                if (*request.cancel)
                    break;
                const size_t end = std::min(tokens.size(), t + width);
                logits = decoder.batch(std::vector<int>(tokens.begin() + t, tokens.begin() + end));
                const double prompt_ms = millis(prompt_start, Clock::now());
                std::cout << "PP " << end << ' ' << tokens.size() << ' ' << prompt_ms << ' '
                          << (prompt_ms > 0 ? end * 1000.0 / prompt_ms : 0.0) << '\n' << std::flush;
            }
            if (use_mtp && !*request.cancel) {
                const auto mtp_start = Clock::now();
                decoder.prepare_mtp(tokens);
                // Allocate verification snapshots before the auto expert cache
                // consumes its budget; keep the existing physical headroom guard.
                decoder.prepare_mtp_history(draft_depth);
                std::cerr << "RESPONSE_PREPARE kind=mtp ms=" << millis(mtp_start, Clock::now()) << '\n';
            }
            if (!*request.cancel) {
                decoder.probe_decode_cache(logits, stops);
                const auto admission_start = Clock::now();
                decoder.prepare_decode_cache();
                std::cerr << "RESPONSE_PREPARE kind=decode_cache ms=" << millis(admission_start, Clock::now()) << '\n';
            }
            decoder.freeze_cache();
            if (!*request.cancel) decoder.prepare_cpu(cpu_prepack_mib * MiB);
            if (!*request.cancel) lock_small_runtime_mappings();
            const auto prefilled = Clock::now();
            decoder.reset_decode_stats();
            int generated = 0;
            std::string finish = "length";
            if (use_mtp && !*request.cancel) {
                auto result = generate_mtp(decoder, logits, count, stops, draft_depth, [&](int token) {
                    if (*request.cancel) throw std::runtime_error("cancelled");
                    std::cout << "T " << token << '\n' << std::flush;
                }, &tokens);
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
            std::cerr << "GLM request failed: " << e.what() << '\n';
            try { decoder.reset(); }
            catch (const std::exception &reset_error) {
                std::cout << "ERR " << e.what() << "; reset failed: " << reset_error.what() << '\n' << std::flush;
                // A broken CUDA context cannot serve another request. Waiting
                // for the stdin reader here would hide the error indefinitely.
                std::_Exit(1);
            }
            if (request.cancel->load())
                std::cout << "DONE 0 0 0 0 stop 0 0 0\n" << std::flush;
            else
                std::cout << "ERR " << e.what() << '\n' << std::flush;
        }
        decoder.set_cancel({});
        try { decoder.flush_expert_observer(); decoder.report_gpu(); decoder.report_cache(); }
        catch (const std::exception &e) {
            std::cout << "ERR GPU reporting failed: " << e.what() << '\n' << std::flush;
            std::_Exit(1);
        }
        std::lock_guard lock(mutex);
        active.reset();
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: strata-glm-decode <shard.gguf> <comma-separated token IDs> [steps=1] "
                     "[dense-cache-MiB=4096] [threads=6] [--prefill-batch=1..8192|auto] "
                     "[--gpu-budget-mib=12288] [--check-prefill] [--check-native-dense] "
                     "[--gpu-devices=0|0,1] [--prefill-expert-cache-mib=auto|N] [--lock-weights] [--check-gpu-split] "
                     "[--decode-prefill-cache] [--decode-bench=3] [--decode-mode-sweep[=mtp]] [--check-verify-graphs] [--profile-decode] [--profile-prefill] "
                     "[--expert-pack=sidecar.gguf] [--expert-pack-profile=retain.json] [--cpu-expert-backend=auto|native|packed-dot|packed-lut] "
                     "[--eval-corpus=sequences.ids] [--eval-prefill] [--eval-save-logits=path | --eval-reference=path] "
                     "[--expert-cache-mib=N] [--remote-tp=HOST:PORT] [--remote-tp-share=0.5] [--remote-tp-reply=f16|f32] [--remote-tp-weights=send|dummy] [--routing-trace=path] [--check-replay] [--dump-logits=path] [--stop-ids=IDs]\n";
        return 2;
    }
    try {
        capture_allocation();
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
        bool check_prefill = false, check_native_dense_q8 = false;
        int expert_mib = 0, decode_cache_mib = 0, decode_cache_slots = 0, check_sequence_count = 0;
        int prefill_batch = 8;
        size_t gpu_budget = 12288 * MiB;
        bool force_gpu = false, warm_weights = false, lock_weights = false, profile = false;
        bool profile_prefill = false, profile_decode = false, decode_graphs = false, check_decode_graphs = false, check_verify_graphs=false, decode_cache_adapt = false;
        int decode_cache_window = 256;
        int repetitions = 1, context = 0, reference_batch = 1, lookup_depth = 0;
        bool check_verify = false, stream_decode = false, pin_cpu = false, use_mtp = false;
        bool mtp_cpu_experts = false, decode_prefill_cache = false, decode_cache_auto = false, decode_cache_extend = false;
        int draft_depth = 3, decode_repetitions = 1;
        bool decode_mode_sweep = false, decode_mtp_sweep = false;
        size_t cpu_prepack_mib = 0, prefill_cache_bytes = 0;
        bool prefill_cache_auto = false, check_gpu_split = false, tensor_experts = false, tensor_batches = false;
        std::string gpu_devices_text = "0";
        std::string expert_pack_path,expert_pack_profile,cpu_expert_backend="auto";
        std::string eval_corpus,eval_save,eval_reference;
        std::string block_screen_profile,block_capture;
        std::vector<int> block_screen_sweep;
        std::vector<int> block_screen_layers;
        bool eval_prefill = false;
        std::string remote_tp_address;
        double remote_tp_share = 0.5;
        std::string remote_tp_reply = "f16";
        std::string remote_tp_weights = "send";
        std::string dump, routing_trace_path;
        std::vector<int> stops;
        for (int i = 6; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--check-native-dense-q8" || flag == "--check-native-dense") {check_native_dense_q8=true;force_gpu=true;}
            else if (flag == "--prefill-experts=f16-batched") { tensor_experts = tensor_batches = true; force_gpu = true; }
            else if (flag == "--prefill-experts=f16") { tensor_experts = true; tensor_batches = false; force_gpu = true; }
            else if (flag == "--prefill-experts=mmq") tensor_experts = tensor_batches = false;
            else if (flag == "--decode-cache-adapt") decode_cache_adapt = true;
            else if (flag == "--check-decode-graphs") { check_decode_graphs = true; force_gpu = true; }
            else if (flag.starts_with("--decode-cache-window=")) decode_cache_window = std::stoi(flag.substr(22));
            else if (flag == "--decode-graphs") { decode_graphs = true; force_gpu = true; }
            else if (flag.starts_with("--remote-tp=")) remote_tp_address = flag.substr(12);
            else if (flag.starts_with("--remote-tp-share=")) remote_tp_share = std::stod(flag.substr(18));
            else if (flag == "--remote-tp-reply=f32" || flag == "--remote-tp-reply=f16") remote_tp_reply = flag.substr(18);
            else if (flag == "--remote-tp-weights=send" || flag == "--remote-tp-weights=dummy") remote_tp_weights = flag.substr(20);
            else if (flag == "--decode-prefill-cache") { decode_prefill_cache = true; force_gpu = true; }
            else if (flag.starts_with("--gpu-devices=")) { gpu_devices_text = flag.substr(14); force_gpu = true; }
            else if (flag.starts_with("--prefill-expert-cache-mib=")) {
                std::tie(prefill_cache_bytes, prefill_cache_auto) = cache_setting(flag.substr(std::string("--prefill-expert-cache-mib=").size()));
                force_gpu = true;
            } else if (flag == "--check-gpu-split") { check_gpu_split = true; force_gpu = true; }
            else if (flag == "--check-replay")
                replay = true;
            else if (flag == "--check-prefill")
                check_prefill = true;
            else if (flag.starts_with("--expert-cache-mib="))
                expert_mib = std::stoi(flag.substr(19));
            else if (flag.starts_with("--decode-cache-mib=")) {
                if (flag.substr(19) == "auto" || flag.substr(19) == "extend") {
                    decode_cache_auto = true; decode_cache_extend = flag.substr(19) == "extend";
                }
                else decode_cache_mib = std::stoi(flag.substr(19));
                force_gpu = true;
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
            else if (flag == "--decode-mode-sweep") decode_mode_sweep = true;
            else if (flag == "--decode-mode-sweep=mtp") decode_mode_sweep = decode_mtp_sweep = true;
            else if (flag.starts_with("--decode-bench=")) decode_repetitions = std::stoi(flag.substr(15));
            else if (flag == "--speculative=lookup")
                lookup_depth = 3;
            else if (flag == "--speculative=none") { lookup_depth = 0; use_mtp = false; }
            else if (flag.starts_with("--check-sequences=")) { check_sequence_count = std::stoi(flag.substr(18)); force_gpu = true; }
            else if (flag == "--check-verify")
                check_verify = true;
            else if (flag.starts_with("--lookup-depth="))
                lookup_depth = std::stoi(flag.substr(15));
            else if (flag.starts_with("--cpu-prepack-mib=")) cpu_prepack_mib = std::stoull(flag.substr(18));
            else if (flag.starts_with("--expert-pack=")) expert_pack_path=flag.substr(14);
            else if (flag.starts_with("--expert-pack-profile=")) expert_pack_profile=flag.substr(22);
            else if (flag.starts_with("--cpu-expert-backend=")) cpu_expert_backend=flag.substr(21);
            else if (flag.starts_with("--eval-corpus=")) eval_corpus=flag.substr(14);
            else if (flag.starts_with("--expert-block-screen=")) block_screen_profile=flag.substr(22);
            else if (flag.starts_with("--expert-block-capture=")) block_capture=flag.substr(23);
            else if (flag.starts_with("--expert-block-sweep=")) block_screen_sweep=token_ids(flag.substr(21));
            else if (flag.starts_with("--expert-block-layer-sweep=")) block_screen_layers=token_ids(flag.substr(27));
            else if (flag == "--eval-prefill") { eval_prefill=true; force_gpu=true; }
            else if (flag.starts_with("--eval-save-logits=")) eval_save=flag.substr(19);
            else if (flag.starts_with("--eval-reference=")) eval_reference=flag.substr(17);
            else if(flag=="--weight-pages=huge")strata::cpu::exl3::huge_pages=true;
            else if(flag=="--weight-pages=4k")strata::cpu::exl3::huge_pages=false;
            else if (flag == "--cpu-affinity=auto" || flag == "--cpu-affinity=numa") pin_cpu = true;
            else if (flag == "--cpu-affinity=none") pin_cpu = false;
            else if (flag.starts_with("--routing-trace=")) routing_trace_path = flag.substr(16);
            // Same as STRATA_GLM_ROUTE_AFFINITY (read when the decoder is built, after the flags).
            else if (flag.starts_with("--route-affinity=")) setenv("STRATA_GLM_ROUTE_AFFINITY", flag.substr(17).c_str(), 1);
            else if(flag=="--check-verify-graphs") {check_verify_graphs=true;force_gpu=true;}
            else if (flag == "--profile")
                profile = true;
            else if (flag == "--profile-decode")
                profile_decode = true;
            else if (flag == "--profile-prefill")
                profile_prefill = true;
            else if (flag == "--gpu-prefill")
                force_gpu = true;
            else if (flag == "--warm-weights")
                warm_weights = true;
            else if (flag == "--lock-weights") { warm_weights = true; lock_weights = true; }
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
        if (decode_mode_sweep) {
            if (use_mtp || lookup_depth || replay || stream_decode || check_prefill || check_verify || check_decode_graphs || check_verify_graphs || check_gpu_split || decode_cache_mib || decode_cache_auto || decode_cache_adapt || decode_prefill_cache || cpu_prepack_mib || expert_mib || std::filesystem::is_directory(argv[1]))
                throw std::invalid_argument("--decode-mode-sweep requires a GGUF CPU target with no other speculative, cache, or parity mode");
            use_mtp = force_gpu = true;
        }
        const auto devices = configure_devices(gpu_devices_text, gpu_budget);
        if (check_gpu_split && (devices.size() != 2 || use_mtp || replay || check_prefill))
            throw std::invalid_argument("--check-gpu-split needs two GPUs, single decode, and no other parity mode");
        if (tokens.empty() || steps < 1 || mib < 64 || threads < 1 || prefill_batch < 0 ||
            prefill_batch > 16384 || repetitions < 1 || context < 0 ||
            (context && tokens.size() + steps > (size_t)context) || expert_mib < 0 || (replay && steps < 2) ||
            (check_prefill && (prefill_batch == 1 || expert_mib != 0)))
            throw std::invalid_argument("invalid arguments");
        if (check_verify && (context ? (size_t)context : tokens.size() + steps) < tokens.size() + 8)
            throw std::invalid_argument("--check-verify needs eight available context positions");
        if (cpu_prepack_mib > 131072 || (cpu_prepack_mib && (stream_decode || expert_mib))) throw std::invalid_argument("invalid CPU packing budget or backend");
        if (direct_upload_enabled() && (!lock_weights || cpu_prepack_mib))
            throw std::invalid_argument("GLM: direct weight upload requires locked original weights");
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
        if (decode_cache_auto && use_mtp)
            throw std::invalid_argument("GLM: automatic decode cache currently requires single-token decoding");
        if (decode_cache_mib < 0 || decode_cache_mib > 12288 ||
            ((decode_cache_mib || decode_cache_auto) && ((use_mtp && !mtp_cpu_experts) || (lookup_depth && (!cached_lookup_enabled() || decode_cache_auto)) || stream_decode || expert_mib || cpu_prepack_mib)))
            throw std::invalid_argument("decode-cache-mib requires CPU target experts, single or CPU-draft MTP, no prepacking, and 0..12288 MiB");
        if (lookup_depth && decode_cache_mib) {
            const char *canonical = std::getenv("STRATA_GLM_CANON"), *pipeline = std::getenv("STRATA_GLM_STEP_PIPELINE");
            if (!canonical || std::string(canonical) != "1" || !pipeline || std::string(pipeline) != "1")
                throw std::invalid_argument("GLM: cached lookup requires canonical pipelined CPU target experts");
        }
        if (decode_cache_slots < 0 || decode_cache_slots > 12960 ||
            (decode_cache_slots && (!decode_cache_mib || decode_cache_auto)))
            throw std::invalid_argument("decode-cache-slots needs a cache budget and 0..12960 slots");
        if(std::filesystem::is_directory(argv[1])&&lookup_depth)throw std::invalid_argument("EXL3: lookup speculation is not qualified");
        if(!block_screen_profile.empty()||!block_capture.empty()) {
            if(eval_corpus.empty()||eval_prefill||use_mtp||lookup_depth||check_sequence_count||stream_decode||
               expert_pack_path.empty()||!expert_pack_profile.empty()||cpu_prepack_mib||expert_mib||
               decode_cache_mib||decode_cache_auto||decode_prefill_cache||!remote_tp_address.empty()||
               !stops.empty()||check_verify||check_verify_graphs)
                throw std::invalid_argument("expert block screen is only for ordinary --eval-corpus with a Q23 pack and no expert cache");
        }
        if(!block_screen_sweep.empty()) {
            if(block_screen_profile.empty()||!block_capture.empty()||!eval_save.empty()||eval_reference.empty())
                throw std::invalid_argument("block sweep requires a screen profile and evaluation reference, without capture or save");
            std::set<int> unique;for(int keep:block_screen_sweep)if(keep<1||keep>128||!unique.insert(keep).second)throw std::invalid_argument("block sweep budgets must be distinct integers in 1..128");
        }
        if(!block_screen_layers.empty()) {
            if(block_screen_sweep.empty())throw std::invalid_argument("layer sensitivity requires block sweep budgets");
            std::set<int> unique;for(int layer:block_screen_layers)if(layer<3||layer>44||!unique.insert(layer).second)throw std::invalid_argument("block sensitivity layers must be distinct integers in 3..44");
        }
        Decoder decoder(argv[1], context ? context : tokens.size() + steps, (size_t)mib * 1024 * 1024,
                        threads, (size_t)expert_mib * 1024 * 1024, pin_cpu,expert_pack_path,expert_pack_profile,cpu_expert_backend);
        if (check_sequence_count) decoder.enable_sequences(check_sequence_count);
        if(check_native_dense_q8) {
            if((context ? size_t(context) : tokens.size()+steps)<tokens.size()+8)
                throw std::invalid_argument("--check-native-dense-q8 requires native EXL3 and eight available positions");
            decoder.set_native_dense_q8(true);
        }
        if (prefill_batch == 0 || prefill_batch > cpu::MAXT || force_gpu)
            prefill_batch = decoder.enable_gpu(prefill_batch, gpu_budget, lookup_depth > 0 || check_verify || check_decode_graphs);
        if (direct_upload_enabled()) decoder.warm_weights(true, devices);
        decoder.configure_prefill(devices, gpu_budget, prefill_cache_bytes, prefill_cache_auto, use_mtp, tensor_experts, tensor_batches);
        decoder.set_gpu_decode_experts(stream_decode);
        if (decode_prefill_cache && (cpu_prepack_mib || expert_mib))
            throw std::invalid_argument("GLM: decode prefill cache cannot be combined with CPU prepacking or legacy expert cache");
        decoder.set_decode_prefill_cache(decode_prefill_cache);
        decoder.set_profile(profile);
        decoder.set_routing_trace(routing_trace_path);
        if (const char *scores = std::getenv("STRATA_GLM_ROUTER_SCORE_TRACE")) decoder.set_router_score_trace(scores);
        decoder.set_decode_cache_budget((size_t)decode_cache_mib * MiB);
        decoder.set_decode_cache_auto(decode_cache_auto, decode_cache_extend);
        decoder.set_decode_graphs(decode_graphs);
        if(check_verify_graphs)decoder.set_verify_graphs(true);
        decoder.set_decode_cache_window(decode_cache_window);
        if (!remote_tp_address.empty()) {
            if (!(remote_tp_share > 0 && remote_tp_share < 1))
                throw std::invalid_argument("--remote-tp-share must be between 0 and 1");
            decoder.enable_remote_tp(remote_tp_address, remote_tp_share, remote_tp_reply != "f32", remote_tp_weights == "send");
        }
        if (decode_cache_adapt && (replay || check_verify || decode_repetitions > 1))
            throw std::invalid_argument("GLM: adaptive cache benchmarks require independent serve requests");
        decoder.set_decode_cache_adapt(decode_cache_adapt);
        decoder.set_decode_cache_slots(decode_cache_slots);
        decoder.set_mtp_cpu_experts(mtp_cpu_experts);
        if(!block_screen_profile.empty()||!block_capture.empty())decoder.enable_block_screen(block_screen_profile,block_capture);
        if(cpu_prepack_mib && std::getenv("STRATA_GLM_Q2_NUMA_WEIGHTS") && std::string(std::getenv("STRATA_GLM_Q2_NUMA_WEIGHTS"))=="1")throw std::invalid_argument("Q2 NUMA weights cannot use CPU format prepacking");
        decoder.prepare_numa_weights();
        if (check_sequence_count) { decoder.check_sequences(tokens); decoder.report_gpu(); return 0; }
        if (use_mtp) decoder.enable_mtp_capture();
        if (warm_weights && !direct_upload_enabled())
            decoder.warm_weights(lock_weights);
        if(eval_prefill && eval_corpus.empty())throw std::invalid_argument("--eval-prefill requires --eval-corpus");
        if(!eval_corpus.empty()) {
            if(use_mtp||lookup_depth||decode_mode_sweep||check_verify||stream_decode)throw std::invalid_argument("evaluation requires ordinary CPU target decode");
            if(block_screen_sweep.empty())evaluate_corpus(decoder,eval_corpus,eval_save,eval_reference,eval_prefill ? prefill_batch : 0);
            else for(int keep:block_screen_sweep){
                if(block_screen_layers.empty()){decoder.set_block_screen_budget(keep);std::cout<<"{\"block_budget\":"<<keep<<"}\n"<<std::flush;evaluate_corpus(decoder,eval_corpus,"",eval_reference,0);}
                else for(int layer:block_screen_layers){decoder.set_block_screen_layer(layer,keep);std::cout<<"{\"block_budget\":"<<keep<<",\"block_layer\":"<<layer<<"}\n"<<std::flush;evaluate_corpus(decoder,eval_corpus,"",eval_reference,0);}
            }
            decoder.report_cache();return 0;
        }
        std::vector<float> logits;
        auto prefill = [&](int width) {
            std::vector<float> last;
            for (size_t t = 0; t < tokens.size(); t += width) {
                const size_t end = std::min(tokens.size(), t + width);
                last = decoder.batch(std::vector<int>(tokens.begin() + t, tokens.begin() + end),
                    !(std::filesystem::is_directory(argv[1]) && width <= cpu::MAXT));
            }
            return last;
        };
        if (check_prefill || check_gpu_split) {
            if (check_gpu_split) decoder.set_secondary_enabled(false);
            auto initial = decoder.snapshot();
            const auto reference_start = std::chrono::steady_clock::now();
            auto expected = prefill(check_gpu_split ? prefill_batch : reference_batch);
            if (check_gpu_split)
                std::cerr << "PREFILL_SPLIT mode=single_gpu tokens=" << tokens.size() << " ms=" <<
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reference_start).count() << '\n';
            auto expected_state = decoder.snapshot();
            decoder.restore(initial);
            if (check_gpu_split) decoder.set_secondary_enabled(true);
            const auto split_start = std::chrono::steady_clock::now();
            logits = prefill(prefill_batch);
            if (check_gpu_split)
                std::cerr << "PREFILL_SPLIT mode=two_gpu tokens=" << tokens.size() << " ms=" <<
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - split_start).count() << '\n';
            auto actual_state = decoder.snapshot();
            auto close = [&](const std::vector<float> &a, const std::vector<float> &b) {
                if (a.size() != b.size())
                    return false;
                if (check_gpu_split || !(force_gpu || prefill_batch > cpu::MAXT))
                    return a == b;
                double error = 0, norm = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (a[i] == -std::numeric_limits<float>::infinity() && b[i] == a[i]) continue;
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
                    if (!close(actual_state.layers[l][i], expected_state.layers[l][i])) {
                        std::cerr << "parity state layer=" << l << " buffer=" << i << '\n';
                        same = false;
                    }
            same &= std::max_element(logits.begin(), logits.end()) - logits.begin() ==
                    std::max_element(expected.begin(), expected.end()) - expected.begin();
            std::cerr << "parity greedy actual="
                      << (std::max_element(logits.begin(), logits.end()) - logits.begin()) << " reference="
                      << (std::max_element(expected.begin(), expected.end()) - expected.begin()) << "\n";
            if (!same)
                throw std::runtime_error(
                    "GLM: batched prefill state or logits differ from sequential execution");
            std::cerr << (check_gpu_split ? "two-GPU prefill state and logits identical\n"
                                          : "batched prefill state and logits passed parity\n");
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
                if (profile_prefill && trial == 0) check(cudaProfilerStart());
                logits = prefill(prefill_batch);
                if (profile_prefill && trial == 0) check(cudaProfilerStop());
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
        if(check_native_dense_q8) {
            const auto quantized_state=decoder.snapshot();const auto quantized_logits=logits;
            decoder.reset();decoder.set_native_dense_q8(false,false);
            auto reference_logits=prefill(prefill_batch);
            decoder.set_native_dense_q8(false,true);
            std::vector<int> window;std::vector<std::vector<float>> reference_rows;
            for(size_t i=0;i<8;++i){window.push_back(greedy(reference_logits));reference_rows.push_back(reference_logits);if(i+1<8)reference_logits=decoder.step(window[i]);}
            decoder.restore(quantized_state);decoder.set_native_dense_q8(true);
            auto actual=quantized_logits;double max_kl=0,max_relative=0;int top1=0;
            for(size_t row=0;row<window.size();++row) {
                const auto& expected=reference_rows[row];double amax=-INFINITY,bmax=-INFINITY,error=0,energy=0;
                for(size_t i=0;i<actual.size();++i) {
                    if(actual[i]==-INFINITY && expected[i]==-INFINITY)continue;
                    if(!std::isfinite(actual[i])||!std::isfinite(expected[i]))throw std::runtime_error("EXL3 Q8: nonfinite quality logits");
                    amax=std::max(amax,double(expected[i]));bmax=std::max(bmax,double(actual[i]));
                    error+=double(actual[i]-expected[i])*(actual[i]-expected[i]);energy+=double(expected[i])*expected[i];
                }
                double asum=0,bsum=0;
                for(size_t i=0;i<actual.size();++i){asum+=std::exp(double(expected[i])-amax);bsum+=std::exp(double(actual[i])-bmax);}
                double az=amax+std::log(asum),bz=bmax+std::log(bsum),kl=0;
                for(size_t i=0;i<actual.size();++i)if(std::isfinite(expected[i]))kl+=std::exp(double(expected[i])-az)*(double(expected[i])-az-double(actual[i])+bz);
                double relative=std::sqrt(error/std::max(energy,1e-30));bool same=greedy(actual)==greedy(expected);top1+=same;
                max_kl=std::max(max_kl,kl);max_relative=std::max(max_relative,relative);
                std::cerr<<"EXL3_Q8_QUALITY row="<<row<<" kl_bf16_to_q8="<<kl<<" relative_L2="<<relative<<" greedy_reference="<<greedy(expected)<<" greedy_actual="<<greedy(actual)<<'\n';
                if(row+1<window.size())actual=decoder.step(window[row]);
            }
            decoder.restore(quantized_state);logits=quantized_logits;
            std::cerr<<"EXL3_Q8_QUALITY_SUMMARY rows="<<window.size()<<" top1_equal="<<top1<<" max_kl="<<max_kl<<" max_relative_L2="<<max_relative<<'\n';
            if(max_kl>.01 || top1!=int(window.size()))throw std::runtime_error("EXL3 dense quantization: quality smoke threshold exceeded (max KL .01, all eight greedy predictions must match)");
        }
        const bool mtp_prepared_for_cache = use_mtp && decoder.prepare_mtp_for_cache(tokens, draft_depth);
        // Own retained rollback history before sizing the tier. Lookup has no draft block,
        // but rejected windows still need the same target-state recovery as MTP.
        if (lookup_depth && decode_cache_mib && cached_lookup_enabled())
            decoder.enable_verify_history(lookup_depth);
        decoder.probe_decode_cache(logits, stops);
        const auto cache_prepare_start = std::chrono::steady_clock::now();
        decoder.prepare_decode_cache();
        decoder.freeze_cache();
        std::cerr << "RESPONSE_PREPARE kind=decode_cache ms=" <<
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cache_prepare_start).count() << '\n';
        decoder.check_exact_cache(logits);
        if (check_decode_graphs) {
            // Compare identical resident slots; adaptation resumes only after
            // the checkpoint is restored, so cache arithmetic cannot drift.
            decoder.set_decode_cache_adapt(false);
            auto cp = decoder.checkpoint();
            decoder.save(cp);
            const std::vector<int> window{greedy(logits), 11, 9647, 0, 1, 2, 3, 4};
            std::vector<std::vector<float>> expected;
            const bool check_device_experts = decoder.uses_device_experts();
            decoder.set_device_experts(false);
            decoder.set_decode_graphs(false);
            for (int token : window) expected.push_back(decoder.step(token));
            decoder.restore(cp);
            decoder.set_decode_graphs(true);
            decoder.set_device_experts(check_device_experts);
            for (size_t i = 0; i < window.size(); ++i)
                if (decoder.step(window[i]) != expected[i])
                    throw std::runtime_error("GLM: CUDA graph decode logits differ from direct launches");
            decoder.restore(cp);
            decoder.set_decode_graphs(decode_graphs);
            decoder.set_decode_cache_adapt(decode_cache_adapt);
            std::cerr << "DECODE_GRAPHS eight sequential logits and rollback identical\n";
            if (std::getenv("STRATA_GLM_CHECK_TIER_ADAPT")) {
                const bool old_tier = decoder.set_tier_adaptation(false);
                decoder.save(cp);
                std::vector<int> inputs;
                std::vector<std::vector<float>> fixed_rows;
                auto current = logits;
                for (int i = 0; i < 32; ++i) {
                    inputs.push_back(greedy(current));
                    current = decoder.step(inputs.back());
                    fixed_rows.push_back(current);
                }
                decoder.restore(cp);
                decoder.set_tier_adaptation(true);
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const auto actual = decoder.step(inputs[i]);
                    if (actual != fixed_rows[i]) {
                        size_t differences = 0; float max_error = 0;
                        for (size_t j = 0; j < actual.size(); ++j) {
                            differences += actual[j] != fixed_rows[i][j];
                            max_error = std::max(max_error, std::abs(actual[j] - fixed_rows[i][j]));
                        }
                        std::cerr << "TIER_PARITY_MISMATCH step=" << i << " logits=" << differences
                                  << " max_abs=" << max_error << " greedy_equal="
                                  << (greedy(actual) == greedy(fixed_rows[i])) << '\n';
                        throw std::runtime_error("GLM: adaptive residency changed target logits");
                    }
                }
                decoder.restore(cp);
                decoder.set_tier_adaptation(old_tier);
                std::cerr << "TIER_PARITY steps=32 all_logits=bitwise_equal\n";
                decoder.set_tier_adaptation(false);
                decoder.set_check_cpu_residency(true);
                decoder.set_decode_graphs(false);
                for (size_t i = 0; i < inputs.size(); ++i)
                    if (decoder.step(inputs[i]) != fixed_rows[i])
                        throw std::runtime_error("GLM: all-CPU residency changed target logits at step " + std::to_string(i));
                decoder.restore(cp);
                decoder.set_check_cpu_residency(false);
                decoder.set_decode_graphs(decode_graphs);
                decoder.set_tier_adaptation(old_tier);
                std::cerr << "CPU_RESIDENCY_PARITY steps=32 all_logits=bitwise_equal\n";
            }
        }
        if(check_verify_graphs) {
            auto cp=decoder.checkpoint();decoder.save(cp);
            for(int allocation=0;allocation<2;++allocation) {
                decoder.enable_verify_history(3);
                for(int width:{1,2,3,4,1,2,4,3})for(bool history:{false,true}) {
                    std::vector<int> window{greedy(logits),11,9647,0};window.resize(width);
                    decoder.restore(cp);decoder.set_decode_graphs(false);
                    auto expected=history?decoder.verify(window):decoder.batch(window,false,true);
                    decoder.set_decode_graphs(true);
                    for(int repetition=0;repetition<3;++repetition) {
                        decoder.restore(cp);auto actual=history?decoder.verify(window):decoder.batch(window,false,true);
                        if(actual!=expected)throw std::runtime_error("GLM: verification graph logits differ from direct batches");
                        if(history)for(int accepted=1;accepted<=std::min(width,decoder.verify_history_slots());++accepted) {
                            if(!decoder.restore_verified(accepted))throw std::runtime_error("GLM: missing graph verification prefix");
                            auto got=decoder.step(17);decoder.restore(cp);decoder.set_decode_graphs(false);
                            decoder.batch(std::vector<int>(window.begin(),window.begin()+accepted),false);
                            auto reference=decoder.step(17);decoder.set_decode_graphs(true);
                            if(got!=reference)throw std::runtime_error("GLM: verification graph rollback changed next logits");
                            // Rebuild retained history before testing the next prefix.
                            decoder.restore(cp);decoder.verify(window);
                        }
                    }
                }
                decoder.clear_verify_history();
            }
            decoder.restore(cp);decoder.set_decode_graphs(decode_graphs);
            std::cerr<<"VERIFY_GRAPHS widths=1,2,3,4 mixed_width_warm_capture_replay_history_reallocation_and_rollback bitwise_equal\n";
        }
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
            if (actual != expected) {
                const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
                const size_t at = mismatch.first - actual.begin();
                std::cerr << "VERIFY_MISMATCH index=" << at << " token=" << at / (expected.size() / window.size())
                          << " actual=" << *mismatch.first << " expected=" << *mismatch.second << '\n';
                throw std::runtime_error("GLM: verify logits differ from sequential decode");
            }
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
        if (const char *value = std::getenv("STRATA_GLM_CHECK_LOOKUP")) {
            size_t consumed = 0; int check_tokens = std::stoi(value, &consumed);
            if (check_tokens == 1) check_tokens = 256;
            if (value[consumed] || check_tokens < 8 || check_tokens > 512)
                throw std::invalid_argument("GLM: lookup check needs 8..512 tokens");
            if (!lookup_depth || use_mtp) throw std::invalid_argument("GLM: lookup check requires lookup-only decoding");
            auto checkpoint = decoder.checkpoint(); decoder.save(checkpoint);
            auto reference_logits = logits;
            std::vector<int> expected, actual;
            for (int i = 0; i < check_tokens; ++i) {
                expected.push_back(greedy(reference_logits));
                if (i + 1 != check_tokens) reference_logits = decoder.step(expected.back());
            }
            decoder.restore(checkpoint);
            auto checked_logits = logits;
            auto checked = generate_lookup(decoder, checked_logits, tokens, check_tokens, {}, lookup_depth,
                                           [&](int token) { actual.push_back(token); }, false);
            if (actual != expected) throw std::runtime_error("GLM: cached lookup changed greedy target output");
            decoder.restore(checkpoint);
            std::cerr << "LOOKUP_PARITY tokens=" << check_tokens << " output=identical proposed=" << checked.proposed
                      << " accepted=" << checked.accepted << " replayed=" << checked.replayed << '\n';
        }
        if (const char *value = std::getenv("STRATA_GLM_CHECK_AFFINITY")) {
            size_t consumed = 0; int rows = std::stoi(value, &consumed);
            if (rows == 1) rows = 64;
            if (value[consumed] || rows < 8 || rows > 256 || use_mtp || lookup_depth)
                throw std::invalid_argument("GLM: affinity quality check needs 8..256 rows and ordinary decoding");
            const bool old_tier = decoder.set_tier_adaptation(false);
            decoder.set_decode_graphs(false);
            auto state = decoder.snapshot(); // host memory: no extra GPU checkpoint
            const float original_affinity = decoder.set_affinity_for_quality(0.f);
            std::vector<float> grid{original_affinity};
            if (const char *specification = std::getenv("STRATA_GLM_AFFINITY_QUALITY_GRID")) {
                grid.clear(); std::istringstream parser(specification); std::string part;
                while (std::getline(parser, part, ',')) {
                    size_t end = 0; const float affinity = std::stof(part, &end);
                    if (part[end] || !(affinity >= 0.f && affinity <= 2.f))
                        throw std::invalid_argument("GLM: invalid affinity quality grid");
                    grid.push_back(affinity);
                }
                if (grid.empty()) throw std::invalid_argument("GLM: empty affinity quality grid");
            }
            decoder.set_route_bias_enabled(false);
            std::vector<int> inputs;
            std::vector<std::vector<float>> reference;
            auto current = logits;
            for (int i = 0; i < rows; ++i) {
                inputs.push_back(greedy(current));
                current = decoder.step(inputs.back()); reference.push_back(current);
            }
            for (float affinity : grid) {
                decoder.restore(state); decoder.set_route_bias_enabled(true);
                decoder.set_affinity_for_quality(affinity);
                double sum_kl = 0, max_kl = 0, sum_relative = 0; int top1 = 0;
                for (int row = 0; row < rows; ++row) {
                    auto actual = decoder.step(inputs[row]); const auto &expected = reference[row];
                    const double amax = *std::max_element(expected.begin(), expected.end());
                    const double bmax = *std::max_element(actual.begin(), actual.end());
                    double az = 0, bz = 0, error = 0, energy = 0;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
                            throw std::runtime_error("GLM: nonfinite affinity quality logits");
                        az += std::exp(double(expected[i]) - amax); bz += std::exp(double(actual[i]) - bmax);
                        const double difference = double(actual[i]) - expected[i];
                        error += difference * difference; energy += double(expected[i]) * expected[i];
                    }
                    az = amax + std::log(az); bz = bmax + std::log(bz); double kl = 0;
                    for (size_t i = 0; i < actual.size(); ++i)
                        kl += std::exp(double(expected[i]) - az) * (double(expected[i]) - az - actual[i] + bz);
                    kl = std::max(0.0, kl); const bool equal = greedy(actual) == greedy(expected);
                    top1 += equal; sum_kl += kl; max_kl = std::max(max_kl, kl);
                    sum_relative += std::sqrt(error / std::max(energy, 1e-30));
                    std::cerr << "AFFINITY_QUALITY_ROW row=" << row << " kl=" << kl << " top1_equal=" << equal << '\n';
                }
                std::cerr << "AFFINITY_QUALITY rows=" << rows << " affinity=" << affinity << " mean_kl=" << sum_kl / rows
                          << " max_kl=" << max_kl << " top1_equal=" << top1 << " top1_share=" << double(top1) / rows
                          << " mean_relative_L2=" << sum_relative / rows << " reference=unbiased_strict_same_prefill_fixed_tier\n";
            }
            decoder.restore(state); decoder.set_route_bias_enabled(true);
            decoder.set_affinity_for_quality(original_affinity);
            decoder.set_decode_graphs(decode_graphs); decoder.set_tier_adaptation(old_tier);
        }
        if (cpu_prepack_mib > 131072) throw std::invalid_argument("CPU packing budget must be at most 131072 MiB");
        const auto cpu_prepare_start = std::chrono::steady_clock::now();
        decoder.prepare_cpu(cpu_prepack_mib * MiB);
        std::cerr << "RESPONSE_PREPARE kind=cpu ms=" <<
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_prepare_start).count() << '\n';
        if (use_mtp && !mtp_prepared_for_cache) {
            const auto start = std::chrono::steady_clock::now();
            decoder.prepare_mtp(tokens);
            std::cerr << "MTP_PRIME ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() << '\n';
        }
        std::unique_ptr<Decoder::Snapshot> decode_start_state;
        const auto decode_start_logits = logits;
        if (decode_repetitions > 1 || decode_mode_sweep) decode_start_state = std::make_unique<Decoder::Snapshot>(decoder.snapshot());
        const int modes = decode_mode_sweep ? (decode_mtp_sweep ? 4 : 7) : 1;
        for (int mode_index = 0; mode_index < modes; ++mode_index) {
            const int mode = decode_mtp_sweep && mode_index ? mode_index + 3 : mode_index;
            const bool trial_mtp = decode_mode_sweep ? mode >= 4 : use_mtp;
            const int trial_lookup = decode_mode_sweep ? (mode >= 1 && mode <= 3 ? mode : 0) : lookup_depth;
            const int trial_depth = decode_mode_sweep ? mode - 3 : draft_depth;
            if (decode_mode_sweep) std::cerr << "DECODE_MODE source=" << (trial_mtp ? "mtp" : trial_lookup ? "lookup" : "none")
                << " depth=" << (trial_mtp ? trial_depth : trial_lookup) << " shared_mtp_allocation=1\n";
        for (int trial = 0; trial < decode_repetitions; ++trial) {
            if (trial || decode_mode_sweep) {
                // A sweep changes the draft depth; plain repetitions keep the allocation the tier was sized around.
                if (decode_mode_sweep) decoder.clear_verify_history();
                else decoder.invalidate_verify_history();
                decoder.restore(*decode_start_state); logits = decode_start_logits;
            }
            lock_small_runtime_mappings();
            std::cerr << "DECODE_TRIAL index=" << trial << '\n';
        if (profile_decode) check(cudaProfilerStart());
        decoder.reset_decode_stats();
        if (trial_mtp) {
            generate_mtp(decoder, logits, steps, stops, trial_depth, [](int token) {
                std::cout << token << '\n' << std::flush;
            }, &tokens);
        } else if (trial_lookup) {
            generate_lookup(decoder, logits, tokens, steps, stops, trial_lookup, [](int token) {
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
                }
        decoder.flush_expert_observer();
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
