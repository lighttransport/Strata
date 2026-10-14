// Shared GLM runner/worker allocation helpers. Included inside the implementation namespace.
#pragma once
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
struct Pinned {
    void *p = nullptr;
    explicit Pinned(size_t bytes) { check(cudaMallocHost(&p, bytes)); }
    ~Pinned() { cudaFreeHost(p); }
};
