// Shared GLM runner/worker allocation helpers. Included inside the implementation namespace.
#pragma once
void check(cudaError_t e, std::source_location where = std::source_location::current()) {
    if (e != cudaSuccess) {
        const char *diagnose = std::getenv("STRATA_GLM_CUDA_CHECK_TRACE");
        if (diagnose && std::string(diagnose) == "1")
            throw std::runtime_error(std::string(cudaGetErrorString(e)) + " at " + where.function_name() +
                                     ":" + std::to_string(where.line()));
        throw std::runtime_error(cudaGetErrorString(e));
    }
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
// Report CUDA's visible ordinals, rather than nvidia-smi's host-wide indices.
// A launcher can restrict each child to a UUID without escaping a Slurm mask.
static void report_gpu_devices(std::ostream &out) {
    int count = 0; check(cudaGetDeviceCount(&count));
    out << "[";
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, device));
        char bus[32]{}; check(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
        if (device) out << ',';
        out << "{\"ordinal\":" << device << ",\"pci_bus_id\":" << std::quoted(bus)
            << ",\"name\":" << std::quoted(props.name)
            << ",\"total_bytes\":" << props.totalGlobalMem << ",\"uuid\":";
#ifdef STRATA_USE_HIP
        out << "null";
#else
        const auto &uuid = props.uuid;
        std::string name = "GPU-";
        constexpr char hex[] = "0123456789abcdef";
        for (int i = 0; i < 16; ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10) name += '-';
            const auto value = static_cast<unsigned char>(uuid.bytes[i]);
            name += hex[value >> 4]; name += hex[value & 15];
        }
        out << std::quoted(name);
#endif
        out << '}';
    }
    out << "]\n";
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
        const auto status = cudaMemcpy(p, x, n, cudaMemcpyHostToDevice);
        const char *registered = std::getenv("STRATA_GLM_PREFILL_REGISTERED_UPLOAD");
        if (status != cudaErrorInvalidValue || !registered || std::string(registered) != "1") {
            check(status);
            return;
        }
        // Registering page-rounded expert ranges can also register a boundary
        // page of a neighbouring dense tensor. CUDA rejects copies spanning
        // registered and pageable memory. Keep its bytes unchanged and bound
        // the temporary storage instead of pinning the whole dense tensor.
        cudaGetLastError();
        const size_t panel = std::min<size_t>(n, 1024 * 1024);
        if (!panel) { check(status); return; }
        void *bounce = nullptr;
        check(cudaMallocHost(&bounce, panel));
        try {
            for (size_t offset = 0; offset < n; offset += panel) {
                const size_t count = std::min(panel, n - offset);
                std::memcpy(bounce, static_cast<const uint8_t *>(x) + offset, count);
                check(cudaMemcpy(static_cast<uint8_t *>(p) + offset, bounce, count, cudaMemcpyHostToDevice));
            }
        } catch (...) { cudaFreeHost(bounce); throw; }
        cudaFreeHost(bounce);
        std::cerr << "REGISTERED_BOUNDARY_COPY bytes=" << n << " panel_bytes=" << panel << '\n';
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
    bool write_combined = false;
    explicit Pinned(size_t bytes, bool portable = false, bool wc = false) : write_combined(wc) {
#ifdef STRATA_USE_HIP
        if (wc) throw std::invalid_argument("GLM: write-combined staging requires CUDA");
        if (portable) check(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
        else check(cudaMallocHost(&p, bytes));
#else
        unsigned flags = (portable ? cudaHostAllocPortable : 0) | (wc ? cudaHostAllocWriteCombined : 0);
        if (flags) check(cudaHostAlloc(&p, bytes, flags));
        else check(cudaMallocHost(&p, bytes));
#endif
    }
    void finish_writes() const { if (write_combined) _mm_sfence(); }
    ~Pinned() { cudaFreeHost(p); }
};
// Only instantiated by opt-in paths. Drain DMA before the mapping is unpinned.
struct HostMappingRegistration {
    void *mapping;
    int owner = 0;
    bool registered = false;
    HostMappingRegistration(void *mapping, size_t bytes, const char *role) : mapping(mapping) {
        check(cudaGetDevice(&owner));
        // The mapping covers whole pages; a protocol size that is not a page multiple is refused otherwise.
        const size_t page=4096;
        check(cudaSetDevice(owner));
        auto status=cudaHostRegister(mapping,(bytes+page-1)/page*page,cudaHostRegisterPortable);
        if (status!=cudaSuccess) {
            std::cerr << "MIXED_REGISTER_RETRY " << role << " first_error=" << cudaGetErrorString(status)
                      << " address=" << mapping << " bytes=" << bytes << '\n';
            cudaGetLastError();
            status=cudaHostRegister(mapping,(bytes+page-1)/page*page,cudaHostRegisterMapped|cudaHostRegisterPortable);
        }
        registered=status==cudaSuccess;
        std::cerr << "MIXED_REGISTER " << role << '=' << registered;
        if (!registered) {cudaGetLastError();std::cerr << " error=" << cudaGetErrorString(status);}
        std::cerr << '\n';
    }
    ~HostMappingRegistration() {
        if (!registered) return;
        int previous=owner;
        cudaGetDevice(&previous);
        cudaSetDevice(owner);
        cudaDeviceSynchronize();
        cudaHostUnregister(mapping);
        cudaSetDevice(previous);
    }
    HostMappingRegistration(const HostMappingRegistration &)=delete;
};
struct ResidentSideStream {
    cudaStream_t stream=nullptr;
    cudaEvent_t fork[2]{}, done[2]{}, consumed[2]{};
    bool consumed_valid[2]{};
    ResidentSideStream() {
        try {
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for (int g=0;g<2;++g) {
                check(cudaEventCreateWithFlags(&fork[g],cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&done[g],cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&consumed[g],cudaEventDisableTiming));
            }
        } catch (...) {release();throw;}
    }
    void release() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        for (int g=0;g<2;++g) {
            if (fork[g]) cudaEventDestroy(fork[g]);
            if (done[g]) cudaEventDestroy(done[g]);
            if (consumed[g]) cudaEventDestroy(consumed[g]);
            fork[g]=done[g]=consumed[g]=nullptr;
        }
        if (stream) cudaStreamDestroy(stream);
        stream=nullptr;
    }
    ~ResidentSideStream() {release();}
    ResidentSideStream(const ResidentSideStream &)=delete;
};
