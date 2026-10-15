// KV-offload feasibility probe for GLM-5.3-Flash (docs/GLM_KV_OFFLOAD.md): how fast can a GPU gather the sparse
// attention rows from pinned host memory instead of VRAM?
//
// Geometry: per token and MLA layer one 2,048-byte latent row; a decode position selects 2,048 rows per layer as 512
// runs of 4 consecutive tokens (index pool 4). One "position" here = 12 layers x 512 runs x 8 KiB = 48 MiB.
//
//   nvcc  -O3 tools/glm_kv_offload_probe.cu -o kv-probe-cuda
//   hipcc -O3 -DSTRATA_PROBE_HIP tools/glm_kv_offload_probe.cu -o kv-probe-hip
//   ./kv-probe-hip [context_tokens=131072] [positions=64] [union=1.0]
// union > 1 adds that factor of extra distinct runs per position (a verify window's tokens select overlapping sets).
// Prints one line per mode: ms per position, effective GB/s.
#ifdef STRATA_PROBE_HIP
#include <hip/hip_runtime.h>
#define API(x) hip##x
#define HOST_ALLOC_FLAGS (hipHostMallocMapped | hipHostMallocPortable)
#define HostAllocCall(p, n) hipHostMalloc(p, n, HOST_ALLOC_FLAGS)
#define HostFreeCall(p) (void)hipHostFree(p)
#define FreeCall(p) (void)hipFree(p)
#else
#include <cuda_runtime.h>
#define API(x) cuda##x
#define HostAllocCall(p, n) cudaHostAlloc(p, n, cudaHostAllocMapped | cudaHostAllocPortable)
#define HostFreeCall(p) (void)cudaFreeHost(p)
#define FreeCall(p) (void)cudaFree(p)
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { auto err = (x); if (err != API(Success)) { std::fprintf(stderr, "%s: %s\n", #x, API(GetErrorString)(err)); std::exit(1); } } while (0)

constexpr int LAYERS = 12, ROW = 2048, POOL = 4, RUNS = 512;
constexpr size_t RUN_BYTES = size_t(ROW) * POOL;   // 8 KiB

// One block per run; 16-byte loads. src is a device pointer (VRAM) or a mapped host pointer (zero copy).
__global__ void gather_runs(const uint4 *src, const int *runs, uint4 *dst, size_t layer_offset_vec) {
    const int r = blockIdx.x;
    const size_t from = layer_offset_vec + size_t(runs[r]) * (RUN_BYTES / 16);
    const size_t to = size_t(r) * (RUN_BYTES / 16);
    for (int i = threadIdx.x; i < int(RUN_BYTES / 16); i += blockDim.x) dst[to + i] = src[from + i];
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const size_t ctx = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 131072;
    const int positions = argc > 2 ? std::atoi(argv[2]) : 64;
    const double uni = argc > 3 ? std::atof(argv[3]) : 1.0;
    const size_t pools = ctx / POOL;
    const int runs = std::min<size_t>(pools, size_t(RUNS * uni));
    if (ctx < 8192 || positions < 4 || runs < 1) return 2;
    const size_t layer_bytes = ctx * ROW, total = layer_bytes * LAYERS;
    std::printf("context=%zu layers=%d cache_MiB=%.0f runs_per_layer=%d position_MiB=%.1f\n", ctx, LAYERS,
                total / 1048576.0, runs, double(runs) * RUN_BYTES * LAYERS / 1048576.0);

    void *host = nullptr;
    CK(HostAllocCall(&host, total));
    std::memset(host, 1, total);
    void *host_dev = nullptr;
    CK(API(HostGetDevicePointer)(&host_dev, host, 0));
    void *vram = nullptr;
    const bool vram_ok = API(Malloc)(&vram, total) == API(Success);
    if (!vram_ok) { (void)API(GetLastError)(); vram = nullptr; }
    void *out = nullptr, *ids = nullptr, *stage = nullptr;
    CK(API(Malloc)(&out, size_t(runs) * RUN_BYTES));
    CK(API(Malloc)(&ids, size_t(runs) * LAYERS * positions * sizeof(int)));
    CK(API(Malloc)(&stage, layer_bytes));

    // Random distinct sorted runs per layer and position; recent pools are favoured as real selections are.
    std::mt19937 rng(20261016);
    std::vector<int> all(size_t(runs) * LAYERS * positions);
    for (int pl = 0; pl < LAYERS * positions; ++pl) {
        std::vector<int> pick;
        std::uniform_int_distribution<size_t> any(0, pools - 1);
        while ((int)pick.size() < runs) {
            size_t p = (rng() & 3) ? any(rng) : pools - 1 - (rng() % std::min<size_t>(pools, 2048));
            pick.push_back(int(p));
            if ((int)pick.size() == runs) { std::sort(pick.begin(), pick.end()); pick.erase(std::unique(pick.begin(), pick.end()), pick.end()); }
        }
        std::copy(pick.begin(), pick.end(), all.begin() + size_t(pl) * runs);
    }
    CK(API(Memcpy)(ids, all.data(), all.size() * sizeof(int), API(MemcpyHostToDevice)));
    API(Stream_t) stream;
    CK(API(StreamCreate)(&stream));

    auto run_gather = [&](const void *src, const char *name) {
        auto one = [&](int pos) {
            for (int l = 0; l < LAYERS; ++l)
                gather_runs<<<runs, 256, 0, stream>>>((const uint4 *)src, (const int *)ids + (size_t(pos) * LAYERS + l) * runs,
                                                       (uint4 *)out, size_t(l) * layer_bytes / 16);
        };
        one(0); CK(API(StreamSynchronize)(stream));
        const double t0 = now_ms();
        for (int p = 0; p < positions; ++p) one(p);
        CK(API(StreamSynchronize)(stream));
        const double ms = (now_ms() - t0) / positions;
        const double bytes = double(runs) * RUN_BYTES * LAYERS;
        std::printf("MODE %-22s ms_per_position=%.3f GB_s=%.2f\n", name, ms, bytes / ms / 1e6);
    };
    run_gather(host_dev, "zero_copy_host_gather");
    if (vram_ok) run_gather(vram, "vram_gather");
    else std::printf("MODE vram_gather skipped (cache does not fit in VRAM)\n");

    // Prefill alternative: DMA one whole layer of the cache to a VRAM staging buffer.
    CK(API(MemcpyAsync)(stage, host, layer_bytes, API(MemcpyHostToDevice), stream));
    CK(API(StreamSynchronize)(stream));
    const int reps = 8;
    const double t0 = now_ms();
    for (int i = 0; i < reps; ++i)
        CK(API(MemcpyAsync)(stage, (char *)host + size_t(i % LAYERS) * layer_bytes, layer_bytes, API(MemcpyHostToDevice), stream));
    CK(API(StreamSynchronize)(stream));
    const double ms = (now_ms() - t0) / reps;
    std::printf("MODE %-22s ms_per_layer=%.3f GB_s=%.2f ms_all_layers=%.1f\n", "dma_layer_stage", ms,
                layer_bytes / ms / 1e6, ms * LAYERS);
    HostFreeCall(host);
    if (vram) FreeCall(vram);
    FreeCall(out); FreeCall(ids); FreeCall(stage);
    return 0;
}
