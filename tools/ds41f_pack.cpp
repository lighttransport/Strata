// DeepSeek V4.1 Flash pack tool: the routed experts of the release checkpoint (safetensors, MXFP4) requantized to a
// Strata expert sidecar GGUF, and the FP8/BF16 trunk to a trunk GGUF. The Engram tables are never touched: they
// stay in the release shards (FP8 rows + E8M0 scales) and are read per token from SSD by the runner.
//
//   strata-ds41f-pack experts  MODEL_DIR OUT.gguf|--chunk-dir=DIR --gate=T --up=T --down=T (--imatrix=FILE | --uncalibrated)
//                              [--keep=FILE] [--layers=0-39] [--threads=N] [--deadline=EPOCH_SECONDS] [--layer-seconds=S]
//                              [--stop-after-layers=N] [--plan-only]
//   strata-ds41f-pack assemble DIR OUT.gguf [--check] [--layers=0-39]   (concatenate a chunk directory into the sidecar;
//                              --check only reports; the default range is all 40 layers)
//   strata-ds41f-pack trunk    MODEL_DIR OUT.gguf [--trunk=q4_k|q5_k|q6_k|q8_0] [--head=q8_0|q4_k] [--threads=N]
//   strata-ds41f-pack verify   [MODEL_DIR] PACK.gguf       (hashes; with MODEL_DIR also the source identity)
//   strata-ds41f-pack dequant  MODEL_DIR LAYER EXPERT gate|up|down OUT.f32   (debug: exact MXFP4 -> f32 of one projection)
//
// Streamable and resumable, for a 100 MB/s shared filesystem and 3-hour job slots: the unit of work is one layer
// (384 experts x 3 projections), the unit of resume is one projection chunk file. A layer's experts are read once,
// in source-file order, by one reader thread (each expert's w1/w3/w2 + scales = 56 MB), quantized by the worker
// threads, and appended in expert order to three `blk.L.ffn_{gate,up,down}_exps.chunk.partial` files; when the
// layer completes each chunk gets a `*.chunk.json` (type, shape, bytes, FNV hash, normalized MSE, kept ids) and is
// renamed atomically. A rerun skips layers whose jsons exist and match `plan.json`, deletes stray partials, and
// stops cleanly (exit 3, "more to do") when the next layer would not finish before --deadline. `OUT.gguf` is the
// chunk directory `OUT.gguf.d` assembled at the end; `assemble` also runs on another machine after the chunks were
// copied there, so a slow node never has to write one 100-150 GB file.
//
// Several processes may fill one chunk directory at once (e.g. one per node, disjoint --layers): plan.json is created
// atomically (hard link of a private temp file) and validated by everyone, each process only clears stale partials of
// its own layers, and `--plan-only` creates/validates the plan (and the imatrix entries of --layers) without packing.
// --threads defaults to the CPUs this process may run on, minus one. Source reads drop their page cache after use.
//
// Identity: the pack binds to the checkpoint through the shards' basenames, sizes and header bytes (FNV-1a-64 in the
// GGUF key, SHA-256 per shard header in plan.json). File mtimes and the directory path are deliberately excluded
// so a re-downloaded or copied checkpoint still matches.
//
// Sidecar format as strata-glm-q2-pack / strata-mimo-q2-pack: GGUF v3 holding blk.L.ffn_{gate,up,down}_exps.weight
// with strata.expert_pack.{version,source,calibrated,architecture="deepseek4",formats,hash.<tensor>}; a REAP-pruned
// layer adds blk.L.ffn_exps_keep (I32, the kept source expert ids).
//
// MXFP4 as released: w{1,3} I8 [2304, 2560] and w2 I8 [5120, 1152] hold two FP4 (E2M1, sign bit 3) per byte, element
// 2c in the low nibble and 2c+1 in the high nibble (ds41f_kernels.c:96); the `.scale` tensor holds one E8M0 (2^(e-127))
// per 32 consecutive inputs. Dense FP8 (F8_E4M3) tensors carry a `.scale` of E8M0 blocks of 32 output rows x
// (cols / scale_cols) inputs (config: weight_block_size [32, 32]). Both dequantizers below are exact in f32.
//
// Calibration: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS require an importance matrix (ggml refuses them
// without one); --imatrix takes llama-imatrix's GGUF (`<pack tensor name>.in_sum2` F32 [cols, experts] and
// `.counts` F32 [experts]). --uncalibrated is allowed for the K-quants only.
#include "strata/artifact/fp8_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/host/ds41f_names.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <cerrno>
#ifdef __linux__
#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {
namespace st = strata::artifact::fp8;
namespace fs = std::filesystem;
using strata::GgufFile;

constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
constexpr int kLayers = STRATA_DS41F_LAYERS, kExperts = STRATA_DS41F_EXPERTS, kHidden = STRATA_DS41F_HIDDEN, kFf = STRATA_DS41F_EXPERT_FF;
constexpr uint32_t kGgufI32 = 26;
constexpr int kExitMoreToDo = 3;

template<class T> void write(std::ostream& out, T value) { out.write(reinterpret_cast<const char*>(&value), sizeof value); }
void string(std::ostream& out, const std::string& value) { write<uint64_t>(out, value.size()); out.write(value.data(), value.size()); }
uint64_t hash_bytes(uint64_t hash, const uint8_t* p, size_t n) { for (size_t i = 0; i < n; ++i) { hash ^= p[i]; hash *= 1099511628211ULL; } return hash; }
bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }
bool ends_with(const std::string& s, const char* suffix) { const size_t n = std::strlen(suffix); return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0; }

// CPUs this process may run on (the job's affinity), minus one for the reader/writer; 1..128.
int default_threads() {
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) return std::clamp(CPU_COUNT(&set) - 1, 1, 128);
#endif
    return std::clamp(int(std::thread::hardware_concurrency()) - 1, 1, 128);
}

// Rows [0, rows) split over `threads` ranges whose starts are multiples of `align`; fn(r0, r1, thread).
template<class F> void parallel_rows(int64_t rows, int threads, int64_t align, F fn) {
    threads = int(std::max<int64_t>(1, std::min<int64_t>(threads, (rows + align - 1) / align)));
    if (threads == 1) { fn(int64_t(0), rows, 0); return; }
    const int64_t blocks = (rows + align - 1) / align;
    std::vector<std::thread> pool; std::exception_ptr failure; std::mutex mu;
    for (int t = 0; t < threads; ++t) {
        const int64_t r0 = std::min(rows, blocks * t / threads * align), r1 = std::min(rows, blocks * (t + 1) / threads * align);
        pool.emplace_back([&, r0, r1, t] { try { if (r1 > r0) fn(r0, r1, t); } catch (...) { std::lock_guard lk(mu); failure = std::current_exception(); } });
    }
    for (auto& th : pool) th.join();
    if (failure) std::rethrow_exception(failure);
}

// CPU seconds (user, system) and peak RSS of this process, for the per-layer log.
struct Usage { double user = 0, sys = 0; long maxrss_kb = 0; };
Usage usage_now() {
    Usage u;
#ifdef __linux__
    rusage r{}; if (getrusage(RUSAGE_SELF, &r) == 0) {
        u.user = double(r.ru_utime.tv_sec) + 1e-6 * double(r.ru_utime.tv_usec); u.sys = double(r.ru_stime.tv_sec) + 1e-6 * double(r.ru_stime.tv_usec); u.maxrss_kb = r.ru_maxrss; }
#endif
    return u;
}

// ---- SHA-256 (plan.json records one per shard header; small, self-contained) ------------------------------------
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64]; size_t len = 0; uint64_t total = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) { const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10); w[i] = w[i - 16] + s0 + w[i - 7] + s1; }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), maj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const uint8_t* p, size_t n) {
        total += n;
        while (n) { const size_t take = std::min(n, 64 - len); std::memcpy(buf + len, p, take); len += take; p += take; n -= take; if (len == 64) { block(buf); len = 0; } }
    }
    std::string hex() {
        const uint64_t bits = total * 8; uint8_t pad = 0x80; update(&pad, 1); const uint8_t zero = 0;
        while (len != 56) update(&zero, 1);
        uint8_t l[8]; for (int i = 0; i < 8; ++i) l[i] = uint8_t(bits >> (56 - 8 * i)); update(l, 8);
        std::ostringstream o; for (uint32_t v : h) o << std::hex << std::setw(8) << std::setfill('0') << v; return o.str();
    }
};

const std::map<std::string, ggml_type> kTypes = {
    {"q2_k", GGML_TYPE_Q2_K}, {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
    {"q8_0", GGML_TYPE_Q8_0}, {"iq1_s", GGML_TYPE_IQ1_S}, {"iq1_m", GGML_TYPE_IQ1_M}, {"iq2_xxs", GGML_TYPE_IQ2_XXS},
    {"iq2_xs", GGML_TYPE_IQ2_XS}, {"iq2_s", GGML_TYPE_IQ2_S}, {"iq3_xxs", GGML_TYPE_IQ3_XXS}, {"iq3_s", GGML_TYPE_IQ3_S},
    {"iq4_xs", GGML_TYPE_IQ4_XS}, {"bf16", GGML_TYPE_BF16}, {"f16", GGML_TYPE_F16}, {"f32", GGML_TYPE_F32},
};

ggml_type type_of(const std::string& name) {
    auto it = kTypes.find(name);
    if (it == kTypes.end()) throw std::runtime_error("unknown quantization type: " + name);
    return it->second;
}

// ---- exact dequantizers of the release formats --------------------------------------------------------------
constexpr float kFp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};   // E2M1 magnitudes

inline float e8m0(uint8_t e) { return e == 0xff ? std::nanf("") : std::ldexp(1.0f, int(e) - 127); }

inline float fp8_e4m3(uint8_t v) {
    const int sign = v >> 7, exp = (v >> 3) & 15, man = v & 7;
    float x;
    if (exp == 0) x = std::ldexp(float(man) / 8.0f, -6);
    else if (exp == 15 && man == 7) x = std::nanf("");
    else x = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
    return sign ? -x : x;
}

void dequant_mxfp4(const uint8_t* packed, const uint8_t* scales, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint8_t* p = packed + size_t(r) * cols / 2;
        const uint8_t* s = scales + size_t(r) * cols / 32;
        float* o = out + size_t(r) * cols;
        for (int c = 0; c < cols; ++c) {
            const unsigned code = (p[c / 2] >> ((c % 2) * 4)) & 15;
            const float mag = kFp4[code & 7];
            o[c] = ((code & 8) ? -mag : mag) * e8m0(s[c / 32]);
        }
    }
}

void dequant_fp8(const uint8_t* w, const uint8_t* scales, int rows, int cols, int scale_rows, int scale_cols, float* out) {
    const int block_cols = cols / scale_cols;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int sr = std::min(r / 32, scale_rows - 1), sc = c / block_cols;
            out[size_t(r) * cols + c] = fp8_e4m3(w[size_t(r) * cols + c]) * e8m0(scales[size_t(sr) * scale_cols + sc]);
        }
}

inline float bf16(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ---- the checkpoint ------------------------------------------------------------------------------------------
struct Checkpoint {
    fs::path dir;
    st::Index index;
    std::map<std::string, std::pair<uint64_t, std::string>> shards;   // basename -> (size, header sha256)
    const st::Tensor& at(const std::string& name) const {
        auto it = index.find(name);
        if (it == index.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    const st::Tensor* find(const std::string& name) const { auto it = index.find(name); return it == index.end() ? nullptr : &it->second; }
    std::vector<uint8_t> read(const st::Tensor& t) const { std::vector<uint8_t> v(t.bytes); pread_drop(t.path, t.offset, v.data(), t.bytes); return v; }
    // Same bytes as st::pread_all; the pages are dropped after the read so streaming ~25 GB of source per node does not
    // fill the (small) node memory with page cache.
    static void pread_drop(const std::string& path, uint64_t offset, uint8_t* out, size_t bytes) {
#ifdef __linux__
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t n = ::pread(fd, out + done, bytes - done, off_t(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { ::close(fd); throw std::runtime_error("short read from " + path); }
            done += size_t(n);
        }
        ::posix_fadvise(fd, off_t(offset), off_t(bytes), POSIX_FADV_DONTNEED);
        ::close(fd);
#else
        st::pread_all(path, offset, out, bytes);
#endif
    }
};

int experts_in_layer(const Checkpoint& c, int layer) {
    int count = 0;
    char name[STRATA_DS41F_NAME_MAX];
    for (int e = 0; e < kExperts; ++e) {
        strata_ds41f_expert_source_name(name, sizeof name, layer, e, STRATA_DS41F_GATE, "weight");
        if (!c.find(name)) break;
        ++count;
    }
    if (count == 0) throw std::runtime_error("layer " + std::to_string(layer) + " has no routed experts in the checkpoint");
    return count;
}

Checkpoint open_checkpoint(const std::string& dir, bool need_experts = true) {
    Checkpoint c;
    c.dir = dir;
    c.index = st::scan(dir);
    if (c.index.empty()) throw std::runtime_error("no safetensors shards in " + dir);
    std::set<std::string> paths;
    for (const auto& [_, t] : c.index) paths.insert(t.path);
    for (const auto& p : paths) {
        std::ifstream f(p, std::ios::binary); uint64_t n = 0;
        if (!f.read(reinterpret_cast<char*>(&n), 8)) throw std::runtime_error("cannot read header of " + p);
        std::vector<uint8_t> h(n); if (!f.read(reinterpret_cast<char*>(h.data()), std::streamsize(n))) throw std::runtime_error("cannot read header of " + p);
        Sha256 s; s.update(h.data(), h.size());
        c.shards[fs::path(p).filename().string()] = {fs::file_size(p), s.hex()};
    }
    if (need_experts) {
        char name[STRATA_DS41F_NAME_MAX];
        strata_ds41f_expert_source_name(name, sizeof name, 0, 0, STRATA_DS41F_GATE, "weight");
        const auto& w1 = c.at(name);
        if (w1.dtype != "I8" || w1.shape != std::vector<uint64_t>{uint64_t(kFf), uint64_t(kHidden / 2)})
            throw std::runtime_error("not a DeepSeek V4.1 Flash checkpoint: " + std::string(name) + " is not I8 [2304, 2560]");
    }
    return c;
}

// Identity of the checkpoint: shard basenames, sizes and header bytes (no mtime, no directory). Binds a pack to its source.
uint64_t source_fingerprint(const Checkpoint& c) {
    uint64_t hash = kFnvBasis;
    for (const auto& [name, info] : c.shards) {
        hash = hash_bytes(hash, reinterpret_cast<const uint8_t*>(name.data()), name.size());
        hash = hash_bytes(hash, reinterpret_cast<const uint8_t*>(&info.first), sizeof info.first);
        hash = hash_bytes(hash, reinterpret_cast<const uint8_t*>(info.second.data()), info.second.size());
    }
    return hash;
}

struct Entry { std::string name; std::vector<uint64_t> shape; uint32_t type; uint64_t offset, bytes, hash = 0; double error = 0, norm = 0; };

std::string header(const std::vector<Entry>& entries, const std::map<std::string, std::string>& kv_strings,
                   const std::map<std::string, uint64_t>& kv_u64, const std::map<std::string, uint32_t>& kv_u32, bool calibrated,
                   const std::map<std::string, uint64_t>& hashes) {
    std::ostringstream out(std::ios::binary);
    write<uint32_t>(out, 0x46554747); write<uint32_t>(out, 3); write<uint64_t>(out, entries.size());
    write<uint64_t>(out, 1 + kv_strings.size() + kv_u64.size() + kv_u32.size() + hashes.size());
    string(out, "strata.expert_pack.calibrated"); write<uint32_t>(out, 7); write<uint8_t>(out, calibrated);
    for (const auto& [k, v] : kv_strings) { string(out, k); write<uint32_t>(out, 8); string(out, v); }
    for (const auto& [k, v] : kv_u64) { string(out, k); write<uint32_t>(out, 10); write<uint64_t>(out, v); }
    for (const auto& [k, v] : kv_u32) { string(out, k); write<uint32_t>(out, 4); write<uint32_t>(out, v); }
    for (const auto& [k, v] : hashes) { string(out, k); write<uint32_t>(out, 10); write<uint64_t>(out, v); }
    for (const auto& e : entries) {
        string(out, e.name); write<uint32_t>(out, uint32_t(e.shape.size()));
        for (auto d : e.shape) write<uint64_t>(out, d);
        write<uint32_t>(out, e.type); write<uint64_t>(out, e.offset);
    }
    auto result = out.str(); result.resize((result.size() + 31) / 32 * 32, 0); return result;
}

void sync_and_drop(const fs::path& path) {
#ifdef __linux__
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) { ::fdatasync(fd); ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
#else
    (void)path;
#endif
}

std::vector<float> imatrix_importance(const GgufFile& im, const std::string& weight, int columns, int experts) {
    const auto* sum = im.find(weight + ".in_sum2"); const auto* cnt = im.find(weight + ".counts");
    if (!sum || !cnt) throw std::runtime_error("imatrix has no entry for " + weight);
    // counts: [experts] (older writers) or {1, experts} as llama-imatrix writes it
    const bool counts_ok = cnt->shape == std::vector<uint64_t>{uint64_t(experts)} || cnt->shape == std::vector<uint64_t>{1, uint64_t(experts)};
    if (sum->type != 0 || cnt->type != 0 || sum->shape != std::vector<uint64_t>{uint64_t(columns), uint64_t(experts)} || !counts_ok)
        throw std::runtime_error("imatrix geometry mismatch for " + weight);
    const float* s = reinterpret_cast<const float*>(im.tensor_data(*sum)); const float* c = reinterpret_cast<const float*>(im.tensor_data(*cnt));
    std::vector<double> pooled(columns, 0.); double total = 0;
    for (int e = 0; e < experts; ++e) { total += c[e]; for (int k = 0; k < columns; ++k) pooled[k] += s[size_t(e) * columns + k]; }
    std::vector<float> out(size_t(columns) * experts);
    for (int e = 0; e < experts; ++e) for (int k = 0; k < columns; ++k) {
        const double v = c[e] > 0 ? s[size_t(e) * columns + k] / c[e] : (total > 0 ? pooled[k] / total : 1.);
        if (!std::isfinite(v) || v < 0) throw std::runtime_error("nonfinite or negative imatrix value for " + weight);
        out[size_t(e) * columns + k] = float(std::max(v, 1e-12));
    }
    return out;
}

// --keep=FILE: lines `<layer>: e0 e1 e2 ...` (kept source expert ids, ascending); layers absent keep every expert.
std::map<int, std::vector<int>> read_keep(const std::string& path, std::string& text) {
    std::map<int, std::vector<int>> keep;
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot read keep map " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        text += line + "\n";
        std::istringstream in(line); int layer; char colon;
        if (!(in >> layer >> colon) || colon != ':' || layer < 0 || layer >= kLayers) throw std::runtime_error("bad keep line: " + line);
        std::vector<int> ids; int e;
        while (in >> e) { if (e < 0 || e >= kExperts || (!ids.empty() && e <= ids.back())) throw std::runtime_error("bad keep ids: " + line); ids.push_back(e); }
        if (ids.empty()) throw std::runtime_error("empty keep list: " + line);
        keep[layer] = ids;
    }
    return keep;
}

std::pair<int, int> parse_layers(const std::string& text) {
    const auto dash = text.find('-');
    if (dash == std::string::npos) { const int l = std::stoi(text); return {l, l}; }
    return {std::stoi(text.substr(0, dash)), std::stoi(text.substr(dash + 1))};
}

struct Projection { int rows, cols; };
Projection geometry(int proj) { return proj == STRATA_DS41F_DOWN ? Projection{kHidden, kFf} : Projection{kFf, kHidden}; }
uint64_t slice_bytes(ggml_type t, int proj) { return uint64_t(ggml_row_size(t, geometry(proj).cols)) * geometry(proj).rows; }

// The six source tensors of one expert.
struct ExpertSource { const st::Tensor* w[3]; const st::Tensor* s[3]; };
ExpertSource expert_source(const Checkpoint& c, int layer, int expert) {
    ExpertSource src{};
    for (int proj = 0; proj < 3; ++proj) {
        char wname[STRATA_DS41F_NAME_MAX], sname[STRATA_DS41F_NAME_MAX];
        strata_ds41f_expert_source_name(wname, sizeof wname, layer, expert, strata_ds41f_proj(proj), "weight");
        strata_ds41f_expert_source_name(sname, sizeof sname, layer, expert, strata_ds41f_proj(proj), "scale");
        src.w[proj] = &c.at(wname); src.s[proj] = &c.at(sname);
        const auto g = geometry(proj);
        if (src.w[proj]->dtype != "I8" || src.w[proj]->bytes != size_t(g.rows) * g.cols / 2 || src.s[proj]->bytes != size_t(g.rows) * g.cols / 32)
            throw std::runtime_error(std::string("unexpected expert tensor layout: ") + wname);
    }
    return src;
}

void dequant_expert(const std::vector<uint8_t>& w, const std::vector<uint8_t>& s, int proj, std::vector<float>& out, const char* what) {
    const auto g = geometry(proj);
    out.resize(size_t(g.rows) * g.cols);
    dequant_mxfp4(w.data(), s.data(), g.rows, g.cols, out.data());
    for (float v : out) if (!std::isfinite(v)) throw std::runtime_error(std::string("nonfinite source weight in ") + what);
}

// ---- chunk directory ----------------------------------------------------------------------------------------------
std::string chunk_name(int layer, int proj) { char n[STRATA_DS41F_NAME_MAX]; strata_ds41f_expert_pack_name(n, sizeof n, layer, strata_ds41f_proj(proj)); return n; }
std::string json_escape(const std::string& s) { std::string o; for (char ch : s) { if (ch == '"' || ch == '\\') o += '\\'; o += ch; } return o; }

// Minimal reader for the flat json files this tool writes (strings, integers, floats, arrays of integers).
struct FlatJson {
    std::map<std::string, std::string> raw;
    static FlatJson parse(const fs::path& p) {
        std::ifstream f(p); if (!f) throw std::runtime_error("cannot read " + p.string());
        std::stringstream ss; ss << f.rdbuf(); const std::string t = ss.str();
        FlatJson j; size_t i = 0;
        auto skip = [&] { while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == ',' || t[i] == '{' || t[i] == '}' || t[i] == '\r' || t[i] == '\t')) ++i; };
        skip();
        while (i < t.size()) {
            if (t[i] != '"') throw std::runtime_error("malformed json " + p.string());
            const size_t e = t.find('"', i + 1); const std::string key = t.substr(i + 1, e - i - 1); i = e + 1;
            while (t[i] == ' ' || t[i] == ':') ++i;
            std::string val;
            if (t[i] == '"') { const size_t ve = t.find('"', i + 1); val = t.substr(i + 1, ve - i - 1); i = ve + 1; }
            else if (t[i] == '[') { const size_t ve = t.find(']', i); val = t.substr(i, ve - i + 1); i = ve + 1; }
            else { size_t ve = i; while (ve < t.size() && t[ve] != ',' && t[ve] != '}' && t[ve] != '\n') ++ve; val = t.substr(i, ve - i); i = ve; }
            j.raw[key] = val; skip();
        }
        return j;
    }
    const std::string& s(const std::string& k) const { auto it = raw.find(k); if (it == raw.end()) throw std::runtime_error("json lacks " + k); return it->second; }
    uint64_t u(const std::string& k) const { return std::stoull(s(k)); }
    double d(const std::string& k) const { return std::stod(s(k)); }
    std::vector<int64_t> list(const std::string& k) const {
        std::vector<int64_t> v; std::string body = s(k); for (char& ch : body) if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
        std::istringstream in(body); int64_t x; while (in >> x) v.push_back(x); return v;
    }
};

struct Plan {
    std::string gate, up, down, keep_text, imatrix_sha; uint64_t fingerprint = 0; std::map<std::string, std::pair<uint64_t, std::string>> shards;
    int first = 0, last = kLayers - 1;   // the layer range the directory is meant to hold (not part of the identity)
    std::string identity() const {
        std::string k = gate + "|" + up + "|" + down + "|keep:" + keep_text + "|imatrix:" + imatrix_sha + "|";
        for (const auto& [n, i] : shards) k += n + ":" + std::to_string(i.first) + ":" + i.second + ";";
        return k;
    }
    std::string identity_sha() const { Sha256 s; const auto id = identity(); s.update(reinterpret_cast<const uint8_t*>(id.data()), id.size()); return s.hex(); }
    // Create plan.json only if it does not exist yet: the complete file is written privately and hard-linked into
    // place (atomic create-if-absent), so concurrent processes never see a torn plan and the first one wins.
    void write_if_absent(const fs::path& p) const {
        char host[256] = "host"; ::gethostname(host, sizeof host - 1);
        const fs::path tmp = p.string() + "." + host + "." + std::to_string(::getpid()) + ".partial";
        write_to(tmp);
        std::error_code ec; fs::create_hard_link(tmp, p, ec);
        if (ec && ec != std::errc::file_exists) {   // filesystems without hard links: rename if still absent
            std::cerr << "PLAN link_unsupported (" << ec.message() << "), using rename\n";
            if (!fs::exists(p)) { fs::rename(tmp, p); return; }
        }
        fs::remove(tmp);
    }
    void write_to(const fs::path& tmp) const {
        std::ofstream o(tmp); o.exceptions(std::ios::failbit | std::ios::badbit);
        o << "{\"architecture\":\"deepseek4\",\"version\":1,\"gate\":\"" << gate << "\",\"up\":\"" << up << "\",\"down\":\"" << down
          << "\",\"source_fingerprint\":" << fingerprint << ",\"imatrix_sha256\":\"" << imatrix_sha << "\",\"keep\":\"" << json_escape(keep_text)
          << "\",\"identity_sha256\":\"" << identity_sha()
          << "\",\"layers\":\"" << first << "-" << last << "\",\"shards\":{";
        bool first = true;
        for (const auto& [n, i] : shards) { if (!first) o << ','; first = false; o << "\"" << n << "\":\"" << i.first << ":" << i.second << "\""; }
        o << "}}\n"; o.close(); sync_and_drop(tmp);
    }
    static std::string stored_identity(const fs::path& p) { return FlatJson::parse(p).s("identity_sha256"); }
};

std::string file_sha256(const fs::path& p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p.string());
    Sha256 s; std::vector<uint8_t> buf(1 << 20);
    while (f) { f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size())); s.update(buf.data(), size_t(f.gcount())); }
    return s.hex();
}

struct ChunkInfo { std::string name; uint32_t type; std::vector<uint64_t> shape; uint64_t bytes, hash; double nmse; std::vector<int64_t> keep; int layer; int proj; };

std::optional<ChunkInfo> read_chunk_json(const fs::path& dir, int layer, int proj) {
    const fs::path p = dir / (chunk_name(layer, proj) + ".chunk.json");
    if (!fs::exists(p)) return std::nullopt;
    const auto j = FlatJson::parse(p);
    ChunkInfo c; c.name = j.s("name"); c.type = uint32_t(j.u("type")); c.bytes = j.u("bytes"); c.hash = j.u("hash"); c.nmse = j.d("normalized_mse");
    for (auto d : j.list("shape")) c.shape.push_back(uint64_t(d)); c.keep = j.list("keep"); c.layer = layer; c.proj = proj;
    const fs::path data = dir / (chunk_name(layer, proj) + ".chunk");
    if (!fs::exists(data) || fs::file_size(data) != c.bytes) return std::nullopt;
    return c;
}

void write_chunk_json(const fs::path& dir, const ChunkInfo& c) {
    const fs::path p = dir / (c.name + ".chunk.json");
    std::ofstream o(p.string() + ".partial"); o.exceptions(std::ios::failbit | std::ios::badbit);
    o << "{\"name\":\"" << c.name << "\",\"layer\":" << c.layer << ",\"proj\":" << c.proj << ",\"type\":" << c.type << ",\"type_name\":\"" << ggml_type_name(ggml_type(c.type))
      << "\",\"shape\":[" << c.shape[0] << "," << c.shape[1] << "," << c.shape[2] << "],\"bytes\":" << c.bytes << ",\"hash\":" << c.hash
      << ",\"normalized_mse\":" << std::setprecision(12) << c.nmse << ",\"keep\":[";
    for (size_t i = 0; i < c.keep.size(); ++i) o << (i ? "," : "") << c.keep[i];
    o << "]}\n"; o.close(); fs::rename(p.string() + ".partial", p);
}

// One layer: read its experts once in source order, quantize all three projections, append to three chunk files.
void pack_layer(const Checkpoint& c, const fs::path& dir, int layer, const std::vector<int>& ids, int present, const ggml_type types[3],
                const GgufFile* imatrix, int threads) {
    std::vector<ExpertSource> sources; sources.reserve(ids.size());
    for (int e : ids) sources.push_back(expert_source(c, layer, e));
    // source-file order of the experts: (shard path, offset) of w1
    std::vector<size_t> order(ids.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::tie(sources[a].w[0]->path, sources[a].w[0]->offset) < std::tie(sources[b].w[0]->path, sources[b].w[0]->offset); });
    std::vector<std::vector<float>> importance(3);
    for (int proj = 0; proj < 3; ++proj) if (imatrix) importance[proj] = imatrix_importance(*imatrix, chunk_name(layer, proj), geometry(proj).cols, present);
    size_t slice[3]; for (int proj = 0; proj < 3; ++proj) slice[proj] = size_t(ggml_row_size(types[proj], geometry(proj).cols)) * geometry(proj).rows;

    struct Raw { size_t idx; std::vector<uint8_t> w[3], s[3]; };
    struct Packed { std::vector<uint8_t> q[3]; double err[3], norm[3]; bool done = false; };
    std::mutex mu; std::condition_variable cv_raw, cv_done;
    std::deque<std::unique_ptr<Raw>> raw; bool reader_done = false; std::exception_ptr failure;
    std::vector<Packed> packed(ids.size());
    const size_t depth = 2;
    std::thread reader; std::vector<std::thread> workers;
    // Joins every thread on any exit; on an exception it first tells them to stop.
    struct JoinAll {
        std::thread& reader; std::vector<std::thread>& workers; std::mutex& mu; std::condition_variable& a; std::condition_variable& b; std::exception_ptr& failure; bool& reader_done;
        ~JoinAll() {
            if (std::uncaught_exceptions() > 0) { std::lock_guard lk(mu); if (!failure) failure = std::make_exception_ptr(std::runtime_error("writer failed")); reader_done = true; a.notify_all(); b.notify_all(); }
            for (auto& t : workers) if (t.joinable()) t.join();
            if (reader.joinable()) reader.join();
        }
    } join_all{reader, workers, mu, cv_raw, cv_done, failure, reader_done};
    reader = std::thread([&] { try {
        for (size_t k : order) {
            auto r = std::make_unique<Raw>(); r->idx = k;
            for (int proj = 0; proj < 3; ++proj) { r->w[proj] = c.read(*sources[k].w[proj]); r->s[proj] = c.read(*sources[k].s[proj]); }
            std::unique_lock lk(mu); cv_raw.wait(lk, [&] { return raw.size() < depth + size_t(threads) || failure; });
            if (failure) return;
            raw.push_back(std::move(r)); cv_raw.notify_all();
        }
        std::lock_guard lk(mu); reader_done = true; cv_raw.notify_all();
    } catch (...) { std::lock_guard lk(mu); failure = std::current_exception(); reader_done = true; cv_raw.notify_all(); } });
    for (int t = 0; t < threads; ++t) workers.emplace_back([&] { try {
        std::vector<float> weights, decoded;
        for (;;) {
            std::unique_ptr<Raw> r;
            { std::unique_lock lk(mu); cv_raw.wait(lk, [&] { return !raw.empty() || reader_done || failure; });
              if (failure || (raw.empty() && reader_done)) return;
              r = std::move(raw.front()); raw.pop_front(); cv_raw.notify_all(); }
            Packed& out = packed[r->idx]; const int expert = ids[r->idx];
            for (int proj = 0; proj < 3; ++proj) {
                const auto g = geometry(proj);
                dequant_expert(r->w[proj], r->s[proj], proj, weights, chunk_name(layer, proj).c_str());
                out.q[proj].resize(slice[proj]);
                const float* im = importance[proj].empty() ? nullptr : importance[proj].data() + size_t(expert) * g.cols;
                const auto written = ggml_quantize_chunk(types[proj], weights.data(), out.q[proj].data(), 0, g.rows, g.cols, im);
                if (written != slice[proj]) throw std::runtime_error("quantizer byte count mismatch");
                const auto* traits = ggml_get_type_traits(types[proj]); decoded.resize(g.cols); out.err[proj] = out.norm[proj] = 0;
                for (int row = 0; row < g.rows; ++row) {
                    traits->to_float(out.q[proj].data() + size_t(row) * slice[proj] / g.rows, decoded.data(), g.cols);
                    for (int k = 0; k < g.cols; ++k) { const double w = weights[size_t(row) * g.cols + k], sc = im ? im[k] : 1., d = double(decoded[k]) - w; out.err[proj] += sc * d * d; out.norm[proj] += sc * w * w; }
                }
            }
            { std::lock_guard lk(mu); out.done = true; cv_done.notify_all(); }
        }
    } catch (...) { std::lock_guard lk(mu); failure = std::current_exception(); reader_done = true; cv_raw.notify_all(); cv_done.notify_all(); } });
    // writer: in ids order, so the chunk is the sidecar's byte layout
    std::ofstream out[3]; uint64_t hash[3]; double err[3] = {0, 0, 0}, norm[3] = {0, 0, 0};
    for (int proj = 0; proj < 3; ++proj) { out[proj].open(dir / (chunk_name(layer, proj) + ".chunk.partial"), std::ios::binary | std::ios::trunc); out[proj].exceptions(std::ios::failbit | std::ios::badbit); hash[proj] = kFnvBasis; }
    for (size_t i = 0; i < ids.size(); ++i) {
        { std::unique_lock lk(mu); cv_done.wait(lk, [&] { return packed[i].done || failure; }); if (failure) std::rethrow_exception(failure); }
        for (int proj = 0; proj < 3; ++proj) {
            out[proj].write(reinterpret_cast<const char*>(packed[i].q[proj].data()), std::streamsize(slice[proj]));
            hash[proj] = hash_bytes(hash[proj], packed[i].q[proj].data(), slice[proj]); err[proj] += packed[i].err[proj]; norm[proj] += packed[i].norm[proj];
            std::vector<uint8_t>().swap(packed[i].q[proj]);
        }
    }
    reader.join(); for (auto& t : workers) t.join();
    if (failure) std::rethrow_exception(failure);
    for (int proj = 0; proj < 3; ++proj) {
        out[proj].close();
        const fs::path partial = dir / (chunk_name(layer, proj) + ".chunk.partial"), final = dir / (chunk_name(layer, proj) + ".chunk");
        sync_and_drop(partial); fs::rename(partial, final);
        ChunkInfo info{chunk_name(layer, proj), uint32_t(types[proj]), {uint64_t(geometry(proj).cols), uint64_t(geometry(proj).rows), ids.size()},
                       slice[proj] * ids.size(), hash[proj], err[proj] / std::max(norm[proj], 1e-30), {}, layer, proj};
        if (ids.size() != size_t(present)) info.keep.assign(ids.begin(), ids.end());
        write_chunk_json(dir, info);
        std::cerr << "PACK chunk=" << info.name << " type=" << ggml_type_name(types[proj]) << " experts=" << ids.size() << " bytes=" << info.bytes
                  << " normalized_mse=" << std::setprecision(9) << info.nmse << '\n';
    }
}

// ---- assemble -----------------------------------------------------------------------------------------------------
std::vector<ChunkInfo> read_chunks(const fs::path& dir, std::vector<int>* missing = nullptr) {
    std::vector<ChunkInfo> chunks;
    for (int l = 0; l < kLayers; ++l) {
        int have = 0; ChunkInfo three[3];
        for (int proj = 0; proj < 3; ++proj) if (auto c = read_chunk_json(dir, l, proj)) { three[proj] = *c; ++have; }
        if (have == 3) for (auto& c : three) chunks.push_back(c);
        else if (have > 0 || fs::exists(dir / (chunk_name(l, 0) + ".chunk.partial"))) { if (missing) missing->push_back(l); }
    }
    return chunks;
}

int cmd_assemble(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("usage: strata-ds41f-pack assemble DIR OUT.gguf [--check] [--layers=A-B]   (default range: all 40 layers)");
    const fs::path dir = argv[2]; bool check = false; std::optional<std::pair<int, int>> range;
    std::string output = argc >= 4 && std::string(argv[3]).rfind("--", 0) != 0 ? argv[3] : "";
    for (int i = 3; i < argc; ++i) { std::string a = argv[i]; if (a == "--check") check = true; else if (starts_with(a, "--layers=")) range = parse_layers(a.substr(9)); else if (i == 3) continue; else throw std::runtime_error("unknown argument " + a); }
    const auto plan = FlatJson::parse(dir / "plan.json");
    std::vector<int> partial_layers; auto chunks = read_chunks(dir, &partial_layers);
    std::set<int> done; for (const auto& c : chunks) done.insert(c.layer);
    // default: the whole model; a partial pack needs an explicit --layers (plan.json's "layers" is informational only,
    // since several processes fill one directory with different ranges)
    if (!range) range = std::make_pair(0, kLayers - 1);
    const int first = range->first, last = range->second;
    std::cout << "ASSEMBLE range=" << first << "-" << last << '\n';
    std::vector<int> missing; for (int l = first; l <= last; ++l) if (!done.count(l)) missing.push_back(l);
    std::cout << "ASSEMBLE dir=" << dir << " layers_done=" << done.size() << " missing=" << missing.size() << (partial_layers.empty() ? "" : " partial_present") << '\n';
    if (check) { for (int l : missing) std::cout << "  missing layer " << l << '\n'; return missing.empty() ? 0 : kExitMoreToDo; }
    if (!missing.empty()) throw std::runtime_error("cannot assemble: " + std::to_string(missing.size()) + " layers missing (first " + std::to_string(missing[0]) + ")");
    if (output.empty()) throw std::runtime_error("assemble needs OUT.gguf");
    const fs::path out_path(output), partial = output + ".partial";
    if (fs::exists(out_path) || fs::exists(partial)) throw std::runtime_error("refusing to overwrite existing pack or partial file");
    std::vector<Entry> entries; std::vector<const ChunkInfo*> payload; uint64_t offset = 0;
    for (const auto& c : chunks) {
        if (c.layer < first || c.layer > last) continue;
        entries.push_back({c.name, c.shape, c.type, offset, c.bytes, c.hash, c.nmse, 1.0}); payload.push_back(&c); offset += c.bytes;
        if (c.proj == 2 && !c.keep.empty()) {
            char name[STRATA_DS41F_NAME_MAX]; strata_ds41f_keep_name(name, sizeof name, c.layer);
            const uint64_t bytes = (c.keep.size() * 4 + 31) / 32 * 32;
            entries.push_back({name, {c.keep.size()}, kGgufI32, offset, bytes}); payload.push_back(&c); offset += bytes;
        }
    }
    const bool calibrated = !plan.s("imatrix_sha256").empty();
    std::map<std::string, std::string> kvs = {{"strata.expert_pack.architecture", "deepseek4"}, {"strata.expert_pack.formats", plan.s("gate") + "," + plan.s("up") + "," + plan.s("down")}};
    std::map<std::string, uint64_t> kvu = {{"strata.expert_pack.source", plan.u("source_fingerprint")}};
    std::map<std::string, uint32_t> kv32 = {{"strata.expert_pack.version", 1}, {"strata.expert_pack.layer_count", uint32_t(kLayers)}, {"strata.expert_pack.expert_count", uint32_t(kExperts)}};
    std::map<std::string, uint64_t> hashes;
    // keep-list payloads are hashed now; chunk hashes come from their jsons and are re-checked while copying
    for (size_t i = 0; i < entries.size(); ++i) {
        auto& e = entries[i];
        if (e.type == kGgufI32) { std::vector<uint8_t> buf(e.bytes, 0); std::vector<int32_t> ids(payload[i]->keep.begin(), payload[i]->keep.end()); std::memcpy(buf.data(), ids.data(), ids.size() * 4); e.hash = hash_bytes(kFnvBasis, buf.data(), buf.size()); }
        hashes["strata.expert_pack.hash." + e.name] = e.hash;
    }
    const auto h = header(entries, kvs, kvu, kv32, calibrated, hashes);
    std::ofstream out(partial, std::ios::binary | std::ios::trunc); out.exceptions(std::ios::failbit | std::ios::badbit); out.write(h.data(), h.size());
    std::vector<uint8_t> buf(8 << 20);
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.type == kGgufI32) { std::vector<uint8_t> b(e.bytes, 0); std::vector<int32_t> ids(payload[i]->keep.begin(), payload[i]->keep.end()); std::memcpy(b.data(), ids.data(), ids.size() * 4); out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size())); continue; }
        std::ifstream in(dir / (e.name + ".chunk"), std::ios::binary); if (!in) throw std::runtime_error("cannot read chunk " + e.name);
        uint64_t hash = kFnvBasis, left = e.bytes;
        while (left) { const size_t n = size_t(std::min<uint64_t>(left, buf.size())); if (!in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n))) throw std::runtime_error("short chunk " + e.name); hash = hash_bytes(hash, buf.data(), n); out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(n)); left -= n; }
        if (hash != e.hash) throw std::runtime_error("chunk hash mismatch while assembling " + e.name);
        std::cerr << "ASSEMBLE tensor=" << e.name << " bytes=" << e.bytes << " done=" << i + 1 << "/" << entries.size() << '\n';
    }
    out.close(); sync_and_drop(partial); fs::rename(partial, out_path);
    std::ofstream report(output + ".quality.json"); report.exceptions(std::ios::failbit | std::ios::badbit);
    report << "{\"architecture\":\"deepseek4\",\"source_fingerprint\":\"" << plan.u("source_fingerprint") << "\",\"calibrated\":" << (calibrated ? "true" : "false")
           << ",\"formats\":[\"" << plan.s("gate") << "\",\"" << plan.s("up") << "\",\"" << plan.s("down") << "\"],\"projections\":[";
    bool firstp = true;
    for (const auto& e : entries) { if (e.type == kGgufI32) continue; if (!firstp) report << ','; firstp = false;
        report << "{\"name\":\"" << e.name << "\",\"experts\":" << e.shape[2] << ",\"bytes\":" << e.bytes << ",\"normalized_mse\":" << std::setprecision(12) << e.error << '}'; }
    report << "]}\n";
    std::cerr << "ASSEMBLE complete path=" << output << " bytes=" << offset + h.size() << '\n';
    return 0;
}

// ---- experts ------------------------------------------------------------------------------------------------------
int cmd_experts(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("usage: strata-ds41f-pack experts MODEL_DIR (OUT.gguf | --chunk-dir=DIR) --gate=T --up=T --down=T (--imatrix=FILE | --uncalibrated) [--keep=FILE] [--layers=0-39] [--threads=N] [--deadline=EPOCH] [--layer-seconds=S] [--stop-after-layers=N] [--plan-only]");
    std::string gate = "iq1_s", up = "iq1_s", down = "iq2_xxs", imatrix_path, keep_path, chunk_dir, output; bool uncalibrated = false;
    int first = 0, last = kLayers - 1, threads = default_threads(), stop_after = -1; int64_t deadline = 0; double seed_seconds = 0; bool plan_only = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (starts_with(a, "--gate=")) gate = a.substr(7); else if (starts_with(a, "--up=")) up = a.substr(5); else if (starts_with(a, "--down=")) down = a.substr(7);
        else if (starts_with(a, "--imatrix=")) imatrix_path = a.substr(10); else if (a == "--uncalibrated") uncalibrated = true;
        else if (starts_with(a, "--keep=")) keep_path = a.substr(7); else if (starts_with(a, "--layers=")) std::tie(first, last) = parse_layers(a.substr(9));
        else if (starts_with(a, "--threads=")) threads = std::stoi(a.substr(10)); else if (starts_with(a, "--chunk-dir=")) chunk_dir = a.substr(12);
        else if (starts_with(a, "--deadline=")) deadline = std::stoll(a.substr(11)); else if (starts_with(a, "--stop-after-layers=")) stop_after = std::stoi(a.substr(20));
        else if (starts_with(a, "--layer-seconds=")) seed_seconds = std::stod(a.substr(16)); else if (a == "--plan-only") plan_only = true;
        else if (a.rfind("--", 0) != 0 && output.empty()) output = a; else throw std::runtime_error("unknown argument " + a);
    }
    if (output.empty() == chunk_dir.empty()) throw std::runtime_error("give either OUT.gguf or --chunk-dir=DIR");
    if (uncalibrated == !imatrix_path.empty() || first < 0 || first > last || last >= kLayers || threads < 1 || threads > 128)
        throw std::runtime_error("invalid settings: choose --imatrix=FILE or --uncalibrated; layers 0..39; threads 1..128");
    const ggml_type types[3] = {type_of(gate), type_of(up), type_of(down)};
    for (auto t : types) if (uncalibrated && ggml_quantize_requires_imatrix(t)) throw std::runtime_error(std::string(ggml_type_name(t)) + " requires --imatrix");
    std::unique_ptr<GgufFile> imatrix; std::string imatrix_sha;
    if (!imatrix_path.empty()) {
        imatrix = std::make_unique<GgufFile>(imatrix_path);
        const auto* type = imatrix->get("general.type");
        if (!type || type->s != "imatrix") throw std::runtime_error("not a llama-imatrix GGUF: " + imatrix_path);
        imatrix_sha = file_sha256(imatrix_path);
    }
    std::string keep_text; const auto keep = keep_path.empty() ? std::map<int, std::vector<int>>{} : read_keep(keep_path, keep_text);
    const auto c = open_checkpoint(argv[2]);
    const fs::path dir = chunk_dir.empty() ? fs::path(output + ".d") : fs::path(chunk_dir);
    if (!output.empty() && (fs::exists(output) || fs::exists(output + ".partial"))) throw std::runtime_error("refusing to overwrite existing pack or partial file");
    fs::create_directories(dir);
    Plan plan{gate, up, down, keep_text, imatrix_sha, source_fingerprint(c), c.shards, first, last};
    const fs::path plan_path = dir / "plan.json";
    if (!fs::exists(plan_path)) plan.write_if_absent(plan_path);
    if (Plan::stored_identity(plan_path) != plan.identity_sha())
        throw std::runtime_error("plan mismatch: " + plan_path.string() + " was made with other formats, keep list, imatrix or source shards");
    if (plan_only) {
        if (imatrix) for (int l = first; l <= last; ++l) { const int present = experts_in_layer(c, l);
            for (int proj = 0; proj < 3; ++proj) imatrix_importance(*imatrix, chunk_name(l, proj), geometry(proj).cols, present); }
        std::cerr << "PACK plan ok dir=" << dir << " identity=" << plan.identity_sha() << (imatrix ? " imatrix=checked" : "") << " layers=" << first << "-" << last << '\n';
        return 0;
    }
    // stale partials of an interrupted run of THIS range only (other processes may be writing other layers here)
    for (int l = first; l <= last; ++l) for (int proj = 0; proj < 3; ++proj) for (const char* suffix : {".chunk.partial", ".chunk.json.partial"}) {
        const fs::path f = dir / (chunk_name(l, proj) + suffix);
        if (fs::exists(f)) { std::cerr << "PACK removing stale " << f.filename() << '\n'; fs::remove(f); }
    }
    ggml_cpu_init();
    const auto t0 = std::chrono::steady_clock::now();
    int done_now = 0; double layer_seconds = seed_seconds; bool more = false;
    for (int l = first; l <= last; ++l) {
        bool have = true; for (int proj = 0; proj < 3; ++proj) if (!read_chunk_json(dir, l, proj)) have = false;
        if (have) { std::cerr << "PACK layer=" << l << " already complete\n"; continue; }
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        if (deadline && (done_now > 0 || layer_seconds > 0) && double(now) + layer_seconds * 1.2 > double(deadline)) { more = true; std::cerr << "PACK deadline: stopping before layer " << l << '\n'; break; }
        if (stop_after >= 0 && done_now >= stop_after) { more = true; std::cerr << "PACK stop-after-layers reached before layer " << l << '\n'; break; }
        const int present = experts_in_layer(c, l);
        std::vector<int> ids;
        if (auto it = keep.find(l); it != keep.end()) { ids = it->second; if (ids.back() >= present) throw std::runtime_error("keep list exceeds the experts present in layer " + std::to_string(l)); }
        else for (int e = 0; e < present; ++e) ids.push_back(e);
        const uint64_t need = uint64_t(ids.size()) * (slice_bytes(types[0], 0) + slice_bytes(types[1], 1) + slice_bytes(types[2], 2));
        if (fs::space(dir).available < need + (uint64_t(1) << 30)) throw std::runtime_error("insufficient disk capacity for the next layer plus 1 GiB reserve");
        const auto l0 = std::chrono::steady_clock::now(); const Usage u0 = usage_now();
        pack_layer(c, dir, l, ids, present, types, imatrix.get(), threads);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count(); const Usage u1 = usage_now();
        layer_seconds = done_now ? 0.5 * (layer_seconds + s) : s; ++done_now;
        std::cerr << "PACK layer=" << l << " seconds=" << std::setprecision(4) << s << " elapsed_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " threads=" << threads << " cpu_user_s=" << std::setprecision(6) << (u1.user - u0.user) << " cpu_sys_s=" << (u1.sys - u0.sys) << " maxrss_kb=" << u1.maxrss_kb << '\n';
    }
    if (more) { std::cerr << "PACK more layers remain (exit 3)\n"; return kExitMoreToDo; }
    if (!output.empty()) {
        std::string lay = std::to_string(first) + "-" + std::to_string(last);
        std::vector<std::string> args = {"strata-ds41f-pack", "assemble", dir.string(), output, "--layers=" + lay};
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        return cmd_assemble(int(av.size()), av.data());
    }
    std::cerr << "PACK chunk directory complete: " << dir << '\n';
    return 0;
}
// ---- trunk ----------------------------------------------------------------------------------------------------
int cmd_trunk(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("usage: strata-ds41f-pack trunk MODEL_DIR OUT.gguf [--trunk=q4_k] [--head=q8_0] [--threads=1]");
    std::string trunk = "q4_k", head = "q8_0"; int threads = 1;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (starts_with(a, "--trunk=")) trunk = a.substr(8); else if (starts_with(a, "--head=")) head = a.substr(7);
        else if (starts_with(a, "--threads=")) threads = std::clamp(std::stoi(a.substr(10)), 1, 128); else throw std::runtime_error("unknown argument " + a);
    }
    const ggml_type trunk_type = type_of(trunk), head_type = type_of(head);
    if (ggml_quantize_requires_imatrix(trunk_type) || ggml_quantize_requires_imatrix(head_type)) throw std::runtime_error("trunk formats must not need an imatrix");
    const auto c = open_checkpoint(argv[2]);
    const fs::path output(argv[3]), partial = output.string() + ".partial";
    if (fs::exists(output) || fs::exists(partial)) throw std::runtime_error("refusing to overwrite existing file");
    struct Item { std::string name; const st::Tensor* w; const st::Tensor* scale; ggml_type type; };
    std::vector<Item> items;
    for (const auto& [name, t] : c.index) {
        if (strata_ds41f_is_routed_expert(name.c_str()) || strata_ds41f_is_engram_table(name.c_str()) || ends_with(name, ".scale")) continue;
        const bool matrix = t.shape.size() == 2 && t.shape[0] >= 256 && t.shape[1] >= 256;
        const st::Tensor* scale = c.find(name.substr(0, name.size() - 7) + ".scale");
        if (t.dtype == "F8_E4M3") { if (!scale) throw std::runtime_error("FP8 tensor without scale: " + name); items.push_back({name, &t, scale, matrix ? trunk_type : GGML_TYPE_F32}); }
        else if (t.dtype == "BF16") items.push_back({name, &t, nullptr, name == "head.weight" ? head_type : (matrix && name != "embed.weight" ? trunk_type : GGML_TYPE_BF16)});
        else if (t.dtype == "F32") items.push_back({name, &t, nullptr, GGML_TYPE_F32});
        else if (t.dtype == "I8" || t.dtype == "F8_E8M0") continue;   // vision tower or DSpark packed tensors: not part of the text trunk
        else throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.name < b.name; });
    std::vector<Entry> entries; uint64_t offset = 0;
    for (const auto& it : items) {
        std::vector<uint64_t> shape(it.w->shape.rbegin(), it.w->shape.rend());
        uint64_t bytes = 1; for (size_t i = 1; i < shape.size(); ++i) bytes *= shape[i];
        bytes *= uint64_t(ggml_row_size(it.type, int64_t(shape[0])));
        bytes = (bytes + 31) / 32 * 32;
        entries.push_back({it.name, shape, uint32_t(it.type), offset, bytes}); offset += bytes;
    }
    const auto fingerprint = source_fingerprint(c);
    std::map<std::string, std::string> kvs = {{"general.architecture", "deepseek4"}, {"strata.trunk.formats", trunk + "," + head}};
    std::map<std::string, uint64_t> kvu = {{"strata.trunk.source", fingerprint}};
    std::map<std::string, uint32_t> kv32 = {{"strata.trunk.version", 1}, {"deepseek4.block_count", uint32_t(kLayers)}, {"deepseek4.embedding_length", uint32_t(kHidden)},
                                            {"deepseek4.expert_count", uint32_t(kExperts)}, {"deepseek4.expert_used_count", 6}, {"deepseek4.expert_feed_forward_length", uint32_t(kFf)}};
    std::map<std::string, uint64_t> hashes; for (const auto& e : entries) hashes["strata.trunk.hash." + e.name] = 0;
    auto h = header(entries, kvs, kvu, kv32, false, hashes);
    std::cerr << "TRUNK plan tensors=" << entries.size() << " bytes=" << offset + h.size() << '\n';
    std::ofstream out(partial, std::ios::binary | std::ios::trunc); out.exceptions(std::ios::failbit | std::ios::badbit); out.write(h.data(), h.size());
    ggml_cpu_init();
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i]; auto& e = entries[i];
        const auto raw = c.read(*it.w);
        std::vector<uint8_t> payload(e.bytes, 0);
        const int64_t ne0 = int64_t(e.shape[0]); int64_t rows = 1; for (size_t k = 1; k < e.shape.size(); ++k) rows *= int64_t(e.shape[k]);
        std::vector<float> f32(size_t(rows) * ne0);
        if (it.w->dtype == "F8_E4M3") {
            const auto sb = c.read(*it.scale);
            const int srows = int(it.scale->shape.size() == 2 ? it.scale->shape[0] : 1), scols = int(it.scale->shape.size() == 2 ? it.scale->shape[1] : it.scale->shape[0]);
            if (it.w->shape.size() != 2 || int64_t(ne0) % scols) throw std::runtime_error("FP8 scale geometry mismatch: " + it.name);
            // row ranges start on 32-row scale blocks, so each range sees its own scale rows
            parallel_rows(rows, threads, 32, [&](int64_t r0, int64_t r1, int) {
                const int sr0 = std::min(int(r0 / 32), srows - 1);
                dequant_fp8(raw.data() + size_t(r0) * ne0, sb.data() + size_t(sr0) * scols, int(r1 - r0), int(ne0), srows - sr0, scols, f32.data() + size_t(r0) * ne0); });
        } else if (it.w->dtype == "BF16") parallel_rows(rows, threads, 1, [&](int64_t r0, int64_t r1, int) {
            for (size_t k = size_t(r0) * ne0; k < size_t(r1) * ne0; ++k) f32[k] = bf16(reinterpret_cast<const uint16_t*>(raw.data())[k]); });
        else std::memcpy(f32.data(), raw.data(), f32.size() * 4);
        for (float v : f32) if (!std::isfinite(v)) throw std::runtime_error("nonfinite value in " + it.name);
        size_t written = 0;
        if (it.type == GGML_TYPE_F32) { std::memcpy(payload.data(), f32.data(), f32.size() * 4); written = f32.size() * 4; }
        else {   // ggml_quantize_chunk writes rows [start/ne0, ...) at start/ne0 * row_size: byte-identical to one call
            std::mutex wmu;
            parallel_rows(rows, threads, 1, [&](int64_t r0, int64_t r1, int) {
                const size_t n = ggml_quantize_chunk(it.type, f32.data(), payload.data(), r0 * ne0, r1 - r0, ne0, nullptr);
                std::lock_guard lk(wmu); written += n; });
        }
        if (written > e.bytes) throw std::runtime_error("quantizer overflow: " + it.name);
        out.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size())); e.hash = hash_bytes(kFnvBasis, payload.data(), payload.size());
        if (ggml_is_quantized(it.type)) {
            const auto* traits = ggml_get_type_traits(it.type); const size_t row_bytes = ggml_row_size(it.type, ne0);
            std::vector<double> err(size_t(threads), 0.), nrm(size_t(threads), 0.);
            parallel_rows(rows, threads, 1, [&](int64_t r0, int64_t r1, int t) {
                std::vector<float> dec(ne0); double se = 0, sn = 0;
                for (int64_t r = r0; r < r1; ++r) { traits->to_float(payload.data() + size_t(r) * row_bytes, dec.data(), ne0);
                    for (int64_t k = 0; k < ne0; ++k) { const double w = f32[size_t(r) * ne0 + k], d = dec[k] - w; se += d * d; sn += w * w; } }
                err[size_t(t)] = se; nrm[size_t(t)] = sn; });
            for (int t = 0; t < threads; ++t) { e.error += err[size_t(t)]; e.norm += nrm[size_t(t)]; }
        }
        std::cerr << "TRUNK tensor=" << it.name << " type=" << ggml_type_name(it.type) << " bytes=" << e.bytes
                  << (ggml_is_quantized(it.type) ? " normalized_mse=" + std::to_string(e.error / std::max(e.norm, 1e-30)) : "") << " done=" << i + 1 << "/" << items.size() << '\n';
    }
    for (const auto& e : entries) hashes["strata.trunk.hash." + e.name] = e.hash;
    h = header(entries, kvs, kvu, kv32, false, hashes); out.seekp(0); out.write(h.data(), h.size()); out.close();
    fs::rename(partial, output);
    std::cerr << "TRUNK complete path=" << output << " bytes=" << offset + h.size() << '\n';
    return 0;
}

// ---- verify ----------------------------------------------------------------------------------------------------
int cmd_verify(int argc, char** argv) {
    if (argc != 3 && argc != 4) throw std::runtime_error("usage: strata-ds41f-pack verify [MODEL_DIR] PACK.gguf");
    GgufFile pack(argv[argc - 1]);
    const bool trunk = pack.get("strata.trunk.version") != nullptr;
    const std::string prefix = trunk ? "strata.trunk." : "strata.expert_pack.";
    const auto* version = pack.get(prefix + "version"); const auto* source = pack.get(prefix + "source");
    const auto* arch = pack.get(trunk ? "general.architecture" : "strata.expert_pack.architecture");
    if (!version || version->u != 1 || !source || !arch || arch->s != "deepseek4") throw std::runtime_error("pack: version or architecture mismatch");
    if (argc == 4) { const auto c = open_checkpoint(argv[2], false); if (source->u != source_fingerprint(c)) throw std::runtime_error("pack: source fingerprint mismatch"); }
    size_t checked = 0; const uint64_t payload = pack.file_size() - pack.data_start();
    for (const auto& t : pack.tensors()) {
        uint64_t bytes;
        if (t.type == kGgufI32) bytes = (t.shape[0] * 4 + 31) / 32 * 32;
        else { uint64_t rows = 1; for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i]; bytes = uint64_t(ggml_row_size(ggml_type(t.type), int64_t(t.shape[0]))) * rows; if (trunk) bytes = (bytes + 31) / 32 * 32; }
        if (!trunk) { int layer; strata_ds41f_proj proj; if (strata_ds41f_parse_pack_name(t.name.c_str(), &layer, &proj) != 1 && t.type != kGgufI32) throw std::runtime_error("pack: unexpected tensor " + t.name); }
        if (t.offset > payload || bytes > payload - t.offset) throw std::runtime_error("pack: invalid payload bounds " + t.name);
        const auto* expected = pack.get(prefix + "hash." + t.name);
        if (!expected || hash_bytes(kFnvBasis, pack.tensor_data(t), bytes) != expected->u) throw std::runtime_error("payload checksum mismatch: " + t.name);
        pack.discard_tensor_pages(t, bytes); ++checked;
    }
    std::cout << "PACK verified architecture=deepseek4 kind=" << (trunk ? "trunk" : "experts") << " tensors=" << checked << (argc == 4 ? " source=matched" : " source=unchecked") << '\n';
    return 0;
}

int cmd_dequant(int argc, char** argv) {
    if (argc != 7) throw std::runtime_error("usage: strata-ds41f-pack dequant MODEL_DIR LAYER EXPERT gate|up|down OUT.f32");
    const auto c = open_checkpoint(argv[2]);
    const std::string which = argv[5]; const int proj = which == "gate" ? 0 : which == "up" ? 1 : which == "down" ? 2 : -1;
    if (proj < 0) throw std::runtime_error("projection must be gate, up or down");
    const auto src = expert_source(c, std::stoi(argv[3]), std::stoi(argv[4]));
    std::vector<float> weights; dequant_expert(c.read(*src.w[proj]), c.read(*src.s[proj]), proj, weights, which.c_str());
    std::ofstream out(argv[6], std::ios::binary); out.write(reinterpret_cast<const char*>(weights.data()), std::streamsize(weights.size() * 4));
    std::cout << "DEQUANT rows=" << geometry(proj).rows << " cols=" << geometry(proj).cols << " values=" << weights.size() << '\n';
    return 0;
}
}  // namespace

int main(int argc, char** argv) { try {
    if (argc < 2) throw std::runtime_error("usage: strata-ds41f-pack experts|assemble|trunk|verify|dequant ...");
    const std::string cmd = argv[1];
    if (cmd == "experts") return cmd_experts(argc, argv);
    if (cmd == "assemble") return cmd_assemble(argc, argv);
    if (cmd == "trunk") return cmd_trunk(argc, argv);
    if (cmd == "verify" || cmd == "--verify") return cmd_verify(argc, argv);
    if (cmd == "dequant") return cmd_dequant(argc, argv);
    throw std::runtime_error("unknown command " + cmd);
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
