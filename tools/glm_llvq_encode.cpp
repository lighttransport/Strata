// strata-glm-llvq-encode: llvq Tetra (Leech lattice, 48-bit words per 24 weights, 2.06 bpw) for every routed expert
// of GLM-5.3-Flash, from the FP8 release. Per matrix (docs/GLM_LLVQ_ENCODE.md):
//   FP8 E4M3 x 128x128 scale -> f32 -> f64 rows -> llvq Rotation (seed) -> two gain centroids (Lloyd) -> row
//   scales -> every full 24-weight block: Tetra word (encoder bit-identical to llvq's f64 encoder) + gain bit ->
//   the 16- or 8-column tail stays exact as BF16.
// The Hessian is the identity: llvq's GPTQ loop then propagates nothing and the matrix quantizes block by block, so
// one matrix is one batch for the CPU or GPU encoder.
//
// Work is split into units (one layer x a group of experts) claimed through lock files in --out, so several
// processes share one run: a CPU process and one GPU process per card (a CUDA and a HIP build cannot share a
// process). A unit is written to a temporary file and renamed when complete; rerunning resumes.
//
//   strata-glm-llvq-encode --model DIR --out DIR [--device cpu|gpu] [--threads N] [--precision f64|f32]
//       [--layers A-B] [--group N] [--seed S] [--iters N] [--units N] [--redo-f32]
//   Resumable: complete units are skipped, claims from another boot or a dead process are taken over, SIGTERM stops
//   at the next matrix boundary. --redo-f32 re-encodes units written by the f32 GPU fallback.
//   strata-glm-llvq-encode --matrix ROWS COLS in.f32 out.bin [--seed S] [--iters N] [--device ..] [--threads N]
//       one f32 matrix through the same pipeline, written as llvq_dump's matrix mode writes it (cross-check)
//   strata-glm-llvq-encode --model DIR --dump-f32 LAYER EXPERT gate|up|down out.f32
//   strata-glm-llvq-encode --model DIR --check UNIT.llvq
//       decode every matrix of a unit and print its relative error against the rotated FP8 weights
#include "strata/artifact/fp8_experts.hpp"
#include "strata/llvq/quantize.hpp"
#include "strata/llvq/tetra_decode.hpp"
#include "strata/llvq/tetra_encoder.hpp"
#include "strata/llvq/tetra_view.hpp"
#ifdef STRATA_LLVQ_GPU
#include "strata/kernels/llvq_tetra.hpp"
#include <cuda_runtime.h>
#endif
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace strata;

namespace {

constexpr int FIRST_MOE_LAYER = 3;  // GLM-5.3-Flash: layers 0-2 are dense, 3-44 routed (42 layers)
constexpr int LAYERS = 45, EXPERTS = 288;
const char* const PROJ[3] = {"gate_proj", "up_proj", "down_proj"};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- FP8 safetensors (strata/artifact/fp8_experts.hpp) --------------------------------------------------------
using Tensor = artifact::fp8::Tensor;
std::map<std::string, Tensor> scan(const fs::path& dir) { return artifact::fp8::scan(dir); }
std::string tensor_name(int layer, int expert, int proj) { return artifact::fp8::expert_name(layer, expert, PROJ[proj]); }
std::vector<float> load_fp8(const std::map<std::string, Tensor>& ts, int layer, int expert, int proj, size_t& rows, size_t& cols) {
    return artifact::fp8::load_expert(ts, layer, expert, PROJ[proj], rows, cols);
}

uint16_t bf16(double v) {
    float f = float(v);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

void parallel_for(size_t n, int threads, const std::function<void(size_t, size_t)>& fn) {
    threads = std::max(1, std::min<int>(threads, int(n)));
    std::vector<std::thread> pool;
    size_t per = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        size_t lo = t * per, hi = std::min(n, lo + per);
        if (lo < hi) pool.emplace_back(fn, lo, hi);
    }
    for (auto& th : pool) th.join();
}

// ---- encoders -----------------------------------------------------------------------------------------------
struct Encoders {
    llvq::Tetra tetra;
    llvq::Encoder cpu{tetra};
    int threads = 1;
    bool gpu = false, f32 = false;
#ifdef STRATA_LLVQ_GPU
    kernels::LlvqTetraGpu* g = nullptr;
    double* dx = nullptr;
    float* dxf = nullptr;
    uint64_t* dw = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;
#endif
    void init_gpu() {
#ifdef STRATA_LLVQ_GPU
        llvq::core::TetraTablesView view;
        std::vector<int16_t> pos5;
        llvq::make_view(tetra, view, pos5);
        g = kernels::llvq_tetra_gpu_create(&view, pos5.data());
        if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("stream");
        gpu = true;
#else
        throw std::runtime_error("this build has no GPU encoder");
#endif
    }
    // x: n blocks x 24 (contiguous), words out (gain bit 0).
    void encode(const double* x, size_t n, uint64_t* words) {
        if (!gpu) {
            std::atomic<size_t> next{0};
            parallel_for(size_t(threads), threads, [&](size_t, size_t) {
                llvq::Scratch sc;
                for (size_t c; (c = next.fetch_add(256)) < n;)
                    for (size_t i = c; i < std::min(n, c + 256); ++i) words[i] = cpu.encode(x + 24 * i, sc).word;
            });
            return;
        }
#ifdef STRATA_LLVQ_GPU
        auto ck = [](cudaError_t e, const char* w) { if (e != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e)); };
        if (n > cap) {
            cudaFree(dx); cudaFree(dxf); cudaFree(dw);
            dx = nullptr; dxf = nullptr;
            if (f32) ck(cudaMalloc((void**)&dxf, n * 24 * 4), "alloc x");
            else ck(cudaMalloc((void**)&dx, n * 24 * 8), "alloc x");
            ck(cudaMalloc((void**)&dw, n * 8), "alloc words");
            cap = n;
        }
        if (f32) {
            std::vector<float> xf(x, x + n * 24);
            ck(cudaMemcpyAsync(dxf, xf.data(), n * 24 * 4, cudaMemcpyHostToDevice, stream), "copy x");
            kernels::llvq_tetra_encode_f32(g, dxf, int64_t(n), dw, nullptr, stream);
            ck(cudaStreamSynchronize(stream), "encode");
        } else {
            ck(cudaMemcpyAsync(dx, x, n * 24 * 8, cudaMemcpyHostToDevice, stream), "copy x");
            kernels::llvq_tetra_encode_f64(g, dx, int64_t(n), dw, nullptr, stream);
        }
        ck(cudaMemcpyAsync(words, dw, n * 8, cudaMemcpyDeviceToHost, stream), "copy words");
        ck(cudaStreamSynchronize(stream), "encode");
#endif
    }
};

// ---- one matrix -------------------------------------------------------------------------------------------
struct Prepared {
    size_t rows = 0, cols = 0, nb = 0, tail = 0;
    std::vector<double> centroids, scales, blocks;  // blocks: rows x nb x 24
    std::vector<uint16_t> tail_bf16;                // rows x tail
};

Prepared prepare(const float* w32, size_t rows, size_t cols, const llvq::Rotation& rot, int iters, int threads) {
    Prepared p;
    p.rows = rows; p.cols = cols; p.nb = cols / 24; p.tail = cols - 24 * p.nb;
    std::vector<double> w(rows * cols);
    parallel_for(rows, threads, [&](size_t lo, size_t hi) {
        for (size_t r = lo; r < hi; ++r) {
            double* row = &w[r * cols];
            for (size_t c = 0; c < cols; ++c) row[c] = double(w32[r * cols + c]);
            rot.apply(row);
        }
    });
    p.centroids = llvq::fit_gain_centroids(w.data(), rows, cols, 1, iters);
    if (p.centroids.size() != 2) p.centroids.resize(2, p.centroids[0]);  // an all-zero matrix: [1, 1]
    p.scales.resize(rows);
    p.blocks.resize(rows * p.nb * 24);
    p.tail_bf16.resize(rows * p.tail);
    parallel_for(rows, threads, [&](size_t lo, size_t hi) {
        for (size_t r = lo; r < hi; ++r) {
            const double* row = &w[r * cols];
            p.scales[r] = llvq::row_scale(row, cols);
            std::memcpy(&p.blocks[r * p.nb * 24], row, p.nb * 24 * 8);
            for (size_t t = 0; t < p.tail; ++t) p.tail_bf16[r * p.tail + t] = bf16(row[24 * p.nb + t]);
        }
    });
    return p;
}

// Words with the gain bit, as TetraShapeGain::quantize chooses: a zero block is word 0, gain 0.
void finish(const Prepared& p, std::vector<uint64_t>& words) {
    for (size_t r = 0; r < p.rows; ++r) {
        double rs = p.scales[r] > 0.0 ? p.scales[r] : 1.0;
        for (size_t b = 0; b < p.nb; ++b) {
            size_t i = r * p.nb + b;
            double nn = llvq::block_norm(&p.blocks[24 * i]);
            if (nn == 0.0) { words[i] = 0; continue; }
            uint64_t gain = llvq::nearest_level_index(p.centroids, nn / rs);
            words[i] = (words[i] & llvq::LABEL_MASK) | gain << 47;
        }
    }
}

// Matrix record: u32 rows, u32 cols, f64 centroids[2], f32 row_scale[rows], words (6 bytes LE, row stride padded to
// a multiple of 8 bytes), BF16 tail [rows][cols % 24].
void write_matrix(std::vector<uint8_t>& out, const Prepared& p, const std::vector<uint64_t>& words) {
    auto put = [&](const void* d, size_t n) { const auto* b = static_cast<const uint8_t*>(d); out.insert(out.end(), b, b + n); };
    uint32_t r32 = uint32_t(p.rows), c32 = uint32_t(p.cols);
    put(&r32, 4); put(&c32, 4);
    put(p.centroids.data(), 16);
    for (double s : p.scales) { float f = float(s); put(&f, 4); }
    const size_t stride = (p.nb * 6 + 7) / 8 * 8;
    std::vector<uint8_t> row(stride);
    for (size_t r = 0; r < p.rows; ++r) {
        std::fill(row.begin(), row.end(), 0);
        for (size_t b = 0; b < p.nb; ++b) std::memcpy(&row[6 * b], &words[r * p.nb + b], 6);
        put(row.data(), stride);
    }
    put(p.tail_bf16.data(), p.tail_bf16.size() * 2);
}

uint64_t fnv1a(const uint8_t* d, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= d[i]; h *= 0x100000001b3ull; }
    return h;
}

// ---- claims and stopping ------------------------------------------------------------------------------------
// A claim holds "pid boot_id". It is live only if it was made in this boot and that pid is still an llvq encoder:
// after a reboot every claim is stale, and a reused pid of another program does not keep a unit claimed.
// Claims of the first format (pid only) count as live only while that pid runs an llvq encoder.
std::string boot_id() {
    std::ifstream f("/proc/sys/kernel/random/boot_id");
    std::string id;
    f >> id;
    return id;
}
bool encoder_alive(long pid) {
    if (pid <= 0 || (::kill(pid_t(pid), 0) != 0 && errno == ESRCH)) return false;
    std::ifstream f("/proc/" + std::to_string(pid) + "/cmdline");
    std::string cmd((std::istreambuf_iterator<char>(f)), {});
    return cmd.find("llvq-encode") != std::string::npos;
}
bool claim(const fs::path& path) {
    static const std::string boot = boot_id();
    for (int attempt = 0; attempt < 2; ++attempt) {
        int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd >= 0) {
            std::string text = std::to_string(::getpid()) + " " + boot + "\n";
            (void)!::write(fd, text.data(), text.size());
            ::close(fd);
            return true;
        }
        std::ifstream f(path);
        long pid = 0;
        std::string their_boot;
        f >> pid >> their_boot;
        if ((their_boot.empty() || their_boot == boot) && encoder_alive(pid)) return false;  // live owner
        fs::remove(path);  // stale: another boot, or a process that is gone
        std::error_code ec;
        fs::path tmp = path;
        tmp.replace_extension(".llvq.tmp");
        fs::remove(tmp, ec);
    }
    return false;
}
// SIGTERM / SIGINT: the unit in progress is abandoned at the next matrix boundary (its claim and temporary file are
// removed) and the process exits; a later run encodes that unit again.
std::atomic<bool> stop_requested{false};
extern "C" void on_stop(int) { stop_requested.store(true); }

// Precision field of an existing unit (header word 5): 0 f64, 1 f32, -1 unreadable.
int unit_precision(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    char head[40];
    if (!f.read(head, 40) || std::memcmp(head, "LLVQTET1", 8)) return -1;
    uint32_t prec;
    std::memcpy(&prec, head + 28, 4);
    return int(prec);
}

struct Args {
    std::string model, out, device = "cpu", precision = "f64";
    int threads = int(std::thread::hardware_concurrency()), layer_lo = FIRST_MOE_LAYER, layer_hi = LAYERS - 1;
    int group = 16, iters = 40, units = -1;
    bool redo_f32 = false;  // --redo-f32: encode f32 units again (with --precision f64: the bit-exact words)
    uint64_t seed = 0x11a7e5eedull;
};

int run(const Args& a) {
    Encoders enc;
    enc.threads = a.threads;
    enc.f32 = a.precision == "f32";
    if (a.device == "gpu") enc.init_gpu();
    else if (enc.f32) throw std::runtime_error("--precision f32 is a GPU mode");
    fs::create_directories(a.out);
    auto ts = scan(a.model);
    std::fprintf(stderr, "llvq-encode: %zu tensors under %s, device %s (%s), %d threads\n", ts.size(), a.model.c_str(),
                 a.device.c_str(), a.precision.c_str(), a.threads);
    std::map<size_t, std::unique_ptr<llvq::Rotation>> rots;
    auto rot_for = [&](size_t n) -> const llvq::Rotation& {
        auto& r = rots[n];
        if (!r) r = std::make_unique<llvq::Rotation>(n, a.seed);
        return *r;
    };
    int done_units = 0;
    double blocks_total = 0, t_start = now();
    for (int layer = a.layer_lo; layer <= a.layer_hi; ++layer)
        for (int e0 = 0; e0 < EXPERTS; e0 += a.group) {
            if (a.units >= 0 && done_units >= a.units) return 0;
            const int ne = std::min(a.group, EXPERTS - e0);
            char stem[64];
            std::snprintf(stem, sizeof stem, "L%02d.e%03d", layer, e0);
            fs::path file = fs::path(a.out) / (std::string(stem) + ".llvq"), lock = fs::path(a.out) / (std::string(stem) + ".claim");
            if (fs::exists(file) && !(a.redo_f32 && !enc.f32 && unit_precision(file) == 1)) continue;
            bool present = true;
            for (int e = e0; e < e0 + ne && present; ++e)
                for (int pr = 0; pr < 3; ++pr) present = present && ts.count(tensor_name(layer, e, pr)) && ts.count(tensor_name(layer, e, pr) + "_scale_inv");
            if (!present || !claim(lock)) continue;
            const double t0 = now();
            std::vector<uint8_t> out;
            auto put = [&](const void* d, size_t n) { const auto* b = static_cast<const uint8_t*>(d); out.insert(out.end(), b, b + n); };
            put("LLVQTET1", 8);
            uint32_t hdr[6] = {1, uint32_t(layer), uint32_t(e0), uint32_t(ne), uint32_t(a.iters), enc.f32 ? 1u : 0u};
            put(hdr, sizeof hdr);
            put(&a.seed, 8);
            // Prepare the next matrix on the host while the current one encodes.
            auto job = [&](int k) {
                size_t rows, cols;
                auto w32 = load_fp8(ts, layer, e0 + k / 3, k % 3, rows, cols);
                return prepare(w32.data(), rows, cols, rot_for(cols), a.iters, enc.gpu ? 4 : a.threads);
            };
            std::future<Prepared> next = std::async(std::launch::async, job, 0);
            size_t blocks = 0;
            bool abandoned = false;
            for (int k = 0; k < 3 * ne; ++k) {
                if (stop_requested.load()) { abandoned = true; break; }
                Prepared p = next.get();
                if (k + 1 < 3 * ne) next = std::async(std::launch::async, job, k + 1);
                std::vector<uint64_t> words(p.rows * p.nb);
                enc.encode(p.blocks.data(), words.size(), words.data());
                finish(p, words);
                write_matrix(out, p, words);
                blocks += words.size();
            }
            if (abandoned) {
                if (next.valid()) next.wait();
                fs::remove(lock);
                std::fprintf(stderr, "%s: stopped by signal; the unit is left for a later run\n", stem);
                return 0;
            }
            fs::path tmp = file;
            tmp += ".tmp";
            {
                std::ofstream f(tmp, std::ios::binary);
                f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
                if (!f) throw std::runtime_error("write " + tmp.string());
            }
            const double dt = now() - t0;
            {
                std::ofstream j(fs::path(a.out) / (std::string(stem) + ".json"));
                j << "{\"layer\":" << layer << ",\"first_expert\":" << e0 << ",\"experts\":" << ne << ",\"blocks\":" << blocks
                  << ",\"bytes\":" << out.size() << ",\"fnv1a64\":\"" << std::hex << fnv1a(out.data(), out.size()) << std::dec
                  << "\",\"device\":\"" << a.device << "\",\"precision\":\"" << a.precision << "\",\"seed\":" << a.seed
                  << ",\"iters\":" << a.iters << ",\"seconds\":" << dt << "}\n";
            }
            fs::rename(tmp, file);  // atomic: a unit file is either absent or complete
            fs::remove(lock);
            if (stop_requested.load()) {
                std::fprintf(stderr, "%s done; stopped by signal\n", stem);
                return 0;
            }
            ++done_units;
            blocks_total += double(blocks);
            std::fprintf(stderr, "%s: %zu blocks in %.1f s (%.0f blocks/s); this process %.0f blocks/s overall\n", stem, blocks, dt,
                         blocks / dt, blocks_total / (now() - t_start));
        }
    return 0;
}

int matrix_mode(const Args& a, size_t rows, size_t cols, const std::string& in, const std::string& outp) {
    std::vector<float> w(rows * cols);
    {
        std::ifstream f(in, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(w.data()), std::streamsize(w.size() * 4))) throw std::runtime_error("read " + in);
    }
    Encoders enc;
    enc.threads = a.threads;
    enc.f32 = a.precision == "f32";
    if (a.device == "gpu") enc.init_gpu();
    llvq::Rotation rot(cols, a.seed);
    double t0 = now();
    Prepared p = prepare(w.data(), rows, cols, rot, a.iters, a.threads);
    double t1 = now();
    std::vector<uint64_t> words(rows * p.nb);
    enc.encode(p.blocks.data(), words.size(), words.data());
    double t2 = now();
    finish(p, words);
    std::ofstream f(outp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p.centroids.data()), 16);
    f.write(reinterpret_cast<const char*>(p.scales.data()), std::streamsize(rows * 8));
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 8));
    std::fprintf(stderr, "matrix %zux%zu: prepare %.2f s, encode %.2f s (%.0f blocks/s)\n", rows, cols, t1 - t0, t2 - t1,
                 words.size() / (t2 - t1));
    return 0;
}

int check_unit(const Args& a, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
    if (bytes.size() < 40 || std::memcmp(bytes.data(), "LLVQTET1", 8)) throw std::runtime_error("not a unit file: " + path);
    uint32_t hdr[6];
    uint64_t seed;
    std::memcpy(hdr, bytes.data() + 8, sizeof hdr);
    std::memcpy(&seed, bytes.data() + 32, 8);
    const int layer = int(hdr[1]), e0 = int(hdr[2]), ne = int(hdr[3]);
    auto ts = scan(a.model);
    llvq::Tetra tetra;
    const auto T = llvq::DecodeTables::build(tetra);
    const uint8_t* p = bytes.data() + 40;
    double worst = 0, sum = 0;
    int n = 0;
    for (int k = 0; k < 3 * ne; ++k) {
        llvq::TetraMatrix m;
        p = m.parse(p);
        size_t rows, cols;
        auto w32 = load_fp8(ts, layer, e0 + k / 3, k % 3, rows, cols);
        if (rows != m.rows || cols != m.cols) throw std::runtime_error("matrix shape differs from the FP8 release");
        llvq::Rotation rot(cols, seed);
        std::vector<double> row(cols);
        std::vector<float> q(cols);
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m.rows; ++r) {
            for (size_t c = 0; c < cols; ++c) row[c] = w32[r * cols + c];
            rot.apply(row.data());
            llvq::dequant_row(T, m, r, q.data());
            for (size_t c = 0; c < cols; ++c) { double d = q[c] - row[c]; num += d * d; den += row[c] * row[c]; }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        worst = std::max(worst, rel);
        sum += rel;
        ++n;
        if (k < 3) std::printf("L%d e%d %s: %ux%u relative error %.4f (normalized MSE %.4f)\n", layer, e0 + k / 3, PROJ[k % 3], m.rows, m.cols, rel, rel * rel);
    }
    if (size_t(p - bytes.data()) != bytes.size()) throw std::runtime_error("trailing bytes after the last matrix");
    std::printf("%s: %d matrices, relative error mean %.4f (normalized MSE %.4f), worst %.4f (seed %llu, precision %s)\n", path.c_str(), n, sum / n, (sum / n) * (sum / n), worst,
                (unsigned long long)seed, hdr[5] ? "f32" : "f64");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a;
        std::vector<std::string> pos;
        std::string dump[4], matrix[4], check;
        bool want_dump = false, want_matrix = false;
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
                return argv[++i];
            };
            if (s == "--model") a.model = val();
            else if (s == "--out") a.out = val();
            else if (s == "--device") a.device = val();
            else if (s == "--precision") a.precision = val();
            else if (s == "--threads") a.threads = std::stoi(val());
            else if (s == "--group") a.group = std::stoi(val());
            else if (s == "--iters") a.iters = std::stoi(val());
            else if (s == "--units") a.units = std::stoi(val());
            else if (s == "--redo-f32") a.redo_f32 = true;
            else if (s == "--seed") a.seed = std::stoull(val(), nullptr, 0);
            else if (s == "--layers") {
                std::string v = val();
                auto d = v.find('-');
                a.layer_lo = std::stoi(v.substr(0, d));
                a.layer_hi = d == std::string::npos ? a.layer_lo : std::stoi(v.substr(d + 1));
            } else if (s == "--dump-f32") { want_dump = true; for (auto& d : dump) d = val(); }
            else if (s == "--matrix") { want_matrix = true; for (auto& m : matrix) m = val(); }
            else if (s == "--check") check = val();
            else throw std::runtime_error("unknown option " + s);
        }
        if (a.device != "cpu" && a.device != "gpu") throw std::runtime_error("--device cpu|gpu");
        if (a.precision != "f64" && a.precision != "f32") throw std::runtime_error("--precision f64|f32");
        if (want_matrix) return matrix_mode(a, std::stoul(matrix[0]), std::stoul(matrix[1]), matrix[2], matrix[3]);
        if (a.model.empty()) throw std::runtime_error("--model DIR is required");
        if (!check.empty()) return check_unit(a, check);
        if (want_dump) {
            auto ts = scan(a.model);
            int proj = dump[2] == "gate" ? 0 : dump[2] == "up" ? 1 : dump[2] == "down" ? 2 : -1;
            if (proj < 0) throw std::runtime_error("projection: gate|up|down");
            size_t rows, cols;
            auto w = load_fp8(ts, std::stoi(dump[0]), std::stoi(dump[1]), proj, rows, cols);
            std::ofstream f(dump[3], std::ios::binary);
            f.write(reinterpret_cast<const char*>(w.data()), std::streamsize(w.size() * 4));
            std::printf("%zu %zu\n", rows, cols);
            return 0;
        }
        if (a.out.empty()) throw std::runtime_error("--out DIR is required");
        std::signal(SIGTERM, on_stop);
        std::signal(SIGINT, on_stop);
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm-llvq-encode: %s\n", e.what());
        return 1;
    }
}
