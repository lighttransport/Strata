// ripple_runner: Ripple-65B Preview (Ricursive dense transformer, RicursiveForCausalLM) text decode on CPU ranks, one rank per
// node (A64FX; MPI with -DRIPPLE_MPI, uTofu collectives with -DRIPPLE_UTOFU and RIPPLE_COMM=utofu).
// 80 layers, hidden 5120, FFN 40960, 80 q heads / 8 kv heads x 128. Per layer (semantics: the checkpoint's modeling_ricursive.py):
//   h += post_attn_norm(attn(input_norm(h)));  h += post_ffn_norm(mlp(pre_ffn_norm(h)))
//   attn: q, k per-head RMSNorm; gate = sigmoid(g_proj(x)) on the attention output; v = v_proj(lerp(emb0, x, sigmoid(lambda)))
//         where emb0 is the scaled token embedding (the "value residual"); sliding layers (window 1024) use RoPE (theta 1e4,
//         half-split), every fifth layer is global attention with no RoPE.
//   mlp: down(silu(min(gate, 20)) * clamp(up, -20, 20)).
// Checkpoints: BF16 (~/models/ripple) or ModelOpt NVFP4 MLPs with BF16 attention (~/models/ripple-fp4), detected from the tensors.
//
// Layout over W ranks (R), exact tensor-parallel with two all-reduces per layer:
//   q / gate projections: rows of this rank's heads (80 over W); o_proj: the columns of those heads (partial sums, all-reduce)
//   k / v projections and the KV cache: replicated on every rank (8 x 128 each; small)
//   gate / up: rows [i0, i0 + in) of the FFN width; down: the matching columns (partial sums, all-reduce)
//   lm_head: rows [v0, v0 + vn) of the vocabulary; the token embedding row is read on demand.
#include "strata/kernels/cpu/ds41_sve.h"
#include "strata/kernels/cpu/ripple_sve.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(RIPPLE_MPI)
#include <mpi.h>
#endif
#if defined(RIPPLE_UTOFU)   // the DS41 runner's uTofu collectives (src/cpu_runner/comm_utofu.c), selected by RIPPLE_COMM=utofu
extern "C" {
int cpu_utofu_init(int rank, int world, int max_count);
int cpu_utofu_max(void);
void cpu_utofu_allreduce(float *x, int n);
int cpu_utofu_xc_init(int shard_cap);
int cpu_utofu_xc_cap(void);
int cpu_utofu_allgather(float *x, const int *off, int canon);
}
#endif

namespace {

using Vec = std::vector<float>;
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int split0(int n, int r, int w) { return int(int64_t(n) * r / w); }
constexpr int RU = 32;   // row-split unit of the FFN width (a multiple of the NVFP4 block of 16)

double rss_bytes() {
    FILE *f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long a = 0, b = 0;
    const int n = std::fscanf(f, "%ld %ld", &a, &b);
    std::fclose(f);
    return n == 2 ? double(b) * double(sysconf(_SC_PAGESIZE)) : 0.0;
}

// ---------------------------------------------------------------- comm
struct Comm {
    int rank = 0, world = 1;
    double t = 0;
    bool ut = false;
    int ut_max = 0, ut_cap = 0;
    std::vector<int> off;
    void init(int *argc, char ***argv) {
#if defined(RIPPLE_MPI)
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
#else
        (void)argc; (void)argv;
#endif
#if defined(RIPPLE_UTOFU)
        const char *e = std::getenv("RIPPLE_COMM");
        if (e && std::strcmp(e, "utofu") == 0 && world > 1) {
            if (cpu_utofu_init(rank, world, 8192) != 0 || cpu_utofu_xc_init(4096) != 0) throw std::runtime_error("ripple: uTofu init failed");
            ut = true; ut_max = cpu_utofu_max(); ut_cap = cpu_utofu_xc_cap();
        }
#endif
    }
    void fini() {
#if defined(RIPPLE_MPI)
        MPI_Finalize();
#endif
    }
    template <class F> void timed(F f) { const double t0 = now(); f(); t += now() - t0; }
    void sum(float *x, int n) {
#if defined(RIPPLE_UTOFU)
        if (ut && n <= ut_max) { timed([&] { cpu_utofu_allreduce(x, n); }); return; }
#endif
#if defined(RIPPLE_MPI)
        if (world > 1) timed([&] { MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); });
#else
        (void)x; (void)n;
#endif
    }
    void dsum(double *x, int n) {
#if defined(RIPPLE_MPI)
        if (world > 1) timed([&] { MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); });
#else
        (void)x; (void)n;
#endif
    }
    void max(float *x, int n) {
#if defined(RIPPLE_MPI)
        if (world > 1) timed([&] { MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD); });
#else
        (void)x; (void)n;
#endif
    }
    // rank r's counts[r] floats at all + displs[r]
    void gatherv(const float *mine, int n, float *all, const std::vector<int> &counts, const std::vector<int> &displs) {
#if defined(RIPPLE_UTOFU)
        if (ut) {
            bool ok = true;
            off.assign(size_t(world) + 1, 0);
            for (int r = 0; r < world; ++r) {
                if (displs[size_t(r)] != off[size_t(r)] || counts[size_t(r)] > ut_cap) { ok = false; break; }
                off[size_t(r) + 1] = off[size_t(r)] + counts[size_t(r)];
            }
            if (ok) {
                if (n) std::memmove(all + displs[size_t(rank)], mine, size_t(n) * sizeof(float));
                timed([&] { if (cpu_utofu_allgather(all, off.data(), 0)) throw std::runtime_error("ripple: uTofu allgather"); });
                return;
            }
        }
#endif
#if defined(RIPPLE_MPI)
        if (world > 1) { timed([&] { MPI_Allgatherv(mine, n, MPI_FLOAT, all, counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD); }); return; }
#endif
        (void)counts;
        std::memcpy(all + displs[0], mine, size_t(n) * sizeof(float));
    }
    void argmax(float &v, int &i) {
#if defined(RIPPLE_MPI)
        if (world > 1) {
            struct { float v; int i; } in{v, i}, out{};
            MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            v = out.v; i = out.i;
        }
#else
        (void)v; (void)i;
#endif
    }
    void barrier() {
#if defined(RIPPLE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
};
Comm comm;

// ---------------------------------------------------------------- tiny JSON (safetensors headers, config)
struct J {
    enum T { NUL, NUM, STR, ARR, OBJ, BOOL } t = NUL;
    double n = 0; std::string s; std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J *get(const std::string &k) const { for (auto &p : o) if (p.first == k) return &p.second; return nullptr; }
    const J &at(const std::string &k) const { const J *j = get(k); if (!j) throw std::runtime_error("ripple: json key " + k); return *j; }
};
struct JP {
    const char *p, *e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    std::string str() {
        std::string r; ++p;
        while (p < e && *p != '"') {
            if (*p == '\\') { ++p; if (*p == 'u') { p += 4; r += '?'; } else r += *p; ++p; }
            else r += *p++;
        }
        ++p; return r;
    }
    J val() {
        ws(); J j;
        if (*p == '{') { j.t = J::OBJ; ++p; ws(); if (*p == '}') { ++p; return j; }
            for (;;) { ws(); std::string k = str(); ws(); ++p; j.o.emplace_back(k, val()); ws(); if (*p == ',') { ++p; continue; } ++p; break; } }
        else if (*p == '[') { j.t = J::ARR; ++p; ws(); if (*p == ']') { ++p; return j; }
            for (;;) { j.a.push_back(val()); ws(); if (*p == ',') { ++p; continue; } ++p; break; } }
        else if (*p == '"') { j.t = J::STR; j.s = str(); }
        else if (*p == 't') { j.t = J::BOOL; j.n = 1; p += 4; }
        else if (*p == 'f') { j.t = J::BOOL; p += 5; }
        else if (*p == 'n') { p += 4; }
        else { j.t = J::NUM; char *q; j.n = std::strtod(p, &q); p = q; }
        return j;
    }
};
J parse_json(const std::string &s) { JP jp{s.data(), s.data() + s.size()}; return jp.val(); }
std::string slurp(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ripple: cannot open " + path);
    std::string s; char buf[1 << 16]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f); return s;
}

// ---------------------------------------------------------------- numerics
inline float bf16_to_f32(uint16_t b) { const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float inv = float(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// ---------------------------------------------------------------- weight source: sharded safetensors
double g_read_s = 0; size_t g_read_b = 0;
struct StSrc {
    struct T { int fd = -1; std::string dtype; std::vector<int64_t> shape; int64_t off = 0; };
    std::map<std::string, T> t;
    std::vector<int> fds;
    // node-local image: every block / fetch result recorded to one file per rank, replayed in the same call order later
    FILE *img = nullptr; bool replay = false; std::string img_path;
    void image(const std::string &dir, int rank, int world, int layers, bool fp4) {
        img_path = dir + "/ripple-r" + std::to_string(rank) + "of" + std::to_string(world) + "-L" + std::to_string(layers) + (fp4 ? "-fp4" : "-bf16") + "-v1.img";
        FILE *ok = std::fopen((img_path + ".ok").c_str(), "r");
        replay = ok != nullptr;
        if (ok) std::fclose(ok);
        img = std::fopen(img_path.c_str(), replay ? "rb" : "wb");
        if (!img) throw std::runtime_error("ripple: cannot open image " + img_path);
        std::setvbuf(img, nullptr, _IOFBF, size_t(8) << 20);
    }
    void img_io(void *p, size_t n) const {
        const double t0 = now();
        if (replay) { if (std::fread(p, 1, n, img) != n) throw std::runtime_error("ripple: short image " + img_path); g_read_s += now() - t0; g_read_b += n; }
        else if (std::fwrite(p, 1, n, img) != n) throw std::runtime_error("ripple: image write " + img_path);
    }
    void image_finish() {
        if (!img || replay) return;
        std::fclose(img); img = nullptr;
        FILE *ok = std::fopen((img_path + ".ok").c_str(), "w"); if (ok) std::fclose(ok);
    }
    explicit StSrc(const std::string &dir) {
        const J idx = parse_json(slurp(dir + "/model.safetensors.index.json"));
        std::map<std::string, bool> files;
        for (auto &p : idx.at("weight_map").o) files[p.second.s] = true;
        for (auto &f : files) {
            const std::string path = dir + "/" + f.first;
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("ripple: cannot open " + path);
            fds.push_back(fd);
            uint64_t hl = 0;
            if (::pread(fd, &hl, 8, 0) != 8) throw std::runtime_error("ripple: short header " + path);
            std::string h(hl, '\0');
            if (::pread(fd, &h[0], hl, 8) != ssize_t(hl)) throw std::runtime_error("ripple: short header " + path);
            const J hj = parse_json(h);
            for (auto &p : hj.o) {
                if (p.first == "__metadata__") continue;
                T x; x.fd = fd; x.dtype = p.second.at("dtype").s;
                for (auto &d : p.second.at("shape").a) x.shape.push_back(int64_t(d.n));
                x.off = int64_t(8 + hl) + int64_t(p.second.at("data_offsets").a[0].n);
                t[p.first] = x;
            }
        }
    }
    ~StSrc() { for (int fd : fds) ::close(fd); }
    bool has(const std::string &n) const { return t.count(n) != 0; }
    const T &at(const std::string &n) const { auto it = t.find(n); if (it == t.end()) throw std::runtime_error("ripple: missing tensor " + n); return it->second; }
    // one contiguous range; a big one outside a parallel region is read by the whole team in 16 MiB pieces
    void read1(const T &x, int64_t off, int64_t n, void *dst) const {
        int64_t done = 0;
        while (done < n) {
            const ssize_t r = ::pread(x.fd, static_cast<char *>(dst) + done, size_t(std::min<int64_t>(n - done, int64_t(256) << 20)), x.off + off + done);
            if (r <= 0) throw std::runtime_error("ripple: read failed");
            done += r;
        }
    }
    void read(const T &x, int64_t off, int64_t n, void *dst) const {
        const double t0 = now();
        constexpr int64_t PIECE = int64_t(16) << 20;
        bool par = n >= 4 * PIECE;
#if defined(_OPENMP)
        par = par && !omp_in_parallel();
#else
        par = false;
#endif
        if (par) {
            const int64_t np = (n + PIECE - 1) / PIECE;
            bool bad = false;
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t i = 0; i < np; ++i) {
                try { read1(x, off + i * PIECE, std::min(PIECE, n - i * PIECE), static_cast<char *>(dst) + i * PIECE); }
                catch (...) { bad = true; }
            }
            if (bad) throw std::runtime_error("ripple: read failed");
        } else {
            read1(x, off, n, dst);
        }
#pragma omp critical(ripple_read_stats)
        { g_read_b += size_t(n); g_read_s += now() - t0; }
    }
    // a small standalone read (norm weights, scalars), through the image
    void fetch(const T &x, int64_t off, int64_t n, void *dst) const {
        if (replay) { img_io(dst, size_t(n)); return; }
        read(x, off, n, dst);
        if (img) img_io(dst, size_t(n));
    }
    // rows [r0, r0 + nr) of a 2D tensor of `esize`-byte elements, columns [c0, c0 + nc); out is [nr][nc]
    void block(const std::string &n, int64_t r0, int64_t nr, int64_t c0, int64_t nc, int esize, void *dst) const {
        if (replay) { img_io(dst, size_t(nr * nc * esize)); return; }
        block_read(n, r0, nr, c0, nc, esize, dst);
        if (img) img_io(dst, size_t(nr * nc * esize));
    }
    void block_read(const std::string &n, int64_t r0, int64_t nr, int64_t c0, int64_t nc, int esize, void *dst) const {
        const T &x = at(n);
        const int64_t cols = x.shape.size() > 1 ? x.shape[1] : x.shape[0];
        if (nc == cols) { read(x, r0 * cols * esize, nr * cols * esize, dst); return; }
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < nr; ++r) read(x, ((r0 + r) * cols + c0) * esize, nc * esize, static_cast<char *>(dst) + r * nc * esize);
    }
};

// ---------------------------------------------------------------- matrices
int g_threads = 1;
size_t g_weight_bytes = 0;
// default-initialising allocator: resize() leaves the pages untouched, so each thread can first-touch (and so home) its own rows
template <class T> struct NoInit : std::allocator<T> {
    template <class U> struct rebind { using other = NoInit<U>; };
    NoInit() = default;
    template <class U> NoInit(const NoInit<U> &) {}
    template <class U> void construct(U *p) { ::new (static_cast<void *>(p)) U; }
    template <class U, class... A> void construct(U *p, A &&...a) { ::new (static_cast<void *>(p)) U(std::forward<A>(a)...); }
};
template <class T> using NVec = std::vector<T, NoInit<T>>;
struct Mat {
    int rows = 0, cols = 0;
    bool fp4 = false;
    NVec<uint16_t> w;        // BF16 [rows][cols]
    NVec<uint8_t> q;         // NVFP4 repacked rows (ripple_sve.h): [rows][cols / 2]
    NVec<uint16_t> sf;       // F16 block scales [rows][cols / 16]
    float s2 = 1.0f;
};
// thread t of nt computes rows [row_lo, row_hi) of every matrix (static split: the same rows every token, so the pages first
// touched by thread t stay in its CMG's memory)
inline int row_lo(int rows, int t, int nt) { return int(int64_t(rows) * t / nt); }
void first_touch(Mat &m) {
#pragma omp parallel
    {
        int t = 0, nt = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
        const int a = row_lo(m.rows, t, nt), b = row_lo(m.rows, t + 1, nt);
        if (m.fp4) {
            std::memset(m.q.data() + size_t(a) * m.cols / 2, 0, size_t(b - a) * m.cols / 2);
            std::memset(m.sf.data() + size_t(a) * m.cols / 16, 0, size_t(b - a) * m.cols / 8);
        } else std::memset(m.w.data() + size_t(a) * m.cols, 0, size_t(b - a) * m.cols * 2);
    }
}
// BF16 rows [r0, r0 + nr), columns [c0, c0 + nc) of tensor `name` (a ".weight" name)
Mat load_bf16(const StSrc &s, const std::string &name, int r0, int nr, int c0, int nc) {
    const auto &x = s.at(name);
    if (x.dtype != "BF16") throw std::runtime_error("ripple: " + name + " is " + x.dtype + ", not BF16");
    Mat m; m.rows = nr; m.cols = nc; m.w.resize(size_t(nr) * nc);
    first_touch(m);
    s.block(name, r0, nr, c0, nc, 2, m.w.data());
    g_weight_bytes += m.w.size() * 2;
    return m;
}
// the NVFP4 projection `base` (".weight" / ".weight_scale" / ".weight_scale_2"), same slice (c0, nc multiples of 32); read in the
// checkpoint's layout, then repacked row by row by the thread that will compute the row (which also homes its pages)
Mat load_fp4(const StSrc &s, const std::string &base, int r0, int nr, int c0, int nc) {
    const auto &x = s.at(base + ".weight");
    if (x.dtype != "U8") throw std::runtime_error("ripple: " + base + ".weight is " + x.dtype + ", not packed U8");
    Mat m; m.rows = nr; m.cols = nc; m.fp4 = true;
    // scratch kept across calls: freeing it every call would hand its (already homed) pages to the next weight arrays
    static NVec<uint8_t> tq, tsc;
    if (tq.size() < size_t(nr) * nc / 2) tq.resize(size_t(nr) * nc / 2);
    if (tsc.size() < size_t(nr) * nc / 16) tsc.resize(size_t(nr) * nc / 16);
    s.block(base + ".weight", r0, nr, c0 / 2, nc / 2, 1, tq.data());
    s.block(base + ".weight_scale", r0, nr, c0 / 16, nc / 16, 1, tsc.data());
    s.fetch(s.at(base + ".weight_scale_2"), 0, 4, &m.s2);
    m.q.resize(size_t(nr) * nc / 2); m.sf.resize(size_t(nr) * nc / 16);
#pragma omp parallel
    {
        int t = 0, nt = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
        for (int r = row_lo(nr, t, nt); r < row_lo(nr, t + 1, nt); ++r)
            ripple_fp4_repack_row(tq.data() + size_t(r) * nc / 2, tsc.data() + size_t(r) * nc / 16, nc, m.q.data() + size_t(r) * nc / 2, m.sf.data() + size_t(r) * nc / 16);
    }
    g_weight_bytes += m.q.size() + m.sf.size() * 2;
    return m;
}
Vec load_vec(const StSrc &s, const std::string &n, int count) {
    const auto &x = s.at(n);
    Vec v(static_cast<size_t>(count));
    if (x.dtype == "F32") { s.fetch(x, 0, int64_t(count) * 4, v.data()); return v; }
    if (x.dtype != "BF16") throw std::runtime_error("ripple: " + n + " is " + x.dtype);
    std::vector<uint16_t> b(static_cast<size_t>(count));
    s.fetch(x, 0, int64_t(count) * 2, b.data());
    for (int i = 0; i < count; ++i) v[size_t(i)] = bf16_to_f32(b[size_t(i)]);
    return v;
}

// one parallel region over (matrix, input, output) jobs; thread t takes its static row range of each matrix. NVFP4 jobs take int8
// activations: the distinct input vectors are quantized first, inside the region (an omp for over their 64-element units).
struct Job { const Mat *m; const float *x; float *y; };   // x: nt vectors of m->cols, m->cols apart; y: [nt][m->rows]
void mv(const std::vector<Job> &jobs, int nt = 1) {
    struct XQ { const float *x; int cols, units; int8_t *q; float *d; };
    static NVec<int8_t> qbuf; static NVec<float> dbuf;
    std::vector<XQ> xs;
    size_t qn = 0, dn = 0;
    for (const Job &J8 : jobs) {
        if (!J8.m->fp4) continue;
        bool have = false;
        for (const XQ &e : xs) have = have || (e.x == J8.x && e.cols == J8.m->cols);
        if (have) continue;
        xs.push_back({J8.x, J8.m->cols, ripple_q8_units(J8.m->cols), nullptr, nullptr});
        qn += size_t(nt) * J8.m->cols; dn += size_t(nt) * size_t(xs.back().units);
    }
    if (qbuf.size() < qn) qbuf.resize(qn);
    if (dbuf.size() < dn) dbuf.resize(dn);
    { size_t qo = 0, dof = 0; for (XQ &e : xs) { e.q = qbuf.data() + qo; e.d = dbuf.data() + dof; qo += size_t(nt) * e.cols; dof += size_t(nt) * size_t(e.units); } }
    const int nx = int(xs.size());
    int total = 0;
    for (const XQ &e : xs) total += nt * e.units;
#pragma omp parallel
    {
        int t = 0, nth = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num(); nth = omp_get_num_threads();
#endif
        if (nx) {
#pragma omp for schedule(static)
            for (int it = 0; it < total; ++it) {
                int k = it, xi = 0;
                while (k >= nt * xs[size_t(xi)].units) { k -= nt * xs[size_t(xi)].units; ++xi; }
                const XQ &e = xs[size_t(xi)];
                const int tok = k / e.units, u = k % e.units;
                ripple_q8_quant_unit(e.x + size_t(tok) * e.cols, e.cols, u, e.q + size_t(tok) * e.cols, e.d + size_t(tok) * e.units + u);
            }
        }
        for (const Job &J8 : jobs) {
            const int r0 = row_lo(J8.m->rows, t, nth), r1 = row_lo(J8.m->rows, t + 1, nth);
            if (r1 <= r0) continue;
            if (J8.m->fp4) {
                const XQ *e = nullptr;
                for (const XQ &c : xs) if (c.x == J8.x && c.cols == J8.m->cols) e = &c;
                const int8_t *xq[8]; const float *xd[8];
                for (int k = 0; k < nt; ++k) { xq[k] = e->q + size_t(k) * e->cols; xd[k] = e->d + size_t(k) * e->units; }
                ripple_fp4_rows_q8(J8.m->q.data(), J8.m->sf.data(), J8.m->cols, r0, r1, xq, xd, nt, J8.m->s2, J8.y, J8.m->rows);
                continue;
            }
            const void *acts[8];
            for (int k = 0; k < nt; ++k) acts[k] = J8.x + size_t(k) * J8.m->cols;
            ds41_rows(DS41_T_BF16, J8.m->w.data(), J8.m->cols, r0, r1, acts, nt, J8.y, J8.m->rows);
        }
    }
}

// ---------------------------------------------------------------- model
struct Cfg {
    int hidden = 5120, inter = 40960, heads = 80, kv_heads = 8, head_dim = 128, layers = 80, vocab = 100352, window = 1024;
    float eps = 1e-6f, theta = 10000.0f, embed_scale = 1.0f, clamp = 20.0f;
    bool valres = true, gated = true, gate_fp32 = true;
    std::vector<int> eos;       // eos_token_id (a number or a list)
    std::vector<bool> global;   // per layer: full attention, no RoPE
};
struct Layer {
    int idx = 0;
    bool global = false;
    Vec in_ln, post_attn_ln, pre_ff_ln, post_ff_ln, qn, kn;
    float blend = 1.0f;             // sigmoid(value_res_lambda)
    Mat wq, wg, wk, wv, wo;         // wq / wg: rows of my heads; wo: columns of my heads
    Mat gate, up, down;             // gate / up: rows [i0, i0 + in); down: columns [i0, i0 + in)
    std::vector<_Float16> kc, vc;   // [slot][kv_heads][head_dim]
    int64_t cap = 0;
};
struct Model {
    Cfg c;
    std::vector<Layer> L;
    Mat head;                       // lm_head rows [v0, v0 + vn)
    Mat emb;                        // token embedding rows [v0, v0 + vn)
    Vec final_norm, inv_freq;
    int h0 = 0, hl = 0, i0 = 0, in = 0, v0 = 0, vn = 0;
    const StSrc *src = nullptr;
    bool fp4 = false;
};

std::string P(int l, const std::string &s) { return "model.layers." + std::to_string(l) + "." + s; }

Cfg read_cfg(const std::string &dir) {
    Cfg c;
    const J cj = parse_json(slurp(dir + "/config.json"));
    auto num = [&](const char *k, double d) { const J *j = cj.get(k); return j && j->t == J::NUM ? j->n : d; };
    c.hidden = int(num("hidden_size", 5120)); c.inter = int(num("intermediate_size", 40960));
    c.heads = int(num("num_attention_heads", 80)); c.kv_heads = int(num("num_key_value_heads", 8));
    c.head_dim = int(num("head_dim", 128)); c.layers = int(num("num_hidden_layers", 80)); c.vocab = int(num("vocab_size", 100352));
    c.window = int(num("sliding_window", 1024)); c.eps = float(num("rms_norm_eps", 1e-6)); c.theta = float(num("rope_theta", 10000.0));
    c.embed_scale = float(num("embed_scale", 1.0)); c.clamp = float(num("ffn_clamp_threshold", 20.0));
    const J *vr = cj.get("use_valres"), *ga = cj.get("use_gated_attention"), *gf = cj.get("gate_full_precision");
    c.valres = vr ? vr->n != 0 : true; c.gated = ga ? ga->n != 0 : true; c.gate_fp32 = gf ? gf->n != 0 : true;
    if (const J *e = cj.get("eos_token_id")) { if (e->t == J::NUM) c.eos.push_back(int(e->n)); else for (auto &x : e->a) c.eos.push_back(int(x.n)); }
    c.global.assign(size_t(c.layers), false);
    const J &lt = cj.at("layer_types");
    for (int l = 0; l < c.layers; ++l) c.global[size_t(l)] = lt.a[size_t(l)].s == "full_attention";
    if (c.heads % c.kv_heads || c.head_dim % 2 || c.inter % RU) throw std::runtime_error("ripple: unsupported head / FFN shape");
    return c;
}

Layer build_layer(const StSrc &s, Model &m, int l, int64_t ctx) {
    const Cfg &c = m.c;
    Layer Y; Y.idx = l; Y.global = c.global[size_t(l)];
    const int H = c.hidden, HD = c.head_dim;
    Y.in_ln = load_vec(s, P(l, "input_layernorm.weight"), H);
    Y.post_attn_ln = load_vec(s, P(l, "post_attention_layernorm.weight"), H);
    Y.pre_ff_ln = load_vec(s, P(l, "pre_feedforward_layernorm.weight"), H);
    Y.post_ff_ln = load_vec(s, P(l, "post_feedforward_layernorm.weight"), H);
    Y.qn = load_vec(s, P(l, "self_attn.q_norm.weight"), HD);
    Y.kn = load_vec(s, P(l, "self_attn.k_norm.weight"), HD);
    if (c.valres) { const Vec lam = load_vec(s, P(l, "self_attn.value_res_lambda"), 1); Y.blend = sigmoidf(lam[0]); }
    const int qd = c.heads * HD, kd = c.kv_heads * HD;
    Y.wq = load_bf16(s, P(l, "self_attn.q_proj.weight"), m.h0 * HD, m.hl * HD, 0, H);
    if (c.gated) Y.wg = load_bf16(s, P(l, "self_attn.g_proj.weight"), m.h0 * HD, m.hl * HD, 0, H);
    Y.wk = load_bf16(s, P(l, "self_attn.k_proj.weight"), 0, kd, 0, H);
    Y.wv = load_bf16(s, P(l, "self_attn.v_proj.weight"), 0, kd, 0, H);
    Y.wo = load_bf16(s, P(l, "self_attn.o_proj.weight"), 0, H, m.h0 * HD, m.hl * HD);
    (void)qd;
    if (m.fp4) {
        Y.gate = load_fp4(s, P(l, "mlp.gate_proj"), m.i0, m.in, 0, H);
        Y.up = load_fp4(s, P(l, "mlp.up_proj"), m.i0, m.in, 0, H);
        Y.down = load_fp4(s, P(l, "mlp.down_proj"), 0, H, m.i0, m.in);
    } else {
        Y.gate = load_bf16(s, P(l, "mlp.gate_proj.weight"), m.i0, m.in, 0, H);
        Y.up = load_bf16(s, P(l, "mlp.up_proj.weight"), m.i0, m.in, 0, H);
        Y.down = load_bf16(s, P(l, "mlp.down_proj.weight"), 0, H, m.i0, m.in);
    }
    Y.cap = Y.global ? ctx : std::min<int64_t>(ctx, int64_t(c.window) + 8);   // +8: a batch of 8 never overwrites a key an earlier token of it still needs
    Y.kc.assign(size_t(Y.cap) * kd, _Float16(0)); Y.vc.assign(size_t(Y.cap) * kd, _Float16(0));
    return Y;
}

// half-split RoPE of one head vector at position pos (fp32, as the reference)
void rope(float *x, int hd, const float *cs, const float *sn) {   // cs / sn: hd / 2 values of this position
    const int h2 = hd / 2;
    for (int i = 0; i < h2; ++i) {
        const float x1 = x[i], x2 = x[i + h2];
        x[i] = x1 * cs[i] - x2 * sn[i]; x[i + h2] = x2 * cs[i] + x1 * sn[i];
    }
}

std::map<std::string, double> g_tm;
struct Lap { double t = now(); void operator()(const char *k) { const double t2 = now(); g_tm[k] += t2 - t; t = t2; } };

// attention of this rank's heads: q [hl][HD] (normed, roped) -> out [hl][HD]
void attend(Model &m, Layer &Y, const float *q, float *out, int64_t pos) {
    const Cfg &c = m.c;
    const int HD = c.head_dim, KD = c.kv_heads * HD, rep = c.heads / c.kv_heads, HL = m.hl;
    const int64_t n = Y.global ? pos + 1 : std::min<int64_t>(pos + 1, c.window), t0 = pos + 1 - n;
    const float scale = 1.0f / std::sqrt(float(HD));
    constexpr int CH = 64;
    const int64_t nch = (n + CH - 1) / CH;
    Vec sc(size_t(HL) * size_t(n));
    auto slot = [&](int64_t t) { return Y.global ? t : t % Y.cap; };
    if (n <= 384) {   // short context: one region, a head per thread (no barriers between scores, softmax and the weighted sum)
#pragma omp parallel for schedule(static)
        for (int h = 0; h < HL; ++h) {
            const int kvh = (m.h0 + h) / rep;
            const float *qh = q + size_t(h) * HD;
            float *s = sc.data() + size_t(h) * size_t(n), *o = out + size_t(h) * HD;
            float mx = -INFINITY;
            for (int64_t i = 0; i < n; ++i) {
                const _Float16 *k = Y.kc.data() + size_t(slot(t0 + i)) * KD + size_t(kvh) * HD;
                float a = 0;
                for (int d = 0; d < HD; ++d) a += qh[d] * float(k[d]);
                s[i] = a * scale; mx = std::max(mx, s[i]);
            }
            double sum = 0;
            for (int64_t i = 0; i < n; ++i) { s[i] = std::exp(s[i] - mx); sum += s[i]; }
            const float inv = float(1.0 / sum);
            for (int d = 0; d < HD; ++d) o[d] = 0;
            for (int64_t i = 0; i < n; ++i) {
                const _Float16 *v = Y.vc.data() + size_t(slot(t0 + i)) * KD + size_t(kvh) * HD;
                const float pr = s[i] * inv;
                for (int d = 0; d < HD; ++d) o[d] += pr * float(v[d]);
            }
        }
        return;
    }
#pragma omp parallel for schedule(static) collapse(2)
    for (int h = 0; h < HL; ++h)
        for (int64_t ch = 0; ch < nch; ++ch) {
            const int kvh = (m.h0 + h) / rep;
            const float *qh = q + size_t(h) * HD;
            for (int64_t i = ch * CH; i < std::min<int64_t>(n, (ch + 1) * CH); ++i) {
                const _Float16 *k = Y.kc.data() + size_t(slot(t0 + i)) * KD + size_t(kvh) * HD;
                float a = 0;
                for (int d = 0; d < HD; ++d) a += qh[d] * float(k[d]);
                sc[size_t(h) * size_t(n) + size_t(i)] = a * scale;
            }
        }
#pragma omp parallel for schedule(static)
    for (int h = 0; h < HL; ++h) {
        float *s = sc.data() + size_t(h) * size_t(n);
        float mx = -INFINITY;
        for (int64_t i = 0; i < n; ++i) mx = std::max(mx, s[i]);
        double sum = 0;
        for (int64_t i = 0; i < n; ++i) { s[i] = std::exp(s[i] - mx); sum += s[i]; }
        const float inv = float(1.0 / sum);
        for (int64_t i = 0; i < n; ++i) s[i] *= inv;
    }
    constexpr int DC = 32;
#pragma omp parallel for schedule(static) collapse(2)
    for (int h = 0; h < HL; ++h)
        for (int d0 = 0; d0 < HD; d0 += DC) {
            const int kvh = (m.h0 + h) / rep;
            float acc[DC] = {};
            const float *s = sc.data() + size_t(h) * size_t(n);
            for (int64_t i = 0; i < n; ++i) {
                const _Float16 *v = Y.vc.data() + size_t(slot(t0 + i)) * KD + size_t(kvh) * HD + d0;
                const float p = s[i];
                for (int d = 0; d < DC; ++d) acc[d] += p * float(v[d]);
            }
            for (int d = 0; d < DC; ++d) out[size_t(h) * HD + size_t(d0) + size_t(d)] = acc[d];
        }
}

struct StepOut { int argmax = 0; double logp_target = 0; };
FILE *g_dump = nullptr;   // rank 0 only
bool g_dump_on = false;   // every rank joins the logits gather
std::vector<int> g_vcnt, g_vdsp;

// nb <= 8 consecutive tokens toks[0..nb) at positions pos0.. through all layers. With `head`, one StepOut per token
// (targets[i] = the token that should follow, or -1); the attention of token i sees the keys of tokens <= its position only.
std::vector<StepOut> forward(Model &m, const int *toks, int nb, int64_t pos0, bool head, const int *targets) {
    Lap lap;
    const Cfg &c = m.c;
    const int H = c.hidden, HD = c.head_dim, KD = c.kv_heads * HD, HL = m.hl, QD = HL * HD;
    auto V = [](size_t n) { return Vec(n); };
    Vec h = V(size_t(nb) * H), emb0, x = V(size_t(nb) * H), vin = V(size_t(nb) * H);
    std::fill(h.begin(), h.end(), 0.0f);
    for (int b = 0; b < nb; ++b) {   // the owner of a row contributes it; the sum gives every rank the rows
        const int tok = toks[b];
        if (tok >= m.v0 && tok < m.v0 + m.vn) {
            const uint16_t *row = m.emb.w.data() + size_t(tok - m.v0) * H;
            for (int i = 0; i < H; ++i) h[size_t(b) * H + size_t(i)] = bf16_to_f32(row[i]) * c.embed_scale;
        }
    }
    comm.sum(h.data(), nb * H);
    emb0 = h;
    // RoPE tables of each token's position (the same for every sliding layer)
    const int h2 = HD / 2;
    Vec rcs(size_t(nb) * h2), rsn(size_t(nb) * h2);
    for (int b = 0; b < nb; ++b)
        for (int i = 0; i < h2; ++i) {
            const double a = double(pos0 + b) * double(m.inv_freq[size_t(i)]);
            rcs[size_t(b) * h2 + size_t(i)] = float(std::cos(a)); rsn[size_t(b) * h2 + size_t(i)] = float(std::sin(a));
        }
    lap("embed");
    Vec q = V(size_t(nb) * QD), g = V(size_t(nb) * QD), k = V(size_t(nb) * KD), v = V(size_t(nb) * KD), att = V(size_t(nb) * QD);
    Vec part = V(size_t(nb) * H), nrm = V(size_t(nb) * H);
    Vec gt = V(size_t(nb) * size_t(std::max(1, m.in))), upv = V(size_t(nb) * size_t(std::max(1, m.in)));
    for (Layer &Y : m.L) {
        // ---- attention
        for (int b = 0; b < nb; ++b) rmsnorm(x.data() + size_t(b) * H, h.data() + size_t(b) * H, Y.in_ln.data(), H, c.eps);
        const float *vsrc = x.data();
        if (c.valres) {
            for (size_t i = 0; i < size_t(nb) * H; ++i) vin[i] = emb0[i] + Y.blend * (x[i] - emb0[i]);
            vsrc = vin.data();
        }
        std::vector<Job> jobs = {{&Y.wq, x.data(), q.data()}, {&Y.wk, x.data(), k.data()}, {&Y.wv, vsrc, v.data()}};
        if (c.gated) jobs.push_back({&Y.wg, x.data(), g.data()});
        mv(jobs, nb);
        lap("attn.proj");
#pragma omp parallel for schedule(static) collapse(2)
        for (int b = 0; b < nb; ++b)
            for (int hh = 0; hh < HL + c.kv_heads; ++hh) {
                const bool is_q = hh < HL;
                float *p = is_q ? q.data() + size_t(b) * QD + size_t(hh) * HD : k.data() + size_t(b) * KD + size_t(hh - HL) * HD;
                rmsnorm(p, p, is_q ? Y.qn.data() : Y.kn.data(), HD, c.eps);
                if (!Y.global) rope(p, HD, rcs.data() + size_t(b) * h2, rsn.data() + size_t(b) * h2);
            }
        for (int b = 0; b < nb; ++b) {
            const int64_t sl = Y.global ? pos0 + b : (pos0 + b) % Y.cap;
            for (int i = 0; i < KD; ++i) { Y.kc[size_t(sl) * KD + size_t(i)] = _Float16(k[size_t(b) * KD + size_t(i)]); Y.vc[size_t(sl) * KD + size_t(i)] = _Float16(v[size_t(b) * KD + size_t(i)]); }
        }
        for (int b = 0; b < nb; ++b) attend(m, Y, q.data() + size_t(b) * QD, att.data() + size_t(b) * QD, pos0 + b);
        if (c.gated) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < nb * QD; ++i) att[size_t(i)] *= sigmoidf(g[size_t(i)]);
        }
        lap("attn.core");
        mv({{&Y.wo, att.data(), part.data()}}, nb);
        lap("attn.o");
        comm.sum(part.data(), nb * H);
        lap("attn.comm");
        for (int b = 0; b < nb; ++b) {
            rmsnorm(nrm.data() + size_t(b) * H, part.data() + size_t(b) * H, Y.post_attn_ln.data(), H, c.eps);
            for (int i = 0; i < H; ++i) h[size_t(b) * H + size_t(i)] += nrm[size_t(b) * H + size_t(i)];
            rmsnorm(x.data() + size_t(b) * H, h.data() + size_t(b) * H, Y.pre_ff_ln.data(), H, c.eps);
        }
        // ---- FFN
        mv({{&Y.gate, x.data(), gt.data()}, {&Y.up, x.data(), upv.data()}}, nb);
        lap("ffn.gateup");
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nb * m.in; ++i) {
            const float a = std::min(gt[size_t(i)], c.clamp), bb = std::max(-c.clamp, std::min(c.clamp, upv[size_t(i)]));
            gt[size_t(i)] = a * sigmoidf(a) * bb;
        }
        mv({{&Y.down, gt.data(), part.data()}}, nb);
        lap("ffn.down");
        comm.sum(part.data(), nb * H);
        lap("ffn.comm");
        for (int b = 0; b < nb; ++b) {
            rmsnorm(nrm.data() + size_t(b) * H, part.data() + size_t(b) * H, Y.post_ff_ln.data(), H, c.eps);
            for (int i = 0; i < H; ++i) h[size_t(b) * H + size_t(i)] += nrm[size_t(b) * H + size_t(i)];
        }
    }
    std::vector<StepOut> outs(static_cast<size_t>(nb));
    if (!head) return outs;
    for (int b = 0; b < nb; ++b) rmsnorm(x.data() + size_t(b) * H, h.data() + size_t(b) * H, m.final_norm.data(), H, c.eps);
    const int VN = std::max(1, m.vn);
    Vec lg(size_t(nb) * VN);
    mv({{&m.head, x.data(), lg.data()}}, nb);
    lap("head");
    for (int b = 0; b < nb; ++b) {
        const float *l = lg.data() + size_t(b) * m.vn;   // mv wrote [nt][rows] with rows = vn
        float best = -INFINITY; int bi = 0;
        for (int i = 0; i < m.vn; ++i) if (l[i] > best) { best = l[i]; bi = m.v0 + i; }
        float lmax = best;
        comm.max(&lmax, 1);
        double se = 0;
        for (int i = 0; i < m.vn; ++i) se += std::exp(double(l[i]) - double(lmax));
        comm.dsum(&se, 1);
        const int target = targets ? targets[b] : -1;
        float tl = 0;
        if (target >= m.v0 && target < m.v0 + m.vn) tl = l[target - m.v0];
        comm.sum(&tl, 1);
        comm.argmax(best, bi);
        outs[size_t(b)].argmax = bi;
        outs[size_t(b)].logp_target = target >= 0 ? double(tl) - double(lmax) - std::log(se) : 0.0;
        if (g_dump_on) {
            Vec all(size_t(c.vocab));
            comm.gatherv(l, m.vn, all.data(), g_vcnt, g_vdsp);
            if (comm.rank == 0) std::fwrite(all.data(), sizeof(float), all.size(), g_dump);
        }
    }
    lap("head.sel");
    return outs;
}
StepOut forward1(Model &m, int tok, int64_t pos, bool head, int target) { return forward(m, &tok, 1, pos, head, &target)[0]; }

uint64_t fnv(const void *p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= static_cast<const uint8_t *>(p)[i]; h *= 1099511628211ull; } return h; }

}  // namespace

int main(int argc, char **argv) {
    std::string model_dir, eval;
    int layers = -1, steps = 6, gen = 0, first = 1;
    int64_t ctx = 4096, ntok = 0;
    bool head = true, no_eos = false, score = true;
    int batch = 8;
    std::string dump, image;
    try {
        comm.init(&argc, &argv);
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("ripple: " + a + " needs a value"); return argv[++i]; };
            if (a == "--model") model_dir = next();
            else if (a == "--layers") layers = std::atoi(next());
            else if (a == "--steps") steps = std::atoi(next());
            else if (a == "--context") ctx = std::atoll(next());
            else if (a == "--eval") eval = next();
            else if (a == "--tokens") ntok = std::atoll(next());
            else if (a == "--generate") gen = std::atoi(next());
            else if (a == "--token") first = std::atoi(next());
            else if (a == "--no-head") head = false;
            else if (a == "--no-eos") no_eos = true;
            else if (a == "--no-score") score = false;   // prompt only: skip the head for all but the last prompt token
            else if (a == "--batch") batch = std::max(1, std::min(8, std::atoi(next())));
            else if (a == "--dump-logits") dump = next();
            else if (a == "--image") image = next();
            else if (a == "--help") {
                std::printf("ripple_runner --model DIR [--layers N] [--context N] [--eval IDS[,IDS] --tokens N] [--generate N]\n"
                            "              [--steps N --token T] [--no-head] [--no-eos] [--no-score] [--batch N] [--dump-logits FILE] [--image DIR]\n");
                comm.fini(); return 0;
            } else throw std::runtime_error("ripple: unknown option " + a);
        }
        if (model_dir.empty()) throw std::runtime_error("ripple: --model DIR");
#if defined(_OPENMP)
        g_threads = omp_get_max_threads();
#endif
        ripple_nvfp4_init();
        std::vector<std::pair<std::string, std::vector<int>>> evals;
        for (size_t p0 = 0; !eval.empty() && p0 <= eval.size();) {
            size_t p1 = eval.find(',', p0); if (p1 == std::string::npos) p1 = eval.size();
            const std::string fn = eval.substr(p0, p1 - p0);
            FILE *f = std::fopen(fn.c_str(), "r");
            if (!f) throw std::runtime_error("ripple: cannot open " + fn);
            std::vector<int> v; int x; while (std::fscanf(f, "%d", &x) == 1) v.push_back(x);
            std::fclose(f);
            if (ntok > 0 && int64_t(v.size()) > ntok) v.resize(static_cast<size_t>(ntok));
            evals.emplace_back(fn, v);
            p0 = p1 + 1;
        }
        int64_t longest = 0;
        for (auto &e : evals) longest = std::max<int64_t>(longest, int64_t(e.second.size()));
        const int64_t need = std::max<int64_t>(steps + 1, longest + gen) + 1;
        if (ctx < need) ctx = need;
        const double tb = now();
        StSrc src(model_dir);
        Model m;
        m.c = read_cfg(model_dir);
        m.src = &src;
        m.fp4 = src.has("model.layers.0.mlp.gate_proj.weight_scale");
        const int W = comm.world, R = comm.rank;
        if (layers < 0 || layers > m.c.layers) layers = m.c.layers;
        m.h0 = split0(m.c.heads, R, W); m.hl = split0(m.c.heads, R + 1, W) - m.h0;
        m.i0 = split0(m.c.inter / RU, R, W) * RU; m.in = split0(m.c.inter / RU, R + 1, W) * RU - m.i0;
        m.v0 = split0(m.c.vocab, R, W); m.vn = split0(m.c.vocab, R + 1, W) - m.v0;
        g_vcnt.resize(static_cast<size_t>(W)); g_vdsp.resize(static_cast<size_t>(W));
        for (int r = 0; r < W; ++r) { g_vdsp[size_t(r)] = split0(m.c.vocab, r, W); g_vcnt[size_t(r)] = split0(m.c.vocab, r + 1, W) - g_vdsp[size_t(r)]; }
        m.inv_freq.resize(size_t(m.c.head_dim / 2));
        for (int i = 0; i < m.c.head_dim / 2; ++i) m.inv_freq[size_t(i)] = float(1.0 / std::pow(double(m.c.theta), double(2 * i) / double(m.c.head_dim)));
        if (!image.empty()) src.image(image, R, W, layers, m.fp4);
        for (int l = 0; l < layers; ++l) {
            m.L.push_back(build_layer(src, m, l, ctx));
            if (R == 0 && (l % 8 == 0 || l == layers - 1))
                std::fprintf(stderr, "RIPPLE loaded layer %d weight_GB=%.2f read_GB=%.2f rss_GB=%.2f t=%.1f\n", l, g_weight_bytes / 1e9, g_read_b / 1e9, rss_bytes() / 1e9, now() - tb);
        }
        m.final_norm = load_vec(src, "model.norm.weight", m.c.hidden);
        m.emb = load_bf16(src, "model.embed_tokens.weight", m.v0, m.vn, 0, m.c.hidden);
        if (head) m.head = load_bf16(src, "lm_head.weight", m.v0, m.vn, 0, m.c.hidden);
        const bool replayed = src.replay;
        src.image_finish();
        g_dump_on = !dump.empty();
        if (g_dump_on && R == 0) g_dump = std::fopen(dump.c_str(), "wb");
        size_t cache_b = 0;
        for (Layer &Y : m.L) cache_b += (Y.kc.size() + Y.vc.size()) * 2;
        comm.barrier();
        if (R == 0)
            std::printf("RUNNER_RIPPLE ready ranks=%d threads=%d layers=%zu heads_mine=%d inter_mine=%d fp4=%d image=%d context=%" PRId64 " weight_GB=%.3f cache_GB=%.3f "
                        "rss_GB=%.3f load_s=%.1f read_GB=%.2f comm=%s\n",
                        W, g_threads, m.L.size(), m.hl, m.in, int(m.fp4), image.empty() ? 0 : (replayed ? 2 : 1), ctx, g_weight_bytes / 1e9, cache_b / 1e9, rss_bytes() / 1e9, now() - tb,
                        g_read_b / 1e9, comm.ut ? "utofu" : "mpi");
        std::fflush(stdout);
        std::vector<int> out;
        double t_steps = 0; int64_t timed = 0;
        for (auto &ev : evals) {
            const std::vector<int> &ids = ev.second;
            std::vector<int> outs;
            double nll = 0; int64_t scored = 0, top1 = 0;
            const double te = now();
            for (size_t i = 0; i < ids.size(); i += size_t(batch)) {
                const int nb = int(std::min<size_t>(size_t(batch), ids.size() - i));
                int tg[8];
                for (int b = 0; b < nb; ++b) tg[b] = i + size_t(b) + 1 < ids.size() ? ids[i + size_t(b) + 1] : -1;
                const bool last_batch = i + size_t(nb) == ids.size();
                const std::vector<StepOut> so = forward(m, ids.data() + i, nb, int64_t(i), head && (score || last_batch), tg);
                if (so.empty() || !(head && (score || last_batch))) continue;
                for (int b = 0; b < nb; ++b) {
                    const bool last = i + size_t(b) + 1 == ids.size();
                    if (!last && score) { nll -= so[size_t(b)].logp_target; top1 += so[size_t(b)].argmax == ids[i + size_t(b) + 1]; ++scored; }
                    if (last) outs.push_back(so[size_t(b)].argmax);
                }
            }
            const double eval_s = now() - te;
            g_tm.clear(); comm.t = 0; t_steps = 0; timed = 0;
            int tok = outs.empty() ? first : outs.back();
            for (int s = 0; s < gen; ++s) {
                comm.barrier(); const double t0 = now();
                const StepOut so = forward1(m, tok, int64_t(ids.size() + s), head, -1);
                comm.barrier(); t_steps += now() - t0; ++timed;
                outs.push_back(so.argmax); tok = so.argmax;
                if (!no_eos && std::find(m.c.eos.begin(), m.c.eos.end(), tok) != m.c.eos.end()) break;   // same token on every rank
            }
            if (R == 0) {
                std::printf("RUNNER_RIPPLE_EVAL file=%s tokens=%zu scored=%" PRId64 " ppl=%.5f top1=%.6f ms_per_token=%.3f full_model=%d\n",
                            ev.first.c_str(), ids.size(), scored, scored ? std::exp(nll / scored) : 0.0, scored ? double(top1) / scored : 0.0,
                            1e3 * eval_s / double(std::max<size_t>(1, ids.size())), int(m.L.size() == size_t(m.c.layers)));
                std::printf("RUNNER_RIPPLE_EVAL_GEN file=%s ms_per_token=%.3f tokens=", ev.first.c_str(), timed ? 1e3 * t_steps / timed : 0.0);
                for (size_t i = 0; i < outs.size(); ++i) std::printf("%s%d", i ? "," : "", outs[i]);
                std::printf("\n");
                std::fflush(stdout);
            }
            out = outs;
        }
        if (evals.empty()) {
            int tok = first;
            for (int s = 0; s < steps; ++s) {
                if (s == 1) { g_tm.clear(); comm.t = 0; }
                comm.barrier(); const double t0 = now();
                const StepOut so = forward1(m, tok, s, head, -1);
                comm.barrier();
                if (s >= 1) { t_steps += now() - t0; ++timed; }
                out.push_back(so.argmax); tok = head ? so.argmax : tok + 1;
            }
        }
        const uint64_t hh = fnv(out.data(), out.size() * sizeof(int));
        if (R == 0) {
            std::printf("RUNNER_RIPPLE_GEN timed=%" PRId64 " ms_per_token=%.3f hash=%016" PRIx64 " rss_GB=%.3f tokens=", timed, timed ? 1e3 * t_steps / timed : 0.0, hh, rss_bytes() / 1e9);
            for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : "", out[i]);
            std::printf("\nRUNNER_RIPPLE_TIMERS");
            const double d = timed ? double(timed) : 1.0;
            for (auto &kv : g_tm) std::printf(" %s=%.3f", kv.first.c_str(), 1e3 * kv.second / d);
            std::printf(" comm=%.3f\n", 1e3 * comm.t / d);
        }
        if (g_dump) std::fclose(g_dump);
        comm.fini();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ripple_runner rank %d: %s\n", comm.rank, e.what());
        comm.fini();
        return 2;
    }
}
