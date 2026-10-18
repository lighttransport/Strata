// tools/a64fx_probe/q23_sdot_bench.cpp - the A64FX SDOT kernels (q23_sdot_sve.c, iq_sdot_sve.c) against ggml's vec_dot.
//
//   q23_sdot_bench [--threads 48] [--mb 2048] [--reps 3] [--rows 2048 --cols 4096] [--tokens 1,2,3,4]
//                  [--types q2_k,q3_k,iq2_xxs,iq1_s] [--kernels ggml,v1,v2,v2r2,reg,gather,iq,tbl,rp,rpc] [--check | --check-only]
//
// Kernels per type: q2_k/q3_k: ggml, v1 (first SDOT kernel), v2 (one row), v2r2 (row pairs); iq2_xxs: ggml, reg
// (signs rebuilt in registers), gather (signs gathered from ksigns64), tbl (gather-free, ggml layout), rp (gather-free,
// repacked 74 B/block), rpc (codes expanded, 106 B); iq1_s: ggml, iq (gather), rpc (codes expanded, 82 B). `--kernels`
// filters. Repacked kernels stream their own layout (repacked before timing); GBps is in source-format bytes (what
// the pack occupies on disk and in the simulator), GBps_streamed in the bytes actually read.
// --check: 1..4 tokens, 64 random rows, every kernel against ggml's vec_dot on the same Q8_K activation (vector
// relative error; same integer sums, different float order) -> PASS at <= 1e-5.
// Timing: every thread streams its own rows (larger than any cache); GB/s counts weight bytes.
#include "ggml.h"
#include "ggml-cpu.h"
#include "q23_sdot_sve.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint64_t mix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_rows(uint8_t* p, size_t bytes, ggml_type t, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t v = mix(seed); std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i)); }
    const size_t bs = ggml_type_size(t);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-2f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d), h2 = ggml_fp32_to_fp16(d * 0.5f);
        if (t == GGML_TYPE_Q2_K) { std::memcpy(p + o + 80, &h, 2); std::memcpy(p + o + 82, &h2, 2); }
        else if (t == GGML_TYPE_Q3_K) std::memcpy(p + o + 108, &h, 2);
        else std::memcpy(p + o, &h, 2);                                   // IQ2_XXS, IQ1_S: d first
    }
}

struct Act { std::vector<uint8_t> q8k; std::vector<q23s_act> perm, nat, quad; };

Act make_act(int cols, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> f(cols); for (auto& v : f) v = nd(rng);
    Act a; a.q8k.resize(ggml_row_size(GGML_TYPE_Q8_K, cols));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(f.data(), a.q8k.data(), cols);
    a.perm.resize(cols / 256); q23s_prepare(a.q8k.data(), cols, a.perm.data());
    a.nat.resize(cols / 256); q23s_prepare_natural(a.q8k.data(), cols, a.nat.data());
    a.quad.resize(cols / 256); q23s_prepare_quad(a.q8k.data(), cols, a.quad.data());
    return a;
}

struct Kernel { std::string name; int rows_per_call; int block_bytes = 0; };   // block_bytes: repacked layout, 0 = source

std::vector<Kernel> kernels_for(ggml_type t) {
    if (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K) return {{"ggml", 1}, {"v1", 1}, {"v2", 1}, {"v2r2", 2}, {"v3", 1}};
    if (t == GGML_TYPE_IQ2_XXS) return {{"ggml", 1}, {"reg", 1}, {"gather", 1}, {"tbl", 1}, {"rp", 1, IQ2XXS_RP_BYTES}, {"rpc", 1, IQ2XXS_RPC_BYTES}, {"v3", 1}};
    return {{"ggml", 1}, {"iq", 1}, {"rpc", 1, IQ1S_RPC_BYTES}, {"v3", 1}};
}

// source rows -> the kernel's repacked layout (same row count)
std::vector<uint8_t> repack(ggml_type t, const Kernel& k, const uint8_t* src, size_t nrows, int cols) {
    const int nb = cols / 256;
    std::vector<uint8_t> out(nrows * nb * k.block_bytes);
    for (size_t r = 0; r < nrows; ++r) {
        const uint8_t* s = src + r * ggml_row_size(t, cols); uint8_t* d = out.data() + r * nb * k.block_bytes;
        if (t == GGML_TYPE_IQ2_XXS) iq2xxs_repack(s, d, nb, k.name == "rp" ? 1 : 2); else iq1s_repack(s, d, nb);
    }
    return out;
}

// one call: rows_per_call rows starting at `row`, out[r * nt + t]
void run(ggml_type t, const std::string& k, int cols, const uint8_t* row, size_t rb, const std::vector<Act>& acts, int nt, float* out) {
    const q23s_act* ap[4]; const q23s_act* an[4]; const q23s_act* aq[4];
    for (int i = 0; i < 4; ++i) { ap[i] = acts[i].perm.data(); an[i] = acts[i].nat.data(); aq[i] = acts[i].quad.data(); }
    if (k == "ggml") {
        const auto* tr = ggml_get_type_traits_cpu(t);
        for (int i = 0; i < nt; ++i) tr->vec_dot(cols, &out[i], 0, row, 0, acts[i].q8k.data(), 0, 1);
    } else if (t == GGML_TYPE_Q2_K) {
        if (k == "v1") q2k_sdot_row(cols, row, ap, nt, out); else if (k == "v3") q2k_sdot_row_v3(cols, row, ap, nt, out);
        else q2k_sdot_rows_v2(cols, row, rb, k == "v2r2" ? 2 : 1, ap, nt, out);
    } else if (t == GGML_TYPE_Q3_K) {
        if (k == "v1") q3k_sdot_row(cols, row, ap, nt, out); else if (k == "v3") q3k_sdot_row_v3(cols, row, ap, nt, out);
        else q3k_sdot_rows_v2(cols, row, rb, k == "v2r2" ? 2 : 1, ap, nt, out);
    } else if (t == GGML_TYPE_IQ2_XXS) {
        if (k == "tbl") iq2xxs_sdot_row_tbl(cols, row, aq, nt, out, 0);
        else if (k == "rp") iq2xxs_sdot_row_tbl(cols, row, aq, nt, out, 1);
        else if (k == "rpc") iq2xxs_sdot_row_tbl(cols, row, aq, nt, out, 2);
        else if (k == "v3") iq2xxs_sdot_row_v3(cols, row, an, nt, out);
        else iq2xxs_sdot_row(cols, row, an, nt, out, k == "gather" ? 1 : 0);
    } else {
        if (k == "rpc") iq1s_sdot_row_rpc(cols, row, aq, nt, out); else if (k == "v3") iq1s_sdot_row_v3(cols, row, an, nt, out);
        else iq1s_sdot_row(cols, row, an, nt, out);
    }
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; size_t p = 0;
    while (p <= s.size()) { size_t e = s.find(',', p); if (e == std::string::npos) e = s.size(); if (e > p) out.push_back(s.substr(p, e - p)); p = e + 1; }
    return out;
}

bool wanted(const std::vector<std::string>& filter, const std::string& k) {
    return filter.empty() || std::find(filter.begin(), filter.end(), k) != filter.end();
}

int check(const std::vector<ggml_type>& types, const std::vector<std::string>& filter, int cols) {
    int failures = 0;
    std::vector<Act> acts; for (int k = 0; k < 4; ++k) acts.push_back(make_act(cols, 101u + k));
    for (ggml_type t : types) {
        const size_t rb = ggml_row_size(t, cols); const int rows = 64;
        std::vector<uint8_t> m(rb * rows); uint64_t seed = 0xc0ffee + t; fill_rows(m.data(), m.size(), t, seed);
        for (auto& kern : kernels_for(t)) {
            if (kern.name == "ggml" || !wanted(filter, kern.name)) continue;
            std::vector<uint8_t> rp; if (kern.block_bytes) rp = repack(t, kern, m.data(), rows, cols);
            const size_t krb = kern.block_bytes ? (size_t) (cols / 256) * kern.block_bytes : rb;
            const uint8_t* kbase = kern.block_bytes ? rp.data() : m.data();
            for (int nt = 1; nt <= 4; ++nt) {
                double num = 0, den = 0;
                for (int r = 0; r < rows; r += kern.rows_per_call) {
                    float got[8], ref[8];
                    run(t, kern.name, cols, kbase + (size_t) r * krb, krb, acts, nt, got);
                    for (int rr = 0; rr < kern.rows_per_call; ++rr) {
                        run(t, "ggml", cols, m.data() + (size_t) (r + rr) * rb, rb, acts, nt, ref);
                        for (int i = 0; i < nt; ++i) { const double e = got[rr * nt + i] - ref[i]; num += e * e; den += (double) ref[i] * ref[i]; }
                    }
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30)); const bool ok = rel <= 1e-5;
                std::printf("  %-8s %-6s tokens=%d vs ggml vec_dot: rel %.2e %s\n", ggml_type_name(t), kern.name.c_str(), nt, rel, ok ? "PASS" : "FAIL");
                failures += !ok;
            }
        }
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int threads = omp_get_max_threads(), mb = 2048, reps = 3, rows = 2048, cols = 4096; bool do_check = false, check_only = false;
    std::string tokens_arg = "1,2,3,4", types_arg = "q2_k,q3_k,iq2_xxs,iq1_s", kernels_arg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--threads") threads = std::stoi(next()); else if (a == "--mb") mb = std::stoi(next()); else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--rows") rows = std::stoi(next()); else if (a == "--cols") cols = std::stoi(next()); else if (a == "--tokens") tokens_arg = next();
        else if (a == "--types") types_arg = next(); else if (a == "--kernels") kernels_arg = next(); else if (a == "--check") do_check = true;
        else if (a == "--check-only") do_check = check_only = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ggml_cpu_init();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (q23s_vl_bytes() != 64) { std::printf("A64FX_SDOT FAIL: SVE vector length %d bytes, kernels need 64\n", q23s_vl_bytes()); return 1; }
    std::vector<ggml_type> types;
    for (auto& s : split(types_arg)) {
        if (s == "q2_k") types.push_back(GGML_TYPE_Q2_K); else if (s == "q3_k") types.push_back(GGML_TYPE_Q3_K);
        else if (s == "iq2_xxs") types.push_back(GGML_TYPE_IQ2_XXS); else if (s == "iq1_s") types.push_back(GGML_TYPE_IQ1_S);
        else { std::fprintf(stderr, "type %s\n", s.c_str()); return 2; }
    }
    const auto filter = split(kernels_arg);
    std::printf("q23_sdot_bench: %d threads, rows %d x cols %d, %d MB of rows per run, %d reps\n", threads, rows, cols, mb, reps);
    if (do_check && check(types, filter, cols)) return 1;
    if (check_only) return 0;
    omp_set_num_threads(threads);
    std::vector<Act> acts; for (int k = 0; k < 4; ++k) acts.push_back(make_act(cols, 11u + k));
    for (ggml_type t : types) {
        const size_t rb = ggml_row_size(t, cols), mat = rb * rows;
        const size_t mats = std::max<size_t>(1, (size_t) mb * 1024 * 1024 / threads / mat);
        std::vector<std::vector<uint8_t>> mem(threads);
        #pragma omp parallel num_threads(threads)
        { const int tid = omp_get_thread_num(); mem[tid].resize(mats * mat); uint64_t seed = 0x5eed + tid + 1000 * t; fill_rows(mem[tid].data(), mem[tid].size(), t, seed); }
        for (auto& kern : kernels_for(t)) {
            if (!wanted(filter, kern.name)) continue;
            const size_t krb = kern.block_bytes ? (size_t) (cols / 256) * kern.block_bytes : rb, kmat = krb * rows;
            std::vector<std::vector<uint8_t>> kmem(kern.block_bytes ? threads : 0);
            if (kern.block_bytes) {
                #pragma omp parallel num_threads(threads)
                { const int tid = omp_get_thread_num(); kmem[tid] = repack(t, kern, mem[tid].data(), mats * rows, cols); }
            }
            for (auto& ts : split(tokens_arg)) {
                const int nt = std::stoi(ts);
                std::vector<float> sink(threads); double best = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    const auto t0 = std::chrono::steady_clock::now();
                    #pragma omp parallel num_threads(threads)
                    {
                        const int tid = omp_get_thread_num(); float acc = 0, o[8];
                        for (size_t m = 0; m < mats; ++m) {
                            const uint8_t* base = (kern.block_bytes ? kmem[tid].data() + m * kmat : mem[tid].data() + m * mat);
                            for (int r = 0; r + kern.rows_per_call <= rows; r += kern.rows_per_call) {
                                run(t, kern.name, cols, base + (size_t) r * krb, krb, acts, nt, o);
                                for (int i = 0; i < nt * kern.rows_per_call; ++i) acc += o[i];
                            }
                        }
                        sink[tid] = acc;
                    }
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                const double bytes = (double) threads * mats * mat, macs = bytes / rb * cols * nt;
                std::printf("A64FX_SDOT kernel=%s type=%s threads=%d rows=%d cols=%d tokens=%d GBps=%.1f GMACs=%.1f GBps_per_thread=%.2f GBps_streamed=%.1f\n",
                            kern.name.c_str(), ggml_type_name(t), threads, rows, cols, nt, bytes / best / 1e9, macs / best / 1e9, bytes / best / 1e9 / threads,
                            bytes * ((double) krb / rb) / best / 1e9);
            }
        }
    }
    return 0;
}
