// src/minimax/m3_cpu_expert_test.cpp - the CPU expert path MiniMax's decode lane uses, against an independent reference:
// ggml's own dequantizers (weights, and the Q8_K activation) and a plain float dot.  Covers swigluoai (NativeFmt::
// swiglu_alpha > 0) with the clamp, the unchanged default silu(g) * u (alpha = 0), and the down rows.  No GPU, no model.
#include "strata/kernels/cpu/native_expert.hpp"

#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace C = strata::kernels::cpu;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// Q8_K activations (ggml has a quantizer but no dequantizer for them): blocks of 256 = float d, int8 qs[256], int16 bsums[16]
static void dequant_q8k(const uint8_t* p, float* out, int n) {
    for (int b = 0; b < n / 256; ++b) {
        const uint8_t* blk = p + (size_t) b * 292;
        float d;
        std::memcpy(&d, blk, 4);
        for (int i = 0; i < 256; ++i) out[b * 256 + i] = d * (float) (int8_t) blk[4 + i];
    }
}

// dequantized row r of `rows` rows of `n` values, `row_bytes` apart
static std::vector<float> dequant_row(int type, const uint8_t* base, size_t row_bytes, int n, int r) {
    std::vector<float> v((size_t) n);
    ggml_get_type_traits((ggml_type) type)->to_float(base + (size_t) r * row_bytes, v.data(), n);
    return v;
}

static void test_pair(int gu_type, int d_type) {
    const int N = 512, FF = 256;
    C::NativeFmt f;
    std::string err;
    if (!C::native_fmt(gu_type, d_type, N, FF, f, err)) {
        std::printf("  native_fmt(%d, %d): %s\n", gu_type, d_type, err.c_str());
        ++g_fail;
        return;
    }
    std::mt19937_64 rng(42 + gu_type * 31 + d_type);
    std::vector<uint8_t> blob(f.bytes);
    for (size_t o = 0; o < blob.size(); o += 8) { uint64_t r = rng(); std::memcpy(blob.data() + o, &r, std::min<size_t>(8, blob.size() - o)); }
    auto set_scales = [&](size_t off, size_t row, size_t rows, size_t blk) {   // fp16 d ~ 0.02 at every block start
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o + blk <= row; o += blk) { blob[off + r * row + o] = 0x1f; blob[off + r * row + o + 1] = 0x25; }
    };
    set_scales(0, f.gu_row, FF, ggml_type_size((ggml_type) gu_type) * 256 / ggml_blck_size((ggml_type) gu_type));
    set_scales(f.up_off, f.gu_row, FF, ggml_type_size((ggml_type) gu_type) * 256 / ggml_blck_size((ggml_type) gu_type));
    set_scales(f.down_off, f.d_row, N, ggml_type_size((ggml_type) d_type) * 256 / ggml_blck_size((ggml_type) d_type));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(N);
    for (float& v : x) v = nd(rng) * 3.f;   // large enough that the clamp at 7 bites for some rows

    std::vector<uint8_t> act(C::kNativeActBytes), hq(C::kNativeHBytes);
    C::native_quant_act(f, x.data(), act.data());
    std::vector<float> xq(N);
    if (ggml_get_type_traits_cpu((ggml_type) gu_type)->vec_dot_type != GGML_TYPE_Q8_K) { std::puts("  gate/up activation is not Q8_K"); ++g_fail; return; }
    dequant_q8k(act.data(), xq.data(), N);

    const std::string tag = "types " + std::to_string(gu_type) + "/" + std::to_string(d_type);
    for (int mode = 0; mode < 2; ++mode) {
        C::NativeFmt fm = f;
        fm.swiglu_alpha = mode ? 1.702f : 0.f;
        fm.swiglu_limit = mode ? 7.f : 0.f;
        std::vector<float> ff(FF), ref(FF);
        float* ffp[1] = {ff.data()};
        const void* actp[1] = {act.data()};
        C::native_gu_rows(fm, blob.data(), actp, 1, ffp, 0, FF);
        int clamped = 0;
        for (int r = 0; r < FF; ++r) {
            const auto gw = dequant_row(gu_type, blob.data(), f.gu_row, N, r);
            const auto uw = dequant_row(gu_type, blob.data() + f.up_off, f.gu_row, N, r);
            double g = 0, u = 0;
            for (int i = 0; i < N; ++i) { g += (double) gw[i] * xq[i]; u += (double) uw[i] * xq[i]; }
            if (mode) {
                clamped += g > 7.0 || u > 7.0 || u < -7.0;
                g = std::fmin(g, 7.0);
                u = std::fmax(-7.0, std::fmin(u, 7.0));
                ref[r] = (float) (g / (1.0 + std::exp(-1.702 * g)) * (1.0 + u));
            } else {
                ref[r] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
        }
        double worst = 0, scale = 1e-6;
        for (int r = 0; r < FF; ++r) { worst = std::fmax(worst, std::fabs(ff[r] - ref[r])); scale = std::fmax(scale, std::fabs(ref[r])); }
        check(worst <= 2e-4 * scale + 1e-4, tag + (mode ? " swigluoai gate/up rows match the dequantized reference (max rel err " : " default silu*up rows match (max rel err ") +
                                              std::to_string(worst / scale) + (mode ? ", " + std::to_string(clamped) + " rows clamped)" : ")"));
        if (mode) {   // the AVX2 multi-token rows against the ggml path (same arithmetic up to summation order)
            C::NativeFmt fq = fm;
            fq.iq256_rows = true;
            std::vector<float> ff2(FF);
            float* ff2p[1] = {ff2.data()};
            C::native_gu_rows(fq, blob.data(), actp, 1, ff2p, 0, FF);
            double w2 = 0, sc2 = 1e-6;
            for (int r = 0; r < FF; ++r) { w2 = std::fmax(w2, std::fabs(ff2[r] - ff[r])); sc2 = std::fmax(sc2, std::fabs(ff[r])); }
            check(w2 <= 1e-5 * sc2 + 1e-6, tag + " iq256 gate/up rows match the ggml path (max rel diff " + std::to_string(w2 / sc2) + ")");
        }
        if (mode) {   // the down rows from this hidden
            C::native_quant_h(fm, ff.data(), hq.data());
            std::vector<float> hqf(FF);
            if (ggml_get_type_traits_cpu((ggml_type) d_type)->vec_dot_type != GGML_TYPE_Q8_K) { std::puts("  down activation is not Q8_K"); ++g_fail; return; }
            dequant_q8k(hq.data(), hqf.data(), FF);
            std::vector<float> out(N);
            float* op[1] = {out.data()};
            const void* hp[1] = {hq.data()};
            C::native_down_rows(fm, blob.data(), hp, 1, op, 0, N);
            double dworst = 0, dscale = 1e-6;
            for (int r = 0; r < N; ++r) {
                const auto dw = dequant_row(d_type, blob.data() + f.down_off, f.d_row, FF, r);
                double acc = 0;
                for (int i = 0; i < FF; ++i) acc += (double) dw[i] * hqf[i];
                dworst = std::fmax(dworst, std::fabs(out[r] - acc));
                dscale = std::fmax(dscale, std::fabs(acc));
            }
            check(dworst <= 2e-4 * dscale + 1e-4, tag + " down rows match the dequantized reference (max rel err " + std::to_string(dworst / dscale) + ")");
            C::NativeFmt fq = fm;
            fq.iq256_rows = true;
            std::vector<float> out2(N);
            float* op2[1] = {out2.data()};
            C::native_down_rows(fq, blob.data(), hp, 1, op2, 0, N);
            double d2 = 0;
            for (int r = 0; r < N; ++r) d2 = std::fmax(d2, std::fabs(out2[r] - out[r]));
            check(d2 <= 1e-5 * dscale + 1e-6, tag + " iq256 down rows match the ggml path (max rel diff " + std::to_string(d2 / dscale) + ")");
        }
    }
}

// A window of tokens through the row kernels at once equals the tokens one at a time (the verify step's CPU lane)
static void test_window(int gu_type, int d_type, bool iq256) {
    const int N = 512, FF = 256, NT = 3;
    C::NativeFmt f;
    std::string err;
    if (!C::native_fmt(gu_type, d_type, N, FF, f, err)) { ++g_fail; return; }
    f.swiglu_alpha = 1.702f;
    f.swiglu_limit = 7.f;
    f.iq256_rows = iq256;
    std::mt19937_64 rng(7 + gu_type + d_type);
    std::vector<uint8_t> blob(f.bytes);
    for (size_t o = 0; o < blob.size(); o += 8) { uint64_t r = rng(); std::memcpy(blob.data() + o, &r, std::min<size_t>(8, blob.size() - o)); }
    auto set_scales = [&](size_t off, size_t row, size_t rows, size_t blk) {
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o + blk <= row; o += blk) { blob[off + r * row + o] = 0x1f; blob[off + r * row + o + 1] = 0x25; }
    };
    set_scales(0, f.gu_row, FF, ggml_type_size((ggml_type) gu_type) * 256 / ggml_blck_size((ggml_type) gu_type));
    set_scales(f.up_off, f.gu_row, FF, ggml_type_size((ggml_type) gu_type) * 256 / ggml_blck_size((ggml_type) gu_type));
    set_scales(f.down_off, f.d_row, N, ggml_type_size((ggml_type) d_type) * 256 / ggml_blck_size((ggml_type) d_type));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(C::kNativeActBytes)), hq(NT, std::vector<uint8_t>(C::kNativeHBytes));
    std::vector<std::vector<float>> ff1(NT, std::vector<float>(FF)), ffn(NT, std::vector<float>(FF)), o1(NT, std::vector<float>(N)), on(NT, std::vector<float>(N));
    for (int t = 0; t < NT; ++t) {
        std::vector<float> x(N);
        for (float& v : x) v = nd(rng) * 2.f;
        C::native_quant_act(f, x.data(), act[t].data());
    }
    for (int t = 0; t < NT; ++t) {   // one token at a time
        const void* ap[1] = {act[t].data()};
        float* fp[1] = {ff1[t].data()};
        C::native_gu_rows(f, blob.data(), ap, 1, fp, 0, FF);
        C::native_quant_h(f, ff1[t].data(), hq[t].data());
        const void* hp[1] = {hq[t].data()};
        float* op[1] = {o1[t].data()};
        C::native_down_rows(f, blob.data(), hp, 1, op, 0, N);
    }
    const void* ap[NT];
    float* fp[NT];
    for (int t = 0; t < NT; ++t) { ap[t] = act[t].data(); fp[t] = ffn[t].data(); }
    C::native_gu_rows(f, blob.data(), ap, NT, fp, 0, FF);
    std::vector<std::vector<uint8_t>> hq2(NT, std::vector<uint8_t>(C::kNativeHBytes));
    const void* hp[NT];
    float* op[NT];
    for (int t = 0; t < NT; ++t) { C::native_quant_h(f, ffn[t].data(), hq2[t].data()); hp[t] = hq2[t].data(); op[t] = on[t].data(); }
    C::native_down_rows(f, blob.data(), hp, NT, op, 0, N);
    double w1 = 0, w2 = 0, s1 = 1e-6, s2 = 1e-6;
    for (int t = 0; t < NT; ++t) {
        for (int r = 0; r < FF; ++r) { w1 = std::fmax(w1, std::fabs(ff1[t][r] - ffn[t][r])); s1 = std::fmax(s1, std::fabs(ff1[t][r])); }
        for (int r = 0; r < N; ++r) { w2 = std::fmax(w2, std::fabs(o1[t][r] - on[t][r])); s2 = std::fmax(s2, std::fabs(o1[t][r])); }
    }
    check(w1 <= 1e-5 * s1 + 1e-6 && w2 <= 1e-5 * s2 + 1e-6,
          "types " + std::to_string(gu_type) + "/" + std::to_string(d_type) + (iq256 ? " iq256" : " ggml") +
              ": a 3-token window equals three single tokens (gate/up " + std::to_string(w1 / s1) + ", down " + std::to_string(w2 / s2) + ")");
}

int main() {
    if (!C::native_experts_available()) { std::puts("SKIP: no ggml-cpu path in this build"); return 77; }
    std::puts("MiniMax CPU expert path:");
    test_pair(17, 18);   // IQ2_XS gate/up, IQ3_XXS down (most layers of UD-Q2_K_XL)
    test_pair(17, 23);   // IQ2_XS, IQ4_XS down
    test_pair(18, 23);   // IQ3_XXS, IQ4_XS down (the last layer)
    for (bool q : {false, true}) {
        test_window(17, 18, q);
        test_window(17, 23, q);
        test_window(18, 23, q);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
