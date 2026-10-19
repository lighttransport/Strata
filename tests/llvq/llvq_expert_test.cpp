// tests/llvq/llvq_expert_test.cpp - llvq Tetra experts on the CPU expert lane (native_expert.hpp, kLlvqType) against a
// double reference: W' rows from dequant_row, the input rotated with llvq::Rotation in double, SwiGLU with GLM's clamp.
//
//   llvq_expert_test UNIT.llvq [expert-in-unit=0]     (optional real unit; otherwise uses a synthetic expert)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/llvq/tetra_decode.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace llvq = strata::llvq;

int main(int argc, char** argv) {
    const bool synthetic = argc < 2;
    const int H = synthetic ? 64 : 4096, FF = synthetic ? 32 : 2048;
    const float limit = 10.0f;
    uint64_t seed = 123;
    uint32_t hdr[6] = {1, 0, 0, 1, 40, 0};
    std::vector<uint8_t> unit;
    const int expert = argc > 2 ? std::atoi(argv[2]) : 0;
    if (!synthetic) {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) { std::printf("cannot read unit file\n"); return 1; }
        unit.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (unit.size() < 40 || std::memcmp(unit.data(), "LLVQTET1", 8) != 0) { std::printf("not an llvq unit\n"); return 1; }
        std::memcpy(hdr, unit.data() + 8, 24);
        std::memcpy(&seed, unit.data() + 32, 8);
    }
    if (hdr[0] != 1 || hdr[5] != 0 || expert < 0 || uint32_t(expert) >= hdr[3]) {
        std::printf("invalid unit header or expert index\n"); return 1;
    }
    cpu::llvq_expert_init(seed);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kLlvqType, cpu::kLlvqType, H, FF, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 1; }
    f.swiglu_limit = limit;
    if (synthetic) {
        unit.resize(40 + f.bytes);
        llvq::SplitMix64 rng(seed);
        for (int projection = 0; projection < 3; ++projection) {
            const uint32_t rows = projection == 2 ? H : FF, cols = projection == 2 ? FF : H;
            auto* p = unit.data() + 40 + (projection == 0 ? 0 : projection == 1 ? f.up_off : f.down_off);
            std::memcpy(p, &rows, 4); std::memcpy(p + 4, &cols, 4);
            const double centroids[] = {0.8, 1.3}; std::memcpy(p + 8, centroids, 16);
            for (uint32_t row = 0; row < rows; ++row) {
                const float scale = float(0.1 + rng.next_f64());
                std::memcpy(p + 24 + row * 4, &scale, 4);
            }
            const size_t stride = (cols / 24 * 6 + 7) / 8 * 8;
            auto* words = p + 24 + rows * 4;
            for (uint32_t row = 0; row < rows; ++row)
                for (uint32_t block = 0; block < cols / 24; ++block) {
                    const uint64_t word = rng.next() & ((uint64_t(1) << 48) - 1);
                    std::memcpy(words + row * stride + block * 6, &word, 6);
                }
            auto* tail = words + rows * stride;
            for (uint32_t i = 0; i < rows * (cols % 24); ++i) {
                const float value = float(rng.next_gaussian() * 0.1);
                uint32_t bits; std::memcpy(&bits, &value, 4);
                const uint16_t bf16 = uint16_t(bits >> 16); std::memcpy(tail + i * 2, &bf16, 2);
            }
        }
    }
    if ((unit.size() - 40) / f.bytes != hdr[3] || (unit.size() - 40) % f.bytes) {
        std::printf("unit size mismatch\n"); return 1;
    }
    const uint8_t* blob = unit.data() + 40 + size_t(expert) * f.bytes;
    for (int projection = 0; projection < 3; ++projection) {
        const auto* p = blob + (projection == 0 ? 0 : projection == 1 ? f.up_off : f.down_off);
        uint32_t shape[2]; std::memcpy(shape, p, sizeof shape);
        if (shape[0] != uint32_t(projection == 2 ? H : FF) || shape[1] != uint32_t(projection == 2 ? FF : H)) {
            std::printf("geometry\n"); return 1;
        }
    }

    // double reference
    llvq::Tetra tetra;
    const auto T = llvq::DecodeTables::build(tetra);
    llvq::TetraMatrix mg, mu, md;
    mg.parse(blob); mu.parse(blob + f.up_off); md.parse(blob + f.down_off);
    if (mg.rows != uint32_t(FF) || mg.cols != uint32_t(H) || md.rows != uint32_t(H) || md.cols != uint32_t(FF)) { std::printf("geometry\n"); return 1; }
    auto rows_of = [&](const llvq::TetraMatrix& m) {
        std::vector<float> w(size_t(m.rows) * m.cols);
        for (uint32_t r = 0; r < m.rows; ++r) llvq::dequant_row(T, m, r, w.data() + size_t(r) * m.cols);
        return w;
    };
    const auto wg = rows_of(mg), wu = rows_of(mu), wd = rows_of(md);
    const llvq::Rotation q_in(H, seed), q_ff(FF, seed);

    std::mt19937 rng(20261019);
    std::normal_distribution<float> nd;
    int fails = 0;
    for (int nt : {1, 2, 3, 4, 5, 8}) {
        std::vector<std::vector<float>> x(nt, std::vector<float>(H));
        for (auto& v : x) for (auto& e : v) e = nd(rng);
        // engine path: activations, gate/up rows in 3 row ranges, hidden, down rows in 2 ranges
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(cpu::kNativeActBytes)), hq(nt, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(nt, std::vector<float>(FF)), out(nt, std::vector<float>(H));
        std::vector<const void*> a(nt), h(nt);
        std::vector<float*> pff(nt), pout(nt);
        for (int t = 0; t < nt; ++t) {
            cpu::native_quant_act(f, x[t].data(), act[t].data());
            a[t] = act[t].data(); h[t] = hq[t].data(); pff[t] = ff[t].data(); pout[t] = out[t].data();
        }
        for (int part = 0; part < 3; ++part) cpu::native_gu_rows(f, blob, a.data(), nt, pff.data(), FF * part / 3, FF * (part + 1) / 3);
        for (int t = 0; t < nt; ++t) cpu::native_quant_h(f, ff[t].data(), hq[t].data());
        for (int part = 0; part < 2; ++part) cpu::native_down_rows(f, blob, h.data(), nt, pout.data(), H * part / 2, H * (part + 1) / 2);
        // Grouping routed tokens into a job must not change an expert's arithmetic.
        for (int t = 0; t < nt; ++t) {
            std::vector<float> single_ff(FF), single_out(H);
            std::vector<uint8_t> single_h(cpu::kNativeHBytes);
            const void* single_a = a[t];
            const void* single_h_ptr = single_h.data();
            float* single_ff_ptr = single_ff.data();
            float* single_out_ptr = single_out.data();
            cpu::native_gu_rows(f, blob, &single_a, 1, &single_ff_ptr, 0, FF);
            cpu::native_quant_h(f, single_ff.data(), single_h.data());
            cpu::native_down_rows(f, blob, &single_h_ptr, 1, &single_out_ptr, 0, H);
            if (std::memcmp(single_ff.data(), ff[t].data(), FF * sizeof(float)) ||
                std::memcmp(single_out.data(), out[t].data(), H * sizeof(float))) {
                std::printf("batch invariance FAIL: batch=%d token=%d\n", nt, t);
                ++fails;
            }
        }
        // reference
        double worst = 0;
        for (int t = 0; t < nt; ++t) {
            std::vector<double> xr(x[t].begin(), x[t].end());
            q_in.apply(xr.data());
            std::vector<double> hid(FF);
            for (int r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int c = 0; c < H; ++c) { g += double(wg[size_t(r) * H + c]) * xr[c]; u += double(wu[size_t(r) * H + c]) * xr[c]; }
                g = std::fmin(g, limit); u = std::fmax(-limit, std::fmin(u, limit));
                hid[size_t(r)] = g / (1 + std::exp(-g)) * u;
            }
            q_ff.apply(hid.data());
            double num = 0, den = 0;
            for (int r = 0; r < H; ++r) {
                double y = 0;
                for (int c = 0; c < FF; ++c) y += double(wd[size_t(r) * FF + c]) * hid[size_t(c)];
                num += (out[t][r] - y) * (out[t][r] - y); den += y * y;
            }
            worst = std::fmax(worst, std::sqrt(num / std::fmax(den, 1e-30)));
        }
        const bool ok = worst < 1e-5;
        std::printf("expert %d, %d token(s): relative error %.3e %s\n", expert, nt, worst, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    std::printf("llvq_expert_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
