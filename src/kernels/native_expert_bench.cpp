// src/kernels/native_expert_bench.cpp - the grouped native experts on real GGUF rows: one AMD layout against
// another, bit for bit, and each one's time.
//
//     build/native_expert_bench <shard1.gguf> <layer[,layer...]> <groups> <tokens per group> <ref mode> <mode> [iters]
//
// Modes are STRATA_EXP_MODE values.  A group is one expert (distinct blobs, ~2 MB each, so G >= 8 does not fit
// in L2); its entries read distinct tokens of an 8-token window.  The output must be bitwise equal to the
// reference mode's: the verify window's text depends on it.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/model.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 7) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> <layer[,layer]> <groups> <tokens> <ref mode> <mode> [iters] [activation-scale=1]\n");
        return 2;
    }
    strata::core::ModelArtifact artifact(argv[1]);
    const auto& model = artifact.descriptor();
    std::vector<int> layers;
    {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int G = std::atoi(argv[3]), T = std::atoi(argv[4]), mref = std::atoi(argv[5]), mode = std::atoi(argv[6]);
    const int iters = argc > 7 ? std::atoi(argv[7]) : 200;
    char* scale_end = nullptr;
    const float activation_scale = argc > 8 ? std::strtof(argv[8], &scale_end) : 1.f;
    const int NTOK = 8;
    if (T < 1 || T > NTOK || G < 1 || G > model.experts || iters < 1 ||
        !std::isfinite(activation_scale) || activation_scale <= 0 || (scale_end && *scale_end)) {
        std::fprintf(stderr, "tokens 1..8, groups 1..expert count, iterations >= 1, positive finite activation scale\n"); return 2;
    }
    const int64_t H = model.hidden;
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int l : layers) {
        if (l < 0 || l >= (int)(model.layers.size() + model.draft_layers.size())) {
            std::printf("layer %d: outside model\n", l); ++failures; continue;
        }
        const auto& layer = l < (int)model.layers.size() ? model.layers.at(l)
                            : model.draft_layers.at(l - model.layers.size());
        const int64_t FF = layer.intermediate, FF_FULL = FF;
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& [name, tensor] : artifact.tensors()) {
            const auto& ti = *tensor.tensor;
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        }
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        f.swiglu_limit = layer.swiglu_limit;
        cpu::NativeFmt ff;    // the full expert's geometry, for strides into the GGUF
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF_FULL, ff, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        const size_t dsz = f.bytes - f.down_off, dsz_full = ff.bytes - ff.down_off;
        const size_t drow = dsz / H, drow_full = dsz_full / H;
        std::vector<uint8_t> blobs((size_t) G * f.bytes);
        for (int g = 0; g < G; ++g) {
            const size_t E = (size_t) ((g * 37 + 5) % model.experts);
            uint8_t* b = blobs.data() + (size_t) g * f.bytes;
            std::memcpy(b, artifact.at(t[0]->name).data() + E * ff.up_off, f.up_off);
            std::memcpy(b + f.up_off, artifact.at(t[1]->name).data() + E * ff.up_off, f.up_off);
            for (int64_t r = 0; r < H; ++r)
                std::memcpy(b + f.down_off + r * drow, artifact.at(t[2]->name).data() + E * dsz_full + r * drow_full, drow);
        }
        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NTOK * H);
        for (auto& v : x) v = nd(rng) * activation_scale;
        const int NE = G * T;
        std::vector<unsigned long long> ptr(G);
        std::vector<int32_t> start(G + 1), dst(NE), tok(NE);
        auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        L.swiglu_limit = layer.swiglu_limit;
        std::printf("shape layer=%d hidden=%lld intermediate=%lld experts=%lld swiglu_limit=%g\n",
                    l, (long long)H, (long long)FF, (long long)model.experts, (double)L.swiglu_limit);
        uint8_t* dblob;
        void *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc((void**) &dblob, blobs.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NTOK * H / 32 * 36);
        cudaMalloc(&dscr, K::native_expert_scratch_bytes(NE, FF));
        cudaMalloc((void**) &dout, (size_t) NE * H * 4);
        cudaMalloc((void**) &dptr, (size_t) G * 8);
        cudaMalloc((void**) &dstart, (size_t) (G + 1) * 4);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, (size_t) NE * 4);
        cudaMalloc((void**) &dtok, (size_t) NE * 4);
        for (int g = 0; g < G; ++g) {
            ptr[g] = (unsigned long long) (dblob + (size_t) g * f.bytes);
            start[g] = g * T;
            for (int j = 0; j < T; ++j) {
                dst[g * T + j] = g * T + j;
                tok[g * T + j] = (g * 3 + j) % NTOK;
            }
        }
        start[G] = NE;
        cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dptr, ptr.data(), (size_t) G * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, s);
        auto run = [&](int m, int phase) {
            K::native_expert_set_mode(m, phase);
            K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s);
        };
        std::vector<float> ref((size_t) NE * H), got((size_t) NE * H);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mref, 0);
        cudaMemcpyAsync(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mode, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error\n", l); return 1; }
        size_t ndiff = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (!std::isfinite(ref[i]) || !std::isfinite(got[i]) ||
                std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double) ref[i] - got[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        auto time = [&](int m, int phase) {
            for (int i = 0; i < 10; ++i) run(m, phase);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) run(m, phase);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double gu_b = (double) G * 2 * f.up_off, d_b = (double) G * dsz;
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)   // interleaved, the minimum: the clocks wander between runs
            for (int k = 0; k < 2; ++k)
                for (int p = 0; p < 3; ++p) us[k][p] = std::fmin(us[k][p], time(k ? mode : mref, p));
        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s) | %s\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    ndiff ? "DIFFERS" : "bitwise equal");
        if (ndiff) {
            std::printf("          %zu of %zu outputs differ, worst rel %.2e\n", ndiff, ref.size(), worst);
            ++failures;
        }
        K::native_expert_set_mode(-1, 0);
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
    std::printf("native_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
