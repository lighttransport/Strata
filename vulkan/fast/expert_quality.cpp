// Whole resident expert against the CPU canonical path, not just dequant dots.
#include "fast/kernels.hpp"
#include "model_checks.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
namespace fast = strata::vulkan::fast;
namespace cpu = strata::kernels::cpu;
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: glm_vulkan_fast_expert_quality MODEL");
    strata::core::ModelArtifact artifact(argv[1]);
    const auto &model = strata::vulkan::checked_model(artifact);
    fast::Device dev("", 1ULL << 30);
    const auto &G = artifact.at("blk.3.ffn_gate_exps.weight"), &U = artifact.at("blk.3.ffn_up_exps.weight"),
               &D = artifact.at("blk.3.ffn_down_exps.weight");
    const auto gl = fast::layout_for(fast::Q2_K, 4096, 2048), dl = fast::layout_for(fast::Q3_K, 2048, 4096);
    const size_t bytes = 2 * gl.head_bytes() + dl.head_bytes();
    std::mt19937 rng(7641); std::normal_distribution<float> normal(0.f, 0.5f);
    cpu::NativeFmt fmt; std::string error;
    if (!cpu::native_fmt(G.tensor->type, D.tensor->type, 4096, 2048, fmt, error)) throw std::runtime_error(error);
    fmt.canon = true; fmt.swiglu_limit = model.layers[3].swiglu_limit;
    for (int expert : {5, 77}) {
      const auto *gate = G.data() + size_t(expert) * G.bytes / 144;
      const auto *up = U.data() + size_t(expert) * U.bytes / 144;
      const auto *down = D.data() + size_t(expert) * D.bytes / 144;
      std::vector<uint8_t> packed(bytes);
      fast::repack(gl, gate, packed.data()); fast::repack(gl, up, packed.data() + gl.head_bytes());
      fast::repack(dl, down, packed.data() + 2 * gl.head_bytes());
      auto arena = dev.buffer(bytes, fast::Memory::Device, true); dev.upload(*arena, packed.data(), bytes);
      const uint64_t address = dev.address(*arena);
      auto table = dev.buffer(8, fast::Memory::Device); dev.upload(*table, &address, 8);
      uint32_t hit_data[192] = {1, 0};
      for (uint32_t t = 0; t < 4; ++t) hit_data[33 + t] = 0x3f800000u; // each token uses this expert
      auto hits = dev.buffer(sizeof(hit_data), fast::Memory::Device); dev.upload(*hits, hit_data, sizeof(hit_data));
      for (uint32_t T : {1u, 4u}) {
        std::vector<float> input(T * 4096), expected(T * 4096), hidden(2048);
        for (float &v : input) v = normal(rng);
        std::vector<uint8_t> act(fmt.act_bytes), hq(fmt.h_bytes);
        for (uint32_t t = 0; t < T; ++t) {
          cpu::native_quant_act(fmt, input.data() + t * 4096, act.data());
          const void *ap[] = {act.data()}; float *hp[] = {hidden.data()};
          cpu::native_gu_rows(fmt, gate, ap, 1, hp, 0, 2048, up);
          cpu::native_quant_h(fmt, hidden.data(), hq.data());
          const void *qp[] = {hq.data()}; float *op[] = {expected.data() + t * 4096};
          cpu::native_down_rows(fmt, gate, qp, 1, op, 0, 4096, down);
        }
        auto x = dev.buffer(T * 4096 * 4, fast::Memory::Device); dev.upload(*x, input.data(), input.size() * 4);
        auto qx = dev.buffer(T * 4096 * 4, fast::Memory::Device);
        auto gu = dev.buffer(T * 4096 * 4, fast::Memory::Device);
        auto h = dev.buffer(T * 2048 * 4, fast::Memory::Device);
        auto qh = dev.buffer(T * 2048 * 4, fast::Memory::Device);
        auto sums = dev.buffer(T * 4096 / 16 * 4, fast::Memory::Device);
        auto out = dev.buffer(T * 4096 * 4, fast::Memory::Host);
        struct Push {uint32_t K, rows, parts, Mp, part_bytes, slot_offset, ksw, side, xofs, xhit, yhit, hofs, columns, xcol, ycol;};
        Push gp{4096, 2048, 2, gl.mp, uint32_t(gl.head_bytes()), 0, 4, uint32_t(gl.chunk_bytes()), 0, 0, T * 4096, 0, T, 4096, 4096};
        Push dp{2048, 4096, 1, dl.mp, uint32_t(dl.head_bytes()), uint32_t(2 * gl.head_bytes()), 4, uint32_t(dl.chunk_bytes()), 0, T * 2048, T * 4096, 0, T, 2048, 4096};
        struct {uint32_t mode, n, uofs, tokens; float limit;} swi{0, 2048, 0, T, fmt.swiglu_limit};
        for (bool rounded : {false, true}) {
          dev.run([&](fast::Recorder &r) {
            if (rounded) r.dispatch("q8k_round", {fast::View(*x), fast::View(*qx), fast::View(*hits)},
                                     std::array<uint32_t, 4>{T * 4096, 0, 0, T}, T * 4096 / 256);
            const auto &a = rounded ? *qx : *x;
            r.dispatch("xsum16", {fast::View(a), fast::View(*sums)}, std::array<uint32_t, 2>{T * 4096 / 16, 0}, (T * 4096 / 16 + 255) / 256);
            r.dispatch("gemv_expert_q2k", {fast::View(*table), fast::View(a), fast::View(*gu), fast::View(*hits), fast::View(*sums)}, gp, 64, 1);
            r.dispatch(rounded ? "expert_mix_canon" : "expert_mix", {fast::View(*gu), fast::View(*h), fast::View(*hits)}, swi, T * 2048 / 256);
            if (rounded) r.dispatch("q8k_round", {fast::View(*h), fast::View(*qh), fast::View(*hits)},
                                     std::array<uint32_t, 4>{T * 2048, 0, 2048, T}, T * 2048 / 256);
            const auto &b = rounded ? *qh : *h;
            r.dispatch("xsum16", {fast::View(b), fast::View(*sums)}, std::array<uint32_t, 2>{T * 2048 / 16, 0}, (T * 2048 / 16 + 255) / 256);
            r.dispatch("gemv_expert_q3k", {fast::View(*table), fast::View(b), fast::View(*out), fast::View(*hits), fast::View(*sums)}, dp, 64, 1);
            r.host_visible();
          });
          double err = 0, norm = 0;
          for (size_t i = 0; i < expected.size(); ++i) {
            const float actual = out->as<float>()[i];
            if (!std::isfinite(actual)) throw std::runtime_error("nonfinite resident expert output");
            err += (double(actual) - expected[i]) * (double(actual) - expected[i]); norm += double(expected[i]) * expected[i];
          }
          const double rel = std::sqrt(err / std::max(norm, 1e-30));
          std::cout << "{\"expert\":" << expert << ",\"columns\":" << T << ",\"q8_grid\":" << rounded << ",\"rel_l2_cpu_canonical\":" << rel << "}\n" << std::flush;
          if (rounded && !(rel < 2e-4)) throw std::runtime_error("Q8-grid expert parity failed");
        }
      }
    }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
