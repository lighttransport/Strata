// CPU-only byte oracle and interleaved packing timings. No Vulkan device is
// created; timings do not qualify end-to-end inference performance.
#include "fast/kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
namespace fast = strata::vulkan::fast;
int main() {
  try {
    std::mt19937 rng(3401);
    for (int type : {fast::F32, fast::Q2_K, fast::Q3_K, fast::Q4_K, fast::Q6_K, fast::Q5_0})
      for (uint32_t K : {256u, 768u, 2048u, 4096u})
        for (uint32_t M : {67u, 4096u}) {
          const auto layout = fast::layout_for(type, K, M);
          const size_t bytes = layout.head_bytes();
          std::vector<uint8_t> source(size_t(M) * fast::row_bytes(type, K) + 3);
          for (auto &v : source) v = uint8_t(rng());
          std::vector<uint8_t> original(bytes + 64, 0xa5), tiled(bytes + 64, 0xa5);
          fast::repack(layout, source.data() + 3, original.data() + 32);
          fast::repack_streamed(layout, source.data() + 3, tiled.data() + 32);
          if (original != tiled) throw std::runtime_error("streamed packing differs from original bytes");
          const uint32_t first_rows = M / 2;
          const size_t first_bytes = size_t(first_rows) * fast::row_bytes(type, K);
          const size_t second_bytes = size_t(M - first_rows) * fast::row_bytes(type, K);
          std::vector<uint8_t> second(second_bytes + 5);
          std::memcpy(second.data() + 5, source.data() + 3 + first_bytes, second_bytes);
          fast::repack_pair_streamed(layout, source.data() + 3, first_rows, second.data() + 5, tiled.data() + 32);
          if (original != tiled) throw std::runtime_error("paired packing differs from concatenated source");
          for (size_t i = 0; i < 32; ++i)
            if (original[i] != 0xa5 || original[bytes + 32 + i] != 0xa5)
              throw std::runtime_error("packing wrote outside its destination");
          std::cout << "{\"type\":" << type << ",\"K\":" << K << ",\"M\":" << M
                    << ",\"byte_identical\":true,\"pair_byte_identical\":true";
          if (M == 4096 && ((type == fast::Q2_K && K == 4096) ||
                           (type == fast::Q3_K && (K == 2048 || K == 4096)))) {
            std::vector<double> old_ms, new_ms;
            auto run = [&](bool streamed) {
              const auto start = std::chrono::steady_clock::now();
              if (streamed) fast::repack_streamed(layout, source.data() + 3, tiled.data() + 32);
              else fast::repack(layout, source.data() + 3, original.data() + 32);
              return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            };
            for (int trial = 0; trial < 9; ++trial) {
              if (trial % 2) {new_ms.push_back(run(true)); old_ms.push_back(run(false));}
              else {old_ms.push_back(run(false)); new_ms.push_back(run(true));}
            }
            if (original != tiled) throw std::runtime_error("repeated packing differs");
            std::sort(old_ms.begin(), old_ms.end()); std::sort(new_ms.begin(), new_ms.end());
            std::cout << ",\"original_median_ms\":" << old_ms[4] << ",\"streamed_median_ms\":" << new_ms[4];
            std::vector<uint8_t> joined(first_bytes + second_bytes);
            std::vector<double> joined_ms, paired_ms;
            auto pack_pair = [&](bool direct) {
              const auto start = std::chrono::steady_clock::now();
              if (direct)
                fast::repack_pair_streamed(layout, source.data() + 3, first_rows, second.data() + 5, tiled.data() + 32);
              else {
                std::memcpy(joined.data(), source.data() + 3, first_bytes);
                std::memcpy(joined.data() + first_bytes, second.data() + 5, second_bytes);
                fast::repack_streamed(layout, joined.data(), original.data() + 32);
              }
              return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            };
            pack_pair(false); pack_pair(true);
            for (int trial = 0; trial < 9; ++trial) {
              if (trial % 2) {paired_ms.push_back(pack_pair(true)); joined_ms.push_back(pack_pair(false));}
              else {joined_ms.push_back(pack_pair(false)); paired_ms.push_back(pack_pair(true));}
            }
            if (original != tiled) throw std::runtime_error("repeated paired packing differs");
            std::sort(joined_ms.begin(), joined_ms.end()); std::sort(paired_ms.begin(), paired_ms.end());
            std::cout << ",\"join_then_pack_median_ms\":" << joined_ms[4]
                      << ",\"direct_pair_median_ms\":" << paired_ms[4];
          }
          std::cout << "}\n" << std::flush;
        }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
