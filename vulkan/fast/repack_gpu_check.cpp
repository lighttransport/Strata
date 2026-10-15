// Byte oracle for native Q2_K/Q3_K GPU packing, including row tails,
// unaligned sources, zero padding and destination sentinels. --cpu uses LVP.
#include "fast/kernels.hpp"
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
namespace fast = strata::vulkan::fast;
int main(int argc, char **argv) {
  try {
    const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu) throw std::invalid_argument("usage: glm_vulkan_fast_repack_gpu_check [--cpu]");
    fast::Device dev(cpu ? "llvmpipe" : "", 64ULL << 20, cpu);
    std::mt19937 rng(4133);
    for (int type : {fast::Q2_K, fast::Q3_K})
      for (uint32_t K : {256u, 768u, 2048u, 4096u})
        for (uint32_t M : {67u, 4096u})
          for (uint32_t offset : {0u, 3u, 5u}) {
            const auto layout = fast::layout_for(type, K, M);
            std::vector<uint8_t> source(offset + size_t(M) * fast::row_bytes(type, K) + 4);
            for (auto &v : source) v = uint8_t(rng());
            std::vector<uint8_t> expected(layout.head_bytes() + 512, 0xa5), actual(expected.size());
            fast::repack(layout, source.data() + offset, expected.data() + 256);
            auto input = dev.buffer(source.size(), fast::Memory::Host);
            auto output = dev.buffer(expected.size(), fast::Memory::Host);
            dev.upload(*input, source.data(), source.size());
            dev.run([&](fast::Recorder &r) {
              r.fill(fast::View(*output), output->size, 0xa5a5a5a5u);
              fast::repack_gpu(r, layout, fast::View(*input), fast::View(*output, 256, layout.head_bytes()), offset);
              r.host_visible();
            });
            dev.download(*output, actual.data(), actual.size());
            if (expected != actual) throw std::runtime_error("GPU repack bytes or sentinels differ from CPU oracle");
            std::cout << "{\"type\":" << type << ",\"K\":" << K << ",\"M\":" << M
                      << ",\"source_offset\":" << offset << ",\"gpu_pack_byte_identical\":true}\n" << std::flush;
          }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
