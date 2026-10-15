// Byte oracle for native Q2_K/Q3_K GPU packing, including row tails,
// unaligned sources, zero padding and destination sentinels. --cpu uses LVP.
#include "fast/kernels.hpp"
#include <array>
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
          for (uint32_t offset : {0u, 3u, 5u})
            for (uint32_t padding : {0u, 4u}) {
              const auto layout = fast::layout_for(type, K, M);
              std::vector<uint8_t> source(offset + size_t(M) * fast::row_bytes(type, K) + padding);
              for (auto &v : source) v = uint8_t(rng());
              std::vector<uint8_t> expected(layout.head_bytes() + 512, 0xa5), actual(expected.size());
              fast::repack(layout, source.data() + offset, expected.data() + 256);
              auto input = dev.buffer(source.size(), fast::Memory::Host);
              auto output = dev.buffer(expected.size(), fast::Memory::Host);
              auto counter = dev.buffer(16, fast::Memory::Host);
              const fast::View bounded_input(*input, 0, (source.size() + 3) / 4 * 4);
              dev.upload(*input, source.data(), source.size());
              auto run = [&](const char *checked_kernel) {
                dev.run([&](fast::Recorder &r) {
                  r.fill(fast::View(*output), output->size, 0xa5a5a5a5u);
                  r.fill(fast::View(*counter), counter->size, 0u);
                  if (checked_kernel)
                    r.dispatch(checked_kernel, {bounded_input, fast::View(*output, 256, layout.head_bytes()), fast::View(*counter)},
                               std::array<uint32_t, 4>{K / 256, M, layout.mp, offset}, layout.mp / 64, K / 256);
                  else
                    fast::repack_gpu(r, layout, bounded_input, fast::View(*output, 256, layout.head_bytes()), offset);
                  r.host_visible();
                });
                dev.download(*output, actual.data(), actual.size());
                if (expected != actual) throw std::runtime_error("GPU repack bytes or sentinels differ from CPU oracle");
              };
              // Check the shader's actual source indices before running its
              // normal variant. The descriptor exposes only the declared range.
              run(type == fast::Q2_K ? "repack_expert_q2k_checked" : "repack_expert_q3k_checked");
              if (counter->as<uint32_t>()[0]) throw std::runtime_error("GPU repack attempted an out-of-range source read");
              run(nullptr);
              std::cout << "{\"type\":" << type << ",\"K\":" << K << ",\"M\":" << M
                        << ",\"source_offset\":" << offset << ",\"source_padding\":" << padding
                        << ",\"checked_source_reads_in_range\":true,\"gpu_pack_byte_identical\":true}\n" << std::flush;
              if (type == fast::Q3_K && K == 2048 && M == 4096 && offset == 0 && padding == 0) {
                run("repack_expert_q3k_checked_legacy");
                if (counter->as<uint32_t>()[0] != 1u)
                  throw std::runtime_error("bounds negative control did not detect the legacy tail read");
                std::cout << "{\"legacy_tail_read_detected\":true,\"invalid_read_attempts\":1,\"unsafe_load_executed\":false}\n" << std::flush;
              }
            }
    return 0;
  } catch (const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
}
