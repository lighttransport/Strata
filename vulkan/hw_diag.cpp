// Model-free Vulkan VRAM/ALU/GEMM diagnostics. Use the guarded launcher on
// physical hardware; --cpu with an explicit LVP ICD is for checker validation.
#include "fast/device.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace fast = strata::vulkan::fast;
using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1ULL << 20;
constexpr uint32_t guard_words = 64, guard_value = 0xa5a5a5a5u;
struct Options {
  bool cpu = false, full = false, full_heap = false;
  std::string device, suite = "all", inject;
  uint32_t memory_mib = 64, chunk_kib = 1024, rounds = 1, compute_rounds = 2, iterations = 64;
  uint32_t compute_repeats = 1;
  uint32_t allocation_limit_mib = 0, logical_base_mib = 0, cpu_allocation_max_mib = 0;
  uint32_t seed = 0x53a9c217u, hold_ms = 0, max_submit_ms = 500;
};
uint32_t number(const std::string &s) {
  size_t used = 0;
  auto value = std::stoull(s, &used, 0);
  if (used != s.size() || value > UINT32_MAX || s.empty() || s[0] == '-')
    throw std::invalid_argument("invalid unsigned argument");
  return uint32_t(value);
}
Options options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "glm_vulkan_hw_diag --device NAME [--suite all|memory|compute] [--memory-mib 1..16384]\n"
                   " [--chunk-kib 4..4096] [--rounds 1..1024] [--compute-rounds 1..10000]\n"
                   " [--iterations 1..256] [--seed UINT32] [--hold-ms 0..1000]\n"
                   " [--compute-repeats 1..64] (read back every repetition)\n"
                   " [--full-heap --allocation-limit-mib MIB] (all allocatable memory; reports gaps)\n"
                   " [--logical-base-mib MIB] (software boundary test only)\n"
                   " [--cpu-allocation-max-mib MIB] (software OOM recovery test only)\n"
                   " [--max-submit-ms 1..1000] [--full-patterns] [--cpu]\n"
                   " [--inject memory|integer|float] (checker self-test; MUST fail)\n"
                   "Use tools/glm_vulkan_hw_diag.py for guarded hardware execution.\n";
      std::exit(0);
    }
    if (arg == "--cpu") {o.cpu = true; continue;}
    if (arg == "--full-patterns") {o.full = true; continue;}
    if (arg == "--full-heap") {o.full_heap = true; continue;}
    if (++i == argc) throw std::invalid_argument("missing argument value");
    const std::string v = argv[i];
    if (arg == "--device") o.device = v;
    else if (arg == "--suite") o.suite = v;
    else if (arg == "--inject") o.inject = v;
    else if (arg == "--memory-mib") o.memory_mib = number(v);
    else if (arg == "--chunk-kib") o.chunk_kib = number(v);
    else if (arg == "--rounds") o.rounds = number(v);
    else if (arg == "--compute-rounds") o.compute_rounds = number(v);
    else if (arg == "--compute-repeats") o.compute_repeats = number(v);
    else if (arg == "--allocation-limit-mib") o.allocation_limit_mib = number(v);
    else if (arg == "--logical-base-mib") o.logical_base_mib = number(v);
    else if (arg == "--cpu-allocation-max-mib") o.cpu_allocation_max_mib = number(v);
    else if (arg == "--iterations") o.iterations = number(v);
    else if (arg == "--seed") o.seed = number(v);
    else if (arg == "--hold-ms") o.hold_ms = number(v);
    else if (arg == "--max-submit-ms") o.max_submit_ms = number(v);
    else throw std::invalid_argument("unknown argument: " + arg);
  }
  if (o.suite != "all" && o.suite != "memory" && o.suite != "compute")
    throw std::invalid_argument("suite must be all, memory or compute");
  if (o.memory_mib < 1 || o.memory_mib > 16384 || o.chunk_kib < 4 || o.chunk_kib > 4096 ||
      o.chunk_kib % 4 || !o.rounds || o.rounds > 1024 || !o.compute_rounds || o.compute_rounds > 10000 ||
      !o.iterations || o.iterations > 256 || o.hold_ms > 1000 || !o.max_submit_ms || o.max_submit_ms > 1000)
    throw std::invalid_argument("argument outside bounded diagnostic range");
  if (!o.compute_repeats || o.compute_repeats > 64)
    throw std::invalid_argument("compute repeats must be 1..64");
  if (uint64_t(o.logical_base_mib) + o.memory_mib > 16384 || (!o.cpu && o.logical_base_mib))
    throw std::invalid_argument("logical range must fit 16 GiB; shifted ranges require --cpu");
  if (o.full_heap && (o.suite == "compute" || !o.allocation_limit_mib || o.allocation_limit_mib > o.memory_mib))
    throw std::invalid_argument("full heap requires memory suite and allocation limit within target");
  if (!o.full_heap && o.allocation_limit_mib)
    throw std::invalid_argument("allocation limit requires --full-heap");
  if (o.cpu_allocation_max_mib && (!o.cpu || !o.full_heap || o.cpu_allocation_max_mib > 64))
    throw std::invalid_argument("allocation OOM emulation requires --cpu --full-heap and maximum <=64 MiB");
  if (!o.inject.empty() && o.inject != "memory" && o.inject != "integer" && o.inject != "float")
    throw std::invalid_argument("unknown injection target");
  if ((o.inject == "memory" && o.suite == "compute") ||
      (!o.inject.empty() && o.inject != "memory" && o.suite == "memory"))
    throw std::invalid_argument("injection target is outside selected suite");
  if (o.cpu && !o.device.empty()) throw std::invalid_argument("--cpu selects llvmpipe; omit --device");
  if (!o.cpu && o.device.empty()) throw std::invalid_argument("physical hardware requires --device NAME");
  return o;
}
uint32_t mix(uint32_t v) {
  v = (v ^ (v >> 16)) * 0x7feb352du;
  v = (v ^ (v >> 15)) * 0x846ca68bu;
  return v ^ (v >> 16);
}
uint32_t pattern(uint32_t id, uint32_t word, uint32_t seed) {
  switch (id) {
  case 0: return 0;
  case 1: return UINT32_MAX;
  case 2: return 0xaaaaaaaau;
  case 3: return 0x55555555u;
  case 4: return word ^ seed;
  case 5: return ~(word ^ seed);
  case 6: return mix(word ^ seed);
  case 7: return uint32_t(1) << ((word + seed) % 32);
  case 8: return ~(uint32_t(1) << ((word + seed) % 32));
  default: return id < 41 ? uint32_t(1) << (id - 9) : ~(uint32_t(1) << (id - 41));
  }
}
[[noreturn]] void mismatch(const std::string &stage, uint64_t word, uint32_t want, uint32_t got) {
  std::cout << "{\"event\":\"mismatch\",\"stage\":" << std::quoted(stage) << ",\"logical_word\":" << word
            << ",\"expected\":" << want << ",\"actual\":" << got << ",\"xor\":" << (want ^ got) << "}\n";
  throw std::runtime_error("data mismatch in " + stage);
}
struct MemoryPush {uint32_t mode, count, offset, global_base, pattern, seed, inverted;};
struct Error {uint32_t count, index, expected, actual;};
class Diagnostic {
public:
  Options o;
  fast::Device dev;
  std::unique_ptr<fast::Buffer> host, copied, errors;
  uint64_t submissions = 0, cpu_words = 0;
  uint64_t memory_payload_bytes = 0, memory_allocation_bytes = 0, memory_padding_bytes = 0;
  uint64_t local_aux_bytes = 0;
  uint32_t allocation_retries = 0;
  double max_ms = 0;
  Clock::time_point started = Clock::now(), heartbeat = started;
  explicit Diagnostic(Options opt) : o(std::move(opt)),
      dev(o.cpu ? "llvmpipe" : o.device, (size_t(o.memory_mib) + 64) * MiB, o.cpu) {
    host = dev.buffer(std::max(size_t(o.chunk_kib) * 1024, size_t(65536) * 4) + 512, fast::Memory::Host);
    copied = dev.buffer(size_t(o.chunk_kib) * 1024 + 512, fast::Memory::Device);
    errors = dev.buffer(sizeof(Error), fast::Memory::Host);
    for (const auto *b : {host.get(), copied.get(), errors.get()})
      if (b->flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) local_aux_bytes += b->allocated;
    std::cout << "{\"event\":\"device\",\"name\":" << std::quoted(dev.name())
              << ",\"cpu_device\":" << (o.cpu ? "true" : "false") << ",\"memory_mib\":" << o.memory_mib
              << ",\"seed\":" << o.seed << ",\"max_submit_ms\":" << o.max_submit_ms << "}\n";
  }
  void progress(const std::string &stage, uint64_t word = 0, bool force = false) {
    auto now = Clock::now();
    if (force || now - heartbeat > std::chrono::seconds(2)) {
      std::cout << "{\"event\":\"progress\",\"stage\":" << std::quoted(stage) << ",\"logical_word\":" << word
                << ",\"submission\":" << submissions << ",\"elapsed_ms\":"
                << std::chrono::duration<double, std::milli>(now - started).count() << "}\n";
      heartbeat = now;
    }
  }
  template<class F> void run(const std::string &stage, F record) {
    progress(stage);
    dev.host_reset_queries(0, 2);
    dev.run([&](fast::Recorder &r) {r.timestamp(0); record(r); r.timestamp(1); r.host_visible();});
    auto ts = dev.read_timestamps(0, 2);
    const double ms = (ts[1] - ts[0]) / 1e6;
    ++submissions; max_ms = std::max(max_ms, ms);
    if (!(ms >= 0) || ms > o.max_submit_ms) {
      std::cout << "{\"event\":\"slow_submission\",\"stage\":" << std::quoted(stage)
                << ",\"submission\":" << submissions << ",\"gpu_ms\":" << ms << "}\n";
      throw std::runtime_error("submission exceeded diagnostic time budget");
    }
  }
  void sentinels(const fast::Buffer &b, size_t payload, const std::string &stage) {
    run(stage, [&](fast::Recorder &r) {
      r.copy(fast::View(b), fast::View(*host), 256);
      r.copy(fast::View(b, payload + 256), fast::View(*host, 256), 256);
    });
    for (uint32_t i = 0; i < 128; ++i)
      if (host->as<uint32_t>()[i] != guard_value) mismatch(stage, i, guard_value, host->as<uint32_t>()[i]);
  }
  void check_errors(uint32_t global, const std::string &stage) {
    const auto e = *errors->as<Error>();
    if (e.count) mismatch(stage, uint64_t(global) + e.index, e.expected, e.actual);
  }
  void memory() {
    std::vector<std::unique_ptr<fast::Buffer>> blocks;
    std::vector<uint32_t> sizes;
    std::vector<uint64_t> bases;
    size_t remaining = size_t(o.memory_mib) * MiB;
    const uint64_t allocation_limit = uint64_t(o.allocation_limit_mib) * MiB;
    uint64_t local_reserved = local_aux_bytes;
    const auto allocation_started = Clock::now();
    progress("memory_allocate", 0, true);
    while (remaining) {
      size_t bytes = std::min(remaining, 64 * MiB);
      std::unique_ptr<fast::Buffer> block;
      while (bytes >= 4096) {
        if (o.full_heap) {
          const uint64_t available = allocation_limit - std::min(allocation_limit, local_reserved);
          if (available <= 512) break;
          bytes = std::min(bytes, size_t((available - 512) & ~uint64_t(4095)));
          if (bytes < 4096) break;
        }
        try {
          if (o.cpu_allocation_max_mib && bytes > size_t(o.cpu_allocation_max_mib) * MiB)
            throw std::runtime_error("Vulkan allocate buffer failed: -2");
          block = dev.buffer(bytes + 512, fast::Memory::Device);
        } catch (const std::runtime_error &error) {
          // Recover only an allocation OOM, not device loss or mapping errors.
          // Every earlier submission completed before this probe.
          if (!o.full_heap || std::string(error.what()) != "Vulkan allocate buffer failed: -2") throw;
        }
        if (block && (!o.full_heap || block->allocated <= allocation_limit - local_reserved)) break;
        block.reset(); ++allocation_retries;
        bytes = (bytes / 2) & ~size_t(4095);
        if (Clock::now() - allocation_started > std::chrono::seconds(60))
          throw std::runtime_error("full heap allocation probe exceeded 60 seconds");
      }
      if (!block) {
        if (!o.full_heap) throw std::runtime_error("VRAM allocation failed");
        break;
      }
      if (!(block->flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        throw std::runtime_error("VRAM test allocation is not DEVICE_LOCAL");
      bases.push_back(uint64_t(o.logical_base_mib) * MiB / 4 + memory_payload_bytes / 4);
      memory_payload_bytes += bytes;
      memory_allocation_bytes += block->allocated;
      memory_padding_bytes += block->allocated - bytes;
      local_reserved += block->allocated;
      blocks.push_back(std::move(block));
      sizes.push_back(uint32_t(bytes / 4)); remaining -= bytes;
      std::cout << "{\"event\":\"memory_block\",\"index\":" << blocks.size() - 1
                << ",\"logical_begin_byte\":" << bases.back() * 4
                << ",\"logical_end_byte_exclusive\":" << bases.back() * 4 + bytes
                << ",\"payload_bytes\":" << bytes << ",\"allocation_bytes\":" << blocks.back()->allocated << "}\n";
      run("memory_guards", [&](fast::Recorder &r) {
        r.fill(fast::View(*blocks.back()), 256, guard_value);
        r.fill(fast::View(*blocks.back(), bytes + 256), 256, guard_value);
      });
    }
    if (blocks.empty()) throw std::runtime_error("No VRAM payload could be allocated");
    run("copy_guards", [&](fast::Recorder &r) {r.fill(fast::View(*copied), copied->size, guard_value);});
    std::cout << "{\"event\":\"memory_allocated\",\"payload_bytes\":" << memory_payload_bytes
              << ",\"target_bytes\":" << size_t(o.memory_mib) * MiB
              << ",\"unavailable_target_bytes\":" << remaining
              << ",\"full_heap\":" << (o.full_heap ? "true" : "false")
              << ",\"allocation_limit_bytes\":" << allocation_limit
              << ",\"local_aux_bytes\":" << local_aux_bytes
              << ",\"payload_allocation_bytes\":" << memory_allocation_bytes
              << ",\"guards_and_padding_bytes\":" << memory_padding_bytes
              << ",\"allocation_retries\":" << allocation_retries
              << ",\"emulated_allocation_max_mib\":" << o.cpu_allocation_max_mib
              << ",\"logical_begin_byte\":" << uint64_t(o.logical_base_mib) * MiB
              << ",\"logical_end_byte_exclusive\":" << uint64_t(o.logical_base_mib) * MiB + memory_payload_bytes
              << ",\"blocks\":" << blocks.size() << ",\"block_mib_max\":64,\"device_local\":true"
                 ",\"physical_addresses_known\":false,\"live_allocation_bytes\":" << dev.live_bytes() << "}\n";
    const uint32_t chunk = o.chunk_kib * 256;
    auto walk = [&](bool reverse, auto op) {
      for (size_t b = 0; b < blocks.size(); ++b) {
        const size_t bi = reverse ? blocks.size() - b - 1 : b;
        for (uint32_t offset = 0; offset < sizes[bi]; offset += chunk) {
          uint32_t count = std::min(chunk, sizes[bi] - offset);
          uint32_t global = uint32_t(bases[bi] + offset);
          progress("memory_window", global);
          op(*blocks[bi], offset, count, global);
        }
      }
    };
    for (uint32_t round = 0; round < o.rounds; ++round) {
      uint32_t seed = o.seed + round * 0x9e3779b9u;
      for (uint32_t pat = 0; pat < (o.full ? 73u : 9u); ++pat) {
        std::cout << "{\"event\":\"memory_pattern\",\"round\":" << round << ",\"pattern\":" << pat
                  << ",\"seed\":" << seed << "}\n";
        progress("memory_fill", 0, true);
        walk(false, [&](fast::Buffer &b, uint32_t offset, uint32_t count, uint32_t global) {
          run("memory_fill", [&](fast::Recorder &r) {
            r.dispatch("hw_memory", {fast::View(b), fast::View(*errors)},
                       MemoryPush{0, count, offset + guard_words, global, pat, seed, 0}, (count + 255) / 256);
          });
        });
        if (o.inject == "memory" && round == 0 && pat == 0) {
          run("inject_memory", [&](fast::Recorder &r) {
            r.fill(fast::View(*blocks[0], (guard_words + 17) * 4), 4, 1);
          });
        }
        if (o.hold_ms) std::this_thread::sleep_for(std::chrono::milliseconds(o.hold_ms));
        progress("memory_verify_invert", 0, true);
        walk(true, [&](fast::Buffer &b, uint32_t offset, uint32_t count, uint32_t global) {
          *errors->as<Error>() = {};
          run("memory_verify_invert", [&](fast::Recorder &r) {
            r.dispatch("hw_memory", {fast::View(b), fast::View(*errors)},
                       MemoryPush{2, count, offset + guard_words, global, pat, seed, 0}, (count + 255) / 256);
          });
          check_errors(global, "memory_verify_invert");
        });
        progress("memory_copy_readback", 0, true);
        walk(false, [&](fast::Buffer &b, uint32_t offset, uint32_t count, uint32_t global) {
          *errors->as<Error>() = {};
          run("memory_copy_readback", [&](fast::Recorder &r) {
            r.dispatch("hw_memory", {fast::View(b), fast::View(*errors)},
                       MemoryPush{1, count, offset + guard_words, global, pat, seed, 1}, (count + 255) / 256);
            r.copy(fast::View(b, (size_t(offset) + guard_words) * 4), fast::View(*copied, 256), size_t(count) * 4);
            r.dispatch("hw_memory", {fast::View(*copied), fast::View(*errors)},
                       MemoryPush{1, count, guard_words, global, pat, seed, 1}, (count + 255) / 256);
            r.copy(fast::View(*copied, 256), fast::View(*host), size_t(count) * 4);
          });
          check_errors(global, "memory_copy_readback");
          for (uint32_t i = 0; i < count; ++i) {
            uint32_t want = ~pattern(pat, global + i, seed), got = host->as<uint32_t>()[i];
            if (want != got) mismatch("memory_cpu_readback", uint64_t(global) + i, want, got);
          }
          cpu_words += count;
        });
        // An independent CPU-generated upload exercises host -> device traffic.
        // Fill every block before verification, retaining cross-block alias coverage.
        if (pat == 6) {
          progress("memory_cpu_upload", 0, true);
          walk(false, [&](fast::Buffer &b, uint32_t offset, uint32_t count, uint32_t global) {
            for (uint32_t i = 0; i < count; ++i) host->as<uint32_t>()[i] = pattern(pat, global + i, seed);
            run("memory_cpu_upload", [&](fast::Recorder &r) {
              r.copy(fast::View(*host), fast::View(b, (size_t(offset) + guard_words) * 4), size_t(count) * 4);
            });
          });
          walk(true, [&](fast::Buffer &b, uint32_t offset, uint32_t count, uint32_t global) {
            *errors->as<Error>() = {};
            run("memory_upload_verify", [&](fast::Recorder &r) {
              r.dispatch("hw_memory", {fast::View(b), fast::View(*errors)},
                         MemoryPush{1, count, offset + guard_words, global, pat, seed, 0}, (count + 255) / 256);
            });
            check_errors(global, "memory_upload_verify");
          });
        }
        for (size_t b = 0; b < blocks.size(); ++b) sentinels(*blocks[b], size_t(sizes[b]) * 4, "memory_sentinels");
        sentinels(*copied, size_t(o.chunk_kib) * 1024, "copy_sentinels");
      }
    }
  }
  void compute() {
    constexpr uint32_t count = 65536, n = 128, k = 256;
    auto input = dev.buffer(size_t(count) * 4, fast::Memory::Device);
    auto output = dev.buffer(size_t(count) * 4 + 512, fast::Memory::Device);
    auto a = dev.buffer(n * k * 4, fast::Memory::Device), b = dev.buffer(k * n * 4, fast::Memory::Device);
    auto c = dev.buffer(n * n * 4 + 512, fast::Memory::Device);
    // Independent integer dot oracle: coefficients are [-8,8]/4, hence
    // every product and sum is exactly representable without FMA assumptions.
    std::vector<int> ai(n * k), bi(k * n), expected_c(n * n);
    std::vector<uint32_t> initial(count), expected(count);
    for (uint32_t round = 0; round < o.compute_rounds; ++round) {
      uint32_t seed = o.seed + round * 0x9e3779b9u;
      progress("compute_round", round, true);
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t x = mix(i ^ seed); initial[i] = x;
        for (uint32_t j = 0; j < o.iterations; ++j) {
          x = std::rotl(x, 11) ^ (seed + j * 0x9e3779b9u);
          x = x * 1664525u + 1013904223u; x ^= x >> 13;
        }
        expected[i] = x;
      }
      std::memcpy(host->mapped, initial.data(), initial.size() * 4);
      run("integer_upload", [&](fast::Recorder &r) {
        r.copy(fast::View(*host), fast::View(*input), size_t(count) * 4);
        r.fill(fast::View(*output), output->size, guard_value);
      });
      for (uint32_t repeat = 0; repeat < o.compute_repeats; ++repeat) {
        run("uint32_alu", [&](fast::Recorder &r) {
          r.dispatch("hw_alu", {fast::View(*input), fast::View(*output)},
                     std::array<uint32_t, 4>{count, o.iterations, seed, guard_words}, count / 256);
        });
        if (o.inject == "integer" && round == 0 && repeat == 0)
          run("inject_integer", [&](fast::Recorder &r) {r.fill(fast::View(*output, (guard_words + 17) * 4), 4, expected[17] ^ 1u);});
        run("integer_readback", [&](fast::Recorder &r) {r.copy(fast::View(*output, 256), fast::View(*host), size_t(count) * 4);});
        for (uint32_t i = 0; i < count; ++i)
          if (host->as<uint32_t>()[i] != expected[i]) mismatch("uint32_alu", i, expected[i], host->as<uint32_t>()[i]);
        cpu_words += count;
        sentinels(*output, size_t(count) * 4, "integer_sentinels");
      }
      for (uint32_t i = 0; i < n * k; ++i) {ai[i] = int(mix(i ^ seed) % 17) - 8; host->as<float>()[i] = float(ai[i]) / 4;}
      run("gemm_upload_a", [&](fast::Recorder &r) {r.copy(fast::View(*host), fast::View(*a), n * k * 4);});
      for (uint32_t i = 0; i < k * n; ++i) {bi[i] = int(mix(i ^ (seed + 12345u)) % 17) - 8; host->as<float>()[i] = float(bi[i]) / 4;}
      run("gemm_upload_b", [&](fast::Recorder &r) {
        r.copy(fast::View(*host), fast::View(*b), k * n * 4); r.fill(fast::View(*c), c->size, guard_value);
      });
      for (uint32_t row = 0; row < n; ++row)
        for (uint32_t col = 0; col < n; ++col) {
          int dot = 0;
          for (uint32_t j = 0; j < k; ++j) dot += ai[row * k + j] * bi[j * n + col];
          expected_c[row * n + col] = dot;
        }
      for (uint32_t repeat = 0; repeat < o.compute_repeats; ++repeat) {
        run("fp32_gemm", [&](fast::Recorder &r) {
          r.dispatch("hw_gemm", {fast::View(*a), fast::View(*b), fast::View(*c)},
                     std::array<uint32_t, 3>{n, k, guard_words}, n / 16, n / 16);
        });
        if (o.inject == "float" && round == 0 && repeat == 0)
          run("inject_float", [&](fast::Recorder &r) {
            uint32_t bits = std::bit_cast<uint32_t>(float(expected_c[17]) / 16) ^ 1u;
            r.fill(fast::View(*c, (guard_words + 17) * 4), 4, bits);
          });
        run("gemm_readback", [&](fast::Recorder &r) {r.copy(fast::View(*c, 256), fast::View(*host), n * n * 4);});
        for (uint32_t i = 0; i < n * n; ++i) {
          uint32_t want = std::bit_cast<uint32_t>(float(expected_c[i]) / 16), got = host->as<uint32_t>()[i];
          // Both signed zeros are arithmetically correct.
          if (want != got && !((want & 0x7fffffffu) == 0 && (got & 0x7fffffffu) == 0)) mismatch("fp32_gemm", i, want, got);
        }
        cpu_words += n * n;
        sentinels(*c, n * n * 4, "gemm_sentinels");
      }
      std::cout << "{\"event\":\"compute_verified\",\"round\":" << round
                << ",\"uint32_elements\":" << count << ",\"alu_iterations\":" << o.iterations
                << ",\"repeats\":" << o.compute_repeats << ",\"gemm_n\":" << n << ",\"gemm_k\":" << k << "}\n";
    }
  }
};
int main(int argc, char **argv) {
  std::cout << std::unitbuf << std::setprecision(10);
  try {
    Diagnostic d(options(argc, argv));
    if (d.o.suite != "compute") d.memory();
    if (d.o.suite != "memory") d.compute();
    if (!d.o.inject.empty()) throw std::runtime_error("injected corruption was not detected");
    std::cout << "{\"event\":\"result\",\"passed\":true,\"suite\":" << std::quoted(d.o.suite)
              << ",\"cpu_device\":" << (d.o.cpu ? "true" : "false")
              << ",\"memory_mib\":" << (d.o.suite == "compute" ? 0 : d.o.memory_mib)
              << ",\"memory_rounds\":" << (d.o.suite == "compute" ? 0 : d.o.rounds)
              << ",\"full_heap\":" << (d.o.full_heap ? "true" : "false")
              << ",\"memory_payload_bytes\":" << d.memory_payload_bytes
              << ",\"memory_target_bytes\":" << (d.o.suite == "compute" ? 0 : uint64_t(d.o.memory_mib) * MiB)
              << ",\"allocation_limit_bytes\":" << uint64_t(d.o.allocation_limit_mib) * MiB
              << ",\"unavailable_target_bytes\":" << (d.o.suite == "compute" ? 0 : uint64_t(d.o.memory_mib) * MiB - d.memory_payload_bytes)
              << ",\"physical_range_complete\":false"
              << ",\"logical_begin_byte\":" << uint64_t(d.o.logical_base_mib) * MiB
              << ",\"logical_end_byte_exclusive\":" << uint64_t(d.o.logical_base_mib) * MiB + d.memory_payload_bytes
              << ",\"patterns_per_round\":" << (d.o.full ? 73 : 9)
              << ",\"compute_rounds\":" << (d.o.suite == "memory" ? 0 : d.o.compute_rounds)
              << ",\"compute_repeats\":" << (d.o.suite == "memory" ? 0 : d.o.compute_repeats)
              << ",\"gpu_submissions\":" << d.submissions << ",\"max_submit_ms\":" << d.max_ms
              << ",\"cpu_verified_words\":" << d.cpu_words << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cout << "{\"event\":\"result\",\"passed\":false,\"error\":" << std::quoted(e.what()) << "}\n";
    return 1;
  }
}
