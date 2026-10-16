// Tetra port checks: llvq's published fingerprint, word <-> point round trip, the encoder's words and t on llvq_dump's
// blocks against a golden hash of the Rust output, and (with a reference file written by
// tools/quant_study/llvq_layer's llvq_dump) every word and t of that file.
//   llvq_tetra_test [reference.bin [threads]]
#include "blocks.hpp"
#include "strata/llvq/tetra_encoder.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

using namespace strata::llvq;

int main(int argc, char** argv) {
    Tetra tetra;
    if (tetra.fingerprint() != Tetra::PUBLISHED_FINGERPRINT) { std::printf("FAIL fingerprint %016llx\n", (unsigned long long)tetra.fingerprint()); return 1; }
    uint64_t z = 0x1234;
    for (int i = 0; i < 200000; ++i) {
        z = z * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t w = (z >> 17) & LABEL_MASK;
        auto back = tetra.encode_point(tetra.decode(w));
        if (!back || *back != w) { std::printf("FAIL round trip at word %llx\n", (unsigned long long)w); return 1; }
    }
    std::printf("ok fingerprint and 200000 round trips\n");
    {
        Encoder enc(tetra);
        Scratch sc;
        auto x = test::dump_blocks(test::GOLDEN_N, test::GOLDEN_SEED);
        test::Fnv h;
        for (size_t i = 0; i < test::GOLDEN_N; ++i) {
            TetraCode c = enc.encode(&x[24 * i], sc);
            h.bytes(&c.word, 8);
            h.bytes(&c.t, 8);
        }
        if (h.h != test::GOLDEN_FNV) { std::printf("FAIL golden hash %016llx\n", (unsigned long long)h.h); return 1; }
        std::printf("ok %zu blocks bit-identical to llvq (golden hash)\n", test::GOLDEN_N);
    }
    if (argc < 2) return 0;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    constexpr size_t REC = 24 * 8 + 8 + 8;
    const size_t n = raw.size() / REC;
    Encoder enc(tetra);
    const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
    std::atomic<size_t> word_bad{0}, t_bad{0}, next{0};
    std::atomic<long long> first_bad{-1};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int th = 0; th < threads; ++th)
        pool.emplace_back([&] {
            Scratch sc;
            for (size_t i; (i = next.fetch_add(1)) < n;) {
                const char* rec = raw.data() + i * REC;
                double x[24];
                uint64_t word;
                double t;
                std::memcpy(x, rec, 192);
                std::memcpy(&word, rec + 192, 8);
                std::memcpy(&t, rec + 200, 8);
                TetraCode c = enc.encode(x, sc);
                if (c.word != word) { ++word_bad; long long e = -1; first_bad.compare_exchange_strong(e, (long long)i); }
                if (std::memcmp(&c.t, &t, 8) != 0) ++t_bad;
            }
        });
    for (auto& t : pool) t.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu blocks: word mismatches %zu, t mismatches %zu (first %lld); %.0f blocks/s on %d threads\n", n,
                word_bad.load(), t_bad.load(), first_bad.load(), n / s, threads);
    return word_bad || t_bad ? 1 : 0;
}
