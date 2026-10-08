#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/prefill/partition.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <iostream>
#include <random>
#include <source_location>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n;
    Buffer(size_t n) : n(n) {
        ck(cudaMalloc(&p, n * sizeof(T)));
        ck(cudaMemset(p, 0, n * sizeof(T)));
    }
    Buffer(const std::vector<T> &v) : Buffer(v.size()) { put(v); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T> &v) {
        if (v.size() != n)
            throw std::runtime_error("size");
        ck(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
void near(const std::vector<float> &a, const std::vector<float> &b, float tolerance = 1e-4,
          const std::source_location site = std::source_location::current()) {
    if (a.size() != b.size())
        throw std::runtime_error("size");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) / (1 + std::abs(b[i])) > tolerance)
            throw std::runtime_error("scaled parity error at " + std::to_string(i) +
                " from line " + std::to_string(site.line()) +
                " actual_bits=" + std::to_string(std::bit_cast<uint32_t>(a[i])) +
                " expected_bits=" + std::to_string(std::bit_cast<uint32_t>(b[i])));
}
void check_vector_gather(bool benchmark) {
    const char* original = std::getenv("STRATA_GLM_MLA_VECTOR_GATHER");
    struct Restore {
        bool present;
        std::string value;
        ~Restore() { if (present) setenv("STRATA_GLM_MLA_VECTOR_GATHER", value.c_str(), 1);
                     else unsetenv("STRATA_GLM_MLA_VECTOR_GATHER"); }
    } restore{original != nullptr, original ? original : ""};
    constexpr int stride = 2052, cache_rows = 8192, guard = 16;
    for (int latent : {7, 32, 512}) for (int queries : {1, 3, 64}) {
        std::vector<float> cache(cache_rows * latent);
        for (size_t i = 0; i < cache.size(); ++i) {
            uint32_t bits = 0x3f000000u | ((uint32_t)(i * 2654435761u) & 0x807fffffu);
            if (i % 1024 == 0) bits = 0x80000000u;
            if (i % 1024 == 1) bits = 0x7fc12345u;
            if (i % 1024 == 2) bits = 0x7f800000u;
            cache[i] = std::bit_cast<float>(bits);
        }
        Buffer<float> dc(cache);
        for (int full : {0, 1}) {
            const int lengths[] = {0, 3, 127, 2051, 2052};
            std::vector<int> counts(queries), ids(queries * stride, INT_MAX);
            const float sentinel = std::bit_cast<float>(0x7fc56789u);
            std::vector<float> expected((size_t)queries * stride * latent + 2 * guard, sentinel);
            for (int q = 0; q < queries; ++q) {
                counts[q] = full ? stride : lengths[q % 5];
                for (int key = 0; key < stride; ++key) {
                    float* dest = expected.data() + guard + ((size_t)q * stride + key) * latent;
                    if (key < counts[q]) {
                        ids[q * stride + key] = (key * 17 + q * 193) % cache_rows;
                        std::copy_n(cache.data() + ids[q * stride + key] * latent, latent, dest);
                    } else std::fill_n(dest, latent, 0.f);
                }
            }
            Buffer<int> di(ids), dn(counts);
            std::vector<float> initial(expected.size(), sentinel);
            Buffer<float> output(initial);
            for (int mode : {0, 1}) {
                setenv("STRATA_GLM_MLA_VECTOR_GATHER", mode ? "1" : "0", 1);
                output.put(initial);
                k::glm_mla_gather(dc.p, di.p, dn.p, output.p + guard, queries, stride, latent, nullptr);
                const auto actual = output.get();
                if (std::memcmp(actual.data(), expected.data(), expected.size() * 4))
                    throw std::runtime_error("vector MLA gather changes bits or allocation guards");
                if (benchmark && latent == 512 && full) {
                    cudaEvent_t begin, end;
                    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
                    std::vector<float> times;
                    for (int trial = 0; trial < 8; ++trial) {
                        ck(cudaEventRecord(begin));
                        for (int repetition = 0; repetition < 8; ++repetition)
                            k::glm_mla_gather(dc.p, di.p, dn.p, output.p + guard, queries, stride, latent, nullptr);
                        ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                        float ms;
                        ck(cudaEventElapsedTime(&ms, begin, end));
                        if (trial) times.push_back(ms / 8);
                    }
                    std::sort(times.begin(), times.end());
                    std::cout << "MLA_VECTOR_BENCH queries=" << queries << " stride=" << stride
                              << " latent=" << latent << " mode=" << mode << " median_ms=" << times[3] << '\n';
                    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
                }
            }
        }
    }
    std::cout << "MLA_VECTOR bits=identical CPU_oracle=passed latent=7,32,512 queries=1,3,64 counts=0,3,127,2051,2052 guards=unchanged\n";
}

void check_batched_resident() {
    for (int nt : {1, 2, 3, 8}) for (int mode : {0, 1, 2, 3}) {
        std::vector<int> ids(nt * 8), starts{0}, dest, token_rows;
        std::vector<unsigned long long> lookup(288), pointers;
        for (int e = 0; e < 288; ++e)
            if (mode && (mode == 3 || e % 3)) lookup[e] = 0x10000ULL + (mode == 2 ? e % 4 : e) * 4096;
        for (int j = 0; j < nt * 8; ++j) ids[j] = mode == 3 ? j * 17 % 288 : (j * 17 + j / 8 * 23) % 13;
        for (int id : ids) if (lookup[id] && std::find(pointers.begin(), pointers.end(), lookup[id]) == pointers.end())
            pointers.push_back(lookup[id]);
        for (auto address : pointers) {
            for (int j = 0; j < nt * 8; ++j) if (lookup[ids[j]] == address) {
                dest.push_back(j); token_rows.push_back(j / 8);
            }
            starts.push_back(dest.size());
        }
        Buffer<int> di(ids), ds(std::vector<int>(67, -99)), dd(std::vector<int>(66, -99)),
                    dt(std::vector<int>(66, -99)), dc(std::vector<int>(3, -99));
        Buffer<unsigned long long> dl(lookup), dp(std::vector<unsigned long long>(66, 999));
        k::glm_resident_routes_batch(di.p, dl.p, dp.p + 1, ds.p + 1, dd.p + 1, dt.p + 1, dc.p + 1, nt, nullptr);
        const auto ap = dp.get(); const auto as = ds.get(); const auto ad = dd.get();
        const auto at = dt.get(); const auto ac = dc.get();
        if (ac[1] != int(pointers.size()) || ac.front() != -99 || ac.back() != -99 ||
            !std::equal(pointers.begin(), pointers.end(), ap.begin() + 1) ||
            !std::equal(starts.begin(), starts.end(), as.begin() + 1) ||
            !std::equal(dest.begin(), dest.end(), ad.begin() + 1) ||
            !std::equal(token_rows.begin(), token_rows.end(), at.begin() + 1) ||
            ap.front() != 999 || ap.back() != 999 || as.front() != -99 || as.back() != -99 ||
            ad.front() != -99 || ad.back() != -99 || at.front() != -99 || at.back() != -99)
            throw std::runtime_error("batched resident metadata differs from independent grouping or overwrites guards");
    }
    for (int nt : {1, 2, 3, 8}) for (int width : {7, 4096}) {
        const int n = nt * 8 * width;
        std::vector<float> cpu(n), primary(n), remote(n), weights(nt * 8), expected(nt * width), initial(expected.size());
        std::vector<int> owners(nt * 8);
        for (int i = 0; i < n; ++i) { cpu[i] = std::sin(i * 0.13f); primary[i] = std::cos(i * 0.11f); remote[i] = std::sin(i * 0.07f); }
        for (int i = 0; i < nt * 8; ++i) { owners[i] = i % 3; weights[i] = (i % 7 - 3) * 0.12345f; }
        for (int i = 0; i < nt * width; ++i) {
            initial[i] = expected[i] = std::cos(i * 0.17f);
            float sum = 0.f;
            for (int j = 0; j < 8; ++j) {
                const int route = i / width * 8 + j;
                const auto &src = owners[route] == 1 ? primary : owners[route] == 2 ? remote : cpu;
                volatile float product = weights[route] * src[route * width + i % width];
                sum += product;
            }
            expected[i] += sum;
        }
        initial.insert(initial.begin(), 999.f); initial.push_back(999.f);
        Buffer<float> c(cpu), p(primary), r(remote), w(weights), out(initial); Buffer<int> own(owners);
        k::glm_moe_reduce_batch(c.p, p.p, r.p, own.p, w.p, out.p + 1, width, nt, nullptr);
        const auto actual = out.get();
        if (actual.front() != 999.f || actual.back() != 999.f ||
            std::memcmp(actual.data() + 1, expected.data(), expected.size() * sizeof(float)))
            throw std::runtime_error("batched reduction changed multiply/add bits or overwrote guards");
    }
    std::cout << "Batched resident metadata and reduction bits/guards passed\n";
}
// Replay the same graph across pool boundaries and backwards positions (MTP rollback).
void check_device_positions() {
    constexpr int dim = 32, pool = 4, capacity = 64, heads = 2, stride = 20;
    for (int width : {1, 2, 3, 8}) {
        std::vector<float> input(width * dim), ape_data(pool * dim), dots_data(capacity / pool * width * heads);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * .13f);
        for (size_t i = 0; i < ape_data.size(); ++i) ape_data[i] = std::cos(float(i) * .17f);
        for (size_t i = 0; i < dots_data.size(); ++i) dots_data[i] = std::sin(float(i) * .07f);
        Buffer<float> x(input), ape(ape_data), dots(dots_data), weights(std::vector<float>(width * heads, .5f));
        Buffer<float> pk(pool * dim), pg(pool * dim), pooled(capacity / pool * dim), cache(capacity * dim);
        Buffer<float> rk(pool * dim), rg(pool * dim), rp(capacity / pool * dim), rc(capacity * dim);
        Buffer<float> scores(capacity / pool * width), rs(capacity / pool * width);
        Buffer<int> ids(width * stride), counts(width), ri(width * stride), rn(width), position(1);
        cudaStream_t stream; ck(cudaStreamCreate(&stream));
        cudaGraph_t graph; cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        k::glm_cache_rows(x.p, cache.p, dim, width, position.p, stream);
        k::glm_index_prepare(x.p, x.p, ape.p, pk.p, pg.p, pooled.p, 0, width, pool, dim, stream, position.p);
        k::glm_index_reduce(dots.p, weights.p, scores.p, heads, capacity / pool, width, 0, pool, dim, stream, position.p);
        k::glm_index_select_batch(scores.p, ids.p, counts.p, capacity / pool, 0, width, pool, 16, stride, stream, position.p);
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int pos : {0, 1, 3, 4, 7, 8, 15, 12, 31, 40, 56}) {
            k::glm_decode_position(position.p, pos, stream);
            ck(cudaGraphLaunch(executable, stream));
            ck(cudaMemcpyAsync(rc.p + pos * dim, x.p, width * dim * 4, cudaMemcpyDeviceToDevice, stream));
            k::glm_index_prepare(x.p, x.p, ape.p, rk.p, rg.p, rp.p, pos, width, pool, dim, stream);
            k::glm_index_reduce(dots.p, weights.p, rs.p, heads, capacity / pool, width, pos, pool, dim, stream);
            k::glm_index_select_batch(rs.p, ri.p, rn.p, capacity / pool, pos, width, pool, 16, stride, stream);
            ck(cudaStreamSynchronize(stream));
            auto equal = [](auto &a, auto &b) {
                auto av = a.get(), bv = b.get();
                if (std::memcmp(av.data(), bv.data(), av.size() * sizeof(av[0])))
                    throw std::runtime_error("device-position graph replay differs from eager execution");
            };
            equal(pk, rk); equal(pg, rg); equal(pooled, rp); equal(cache, rc);
            equal(scores, rs); equal(ids, ri); equal(counts, rn);
        }
        ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    }
    std::cout << "device-position graph replay: bitwise parity\n";
}
int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    if (argc > 1 && !benchmark) return 2;
    try {
        check_device_positions();
        check_vector_gather(benchmark);
        check_batched_resident();
        for (int primary = 1; primary < 18; ++primary) {
            int counts[2]{};
            for (int group = 0; group < 18; ++group) {
                const bool a = strata::prefill::owns_expert_group(group, 0, 2, primary);
                const bool b = strata::prefill::owns_expert_group(group, 1, 2, primary);
                if (a == b || (primary == 9 && a != (group % 2 == 0)) ||
                    !strata::prefill::owns_expert_group(group, 0, 1, primary) ||
                    strata::prefill::owns_expert_group(group, 1, 1, primary))
                    throw std::runtime_error("prefill partition duplicates or loses an expert group");
                ++counts[a ? 0 : 1];
            }
            if (counts[0] != primary || counts[1] != 18 - primary)
                throw std::runtime_error("prefill partition count");
        }
        std::mt19937 rng(53);
        std::uniform_real_distribution<float> rf(-1, 1);
        auto random = [&](int n) {
            std::vector<float> v(n);
            for (auto &x : v)
                x = rf(rng);
            return v;
        };
        for (int width : {7, 4096}) for (int mode : {0, 1, 2, 3}) {
            auto cpu = random(8 * width), primary = random(8 * width), remote = random(8 * width);
            auto weights = random(8), initial = random(width), expected = initial;
            std::vector<int> owners(8);
            for (int j = 0; j < 8; ++j) owners[j] = mode == 3 ? j % 3 : mode;
            for (int col = 0; col < width; ++col) {
                float sum = 0.f;
                for (int j = 0; j < 8; ++j) {
                    const auto &source = owners[j] == 1 ? primary : owners[j] == 2 ? remote : cpu;
                    volatile float product = weights[j] * source[j * width + col];
                    sum += product;
                }
                expected[col] += sum;
            }
            Buffer<float> c(cpu), p(primary), r(remote), w(weights), out(initial);
            Buffer<int> ownership(owners);
            k::glm_moe_reduce(c.p, p.p, r.p, ownership.p, w.p, out.p, width, nullptr);
            near(out.get(), expected, 0);
        }
        // Split-K mHC projection against an independent double-precision dot.
        for (int resident : {0, 1, 4, 8}) {
            std::vector<int> ids{17, 4, 287, 9, 25, 63, 128, 0};
            std::vector<unsigned long long> lookup(288), expected;
            for (int j = 0; j < resident; ++j) lookup[ids[j]] = 0x10000ULL + j * 0x4000;
            Buffer<int> di(ids), starts(9), destination(8), tokens(8), count(1);
            Buffer<unsigned long long> dl(lookup), pointers(8);
            k::glm_resident_routes(di.p, dl.p, pointers.p, starts.p, destination.p, tokens.p, count.p, nullptr);
            if (count.get()[0] != resident) throw std::runtime_error("resident lookup count");
            auto s = starts.get(), d = destination.get(), t = tokens.get();
            auto p = pointers.get();
            for (int j = 0; j < resident; ++j)
                if (s[j] != j || s[j + 1] != j + 1 || d[j] != j || t[j] != 0 || p[j] != lookup[ids[j]])
                    throw std::runtime_error("resident lookup metadata");
        }
        for (int width : {7, 513, 8192, 16384}) {
            auto x = random(width), w = random(24 * width);
            Buffer<float> dx(x), dw(w), result(24), scratch(24 * 32);
            k::glm_hc_project(dx.p, dw.p, result.p, scratch.p, width, nullptr);
            auto actual = result.get();
            for (int row = 0; row < 24; ++row) {
                double sum = 0, magnitude = 0;
                for (int col = 0; col < width; ++col) {
                    const double term = double(x[col]) * w[row * width + col];
                    sum += term; magnitude += std::abs(term);
                }
                if (!std::isfinite(actual[row]) || std::abs(actual[row] - sum) > 2e-6 * magnitude)
                    throw std::runtime_error("HC split projection differs from double oracle");
            }
        }
        // Tensor-core route scratch starts at row zero, even for nonzero group offsets.
        for (int width : {7, 4096}) for (int offset : {0, 3, 6, 9}) for (unsigned mask : {0u, 0x15u, 0x8u}) {
            constexpr int first = 1, rows = 3, total = 20;
            const int groups = mask ? std::popcount(mask) : 4;
            std::vector<int> experts;
            for (int e = 0; e < (mask ? 5 : 4); ++e) if (!mask || (mask & (1u << e))) experts.push_back(first + e);
            std::vector<int> bounds{0, 2, 2, 7, 8, 17, total}, source(total), dest(total);
            for (int j = 0; j < total; ++j) { source[j] = (j * 7) % total; dest[j] = (j * 3) % total; }
            auto input = random(total * width);
            std::vector<float> dense(groups * rows * width), expected(total * width, -13.f);
            for (int e = 0; e < groups; ++e) for (int row = 0; row < rows; ++row) {
                const int route = bounds[experts[e]] + offset + row;
                if (route >= bounds[experts[e] + 1]) continue;
                for (int col = 0; col < width; ++col) {
                    const float value = input[source[route] * width + col];
                    dense[(e * rows + row) * width + col] = value;
                    expected[dest[route] * width + col] = value;
                }
            }
            Buffer<float> x(input), ref(dense), y(std::vector<float>(total * width, -13.f));
            Buffer<int> b(bounds), s(source), d(dest);
            Buffer<uint16_t> half(dense.size()), expected_half(dense.size());
            k::glm_gather_expert_f16(x.p, half.p, b.p, s.p, first, groups, offset, rows, width, nullptr, mask);
            k::glm_f16(ref.p, expected_half.p, dense.size(), nullptr);
            if (half.get() != expected_half.get()) throw std::runtime_error("batched expert gather padding/offset");
            k::glm_scatter_expert_rows(ref.p, y.p, b.p, d.p, first, groups, offset, rows, width, nullptr, mask);
            near(y.get(), expected, 0);
        }
        for (int width : {7, 4096}) {
            const int rows = 19, begin = 3, count = 11;
            auto x = random(rows * width);
            std::vector<int> routes(rows);
            for (int i = 0; i < rows; ++i) routes[i] = (i * 7) % rows;
            std::vector<float> gathered(count * width), scattered(rows * width, -13.f);
            for (int row = 0; row < count; ++row) {
                std::copy_n(x.data() + routes[begin + row] * width, width, gathered.data() + row * width);
                std::copy_n(gathered.data() + row * width, width, scattered.data() + routes[begin + row] * width);
            }
            Buffer<float> source(x), expected_gather(gathered), destination(std::vector<float>(rows * width, -13.f));
            Buffer<uint16_t> half(count * width), expected_half(count * width);
            Buffer<int> ids(routes);
            k::glm_gather_f16(source.p, half.p, ids.p, begin, count, width, nullptr);
            k::glm_f16(expected_gather.p, expected_half.p, count * width, nullptr);
            if (half.get() != expected_half.get()) throw std::runtime_error("FP16 route gather differs");
            k::glm_scatter_rows(expected_gather.p, destination.p, ids.p, begin, count, width, nullptr);
            near(destination.get(), scattered, 0);
        }
        // Route transport must preserve NaN payloads, signed zero, subnormals,
        // and untouched rows, including a partial group with a nonzero offset.
        for (int width : {7, 4096}) {
            const int rows = 19, begin = 3, count = 11;
            std::vector<int> routes(rows);
            for (int i = 0; i < rows; ++i) routes[i] = (i * 7) % rows;
            std::vector<float> input(rows * width), sentinel(rows * width, -13.f);
            const unsigned patterns[] = {0x00000000, 0x80000000, 0x00000001,
                                         0x7f800000, 0xff800000, 0x7fc01234, 0x3f800000};
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = std::bit_cast<float>(patterns[i % 7]);
            Buffer<float> source(input), grouped(sentinel), scattered(sentinel);
            Buffer<int> ids(routes);
            k::glm_copy_route_rows(source.p, grouped.p, ids.p, begin, count, width, true, nullptr);
            auto expected = sentinel;
            for (int row = begin; row < begin + count; ++row)
                std::memcpy(expected.data() + row * width, input.data() + routes[row] * width, width * 4);
            auto actual = grouped.get();
            if (std::memcmp(actual.data(), expected.data(), actual.size() * 4))
                throw std::runtime_error("route gather changed bits or untouched rows");
            k::glm_copy_route_rows(grouped.p, scattered.p, ids.p, begin, count, width, false, nullptr);
            expected = sentinel;
            for (int row = begin; row < begin + count; ++row)
                std::memcpy(expected.data() + routes[row] * width, input.data() + routes[row] * width, width * 4);
            actual = scattered.get();
            if (std::memcmp(actual.data(), expected.data(), actual.size() * 4))
                throw std::runtime_error("route scatter changed bits or untouched rows");
        }
        // Decode's warp Sinkhorn keeps the serial prefill arithmetic order,
        // including saturated logits and different iteration/epsilon settings.
        for (int iterations : {1, 2, 20, 80})
            for (float amplitude : {0.f, 1.f, 20.f, 100.f}) {
                constexpr int B = 7, H = 64;
                auto projected = random(24 * B);
                for (auto &v : projected) v *= amplitude;
                Buffer<float> streams(random(4 * H * B)), p(projected), base(random(24)),
                    scale(std::vector<float>{.1, .2, .3}), batch(24 * B), single(24 * B),
                    output(H * B), reference(H * B);
                k::glm_mhc_read_batch(streams.p, p.p, base.p, scale.p, batch.p, output.p,
                                      H, iterations, 1e-6, B, nullptr);
                for (int t = 0; t < B; ++t)
                    k::glm_mhc_read(streams.p + t * 4 * H, p.p + t * 24, base.p, scale.p,
                                    single.p + t * 24, reference.p + t * H, H, iterations, 1e-6, nullptr);
                near(batch.get(), single.get(), 0);
                near(output.get(), reference.get(), 0);
            }
        for (int B : {1, 3, 4, 5, 9}) {
            int n = 128;
            Buffer<float> x(random(n * B)), w(random(n * 4)), h(random(n * 3)), seq(h.get()), a(n * B),
                b(n * B);
            k::glm_conv_batch(x.p, w.p, h.p, a.p, n, 4, B, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_conv(x.p + t * n, w.p, seq.p, b.p + t * n, n, 4, nullptr);
            near(a.get(), b.get(), 0);
            near(h.get(), seq.get(), 0);
            int H = 64;
            Buffer<float> r(random(4 * H * B)), p(random(24 * B)), base(random(24)),
                scale(std::vector<float>{.1, .2, .3}), c(24 * B), cs(24 * B), y(random(H * B)), read(H * B),
                reads(H * B), write(4 * H * B), writes(4 * H * B);
            k::glm_mhc_read_batch(r.p, p.p, base.p, scale.p, c.p, read.p, H, 20, 1e-6, B, nullptr);
            k::glm_mhc_write_batch(r.p, c.p, y.p, write.p, H, B, nullptr);
            for (int t = 0; t < B; ++t) {
                k::glm_mhc_read(r.p + t * 4 * H, p.p + t * 24, base.p, scale.p, cs.p + t * 24,
                                reads.p + t * H, H, 20, 1e-6, nullptr);
                k::glm_mhc_write(r.p + t * 4 * H, cs.p + t * 24, y.p + t * H, writes.p + t * 4 * H, H,
                                 nullptr);
            }
            near(c.get(), cs.get(), 0);
            near(read.get(), reads.get(), 0);
            near(write.get(), writes.get(), 0);
            int heads = 2, dim = 128;
            Buffer<float> raw(random(B * heads * dim)), bias(random(heads * dim)),
                neg_a(std::vector<float>(heads, -.8)), decay(B * heads * dim), decays(B * heads * dim),
                norm(random(dim)), out(B * heads * dim), outs(B * heads * dim);
            k::glm_kda_gate_batch(raw.p, bias.p, neg_a.p, decay.p, heads, dim, -5, B, nullptr);
            k::glm_kda_output_batch(raw.p, raw.p, norm.p, out.p, heads, dim, 1e-5, B, nullptr);
            for (int t = 0; t < B; ++t) {
                k::glm_kda_gate(raw.p + t * heads * dim, bias.p, neg_a.p, decays.p + t * heads * dim, heads,
                                dim, -5, nullptr);
                k::glm_kda_output(raw.p + t * heads * dim, raw.p + t * heads * dim, norm.p,
                                  outs.p + t * heads * dim, heads, dim, 1e-5, nullptr);
            }
            near(decay.get(), decays.get(), 0);
            near(out.get(), outs.get(), 0);
        }
        for (int B : {1, 9, 64, 129, 8192}) {
            int H = B == 64 ? 64 : 2, D = 128, N = H * D;
            Buffer<float> state(random(H * D * D)), seq(state.get()), q(random(B * N)), key(random(B * N)),
                v(random(B * N)), decay(std::vector<float>(B * N, -.02f)), beta(random(B * H)), out(B * N),
                ref(B * N);
            const auto initial = state.get();
            for (int t = 0; t < B; t += 64)
                k::glm_kda_chunk(state.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                 beta.p + t * H, out.p + t * N, H, D, std::min(64, B - t), nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_kda_step(seq.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                beta.p + t * H, ref.p + t * N, H, D, nullptr);
            near(state.get(), seq.get(), 1e-6);
            near(out.get(), ref.get(), 1e-6);
            const auto expected_state = state.get(), expected_out = out.get();
            // Hoist only token-independent normalization and gates. Every
            // recurrence output and final state must retain the original bits.
            for (int columns : {32, 64, 128}) {
                Buffer<float> prepared_state(initial), prepared_out(B * N), pk(key.get()),
                    pg(decay.get()), pb(beta.get()), qi(B * H);
                k::glm_kda_prepare(q.p, pk.p, pg.p, pb.p, qi.p, H, B, nullptr);
                for (int t = 0; t < B; t += 64)
                    k::glm_kda_chunk(prepared_state.p, q.p + t * N, pk.p + t * N, v.p + t * N,
                        pg.p + t * N, pb.p + t * H, prepared_out.p + t * N, H, D, std::min(64, B - t),
                        nullptr, columns, 1, qi.p + t * H);
                const auto actual_state = prepared_state.get(), actual_out = prepared_out.get();
                if (std::memcmp(actual_state.data(), expected_state.data(), expected_state.size() * 4) ||
                    std::memcmp(actual_out.data(), expected_out.data(), expected_out.size() * 4))
                    throw std::runtime_error("prepared KDA column kernel differs from original float bits");
                if (B == 64 && benchmark) {
                    cudaEvent_t begin, end;
                    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
                    float total_ms = 0;
                    for (int repeat = 0; repeat < 50; ++repeat) {
                        // Restore preparation inputs outside the timed range.
                        ck(cudaMemcpyAsync(pk.p, key.p, (size_t)B * N * 4, cudaMemcpyDeviceToDevice));
                        ck(cudaMemcpyAsync(pg.p, decay.p, (size_t)B * N * 4, cudaMemcpyDeviceToDevice));
                        ck(cudaMemcpyAsync(pb.p, beta.p, (size_t)B * H * 4, cudaMemcpyDeviceToDevice));
                        ck(cudaEventRecord(begin));
                        k::glm_kda_prepare(q.p, pk.p, pg.p, pb.p, qi.p, H, B, nullptr);
                        k::glm_kda_chunk(prepared_state.p, q.p, pk.p, v.p, pg.p, pb.p, prepared_out.p,
                            H, D, B, nullptr, columns, 1, qi.p);
                        ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                        float ms; ck(cudaEventElapsedTime(&ms, begin, end)); total_ms += ms;
                    }
                    std::cout << "KDA_PREPARED_TIME columns=" << columns << " heads=" << H
                              << " tokens=" << B << " ms=" << total_ms / 50 << '\n';
                    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
                }
            }
            for (int columns : {32, 64}) {
                Buffer<float> split(initial), split_out(B * N);
                for (int t = 0; t < B; t += 64)
                    k::glm_kda_chunk(split.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                     beta.p + t * H, split_out.p + t * N, H, D, std::min(64, B - t), nullptr, columns);
                const auto actual_state = split.get(), actual_out = split_out.get();
                if (std::memcmp(actual_state.data(), expected_state.data(), expected_state.size() * 4) ||
                    std::memcmp(actual_out.data(), expected_out.data(), expected_out.size() * 4))
                    throw std::runtime_error("KDA split columns differ from original float bits");
                if (B == 64 && benchmark) {
                    cudaEvent_t begin, end;
                    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
                    for (int variant : {128, columns}) {
                        ck(cudaEventRecord(begin));
                        for (int repeat = 0; repeat < 50; ++repeat)
                            k::glm_kda_chunk(split.p, q.p, key.p, v.p, decay.p, beta.p, split_out.p,
                                             H, D, B, nullptr, variant);
                        ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                        float ms; ck(cudaEventElapsedTime(&ms, begin, end));
                        std::cout << "KDA_TIME columns=" << variant << " heads=" << H << " tokens=" << B << " ms=" << ms / 50 << '\n';
                    }
                    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
                }
            }
        }
        // Row-partitioned dots deliberately change addition order. Keep the
        // exact-bit tests above and enforce a separate FP32 error budget here.
        for (int B : {1, 9, 64, 129, 8192}) {
            const int H = B == 64 ? 64 : 2, D = 128, N = H * D;
            Buffer<float> original(random(H * D * D)), q(random(B * N)), key(random(B * N)),
                v(random(B * N)), decay(std::vector<float>(B * N, -.02f)), beta(random(B * H)), expected(B * N);
            const auto initial = original.get();
            for (int t = 0; t < B; t += 64)
                k::glm_kda_chunk(original.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                 beta.p + t * H, expected.p + t * N, H, D, std::min(64, B - t), nullptr);
            const auto expected_state = original.get(), expected_out = expected.get();
            for (int parts : {4, 8}) {
                Buffer<float> candidate(initial), out(B * N);
                for (int t = 0; t < B; t += 64)
                    k::glm_kda_chunk(candidate.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                     beta.p + t * H, out.p + t * N, H, D, std::min(64, B - t), nullptr, 128, parts);
                near(candidate.get(), expected_state, 1e-5);
                near(out.get(), expected_out, 1e-5);
                Buffer<float> prepared(initial), prepared_key(key.get()), prepared_decay(decay.get()),
                    prepared_beta(beta.get()), qi(B * H), prepared_out(B * N);
                k::glm_kda_prepare(q.p, prepared_key.p, prepared_decay.p, prepared_beta.p, qi.p, H, B, nullptr);
                for (int t = 0; t < B; t += 64)
                    k::glm_kda_chunk(prepared.p, q.p + t * N, prepared_key.p + t * N, v.p + t * N,
                                     prepared_decay.p + t * N, prepared_beta.p + t * H, prepared_out.p + t * N,
                                     H, D, std::min(64, B - t), nullptr, 128, parts, qi.p + t * H);
                const auto prepared_state = prepared.get(), row_state = candidate.get();
                const auto prepared_values = prepared_out.get(), row_values = out.get();
                if (std::memcmp(prepared_state.data(), row_state.data(), row_state.size() * sizeof(float)) ||
                    std::memcmp(prepared_values.data(), row_values.data(), row_values.size() * sizeof(float)))
                    throw std::runtime_error("KDA preparation changed FP32 output or state bits");
                if (B == 64 && benchmark) {
                    cudaEvent_t begin, end;
                    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
                    for (int variant : {1, parts}) {
                        ck(cudaEventRecord(begin));
                        for (int repeat = 0; repeat < 50; ++repeat)
                            k::glm_kda_chunk(candidate.p, q.p, key.p, v.p, decay.p, beta.p, out.p,
                                             H, D, B, nullptr, 128, variant);
                        ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                        float ms; ck(cudaEventElapsedTime(&ms, begin, end));
                        std::cout << "KDA_ROWS parts=" << variant << " heads=" << H << " tokens=" << B
                                  << " ms=" << ms / 50 << '\n';
                    }
                    std::vector<float> samples;
                    for (int repeat = 0; repeat < 20; ++repeat) {
                        ck(cudaMemcpyAsync(prepared_key.p, key.p, B * N * sizeof(float), cudaMemcpyDeviceToDevice));
                        ck(cudaMemcpyAsync(prepared_decay.p, decay.p, B * N * sizeof(float), cudaMemcpyDeviceToDevice));
                        ck(cudaMemcpyAsync(prepared_beta.p, beta.p, B * H * sizeof(float), cudaMemcpyDeviceToDevice));
                        ck(cudaEventRecord(begin));
                        k::glm_kda_prepare(q.p, prepared_key.p, prepared_decay.p, prepared_beta.p, qi.p, H, B, nullptr);
                        k::glm_kda_chunk(prepared.p, q.p, prepared_key.p, v.p, prepared_decay.p,
                                         prepared_beta.p, prepared_out.p, H, D, B, nullptr, 128, parts, qi.p);
                        ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                        float ms; ck(cudaEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
                    }
                    std::sort(samples.begin(), samples.end());
                    std::cout << "KDA_PREPARED parts=" << parts << " heads=" << H << " tokens=" << B
                              << " ms=" << samples[samples.size() / 2] << " exact=1 includes_prepare=1\n";
                    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
                }
            }
        }
        for (int pos : {0, 1, 2, 3, 4, 4093}) {
            int B = 9, dim = 32, pool = 4, total = (pos + B) / pool;
            Buffer<float> keys(random(B * dim)), gates(random(B * dim)), ape(random(pool * dim)),
                pk(random(pool * dim)), pg(random(pool * dim)), sk(pk.get()), sg(pg.get()),
                pooled(std::max(1, total) * dim), expected(std::max(1, total) * dim);
            k::glm_index_prepare(keys.p, gates.p, ape.p, pk.p, pg.p, pooled.p, pos, B, pool, dim, nullptr);
            for (int t = 0; t < B; ++t) {
                ck(cudaMemcpy(sk.p + ((pos + t) % pool) * dim, keys.p + t * dim, dim * 4,
                              cudaMemcpyDeviceToDevice));
                ck(cudaMemcpy(sg.p + ((pos + t) % pool) * dim, gates.p + t * dim, dim * 4,
                              cudaMemcpyDeviceToDevice));
                if ((pos + t + 1) % pool == 0)
                    k::glm_index_pool(sk.p, sg.p, ape.p, expected.p + ((pos + t + 1) / pool - 1) * dim, pool,
                                      dim, nullptr);
            }
            near(pk.get(), sk.get(), 0);
            near(pg.get(), sg.get(), 0);
            near(pooled.get(), expected.get(), 0);
        }
        for (bool ties : {false, true}) {
            int B = 9, pos = 4093, pools = (pos + B) / 4, stride = 2052;
            auto hs = ties ? std::vector<float>(B * pools, 0) : random(B * pools);
            Buffer<float> scores(hs);
            Buffer<int> ids(B * stride), counts(B), ref(B * stride), rc(B);
            k::glm_index_select_batch(scores.p, ids.p, counts.p, pools, pos, B, 4, 2048, stride, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_index_select(scores.p + t * pools, ref.p + t * stride, rc.p + t, pos + t + 1, 4, 2048,
                                    nullptr);
            auto a = ids.get(), b = ref.get(), ns = counts.get();
            if (ns != rc.get())
                throw std::runtime_error("selection counts");
            for (int t = 0; t < B; ++t)
                for (int i = 0; i < ns[t]; ++i)
                    if (a[t * stride + i] != b[t * stride + i])
                        throw std::runtime_error("selection order");
        }
        {
            int B = 9, E = 288, K = 8;
            Buffer<float> logits(random(B * E)), bias(random(E)), weights(B * K), seqw(B * K);
            Buffer<int> ids(B * K), seqids(B * K), bounds(E + 1), dest(B * K), source(B * K), cursor(E);
            k::glm_route_batch(logits.p, bias.p, ids.p, weights.p, E, K, 2.5, B, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_router(logits.p + t * E, bias.p, seqids.p + t * K, seqw.p + t * K, E, K, 2.5, nullptr);
            if (ids.get() != seqids.get())
                throw std::runtime_error("routing IDs");
            near(weights.get(), seqw.get(), 0);
            k::glm_group_routes(ids.p, bounds.p, dest.p, source.p, cursor.p, E, K, B, nullptr);
            auto hb = bounds.get(), hd = dest.get(), hs = source.get(), hi = ids.get();
            std::vector<int> seen(B * K);
            for (int e = 0; e < E; ++e)
                for (int j = hb[e]; j < hb[e + 1]; ++j) {
                    if (hi[hd[j]] != e || hs[j] != hd[j] / K || seen[hd[j]]++)
                        throw std::runtime_error("CSR routing");
                }
            for (int v : seen)
                if (v != 1)
                    throw std::runtime_error("missing route");
        }
        for (int E : {1, 7, 288}) for (int B : {1, 2, 31, 32, 33, 255, 256, 257, 4096}) {
            const int K = std::min(E, 8), routes = B * K, blocks = (routes + 255) / 256;
            std::vector<int> ids(routes), bounds(E + 1), dest, source;
            for (int i = 0; i < routes; ++i) ids[i] = i % 2 ? 0 : (i * 17) % E;
            for (int e = 0; e < E; ++e) {
                for (int i = 0; i < routes; ++i) if (ids[i] == e) {dest.push_back(i); source.push_back(i / K);}
                bounds[e + 1] = dest.size();
            }
            Buffer<int> di(ids), db(std::vector<int>(E + 3, -99)), dd(std::vector<int>(routes + 2, -99)),
                ds(std::vector<int>(routes + 2, -99)), scratch(std::vector<int>(E * (blocks + 1) + 2, -99));
            for (int repeat = 0; repeat < 4; ++repeat) {
                k::glm_group_routes_stable(di.p, db.p + 1, dd.p + 1, ds.p + 1, scratch.p + 1, E, K, B, nullptr);
                const auto actual_b = db.get(), actual_d = dd.get(), actual_s = ds.get();
                const auto actual_scratch = scratch.get();
                if (!std::equal(bounds.begin(), bounds.end(), actual_b.begin() + 1) ||
                    !std::equal(dest.begin(), dest.end(), actual_d.begin() + 1) ||
                    !std::equal(source.begin(), source.end(), actual_s.begin() + 1) ||
                    actual_b.front() != -99 || actual_b.back() != -99 ||
                    actual_d.front() != -99 || actual_d.back() != -99 ||
                    actual_s.front() != -99 || actual_s.back() != -99 ||
                    actual_scratch.front() != -99 || actual_scratch.back() != -99)
                    throw std::runtime_error("stable routes differ from ordered CPU oracle or overwrite guard");
            }
        }
        {
            int B = 3, H = 2, L = 32, C = 2052, N = 512;
            Buffer<float> cache(random(N * L)), q(random(B * H * L)), gathered(B * C * L), scores(B * H * C),
                out(B * H * L), ref(B * H * L);
            std::vector<int> ids(B * C, 0), counts{3, 127, 512};
            for (int t = 0; t < B; ++t)
                for (int i = 0; i < counts[t]; ++i)
                    ids[t * C + i] = (i * 7) % N;
            Buffer<int> di(ids), dc(counts);
            k::glm_mla_gather(cache.p, di.p, dc.p, gathered.p, B, C, L, nullptr);
            cublasHandle_t h;
            cublasCreate(&h);
#ifdef STRATA_USE_HIP
            cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
#else
            cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
#endif
            float one = 1, zero = 0;
            auto cb = [](cublasStatus_t e) {
                if (e != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("blas");
            };
            cb(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, C, H, L, &one, gathered.p, L, C * L,
                                         q.p, L, H * L, &zero, scores.p, C, H * C, B));
            k::glm_mla_softmax(scores.p, dc.p, H, C, B, 1.f / 16, nullptr);
            cb(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, L, H, C, &one, gathered.p, L, C * L,
                                         scores.p, C, H * C, &zero, out.p, L, H * L, B));
            for (int t = 0; t < B; ++t)
                k::glm_mla(q.p + t * H * L, cache.p, di.p + t * C, dc.p + t, ref.p + t * H * L, H, L,
                           1.f / 16, nullptr);
            near(out.get(), ref.get());
            cublasDestroy(h);
        }
        ck(cudaDeviceSynchronize());
        std::cout << "GLM prefill batch, boundary, CSR, tie and sparse attention parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
