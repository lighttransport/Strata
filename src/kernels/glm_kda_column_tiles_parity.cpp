// Compare tiled columns with the original KDA reduction, including retained state.
#include "strata/kernels/glm_prefill.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    float *base = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { ck(cudaMalloc(&base, (n + 32) * 4)); }
    ~Buffer() { cudaFree(base); }
    float *data() { return base + 16; }
    void put(const std::vector<float> &x, cudaStream_t s) {
        if (x.size() != count)
            throw std::runtime_error("size");
        ck(cudaMemsetAsync(base, 0xa5, (count + 32) * 4, s));
        ck(cudaMemcpyAsync(data(), x.data(), count * 4, cudaMemcpyHostToDevice, s));
    }
    std::vector<unsigned char> get(cudaStream_t s) {
        ck(cudaStreamSynchronize(s));
        std::vector<unsigned char> out((count + 32) * 4);
        ck(cudaMemcpy(out.data(), base, out.size(), cudaMemcpyDeviceToHost));
        if (!std::all_of(out.begin(), out.begin() + 64, [](auto b) { return b == 0xa5; }) ||
            !std::all_of(out.end() - 64, out.end(), [](auto b) { return b == 0xa5; }))
            throw std::runtime_error("guard overwritten");
        for (size_t i = 0; i < count; ++i) {
            float value;
            std::memcpy(&value, out.data() + 64 + i * 4, 4);
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite");
        }
        return out;
    }
};
int main() {
    cudaStream_t stream = nullptr;
    const char *flag = std::getenv("STRATA_GLM_KDA_COLUMN_TILES");
    const bool present = flag != nullptr;
    const std::string saved = flag ? flag : "";
    auto restore = [&] {
        if (present)
            setenv("STRATA_GLM_KDA_COLUMN_TILES", saved.c_str(), 1);
        else
            unsetenv("STRATA_GLM_KDA_COLUMN_TILES");
    };
    try {
        ck(cudaStreamCreate(&stream));
        std::mt19937 rng(5322);
        std::normal_distribution<float> random(0, .1f);
        for (int heads : {1, 3, 64})
            for (int nt : {1, 2, 3, 16, 64})
                for (int parts : {4, 8})
                    for (bool prepared : {false, true}) {
                        const size_t n = size_t(heads) * nt * 128;
                        std::vector<float> q(n), key(n), v(n), g(n, -.08f), beta(heads * nt, .2f),
                            initial(size_t(heads) * 128 * 128), zeros(n, 0), qi(heads * nt, 0);
                        for (auto &x : q)
                            x = random(rng);
                        for (auto &x : key)
                            x = random(rng);
                        for (auto &x : v)
                            x = random(rng);
                        for (auto &x : initial)
                            x = random(rng) * .1f;
                        Buffer dq(n), dk(n), dv(n), dg(n), db(beta.size()), di(qi.size()),
                            state(initial.size()), out(n);
                        dq.put(q, stream);
                        dk.put(key, stream);
                        dv.put(v, stream);
                        dg.put(g, stream);
                        db.put(beta, stream);
                        di.put(qi, stream);
                        if (prepared)
                            k::glm_kda_prepare(dq.data(), dk.data(), dg.data(), db.data(), di.data(), heads,
                                               nt, stream);
                        std::vector<unsigned char> reference_state, reference_out;
                        for (int tiled : {0, 1}) {
                            if (tiled)
                                setenv("STRATA_GLM_KDA_COLUMN_TILES", "1", 1);
                            else
                                unsetenv("STRATA_GLM_KDA_COLUMN_TILES");
                            state.put(initial, stream);
                            out.put(zeros, stream);
                            k::glm_kda_chunk(state.data(), dq.data(), dk.data(), dv.data(), dg.data(),
                                             db.data(), out.data(), heads, 128, nt, stream, 128, parts,
                                             prepared ? di.data() : nullptr);
                            auto actual_state = state.get(stream), actual_out = out.get(stream);
                            if (!tiled) {
                                reference_state = actual_state;
                                reference_out = actual_out;
                            } else if (actual_state != reference_state || actual_out != reference_out)
                                throw std::runtime_error("tiled columns changed bits");
                            if (heads == 64 && nt == 64 && parts == 8 && prepared) {
                                cudaEvent_t a, b;
                                ck(cudaEventCreate(&a));
                                ck(cudaEventCreate(&b));
                                std::vector<float> times;
                                for (int trial = 0; trial < 8; ++trial) {
                                    ck(cudaEventRecord(a, stream));
                                    for (int i = 0; i < 32; ++i)
                                        k::glm_kda_chunk(state.data(), dq.data(), dk.data(), dv.data(),
                                                         dg.data(), db.data(), out.data(), heads, 128, nt,
                                                         stream, 128, parts, di.data());
                                    ck(cudaEventRecord(b, stream));
                                    ck(cudaEventSynchronize(b));
                                    float ms;
                                    ck(cudaEventElapsedTime(&ms, a, b));
                                    if (trial)
                                        times.push_back(ms / 32);
                                }
                                std::sort(times.begin(), times.end());
                                std::cout << "KDA_COLUMN_BENCH tiled=" << tiled
                                          << " median_us=" << times[3] * 1000 << '\n';
                                ck(cudaEventDestroy(a));
                                ck(cudaEventDestroy(b));
                            }
                        }
                        std::cout << "KDA_COLUMN_PARITY heads=" << heads << " nt=" << nt << " parts=" << parts
                                  << " prepared=" << prepared
                                  << " output_state_bits=identical guards=unchanged\n";
                    }
        restore();
        ck(cudaStreamDestroy(stream));
        std::cout << "KDA column tiles parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        restore();
        if (stream)
            cudaStreamDestroy(stream);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
