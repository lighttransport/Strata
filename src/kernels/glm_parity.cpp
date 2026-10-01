#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_reference.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace ref = k::glm_reference;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n;
    explicit Buffer(size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(T))); }
    explicit Buffer(const std::vector<T> &v) : Buffer(v.size()) { put(v); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void put(const std::vector<T> &v) { ck(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
void near(const char *name, const std::vector<float> &got, const std::vector<float> &expected,
          float tolerance = 3e-5f) {
    float error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]))
            throw std::runtime_error(std::string(name) + ": non-finite result");
        error = std::max(error, std::abs(got[i] - expected[i]) / (1 + std::abs(expected[i])));
    }
    std::cout << name << " max scaled error=" << error << '\n';
    if (error > tolerance)
        throw std::runtime_error(std::string(name) + ": parity failure");
}
int main() {
    try {
        std::mt19937 rng(53);
        std::normal_distribution<float> nd(0, 1);
        auto random = [&](int n) {
            std::vector<float> v(n);
            for (auto &x : v)
                x = nd(rng);
            return v;
        };
        const int H = 4096, heads = 64, dim = 128, n = heads * dim;
        auto x = random(H), w = random(H);
        Buffer<float> dx(x), dw(w), dy(H);
        double sum = 0;
        for (float v : x)
            sum += (double)v * v;
        std::vector<float> expected(H);
        for (int i = 0; i < H; ++i)
            expected[i] = x[i] * w[i] / std::sqrt(sum / H + 1e-5);
        k::glm_rms_norm(dx.p, dw.p, dy.p, H, 1, 1e-5, nullptr);
        near("RMSNorm", dy.get(), expected);
        auto b = random(H);
        Buffer<float> db(b);
        double mean = 0;
        for (float v : x)
            mean += v;
        mean /= H;
        sum = 0;
        for (float v : x)
            sum += (v - mean) * (v - mean);
        for (int i = 0; i < H; ++i)
            expected[i] = (x[i] - mean) * w[i] / std::sqrt(sum / H + 1e-6) + b[i];
        k::glm_layer_norm(dx.p, dw.p, db.p, dy.p, H, 1e-6, nullptr);
        near("LayerNorm", dy.get(), expected);
        auto gate = random(H), up = random(H);
        for (auto &v : gate)
            v *= 25;
        for (auto &v : up)
            v *= 25;
        Buffer<float> dg(gate), du(up);
        for (int i = 0; i < H; ++i)
            expected[i] = ref::swiglu(gate[i], up[i], 10);
        k::glm_swiglu(dg.p, du.p, dy.p, H, 10, nullptr);
        near("asymmetric SwiGLU clamp", dy.get(), expected);
        auto logits = random(288), bias = random(288);
        Buffer<float> dl(logits), dcorr(bias), drw(8);
        Buffer<int> did(8);
        std::vector<int> ids(8);
        std::vector<float> weights(8);
        ref::router(logits.data(), bias.data(), 288, 8, 2.5, ids.data(), weights.data());
        k::glm_router(dl.p, dcorr.p, did.p, drw.p, 288, 8, 2.5, nullptr);
        if (did.get() != ids)
            throw std::runtime_error("routing IDs differ");
        near("unbiased route weights", drw.get(), weights);
        // Equal scores must have deterministic ascending expert IDs.
        dl.put(std::vector<float>(288));
        dcorr.put(std::vector<float>(288));
        k::glm_router(dl.p, dcorr.p, did.p, drw.p, 288, 8, 2.5, nullptr);
        for (int i = 0; i < 8; ++i)
            if (did.get()[i] != i)
                throw std::runtime_error("routing tie failure");
        // Warp boundaries and the serial fallback must retain first-index ties.
        for (int ne : {1, 31, 32, 33, 127, 128, 287, 288, 511, 512, 513}) {
            const int count = std::min(8, ne);
            auto scores = random(ne), correction = random(ne);
            Buffer<float> input(scores), bias_input(correction), output_weights(count);
            Buffer<int> output_ids(count);
            std::vector<int> expected_ids(count);
            std::vector<float> expected_weights(count);
            for (int equal = 0; equal < 2; ++equal) {
                if (equal) {
                    std::fill(scores.begin(), scores.end(), 0);
                    std::fill(correction.begin(), correction.end(), 0);
                    input.put(scores); bias_input.put(correction);
                }
                ref::router(scores.data(), correction.data(), ne, count, 2.5,
                            expected_ids.data(), expected_weights.data());
                k::glm_router(input.p, bias_input.p, output_ids.p, output_weights.p,
                              ne, count, 2.5, nullptr);
                if (output_ids.get() != expected_ids)
                    throw std::runtime_error("routing boundary/tie IDs differ");
                near("routing boundary weights", output_weights.get(), expected_weights);
            }
        }
        auto streams = random(4 * H), projected = random(24), base = random(24);
        std::vector<float> scales = {.1, .2, .3}, coeff(24), collapsed(H), mixed(4 * H);
        Buffer<float> ds(streams), dp(projected), dbase(base), dscale(scales), dcoeff(24);
        ref::mhc(projected.data(), base.data(), scales.data(), coeff.data(), 20, 1e-6);
        k::glm_mhc_read(ds.p, dp.p, dbase.p, dscale.p, dcoeff.p, dy.p, H, 20, 1e-6, nullptr);
        near("mHC coefficients", dcoeff.get(), coeff);
        for (int d = 0; d < H; ++d)
            for (int i = 0; i < 4; ++i)
                collapsed[d] += coeff[i] * streams[i * H + d];
        near("mHC collapse", dy.get(), collapsed);
        for (int d = 0; d < H; ++d)
            for (int j = 0; j < 4; ++j) {
                mixed[j * H + d] = coeff[4 + j] * x[d];
                for (int i = 0; i < 4; ++i)
                    mixed[j * H + d] += coeff[8 + i * 4 + j] * streams[i * H + d];
            }
        k::glm_mhc_write(ds.p, dcoeff.p, dx.p, ds.p, H, nullptr);
        near("in-place mHC residual", ds.get(), mixed);
        for (int d = 0; d < H; ++d)
            expected[d] = (mixed[d] + mixed[H + d] + mixed[2 * H + d] + mixed[3 * H + d]) * .25f;
        k::glm_hyper_head(ds.p, dy.p, H, nullptr);
        near("hyper head", dy.get(), expected);
        // Recurrent parity across eight tokens catches state layout and update-order mistakes.
        std::vector<float> state(heads * dim * dim), history(n * 3);
        auto convw = random(n * 4);
        Buffer<float> dstate(state), dhistory(history), dconvw(convw), dq(n), dk(n), dv(n), ddecay(n),
            dbeta(heads), dout(n);
        auto alog = random(heads), dt = random(n);
        std::vector<float> gguf_a(heads);
        for (int h = 0; h < heads; ++h)
            gguf_a[h] = -std::exp(alog[h]);
        Buffer<float> da(gguf_a), ddt(dt);
        for (int token = 0; token < 8; ++token) {
            auto raw = random(n);
            dq.put(raw);
            std::vector<float> cq(n);
            for (int i = 0; i < n; ++i) {
                float s = raw[i] * convw[i * 4 + 3];
                for (int j = 0; j < 3; ++j)
                    s += history[i * 3 + j] * convw[i * 4 + j];
                cq[i] = s * ref::sigmoid(s);
                history[i * 3] = history[i * 3 + 1];
                history[i * 3 + 1] = history[i * 3 + 2];
                history[i * 3 + 2] = raw[i];
            }
            k::glm_conv(dq.p, dconvw.p, dhistory.p, dout.p, n, 4, nullptr);
            near("KDA conv", dout.get(), cq);
            near("conv history", dhistory.get(), history);
            auto q = random(n), key = random(n), v = random(n), project = random(n), beta = random(heads);
            std::vector<float> decay(n), out(n);
            for (int i = 0; i < n; ++i)
                decay[i] = -5 * ref::sigmoid(std::exp(alog[i / dim]) * (project[i] + dt[i]));
            dq.put(project);
            k::glm_kda_gate(dq.p, ddt.p, da.p, ddecay.p, heads, dim, -5, nullptr);
            near("KDA gate", ddecay.get(), decay);
            dq.put(q);
            dk.put(key);
            dv.put(v);
            dbeta.put(beta);
            ref::kda(state.data(), q.data(), key.data(), v.data(), decay.data(), beta.data(), out.data(),
                     heads, dim);
            k::glm_kda_step(dstate.p, dq.p, dk.p, dv.p, ddecay.p, dbeta.p, dout.p, heads, dim, nullptr);
            near("KDA output", dout.get(), out);
            near("KDA state", dstate.get(), state);
        }
        auto normw = random(dim), outgate = random(n);
        Buffer<float> dnormw(normw), dog(outgate);
        auto last = dout.get();
        std::vector<float> final(n);
        for (int h = 0; h < heads; ++h) {
            double s = 0;
            for (int d = 0; d < dim; ++d)
                s += (double)last[h * dim + d] * last[h * dim + d];
            for (int d = 0; d < dim; ++d)
                final[h * dim + d] = last[h * dim + d] / std::sqrt(s / dim + 1e-5) * normw[d] *
                                     ref::sigmoid(outgate[h * dim + d]);
        }
        k::glm_kda_output(dout.p, dog.p, dnormw.p, dq.p, heads, dim, 1e-5, nullptr);
        near("KDA sigmoid output gate", dq.get(), final);
        auto keys = random(4 * dim), gates = random(4 * dim), ape = random(4 * dim);
        std::vector<float> pooled(dim);
        Buffer<float> dkeys(keys), dgates(gates), dape(ape), dpooled(dim);
        ref::pool(keys.data(), gates.data(), ape.data(), pooled.data(), 4, dim);
        k::glm_index_pool(dkeys.p, dgates.p, dape.p, dpooled.p, 4, dim, nullptr);
        near("learned per-channel pool", dpooled.get(), pooled);
        const int ih = 32, np = 5;
        auto iq = random(ih * dim), iw = random(ih), ik = random(np * dim);
        Buffer<float> diq(iq), diw(iw), dik(ik), scores(np);
        std::vector<float> score(np);
        for (int p = 0; p < np; ++p)
            for (int h = 0; h < ih; ++h) {
                double dot = 0;
                for (int d = 0; d < dim; ++d)
                    dot += (double)iq[h * dim + d] * ik[p * dim + d];
                score[p] += iw[h] * std::max(0., dot / std::sqrt((double)dim)) / std::sqrt((double)ih);
            }
        k::glm_index_score(diq.p, diw.p, dik.p, scores.p, ih, dim, np, nullptr);
        near("IndexPool scores", scores.get(), score);
        Buffer<int> selected(11), count(1);
        for (int tokens : {1, 3, 4, 7, 20, 23}) {
            k::glm_index_select(scores.p, selected.p, count.p, tokens, 4, 8, nullptr);
            const int pools = tokens / 4, chosen = std::min(2, pools);
            std::vector<int> order(pools);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
            std::vector<int> want;
            for (int i = 0; i < chosen; ++i)
                for (int j = 0; j < 4; ++j)
                    want.push_back(order[i] * 4 + j);
            for (int i = pools * 4; i < tokens; ++i)
                want.push_back(i);
            auto got = selected.get();
            got.resize(count.get()[0]);
            if (got != want)
                throw std::runtime_error("pool/tail selection differs");
        }
        {
            const int pools = 2050, budget = 2048;
            std::vector<float> tied(pools);
            for (int p = 0; p < pools; ++p)
                tied[p] = (float)(p % 17 - 8);
            Buffer<float> dlong(tied);
            Buffer<int> long_ids(budget + 3), long_count(1);
            k::glm_index_select(dlong.p, long_ids.p, long_count.p, pools * 4 + 3, 4, budget, nullptr);
            std::vector<int> order(pools), want;
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return tied[a] > tied[b]; });
            for (int i = 0; i < budget / 4; ++i)
                for (int j = 0; j < 4; ++j)
                    want.push_back(order[i] * 4 + j);
            for (int i = pools * 4; i < pools * 4 + 3; ++i)
                want.push_back(i);
            auto got = long_ids.get();
            got.resize(long_count.get()[0]);
            if (got != want)
                throw std::runtime_error("radix sparse budget, negative scores, or ties differ");
            std::cout << "long-context IndexPool budget and ties passed\n";
        }
        // Absorbed MLA against a double-precision dense softmax oracle.
        const int latent = 512, mh = 64, nt = 23;
        auto mq = random(mh * latent), cache = random(nt * latent);
        Buffer<float> dmq(mq), dcache(cache), mo(mh * latent);
        auto selection = selected.get();
        selection.resize(count.get()[0]);
        std::vector<float> mla(mh * latent);
        for (int h = 0; h < mh; ++h) {
            std::vector<double> s(selection.size());
            double maximum = -INFINITY, denom = 0;
            for (size_t i = 0; i < s.size(); ++i) {
                s[i] = 0;
                for (int d = 0; d < latent; ++d)
                    s[i] += (double)mq[h * latent + d] * cache[selection[i] * latent + d] / 16.;
                maximum = std::max(maximum, s[i]);
            }
            for (auto &v : s) {
                v = std::exp(v - maximum);
                denom += v;
            }
            for (int d = 0; d < latent; ++d) {
                double value = 0;
                for (size_t i = 0; i < s.size(); ++i)
                    value += s[i] / denom * cache[selection[i] * latent + d];
                mla[h * latent + d] = value;
            }
        }
        k::glm_mla(dmq.p, dcache.p, selected.p, count.p, mo.p, mh, latent, 1.f / 16, nullptr);
        near("absorbed NoPE MLA", mo.get(), mla);
        ck(cudaDeviceSynchronize());
        std::cout << "GLM primitive parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
