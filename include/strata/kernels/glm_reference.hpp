#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace strata::kernels::glm_reference {

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}
inline float swiglu(float g, float u, float limit) {
    if (limit > 0) {
        g = std::min(g, limit);
        u = std::clamp(u, -limit, limit);
    }
    return g * sigmoid(g) * u;
}
inline void router(const float *logits, const float *bias, int ne, int k, float scale, int *ids,
                   float *weights) {
    std::vector<float> scores(ne);
    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    for (int i = 0; i < ne; ++i)
        scores[i] = sigmoid(logits[i]);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] + bias[a] > scores[b] + bias[b]; });
    float sum = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = order[i];
        sum += scores[ids[i]];
    }
    for (int i = 0; i < k; ++i)
        weights[i] = scale * scores[ids[i]] / (sum + 1e-20f);
}
inline void mhc(const float *p, const float *base, const float *scale, float *c, int iters, float eps) {
    for (int i = 0; i < 4; ++i) {
        c[i] = sigmoid(p[i] * scale[0] + base[i]) + eps;
        c[4 + i] = 2 * sigmoid(p[4 + i] * scale[1] + base[4 + i]);
    }
    for (int i = 0; i < 4; ++i) {
        float mx = -INFINITY, sum = 0;
        for (int j = 0; j < 4; ++j)
            mx = std::max(mx, p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
        for (int j = 0; j < 4; ++j) {
            c[8 + i * 4 + j] = std::exp(p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j] - mx);
            sum += c[8 + i * 4 + j];
        }
        for (int j = 0; j < 4; ++j)
            c[8 + i * 4 + j] = c[8 + i * 4 + j] / sum + eps;
    }
    for (int n = 0; n < iters; ++n) {
        if (n > 0)
            for (int i = 0; i < 4; ++i) {
                float sum = eps;
                for (int j = 0; j < 4; ++j)
                    sum += c[8 + i * 4 + j];
                for (int j = 0; j < 4; ++j)
                    c[8 + i * 4 + j] /= sum;
            }
        for (int j = 0; j < 4; ++j) {
            float sum = eps;
            for (int i = 0; i < 4; ++i)
                sum += c[8 + i * 4 + j];
            for (int i = 0; i < 4; ++i)
                c[8 + i * 4 + j] /= sum;
        }
    }
}
inline void kda(float *state, const float *q, const float *k, const float *v, const float *g,
                const float *beta, float *out, int heads, int dim) {
    for (int h = 0; h < heads; ++h) {
        std::vector<float> nq(dim), nk(dim), delta(dim);
        float qs = 1e-6f, ks = 1e-6f;
        for (int i = 0; i < dim; ++i) {
            qs += q[h * dim + i] * q[h * dim + i];
            ks += k[h * dim + i] * k[h * dim + i];
        }
        for (int i = 0; i < dim; ++i) {
            nq[i] = q[h * dim + i] / std::sqrt(qs) / std::sqrt((float)dim);
            nk[i] = k[h * dim + i] / std::sqrt(ks);
            for (int j = 0; j < dim; ++j)
                state[(h * dim + i) * dim + j] *= std::exp(g[h * dim + i]);
        }
        for (int j = 0; j < dim; ++j) {
            float mem = 0;
            for (int i = 0; i < dim; ++i)
                mem += state[(h * dim + i) * dim + j] * nk[i];
            delta[j] = (v[h * dim + j] - mem) * sigmoid(beta[h]);
            out[h * dim + j] = 0;
            for (int i = 0; i < dim; ++i) {
                float &s = state[(h * dim + i) * dim + j];
                s += nk[i] * delta[j];
                out[h * dim + j] += s * nq[i];
            }
        }
    }
}
inline void pool(const float *keys, const float *gates, const float *ape, float *out, int p, int dim) {
    for (int d = 0; d < dim; ++d) {
        float mx = -INFINITY, sum = 0, value = 0;
        for (int i = 0; i < p; ++i)
            mx = std::max(mx, gates[i * dim + d] + ape[i * dim + d]);
        for (int i = 0; i < p; ++i) {
            const float w = std::exp(gates[i * dim + d] + ape[i * dim + d] - mx);
            sum += w;
            value += w * keys[i * dim + d];
        }
        out[d] = value / sum;
    }
}
} // namespace strata::kernels::glm_reference
