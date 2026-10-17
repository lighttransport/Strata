// llvq's shape-gain matrix quantizer around the Tetra encoder, ported for identical words
// (llvq-quant/src/{quantizer,rotation}.rs and llvq-core/src/rng.rs @ ddecad31):
//   - SplitMix64 and its Box-Muller Gaussian, the incoherence Rotation (Gaussian signs, a scaled Walsh-Hadamard
//     transform per power-of-two group, a Gram-Schmidt mix across groups);
//   - row_scale, fit_gain_centroids (Lloyd on block norm / row scale), the gain choice (nearest level, first wins)
//     and reconstruct_shape_gain.
// With an identity Hessian llvq's GPTQ loop propagates nothing, so a matrix quantizes block by block: rotate rows,
// fit the two centroids, then each full 24-weight block -> (Tetra word, gain bit); a trailing partial block stays
// exact (TailPolicy::KeepExact).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::llvq {

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double next_f64() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
    double next_gaussian() {
        double u1 = (double(next() >> 11) + 1.0) * (1.0 / 9007199254740993.0);
        double u2 = next_f64();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

class Rotation {
public:
    Rotation(size_t n, uint64_t seed) : n_(n) {
        if (n == 0) throw std::invalid_argument("llvq rotation: dimension must be positive");
        m_ = size_t(1) << __builtin_ctzll(n);
        k_ = n / m_;
        SplitMix64 rng(seed);
        signs_.resize(n);
        for (auto& s : signs_) s = rng.next_gaussian() < 0.0 ? -1.0 : 1.0;
        small_.assign(k_ * k_, 0.0);
        for (size_t i = 0; i < k_; ++i) {
            std::vector<double> row(k_);
            for (auto& r : row) r = rng.next_gaussian();
            for (size_t p = 0; p < i; ++p) {
                const double* prev = &small_[p * k_];
                double d = 0.0;
                for (size_t t = 0; t < k_; ++t) d += row[t] * prev[t];
                for (size_t t = 0; t < k_; ++t) row[t] -= d * prev[t];
            }
            double nrm = 0.0;
            for (double a : row) nrm += a * a;
            nrm = std::sqrt(nrm);
            if (!(nrm > 1e-9)) throw std::runtime_error("llvq rotation: degenerate Gram-Schmidt");
            for (size_t t = 0; t < k_; ++t) small_[i * k_ + t] = row[t] / nrm;
        }
    }
    size_t dim() const { return n_; }
    // v <- Q v (Rotation::apply); rows of W become rows of W Q^T.
    void apply(double* v) const {
        for (size_t i = 0; i < n_; ++i) v[i] *= signs_[i];
        for (size_t g = 0; g < k_; ++g) wht(v + g * m_, m_);
        mix(v, false);
    }
    // v <- Q^T v (Rotation::apply_transpose).
    void apply_transpose(double* v) const {
        mix(v, true);
        for (size_t g = 0; g < k_; ++g) wht(v + g * m_, m_);
        for (size_t i = 0; i < n_; ++i) v[i] *= signs_[i];
    }
    const std::vector<double>& signs() const { return signs_; }

private:
    size_t n_, m_, k_;
    std::vector<double> signs_, small_;
    static void wht(double* v, size_t n) {
        for (size_t len = 1; len < n; len <<= 1)
            for (size_t i = 0; i < n; i += len << 1)
                for (size_t j = i; j < i + len; ++j) {
                    double a = v[j], b = v[j + len];
                    v[j] = a + b;
                    v[j + len] = a - b;
                }
        double s = 1.0 / std::sqrt(double(n));
        for (size_t i = 0; i < n; ++i) v[i] *= s;
    }
    void mix(double* v, bool transpose) const {
        if (k_ == 1) return;
        std::vector<double> col(k_);
        for (size_t j = 0; j < m_; ++j) {
            for (size_t g = 0; g < k_; ++g) col[g] = v[g * m_ + j];
            for (size_t g = 0; g < k_; ++g) {
                double acc = 0.0;
                for (size_t t = 0; t < k_; ++t) acc += (transpose ? small_[t * k_ + g] : small_[g * k_ + t]) * col[t];
                v[g * m_ + j] = acc;
            }
        }
    }
};

inline double row_scale(const double* row, size_t n) {
    if (n == 0) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += row[i] * row[i];
    return std::sqrt(s / double(std::max<size_t>(n / 24, 1)));
}

inline double block_norm(const double* b) {
    double s = 0.0;
    for (int i = 0; i < 24; ++i) s += b[i] * b[i];
    return std::sqrt(s);
}

// Two (k_bits = 1) gain levels for a whole matrix: Lloyd on the block norms relative to their row's scale.
inline std::vector<double> fit_gain_centroids(const double* w, size_t d_out, size_t d_in, uint32_t k_bits, int iters) {
    std::vector<double> g;
    g.reserve(d_out * (d_in / 24));
    for (size_t i = 0; i < d_out; ++i) {
        const double* row = w + i * d_in;
        double scale = row_scale(row, d_in);
        if (scale <= 0.0) continue;
        for (size_t b = 0; b + 24 <= d_in; b += 24) {
            double nn = block_norm(row + b);
            if (nn > 0.0) g.push_back(nn / scale);
        }
    }
    if (g.empty()) return {1.0};
    const size_t k = size_t(1) << k_bits;
    std::sort(g.begin(), g.end());  // finite, non-negative: the same order as total_cmp
    std::vector<double> c(k);
    for (size_t j = 0; j < k; ++j) c[j] = g[(2 * j + 1) * g.size() / (2 * k)];
    for (int it = 0; it < iters; ++it) {
        std::vector<double> sums(k, 0.0);
        std::vector<uint64_t> counts(k, 0);
        for (double v : g) {
            size_t j = 0;
            for (size_t a = 1; a < k; ++a)
                if (std::fabs(v - c[a]) < std::fabs(v - c[j])) j = a;  // min_by: the first minimum
            sums[j] += v;
            counts[j] += 1;
        }
        for (size_t j = 0; j < k; ++j)
            if (counts[j] > 0) c[j] = sums[j] / double(counts[j]);
        std::sort(c.begin(), c.end());
    }
    return c;
}

inline uint32_t nearest_level_index(const std::vector<double>& levels, double g) {
    double best = INFINITY;
    uint32_t bi = 0;
    for (uint32_t i = 0; i < levels.size(); ++i) {
        double d = std::fabs(g - levels[i]);
        if (d < best) { best = d; bi = i; }
    }
    return bi;
}

// reconstruct_shape_gain: point (natural order) and gain level -> 24 weights.
inline void reconstruct_shape_gain(const int32_t* point, uint32_t gain, const std::vector<double>& centroids,
                                   double row_scale_v, double* out) {
    int64_t ss = 0;
    for (int i = 0; i < 24; ++i) ss += int64_t(point[i]) * point[i];
    if (ss == 0 || ss % 16 != 0) {
        for (int i = 0; i < 24; ++i) out[i] = 0.0;
        return;
    }
    int64_t m = ss / 16;
    double picked = centroids[gain] * row_scale_v;
    double scale = picked / std::sqrt(double(16 * m));
    for (int i = 0; i < 24; ++i) out[i] = double(point[i]) * scale;
}

}  // namespace strata::llvq
