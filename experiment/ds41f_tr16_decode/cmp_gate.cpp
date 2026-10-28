// cmp_gate REF.f32 CAND.f32 TOKENS.txt: KL(ref || cand), p99, top-1 agreement, next-token accuracy of both, both PPLs
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: cmp_gate REF.f32 CAND.f32 TOKENS.txt\n"); return 2; }
    const size_t V = 129280;
    auto load = [&](const char* p) { std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = size_t(f.tellg()) / 4; std::vector<float> v(n); f.seekg(0); f.read((char*) v.data(), std::streamsize(n * 4)); return v; };
    auto r = load(argv[1]), g = load(argv[2]);
    std::vector<int> toks; { std::ifstream t(argv[3]); int x; while (t >> x) toks.push_back(x); }
    const size_t n = std::min(r.size(), g.size()) / V;
    std::vector<double> kls; double nll_r = 0, nll_g = 0; size_t agree = 0, acc_r = 0, acc_g = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        const float *R = &r[i * V], *G = &g[i * V];
        double mr = -1e30, mg = -1e30; size_t ar = 0, ag = 0;
        for (size_t j = 0; j < V; ++j) { if (R[j] > mr) { mr = R[j]; ar = j; } if (G[j] > mg) { mg = G[j]; ag = j; } }
        double sr = 0, sg = 0; for (size_t j = 0; j < V; ++j) { sr += std::exp(R[j] - mr); sg += std::exp(G[j] - mg); }
        const double lzr = mr + std::log(sr), lzg = mg + std::log(sg);
        double kl = 0; for (size_t j = 0; j < V; ++j) { const double lr = R[j] - lzr, lg = G[j] - lzg; kl += std::exp(lr) * (lr - lg); }
        kls.push_back(kl); agree += ar == ag;
        if (i + 1 < toks.size()) {
            const int t = toks[i + 1];
            nll_r -= R[t] - lzr; nll_g -= G[t] - lzg; acc_r += size_t(ar) == size_t(t); acc_g += size_t(ag) == size_t(t); ++m;
        }
    }
    double mean = 0; for (double k : kls) mean += k; mean /= double(n);
    std::vector<double> s = kls; std::sort(s.begin(), s.end());
    std::printf("n=%zu KL(ref||cand) mean %.4f p99 %.3f  top1_agree %.1f%%  next_acc ref %.1f%% cand %.1f%% (%+.1f pp)  PPL ref %.3f cand %.3f (%+.2f%%)\n",
                n, mean, s[size_t(0.99 * double(n - 1))], 100.0 * double(agree) / double(n),
                100.0 * double(acc_r) / double(m), 100.0 * double(acc_g) / double(m), 100.0 * (double(acc_g) - double(acc_r)) / double(m),
                std::exp(nll_r / double(m)), std::exp(nll_g / double(m)), 100.0 * (std::exp((nll_g - nll_r) / double(m)) - 1.0));
}
