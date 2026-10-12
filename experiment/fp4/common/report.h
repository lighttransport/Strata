// Timing statistics and CSV output shared by both benchmarks.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace fp4 {

struct Stats { double med = 0, p10 = 0, p90 = 0, min = 0; };
inline Stats stats_of(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, size_t(f * (v.size() - 1) + 0.5))]; };
    s.med = q(0.5); s.p10 = q(0.1); s.p90 = q(0.9); s.min = v.front();
    return s;
}

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One CSV row per measured (kernel, shape, config). Columns are fixed so results/*.csv concatenate.
struct Csv {
    FILE* f = nullptr;
    explicit Csv(const std::string& path) {
        if (path.empty()) return;
        FILE* t = fopen(path.c_str(), "r");
        bool exists = t != nullptr;
        if (t) fclose(t);
        f = fopen(path.c_str(), "a");
        if (f && !exists)
            fprintf(f, "device,kernel,op,weights,n_tok,threads_or_cfg,us_med,us_p10,us_p90,bytes,gbps,pct_peak,tflops,rel_l2,max_abs,hash\n");
    }
    ~Csv() { if (f) fclose(f); }
    void row(const char* dev, const std::string& kernel, const std::string& op, const char* weights, int n_tok,
             const std::string& cfg, const Stats& s, double bytes, double peak_gbps, double flops, double rel_l2,
             double max_abs, unsigned long long hash) {
        double gbps = bytes / (s.med * 1e-6) / 1e9;
        double tf = flops / (s.med * 1e-6) / 1e12;
        if (f)
            fprintf(f, "%s,%s,%s,%s,%d,%s,%.2f,%.2f,%.2f,%.0f,%.1f,%.1f,%.2f,%.3e,%.3e,%016llx\n", dev, kernel.c_str(),
                    op.c_str(), weights, n_tok, cfg.c_str(), s.med, s.p10, s.p90, bytes, gbps,
                    peak_gbps > 0 ? 100 * gbps / peak_gbps : 0, tf, rel_l2, max_abs, hash);
        printf("%-4s %-22s %-10s N=%-5d %-14s %9.1f us [%8.1f..%8.1f] %7.1f GB/s (%5.1f%%) %7.2f TF  relL2=%.2e%s\n",
               dev, kernel.c_str(), op.c_str(), n_tok, cfg.c_str(), s.med, s.p10, s.p90, gbps,
               peak_gbps > 0 ? 100 * gbps / peak_gbps : 0, tf, rel_l2, rel_l2 > 0.05 ? "  <-- HIGH" : "");
        fflush(stdout);
    }
};

}  // namespace fp4
