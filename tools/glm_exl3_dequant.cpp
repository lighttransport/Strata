// Dense FP32 reconstruction of selected routed experts of the GLM-5.3-Flash EXL3 K2 pack (quantization study,
// docs/GLM_QUANT_LEECH_GSQ_STUDY.md). The pack stores y = diag(svh) H T^T H diag(suh) x with T the decoded
// trellis tiles (input x output) and H the orthonormal 128-point Walsh-Hadamard per block; this tool forms
// W = diag(svh) H T^T H diag(suh) (output x input), checks it against GlmExl3::Matrix::apply on random inputs, and
// writes F32 safetensors named like the BF16 checkpoint (model.language_model.layers.L.mlp.experts.E.P.weight).
//
//   strata-glm-exl3-dequant MODEL_DIR LAYER EXPERTS(comma) OUT.safetensors
#include "strata/artifact/glm_exl3.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace e = strata::cpu::exl3;

static std::vector<float> dense(const strata::artifact::GlmExl3::Matrix& m) {
    const size_t in = m.in, out = m.out;
    // T^T as out x in: row o holds T[:, o].
    std::vector<float> w(out * in), tile(256);
    for (size_t it = 0; it < in / 16; ++it)
        for (size_t ot = 0; ot < out / 16; ++ot) {
            e::tile(m.trellis.data() + (it * (out / 16) + ot) * 64, tile.data());
            for (size_t r = 0; r < 16; ++r)
                for (size_t c = 0; c < 16; ++c) w[(ot * 16 + c) * in + it * 16 + r] = tile[r * 16 + c];
        }
    // Right factor H diag(suh): each row v -> (v H) diag(suh); H is symmetric, so apply it to the row as a vector.
    for (size_t o = 0; o < out; ++o) {
        float* row = w.data() + o * in;
        e::hadamard(row, in);
        for (size_t i = 0; i < in; ++i) row[i] *= e::half_value(m.suh[i]);
    }
    // Left factor diag(svh) H: transform every column.
    std::vector<float> col(out);
    for (size_t i = 0; i < in; ++i) {
        for (size_t o = 0; o < out; ++o) col[o] = w[o * in + i];
        e::hadamard(col.data(), out);
        for (size_t o = 0; o < out; ++o) w[o * in + i] = col[o] * e::half_value(m.svh[o]);
    }
    return w;
}

static double check(const strata::artifact::GlmExl3::Matrix& m, const std::vector<float>& w, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    std::vector<float> x(m.in), y(m.out);
    for (auto& v : x) v = n(rng);
    m.apply(x.data(), y.data());
    double err = 0, ref = 0;
    for (size_t o = 0; o < m.out; ++o) {
        double s = 0;
        for (size_t i = 0; i < m.in; ++i) s += double(w[o * m.in + i]) * x[i];
        err = std::max(err, std::abs(s - y[o]));
        ref = std::max(ref, std::abs(double(y[o])));
    }
    return err / std::max(ref, 1e-30);
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: strata-glm-exl3-dequant MODEL_DIR LAYER EXPERTS(comma) OUT.safetensors");
        strata::artifact::GlmExl3 model(argv[1]);
        const unsigned layer = std::stoul(argv[2]);
        std::vector<unsigned> experts;
        std::stringstream list(argv[3]);
        for (std::string item; std::getline(list, item, ',');) experts.push_back(std::stoul(item));
        struct Entry { std::string name; size_t out, in; std::vector<float> w; };
        std::vector<Entry> entries;
        double worst = 0;
        for (unsigned ex : experts)
            for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
                auto m = model.load(layer, ex, p);
                auto w = dense(m);
                const double rel = check(m, w, layer * 1000 + ex);
                worst = std::max(worst, rel);
                if (rel > 1e-4) throw std::runtime_error("dequant mismatch " + std::to_string(rel) + " at expert " + std::to_string(ex) + " " + p);
                entries.push_back({"model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(ex) + "." + p + ".weight",
                                   m.out, m.in, std::move(w)});
                std::cerr << "expert " << ex << ' ' << p << " " << m.out << "x" << m.in << " max_rel_err=" << rel << '\n';
            }
        std::ostringstream h;
        h << '{';
        size_t offset = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            const size_t bytes = entries[i].w.size() * 4;
            h << (i ? "," : "") << '"' << entries[i].name << "\":{\"dtype\":\"F32\",\"shape\":[" << entries[i].out << ','
              << entries[i].in << "],\"data_offsets\":[" << offset << ',' << offset + bytes << "]}";
            offset += bytes;
        }
        h << '}';
        std::string head = h.str();
        head.append((8 - head.size() % 8) % 8, ' ');
        std::ofstream f(argv[4], std::ios::binary | std::ios::trunc);
        const uint64_t n = head.size();
        f.write(reinterpret_cast<const char*>(&n), 8);
        f.write(head.data(), head.size());
        for (const auto& en : entries) f.write(reinterpret_cast<const char*>(en.w.data()), en.w.size() * 4);
        if (!f) throw std::runtime_error("write failed");
        std::cout << "wrote " << argv[4] << " tensors=" << entries.size() << " bytes=" << offset << " worst_rel_err=" << worst << '\n';
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
