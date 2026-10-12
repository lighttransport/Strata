// Expert weight sources: seeded dummy MXFP4, or real MiMo-V2.6 expert tensors read straight from
// the safetensors shards (mmap, minimal header parser, no dependencies).
#pragma once
#include "mxfp4.h"
#include "shapes.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace fp4 {

// One routed expert. gate and up are stacked into one [4096 x 4096] "gu" matrix
// (rows 0..2047 gate, 2048..4095 up) so one GEMV covers both.
struct Expert {
    std::vector<uint8_t> guq, gue, dnq, dne;
    MxView gu() const { return {2 * mimo::kExpertInter, mimo::kHidden, guq.data(), gue.data()}; }
    MxView dn() const { return {mimo::kHidden, mimo::kExpertInter, dnq.data(), dne.data()}; }
    size_t bytes() const { return guq.size() + gue.size() + dnq.size() + dne.size(); }
};

inline void alloc_expert(Expert& x) {
    const size_t I = mimo::kExpertInter, H = mimo::kHidden;
    x.guq.resize(2 * I * H / 2);
    x.gue.resize(2 * I * H / QK);
    x.dnq.resize(H * I / 2);
    x.dne.resize(H * I / QK);
}

// Dummy: uniform random codes; E8M0 drawn from 118..123, the range seen in the real checkpoint.
inline Expert make_dummy_expert(uint64_t seed) {
    Expert x;
    alloc_expert(x);
    std::mt19937_64 rng(seed);
    auto fill = [&](std::vector<uint8_t>& v) {
        for (size_t i = 0; i < v.size(); i += 8) {
            uint64_t r = rng();
            std::memcpy(&v[i], &r, std::min<size_t>(8, v.size() - i));
        }
    };
    fill(x.guq);
    fill(x.dnq);
    for (auto& b : x.gue) b = 118 + rng() % 6;
    for (auto& b : x.dne) b = 118 + rng() % 6;
    return x;
}

// ---- safetensors --------------------------------------------------------------------------------
struct StTensor { std::string dtype; std::vector<int64_t> shape; uint64_t begin = 0, end = 0; };

class StFile {
public:
    explicit StFile(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open " + path);
        struct stat st;
        fstat(fd_, &st);
        size_ = st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap " + path);
        uint64_t n;
        std::memcpy(&n, base_, 8);
        parse(std::string((const char*)base_ + 8, n));
        data_ = base_ + 8 + n;
    }
    ~StFile() { munmap((void*)base_, size_); ::close(fd_); }
    const StTensor& get(const std::string& name) const {
        auto it = t_.find(name);
        if (it == t_.end()) throw std::runtime_error("tensor not found: " + name + " in " + path_);
        return it->second;
    }
    const uint8_t* data(const StTensor& t) const { return data_ + t.begin; }

private:
    // Header is {"name": {"dtype": "U8", "shape": [a, b], "data_offsets": [s, e]}, "__metadata__": {...}}
    void parse(const std::string& h) {
        size_t p = 0;
        auto skip_ws = [&] { while (p < h.size() && isspace((unsigned char)h[p])) ++p; };
        auto str = [&] {
            skip_ws();
            if (h[p] != '"') throw std::runtime_error("st header: expected string");
            size_t q = h.find('"', p + 1);
            std::string s = h.substr(p + 1, q - p - 1);
            p = q + 1;
            return s;
        };
        auto ints = [&] {
            std::vector<int64_t> v;
            skip_ws(); ++p;  // [
            for (;;) {
                skip_ws();
                if (h[p] == ']') { ++p; break; }
                if (h[p] == ',') { ++p; continue; }
                size_t q;
                v.push_back(std::stoll(h.substr(p), &q));
                p += q;
            }
            return v;
        };
        skip_ws(); ++p;  // {
        for (;;) {
            skip_ws();
            if (h[p] == '}') break;
            if (h[p] == ',') { ++p; continue; }
            std::string name = str();
            skip_ws(); ++p;  // :
            skip_ws();
            if (name == "__metadata__") {  // flat string map
                size_t q = h.find('}', p);
                p = q + 1;
                continue;
            }
            ++p;  // {
            StTensor t;
            for (;;) {
                skip_ws();
                if (h[p] == '}') { ++p; break; }
                if (h[p] == ',') { ++p; continue; }
                std::string k = str();
                skip_ws(); ++p;
                if (k == "dtype") t.dtype = str();
                else if (k == "shape") t.shape = ints();
                else if (k == "data_offsets") { auto v = ints(); t.begin = v[0]; t.end = v[1]; }
            }
            t_[name] = t;
        }
    }
    std::string path_;
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* base_ = nullptr;
    const uint8_t* data_ = nullptr;
    std::map<std::string, StTensor> t_;
};

// MiMo-V2.6 checkpoint: expert e of every layer lives in model_pp0_ep{e/4}_shard0.safetensors
// (verified against model.safetensors.index.json for layers 1 and 47).
class MimoCheckpoint {
public:
    explicit MimoCheckpoint(std::string dir) : dir_(std::move(dir)) {}
    Expert load_expert(int layer, int expert) {
        StFile& f = shard(expert / 4);
        Expert x;
        alloc_expert(x);
        const std::string pre = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + ".";
        auto copy = [&](const std::string& name, uint8_t* dst, size_t want, int64_t r, int64_t c) {
            const StTensor& t = f.get(pre + name);
            if (t.dtype != "U8" || t.shape.size() != 2 || t.shape[0] != r || t.shape[1] != c || t.end - t.begin != want)
                throw std::runtime_error("unexpected tensor layout: " + pre + name);
            std::memcpy(dst, f.data(t), want);
        };
        const int64_t I = mimo::kExpertInter, H = mimo::kHidden;
        copy("gate_proj.weight", x.guq.data(), I * H / 2, I, H / 2);
        copy("up_proj.weight", x.guq.data() + I * H / 2, I * H / 2, I, H / 2);
        copy("gate_proj.weight_scale", x.gue.data(), I * H / QK, I, H / QK);
        copy("up_proj.weight_scale", x.gue.data() + I * H / QK, I * H / QK, I, H / QK);
        copy("down_proj.weight", x.dnq.data(), H * I / 2, H, I / 2);
        copy("down_proj.weight_scale", x.dne.data(), H * I / QK, H, I / QK);
        return x;
    }

private:
    StFile& shard(int ep) {
        auto it = files_.find(ep);
        if (it != files_.end()) return *it->second;
        auto f = std::make_unique<StFile>(dir_ + "/model_pp0_ep" + std::to_string(ep) + "_shard0.safetensors");
        return *(files_[ep] = std::move(f));
    }
    std::string dir_;
    std::map<int, std::unique_ptr<StFile>> files_;
};

// Activations that look like post-RMSNorm hidden states: N(0,1) with a few large outlier channels.
inline std::vector<float> make_activation(int n, uint64_t seed, int outliers = 8, float outlier_mag = 20.f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    for (int i = 0; i < outliers; ++i) x[rng() % n] *= outlier_mag;
    return x;
}

}  // namespace fp4
