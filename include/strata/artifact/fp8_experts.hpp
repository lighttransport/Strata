#pragma once
// GLM-5.3-Flash routed experts from the FP8 release (zai-org/GLM-5.3-Flash): every *.safetensors shard in a
// directory is indexed from its header (the release ships no index JSON), and one expert projection is
// dequantized as fp8_e4m3 x its 128x128 block's weight_scale_inv, rounded to f32 as the reference dequantizer does.
// Layout: model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight, row-major [out][in].
#include "strata/artifact/json.hpp"
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace strata::artifact::fp8 {

struct Tensor {
    std::string path, dtype;
    uint64_t offset = 0, bytes = 0;
    std::vector<uint64_t> shape;
};
using Index = std::map<std::string, Tensor>;

inline Index scan(const std::filesystem::path& dir) {
    Index out;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() != ".safetensors") continue;
        std::ifstream f(e.path(), std::ios::binary);
        uint64_t n = 0;
        if (!f.read(reinterpret_cast<char*>(&n), 8) || n > (64u << 20)) continue;
        std::string text(n, '\0');
        if (!f.read(text.data(), std::streamsize(n))) continue;
        auto j = parse_json(text);
        for (const auto& [name, t] : j.object) {
            if (name == "__metadata__") continue;
            Tensor v;
            v.path = e.path().string();
            v.dtype = t.at("dtype").string();
            for (const auto& d : t.at("shape").array) v.shape.push_back(d.integer());
            const auto& off = t.at("data_offsets").array;
            v.offset = 8 + n + off.at(0).integer();
            v.bytes = off.at(1).integer() - off.at(0).integer();
            out[name] = v;
        }
    }
    return out;
}

inline void pread_all(const std::string& path, uint64_t offset, void* dst, size_t bytes) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("fp8: open " + path);
    auto* p = static_cast<char*>(dst);
    while (bytes) {
        ssize_t r = ::pread(fd, p, bytes, off_t(offset));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) { ::close(fd); throw std::runtime_error("fp8: read " + path); }
        p += r; offset += uint64_t(r); bytes -= size_t(r);
    }
    ::close(fd);
}

inline float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = std::ldexp(float(m) / 8.0f, -6);
    else v = std::ldexp(1.0f + float(m) / 8.0f, e - 7);
    return s ? -v : v;
}

// projection: "gate_proj", "up_proj" or "down_proj".
inline std::string expert_name(int layer, int expert, const std::string& projection) {
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." + projection + ".weight";
}
inline bool has_expert(const Index& ts, int layer, int expert, const std::string& projection) {
    const auto n = expert_name(layer, expert, projection);
    return ts.count(n) && ts.count(n + "_scale_inv");
}

inline std::vector<float> load_expert(const Index& ts, int layer, int expert, const std::string& projection, size_t& rows, size_t& cols) {
    const auto name = expert_name(layer, expert, projection);
    auto w = ts.find(name), s = ts.find(name + "_scale_inv");
    if (w == ts.end() || s == ts.end()) throw std::runtime_error("fp8: missing " + name);
    if (w->second.dtype != "F8_E4M3" || w->second.shape.size() != 2 || s->second.dtype != "F32" || s->second.shape.size() != 2)
        throw std::runtime_error("fp8: unexpected dtype/shape for " + name);
    rows = w->second.shape[0];
    cols = w->second.shape[1];
    const size_t srows = s->second.shape[0], scols = s->second.shape[1];
    if (srows != (rows + 127) / 128 || scols != (cols + 127) / 128) throw std::runtime_error("fp8: scale grid of " + name);
    std::vector<uint8_t> raw(rows * cols);
    std::vector<float> scale(srows * scols);
    pread_all(w->second.path, w->second.offset, raw.data(), raw.size());
    pread_all(s->second.path, s->second.offset, scale.data(), scale.size() * 4);
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = e4m3(uint8_t(i));
    std::vector<float> out(rows * cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) out[r * cols + c] = lut[raw[r * cols + c]] * scale[(r / 128) * scols + c / 128];
    return out;
}

}  // namespace strata::artifact::fp8
