// DeepSeek V4.1 Flash pack tool: the routed experts of the release checkpoint (safetensors, MXFP4) requantized to a
// Strata expert sidecar GGUF, and the FP8/BF16 trunk to a trunk GGUF. The Engram tables are never touched: they
// stay in the release shards (FP8 rows + E8M0 scales) and are read per token from SSD by the runner.
//
//   strata-ds41f-pack experts MODEL_DIR OUT.gguf --gate=iq1_s --up=iq1_s --down=iq2_xxs (--imatrix=FILE | --uncalibrated)
//                     [--keep=FILE] [--layers=0-39] [--threads=16]
//   strata-ds41f-pack trunk   MODEL_DIR OUT.gguf [--trunk=q4_k|q5_k|q6_k|q8_0] [--head=q8_0|q4_k] [--threads=16]
//   strata-ds41f-pack verify  MODEL_DIR PACK.gguf
//   strata-ds41f-pack dequant MODEL_DIR LAYER EXPERT gate|up|down OUT.f32        (debug: exact MXFP4 -> f32 of one projection)
//
// Same sidecar format as strata-glm-q2-pack / strata-mimo-q2-pack: GGUF v3 holding blk.L.ffn_{gate,up,down}_exps.weight
// with strata.expert_pack.{version,source,calibrated,architecture="deepseek4",hash.<tensor>}; a REAP-pruned layer
// adds blk.L.ffn_exps_keep (I32, the kept source expert ids). The source fingerprint hashes every shard's path,
// size, mtime and header, so a pack binds to the checkpoint it was made from.
//
// MXFP4 as released: w{1,3} I8 [2304, 2560] and w2 I8 [5120, 1152] hold two FP4 (E2M1, sign bit 3) per byte, element
// 2c in the low nibble and 2c+1 in the high nibble (ds41f_kernels.c:96); the `.scale` tensor holds one E8M0 (2^(e-127))
// per 32 consecutive inputs. Dense FP8 (F8_E4M3) tensors carry a `.scale` of E8M0 blocks of 32 output rows x
// (cols / scale_cols) inputs (ds41f_kernels.c:157). Both dequantizers below are exact in f32.
//
// Calibration: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS require an importance matrix (ggml refuses them
// without one); --imatrix takes llama-imatrix's GGUF (`<pack tensor name>.in_sum2` F32 [cols, experts] and
// `.counts` F32 [experts]), produced by the exact A64FX run or by a Strata prefill. --uncalibrated is allowed for
// the K-quants only.
#include "strata/artifact/fp8_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/host/ds41f_names.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
namespace st = strata::artifact::fp8;
using strata::GgufFile;

constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
constexpr int kLayers = STRATA_DS41F_LAYERS, kExperts = STRATA_DS41F_EXPERTS, kHidden = STRATA_DS41F_HIDDEN, kFf = STRATA_DS41F_EXPERT_FF;
constexpr uint32_t kGgufI32 = 26;

template<class T> void write(std::ostream& out, T value) { out.write(reinterpret_cast<const char*>(&value), sizeof value); }
void string(std::ostream& out, const std::string& value) { write<uint64_t>(out, value.size()); out.write(value.data(), value.size()); }
uint64_t hash_bytes(uint64_t hash, const uint8_t* p, size_t n) { for (size_t i = 0; i < n; ++i) { hash ^= p[i]; hash *= 1099511628211ULL; } return hash; }

const std::map<std::string, ggml_type> kTypes = {
    {"q2_k", GGML_TYPE_Q2_K}, {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
    {"q8_0", GGML_TYPE_Q8_0}, {"iq1_s", GGML_TYPE_IQ1_S}, {"iq1_m", GGML_TYPE_IQ1_M}, {"iq2_xxs", GGML_TYPE_IQ2_XXS},
    {"iq2_xs", GGML_TYPE_IQ2_XS}, {"iq2_s", GGML_TYPE_IQ2_S}, {"iq3_xxs", GGML_TYPE_IQ3_XXS}, {"iq3_s", GGML_TYPE_IQ3_S},
    {"iq4_xs", GGML_TYPE_IQ4_XS}, {"bf16", GGML_TYPE_BF16}, {"f16", GGML_TYPE_F16}, {"f32", GGML_TYPE_F32},
};

ggml_type type_of(const std::string& name) {
    auto it = kTypes.find(name);
    if (it == kTypes.end()) throw std::runtime_error("unknown quantization type: " + name);
    return it->second;
}

// ---- exact dequantizers of the release formats --------------------------------------------------------------
constexpr float kFp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};   // E2M1 magnitudes

inline float e8m0(uint8_t e) { return e == 0xff ? std::nanf("") : std::ldexp(1.0f, int(e) - 127); }

inline float fp8_e4m3(uint8_t v) {
    const int sign = v >> 7, exp = (v >> 3) & 15, man = v & 7;
    float x;
    if (exp == 0) x = std::ldexp(float(man) / 8.0f, -6);
    else if (exp == 15 && man == 7) x = std::nanf("");
    else x = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
    return sign ? -x : x;
}

// rows x cols MXFP4 (packed bytes [rows][cols/2], scales [rows][cols/32]) -> f32 [rows][cols]
void dequant_mxfp4(const uint8_t* packed, const uint8_t* scales, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint8_t* p = packed + size_t(r) * cols / 2;
        const uint8_t* s = scales + size_t(r) * cols / 32;
        float* o = out + size_t(r) * cols;
        for (int c = 0; c < cols; ++c) {
            const unsigned code = (p[c / 2] >> ((c % 2) * 4)) & 15;
            const float mag = kFp4[code & 7];
            o[c] = ((code & 8) ? -mag : mag) * e8m0(s[c / 32]);
        }
    }
}

// rows x cols FP8 E4M3 with E8M0 scales in blocks of 32 rows x (cols / scale_cols) columns
void dequant_fp8(const uint8_t* w, const uint8_t* scales, int rows, int cols, int scale_rows, int scale_cols, float* out) {
    const int block_cols = cols / scale_cols;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int sr = std::min(r / 32, scale_rows - 1), sc = c / block_cols;
            out[size_t(r) * cols + c] = fp8_e4m3(w[size_t(r) * cols + c]) * e8m0(scales[size_t(sr) * scale_cols + sc]);
        }
}

inline float bf16(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ---- the checkpoint ------------------------------------------------------------------------------------------
struct Checkpoint {
    std::filesystem::path dir;
    st::Index index;
    const st::Tensor& at(const std::string& name) const {
        auto it = index.find(name);
        if (it == index.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    const st::Tensor* find(const std::string& name) const { auto it = index.find(name); return it == index.end() ? nullptr : &it->second; }
    std::vector<uint8_t> read(const st::Tensor& t) const { std::vector<uint8_t> v(t.bytes); st::pread_all(t.path, t.offset, v.data(), t.bytes); return v; }
};

// Experts present for a layer in the checkpoint (384 in the release; a synthetic or pruned checkpoint may hold fewer).
int experts_in_layer(const Checkpoint& c, int layer) {
    int count = 0;
    char name[STRATA_DS41F_NAME_MAX];
    for (int e = 0; e < kExperts; ++e) {
        strata_ds41f_expert_source_name(name, sizeof name, layer, e, STRATA_DS41F_GATE, "weight");
        if (!c.find(name)) break;
        ++count;
    }
    if (count == 0) throw std::runtime_error("layer " + std::to_string(layer) + " has no routed experts in the checkpoint");
    return count;
}

Checkpoint open_checkpoint(const std::string& dir) {
    Checkpoint c;
    c.dir = dir;
    c.index = st::scan(dir);
    if (c.index.empty()) throw std::runtime_error("no safetensors shards in " + dir);
    char name[STRATA_DS41F_NAME_MAX];
    strata_ds41f_expert_source_name(name, sizeof name, 0, 0, STRATA_DS41F_GATE, "weight");
    const auto& w1 = c.at(name);
    if (w1.dtype != "I8" || w1.shape != std::vector<uint64_t>{uint64_t(kFf), uint64_t(kHidden / 2)})
        throw std::runtime_error("not a DeepSeek V4.1 Flash checkpoint: " + std::string(name) + " is not I8 [2304, 2560]");
    return c;
}

// Fingerprint of the shards: path, size, mtime and the safetensors header of every shard. Binds a pack to its source.
uint64_t source_fingerprint(const Checkpoint& c) {
    std::set<std::string> paths;
    for (const auto& [_, t] : c.index) paths.insert(t.path);
    uint64_t hash = kFnvBasis;
    auto add = [&](const void* data, size_t size) { hash = hash_bytes(hash, static_cast<const uint8_t*>(data), size); };
    for (const auto& p : paths) {
        add(p.data(), p.size());
        const auto size = std::filesystem::file_size(p); add(&size, sizeof size);
        const auto stamp = std::filesystem::last_write_time(p).time_since_epoch().count(); add(&stamp, sizeof stamp);
        std::ifstream f(p, std::ios::binary); uint64_t n = 0;
        if (!f.read(reinterpret_cast<char*>(&n), 8)) throw std::runtime_error("cannot fingerprint " + p);
        std::vector<char> h(n); if (!f.read(h.data(), std::streamsize(n))) throw std::runtime_error("cannot fingerprint " + p);
        add(h.data(), h.size());
    }
    return hash;
}

struct Entry { std::string name; std::vector<uint64_t> shape; uint32_t type; uint64_t offset, bytes, hash = 0; double error = 0, norm = 0; };

std::string header(const std::vector<Entry>& entries, const std::map<std::string, std::string>& kv_strings,
                   const std::map<std::string, uint64_t>& kv_u64, const std::map<std::string, uint32_t>& kv_u32, bool calibrated,
                   const std::map<std::string, uint64_t>& hashes) {
    std::ostringstream out(std::ios::binary);
    write<uint32_t>(out, 0x46554747); write<uint32_t>(out, 3); write<uint64_t>(out, entries.size());
    write<uint64_t>(out, 1 + kv_strings.size() + kv_u64.size() + kv_u32.size() + hashes.size());
    string(out, "strata.expert_pack.calibrated"); write<uint32_t>(out, 7); write<uint8_t>(out, calibrated);
    for (const auto& [k, v] : kv_strings) { string(out, k); write<uint32_t>(out, 8); string(out, v); }
    for (const auto& [k, v] : kv_u64) { string(out, k); write<uint32_t>(out, 10); write<uint64_t>(out, v); }
    for (const auto& [k, v] : kv_u32) { string(out, k); write<uint32_t>(out, 4); write<uint32_t>(out, v); }
    for (const auto& [k, v] : hashes) { string(out, k); write<uint32_t>(out, 10); write<uint64_t>(out, v); }
    for (const auto& e : entries) {
        string(out, e.name); write<uint32_t>(out, uint32_t(e.shape.size()));
        for (auto d : e.shape) write<uint64_t>(out, d);
        write<uint32_t>(out, e.type); write<uint64_t>(out, e.offset);
    }
    auto result = out.str(); result.resize((result.size() + 31) / 32 * 32, 0); return result;
}

void drop_pages(const std::filesystem::path& path) {
#ifdef __linux__
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) { ::fdatasync(fd); ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
#endif
}

// llama-imatrix GGUF: per-expert mean squared input per column (the quantity ggml weights the error with).
std::vector<float> imatrix_importance(const GgufFile& im, const std::string& weight, int columns, int experts) {
    const auto* sum = im.find(weight + ".in_sum2"); const auto* cnt = im.find(weight + ".counts");
    if (!sum || !cnt) throw std::runtime_error("imatrix has no entry for " + weight);
    if (sum->type != 0 || cnt->type != 0 || sum->shape != std::vector<uint64_t>{uint64_t(columns), uint64_t(experts)} ||
        cnt->shape != std::vector<uint64_t>{uint64_t(experts)})
        throw std::runtime_error("imatrix geometry mismatch for " + weight);
    const float* s = reinterpret_cast<const float*>(im.tensor_data(*sum)); const float* c = reinterpret_cast<const float*>(im.tensor_data(*cnt));
    std::vector<double> pooled(columns, 0.); double total = 0;
    for (int e = 0; e < experts; ++e) { total += c[e]; for (int k = 0; k < columns; ++k) pooled[k] += s[size_t(e) * columns + k]; }
    std::vector<float> out(size_t(columns) * experts);
    for (int e = 0; e < experts; ++e) for (int k = 0; k < columns; ++k) {
        const double v = c[e] > 0 ? s[size_t(e) * columns + k] / c[e] : (total > 0 ? pooled[k] / total : 1.);
        if (!std::isfinite(v) || v < 0) throw std::runtime_error("nonfinite or negative imatrix value for " + weight);
        out[size_t(e) * columns + k] = float(std::max(v, 1e-12));
    }
    return out;
}

// --keep=FILE: lines `<layer>: e0 e1 e2 ...` (kept source expert ids, ascending); layers absent keep every expert.
std::map<int, std::vector<int>> read_keep(const std::string& path) {
    std::map<int, std::vector<int>> keep;
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot read keep map " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line); int layer; char colon;
        if (!(in >> layer >> colon) || colon != ':' || layer < 0 || layer >= kLayers) throw std::runtime_error("bad keep line: " + line);
        std::vector<int> ids; int e;
        while (in >> e) { if (e < 0 || e >= kExperts || (!ids.empty() && e <= ids.back())) throw std::runtime_error("bad keep ids: " + line); ids.push_back(e); }
        if (ids.empty()) throw std::runtime_error("empty keep list: " + line);
        keep[layer] = ids;
    }
    return keep;
}

std::pair<int, int> parse_layers(const std::string& text) {
    const auto dash = text.find('-');
    if (dash == std::string::npos) { const int l = std::stoi(text); return {l, l}; }
    return {std::stoi(text.substr(0, dash)), std::stoi(text.substr(dash + 1))};
}

struct Projection { int rows, cols; };
Projection geometry(int proj) { return proj == STRATA_DS41F_DOWN ? Projection{kHidden, kFf} : Projection{kFf, kHidden}; }

void load_expert(const Checkpoint& c, int layer, int expert, int proj, std::vector<float>& weights) {
    char wname[STRATA_DS41F_NAME_MAX], sname[STRATA_DS41F_NAME_MAX];
    strata_ds41f_expert_source_name(wname, sizeof wname, layer, expert, strata_ds41f_proj(proj), "weight");
    strata_ds41f_expert_source_name(sname, sizeof sname, layer, expert, strata_ds41f_proj(proj), "scale");
    const auto& w = c.at(wname); const auto& s = c.at(sname);
    const auto g = geometry(proj);
    if (w.dtype != "I8" || w.bytes != size_t(g.rows) * g.cols / 2 || s.bytes != size_t(g.rows) * g.cols / 32)
        throw std::runtime_error(std::string("unexpected expert tensor layout: ") + wname);
    const auto wb = c.read(w), sb = c.read(s);
    weights.resize(size_t(g.rows) * g.cols);
    dequant_mxfp4(wb.data(), sb.data(), g.rows, g.cols, weights.data());
    for (float v : weights) if (!std::isfinite(v)) throw std::runtime_error(std::string("nonfinite source weight in ") + wname);
}

// ---- experts -------------------------------------------------------------------------------------------------
int cmd_experts(int argc, char** argv) {
    if (argc < 5) throw std::runtime_error("usage: strata-ds41f-pack experts MODEL_DIR OUT.gguf --gate=T --up=T --down=T (--imatrix=FILE | --uncalibrated) [--keep=FILE] [--layers=0-39] [--threads=16]");
    std::string gate = "iq1_s", up = "iq1_s", down = "iq2_xxs", imatrix_path, keep_path; bool uncalibrated = false; int first = 0, last = kLayers - 1, threads = 16;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a.starts_with("--gate=")) gate = a.substr(7); else if (a.starts_with("--up=")) up = a.substr(5); else if (a.starts_with("--down=")) down = a.substr(7);
        else if (a.starts_with("--imatrix=")) imatrix_path = a.substr(10); else if (a == "--uncalibrated") uncalibrated = true;
        else if (a.starts_with("--keep=")) keep_path = a.substr(7); else if (a.starts_with("--layers=")) std::tie(first, last) = parse_layers(a.substr(9));
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10)); else throw std::runtime_error("unknown argument " + a);
    }
    if (uncalibrated == !imatrix_path.empty() || first < 0 || first > last || last >= kLayers || threads < 1 || threads > 64)
        throw std::runtime_error("invalid settings: choose --imatrix=FILE or --uncalibrated; layers 0..39; threads 1..64");
    const ggml_type types[3] = {type_of(gate), type_of(up), type_of(down)};
    for (auto t : types) if (uncalibrated && ggml_quantize_requires_imatrix(t)) throw std::runtime_error(std::string(ggml_type_name(t)) + " requires --imatrix");
    std::unique_ptr<GgufFile> imatrix;
    if (!imatrix_path.empty()) {
        imatrix = std::make_unique<GgufFile>(imatrix_path);
        const auto* type = imatrix->get("general.type");
        if (!type || type->s != "imatrix") throw std::runtime_error("not a llama-imatrix GGUF: " + imatrix_path);
    }
    const auto keep = keep_path.empty() ? std::map<int, std::vector<int>>{} : read_keep(keep_path);
    const auto c = open_checkpoint(argv[2]);
    const std::filesystem::path output(argv[3]), partial = output.string() + ".partial";
    if (std::filesystem::exists(output) || std::filesystem::exists(partial)) throw std::runtime_error("refusing to overwrite existing pack or partial file");
    // plan: per layer the three projections (ggml ne0 = inputs), plus the keep list of a pruned layer
    std::vector<Entry> entries; std::vector<std::vector<int>> keep_lists; std::vector<int> source_counts; uint64_t offset = 0;
    for (int l = first; l <= last; ++l) {
        const int present = experts_in_layer(c, l);
        std::vector<int> ids;
        if (auto it = keep.find(l); it != keep.end()) { ids = it->second; if (ids.back() >= present) throw std::runtime_error("keep list exceeds the experts present in layer " + std::to_string(l)); }
        else for (int e = 0; e < present; ++e) ids.push_back(e);
        for (int proj = 0; proj < 3; ++proj) {
            char name[STRATA_DS41F_NAME_MAX]; strata_ds41f_expert_pack_name(name, sizeof name, l, strata_ds41f_proj(proj));
            const auto g = geometry(proj);
            const uint64_t bytes = uint64_t(ggml_row_size(types[proj], g.cols)) * g.rows * ids.size();
            entries.push_back({name, {uint64_t(g.cols), uint64_t(g.rows), ids.size()}, uint32_t(types[proj]), offset, bytes}); offset += bytes;
            keep_lists.push_back(ids); source_counts.push_back(present);
        }
        if (ids.size() != size_t(present)) {
            char name[STRATA_DS41F_NAME_MAX]; strata_ds41f_keep_name(name, sizeof name, l);
            const uint64_t bytes = (ids.size() * 4 + 31) / 32 * 32;
            entries.push_back({name, {ids.size()}, kGgufI32, offset, bytes}); offset += bytes; keep_lists.push_back(ids); source_counts.push_back(present);
        }
    }
    if (std::filesystem::space(output.parent_path().empty() ? "." : output.parent_path()).available < offset + (uint64_t(2) << 30))
        throw std::runtime_error("insufficient disk capacity for pack plus 2 GiB reserve");
    const auto fingerprint = source_fingerprint(c);
    std::map<std::string, std::string> kvs = {{"strata.expert_pack.architecture", "deepseek4"}, {"strata.expert_pack.formats", gate + "," + up + "," + down}};
    std::map<std::string, uint64_t> kvu = {{"strata.expert_pack.source", fingerprint}};
    std::map<std::string, uint32_t> kv32 = {{"strata.expert_pack.version", 1}, {"strata.expert_pack.layer_count", uint32_t(kLayers)},
                                            {"strata.expert_pack.expert_count", uint32_t(kExperts)}};
    std::map<std::string, uint64_t> hashes;
    for (const auto& e : entries) hashes["strata.expert_pack.hash." + e.name] = 0;
    auto h = header(entries, kvs, kvu, kv32, bool(imatrix), hashes);
    std::cerr << "PACK plan tensors=" << entries.size() << " bytes=" << offset + h.size() << " source_fingerprint=" << fingerprint << '\n';
    std::ofstream out(partial, std::ios::binary | std::ios::trunc); out.exceptions(std::ios::failbit | std::ios::badbit); out.write(h.data(), h.size());
    ggml_cpu_init();
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t ei = 0; ei < entries.size(); ++ei) {
        auto& e = entries[ei]; const auto& ids = keep_lists[ei];
        e.hash = kFnvBasis;
        if (e.type == kGgufI32) {
            std::vector<uint8_t> buf(e.bytes, 0); std::memcpy(buf.data(), ids.data(), ids.size() * 4);
            out.write(reinterpret_cast<const char*>(buf.data()), buf.size()); e.hash = hash_bytes(e.hash, buf.data(), buf.size()); continue;
        }
        int layer = 0; strata_ds41f_proj proj; strata_ds41f_parse_pack_name(e.name.c_str(), &layer, &proj);
        const auto g = geometry(proj); const size_t slice = e.bytes / ids.size();
        const std::vector<float> importance = imatrix ? imatrix_importance(*imatrix, e.name, g.cols, source_counts[ei]) : std::vector<float>{};
        for (size_t base = 0; base < ids.size(); base += threads) {
            const int count = int(std::min<size_t>(threads, ids.size() - base));
            std::vector<std::vector<uint8_t>> packed(count, std::vector<uint8_t>(slice));
            std::vector<std::jthread> workers; std::vector<std::exception_ptr> errors(count); std::vector<double> err(count), norm(count);
            for (int j = 0; j < count; ++j) workers.emplace_back([&, j] { try {
                const int expert = ids[base + j]; std::vector<float> weights, decoded(g.cols);
                load_expert(c, layer, expert, proj, weights);
                const float* im = importance.empty() ? nullptr : importance.data() + size_t(expert) * g.cols;
                const auto written = ggml_quantize_chunk(ggml_type(e.type), weights.data(), packed[j].data(), 0, g.rows, g.cols, im);
                if (written != slice) throw std::runtime_error("quantizer byte count mismatch");
                const auto* traits = ggml_get_type_traits(ggml_type(e.type));
                for (int r = 0; r < g.rows; ++r) {
                    traits->to_float(packed[j].data() + size_t(r) * slice / g.rows, decoded.data(), g.cols);
                    for (int k = 0; k < g.cols; ++k) { const double w = weights[size_t(r) * g.cols + k], scale = im ? im[k] : 1., d = double(decoded[k]) - w; err[j] += scale * d * d; norm[j] += scale * w * w; }
                }
            } catch (...) { errors[j] = std::current_exception(); } });
            for (auto& w : workers) w.join();
            for (auto& error : errors) if (error) std::rethrow_exception(error);
            for (int j = 0; j < count; ++j) { out.write(reinterpret_cast<const char*>(packed[j].data()), slice); e.hash = hash_bytes(e.hash, packed[j].data(), slice); e.error += err[j]; e.norm += norm[j]; }
        }
        out.flush(); drop_pages(partial);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cerr << "PACK tensor=" << e.name << " type=" << ggml_type_name(ggml_type(e.type)) << " experts=" << ids.size() << " bytes=" << e.bytes
                  << " normalized_mse=" << std::setprecision(9) << e.error / std::max(e.norm, 1e-30) << " done=" << ei + 1 << "/" << entries.size()
                  << " elapsed_s=" << std::setprecision(5) << s << '\n';
    }
    if (source_fingerprint(c) != fingerprint) throw std::runtime_error("source identity changed during conversion");
    for (const auto& e : entries) hashes["strata.expert_pack.hash." + e.name] = e.hash;
    h = header(entries, kvs, kvu, kv32, bool(imatrix), hashes); out.seekp(0); out.write(h.data(), h.size()); out.close();
    std::filesystem::rename(partial, output);
    std::ofstream report(output.string() + ".quality.json"); report.exceptions(std::ios::failbit | std::ios::badbit);
    report << "{\"architecture\":\"deepseek4\",\"source_fingerprint\":\"" << fingerprint << "\",\"calibrated\":" << (imatrix ? "true" : "false")
           << ",\"formats\":[\"" << gate << "\",\"" << up << "\",\"" << down << "\"],\"projections\":[";
    bool firstp = true;
    for (const auto& e : entries) { if (e.type == kGgufI32) continue; if (!firstp) report << ','; firstp = false;
        report << "{\"name\":\"" << e.name << "\",\"experts\":" << e.shape[2] << ",\"bytes\":" << e.bytes << ",\"normalized_mse\":" << std::setprecision(12) << e.error / std::max(e.norm, 1e-30) << '}'; }
    report << "]}\n";
    std::cerr << "PACK complete path=" << output << " bytes=" << offset + h.size() << '\n';
    return 0;
}

// ---- trunk ----------------------------------------------------------------------------------------------------
// Every tensor that is not a routed expert and not an Engram table: FP8 (+ E8M0 scale) and BF16 matrices are
// dequantized to f32 and requantized (--trunk for the projections, --head for head.weight); embed.weight and every
// vector (norms, biases, hc_* parameters, scales) are copied as BF16/F32. Names stay the safetensors names.
int cmd_trunk(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("usage: strata-ds41f-pack trunk MODEL_DIR OUT.gguf [--trunk=q4_k] [--head=q8_0] [--threads=16]");
    std::string trunk = "q4_k", head = "q8_0"; int threads = 16;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a.starts_with("--trunk=")) trunk = a.substr(8); else if (a.starts_with("--head=")) head = a.substr(7);
        else if (a.starts_with("--threads=")) threads = std::stoi(a.substr(10)); else throw std::runtime_error("unknown argument " + a);
    }
    const ggml_type trunk_type = type_of(trunk), head_type = type_of(head);
    if (ggml_quantize_requires_imatrix(trunk_type) || ggml_quantize_requires_imatrix(head_type)) throw std::runtime_error("trunk formats must not need an imatrix");
    const auto c = open_checkpoint(argv[2]);
    const std::filesystem::path output(argv[3]), partial = output.string() + ".partial";
    if (std::filesystem::exists(output) || std::filesystem::exists(partial)) throw std::runtime_error("refusing to overwrite existing file");
    struct Item { std::string name; const st::Tensor* w; const st::Tensor* scale; ggml_type type; bool copy; };
    std::vector<Item> items;
    for (const auto& [name, t] : c.index) {
        if (strata_ds41f_is_routed_expert(name.c_str()) || strata_ds41f_is_engram_table(name.c_str()) || name.ends_with(".scale")) continue;
        const bool matrix = t.shape.size() == 2 && t.shape[0] >= 256 && t.shape[1] >= 256;
        const st::Tensor* scale = c.find(name.substr(0, name.size() - 7) + ".scale");
        if (t.dtype == "F8_E4M3") { if (!scale) throw std::runtime_error("FP8 tensor without scale: " + name); items.push_back({name, &t, scale, matrix ? trunk_type : GGML_TYPE_F32, false}); }
        else if (t.dtype == "BF16") items.push_back({name, &t, nullptr, name == "head.weight" ? head_type : (matrix && name != "embed.weight" ? trunk_type : GGML_TYPE_BF16), name != "head.weight" && !(matrix && name != "embed.weight")});
        else if (t.dtype == "F32") items.push_back({name, &t, nullptr, GGML_TYPE_F32, true});
        else if (t.dtype == "I8" || t.dtype == "F8_E8M0") continue;   // vision tower or DSpark packed tensors: not part of the text trunk
        else throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.name < b.name; });
    std::vector<Entry> entries; uint64_t offset = 0;
    for (const auto& it : items) {
        std::vector<uint64_t> shape(it.w->shape.rbegin(), it.w->shape.rend());   // ggml ne0 = innermost
        uint64_t bytes = 1; for (size_t i = 1; i < shape.size(); ++i) bytes *= shape[i];
        bytes *= uint64_t(ggml_row_size(it.type, int64_t(shape[0])));
        bytes = (bytes + 31) / 32 * 32;
        entries.push_back({it.name, shape, uint32_t(it.type), offset, bytes}); offset += bytes;
    }
    const auto fingerprint = source_fingerprint(c);
    std::map<std::string, std::string> kvs = {{"general.architecture", "deepseek4"}, {"strata.trunk.formats", trunk + "," + head}};
    std::map<std::string, uint64_t> kvu = {{"strata.trunk.source", fingerprint}};
    std::map<std::string, uint32_t> kv32 = {{"strata.trunk.version", 1}, {"deepseek4.block_count", uint32_t(kLayers)}, {"deepseek4.embedding_length", uint32_t(kHidden)},
                                            {"deepseek4.expert_count", uint32_t(kExperts)}, {"deepseek4.expert_used_count", 6}, {"deepseek4.expert_feed_forward_length", uint32_t(kFf)}};
    std::map<std::string, uint64_t> hashes; for (const auto& e : entries) hashes["strata.trunk.hash." + e.name] = 0;
    auto h = header(entries, kvs, kvu, kv32, false, hashes);
    std::cerr << "TRUNK plan tensors=" << entries.size() << " bytes=" << offset + h.size() << '\n';
    std::ofstream out(partial, std::ios::binary | std::ios::trunc); out.exceptions(std::ios::failbit | std::ios::badbit); out.write(h.data(), h.size());
    ggml_cpu_init();
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i]; auto& e = entries[i];
        const auto raw = c.read(*it.w);
        std::vector<uint8_t> payload(e.bytes, 0);
        const int64_t ne0 = int64_t(e.shape[0]); int64_t rows = 1; for (size_t k = 1; k < e.shape.size(); ++k) rows *= int64_t(e.shape[k]);
        std::vector<float> f32(size_t(rows) * ne0);
        if (it.w->dtype == "F8_E4M3") {
            const auto sb = c.read(*it.scale);
            const int srows = int(it.scale->shape.size() == 2 ? it.scale->shape[0] : 1), scols = int(it.scale->shape.size() == 2 ? it.scale->shape[1] : it.scale->shape[0]);
            if (it.w->shape.size() != 2 || int64_t(ne0) % scols) throw std::runtime_error("FP8 scale geometry mismatch: " + it.name);
            dequant_fp8(raw.data(), sb.data(), int(rows), int(ne0), srows, scols, f32.data());
        } else if (it.w->dtype == "BF16") { for (size_t k = 0; k < f32.size(); ++k) f32[k] = bf16(reinterpret_cast<const uint16_t*>(raw.data())[k]); }
        else std::memcpy(f32.data(), raw.data(), f32.size() * 4);
        for (float v : f32) if (!std::isfinite(v)) throw std::runtime_error("nonfinite value in " + it.name);
        const size_t written = it.type == GGML_TYPE_F32 ? (std::memcpy(payload.data(), f32.data(), f32.size() * 4), f32.size() * 4)
                             : ggml_quantize_chunk(it.type, f32.data(), payload.data(), 0, rows, ne0, nullptr);
        if (written > e.bytes) throw std::runtime_error("quantizer overflow: " + it.name);
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size()); e.hash = hash_bytes(kFnvBasis, payload.data(), payload.size());
        if (ggml_is_quantized(it.type)) {
            std::vector<float> dec(ne0); const auto* traits = ggml_get_type_traits(it.type); const size_t row_bytes = ggml_row_size(it.type, ne0);
            for (int64_t r = 0; r < rows; ++r) { traits->to_float(payload.data() + size_t(r) * row_bytes, dec.data(), ne0);
                for (int64_t k = 0; k < ne0; ++k) { const double w = f32[size_t(r) * ne0 + k], d = dec[k] - w; e.error += d * d; e.norm += w * w; } }
        }
        std::cerr << "TRUNK tensor=" << it.name << " type=" << ggml_type_name(it.type) << " bytes=" << e.bytes
                  << (ggml_is_quantized(it.type) ? " normalized_mse=" + std::to_string(e.error / std::max(e.norm, 1e-30)) : "") << " done=" << i + 1 << "/" << items.size() << '\n';
        (void)threads;
    }
    for (const auto& e : entries) hashes["strata.trunk.hash." + e.name] = e.hash;
    h = header(entries, kvs, kvu, kv32, false, hashes); out.seekp(0); out.write(h.data(), h.size()); out.close();
    std::filesystem::rename(partial, output);
    std::cerr << "TRUNK complete path=" << output << " bytes=" << offset + h.size() << '\n';
    return 0;
}

// ---- verify ----------------------------------------------------------------------------------------------------
int cmd_verify(int argc, char** argv) {
    if (argc != 4) throw std::runtime_error("usage: strata-ds41f-pack verify MODEL_DIR PACK.gguf");
    const auto c = open_checkpoint(argv[2]);
    GgufFile pack(argv[3]);
    const bool trunk = pack.get("strata.trunk.version") != nullptr;
    const std::string prefix = trunk ? "strata.trunk." : "strata.expert_pack.";
    const auto* version = pack.get(prefix + "version"); const auto* source = pack.get(prefix + "source");
    const auto* arch = pack.get(trunk ? "general.architecture" : "strata.expert_pack.architecture");
    if (!version || version->u != 1 || !source || source->u != source_fingerprint(c) || !arch || arch->s != "deepseek4")
        throw std::runtime_error("pack: version, architecture or source fingerprint mismatch");
    size_t checked = 0; const uint64_t payload = pack.file_size() - pack.data_start();
    for (const auto& t : pack.tensors()) {
        uint64_t bytes;
        if (t.type == kGgufI32) bytes = (t.shape[0] * 4 + 31) / 32 * 32;
        else { uint64_t rows = 1; for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i]; bytes = uint64_t(ggml_row_size(ggml_type(t.type), int64_t(t.shape[0]))) * rows; if (trunk) bytes = (bytes + 31) / 32 * 32; }
        if (!trunk) { int layer; strata_ds41f_proj proj; if (strata_ds41f_parse_pack_name(t.name.c_str(), &layer, &proj) != 1 && t.type != kGgufI32) throw std::runtime_error("pack: unexpected tensor " + t.name); }
        if (t.offset > payload || bytes > payload - t.offset) throw std::runtime_error("pack: invalid payload bounds " + t.name);
        const auto* expected = pack.get(prefix + "hash." + t.name);
        if (!expected || hash_bytes(kFnvBasis, pack.tensor_data(t), bytes) != expected->u) throw std::runtime_error("payload checksum mismatch: " + t.name);
        pack.discard_tensor_pages(t, bytes); ++checked;
    }
    std::cout << "PACK verified architecture=deepseek4 kind=" << (trunk ? "trunk" : "experts") << " tensors=" << checked << '\n';
    return 0;
}

int cmd_dequant(int argc, char** argv) {
    if (argc != 7) throw std::runtime_error("usage: strata-ds41f-pack dequant MODEL_DIR LAYER EXPERT gate|up|down OUT.f32");
    const auto c = open_checkpoint(argv[2]);
    const std::string which = argv[5]; const int proj = which == "gate" ? 0 : which == "up" ? 1 : which == "down" ? 2 : -1;
    if (proj < 0) throw std::runtime_error("projection must be gate, up or down");
    std::vector<float> weights; load_expert(c, std::stoi(argv[3]), std::stoi(argv[4]), proj, weights);
    std::ofstream out(argv[6], std::ios::binary); out.write(reinterpret_cast<const char*>(weights.data()), weights.size() * 4);
    std::cout << "DEQUANT rows=" << geometry(proj).rows << " cols=" << geometry(proj).cols << " values=" << weights.size() << '\n';
    return 0;
}
}  // namespace

int main(int argc, char** argv) { try {
    if (argc < 2) throw std::runtime_error("usage: strata-ds41f-pack experts|trunk|verify|dequant ...");
    const std::string cmd = argv[1];
    if (cmd == "experts") return cmd_experts(argc, argv);
    if (cmd == "trunk") return cmd_trunk(argc, argv);
    if (cmd == "verify" || cmd == "--verify") return cmd_verify(argc, argv);
    if (cmd == "dequant") return cmd_dequant(argc, argv);
    throw std::runtime_error("unknown command " + cmd);
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
