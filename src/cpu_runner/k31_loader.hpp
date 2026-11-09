// k31_loader.hpp: safetensors reading and weight packing for k31pro_runner's real-weight load.
//
// A piece is one rank's part of one checkpoint tensor, written to that rank's arena at a 256-byte aligned offset:
//   ROWS  rows [r0, r0 + rn) of a BF16 [rows, cols] matrix (contiguous)
//   COLS  columns [c0, c0 + cn) of a BF16 [rows, cols] matrix, stored [rows][cn]
//   MX    rows [r0, r0 + rn), 32-blocks [b0, b0 + bn) of a compressed-tensors MXFP4 matrix (U8 weight_packed
//         [rows, cols / 2] + U8 E8M0 weight_scale [rows, cols / 32]), stored in the GGUF block layout ds41_rows reads
//         (E8M0 byte + 16 bytes; low nibble element j, high nibble element j + 16)
//   VEC   elements [c0, c0 + cn) of a tensor (any shape) converted to F32
// Pieces are packed in row ranges, so a source never holds a whole large tensor; pack() of row range [a, b) writes
// exactly the destination bytes of those rows. hash_words() is additive over 8-byte words keyed by their arena word
// index, so the source (packing chunks) and the destination (its arena) compute the same sum.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace k31ld {

enum Dtype { DT_BF16, DT_F16, DT_F32, DT_U8, DT_OTHER };
inline int dtype_size(Dtype d) { return d == DT_F32 ? 4 : d == DT_U8 ? 1 : 2; }

struct Tensor {
    std::string name;
    Dtype dt = DT_OTHER;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;   // absolute file offsets
    int64_t rows() const { return shape.empty() ? 1 : shape[0]; }
    int64_t numel() const { int64_t n = 1; for (int64_t s : shape) n *= s; return n; }
    int64_t row_elems() const { return rows() ? numel() / rows() : 0; }
};

// ------------------------------------------------------------------ minimal JSON (safetensors headers, index)
struct Json {
    const char *p, *e;
    explicit Json(const std::string &s) : p(s.data()), e(s.data() + s.size()) {}
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    [[noreturn]] void fail(const char *what) { throw std::runtime_error(std::string("k31 json: ") + what); }
    void expect(char c) { ws(); if (p >= e || *p != c) fail("unexpected character"); ++p; }
    bool peek(char c) { ws(); return p < e && *p == c; }
    std::string str() {
        expect('"');
        std::string s;
        while (p < e && *p != '"') {
            if (*p == '\\') { ++p; if (p >= e) fail("bad escape"); s.push_back(*p == 'n' ? '\n' : *p); ++p; }
            else s.push_back(*p++);
        }
        if (p >= e) fail("unterminated string");
        ++p;
        return s;
    }
    int64_t num() {
        ws();
        char *q = nullptr;
        const long long v = std::strtoll(p, &q, 10);
        if (q == p) fail("number expected");
        p = q;
        while (p < e && (*p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-' || (*p >= '0' && *p <= '9'))) ++p;
        return v;
    }
    void skip() {
        ws();
        if (p >= e) fail("value expected");
        if (*p == '"') { str(); return; }
        if (*p == '{') { ++p; if (peek('}')) { ++p; return; } for (;;) { str(); expect(':'); skip(); if (peek(',')) { ++p; continue; } expect('}'); return; } }
        if (*p == '[') { ++p; if (peek(']')) { ++p; return; } for (;;) { skip(); if (peek(',')) { ++p; continue; } expect(']'); return; } }
        while (p < e && *p != ',' && *p != '}' && *p != ']') ++p;
    }
};

inline Dtype parse_dtype(const std::string &s) {
    if (s == "BF16") return DT_BF16;
    if (s == "F16") return DT_F16;
    if (s == "F32") return DT_F32;
    if (s == "U8") return DT_U8;
    return DT_OTHER;
}

// parse a safetensors header (the JSON after the 8-byte length); data offsets made absolute with base
inline std::vector<Tensor> parse_header(const std::string &js, uint64_t base) {
    Json j(js);
    std::vector<Tensor> out;
    j.expect('{');
    if (j.peek('}')) return out;
    for (;;) {
        Tensor t;
        t.name = j.str();
        j.expect(':');
        if (t.name == "__metadata__") j.skip();
        else {
            j.expect('{');
            for (;;) {
                const std::string k = j.str();
                j.expect(':');
                if (k == "dtype") t.dt = parse_dtype(j.str());
                else if (k == "shape") {
                    j.expect('[');
                    if (!j.peek(']')) for (;;) { t.shape.push_back(j.num()); if (j.peek(',')) { ++j.p; continue; } break; }
                    j.expect(']');
                } else if (k == "data_offsets") {
                    j.expect('['); t.begin = uint64_t(j.num()) + base; j.expect(','); t.end = uint64_t(j.num()) + base; j.expect(']');
                } else j.skip();
                if (j.peek(',')) { ++j.p; continue; }
                j.expect('}');
                break;
            }
            out.push_back(std::move(t));
        }
        if (j.peek(',')) { ++j.p; continue; }
        j.expect('}');
        break;
    }
    return out;
}

inline void pread_all(int fd, void *dst, size_t n, uint64_t off, const std::string &path) {
    char *d = static_cast<char *>(dst);
    while (n) {
        const ssize_t r = ::pread(fd, d, n, off_t(off));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("k31: read failed: " + path);
        d += r; n -= size_t(r); off += uint64_t(r);
    }
}

struct File {
    std::string path;
    int fd = -1;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> by_name;
    void open(const std::string &p) {
        path = p;
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("k31: cannot open " + p);
        uint64_t n = 0;
        pread_all(fd, &n, 8, 0, p);
        if (n == 0 || n > (1ull << 31)) throw std::runtime_error("k31: bad safetensors header size in " + p);
        std::string js(size_t(n), '\0');
        pread_all(fd, &js[0], size_t(n), 8, p);
        tensors = parse_header(js, 8 + n);
        for (size_t i = 0; i < tensors.size(); ++i) by_name[tensors[i].name] = i;
    }
    const Tensor *find(const std::string &n) const {
        auto it = by_name.find(n);
        return it == by_name.end() ? nullptr : &tensors[it->second];
    }
    void close() { if (fd >= 0) ::close(fd); fd = -1; }
};

// model.safetensors.index.json: tensor name -> file name
inline std::unordered_map<std::string, std::string> read_index(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("k31: cannot open " + path);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    std::unordered_map<std::string, std::string> m;
    Json j(s);
    j.expect('{');
    for (;;) {
        const std::string k = j.str();
        j.expect(':');
        if (k == "weight_map") {
            j.expect('{');
            if (!j.peek('}')) for (;;) { std::string a = j.str(); j.expect(':'); m[a] = j.str(); if (j.peek(',')) { ++j.p; continue; } break; }
            j.expect('}');
        } else j.skip();
        if (j.peek(',')) { ++j.p; continue; }
        j.expect('}');
        break;
    }
    return m;
}

// ------------------------------------------------------------------ pieces
enum Kind { ROWS, COLS, MX, VEC };
// tensor names interned: a piece carries an id (plans of expert-TP layouts hold millions of pieces)
inline std::vector<std::string> &names() { static std::vector<std::string> v; return v; }
inline uint32_t name_id(const std::string &n) {
    static std::unordered_map<std::string, uint32_t> m;
    auto it = m.find(n);
    if (it != m.end()) return it->second;
    const uint32_t id = uint32_t(names().size());
    names().push_back(n);
    m.emplace(n, id);
    return id;
}
struct Piece {
    Kind kind = ROWS;
    uint32_t tid = 0;        // checkpoint name id (MX: the base, without .weight_packed / .weight_scale)
    int32_t r0 = 0, rn = 0;  // ROWS / MX rows; COLS: all rows (rn = rows)
    int32_t c0 = 0, cn = 0;  // COLS columns, VEC elements, MX: blocks b0 = c0, bn = cn
    int32_t cols = 0;        // global columns (ROWS / COLS / MX)
    int32_t dst = 0;
    uint64_t off = 0, bytes = 0;
    const std::string &tensor() const { return names()[tid]; }
};
// planar MXFP4 rows (k31_mxfp4p_rows): pieces with a multiple of 4 blocks, rows padded to 64 bytes
inline bool g_planar = false;
inline bool mx_planar(int64_t bn) { return g_planar && bn % 4 == 0; }
inline uint64_t mx_row_bytes(int64_t bn) { return mx_planar(bn) ? (uint64_t(bn) * 17 + 63) & ~uint64_t(63) : uint64_t(bn) * 17; }
inline uint64_t piece_row_bytes(const Piece &p) {
    switch (p.kind) {
    case ROWS: return uint64_t(p.cols) * 2;
    case COLS: return uint64_t(p.cn) * 2;
    case MX: return mx_row_bytes(p.cn);
    case VEC: return 0;
    }
    return 0;
}
inline uint64_t piece_bytes(const Piece &p) {
    return p.kind == VEC ? uint64_t(p.cn) * 4 : piece_row_bytes(p) * uint64_t(p.rn);
}

inline float bf16_to_f32(uint16_t b) { const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int ee = -1; uint32_t mm = m; do { ++ee; mm <<= 1; } while (!(mm & 0x400)); u = s | uint32_t(127 - 15 - ee) << 23 | (mm & 0x3ff) << 13; }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e + 112) << 23 | m << 13;
    float f; std::memcpy(&f, &u, 4); return f;
}

// MXFP4 rows: compressed-tensors (packed: element 2i low nibble, 2i + 1 high nibble; scale: one E8M0 per 32) ->
// GGUF block_mxfp4 (e, qs[16]: qs[j] = q[j] | q[j + 16] << 4). packed / scale point at row 0 of the given rows.
// (planar: per row the bn x 16 nibble bytes, then bn E8M0 bytes, zero padding to the row stride)
inline void repack_mx_rows(uint8_t *dst, const uint8_t *packed, const uint8_t *scale, int64_t rows, int64_t cols,
                           int64_t b0, int64_t bn) {
    const int64_t pb = cols / 2, sb = cols / 32;
    const bool pl = mx_planar(bn);
    const uint64_t rb = mx_row_bytes(bn);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t *pr = packed + r * pb, *sr = scale + r * sb;
        uint8_t *dr = dst + uint64_t(r) * rb;
        if (pl) std::memset(dr + bn * 17, 0, size_t(rb - uint64_t(bn) * 17));
        for (int64_t b = 0; b < bn; ++b) {
            const int64_t gb = b0 + b;
            const uint8_t *q = pr + gb * 16;   // 16 bytes = 32 elements
            uint8_t *qs = pl ? dr + b * 16 : dr + b * 17 + 1;
            (pl ? dr[bn * 16 + b] : dr[b * 17]) = sr[gb];
            for (int j = 0; j < 16; ++j) {
                const int lo = j, hi = j + 16;   // element indices within the block
                const uint8_t ql = (q[lo / 2] >> ((lo & 1) * 4)) & 15, qh = (q[hi / 2] >> ((hi & 1) * 4)) & 15;
                qs[j] = uint8_t(ql | qh << 4);
            }
        }
    }
}

// pack piece rows [a, b) (relative to the piece) from raw source bytes. src: ROWS / COLS: the tensor's rows
// [p.r0 + a, p.r0 + b) (COLS: all columns); MX: packed rows, scale_src: scale rows. Returns the bytes written.
inline uint64_t pack_rows(const Piece &p, uint8_t *dst, const uint8_t *src, const uint8_t *scale_src, int64_t a, int64_t b) {
    const int64_t n = b - a;
    switch (p.kind) {
    case ROWS: std::memcpy(dst, src, size_t(n * p.cols * 2)); return uint64_t(n * p.cols * 2);
    case COLS:
        for (int64_t r = 0; r < n; ++r) std::memcpy(dst + r * p.cn * 2, src + (r * p.cols + p.c0) * 2, size_t(p.cn * 2));
        return uint64_t(n * p.cn * 2);
    case MX: repack_mx_rows(dst, src, scale_src, n, p.cols, p.c0, p.cn); return uint64_t(n) * mx_row_bytes(p.cn);
    case VEC: break;
    }
    return 0;
}

// VEC: elements [c0, c0 + cn) of src (the tensor's full data) as F32
inline void pack_vec(const Piece &p, Dtype dt, uint8_t *dst, const uint8_t *src) {
    float *o = reinterpret_cast<float *>(dst);
    for (int64_t i = 0; i < p.cn; ++i) {
        const int64_t k = p.c0 + i;
        uint16_t h;
        switch (dt) {
        case DT_F32: std::memcpy(&o[i], src + k * 4, 4); break;
        case DT_BF16: std::memcpy(&h, src + k * 2, 2); o[i] = bf16_to_f32(h); break;
        case DT_F16: std::memcpy(&h, src + k * 2, 2); o[i] = f16_to_f32(h); break;
        default: throw std::runtime_error("k31: unsupported vector dtype for " + p.tensor());
        }
    }
}

// sum over 8-byte words of mix(word ^ word index); bytes [0, n) at arena offset off (off % 8 == 0); the last partial
// word zero-padded
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
inline uint64_t hash_words(const uint8_t *p, uint64_t n, uint64_t off) {
    uint64_t h = 0, w0 = off / 8;
    const uint64_t full = n / 8;
    for (uint64_t i = 0; i < full; ++i) { uint64_t w; std::memcpy(&w, p + i * 8, 8); h += mix64(w ^ (w0 + i) * 0x9e3779b97f4a7c15ull); }
    if (n % 8) { uint64_t w = 0; std::memcpy(&w, p + full * 8, size_t(n % 8)); h += mix64(w ^ (w0 + full) * 0x9e3779b97f4a7c15ull); }
    return h;
}

}  // namespace k31ld
