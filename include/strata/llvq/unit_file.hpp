// include/strata/llvq/unit_file.hpp - the routed experts of strata-glm-llvq-encode (docs/GLM_LLVQ.md) mapped in place.
//
// A unit file Lxx.eyyy.llvq holds `count` experts of layer xx from expert yyy: a 40-byte header ("LLVQTET1", u32
// version 1, layer, first expert, count, Lloyd iterations, precision (0 f64, 1 f32), u64 rotation seed at byte 32),
// then per expert the gate, up and down records back to back. A record is u32 rows, u32 cols, f64 centroids[2],
// f32 row scales[rows], the word rows (6 bytes per 24 columns, row stride padded to 8 bytes), a BF16 tail
// [rows][cols % 24]. So one expert is one contiguous [gate | up | down] slice of a file.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::llvq {

inline size_t unit_size_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("llvq experts: size overflow");
    return a + b;
}

inline size_t unit_size_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error("llvq experts: size overflow");
    return a * b;
}

inline size_t unit_record_bytes(uint64_t rows, uint64_t cols) {
    if (!rows || !cols || rows > UINT32_MAX || cols > UINT32_MAX)
        throw std::invalid_argument("llvq experts: invalid record geometry");
    const uint64_t nb = cols / 24, tail = cols - 24 * nb, stride = (nb * 6 + 7) / 8 * 8;
    return unit_size_add(24, unit_size_mul(size_t(rows), size_t(4 + stride + 2 * tail)));
}

class UnitSet {
public:
    UnitSet() = default;
    UnitSet(const UnitSet&) = delete;
    UnitSet& operator=(const UnitSet&) = delete;
    ~UnitSet() {
        for (const auto& m : maps_) ::munmap(m.first, m.second);
    }

    /// Maps the units of layers [first, last] with `experts` experts each (n_embd x n_ff experts). Every unit must be
    /// present, f64, and share one seed; throws otherwise.
    void open(const std::filesystem::path& dir, int first, int last, int experts, uint64_t n_embd, uint64_t n_ff) {
        // Publish only a complete set. Failure releases temporary maps and preserves any old set.
        UnitSet next;
        next.load(dir, first, last, experts, n_embd, n_ff);
        using std::swap;
        swap(first_, next.first_); swap(experts_, next.experts_);
        swap(gu_, next.gu_); swap(dn_, next.dn_); swap(per_, next.per_);
        swap(bytes_, next.bytes_); swap(seed_, next.seed_);
        table_.swap(next.table_); maps_.swap(next.maps_);
    }

private:
    void load(const std::filesystem::path& dir, int first, int last, int experts, uint64_t n_embd, uint64_t n_ff) {
        if (first < 0 || last < first || experts <= 0)
            throw std::invalid_argument("llvq experts: invalid layer or expert range");
        first_ = first;
        experts_ = experts;
        gu_ = unit_record_bytes(n_ff, n_embd);
        dn_ = unit_record_bytes(n_embd, n_ff);
        per_ = unit_size_add(unit_size_mul(2, gu_), dn_);
        table_.assign(unit_size_mul(size_t(int64_t(last) - first + 1), size_t(experts)), nullptr);
        bool have_seed = false;
        for (int64_t l = first; l <= last; ++l)
            for (int e0 = 0; e0 < experts;) {
                char name[32];
                std::snprintf(name, sizeof name, "L%02d.e%03d.llvq", int(l), e0);
                const auto path = dir / name;
                const int fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("llvq experts: missing unit " + path.string());
                struct stat st{};
                if (::fstat(fd, &st) != 0 || st.st_size < 40 || uintmax_t(st.st_size) > std::numeric_limits<size_t>::max()) {
                    ::close(fd);
                    throw std::runtime_error("llvq experts: unreadable unit " + path.string());
                }
                void* p = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
                ::close(fd);
                if (p == MAP_FAILED) throw std::runtime_error("llvq experts: cannot map " + path.string());
                try { maps_.push_back({p, size_t(st.st_size)}); }
                catch (...) { ::munmap(p, size_t(st.st_size)); throw; }
                const auto* b = static_cast<const uint8_t*>(p);
                uint32_t h[6];
                uint64_t seed;
                std::memcpy(h, b + 8, 24);
                std::memcpy(&seed, b + 32, 8);
                if (std::memcmp(b, "LLVQTET1", 8) || h[0] != 1 || h[1] != uint32_t(l) || h[2] != uint32_t(e0) || h[3] == 0 ||
                    h[3] > uint32_t(experts - e0))
                    throw std::runtime_error("llvq experts: unit header mismatch in " + path.string());
                if (h[5] != 0) throw std::runtime_error("llvq experts: f32-encoded unit (re-encode in f64) " + path.string());
                if (size_t(st.st_size) != unit_size_add(40, unit_size_mul(size_t(h[3]), per_)))
                    throw std::runtime_error("llvq experts: unit size does not match the model's expert geometry: " + path.string());
                if (have_seed && seed != seed_) throw std::runtime_error("llvq experts: units with different rotation seeds");
                seed_ = seed;
                have_seed = true;
                for (uint32_t k = 0; k < h[3]; ++k) {
                    const uint8_t* x = b + 40 + size_t(k) * per_;
                    uint32_t rc[2];
                    std::memcpy(rc, x, 8);
                    if (rc[0] != n_ff || rc[1] != n_embd) throw std::runtime_error("llvq experts: gate record shape in " + path.string());
                    std::memcpy(rc, x + gu_, 8);
                    if (rc[0] != n_ff || rc[1] != n_embd) throw std::runtime_error("llvq experts: up record shape in " + path.string());
                    std::memcpy(rc, x + 2 * gu_, 8);
                    if (rc[0] != n_embd || rc[1] != n_ff) throw std::runtime_error("llvq experts: down record shape in " + path.string());
                    table_[size_t(l - first) * size_t(experts) + e0 + k] = x;
                }
                bytes_ = unit_size_add(bytes_, size_t(st.st_size));
                e0 += int(h[3]);
            }
    }
public:
    bool has(int layer) const { return !table_.empty() && layer >= first_ && uint64_t(int64_t(layer) - first_) < table_.size() / size_t(experts_); }
    /// The expert's [gate | up | down] records.
    const uint8_t* expert(int layer, int e) const {
        if (!has(layer) || e < 0 || e >= experts_) throw std::out_of_range("llvq experts: no such expert");
        return table_[size_t(layer - first_) * size_t(experts_) + size_t(e)];
    }
    uint64_t seed() const { return seed_; }
    size_t gate_bytes() const { return gu_; }
    size_t down_bytes() const { return dn_; }
    size_t expert_bytes() const { return per_; }
    size_t mapped_bytes() const { return bytes_; }

private:
    int first_ = 0, experts_ = 0;
    size_t gu_ = 0, dn_ = 0, per_ = 0, bytes_ = 0;
    uint64_t seed_ = 0;
    std::vector<const uint8_t*> table_;
    std::vector<std::pair<void*, size_t>> maps_;
};

}  // namespace strata::llvq
