#include "strata/llvq/unit_file.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

namespace fs = std::filesystem;
using strata::llvq::UnitSet;

namespace {
constexpr uint32_t H = 32, F = 16, experts = 19;
constexpr uint64_t seed = 123;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void patch_u32(const fs::path& path, size_t offset, uint32_t value) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.seekp(offset);
    file.write(reinterpret_cast<const char*>(&value), sizeof value);
}

void write_units(const fs::path& dir, uint32_t group, bool bad_up = false) {
    fs::create_directories(dir);
    const size_t gu = strata::llvq::unit_record_bytes(F, H);
    const size_t dn = strata::llvq::unit_record_bytes(H, F);
    for (uint32_t first = 0; first < experts; first += group) {
        const uint32_t count = std::min(group, experts - first);
        std::vector<uint8_t> bytes(40 + count * (2 * gu + dn));
        std::memcpy(bytes.data(), "LLVQTET1", 8);
        const uint32_t header[] = {1, 0, first, count, 40, 0};
        std::memcpy(bytes.data() + 8, header, sizeof header);
        std::memcpy(bytes.data() + 32, &seed, sizeof seed);
        for (uint32_t e = 0; e < count; ++e) {
            auto* record = bytes.data() + 40 + e * (2 * gu + dn);
            for (int projection = 0; projection < 3; ++projection) {
                uint32_t shape[] = {projection == 2 ? H : F, projection == 2 ? F : H};
                if (bad_up && projection == 1) shape[0] = shape[1] = std::numeric_limits<uint32_t>::max();
                std::memcpy(record, shape, sizeof shape);
                const float marker = float(3 * (first + e) + projection);
                std::memcpy(record + 24, &marker, sizeof marker);
                record += projection == 2 ? dn : gu;
            }
        }
        char name[32];
        std::snprintf(name, sizeof name, "L00.e%03u.llvq", first);
        std::ofstream out(dir / name, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
}
}

int main() {
    char path[] = "/tmp/strata-llvq-unit-test-XXXXXX";
    if (!::mkdtemp(path)) return 1;
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{path};
    try {
        // Cross unit boundaries and a short final unit, including the historical group of 16.
        for (uint32_t group : {1u, 8u, 16u, 32u}) {
            const auto dir = cleanup.path / std::to_string(group);
            write_units(dir, group);
            UnitSet units;
            units.open(dir, 0, 0, experts, H, F);
            require(units.seed() == seed, "rotation seed");
            for (uint32_t e = 0; e < experts; ++e) {
                const auto* record = units.expert(0, e);
                for (int projection = 0; projection < 3; ++projection) {
                    float marker;
                    std::memcpy(&marker, record + 24, sizeof marker);
                    require(marker == float(3 * e + projection), "wrong expert or projection");
                    record += projection == 2 ? units.down_bytes() : units.gate_bytes();
                }
            }
        }
        const auto bad = cleanup.path / "bad-up";
        write_units(bad, 8, true);
        bool rejected = false;
        try {
            UnitSet units;
            units.open(bad, 0, 0, experts, H, F);
        } catch (const std::runtime_error& e) {
            rejected = std::string(e.what()).find("up record shape") != std::string::npos;
        }
        require(rejected, "malformed up record accepted");

        // A failed load must not expose a partially populated expert table.
        const auto partial = cleanup.path / "partial";
        write_units(partial, 8);
        fs::remove(partial / "L00.e008.llvq");
        UnitSet retry;
        rejected = false;
        try { retry.open(partial, 0, 0, experts, H, F); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected && !retry.has(0) && retry.mapped_bytes() == 0,
                "failed open exposes partial state");
        const auto valid = cleanup.path / "8";
        retry.open(valid, 0, 0, experts, H, F);
        const auto* saved = retry.expert(0, 18);
        const size_t mapped = retry.mapped_bytes();
        try { retry.open(partial, 0, 0, experts, H, F); }
        catch (const std::runtime_error&) {}
        require(retry.expert(0, 18) == saved && retry.mapped_bytes() == mapped,
                "failed reopen damages existing units");
        retry.open(valid, 0, 0, experts, H, F);
        require(retry.mapped_bytes() == mapped, "reopen retains previous mappings");
        require(!retry.has(-1) && !retry.has(1), "layer bounds");
        for (int e : {-1, int(experts)}) {
            rejected = false;
            try { (void)retry.expert(0, e); }
            catch (const std::out_of_range&) { rejected = true; }
            require(rejected, "expert bounds");
        }
        for (int kind = 0; kind < 6; ++kind) {
            rejected = false;
            try {
                UnitSet units;
                units.open(valid, kind == 0 ? -1 : 0, kind == 1 ? -1 : 0,
                           kind == 2 ? 0 : kind == 3 ? -1 : int(experts),
                           kind == 4 ? 0 : H,
                           kind == 5 ? uint64_t(UINT32_MAX) + 1 : F);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid model geometry not rejected before loading");
        }
        // Corrupt each independently validated field, including an overflowing expert count.
        const size_t gu = strata::llvq::unit_record_bytes(F, H);
        struct Corruption { size_t offset; uint32_t value; const char* error; };
        const Corruption cases[] = {
            {0, 0, "header mismatch"}, {8, 2, "header mismatch"},
            {12, 1, "header mismatch"}, {16, 1, "header mismatch"},
            {20, 0, "header mismatch"}, {20, UINT32_MAX, "header mismatch"},
            {28, 1, "f32-encoded"}, {40, H, "gate record shape"},
            {40 + gu + 4, F, "up record shape"}, {40 + 2 * gu, F, "down record shape"}
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            const auto dir = cleanup.path / ("corrupt-" + std::to_string(i));
            write_units(dir, 8);
            patch_u32(dir / "L00.e000.llvq", cases[i].offset, cases[i].value);
            rejected = false;
            try { retry.open(dir, 0, 0, experts, H, F); }
            catch (const std::runtime_error& e) {
                rejected = std::string(e.what()).find(cases[i].error) != std::string::npos;
            }
            require(rejected && retry.mapped_bytes() == mapped && retry.expert(0, 18),
                    "corrupt unit accepted or existing set damaged");
        }
        for (int kind = 0; kind < 4; ++kind) {
            const auto dir = cleanup.path / ("size-seed-" + std::to_string(kind));
            write_units(dir, 8);
            const auto file = dir / "L00.e008.llvq";
            if (kind == 0) patch_u32(file, 32, uint32_t(seed + 1));
            else fs::resize_file(file, kind == 1 ? 39 : kind == 2 ? fs::file_size(file) - 1 : fs::file_size(file) + 1);
            rejected = false;
            try { retry.open(dir, 0, 0, experts, H, F); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected && retry.mapped_bytes() == mapped, "invalid size or seed accepted");
        }
        for (bool multiply : {false, true}) {
            rejected = false;
            try {
                if (multiply) (void)strata::llvq::unit_size_mul(std::numeric_limits<size_t>::max(), 2);
                else (void)strata::llvq::unit_size_add(std::numeric_limits<size_t>::max(), 1);
            } catch (const std::overflow_error&) { rejected = true; }
            require(rejected, "size arithmetic overflow accepted");
        }
        const auto highest = cleanup.path / "highest-layer";
        write_units(highest, 8);
        for (int e = 0; e < int(experts); e += 8) {
            char old_name[32], new_name[32];
            std::snprintf(old_name, sizeof old_name, "L00.e%03d.llvq", e);
            std::snprintf(new_name, sizeof new_name, "L%d.e%03d.llvq", std::numeric_limits<int>::max(), e);
            patch_u32(highest / old_name, 12, uint32_t(std::numeric_limits<int>::max()));
            fs::rename(highest / old_name, highest / new_name);
        }
        retry.open(highest, std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), experts, H, F);
        require(retry.has(std::numeric_limits<int>::max()) && !retry.has(0) &&
                retry.expert(std::numeric_limits<int>::max(), 18), "highest layer overflow");
        std::cout << "llvq_unit_file_test: ok\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
