// include/strata/net/tp_storage.hpp - a tensor-parallel worker's expert rows: anonymous memory or a kept --weights-dir
// file (strata-glm-tp-worker, strata-ds41-worker; include/strata/net/expert_tp.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::net {

inline constexpr double kGiB = double(1ull << 30);

inline size_t mem_available() {
    std::ifstream info("/proc/meminfo");
    std::string key;
    size_t kib = 0;
    while (info >> key >> kib) {
        if (key == "MemAvailable:") return kib * 1024;
        info.ignore(256, '\n');
    }
    return (size_t)sysconf(_SC_AVPHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
}

// The expert rows: anonymous memory, or a shared mapping of a --weights-dir file (kept for the next worker).
struct Storage {
    uint8_t *data = nullptr;
    size_t bytes = 0;
    std::string path; // the complete file; rows are received into path + ".partial" and renamed when complete
    ~Storage() {
        if (data) munmap(data, bytes);
    }
    void map_anonymous() {
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot allocate the expert rows");
        data = (uint8_t *)p;
    }
    void map_file(const std::string &file, bool create) {
        const int fd = open(file.c_str(), create ? O_RDWR | O_CREAT | O_TRUNC : O_RDWR, 0600);
        if (fd < 0) throw std::runtime_error("cannot open " + file);
        if (create && ftruncate(fd, (off_t)bytes) != 0) {
            close(fd);
            throw std::runtime_error("cannot size " + file);
        }
        // Kept rows are mapped populated: faulting 4 KiB pages in on first use from every pool thread at once
        // stalled the first decode for tens of seconds.
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | (create ? 0 : MAP_POPULATE), fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + file);
        data = (uint8_t *)p;
    }
    void complete() { // the received rows become reusable
        if (path.empty()) return;
        if (msync(data, bytes, MS_SYNC) != 0 || std::rename((path + ".partial").c_str(), path.c_str()) != 0)
            throw std::runtime_error("cannot finish " + path);
    }
};

inline uint64_t fnv(const void *p, size_t n, uint64_t h = 14695981039346656037ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t *)p)[i]) * 1099511628211ull;
    return h;
}

} // namespace strata::net
