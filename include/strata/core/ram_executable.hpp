#pragma once
#include "strata/host/glm_env.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
extern char **environ;
#endif

namespace strata::core {
// Call at main entry, before engine threads or model loading. Exec preserves
// the PID, arguments and transport descriptors. The only copied data is this
// executable, never model weights. A sealed memfd cannot be invalidated by a
// shared filesystem's executable-page cache coherency operations.
inline void glm_ram_executable(char *const argv[]) {
    if (!strata_env_is_one("STRATA_GLM_RAM_EXECUTABLE")) return;
#ifdef __linux__
    struct Descriptor {
        int fd;
        explicit Descriptor(int value) : fd(value) {}
        ~Descriptor() { if (fd >= 0) close(fd); }
        Descriptor(const Descriptor&) = delete;
        Descriptor& operator=(const Descriptor&) = delete;
    };
    const auto fail = [](const char *operation) {
        throw std::runtime_error(std::string("RAM executable: ") + operation + ": " + std::strerror(errno));
    };
    Descriptor source(open("/proc/self/exe", O_RDONLY | O_CLOEXEC));
    if (source.fd < 0) fail("open self");
    struct stat info{};
    if (fstat(source.fd, &info)) fail("stat self");
    constexpr size_t maximum = 128 * 1024 * 1024;
    if (!S_ISREG(info.st_mode) || info.st_size < 1 || uint64_t(info.st_size) > maximum)
        throw std::runtime_error("RAM executable: image must be a regular file of at most 128 MiB");
    constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    std::array<char,4096> link{};
    const ssize_t length = readlink("/proc/self/exe", link.data(), link.size());
    if (length < 0) fail("read self link");
    if (length == ssize_t(link.size())) throw std::runtime_error("RAM executable: self link too long");
    const std::string image(link.data(), size_t(length));
    if (image.starts_with("/memfd:strata-glm-executable")) {
        const int actual = fcntl(source.fd, F_GET_SEALS);
        if (actual < 0 || (actual & seals) != seals)
            throw std::runtime_error("RAM executable: image is not sealed");
        std::cerr << "RAM_EXECUTABLE active=1 bytes=" << info.st_size << " sealed=1\n";
        return;
    }
    Descriptor target{int(syscall(SYS_memfd_create, "strata-glm-executable", MFD_CLOEXEC | MFD_ALLOW_SEALING))};
    if (target.fd < 0) fail("create memfd");
    std::array<char,65536> buffer{};
    size_t copied = 0;
    while (copied < size_t(info.st_size)) {
        const size_t wanted = std::min(buffer.size(), size_t(info.st_size) - copied);
        const ssize_t count = read(source.fd, buffer.data(), wanted);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) fail("read image");
        if (!count) throw std::runtime_error("RAM executable: truncated self image");
        size_t written = 0;
        while (written < size_t(count)) {
            const ssize_t n = write(target.fd, buffer.data() + written, size_t(count) - written);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) fail("write image");
            if (!n) throw std::runtime_error("RAM executable: incomplete image write");
            written += size_t(n);
        }
        copied += size_t(count);
    }
    if (fchmod(target.fd, S_IRUSR | S_IXUSR)) fail("set image mode");
    if (fcntl(target.fd, F_ADD_SEALS, seals)) fail("seal image");
    fexecve(target.fd, argv, ::environ);
    fail("exec sealed image");
#else
    (void)argv;
    throw std::invalid_argument("RAM executable requires Linux");
#endif
}
} // namespace strata::core
