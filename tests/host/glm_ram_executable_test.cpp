#include "strata/core/ram_executable.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#ifdef __linux__
static uint64_t fingerprint(const char *path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    uint64_t value = 14695981039346656037ULL;
    for (char c; input.get(c);) value = (value ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
    assert(input.eof());
    return value;
}
#endif

int main(int argc, char **argv) {
#ifdef __linux__
    const bool off = argc == 2 && std::string(argv[1]) == "--off";
    if (!std::getenv("STRATA_RAM_EXEC_TEST_PID")) {
        const std::string pid = std::to_string(getpid());
        assert(setenv("STRATA_RAM_EXEC_TEST_PID", pid.c_str(), 1) == 0);
    }
    strata::core::glm_ram_executable(argv);
    assert(std::stoi(std::getenv("STRATA_RAM_EXEC_TEST_PID")) == getpid());
    const int self = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    assert(self >= 0);
    const int seals = fcntl(self, F_GET_SEALS);
    if (off) {
        struct stat original{}, current{};
        assert(stat(argv[0], &original) == 0 && fstat(self, &current) == 0);
        assert(original.st_dev == current.st_dev && original.st_ino == current.st_ino);
    } else assert(seals >= 0 && (seals & (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) ==
                (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL));
    close(self);
    assert(fingerprint(argv[0]) == fingerprint("/proc/self/exe"));
    std::cout << "glm_ram_executable_test: passed pid=" << getpid() << " off=" << off << '\n';
    return 0;
#else
    (void)argc; (void)argv;
    return 77;
#endif
}
