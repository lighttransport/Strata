#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace strata::ds4 {
struct EgRead;
// Calls are serialized by EgReader's single remote worker.
class EngramRemote {
public:
    EngramRemote();
    ~EngramRemote();
    bool connect(const std::string& address, uint64_t size, const std::string& sha256, std::string& error);
    bool read(EgRead* rows, size_t count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
