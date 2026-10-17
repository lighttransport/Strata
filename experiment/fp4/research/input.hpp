#pragma once
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace research {
// Raw little-endian, row-major IEEE float32. No model-format assumptions.
inline std::vector<float> read_f32(const std::string& path,size_t count=0) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    if constexpr(std::endian::native!=std::endian::little)
        throw std::runtime_error("raw input requires a little-endian host");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("cannot open float32 input");
    auto size=file.tellg();
    if(size<=0 || uint64_t(size)%4 || uint64_t(size)>512ull*1024*1024)
        throw std::runtime_error("float32 input must be nonempty, <=512 MiB, and contain complete values");
    size_t actual=size_t(size)/4;
    if(count && actual!=count)throw std::runtime_error("float32 input size does not match geometry");
    std::vector<float> values(actual);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(values.data()),size))throw std::runtime_error("short float32 read");
    for(float v:values)if(!std::isfinite(v))throw std::runtime_error("nonfinite float32 input");
    return values;
}
} // namespace research
