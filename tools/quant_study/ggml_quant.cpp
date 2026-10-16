// Quantize one FP32 matrix (rows x cols, row-major) to a ggml type and back, for the quantization study
// (docs/GLM_QUANT_LEECH_GSQ_STUDY.md). Optional importance weights (one per column, e.g. E[x^2] of the inputs)
// are passed to ggml_quantize_chunk as llama.cpp's imatrix. Prints the bits per weight of the encoded tensor.
//
//   g++ -O2 -std=c++17 tools/quant_study/ggml_quant.cpp -I$LLAMA/ggml/include -L$LLAMA/build/bin -lggml-base \
//       -Wl,-rpath,$LLAMA/build/bin -o ggml_quant
//   ggml_quant TYPE ROWS COLS in.f32 out.f32 [importance.f32]
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read(const char* path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * 4)) { std::fprintf(stderr, "cannot read %zu floats from %s\n", n, path); std::exit(2); }
    return v;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) { std::fprintf(stderr, "usage: ggml_quant TYPE ROWS COLS in.f32 out.f32 [importance.f32]\n"); return 2; }
    ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const auto* tr = ggml_get_type_traits((ggml_type)t);
        if (tr && tr->type_name && std::strcmp(tr->type_name, argv[1]) == 0) type = (ggml_type)t;
    }
    // ggml spells types in lower case (q2_K); accept either spelling.
    if (type == GGML_TYPE_COUNT)
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const auto* tr = ggml_get_type_traits((ggml_type)t);
            if (tr && tr->type_name && strcasecmp(tr->type_name, argv[1]) == 0) type = (ggml_type)t;
        }
    if (type == GGML_TYPE_COUNT) { std::fprintf(stderr, "unknown type %s\n", argv[1]); return 2; }
    const int64_t rows = std::atoll(argv[2]), cols = std::atoll(argv[3]);
    auto w = read(argv[4], size_t(rows * cols));
    std::vector<float> imat;
    if (argc == 7) imat = read(argv[6], size_t(cols));
    if (ggml_quantize_requires_imatrix(type) && imat.empty()) { std::fprintf(stderr, "%s requires importance weights\n", argv[1]); return 3; }
    const size_t row_bytes = ggml_row_size(type, cols);
    std::vector<uint8_t> q(row_bytes * rows);
    ggml_quantize_init(type);
    const size_t bytes = ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, cols, imat.empty() ? nullptr : imat.data());
    std::vector<float> back(size_t(rows * cols));
    const auto* tr = ggml_get_type_traits(type);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(q.data() + r * row_bytes, back.data() + r * cols, cols);
    std::ofstream o(argv[5], std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(back.data()), back.size() * 4);
    if (!o) { std::fprintf(stderr, "write failed\n"); return 4; }
    std::printf("{\"type\":\"%s\",\"bytes\":%zu,\"bpw\":%.4f}\n", tr->type_name, bytes, bytes * 8.0 / double(rows * cols));
    return 0;
}
