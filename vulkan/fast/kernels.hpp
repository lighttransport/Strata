#pragma once
// Record-time helpers for the fast engine's custom kernels.
#include "fast/device.hpp"
#include "strata/core/model.hpp"
namespace strata::vulkan::fast {

// GGML type ids used by the GLM REAP50 weights.
enum : int { F32 = 0, Q5_0 = 6, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q6_K = 14 };

size_t row_bytes(int type, uint32_t k);
uint32_t block_elements(int type);

// Repacked quantized layout (see gemv_rows.comp): per head, 16-byte chunks
// ordered [block][chunk][row] followed by a side plane of words ordered
// [block][word][row]; rows are padded to a multiple of 64. F32 stays row-major.
struct Layout {
  int type = -1;
  uint32_t k = 0, m = 0, mp = 0, chunks = 0, side_words = 0;
  size_t chunk_bytes() const { return size_t(k / block_elements(type)) * chunks * mp * 16; }
  size_t side_bytes() const { return size_t(k / block_elements(type)) * side_words * mp * 4; }
  size_t head_bytes() const;  // padded to 256
};
Layout layout_for(int type, uint32_t k, uint32_t m);
// Repacks rows [0,m) of GGUF-ordered blocks into `out` (head_bytes() bytes).
void repack(const Layout &, const uint8_t *src, uint8_t *out);
// Same bytes, with rows tiled for repeated streaming during large prefill.
// The original repack traversal remains the default for existing callers.
void repack_streamed(const Layout &, const uint8_t *src, uint8_t *out);
// Same output as concatenating first_rows from first and the remaining rows
// from second, without creating a temporary joined GGUF tensor.
void repack_pair_streamed(const Layout &, const uint8_t *first, uint32_t first_rows,
                          const uint8_t *second, uint8_t *out);
// GPU byte conversion for native Q2_K/Q3_K rows. The source binding includes
// padding through the last four-byte word; source_offset may be unaligned.
void repack_gpu(Recorder &, const Layout &, View source, View output, uint32_t source_offset = 0);

// A resident weight: one or more GGUF tensors of the same type and K stacked
// along rows, optionally with heads (3-D tensors).
struct Matrix {
  std::unique_ptr<Buffer> data;
  Layout layout;
  uint32_t heads = 1;
  uint32_t k() const { return layout.k; }
  uint32_t m() const { return layout.m; }
  size_t bytes() const { return layout.head_bytes() * heads; }
};
Matrix upload_matrix(Device &, const core::ArtifactTensor &);
Matrix upload_stacked(Device &, std::span<const core::ArtifactTensor *const>);
Matrix upload_matrix(Device &, int type, uint32_t k, uint32_t m, uint32_t heads, const void *data);

// y[n*ys + head*hy + row] (+)= sum_k W[head,row,k] * x[n*xs + head*hx + k], n < columns.
// Strides and offsets are in floats; x offsets/strides must be multiples of four.
struct GemvArgs {
  uint32_t columns = 1, xs = 0, ys = 0, hx = 0, hy = 0, xofs = 0, yofs = 0;
  bool accumulate = false;
  uint32_t ksw = 0, kparts = 0;  // 0 = automatic; explicit values are for benchmarking
  View scratch;  // partial sums for a workgroup K-split; without it no split is used
};
// Floats of scratch a K-split GEMV of this matrix may use for `columns`.
size_t gemv_scratch_floats(const Matrix &, uint32_t columns);
void gemv(Recorder &, const Matrix &, View x, View y, const GemvArgs & = {});
// Tiled matrix product for larger prefill batches, with the same strides.
void gemm(Recorder &, const Matrix &, View x, View y, const GemvArgs &, uint32_t tile_columns = 64);
// Keep dense activation dispatches and bound views within a 1024-token slice.
// The original shader and single dispatch are retained at smaller widths.
void swiglu(Recorder &, View gate_up, View hidden, uint32_t units, uint32_t tokens, float limit);

} // namespace strata::vulkan::fast
