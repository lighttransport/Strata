#include "fast/kernels.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace strata::vulkan::fast {
namespace {
void require(bool v, const char *message) {
  if (!v) throw std::invalid_argument(message);
}
} // namespace

uint32_t block_elements(int type) {
  switch (type) {
  case F32: return 4;
  case Q5_0: return 32;
  case Q2_K: case Q3_K: case Q4_K: case Q6_K: return 256;
  default: throw std::invalid_argument("Vulkan fast block type unsupported");
  }
}
size_t row_bytes(int type, uint32_t k) {
  require(k % block_elements(type) == 0, "Vulkan fast row width is not a whole number of blocks");
  switch (type) {
  case F32: return size_t(k) * 4;
  case Q5_0: return size_t(k / 32) * 22;
  case Q2_K: return size_t(k / 256) * 84;
  case Q3_K: return size_t(k / 256) * 110;
  case Q4_K: return size_t(k / 256) * 144;
  case Q6_K: return size_t(k / 256) * 210;
  default: throw std::invalid_argument("Vulkan fast row type unsupported");
  }
}
size_t Layout::head_bytes() const {
  const size_t raw = type == F32 ? size_t(k) * m * 4 : chunk_bytes() + side_bytes();
  return (raw + 255) & ~size_t(255);
}
Layout layout_for(int type, uint32_t k, uint32_t m) {
  Layout l;
  l.type = type;
  l.k = k;
  l.m = m;
  l.mp = (m + 63) / 64 * 64;
  require(k > 0 && m > 0 && k % block_elements(type) == 0, "Vulkan fast layout geometry invalid");
  switch (type) {
  case F32: l.mp = m; break;
  case Q4_K: l.chunks = 9; break;
  case Q6_K: l.chunks = 13; l.side_words = 1; break;
  case Q5_0: l.chunks = 1; l.side_words = 2; break;
  case Q2_K: l.chunks = 5; l.side_words = 1; break;
  case Q3_K: l.chunks = 6; l.side_words = 4; break;
  default: throw std::invalid_argument("Vulkan fast layout type unsupported");
  }
  return l;
}
static void repack_impl(const Layout &l, const uint8_t *src, uint8_t *out, bool tiled,
                         const uint8_t *second = nullptr, uint32_t first_rows = 0) {
  std::memset(out, 0, l.head_bytes());
  const size_t rb = row_bytes(l.type, l.k);
  if (l.type == F32) {
    if (second) {
      std::memcpy(out, src, rb * first_rows);
      std::memcpy(out + rb * first_rows, second, rb * (l.m - first_rows));
    } else std::memcpy(out, src, rb * l.m);
    return;
  }
  const uint32_t nb = l.k / block_elements(l.type);
  const size_t bb = rb / nb;
  uint8_t *side = out + l.chunk_bytes();
  auto put_side = [&](uint32_t b, uint32_t w, uint32_t row, const uint8_t *bytes, size_t n) {
    uint32_t v = 0;
    std::memcpy(&v, bytes, n);
    std::memcpy(side + ((size_t(b) * l.side_words + w) * l.mp + row) * 4, &v, 4);
  };
  auto copy_block = [&](uint32_t row, uint32_t b, const uint8_t *s) {
      auto chunk = [&](uint32_t c, const uint8_t *bytes) {
        std::memcpy(out + ((size_t(b) * l.chunks + c) * l.mp + row) * 16, bytes, 16);
      };
      switch (l.type) {
      case Q4_K:
        for (uint32_t c = 0; c < 9; ++c) chunk(c, s + 16 * c);
        break;
      case Q6_K:
        for (uint32_t c = 0; c < 13; ++c) chunk(c, s + 16 * c);
        put_side(b, 0, row, s + 208, 2);
        break;
      case Q5_0:
        chunk(0, s + 6);
        put_side(b, 0, row, s + 2, 4);
        put_side(b, 1, row, s, 2);
        break;
      case Q2_K:
        for (uint32_t c = 0; c < 5; ++c) chunk(c, s + 16 * c);
        put_side(b, 0, row, s + 80, 4);
        break;
      case Q3_K:
        for (uint32_t c = 0; c < 6; ++c) chunk(c, s + 16 * c);
        for (uint32_t w = 0; w < 3; ++w) put_side(b, w, row, s + 96 + 4 * w, 4);
        put_side(b, 3, row, s + 108, 2);
        break;
      }
  };
  auto copy_rows = [&](const uint8_t *data, uint32_t begin, uint32_t count) {
    if (tiled) {
      // Fill neighboring rows in each output plane before visiting the next
      // superblock. Keep the two source tensors in separate row ranges so
      // their address selection does not branch once per copied block.
      for (uint32_t first = begin; first < begin + count; first += 64)
        for (uint32_t b = 0; b < nb; ++b)
          for (uint32_t row = first; row < std::min(first + 64, begin + count); ++row)
            copy_block(row, b, data + (row - begin) * rb + b * bb);
    } else {
      for (uint32_t row = begin; row < begin + count; ++row)
        for (uint32_t b = 0; b < nb; ++b) copy_block(row, b, data + (row - begin) * rb + b * bb);
    }
  };
  if (second) { copy_rows(src, 0, first_rows); copy_rows(second, first_rows, l.m - first_rows); }
  else copy_rows(src, 0, l.m);
}
void repack(const Layout &l, const uint8_t *src, uint8_t *out) { repack_impl(l, src, out, false); }
void repack_streamed(const Layout &l, const uint8_t *src, uint8_t *out) { repack_impl(l, src, out, true); }
void repack_pair_streamed(const Layout &l, const uint8_t *first, uint32_t first_rows,
                          const uint8_t *second, uint8_t *out) {
  require(first && second && first_rows > 0 && first_rows < l.m, "Vulkan paired repack split invalid");
  repack_impl(l, first, out, true, second, first_rows);
}

void repack_gpu(Recorder &r, const Layout &l, View source, View output, uint32_t source_offset) {
  require(l.type == Q2_K || l.type == Q3_K, "Vulkan GPU repack requires Q2_K/Q3_K rows");
  const auto expected = layout_for(l.type, l.k, l.m);
  require(l.mp == expected.mp && l.chunks == expected.chunks && l.side_words == expected.side_words,
          "Vulkan GPU repack layout invalid");
  auto available = [](View v) {
    require(v.buffer && v.offset <= v.buffer->size, "Vulkan GPU repack binding invalid");
    const size_t remaining = v.buffer->size - v.offset;
    require(!v.range || v.range <= remaining, "Vulkan GPU repack range invalid");
    return v.range ? v.range : remaining;
  };
  const size_t input_bytes = size_t(source_offset) + row_bytes(l.type, l.k) * l.m;
  require((input_bytes + 3) / 4 * 4 <= available(source) && l.head_bytes() <= available(output),
          "Vulkan GPU repack buffer too small");
  require(input_bytes <= UINT32_MAX && l.head_bytes() <= UINT32_MAX,
          "Vulkan GPU repack exceeds shader address range");
  r.dispatch(l.type == Q2_K ? "repack_expert_q2k" : "repack_expert_q3k", {source, output},
             std::array<uint32_t, 4>{l.k / 256, l.m, l.mp, source_offset}, l.mp / 64, l.k / 256);
}

Matrix upload_matrix(Device &d, int type, uint32_t k, uint32_t m, uint32_t heads, const void *data) {
  require(heads > 0, "Vulkan fast matrix heads");
  Matrix out;
  out.layout = layout_for(type, k, m);
  out.heads = heads;
  const size_t hb = out.layout.head_bytes(), rb = row_bytes(type, k);
  out.data = d.buffer(out.bytes(), Memory::Device);
  std::vector<uint8_t> packed(hb);
  for (uint32_t h = 0; h < heads; ++h) {
    repack(out.layout, static_cast<const uint8_t *>(data) + size_t(h) * rb * m, packed.data());
    d.upload(*out.data, packed.data(), hb, size_t(h) * hb);
  }
  return out;
}
Matrix upload_matrix(Device &d, const core::ArtifactTensor &t) {
  const auto &s = t.tensor->shape;
  require(s.size() >= 2 && s.size() <= 3, "Vulkan fast matrix rank");
  const uint32_t heads = s.size() == 3 ? uint32_t(s[2]) : 1;
  require(row_bytes(int(t.tensor->type), uint32_t(s[0])) * s[1] * heads == t.bytes,
          "Vulkan fast matrix byte count differs from GGUF");
  return upload_matrix(d, int(t.tensor->type), uint32_t(s[0]), uint32_t(s[1]), heads, t.data());
}
Matrix upload_stacked(Device &d, std::span<const core::ArtifactTensor *const> parts) {
  require(!parts.empty(), "Vulkan fast stacked matrix needs parts");
  const int type = int(parts[0]->tensor->type);
  const uint32_t k = uint32_t(parts[0]->tensor->shape[0]);
  std::vector<uint8_t> joined;
  uint32_t m = 0;
  for (auto *t : parts) {
    require(int(t->tensor->type) == type && t->tensor->shape.size() == 2 && t->tensor->shape[0] == k &&
                row_bytes(type, k) * t->tensor->shape[1] == t->bytes,
            "Vulkan fast stacked parts differ in type, width or size");
    joined.insert(joined.end(), t->data(), t->data() + t->bytes);
    m += uint32_t(t->tensor->shape[1]);
  }
  return upload_matrix(d, type, k, m, 1, joined.data());
}

void gemv(Recorder &r, const Matrix &w, View x, View y, const GemvArgs &a, uint32_t panel_columns) {
  const auto &l = w.layout;
  require((panel_columns == 4 || panel_columns == 8 || panel_columns == 16) &&
              a.columns >= 1 && a.columns <= panel_columns && a.xofs % 4 == 0 && a.xs % 4 == 0 && a.hx % 4 == 0,
          "Vulkan fast GEMV arguments invalid");
  const std::string suffix = panel_columns == 4 ? "" : "_n" + std::to_string(panel_columns);
  const uint32_t xs = a.xs ? a.xs : l.k * w.heads, ys = a.ys ? a.ys : l.m * w.heads;
  const uint32_t hx = w.heads > 1 ? (a.hx ? a.hx : l.k) : 0, hy = w.heads > 1 ? (a.hy ? a.hy : l.m) : 0;
  if (panel_columns > 4) {
    auto available = [](View v) {
      require(v.buffer && v.offset <= v.buffer->size, "Vulkan GEMV panel view invalid");
      const size_t left = v.buffer->size - v.offset;
      require(!v.range || v.range <= left, "Vulkan GEMV panel range invalid");
      return v.range ? v.range : left;
    };
    const size_t inputs = size_t(a.xofs) + size_t(a.columns - 1) * xs + size_t(w.heads - 1) * hx + l.k;
    const size_t outputs = size_t(a.yofs) + size_t(a.columns - 1) * ys + size_t(w.heads - 1) * hy + l.m;
    require(inputs <= available(x) / 4 && outputs <= available(y) / 4, "Vulkan GEMV panel buffer too small");
    if (a.scratch.buffer) available(a.scratch);
  }
  if (l.type == F32) {
    require(w.heads == 1, "Vulkan fast F32 GEMV has no heads");
    const uint32_t rs = l.m >= 64 ? 2 : 1;
    struct {
      uint32_t K, M, N, xs, ys, RS, xofs, yofs, accumulate, pad0, pad1, pad2;
    } push{l.k, l.m, a.columns, xs, ys, rs, a.xofs, a.yofs, a.accumulate ? 1u : 0u, 0, 0, 0};
    r.dispatch("gemv_f32" + suffix, {View(*w.data), x, y}, push, (l.m + rs - 1) / rs);
    return;
  }
  const uint32_t nb = l.k / block_elements(l.type);
  // Four waves per workgroup (a 1024-thread group caps GCN at 64 VGPRs and
  // spills). Small row counts split K across workgroups instead.
  // Measured on RX570 (glm_vulkan_fast_bench): Q4_K prefers four K-split
  // waves from 16 blocks, Q6_K one or two.
  uint32_t ksw = a.ksw;
  if (!ksw) {
    if (l.type == Q6_K) ksw = nb >= 32 ? 2 : 1;
    else ksw = nb >= 16 ? 4 : (nb >= 4 ? 2 : 1);
  }
  const uint32_t groups = (l.mp + 64 * (4 / ksw) - 1) / (64 * (4 / ksw));
  uint32_t kparts = a.kparts ? a.kparts : 1;
  if (!a.kparts && a.scratch.buffer && w.heads == 1)
    while (groups * kparts * 4 < 512 && nb >= ksw * kparts * 2 && kparts < 16) kparts *= 2;
  require(ksw >= 1 && 4 % ksw == 0 && kparts >= 1 && (kparts == 1 || (a.scratch.buffer && w.heads == 1)),
          "Vulkan fast GEMV split invalid");
  struct {
    uint32_t K, M, Mp, N, xs, ys, xofs, yofs, accumulate, ksw, hb, hx, hy, side, kparts, pad1;
  } push{l.k, l.m, l.mp, a.columns, xs, ys, a.xofs, a.yofs, a.accumulate ? 1u : 0u, ksw,
         uint32_t(l.head_bytes()), hx, hy, uint32_t(l.chunk_bytes()), kparts, 0};
  const char *kernel = nullptr;
  switch (l.type) {
  case Q4_K: kernel = "gemv_rows_q4k"; break;
  case Q6_K: kernel = "gemv_rows_q6k"; break;
  case Q5_0: kernel = "gemv_rows_q50"; break;
  default: throw std::invalid_argument("Vulkan fast GEMV type unsupported");
  }
  View part = kparts > 1 ? a.scratch : y;
  if (kparts > 1)
    require(part.buffer->size - part.offset >= size_t(kparts) * l.m * a.columns * 4, "Vulkan fast GEMV scratch too small");
  if (kparts > 1 && panel_columns > 4 && part.range)
    require(part.range >= size_t(kparts) * l.m * a.columns * 4, "Vulkan GEMV panel scratch range too small");
  r.dispatch(std::string(kernel) + suffix, {View(*w.data), x, y, part}, push, groups, w.heads, kparts);
  if (kparts > 1) {
    r.barrier();  // the reduction reads the partials, also inside a concurrent group
    struct {
      uint32_t M, N, ys, yofs, parts, accumulate;
    } reduce{l.m, a.columns, ys, a.yofs, kparts, a.accumulate ? 1u : 0u};
    r.dispatch("reduce_parts", {part, y}, reduce, (l.m * a.columns + 255) / 256);
  }
}
size_t gemv_scratch_floats(const Matrix &w, uint32_t columns) { return size_t(16) * w.m() * columns; }

void gemm(Recorder &r, const Matrix &w, View x, View y, const GemvArgs &a, uint32_t tile_columns) {
  require(tile_columns == 64 || tile_columns == 128 || tile_columns == 256, "Vulkan fast GEMM tile width invalid");
  require(a.columns >= 1 && a.columns <= 4096, "Vulkan fast GEMM column count invalid");
  const uint32_t xs = a.xs ? a.xs : w.k() * w.heads, ys = a.ys ? a.ys : w.m() * w.heads;
  const uint32_t hx = w.heads > 1 ? (a.hx ? a.hx : w.k()) : 0;
  const uint32_t hy = w.heads > 1 ? (a.hy ? a.hy : w.m()) : 0;
  const size_t inputs = size_t(a.xofs) + size_t(a.columns - 1) * xs + size_t(w.heads - 1) * hx + w.k();
  const size_t outputs = size_t(a.yofs) + size_t(a.columns - 1) * ys + size_t(w.heads - 1) * hy + w.m();
  auto available = [](View v) {
    require(v.buffer && v.offset <= v.buffer->size, "Vulkan fast GEMM view offset invalid");
    const size_t remaining = v.buffer->size - v.offset;
    require(v.range == 0 || v.range <= remaining, "Vulkan fast GEMM view range invalid");
    return v.range ? v.range : remaining;
  };
  require(inputs <= available(x) / 4 && outputs <= available(y) / 4,
          "Vulkan fast GEMM activation buffer too small");
  const char *kernel = nullptr;
  switch (w.layout.type) {
  case F32: kernel = "gemm_f32"; break;
  case Q4_K: kernel = "gemm_q4k"; break;
  case Q6_K: kernel = "gemm_q6k"; break;
  case Q5_0: kernel = "gemm_q50"; break;
  case Q2_K: kernel = "gemm_q2k"; break;
  case Q3_K: kernel = "gemm_q3k"; break;
  default: throw std::invalid_argument("Vulkan fast GEMM type unsupported");
  }
  uint32_t kparts = 1;
  if (w.layout.type == F32 && w.heads == 1 && a.scratch.buffer)
    while ((w.m() + 63) / 64 * ((a.columns + tile_columns - 1) / tile_columns) * kparts < 128 &&
            kparts < 16 && w.k() >= kparts * 64) kparts *= 2;
  View part = kparts > 1 ? a.scratch : y;
  require(kparts == 1 || size_t(kparts) * w.m() * a.columns <= available(part) / 4, "Vulkan GEMM partial buffer too small");
  struct {
    uint32_t K, M, Mp, N, xs, ys, xofs, yofs, accumulate, hb, hx, hy, side, kparts;
  } push{w.k(), w.m(), w.layout.mp, a.columns, xs, ys, a.xofs, a.yofs, a.accumulate ? 1u : 0u,
         uint32_t(w.layout.head_bytes()), hx, hy, uint32_t(w.layout.chunk_bytes()), kparts};
  r.dispatch(std::string(kernel) + (tile_columns == 256 ? "_wider" : tile_columns == 128 ? "_wide" : ""), {View(*w.data), x, y, part}, push,
             (w.m() + 63) / 64, (a.columns + tile_columns - 1) / tile_columns, w.heads * kparts);
  if (kparts > 1) {
    struct {uint32_t M, N, ys, yofs, parts, accumulate;} reduce{w.m(), a.columns, ys, a.yofs, kparts, a.accumulate ? 1u : 0u};
    r.barrier();
    r.dispatch("reduce_parts", {part, y}, reduce, (w.m() * a.columns + 255) / 256);
  }
}

void swiglu(Recorder &r, View gu, View hidden, uint32_t units, uint32_t tokens, float limit) {
  require(units > 0 && units <= 12288 && units % 64 == 0 && tokens > 0 && tokens <= 2048,
          "Vulkan fast SwiGLU geometry invalid");
  auto available = [](View v) {
    require(v.buffer && v.offset <= v.buffer->size, "Vulkan fast SwiGLU binding invalid");
    const size_t remaining = v.buffer->size - v.offset;
    require(!v.range || v.range <= remaining, "Vulkan fast SwiGLU range invalid");
    return v.range ? v.range : remaining;
  };
  const size_t row_bytes = size_t(units) * 4;
  require(size_t(tokens) * 2 * row_bytes <= available(gu) && size_t(tokens) * row_bytes <= available(hidden),
          "Vulkan fast SwiGLU buffer too small");
  for (uint32_t first = 0; first < tokens; first += 1024) {
    const uint32_t count = std::min(1024u, tokens - first);
    View x = gu, y = hidden;
    if (tokens > 1024) {
      x = View(*gu.buffer, gu.offset + size_t(first) * 2 * row_bytes, size_t(count) * 2 * row_bytes);
      y = View(*hidden.buffer, hidden.offset + size_t(first) * row_bytes, size_t(count) * row_bytes);
    }
    struct {uint32_t n, tokens; float limit;} push{units, count, limit};
    r.dispatch("swiglu", {x, y}, push, (units * count + 255) / 256);
  }
}

} // namespace strata::vulkan::fast
