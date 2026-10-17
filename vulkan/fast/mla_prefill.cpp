#include "fast/mla_prefill.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strata::vulkan::fast {
void MlaPrefill::run(Recorder &r, View cache, View queries, View ids, View output,
                     uint32_t tokens, uint32_t position, float scale, uint32_t istride, bool bounded_keys) {
  if (!tokens || tokens > 8192 || istride < 2113 || istride > 4096 ||
      position > std::numeric_limits<uint32_t>::max() - tokens - 63 || !std::isfinite(scale))
    throw std::invalid_argument("batched MLA geometry invalid");
  constexpr uint32_t latent = 512, heads = 64, batch = 8, max_keys = 2112;
  auto fits = [](View view, size_t bytes) {
    if (!view.buffer || view.offset > view.buffer->size) return false;
    const auto remaining = view.buffer->size - view.offset;
    return (!view.range || view.range <= remaining) && bytes <= (view.range ? view.range : remaining);
  };
  if (!fits(queries, size_t(tokens) * heads * latent * 4) ||
      !fits(output, size_t(tokens) * heads * latent * 4) ||
      !fits(ids, size_t(tokens) * istride * 4) || !fits(cache, latent * 4))
    throw std::invalid_argument("batched MLA buffer too small");
  if (!keys_.data) {
    keys_.data = device_.buffer(size_t(batch) * max_keys * latent * 4, Memory::Device);
    scores_ = device_.buffer(size_t(batch) * max_keys * heads * 4, Memory::Device);
  }
  for (uint32_t first = 0; first < tokens; first += batch) {
    const uint32_t count = std::min(batch, tokens - first);
    const uint32_t width = bounded_keys ? std::min(max_keys, (position + first + count + 63) / 64 * 64) : max_keys;
    View q = queries, id = ids, out = output;
    auto advance = [](View &view, size_t bytes) {
      view.offset += bytes;
      if (view.range) view.range -= bytes;
    };
    advance(q, size_t(first) * heads * latent * 4);
    advance(id, size_t(first) * istride * 4);
    advance(out, size_t(first) * heads * latent * 4);
    keys_.layout = layout_for(F32, latent, width); keys_.heads = count;
    r.dispatch("mla_gather", {cache, id, View(*keys_.data)},
               std::array<uint32_t, 3>{width, istride, count}, latent / 256, width, count);
    GemvArgs a; a.columns = heads; a.xs = latent; a.ys = width;
    a.hx = heads * latent; a.hy = heads * width;
    gemm(r, keys_, q, View(*scores_), a, 64, true);
    struct Softmax {uint32_t width, stride; float scale;} softmax{width, istride, scale};
    r.dispatch("mla_softmax", {View(*scores_), id}, softmax, heads, count);
    a.xs = width; a.ys = latent; a.hx = heads * width; a.hy = heads * latent;
    // Read the gathered keys as transposed weights. No second cache copy.
    keys_.layout = layout_for(F32, width, latent);
    gemm(r, keys_, View(*scores_), out, a, 64, false, true);
  }
}
}
