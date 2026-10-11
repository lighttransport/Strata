#include "projections.hpp"
#include "runtime.hpp"
#include "strata/kernels/glm_reference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
namespace ref = strata::kernels::glm_reference;
void close(const std::vector<float> &a, const std::vector<float> &b,
           float tolerance, const char *name) {
  if (a.size() != b.size())
    throw std::runtime_error("parity geometry differs");
  double error = 0, energy = 0;
  float maximum = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("parity nonfinite");
    error += (a[i] - b[i]) * (a[i] - b[i]);
    energy += b[i] * b[i];
    maximum = std::max(maximum, std::abs(a[i] - b[i]));
  }
  double relative = std::sqrt(error / std::max(1e-20, energy));
  std::cerr << "PARITY name=" << name << " relative_l2=" << relative
            << " max_abs=" << maximum << '\n';
  if (relative > tolerance)
    throw std::runtime_error(std::string(name) + " parity failed");
}
int main(int argc, char **argv) {
  try {
    strata::vulkan::Projections projections(0, 512ULL * 1024 * 1024);
    strata::vulkan::Compute gpu(projections.device_name());
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> random(-.2f, .2f);
    auto values = [&](size_t n) {
      std::vector<float> v(n);
      for (float &x : v)
        x = random(rng);
      return v;
    };
    for (int nt : {1, 2, 8, 64}) {
      int in = 256, out = 128;
      auto w = values(in * out), x = values(in * nt);
      std::span<const uint8_t> bytes(
          reinterpret_cast<const uint8_t *>(w.data()), w.size() * 4);
      auto actual = projections.synthetic("f32-" + std::to_string(nt),
                                          GGML_TYPE_F32, bytes, in, out, x, nt);
      std::vector<float> expected(out * nt);
      for (int t = 0; t < nt; t++)
        for (int r = 0; r < out; r++)
          for (int i = 0; i < in; i++)
            expected[t * out + r] += w[r * in + i] * x[t * in + i];
      close(actual, expected, 2e-5, "projection-f32");
    }
    for (auto type :
         {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
          GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) {
      const int in = 4096, out = 128;
      auto w = values(in * out), x = values(in);
      std::vector<uint8_t> packed(ggml_row_size(type, in) * out);
      const size_t written = ggml_quantize_chunk(type, w.data(), packed.data(),
                                                 0, out, in, nullptr);
      if (written != packed.size())
        throw std::runtime_error("quantized fixture size differs");
      std::vector<float> decoded(w.size()), expected(out);
      auto quantized_x = strata::vulkan::decode_activations(x);
      ggml_get_type_traits(type)->to_float(packed.data(), decoded.data(),
                                           decoded.size());
      for (int r = 0; r < out; ++r)
        for (int i = 0; i < in; ++i)
          expected[r] += decoded[r * in + i] * quantized_x[i];
      if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
        const int blocks = in / 32;
        for (int r = 0; r < out; ++r)
          for (int b = 0; b < blocks; ++b) {
            float reconstructed = 0;
            for (int i = 0; i < 32; ++i)
              reconstructed += quantized_x[b * 32 + i];
            auto sum = [&](auto &&self, int first, int stride) -> float {
              if (stride == 32)
                return x[b * 32 + first];
              return self(self, first, stride * 2) +
                     self(self, first + stride, stride * 2);
            };
            float original = sum(sum, 0, 1);
            original = ggml_fp16_to_fp32(ggml_fp32_to_fp16(original));
            ggml_fp16_t d;
            std::memcpy(&d,
                        packed.data() +
                            (size_t(r) * blocks + b) * ggml_type_size(type),
                        sizeof(d));
            expected[r] += (type == GGML_TYPE_Q4_0 ? -8.f : -16.f) *
                           ggml_fp16_to_fp32(d) * (original - reconstructed);
          }
      }
      auto actual = projections.synthetic("quant-" + std::to_string(type), type,
                                          packed, in, out, x);
      close(actual, expected, 2e-5f, "projection-quantized");
      auto repeat = projections.synthetic("quant-" + std::to_string(type), type,
                                          packed, in, out, x);
      if (actual != repeat)
        throw std::runtime_error("Cached projection repeat differs");
    }
    if (const char *check = std::getenv("STRATA_VULKAN_AVX2_ACTIVATIONS");
        check && std::string(check) == "1") {
      auto input = values(32 * 128);
      for (int block = 0; block < 64; ++block) {
        input[block * 32 + 31] = 127.f;
        input[block * 32] = .5f;
        input[block * 32 + 1] = -.5f;
        input[block * 32 + 2] = std::nextafter(.5f, 0.f);
        input[block * 32 + 3] = std::nextafter(.5f, 1.f);
        input[block * 32 + 4] = -0.f;
      }
      auto vectorized = strata::vulkan::decode_activations(input);
#ifdef _WIN32
      _putenv_s("STRATA_VULKAN_AVX2_ACTIVATIONS", "0");
#else
      setenv("STRATA_VULKAN_AVX2_ACTIVATIONS", "0", 1);
#endif
      auto scalar = strata::vulkan::decode_activations(input);
#ifdef _WIN32
      _putenv_s("STRATA_VULKAN_AVX2_ACTIVATIONS", "1");
#else
      setenv("STRATA_VULKAN_AVX2_ACTIVATIONS", "1", 1);
#endif
      if (std::memcmp(scalar.data(), vectorized.data(),
                      scalar.size() * sizeof(float)))
        throw std::runtime_error("AVX2 activation bytes differ from scalar");
      std::cerr << "PARITY name=activation-avx2 bitwise_equal=1 values="
                << input.size() << '\n';
    }
    const int heads = 64, dim = 128, n = heads * dim;
    std::vector<float> state(size_t(n) * dim), actual(n), expected(n);
    for (int step = 0; step < 5; step++) {
      auto q = values(n), k = values(n), v = values(n), g = values(n),
           beta = values(heads);
      for (float &x : g)
        x = -std::abs(x);
      gpu.kda(0, q, k, v, g, beta, actual, heads, dim);
      ref::kda(state.data(), q.data(), k.data(), v.data(), g.data(),
               beta.data(), expected.data(), heads, dim);
      close(actual, expected, 2e-5, "kda-output");
      close(gpu.kda_state(0), state, 2e-5, "kda-state");
    }
    // Fused recurrent path: convolution history, gate equations and output RMS.
    for (int convolution : {1, 4}) {
      const int count = heads * dim, layer = 20 + convolution;
      auto weights = values(3 * count * convolution), bias = values(count),
           a = values(heads), norm = values(dim);
      for (float &value : a)
        value = -std::abs(value) - 0.1f;
      std::vector<float> history(3 * count * (convolution - 1)),
          expected_state(size_t(count) * dim), output(count);
      for (int step = 0; step < 5; ++step) {
        auto raw = values(3 * count), decay = values(count),
             beta = values(heads), gate = values(count);
        std::array<std::vector<float>, 3> convolved;
        for (int part = 0; part < 3; ++part) {
          convolved[part].resize(count);
          for (int channel = 0; channel < count; ++channel) {
            const int wi = (part * count + channel) * convolution,
                      hi = (part * count + channel) * (convolution - 1);
            float sum =
                raw[part * count + channel] * weights[wi + convolution - 1];
            for (int k = 0; k < convolution - 1; ++k)
              sum += history[hi + k] * weights[wi + k];
            for (int k = 0; k < convolution - 2; ++k)
              history[hi + k] = history[hi + k + 1];
            if (convolution > 1)
              history[hi + convolution - 2] = raw[part * count + channel];
            convolved[part][channel] = sum * ref::sigmoid(sum);
          }
        }
        auto g = decay, b = beta;
        std::vector<float> expected(count);
        for (int channel = 0; channel < count; ++channel)
          g[channel] = -5.f * ref::sigmoid(-a[channel / dim] *
                                           (decay[channel] + bias[channel]));
        ref::kda(expected_state.data(), convolved[0].data(),
                 convolved[1].data(), convolved[2].data(), g.data(), b.data(),
                 expected.data(), heads, dim);
        for (int h = 0; h < heads; ++h) {
          float energy = 0;
          for (int j = 0; j < dim; ++j)
            energy += expected[h * dim + j] * expected[h * dim + j];
          const float inv = 1 / std::sqrt(energy / dim + 1e-6f);
          for (int j = 0; j < dim; ++j)
            expected[h * dim + j] = expected[h * dim + j] * inv * norm[j] *
                                    ref::sigmoid(gate[h * dim + j]);
        }
        gpu.kda_fused(layer, raw, decay, beta, gate, weights, bias, a, norm,
                      output, heads, convolution, -5.f, 1e-6f);
        close(output, expected, 2e-5, "kda-fused-output");
        close(gpu.kda_state(layer), expected_state, 2e-5, "kda-fused-state");
        if (convolution > 1)
          close(gpu.kda_history(layer), history, 0, "kda-fused-history");
      }
    }
    auto r = values(4 * 256), p = values(24), base = values(24),
         scale = values(3), c = values(24), x = values(256), ec = values(24),
         ex = values(256);
    ref::mhc(p.data(), base.data(), scale.data(), ec.data(), 20, 1e-6f);
    for (int d = 0; d < 256; d++) {
      ex[d] = 0;
      for (int i = 0; i < 4; i++)
        ex[d] += ec[i] * r[i * 256 + d];
    }
    gpu.mhc_read(r, p, base, scale, c, x, 20, 1e-6f);
    close(c, ec, 2e-5, "mhc-coefficients");
    close(x, ex, 2e-5, "mhc-read");
    auto y = values(256);
    std::vector<float> out(1024), eo(1024);
    for (int j = 0; j < 4; j++)
      for (int d = 0; d < 256; d++) {
        eo[j * 256 + d] = c[4 + j] * y[d];
        for (int i = 0; i < 4; i++)
          eo[j * 256 + d] += c[8 + i * 4 + j] * r[i * 256 + d];
      }
    gpu.mhc_write(r, c, y, out);
    close(out, eo, 2e-5, "mhc-write");
    int latent = 512;
    std::vector<float> cache;
    for (int pos = 0; pos < 17; pos++) {
      auto row = values(latent), q = values(heads * latent);
      cache.insert(cache.end(), row.begin(), row.end());
      std::vector<int> ids;
      for (int i = 0; i <= pos; i++)
        if (i % 3 != 1)
          ids.push_back(i);
      std::vector<float> a(heads * latent), e(heads * latent);
      gpu.mla(1, row, pos, 32, q, ids, a, heads, .125f);
      for (int h = 0; h < heads; h++) {
        std::vector<float> scores;
        for (int id : ids) {
          float s = 0;
          for (int d = 0; d < latent; d++)
            s += q[h * latent + d] * cache[id * latent + d];
          scores.push_back(s * .125f);
        }
        float mx = *std::max_element(scores.begin(), scores.end()), total = 0;
        for (float &v : scores) {
          v = std::exp(v - mx);
          total += v;
        }
        for (size_t s = 0; s < scores.size(); s++)
          for (int d = 0; d < latent; d++)
            e[h * latent + d] += scores[s] / total * cache[ids[s] * latent + d];
      }
      close(a, e, 2e-5, "mla");
    }
    bool rejected = false;
    try {
      std::vector<float> q(heads * latent), y(heads * latent), row(latent);
      std::vector<int> ids{999};
      gpu.mla(1, row, 17, 32, q, ids, y, heads, 1);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("MLA invalid ID not rejected");
    if (argc == 2) {
      strata::core::ModelArtifact artifact(argv[1]);
      std::vector<std::pair<std::string, const strata::core::ArtifactTensor *>>
          items;
      for (const auto *part :
           {"attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_g_a", "ssm_beta"}) {
        std::string name = std::string("blk.1.") + part + ".weight";
        items.emplace_back(name, &artifact.at(name));
      }
      auto input = values(artifact.descriptor().hidden);
      auto grouped = projections.joined(items, input);
      auto repeated = projections.joined(items, input);
      for (size_t i = 0; i < items.size(); ++i) {
        auto individual =
            projections.mat(items[i].first, *items[i].second, input);
        close(grouped[i], individual, 2e-5, "projection-group");
        if (grouped[i] != repeated[i])
          throw std::runtime_error("Grouped projection repeat differs");
      }
      auto &fa = artifact.at("blk.1.ssm_f_b.weight");
      auto &ga = artifact.at("blk.1.ssm_g_b.weight");
      auto f = values(fa.tensor->shape[0]), g = values(ga.tensor->shape[0]);
      auto pair = projections.multiple(
          {{"blk.1.ssm_f_b.weight", &fa}, {"blk.1.ssm_g_b.weight", &ga}},
          {f, g});
      close(pair[0], projections.mat("blk.1.ssm_f_b.weight", fa, f), 2e-5,
            "projection-distinct-input-f");
      close(pair[1], projections.mat("blk.1.ssm_g_b.weight", ga, g), 2e-5,
            "projection-distinct-input-g");
    } else if (argc != 1) {
      throw std::invalid_argument("Usage: glm_vulkan_parity [model.gguf]");
    }
    gpu.reset();
    for (int layer : {21, 24}) {
      auto state = gpu.kda_state(layer), history = gpu.kda_history(layer);
      if (std::any_of(state.begin(), state.end(),
                      [](float value) { return value != 0; }) ||
          std::any_of(history.begin(), history.end(),
                      [](float value) { return value != 0; }))
        throw std::runtime_error("Fused KDA reset left recurrent history");
    }
    std::cout << "{\"backend\":\"vulkan\",\"passed\":true}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
