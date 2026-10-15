#pragma once
#include "strata/core/model.hpp"
#include <ggml-backend.h>
#include <ggml.h>
#include <map>
#include <span>
#include <string>
#include <vector>
namespace strata::vulkan {
// Match the existing runner's 32-value Q8_1 activation scale and rounding.
std::vector<float> decode_activations(std::span<const float> input);
class Projections {
public:
  explicit Projections(int device, size_t budget_bytes);
  ~Projections();
  Projections(const Projections &) = delete;
  Projections &operator=(const Projections &) = delete;
  std::vector<float> mat(const std::string &name,
                         const strata::core::ArtifactTensor &,
                         std::span<const float> input, int heads = 1,
                         int tokens = 1);
  std::vector<std::vector<float>>
  joined(const std::vector<
             std::pair<std::string, const strata::core::ArtifactTensor *>>
             &weights,
         std::span<const float> input);
  std::vector<std::vector<float>>
  multiple(const std::vector<
               std::pair<std::string, const strata::core::ArtifactTensor *>>
               &weights,
           const std::vector<std::span<const float>> &inputs);
  std::vector<float> synthetic(const std::string &key, ggml_type type,
                               std::span<const uint8_t> bytes, int in, int out,
                               std::span<const float> input, int tokens = 1,
                               int heads = 1);
  void release_plans();
  std::string device_name() const;
  size_t resident_bytes() const { return resident_; }
  double compute_ms = 0, transfer_ms = 0;
  double affine_cpu_ms = 0;
  uint64_t calls = 0, submissions = 0;

private:
  struct Weight {
    ggml_context *context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *tensor = nullptr;
    size_t bytes = 0;
    size_t allocated = 0;
    ggml_context *plan_context = nullptr;
    ggml_backend_buffer_t plan_buffer = nullptr;
    ggml_tensor *input = nullptr, *output = nullptr;
    ggml_cgraph *graph = nullptr;
    size_t plan_bytes = 0;
    int tokens = 0;
    std::vector<float> affine_bias;
  };
  ggml_backend_t backend_ = nullptr;
  struct Group {
    ggml_context *context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> inputs;
    ggml_cgraph *graph = nullptr;
    std::vector<ggml_tensor *> outputs;
  };
  std::map<std::string, Group> groups_;
  std::map<std::string, Weight> weights_;
  size_t budget_, resident_ = 0;
  int device_;
  bool async_staging_ = false, pending_ = false;
  ggml_backend_buffer_t upload_ = nullptr, download_ = nullptr;
  void ensure_staging(ggml_backend_buffer_t &buffer, size_t bytes);
  std::vector<float> product(const std::string &, ggml_type, const void *,
                             size_t, int, int, std::span<const float>, int,
                             int);
};
} // namespace strata::vulkan
