#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
namespace strata::vulkan {
class Compute {
public:
  struct Buffer {
    Compute *owner = nullptr;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    size_t size = 0, allocated = 0;
    VkMemoryPropertyFlags memory_flags = 0;
    Buffer(Compute &, size_t, bool device_local = false);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void put(const void *, size_t, size_t offset = 0);
    void get(void *, size_t, size_t offset = 0) const;
  };
  struct Kernel {
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
  };
  explicit Compute(std::string device_name,
                   size_t budget_bytes = 1024ULL * 1024 * 1024);
  ~Compute();
  Compute(const Compute &) = delete;
  Compute &operator=(const Compute &) = delete;
  void kda(int layer, std::span<const float> q, std::span<const float> k,
           std::span<const float> v, std::span<const float> g,
           std::span<const float> beta, std::span<float> y, int heads, int dim);
  void kda_fused(int layer, std::span<const float> raw_qkv,
                 std::span<const float> decay, std::span<const float> beta,
                 std::span<const float> gate,
                 std::span<const float> conv_weights,
                 std::span<const float> bias, std::span<const float> negative_a,
                 std::span<const float> norm, std::span<float> output,
                 int heads, int kernel_size, float lower, float eps);
  std::vector<float> kda_history(int layer) const;
  void mla(int layer, std::span<const float> row, int position, int capacity,
           std::span<const float> q, std::span<const int> ids,
           std::span<float> y, int heads, float scale);
  void mhc_read(std::span<const float> r, std::span<const float> projected,
                std::span<const float> base, std::span<const float> scale,
                std::span<float> c, std::span<float> x, int iterations,
                float eps);
  void mhc_write(std::span<const float> r, std::span<const float> c,
                 std::span<const float> y, std::span<float> out);
  std::vector<float> kda_state(int layer) const;
  void reset();
  size_t peak_bytes() const { return peak_; }
  double dispatch_ms = 0;
  const std::string &device_name() const { return name_; }

private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool commands_ = VK_NULL_HANDLE;
  VkCommandBuffer command_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  size_t budget_, live_ = 0, peak_ = 0;
  std::string name_;
  std::map<std::string, Kernel> kernels_;
  std::array<std::unique_ptr<Buffer>, 8> scratch_;
  std::map<int, std::unique_ptr<Buffer>> state_, cache_, conv_, kda_weights_;
  std::map<int, int> positions_;
  void dispatch(const std::string &, const std::array<Buffer *, 8> &,
                const void *, uint32_t);
  Buffer &scratch(int, size_t);
  Kernel &kernel(const std::string &);
  void cleanup() noexcept;
  void destroy_kernel(Kernel &) noexcept;
  friend struct Buffer;
};
} // namespace strata::vulkan
