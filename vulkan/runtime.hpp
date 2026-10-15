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
    Buffer(Compute &, size_t, bool device_local = false, bool host_access = true);
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
  std::vector<float> mla_batch(int layer, std::span<const float> rows, int position,
                               int capacity, std::span<const float> q,
                               const std::vector<std::vector<int>> &ids,
                               int heads, float scale);
  std::vector<float> index_scores(std::span<const float> pooled, std::span<const float> queries,
                                   std::span<const float> weights, std::span<const int> counts,
                                   int heads,int dim);
  void mhc_read(std::span<const float> r, std::span<const float> projected,
                std::span<const float> base, std::span<const float> scale,
                std::span<float> c, std::span<float> x, int iterations,
                float eps);
  void mhc_write(std::span<const float> r, std::span<const float> c,
                 std::span<const float> y, std::span<float> out);
  std::vector<float> kda_state(int layer) const;
  void reset();
  void release_working_buffers();
  size_t upload_expert(int key, std::span<const uint8_t> gate,
                     std::span<const uint8_t> up, std::span<const uint8_t> down,
                       size_t max_bytes = size_t(-1));
  std::vector<float> expert_dot(int key, std::span<const uint8_t> activation,
                                int columns, int rows, int type, size_t offset, int tokens = 1);
  void clear_experts();
  void release_expert(int key);
  struct Snapshot {
    Compute *owner = nullptr;
    std::map<int, std::unique_ptr<Buffer>> state, convolution;
    std::map<int, int> positions;
  };
  void rewind_mla(int layer, int position);
  Snapshot checkpoint();
  void restore(const Snapshot &);

  struct TransferMeasurement {
    std::string direction;
    size_t bytes;
    uint32_t source_flags, destination_flags;
    std::vector<double> milliseconds;
  };
  std::vector<TransferMeasurement> benchmark_transfers(size_t bytes, int repeats);
  size_t peak_bytes() const { return peak_; }
  double dispatch_ms = 0, transfer_ms = 0;
  const std::string &device_name() const { return name_; }

private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool commands_ = VK_NULL_HANDLE;
  VkCommandBuffer command_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0, max_groups_x_ = 0, max_groups_y_ = 0;
  size_t budget_, live_ = 0, peak_ = 0;
  bool float64_ = false, device_state_ = false;
  std::string name_;
  std::map<std::string, Kernel> kernels_;
  std::array<std::unique_ptr<Buffer>, 8> scratch_;
  std::map<int, std::unique_ptr<Buffer>> state_, cache_, conv_, kda_weights_;
  std::map<int, int> positions_;
  std::map<int, std::unique_ptr<Buffer>> experts_;
  void copy(Buffer &source, Buffer &destination, size_t bytes, size_t offset = 0, size_t source_offset = 0);
  void zero(Buffer &);
  std::unique_ptr<Buffer> persistent(size_t bytes);
  void dispatch(const std::string &, const std::array<Buffer *, 8> &,
                const void *, uint32_t, uint32_t groups_y = 1);
  void dispatch_grid(const std::string &, const std::array<Buffer *, 8> &,
                     const void *, uint32_t);
  Buffer &scratch(int, size_t);
  Kernel &kernel(const std::string &);
  void cleanup() noexcept;
  void destroy_kernel(Kernel &) noexcept;
  friend struct Buffer;
};
} // namespace strata::vulkan
