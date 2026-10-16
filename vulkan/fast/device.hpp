#pragma once
// Single-device Vulkan runtime for the opt-in fast GLM engine. Unlike
// Compute (runtime.hpp), work is recorded into command buffers that hold many
// dispatches, descriptors are pushed at record time, and host/GPU handoff uses
// timeline semaphores instead of one fence per dispatch.
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
namespace strata::vulkan::fast {

enum class Memory {
  Device,  // device-local, not mapped
  Host,    // host-visible coherent system memory, cached when available
  Shared,  // device-local and host-visible (large BAR); falls back to Host
};

class Device;

struct Buffer {
  Device *owner = nullptr;
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  size_t size = 0, allocated = 0;
  VkMemoryPropertyFlags flags = 0;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer();
  template <class T> T *as() const { return static_cast<T *>(mapped); }
};

// A bound range. range==0 binds from offset to the end of the buffer.
struct View {
  const Buffer *buffer = nullptr;
  size_t offset = 0, range = 0;
  View() = default;
  View(const Buffer &b, size_t o = 0, size_t r = 0) : buffer(&b), offset(o), range(r) {}
  View(const std::unique_ptr<Buffer> &b, size_t o = 0, size_t r = 0) : buffer(b.get()), offset(o), range(r) {}
};

struct Pipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
};

class Device;

// Records dispatches into one command buffer. Every dispatch is followed by a
// compute barrier unless it is placed inside a concurrent() group, in which
// case only the group end inserts the barrier.
class Recorder {
public:
  Recorder(Device &, VkCommandBuffer);
  // Immediate, bounded prefill recording. Each dispatch completes before the
  // command buffer is reset; copies are drained at the next dispatch or flush.
  void bounded(uint32_t limit_ms, bool profile = false) { completed_limit_ms_ = limit_ms; bounded_profile_ = profile; }
  void flush();
  double submission_ms = 0, maximum_submission_ms = 0;
  uint64_t submissions = 0;
  std::map<std::string, double> kernel_ms;
  void dispatch(const std::string &kernel, std::initializer_list<View> bindings,
                const void *push, uint32_t push_bytes, uint32_t gx,
                uint32_t gy = 1, uint32_t gz = 1);
  template <class P>
  void dispatch(const std::string &kernel, std::initializer_list<View> bindings,
                const P &push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    static_assert(sizeof(P) <= 128 && sizeof(P) % 4 == 0);
    dispatch(kernel, bindings, &push, sizeof(P), gx, gy, gz);
  }
  void begin_concurrent();
  void end_concurrent();
  void barrier();
  void copy(View source, View destination, size_t bytes);
  void fill(View destination, size_t bytes, uint32_t value = 0);
  void timestamp(uint32_t query);
  // Makes prior shader/transfer writes visible to host reads after the
  // submission's semaphore or fence completes.
  void host_visible();
  VkCommandBuffer handle() const { return command_; }
  // Profiling: when set, every dispatch is followed by a timestamp at query
  // (*next)++ and its kernel name is appended to *names.
  std::vector<std::string> *names = nullptr;
  uint32_t *next = nullptr;

private:
  Device &device_;
  VkCommandBuffer command_;
  bool concurrent_ = false, pending_ = false;
  uint32_t completed_limit_ms_ = 0;
  bool bounded_profile_ = false;
};

struct Wait {
  VkSemaphore semaphore = VK_NULL_HANDLE;
  uint64_t value = 0;
};
struct Submission {
  VkCommandBuffer command = VK_NULL_HANDLE;
  Wait wait, signal;  // null semaphore: none
};

class Device {
public:
  explicit Device(const std::string &name_filter = "", size_t budget = 15ULL << 30, bool allow_cpu = false);
  ~Device();
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  std::unique_ptr<Buffer> buffer(size_t bytes, Memory kind, bool addressable = false);
  uint64_t address(const Buffer &) const;
  // Uploads through a reusable staging buffer when the destination is not mapped.
  void upload(const Buffer &destination, const void *data, size_t bytes, size_t offset = 0);
  void download(const Buffer &source, void *data, size_t bytes, size_t offset = 0);

  VkCommandBuffer allocate_command(bool upload_queue = false);
  void free_command(VkCommandBuffer, bool upload_queue = false);
  void begin(VkCommandBuffer, bool one_time = true);
  void end(VkCommandBuffer);
  // Submits one command buffer. Waits/signals are timeline values.
  void submit(VkCommandBuffer, std::span<const Wait> waits,
              std::span<const Wait> signals, bool upload_queue = false,
              VkFence fence = VK_NULL_HANDLE);
  // Records, submits and waits (60 s bound). Used for setup work.
  template <class F> void run(F &&record, bool upload_queue = false) {
    auto c = allocate_command(upload_queue);
    begin(c);
    Recorder r(*this, c);
    record(r);
    end(c);
    submit_wait(c, upload_queue);
    free_command(c, upload_queue);
  }
  void submit_wait(VkCommandBuffer, bool upload_queue = false);
  // One vkQueueSubmit for several pre-recorded command buffers, in order.
  void submit_batch(std::span<const Submission>);

  VkSemaphore timeline(uint64_t initial = 0);
  void destroy(VkSemaphore);
  void signal(VkSemaphore, uint64_t value);
  // Returns false on timeout. Exits on device loss.
  bool wait(VkSemaphore, uint64_t value, uint64_t timeout_ns = 60ULL * 1000000000);
  uint64_t value(VkSemaphore);
  void wait_idle();

  // GPU timestamps, nanoseconds.
  void reset_queries(VkCommandBuffer, uint32_t first, uint32_t count);
  std::vector<double> read_timestamps(uint32_t first, uint32_t count);
  void host_reset_queries(uint32_t first, uint32_t count);

  const Pipeline &pipeline(const std::string &name);
  VkPipelineLayout layout() const { return layout_; }
  VkDevice handle() const { return device_; }
  const std::string &name() const { return name_; }
  uint32_t subgroup_size() const {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_, &properties);
    return subgroup.subgroupSize;
  }
  size_t live_bytes() const { return live_; }
  size_t peak_bytes() const { return peak_; }
  size_t budget() const { return budget_; }
  bool shared_heap_large() const { return shared_large_; }
  bool fp64_expert_scale() const { return fp64_expert_scale_; }
  bool fp64_expert_math() const { return fp64_expert_math_; }
  bool fp32_expert_math() const { return fp32_expert_math_; }
  bool native_expert_fma() const { return native_expert_fma_; }
  uint32_t max_groups_x() const { return max_groups_x_; }
  VkQueryPool queries() const { return queries_; }
  static constexpr uint32_t query_capacity = 4096;

private:
  friend struct Buffer;
  friend class Recorder;
  bool fp64_expert_scale_ = false;
  bool fp64_expert_math_ = false;
  bool fp32_expert_math_ = false;
  bool native_expert_fma_ = false;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE, upload_queue_ = VK_NULL_HANDLE;
  uint32_t family_ = 0;
  VkCommandPool pool_ = VK_NULL_HANDLE, upload_pool_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptors_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkQueryPool queries_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE, upload_fence_ = VK_NULL_HANDLE;
  PFN_vkCmdPushDescriptorSetKHR push_descriptors_ = nullptr;
  std::map<std::string, Pipeline> pipelines_;
  std::unique_ptr<Buffer> staging_;
  VkPhysicalDeviceMemoryProperties memory_{};
  size_t budget_, live_ = 0, peak_ = 0;
  uint32_t max_groups_x_ = 0;
  double timestamp_period_ = 1;
  bool shared_large_ = false;
  bool cpu_device_ = false;
  VkDeviceSize max_storage_range_ = 0;
  std::string name_;
  void cleanup() noexcept;
};

[[noreturn]] void fatal(const std::string &message);
void check(VkResult, const char *operation);

} // namespace strata::vulkan::fast
