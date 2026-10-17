#include "fast/device.hpp"
#include "fast_shaders.hpp"
#include "strata/host/glm_env.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace strata::vulkan::fast {

void fatal(const std::string &message) {
  // A failed submission or wait may leave GPU work in flight. Exit without
  // freeing memory the device could still access.
  std::cerr << "VULKAN_FAST_ERROR " << message << "; exiting without freeing in-flight buffers\n";
  std::_Exit(74);
}
void check(VkResult r, const char *operation) {
  if (r != VK_SUCCESS)
    throw std::runtime_error(std::string("Vulkan ") + operation + " failed: " + std::to_string(r));
}
namespace {
void require(bool v, const char *message) {
  if (!v) throw std::invalid_argument(message);
}
} // namespace

Buffer::~Buffer() {
  if (!owner) return;
  if (mapped) vkUnmapMemory(owner->device_, memory);
  if (handle) vkDestroyBuffer(owner->device_, handle, nullptr);
  if (memory) vkFreeMemory(owner->device_, memory, nullptr);
  owner->live_ -= allocated;
}

Device::Device(const std::string &filter, size_t budget, bool allow_cpu) : budget_(budget) {
  try {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Strata GLM Vulkan fast";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    check(vkCreateInstance(&info, nullptr, &instance_), "create instance");
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "enumerate devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "enumerate devices");
    int matches = 0;
    for (auto d : devices) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(d, &p);
      if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !allow_cpu) continue;
      if (!filter.empty() && std::string(p.deviceName).find(filter) == std::string::npos) continue;
      if (!physical_) {
        physical_ = d;
        name_ = p.deviceName;
        cpu_device_ = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
      }
      ++matches;
    }
    require(physical_ && (filter.empty() || matches == 1), "Vulkan fast device not found or ambiguous");
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_, &properties);
    require(properties.apiVersion >= VK_API_VERSION_1_2, "Vulkan 1.2 required");
    max_groups_x_ = properties.limits.maxComputeWorkGroupCount[0];
    max_storage_range_ = properties.limits.maxStorageBufferRange;
    timestamp_period_ = properties.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
      const auto f = memory_.memoryTypes[i].propertyFlags;
      if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          memory_.memoryHeaps[memory_.memoryTypes[i].heapIndex].size >= (1ULL << 30))
        shared_large_ = true;
    }

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qs(families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, qs.data());
    int chosen = -1;
    for (uint32_t i = 0; i < families; ++i)  // prefer a compute-only family with two queues
      if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          qs[i].queueCount >= 2 && qs[i].timestampValidBits) {
        chosen = int(i);
        break;
      }
    if (chosen < 0)
      for (uint32_t i = 0; i < families; ++i)
        if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qs[i].timestampValidBits) {
          chosen = int(i);
          break;
        }
    require(chosen >= 0, "Vulkan compute queue with timestamps unavailable");
    family_ = uint32_t(chosen);
    const uint32_t queue_count = std::min<uint32_t>(2, qs[family_].queueCount);
    const std::array<float, 2> priorities{1.f, 1.f};
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = family_;
    qi.queueCount = queue_count;
    qi.pQueuePriorities = priorities.data();

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    require(f12.timelineSemaphore && f12.storageBuffer8BitAccess && f12.shaderInt8 &&
                f11.storageBuffer16BitAccess && f2.features.shaderInt16,
            "Vulkan fast engine needs timeline semaphores and 8/16-bit storage");
    VkPhysicalDeviceVulkan12Features e12{};
    e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    e12.timelineSemaphore = VK_TRUE;
    e12.storageBuffer8BitAccess = VK_TRUE;
    e12.uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess;
    e12.shaderInt8 = VK_TRUE;
    e12.hostQueryReset = VK_TRUE;
    e12.bufferDeviceAddress = f12.bufferDeviceAddress;
    require(f12.hostQueryReset, "Vulkan fast engine needs host query reset");
    VkPhysicalDeviceVulkan11Features e11{};
    e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    e11.storageBuffer16BitAccess = VK_TRUE;
    e11.uniformAndStorageBuffer16BitAccess = f11.uniformAndStorageBuffer16BitAccess;
    e11.pNext = &e12;
    VkPhysicalDeviceFeatures2 e2{};
    e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    e2.pNext = &e11;
    e2.features.shaderInt16 = VK_TRUE;
    e2.features.shaderInt64 = f2.features.shaderInt64;
    fp64_expert_math_ = strata_env_is_one("STRATA_VULKAN_FAST_EXPERT_FP64_MATH");
    fp64_expert_scale_ = strata_env_is_one("STRATA_VULKAN_FAST_EXPERT_FP64_SCALE") || fp64_expert_math_;
    fp32_expert_math_ = strata_env_is_one("STRATA_VULKAN_FAST_EXPERT_FP32_MATH");
    require(!fp32_expert_math_ || !fp64_expert_scale_, "FP32 expert math cannot be combined with FP64 expert modes");
    require(!fp64_expert_scale_ || f2.features.shaderFloat64, "Vulkan FP64 expert scale diagnostic unavailable");
    e2.features.shaderFloat64 = fp64_expert_scale_ ? VK_TRUE : VK_FALSE;
    const bool robust = strata_env_is_one("STRATA_VULKAN_FAST_ROBUST_BUFFERS");
    require(!robust || f2.features.robustBufferAccess, "Vulkan robust buffer access unavailable");
    e2.features.robustBufferAccess = robust ? VK_TRUE : VK_FALSE;
    const char *extensions[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, nullptr};
    uint32_t enabled_extensions = 1;
    native_expert_fma_ = strata_env_is_one("STRATA_VULKAN_FAST_EXPERT_NATIVE_FMA");
    require(!native_expert_fma_ || fp64_expert_math_ || fp32_expert_math_, "Native expert FMA requires FP64 or FP32 expert math mode");
    require(!fp32_expert_math_ || native_expert_fma_, "FP32 expert math requires native expert FMA");
#if defined(STRATA_VULKAN_NATIVE_FMA) && defined(VK_KHR_shader_fma)
    VkPhysicalDeviceShaderFmaFeaturesKHR efma{};
    efma.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR;
    if (native_expert_fma_) {
      uint32_t count = 0;
      check(vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count, nullptr), "count device extensions");
      std::vector<VkExtensionProperties> supported(count);
      check(vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count, supported.data()), "device extensions");
      require(std::any_of(supported.begin(), supported.end(), [](const auto &extension) {
                return std::strcmp(extension.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME) == 0;
              }), "VK_KHR_shader_fma unavailable on selected Vulkan driver");
      VkPhysicalDeviceFeatures2 queried{};
      queried.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      queried.pNext = &efma;
      vkGetPhysicalDeviceFeatures2(physical_, &queried);
      require(efma.shaderFmaFloat32, "Vulkan driver lacks guaranteed FP32 FMA");
      efma.shaderFmaFloat16 = VK_FALSE;
      efma.shaderFmaFloat64 = VK_FALSE;
      efma.pNext = e2.pNext;
      e2.pNext = &efma;
      extensions[enabled_extensions++] = VK_KHR_SHADER_FMA_EXTENSION_NAME;
    }
#else
    require(!native_expert_fma_, "Native expert FMA unavailable in this build; enable STRATA_VULKAN_NATIVE_FMA with a supported SDK");
#endif
    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.pNext = &e2;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = enabled_extensions;
    di.ppEnabledExtensionNames = extensions;
    check(vkCreateDevice(physical_, &di, nullptr, &device_), "create device");
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    vkGetDeviceQueue(device_, family_, queue_count - 1, &upload_queue_);
    push_descriptors_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR"));
    require(push_descriptors_, "VK_KHR_push_descriptor unavailable");

    for (VkCommandPool *pool : {&pool_, &upload_pool_}) {
      VkCommandPoolCreateInfo pi{};
      pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pi.queueFamilyIndex = family_;
      check(vkCreateCommandPool(device_, &pi, nullptr, pool), "command pool");
    }
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dl.bindingCount = uint32_t(bindings.size());
    dl.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &dl, nullptr, &descriptors_), "descriptor layout");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1;
    li.pSetLayouts = &descriptors_;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device_, &li, nullptr, &layout_), "pipeline layout");
    VkQueryPoolCreateInfo qp{};
    qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount = query_capacity;
    check(vkCreateQueryPool(device_, &qp, nullptr, &queries_), "query pool");
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(vkCreateFence(device_, &fi, nullptr, &fence_), "fence");
    check(vkCreateFence(device_, &fi, nullptr, &upload_fence_), "fence");
    std::cerr << "VULKAN_FAST device=" << name_ << " family=" << family_ << " queues=" << queue_count
              << " large_bar=" << shared_large_ << " budget_MiB=" << (budget_ >> 20)
              << " max_groups_x=" << max_groups_x_ << " max_storage_range=" << max_storage_range_
              << " robust_buffers=" << robust << '\n';
  } catch (...) {
    cleanup();
    throw;
  }
}
Device::~Device() { cleanup(); }
void Device::cleanup() noexcept {
  if (device_) vkDeviceWaitIdle(device_);
  staging_.reset();
  for (auto &[name, p] : pipelines_)
    if (p.pipeline) vkDestroyPipeline(device_, p.pipeline, nullptr);
  pipelines_.clear();
  if (queries_) vkDestroyQueryPool(device_, queries_, nullptr);
  if (fence_) vkDestroyFence(device_, fence_, nullptr);
  if (upload_fence_) vkDestroyFence(device_, upload_fence_, nullptr);
  if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
  if (descriptors_) vkDestroyDescriptorSetLayout(device_, descriptors_, nullptr);
  if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
  if (upload_pool_) vkDestroyCommandPool(device_, upload_pool_, nullptr);
  if (device_) vkDestroyDevice(device_, nullptr);
  if (instance_) vkDestroyInstance(instance_, nullptr);
  queries_ = VK_NULL_HANDLE;
  fence_ = upload_fence_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  descriptors_ = VK_NULL_HANDLE;
  pool_ = upload_pool_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  instance_ = VK_NULL_HANDLE;
}

uint64_t Device::address(const Buffer &b) const {
  VkBufferDeviceAddressInfo i{};
  i.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  i.buffer = b.handle;
  return vkGetBufferDeviceAddress(device_, &i);
}
std::unique_ptr<Buffer> Device::buffer(size_t bytes, Memory kind, bool addressable) {
  auto b = std::make_unique<Buffer>();
  b->owner = this;
  b->size = std::max<size_t>(16, (bytes + 15) & ~size_t(15));
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = b->size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT | (addressable ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device_, &info, nullptr, &b->handle), "create buffer");
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, b->handle, &req);
  require(req.size <= budget_ - std::min(budget_, live_), "Vulkan fast allocation exceeds budget");
  // Candidate memory types, best first.
  std::vector<std::pair<int, uint32_t>> candidates;
  for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
    if (!(req.memoryTypeBits & (1u << i))) continue;
    const auto f = memory_.memoryTypes[i].propertyFlags;
    const bool local = f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const bool visible = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const bool cached = f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    int score = -1;
    if (kind == Memory::Device) score = local ? (visible ? 2 : 3) : -1;
    else if (kind == Memory::Host) score = visible && (!local || cpu_device_) ? (cached ? 3 : 2) : -1;
    else score = visible ? (local ? 3 : (cached ? 1 : 0)) : -1;
    if (score >= 0) candidates.emplace_back(score, i);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.first > b.first; });
  VkMemoryAllocateFlagsInfo flags{};
  flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.pNext = addressable ? &flags : nullptr;
  alloc.allocationSize = req.size;
  VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  for (auto [score, type] : candidates) {
    alloc.memoryTypeIndex = type;
    result = vkAllocateMemory(device_, &alloc, nullptr, &b->memory);
    if (result == VK_SUCCESS) {
      b->flags = memory_.memoryTypes[type].propertyFlags;
      break;
    }
    b->memory = VK_NULL_HANDLE;
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) break;
  }
  if (result != VK_SUCCESS) {
    vkDestroyBuffer(device_, b->handle, nullptr);
    b->handle = VK_NULL_HANDLE;
    b->owner = nullptr;
    check(result, "allocate buffer");
  }
  b->allocated = req.size;
  live_ += req.size;
  peak_ = std::max(peak_, live_);
  check(vkBindBufferMemory(device_, b->handle, b->memory, 0), "bind buffer");
  if (b->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    check(vkMapMemory(device_, b->memory, 0, VK_WHOLE_SIZE, 0, &b->mapped), "map buffer");
  return b;
}

void Device::upload(const Buffer &dst, const void *data, size_t bytes, size_t offset) {
  require(offset <= dst.size && bytes <= dst.size - offset, "Vulkan fast upload out of bounds");
  if (!bytes) return;
  if (dst.mapped) {
    std::memcpy(static_cast<char *>(dst.mapped) + offset, data, bytes);
    return;
  }
  constexpr size_t chunk = 64ULL << 20;
  if (!staging_) staging_ = buffer(chunk, Memory::Host);
  for (size_t done = 0; done < bytes; done += chunk) {
    const size_t n = std::min(chunk, bytes - done);
    std::memcpy(staging_->mapped, static_cast<const char *>(data) + done, n);
    run([&](Recorder &r) { r.copy(View(*staging_), View(dst, offset + done), n); });
  }
}
void Device::download(const Buffer &src, void *data, size_t bytes, size_t offset) {
  require(offset <= src.size && bytes <= src.size - offset, "Vulkan fast download out of bounds");
  if (!bytes) return;
  if (src.mapped) {
    std::memcpy(data, static_cast<const char *>(src.mapped) + offset, bytes);
    return;
  }
  constexpr size_t chunk = 64ULL << 20;
  if (!staging_) staging_ = buffer(chunk, Memory::Host);
  for (size_t done = 0; done < bytes; done += chunk) {
    const size_t n = std::min(chunk, bytes - done);
    run([&](Recorder &r) {
      r.copy(View(src, offset + done), View(*staging_), n);
      r.host_visible();
    });
    std::memcpy(static_cast<char *>(data) + done, staging_->mapped, n);
  }
}

VkCommandBuffer Device::allocate_command(bool upload_queue) {
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = upload_queue ? upload_pool_ : pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer c;
  check(vkAllocateCommandBuffers(device_, &ai, &c), "allocate command buffer");
  return c;
}
void Device::free_command(VkCommandBuffer c, bool upload_queue) {
  vkFreeCommandBuffers(device_, upload_queue ? upload_pool_ : pool_, 1, &c);
}
void Device::begin(VkCommandBuffer c, bool one_time) {
  check(vkResetCommandBuffer(c, 0), "reset command buffer");
  VkCommandBufferBeginInfo b{};
  b.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  b.flags = one_time ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
  check(vkBeginCommandBuffer(c, &b), "begin command buffer");
  // Order against host writes and against work in earlier submissions on this
  // queue (pre-submitted segments of one token depend on each other).
  VkMemoryBarrier m{};
  m.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  m.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0,
                       nullptr, 0, nullptr);
}
void Device::end(VkCommandBuffer c) { check(vkEndCommandBuffer(c), "end command buffer"); }
void Device::submit(VkCommandBuffer c, std::span<const Wait> waits, std::span<const Wait> signals,
                    bool upload_queue, VkFence fence) {
  std::vector<VkSemaphore> ws, ss;
  std::vector<uint64_t> wv, sv;
  std::vector<VkPipelineStageFlags> stages;
  for (auto w : waits) {
    ws.push_back(w.semaphore);
    wv.push_back(w.value);
    stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
  for (auto s : signals) {
    ss.push_back(s.semaphore);
    sv.push_back(s.value);
  }
  VkTimelineSemaphoreSubmitInfo ti{};
  ti.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  ti.waitSemaphoreValueCount = uint32_t(wv.size());
  ti.pWaitSemaphoreValues = wv.data();
  ti.signalSemaphoreValueCount = uint32_t(sv.size());
  ti.pSignalSemaphoreValues = sv.data();
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.pNext = &ti;
  si.waitSemaphoreCount = uint32_t(ws.size());
  si.pWaitSemaphores = ws.data();
  si.pWaitDstStageMask = stages.data();
  si.commandBufferCount = c ? 1 : 0;
  si.pCommandBuffers = &c;
  si.signalSemaphoreCount = uint32_t(ss.size());
  si.pSignalSemaphores = ss.data();
  const auto r = vkQueueSubmit(upload_queue ? upload_queue_ : queue_, 1, &si, fence);
  if (r != VK_SUCCESS) fatal("queue submit failed: " + std::to_string(r));
}
void Device::submit_batch(std::span<const Submission> list) {
  if (list.empty()) return;
  const size_t n = list.size();
  std::vector<VkTimelineSemaphoreSubmitInfo> ti(n);
  std::vector<VkSubmitInfo> si(n);
  std::vector<VkSemaphore> ws(n), ss(n);
  std::vector<uint64_t> wv(n), sv(n);
  std::vector<VkPipelineStageFlags> stage(n, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  for (size_t i = 0; i < n; ++i) {
    const auto &s = list[i];
    ws[i] = s.wait.semaphore;
    wv[i] = s.wait.value;
    ss[i] = s.signal.semaphore;
    sv[i] = s.signal.value;
    ti[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ti[i].waitSemaphoreValueCount = s.wait.semaphore ? 1 : 0;
    ti[i].pWaitSemaphoreValues = &wv[i];
    ti[i].signalSemaphoreValueCount = s.signal.semaphore ? 1 : 0;
    ti[i].pSignalSemaphoreValues = &sv[i];
    si[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si[i].pNext = &ti[i];
    si[i].waitSemaphoreCount = s.wait.semaphore ? 1 : 0;
    si[i].pWaitSemaphores = &ws[i];
    si[i].pWaitDstStageMask = &stage[i];
    si[i].commandBufferCount = 1;
    si[i].pCommandBuffers = &s.command;
    si[i].signalSemaphoreCount = s.signal.semaphore ? 1 : 0;
    si[i].pSignalSemaphores = &ss[i];
  }
  const auto r = vkQueueSubmit(queue_, uint32_t(n), si.data(), VK_NULL_HANDLE);
  if (r != VK_SUCCESS) fatal("batched queue submit failed: " + std::to_string(r));
}
void Device::submit_wait(VkCommandBuffer c, bool upload_queue) {
  VkFence f = upload_queue ? upload_fence_ : fence_;
  check(vkResetFences(device_, 1, &f), "reset fence");
  submit(c, {}, {}, upload_queue, f);
  const auto r = vkWaitForFences(device_, 1, &f, VK_TRUE, 60ULL * 1000000000);
  if (r != VK_SUCCESS) fatal("fence wait failed: " + std::to_string(r));
}

VkSemaphore Device::timeline(uint64_t initial) {
  VkSemaphoreTypeCreateInfo t{};
  t.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  t.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  t.initialValue = initial;
  VkSemaphoreCreateInfo i{};
  i.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  i.pNext = &t;
  VkSemaphore s;
  check(vkCreateSemaphore(device_, &i, nullptr, &s), "create timeline semaphore");
  return s;
}
void Device::destroy(VkSemaphore s) {
  if (s) vkDestroySemaphore(device_, s, nullptr);
}
void Device::signal(VkSemaphore s, uint64_t v) {
  VkSemaphoreSignalInfo i{};
  i.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  i.semaphore = s;
  i.value = v;
  const auto r = vkSignalSemaphore(device_, &i);
  if (r != VK_SUCCESS) fatal("host semaphore signal failed: " + std::to_string(r));
}
bool Device::wait(VkSemaphore s, uint64_t v, uint64_t timeout) {
  VkSemaphoreWaitInfo i{};
  i.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  i.semaphoreCount = 1;
  i.pSemaphores = &s;
  i.pValues = &v;
  const auto r = vkWaitSemaphores(device_, &i, timeout);
  if (r == VK_TIMEOUT) return false;
  if (r != VK_SUCCESS) fatal("semaphore wait failed: " + std::to_string(r));
  return true;
}
uint64_t Device::value(VkSemaphore s) {
  uint64_t v = 0;
  const auto r = vkGetSemaphoreCounterValue(device_, s, &v);
  if (r != VK_SUCCESS) fatal("semaphore query failed: " + std::to_string(r));
  return v;
}
void Device::wait_idle() {
  const auto r = vkDeviceWaitIdle(device_);
  if (r != VK_SUCCESS) fatal("device wait idle failed: " + std::to_string(r));
}
void Device::reset_queries(VkCommandBuffer c, uint32_t first, uint32_t count) {
  vkCmdResetQueryPool(c, queries_, first, count);
}
void Device::host_reset_queries(uint32_t first, uint32_t count) { vkResetQueryPool(device_, queries_, first, count); }
std::vector<double> Device::read_timestamps(uint32_t first, uint32_t count) {
  std::vector<uint64_t> raw(count);
  check(vkGetQueryPoolResults(device_, queries_, first, count, raw.size() * 8, raw.data(), 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "query results");
  std::vector<double> ns(count);
  for (uint32_t i = 0; i < count; ++i) ns[i] = double(raw[i]) * timestamp_period_;
  return ns;
}

const Pipeline &Device::pipeline(const std::string &name) {
  if (auto f = pipelines_.find(name); f != pipelines_.end()) return f->second;
  auto bytes = fast_shader(name);
  require(!bytes.empty() && bytes.size() % 4 == 0, "Unknown Vulkan fast shader");
  std::vector<uint32_t> code(bytes.size() / 4);
  std::memcpy(code.data(), bytes.data(), bytes.size());
  VkShaderModuleCreateInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  si.codeSize = bytes.size();
  si.pCode = code.data();
  VkShaderModule module;
  check(vkCreateShaderModule(device_, &si, nullptr, &module), "shader module");
  VkComputePipelineCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = module;
  ci.stage.pName = "main";
  ci.layout = layout_;
  Pipeline p;
  const auto r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline);
  vkDestroyShaderModule(device_, module, nullptr);
  check(r, ("compute pipeline " + name).c_str());
  return pipelines_.emplace(name, p).first->second;
}

Recorder::Recorder(Device &d, VkCommandBuffer c) : device_(d), command_(c) {}
void Recorder::flush() {
  require(completed_limit_ms_ > 0, "flush requires bounded recording");
  barrier();
  host_visible();
  device_.end(command_);
  const auto start = std::chrono::steady_clock::now();
  device_.submit_wait(command_);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  submission_ms += ms; maximum_submission_ms = std::max(maximum_submission_ms, ms); ++submissions;
  // A completed slow command is safe to release. This is not preemption.
  if (ms > completed_limit_ms_)
    throw std::runtime_error("bounded prefill submission exceeded completed-time limit: " + std::to_string(ms) + " ms");
  // begin() resets the completed command buffer before starting recording.
  device_.begin(command_);
  barrier();
}
void Recorder::dispatch(const std::string &kernel, std::initializer_list<View> bindings, const void *push,
                        uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz) {
  require(bindings.size() <= 8 && push_bytes <= 128 && gx > 0 && gy > 0 && gz > 0 &&
              gx <= device_.max_groups_x_,
          "Vulkan fast dispatch geometry invalid");
  if (pending_ && !concurrent_) barrier();
  const auto &p = device_.pipeline(kernel);
  vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
  std::array<VkDescriptorBufferInfo, 8> infos{};
  std::array<VkWriteDescriptorSet, 8> writes{};
  uint32_t n = 0, binding = 0;
  for (const auto &v : bindings) {
    if (v.buffer) {
      require(v.offset <= v.buffer->size, "Vulkan fast binding offset out of range");
      const size_t remaining = v.buffer->size - v.offset;
      const size_t range = v.range ? v.range : remaining;
      require(range > 0 && range <= remaining && range <= device_.max_storage_range_,
              "Vulkan fast storage binding range invalid or exceeds device limit");
      infos[n] = {v.buffer->handle, v.offset, range};
      writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[n].dstBinding = binding;
      writes[n].descriptorCount = 1;
      writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[n].pBufferInfo = &infos[n];
      ++n;
    }
    ++binding;
  }
  if (n) device_.push_descriptors_(command_, VK_PIPELINE_BIND_POINT_COMPUTE, device_.layout_, 0, n, writes.data());
  if (push_bytes) vkCmdPushConstants(command_, device_.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
  if (bounded_profile_) {
    device_.host_reset_queries(0, 2);
    vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device_.queries_, 0);
  }
  vkCmdDispatch(command_, gx, gy, gz);
  if (bounded_profile_) vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, device_.queries_, 1);
  pending_ = true;
  if (names && next) {
    require(*next < Device::query_capacity, "Vulkan profile timestamp capacity exceeded");
    if (!concurrent_) barrier();
    vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, device_.queries_, (*next)++);
    names->push_back(kernel);
  }
  if (completed_limit_ms_) {
    try {flush();}
    catch (const std::exception &e) {throw std::runtime_error(kernel + ": " + e.what());}
  }
  if (bounded_profile_) {
    const auto times = device_.read_timestamps(0, 2);
    kernel_ms[kernel] += (times[1] - times[0]) / 1e6;
  }
}
void Recorder::begin_concurrent() {
  if (pending_) barrier();
  concurrent_ = true;
}
void Recorder::end_concurrent() {
  concurrent_ = false;
  if (pending_) barrier();
}
void Recorder::barrier() {
  VkMemoryBarrier m{};
  m.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0,
                       nullptr, 0, nullptr);
  pending_ = false;
}
void Recorder::copy(View s, View d, size_t bytes) {
  require(s.offset <= s.buffer->size && bytes <= s.buffer->size - s.offset && d.offset <= d.buffer->size &&
              bytes <= d.buffer->size - d.offset,
          "Vulkan fast copy out of bounds");
  if (pending_ && !concurrent_) barrier();
  VkBufferCopy region{s.offset, d.offset, bytes};
  vkCmdCopyBuffer(command_, s.buffer->handle, d.buffer->handle, 1, &region);
  pending_ = true;
}
void Recorder::fill(View d, size_t bytes, uint32_t value) {
  require(d.offset % 4 == 0 && bytes % 4 == 0 && d.offset <= d.buffer->size && bytes <= d.buffer->size - d.offset,
          "Vulkan fast fill out of bounds");
  if (pending_ && !concurrent_) barrier();
  vkCmdFillBuffer(command_, d.buffer->handle, d.offset, bytes, value);
  pending_ = true;
}
void Recorder::timestamp(uint32_t query) {
  require(query < Device::query_capacity, "Vulkan fast timestamp index out of range");
  if (pending_ && !concurrent_) barrier();
  vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, device_.queries_, query);
}
void Recorder::host_visible() {
  VkMemoryBarrier m{};
  m.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  m.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
  pending_ = false;
}

} // namespace strata::vulkan::fast
