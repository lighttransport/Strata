#include "runtime.hpp"
#include "kda_fused_spv.hpp"
#include "kda_spv.hpp"
#include "mhc_spv.hpp"
#include "mla_spv.hpp"
#include "task_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace strata::vulkan {
namespace {
void ok(VkResult r, const char *op) {
  if (r != VK_SUCCESS)
    throw std::runtime_error(std::string("Vulkan ") + op +
                             " failed: " + std::to_string(r));
}
void require(bool v, const char *msg) {
  if (!v)
    throw std::invalid_argument(msg);
}
} // namespace
Compute::Buffer::Buffer(Compute &c, size_t bytes, bool device_local)
    : owner(&c), size(std::max<size_t>(4, bytes)) {
  require(size <= c.budget_ - c.live_, "Vulkan custom-buffer budget exceeded");
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(c.physical_, &properties);
  require(size <= properties.limits.maxStorageBufferRange,
          "Vulkan storage-buffer range exceeded");
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ok(vkCreateBuffer(c.device_, &info, nullptr, &handle), "create buffer");
  try {
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c.device_, handle, &req);
    allocated = req.size;
    require(allocated <= c.budget_ - c.live_,
            "Vulkan allocation budget exceeded");
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(c.physical_, &props);
    std::vector<std::pair<int, uint32_t>> candidates;
    const auto needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < props.memoryTypeCount; i++)
      if ((req.memoryTypeBits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & needed) == needed) {
        const auto flags = props.memoryTypes[i].propertyFlags;
        const bool local = flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const bool cached = flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        const int score = device_local ? (local ? 4 : 0) + (cached ? 1 : 0)
                                       : (cached ? 4 : 0) + (local ? 0 : 2);
        candidates.emplace_back(score, i);
      }
    require(!candidates.empty(),
            "Vulkan needs host-visible coherent storage buffers");
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](auto a, auto b) { return a.first > b.first; });
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = allocated;
    VkResult allocation = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (auto [score, selected] : candidates) {
      alloc.memoryTypeIndex = selected;
      allocation = vkAllocateMemory(c.device_, &alloc, nullptr, &memory);
      if (allocation == VK_SUCCESS) {
        memory_flags = props.memoryTypes[selected].propertyFlags;
        break;
      }
      memory = VK_NULL_HANDLE;
      // Small visible-VRAM heaps on older cards can fill before total VRAM.
      // Try coherent system memory; a driver/device error must stop the run.
      if (allocation != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
          allocation != VK_ERROR_OUT_OF_HOST_MEMORY)
        break;
    }
    ok(allocation, "allocate buffer");
    ok(vkBindBufferMemory(c.device_, handle, memory, 0), "bind buffer");
    ok(vkMapMemory(c.device_, memory, 0, size, 0, &mapped), "map buffer");
    c.live_ += allocated;
    c.peak_ = std::max(c.peak_, c.live_);
    std::memset(mapped, 0, size);
  } catch (...) {
    if (mapped)
      vkUnmapMemory(c.device_, memory);
    vkDestroyBuffer(c.device_, handle, nullptr);
    if (memory)
      vkFreeMemory(c.device_, memory, nullptr);
    throw;
  }
}
Compute::Buffer::~Buffer() {
  if (mapped)
    vkUnmapMemory(owner->device_, memory);
  if (handle)
    vkDestroyBuffer(owner->device_, handle, nullptr);
  if (memory)
    vkFreeMemory(owner->device_, memory, nullptr);
  owner->live_ -= allocated;
}
void Compute::Buffer::put(const void *p, size_t n, size_t off) {
  require(off <= size && n <= size - off, "Vulkan buffer upload out of bounds");
  if (n)
    std::memcpy(static_cast<char *>(mapped) + off, p, n);
}
void Compute::Buffer::get(void *p, size_t n, size_t off) const {
  require(off <= size && n <= size - off,
          "Vulkan buffer download out of bounds");
  if (n)
    std::memcpy(p, static_cast<const char *>(mapped) + off, n);
}
Compute::Compute(std::string requested, size_t budget) : budget_(budget) {
  try {
    require(!requested.empty() && budget >= 4,
            "Vulkan device name or budget invalid");
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Strata GLM Vulkan";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    ok(vkCreateInstance(&info, nullptr, &instance_), "create instance");
    uint32_t count = 0;
    ok(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
       "enumerate devices");
    std::vector<VkPhysicalDevice> devices(count);
    ok(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
       "enumerate devices");
    unsigned matches = 0;
    for (auto d : devices) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(d, &p);
      if (requested.find(p.deviceName) != std::string::npos ||
          std::string(p.deviceName).find(requested) != std::string::npos) {
        physical_ = d;
        name_ = p.deviceName;
        ++matches;
      }
    }
    require(physical_ != VK_NULL_HANDLE && matches == 1,
            "Vulkan GGML and custom-compute device names do not match");
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_, &properties);
    require(properties.apiVersion >= VK_API_VERSION_1_2,
            "Vulkan 1.2 compute is required");
    require(properties.limits.maxComputeWorkGroupInvocations >= 128 &&
                properties.limits.maxComputeSharedMemorySize >= 2112 * 4,
            "Vulkan compute limits insufficient");
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qs(families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, qs.data());
    bool found = false;
    for (uint32_t i = 0; i < families; i++)
      if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queue_family_ = i;
        found = true;
        break;
      }
    require(found, "Vulkan compute queue unavailable");
    float priority = 1;
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = queue_family_;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    ok(vkCreateDevice(physical_, &di, nullptr, &device_), "create device");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = queue_family_;
    ok(vkCreateCommandPool(device_, &pi, nullptr, &commands_), "command pool");
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = commands_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    ok(vkAllocateCommandBuffers(device_, &ai, &command_), "command buffer");
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ok(vkCreateFence(device_, &fi, nullptr, &fence_), "fence");
    for (auto &b : scratch_)
      b = std::make_unique<Buffer>(*this, 4);
    std::cerr << "VULKAN_CUSTOM device=" << name_
              << " budget_MiB=" << budget_ / 1048576
              << " staging_memory_flags=" << scratch_[0]->memory_flags << '\n';
  } catch (...) {
    cleanup();
    throw;
  }
}
Compute::~Compute() { cleanup(); }
void Compute::cleanup() noexcept {
  conv_.clear();
  kda_weights_.clear();
  state_.clear();
  cache_.clear();
  for (auto &b : scratch_)
    b.reset();
  for (auto &[name, k] : kernels_)
    destroy_kernel(k);
  if (fence_)
    vkDestroyFence(device_, fence_, nullptr);
  if (commands_)
    vkDestroyCommandPool(device_, commands_, nullptr);
  if (device_)
    vkDestroyDevice(device_, nullptr);
  if (instance_)
    vkDestroyInstance(instance_, nullptr);
}
Compute::Kernel &Compute::kernel(const std::string &name) {
  auto found = kernels_.find(name);
  if (found != kernels_.end())
    return found->second;
  Kernel k;
  try {
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (uint32_t i = 0; i < 8; i++)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 8;
    dl.pBindings = bindings.data();
    ok(vkCreateDescriptorSetLayout(device_, &dl, nullptr, &k.descriptors),
       "descriptor layout");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1;
    li.pSetLayouts = &k.descriptors;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &push;
    ok(vkCreatePipelineLayout(device_, &li, nullptr, &k.layout),
       "pipeline layout");
    std::span<const unsigned char> bytes;
    if (name == "kda")
      bytes = kda_spv;
    else if (name == "kda_fused")
      bytes = kda_fused_spv;
    else if (name == "mla")
      bytes = mla_spv;
    else if (name == "mhc")
      bytes = mhc_spv;
    else
      throw std::invalid_argument("Unknown Vulkan shader");
    require(!bytes.empty() && bytes.size() % 4 == 0,
            "Vulkan shader size invalid");
    std::vector<uint32_t> code(bytes.size() / 4);
    std::memcpy(code.data(), bytes.data(), bytes.size());
    VkShaderModuleCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = bytes.size();
    si.pCode = code.data();
    VkShaderModule shader;
    ok(vkCreateShaderModule(device_, &si, nullptr, &shader), "shader module");
    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = {};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shader;
    ci.stage.pName = "main";
    ci.layout = k.layout;
    auto result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci,
                                           nullptr, &k.pipeline);
    vkDestroyShaderModule(device_, shader, nullptr);
    ok(result, "compute pipeline");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo pc{};
    pc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pc.maxSets = 1;
    pc.poolSizeCount = 1;
    pc.pPoolSizes = &ps;
    ok(vkCreateDescriptorPool(device_, &pc, nullptr, &k.pool),
       "descriptor pool");
    VkDescriptorSetAllocateInfo sa{};
    sa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sa.descriptorPool = k.pool;
    sa.descriptorSetCount = 1;
    sa.pSetLayouts = &k.descriptors;
    ok(vkAllocateDescriptorSets(device_, &sa, &k.set), "descriptor set");
    return kernels_.emplace(name, k).first->second;
  } catch (...) {
    destroy_kernel(k);
    throw;
  }
}
void Compute::destroy_kernel(Kernel &k) noexcept {
  if (k.pipeline)
    vkDestroyPipeline(device_, k.pipeline, nullptr);
  if (k.layout)
    vkDestroyPipelineLayout(device_, k.layout, nullptr);
  if (k.pool)
    vkDestroyDescriptorPool(device_, k.pool, nullptr);
  if (k.descriptors)
    vkDestroyDescriptorSetLayout(device_, k.descriptors, nullptr);
}
Compute::Buffer &Compute::scratch(int i, size_t bytes) {
  if (scratch_[i]->size < std::max<size_t>(4, bytes))
    scratch_[i] = std::make_unique<Buffer>(*this, bytes);
  return *scratch_[i];
}
void Compute::dispatch(const std::string &name,
                       const std::array<Buffer *, 8> &buffers,
                       const void *params, uint32_t groups) {
  const auto started = std::chrono::steady_clock::now();
  auto &k = kernel(name);
  std::array<VkDescriptorBufferInfo, 8> bi{};
  std::array<VkWriteDescriptorSet, 8> wi{};
  for (uint32_t i = 0; i < 8; i++) {
    auto *b = buffers[i] ? buffers[i] : scratch_[i].get();
    bi[i] = {b->handle, 0, b->size};
    wi[i] = {};
    wi[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wi[i].dstSet = k.set;
    wi[i].dstBinding = i;
    wi[i].descriptorCount = 1;
    wi[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wi[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(device_, 8, wi.data(), 0, nullptr);
  ok(vkResetCommandBuffer(command_, 0), "reset command");
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  ok(vkBeginCommandBuffer(command_, &begin), "begin command");
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command_,
                       VK_PIPELINE_STAGE_HOST_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
  vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0,
                          1, &k.set, 0, nullptr);
  vkCmdPushConstants(command_, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                     params);
  vkCmdDispatch(command_, groups, 1, 1);
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);
  ok(vkEndCommandBuffer(command_), "end command");
  ok(vkResetFences(device_, 1, &fence_), "reset fence");
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_;
  operation_trace("kernel_submit_begin", name);
  const auto submitted = vkQueueSubmit(queue_, 1, &submit, fence_);
  if (submitted != VK_SUCCESS) {
    std::cerr << "VULKAN_ERROR queue submit failed: " << submitted << '\n';
    std::_Exit(74);
  }
  const auto waited =
      vkWaitForFences(device_, 1, &fence_, VK_TRUE, 60ULL * 1000000000);
  if (waited != VK_SUCCESS) {
    std::cerr << "VULKAN_ERROR wait fence failed: " << waited
              << "; exiting without freeing in-flight buffers\n";
    std::_Exit(74);
  }
  operation_trace("kernel_complete", name);
  dispatch_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - started)
                     .count();
}
void Compute::kda(int layer, std::span<const float> q,
                  std::span<const float> key, std::span<const float> v,
                  std::span<const float> g, std::span<const float> beta,
                  std::span<float> y, int heads, int dim) {
  require(heads > 0 && heads <= 256 && dim == 128,
          "Vulkan KDA requires positive heads and dimension 128");
  size_t n = size_t(heads) * dim;
  require(q.size() == n && key.size() == n && v.size() == n && g.size() == n &&
              beta.size() == size_t(heads) && y.size() == n,
          "Vulkan KDA buffer geometry invalid");
  auto &s = state_[layer];
  if (!s)
    s = std::make_unique<Buffer>(*this, n * dim * 4, true);
  require(s->size == n * dim * 4, "Vulkan KDA state geometry changed");
  std::array<Buffer *, 8> b{};
  b[0] = s.get();
  std::array<std::span<const float>, 5> inputs{q, key, v, g, beta};
  for (int i = 0; i < 5; i++) {
    b[i + 1] = &scratch(i + 1, inputs[i].size_bytes());
    b[i + 1]->put(inputs[i].data(), inputs[i].size_bytes());
  }
  b[6] = &scratch(6, y.size_bytes());
  std::array<uint32_t, 4> params{uint32_t(heads), uint32_t(dim), 0, 0};
  dispatch("kda", b, params.data(), heads);
  b[6]->get(y.data(), y.size_bytes());
}
void Compute::kda_fused(int layer, std::span<const float> raw,
                        std::span<const float> decay,
                        std::span<const float> beta,
                        std::span<const float> gate,
                        std::span<const float> weights,
                        std::span<const float> bias, std::span<const float> a,
                        std::span<const float> norm, std::span<float> y,
                        int heads, int kernel_size, float lower, float eps) {
  require(layer >= 0 && heads > 0 && heads <= 256 && kernel_size >= 1 &&
              kernel_size <= 32 && std::isfinite(lower) && lower <= 0 &&
              std::isfinite(eps) && eps > 0,
          "Vulkan fused KDA dimensions invalid");
  const size_t n = size_t(heads) * 128;
  require(raw.size() == 3 * n && decay.size() == n &&
              beta.size() == size_t(heads) && gate.size() == n &&
              weights.size() == 3 * n * kernel_size && bias.size() == n &&
              a.size() == size_t(heads) && norm.size() == 128 && y.size() == n,
          "Vulkan fused KDA buffer geometry invalid");
  auto &state = state_[layer];
  if (!state)
    state = std::make_unique<Buffer>(*this, n * 128 * sizeof(float), true);
  auto &history = conv_[layer];
  const size_t history_bytes =
      std::max(size_t(1), 3 * n * (kernel_size - 1)) * sizeof(float);
  if (!history)
    history = std::make_unique<Buffer>(*this, history_bytes, true);
  auto &params = kda_weights_[layer];
  const size_t params_bytes =
      (weights.size() + bias.size() + a.size() + norm.size()) * sizeof(float);
  if (!params) {
    std::vector<float> packed;
    packed.reserve(params_bytes / sizeof(float));
    for (auto input : {weights, bias, a, norm})
      packed.insert(packed.end(), input.begin(), input.end());
    params = std::make_unique<Buffer>(*this, params_bytes, true);
    params->put(packed.data(), params_bytes);
  }
  require(state->size == n * 128 * sizeof(float) &&
              history->size == history_bytes && params->size == params_bytes,
          "Vulkan fused KDA cached geometry changed");
  std::array<Buffer *, 8> b{};
  b[0] = state.get();
  b[5] = params.get();
  b[7] = history.get();
  std::array<std::span<const float>, 4> inputs{raw, decay, beta, gate};
  for (int i = 0; i < 4; ++i) {
    b[i + 1] = &scratch(i + 1, inputs[i].size_bytes());
    b[i + 1]->put(inputs[i].data(), inputs[i].size_bytes());
  }
  b[6] = &scratch(6, y.size_bytes());
  struct {
    uint32_t heads, kernel;
    float lower, eps;
  } push{uint32_t(heads), uint32_t(kernel_size), lower, eps};
  dispatch("kda_fused", b, &push, heads);
  b[6]->get(y.data(), y.size_bytes());
}
std::vector<float> Compute::kda_history(int layer) const {
  const auto &b = *conv_.at(layer);
  std::vector<float> out(b.size / sizeof(float));
  b.get(out.data(), b.size);
  return out;
}
void Compute::mla(int layer, std::span<const float> row, int position,
                  int capacity, std::span<const float> q,
                  std::span<const int> ids, std::span<float> y, int heads,
                  float scale) {
  require(capacity > 0 && position >= 0 && position < capacity && heads > 0 &&
              !row.empty() && row.size() <= 4096,
          "Vulkan MLA geometry invalid");
  require(q.size() == row.size() * heads && y.size() == q.size() &&
              !ids.empty() && ids.size() <= 2112,
          "Vulkan MLA query/selection invalid");
  for (int i : ids)
    require(i >= 0 && i <= position, "Vulkan MLA selected row out of bounds");
  auto &cache = cache_[layer];
  if (!cache) {
    require(position == 0, "Vulkan MLA cache must begin at position zero");
    cache = std::make_unique<Buffer>(*this, size_t(capacity) * row.size_bytes(),
                                     true);
    positions_[layer] = 0;
  }
  require(position == positions_[layer] &&
              cache->size == size_t(capacity) * row.size_bytes(),
          "Vulkan MLA cache append order changed");
  cache->put(row.data(), row.size_bytes(), size_t(position) * row.size_bytes());
  positions_[layer]++;
  std::array<Buffer *, 8> b{};
  b[0] = cache.get();
  b[1] = &scratch(1, q.size_bytes());
  b[1]->put(q.data(), q.size_bytes());
  b[2] = &scratch(2, ids.size_bytes());
  b[2]->put(ids.data(), ids.size_bytes());
  b[6] = &scratch(6, y.size_bytes());
  struct Params {
    uint32_t heads, latent, count;
    float scale;
  } params{uint32_t(heads), uint32_t(row.size()), uint32_t(ids.size()), scale};
  dispatch("mla", b, &params, heads);
  b[6]->get(y.data(), y.size_bytes());
}
void Compute::mhc_read(std::span<const float> r, std::span<const float> p,
                       std::span<const float> base,
                       std::span<const float> scale, std::span<float> c,
                       std::span<float> x, int iterations, float eps) {
  require(!x.empty() && r.size() == 4 * x.size() && p.size() == 24 &&
              base.size() == 24 && scale.size() == 3 && c.size() == 24 &&
              iterations > 0 && iterations <= 128,
          "Vulkan mHC read geometry invalid");
  std::array<float, 51> packed{};
  std::copy(p.begin(), p.end(), packed.begin());
  std::copy(base.begin(), base.end(), packed.begin() + 24);
  std::copy(scale.begin(), scale.end(), packed.begin() + 48);
  std::array<Buffer *, 8> b{};
  b[0] = &scratch(0, r.size_bytes());
  b[0]->put(r.data(), r.size_bytes());
  b[1] = &scratch(1, sizeof(packed));
  b[1]->put(packed.data(), sizeof(packed));
  b[2] = &scratch(2, c.size_bytes());
  b[6] = &scratch(6, x.size_bytes());
  struct Params {
    uint32_t width, iterations;
    float eps;
    uint32_t mode;
  } params{uint32_t(x.size()), uint32_t(iterations), eps, 0};
  dispatch("mhc", b, &params, 1);
  b[2]->get(c.data(), c.size_bytes());
  b[6]->get(x.data(), x.size_bytes());
}
void Compute::mhc_write(std::span<const float> r, std::span<const float> c,
                        std::span<const float> y, std::span<float> out) {
  require(!y.empty() && r.size() == 4 * y.size() && out.size() == r.size() &&
              c.size() == 24,
          "Vulkan mHC write geometry invalid");
  std::array<Buffer *, 8> b{};
  b[0] = &scratch(0, r.size_bytes());
  b[0]->put(r.data(), r.size_bytes());
  b[2] = &scratch(2, c.size_bytes());
  b[2]->put(c.data(), c.size_bytes());
  b[3] = &scratch(3, y.size_bytes());
  b[3]->put(y.data(), y.size_bytes());
  b[6] = &scratch(6, out.size_bytes());
  struct Params {
    uint32_t width, iterations;
    float eps;
    uint32_t mode;
  } params{uint32_t(y.size()), 0, 0, 1};
  dispatch("mhc", b, &params, 1);
  b[6]->get(out.data(), out.size_bytes());
}
std::vector<float> Compute::kda_state(int layer) const {
  auto &b = *state_.at(layer);
  std::vector<float> out(b.size / 4);
  b.get(out.data(), b.size);
  return out;
}
void Compute::reset() {
  for (auto &[i, b] : conv_)
    std::memset(b->mapped, 0, b->size);
  for (auto &[i, b] : state_)
    std::memset(b->mapped, 0, b->size);
  for (auto &[i, b] : cache_)
    std::memset(b->mapped, 0, b->size);
  for (auto &[i, pos] : positions_)
    pos = 0;
}
} // namespace strata::vulkan
