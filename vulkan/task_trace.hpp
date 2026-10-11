#pragma once
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif
namespace strata::vulkan {
inline thread_local int operation_position = -1, operation_layer = -1;
inline void set_operation_context(int position, int layer) noexcept {
  operation_position = position;
  operation_layer = layer;
}
// These markers bracket already-synchronous calls. No GPU work or file I/O
// is added when tracing is unset; names identify the projection or shader.
inline void operation_trace(const char *phase, const std::string &operation) {
  const char *path = std::getenv("STRATA_GLM_TASK_TRACE");
  if (!path || !*path)
    return;
  const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  std::ofstream out(path, std::ios::app);
  if (!out)
    throw std::runtime_error("Vulkan operation trace unavailable");
  out << "{\"unix_ms\":" << time << ",\"phase\":" << std::quoted(phase)
      << ",\"operation\":" << std::quoted(operation)
      << ",\"position\":" << operation_position
      << ",\"layer\":" << operation_layer << "}\n";
  out.flush();
  if (!out)
    throw std::runtime_error("Vulkan operation trace write failed");
#ifdef __linux__
  const int fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0 || fsync(fd)) {
    if (fd >= 0)
      close(fd);
    throw std::runtime_error("Vulkan operation trace sync failed");
  }
  close(fd);
#endif
}
} // namespace strata::vulkan
