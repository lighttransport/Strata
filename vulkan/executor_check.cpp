#include "cpu_executor.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
int main() {
  try {
    std::atomic<int> completed{0};
    {
      strata::vulkan::CpuExecutor executor(-1);
      executor.submit([&] { completed.fetch_add(1); }).get();
      bool propagated = false;
      try {
        executor.submit([] { throw std::runtime_error("expected"); }).get();
      } catch (const std::runtime_error &) {
        propagated = true;
      }
      if (!propagated)
        throw std::runtime_error("CPU exception was lost");
      executor.submit([&] { completed.fetch_add(1); }).get();
      // Destroy without consuming the final future: queued storage must finish
      // before the caller's captured object can leave its scope.
      executor.submit([&] { completed.fetch_add(1); });
    }
    if (completed.load() != 3)
      throw std::runtime_error("CPU coordinator failed to join queued work");
    std::cout << "CPU coordinator lifetime and exception checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
