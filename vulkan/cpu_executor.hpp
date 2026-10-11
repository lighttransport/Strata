#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
namespace strata::vulkan {
// One layer at a time. The caller keeps its captured storage alive through
// get().
class CpuExecutor {
  std::mutex mutex_;
  std::condition_variable ready_;
  std::function<void()> task_;
  bool stop_ = false;
  std::thread thread_;

public:
  explicit CpuExecutor(int cpu)
      : thread_([this, cpu] {
          auto affinity = strata::kernels::cpu::pin_current_thread(cpu);
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock lock(mutex_);
              ready_.wait(lock, [&] { return stop_ || bool(task_); });
              if (!task_)
                break;
              task = std::move(task_);
              task_ = {};
            }
            task();
          }
          strata::kernels::cpu::restore_thread_affinity(affinity);
        }) {}
  ~CpuExecutor() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    ready_.notify_one();
    thread_.join();
  }
  CpuExecutor(const CpuExecutor &) = delete;
  CpuExecutor &operator=(const CpuExecutor &) = delete;
  template <class F> std::future<void> submit(F &&function) {
    auto task =
        std::make_shared<std::packaged_task<void()>>(std::forward<F>(function));
    auto result = task->get_future();
    {
      std::lock_guard lock(mutex_);
      if (stop_ || task_)
        throw std::logic_error("CPU coordinator already has a queued layer");
      task_ = [task] { (*task)(); };
    }
    ready_.notify_one();
    return result;
  }
};
} // namespace strata::vulkan
