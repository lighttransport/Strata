#pragma once
// Immutable job indices; one reader owns each job, consumers acquire its bytes.
// The caller owns job data until destruction has joined every reader.
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
namespace strata::cpu_runner {
class RowPrefetch {
    size_t count_;
    std::unique_ptr<std::atomic<unsigned char>[]> ready_;
    std::atomic<size_t> next_{0}, completed_{0};
    std::atomic<bool> failed_{false};
    std::function<bool(size_t)> read_;
    std::function<bool(int)> setup_;
    std::vector<std::thread> workers_;
    void join() noexcept { for (auto &t : workers_) if (t.joinable()) t.join(); }
    void worker(int worker_id) noexcept {
        try {
            if (!setup_(worker_id)) { failed_.store(true, std::memory_order_release); return; }
            while (!failed_.load(std::memory_order_acquire)) {
                const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
                if (i >= count_) return;
                if (!read_(i)) { failed_.store(true, std::memory_order_release); return; }
                ready_[i].store(1, std::memory_order_release);
                completed_.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) { failed_.store(true, std::memory_order_release); }
    }
public:
    RowPrefetch(size_t count, int workers, std::function<bool(size_t)> read,
                std::function<bool(int)> setup)
        : count_(count), ready_(count ? new std::atomic<unsigned char>[count] : nullptr),
          read_(std::move(read)), setup_(std::move(setup)) {
        if (workers < 1 || workers > 16 || !read_ || !setup_) throw std::invalid_argument("row prefetch configuration");
        for (size_t i = 0; i < count_; ++i) ready_[i].store(0, std::memory_order_relaxed);
        try {
            for (int i = 0; i < workers; ++i) workers_.emplace_back([this, i] { worker(i); });
        } catch (...) {
            failed_.store(true, std::memory_order_release);
            join();
            throw;
        }
    }
    RowPrefetch(const RowPrefetch &) = delete;
    RowPrefetch &operator=(const RowPrefetch &) = delete;
    ~RowPrefetch() { join(); }
    bool wait(size_t i) const noexcept {
        if (i >= count_) return false;
        while (!ready_[i].load(std::memory_order_acquire)) {
            if (failed_.load(std::memory_order_acquire)) return false;
#if defined(__aarch64__)
            __asm__ volatile("yield");
#elif defined(__x86_64__)
            __asm__ volatile("pause");
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
        }
        return true;
    }
    // Call finish before releasing any storage or reading final completion counts.
    bool finish() noexcept { join(); return !failed_.load(std::memory_order_acquire) && completed() == count_; }
    size_t completed() const noexcept { return completed_.load(std::memory_order_relaxed); }
};
}
