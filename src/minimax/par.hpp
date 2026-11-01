// src/minimax/par.hpp - a small persistent worker pool for the MiniMax CPU expert lane (and its benchmark).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

/// A small persistent worker pool: run(n, fn) calls fn(0..n-1) on the workers and the caller and returns when all are
/// done.  A claim carries its batch number, so a worker that wakes late for batch g can never take an index of g + 1.
class Par {
public:
    /// `cpus` (optional): logical CPUs to pin worker i to (the calling thread is pinned to cpus[workers] when given).
    explicit Par(int workers, const std::vector<int>& cpus = {}) {
        for (int i = 0; i < workers; ++i) {
            threads_.emplace_back([this] { loop(); });
#if defined(__linux__)
            if ((size_t) i < cpus.size()) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpus[(size_t) i], &set);
                pthread_setaffinity_np(threads_.back().native_handle(), sizeof set, &set);
            }
#endif
        }
#if defined(__linux__)
        if ((size_t) workers < cpus.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[(size_t) workers], &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        }
#endif
    }
    ~Par() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    int size() const { return (int) threads_.size() + 1; }
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        uint64_t g;
        {
            std::lock_guard<std::mutex> lk(mu_);
            g = ++gen_;
            job_ = &fn;
            n_ = n;
            head_.store(g << 32);
            remaining_.store(n);
        }
        cv_.notify_all();
        work(g, n, fn);
        while (remaining_.load(std::memory_order_acquire) > 0) std::this_thread::yield();
    }

private:
    int claim(uint64_t g, int n) {
        uint64_t cur = head_.load();
        for (;;) {
            if ((cur >> 32) != g || (int) (cur & 0xffffffffu) >= n) return -1;
            if (head_.compare_exchange_weak(cur, cur + 1)) return (int) (cur & 0xffffffffu);
        }
    }
    void work(uint64_t g, int n, const std::function<void(int)>& fn) {
        for (int i; (i = claim(g, n)) >= 0;) {
            fn(i);
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* fn;
            int n;
            uint64_t g;
            {
                // a short spin before sleeping: the two phases of a layer come within microseconds of each other
                for (int spin = 0; spin < 4000 && gen_atomic() == seen; ++spin) std::this_thread::yield();
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                g = seen = gen_;
                fn = job_;
                n = n_;
            }
            work(g, n, *fn);
        }
    }
    uint64_t gen_atomic() { return head_.load(std::memory_order_relaxed) >> 32; }

    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t gen_ = 0;
    const std::function<void(int)>* job_ = nullptr;
    int n_ = 0;
    bool stop_ = false;
    std::atomic<uint64_t> head_{0};
    std::atomic<int> remaining_{0};
};
