#pragma once
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include <stdexcept>
// Opt-in bounded FP4 tree runner: one serialized owner per transport context.
class ServiceLane {
    using Task=std::packaged_task<std::vector<float>()>;
    std::mutex mutex_;
    std::condition_variable wake_;
    Task pending_;
    bool stop_=false;
    std::thread worker_;
public:
    ServiceLane():worker_([this] {
        for(;;) {
            Task task;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock,[this]{return stop_||pending_.valid();});
                if(stop_&&!pending_.valid())return;
                task=std::move(pending_);
            }
            task(); // Exceptions are carried by the packaged task's future.
        }
    }) {}
    ~ServiceLane() {
        {std::lock_guard lock(mutex_);stop_=true;}
        wake_.notify_one();worker_.join();
    }
    ServiceLane(const ServiceLane&)=delete;
    ServiceLane& operator=(const ServiceLane&)=delete;
    template<class F> std::future<std::vector<float>> submit(F fn) {
        Task task(std::move(fn));auto future=task.get_future();
        {std::lock_guard lock(mutex_);
         if(stop_||pending_.valid())throw std::logic_error("service lane already queued or stopped");
         pending_=std::move(task);}
        wake_.notify_one();return future;
    }
};
