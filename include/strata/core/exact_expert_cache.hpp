#pragma once
// Linux prototype: immutable expert bytes, direct I/O, bounded exclusive CPU tier.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::core {
class ExactExpertCache {
public:
    enum class Policy { lru, lfu };
    struct Span { std::string path; uint64_t offset=0; size_t bytes=0; };
    using Source = std::array<Span,3>;
    struct Entry {
        std::array<std::unique_ptr<unsigned char, decltype(&std::free)>,3> storage{{{nullptr,std::free},{nullptr,std::free},{nullptr,std::free}}};
        std::array<size_t,3> displacement{};
        size_t bytes=0;
        // Published by wait(), not by the readers. A begin() hit is safe to compute
        // while this batch's misses load; new entries cannot be consumed before wait().
        bool ready=false;
        std::shared_ptr<std::atomic<size_t>> allocated;
        ~Entry() { if (allocated) allocated->fetch_sub(bytes); }
        const unsigned char* data(int i) const { return storage[i].get()+displacement[i]; }
    };
    struct Stats { uint64_t hits=0, misses=0, evictions=0, read_bytes=0; double wait_ms=0; } stats;
    explicit ExactExpertCache(size_t budget, unsigned readers=4, Policy policy=Policy::lru, int group_size=0)
        : budget_(budget), policy_(policy), group_size_(group_size) {
#ifndef __linux__
        throw std::runtime_error("exact expert cache requires Linux O_DIRECT");
#else
        if (!budget || readers<1 || readers>4 || group_size<0) throw std::invalid_argument("exact cache budget/readers/groups");
        try { for(unsigned i=0;i<readers;++i) workers_.emplace_back([this]{ worker(); }); }
        catch (...) { stop(); throw; }
#endif
    }
    ~ExactExpertCache() {
        // A caller may release a batch without waiting. Readers still own its memory.
        try { wait(); } catch (...) {}
        stop();
#ifdef __linux__
        for(auto& [path,fd]: files_) ::close(fd);
#endif
    }
    ExactExpertCache(const ExactExpertCache&)=delete;
    size_t resident_bytes() const { return allocated_->load(); }
    size_t budget_bytes() const { return budget_; }
    static size_t allocation_bytes(const Source& source) {
        size_t bytes=0;
        for (const auto& span:source) {
            if (!span.bytes || span.bytes>64*1024*1024 || span.offset>UINT64_MAX-span.bytes-4095)
                throw std::runtime_error("exact cache invalid source span");
            bytes+=(span.offset%4096+span.bytes+4095)/4096*4096;
        }
        return bytes;
    }
    // A bounded training prior (in expected accesses), not a permanent pin.
    void seed_heat(int key, double heat) {
        if (key<0 || !std::isfinite(heat) || heat<0) throw std::invalid_argument("exact cache heat");
        auto found=entries_.find(key);
        if (policy_==Policy::lfu && found!=entries_.end()) frequency_.erase({heat_[key],found->second.tick,key});
        heat_[key]=heat;
        if (policy_==Policy::lfu && found!=entries_.end()) frequency_.emplace(heat,found->second.tick,key);
    }
    // Single service-thread API. Returned leases pin all current jobs until CPU execution finishes.
    std::vector<std::shared_ptr<Entry>> acquire(const std::vector<std::pair<int,Source>>& requests) {
        auto result=begin(requests);
        wait();
        return result;
    }
    // One outstanding batch. Previously ready entries may be used immediately;
    // new entries must wait(). Outstanding I/O retains its own leases.
    std::vector<std::shared_ptr<Entry>> begin(const std::vector<std::pair<int,Source>>& requests) try {
        wait();
        std::vector<std::shared_ptr<Entry>> result;
        std::vector<Read> reads;
        for(const auto& [key,source]: requests) {
            if (key<0) throw std::invalid_argument("exact cache key");
            if (policy_==Policy::lfu) touch_heat(key);
            auto found=entries_.find(key);
            if(found!=entries_.end()) {
                ++stats.hits; lru_.splice(lru_.begin(),lru_,found->second.order);
                result.push_back(found->second.entry); continue;
            }
            auto entry=std::make_shared<Entry>();
            std::array<size_t,3> lengths{};
            entry->bytes=allocation_bytes(source);
            for(int i=0;i<3;++i) {
                entry->displacement[i]=source[i].offset%4096;
                lengths[i]=(entry->displacement[i]+source[i].bytes+4095)/4096*4096;
            }
            if(entry->bytes>budget_) throw std::runtime_error("exact cache budget smaller than one expert");
            while(allocated_->load()>budget_-entry->bytes) {
                auto victim=lru_.end();
                if (policy_==Policy::lru) {
                    for(auto it=lru_.end();it!=lru_.begin();) {
                        --it;
                        if(entries_.at(*it).entry.use_count()==1) { victim=it; break; }
                    }
                } else {
                    for (const auto& [heat,tick,k]:frequency_) {
                        (void)heat; (void)tick;
                        auto v=entries_.find(k);
                        if(v->second.entry.use_count()==1) { victim=v->second.order; break; }
                    }
                }
                if(victim==lru_.end()) throw std::runtime_error("exact cache budget smaller than active expert leases");
                auto v=entries_.find(*victim); used_-=v->second.entry->bytes;
                if (policy_==Policy::lfu) frequency_.erase({heat_[*victim],v->second.tick,*victim});
                entries_.erase(v); lru_.erase(victim); ++stats.evictions;
            }
            entry->allocated=allocated_;
            allocated_->fetch_add(entry->bytes);
#ifdef __linux__
            for(int i=0;i<3;++i) {
                void* p=nullptr;
                if(posix_memalign(&p,4096,lengths[i])) throw std::bad_alloc();
                entry->storage[i].reset(static_cast<unsigned char*>(p));
                int fd;
                auto file=files_.find(source[i].path);
                if(file==files_.end()) {
                    fd=::open(source[i].path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
                    if(fd<0) throw std::runtime_error("exact cache O_DIRECT open: "+source[i].path+": "+std::strerror(errno));
                    files_.emplace(source[i].path,fd);
                } else fd=file->second;
                reads.push_back({fd,p,lengths[i],source[i].offset-entry->displacement[i]});
                stats.read_bytes+=lengths[i];
            }
#endif
            lru_.push_front(key);
            const uint64_t tick=++clock_;
            entries_.emplace(key,Record{entry,lru_.begin(),tick}); used_+=entry->bytes;
            if (policy_==Policy::lfu) frequency_.emplace(heat_[key],tick,key);
            ++stats.misses; result.push_back(std::move(entry));
        }
        if(!reads.empty()) {
            pending_entries_=result;
            std::unique_lock lock(mutex_);
            reads_=std::move(reads); next_=0; pending_=reads_.size(); failure_=nullptr;
            inflight_=true;
            ready_.notify_all();
        }
        return result;
    } catch (...) {
        entries_.clear(); lru_.clear(); frequency_.clear(); used_=0;
        throw;
    }
    void wait() {
        if (!inflight_) return;
        const auto start=std::chrono::steady_clock::now();
        std::unique_lock lock(mutex_);
        done_.wait(lock,[this]{return pending_==0;});
        reads_.clear();
        inflight_=false;
        if (!failure_) for (auto& e:pending_entries_) e->ready=true;
        pending_entries_.clear();
        stats.wait_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if (failure_) {
            entries_.clear(); lru_.clear(); frequency_.clear(); used_=0;
            std::rethrow_exception(failure_);
        }
    }
private:
    struct Record { std::shared_ptr<Entry> entry; std::list<int>::iterator order; uint64_t tick; };
    struct Read { int fd; void* dest; size_t bytes; uint64_t offset; };
    size_t budget_,used_=0;
    Policy policy_;
    int group_size_;
    std::map<int,double> heat_;
    std::map<int,uint64_t> group_accesses_;
    uint64_t clock_=0;
    std::set<std::tuple<double,uint64_t,int>> frequency_;
    std::shared_ptr<std::atomic<size_t>> allocated_=std::make_shared<std::atomic<size_t>>(0);
    std::map<int,Record> entries_;
    std::list<int> lru_;
    std::map<std::string,int> files_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_,done_;
    std::vector<Read> reads_;
    size_t next_=0,pending_=0;
    bool inflight_=false;
    std::vector<std::shared_ptr<Entry>> pending_entries_;
    bool stopping_=false;
    std::exception_ptr failure_;
    void touch_heat(int key) {
        seed_heat(key,heat_[key]+1);
        auto found=entries_.find(key);
        if (found!=entries_.end()) {
            frequency_.erase({heat_[key],found->second.tick,key});
            found->second.tick=++clock_;
            frequency_.emplace(heat_[key],found->second.tick,key);
        }
        const int group=group_size_ ? key/group_size_ : 0;
        if (++group_accesses_[group]%240) return;
        auto first=group_size_ ? heat_.lower_bound(group*group_size_) : heat_.begin();
        for (auto it=first; it!=heat_.end() && (!group_size_ || it->first/group_size_==group); ++it)
            seed_heat(it->first,it->second*0.97);
    }
    void stop() {
        { std::lock_guard lock(mutex_); stopping_=true; }
        ready_.notify_all(); for(auto& t:workers_) if(t.joinable())t.join();
    }
    void worker() {
#ifdef __linux__
        for(;;) {
            Read read;
            { std::unique_lock lock(mutex_); ready_.wait(lock,[this]{return stopping_ || next_<reads_.size();});
              if(stopping_)return;
              read=reads_[next_++]; }
            std::exception_ptr error;
            try {
                ssize_t n;
                do { n=::pread(read.fd,read.dest,read.bytes,read.offset); } while(n<0 && errno==EINTR);
                if(n!=static_cast<ssize_t>(read.bytes)) throw std::runtime_error("exact cache direct read failed/short (no buffered fallback)");
            } catch(...) { error=std::current_exception(); }
            { std::lock_guard lock(mutex_); if(error && !failure_)failure_=error; if(!--pending_)done_.notify_one(); }
        }
#endif
    }
};
}
