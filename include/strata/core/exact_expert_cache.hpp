#pragma once
// Linux prototype: immutable expert bytes, demand-only I/O, bounded exclusive CPU tier.
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
    struct Span { std::string path; uint64_t offset=0; size_t bytes=0; };
    using Source = std::array<Span,3>;
    struct Entry {
        std::array<std::unique_ptr<unsigned char, decltype(&std::free)>,3> storage{{{nullptr,std::free},{nullptr,std::free},{nullptr,std::free}}};
        std::array<size_t,3> displacement{};
        size_t bytes=0;
        std::shared_ptr<std::atomic<size_t>> allocated;
        ~Entry() { if (allocated) allocated->fetch_sub(bytes); }
        const unsigned char* data(int i) const { return storage[i].get()+displacement[i]; }
    };
    struct Stats { uint64_t hits=0, misses=0, evictions=0, read_bytes=0; double wait_ms=0; } stats;
    explicit ExactExpertCache(size_t budget, unsigned readers=4): budget_(budget) {
#ifndef __linux__
        throw std::runtime_error("exact expert cache requires Linux O_DIRECT");
#else
        if (!budget || readers<1 || readers>4) throw std::invalid_argument("exact cache budget/readers");
        try { for(unsigned i=0;i<readers;++i) workers_.emplace_back([this]{ worker(); }); }
        catch (...) { stop(); throw; }
#endif
    }
    ~ExactExpertCache() { stop();
#ifdef __linux__
        for(auto& [path,fd]: files_) ::close(fd);
#endif
    }
    ExactExpertCache(const ExactExpertCache&)=delete;
    size_t resident_bytes() const { return allocated_->load(); }
    // Single service-thread API. Returned leases pin all current jobs until CPU execution finishes.
    std::vector<std::shared_ptr<Entry>> acquire(const std::vector<std::pair<int,Source>>& requests) try {
        auto start=std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<Entry>> result;
        std::vector<Read> reads;
        for(const auto& [key,source]: requests) {
            auto found=entries_.find(key);
            if(found!=entries_.end()) {
                ++stats.hits; lru_.splice(lru_.begin(),lru_,found->second.order);
                result.push_back(found->second.entry); continue;
            }
            auto entry=std::make_shared<Entry>();
            std::array<size_t,3> lengths{};
            for(int i=0;i<3;++i) {
                if(!source[i].bytes || source[i].bytes>64*1024*1024 || source[i].offset>UINT64_MAX-source[i].bytes-4095)
                    throw std::runtime_error("exact cache invalid source span");
                entry->displacement[i]=source[i].offset%4096;
                lengths[i]=(entry->displacement[i]+source[i].bytes+4095)/4096*4096;
                entry->bytes+=lengths[i];
            }
            if(entry->bytes>budget_) throw std::runtime_error("exact cache budget smaller than one expert");
            while(allocated_->load()>budget_-entry->bytes) {
                auto victim=lru_.end();
                for(auto it=lru_.end();it!=lru_.begin();) {
                    --it; if(entries_.at(*it).entry.use_count()==1) { victim=it; break; }
                }
                if(victim==lru_.end()) throw std::runtime_error("exact cache budget smaller than active expert leases");
                auto v=entries_.find(*victim); used_-=v->second.entry->bytes;
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
            lru_.push_front(key); entries_.emplace(key,Record{entry,lru_.begin()}); used_+=entry->bytes;
            ++stats.misses; result.push_back(std::move(entry));
        }
        if(!reads.empty()) {
            std::unique_lock lock(mutex_);
            reads_=std::move(reads); next_=0; pending_=reads_.size(); failure_=nullptr;
            ready_.notify_all(); done_.wait(lock,[this]{return pending_==0;});
            reads_.clear();
            if(failure_) { entries_.clear(); lru_.clear(); used_=0; std::rethrow_exception(failure_); }
        }
        stats.wait_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return result;
    } catch (...) {
        entries_.clear(); lru_.clear(); used_=0;
        throw;
    }
private:
    struct Record { std::shared_ptr<Entry> entry; std::list<int>::iterator order; };
    struct Read { int fd; void* dest; size_t bytes; uint64_t offset; };
    size_t budget_,used_=0;
    std::shared_ptr<std::atomic<size_t>> allocated_=std::make_shared<std::atomic<size_t>>(0);
    std::map<int,Record> entries_;
    std::list<int> lru_;
    std::map<std::string,int> files_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_,done_;
    std::vector<Read> reads_;
    size_t next_=0,pending_=0;
    bool stopping_=false;
    std::exception_ptr failure_;
    void stop() {
        { std::lock_guard lock(mutex_); stopping_=true; }
        ready_.notify_all(); for(auto& t:workers_) if(t.joinable())t.join();
    }
    void worker() {
#ifdef __linux__
        for(;;) {
            Read read;
            { std::unique_lock lock(mutex_); ready_.wait(lock,[this]{return stopping_ || next_<reads_.size();});
              if(stopping_)return; read=reads_[next_++]; }
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
