#pragma once
// Opt-in bounded local draft transport. No shell, and no fork after HIP initialization.
#include "strata/artifact/json.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#ifdef __linux__
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif
namespace strata::spec {
class ExternalWorker {
#ifdef __linux__
    int fd_ = -1;
    pid_t pid_ = -1;
    int timeout_ms_ = 100;
    uint64_t request_ = 0;
    int protocol_ = 1;
    std::string buffered_;
    using Clock = std::chrono::steady_clock;
    void ready(short events, Clock::time_point deadline) {
        for (;;) {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
            if (ms <= 0) throw std::runtime_error("external draft timeout");
            pollfd p{fd_, events, 0};
            int n = poll(&p, 1, static_cast<int>(ms));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0 || !(p.revents & events)) throw std::runtime_error("external draft unavailable");
            return;
        }
    }
    void send_line(const std::string& line, Clock::time_point deadline) {
        size_t used=0;
        while (used<line.size()) {
            ready(POLLOUT,deadline);
            ssize_t n=send(fd_,line.data()+used,line.size()-used,MSG_NOSIGNAL|MSG_DONTWAIT);
            if(n<0&&(errno==EINTR||errno==EAGAIN))continue;
            if(n<=0)throw std::runtime_error("external draft write failed (errno="+std::to_string(errno)+")");
            used+=static_cast<size_t>(n);
        }
    }
    void receive_bytes(void* destination, size_t size, Clock::time_point deadline) {
        auto* out=static_cast<char*>(destination);
        const size_t buffered=std::min(size,buffered_.size());
        std::memcpy(out,buffered_.data(),buffered);buffered_.erase(0,buffered);
        size_t used=buffered;
        while(used<size) {
            ready(POLLIN,deadline);
            const ssize_t n=recv(fd_,out+used,size-used,MSG_DONTWAIT);
            if(n<0&&(errno==EINTR||errno==EAGAIN))continue;
            if(n<=0)throw std::runtime_error("draft binary response closed");
            used+=static_cast<size_t>(n);
        }
    }
    strata::artifact::Json receive(Clock::time_point deadline) {
        for (;;) {
            const auto end=buffered_.find('\n');
            if(end!=std::string::npos){if(end>8192)throw std::runtime_error("external draft response too large");auto line=buffered_.substr(0,end);buffered_.erase(0,end+1);return strata::artifact::parse_json(line);}
            if(buffered_.size()>8192)throw std::runtime_error("external draft response too large");
            ready(POLLIN,deadline);char bytes[1024];ssize_t n=recv(fd_,bytes,sizeof(bytes),MSG_DONTWAIT);
            if(n<0&&(errno==EINTR||errno==EAGAIN))continue;
            if(n<=0)throw std::runtime_error("external draft closed");
            buffered_.append(bytes,static_cast<size_t>(n));
        }
    }
    void stop() noexcept {
        if(fd_>=0){close(fd_);fd_=-1;}
        if(pid_>0){kill(pid_,SIGKILL);for(int i=0;i<100;++i){if(waitpid(pid_,nullptr,WNOHANG)!=0)break;usleep(1000);}pid_=-1;}
    }
#endif
public:
    explicit ExternalWorker(const std::string& path, bool dflash2=false) {
#ifdef __linux__
        protocol_=dflash2?2:1;
        auto c=strata::artifact::read_json(path);
        const auto& args=c.at("argv");
        if(args.kind!=strata::artifact::Json::Array||args.array.empty())throw std::runtime_error("draft worker argv required");
        std::vector<std::string> values;for(const auto& a:args.array){values.push_back(a.string());if(values.back().find('\0')!=std::string::npos)throw std::runtime_error("draft argv contains NUL");}
        std::vector<char*> argv;for(auto& a:values)argv.push_back(a.data());argv.push_back(nullptr);
        if(auto t=c.find("timeout_ms")){auto n=t->integer();if(n<1||n>1000)throw std::runtime_error("draft timeout must be 1..1000ms");timeout_ms_=static_cast<int>(n);}
        int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,pair))throw std::runtime_error("draft socketpair failed");
        // Launchers may close stdin/stdout. Keep transport descriptors away from
        // 0/1/2 so file actions cannot close a just-duplicated child endpoint.
        for (int i=0;i<2;++i) if(pair[i]<3) {
            const int replacement=fcntl(pair[i],F_DUPFD_CLOEXEC,3);
            if(replacement<0){close(pair[0]);close(pair[1]);throw std::runtime_error("draft descriptor allocation failed");}
            close(pair[i]);pair[i]=replacement;
        }
        posix_spawn_file_actions_t actions;
        int error=posix_spawn_file_actions_init(&actions);
        if(error){close(pair[0]);close(pair[1]);throw std::runtime_error("draft spawn actions failed");}
        error=posix_spawn_file_actions_adddup2(&actions,pair[1],STDIN_FILENO);
        if(!error)error=posix_spawn_file_actions_adddup2(&actions,pair[1],STDOUT_FILENO);
        if(!error)error=posix_spawn_file_actions_addclose(&actions,pair[0]);
        if(!error)error=posix_spawn_file_actions_addclose(&actions,pair[1]);
        if(!error)error=posix_spawnp(&pid_,argv[0],&actions,nullptr,argv.data(),environ);
        posix_spawn_file_actions_destroy(&actions);close(pair[1]);fd_=pair[0];
        if(error){stop();throw std::runtime_error("draft spawn failed");}
        try {
            auto hello=receive(Clock::now()+std::chrono::seconds(dflash2?180:30));
            if(hello.at("protocol").integer()!=static_cast<uint64_t>(protocol_)||!hello.at("ready").flag())throw std::runtime_error("draft handshake invalid");
            const double mib=hello.at("vram_mib").real();
            if(!std::isfinite(mib)||mib<0||mib>(dflash2?4096:768))throw std::runtime_error("draft worker exceeds VRAM cap");
            if(dflash2 && (hello.at("hidden_size").integer()!=4096 || hello.at("block_size").integer()!=8 ||
                hello.at("vocab_size").integer()!=154880 || hello.at("mask_token_id").integer()!=154856))
                throw std::runtime_error("DFlash2 checkpoint geometry mismatch");
            std::cerr<<"EXTERNAL_READY vram_mib="<<mib<<'\n';
        }
        catch(...){stop();throw;}
#else
        (void)path;(void)dflash2;throw std::runtime_error("external draft worker currently requires Linux");
#endif
    }
    ~ExternalWorker(){
#ifdef __linux__
        stop();
#endif
    }
    ExternalWorker(const ExternalWorker&)=delete;
    ExternalWorker& operator=(const ExternalWorker&)=delete;
    bool active() const {
#ifdef __linux__
        return fd_ >= 0;
#else
        return false;
#endif
    }
    // Protocol 2 uses bounded little-endian float32 frames. Target features are
    // committed rows only; speculative draft KV is never retained by the worker.
    // The target's existing embedding and head remain in the engine process.
    strata::artifact::Json exchange(const std::string& fields, const float* data=nullptr,
                                   size_t count=0, std::vector<float>* output=nullptr,
                                   bool prepare=false) {
#ifdef __linux__
        if(protocol_!=2||fd_<0||count>size_t(2048)*20480+8*4096)
            throw std::runtime_error("DFlash2 frame bounds");
        const auto id=++request_;
        const auto deadline=Clock::now()+std::chrono::milliseconds(prepare?30000:timeout_ms_);
        send_line("{\"protocol\":2,\"request_id\":"+std::to_string(id)+","+fields+"}\n",deadline);
        if(count)send_line(std::string(reinterpret_cast<const char*>(data),count*sizeof(float)),deadline);
        auto response=receive(deadline);
        if(response.at("protocol").integer()!=2||response.at("request_id").integer()!=id)
            throw std::runtime_error("DFlash2 response identity mismatch");
        if(output) {
            if(response.at("rows").integer()!=7||response.at("columns").integer()!=4096)
                throw std::runtime_error("DFlash2 hidden response shape mismatch");
            output->resize(7*4096);receive_bytes(output->data(),output->size()*sizeof(float),deadline);
            for(float v:*output)if(!std::isfinite(v))throw std::runtime_error("DFlash2 nonfinite hidden response");
        }
        return response;
#else
        (void)fields;(void)data;(void)count;(void)output;(void)prepare;
        throw std::runtime_error("DFlash2 requires Linux");
#endif
    }
    void prime(const std::vector<int>& history) { (void)propose(history,0,0,true); }
    std::vector<int> propose(const std::vector<int>& history,int depth,int vocabulary,bool prime=false) {
#ifdef __linux__
        if(fd_<0||(!prime&&depth<=0))return {};
        try {
            if(history.size()>131072||depth>7)throw std::runtime_error("external draft request bounds");
            const auto id=++request_;std::string line="{\"protocol\":1,\"request_id\":"+std::to_string(id)+",\"prime\":"+(prime?"true":"false")+",\"max_tokens\":"+std::to_string(depth)+",\"context\":[";
            for(size_t i=0;i<history.size();++i){if(i)line+=',';line+=std::to_string(history[i]);}line+="]}\n";
            auto deadline=Clock::now()+std::chrono::milliseconds(prime?30000:timeout_ms_);send_line(line,deadline);auto response=receive(deadline);
            if(response.at("protocol").integer()!=1||response.at("request_id").integer()!=id)throw std::runtime_error("external draft response identity mismatch");
            if(prime&&!response.at("primed").flag())throw std::runtime_error("external draft priming failed");
            const auto& tokens=response.at("tokens");if(tokens.kind!=strata::artifact::Json::Array||tokens.array.size()>static_cast<size_t>(depth))throw std::runtime_error("external draft length invalid");
            std::vector<int> result;for(const auto& t:tokens.array){auto v=t.integer();if(v>=static_cast<uint64_t>(vocabulary))throw std::runtime_error("external draft token invalid");result.push_back(static_cast<int>(v));}return result;
        } catch(const std::exception& e){std::cerr<<"EXTERNAL_DRAFT_DISABLED reason="<<e.what()<<'\n';stop();return {};}
#else
        (void)history;(void)depth;(void)vocabulary;(void)prime;return {};
#endif
    }
};
}
