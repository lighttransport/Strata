#pragma once
// Indexed bounded reads follow gemm/common/glm53f_safetensors.h; no payload mapping at inspection.
#include "strata/artifact/json.hpp"
#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::artifact {
enum class TensorDtype { F32, BF16, F16, I16, I32, I64, U8, I8, Bool };
inline size_t dtype_bytes(const std::string& name) {
    if(name=="F32"||name=="I32") return 4;
    if(name=="BF16"||name=="F16"||name=="I16") return 2;
    if(name=="I64") return 8;
    if(name=="U8"||name=="I8"||name=="BOOL") return 1;
    throw std::runtime_error("safetensors: unsupported dtype " + name);
}
struct TensorView {
    std::string name, dtype;
    std::vector<uint64_t> shape; // source row-major dimensions, never GGUF dimensions
    uint64_t offset=0, bytes=0;
    size_t shard=0;
};
class Safetensors {
    struct Shard {
        std::filesystem::path path;
        uint64_t start=0, size=0;
#ifndef _WIN32
        int fd=-1;
        ~Shard() { if(fd>=0) ::close(fd); }
#else
        mutable std::ifstream file;
        mutable std::mutex mutex;
#endif
        void read(uint64_t offset, void* destination, size_t bytes) const {
            if(offset>size || bytes>size-offset) throw std::runtime_error("safetensors: read exceeds shard " + path.string());
#ifndef _WIN32
            auto* out=static_cast<char*>(destination);
            while(bytes) {
                auto n=::pread(fd,out,bytes,static_cast<off_t>(offset));
                if(n<0 && errno==EINTR) continue;
                if(n<=0) throw std::runtime_error("safetensors: failed read " + path.string());
                out+=n; offset+=static_cast<uint64_t>(n); bytes-=static_cast<size_t>(n);
            }
#else
            std::lock_guard lock(mutex); file.clear(); file.seekg(static_cast<std::streamoff>(offset));
            if(!file.read(static_cast<char*>(destination),static_cast<std::streamsize>(bytes))) throw std::runtime_error("safetensors: failed read " + path.string());
#endif
        }
    };
    std::vector<std::unique_ptr<Shard>> shards_;
    std::map<std::string,TensorView> tensors_;
    uint64_t payload_=0, file_bytes_=0;
public:
    explicit Safetensors(const std::filesystem::path& directory) {
        auto root=std::filesystem::canonical(directory);
        Json index=read_json((root/"model.safetensors.index.json").string());
        const auto& map=index.at("weight_map");
        if(map.kind!=Json::Object || map.object.empty()) throw std::runtime_error("safetensors: empty weight map");
        std::set<std::string> paths;
        for(const auto& [name,shard] : map.object) {
            auto p=std::filesystem::path(shard.string());
            if(name.empty() || p.empty() || p.is_absolute() || p.has_parent_path() || p.extension()!=".safetensors") throw std::runtime_error("safetensors: invalid shard path");
            paths.insert(p.string());
        }
        if(paths.size()>1024) throw std::runtime_error("safetensors: too many shards");
        for(const auto& filename:paths) {
            auto shard=std::make_unique<Shard>(); shard->path=std::filesystem::canonical(root/filename);
            if(shard->path.parent_path()!=root) throw std::runtime_error("safetensors: shard escapes model directory");
            shard->size=std::filesystem::file_size(shard->path);
            file_bytes_+=shard->size;
            if(shard->size<8 || shard->size>uint64_t(INT64_MAX)) throw std::runtime_error("safetensors: invalid shard size");
#ifndef _WIN32
            shard->fd=::open(shard->path.c_str(),O_RDONLY|O_CLOEXEC);
            if(shard->fd<0) throw std::runtime_error("safetensors: cannot open shard");
#else
            shard->file.open(shard->path,std::ios::binary); if(!shard->file) throw std::runtime_error("safetensors: cannot open shard");
#endif
            std::array<uint8_t,8> length{}; shard->read(0,length.data(),8); uint64_t n=0;
            for(unsigned i=0;i<8;++i) n|=uint64_t(length[i])<<(8*i);
            if(n>64*1024*1024 || n>shard->size-8) throw std::runtime_error("safetensors: truncated or oversized header");
            std::string text(static_cast<size_t>(n),'\0'); shard->read(8,text.data(),text.size());
            auto header=parse_json(text); if(header.kind!=Json::Object) throw std::runtime_error("safetensors: header is not an object");
            shard->start=8+n;
            std::map<uint64_t,uint64_t> spans;
            for(const auto& [name,entry]:header.object) {
                if(name=="__metadata__") continue;
                TensorView t; t.name=name; t.dtype=entry.at("dtype").string(); t.shard=shards_.size();
                const auto& shape=entry.at("shape"); const auto& offsets=entry.at("data_offsets");
                if(shape.kind!=Json::Array || shape.array.size()>8 || offsets.kind!=Json::Array || offsets.array.size()!=2) throw std::runtime_error("safetensors: invalid shape/offsets " + name);
                uint64_t bytes=dtype_bytes(t.dtype);
                for(const auto& dim:shape.array) {
                    uint64_t d=dim.integer(); if(d && bytes>UINT64_MAX/d) throw std::runtime_error("safetensors: extent overflow " + name);
                    bytes*=d; t.shape.push_back(d);
                }
                uint64_t begin=offsets.array[0].integer(),end=offsets.array[1].integer();
                if(end<begin || end-begin!=bytes || end>shard->size-shard->start) throw std::runtime_error("safetensors: invalid tensor span " + name);
                t.offset=begin; t.bytes=bytes;
                if(bytes && !spans.emplace(begin,end).second) throw std::runtime_error("safetensors: duplicate span " + name);
                auto declared=map.object.find(name);
                if(declared==map.object.end() || declared->second.string()!=filename) throw std::runtime_error("safetensors: index/header disagreement " + name);
                if(!tensors_.emplace(name,std::move(t)).second) throw std::runtime_error("safetensors: duplicate tensor " + name);
                if(payload_>UINT64_MAX-bytes) throw std::runtime_error("safetensors: payload overflow");
                payload_+=bytes;
            }
            uint64_t end=0;
            for(const auto& [begin,finish]:spans) { if(begin!=end) throw std::runtime_error("safetensors: overlapping spans or payload holes"); end=finish; }
            if(end!=shard->size-shard->start) throw std::runtime_error("safetensors: trailing payload bytes");
            shards_.push_back(std::move(shard));
        }
        if(tensors_.size()!=map.object.size()) throw std::runtime_error("safetensors: missing indexed tensor");
        if(auto m=index.find("metadata")) if(auto size=m->find("total_size")) if(size->integer()!=payload_ && size->integer()!=file_bytes_) throw std::runtime_error("safetensors: total_size mismatch");
    }
    const std::map<std::string,TensorView>& tensors() const { return tensors_; }
    const TensorView& at(const std::string& name) const {
        auto i=tensors_.find(name); if(i==tensors_.end()) throw std::runtime_error("safetensors: missing tensor " + name); return i->second;
    }
    uint64_t bytes() const { return payload_; }
    size_t shards() const { return shards_.size(); }
    void expect(const std::string& name,const std::string& dtype,const std::vector<uint64_t>& shape) const {
        const auto& t=at(name); if(t.dtype!=dtype || t.shape!=shape) throw std::runtime_error("safetensors: incompatible dtype/shape " + name);
    }
    void read(const TensorView& t,uint64_t offset,void* out,size_t bytes) const {
        if(offset>t.bytes || bytes>t.bytes-offset || (bytes && !out)) throw std::runtime_error("safetensors: read exceeds tensor " + t.name);
        shards_.at(t.shard)->read(shards_[t.shard]->start+t.offset+offset,out,bytes);
    }
    void read(const std::string& name,uint64_t offset,void* out,size_t bytes) const { read(at(name),offset,out,bytes); }
};
} // namespace strata::artifact
