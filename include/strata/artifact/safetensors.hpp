#pragma once
// Indexed bounded reads follow gemm/common/glm53f_safetensors.h; no payload mapping at inspection.
#include "strata/artifact/json.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#include <sys/stat.h>
#endif
#endif

namespace strata::artifact {
enum class TensorDtype { F32, BF16, F16, I16, I32, I64, U8, I8, Bool, U16, U32 };
inline size_t dtype_bytes(const std::string& name) {
    if(name=="F32"||name=="I32"||name=="U32") return 4;
    if(name=="BF16"||name=="F16"||name=="I16"||name=="U16") return 2;
    if(name=="I64") return 8;
    if(name=="U8"||name=="I8"||name=="BOOL"||name=="F8_E4M3") return 1;
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
        mutable void* mapping = MAP_FAILED;
        mutable std::mutex mapping_mutex;
        ~Shard() { if(mapping != MAP_FAILED) ::munmap(mapping, size); if(fd>=0) ::close(fd); }
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
#ifndef _WIN32
        const uint8_t* map(bool resident_copy) const {
            std::lock_guard guard(mapping_mutex);
            if(mapping!=MAP_FAILED)return static_cast<const uint8_t*>(mapping);
            if(!resident_copy) {
                mapping=::mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0);
                if(mapping==MAP_FAILED)throw std::runtime_error("safetensors: cannot map shard");
                return static_cast<const uint8_t*>(mapping);
            }
            // One immutable compressed copy shared by this process's workers.
            // Read directly into the final mapping and release source cache
            // pages in bounded chunks, rather than retaining two weight sets.
            void* pages=::mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
            if(pages==MAP_FAILED)throw std::runtime_error("safetensors: cannot allocate shared RAM weights");
            try {
                constexpr uint64_t chunk=4*1024*1024;
                for(uint64_t offset=0;offset<size;offset+=chunk) {
                    const size_t bytes=static_cast<size_t>(std::min(chunk,size-offset));
                    read(offset,static_cast<uint8_t*>(pages)+offset,bytes);
                    ::posix_fadvise(fd,static_cast<off_t>(offset),static_cast<off_t>(bytes),POSIX_FADV_DONTNEED);
                }
                if(::mprotect(pages,size,PROT_READ))throw std::runtime_error("safetensors: cannot protect shared RAM weights");
            }catch(...) {::munmap(pages,size);throw;}
            mapping=pages;
            std::fprintf(stderr,"FP4_HOST_STAGE bytes=%llu backing=shared_ram readonly=1\n",static_cast<unsigned long long>(size));
            return static_cast<const uint8_t*>(mapping);
        }
#endif
    };
    std::vector<std::unique_ptr<Shard>> shards_;
    std::map<std::string,TensorView> tensors_;
    uint64_t payload_=0, file_bytes_=0;
    bool resident_copy_=false,ram_fds_=false;
public:
    explicit Safetensors(const std::filesystem::path& directory,bool indexed_subsets=false,bool resident_copy=false,bool ram_fds=false)
        :resident_copy_(resident_copy),ram_fds_(ram_fds) {
        if(resident_copy_&&ram_fds_)throw std::invalid_argument("safetensors: choose RAM copy or sealed RAM files");
        auto root=std::filesystem::canonical(directory);
        Json index=read_json((root/"model.safetensors.index.json").string());
        const auto& map=index.at("weight_map");
        if(map.kind!=Json::Object || map.object.empty()) throw std::runtime_error("safetensors: empty weight map");
        std::set<std::string> paths;
        for(const auto& [name,shard] : map.object) {
            auto p=std::filesystem::path(shard.string());
            bool parent=false;for(const auto& component:p)if(component=="..")parent=true;
            if(name.empty() || p.empty() || p.is_absolute() || parent ||
               (!indexed_subsets&&p.has_parent_path()) || p.extension()!=".safetensors") throw std::runtime_error("safetensors: invalid shard path");
            paths.insert(p.string());
        }
        if(paths.size()>1024) throw std::runtime_error("safetensors: too many shards");
        for(const auto& filename:paths) {
            auto shard=std::make_unique<Shard>();
            if(ram_fds_)shard->path=root/filename;
            else {
                shard->path=std::filesystem::canonical(root/filename);
                const auto relative=shard->path.lexically_relative(root);
                if(relative.empty() || relative.is_absolute() || *relative.begin()==".." ||
                   (!indexed_subsets&&shard->path.parent_path()!=root))throw std::runtime_error("safetensors: shard escapes model directory");
            }
            shard->size=std::filesystem::file_size(shard->path);
            if(file_bytes_>UINT64_MAX-shard->size)throw std::runtime_error("safetensors: shard size sum overflow");
            file_bytes_+=shard->size;
            if(shard->size<8 || shard->size>uint64_t(INT64_MAX)) throw std::runtime_error("safetensors: invalid shard size");
#ifndef _WIN32
            shard->fd=::open(shard->path.c_str(),O_RDONLY|O_CLOEXEC);
            if(shard->fd<0) throw std::runtime_error("safetensors: cannot open shard");
            if(ram_fds_) {
#ifdef __linux__
                struct stat st{};struct statfs fs{};
                const int required=F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL;
                const int seals=::fcntl(shard->fd,F_GET_SEALS);
                if(::fstat(shard->fd,&st)||::fstatfs(shard->fd,&fs)||!S_ISREG(st.st_mode)||st.st_uid!=::geteuid()||
                   uint64_t(st.st_size)!=shard->size||fs.f_type!=TMPFS_MAGIC||seals<0||(seals&required)!=required)
                    throw std::runtime_error("safetensors: RAM shard must be an immutable sealed memfd owned by this user");
#else
                throw std::runtime_error("safetensors: sealed RAM files require Linux");
#endif
            }
#else
            if(ram_fds_)throw std::runtime_error("safetensors: sealed RAM files require Linux");
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
                if(declared==map.object.end() || declared->second.string()!=filename) {
                    if(indexed_subsets)continue; // the index explicitly selects a local replacement
                    throw std::runtime_error("safetensors: index/header disagreement " + name);
                }
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
    uint64_t file_bytes() const { return file_bytes_; }
    size_t shards() const { return shards_.size(); }
    void expect(const std::string& name,const std::string& dtype,const std::vector<uint64_t>& shape) const {
        const auto& t=at(name); if(t.dtype!=dtype || t.shape!=shape) throw std::runtime_error("safetensors: incompatible dtype/shape " + name);
    }
    void read(const TensorView& t,uint64_t offset,void* out,size_t bytes) const {
        if(offset>t.bytes || bytes>t.bytes-offset || (bytes && !out)) throw std::runtime_error("safetensors: read exceeds tensor " + t.name);
        if((resident_copy_||ram_fds_) && bytes) {std::memcpy(out,data(t)+offset,bytes);return;}
        shards_.at(t.shard)->read(shards_[t.shard]->start+t.offset+offset,out,bytes);
    }
    void read(const std::string& name,uint64_t offset,void* out,size_t bytes) const { read(at(name),offset,out,bytes); }
    // Read-only pages are shared by all native workers. The default maps the
    // source file; the opt-in RAM copy keeps its original compressed bytes.
    // Mapping lifetime equals this reader's lifetime.
    const uint8_t* data(const TensorView& t) const {
#ifndef _WIN32
        const auto& s=*shards_.at(t.shard);
        return s.map(resident_copy_)+s.start+t.offset;
#else
        (void)t;
        throw std::runtime_error("safetensors: shared native mappings require Linux");
#endif
    }
    // Only immutable RAM-backed checkpoints may opt in to long-lived DMA
    // registration. Returned mappings stay owned by this reader.
    std::vector<std::pair<void*,size_t>> ram_mappings()const {
        if(!resident_copy_&&!ram_fds_)throw std::runtime_error("safetensors: direct DMA requires RAM-backed shards");
#ifndef _WIN32
        std::vector<std::pair<void*,size_t>> result;
        for(const auto& shard:shards_)result.emplace_back(const_cast<uint8_t*>(shard->map(resident_copy_)),shard->size);
        return result;
#else
        throw std::runtime_error("safetensors: shared RAM mappings require Linux");
#endif
    }
    const uint8_t* data(const std::string& name) const { return data(at(name)); }
    void warm(bool lock=false)const {
#ifndef _WIN32
        uint64_t bytes=0;unsigned checksum=0;
        for(const auto& holder:shards_) {
            const auto& s=*holder;
            s.map(resident_copy_);
            std::lock_guard guard(s.mapping_mutex);
            if(lock && ::mlock(s.mapping,s.size))throw std::runtime_error("safetensors: cannot lock shared weight pages");
            const auto* p=static_cast<const volatile uint8_t*>(s.mapping);
            for(uint64_t offset=0;offset<s.size;offset+=4096)checksum^=p[offset];
            bytes+=s.size;
        }
        std::fprintf(stderr,"FP4_HOST_WEIGHTS bytes=%llu locked=%d shared_readonly=1 checksum=%u backing=%s\n",
                     static_cast<unsigned long long>(bytes),int(lock),checksum,ram_fds_?"sealed_ram":resident_copy_?"shared_ram":"source_file");
#else
        (void)lock;throw std::runtime_error("safetensors: resident shared weights require Linux");
#endif
    }
    // Drop a tensor's clean pages from the page cache (a hint; a weight set larger than RAM is streamed once per use).
    void drop(const TensorView& t) const {
#ifndef _WIN32
        if(resident_copy_||ram_fds_)return;
        const auto& s=*shards_.at(t.shard);
        ::posix_fadvise(s.fd,static_cast<off_t>(s.start+t.offset),static_cast<off_t>(t.bytes),POSIX_FADV_DONTNEED);
#else
        (void)t;
#endif
    }
};
} // namespace strata::artifact
