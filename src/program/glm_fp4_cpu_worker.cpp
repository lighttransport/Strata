// CPU-only native FP4 worker. No CUDA/HIP dependency or GPU initialization.
#include "strata/artifact/glm_fp4_partition.hpp"
#include "strata/kernels/cpu/glm_fp4_exact.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/host/glm_fp4_worker.h"
#include "strata/net/glm_fp4_worker.hpp"
#include "ucomm.h"
#include <array>
#include <cstring>
#include <iostream>
#include <list>
#include <thread>

namespace {
void ok(ucomm_status code) {if(code!=UCOMM_OK)strata::net::glm_fp4_transport_failure(code);}
void receive(ucomm_t* c,unsigned tag,void* p,size_t n) {
    size_t got=0;ok(ucomm_recv(c,0,tag,p,n,&got));if(got!=n)throw std::runtime_error("truncated FP4 worker message");
}
using Expert=strata::kernels::cpu::GlmFp4ExactExpert;
struct ExpertCache {
    std::map<std::pair<unsigned,unsigned>,std::unique_ptr<Expert>> entries;
    std::list<std::pair<unsigned,unsigned>> lru;
    size_t bytes=0,hits=0,misses=0;
    static constexpr size_t limit=size_t(2)<<30;
    Expert& get(strata::artifact::GlmFp4Partition& part,unsigned layer,unsigned id,bool qpn) {
        const auto key=std::make_pair(layer,id);
        if(!entries.count(key)) {
            ++misses;auto expert=std::make_unique<Expert>(part.model,layer,id,qpn);
            const size_t charge=expert->scale_bytes+sizeof(Expert);
            if(charge>limit)throw std::runtime_error("FP4 worker expert exceeds cache budget");
            while(!lru.empty()&&bytes+charge>limit) {
                const auto old=lru.back();lru.pop_back();bytes-=entries.at(old)->scale_bytes+sizeof(Expert);entries.erase(old);
            }
            bytes+=charge;entries.emplace(key,std::move(expert));
        }else ++hits;
        lru.remove(key);lru.push_front(key);return *entries.at(key);
    }
};
void serve(strata::artifact::GlmFp4Partition& part,const std::string& address,unsigned threads,bool qpn,bool fetch,bool spin,ExpertCache* experts=nullptr) {
    ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=1;cfg.world=2;cfg.root_addr=address.c_str();
    if(cfg.timeout_ms<=0)cfg.timeout_ms=60000;
    ucomm_t* c=nullptr;ok(ucomm_init(&cfg,&c));
    // Fingerprints and arithmetic mode must agree before processing requests.
    std::array<char,132> hello{};
    std::memcpy(hello.data(),part.report.at("layout_sha256").string().data(),64);
    std::memcpy(hello.data()+64,part.report.at("index_sha256").string().data(),64);
    hello[128]=char(qpn);hello[129]=char(fetch);hello[130]=STRATA_GLM_FP4_WORKER_VERSION;
    std::array<char,132> peer{};receive(c,20,peer.data(),peer.size());
    if(peer!=hello)throw std::runtime_error("FP4 worker handshake mismatch");
    ok(ucomm_send(c,0,21,hello.data(),hello.size()));
    std::unique_ptr<strata::cpu::exl3::Pool> pool;
    if(!fetch)pool=std::make_unique<strata::cpu::exl3::Pool>(threads,true,spin);
    uint32_t sequence=1;
    for(;;++sequence) {
        uint32_t header[8];strata::net::glm_fp4_receive_idle(c,0,22,header,sizeof header);
        if(!strata_glm_fp4_worker_command(header,sequence,part.remote_count,part.ordered.size(),fetch))
            throw std::runtime_error("invalid FP4 worker command");
        if(!header[2])break;
        if(header[2]==2) {
            const auto& tensor=*part.ordered.at(header[7]);
            std::vector<uint8_t> bytes(tensor.bytes);part.tensors.read(tensor,0,bytes.data(),bytes.size());
            ok(ucomm_send(c,0,24,bytes.data(),bytes.size()));continue;
        }
        auto& expert=experts->get(part,header[4],header[5],qpn);
        std::vector<float> x(size_t(header[6])*4096);receive(c,23,x.data(),x.size()*4);
        const float clamp=part.config.at("text_config").at("swiglu_limit").real();
        auto y=expert.run(*pool,x,header[6],clamp);
        ok(ucomm_send(c,0,24,y.data(),y.size()*4));
    }
    ucomm_finalize(c);
}
}
int main(int argc,char** argv) {
    try {
        std::string model,address,fetch;unsigned threads=14,sessions=1;bool qpn=false,ram_fds=false,spin=false,prewarm=false;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg.starts_with("--model="))model=arg.substr(8);
            else if(arg.starts_with("--root="))address=arg.substr(7);
            else if(arg.starts_with("--fetch-root="))fetch=arg.substr(13);
            else if(arg.starts_with("--threads="))threads=std::stoul(arg.substr(10));
            else if(arg.starts_with("--sessions="))sessions=std::stoul(arg.substr(11));
            else if(arg=="--spin")spin=true;
            else if(arg=="--prewarm")prewarm=true;
            else if(arg=="--qpn")qpn=true;
            else if(arg=="--ram-fds")ram_fds=true;
            else throw std::invalid_argument("unknown FP4 worker argument");
        }
        if(model.empty()||address.empty()||threads<1||threads>31||sessions<1||sessions>16)throw std::invalid_argument(
            "usage: strata-glm-fp4-cpu-worker --model=PARTITION --root=HOST:PORT [--fetch-root=HOST:PORT] [--threads=14] [--qpn] [--ram-fds] [--sessions=1..16] [--spin] [--prewarm]");
        strata::artifact::GlmFp4Partition part(model,ram_fds);
        const auto reserve=part.report.at("reserve_bytes").integer();
        const auto memory=strata::core::sample_host_memory();
        // Sealed RAM files were already populated and charged by the staging
        // owner. Charge their bytes once, and require room for this worker's
        // remaining runtime reserve rather than demanding a second weight set.
        if(ram_fds) {
            const auto index=strata::artifact::read_json((std::filesystem::path(model)/"model.safetensors.index.json").string());
            std::set<std::string> names;
            for(const auto& [name,file]:index.at("weight_map").object)names.insert(file.string());
            for(const auto& name:names) {
                struct stat info{};
                if(stat((std::filesystem::path(model)/name).c_str(),&info) ||
                   uint64_t(info.st_blocks)*512<uint64_t(info.st_size))
                    throw std::runtime_error("FP4 worker RAM file has unpopulated sparse pages");
            }
        }
        if(!memory || !strata_glm_fp4_layout_admit(ram_fds?0:part.tensors.file_bytes(),reserve,memory->available))
            throw std::runtime_error("FP4 worker weights and reserve exceed available RAM");
        // Compact files only: never warm absent tensors or the original full checkpoint.
        part.tensors.warm(true);
        const auto resident_memory=strata::core::sample_host_memory();
        if(!resident_memory || !strata_glm_fp4_layout_admit(0,reserve,resident_memory->available))
            throw std::runtime_error("FP4 worker runtime reserve unavailable after locking resident weights");
        ExpertCache experts;
        if(prewarm) {
            if(!strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE"))throw std::invalid_argument("FP4 worker --prewarm requires CPU_SPARSE=1");
            for(unsigned layer=3;layer<45;++layer)for(unsigned id=0;id<part.remote_count;++id)experts.get(part,layer,id,qpn);
            if(experts.entries.size()!=size_t(42)*part.remote_count)throw std::runtime_error("FP4 worker prewarm exceeds cache budget");
            std::cerr<<"FP4_WORKER prewarm_experts="<<experts.entries.size()<<" cache_bytes="<<experts.bytes<<'\n';
        }
        const auto ready_memory=strata::core::sample_host_memory();
        if(!ready_memory || !strata_glm_fp4_layout_admit(0,reserve,ready_memory->available))
            throw std::runtime_error("FP4 worker runtime reserve unavailable after cache prewarm");
        std::cerr<<"FP4_WORKER ready payload_bytes="<<part.tensors.bytes()<<" no_GPU=1\n";
        for(unsigned session=0;session<sessions;++session) {
        std::cerr<<"FP4_WORKER session="<<session<<" begin\n";
        std::thread fetcher;
        if(!fetch.empty())fetcher=std::thread([&] {
            try {serve(part,fetch,1,qpn,true,false);}catch(const std::exception& e) {
                std::cerr<<"FP4 fetch worker: "<<e.what()<<'\n';std::_Exit(1);
            }
        });
        try {serve(part,address,threads,qpn,false,spin,&experts);}
        catch(const std::exception& e) {
            std::cerr<<"FP4 compute worker: "<<e.what()<<'\n';std::cerr.flush();std::_Exit(1);
        }
        if(fetcher.joinable())fetcher.join();
        std::cerr<<"FP4_WORKER session="<<session<<" end cache_hits="<<experts.hits<<" cache_misses="<<experts.misses<<" cache_bytes="<<experts.bytes<<'\n';
        }
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FP4 worker: "<<e.what()<<'\n';std::cerr.flush();std::_Exit(1);
    }
}
