namespace {
// Contiguous GLM stages using the existing GGUF arithmetic. The scheduling idea
// follows project-maya (MIT), pinned and credited in docs/GLM_MULTI_GPU.md.
// No quantization or model-pack conversion is performed here.
struct GlmLayerOptions {
    std::string devices = "0,1", split = "auto", pipeline = "none", memory = "interleave", plan, executable;
    int context = 4096, dense = 4096, threads = 6, width = 8, steps = 1, repeats = 1;
    int depth = 1, cache_window = 256, force_miss = 0, emulate = 0;
    size_t gpu_budget = 12288 * MiB, cache = 0;
    bool mtp = false, draft_cpu = false, lock = false, graphs = false, pin = true, check = false, check_pipeline = false, adapt = false;
    std::string prefill_dump;
    std::string eval_corpus, eval_save, eval_reference;
    bool eval_prefill = false;
    std::vector<int> stage_workers, stage_cache_mib;
    bool pin_coordinator = false;
    std::vector<int> stops;
};

static std::string layer_measured_split(const GlmLayerOptions &options,const std::string &path,
                                        const std::vector<int> &devices) {
    if(options.plan.empty())return options.split;
    if(options.split!="auto")throw std::invalid_argument("GLM: layer-plan requires layer-split=auto");
    const auto plan=strata::artifact::read_json(options.plan);
    const bool v2=plan.at("schema").string()=="strata.glm.layer-plan.v2";
    if((!v2 && plan.at("schema").string()!="strata.glm.layer-plan.v1") ||
       plan.at("model_filename").string()!=std::filesystem::path(path).filename().string() ||
       plan.at("model_bytes").integer()!=std::filesystem::file_size(path))
        throw std::invalid_argument("GLM: measured layer plan belongs to a different model");
    std::string executable=options.executable;
#ifdef __linux__
    executable="/proc/self/exe";
#endif
    std::ifstream binary(executable,std::ios::binary);
    if(!binary)throw std::runtime_error("GLM: cannot read decoder identity for measured plan");
    uint64_t fingerprint=14695981039346656037ULL;std::array<char,131072> buffer{};
    while(binary){binary.read(buffer.data(),buffer.size());for(std::streamsize i=0;i<binary.gcount();++i){fingerprint^=(unsigned char)buffer[i];fingerprint*=1099511628211ULL;}}
    if(!binary.eof() || fingerprint!=plan.at("decoder_fnv64").integer())throw std::invalid_argument("GLM: measured layer plan belongs to a different decoder build");
    const auto &cfg=plan.at("configuration");
    if(!v2 && (!options.stage_workers.empty() || !options.stage_cache_mib.empty() || options.pin_coordinator))
        throw std::invalid_argument("GLM: per-stage budgets or coordinator affinity require a v2 measured plan");
    if(v2) {
        auto list_matches=[&](const char *name,const std::vector<int> &values) {
            const auto &recorded=cfg.at(name);
            if(values.empty() && recorded.kind==strata::artifact::Json::Null)return;
            std::string text;for(int value:values){if(!text.empty())text+=',';text+=std::to_string(value);}
            if(recorded.kind!=strata::artifact::Json::String || recorded.string()!=text)
                throw std::invalid_argument(std::string("GLM: layer plan setting differs: ")+name);
        };
        list_matches("layer_workers",options.stage_workers);list_matches("layer_cache_mib",options.stage_cache_mib);
        if(cfg.at("layer_pin_coordinator").flag()!=options.pin_coordinator)
            throw std::invalid_argument("GLM: layer plan coordinator affinity differs");
    }
    auto number=[&](const char *name,uint64_t expected){if(cfg.at(name).integer()!=expected)throw std::invalid_argument(std::string("GLM: layer plan setting differs: ")+name);};
    auto text=[&](const char *name,const std::string &expected){if(cfg.at(name).string()!=expected)throw std::invalid_argument(std::string("GLM: layer plan setting differs: ")+name);};
    number("context",options.context);number("threads",options.threads);number("gpu_budget_mib",options.gpu_budget/MiB);
    number("decode_cache_window",options.cache_window);number("draft_depth",options.depth);number("prefill_batch",options.width?options.width:256);
    text("gpu_devices",options.devices);text("decode_cache_mib",std::to_string(options.cache/MiB));
    text("speculative",options.mtp?"mtp":"none");text("layer_pipeline",options.pipeline);text("layer_memory",options.memory);
    if(cfg.at("decode_cache_adapt").flag()!=options.adapt || cfg.at("decode_graphs").flag()!=options.graphs || cfg.at("pin_cpu").flag()!=options.pin)
        throw std::invalid_argument("GLM: layer plan cache or graph policy differs");
    const auto &environment=plan.at("environment");
    for(const auto &[name,value]:environment.object){
        const char *actual=std::getenv(name.c_str());
        if(value.kind==strata::artifact::Json::Null ? actual!=nullptr : (!actual || value.string()!=actual))
            throw std::invalid_argument("GLM: layer plan environment differs: "+name);
    }
#ifdef __linux__
    for(char **entry=::environ;entry && *entry;++entry){
        const std::string item=*entry;const auto at=item.find('=');const std::string name=item.substr(0,at);
        const bool relevant=name.starts_with("STRATA_GLM_") || name.starts_with("STRATA_NATIVE_") ||
                            name.starts_with("STRATA_POOL_") || name=="STRATA_OLD_IQ_MMVQ" || name=="STRATA_GROUPED_V1";
        if(relevant && !environment.object.count(name))throw std::invalid_argument("GLM: layer plan has no matching environment setting: "+name);
    }
#endif
    const auto &hardware=plan.at("hardware");const auto &names=hardware.at("gpus").array;
    for(int device:devices){cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
        if((size_t)device>=names.size() || names[device].string()!=props.name)throw std::invalid_argument("GLM: layer plan GPU hardware differs");}
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");std::string line,cpu;
    while(std::getline(cpuinfo,line))if(line.starts_with("model name")){
        cpu=line.substr(line.find(':')+1);cpu.erase(0,cpu.find_first_not_of(" \t"));break;
    }
    if(hardware.at("cpu").string()!=cpu)throw std::invalid_argument("GLM: layer plan CPU hardware differs");
#endif
    std::cerr<<"LAYER_PLAN source="<<options.plan<<" selected="<<plan.at("layer_split").string()<<" objective=measured_single_stream_decode\n";
    return plan.at("layer_split").string();
}

// Keep the allocation's physical cores: in particular, do not equate NUMA node
// with socket on hosts with two NUMA domains per socket. Socket is used only as
// a preference; physical_cores already respects the caller's allowed cpuset.
static std::vector<cpu::CpuTopology> layer_cpu_placement(const std::vector<int> &devices, int threads,
        const std::vector<int> &specified = {}, int *coordinator = nullptr) {
    const int stages=(int)devices.size();
    auto cores = cpu::physical_cores(false);
    if (cores.size() < (size_t)(2 * stages + 1))
        throw std::invalid_argument("GLM layer layout needs a coordinator, one submitter and one worker core per stage");
    std::vector<cpu::CpuTopology> result(stages);
    if(coordinator)*coordinator=cores.front();
    cores.erase(cores.begin()); // coordinator
    const int budget = std::min(threads, (int)cores.size() - stages);
    if (budget < stages) throw std::invalid_argument("GLM layer threads must provide at least one worker per stage");
    if(!specified.empty() && (specified.size()!=devices.size() ||
            std::any_of(specified.begin(),specified.end(),[](int n){return n<1;}) ||
            std::accumulate(specified.begin(),specified.end(),uint64_t(0))>(uint64_t)budget))
        throw std::invalid_argument("GLM: layer-workers must give one positive count per stage within the worker budget");
    auto package=[](int core) {
        int id=-1;
#ifdef __linux__
        std::ifstream file("/sys/devices/system/cpu/cpu"+std::to_string(core)+"/topology/physical_package_id");file>>id;
#else
        (void)core;
#endif
        return id;
    };
    std::vector<int> preferred(stages,-1);
#ifdef __linux__
    for(int i=0;i<stages;++i) {
        char pci[32]{}; check(cudaDeviceGetPCIBusId(pci,sizeof(pci),devices[i]));
        std::string address=pci;
        std::transform(address.begin(),address.end(),address.begin(),[](unsigned char c){return (char)std::tolower(c);});
        int node=-1;std::ifstream file("/sys/bus/pci/devices/"+address+"/numa_node");file>>node;
        if(node>=0)for(int core:cores)if(std::filesystem::exists("/sys/devices/system/cpu/cpu"+std::to_string(core)+"/node"+std::to_string(node))) {
            preferred[i]=package(core);break;
        }
    }
#endif
    auto take=[&](int preferred_package) {
        auto it=std::find_if(cores.begin(),cores.end(),[&](int core){return preferred_package>=0 && package(core)==preferred_package;});
        if(it==cores.end())it=cores.begin();
        const int core=*it;cores.erase(it);return core;
    };
    // Reserve every submitter before assigning workers so a large first pool
    // cannot consume a later stage's control core.
    for(int i=0;i<stages;++i)result[i].host_core=take(preferred[i]);
    for (int i = 0; i < stages; ++i) {
        auto &topo = result[i];
        const int n = specified.empty()?budget / stages + (i < budget % stages):specified[i];
        for(int j=0;j<n;++j)topo.worker_cores.push_back(take(preferred[i]));
        std::cerr << "LAYER_CPU stage=" << i << " socket=" << preferred[i] << " host=" << topo.host_core << " workers=";
        for (int c : topo.worker_cores) std::cerr << c << ',';
        std::cerr << '\n';
    }
    std::cerr << "LAYER_CPU requested_workers=" << threads << " workers=" << budget << " reserve_submitters=" << stages << '\n';
    return result;
}

struct LayerCoordinatorAffinity {
#ifdef __linux__
    cpu_set_t previous{};
    explicit LayerCoordinatorAffinity(int core) {
        if(core<0 || core>=CPU_SETSIZE || sched_getaffinity(0,sizeof previous,&previous)!=0 || !CPU_ISSET(core,&previous))
            throw std::runtime_error("GLM: coordinator CPU is outside the allocation affinity");
        cpu_set_t selected;CPU_ZERO(&selected);CPU_SET(core,&selected);
        if(sched_setaffinity(0,sizeof selected,&selected)!=0)throw std::runtime_error("GLM: cannot pin layer coordinator");
        std::cerr<<"LAYER_COORDINATOR cpu="<<core<<'\n';
    }
    ~LayerCoordinatorAffinity(){(void)sched_setaffinity(0,sizeof previous,&previous);}
#else
    explicit LayerCoordinatorAffinity(int){throw std::runtime_error("GLM: coordinator affinity requires Linux");}
#endif
};

// Apply NUMA policy only while the single shared mapping is materialized.
// MPOL_F_MEMS_ALLOWED handles restricted Slurm memory masks and any number of
// NUMA domains. Pinned transport keeps the existing GPU locality policy.
class LayerWeightPolicy {
#ifdef __linux__
    int previous=MPOL_DEFAULT;
    std::array<unsigned long,16> saved{};
    bool changed=false;
#endif
  public:
    explicit LayerWeightPolicy(int socket_core=-1) {
#ifdef __linux__
        std::array<unsigned long,16> allowed{};
        if (!syscall(SYS_get_mempolicy,&previous,saved.data(),1024,nullptr,0) &&
            !syscall(SYS_get_mempolicy,nullptr,allowed.data(),1024,nullptr,MPOL_F_MEMS_ALLOWED)) {
            if(socket_core>=0) {
                int package=-1;
                std::ifstream file("/sys/devices/system/cpu/cpu"+std::to_string(socket_core)+"/topology/physical_package_id");file>>package;
                std::array<unsigned long,16> local{};
                for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allocation_cpus)) {
                    int id=-1;std::ifstream info("/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/physical_package_id");info>>id;
                    if(id!=package || package<0)continue;
                    for(int node=0;node<1024;++node)if((allowed[node/64]>>(node%64)&1UL) &&
                        std::filesystem::exists("/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/node"+std::to_string(node)))
                        local[node/64]|=1UL<<(node%64);
                }
                if(std::any_of(local.begin(),local.end(),[](unsigned long bits){return bits!=0;}))allowed=local;
            }
            changed=!syscall(SYS_set_mempolicy,MPOL_INTERLEAVE,allowed.data(),1024);
        }
        std::cerr << "LAYER_WEIGHT_POLICY requested=" << (socket_core>=0?"socket":"interleave") << " applied=" << changed << " nodes=";
        for(int node=0;node<1024;++node)if(allowed[node/64]>>(node%64)&1UL)std::cerr<<node<<',';
        std::cerr<<'\n';
#else
        (void)socket_core;
#endif
    }
    ~LayerWeightPolicy() {
#ifdef __linux__
        if(changed) syscall(SYS_set_mempolicy,previous,previous==MPOL_DEFAULT?nullptr:saved.data(),1024);
#endif
    }
};
static void layer_lock_budget(const strata::core::ModelArtifact &model,bool draft,size_t transport_bytes,size_t stages) {
#ifdef __linux__
    size_t bytes=0;
    for(const auto &[name,tensor]:model.tensors()) {
        int layer=0; const char *suffix=nullptr;
        if(!draft && strata_parse_blk(name.c_str(),&layer,&suffix)==1 && layer==(int)model.descriptor().layers.size()) continue;
        if(tensor.bytes>SIZE_MAX-bytes)throw std::overflow_error("GLM layer host budget overflow");
        bytes+=tensor.bytes;
    }
    uint64_t limit=uint64_t(sysconf(_SC_PHYS_PAGES))*uint64_t(sysconf(_SC_PAGESIZE));
    for(const char *path:{"/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream file(path); std::string text; uint64_t value=0;
        if(file>>text && text!="max") {try{value=std::stoull(text);}catch(...){continue;} if(value)limit=std::min(limit,value);}
    }
    // Slurm can expose a nested or unmounted memory controller. Its requested
    // node allowance is still an upper bound even when memory.max is absent.
    if(const char *text=std::getenv("SLURM_MEM_PER_NODE")) {
        try {
            size_t used=0;const auto mib=std::stoull(text,&used);
            if(text[used]=='\0' && mib && mib<=UINT64_MAX/MiB)limit=std::min(limit,uint64_t(mib)*MiB);
        } catch(const std::exception &) {}
    }
    const uint64_t staging=transport_bytes+stages*512*MiB;
    rlimit locked{};
    if(getrlimit(RLIMIT_MEMLOCK,&locked) || (locked.rlim_cur!=RLIM_INFINITY && bytes+staging>locked.rlim_cur))
        throw std::runtime_error("GLM: layer weights and staging need a larger locked-memory allowance");
    const uint64_t reserve=4ULL*1024*MiB+staging;
    if(limit<reserve || bytes>limit-reserve)throw std::runtime_error("GLM layer locked weights plus 4 GiB host reserve exceed memory allowance");
    std::cerr << "LAYER_HOST_BUDGET weights_MiB=" << bytes/MiB << " reserve_MiB=" << reserve/MiB << " limit_MiB=" << limit/MiB << '\n';
#else
    (void)model;(void)draft;(void)transport_bytes;(void)stages;
#endif
}

class LayerDecoder {
    struct Stage {
        int device;
        uint64_t decode_rows=0;
        double decode_ms=0, draft_ms=0;
        std::unique_ptr<Decoder> decoder;
        std::shared_ptr<GpuWorker> worker;
        Stage(int id, std::shared_ptr<GpuWorker> submitter) : device(id), worker(std::move(submitter)) {}
        ~Stage() {
            // A worker may still refer to its decoder during exception unwinding.
            if (worker) worker->submit([this] { decoder.reset(); }).get();
            worker.reset();
        }
    };
    struct Hop {
        std::unique_ptr<Pinned> data;
        uint64_t generation = 0;
        bool occupied = false;
        explicit Hop(size_t bytes) : data(std::make_unique<Pinned>(bytes, true)) {}
        float *write(uint64_t gen) {
            if (occupied) throw std::logic_error("GLM: live layer hop overwritten");
            generation = gen; occupied = true;
            return (float *)data->p;
        }
        const float *read(uint64_t gen) const {
            if (!occupied || generation != gen) throw std::logic_error("GLM: stale layer hop generation");
            return (const float *)data->p;
        }
        void release(uint64_t gen) {
            (void)read(gen); occupied = false;
        }
    };
    std::shared_ptr<strata::core::ModelArtifact> model;
    std::vector<std::unique_ptr<Stage>> stages;
    std::vector<std::array<std::unique_ptr<Hop>, 2>> hops;
    std::unique_ptr<LayerCoordinatorAffinity> coordinator_affinity;
    GlmLayerOptions options;
    uint64_t generation = 0;
    int chunk = 1;
    Decoder::Checkpoint head_checkpoint;
    uint64_t proposed = 0, accepted = 0, draft_count = 0;
    uint64_t evaluation_rows = 0;
    std::vector<std::vector<uint64_t>> cache_history;
    static int pick(const std::vector<float> &logits) {
        if (logits.empty()) throw std::logic_error("GLM: missing tail logits");
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    int proposal(Decoder &decoder,int token,bool perturb=true) {
        const auto start=std::chrono::steady_clock::now();
        int next=pick(decoder.mtp_propose(token,true));
        stages.back()->draft_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(perturb && options.force_miss && ++draft_count%(unsigned)options.force_miss==0)
            next=(next+1)%(int)model->descriptor().vocab;
        return next;
    }
    void call(size_t i, const std::function<void(Decoder &)> &fn) {
        stages[i]->worker->submit([&, i] { fn(*stages[i]->decoder); }).get();
    }
    std::future<void> head(int token, Hop &out, uint64_t gen) {
        float *dst = out.write(gen);
        return stages[0]->worker->submit([this, token, dst] {
            const auto start=std::chrono::steady_clock::now();
            stages[0]->decoder->batch({token}, false, false, {}, nullptr, nullptr, dst);
            stages[0]->decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            ++stages[0]->decode_rows;
        });
    }
    std::future<void> tail(int token, Hop &in, uint64_t gen, std::vector<float> &logits) {
        const float *src = in.read(gen);
        return stages.back()->worker->submit([this, token, src, &logits] {
            const auto start=std::chrono::steady_clock::now();
            logits = stages.back()->decoder->batch({token}, false, false, {}, nullptr, src);
            stages.back()->decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            ++stages.back()->decode_rows;
        });
    }
  public:
    LayerDecoder(const std::string &path, GlmLayerOptions config) : options(std::move(config)) {
        if (std::filesystem::is_directory(path)) throw std::invalid_argument("GLM layer layout currently requires GGUF");
        // These paths own another GPU, remap/discard shared source pages, change
        // target arithmetic, or mutate residency inside an uncommitted window.
        for (const char *name : {"STRATA_GLM_LAYER_RANGE", "STRATA_GLM_STAGE_SERVE", "STRATA_GLM_STAGE_MEMORY_FD",
                "STRATA_GLM_STAGE_SOCKET_FD", "STRATA_GLM_MIXED_MEMORY_FD", "STRATA_GLM_MIXED_SOCKET_FD",
                "STRATA_GLM_DECODE_TP", "STRATA_GLM_KDA_SPLIT", "STRATA_GLM_MLA_SPLIT",
                "STRATA_GLM_MTP_BATCHED", "STRATA_GLM_TIER_ADAPT", "STRATA_GLM_REMOTE_TP",
                "STRATA_GLM_PREFER_RESIDENT_MIB", "STRATA_GLM_PRIMARY_GROUPS"})
            if (strata_env_is_set(name))
                throw std::invalid_argument(std::string("GLM layer layout is incompatible with ") + name);
        for (const char *name : {
                "STRATA_GLM_SPLIT_VERIFY", "STRATA_GLM_BATCH_SPLIT", "STRATA_GLM_DIRECT_WEIGHT_UPLOAD",
                "STRATA_GLM_Q2_NUMA_WEIGHTS", "STRATA_GLM_MAPPED_OWNED",
                "STRATA_GLM_RAM_TIER", "STRATA_GLM_RELEASE_GPU_DENSE_HOST",
                "STRATA_GLM_SPLIT_PROJECTIONS", "STRATA_GLM_RAM_MIB",
                "STRATA_GLM_PREFILL_ROUTE_BONUS", "STRATA_GLM_TIER_DIRECT",
                "STRATA_GLM_FLOATING_Q8", "STRATA_GLM_DENSE_Q4", "STRATA_GLM_ROUTE_AFFINITY"})
            if (strata_env_is_set(name) && !strata_env_equals(name, "0"))
                throw std::invalid_argument(std::string("GLM layer layout is incompatible with ") + name);
        if(options.memory!="interleave" && options.memory!="socket")throw std::invalid_argument("layer-memory must be interleave or socket");
        if(options.memory=="socket" && !options.lock)throw std::invalid_argument("layer-memory=socket requires locked COW model weights");
        if (options.pipeline != "none" && options.pipeline != "mtp") throw std::invalid_argument("layer-pipeline must be none or mtp");
        if (options.pipeline == "mtp" && (!options.mtp || options.depth != 1))
            throw std::invalid_argument("layer-pipeline=mtp needs speculative=mtp and draft-depth=1");
        if (options.mtp && options.depth != 1) throw std::invalid_argument("GLM layer MTP currently supports draft-depth=1");
        auto devices = configure_devices(options.devices, options.gpu_budget, STRATA_GLM_MAX_STAGES);
        if(options.emulate){
            if(options.emulate<2 || options.emulate>STRATA_GLM_MAX_STAGES || options.mtp || options.adapt || !options.plan.empty())
                throw std::invalid_argument("GLM: stage emulation is a 2..16-stage non-MTP parity diagnostic");
            const auto physical=devices;devices.resize(options.emulate);
            std::cerr<<"LAYER_EMULATION physical_devices="<<physical.size()<<" logical_stages="<<options.emulate<<'\n';
            for(size_t i=0;i<devices.size();++i)devices[i]=physical[i%physical.size()];
        }
        if (devices.size() < 2 || ((options.pipeline == "mtp" || options.check_pipeline) && devices.size() != 2))
            throw std::invalid_argument("GLM layer layout needs 2..16 GPUs; overlapped MTP needs exactly two");
        for(int device:devices){cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));std::cerr<<"LAYER_GPU device="<<device<<" name="<<props.name<<'\n';}
        model = std::make_shared<strata::core::ModelArtifact>(path);
        int bounds[STRATA_GLM_MAX_STAGES + 1];
        const auto split=layer_measured_split(options,path,devices);
        if (strata_glm_layer_bounds(split.c_str(), (int)model->descriptor().layers.size(), (int)devices.size(), bounds))
            throw std::invalid_argument("GLM: layer-split must be auto or strictly increasing first-layer cuts");
        if(!options.stage_cache_mib.empty() && (options.stage_cache_mib.size()!=devices.size() ||
                std::any_of(options.stage_cache_mib.begin(),options.stage_cache_mib.end(),[&](int mib){return mib<0 || size_t(mib)>options.gpu_budget/MiB;})))
            throw std::invalid_argument("GLM: layer-cache-mib needs one budget per stage within the GPU limit");
        if(options.pin_coordinator && !options.pin)throw std::invalid_argument("GLM: coordinator pinning requires CPU affinity");
        int coordinator=-1;
        auto placement = layer_cpu_placement(devices, options.threads,options.stage_workers,&coordinator);
        std::cerr << "LAYER_CPU pinned=" << options.pin << '\n';
        strata::prefill::Gemm::lossy_allowed = false;
        k::glm_set_prefill_lossy(false);
        chunk = options.width ? options.width : 256;
        // Sequential construction also fixes the shared staging geometry before
        // any stage executes. Each submission thread owns all CUDA resource use.
        std::map<int,std::shared_ptr<GpuWorker>> submitters;
        for (size_t i = 0; i < devices.size(); ++i) {
            auto &worker=submitters[devices[i]];
            if(!worker)worker=std::make_shared<GpuWorker>(devices[i],true,options.pin?placement[i].host_core:-1,options.pin);
            stages.push_back(std::make_unique<Stage>(devices[i],worker));
        }
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->worker->submit([&, i] {
                stages[i]->decoder = std::make_unique<Decoder>(path, options.context, (size_t)options.dense * MiB,
                    (int)placement[i].worker_cores.size(), 0, options.pin, "", "", "auto", model,
                    bounds[i], bounds[i + 1], &placement[i]);
            }).get();
            std::cerr << "LAYER_STAGE stage=" << i << " device=" << devices[i] << " first=" << bounds[i] << " last=" << bounds[i+1] << '\n';
        }
        // Shared COW mapping: one touch/lock pass, including the draft block.
        if (options.lock) {
            const size_t hop_bytes=(stages.size()-1)*2*(size_t)chunk*model->descriptor().hidden*4*sizeof(float);
            layer_lock_budget(*model,options.mtp,hop_bytes,stages.size());
            if(options.memory=="socket")for(size_t i=0;i<stages.size();++i)call(i,[&,i](Decoder &d) {
                LayerWeightPolicy policy(placement[i].host_core);
                d.warm_weights(true,{},options.mtp,[&](const std::string &name) {
                    return strata_glm_layer_tensor_owned(name.c_str(),bounds[i],bounds[i+1],(int)model->descriptor().layers.size(),options.mtp);
                });
            });
            else call(0, [&](Decoder &d) { LayerWeightPolicy policy; d.warm_weights(true, {}, options.mtp); });
        }
        for (size_t i = 0; i < stages.size(); ++i) call(i, [&](Decoder &d) {
            if (options.mtp && i + 1 == stages.size()) { d.enable_mtp_capture(); d.set_mtp_cpu_experts(options.draft_cpu); }
            chunk = std::min(chunk, d.enable_gpu(chunk, options.gpu_budget, options.pipeline == "mtp" || options.check_pipeline));
            d.configure_prefill({devices[i]}, options.gpu_budget, 0, false, false);
            d.set_decode_cache_budget(options.stage_cache_mib.empty()?options.cache:size_t(options.stage_cache_mib[i])*MiB);
            d.set_decode_cache_window(options.cache_window);
            d.enable_layer_cache_windows(options.adapt);
            d.set_decode_graphs(options.graphs);
        });
        for (size_t i = 1; i < stages.size(); ++i) {
            std::array<std::unique_ptr<Hop>, 2> pair;
            for (auto &hop : pair) hop = std::make_unique<Hop>((size_t)chunk * model->descriptor().hidden * 4 * sizeof(float));
            hops.push_back(std::move(pair));
        }
        if (options.pipeline == "mtp" || options.check_pipeline) call(0, [&](Decoder &d) { head_checkpoint = d.checkpoint(); });
        if(options.pin_coordinator)coordinator_affinity=std::make_unique<LayerCoordinatorAffinity>(coordinator);
    }
    ~LayerDecoder() { stages.clear(); }
    int width() const { return chunk; }
    int vocabulary() const { return (int)model->descriptor().vocab; }
    bool has_decode_cache() const { return options.stage_cache_mib.empty()?options.cache!=0:
        std::any_of(options.stage_cache_mib.begin(),options.stage_cache_mib.end(),[](int n){return n>0;}); }
    void prepare_decode_cache() { prepare({}); }
    void flush_expert_observer() {
        for(size_t i=0;i<stages.size();++i)call(i,[](Decoder &d){d.flush_expert_observer();});
    }
    void report_gpu() { report(); }
    void after_evaluation_batch(size_t rows) {
        const auto previous=evaluation_rows; evaluation_rows+=rows;
        if(options.adapt && previous/(unsigned)options.cache_window != evaluation_rows/(unsigned)options.cache_window)
            rebalance();
    }
    void reset() {
        for (size_t i = 0; i < stages.size(); ++i) call(i, [](Decoder &d) { d.reset(); });
        for (auto &pair : hops) for (auto &h : pair) h->occupied = false;
        proposed = accepted = draft_count = 0;
        evaluation_rows = 0;
        cache_history.clear();
        for(auto &s:stages){s->decode_rows=0;s->decode_ms=s->draft_ms=0;}
    }
    void set_cancel(const std::function<bool()> &cancel) {
        for (size_t i = 0; i < stages.size(); ++i) call(i, [&](Decoder &d) { d.set_cancel(cancel); });
    }
    std::vector<float> batch(const std::vector<int> &tokens, bool prefill = true, bool all_logits = false,
                            const std::function<void(const float *, int, int)> &consume_logits = {}) {
        const uint64_t gen = ++generation;
        std::vector<float> logits;
        for (size_t i = 0; i < stages.size(); ++i) {
            const float *src = i ? hops[i-1][gen & 1]->read(gen) : nullptr;
            float *dst = i+1 < stages.size() ? hops[i][gen & 1]->write(gen) : nullptr;
            call(i, [&](Decoder &d) {
                const auto start=std::chrono::steady_clock::now();
                logits = d.batch(tokens, prefill, all_logits,
                    i+1==stages.size()?consume_logits:std::function<void(const float *,int,int)>{}, nullptr, src, dst);
                if(!prefill){stages[i]->decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();stages[i]->decode_rows+=tokens.size();}
            });
            // batch completes the stream before returning: its future is a host
            // publication fence stronger than an event wait on a pending copy.
            if (i) hops[i-1][gen & 1]->release(gen);
        }
        return logits;
    }
    std::vector<float> prefill(const std::vector<int> &tokens,
                              const std::function<void(size_t)> &progress={}) {
        struct Backend {
            LayerDecoder &owner;
            std::vector<std::vector<int>> panels;
            std::vector<uint64_t> generations;
            std::vector<float> logits;
            const std::function<void(size_t)> &progress;
            std::future<void> start(size_t stage,size_t panel) {
                const uint64_t gen=generations[panel];
                const float *src=stage?owner.hops[stage-1][gen&1]->read(gen):nullptr;
                float *dst=stage+1<owner.stages.size()?owner.hops[stage][gen&1]->write(gen):nullptr;
                return owner.stages[stage]->worker->submit([this,stage,panel,src,dst] {
                    auto result=owner.stages[stage]->decoder->batch(panels[panel],true,false,{},nullptr,src,dst);
                    if(stage+1==owner.stages.size())logits=std::move(result);
                });
            }
            void complete(size_t stage,size_t panel) {
                const uint64_t gen=generations[panel];
                if(stage)owner.hops[stage-1][gen&1]->release(gen);
                if(stage+1==owner.stages.size() && progress)
                    progress(std::min((panel+1)*(size_t)owner.chunk,total()));
            }
            size_t total()const {return (panels.size()-1)*(size_t)owner.chunk+panels.back().size();}
        } backend{*this,{},{},{},progress};
        for(size_t t=0;t<tokens.size();t+=chunk) {
            const size_t end=std::min(tokens.size(),t+chunk);
            backend.panels.emplace_back(tokens.begin()+t,tokens.begin()+end);
            backend.generations.push_back(++generation);
        }
        strata::spec::run_layer_prefill(backend,backend.panels.size(),stages.size());
        return backend.logits;
    }
    void prepare(const std::vector<int> &prompt) {
        if (options.mtp) call(stages.size()-1, [&](Decoder &d) { d.prepare_mtp(prompt); });
        for (size_t i=0;i<stages.size();++i) call(i, [](Decoder &d) {
            d.prepare_decode_cache(); d.verify_layer_cache(); d.freeze_cache(); d.reset_decode_stats();
        });
    }
    void rebalance() {
        std::vector<uint64_t> fingerprints(stages.size());std::vector<std::future<void>> tasks;
        try{
            for(size_t i=0;i<stages.size();++i)tasks.push_back(stages[i]->worker->submit([&,i]{
                auto &d=*stages[i]->decoder;d.rebalance_layer_cache();fingerprints[i]=d.layer_cache_fingerprint();
            }));
        }catch(...){for(auto &task:tasks)if(task.valid())task.wait();throw;}
        std::exception_ptr error;
        for(auto &task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
        cache_history.push_back(std::move(fingerprints));
    }
    void report() {
        for(size_t i=0;i<stages.size();++i) call(i,[&](Decoder &d) {
            std::cerr << "LAYER_WORK stage=" << i << " device=" << stages[i]->device
                      << " decode_rows=" << stages[i]->decode_rows << " decode_wall_ms=" << stages[i]->decode_ms
                      << " draft_wall_ms=" << stages[i]->draft_ms << " scope=host_submission_through_completion\n";
            d.report_gpu();d.report_cache();
        });
    }
    void check_serial(const std::vector<int> &prompt) {
        DeviceScope scope(stages.front()->device);
        // The reference uses the same shared mapping and unmodified full-layer
        // arithmetic. Both arms disable lossy prefill and use the same panels.
        Decoder reference("", options.context, (size_t)options.dense*MiB, 1, 0, false,
                          "", "", "auto", model, 0, model->descriptor().layers.size());
        reference.enable_gpu(chunk,options.gpu_budget);
        reference.configure_prefill({stages.front()->device},options.gpu_budget,0,false,false);
        reset();
        std::vector<float> actual,expected;
        auto compare=[&] {
            if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))
                throw std::runtime_error("GLM: serial layer logits differ from full-layer reference");
            const auto full=reference.snapshot();
            for(size_t i=0;i<stages.size();++i) call(i,[&](Decoder &d) {
                const auto part=d.snapshot();
                if(part.position!=full.position)throw std::runtime_error("GLM: stage position differs from full-layer reference");
                for(size_t l=0;l<model->descriptor().layers.size();++l)for(size_t b=0;b<8;++b)
                    if(!part.layers[l][b].empty() && part.layers[l][b]!=full.layers[l][b])
                        throw std::runtime_error("GLM: layer state differs from full-layer reference at layer "+std::to_string(l));
            });
        };
        for(size_t t=0;t<prompt.size();t+=chunk) {
            size_t end=std::min(prompt.size(),t+chunk);
            std::vector<int> panel(prompt.begin()+t,prompt.begin()+end);
            expected=reference.batch(panel);
        }
        actual=prefill(prompt);compare();
        for(int i=0;i<8 && prompt.size()+i<(size_t)options.context;++i) {
            int token=pick(expected);
            actual=batch({token},false);expected=reference.step(token);compare();
        }
        std::cerr << "LAYER_SERIAL_PARITY logits=bitwise_equal state=bitwise_equal decode_steps="
                  << std::min<size_t>(8,(size_t)options.context-prompt.size()) << '\n';
        reset();
    }
    void check_pipeline(const std::vector<int> &prompt,int count) {
        if(!options.mtp || stages.size()!=2)throw std::invalid_argument("pipeline check needs two GPUs and speculative=mtp");
        const std::string original=options.pipeline;
        for(size_t i=0;i<stages.size();++i)call(i,[](Decoder &d){d.set_layer_cache_checks(true);});
        std::vector<int> expected,actual;
        std::vector<Decoder::Snapshot> saved;
        options.pipeline="none";reset();auto logits=prefill(prompt);prepare(prompt);
        generate(std::move(logits),count,[&](int token){expected.push_back(token);});
        std::vector<uint64_t> route_fingerprints;
        const auto expected_history=cache_history;
        for(size_t i=0;i<stages.size();++i)call(i,[&](Decoder &d){saved.push_back(d.snapshot());route_fingerprints.push_back(d.layer_routes_fingerprint());});
        options.pipeline="mtp";reset();logits=prefill(prompt);prepare(prompt);
        generate(std::move(logits),count,[&](int token){actual.push_back(token);});
        if(actual!=expected)throw std::runtime_error("GLM: layer MTP tokens differ from serial control");
        if(cache_history!=expected_history)throw std::runtime_error("GLM: layer MTP cache policy differs from serial control");
        for(size_t i=0;i<stages.size();++i)call(i,[&](Decoder &d){
            const auto state=d.snapshot();const auto &reference=saved[i];
            if(d.layer_routes_fingerprint()!=route_fingerprints[i])throw std::runtime_error("GLM: speculative routes leaked into committed statistics");
            if(state.position!=reference.position || state.mtp_position!=reference.mtp_position)
                throw std::runtime_error("GLM: layer MTP frontiers differ");
            for(size_t l=0;l<state.layers.size();++l)for(size_t b=0;b<8;++b){
                const auto &a=state.layers[l][b],&e=reference.layers[l][b];
                size_t visible=a.size();
                const int position=l==model->descriptor().layers.size()?state.mtp_position:state.position;
                if(b==4)visible=std::min(visible,(size_t)position*model->descriptor().kv_rank);
                if(b==7)visible=std::min(visible,(size_t)(position/model->descriptor().index_pool)*model->descriptor().index_dim);
                if(a.size()!=e.size() || !std::equal(a.begin(),a.begin()+visible,e.begin()))
                    throw std::runtime_error("GLM: layer MTP committed state differs at layer "+std::to_string(l)+" buffer "+std::to_string(b));
            }
        });
        std::cerr << "LAYER_PIPELINE_PARITY tokens=" << actual.size() << " token_ids=equal committed_state=bitwise_equal forced_miss_every=" << options.force_miss << '\n';
        options.pipeline=original;reset();
        for(size_t i=0;i<stages.size();++i)call(i,[](Decoder &d){d.set_layer_cache_checks(false);});
    }
    int generate(std::vector<float> logits, int count, const std::function<void(int)> &emit) {
        int current = pick(logits), generated = 0;
        auto output = [&](int token) {
            emit(token); ++generated;
            return generated == count || std::find(options.stops.begin(), options.stops.end(), token) != options.stops.end();
        };
        if (output(current)) return generated;
        if (!options.mtp) {
            while (generated < count) {
                logits = batch({current}, false); current = pick(logits); if (output(current)) break;
                if(options.adapt && (generated-1)%(unsigned)options.cache_window==0)rebalance();
            }
            return generated;
        }
        // Serialized MTP is the exact reference: same draft computations, but
        // no work is issued to the head before the target token is confirmed.
        if (options.pipeline == "none") {
            while (generated < count) {
                call(stages.size()-1, [&](Decoder &d) { (void)proposal(d,current,false); });
                logits = batch({current}, false); current = pick(logits); if (output(current)) break;
                if(options.adapt && (generated-1)%(unsigned)options.cache_window==0)rebalance();
            }
            return generated;
        }
        struct Backend {
            LayerDecoder &owner;
            std::vector<float> logits;
            std::future<void> start_head(int token, int slot, uint64_t gen) {
                return owner.head(token, *owner.hops[0][slot], gen);
            }
            int draft(int token) {
                int result=-1;
                owner.call(1, [&](Decoder &d) {
                    result=owner.proposal(d,token);
                });
                return result;
            }
            std::future<void> start_draft(int token, int &result) {
                return owner.stages[1]->worker->submit([this,token,&result] {
                    result=owner.proposal(*owner.stages[1]->decoder,token);
                });
            }
            bool window_due(uint64_t consumed)const {return owner.options.adapt && consumed%(unsigned)owner.options.cache_window==0;}
            void rebalance(){owner.rebalance();}
            void save_head() { owner.call(0,[&](Decoder &d){d.save(owner.head_checkpoint);}); }
            void restore_head() { owner.call(0,[&](Decoder &d){d.restore(owner.head_checkpoint);}); }
            void release(int slot,uint64_t gen) { owner.hops[0][slot]->release(gen); }
            int confirm(int token,int slot,uint64_t gen) {
                owner.tail(token,*owner.hops[0][slot],gen,logits).get();
                release(slot,gen); return pick(logits);
            }
        } backend{*this,{}};
        const auto stats=strata::spec::run_layer_pipeline(backend,current,output);
        proposed+=stats.proposed; accepted+=stats.accepted;
        std::cerr << "LAYER_MTP proposed=" << proposed << " accepted=" << accepted << " depth=1\n";
        return generated;
    }
};

static void layer_binary_dump(const std::string &path, const std::vector<float> &values) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary);
    out.write((const char *)values.data(), values.size() * sizeof(float));
    if (!out) throw std::runtime_error("GLM: layer logits write failed");
}

static int serve_layers(const std::string &path, GlmLayerOptions config) {
    LayerDecoder decoder(path, config);
    struct Request { std::string line; std::shared_ptr<std::atomic<bool>> cancel; };
    std::mutex mutex; std::condition_variable ready; std::deque<Request> requests;
    std::shared_ptr<std::atomic<bool>> active;
    bool ended = false;
    std::jthread reader([&] {
        for (std::string line; std::getline(std::cin, line);) {
            std::lock_guard lock(mutex);
            if (line == "STOP") {
                if (active) active->store(true);
                else if (!requests.empty()) requests.front().cancel->store(true);
                continue;
            }
            bool quit = line == "QUIT";
            requests.push_back({std::move(line), std::make_shared<std::atomic<bool>>(false)});
            ready.notify_one(); if (quit) break;
        }
        std::lock_guard lock(mutex); ended = true; ready.notify_one();
    });
    using Clock = std::chrono::steady_clock;
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b-a).count(); };
    std::cout << "INFO engine=glm-layer-experimental sampling=greedy\nREADY " << config.context << " stop prefill-includes-reset\n" << std::flush;
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex); ready.wait(lock, [&] { return ended || !requests.empty(); });
            if (requests.empty()) break;
            request = std::move(requests.front()); requests.pop_front(); active = request.cancel;
        }
        if (request.line == "QUIT") break;
        int sent = 0; size_t prompt_count = 0;
        // Publish the idle frontier before DONE/ERR: the client may immediately
        // queue another GEN and STOP after reading the terminal response.
        auto finish_active = [&] { std::lock_guard lock(mutex); active.reset(); };
        try {
            std::istringstream input(request.line); std::string op, ids, extra; int count = 0;
            if (!(input >> op >> count >> ids) || op != "GEN" || input >> extra) throw std::invalid_argument("expected GEN <max_new> <IDs>");
            auto tokens = token_ids(ids); prompt_count = tokens.size();
            if (tokens.empty() || count < 1 || tokens.size() + (size_t)count > (size_t)config.context)
                throw std::invalid_argument("request exceeds context or has invalid token counts");
            auto start = Clock::now(); decoder.reset();
            decoder.set_cancel([cancel = request.cancel] { return cancel->load(); });
            std::vector<float> logits;
            const auto prompt_start = Clock::now();
            logits=decoder.prefill(tokens,[&](size_t end) {
                if (request.cancel->load()) throw std::runtime_error("cancelled");
                double elapsed = ms(prompt_start, Clock::now());
                std::cout << "PP " << end << ' ' << tokens.size() << ' ' << elapsed << ' ' << (elapsed>0?end*1000.0/elapsed:0) << '\n' << std::flush;
            });
            const auto preparing=Clock::now();
            decoder.prepare(tokens);
            std::cerr << "RESPONSE_PREPARE kind=decode_cache ms=" << ms(preparing,Clock::now()) << '\n';
            const auto prefilled = Clock::now(); std::string finish = "length";
            decoder.generate(std::move(logits), count, [&](int token) {
                if (request.cancel->load()) throw std::runtime_error("cancelled");
                ++sent; std::cout << "T " << token << '\n' << std::flush;
                if (std::find(config.stops.begin(), config.stops.end(), token) != config.stops.end()) finish = "stop";
            });
            finish_active();
            std::cout << "DONE " << sent << ' ' << tokens.size() << ' ' << ms(start,prefilled) << ' ' << ms(prefilled,Clock::now()) << ' ' << finish << " 0 0 0\n" << std::flush;
        } catch (const std::exception &e) {
            decoder.set_cancel({});
            try { decoder.reset(); } catch (...) { std::cerr << "GLM: layer reset failed\n"; std::_Exit(1); }
            finish_active();
            if (request.cancel->load() && std::string(e.what()) == "cancelled")
                std::cout << "DONE " << sent << ' ' << prompt_count << " 0 0 cancel 0 0 0\n" << std::flush;
            else std::cout << "ERR " << e.what() << '\n' << std::flush;
        }
        decoder.set_cancel({});
        decoder.report();
    }
    return 0;
}

static std::pair<bool,int> layer_entry(int &argc, char **argv) {
    std::string layout = "legacy";
    for (int i=1;i<argc;++i) if (std::string(argv[i]).starts_with("--gpu-layout=")) layout=std::string(argv[i]).substr(13);
    if (layout != "legacy" && layout != "layer") throw std::invalid_argument("gpu-layout must be legacy or layer");
    if (layout == "legacy") {
        for (int i=1;i<argc;) {
            const std::string flag=argv[i];
            if (flag.starts_with("--layer-split=") || flag.starts_with("--layer-pipeline="))
                throw std::invalid_argument("GLM layer options require --gpu-layout=layer");
            if (flag.starts_with("--gpu-layout=")) { for(int j=i;j<argc;++j) argv[j]=argv[j+1]; --argc; }
            else ++i;
        }
        return {false,0};
    }
    const bool serving = std::string(argv[1]) == "--serve";
    GlmLayerOptions o;o.executable=argv[0];
    std::string path;
    int named = 6;
    auto nonnegative = [](const std::string &s) {
        size_t used=0; int n=std::stoi(s,&used);
        if (used!=s.size() || n<0) throw std::invalid_argument("GLM: expected a nonnegative integer");
        return n;
    };
    auto counts = [&](const std::string &text) {
        std::vector<int> values;size_t start=0;
        for(;;){
            const auto end=text.find(',',start);const auto word=text.substr(start,end-start);
            if(word.empty() || !std::all_of(word.begin(),word.end(),[](unsigned char c){return c>='0'&&c<='9';}))
                throw std::invalid_argument("GLM: layer budgets require comma-separated nonnegative integers");
            values.push_back(nonnegative(word));
            if(end==std::string::npos)break;
            start=end+1;
        }
        return values;
    };
    if (serving) {
        if (argc < 29) throw std::invalid_argument("GLM layer serve requires all legacy positional fields before named options");
        path=argv[2]; o.context=nonnegative(argv[3]); o.dense=nonnegative(argv[4]); o.threads=nonnegative(argv[5]);
        o.width=prefill_width(argv[7]); o.stops=token_ids(argv[8]); o.gpu_budget=gpu_budget_bytes(argv[9]);
        o.mtp=std::string(argv[12])=="mtp"; o.depth=nonnegative(argv[13]); o.pin=std::string(argv[14])!="none";
        o.devices=argv[16]; o.lock=std::string(argv[18])=="1"; o.cache=cache_setting(argv[20]).first;
        o.graphs=std::string(argv[21])=="1"; o.cache_window=nonnegative(argv[22]);o.adapt=std::string(argv[23])=="1";
        if (std::string(argv[6])!="0" || std::string(argv[10])!="0" || std::string(argv[11])!="0" ||
            std::string(argv[15])!="0" || std::string(argv[17])!="0" || std::string(argv[19])!="0" ||
            std::string(argv[24])!="mmq" || std::string(argv[26])!="" ||
            std::string(argv[27])!="" || std::string(argv[28])!="auto" ||
            (std::string(argv[20])=="auto" || std::string(argv[20])=="extend"))
            throw std::invalid_argument("GLM layer recipe requires fixed decode cache, mmq prefill, native assembled GGUF and no legacy helper options");
        if (std::string(argv[12])!="none" && !o.mtp) throw std::invalid_argument("GLM layer speculation must be none or mtp");
        named=29;
    } else {
        if (argc<6) throw std::invalid_argument("GLM layer CLI requires model, IDs, steps, dense MiB and threads");
        path=argv[1]; o.steps=nonnegative(argv[3]); o.dense=nonnegative(argv[4]); o.threads=nonnegative(argv[5]);
    }
    for (int i=named;i<argc;++i) {
        std::string f=argv[i];
        if (f=="--gpu-layout=layer") {}
        else if (f.starts_with("--gpu-devices=")) o.devices=f.substr(14);
        else if (f.starts_with("--layer-split=")) o.split=f.substr(14);
        else if (f.starts_with("--layer-pipeline=")) o.pipeline=f.substr(17);
        else if (f.starts_with("--layer-memory=")) o.memory=f.substr(15);
        else if (f.starts_with("--layer-plan=")) o.plan=f.substr(13);
        else if (f.starts_with("--layer-workers=")) o.stage_workers=counts(f.substr(16));
        else if (f.starts_with("--layer-cache-mib=")) o.stage_cache_mib=counts(f.substr(18));
        else if (f=="--layer-pin-coordinator") o.pin_coordinator=true;
        else if (f.starts_with("--context=")) o.context=nonnegative(f.substr(10));
        else if (f.starts_with("--prefill-batch=")) o.width=prefill_width(f.substr(16));
        else if (f.starts_with("--gpu-budget-mib=")) o.gpu_budget=gpu_budget_bytes(f.substr(17));
        else if (f.starts_with("--decode-cache-mib=")) o.cache=(size_t)nonnegative(f.substr(19))*MiB;
        else if (f=="--speculative=mtp") o.mtp=true;
        else if (f=="--speculative=none") o.mtp=false;
        else if (f.starts_with("--draft-depth=")) o.depth=nonnegative(f.substr(14));
        else if (f=="--lock-weights") o.lock=true;
        else if (f=="--decode-graphs") o.graphs=true;
        else if (f=="--decode-cache-adapt") o.adapt=true;
        else if (f=="--cpu-affinity=none") o.pin=false;
        else if (f=="--cpu-affinity=numa" || f=="--cpu-affinity=auto") o.pin=true;
        else if (f.starts_with("--stop-ids=")) o.stops=token_ids(f.substr(11));
        else if (f.starts_with("--dump-logits=")) throw std::invalid_argument("GLM layer path: use --dump-prefill-logits for reference logits");
        else if (f.starts_with("--dump-prefill-logits=")) o.prefill_dump=f.substr(22);
        else if (f.starts_with("--eval-corpus=")) o.eval_corpus=f.substr(14);
        else if (f.starts_with("--eval-save-logits=")) o.eval_save=f.substr(19);
        else if (f.starts_with("--eval-reference=")) o.eval_reference=f.substr(17);
        else if (f=="--eval-prefill") o.eval_prefill=true;
        else if (f.starts_with("--bench=")) o.repeats=nonnegative(f.substr(8));
        else if (f=="--check-layer-parity") o.check=true;
        else if (f.starts_with("--check-layer-stages=")) {
            o.emulate=nonnegative(f.substr(21));o.check=true;
            if(o.emulate<2 || o.emulate>STRATA_GLM_MAX_STAGES)throw std::invalid_argument("GLM: emulated stage count must be 2..16");
        }
        else if (f=="--check-layer-pipeline") o.check_pipeline=true;
        else if (f.starts_with("--layer-force-miss-every=")) o.force_miss=nonnegative(f.substr(25));
        else if (f.starts_with("--decode-cache-window=")) o.cache_window=nonnegative(f.substr(22));
        else if (f=="--lookup-depth=0" || f=="--decode-experts=cpu" || f=="--expert-cache-mib=0" ||
                 f=="--cpu-prepack-mib=0" || f=="--prefill-expert-cache-mib=0" || f=="--prefill-experts=mmq") {}
        else if (f=="--mtp-experts=cpu") o.draft_cpu=true;
        else if (f=="--mtp-experts=gpu") o.draft_cpu=false;
        else if (f=="--gpu-prefill") {}
        else throw std::invalid_argument("unsupported GLM layer option: "+f);
    }
    if (o.context<1 || o.dense<64 || o.threads<1 || o.steps<1 || o.repeats<1 || o.cache>o.gpu_budget ||
        o.cache_window>8192 || (o.adapt && ((!o.cache && std::none_of(o.stage_cache_mib.begin(),o.stage_cache_mib.end(),[](int n){return n>0;})) || o.cache_window<1)))
        throw std::invalid_argument("GLM: invalid layer configuration");
    if(serving && (o.check || o.check_pipeline))throw std::invalid_argument("GLM: parity diagnostics use the generation CLI");
    if (!o.eval_corpus.empty() || !o.eval_save.empty() || !o.eval_reference.empty() || o.eval_prefill) {
        if(serving || o.eval_corpus.empty() || o.mtp || o.pipeline!="none" || o.check || o.check_pipeline)
            throw std::invalid_argument("GLM: layer corpus evaluation requires ordinary target decode and --eval-corpus");
        if(o.adapt && strata_env_value_ll("STRATA_GLM_EVAL_WIDTH",4)!=1)
            throw std::invalid_argument("GLM: adaptive layer corpus evaluation requires STRATA_GLM_EVAL_WIDTH=1");
        LayerDecoder decoder(path,o);
        evaluate_corpus(decoder,o.eval_corpus,o.eval_save,o.eval_reference,o.eval_prefill?decoder.width():0);
        return {true,0};
    }
    if (serving) return {true,serve_layers(path,o)};
    std::string ids=argv[2];
    if (ids.starts_with("@")) {
        std::ifstream file(ids.substr(1)); if (!file) throw std::runtime_error("cannot read token file");
        ids.assign(std::istreambuf_iterator<char>(file),{});
    }
    const auto tokens=token_ids(ids);
    if (tokens.empty() || tokens.size()+(size_t)o.steps>(size_t)o.context) throw std::invalid_argument("GLM layer request exceeds context");
    LayerDecoder decoder(path,o);
    if(o.check) {decoder.check_serial(tokens);return {true,0};}
    if(o.check_pipeline) {decoder.check_pipeline(tokens,o.steps);return {true,0};}
    using Clock=std::chrono::steady_clock;
    for(int r=0;r<o.repeats;++r) {
        auto start=Clock::now(); decoder.reset(); std::vector<float> logits;
        logits=decoder.prefill(tokens);
        layer_binary_dump(o.prefill_dump,logits);
        decoder.prepare(tokens); auto prefilled=Clock::now();
        std::vector<int> output;
        int n=decoder.generate(logits,o.steps,[&](int token){output.push_back(token);});
        auto end=Clock::now();
        const double pp=std::chrono::duration<double,std::milli>(prefilled-start).count();
        const double tg=std::chrono::duration<double,std::milli>(end-prefilled).count();
        for(int token:output)std::cout<<token<<'\n';
        std::cout<<std::flush;
        std::cerr << "PREFILL tokens=" << tokens.size() << " ms=" << pp << " tok_s=" << tokens.size()*1000.0/pp << '\n'
                  << "DECODE steps=" << n-1 << " ms=" << tg << " tok_s=" << (tg>0?(n-1)*1000.0/tg:0) << '\n';
        decoder.report();
    }
    return {true,0};
}

} // namespace
