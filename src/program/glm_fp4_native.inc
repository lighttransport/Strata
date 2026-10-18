// Native ModelOpt execution. Included after the GLM GPU allocation helpers.
// One owner shares read-only checkpoint mappings across the two device streams.
#ifndef STRATA_USE_HIP
static bool fp4_sm70_kernel = false;
static bool fp4_kernel_selected = false;
class Fp4NativeExecutor {
    using Format=strata::artifact::ModelOptFormat;
    using Packed=strata::artifact::ModelOptPacked;
    static constexpr size_t expert_bytes=3*(size_t(4096)*2048/2+size_t(4096)*2048/16);
    struct RegisteredRam {
        int device;
        std::vector<std::pair<void*,size_t>> ranges;
        RegisteredRam(const strata::artifact::Safetensors& tensors,const std::vector<int>& devices):device(devices.front()) {
#ifdef __linux__
            DeviceScope scope(device);
            for(int id:devices) {
                int supported=0;check(cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterReadOnlySupported,id));
                if(!supported)throw std::runtime_error("GLM FP4: GPU cannot register read-only RAM weights");
            }
            const auto start=std::chrono::steady_clock::now();size_t bytes=0;
            const auto mappings=tensors.ram_mappings();ranges.reserve(mappings.size());
            const size_t page=sysconf(_SC_PAGESIZE);
            try {
                for(auto [pointer,size]:mappings) {
                    check(cudaHostRegister(pointer,(size+page-1)/page*page,cudaHostRegisterPortable|cudaHostRegisterReadOnly));
                    ranges.emplace_back(pointer,size);bytes+=size;
                }
            }catch(...) {for(auto [p,n]:ranges)cudaHostUnregister(p);throw;}
            std::cerr<<"FP4_HOST_DMA registered_bytes="<<bytes<<" shards="<<ranges.size()<<" ms="
                <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
#else
            throw std::runtime_error("GLM FP4: read-only RAM registration requires Linux");
#endif
        }
        ~RegisteredRam(){
            const auto start=std::chrono::steady_clock::now();
            int previous=device;cudaGetDevice(&previous);cudaSetDevice(device);cudaDeviceSynchronize();
            for(auto [p,n]:ranges)cudaHostUnregister(p);
            cudaSetDevice(previous);
            std::cerr<<"FP4_HOST_DMA_RELEASE shards="<<ranges.size()<<" ms="
                <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
        }
    };
    struct Matrix {
        k::GlmPackedFormat format;
        int rows=0,columns=0;
        float global=1;
        size_t scale_offset=0;
        std::unique_ptr<Device> data;
        const void* scales()const {return static_cast<const char*>(data->p)+scale_offset;}
    };
    struct Expert {
        std::array<Matrix,3> projection;
        size_t bytes=0;
        uint64_t use=0;
    };
    struct ExpertSource {
        int rows=0,columns=0;
        float global=1;
        bool csf=false;
        std::array<const void*,6> data{};
        std::array<size_t,6> bytes{};
    };
    struct CpuExpert {
        std::array<Packed,3> projection;
        std::array<std::unique_ptr<cpu::GlmFp4Csf32Rows>,3> csf32;
        std::array<std::unique_ptr<cpu::GlmFp4Csf32SparseRows>,3> sparse;
        alignas(64) std::array<std::array<float,cpu::glm_fp4_weight_table_floats>,3> scales;
        size_t bytes=0;
        std::list<std::pair<int,int>>::iterator order;
    };
    struct Context {
        int device;
        cudaStream_t stream=nullptr;
        cublasHandle_t blas=nullptr;
        std::thread worker;
        std::mutex work_mutex;
        std::condition_variable wake;
        std::packaged_task<void()> task;
        bool stop=false;
        std::map<std::pair<int,int>,Expert> experts;
        std::unique_ptr<Expert> temporary;
        std::map<int,std::set<int>> keep;
        std::map<int,std::vector<uint64_t>> frequency;
        std::map<std::string,Matrix> dense;
        std::map<std::string,std::unique_ptr<Device>> scratch;
        cudaStream_t copy_stream=nullptr;
        static constexpr size_t upload_capacity=16*MiB;
        std::array<std::unique_ptr<Pinned>,2> upload_host;
        std::array<cudaEvent_t,2> upload_ready{};
        std::unique_ptr<Device> upload_device;
        std::set<std::pair<int,int>> protected_experts;
        int upload_slot=0;
        size_t cache_bytes=0,cache_limit=0,prefill_cache_limit=0;
        uint64_t clock=0,hits=0,misses=0,uploaded=0,mma_calls=0,reference_calls=0,prefill_calls=0;
        explicit Context(int device):device(device) {
            DeviceScope scope(device);
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cublasCreate(&blas));
            check(cublasSetStream(blas,stream));check(cublasSetMathMode(blas,CUBLAS_TENSOR_OP_MATH));
            std::promise<void> ready;auto started=ready.get_future();
            worker=std::thread([this,&ready] {
                std::unique_ptr<DeviceScope> scope;
                std::unique_ptr<GpuLocality> locality;
                try {scope=std::make_unique<DeviceScope>(this->device);locality=std::make_unique<GpuLocality>();}
                catch(...){ready.set_exception(std::current_exception());return;}
                ready.set_value();
                for(;;) {
                    std::unique_lock lock(work_mutex);wake.wait(lock,[&]{return stop||task.valid();});
                    if(stop)return;
                    auto job=std::move(task);lock.unlock();job();
                }
            });
            try {started.get();}catch(...){worker.join();if(blas)cublasDestroy(blas);if(stream)cudaStreamDestroy(stream);throw;}
        }
        ~Context(){
            {std::lock_guard lock(work_mutex);stop=true;}wake.notify_one();if(worker.joinable())worker.join();
            DeviceScope scope(device);if(stream)cudaStreamSynchronize(stream);if(copy_stream)cudaStreamSynchronize(copy_stream);
            temporary.reset();experts.clear();dense.clear();scratch.clear();release_upload();
            for(auto e:upload_ready)if(e)cudaEventDestroy(e);
            if(copy_stream)cudaStreamDestroy(copy_stream);
            if(blas)cublasDestroy(blas);
            if(stream)cudaStreamDestroy(stream);
        }
        std::future<void> submit(std::function<void()> job) {
            std::lock_guard lock(work_mutex);
            if(task.valid()||stop)throw std::logic_error("GLM FP4: GPU worker already has a pending task");
            task=std::packaged_task<void()>(std::move(job));auto done=task.get_future();wake.notify_one();return done;
        }
        Device& buffer(const std::string& name,size_t bytes) {
            auto& b=scratch[name];
            if(!b||b->bytes<bytes) {
                check(cudaStreamSynchronize(stream));b.reset();room(std::max<size_t>(bytes,256));
                b=std::make_unique<Device>(std::max<size_t>(bytes,256));
            }
            return *b;
        }
        void evict() {
            auto victim=experts.end();
            for(auto it=experts.begin();it!=experts.end();++it)
                if(!protected_experts.count(it->first)&&(victim==experts.end()||it->second.use<victim->second.use))victim=it;
            if(victim==experts.end())throw std::runtime_error("GLM FP4: insufficient GPU memory for bounded working set");
            check(cudaStreamSynchronize(stream));cache_bytes-=victim->second.bytes;experts.erase(victim);
        }
        void init_upload() {
            if(!copy_stream)check(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
            if(!upload_device){room(upload_capacity);upload_device=std::make_unique<Device>(upload_capacity);}
            for(int i=0;i<2;++i) {
                if(!upload_host[i])upload_host[i]=std::make_unique<Pinned>(upload_capacity,true);
                if(!upload_ready[i])check(cudaEventCreateWithFlags(&upload_ready[i],cudaEventDisableTiming));
            }
        }
        void release_upload() {
            if(copy_stream)cudaStreamSynchronize(copy_stream);
            upload_device.reset();for(auto& p:upload_host)p.reset();
        }
        void room(size_t bytes) {
            if(auto* pool=Device::pool()) {
                // The pool already owns its physical capacity. A driver-wide
                // memory query here can serialize the independent copy stream.
                while(pool->largest()<bytes||Device::limit()<Device::live()+bytes)evict();
                return;
            }
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=Device::prealloc()?0:Device::physical_reserve()+256*MiB;
            while(free<bytes+reserve || Device::limit()<Device::live()+bytes) {
                evict();check(Device::mem_info(&free,&total));
            }
        }
    };
    strata::core::ModelArtifact& artifact;
    strata::artifact::GlmModelOpt& model;
    strata::cpu::exl3::Pool& cpu_pool;
    bool sm70,reserve_draft;
    const bool profile=strata_env_is_one("STRATA_GLM_FP4_PROFILE");
    const bool device_routing_enabled=strata_env_is_one("STRATA_GLM_FP4_DEVICE_ROUTING");
    bool decode_device_routing_enabled=strata_env_is_one("STRATA_GLM_FP4_DECODE_ROUTING");
    const bool async_upload_enabled=strata_env_is_one("STRATA_GLM_FP4_ASYNC_UPLOAD");
    const bool cpu_csf32_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_CSF32");
    const bool cpu_sparse_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE")||strata_env_is_one("STRATA_GLM_FP4_CPU_Q8");
    const bool cpu_q8_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_Q8");
    const bool cpu_fused_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_FUSED");
    const bool cpu_borrow_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_BORROW");
    const double route_drop_fraction=strata_env_value_double("STRATA_GLM_FP4_ROUTE_DROP",0);
    const long long route_drop_max=strata_env_value_ll("STRATA_GLM_FP4_ROUTE_DROP_MAX",2);
    const bool route_renorm_enabled=strata_env_is_one("STRATA_GLM_FP4_ROUTE_RENORM");
    const long long decode_reserve_mib=strata_env_value_ll("STRATA_GLM_FP4_DECODE_RESERVE_MIB",-1);
    uint64_t route_drop_tokens=0,route_drop_candidates=0,route_drop_count=0;
    double route_drop_share_sum=0,route_drop_share_max=0;
    double route_drop_multiplier_max=1;
    uint64_t decode_device_calls=0,decode_host_calls=0;
    bool decode_phase=false;
    using Clock=std::chrono::steady_clock;
    Clock::time_point tick()const {return profile?Clock::now():Clock::time_point{};}
    double elapsed(Clock::time_point start)const {
        return profile?std::chrono::duration<double,std::milli>(Clock::now()-start).count():0;
    }
    uint64_t cpu_experts=0;
    std::mutex cpu_cache_mutex;
    std::map<std::pair<int,int>,std::shared_ptr<CpuExpert>> cpu_cache;
    std::list<std::pair<int,int>> cpu_lru;
    size_t cpu_cache_bytes=0;
    const size_t cpu_scale_limit=strata::core::glm_fp4_cpu_scale_cache_bytes();
    uint64_t cpu_cache_hits=0,cpu_cache_misses=0;
    // Destroy workers and their DMA streams before unregistering the source.
    std::unique_ptr<RegisteredRam> registered_ram;
    std::array<std::array<std::array<std::unique_ptr<ExpertSource>,3>,288>,46> expert_sources;
    std::vector<std::unique_ptr<Context>> contexts;
    std::unique_ptr<Pinned> host_x,host_routes,host_weights;
    size_t host_rows=0;
    std::vector<std::vector<float>> sums;
    static size_t raw_bytes(const strata::artifact::ModelOptMatrix& m) {return m.weights->bytes;}
    const void* scale_source(const strata::artifact::ModelOptMatrix& desc,const char* part,const void* fallback)const {
        return registered_ram?artifact.exl3().tensors.data(desc.name+"_scale.csf32_"+part):fallback;
    }
    const ExpertSource& expert_source(int layer,int id,int projection) {
        auto& saved=expert_sources.at(layer).at(id).at(projection);
        if(saved)return *saved;
        const char* names[]={"gate_proj","up_proj","down_proj"};
        const auto desc=model.matrix(strata::artifact::GlmExl3::prefix(layer,id,names[projection])+".weight");
        if(desc.format!=Format::Nvfp4)throw std::runtime_error("GLM FP4: asynchronous expert upload requires NVFP4");
        auto source=std::make_unique<ExpertSource>();source->rows=desc.rows;source->columns=desc.columns;
        source->data[0]=artifact.exl3().tensors.data(*desc.weights);source->bytes[0]=raw_bytes(desc);
        if(model.has_csf32(desc)) {
            const auto h=model.csf32(desc);source->global=h.global;source->csf=true;
            source->data[1]=h.bases;source->bytes[1]=desc.rows;
            source->data[2]=h.codes;source->bytes[2]=h.codes_count;
            source->data[3]=scale_source(desc,"offsets",nullptr);source->bytes[3]=h.offsets.size()*2;
            source->data[4]=scale_source(desc,"page_bases",nullptr);source->bytes[4]=h.pages.size()*4;
            source->data[5]=scale_source(desc,"exceptions",nullptr);source->bytes[5]=h.exceptions.size()*2;
        }else {
            const auto h=model.load(desc,true);source->global=h.global;
            source->data[1]=artifact.exl3().tensors.data(desc.name+"_scale");source->bytes[1]=h.scales.size();
        }
        // This cache is only used after read-only RAM registration. The
        // checkpoint cannot change after its structural validation succeeds.
        saved=std::move(source);return *saved;
    }
    std::shared_ptr<CpuExpert> cpu_expert(int layer,int id) {
        const auto key=std::make_pair(layer,id);
        {
            std::lock_guard lock(cpu_cache_mutex);
            auto it=cpu_cache.find(key);
            if(it!=cpu_cache.end()) {++cpu_cache_hits;cpu_lru.splice(cpu_lru.begin(),cpu_lru,it->second->order);return it->second;}
            ++cpu_cache_misses;
        }
        auto e=std::make_shared<CpuExpert>();
        e->bytes=sizeof(CpuExpert);
        const char* parts[]={"gate_proj","up_proj","down_proj"};
        for(int p=0;p<3;++p) {
            const auto name=strata::artifact::GlmExl3::prefix(layer,id,parts[p])+".weight";
            e->projection[p]=model.load(name,true);
            if((cpu_csf32_enabled||cpu_sparse_enabled)&&model.has_csf32(model.matrix(name))) {
                auto& h=e->projection[p];const auto stem=name+"_scale.csf32_";
                const auto* bases=artifact.exl3().tensors.data(stem+"bases");
                const auto* codes=artifact.exl3().tensors.data(stem+"codes");
                if(cpu_sparse_enabled) {
                    e->sparse[p]=std::make_unique<cpu::GlmFp4Csf32SparseRows>(cpu::glm_fp4_csf32_sparse_rows(h.scales.data(),bases,codes,h.rows,h.columns/16,cpu_borrow_enabled));
                    e->bytes+=sizeof(cpu::GlmFp4Csf32SparseRows)+e->sparse[p]->bytes();
                }else {
                    e->csf32[p]=std::make_unique<cpu::GlmFp4Csf32Rows>(cpu::glm_fp4_csf32_rows(h.scales.data(),bases,codes,h.rows,h.columns/16));
                    e->bytes+=sizeof(cpu::GlmFp4Csf32Rows)+e->csf32[p]->bytes();
                }
                std::vector<uint8_t>().swap(h.scales);
            }
            e->bytes+=e->projection[p].scales.size();
            cpu::glm_fp4_weight_table(e->projection[p].global,sm70,e->scales[p].data());
        }
        std::lock_guard lock(cpu_cache_mutex);
        while(cpu_cache_bytes+e->bytes>cpu_scale_limit && !cpu_cache.empty()) {
            auto victim=cpu_cache.find(cpu_lru.back());
            cpu_cache_bytes-=victim->second->bytes;cpu_cache.erase(victim);cpu_lru.pop_back();
        }
        cpu_lru.push_front(key);e->order=cpu_lru.begin();cpu_cache_bytes+=e->bytes;cpu_cache.emplace(key,e);return e;
    }
    static void cpu_product(const CpuExpert& e,int projection,const float* x,float* y,
                            int columns,int rows,int tokens,int first,int last,const cpu::GlmFp4Q8* q8=nullptr) {
        const auto& packed=e.projection[projection];
        if(e.sparse[projection]&&q8)cpu::glm_fp4_rows_csf32_q8(packed.data(),*e.sparse[projection],e.scales[projection].data(),*q8,y,columns,rows,tokens,first,last);
        else if(e.sparse[projection])cpu::glm_fp4_rows_csf32_sparse_table(packed.data(),*e.sparse[projection],e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
        else if(e.csf32[projection])cpu::glm_fp4_rows_csf32_table(packed.data(),*e.csf32[projection],e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
        else cpu::glm_fp4_rows_table(packed.data(),packed.scales.data(),e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
    }
    Matrix upload(Context& c,const strata::artifact::ModelOptMatrix& desc) {
        if(model.has_csf32(desc)) {
            const auto h=model.csf32(desc);
            Matrix m;m.format=k::GlmPackedFormat::Nvfp4;m.rows=desc.rows;m.columns=desc.columns;m.global=h.global;
            if(sm70 && (!std::isfinite(h.global*16384.f) || h.global*16384.f>65504.f))
                throw std::runtime_error("GLM FP4: global scale exceeds SM70 decoder range");
            const size_t bytes=raw_bytes(desc),scales=desc.rows*desc.columns/16;
            m.scale_offset=(bytes+15)/16*16;
            c.room(m.scale_offset+scales);m.data=std::make_unique<Device>(m.scale_offset+scales);
            auto align=[](size_t n){return (n+15)/16*16;};
            const size_t base=align(bytes),code=align(base+desc.rows),offset=align(code+h.codes_count);
            const size_t page=align(offset+h.offsets.size()*2),exception=align(page+h.pages.size()*4);
            const size_t flag=align(exception+h.exceptions.size()*2);
            auto& staging=c.buffer("packed_source",flag+sizeof(int));
            auto* p=static_cast<uint8_t*>(staging.p);
            const auto* raw=artifact.exl3().tensors.data(*desc.weights);
            check(cudaMemcpyAsync(p,raw,bytes,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(p+base,h.bases,desc.rows,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(p+code,h.codes,h.codes_count,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(p+offset,scale_source(desc,"offsets",h.offsets.data()),h.offsets.size()*2,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(p+page,scale_source(desc,"page_bases",h.pages.data()),h.pages.size()*4,cudaMemcpyHostToDevice,c.stream));
            if(!h.exceptions.empty())check(cudaMemcpyAsync(p+exception,scale_source(desc,"exceptions",h.exceptions.data()),h.exceptions.size()*2,cudaMemcpyHostToDevice,c.stream));
            k::glm_fp4_prepack(p,nullptr,static_cast<uint8_t*>(m.data->p),nullptr,m.rows,m.columns,c.stream);
            k::glm_fp4_csf32_scales(p+base,p+code,reinterpret_cast<const uint16_t*>(p+offset),
                reinterpret_cast<const uint32_t*>(p+page),reinterpret_cast<const uint16_t*>(p+exception),
                static_cast<uint8_t*>(m.data->p)+m.scale_offset,reinterpret_cast<int*>(p+flag),m.rows,m.columns,c.stream);
            int invalid=0;check(cudaMemcpyAsync(&invalid,p+flag,sizeof(int),cudaMemcpyDeviceToHost,c.stream));
            check(cudaStreamSynchronize(c.stream));
            if(invalid)throw std::runtime_error("GLM ModelOpt: invalid FP4 group scale: "+desc.name);
            c.uploaded+=m.data->bytes;return m;
        }
        const auto h=model.load(desc,true);
        Matrix m;m.format=static_cast<k::GlmPackedFormat>(h.format);m.rows=h.rows;m.columns=h.columns;m.global=h.global;
        const size_t bytes=raw_bytes(desc);m.scale_offset=(bytes+15)/16*16;
        // Measured on V100: MXFP8 benefits from coalesced lane fragments.
        // The block-FP8 shared-expert shape is faster in its row-major layout.
        const bool packed_mxfp8=h.format==Format::Mxfp8&&h.rows%32==0;
        const size_t scales=packed_mxfp8?h.scales.size()*2:h.scales.size();
        c.room(m.scale_offset+scales);m.data=std::make_unique<Device>(m.scale_offset+scales);
        if(h.format==Format::Nvfp4) {
            if(sm70 && (!std::isfinite(h.global*16384.f) || h.global*16384.f>65504.f))
                throw std::runtime_error("GLM FP4: global scale exceeds SM70 decoder range");
            auto& staging=c.buffer("packed_source",bytes+h.scales.size());
            check(cudaMemcpyAsync(staging.p,h.data(),bytes,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(static_cast<char*>(staging.p)+bytes,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,c.stream));
            k::glm_fp4_prepack(static_cast<const uint8_t*>(staging.p),static_cast<const uint8_t*>(staging.p)+bytes,
                static_cast<uint8_t*>(m.data->p),static_cast<uint8_t*>(m.data->p)+m.scale_offset,m.rows,m.columns,c.stream);
        }else if(packed_mxfp8) {
            auto& staging=c.buffer("packed_source",bytes+h.scales.size());
            check(cudaMemcpyAsync(staging.p,h.data(),bytes,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(static_cast<char*>(staging.p)+bytes,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,c.stream));
            k::glm_fp8_prepack(m.format,static_cast<const uint8_t*>(staging.p),static_cast<char*>(staging.p)+bytes,
                static_cast<uint8_t*>(m.data->p),static_cast<char*>(m.data->p)+m.scale_offset,m.rows,m.columns,c.stream);
            m.format=k::GlmPackedFormat::Mxfp8Qpn;
        }else {
            check(cudaMemcpyAsync(m.data->p,h.data(),bytes,cudaMemcpyHostToDevice,c.stream));
            if(!h.scales.empty())check(cudaMemcpyAsync(static_cast<char*>(m.data->p)+m.scale_offset,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,c.stream));
            if(h.format==Format::Bf16) {
                // Cache the exact FP16 operand the W4A16 product would form.
                // BF16 and FP16 occupy the same bytes; no second matrix lives
                // on either the host or the device.
                k::glm_fp4_dequant_f16(m.format,m.data->p,m.scales(),m.global,
                    static_cast<uint16_t*>(m.data->p),m.rows,m.columns,c.stream);
                m.format=k::GlmPackedFormat::F16;
            }
        }
        // Scale vectors are bounded host temporaries; complete their DMA before
        // releasing them. The mapped source bytes remain owned by the artifact.
        check(cudaStreamSynchronize(c.stream));c.uploaded+=m.data->bytes;
        return m;
    }
    Expert& expert(Context& c,int layer,int id,bool admit=true) {
        c.temporary.reset();
        const auto key=std::make_pair(layer,id);auto it=c.experts.find(key);
        if(it!=c.experts.end()){++c.hits;it->second.use=++c.clock;return it->second;}
        ++c.misses;
        // Fixed per-device admission leaves room for scratch and future dense
        // operands. Eviction completes readers before releasing their storage.
        while(admit && !c.experts.empty() && c.cache_bytes+expert_bytes>c.cache_limit)c.evict();
        Expert e;const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int i=0;i<3;++i) {
            e.projection[i]=upload(c,model.matrix(strata::artifact::GlmExl3::prefix(layer,id,names[i])+".weight"));
            e.bytes+=e.projection[i].data->bytes;
        }
        if(!admit){c.temporary=std::make_unique<Expert>(std::move(e));return *c.temporary;}
        e.use=++c.clock;c.cache_bytes+=e.bytes;
        return c.experts.emplace(key,std::move(e)).first->second;
    }
    struct PendingExpert {
        Context* context=nullptr;
        Expert* value=nullptr;
        std::unique_ptr<Expert> owned;
        std::pair<int,int> key;
        int slot=-1;
        PendingExpert()=default;
        PendingExpert(PendingExpert&&)=default;
        PendingExpert& operator=(PendingExpert&&)=default;
        ~PendingExpert() {
            // An exception may leave either DMA or a matrix reader queued.
            if(owned&&context) {
                cudaStreamSynchronize(context->copy_stream);
                cudaStreamSynchronize(context->stream);
            }
        }
        void wait() {
            if(slot<0)return;
            check(cudaEventSynchronize(context->upload_ready[slot]));
            const auto* flags=reinterpret_cast<const int*>(static_cast<const char*>(context->upload_host[slot]->p)+Context::upload_capacity-16);
            if(flags[0]||flags[1]||flags[2])throw std::runtime_error("GLM ModelOpt: invalid FP4 group scale in asynchronous upload");
        }
    };
    PendingExpert prepare_expert(Context& c,int layer,int id) {
        PendingExpert result;result.context=&c;result.key={layer,id};
        auto cached=c.experts.find(result.key);
        if(cached!=c.experts.end()) {
            ++c.hits;cached->second.use=++c.clock;c.protected_experts.insert(result.key);
            result.value=&cached->second;return result;
        }
        ++c.misses;c.init_upload();
        result.slot=c.upload_slot;c.upload_slot^=1;
        // The previous consumer of this slot has already waited for its event.
        auto* host=static_cast<uint8_t*>(c.upload_host[result.slot]->p);
        auto* device=static_cast<uint8_t*>(c.upload_device->p);
        constexpr size_t flag=Context::upload_capacity-16;
        size_t used=0;
        struct Copy {const void* source;size_t offset,bytes;};std::vector<Copy> copies;
        auto append=[&](const void* source,size_t bytes,bool mapped=true) {
            const size_t offset=(used+15)/16*16;
            if(offset>flag||bytes>flag-offset)throw std::runtime_error("GLM FP4: expert exceeds bounded upload frame");
            if(bytes) {
                if(registered_ram&&mapped)copies.push_back({source,offset,bytes});
                else {std::memcpy(host+offset,source,bytes);if(registered_ram)copies.push_back({host+offset,offset,bytes});}
            }
            used=offset+bytes;return offset;
        };
        struct Source {size_t weight,base,code,offset,page,exception;bool csf;};
        std::array<Source,3> sources{};
        result.owned=std::make_unique<Expert>();result.value=result.owned.get();
        const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int i=0;i<3;++i) {
            auto& m=result.value->projection[i];auto& s=sources[i];
            if(registered_ram) {
                const auto& h=expert_source(layer,id,i);
                m.format=k::GlmPackedFormat::Nvfp4;m.rows=h.rows;m.columns=h.columns;m.global=h.global;
                s.csf=h.csf;m.scale_offset=(h.bytes[0]+15)/16*16;
                s.weight=append(h.data[0],h.bytes[0]);s.base=append(h.data[1],h.bytes[1]);
                if(s.csf) {
                    s.code=append(h.data[2],h.bytes[2]);s.offset=append(h.data[3],h.bytes[3]);
                    s.page=append(h.data[4],h.bytes[4]);s.exception=append(h.data[5],h.bytes[5]);
                }
            }else {
                const auto desc=model.matrix(strata::artifact::GlmExl3::prefix(layer,id,names[i])+".weight");
                if(desc.format!=Format::Nvfp4)throw std::runtime_error("GLM FP4: asynchronous expert upload requires NVFP4");
                m.format=k::GlmPackedFormat::Nvfp4;m.rows=desc.rows;m.columns=desc.columns;
                const size_t bytes=raw_bytes(desc);m.scale_offset=(bytes+15)/16*16;
                if(model.has_csf32(desc)) {
                    const auto h=model.csf32(desc);m.global=h.global;s.csf=true;
                    s.weight=append(artifact.exl3().tensors.data(*desc.weights),bytes);
                    s.base=append(h.bases,desc.rows);s.code=append(h.codes,h.codes_count);
                    s.offset=append(scale_source(desc,"offsets",h.offsets.data()),h.offsets.size()*2);
                    s.page=append(scale_source(desc,"page_bases",h.pages.data()),h.pages.size()*4);
                    s.exception=append(scale_source(desc,"exceptions",h.exceptions.data()),h.exceptions.size()*2);
                }else {
                    const auto h=model.load(desc,true);m.global=h.global;
                    s.weight=append(h.data(),bytes);s.base=append(h.scales.data(),h.scales.size(),false);
                }
            }
            if(sm70&&(!std::isfinite(m.global*16384.f)||m.global*16384.f>65504.f))
                throw std::runtime_error("GLM FP4: global scale exceeds SM70 decoder range");
            c.room(m.scale_offset+size_t(m.rows)*m.columns/16);
            m.data=std::make_unique<Device>(m.scale_offset+size_t(m.rows)*m.columns/16);
            result.value->bytes+=m.data->bytes;
        }
        if(registered_ram) {
            for(const auto& copy:copies)check(cudaMemcpyAsync(device+copy.offset,copy.source,copy.bytes,cudaMemcpyHostToDevice,c.copy_stream));
        }else check(cudaMemcpyAsync(device,host,used,cudaMemcpyHostToDevice,c.copy_stream));
        check(cudaMemsetAsync(device+flag,0,3*sizeof(int),c.copy_stream));
        for(int i=0;i<3;++i) {
            const auto& s=sources[i];auto& m=result.value->projection[i];
            auto* output=static_cast<uint8_t*>(m.data->p);
            k::glm_fp4_prepack(device+s.weight,s.csf?nullptr:device+s.base,
                output,s.csf?nullptr:output+m.scale_offset,m.rows,m.columns,c.copy_stream);
            if(s.csf)k::glm_fp4_csf32_scales(device+s.base,device+s.code,
                reinterpret_cast<const uint16_t*>(device+s.offset),reinterpret_cast<const uint32_t*>(device+s.page),
                reinterpret_cast<const uint16_t*>(device+s.exception),output+m.scale_offset,
                reinterpret_cast<int*>(device+flag)+i,m.rows,m.columns,c.copy_stream);
        }
        check(cudaGetLastError());
        check(cudaMemcpyAsync(host+flag,device+flag,3*sizeof(int),cudaMemcpyDeviceToHost,c.copy_stream));
        check(cudaEventRecord(c.upload_ready[result.slot],c.copy_stream));
        c.uploaded+=result.value->bytes;return result;
    }
    void finish_expert(Context& c,PendingExpert& pending,bool admit) {
        c.protected_experts.erase(pending.key);
        if(!pending.owned||!admit)return;
        while(!c.experts.empty()&&c.cache_bytes+pending.value->bytes>c.cache_limit)c.evict();
        pending.value->use=++c.clock;c.cache_bytes+=pending.value->bytes;
        c.experts.emplace(pending.key,std::move(*pending.owned));pending.owned.reset();
    }
    void select(Context& c,int layer) {
        const auto& frequency=c.frequency.at(layer);
        std::vector<int> ids;
        for(int id=0;id<288;++id)if(size_t(id)%contexts.size()==size_t(&c==contexts.front().get()?0:1))ids.push_back(id);
        std::sort(ids.begin(),ids.end(),[&](int a,int b){return frequency[a]!=frequency[b]?frequency[a]>frequency[b]:a<b;});
        const size_t main_limit=reserve_draft?(c.cache_limit>2048*MiB?c.cache_limit-2048*MiB:0):c.cache_limit;
        const size_t count=layer==45?ids.size():std::min(ids.size(),main_limit/(42*expert_bytes));
        c.keep[layer]=std::set<int>(ids.begin(),ids.begin()+count);
    }
    void product(Context& c,const Matrix& w,const uint16_t* x,float* y,int tokens) {
        if(tokens<=8) {
            float* partials=nullptr;
            if(sm70 && w.format!=k::GlmPackedFormat::Nvfp4 && w.rows<=512 && w.columns>=4096)
                partials=c.buffer("matrix_partial",k::glm_fp4_partial_bytes(512)).f();
            k::glm_fp4_gemm(w.format,w.data->p,w.scales(),w.global,x,y,w.rows,w.columns,tokens,sm70,c.stream,partials);
            (sm70?c.mma_calls:c.reference_calls)++;
        }else {
            const void* operand=w.data->p;
            if(w.format!=k::GlmPackedFormat::F16) {
                auto& half=c.buffer("dequant_half",size_t(w.rows)*w.columns*2);
                k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,static_cast<uint16_t*>(half.p),w.rows,w.columns,c.stream,sm70);
                operand=half.p;
            }
            const float one=1,zero=0;
            check(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,w.rows,tokens,w.columns,&one,
                operand,CUDA_R_16F,w.columns,x,CUDA_R_16F,w.columns,&zero,y,CUDA_R_32F,w.rows,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));++c.prefill_calls;
        }
        check(cudaGetLastError());
    }
public:
    explicit Fp4NativeExecutor(strata::core::ModelArtifact& a,strata::cpu::exl3::Pool& pool,
                              const std::vector<int>& devices,size_t gpu_budget,int width,bool draft)
        :artifact(a),model(*a.exl3().modelopt),cpu_pool(pool),sm70(fp4_sm70_kernel),reserve_draft(draft) {
        if(devices.empty()||devices.size()>2)throw std::invalid_argument("GLM FP4: expected one or two GPUs");
        if(decode_device_routing_enabled)std::cerr<<"FP4_DECODE_ROUTING enabled=1 deferred_expert_sync=1\n";
        if(!std::isfinite(route_drop_fraction)||route_drop_fraction<0||route_drop_fraction>0.5||route_drop_max<0||route_drop_max>7)
            throw std::invalid_argument("GLM FP4: ROUTE_DROP must be in [0,0.5], ROUTE_DROP_MAX in [0,7]");
        if(decode_reserve_mib!=-1 && (decode_reserve_mib<256 || decode_reserve_mib>8192))
            throw std::invalid_argument("GLM FP4: DECODE_RESERVE_MIB must be in [256,8192], or -1 for the original per-device reserves");
        if(route_drop_fraction>0)std::cerr<<"FP4_ROUTE_DROP lossy=1 cpu_only=1 fraction="<<route_drop_fraction
            <<" max_routes="<<route_drop_max<<" renormalize="<<route_renorm_enabled<<" draft=0\n";
        if(cpu_q8_enabled)std::cerr<<"FP4_CPU_Q8 lossy=1 mode="<<(cpu::glm_fp4_q8_vnni_available()?"vnni":"scalar")<<'\n';
        if(strata_env_is_one("STRATA_GLM_FP4_REGISTER_WEIGHTS"))
            registered_ram=std::make_unique<RegisteredRam>(artifact.exl3().tensors,devices);
        for(int d:devices) {
            DeviceScope scope(d);cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,d));
            if(p.major<7)throw std::invalid_argument("GLM FP4: tensor core path requires SM70 or newer");
            auto c=std::make_unique<Context>(d);
            if(Device::limit()==SIZE_MAX)Device::limit()=std::min<size_t>(gpu_budget,p.totalGlobalMem)-Device::budget_reserve();
            contexts.push_back(std::move(c));
        }
        // Compressed fixed matrices are resident, not expanded into FP32 on
        // the host. Their exact published scale grids accompany their codes.
        Context& c=*contexts.front();
        {DeviceScope scope(c.device);
            for(const auto& [name,t]:artifact.tensors()) {
                if(name=="token_embd.weight" || name.starts_with("blk.45.") || !t.native || t.tensor->shape.size()!=2)continue;
                c.dense.emplace(name,upload(c,model.matrix(t.native->source)));
            }
        }
        for(auto& ctx:contexts) {
            DeviceScope scope(ctx->device);size_t free,total;check(Device::mem_info(&free,&total));
            // Prefill scratch scales with tokens. The coordinator additionally
            // holds attention state and its GpuPrefill arena outside this class.
            const size_t reserve=std::max<size_t>((ctx.get()==contexts.front().get()?4096:1024)*MiB,size_t(width)*4096*24);
            const size_t available=std::min(free,Device::limit()>Device::live()?Device::limit()-Device::live():0);
            ctx->cache_limit=available>reserve?available-reserve:0;
            ctx->prefill_cache_limit=ctx->cache_limit;
            if(ctx->cache_limit<64*MiB)throw std::runtime_error("GLM FP4: less than 64 MiB available for expert cache");
            std::cerr<<"FP4_DEVICE device="<<ctx->device<<" cache_limit_MiB="<<ctx->cache_limit/MiB
                     <<" fixed_matrices="<<ctx->dense.size()<<" kernel="<<(sm70?"sm70":"reference")<<'\n';
        }
        sums.resize(contexts.size());
    }
    ~Fp4NativeExecutor(){report();}
    bool set_decode_device_routing(bool enabled) {
        const bool previous=decode_device_routing_enabled;decode_device_routing_enabled=enabled;return previous;
    }
    std::vector<int> devices()const {
        std::vector<int> result;for(const auto& c:contexts)result.push_back(c->device);return result;
    }
    void begin_request() {
        decode_phase=false;
        for(auto& c:contexts){c->frequency.clear();c->keep.clear();}
    }
    void begin_prefill(size_t coordinator_bytes) {
        decode_phase=false;
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=contexts[rank];
            DeviceScope scope(c->device);check(cudaStreamSynchronize(c->stream));
            c->scratch.clear();c->temporary.reset();c->cache_limit=c->prefill_cache_limit;
            while(c->cache_bytes>c->cache_limit)c->evict();
            if(Device::prealloc()) {
                // The old arena's exact-sized hole is reusable. Extra scratch
                // uses separate blocks and need not be contiguous with it.
                c->room(rank==0?coordinator_bytes:256*MiB);
                continue;
            }
            // Driver/library allocations can grow after the first prefill.
            // Make room against current physical free memory before the
            // coordinator restores its arena, then bound later admissions.
            const size_t future=rank==0?coordinator_bytes+512*MiB:256*MiB;
            c->room(future);
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=Device::physical_reserve()+256*MiB+future;
            const size_t physical=free>reserve?free-reserve:0;
            const size_t logical=Device::limit()>Device::live()+future?Device::limit()-Device::live()-future:0;
            c->cache_limit=std::min(c->cache_limit,c->cache_bytes+std::min(physical,logical));
        }
    }
    void compact() {
        decode_phase=true;
        for(auto& c:contexts) {
            DeviceScope scope(c->device);check(cudaStreamSynchronize(c->stream));c->scratch.clear();c->release_upload();
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=size_t(decode_reserve_mib<0?
                (c.get()==contexts.front().get()?3072:1024):decode_reserve_mib)*MiB;
            if(decode_reserve_mib>=0)if(auto* pool=Device::pool()) {
                // An expert consists of many small allocations, so it can use
                // separate holes left by prefill. The largest hole is needed
                // for an individual allocation, not the complete cache budget.
                std::lock_guard lock(pool->mutex);
                free=0;for(const auto& [offset,bytes]:pool->free_blocks)free+=bytes;
            }
            const size_t available=std::min(free,Device::limit()>Device::live()?Device::limit()-Device::live():0);
            c->cache_limit=c->cache_bytes+(available>reserve?available-reserve:0);
            if(decode_reserve_mib>=0)std::cerr<<"FP4_DECODE_CACHE device="<<c->device
                <<" reserve_MiB="<<decode_reserve_mib<<" limit_MiB="<<c->cache_limit/MiB<<'\n';
            for(const auto& [layer,_]:c->frequency)select(*c,layer);
        }
        host_x.reset();host_routes.reset();host_weights.reset();host_rows=0;
    }
    void dense(const std::string& name,const float* x,float* y,int nt,cudaStream_t caller) {
        Context& c=*contexts.front();DeviceScope scope(c.device);
        auto it=c.dense.find(name);
        if(it==c.dense.end()) {
            const auto& t=artifact.at(name);
            if(!t.native)throw std::logic_error("GLM FP4: expected native dense binding");
            it=c.dense.emplace(name,upload(c,model.matrix(t.native->source))).first;
        }
        // Dense products stay on the coordinator stream. This preserves its
        // producer ordering and permits recurrent CUDA graph capture after
        // the buffers and fixed matrix have been warmed.
        struct Restore {
            Context& c;cudaStream_t previous;
            ~Restore(){c.stream=previous;}
        } restore{c,c.stream};
        c.stream=caller;
        // All narrow dense calls share this buffer. Keep its address stable
        // across verification widths and the model's largest (4H) input.
        const size_t input_bytes=nt<=8?size_t(8)*16384*2:size_t(nt)*it->second.columns*2;
        auto& hx=c.buffer("dense_x_half",input_bytes);
        k::glm_f16(x,static_cast<uint16_t*>(hx.p),size_t(nt)*it->second.columns,c.stream);
        if(nt>8)check(cublasSetStream(c.blas,caller));
        product(c,it->second,static_cast<uint16_t*>(hx.p),y,nt);
        if(nt>8)check(cublasSetStream(c.blas,restore.previous));
    }
    void routed(int layer,const float* x,const int* ids,const float* weights,float* out,int nt,
                float limit,cudaStream_t caller) {
        const auto started=tick();
        struct alignas(64) Times {double upload=0,gather=0,product=0,reduce=0,fused=0;};
        std::array<Times,2> times{};Times cpu_times;
        if(size_t(nt)>host_rows) {
            check(cudaStreamSynchronize(caller));host_rows=nt;
            host_x=std::make_unique<Pinned>(size_t(nt)*4096*4,true);
            host_routes=std::make_unique<Pinned>(size_t(nt)*8*4,true);
            host_weights=std::make_unique<Pinned>(size_t(nt)*8*4,true);
        }
        check(cudaMemcpyAsync(host_x->p,x,size_t(nt)*4096*4,cudaMemcpyDeviceToHost,caller));
        check(cudaMemcpyAsync(host_routes->p,ids,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
        check(cudaMemcpyAsync(host_weights->p,weights,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
        check(cudaStreamSynchronize(caller));
        const auto* selected=static_cast<const int*>(host_routes->p);
        auto* routing=static_cast<float*>(host_weights->p);
        const auto* activation=static_cast<const float*>(host_x->p);
        std::map<int,std::vector<int>> grouped;
        for(int i=0;i<nt*8;++i) {
            if(selected[i]<0||selected[i]>=288)throw std::runtime_error("GLM FP4: invalid expert route");
            grouped[selected[i]].push_back(i);
        }
        const bool decode=nt<=8;
        bool device_routing=decode?decode_device_routing_enabled:device_routing_enabled;
        if(device_routing)for(const auto& [id,routes]:grouped)
            for(size_t j=1;j<routes.size();++j)if(routes[j]/8==routes[j-1]/8)device_routing=false;
        if(decode){if(device_routing)++decode_device_calls;else ++decode_host_calls;}
        for(auto& c:contexts)if(!decode || !c->frequency.count(layer)) {
            auto& counts=c->frequency[layer];counts.assign(288,0);
            for(const auto& [id,routes]:grouped)counts[id]=routes.size();
            select(*c,layer);
        }
        if(decode_phase && decode && layer<45 && route_drop_fraction>0 && route_drop_max>0) {
            std::array<uint8_t,8> masks{};
            for(int t=0;t<nt;++t) {
                uint8_t eligible=0;
                for(int j=0;j<8;++j) {
                    const int id=selected[t*8+j];
                    if(!contexts[size_t(id)%contexts.size()]->keep.at(layer).count(id)) {
                        eligible|=uint8_t(1u<<j);++route_drop_candidates;
                    }
                }
                double share=0;
                masks[t]=strata_glm_route_drop(routing+t*8,eligible,route_drop_fraction,
                                              unsigned(route_drop_max),&share);
                ++route_drop_tokens;route_drop_share_sum+=share;
                route_drop_share_max=std::max(route_drop_share_max,share);
                if(route_renorm_enabled && masks[t]) {
                    const auto factor=strata_glm_route_renormalize(routing+t*8,masks[t]);
                    if(!factor)throw std::runtime_error("GLM FP4: invalid route normalization");
                    route_drop_multiplier_max=std::max(route_drop_multiplier_max,factor);
                }
            }
            for(auto it=grouped.begin();it!=grouped.end();) {
                auto& routes=it->second;
                routes.erase(std::remove_if(routes.begin(),routes.end(),[&](int route) {
                    if(!(masks[route/8]&(1u<<(route%8))))return false;
                    ++route_drop_count;return true;
                }),routes.end());
                if(routes.empty())it=grouped.erase(it);else ++it;
            }
        }
        struct CpuJob {
            int id;const std::vector<int>* routes;
            std::shared_ptr<CpuExpert> weights;
            std::vector<float> input,gate,up,hidden,result;
            cpu::GlmFp4Q8 qinput,qhidden;
        };
        std::vector<CpuJob> cpu_jobs;
        if(decode)for(const auto& [id,routes]:grouped) {
            auto& c=*contexts[size_t(id)%contexts.size()];
            if(!c.keep.at(layer).count(id))cpu_jobs.push_back({id,&routes,{},{},{},{},{},{},{},{}});
        }
        std::vector<float> cpu_sum(cpu_jobs.empty()?0:size_t(nt)*4096,0);
        const double input_wait_ms=elapsed(started);
        auto run_cpu=[&] {
            if(cpu_jobs.empty())return;
            cpu_experts+=cpu_jobs.size();
            auto begin=tick();
            // Packed codes stay mapped. A bounded LRU retains logical
            // scale planes so repeat CPU misses do not re-expand CSF32.
            cpu_pool.run([&](size_t rank) {
                for(size_t i=rank;i<cpu_jobs.size();i+=cpu_pool.size()) {
                    auto& job=cpu_jobs[i];const size_t count=job.routes->size();
                    job.weights=cpu_expert(layer,job.id);
                    job.input.resize(count*4096);job.gate.resize(count*2048);job.up.resize(count*2048);
                    job.hidden.resize(count*2048);job.result.resize(count*4096);
                    for(size_t t=0;t<count;++t) {
                        const auto* source=activation+size_t((*job.routes)[t]/8)*4096;
                        auto* dest=job.input.data()+t*4096;
                        if(cpu_q8_enabled||cpu_fused_enabled) {
                            std::copy_n(source,4096,dest);cpu::glm_fp4_round_f16(dest,4096);
                        }else for(int h=0;h<4096;++h)
                            dest[h]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(source[h]));
                    }
                    if(cpu_q8_enabled)job.qinput=cpu::glm_fp4_quantize_q8(job.input.data(),4096,count);
                }
            });
            if(profile)cpu_times.upload+=elapsed(begin);
            begin=tick();
            auto gate_up=[&](size_t rank) {
                const int first=2048*rank/cpu_pool.size(),last=2048*(rank+1)/cpu_pool.size();
                for(auto& job:cpu_jobs) {
                    const int count=job.routes->size();
                    for(int p=0;p<2;++p)cpu_product(*job.weights,p,
                        job.input.data(),p?job.up.data():job.gate.data(),4096,2048,count,first,last,cpu_q8_enabled?&job.qinput:nullptr);
                    for(int t=0;t<count;++t) {
                        for(int r=first;r<last;++r) {
                            const size_t i=size_t(t)*2048+r;
                            const float g=std::min(job.gate[i],limit),u=std::clamp(job.up[i],-limit,limit);
                            const float hidden=(g/(1.f+std::exp(-g)))*u;
                            job.hidden[i]=(cpu_q8_enabled||cpu_fused_enabled)?hidden:
                                strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(hidden));
                        }
                        if(cpu_q8_enabled||cpu_fused_enabled)cpu::glm_fp4_round_f16(job.hidden.data()+size_t(t)*2048+first,last-first);
                    }
                }
            };
            auto quant_hidden=[&](size_t rank) {
                for(size_t i=rank;i<cpu_jobs.size();i+=cpu_pool.size()) {
                    auto& job=cpu_jobs[i];job.qhidden=cpu::glm_fp4_quantize_q8(job.hidden.data(),2048,job.routes->size());
                }
            };
            auto down=[&](size_t rank) {
                const int first=4096*rank/cpu_pool.size(),last=4096*(rank+1)/cpu_pool.size();
                for(auto& job:cpu_jobs)cpu_product(*job.weights,2,
                    job.hidden.data(),job.result.data(),2048,4096,job.routes->size(),first,last,cpu_q8_enabled?&job.qhidden:nullptr);
            };
            if(cpu_fused_enabled) {
                // Keep the same row partitions and arithmetic while avoiding
                // sleeping and waking every worker between the small products.
                // A failed phase releases peers before Pool rethrows its error.
                std::atomic<size_t> arrived{0};std::atomic<bool> cancelled{false};
                auto barrier=[&](size_t phase) {
                    arrived.fetch_add(1,std::memory_order_acq_rel);
                    while(arrived.load(std::memory_order_acquire)<cpu_pool.size()*phase) {
                        if(cancelled.load(std::memory_order_acquire))return false;
#if defined(__x86_64__)
                        _mm_pause();
#else
                        std::this_thread::yield();
#endif
                    }
                    return !cancelled.load(std::memory_order_acquire);
                };
                auto fused_work=[&](size_t rank) {
                    try {
                        gate_up(rank);if(!barrier(1))return;
                        if(cpu_q8_enabled){quant_hidden(rank);if(!barrier(2))return;}
                        down(rank);
                    }catch(...){cancelled.store(true,std::memory_order_release);throw;}
                };
                // Worker-side callable copies must not allocate before entering
                // the cancellation handler: peers may already be at a barrier.
                cpu_pool.run(std::ref(fused_work));
                if(profile)cpu_times.fused+=elapsed(begin);
            }else {
                cpu_pool.run(gate_up);
                if(cpu_q8_enabled)cpu_pool.run(quant_hidden);
                if(profile)cpu_times.gather+=elapsed(begin);
                begin=tick();cpu_pool.run(down);
                if(profile)cpu_times.product+=elapsed(begin);
            }
            begin=tick();
            for(const auto& job:cpu_jobs)for(size_t t=0;t<job.routes->size();++t) {
                const int route=(*job.routes)[t];float* dst=cpu_sum.data()+size_t(route/8)*4096;
                for(int h=0;h<4096;++h)dst[h]+=routing[route]*job.result[t*4096+h];
            }
            if(profile)cpu_times.reduce+=elapsed(begin);
        };
        auto run=[&](size_t rank) {
            Context& c=*contexts[rank];DeviceScope scope(c.device);
            struct Drain {
                Context& c;
                ~Drain(){if(c.copy_stream||!c.protected_experts.empty())cudaStreamSynchronize(c.stream);if(c.copy_stream)cudaStreamSynchronize(c.copy_stream);c.protected_experts.clear();}
            } drain{c};
            auto& sum=sums[rank];sum.assign(size_t(nt)*4096,0);
            std::vector<float> input,result;
            const float* device_input=x;const float* device_weights=weights;
            float* device_sum=nullptr;
            if(device_routing) {
                device_sum=c.buffer("prefill_route_sum",sum.size()*4).f();
                check(cudaMemsetAsync(device_sum,0,sum.size()*4,c.stream));
                if(rank) {
                    device_input=c.buffer("prefill_route_input",size_t(nt)*4096*4).f();
                    device_weights=c.buffer("prefill_route_weights",size_t(nt)*8*4).f();
                    check(cudaMemcpyAsync(const_cast<float*>(device_input),host_x->p,size_t(nt)*4096*4,cudaMemcpyHostToDevice,c.stream));
                    check(cudaMemcpyAsync(const_cast<float*>(device_weights),host_weights->p,size_t(nt)*8*4,cudaMemcpyHostToDevice,c.stream));
                }
                if(decode && !rank) {
                    // Decode may renormalize the copied routing coefficients.
                    // The checkpoint/router's device coefficients stay intact.
                    device_weights=c.buffer("decode_route_weights",size_t(nt)*8*4).f();
                    check(cudaMemcpyAsync(const_cast<float*>(device_weights),host_weights->p,size_t(nt)*8*4,cudaMemcpyHostToDevice,c.stream));
                }
            }
            // Stable expert ownership prevents duplicated resident expert
            // weights and keeps arithmetic independent of route arrival order.
            const bool pipeline=async_upload_enabled&&device_routing&&!decode&&Device::prealloc();
            std::vector<int> jobs;
            for(const auto& [id,routes]:grouped) {
                if(size_t(id)%contexts.size()!=rank)continue;
                if(decode&&!c.keep.at(layer).count(id))continue;
                jobs.push_back(id);
            }
            PendingExpert pending;
            if(pipeline&&!jobs.empty())pending=prepare_expert(c,layer,jobs.front());
            for(size_t job=0;job<jobs.size();++job) {
                const int id=jobs[job];const auto& routes=grouped.at(id);
                const bool admit=c.keep.at(layer).count(id);
                auto begin=tick();
                if(pipeline)pending.wait();
                auto& e=pipeline?*pending.value:expert(c,layer,id,admit);const int count=routes.size();
                c.protected_experts.insert({layer,id});
                if(profile)times[rank].upload+=elapsed(begin);
                begin=tick();
                if(!device_routing) {
                    input.resize(size_t(count)*4096);result.resize(size_t(count)*4096);
                    for(int j=0;j<count;++j)std::copy_n(activation+size_t(routes[j]/8)*4096,4096,input.data()+size_t(j)*4096);
                }
                if(profile)times[rank].gather+=elapsed(begin);
                begin=tick();
                auto& hx=c.buffer("expert_x_half",size_t(count)*4096*2);
                auto& gate=c.buffer("expert_gate",size_t(count)*2048*4);
                auto& up=c.buffer("expert_up",size_t(count)*2048*4);
                auto& hidden=c.buffer("expert_hidden_half",size_t(count)*2048*2);
                auto& dy=c.buffer("expert_result",size_t(count)*4096*4);
                int* device_routes=nullptr;
                if(device_routing) {
                    device_routes=static_cast<int*>(c.buffer("expert_route_ids",size_t(count)*4).p);
                    check(cudaMemcpyAsync(device_routes,routes.data(),size_t(count)*4,cudaMemcpyHostToDevice,c.stream));
                    k::glm_fp4_route_gather(device_input,device_routes,static_cast<uint16_t*>(hx.p),count,4096,8,c.stream);
                }else {
                    auto& dx=c.buffer("expert_x",input.size()*4);
                    check(cudaMemcpyAsync(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice,c.stream));
                    k::glm_f16(dx.f(),static_cast<uint16_t*>(hx.p),input.size(),c.stream);
                }
                product(c,e.projection[0],static_cast<uint16_t*>(hx.p),gate.f(),count);
                product(c,e.projection[1],static_cast<uint16_t*>(hx.p),up.f(),count);
                k::glm_swiglu(gate.f(),up.f(),gate.f(),count*2048,limit,c.stream);
                k::glm_f16(gate.f(),static_cast<uint16_t*>(hidden.p),size_t(count)*2048,c.stream);
                product(c,e.projection[2],static_cast<uint16_t*>(hidden.p),dy.f(),count);
                if(device_routing)k::glm_fp4_route_scatter(dy.f(),device_routes,device_weights,device_sum,count,4096,8,c.stream);
                else check(cudaMemcpyAsync(result.data(),dy.p,result.size()*4,cudaMemcpyDeviceToHost,c.stream));
                PendingExpert next;
                if(pipeline&&job+1<jobs.size())next=prepare_expert(c,layer,jobs[job+1]);
                if(!decode||!device_routing)check(cudaStreamSynchronize(c.stream));
                if(pipeline){finish_expert(c,pending,admit);pending=std::move(next);}
                else if(!decode||!device_routing)c.protected_experts.erase({layer,id});
                if(profile)times[rank].product+=elapsed(begin);
                begin=tick();
                if(!device_routing)for(int j=0;j<count;++j) {
                    float* dest=sum.data()+size_t(routes[j]/8)*4096;
                    const float scale=routing[routes[j]];
                    for(int h=0;h<4096;++h)dest[h]+=scale*result[size_t(j)*4096+h];
                }
                if(profile)times[rank].reduce+=elapsed(begin);
            }
            if(device_routing) {
                const auto begin=tick();
                check(cudaMemcpyAsync(sum.data(),device_sum,sum.size()*4,cudaMemcpyDeviceToHost,c.stream));
                check(cudaStreamSynchronize(c.stream));
                if(profile)times[rank].product+=elapsed(begin);
            }
        };
        // Independent device streams operate concurrently; shared model pages
        // and bounded CPU staging are owned by this executor.
        std::vector<std::future<void>> work;
        try {
            for(size_t rank=0;rank<contexts.size();++rank)work.push_back(contexts[rank]->submit([&,rank]{run(rank);}));
            run_cpu();for(auto& f:work)f.get();
        }catch(...){for(auto& f:work)if(f.valid())f.wait();throw;}
        auto& c=*contexts.front();DeviceScope scope(c.device);
        if(sums.size()==2)for(size_t i=0;i<sums[0].size();++i)sums[0][i]+=sums[1][i];
        if(!cpu_jobs.empty())for(size_t i=0;i<sums[0].size();++i)sums[0][i]+=cpu_sum[i];
        auto& result=c.buffer("route_sum",sums[0].size()*4);
        check(cudaMemcpyAsync(result.p,sums[0].data(),sums[0].size()*4,cudaMemcpyHostToDevice,caller));
        const float one=1;
        check(cublasSetStream(c.blas,caller));
        check(cublasSaxpy(c.blas,sums[0].size(),&one,result.f(),1,out,1));
        check(cublasSetStream(c.blas,c.stream));
        check(cudaGetLastError());
        if(profile) {
            std::ostringstream line;
            line<<"FP4_PROFILE layer="<<layer<<" tokens="<<nt<<" wall_ms="<<elapsed(started)
                <<" input_wait_ms="<<input_wait_ms<<" cpu_groups="<<cpu_jobs.size()
                <<" cpu_load_ms="<<cpu_times.upload<<" cpu_gu_ms="<<cpu_times.gather
                <<" cpu_down_ms="<<cpu_times.product<<" cpu_reduce_ms="<<cpu_times.reduce
                <<" cpu_fused_ms="<<cpu_times.fused;
            for(size_t rank=0;rank<contexts.size();++rank)line
                <<" gpu"<<rank<<"_upload_ms="<<times[rank].upload
                <<" gpu"<<rank<<"_gather_ms="<<times[rank].gather
                <<" gpu"<<rank<<"_product_ms="<<times[rank].product
                <<" gpu"<<rank<<"_reduce_ms="<<times[rank].reduce;
            line<<'\n';std::cerr<<line.str();
        }
    }
    void report()const {
        if(decode_device_routing_enabled)std::cerr<<"FP4_DECODE_ROUTING device_calls="<<decode_device_calls
            <<" host_calls="<<decode_host_calls<<'\n';
        if(route_drop_fraction>0)std::cerr<<"FP4_ROUTE_DROP_STATS token_layers="<<route_drop_tokens
            <<" cpu_candidates="<<route_drop_candidates<<" dropped="<<route_drop_count
            <<" mean_share="<<(route_drop_tokens?route_drop_share_sum/route_drop_tokens:0)
            <<" max_share="<<route_drop_share_max<<" max_multiplier="<<route_drop_multiplier_max<<'\n';
        std::cerr<<"FP4_CPU expert_groups="<<cpu_experts<<" activation_format="
            <<(cpu_q8_enabled?"q8_block64_from_f16":"f16")<<" weight_format=native_nvfp4\n";
        std::cerr<<"FP4_CPU_SCALES hits="<<cpu_cache_hits<<" misses="<<cpu_cache_misses<<" bytes="<<cpu_cache_bytes<<'\n';
        for(const auto& c:contexts)std::cerr<<"FP4_ROUTES device="<<c->device<<" hits="<<c->hits<<" misses="<<c->misses
            <<" uploaded_bytes="<<c->uploaded<<" cache_bytes="<<c->cache_bytes<<" mma="<<c->mma_calls
            <<" reference="<<c->reference_calls<<" prefill="<<c->prefill_calls<<'\n';
    }
};
#endif
