// Native ModelOpt execution. Included after the GLM GPU allocation helpers.
// One owner shares read-only checkpoint mappings across the two device streams.
#ifndef STRATA_USE_HIP
static bool fp4_sm70_kernel = false;
static bool fp4_kernel_selected = false;
class Fp4NativeExecutor {
    using Format=strata::artifact::ModelOptFormat;
    using Packed=strata::artifact::ModelOptPacked;
    static constexpr size_t expert_bytes=3*(size_t(4096)*2048/2+size_t(4096)*2048/16);
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
    struct CpuExpert {
        std::array<Packed,3> projection;
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
            DeviceScope scope(device);if(stream)cudaStreamSynchronize(stream);temporary.reset();experts.clear();dense.clear();scratch.clear();if(blas)cublasDestroy(blas);if(stream)cudaStreamDestroy(stream);
        }
        std::future<void> submit(std::function<void()> job) {
            std::lock_guard lock(work_mutex);
            if(task.valid()||stop)throw std::logic_error("GLM FP4: GPU worker already has a pending task");
            task=std::packaged_task<void()>(std::move(job));auto done=task.get_future();wake.notify_one();return done;
        }
        Device& buffer(const std::string& name,size_t bytes) {
            auto& b=scratch[name];
            if(!b||b->bytes<bytes) {check(cudaStreamSynchronize(stream));b=std::make_unique<Device>(std::max<size_t>(bytes,256));}
            return *b;
        }
        void evict() {
            if(experts.empty())throw std::runtime_error("GLM FP4: insufficient GPU memory for bounded working set");
            auto victim=std::min_element(experts.begin(),experts.end(),[](const auto& a,const auto& b){return a.second.use<b.second.use;});
            check(cudaStreamSynchronize(stream));cache_bytes-=victim->second.bytes;experts.erase(victim);
        }
        void room(size_t bytes) {
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=Device::physical_reserve()+256*MiB;
            while(free<bytes+reserve || Device::limit()<Device::live()+bytes) {
                evict();check(Device::mem_info(&free,&total));
            }
        }
    };
    strata::core::ModelArtifact& artifact;
    strata::artifact::GlmModelOpt& model;
    strata::cpu::exl3::Pool& cpu_pool;
    bool sm70,reserve_draft;
    uint64_t cpu_experts=0;
    std::mutex cpu_cache_mutex;
    std::map<std::pair<int,int>,std::shared_ptr<CpuExpert>> cpu_cache;
    std::list<std::pair<int,int>> cpu_lru;
    size_t cpu_cache_bytes=0;
    uint64_t cpu_cache_hits=0,cpu_cache_misses=0;
    std::vector<std::unique_ptr<Context>> contexts;
    std::unique_ptr<Pinned> host_x,host_routes,host_weights;
    size_t host_rows=0;
    std::vector<std::vector<float>> sums;
    static size_t raw_bytes(const strata::artifact::ModelOptMatrix& m) {return m.weights->bytes;}
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
            e->projection[p]=model.load(strata::artifact::GlmExl3::prefix(layer,id,parts[p])+".weight",true);
            e->bytes+=e->projection[p].scales.size();
            cpu::glm_fp4_weight_table(e->projection[p].global,sm70,e->scales[p].data());
        }
        std::lock_guard lock(cpu_cache_mutex);
        constexpr size_t scale_limit=2048*MiB;
        while(cpu_cache_bytes+e->bytes>scale_limit && !cpu_cache.empty()) {
            auto victim=cpu_cache.find(cpu_lru.back());
            cpu_cache_bytes-=victim->second->bytes;cpu_cache.erase(victim);cpu_lru.pop_back();
        }
        cpu_lru.push_front(key);e->order=cpu_lru.begin();cpu_cache_bytes+=e->bytes;cpu_cache.emplace(key,e);return e;
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
            check(cudaMemcpyAsync(p+offset,h.offsets.data(),h.offsets.size()*2,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(p+page,h.pages.data(),h.pages.size()*4,cudaMemcpyHostToDevice,c.stream));
            if(!h.exceptions.empty())check(cudaMemcpyAsync(p+exception,h.exceptions.data(),h.exceptions.size()*2,cudaMemcpyHostToDevice,c.stream));
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
                k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,static_cast<uint16_t*>(half.p),w.rows,w.columns,c.stream);
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
    std::vector<int> devices()const {
        std::vector<int> result;for(const auto& c:contexts)result.push_back(c->device);return result;
    }
    void begin_request() {
        for(auto& c:contexts){c->frequency.clear();c->keep.clear();}
    }
    void begin_prefill() {
        for(auto& c:contexts) {
            DeviceScope scope(c->device);check(cudaStreamSynchronize(c->stream));
            c->scratch.clear();c->temporary.reset();c->cache_limit=c->prefill_cache_limit;
            while(c->cache_bytes>c->cache_limit)c->evict();
        }
    }
    void compact() {
        for(auto& c:contexts) {
            DeviceScope scope(c->device);check(cudaStreamSynchronize(c->stream));c->scratch.clear();
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=(c.get()==contexts.front().get()?3072:1024)*MiB;
            const size_t available=std::min(free,Device::limit()>Device::live()?Device::limit()-Device::live():0);
            c->cache_limit=c->cache_bytes+(available>reserve?available-reserve:0);
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
        const auto* routing=static_cast<const float*>(host_weights->p);
        const auto* activation=static_cast<const float*>(host_x->p);
        std::map<int,std::vector<int>> grouped;
        for(int i=0;i<nt*8;++i) {
            if(selected[i]<0||selected[i]>=288)throw std::runtime_error("GLM FP4: invalid expert route");
            grouped[selected[i]].push_back(i);
        }
        const bool decode=nt<=8;
        for(auto& c:contexts)if(!decode || !c->frequency.count(layer)) {
            auto& counts=c->frequency[layer];counts.assign(288,0);
            for(const auto& [id,routes]:grouped)counts[id]=routes.size();
            select(*c,layer);
        }
        struct CpuJob {
            int id;const std::vector<int>* routes;
            std::shared_ptr<CpuExpert> weights;
            std::vector<float> input,gate,up,hidden,result;
        };
        std::vector<CpuJob> cpu_jobs;
        if(decode)for(const auto& [id,routes]:grouped) {
            auto& c=*contexts[size_t(id)%contexts.size()];
            if(!c.keep.at(layer).count(id))cpu_jobs.push_back({id,&routes,{},{},{},{},{},{}});
        }
        std::vector<float> cpu_sum(cpu_jobs.empty()?0:size_t(nt)*4096,0);
        auto run_cpu=[&] {
            if(cpu_jobs.empty())return;
            cpu_experts+=cpu_jobs.size();
            // Packed codes stay mapped. A bounded 2 GiB LRU retains logical
            // scale planes so repeat CPU misses do not re-expand CSF32.
            cpu_pool.run([&](size_t rank) {
                for(size_t i=rank;i<cpu_jobs.size();i+=cpu_pool.size()) {
                    auto& job=cpu_jobs[i];const size_t count=job.routes->size();
                    job.weights=cpu_expert(layer,job.id);
                    job.input.resize(count*4096);job.gate.resize(count*2048);job.up.resize(count*2048);
                    job.hidden.resize(count*2048);job.result.resize(count*4096);
                    for(size_t t=0;t<count;++t)for(int h=0;h<4096;++h)
                        job.input[t*4096+h]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(activation[size_t((*job.routes)[t]/8)*4096+h]));
                }
            });
            cpu_pool.run([&](size_t rank) {
                const int first=2048*rank/cpu_pool.size(),last=2048*(rank+1)/cpu_pool.size();
                for(auto& job:cpu_jobs) {
                    const int count=job.routes->size();
                    for(int p=0;p<2;++p)cpu::glm_fp4_rows_table(job.weights->projection[p].data(),job.weights->projection[p].scales.data(),job.weights->scales[p].data(),
                        job.input.data(),p?job.up.data():job.gate.data(),4096,2048,count,first,last);
                    for(int t=0;t<count;++t)for(int r=first;r<last;++r) {
                        const size_t i=size_t(t)*2048+r;
                        const float g=std::min(job.gate[i],limit),u=std::clamp(job.up[i],-limit,limit);
                        const float hidden=(g/(1.f+std::exp(-g)))*u;
                        job.hidden[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(hidden));
                    }
                }
            });
            cpu_pool.run([&](size_t rank) {
                const int first=4096*rank/cpu_pool.size(),last=4096*(rank+1)/cpu_pool.size();
                for(auto& job:cpu_jobs)cpu::glm_fp4_rows_table(job.weights->projection[2].data(),job.weights->projection[2].scales.data(),job.weights->scales[2].data(),
                    job.hidden.data(),job.result.data(),2048,4096,job.routes->size(),first,last);
            });
            for(const auto& job:cpu_jobs)for(size_t t=0;t<job.routes->size();++t) {
                const int route=(*job.routes)[t];float* dst=cpu_sum.data()+size_t(route/8)*4096;
                for(int h=0;h<4096;++h)dst[h]+=routing[route]*job.result[t*4096+h];
            }
        };
        auto run=[&](size_t rank) {
            Context& c=*contexts[rank];DeviceScope scope(c.device);
            auto& sum=sums[rank];sum.assign(size_t(nt)*4096,0);
            std::vector<float> input,result;
            // Stable expert ownership prevents duplicated resident expert
            // weights and keeps arithmetic independent of route arrival order.
            for(const auto& [id,routes]:grouped) {
                if(size_t(id)%contexts.size()!=rank)continue;
                const bool admit=c.keep.at(layer).count(id);
                if(decode&&!admit)continue;
                auto& e=expert(c,layer,id,admit);const int count=routes.size();
                input.resize(size_t(count)*4096);result.resize(size_t(count)*4096);
                for(int j=0;j<count;++j)std::copy_n(activation+size_t(routes[j]/8)*4096,4096,input.data()+size_t(j)*4096);
                auto& dx=c.buffer("expert_x",input.size()*4);
                auto& hx=c.buffer("expert_x_half",input.size()*2);
                auto& gate=c.buffer("expert_gate",size_t(count)*2048*4);
                auto& up=c.buffer("expert_up",size_t(count)*2048*4);
                auto& hidden=c.buffer("expert_hidden_half",size_t(count)*2048*2);
                auto& dy=c.buffer("expert_result",result.size()*4);
                check(cudaMemcpyAsync(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice,c.stream));
                k::glm_f16(dx.f(),static_cast<uint16_t*>(hx.p),input.size(),c.stream);
                product(c,e.projection[0],static_cast<uint16_t*>(hx.p),gate.f(),count);
                product(c,e.projection[1],static_cast<uint16_t*>(hx.p),up.f(),count);
                k::glm_swiglu(gate.f(),up.f(),gate.f(),count*2048,limit,c.stream);
                k::glm_f16(gate.f(),static_cast<uint16_t*>(hidden.p),size_t(count)*2048,c.stream);
                product(c,e.projection[2],static_cast<uint16_t*>(hidden.p),dy.f(),count);
                check(cudaMemcpyAsync(result.data(),dy.p,result.size()*4,cudaMemcpyDeviceToHost,c.stream));
                check(cudaStreamSynchronize(c.stream));
                for(int j=0;j<count;++j) {
                    float* dest=sum.data()+size_t(routes[j]/8)*4096;
                    const float scale=routing[routes[j]];
                    for(int h=0;h<4096;++h)dest[h]+=scale*result[size_t(j)*4096+h];
                }
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
    }
    void report()const {
        std::cerr<<"FP4_CPU expert_groups="<<cpu_experts<<" activation_format=f16 weight_format=native_nvfp4\n";
        std::cerr<<"FP4_CPU_SCALES hits="<<cpu_cache_hits<<" misses="<<cpu_cache_misses<<" bytes="<<cpu_cache_bytes<<'\n';
        for(const auto& c:contexts)std::cerr<<"FP4_ROUTES device="<<c->device<<" hits="<<c->hits<<" misses="<<c->misses
            <<" uploaded_bytes="<<c->uploaded<<" cache_bytes="<<c->cache_bytes<<" mma="<<c->mma_calls
            <<" reference="<<c->reference_calls<<" prefill="<<c->prefill_calls<<'\n';
    }
};
#endif
