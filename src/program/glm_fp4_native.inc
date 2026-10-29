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
    struct PrefillBank {
        static constexpr size_t count=144,frame_bytes=16*MiB,flag_offset=frame_bytes-4096;
        int device,layer=-1;
        cudaStream_t stream=nullptr;
        // Declared before the views so that they are destroyed first.
        std::unique_ptr<Device> slab;
        std::array<Expert,count> experts;
        std::array<Expert*,count> cached{};
        std::array<std::array<const ExpertSource*,3>,count> sources{};
        std::unique_ptr<Device> frame;
        std::unique_ptr<Pinned> flags;
        std::future<void> loading;
        size_t allocated_bytes=0,cached_count=0,wire_bytes=0;
        explicit PrefillBank(int device):device(device) {
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        }
        ~PrefillBank(){DeviceScope scope(device);if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    };
    // Streamed prefill keeps raw expert matrices flowing through a ring of
    // device frames on the copy stream, also while attention runs. A frame is
    // refilled only after the operand kernels reading it have been enqueued.
    struct PrefillStream {
        static constexpr size_t operand_slots=2,operand_elements=size_t(2048)*4096;
        static constexpr size_t flag_count=2*288*3;
        struct Item {
            int layer=-1,id=-1,frame=-1;
            std::array<const ExpertSource*,3> sources{};
            std::array<std::array<size_t,6>,3> offsets{};
        };
        int device;
        cudaStream_t prep=nullptr;
        size_t allocated_bytes=0;
        // Frames are the lent draft experts' own projections, so the arena's
        // free blocks stay those of the original path. Operand slots and
        // flags come from the driver margin outside the arena.
        std::vector<void*> outside;
        std::vector<std::array<uint8_t*,3>> frames;
        std::vector<cudaEvent_t> full,consumed;
        std::vector<char> refillable;
        // A lent entry the cache evicts while its frame is idle frees its
        // memory at once, as on the original path; the frame is then skipped.
        std::vector<char> dead;
        std::map<std::pair<int,int>,int> lent_frame;
        size_t alive=0;
        std::mutex mutex;
        std::array<std::array<uint16_t*,3>,operand_slots> operands{};
        std::array<cudaEvent_t,operand_slots> operand_ready{},operand_free{};
        cudaEvent_t input_ready=nullptr;
        int* flags=nullptr;
        std::unique_ptr<Pinned> flags_host;
        // One pinned slot per frame gathers a projection's small CSF parts so
        // they cross in one copy: eighteen small copies per expert fill the
        // driver queue and then block kernel launches.
        std::unique_ptr<Pinned> staging;
        // Route tables cross by glm_fp4_copy_host_words: a copy-engine upload
        // would queue behind the ring. Two buffers alternate by layer.
        std::array<std::unique_ptr<Pinned>,2> route_host;std::array<size_t,2> route_host_words{};
        std::array<cudaEvent_t,2> route_read{};
        std::deque<std::pair<int,int>> planned;
        std::deque<Item> queued;
        size_t head=0,operand=0;
        explicit PrefillStream(int device):device(device) {
            check(cudaStreamCreateWithFlags(&prep,cudaStreamNonBlocking));
        }
        void create_events(size_t count) {
            full.assign(count,nullptr);consumed.assign(count,nullptr);refillable.assign(count,1);dead.assign(count,0);
            for(size_t i=0;i<count;++i) {
                check(cudaEventCreateWithFlags(&full[i],cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&consumed[i],cudaEventDisableTiming));
            }
            check(cudaEventCreateWithFlags(&input_ready,cudaEventDisableTiming));
            for(auto& e:route_read)check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
            for(size_t i=0;i<operand_slots;++i) {
                check(cudaEventCreateWithFlags(&operand_ready[i],cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&operand_free[i],cudaEventDisableTiming));
            }
        }
        ~PrefillStream() {
            DeviceScope scope(device);if(prep)cudaStreamSynchronize(prep);
            for(auto e:full)if(e)cudaEventDestroy(e);
            for(auto e:consumed)if(e)cudaEventDestroy(e);
            for(auto e:operand_ready)if(e)cudaEventDestroy(e);
            for(auto e:operand_free)if(e)cudaEventDestroy(e);
            for(auto e:route_read)if(e)cudaEventDestroy(e);
            if(input_ready)cudaEventDestroy(input_ready);
            if(prep)cudaStreamDestroy(prep);
            for(auto p:outside)cudaFree(p);
        }
    };
    struct CpuExpert {
        std::array<Packed,3> projection;
        std::array<std::unique_ptr<cpu::GlmFp4Csf32Rows>,3> csf32;
        std::array<std::unique_ptr<cpu::GlmFp4Csf32SparseRows>,3> sparse;
        alignas(64) std::array<std::array<float,cpu::glm_fp4_weight_table_floats>,3> scales;
        size_t bytes=0;
        std::list<std::pair<int,int>>::iterator order;
    };
    struct HeadPartition {
        Matrix first;
        int primary=0,secondary=0;
        cudaEvent_t ready=nullptr,done=nullptr;
        uint64_t calls=0;
        ~HeadPartition() {
            if(done){DeviceScope scope(secondary);cudaEventSynchronize(done);cudaEventDestroy(done);}
            if(ready){DeviceScope scope(primary);cudaEventDestroy(ready);}
        }
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
        std::unique_ptr<Pinned> result_host;size_t result_host_bytes=0;
        std::unique_ptr<Pinned> route_ids_host;
        std::unique_ptr<Pinned> grouped_host;
        uint64_t grouped_calls=0,grouped_experts=0,grouped_rows=0,grouped_scatters=0;
        uint64_t route_id_uploads=0,route_id_groups=0,route_id_entries=0;
        std::map<int,std::set<int>> keep;
        std::map<int,std::vector<uint64_t>> frequency;
        std::map<std::string,Matrix> dense;
        size_t half_cache_extra=0,half_cache_matrices=0,half_cache_skipped=0;
        std::map<std::string,std::unique_ptr<Device>> scratch;
        std::unique_ptr<PrefillBank> bank;
        std::unique_ptr<PrefillStream> stream_ring;
        uint64_t stream_prefetched=0,stream_consumed=0,stream_dropped=0,stream_demand=0,stream_admitted=0,stream_layers=0,evictions=0,room_evictions=0,stream_restored=0;
        size_t stream_frames_reported=0;std::set<std::pair<int,int>> lent;
        // An evicted lent entry leaves the cache, but its memory is a ring
        // frame until the pass ends.
        std::vector<std::unique_ptr<Device>> lent_retired;uint64_t lent_evicted=0,lent_freed=0;
        std::vector<std::pair<const std::pair<int,int>,Expert>*> restoring;
        uint64_t stream_small_direct=0,stream_small_staged=0;
        uint64_t bank_layers=0,bank_unused_experts=0;
        double bank_wait_ms=0;
        struct Trace {
            cudaEvent_t first=nullptr,last=nullptr;const char* stage;
            int rows=0,columns=0,tokens=0,format=-1;bool ended=false;
        };
        std::vector<Trace> traces;bool tracing=false;
        size_t trace_start(const char* stage,cudaStream_t s,int rows=0,int columns=0,int tokens=0,int format=-1) {
            if(!tracing)return SIZE_MAX;
            cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(s,&capture));
            if(capture!=cudaStreamCaptureStatusNone)return SIZE_MAX;
            Trace t{};t.stage=stage;t.rows=rows;t.columns=columns;t.tokens=tokens;t.format=format;
            check(cudaEventCreate(&t.first));
            try {check(cudaEventCreate(&t.last));check(cudaEventRecord(t.first,s));traces.push_back(t);}
            catch(...){cudaEventDestroy(t.first);if(t.last)cudaEventDestroy(t.last);throw;}
            return traces.size()-1;
        }
        void trace_end(size_t index,cudaStream_t s){if(index!=SIZE_MAX) {
            check(cudaEventRecord(traces[index].last,s));traces[index].ended=true;
        }}
        void trace_report(int layer,int tokens) {
            if(traces.empty())return;
            std::map<std::string,std::pair<double,size_t>> totals;
            for(const auto& t:traces) {
                if(!t.ended)continue;
                check(cudaEventSynchronize(t.last));
                float ms=0;check(cudaEventElapsedTime(&ms,t.first,t.last));
                if(t.rows) {
                    std::cerr<<"FP4_DENSE_INTERVAL device="<<device<<" tensor="<<t.stage
                        <<" rows="<<t.rows<<" columns="<<t.columns<<" tokens="<<t.tokens
                        <<" format="<<t.format<<" ms="<<ms<<'\n';
                    continue;
                }
                auto& value=totals[t.stage];value.first+=ms;++value.second;
            }
            for(auto& t:traces){cudaEventDestroy(t.first);cudaEventDestroy(t.last);}traces.clear();
            for(const auto& [stage,value]:totals)std::cerr<<"FP4_GPU_INTERVAL layer="<<layer<<" tokens="<<tokens
                <<" device="<<device<<" stage="<<stage<<" ms="<<value.first<<" calls="<<value.second<<'\n';
        }
        cudaStream_t copy_stream=nullptr;
        static constexpr size_t upload_capacity=16*MiB;
        std::array<std::unique_ptr<Pinned>,8> upload_host;
        std::array<cudaEvent_t,8> upload_ready{}, consumed{};
        int upload_slots=2;bool async_eviction=false;
        std::unique_ptr<Device> upload_device;
        std::set<std::pair<int,int>> protected_experts;
        int upload_slot=0;
        size_t cache_bytes=0,cache_limit=0,prefill_cache_limit=0;
        std::array<uint64_t,3> wire_bytes{};int phase=0;
        uint64_t batch_groups=0,batch_experts=0,batch_actual_rows=0,batch_padded_rows=0;
        uint64_t operand_groups=0,operand_rows=0;
        uint64_t fused_prefill_calls=0;
        uint64_t clock=0,hits=0,misses=0,uploaded=0,mma_calls=0,reference_calls=0,prefill_calls=0;
        explicit Context(int device,int worker_cpu=-1):device(device) {
            DeviceScope scope(device);
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cublasCreate(&blas));
            check(cublasSetStream(blas,stream));check(cublasSetMathMode(blas,CUBLAS_TENSOR_OP_MATH));
            std::promise<void> ready;auto started=ready.get_future();
            worker=std::thread([this,&ready,worker_cpu] {
                std::unique_ptr<DeviceScope> scope;
                std::unique_ptr<GpuLocality> locality;
                try {
                    scope=std::make_unique<DeviceScope>(this->device);locality=std::make_unique<GpuLocality>();
#ifdef __linux__
                    if(worker_cpu>=0) {
                        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(worker_cpu,&mask);
                        if(sched_setaffinity(0,sizeof(mask),&mask))throw std::runtime_error("GLM FP4: GPU worker CPU pinning failed");
                    }
#endif
                }
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
            if(bank&&bank->loading.valid())bank->loading.wait();
            {std::lock_guard lock(work_mutex);stop=true;}wake.notify_one();if(worker.joinable())worker.join();
            DeviceScope scope(device);if(stream)cudaStreamSynchronize(stream);if(copy_stream)cudaStreamSynchronize(copy_stream);
            bank.reset();stream_ring.reset();
            temporary.reset();experts.clear();dense.clear();scratch.clear();release_upload();
            for(auto& t:traces){cudaEventDestroy(t.first);cudaEventDestroy(t.last);}
            for(auto e:upload_ready)if(e)cudaEventDestroy(e);
            for(auto e:consumed)if(e)cudaEventDestroy(e);
            if(copy_stream)cudaStreamDestroy(copy_stream);
            if(blas)cublasDestroy(blas);
            if(stream)cudaStreamDestroy(stream);
        }
        std::future<void> submit(std::function<void()> job) {
            std::lock_guard lock(work_mutex);
            if(task.valid()||stop)throw std::logic_error("GLM FP4: GPU worker already has a pending task");
            task=std::packaged_task<void()>(std::move(job));auto done=task.get_future();wake.notify_one();return done;
        }
        size_t storage_limit()const {
            // The streamed ring lives outside the cache budget (see ensure_prefill_stream).
            const size_t reserved=bank?bank->allocated_bytes:0;
            return cache_limit>reserved?cache_limit-reserved:0;
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
            // Lent entries go last: they hold the oldest clocks, and a prompt
            // longer than one batch lends them again for every batch.
            for(int pass=0;pass<2&&victim==experts.end();++pass)
                for(auto it=experts.begin();it!=experts.end();++it)
                    if(!protected_experts.count(it->first)&&(pass||!lent.count(it->first))&&
                       (victim==experts.end()||it->second.use<victim->second.use))victim=it;
            if(victim==experts.end())throw std::runtime_error("GLM FP4: insufficient GPU memory for bounded working set");
            if(!async_eviction)check(cudaStreamSynchronize(stream));cache_bytes-=victim->second.bytes;
            if(lent.erase(victim->first)) {
                bool freed=false;
                if(stream_ring) {
                    auto& s=*stream_ring;std::lock_guard lock(s.mutex);
                    if(auto f=s.lent_frame.find(victim->first);f!=s.lent_frame.end()) {
                        const int frame=f->second;s.lent_frame.erase(f);
                        // Its last reader is already enqueued; keep at least eight frames.
                        if(s.refillable[frame]&&s.alive>8) {
                            check(cudaEventSynchronize(s.consumed[frame]));
                            s.dead[frame]=1;s.refillable[frame]=0;--s.alive;freed=true;
                        }
                    }
                }
                if(freed)++lent_freed;
                else {for(auto& m:victim->second.projection)lent_retired.push_back(std::move(m.data));++lent_evicted;}
            }
            experts.erase(victim);++evictions;
        }
        void init_upload() {
            if(!copy_stream)check(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
            if(!upload_device){room(upload_capacity);upload_device=std::make_unique<Device>(upload_capacity);}
            for(int i=0;i<upload_slots;++i) {
                if(!consumed[i])check(cudaEventCreateWithFlags(&consumed[i],cudaEventDisableTiming));
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
                while(pool->largest()<bytes||Device::limit()<Device::live()+bytes){evict();++room_evictions;}
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
    const bool dense_split_enabled=strata_env_is_one("STRATA_GLM_FP4_DENSE_SPLIT");
    const bool kda_qkv_batch=strata_env_is_one("STRATA_GLM_FP4_KDA_QKV_BATCH");
    const bool gpu_reduce_enabled=strata_env_is_one("STRATA_GLM_FP4_GPU_REDUCE");
    const bool expert_fusion=strata_env_is_one("STRATA_GLM_FP4_EXPERT_FUSION");
    const bool early_readback=strata_env_is_one("STRATA_GLM_FP4_EARLY_READBACK");
    const bool route_id_batch=strata_env_is_one("STRATA_GLM_FP4_ROUTE_ID_BATCH");
    const bool resident_grouped=strata_env_is_one("STRATA_GLM_FP4_RESIDENT_GROUPED");
    const bool grouped_scatter=strata_env_is_one("STRATA_GLM_FP4_GROUPED_SCATTER");
    const bool head_tp_enabled=strata_env_is_one("STRATA_GLM_FP4_HEAD_TP");
    const bool head_tp_borrow=strata_env_is_one("STRATA_GLM_FP4_HEAD_TP_BORROW");
    const long long fixed_half_mask=strata_env_value_ll("STRATA_GLM_FP4_FIXED_HALF",0);
    const long long fixed_half_max_mib=strata_env_value_ll("STRATA_GLM_FP4_FIXED_HALF_MAX_MIB",0);
    const bool fixed_half_borrow=strata_env_is_one("STRATA_GLM_FP4_FIXED_HALF_BORROW");
    const long long head_fp8_accumulators=strata_env_value_ll("STRATA_GLM_FP4_HEAD_ACCUMULATORS",0);
    const bool block_gate_pack=strata_env_is_one("STRATA_GLM_FP4_BLOCK_GATE_PACK");
    const bool prefill_bank_enabled=strata_env_is_one("STRATA_GLM_FP4_PREFILL_BANK");
    const bool prefill_bank_slab=strata_env_is_one("STRATA_GLM_FP4_PREFILL_BANK_SLAB");
    const bool prefill_operand_pipeline=strata_env_is_one("STRATA_GLM_FP4_PREFILL_OPERAND_PIPELINE");
    const bool prefill_operand_overlap=strata_env_is_one("STRATA_GLM_FP4_PREFILL_OPERAND_OVERLAP");
    const bool prefill_stream_enabled=strata_env_is_one("STRATA_GLM_FP4_PREFILL_STREAM");
    const long long prefill_stream_frames=strata_env_value_ll("STRATA_GLM_FP4_PREFILL_STREAM_FRAMES",64);
    // The secondary keeps little physical headroom outside its cache budget.
    const long long prefill_stream_frames1=strata_env_value_ll("STRATA_GLM_FP4_PREFILL_STREAM_FRAMES1",prefill_stream_frames);
    // Hold the primary's next-layer uploads until the secondary's partial sum
    // has crossed the same inbound PCIe link; they then fill during attention.
    const bool prefill_stream_hold=strata_env_is_one("STRATA_GLM_FP4_PREFILL_STREAM_HOLD");
    // Send the secondary its FP16 activations before the router and shared expert.
    const bool prefill_stream_early=strata_env_is_one("STRATA_GLM_FP4_PREFILL_STREAM_EARLY");
    const bool eviction_stats=strata_env_is_one("STRATA_GLM_FP4_EVICTION_STATS");
    // Memory kept free for the pass's later scratch allocations.
    int early_input_layer=-1;
    // The primary's held uploads are enqueued by its worker: a full copy
    // queue blocks the enqueuing thread, which must not be the coordinator.
    std::future<void> held_fill;
    void join_held_fill(){if(held_fill.valid())held_fill.get();}
    std::array<size_t,3> stream_projection_bytes{},stream_small_bytes{};int stream_planned_layer=-1;uint64_t stream_skipped=0;
    bool peer_enabled=false;
    const bool device_routing_enabled=strata_env_is_one("STRATA_GLM_FP4_DEVICE_ROUTING");
    bool decode_device_routing_enabled=strata_env_is_one("STRATA_GLM_FP4_DECODE_ROUTING");
    const bool async_upload_enabled=strata_env_is_one("STRATA_GLM_FP4_ASYNC_UPLOAD");
    const long long prefill_batch_experts=strata_env_value_ll("STRATA_GLM_FP4_PREFILL_BATCH_EXPERTS",1);
    const long long prefill_queue_depth=strata_env_value_ll("STRATA_GLM_FP4_PREFILL_QUEUE",1);
    const long long prefill_mma_tile=strata_env_value_ll("STRATA_GLM_FP4_PREFILL_MMA",0);
    const long long global_cache_mode=strata_env_value_ll("STRATA_GLM_FP4_GLOBAL_CACHE",0);
    const long long fp8_accumulators=strata_env_value_ll("STRATA_GLM_FP4_FP8_ACCUMULATORS",1);
    const long long cache_adapt_rows=strata_env_value_ll("STRATA_GLM_FP4_CACHE_ADAPT_ROWS",0);
    // Opt-in: once per request and layer, after this many decode rows, choose the layer's GPU experts again from
    // the last prefill batch's routes plus the decode routes so far (0 = off, the default).
    const long long keep_reselect_rows=strata_env_value_ll("STRATA_GLM_FP4_KEEP_RESELECT_ROWS",0);
    // ... at most this many swaps per device and layer (0 = no cap), and this many background uploads of the newly
    // kept experts per decode routed() call (0 = each is uploaded when it is first routed)
    const long long keep_reselect_swaps=strata_env_value_ll("STRATA_GLM_FP4_KEEP_RESELECT_SWAPS",0);
    const long long keep_reselect_prefetch=strata_env_value_ll("STRATA_GLM_FP4_KEEP_RESELECT_PREFETCH",0);
    const bool cache_total_free=strata_env_is_one("STRATA_GLM_FP4_CACHE_TOTAL_FREE");
    const bool cpu_logical=strata_env_is_one("STRATA_GLM_FP4_CPU_LOGICAL");
    const std::array<long long,2> worker_cpus={strata_env_value_ll("STRATA_GLM_FP4_WORKER_CPU0",-1),
                                             strata_env_value_ll("STRATA_GLM_FP4_WORKER_CPU1",-1)};
    const bool cpu_csf32_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_CSF32");
    const bool cpu_sparse_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE")||strata_env_is_one("STRATA_GLM_FP4_CPU_Q8");
    const bool cpu_q8_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_Q8");
    const bool cpu_fused_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_FUSED");
    const bool cpu_scale_quartets=strata_env_is_one("STRATA_GLM_FP4_CPU_SCALE_QUARTETS");
    const bool cpu_scale_local=strata_env_is_one("STRATA_GLM_FP4_CPU_SCALE_LOCAL");
    const bool cpu_scale_local_m1=strata_env_is_one("STRATA_GLM_FP4_CPU_SCALE_LOCAL_M1");
    const bool cpu_input_reuse=strata_env_is_one("STRATA_GLM_FP4_CPU_INPUT_REUSE");
    const bool cpu_q8_prepare=strata_env_is_one("STRATA_GLM_FP4_CPU_Q8_PREPARE");
    const bool cpu_q8_bias=strata_env_is_one("STRATA_GLM_FP4_CPU_Q8_BIAS");
    const bool cpu_q8_policy=strata_env_is_one("STRATA_GLM_FP4_CPU_Q8_POLICY");
    const long long cpu_team_size=strata_env_value_ll("STRATA_GLM_FP4_CPU_TEAM_SIZE",0);
    const long long cpu_row_tile=strata_env_value_ll("STRATA_GLM_FP4_CPU_ROW_TILE",1);
    const bool cpu_borrow_enabled=strata_env_is_one("STRATA_GLM_FP4_CPU_BORROW");
    const double route_drop_fraction=strata_env_value_double("STRATA_GLM_FP4_ROUTE_DROP",0);
    const double route_drop_cap=strata_env_value_double("STRATA_GLM_FP4_ROUTE_DROP_CAP",0.5);
    const long long route_drop_max=strata_env_value_ll("STRATA_GLM_FP4_ROUTE_DROP_MAX",2);
    const bool route_renorm_enabled=strata_env_is_one("STRATA_GLM_FP4_ROUTE_RENORM");
    const double route_affinity=strata_env_value_double("STRATA_GLM_FP4_ROUTE_AFFINITY",0);
    const bool route_affinity_secondary=strata_env_is_one("STRATA_GLM_FP4_ROUTE_AFFINITY_SECONDARY");
    std::map<int,std::unique_ptr<Device>> route_affinity_bonuses;
    const long long decode_reserve_mib=strata_env_value_ll("STRATA_GLM_FP4_DECODE_RESERVE_MIB",-1);
    // STRATA_GLM_FP4_CACHE_SHRINK0_MIB (diagnostic): GPU0's prefill and decode expert caches are this much
    // smaller, as a long context's latent cache would make them, with the attention state of a short one.
    const long long cache_shrink0_mib=strata_env_value_ll("STRATA_GLM_FP4_CACHE_SHRINK0_MIB",0);
    size_t shrink0(const Context& c,size_t limit)const {
        if(&c!=contexts.front().get()||!cache_shrink0_mib)return limit;
        const size_t bytes=size_t(cache_shrink0_mib)*MiB;return limit>bytes?limit-bytes:0;
    }
    uint64_t route_drop_tokens=0,route_drop_candidates=0,route_drop_count=0;
    double route_drop_share_sum=0,route_drop_share_max=0;
    double route_drop_multiplier_max=1;
    uint64_t decode_device_calls=0,decode_host_calls=0;
    uint64_t early_readback_calls=0;
    bool decode_phase=false;
    std::map<int,std::array<uint8_t,288>> decode_owners;
    std::map<int,std::vector<uint64_t>> priority_frequency;
    std::map<int,std::array<uint64_t,288>> decode_heat;
    std::map<int,size_t> decode_heat_rows;
    uint64_t cache_promotions=0,cache_reclaimed_bytes=0;
    // KEEP_RESELECT_ROWS: this request's decode routes per layer (those that survive the all-CPU drop budget, as
    // adapt_cache counts them), its rows, and the layers already chosen again
    std::map<int,std::array<uint64_t,288>> reselect_heat;
    std::map<int,size_t> reselect_rows;
    std::set<int> reselected,stale_bonuses;
    uint64_t reselect_layers=0,reselect_swaps=0,reselect_reclaimed_bytes=0;
    uint64_t reselect_prefetched=0,reselect_prefetch_skipped=0,reselect_prefetch_dropped=0;
    size_t owner(int layer,int id,bool decode)const {
        if(decode&&global_cache_mode>=2) {
            const auto it=decode_owners.find(layer);
            if(it!=decode_owners.end())return it->second.at(size_t(id));
        }
        return size_t(id)%contexts.size();
    }
    using Clock=std::chrono::steady_clock;
    Clock::time_point tick()const {return profile?Clock::now():Clock::time_point{};}
    double elapsed(Clock::time_point start)const {
        return profile?std::chrono::duration<double,std::milli>(Clock::now()-start).count():0;
    }
    uint64_t cpu_experts=0;
    std::array<uint64_t,9> cpu_width_groups{};
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
    std::unique_ptr<HeadPartition> head_partition;
    std::unique_ptr<Pinned> host_x,host_routes,host_weights;
    size_t host_rows=0,host_x_rows=0;
    // STRATA_GLM_PREFILL_LEAN_HOST=1: a device-routed prefill never reads the host activation rows, so it
    // skips their pinned buffer (128 MiB at 8,192 tokens, allocated per prompt and freed at compact).
    const bool lean_host=strata_env_truthy_one("STRATA_GLM_PREFILL_LEAN_HOST");
    void ensure_host_rows(int nt,cudaStream_t caller,bool activation) {
        const bool rows=size_t(nt)>host_rows,x=activation&&size_t(nt)>host_x_rows;
        if(!rows&&!x)return;
        check(cudaStreamSynchronize(caller));
        if(rows) {
            host_rows=nt;
            host_routes=std::make_unique<Pinned>(size_t(nt)*8*4,true);
            host_weights=std::make_unique<Pinned>(size_t(nt)*8*4,true);
        }
        if(x){host_x_rows=nt;host_x=std::make_unique<Pinned>(size_t(nt)*4096*4,true);}
    }
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
                    if(cpu_scale_quartets)cpu::glm_fp4_cache_scale_quartets(*e->sparse[p]);
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
    void cpu_product(const CpuExpert& e,int projection,const float* x,float* y,
                            int columns,int rows,int tokens,int first,int last,const cpu::GlmFp4Q8* q8=nullptr) {
        const auto& packed=e.projection[projection];
        if(e.sparse[projection]&&q8)cpu::glm_fp4_rows_csf32_q8(packed.data(),*e.sparse[projection],e.scales[projection].data(),*q8,y,columns,rows,tokens,first,last,int(cpu_row_tile),cpu_scale_local||(cpu_scale_local_m1&&tokens==1),cpu_q8_policy);
        else if(e.sparse[projection])cpu::glm_fp4_rows_csf32_sparse_table(packed.data(),*e.sparse[projection],e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
        else if(e.csf32[projection])cpu::glm_fp4_rows_csf32_table(packed.data(),*e.csf32[projection],e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
        else cpu::glm_fp4_rows_table(packed.data(),packed.scales.data(),e.scales[projection].data(),x,y,columns,rows,tokens,first,last);
    }
    // `into` re-packs a matrix in its existing allocation; `stage` replaces
    // the packed_source scratch. Both leave the arena's free blocks alone.
    Matrix upload(Context& c,const strata::artifact::ModelOptMatrix& desc,std::unique_ptr<Device> into={},Device* stage=nullptr,
                  cudaStream_t stream=nullptr) {
        // `stream` replaces the context stream, which dense() may swap meanwhile.
        cudaStream_t on=stream?stream:c.stream;
        auto copy=[&](void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) {
            if(kind==cudaMemcpyHostToDevice)c.wire_bytes[c.phase]+=bytes;
            return cudaMemcpyAsync(dst,src,bytes,kind,stream);
        };
        auto allocate=[&](Matrix& m,size_t bytes) {
            if(!into){c.room(bytes);m.data=std::make_unique<Device>(bytes);return;}
            if(into->bytes!=bytes)throw std::logic_error("GLM FP4: re-packed matrix changed size: "+desc.name);
            m.data=std::move(into);
        };
        auto staging_buffer=[&](size_t bytes)->Device& {
            if(!stage)return c.buffer("packed_source",bytes);
            if(stage->bytes<bytes)throw std::logic_error("GLM FP4: re-pack staging is too small: "+desc.name);
            return *stage;
        };
        if(model.has_csf32(desc)) {
            const auto h=model.csf32(desc);
            Matrix m;m.format=k::GlmPackedFormat::Nvfp4;m.rows=desc.rows;m.columns=desc.columns;m.global=h.global;
            if(sm70 && (!std::isfinite(h.global*16384.f) || h.global*16384.f>65504.f))
                throw std::runtime_error("GLM FP4: global scale exceeds SM70 decoder range");
            const size_t bytes=raw_bytes(desc),scales=desc.rows*desc.columns/16;
            m.scale_offset=(bytes+15)/16*16;
            allocate(m,m.scale_offset+scales);
            auto align=[](size_t n){return (n+15)/16*16;};
            const size_t base=align(bytes),code=align(base+desc.rows),offset=align(code+h.codes_count);
            const size_t page=align(offset+h.offsets.size()*2),exception=align(page+h.pages.size()*4);
            const size_t flag=align(exception+h.exceptions.size()*2);
            auto& staging=staging_buffer(flag+sizeof(int));
            auto* p=static_cast<uint8_t*>(staging.p);
            const auto* raw=artifact.exl3().tensors.data(*desc.weights);
            check(copy(p,raw,bytes,cudaMemcpyHostToDevice,on));
            check(copy(p+base,h.bases,desc.rows,cudaMemcpyHostToDevice,on));
            check(copy(p+code,h.codes,h.codes_count,cudaMemcpyHostToDevice,on));
            check(copy(p+offset,scale_source(desc,"offsets",h.offsets.data()),h.offsets.size()*2,cudaMemcpyHostToDevice,on));
            check(copy(p+page,scale_source(desc,"page_bases",h.pages.data()),h.pages.size()*4,cudaMemcpyHostToDevice,on));
            if(!h.exceptions.empty())check(copy(p+exception,scale_source(desc,"exceptions",h.exceptions.data()),h.exceptions.size()*2,cudaMemcpyHostToDevice,on));
            k::glm_fp4_prepack(p,nullptr,static_cast<uint8_t*>(m.data->p),nullptr,m.rows,m.columns,on);
            k::glm_fp4_csf32_scales(p+base,p+code,reinterpret_cast<const uint16_t*>(p+offset),
                reinterpret_cast<const uint32_t*>(p+page),reinterpret_cast<const uint16_t*>(p+exception),
                static_cast<uint8_t*>(m.data->p)+m.scale_offset,reinterpret_cast<int*>(p+flag),m.rows,m.columns,on);
            int invalid=0;check(copy(&invalid,p+flag,sizeof(int),cudaMemcpyDeviceToHost,on));
            check(cudaStreamSynchronize(on));
            if(invalid)throw std::runtime_error("GLM ModelOpt: invalid FP4 group scale: "+desc.name);
            c.uploaded+=m.data->bytes;return m;
        }
        const auto h=model.load(desc,true);
        Matrix m;m.format=static_cast<k::GlmPackedFormat>(h.format);m.rows=h.rows;m.columns=h.columns;m.global=h.global;
        const size_t bytes=raw_bytes(desc);m.scale_offset=(bytes+15)/16*16;
        // Measured on V100: MXFP8 benefits from coalesced lane fragments.
        // The block-FP8 shared-expert shape is faster in its row-major layout.
        const bool packed_mxfp8=h.format==Format::Mxfp8&&h.rows%32==0;
        const bool packed_block_gate=block_gate_pack&&h.format==Format::BlockFp8&&h.rows==2048&&h.columns==4096;
        const size_t scales=packed_mxfp8?h.scales.size()*2:h.scales.size();
        if(into&&h.format!=Format::Nvfp4)throw std::logic_error("GLM FP4: only FP4 matrices are re-packed in place");
        allocate(m,m.scale_offset+scales);
        if(h.format==Format::Nvfp4) {
            if(sm70 && (!std::isfinite(h.global*16384.f) || h.global*16384.f>65504.f))
                throw std::runtime_error("GLM FP4: global scale exceeds SM70 decoder range");
            auto& staging=staging_buffer(bytes+h.scales.size());
            check(copy(staging.p,h.data(),bytes,cudaMemcpyHostToDevice,on));
            check(copy(static_cast<char*>(staging.p)+bytes,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,on));
            k::glm_fp4_prepack(static_cast<const uint8_t*>(staging.p),static_cast<const uint8_t*>(staging.p)+bytes,
                static_cast<uint8_t*>(m.data->p),static_cast<uint8_t*>(m.data->p)+m.scale_offset,m.rows,m.columns,on);
        }else if(packed_mxfp8||packed_block_gate) {
            auto& staging=c.buffer("packed_source",bytes+h.scales.size());
            check(copy(staging.p,h.data(),bytes,cudaMemcpyHostToDevice,on));
            check(copy(static_cast<char*>(staging.p)+bytes,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,on));
            k::glm_fp8_prepack(m.format,static_cast<const uint8_t*>(staging.p),static_cast<char*>(staging.p)+bytes,
                static_cast<uint8_t*>(m.data->p),static_cast<char*>(m.data->p)+m.scale_offset,m.rows,m.columns,on);
            m.format=packed_mxfp8?k::GlmPackedFormat::Mxfp8Qpn:k::GlmPackedFormat::BlockFp8Qpn;
        }else {
            check(copy(m.data->p,h.data(),bytes,cudaMemcpyHostToDevice,on));
            if(!h.scales.empty())check(copy(static_cast<char*>(m.data->p)+m.scale_offset,h.scales.data(),h.scales.size(),cudaMemcpyHostToDevice,on));
            if(h.format==Format::Bf16) {
                // Cache the exact FP16 operand the W4A16 product would form.
                // BF16 and FP16 occupy the same bytes; no second matrix lives
                // on either the host or the device.
                k::glm_fp4_dequant_f16(m.format,m.data->p,m.scales(),m.global,
                    static_cast<uint16_t*>(m.data->p),m.rows,m.columns,on);
                m.format=k::GlmPackedFormat::F16;
            }
        }
        // Scale vectors are bounded host temporaries; complete their DMA before
        // releasing them. The mapped source bytes remain owned by the artifact.
        check(cudaStreamSynchronize(on));c.uploaded+=m.data->bytes;
        return m;
    }
    Matrix upload_head_tail(Context& c,const strata::artifact::ModelOptMatrix& desc,int first,int rows) {
        const auto h=model.load(desc,true);
        if(h.format!=Format::Mxfp8||first<0||rows<1||first%32||rows%32||
           size_t(first)>h.rows||size_t(rows)>h.rows-size_t(first))
            throw std::invalid_argument("GLM FP4: head partition requires aligned MXFP8 rows");
        Matrix m;m.format=k::GlmPackedFormat::Mxfp8Qpn;m.rows=rows;m.columns=h.columns;m.global=h.global;
        const size_t bytes=size_t(rows)*h.columns,scale_bytes=bytes/32;
        m.scale_offset=bytes;
        c.room(bytes+2*scale_bytes);m.data=std::make_unique<Device>(bytes+2*scale_bytes);
        auto& staging=c.buffer("packed_source",bytes+scale_bytes);
        auto* source=static_cast<uint8_t*>(staging.p);
        try {
            check(cudaMemcpyAsync(source,h.data()+size_t(first)*h.columns,bytes,cudaMemcpyHostToDevice,c.stream));
            check(cudaMemcpyAsync(source+bytes,h.scales.data()+size_t(first)*h.columns/32,
                                  scale_bytes,cudaMemcpyHostToDevice,c.stream));
            c.wire_bytes[c.phase]+=bytes+scale_bytes;
            k::glm_fp8_prepack(k::GlmPackedFormat::Mxfp8,source,source+bytes,
                static_cast<uint8_t*>(m.data->p),static_cast<uint8_t*>(m.data->p)+m.scale_offset,rows,m.columns,c.stream);
            check(cudaGetLastError());check(cudaStreamSynchronize(c.stream));
        }catch(...){cudaStreamSynchronize(c.stream);throw;}
        c.uploaded+=m.data->bytes;return m;
    }
    Expert& expert(Context& c,int layer,int id,bool admit=true) {
        c.temporary.reset();
        const auto key=std::make_pair(layer,id);auto it=c.experts.find(key);
        if(it!=c.experts.end()){++c.hits;it->second.use=++c.clock;return it->second;}
        ++c.misses;
        // Fixed per-device admission leaves room for scratch and future dense
        // operands. Eviction completes readers before releasing their storage.
        while(admit && !c.experts.empty() && c.cache_bytes+expert_bytes>c.storage_limit())c.evict();
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
        cudaStream_t reader_stream=nullptr;
        int slot=-1,done_slot=-1;bool completed=false;
        PendingExpert()=default;
        PendingExpert(PendingExpert&&)=default;
        PendingExpert& operator=(PendingExpert&&)=default;
        ~PendingExpert() {
            // An exception may leave either DMA or a matrix reader queued.
            if(owned&&context&&!completed) {
                cudaStreamSynchronize(context->copy_stream);
                if(reader_stream)cudaStreamSynchronize(reader_stream);
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
        result.slot=c.upload_slot;c.upload_slot=(c.upload_slot+1)%c.upload_slots;
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
        const auto copy_trace=c.trace_start("matrix_h2d",c.copy_stream);
        if(registered_ram) {
            for(const auto& copy:copies){c.wire_bytes[c.phase]+=copy.bytes;check(cudaMemcpyAsync(device+copy.offset,copy.source,copy.bytes,cudaMemcpyHostToDevice,c.copy_stream));}
        }else {c.wire_bytes[c.phase]+=used;check(cudaMemcpyAsync(device,host,used,cudaMemcpyHostToDevice,c.copy_stream));}
        c.trace_end(copy_trace,c.copy_stream);
        const auto pack_trace=c.trace_start("prepack_scales",c.copy_stream);
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
        c.trace_end(pack_trace,c.copy_stream);
        check(cudaMemcpyAsync(host+flag,device+flag,3*sizeof(int),cudaMemcpyDeviceToHost,c.copy_stream));
        check(cudaEventRecord(c.upload_ready[result.slot],c.copy_stream));
        c.uploaded+=result.value->bytes;return result;
    }
    void finish_expert(Context& c,PendingExpert& pending,bool admit) {
        c.protected_experts.erase(pending.key);
        if(!pending.owned||!admit)return;
        while(!c.experts.empty()&&c.cache_bytes+pending.value->bytes>c.storage_limit())c.evict();
        pending.value->use=++c.clock;c.cache_bytes+=pending.value->bytes;
        c.experts.emplace(pending.key,std::move(*pending.owned));pending.owned.reset();
    }
    static void upload_bank(PrefillBank& b) {
        // The coordinator can allocate dense scratch while this runs. Touch
        // only the bank and its stream; never use Context::stream or allocator.
        DeviceScope scope(b.device);
        auto* frame=static_cast<uint8_t*>(b.frame->p);
        try {
            check(cudaMemsetAsync(frame+PrefillBank::flag_offset,0,PrefillBank::count*3*4,b.stream));
            for(size_t slot=0;slot<PrefillBank::count;++slot) {
                if(b.cached[slot])continue;
                struct Copy {const void* data;size_t offset,bytes;};
                std::array<Copy,18> copies{};size_t used=0,ncopy=0;
                std::array<std::array<size_t,6>,3> offsets{};
                for(int p=0;p<3;++p) {
                    const auto& source=*b.sources[slot][p];
                    for(int part=0;part<(source.csf?6:2);++part) {
                        const size_t at=(used+15)/16*16,bytes=source.bytes[part];
                        if(at>PrefillBank::flag_offset||bytes>PrefillBank::flag_offset-at||
                           (bytes&&!source.data[part]))
                            throw std::runtime_error("GLM FP4: prefill bank upload frame overflow");
                        offsets[p][part]=at;used=at+bytes;
                        if(bytes)copies[ncopy++]={source.data[part],at,bytes};
                    }
                }
                for(size_t i=0;i<ncopy;++i) {
                    const auto& copy=copies[i];
                    check(cudaMemcpyAsync(frame+copy.offset,copy.data,copy.bytes,cudaMemcpyHostToDevice,b.stream));
                    b.wire_bytes+=copy.bytes;
                }
                for(int p=0;p<3;++p) {
                    const auto& source=*b.sources[slot][p];const auto& at=offsets[p];
                    auto& m=b.experts[slot].projection[p];auto* out=static_cast<uint8_t*>(m.data->p);
                    k::glm_fp4_prepack(frame+at[0],source.csf?nullptr:frame+at[1],out,
                        source.csf?nullptr:out+m.scale_offset,m.rows,m.columns,b.stream);
                    if(source.csf)k::glm_fp4_csf32_scales(frame+at[1],frame+at[2],
                        reinterpret_cast<const uint16_t*>(frame+at[3]),reinterpret_cast<const uint32_t*>(frame+at[4]),
                        reinterpret_cast<const uint16_t*>(frame+at[5]),out+m.scale_offset,
                        reinterpret_cast<int*>(frame+PrefillBank::flag_offset)+slot*3+p,m.rows,m.columns,b.stream);
                }
                check(cudaGetLastError());
            }
            check(cudaMemcpyAsync(b.flags->p,frame+PrefillBank::flag_offset,PrefillBank::count*3*4,
                                  cudaMemcpyDeviceToHost,b.stream));
            check(cudaStreamSynchronize(b.stream));
            const auto* flags=static_cast<const int*>(b.flags->p);
            for(size_t i=0;i<PrefillBank::count*3;++i)if(flags[i])
                throw std::runtime_error("GLM ModelOpt: invalid FP4 scale in prefill bank");
        }catch(...){cudaStreamSynchronize(b.stream);throw;}
    }
    void join_bank(Context& c) {
        if(!c.bank||!c.bank->loading.valid())return;
        auto& b=*c.bank;const auto begin=Clock::now();b.loading.get();
        c.bank_wait_ms+=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        c.wire_bytes[1]+=b.wire_bytes;c.hits+=b.cached_count;
        const size_t cold=PrefillBank::count-b.cached_count;
        c.misses+=cold;c.uploaded+=cold*expert_bytes;
        ++c.bank_layers;c.bank_unused_experts+=PrefillBank::count;
    }
    void release_banks() {
        for(auto& c:contexts) {
            join_bank(*c);
            if(c->bank){DeviceScope scope(c->device);c->bank.reset();c->protected_experts.clear();}
        }
    }
    void start_bank_layer(int layer) {
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=*contexts[rank];auto& b=*c.bank;
            if(b.loading.valid()||!c.protected_experts.empty())
                throw std::logic_error("GLM FP4: prefill bank has active readers");
            b.layer=layer;b.cached_count=0;b.wire_bytes=0;b.cached.fill(nullptr);
            for(size_t slot=0;slot<PrefillBank::count;++slot) {
                const int id=int(slot*2+rank);const auto key=std::make_pair(layer,id);
                if(auto it=c.experts.find(key);it!=c.experts.end()) {
                    b.cached[slot]=&it->second;++b.cached_count;c.protected_experts.insert(key);
                }
                for(int p=0;p<3;++p) {
                    const auto& source=expert_source(layer,id,p);auto& m=b.experts[slot].projection[p];
                    if(source.rows!=m.rows||source.columns!=m.columns||source.bytes[0]!=m.scale_offset||
                       !std::isfinite(source.global*16384.f)||source.global*16384.f>65504.f)
                        throw std::runtime_error("GLM FP4: unsupported prefill bank matrix");
                    m.global=source.global;b.sources[slot][p]=&source;
                }
            }
            b.loading=c.submit([&b]{upload_bank(b);});
        }
    }
    void ensure_prefill_banks() {
        constexpr size_t reserved=PrefillBank::count*expert_bytes+PrefillBank::frame_bytes;
        static_assert(reserved<2048*MiB);
        for(auto& ctx:contexts) {
            auto& c=*ctx;DeviceScope scope(c.device);
            if(c.bank||!c.protected_experts.empty())throw std::logic_error("GLM FP4: prefill bank already allocated");
            if(c.cache_limit<reserved+64*MiB)throw std::runtime_error("GLM FP4: prefill bank needs a draft reserve");
            // Borrow the main pass's unused draft reserve; restore it before
            // batched draft priming. Keep select()'s nominal budget unchanged.
            for(auto it=c.experts.begin();it!=c.experts.end();) {
                if(it->first.first==45){c.cache_bytes-=it->second.bytes;it=c.experts.erase(it);}
                else ++it;
            }
            while(c.cache_bytes>c.cache_limit-reserved)c.evict();
            auto b=std::make_unique<PrefillBank>(c.device);
            if(prefill_bank_slab) {
                c.room(reserved);b->slab=std::make_unique<Device>(reserved);
            }
            for(auto& expert:b->experts)for(int p=0;p<3;++p) {
                auto& m=expert.projection[p];m.format=k::GlmPackedFormat::Nvfp4;
                m.rows=p==2?4096:2048;m.columns=p==2?2048:4096;
                m.scale_offset=size_t(m.rows)*m.columns/2;
                const size_t bytes=m.scale_offset+size_t(m.rows)*m.columns/16;
                if(b->slab)m.data=std::make_unique<Device>(
                    static_cast<uint8_t*>(b->slab->p)+b->allocated_bytes,bytes);
                else {c.room(bytes);m.data=std::make_unique<Device>(bytes);}
                b->allocated_bytes+=bytes;expert.bytes+=bytes;
            }
            if(b->slab)b->frame=std::make_unique<Device>(
                static_cast<uint8_t*>(b->slab->p)+b->allocated_bytes,PrefillBank::frame_bytes);
            else {c.room(PrefillBank::frame_bytes);b->frame=std::make_unique<Device>(PrefillBank::frame_bytes);}
            b->allocated_bytes+=PrefillBank::frame_bytes;
            if(b->allocated_bytes!=reserved)throw std::logic_error("GLM FP4: prefill bank size mismatch");
            if(prefill_bank_slab)std::cerr<<"FP4_PREFILL_BANK_SLAB device="<<c.device
                <<" bytes="<<b->allocated_bytes<<'\n';
            b->flags=std::make_unique<Pinned>(PrefillBank::count*3*4,true);c.bank=std::move(b);
        }
        start_bank_layer(3);
    }
    // Each projection's parts keep prepare_expert's 16-byte alignment.
    static std::array<size_t,3> stream_layout(const std::array<const ExpertSource*,3>& sources,std::array<std::array<size_t,6>,3>& offsets) {
        std::array<size_t,3> used{};
        for(int p=0;p<3;++p)for(int part=0;part<(sources[p]->csf?6:2);++part) {
            const size_t at=(used[p]+15)/16*16;offsets[p][part]=at;used[p]=at+sources[p]->bytes[part];
        }
        return used;
    }
    bool ensure_prefill_stream() {
        if(!stream_projection_bytes[0]) {
            // Resolve every routed source once. Workers then only read the
            // cached descriptors while they refill frames.
            std::array<std::array<size_t,6>,3> offsets;
            for(int layer=3;layer<45;++layer)for(int id=0;id<288;++id) {
                std::array<const ExpertSource*,3> sources;
                for(int p=0;p<3;++p) {
                    const auto& source=expert_source(layer,id,p);sources[p]=&source;
                    if(source.rows!=(p==2?4096:2048)||source.columns!=(p==2?2048:4096)||
                       !std::isfinite(source.global*16384.f)||source.global*16384.f>65504.f)
                        throw std::runtime_error("GLM FP4: unsupported streamed prefill matrix");
                    for(int part=0;part<(source.csf?6:2);++part)
                        if(source.bytes[part]&&!source.data[part])throw std::runtime_error("GLM FP4: streamed prefill needs registered sources");
                }
                const auto used=stream_layout(sources,offsets);
                for(int p=0;p<3;++p) {
                    stream_projection_bytes[p]=std::max(stream_projection_bytes[p],(used[p]+255)/256*256);
                    if(sources[p]->csf)stream_small_bytes[p]=std::max(stream_small_bytes[p],(used[p]-offsets[p][1]+255)/256*256);
                }
            }
        }
        constexpr size_t operand_bytes=PrefillStream::operand_elements*2;
        const size_t frame_bytes=stream_projection_bytes[0]+stream_projection_bytes[1]+stream_projection_bytes[2];
        bool usable=true;
        for(auto& ctx:contexts) {
            auto& c=*ctx;DeviceScope scope(c.device);
            const size_t limit=size_t(&c==contexts.front().get()?prefill_stream_frames:prefill_stream_frames1);
            if(c.stream_ring||c.bank||!c.protected_experts.empty()||!c.lent.empty()||!c.lent_retired.empty())
                throw std::logic_error("GLM FP4: prefill stream already allocated");
            c.init_upload();
            auto s=std::make_unique<PrefillStream>(c.device);s->create_events(limit);
            // The ring never takes arena memory: the original path's later
            // scratch and admissions must find the same free blocks, and the
            // decode budget is set from the largest one. The resident draft
            // experts are unused until this pass ends; their projections hold
            // the frames and are re-packed in place afterwards. Lent entries
            // keep their budget bytes and LRU clock.
            size_t lent_bytes=0;
            // Lend the most recently used entries: eviction takes the least
            // recently used, and an evicted lent entry cannot free its memory
            // until the pass ends.
            std::vector<std::pair<const std::pair<int,int>,Expert>*> draft;
            for(auto& entry:c.experts) {
                if(entry.first.first!=45)continue;
                bool fits=true;
                for(int p=0;p<3;++p)fits=fits&&entry.second.projection[p].data&&entry.second.projection[p].data->bytes>=stream_projection_bytes[p];
                if(fits)draft.push_back(&entry);
            }
            std::stable_sort(draft.begin(),draft.end(),[](auto* a,auto* b){return a->second.use>b->second.use;});
            for(auto* entry:draft) {
                if(s->frames.size()>=limit)break;
                auto& e=entry->second;
                s->frames.push_back({static_cast<uint8_t*>(e.projection[0].data->p),static_cast<uint8_t*>(e.projection[1].data->p),
                                     static_cast<uint8_t*>(e.projection[2].data->p)});
                c.lent.insert(entry->first);lent_bytes+=e.bytes;
                s->lent_frame[entry->first]=int(s->frames.size()-1);
            }
            s->alive=s->frames.size();
            bool ready=s->frames.size()>=8;
            // Operand slots and flags use the driver margin outside the arena;
            // keep most of it for the driver and libraries.
            const size_t outside_bytes=PrefillStream::operand_slots*3*operand_bytes+PrefillStream::flag_count*4;
            size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
            if(ready&&free<outside_bytes+384*MiB)ready=false;
            auto take=[&](size_t bytes)->void* {
                void* p=nullptr;check(cudaMalloc(&p,bytes));s->outside.push_back(p);s->allocated_bytes+=bytes;return p;
            };
            if(ready) {
                for(auto& slot:s->operands)for(auto& part:slot)part=static_cast<uint16_t*>(take(operand_bytes));
                s->flags=static_cast<int*>(take(PrefillStream::flag_count*4));
            }
            if(s->frames.size()!=c.stream_frames_reported)std::cerr<<"FP4_PREFILL_STREAM_RING device="<<c.device
                <<" frames="<<s->frames.size()<<" limit="<<limit<<" outside_bytes="<<s->allocated_bytes
                <<" outside_free="<<free<<" lent_experts="<<c.lent.size()<<" lent_bytes="<<lent_bytes
                <<" frame_bytes="<<frame_bytes<<'\n';
            c.stream_frames_reported=s->frames.size();
            s->flags_host=std::make_unique<Pinned>(PrefillStream::flag_count*4,true);
            if(ready)s->staging=std::make_unique<Pinned>(std::max<size_t>(s->frames.size()*stream_stage_bytes(),1),true);
            c.stream_ring=std::move(s);
            if(!ready)usable=false;
        }
        // Too few draft experts or margin: run this pass on the original path.
        // Nothing was written to the lent projections yet.
        if(!usable){for(auto& c:contexts)c->lent.clear();release_prefill_stream();++stream_skipped;return false;}
        stream_planned_layer=-1;return true;
    }
    void restore_lent(Context& c) {
        // Re-pack lent entries the budget still holds in their own memory; an
        // evicted entry stays evicted, exactly as on the original path. An
        // operand slot (outside the arena) stages the source, and the copy
        // stream carries the work: dense() may swap the context stream.
        const char* names[]={"gate_proj","up_proj","down_proj"};
        Device stage(c.stream_ring->operands[0][0],PrefillStream::operand_elements*2);
        for(auto* e:c.restoring) {
            for(int p=0;p<3;++p) {
                auto data=std::move(e->second.projection[p].data);
                e->second.projection[p]=upload(c,model.matrix(strata::artifact::GlmExl3::prefix(e->first.first,e->first.second,names[p])+".weight"),
                                               std::move(data),&stage,c.copy_stream);
            }
            ++c.stream_restored;
        }
    }
    // The last routed layer releases the ring with `overlap`: the workers
    // re-pack the lent entries while the coordinator runs the output head and
    // the draft priming, which reads no draft experts in native FP4. Lent
    // entries stay protected until join_restore(); the next routed layer,
    // prefill or decode transition joins first.
    std::vector<std::future<void>> restore_work;
    bool restore_pending=false;
    // Streamed batches since the last decode. A batch that follows another
    // meets the layers in the same cyclic order, so LRU admissions would
    // evict the layer's cached experts before their jobs reach them.
    int stream_followups=0;
    void join_restore() {
        if(!restore_pending)return;
        restore_pending=false;
        std::exception_ptr failure;
        for(auto& f:restore_work)try{f.get();}catch(...){if(!failure)failure=std::current_exception();}
        restore_work.clear();
        for(auto& c:contexts)if(c->stream_ring) {
            for(const auto& key:c->lent)c->protected_experts.erase(key);
            c->lent.clear();c->restoring.clear();
            DeviceScope scope(c->device);c->lent_retired.clear();c->stream_ring.reset();
        }
        if(failure)std::rethrow_exception(failure);
    }
    void release_prefill_stream(bool overlap=false) {
        join_held_fill();join_restore();
        for(auto& c:contexts)if(c->stream_ring) {
            DeviceScope scope(c->device);
            if(c->copy_stream)cudaStreamSynchronize(c->copy_stream);
            cudaStreamSynchronize(c->stream_ring->prep);cudaStreamSynchronize(c->stream);
            c->protected_experts.clear();
            // Resolve the entries here: the workers then touch no map.
            for(const auto& key:c->lent) {
                auto it=c->experts.find(key);
                if(it==c->experts.end())throw std::logic_error("GLM FP4: lent expert left the cache unrecorded");
                c->restoring.push_back(&*it);c->protected_experts.insert(key);
            }
        }
        // Restore concurrently; each worker owns its device stream.
        restore_pending=true;
        try {
            for(auto& c:contexts)if(!c->restoring.empty())
                restore_work.push_back(c->submit([this,&c]{DeviceScope scope(c->device);restore_lent(*c);}));
        }catch(...){join_restore();throw;}
        if(!overlap)join_restore();
        stream_planned_layer=-1;
    }
    // Ownership is the prefill parity split. Plan on the coordinator while
    // workers are idle; the cache state is read, never changed, here.
    void plan_stream_layer(int layer) {
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=*contexts[rank];
            for(int id=0;id<288;++id)
                if(owner(layer,id,false)==rank&&!c.experts.count({layer,id}))c.stream_ring->planned.emplace_back(layer,id);
        }
        stream_planned_layer=layer;
    }
    size_t stream_stage_bytes()const {return stream_small_bytes[0]+stream_small_bytes[1]+stream_small_bytes[2];}
    void stream_fill(Context& c,int last_layer=INT_MAX) {
        auto& s=*c.stream_ring;
        std::lock_guard lock(s.mutex);
        for(;;) {
            while(s.dead[s.head])s.head=(s.head+1)%s.frames.size();
            if(s.planned.empty()||s.planned.front().first>last_layer||!s.refillable[s.head])break;
            PrefillStream::Item item;
            std::tie(item.layer,item.id)=s.planned.front();s.planned.pop_front();
            item.frame=int(s.head);
            for(int p=0;p<3;++p)item.sources[p]=&expert_source(item.layer,item.id,p);
            const auto used=stream_layout(item.sources,item.offsets);
            for(int p=0;p<3;++p)if(used[p]>stream_projection_bytes[p])throw std::logic_error("GLM FP4: streamed expert exceeds its frame");
            const auto& frame=s.frames[s.head];
            check(cudaStreamWaitEvent(c.copy_stream,s.consumed[s.head],0));
            // The slot's previous copy was enqueued a ring ago; normally done.
            auto* slot=static_cast<uint8_t*>(s.staging->p)+size_t(s.head)*stream_stage_bytes();
            check(cudaEventSynchronize(s.full[s.head]));
            const auto trace=c.trace_start("matrix_h2d",c.copy_stream);
            for(int p=0;p<3;++p) {
                const auto& source=*item.sources[p];const auto& at=item.offsets[p];
                auto copy=[&](size_t offset,const void* from,size_t bytes) {
                    if(!bytes)return;
                    check(cudaMemcpyAsync(frame[p]+offset,from,bytes,cudaMemcpyHostToDevice,c.copy_stream));
                    c.wire_bytes[c.phase]+=bytes;
                };
                copy(at[0],source.data[0],source.bytes[0]);
                if(!source.csf){copy(at[1],source.data[1],source.bytes[1]);continue;}
                // Parts 1-5 occupy [at[1], end) in the frame. Copy them from
                // the source directly when it has the same spacing.
                const size_t end=at[5]+source.bytes[5];
                bool direct=true;
                for(int part=2;part<6;++part)if(source.bytes[part])
                    direct=direct&&static_cast<const uint8_t*>(source.data[part])==static_cast<const uint8_t*>(source.data[1])+(at[part]-at[1]);
                if(direct){copy(at[1],source.data[1],end-at[1]);++c.stream_small_direct;continue;}
                auto* staged=slot+(p?stream_small_bytes[0]:0)+(p==2?stream_small_bytes[1]:0);
                if(end-at[1]>stream_small_bytes[p])throw std::logic_error("GLM FP4: streamed scale parts exceed their slot");
                for(int part=1;part<6;++part)if(source.bytes[part])std::memcpy(staged+(at[part]-at[1]),source.data[part],source.bytes[part]);
                copy(at[1],staged,end-at[1]);++c.stream_small_staged;
            }
            c.trace_end(trace,c.copy_stream);
            check(cudaEventRecord(s.full[s.head],c.copy_stream));
            s.refillable[s.head]=0;s.head=(s.head+1)%s.frames.size();
            s.queued.push_back(item);++c.stream_prefetched;
        }
    }
    // An unused frame needs no reader; later uploads follow on the same stream.
    void stream_drop(Context& c) {
        auto& s=*c.stream_ring;const int frame=s.queued.front().frame;s.queued.pop_front();
        std::lock_guard lock(s.mutex);
        check(cudaEventRecord(s.consumed[frame],c.copy_stream));s.refillable[frame]=1;++c.stream_dropped;
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
    void select_global() {
        if(!global_cache_mode)return;
        std::array<size_t,2> slots{};
        std::array<uint64_t,2> previous_hits{},selected_hits{},priority_hits{};
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=*contexts[rank];
            const size_t limit=reserve_draft?(c.cache_limit>2048*MiB?c.cache_limit-2048*MiB:0):c.cache_limit;
            slots[rank]=limit/expert_bytes;
            for(auto& [layer,kept]:c.keep)if(layer<45) {
                for(int id:kept)previous_hits[rank]+=c.frequency.at(layer).at(size_t(id));
                kept.clear();
            }
        }
        std::vector<strata_glm_fp4_cache_entry> entries;
        for(const auto& [layer,counts]:contexts.front()->frequency)if(layer<45) {
            const auto priority=priority_frequency.find(layer);
            const auto& scores=global_cache_mode==3&&priority!=priority_frequency.end()?priority->second:counts;
            auto& owners=decode_owners[layer];
            for(int id=0;id<288;++id) {
                owners[size_t(id)]=uint8_t(size_t(id)%contexts.size());
                entries.push_back({layer,id,scores.at(size_t(id)),-1});
            }
        }
        if(!strata_glm_fp4_cache_select(entries.data(),entries.size(),slots.data(),unsigned(contexts.size()),global_cache_mode==1?1:2))
            throw std::logic_error("GLM FP4: invalid global cache admission");
        std::array<size_t,2> selected{},moved{};
        for(const auto& e:entries)if(e.owner>=0) {
            const size_t rank=size_t(e.owner);
            contexts[rank]->keep[e.layer].insert(e.expert);
            decode_owners.at(e.layer).at(size_t(e.expert))=uint8_t(rank);
            ++selected[rank];priority_hits[rank]+=e.frequency;
            selected_hits[rank]+=contexts.front()->frequency.at(e.layer).at(size_t(e.expert));
            if(rank!=size_t(e.expert)%contexts.size())++moved[rank];
        }
        for(size_t rank=0;rank<contexts.size();++rank)
            std::cerr<<"FP4_CACHE_POLICY mode="<<global_cache_mode<<" device="<<contexts[rank]->device
                <<" slots="<<slots[rank]<<" selected="<<selected[rank]<<" moved="<<moved[rank]
                <<" prompt_hits="<<selected_hits[rank]<<" uniform_hits="<<previous_hits[rank]
                <<" priority_hits="<<priority_hits[rank]<<'\n';
    }
    void adapt_cache(int layer,const int* ids,const float* weights,int nt) {
        if(!cache_adapt_rows||!decode_phase||nt>8||layer>=45)return;
        auto& heat=decode_heat[layer];
        for(int t=0;t<nt;++t) {
            const uint8_t mask=strata_glm_route_drop_capped(weights+t*8,0xff,route_drop_fraction,
                                                    unsigned(route_drop_max),route_drop_cap,nullptr);
            for(int j=0;j<8;++j)if(!(mask&(1u<<j)))++heat[size_t(ids[t*8+j])];
        }
        auto& rows=decode_heat_rows[layer];rows+=size_t(nt);
        if(rows<size_t(cache_adapt_rows))return;
        rows=0;
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=*contexts[rank];auto& kept=c.keep.at(layer);
            std::array<uint8_t,288> mask{};for(int id:kept)mask[size_t(id)]=1;
            std::array<int,288> removed{},added{};
            const int n=strata_glm_fp4_cache_adapt(heat.data(),mask.data(),unsigned(rank),unsigned(contexts.size()),
                                                  2,2,removed.data(),added.data());
            if(n<0)throw std::logic_error("GLM FP4: invalid adaptive cache admission");
            if(!n)continue;
            // All preceding expert readers have retired before routed() starts
            // another layer. Reclaim demoted weights now, so global LRU cannot
            // evict a different layer's retained set to admit the replacements.
            if(!c.protected_experts.empty())throw std::logic_error("GLM FP4: adaptive cache has active readers");
            DeviceScope scope(c.device);
            auto release=[&](auto it) {
                const size_t bytes=it->second.bytes;
                if(c.cache_bytes<bytes)throw std::logic_error("GLM FP4: adaptive cache byte accounting");
                c.cache_bytes-=bytes;cache_reclaimed_bytes+=bytes;c.experts.erase(it);
            };
            size_t missing=0;
            for(int i=0;i<n;++i) {
                kept.erase(removed[size_t(i)]);kept.insert(added[size_t(i)]);
                const auto old=c.experts.find({layer,removed[size_t(i)]});
                if(old!=c.experts.end())release(old);
                if(!c.experts.count({layer,added[size_t(i)]}))++missing;
            }
            const size_t needed=missing*expert_bytes;
            while(c.cache_bytes+needed>c.cache_limit) {
                auto victim=c.experts.end();
                for(auto it=c.experts.begin();it!=c.experts.end();++it) {
                    if(it->first.first>=45||c.protected_experts.count(it->first))continue;
                    const auto k=c.keep.find(it->first.first);
                    if(k!=c.keep.end()&&k->second.count(it->first.second))continue;
                    if(victim==c.experts.end()||it->second.use<victim->second.use)victim=it;
                }
                if(victim==c.experts.end())break;
                release(victim);
            }
            cache_promotions+=unsigned(n);
        }
        for(auto& value:heat)value/=2;
    }
    // KEEP_RESELECT_ROWS: the GPU set of each layer comes from the prompt's last prefill batch (select()), which
    // covers fewer decode routes than the decode routes themselves would. After the layer has seen this many decode
    // rows, it is chosen once more: each device swaps kept experts for hotter ones of its own (at most
    // KEEP_RESELECT_SWAPS, each by a margin of two decode routes), ranked by the prefill share plus the decode share.
    // A dropped expert stays cached until room is needed; a new one is uploaded in the background
    // (KEEP_RESELECT_PREFETCH) or when first routed.
    void reselect_keep(int layer,const int* ids,const float* weights,int nt) {
        if(!keep_reselect_rows||!decode_phase||nt>8||layer>=45||reselected.count(layer))return;
        auto& heat=reselect_heat[layer];
        for(int t=0;t<nt;++t) {
            const uint8_t mask=strata_glm_route_drop_capped(weights+t*8,0xff,route_drop_fraction,
                                                    unsigned(route_drop_max),route_drop_cap,nullptr);
            for(int j=0;j<8;++j)if(!(mask&(1u<<j)))++heat[size_t(ids[t*8+j])];
        }
        auto& rows=reselect_rows[layer];rows+=size_t(nt);
        if(rows<size_t(keep_reselect_rows))return;
        reselected.insert(layer);++reselect_layers;
        uint64_t decoded=0;for(auto v:heat)decoded+=v;
        for(size_t rank=0;rank<contexts.size();++rank) {
            auto& c=*contexts[rank];
            const auto f=c.frequency.find(layer);
            const auto k=c.keep.find(layer);
            if(f==c.frequency.end()||k==c.keep.end())continue;
            uint64_t prompted=0;for(auto v:f->second)prompted+=v;
            prompted=std::max<uint64_t>(prompted,1);
            // heat/decoded + prefill/prompted, multiplied by decoded*prompted to stay in integers
            std::array<uint64_t,288> score{};
            for(size_t id=0;id<288;++id)score[id]=heat[id]*prompted+uint64_t(f->second[id])*decoded;
            std::array<uint8_t,288> mask{};for(int id:k->second)mask[size_t(id)]=1;
            std::array<int,288> removed{},added{};
            const int n=strata_glm_fp4_cache_adapt(score.data(),mask.data(),unsigned(rank),unsigned(contexts.size()),
                keep_reselect_swaps?unsigned(keep_reselect_swaps):288u,2*prompted,removed.data(),added.data());
            if(n<0)throw std::logic_error("GLM FP4: invalid keep reselection");
            if(!n)continue;
            // All preceding expert readers have retired before routed() starts another layer (as in adapt_cache).
            if(!c.protected_experts.empty())throw std::logic_error("GLM FP4: keep reselection has active readers");
            DeviceScope scope(c.device);
            for(int i=0;i<n;++i) {
                k->second.erase(removed[size_t(i)]);k->second.insert(added[size_t(i)]);++reselect_swaps;
                if(keep_reselect_prefetch&&!c.experts.count({layer,added[size_t(i)]}))
                    prefetch_queue[rank].push_back({layer,added[size_t(i)]});
            }
            // Room for the new ones from experts no layer keeps (the dropped ones among them, least recently used
            // first), so global LRU cannot take another layer's set.
            size_t missing=0;
            for(int i=0;i<n;++i)if(!c.experts.count({layer,added[size_t(i)]}))++missing;
            while(c.cache_bytes+missing*expert_bytes>c.cache_limit) {
                auto victim=c.experts.end();
                for(auto it=c.experts.begin();it!=c.experts.end();++it) {
                    if(it->first.first>=45||c.protected_experts.count(it->first))continue;
                    const auto kept=c.keep.find(it->first.first);
                    if(kept!=c.keep.end()&&kept->second.count(it->first.second))continue;
                    if(victim==c.experts.end()||it->second.use<victim->second.use)victim=it;
                }
                if(victim==c.experts.end())break;
                const size_t bytes=victim->second.bytes;
                c.cache_bytes-=bytes;reselect_reclaimed_bytes+=bytes;c.experts.erase(victim);
            }
        }
        // The router's bonus for kept experts (ROUTE_AFFINITY) follows the new sets: routing_bonus() writes it again,
        // into the same buffer, where it first wrote it.
        if(route_affinity>0)stale_bonuses.insert(layer);
    }
    // KEEP_RESELECT_PREFETCH: the newly kept experts not yet cached, and the uploads in flight, per device.  Uploads are
    // issued after a decode routed() and finished before the next routed() begins, so the slots and the single upload
    // frame are free again before any upload of routed() itself, and no expert is uploaded twice.
    std::array<std::deque<std::pair<int,int>>,2> prefetch_queue;
    std::array<std::vector<PendingExpert>,2> prefetching;
    void finish_prefetches() {
        for(size_t rank=0;rank<contexts.size()&&rank<prefetching.size();++rank) {
            auto& list=prefetching[rank];
            if(list.empty())continue;
            auto& c=*contexts[rank];DeviceScope scope(c.device);
            for(auto& p:list) {
                p.wait();p.completed=true;
                const auto k=c.keep.find(p.key.first);
                finish_expert(c,p,k!=c.keep.end()&&k->second.count(p.key.second)&&!c.experts.count(p.key));
            }
            list.clear();
        }
    }
    void issue_prefetches() {
        for(size_t rank=0;rank<contexts.size()&&rank<prefetching.size();++rank) {
            auto& c=*contexts[rank];auto& queue=prefetch_queue[rank];auto& list=prefetching[rank];
            const size_t most=std::min<size_t>(size_t(keep_reselect_prefetch),size_t(c.upload_slots));
            if(queue.empty())continue;
            DeviceScope scope(c.device);
            while(!queue.empty()&&list.size()<most) {
                const auto key=queue.front();queue.pop_front();
                const auto k=c.keep.find(key.first);
                if(k==c.keep.end()||!k->second.count(key.second)||c.experts.count(key)){++reselect_prefetch_skipped;continue;}
                list.push_back(prepare_expert(c,key.first,key.second));++reselect_prefetched;
            }
        }
    }
    size_t cache_reserve(const Context& c,size_t reserve)const {
        if(fixed_half_borrow&&c.half_cache_extra) {
            if(reserve<c.half_cache_extra+256*MiB)
                throw std::runtime_error("GLM FP4: fixed FP16 cache must retain at least 256 MiB scratch");
            reserve-=c.half_cache_extra;
        }
        if(!head_tp_borrow||&c!=contexts[1].get())return reserve;
        const size_t extra=c.dense.at("output.weight").data->bytes;
        if(reserve<extra+256*MiB)
            throw std::runtime_error("GLM FP4: head reserve must retain at least 256 MiB scratch");
        return reserve-extra;
    }
    void product(Context& c,const Matrix& w,const uint16_t* x,float* y,int tokens,int accumulators=0,const void* reconstructed=nullptr) {
        if(tokens<=8) {
            const auto trace=c.trace_start("skinny",c.stream);
            float* partials=nullptr;
            const bool split_gate=block_gate_pack&&w.format==k::GlmPackedFormat::BlockFp8Qpn&&
                w.rows==2048&&w.columns==4096;
            if(sm70 && w.format!=k::GlmPackedFormat::Nvfp4 &&
               (w.rows<=(dense_split_enabled?8192:512)||split_gate) && w.columns>=4096)
                // The largest medium split (8192 rows, 8 tokens, 2 parts) also fits this scratch.
                partials=c.buffer("matrix_partial",k::glm_fp4_partial_bytes(512)).f();
            k::glm_fp4_gemm(w.format,w.data->p,w.scales(),w.global,x,y,w.rows,w.columns,tokens,sm70,c.stream,partials,
                           accumulators?accumulators:int(fp8_accumulators));
            c.trace_end(trace,c.stream);
            (sm70?c.mma_calls:c.reference_calls)++;
        }else {
            const bool selective_mma=prefill_mma_tile==1;
            const bool use_prefill_mma=prefill_mma_tile&&(!selective_mma||
                (w.rows==4096&&w.columns==2048&&tokens>=16&&tokens<=64));
            if(!reconstructed&&sm70&&use_prefill_mma&&w.format==k::GlmPackedFormat::Nvfp4&&w.rows%32==0&&w.columns%64==0) {
                const auto trace=c.trace_start("fused_prefill",c.stream);
                k::glm_fp4_prefill_mma(static_cast<const uint8_t*>(w.data->p),static_cast<const uint8_t*>(w.scales()),
                    w.global,x,y,w.rows,w.columns,tokens,selective_mma?32:int(prefill_mma_tile),c.stream);
                c.trace_end(trace,c.stream);++c.prefill_calls;++c.fused_prefill_calls;check(cudaGetLastError());return;
            }
            const void* operand=reconstructed?reconstructed:w.data->p;
            if(!reconstructed&&w.format!=k::GlmPackedFormat::F16) {
                auto& half=c.buffer("dequant_half",size_t(w.rows)*w.columns*2);
                const auto trace=c.trace_start("reconstruct",c.stream);
                k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,static_cast<uint16_t*>(half.p),w.rows,w.columns,c.stream,sm70);
                c.trace_end(trace,c.stream);
                operand=half.p;
            }
            const float one=1,zero=0;
            const auto trace=c.trace_start("gemm",c.stream);
            check(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,w.rows,tokens,w.columns,&one,
                operand,CUDA_R_16F,w.columns,x,CUDA_R_16F,w.columns,&zero,y,CUDA_R_32F,w.rows,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));++c.prefill_calls;
            c.trace_end(trace,c.stream);
        }
        check(cudaGetLastError());
    }
public:
    explicit Fp4NativeExecutor(strata::core::ModelArtifact& a,strata::cpu::exl3::Pool& pool,
                              const std::vector<int>& devices,size_t gpu_budget,int width,bool draft)
        :artifact(a),model(*a.exl3().modelopt),cpu_pool(pool),sm70(fp4_sm70_kernel),reserve_draft(draft) {
        if(devices.empty()||devices.size()>2)throw std::invalid_argument("GLM FP4: expected one or two GPUs");
        if(fixed_half_mask<0||fixed_half_mask>7)
            throw std::invalid_argument("GLM FP4: FIXED_HALF must be a mask from 0 through 7");
        if(fixed_half_max_mib<0||fixed_half_max_mib>8192)
            throw std::invalid_argument("GLM FP4: FIXED_HALF_MAX_MIB must be from 0 through 8192");
        if(fixed_half_mask&&(!sm70||head_tp_enabled||fp8_accumulators!=1||
           (head_fp8_accumulators!=0&&head_fp8_accumulators!=1)))
            throw std::invalid_argument("GLM FP4: FIXED_HALF requires SM70, no head split and single FP8 accumulators");
        if(fixed_half_borrow&&(!fixed_half_mask||!Device::prealloc()||decode_reserve_mib>=0))
            throw std::invalid_argument("GLM FP4: FIXED_HALF_BORROW requires FIXED_HALF, arena and the original decode reserves");
        if(prefill_batch_experts!=1&&prefill_batch_experts!=2&&prefill_batch_experts!=4)
            throw std::invalid_argument("GLM FP4: PREFILL_BATCH_EXPERTS must be 1, 2 or 4");
        if(prefill_queue_depth!=1&&prefill_queue_depth!=2&&prefill_queue_depth!=4)
            throw std::invalid_argument("GLM FP4: PREFILL_QUEUE must be 1, 2 or 4");
        if(prefill_bank_slab&&!prefill_bank_enabled)
            throw std::invalid_argument("GLM FP4: PREFILL_BANK_SLAB requires PREFILL_BANK");
        if(prefill_operand_overlap&&!prefill_operand_pipeline)
            throw std::invalid_argument("GLM FP4: PREFILL_OPERAND_OVERLAP requires PREFILL_OPERAND_PIPELINE");
        if(prefill_operand_pipeline&&(!sm70||!Device::prealloc()||!device_routing_enabled||
           !async_upload_enabled||prefill_bank_enabled||prefill_mma_tile||
           prefill_queue_depth!=1||prefill_batch_experts!=1))
            throw std::invalid_argument("GLM FP4: PREFILL_OPERAND_PIPELINE requires SM70, arena, device routing, async uploads, no bank/fused prefill and queue/batch 1");
        if(prefill_bank_enabled&&(devices.size()!=2||!draft||!sm70||!Device::prealloc()||
           !gpu_reduce_enabled||!device_routing_enabled||!async_upload_enabled||
           prefill_queue_depth!=1||prefill_batch_experts!=1||!strata_env_is_one("STRATA_GLM_FP4_REGISTER_WEIGHTS")))
            throw std::invalid_argument("GLM FP4: PREFILL_BANK requires two SM70 GPUs, MTP, arena, registered RAM, GPU reduction, device routing, async uploads and queue/batch 1");
        if(prefill_stream_frames<8||prefill_stream_frames>136||prefill_stream_frames1<8||prefill_stream_frames1>136)
            throw std::invalid_argument("GLM FP4: PREFILL_STREAM_FRAMES and FRAMES1 must be from 8 through 136");
        if(prefill_stream_enabled&&(devices.size()!=2||!draft||!sm70||!Device::prealloc()||
           !gpu_reduce_enabled||!device_routing_enabled||!async_upload_enabled||prefill_bank_enabled||
           prefill_operand_pipeline||prefill_mma_tile||prefill_queue_depth!=1||prefill_batch_experts!=1||
           !strata_env_is_one("STRATA_GLM_FP4_REGISTER_WEIGHTS")))
            throw std::invalid_argument("GLM FP4: PREFILL_STREAM requires two SM70 GPUs, MTP, arena, registered RAM, GPU reduction, device routing, async uploads, no bank/operand pipeline/fused prefill and queue/batch 1");
        if(prefill_mma_tile!=0&&prefill_mma_tile!=1&&prefill_mma_tile!=32&&prefill_mma_tile!=64)
            throw std::invalid_argument("GLM FP4: PREFILL_MMA must be 0, 1, 32 or 64");
        if(global_cache_mode<0||global_cache_mode>3)
            throw std::invalid_argument("GLM FP4: GLOBAL_CACHE must be 0, 1, 2 or 3");
        if(fp8_accumulators!=1&&fp8_accumulators!=2&&fp8_accumulators!=4)
            throw std::invalid_argument("GLM FP4: FP8_ACCUMULATORS must be 1, 2 or 4");
        if(fp8_accumulators!=1)std::cerr<<"FP4_FP8_ACCUMULATORS count="<<fp8_accumulators<<'\n';
        if(head_fp8_accumulators!=0&&head_fp8_accumulators!=1&&head_fp8_accumulators!=2&&head_fp8_accumulators!=4)
            throw std::invalid_argument("GLM FP4: HEAD_ACCUMULATORS must be 0, 1, 2 or 4");
        if(head_tp_borrow&&(!head_tp_enabled||decode_reserve_mib>=0))
            throw std::invalid_argument("GLM FP4: HEAD_TP_BORROW requires HEAD_TP and the original decode reserves");
        if(cache_total_free&&!Device::prealloc())
            throw std::invalid_argument("GLM FP4: CACHE_TOTAL_FREE requires the GPU arena");
        if(cache_adapt_rows!=0&&cache_adapt_rows!=32&&cache_adapt_rows!=64&&cache_adapt_rows!=128)
            throw std::invalid_argument("GLM FP4: CACHE_ADAPT_ROWS must be 0, 32, 64 or 128");
        if(cache_adapt_rows&&global_cache_mode)
            throw std::invalid_argument("GLM FP4: adaptive cache and global cache admission are separate experiments");
        if(keep_reselect_rows<0||keep_reselect_rows>65536||(keep_reselect_rows&&(cache_adapt_rows||global_cache_mode)))
            throw std::invalid_argument("GLM FP4: KEEP_RESELECT_ROWS must be 0..65536 and excludes CACHE_ADAPT_ROWS "
                                        "and GLOBAL_CACHE");
        if(keep_reselect_swaps<0||keep_reselect_swaps>288||keep_reselect_prefetch<0||keep_reselect_prefetch>8||
           ((keep_reselect_swaps||keep_reselect_prefetch)&&!keep_reselect_rows))
            throw std::invalid_argument("GLM FP4: KEEP_RESELECT_SWAPS must be 0..288 and KEEP_RESELECT_PREFETCH 0..8, "
                                        "both with KEEP_RESELECT_ROWS");
        if(!std::isfinite(route_affinity)||route_affinity<0||route_affinity>0.5||
           (route_affinity>0&&cache_adapt_rows))
            throw std::invalid_argument("GLM FP4: ROUTE_AFFINITY must be0..0.5 and requires fixed decode admission");
        if(route_affinity>0&&route_affinity_secondary&&devices.size()!=2)
            throw std::invalid_argument("GLM FP4: secondary route affinity requires two configured GPUs");
        if(route_affinity>0)std::cerr<<"FP4_ROUTE_AFFINITY lossy=1 bonus="<<route_affinity
            <<" secondary_only="<<route_affinity_secondary<<" selection_only=1 prefill=0 draft=0\n";
        if((global_cache_mode||cache_adapt_rows)&&(devices.empty()||devices.size()>2))
            throw std::invalid_argument("GLM FP4: native expert execution requires one or two devices");
        for(long long cpu:worker_cpus) {
#ifdef __linux__
            if(cpu<-1||cpu>=CPU_SETSIZE||(cpu>=0&&!CPU_ISSET(int(cpu),&allocation_cpus)))
                throw std::invalid_argument("GLM FP4: worker CPU must be -1 or belong to the process allocation");
#else
            if(cpu!=-1)throw std::invalid_argument("GLM FP4: GPU worker CPU pinning requires Linux");
#endif
        }
        if(cpu_logical) {
#ifdef __linux__
            std::vector<int> logical;
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allocation_cpus))logical.push_back(cpu);
            if(logical.size()<2||cpu_pool.size()>logical.size()-1)
                throw std::invalid_argument("GLM FP4: CPU_LOGICAL needs a distinct allocated CPU per worker plus one coordinator CPU");
            const int reserved=logical.front();logical.erase(logical.begin());
            cpu_pool.run([&](size_t rank) {
                cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(logical.at(rank),&mask);
                if(sched_setaffinity(0,sizeof(mask),&mask))
                    throw std::runtime_error("GLM FP4: cannot pin a CPU expert worker to its allocated logical CPU");
            });
            std::cerr<<"FP4_CPU_LOGICAL workers="<<cpu_pool.size()<<" reserved_cpu="<<reserved<<" cpus=";
            for(size_t rank=0;rank<cpu_pool.size();++rank)std::cerr<<(rank?",":"")<<logical[rank];
            std::cerr<<'\n';
#else
            throw std::invalid_argument("GLM FP4: CPU_LOGICAL requires Linux");
#endif
        }
        if(cpu_team_size!=0&&cpu_team_size!=2&&cpu_team_size!=4&&cpu_team_size!=8)
            throw std::invalid_argument("GLM FP4: CPU_TEAM_SIZE must be 0, 2, 4 or 8");
        if(expert_fusion)std::cerr<<"FP4_EXPERT_FUSION enabled=1 sm70="<<sm70<<" narrow_only=1\n";
        if(early_readback)std::cerr<<"FP4_EARLY_READBACK enabled=1 narrow_only=1\n";
        if(route_id_batch)std::cerr<<"FP4_ROUTE_ID_BATCH enabled=1 maximum_ids=64 host_bytes_per_device=256\n";
        if(cpu_row_tile!=1&&cpu_row_tile!=2&&cpu_row_tile!=4)
            throw std::invalid_argument("GLM FP4: CPU_ROW_TILE must be 1, 2 or 4");
        if(cpu_q8_bias&&!cpu_q8_enabled)throw std::invalid_argument("GLM FP4: CPU_Q8_BIAS requires CPU_Q8=1");
        if(cpu_scale_local&&!cpu_q8_enabled)throw std::invalid_argument("GLM FP4: CPU_SCALE_LOCAL requires CPU_Q8=1");
        if(cpu_scale_local_m1&&!cpu_q8_enabled)throw std::invalid_argument("GLM FP4: CPU_SCALE_LOCAL_M1 requires CPU_Q8=1");
        if(cpu_q8_policy&&(!cpu_q8_enabled||cpu_q8_bias||cpu_q8_prepare||cpu_scale_local||cpu_scale_local_m1||cpu_row_tile!=1))
            throw std::invalid_argument("GLM FP4: CPU_Q8_POLICY requires CPU_Q8=1, ROW_TILE=1 and global Q8_BIAS/Q8_PREPARE/SCALE_LOCAL/SCALE_LOCAL_M1 off");
        if(cpu_q8_bias)std::cerr<<"FP4_CPU_Q8_BIAS enabled=1\n";
        if(cpu_scale_local)std::cerr<<"FP4_CPU_SCALE_LOCAL enabled=1 max_row_bytes=256\n";
        if(cpu_scale_local_m1)std::cerr<<"FP4_CPU_SCALE_LOCAL_M1 enabled=1 max_row_bytes=256\n";
        if(cpu_q8_policy)std::cerr<<"FP4_CPU_Q8_POLICY enabled=1 unchanged_q8_arithmetic=1\n";
        if(cpu_row_tile!=1||cpu_q8_prepare)std::cerr<<"FP4_CPU_REUSE row_tile="<<cpu_row_tile<<" prepared="<<cpu_q8_prepare<<'\n';
        if(decode_device_routing_enabled)std::cerr<<"FP4_DECODE_ROUTING enabled=1 deferred_expert_sync=1\n";
        if(!std::isfinite(route_drop_cap)||route_drop_cap<0.5||route_drop_cap>0.75||
           !std::isfinite(route_drop_fraction)||route_drop_fraction<0||route_drop_fraction>route_drop_cap||
           route_drop_max<0||route_drop_max>7)
            throw std::invalid_argument("GLM FP4: ROUTE_DROP must be in [0,ROUTE_DROP_CAP], CAP in [0.5,0.75], MAX in [0,7]");
        if(decode_reserve_mib!=-1 && (decode_reserve_mib<256 || decode_reserve_mib>8192))
            throw std::invalid_argument("GLM FP4: DECODE_RESERVE_MIB must be in [256,8192], or -1 for the original per-device reserves");
        if(cache_shrink0_mib<0||cache_shrink0_mib>16384)throw std::invalid_argument("GLM FP4: CACHE_SHRINK0_MIB must be in [0,16384]");
        if(route_drop_fraction>0)std::cerr<<"FP4_ROUTE_DROP lossy=1 cpu_only=1 fraction="<<route_drop_fraction
            <<" cap="<<route_drop_cap<<" max_routes="<<route_drop_max<<" renormalize="<<route_renorm_enabled<<" draft=0\n";
        if(cpu_q8_enabled)std::cerr<<"FP4_CPU_Q8 lossy=1 mode="<<(cpu::glm_fp4_q8_vnni_available()?"vnni":"scalar")<<'\n';
        if(strata_env_is_one("STRATA_GLM_FP4_REGISTER_WEIGHTS"))
            registered_ram=std::make_unique<RegisteredRam>(artifact.exl3().tensors,devices);
        for(int d:devices) {
            DeviceScope scope(d);cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,d));
            if(p.major<7)throw std::invalid_argument("GLM FP4: tensor core path requires SM70 or newer");
            const int worker_cpu=contexts.size()<worker_cpus.size()?int(worker_cpus[contexts.size()]):-1;
            auto c=std::make_unique<Context>(d,worker_cpu);
            if(worker_cpu>=0)std::cerr<<"FP4_WORKER_CPU rank="<<contexts.size()<<" cpu="<<worker_cpu<<'\n';
            c->upload_slots=int(std::max<long long>(2*prefill_batch_experts,prefill_queue_depth));c->tracing=profile;
            if(Device::limit()==SIZE_MAX)Device::limit()=std::min<size_t>(gpu_budget,p.totalGlobalMem)-Device::budget_reserve();
            contexts.push_back(std::move(c));
        }
        if((gpu_reduce_enabled||head_tp_enabled)&&contexts.size()==2) {
            int forward=0,backward=0;
            check(cudaDeviceCanAccessPeer(&forward,devices[0],devices[1]));
            check(cudaDeviceCanAccessPeer(&backward,devices[1],devices[0]));
            peer_enabled=forward&&backward;
            if(peer_enabled)for(size_t i=0;i<2;++i) {
                DeviceScope scope(devices[i]);
                const auto status=cudaDeviceEnablePeerAccess(devices[1-i],0);
                if(status==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(status);
            }
        }
        if(head_tp_enabled&&(!sm70||contexts.size()!=2||!peer_enabled))
            throw std::invalid_argument("GLM FP4: HEAD_TP requires SM70 and two peer-accessible GPUs");
        if(prefill_bank_enabled&&!peer_enabled)
            throw std::invalid_argument("GLM FP4: PREFILL_BANK requires peer access");
        if(prefill_stream_enabled&&!peer_enabled)
            throw std::invalid_argument("GLM FP4: PREFILL_STREAM requires peer access");
        if(prefill_stream_enabled)std::cerr<<"FP4_PREFILL_STREAM enabled=1 frames="<<prefill_stream_frames
            <<" frames1="<<prefill_stream_frames1<<" hold="<<prefill_stream_hold<<" early="<<prefill_stream_early
            <<" min_tokens=4096\n";
        if(gpu_reduce_enabled)std::cerr<<"FP4_GPU_REDUCE peer="<<peer_enabled<<'\n';
        // Compressed fixed matrices are resident, not expanded into FP32 on
        // the host. Their exact published scale grids accompany their codes.
        Context& c=*contexts.front();
        {DeviceScope scope(c.device);
            for(const auto& [name,t]:artifact.tensors()) {
                if(name=="token_embd.weight" || name.starts_with("blk.45.") || !t.native || t.tensor->shape.size()!=2)continue;
                auto matrix=upload(c,model.matrix(t.native->source));
                const bool head=(fixed_half_mask&1)&&name=="output.weight";
                const bool attention=(fixed_half_mask&2)&&name!="output.weight"&&
                    matrix.format==k::GlmPackedFormat::Mxfp8Qpn&&matrix.rows<=8192&&matrix.columns>=4096;
                const bool shared=(fixed_half_mask&4)&&
                    (matrix.format==k::GlmPackedFormat::BlockFp8||matrix.format==k::GlmPackedFormat::BlockFp8Qpn)&&
                    matrix.rows==2048&&matrix.columns==4096;
                if(head||attention||shared) {
                    if(matrix.format!=k::GlmPackedFormat::Mxfp8Qpn&&
                       matrix.format!=k::GlmPackedFormat::BlockFp8&&matrix.format!=k::GlmPackedFormat::BlockFp8Qpn)
                        throw std::invalid_argument("GLM FP4: selected fixed FP16 cache requires an FP8 source");
                    const size_t bytes=size_t(matrix.rows)*matrix.columns*2;
                    const size_t original=matrix.data->bytes;
                    if(bytes<original)throw std::logic_error("GLM FP4: fixed FP16 cache byte accounting");
                    // Selection follows the existing fixed-matrix traversal.
                    // Bound extra resident bytes before allocating the cache;
                    // skipped matrices retain their original format and data.
                    if(fixed_half_max_mib&&c.half_cache_extra+bytes-original>size_t(fixed_half_max_mib)*MiB) {
                        ++c.half_cache_skipped;c.dense.emplace(name,std::move(matrix));continue;
                    }
                    c.room(bytes);auto cached=std::make_unique<Device>(bytes);
                    k::glm_fp8_cache_f16_qpn(matrix.format,matrix.data->p,matrix.scales(),
                        static_cast<uint16_t*>(cached->p),matrix.rows,matrix.columns,c.stream);
                    check(cudaGetLastError());check(cudaStreamSynchronize(c.stream));
                    matrix.data=std::move(cached);matrix.scale_offset=0;matrix.format=k::GlmPackedFormat::F16Qpn;
                    c.half_cache_extra+=bytes-original;++c.half_cache_matrices;
                }
                c.dense.emplace(name,std::move(matrix));
            }
        }
        if(fixed_half_mask)std::cerr<<"FP4_FIXED_HALF mask="<<fixed_half_mask
            <<" matrices="<<c.half_cache_matrices<<" extra_bytes="<<c.half_cache_extra
            <<" borrow="<<fixed_half_borrow<<" max_MiB="<<fixed_half_max_mib
            <<" skipped="<<c.half_cache_skipped<<'\n';
        if(head_tp_enabled) {
            const auto& full=c.dense.at("output.weight");
            if(full.format!=k::GlmPackedFormat::Mxfp8Qpn||full.rows%64)
                throw std::invalid_argument("GLM FP4: HEAD_TP requires an MXFP8 head with rows divisible by 64");
            auto split=std::make_unique<HeadPartition>();split->primary=c.device;split->secondary=contexts[1]->device;
            split->first.format=full.format;split->first.rows=full.rows/2;
            split->first.columns=full.columns;split->first.global=full.global;split->first.scale_offset=full.scale_offset;
            // Retain the original primary allocation for wide-prefill fallback.
            // This view borrows its codes and original separate scale plane.
            {DeviceScope scope(c.device);
                split->first.data=std::make_unique<Device>(full.data->p,full.data->bytes);
                check(cudaEventCreateWithFlags(&split->ready,cudaEventDisableTiming));}
            {auto& other=*contexts[1];DeviceScope scope(other.device);
                check(cudaEventCreateWithFlags(&split->done,cudaEventDisableTiming));
                other.dense.emplace("output.weight",upload_head_tail(other,
                    model.matrix(artifact.at("output.weight").native->source),full.rows/2,full.rows/2));
                other.scratch.erase("packed_source");}
            head_partition=std::move(split);
            std::cerr<<"FP4_HEAD_TP rows_per_gpu="<<full.rows/2
                <<" extra_weight_bytes="<<contexts[1]->dense.at("output.weight").data->bytes<<'\n';
        }
        for(auto& ctx:contexts) {
            DeviceScope scope(ctx->device);size_t free,total;check(Device::mem_info(&free,&total));
            // Prefill scratch scales with tokens. The coordinator additionally
            // holds attention state and its GpuPrefill arena outside this class.
            const size_t reserve=cache_reserve(*ctx,std::max<size_t>(
                (ctx.get()==contexts.front().get()?4096:1024)*MiB,size_t(width)*4096*24));
            const size_t available=std::min(free,Device::limit()>Device::live()?Device::limit()-Device::live():0);
            ctx->cache_limit=shrink0(*ctx,available>reserve?available-reserve:0);
            ctx->prefill_cache_limit=ctx->cache_limit;
            if(ctx->cache_limit<64*MiB)throw std::runtime_error("GLM FP4: less than 64 MiB available for expert cache");
            std::cerr<<"FP4_DEVICE device="<<ctx->device<<" cache_limit_MiB="<<ctx->cache_limit/MiB
                     <<" fixed_matrices="<<ctx->dense.size()<<" kernel="<<(sm70?"sm70":"reference")<<'\n';
        }
        sums.resize(contexts.size());
    }
    ~Fp4NativeExecutor(){if(held_fill.valid())held_fill.wait();for(auto& f:restore_work)if(f.valid())f.wait();report(false);}
    bool set_decode_device_routing(bool enabled) {
        const bool previous=decode_device_routing_enabled;decode_device_routing_enabled=enabled;return previous;
    }
    std::vector<int> devices()const {
        std::vector<int> result;for(const auto& c:contexts)result.push_back(c->device);return result;
    }
    void begin_request() {
        if(prefill_bank_enabled)release_banks();
        if(prefill_stream_enabled)release_prefill_stream();
        decode_phase=false;
        route_affinity_bonuses.clear();
        decode_owners.clear();
        priority_frequency.clear();
        decode_heat.clear();decode_heat_rows.clear();
        reselect_heat.clear();reselect_rows.clear();reselected.clear();stale_bonuses.clear();
        finish_prefetches();
        for(auto& q:prefetch_queue){reselect_prefetch_dropped+=q.size();q.clear();}
        for(auto& c:contexts){c->phase=1;c->frequency.clear();c->keep.clear();}
    }
    void begin_prefill(size_t coordinator_bytes) {
        if(prefill_bank_enabled)release_banks();
        if(prefill_stream_enabled)release_prefill_stream();
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
        route_affinity_bonuses.clear();
        if(prefill_bank_enabled)release_banks();
        if(prefill_stream_enabled)release_prefill_stream();
        decode_phase=true;stream_followups=0;
        for(auto& c:contexts) {
            c->phase=2;
            DeviceScope scope(c->device);check(cudaStreamSynchronize(c->stream));c->scratch.clear();c->release_upload();
            size_t free,total;check(Device::mem_info(&free,&total));
            const size_t reserve=cache_reserve(*c,size_t(decode_reserve_mib<0?
                (c.get()==contexts.front().get()?3072:1024):decode_reserve_mib)*MiB);
            if(head_tp_borrow&&c.get()==contexts[1].get())
                std::cerr<<"FP4_HEAD_TP_RESERVE device="<<c->device
                         <<" borrowed_bytes="<<c->dense.at("output.weight").data->bytes
                         <<" remaining_bytes="<<reserve<<'\n';
            if(decode_reserve_mib>=0||cache_total_free)if(auto* pool=Device::pool()) {
                // An expert consists of many small allocations, so it can use
                // separate holes left by prefill. The largest hole is needed
                // for an individual allocation, not the complete cache budget.
                std::lock_guard lock(pool->mutex);
                free=0;size_t largest=0;
                constexpr size_t matrix_bytes=expert_bytes/3;
                for(const auto& [offset,bytes]:pool->free_blocks) {
                    largest=std::max(largest,bytes);
                    // A projection must fit one hole. Ignore tails smaller
                    // than a packed projection rather than counting every
                    // free byte as expert capacity.
                    free+=cache_total_free?bytes/matrix_bytes*matrix_bytes:bytes;
                }
                if(cache_total_free)std::cerr<<"FP4_CACHE_FREE device="<<c->device
                    <<" usable_bytes="<<free<<" largest_bytes="<<largest
                    <<" matrix_bytes="<<matrix_bytes<<'\n';
            }
            const size_t available=std::min(free,Device::limit()>Device::live()?Device::limit()-Device::live():0);
            c->cache_limit=c->cache_bytes+(available>reserve?available-reserve:0);
            if(cache_shrink0_mib) {
                c->cache_limit=shrink0(*c,c->cache_limit);
                while(c->cache_bytes>c->cache_limit)c->evict();
                std::cerr<<"FP4_CACHE_SHRINK device="<<c->device<<" MiB="<<cache_shrink0_mib<<" limit_MiB="<<c->cache_limit/MiB<<'\n';
            }
            if(decode_reserve_mib>=0||cache_total_free)std::cerr<<"FP4_DECODE_CACHE device="<<c->device
                <<" reserve_MiB="<<reserve/MiB<<" limit_MiB="<<c->cache_limit/MiB<<'\n';
            for(const auto& [layer,_]:c->frequency)select(*c,layer);
        }
        select_global();
        host_x.reset();host_routes.reset();host_weights.reset();host_rows=0;host_x_rows=0;
    }
    const float* routing_bonus(int layer,int tokens) {
        if(route_affinity==0||!decode_phase||tokens>8||layer>=45)return nullptr;
        auto& saved=route_affinity_bonuses[layer];
        const bool stale=stale_bonuses.erase(layer)>0;      // KEEP_RESELECT_ROWS changed the layer's set
        if(!saved||stale) {
            // Fixed admission for this request, including optional cross-device
            // ownership. The router uses the bonus only to select IDs; retained
            // weights still come from their original unbiased sigmoid scores.
            std::array<float,288> bonus{};
            for(int id=0;id<288;++id) {
                const size_t rank=owner(layer,id,true);
                if((!route_affinity_secondary||rank==1)&&contexts[rank]->keep.at(layer).count(id))
                    bonus[id]=float(route_affinity);
            }
            DeviceScope scope(contexts.front()->device);
            if(!saved)saved=std::make_unique<Device>(sizeof(bonus));
            saved->put(bonus.data(),sizeof(bonus));
        }
        return saved->f();
    }
    // The same rounded rows routed() would send, enqueued before the primary's
    // router and shared expert so the peer copy overlaps them.
    void prefetch_routed_input(int layer,const float* x,int nt,cudaStream_t caller) {
        if(!prefill_stream_early||decode_phase||nt<=8||layer>=45||contexts.size()!=2||!peer_enabled||
           !gpu_reduce_enabled||!contexts.front()->stream_ring||stream_planned_layer!=layer)return;
        auto& c=*contexts.front();auto& other=*contexts[1];uint16_t* half=nullptr;
        {
            DeviceScope scope(c.device);
            half=static_cast<uint16_t*>(c.buffer("route_input_half",size_t(nt)*4096*2).p);
            k::glm_f16(x,half,int64_t(nt)*4096,caller);
            check(cudaEventRecord(c.stream_ring->input_ready,caller));
        }
        DeviceScope scope(other.device);
        auto* rows=other.buffer("prefill_route_input_half",size_t(nt)*4096*2).p;
        check(cudaStreamWaitEvent(other.stream,c.stream_ring->input_ready,0));
        check(cudaMemcpyPeerAsync(rows,other.device,half,c.device,size_t(nt)*4096*2,other.stream));
        early_input_layer=layer;
    }
    bool kda_qkv_batch_enabled()const {return kda_qkv_batch;}
    bool can_batch_kda_qkv(const std::array<std::string,3>& names,int rows,int columns,int nt)const {
        if(!kda_qkv_batch||!sm70||dense_split_enabled||fp8_accumulators!=1||nt<1||nt>8||
           rows<=512||rows%32||columns%64)return false;
        const auto& c=*contexts.front();
        for(const auto& name:names) {
            const auto it=c.dense.find(name);
            if(it==c.dense.end()||it->second.format!=k::GlmPackedFormat::Mxfp8Qpn||
               it->second.rows!=rows||it->second.columns!=columns)return false;
        }
        return true;
    }
    void batch_kda_qkv(const std::array<std::string,3>& names,const float* x,
                       const std::array<float*,3>& y,int nt,cudaStream_t caller) {
        Context& c=*contexts.front();DeviceScope scope(c.device);
        const auto& a=c.dense.at(names[0]);
        auto& hx=c.buffer("dense_x_half",size_t(8)*16384*2);
        const auto trace=profile?c.trace_start("kda_qkv_batch",caller,a.rows,a.columns,nt,int(a.format)):SIZE_MAX;
        k::glm_f16(x,static_cast<uint16_t*>(hx.p),size_t(nt)*a.columns,caller);
        std::array<k::GlmFp8Projection,3> p;
        for(int i=0;i<3;++i) {
            const auto& w=c.dense.at(names[i]);p[i]={w.data->p,w.scales(),y[i]};
        }
        k::glm_mxfp8_gemm_three(p[0],p[1],p[2],static_cast<uint16_t*>(hx.p),a.rows,a.columns,nt,caller);
        c.trace_end(trace,caller);c.mma_calls+=3;check(cudaGetLastError());
    }
    void dense(const std::string& name,const float* x,float* y,int nt,cudaStream_t caller) {
        if(prefill_bank_enabled&&!decode_phase&&nt>=4096&&name.starts_with("blk.0.")&&!contexts.front()->bank)
            ensure_prefill_banks();
        // A prompt longer than one batch: the previous batch's restore may
        // still be pending; finish it so this batch can lend the entries again.
        if(prefill_stream_enabled&&!decode_phase&&nt>=4096&&name.starts_with("blk.0."))join_restore();
        if(prefill_stream_enabled&&!decode_phase&&nt>=4096&&name.starts_with("blk.0.")&&!contexts.front()->stream_ring) {
            // Start the first routed layer's uploads behind the dense layers.
            if(ensure_prefill_stream()) {
                plan_stream_layer(3);
                for(auto& ctx:contexts){DeviceScope scope(ctx->device);stream_fill(*ctx);}
            }
        }
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
        // std::map keys stay alive until the context is destroyed, including
        // events flushed after the final head projection of a request.
        const size_t dense_trace=profile?c.trace_start(it->first.c_str(),caller,
            it->second.rows,it->second.columns,nt,int(it->second.format)):SIZE_MAX;
        k::glm_f16(x,static_cast<uint16_t*>(hx.p),size_t(nt)*it->second.columns,c.stream);
        if(head_partition&&name=="output.weight"&&nt<=8) {
            join_restore();
            cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(caller,&capture));
            if(capture!=cudaStreamCaptureStatusNone)
                throw std::runtime_error("GLM FP4: HEAD_TP does not support CUDA graph capture");
            auto& split=*head_partition;auto& other=*contexts[1];
            const int rows=split.first.rows,columns=split.first.columns;
            auto& left=c.buffer("tp_head_left",size_t(8)*rows*4);
            check(cudaEventRecord(split.ready,caller));
            float* right=nullptr;
            auto work=other.submit([&] {
                DeviceScope scope(other.device);
                try {
                    auto& input=other.buffer("tp_head_input",size_t(8)*columns*2);
                    right=other.buffer("tp_head_result",size_t(8)*rows*4).f();
                    check(cudaStreamWaitEvent(other.stream,split.ready,0));
                    check(cudaMemcpyPeerAsync(input.p,other.device,hx.p,c.device,size_t(nt)*columns*2,other.stream));
                    product(other,other.dense.at(name),static_cast<uint16_t*>(input.p),right,nt,int(head_fp8_accumulators));
                    check(cudaEventRecord(split.done,other.stream));
                }catch(...){cudaStreamSynchronize(other.stream);throw;}
            });
            try {
                product(c,split.first,static_cast<uint16_t*>(hx.p),left.f(),nt,int(head_fp8_accumulators));
                work.get();
                check(cudaMemcpy2DAsync(y,size_t(rows)*8,left.p,size_t(rows)*4,
                    size_t(rows)*4,nt,cudaMemcpyDeviceToDevice,caller));
                check(cudaStreamWaitEvent(caller,split.done,0));
                for(int t=0;t<nt;++t)check(cudaMemcpyPeerAsync(y+size_t(t)*rows*2+rows,c.device,
                    right+size_t(t)*rows,other.device,size_t(rows)*4,caller));
            }catch(...) {
                if(work.valid())work.wait();
                {DeviceScope scope(other.device);cudaStreamSynchronize(other.stream);}
                cudaStreamSynchronize(caller);throw;
            }
            c.trace_end(dense_trace,caller);
            ++split.calls;check(cudaGetLastError());return;
        }
        if(nt>8)check(cublasSetStream(c.blas,caller));
        product(c,it->second,static_cast<uint16_t*>(hx.p),y,nt,
                name=="output.weight"?int(head_fp8_accumulators):0);
        c.trace_end(dense_trace,caller);
        if(nt>8)check(cublasSetStream(c.blas,restore.previous));
    }
    bool prepare_routed_inputs(const float* x,const int* ids,const float* weights,int nt,
                               cudaStream_t caller,cudaEvent_t ready) {
        if(!early_readback||nt<1||nt>8)return false;
        ensure_host_rows(nt,caller,true);
        // The router and normalized input precede the independent shared FFN.
        // This event fences only its readback, so CPU expert work can start
        // while the shared FFN remains queued on the coordinator stream.
        try {
            check(cudaMemcpyAsync(host_x->p,x,size_t(nt)*4096*4,cudaMemcpyDeviceToHost,caller));
            check(cudaMemcpyAsync(host_routes->p,ids,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
            check(cudaMemcpyAsync(host_weights->p,weights,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
            check(cudaEventRecord(ready,caller));
        }catch(...){cudaStreamSynchronize(caller);throw;}
        ++early_readback_calls;return true;
    }
    void routed(int layer,const float* x,const int* ids,const float* weights,float* out,int nt,
                float limit,cudaStream_t caller,cudaEvent_t inputs_ready=nullptr) {
        if(!keep_reselect_prefetch)return routed_body(layer,x,ids,weights,out,nt,limit,caller,inputs_ready);
        finish_prefetches();                               // before any upload of this call
        routed_body(layer,x,ids,weights,out,nt,limit,caller,inputs_ready);
        if(decode_phase&&nt<=8&&layer<45)issue_prefetches();
    }
    void routed_body(int layer,const float* x,const int* ids,const float* weights,float* out,int nt,
                     float limit,cudaStream_t caller,cudaEvent_t inputs_ready) {
        join_held_fill();join_restore();
        const auto started=tick();
        struct alignas(64) Times {double upload=0,gather=0,product=0,reduce=0,fused=0;};
        std::array<Times,2> times{};Times cpu_times;
        const bool skip_activation=gpu_reduce_enabled&&nt>8&&device_routing_enabled&&
            (contexts.size()==1||peer_enabled);
        if(inputs_ready) {
            if(!early_readback||nt<1||nt>8||size_t(nt)>host_rows||size_t(nt)>host_x_rows)
                throw std::logic_error("GLM FP4: invalid early route readback");
            check(cudaEventSynchronize(inputs_ready));
        }else {
            ensure_host_rows(nt,caller,!(lean_host&&skip_activation));
            if(!skip_activation)check(cudaMemcpyAsync(host_x->p,x,size_t(nt)*4096*4,cudaMemcpyDeviceToHost,caller));
            check(cudaMemcpyAsync(host_routes->p,ids,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
            check(cudaMemcpyAsync(host_weights->p,weights,size_t(nt)*8*4,cudaMemcpyDeviceToHost,caller));
            check(cudaStreamSynchronize(caller));
        }
        const auto* selected=static_cast<const int*>(host_routes->p);
        auto* routing=static_cast<float*>(host_weights->p);
        const float* activation=size_t(nt)<=host_x_rows?static_cast<const float*>(host_x->p):nullptr;
        std::map<int,std::vector<int>> grouped;
        for(int i=0;i<nt*8;++i) {
            if(selected[i]<0||selected[i]>=288)throw std::runtime_error("GLM FP4: invalid expert route");
            grouped[selected[i]].push_back(i);
        }
        const bool decode=nt<=8;
        bool device_routing=decode?decode_device_routing_enabled:device_routing_enabled;
        if(device_routing)for(const auto& [id,routes]:grouped)
            for(size_t j=1;j<routes.size();++j)if(routes[j]/8==routes[j-1]/8)device_routing=false;
        if(skip_activation&&!device_routing) {
            ensure_host_rows(nt,caller,true);activation=static_cast<const float*>(host_x->p);
            check(cudaMemcpyAsync(host_x->p,x,size_t(nt)*4096*4,cudaMemcpyDeviceToHost,caller));
            check(cudaStreamSynchronize(caller));
        }
        const bool gpu_reduce=gpu_reduce_enabled&&device_routing;
        if(decode){if(device_routing)++decode_device_calls;else ++decode_host_calls;}
        if(global_cache_mode==3&&!decode&&layer<45) {
            // Cache contributions that would survive the existing drop budget
            // if every route were on CPU. Otherwise a frequent, weak route can
            // consume VRAM even though it would cost no CPU dot product.
            auto& counts=priority_frequency[layer];counts.assign(288,0);
            for(int t=0;t<nt;++t) {
                const uint8_t mask=strata_glm_route_drop_capped(routing+t*8,0xff,route_drop_fraction,
                                                        unsigned(route_drop_max),route_drop_cap,nullptr);
                for(int j=0;j<8;++j)if(!(mask&(1u<<j)))++counts[size_t(selected[t*8+j])];
            }
        }
        for(auto& c:contexts)if(!decode || !c->frequency.count(layer)) {
            auto& counts=c->frequency[layer];counts.assign(288,0);
            for(const auto& [id,routes]:grouped)counts[id]=routes.size();
            select(*c,layer);
        }
        if(prefill_bank_enabled&&!decode&&layer<45)for(auto& c:contexts)if(c->bank) {
            if(c->bank->layer!=layer)throw std::logic_error("GLM FP4: wrong prefill bank layer");
            join_bank(*c);
        }
        adapt_cache(layer,selected,routing,nt);
        reselect_keep(layer,selected,routing,nt);
        if(decode_phase && decode && layer<45 && route_drop_fraction>0 && route_drop_max>0) {
            std::array<uint8_t,8> masks{};
            for(int t=0;t<nt;++t) {
                uint8_t eligible=0;
                for(int j=0;j<8;++j) {
                    const int id=selected[t*8+j];
                    if(!contexts[owner(layer,id,decode)]->keep.at(layer).count(id)) {
                        eligible|=uint8_t(1u<<j);++route_drop_candidates;
                    }
                }
                double share=0;
                masks[t]=strata_glm_route_drop_capped(routing+t*8,eligible,route_drop_fraction,
                                              unsigned(route_drop_max),route_drop_cap,&share);
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
            auto& c=*contexts[owner(layer,id,decode)];
            if(!c.keep.at(layer).count(id))cpu_jobs.push_back({id,&routes,{},{},{},{},{},{},{},{}});
        }
        std::vector<float> cpu_sum(cpu_jobs.empty()?0:size_t(nt)*4096,0);
        const double input_wait_ms=elapsed(started);
        auto run_cpu=[&] {
            if(cpu_jobs.empty())return;
            cpu_experts+=cpu_jobs.size();
            if(cpu_q8_policy||cpu_scale_local_m1)for(const auto& job:cpu_jobs)++cpu_width_groups.at(job.routes->size());
            auto begin=tick();
            cpu::GlmFp4Q8 shared_input;
            if(cpu_input_reuse&&cpu_q8_enabled) {
                std::vector<float> rounded(activation,activation+size_t(nt)*4096);
                cpu::glm_fp4_round_f16(rounded.data(),rounded.size());
                shared_input=cpu::glm_fp4_quantize_q8(rounded.data(),4096,nt);
                if(cpu_q8_bias||cpu_q8_policy)cpu::glm_fp4_prepare_q8_bias(shared_input);
                else if(cpu_q8_prepare)cpu::glm_fp4_prepare_q8(shared_input);
            }
            // Packed codes stay mapped. A bounded LRU retains logical
            // scale planes so repeat CPU misses do not re-expand CSF32.
            cpu_pool.run([&](size_t rank) {
                for(size_t i=rank;i<cpu_jobs.size();i+=cpu_pool.size()) {
                    auto& job=cpu_jobs[i];const size_t count=job.routes->size();
                    job.weights=cpu_expert(layer,job.id);
                    const bool reuse=shared_input.tokens&&job.weights->sparse[0]&&job.weights->sparse[1];
                    if(!reuse)job.input.resize(count*4096);
                    job.gate.resize(count*2048);job.up.resize(count*2048);
                    job.hidden.resize(count*2048);job.result.resize(count*4096);
                    if(reuse) {
                        std::array<int,8> rows{};
                        if(count>rows.size())throw std::runtime_error("GLM FP4: Q8 route width exceeds eight");
                        for(size_t t=0;t<count;++t)rows[t]=(*job.routes)[t]/8;
                        job.qinput=cpu::glm_fp4_gather_q8(shared_input,rows.data(),int(count));
                    }else {
                    for(size_t t=0;t<count;++t) {
                        const auto* source=activation+size_t((*job.routes)[t]/8)*4096;
                        auto* dest=job.input.data()+t*4096;
                        if(cpu_q8_enabled||cpu_fused_enabled) {
                            std::copy_n(source,4096,dest);cpu::glm_fp4_round_f16(dest,4096);
                        }else for(int h=0;h<4096;++h)
                            dest[h]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(source[h]));
                    }
                    if(cpu_q8_enabled) {
                        job.qinput=cpu::glm_fp4_quantize_q8(job.input.data(),4096,count);
                        if(cpu_q8_bias||cpu_q8_policy)cpu::glm_fp4_prepare_q8_bias(job.qinput);
                        else if(cpu_q8_prepare)cpu::glm_fp4_prepare_q8(job.qinput);
                    }
                    }
                }
            });
            if(profile)cpu_times.upload+=elapsed(begin);
            begin=tick();
            auto gate_up_range=[&](size_t rank,size_t workers,size_t begin_job,size_t end_job) {
                const int first=2048*rank/workers,last=2048*(rank+1)/workers;
                for(size_t j=begin_job;j<end_job;++j) {
                    auto& job=cpu_jobs[j];
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
                    if(cpu_q8_bias||cpu_q8_policy)cpu::glm_fp4_prepare_q8_bias(job.qhidden);
                    else if(cpu_q8_prepare)cpu::glm_fp4_prepare_q8(job.qhidden);
                }
            };
            auto down_range=[&](size_t rank,size_t workers,size_t begin_job,size_t end_job) {
                const int first=4096*rank/workers,last=4096*(rank+1)/workers;
                for(size_t j=begin_job;j<end_job;++j) {
                    auto& job=cpu_jobs[j];cpu_product(*job.weights,2,
                    job.hidden.data(),job.result.data(),2048,4096,job.routes->size(),first,last,cpu_q8_enabled?&job.qhidden:nullptr);
                }
            };
            auto gate_up=[&](size_t rank){gate_up_range(rank,cpu_pool.size(),0,cpu_jobs.size());};
            auto down=[&](size_t rank){down_range(rank,cpu_pool.size(),0,cpu_jobs.size());};
            if(cpu_team_size) {
                const size_t size=size_t(cpu_team_size),teams=(cpu_pool.size()+size-1)/size;
                struct alignas(64) TeamCounter {std::atomic<size_t> value{0};};
                auto arrived=std::make_unique<TeamCounter[]>(teams);
                std::atomic<bool> cancelled{false};
                auto team_work=[&](size_t rank) {
                    // Spread small teams across the worker pool so a short expert list
                    // does not leave every worker on the second socket idle.
                    const size_t team=rank%teams,member=rank/teams;
                    const size_t workers=(cpu_pool.size()-1-team)/teams+1;
                    size_t phase=0;
                    auto barrier=[&] {
                        const size_t target=workers*(++phase);
                        arrived[team].value.fetch_add(1,std::memory_order_acq_rel);
                        while(arrived[team].value.load(std::memory_order_acquire)<target) {
                            if(cancelled.load(std::memory_order_acquire))return false;
#if defined(__x86_64__)
                            _mm_pause();
#else
                            std::this_thread::yield();
#endif
                        }
                        return !cancelled.load(std::memory_order_acquire);
                    };
                    try {
                        for(size_t j=team;j<cpu_jobs.size();j+=teams) {
                            if(cancelled.load(std::memory_order_acquire))return;
                            gate_up_range(member,workers,j,j+1);if(!barrier())return;
                            if(cpu_q8_enabled) {
                                if(member==0) {
                                    auto& job=cpu_jobs[j];
                                    job.qhidden=cpu::glm_fp4_quantize_q8(job.hidden.data(),2048,job.routes->size());
                                    if(cpu_q8_bias||cpu_q8_policy)cpu::glm_fp4_prepare_q8_bias(job.qhidden);
                                    else if(cpu_q8_prepare)cpu::glm_fp4_prepare_q8(job.qhidden);
                                }
                                if(!barrier())return;
                            }
                            down_range(member,workers,j,j+1);
                        }
                    }catch(...){cancelled.store(true,std::memory_order_release);throw;}
                };
                cpu_pool.run(std::ref(team_work));
                if(profile)cpu_times.fused+=elapsed(begin);
            }else if(cpu_fused_enabled) {
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
        // A streamed layer leaves the next layer's uploads queued behind it.
        const bool stream_layer=prefill_stream_enabled&&!decode&&layer<45&&
            contexts.front()->stream_ring&&stream_planned_layer==layer;
        const bool streaming=stream_layer&&device_routing;
        const std::array<int,2> fill_limit{prefill_stream_hold&&contexts.size()==2?layer:INT_MAX,INT_MAX};
        // Allocated by the coordinator: the primary worker owns its scratch map.
        void* pushed_sum=nullptr;uint16_t* half_input=nullptr;
        if(streaming&&gpu_reduce&&peer_enabled&&contexts.size()==2) {
            DeviceScope scope(contexts.front()->device);
            pushed_sum=contexts.front()->buffer("route_other_sum",size_t(nt)*4096*4).p;
            // The secondary gathers FP16 rows, so send the rounded copy:
            // glm_f16 and route_gather round identically, at half the bytes.
            half_input=static_cast<uint16_t*>(contexts.front()->buffer("route_input_half",size_t(nt)*4096*2).p);
            if(early_input_layer!=layer) {
                k::glm_f16(x,half_input,int64_t(nt)*4096,caller);
                check(cudaEventRecord(contexts.front()->stream_ring->input_ready,caller));
            }
        }
        // A wider or skipped layer must not reuse an earlier prefetch.
        const bool early_input=half_input&&early_input_layer==layer;early_input_layer=-1;
        auto run=[&](size_t rank) {
            Context& c=*contexts[rank];DeviceScope scope(c.device);
            struct Drain {
                Context& c;bool streaming;int exceptions=std::uncaught_exceptions();
                ~Drain(){
                    const bool failed=std::uncaught_exceptions()>exceptions;
                    if(c.copy_stream||!c.protected_experts.empty())cudaStreamSynchronize(c.stream);
                    if(c.copy_stream&&(!streaming||failed))cudaStreamSynchronize(c.copy_stream);
                    if(failed&&c.stream_ring)cudaStreamSynchronize(c.stream_ring->prep);
                    c.protected_experts.clear();
                }
            } drain{c,stream_layer};
            if(stream_layer&&!streaming) {
                auto& s=*c.stream_ring;
                while(!s.queued.empty()&&s.queued.front().layer==layer)stream_drop(c);
                while(!s.planned.empty()&&s.planned.front().first==layer)s.planned.pop_front();
                stream_fill(c);
            }
            const bool batch_routes=route_id_batch&&decode&&device_routing&&(!c.bank||c.bank->layer!=layer);
            struct DrainRouteFailure {
                cudaStream_t stream;bool active;int exceptions;
                ~DrainRouteFailure(){if(active&&std::uncaught_exceptions()>exceptions)cudaStreamSynchronize(stream);}
            } route_failure{c.stream,batch_routes||resident_grouped,
                            (batch_routes||resident_grouped)?std::uncaught_exceptions():0};
            const size_t sum_count=size_t(nt)*4096;auto& sum=sums[rank];
            if(gpu_reduce)sum.clear();else sum.assign(sum_count,0);
            std::vector<float> input,result;
            const float* device_input=x;const float* device_weights=weights;
            const uint16_t* device_input_half=nullptr;
            float* device_sum=nullptr;
            if(device_routing) {
                device_sum=c.buffer("prefill_route_sum",sum_count*4).f();
                check(cudaMemsetAsync(device_sum,0,sum_count*4,c.stream));
                if(rank&&half_input) {
                    auto* rows=static_cast<uint16_t*>(c.buffer("prefill_route_input_half",size_t(nt)*4096*2).p);
                    device_weights=c.buffer("prefill_route_weights",size_t(nt)*8*4).f();
                    if(!early_input) {
                        check(cudaStreamWaitEvent(c.stream,contexts.front()->stream_ring->input_ready,0));
                        check(cudaMemcpyPeerAsync(rows,c.device,half_input,contexts.front()->device,size_t(nt)*4096*2,c.stream));
                    }
                    // The ring's copies hold this GPU's copy engine.
                    k::glm_fp4_copy_host_words(host_weights->p,const_cast<float*>(device_weights),size_t(nt)*8,c.stream);
                    device_input=nullptr;device_input_half=rows;
                }else if(rank) {
                    device_input=c.buffer("prefill_route_input",size_t(nt)*4096*4).f();
                    device_weights=c.buffer("prefill_route_weights",size_t(nt)*8*4).f();
                    if(gpu_reduce&&peer_enabled)check(cudaMemcpyPeerAsync(const_cast<float*>(device_input),c.device,x,contexts.front()->device,size_t(nt)*4096*4,c.stream));
                    else check(cudaMemcpyAsync(const_cast<float*>(device_input),host_x->p,size_t(nt)*4096*4,cudaMemcpyHostToDevice,c.stream));
                    check(cudaMemcpyAsync(const_cast<float*>(device_weights),host_weights->p,size_t(nt)*8*4,cudaMemcpyHostToDevice,c.stream));
                }
                if(decode && !rank) {
                    // Decode may renormalize the copied routing coefficients.
                    // The checkpoint/router's device coefficients stay intact.
                    device_weights=c.buffer("decode_route_weights",size_t(nt)*8*4).f();
                    check(cudaMemcpyAsync(const_cast<float*>(device_weights),host_weights->p,size_t(nt)*8*4,cudaMemcpyHostToDevice,c.stream));
                }
            }
            // Ownership stays fixed for each phase of a request. Experimental
            // global admission can reassign decode weights after prefill.
            const bool pipeline=async_upload_enabled&&device_routing&&!decode&&Device::prealloc();
            std::vector<int> jobs;
            for(const auto& [id,routes]:grouped) {
                if(owner(layer,id,decode)!=rank)continue;
                if(decode&&!c.keep.at(layer).count(id))continue;
                jobs.push_back(id);
            }
            // Do not admit or upload new weights in this path. Protect every
            // resident operand before allocating bounded scratch, which can evict.
            bool group_ready=resident_grouped&&sm70&&decode&&device_routing&&gpu_reduce&&peer_enabled&&
                !expert_fusion&&nt<=8&&jobs.size()>1&&(!c.bank||c.bank->layer!=layer);
            size_t total_rows=0;
            if(group_ready)for(int id:jobs) {
                const auto it=c.experts.find({layer,id});
                const auto& rows=grouped.at(id);
                if(it==c.experts.end()||rows.empty()||rows.size()>8||rows.size()>64-total_rows){group_ready=false;break;}
                for(int row:rows)if(row<0||row>=nt*8)group_ready=false;
                for(int p=0;p<3;++p) {
                    const auto& w=it->second.projection[p];
                    if(w.format!=k::GlmPackedFormat::Nvfp4||w.rows!=(p==2?4096:2048)||w.columns!=(p==2?2048:4096))group_ready=false;
                }
                total_rows+=rows.size();
            }
            if(group_ready) {
                for(int id:jobs)c.protected_experts.insert({layer,id});
                constexpr size_t descriptor_bytes=64*sizeof(k::GlmFp4ExpertGroup),metadata_bytes=descriptor_bytes+64*sizeof(int);
                size_t bytes=metadata_bytes;
                auto reserve=[&](size_t n){const size_t at=(bytes+255)/256*256;bytes=at+n;return at;};
                const size_t xo=reserve(total_rows*4096*2),go=reserve(total_rows*2048*4),
                    uo=reserve(total_rows*2048*4),ho=reserve(total_rows*2048*2),yo=reserve(total_rows*4096*4);
                auto* base=static_cast<uint8_t*>(c.buffer("decode_grouped_workspace",bytes).p);
                if(!c.grouped_host)c.grouped_host=std::make_unique<Pinned>(metadata_bytes,true);
                // The previous call's final stream synchronization retires this
                // portable pinned source before reuse. Drain also covers exceptions.
                std::memset(c.grouped_host->p,0,metadata_bytes);
                auto* desc=static_cast<k::GlmFp4ExpertGroup*>(c.grouped_host->p);
                auto* routes=reinterpret_cast<int*>(static_cast<uint8_t*>(c.grouped_host->p)+descriptor_bytes);
                size_t offset=0;
                for(size_t i=0;i<jobs.size();++i) {
                    const int id=jobs[i];auto& e=c.experts.at({layer,id});const auto& rows=grouped.at(id);
                    ++c.hits;e.use=++c.clock;
                    desc[i]={};desc[i].count=int(rows.size());desc[i].offset=int(offset);
                    for(int p=0;p<3;++p) {
                        const auto& w=e.projection[p];desc[i].weights[p]=static_cast<const uint8_t*>(w.data->p);
                        desc[i].scales[p]=static_cast<const uint8_t*>(w.scales());desc[i].global[p]=w.global;
                    }
                    std::copy(rows.begin(),rows.end(),routes+offset);offset+=rows.size();
                }
                check(cudaMemcpyAsync(base,c.grouped_host->p,metadata_bytes,cudaMemcpyHostToDevice,c.stream));
                auto* dg=reinterpret_cast<const k::GlmFp4ExpertGroup*>(base);
                auto* dr=reinterpret_cast<const int*>(base+descriptor_bytes);
                auto* hx=reinterpret_cast<uint16_t*>(base+xo);auto* gate=reinterpret_cast<float*>(base+go);
                auto* up=reinterpret_cast<float*>(base+uo);auto* hidden=reinterpret_cast<uint16_t*>(base+ho);
                auto* result=reinterpret_cast<float*>(base+yo);
                k::glm_fp4_route_gather(device_input,dr,hx,int(total_rows),4096,8,c.stream);
                k::glm_fp4_expert_group_products(dg,int(jobs.size()),false,hx,gate,up,c.stream);
                k::glm_swiglu(gate,up,gate,int(total_rows)*2048,limit,c.stream);
                k::glm_f16(gate,hidden,total_rows*2048,c.stream);
                k::glm_fp4_expert_group_products(dg,int(jobs.size()),true,hidden,result,nullptr,c.stream);
                if(grouped_scatter) {
                    k::glm_fp4_expert_group_scatter(dg,int(jobs.size()),dr,device_weights,result,device_sum,nt,c.stream);
                    ++c.grouped_scatters;
                }else {
                offset=0;
                // Retain the existing scatter kernel, once per original expert,
                // so multiply/add rounding and accumulation order stay identical.
                for(int id:jobs) {
                    const int count=int(grouped.at(id).size());
                    k::glm_fp4_route_scatter(result+offset*4096,dr+offset,device_weights,device_sum,count,4096,8,c.stream);
                    offset+=count;
                }
                }
                check(cudaGetLastError());check(cudaStreamSynchronize(c.stream));
                ++c.grouped_calls;c.grouped_experts+=jobs.size();c.grouped_rows+=total_rows;c.mma_calls+=3*jobs.size();
                return;
            }
            const int* batched_device_routes=nullptr;
            std::array<int,288> route_offsets;
            if(batch_routes&&!jobs.empty()) {
                route_offsets.fill(-1);
                if(!c.route_ids_host)c.route_ids_host=std::make_unique<Pinned>(64*sizeof(int),true);
                auto* packed=static_cast<int*>(c.route_ids_host->p);size_t count=0;
                for(int id:jobs) {
                    const auto& rows=grouped.at(id);
                    if(id<0||id>=288||rows.size()>size_t(nt)*8-count)
                        throw std::logic_error("GLM FP4: route ID batch exceeds narrow bounds");
                    route_offsets[id]=int(count);
                    for(int row:rows) {
                        if(row<0||row>=nt*8)throw std::logic_error("GLM FP4: invalid batched route ID");
                        packed[count++]=row;
                    }
                }
                // This buffer is not resized by execute while the batch is active.
                // The pinned source remains owned by this context until its stream retires.
                auto* target=static_cast<int*>(c.buffer("expert_route_ids",64*sizeof(int)).p);
                check(cudaMemcpyAsync(target,packed,count*sizeof(int),cudaMemcpyHostToDevice,c.stream));
                batched_device_routes=target;++c.route_id_uploads;c.route_id_groups+=jobs.size();c.route_id_entries+=count;
            }
            auto execute=[&](Expert& e,const std::vector<int>& routes,const uint16_t* operands=nullptr,
                             const int* uploaded_routes=nullptr,const std::array<uint16_t*,3>* parts=nullptr) {
                const int count=int(routes.size());
                auto& hx=c.buffer("expert_x_half",size_t(count)*4096*2);
                auto& gate=c.buffer("expert_gate",size_t(count)*2048*4);
                auto& up=c.buffer("expert_up",size_t(count)*2048*4);
                auto& hidden=c.buffer("expert_hidden_half",size_t(count)*2048*2);
                auto& dy=c.buffer("expert_result",size_t(count)*4096*4);
                const int* device_routes=uploaded_routes;
                if(device_routing) {
                    if(!device_routes) {
                        auto* target=static_cast<int*>(c.buffer("expert_route_ids",size_t(count)*4).p);
                        check(cudaMemcpyAsync(target,routes.data(),size_t(count)*4,cudaMemcpyHostToDevice,c.stream));
                        device_routes=target;
                    }
                    if(device_input_half)k::glm_fp4_route_gather_half(device_input_half,device_routes,static_cast<uint16_t*>(hx.p),count,4096,8,c.stream);
                    else k::glm_fp4_route_gather(device_input,device_routes,static_cast<uint16_t*>(hx.p),count,4096,8,c.stream);
                }else {
                    auto& dx=c.buffer("expert_x",input.size()*4);
                    check(cudaMemcpyAsync(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice,c.stream));
                    k::glm_f16(dx.f(),static_cast<uint16_t*>(hx.p),input.size(),c.stream);
                }
                constexpr size_t elements=size_t(2048)*4096;
                const auto part=[&](int i)->const uint16_t* {return parts?(*parts)[i]:operands?operands+i*elements:nullptr;};
                if(expert_fusion&&sm70&&decode&&e.projection[0].format==k::GlmPackedFormat::Nvfp4&&
                   e.projection[1].format==k::GlmPackedFormat::Nvfp4) {
                    const auto& g=e.projection[0];const auto& u=e.projection[1];
                    const auto trace=c.trace_start("fused_gate_up",c.stream);
                    k::glm_fp4_gate_up_f16(static_cast<const uint8_t*>(g.data->p),static_cast<const uint8_t*>(g.scales()),g.global,
                        static_cast<const uint8_t*>(u.data->p),static_cast<const uint8_t*>(u.scales()),u.global,
                        static_cast<uint16_t*>(hx.p),static_cast<uint16_t*>(hidden.p),2048,4096,count,limit,c.stream);
                    c.trace_end(trace,c.stream);c.mma_calls+=2;
                }else {
                product(c,e.projection[0],static_cast<uint16_t*>(hx.p),gate.f(),count,0,part(0));
                product(c,e.projection[1],static_cast<uint16_t*>(hx.p),up.f(),count,0,part(1));
                k::glm_swiglu(gate.f(),up.f(),gate.f(),count*2048,limit,c.stream);
                k::glm_f16(gate.f(),static_cast<uint16_t*>(hidden.p),size_t(count)*2048,c.stream);
                }
                product(c,e.projection[2],static_cast<uint16_t*>(hidden.p),dy.f(),count,0,part(2));
                if(device_routing)k::glm_fp4_route_scatter(dy.f(),device_routes,device_weights,device_sum,count,4096,8,c.stream);
                else check(cudaMemcpyAsync(result.data(),dy.p,result.size()*4,cudaMemcpyDeviceToHost,c.stream));
            };
            auto execute_batch=[&](std::vector<PendingExpert>& experts,const std::vector<int>& ids) {
                const int batch=int(ids.size());int count=0;
                for(int id:ids)count=std::max(count,int(grouped.at(id).size()));
                const int padded=(count+15)/16*16;
                ++c.batch_groups;c.batch_experts+=batch;c.batch_padded_rows+=size_t(batch)*padded;
                for(int id:ids)c.batch_actual_rows+=grouped.at(id).size();
                constexpr size_t elements=size_t(2048)*4096;
                size_t bytes=0;
                auto reserve=[&](size_t n){const size_t at=(bytes+255)/256*256;bytes=at+n;return at;};
                const size_t wo=reserve(2*batch*elements*2),xo=reserve(size_t(batch)*padded*4096*2);
                const size_t go=reserve(size_t(2*batch)*padded*2048*4),ho=reserve(size_t(batch)*padded*2048*2);
                const size_t yo=reserve(size_t(batch)*padded*4096*4),ro=reserve(size_t(batch)*padded*4);
                const size_t po=reserve(size_t(9*batch)*sizeof(void*));
                if(bytes+size_t(4)*expert_bytes>256*MiB)throw std::logic_error("GLM FP4: batch scratch admission mismatch");
                auto* base=static_cast<uint8_t*>(c.buffer("prefill_batch_workspace",bytes).p);
                auto* half=reinterpret_cast<uint16_t*>(base+wo);auto* hx=reinterpret_cast<uint16_t*>(base+xo);
                auto* gu=reinterpret_cast<float*>(base+go);auto* hidden=reinterpret_cast<uint16_t*>(base+ho);
                auto* result=reinterpret_cast<float*>(base+yo);auto* routes=reinterpret_cast<int*>(base+ro);
                auto* pointers=reinterpret_cast<void**>(base+po);
                std::vector<void*> host(9*batch);
                // Keep pointer-table sources alive while unwinding a failed enqueue.
                struct DrainOnFailure {cudaStream_t stream;int exceptions=std::uncaught_exceptions();
                    ~DrainOnFailure(){if(std::uncaught_exceptions()>exceptions)cudaStreamSynchronize(stream);}} drain{c.stream};
                check(cudaMemsetAsync(hx,0,size_t(batch)*padded*4096*2,c.stream));
                for(int b=0;b<batch;++b) {
                    const auto& route=grouped.at(ids[b]);
                    check(cudaMemcpyAsync(routes+size_t(b)*padded,route.data(),route.size()*4,cudaMemcpyHostToDevice,c.stream));
                    k::glm_fp4_route_gather(device_input,routes+size_t(b)*padded,hx+size_t(b)*padded*4096,route.size(),4096,8,c.stream);
                    for(int p=0;p<2;++p) {
                        const int i=2*b+p;const auto& w=experts[b].value->projection[p];
                        const auto trace=c.trace_start("reconstruct",c.stream);
                        k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,half+size_t(i)*elements,w.rows,w.columns,c.stream,sm70);
                        c.trace_end(trace,c.stream);
                        host[i]=half+size_t(i)*elements;host[2*batch+i]=hx+size_t(b)*padded*4096;
                        host[4*batch+i]=gu+size_t(i)*padded*2048;
                    }
                    host[6*batch+b]=half+size_t(b)*elements;
                    host[7*batch+b]=hidden+size_t(b)*padded*2048;
                    host[8*batch+b]=result+size_t(b)*padded*4096;
                }
                check(cudaMemcpyAsync(pointers,host.data(),host.size()*sizeof(void*),cudaMemcpyHostToDevice,c.stream));
                const float one=1,zero=0;
                auto trace=c.trace_start("batched_gemm",c.stream);
                check(cublasGemmBatchedEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,2048,padded,4096,&one,
                    reinterpret_cast<const void* const*>(pointers),CUDA_R_16F,4096,
                    reinterpret_cast<const void* const*>(pointers+2*batch),CUDA_R_16F,4096,
                    &zero,pointers+4*batch,CUDA_R_32F,2048,2*batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                c.trace_end(trace,c.stream);
                for(int b=0;b<batch;++b) {
                    auto* gate=gu+size_t(2*b)*padded*2048;auto* up=gate+size_t(padded)*2048;
                    k::glm_swiglu(gate,up,gate,padded*2048,limit,c.stream);
                    k::glm_f16(gate,hidden+size_t(b)*padded*2048,size_t(padded)*2048,c.stream);
                    const auto& w=experts[b].value->projection[2];
                    trace=c.trace_start("reconstruct",c.stream);
                    k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,half+size_t(b)*elements,w.rows,w.columns,c.stream,sm70);
                    c.trace_end(trace,c.stream);
                }
                trace=c.trace_start("batched_gemm",c.stream);
                check(cublasGemmBatchedEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,4096,padded,2048,&one,
                    reinterpret_cast<const void* const*>(pointers+6*batch),CUDA_R_16F,2048,
                    reinterpret_cast<const void* const*>(pointers+7*batch),CUDA_R_16F,2048,
                    &zero,pointers+8*batch,CUDA_R_32F,4096,batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                c.trace_end(trace,c.stream);c.prefill_calls+=3*batch;
                for(int b=0;b<batch;++b)k::glm_fp4_route_scatter(result+size_t(b)*padded*4096,
                    routes+size_t(b)*padded,device_weights,device_sum,grouped.at(ids[b]).size(),4096,8,c.stream);
                return host; // Keep asynchronous pointer-table upload sources alive until retirement.
            };
            size_t max_rows=0;for(int id:jobs)max_rows=std::max(max_rows,grouped.at(id).size());
            const size_t working_bytes=max_rows*(4096*2+2048*4*2+2048*2+4096*4+4)+16*MiB;
            const size_t queue_depth=pipeline&&prefill_queue_depth>1&&
                working_bytes+size_t(prefill_queue_depth)*expert_bytes<=256*MiB?size_t(prefill_queue_depth):1;
            size_t batch_limit=pipeline?size_t(prefill_batch_experts):1;
            // Two pairs or one quartet retain at most four uploaded experts.
            // Admit each group by its own row count, not the hottest expert in the layer.
            const size_t batch_window=prefill_batch_experts==4?1:2;
            if(streaming) {
                auto& s=*c.stream_ring;const auto stream_begin=tick();
                // One route upload per layer: per-expert copies would queue
                // behind the large frame transfers on the copy engine.
                std::vector<int> route_table;std::array<int,288> route_at{};
                for(int id:jobs){const auto& r=grouped.at(id);route_at[id]=int(route_table.size());route_table.insert(route_table.end(),r.begin(),r.end());}
                // Cache hits, LRU clocks, protection and admissions follow the
                // original pipeline: job j+1 is prepared before job j finishes.
                std::vector<char> hit(jobs.size()+1,0);
                auto prepare_hit=[&](size_t j) {
                    if(j>=jobs.size()||hit[j])return;
                    const auto key=std::make_pair(layer,jobs[j]);
                    if(auto it=c.experts.find(key);it!=c.experts.end()) {
                        ++c.hits;it->second.use=++c.clock;c.protected_experts.insert(key);hit[j]=1;
                    }
                };
                prepare_hit(0);
                // A following batch marks every hit first (see stream_followups).
                if(stream_followups>0)for(size_t j=1;j<jobs.size();++j)prepare_hit(j);
                auto* table=static_cast<int*>(c.buffer("stream_route_ids",std::max<size_t>(route_table.size(),1)*4).p);
                c.buffer("expert_x_half",max_rows*4096*2);
                c.buffer("expert_gate",max_rows*2048*4);c.buffer("expert_up",max_rows*2048*4);
                c.buffer("expert_hidden_half",max_rows*2048*2);
                c.buffer("expert_result",max_rows*4096*4);
                c.buffer("dequant_half",size_t(2048)*4096*2);
                if(!route_table.empty()) {
                    const int b=layer&1;auto& host=s.route_host[b];
                    check(cudaEventSynchronize(s.route_read[b]));
                    if(route_table.size()>s.route_host_words[b]) {
                        host=std::make_unique<Pinned>(route_table.size()*4,true);s.route_host_words[b]=route_table.size();
                    }
                    std::memcpy(host->p,route_table.data(),route_table.size()*4);
                    k::glm_fp4_copy_host_words(host->p,table,route_table.size(),c.stream);
                    check(cudaEventRecord(s.route_read[b],c.stream));
                }
                check(cudaMemsetAsync(s.flags,0,PrefillStream::flag_count*4,s.prep));
                // Storage read by queued kernels stays alive until the final fence.
                std::vector<PendingExpert> demands;
                std::vector<std::unique_ptr<Expert>> narrow;
                int position=0;
                for(size_t j=0;j<jobs.size();++j) {
                    const int id=jobs[j];const auto key=std::make_pair(layer,id);
                    // Skip unrouted uploads; refill so a planned expert is queued.
                    for(;;) {
                        while(!s.queued.empty()&&s.queued.front().layer==layer&&s.queued.front().id<id)stream_drop(c);
                        if(!s.queued.empty()||s.planned.empty())break;
                        stream_fill(c,fill_limit[rank]);
                        if(s.queued.empty())break;
                    }
                    const auto& routes=grouped.at(id);const int* rows=table+route_at[id];
                    const bool admit=c.keep.at(layer).count(id);
                    const bool queued=!s.queued.empty()&&s.queued.front().layer==layer&&s.queued.front().id==id;
                    std::unique_ptr<Expert> packed;PendingExpert demand;
                    if(hit[j]) {
                        if(queued)stream_drop(c);
                        execute(c.experts.at(key),routes,nullptr,rows);
                    }else if(queued) {
                        const auto item=s.queued.front();s.queued.pop_front();
                        if(position>=288)throw std::logic_error("GLM FP4: streamed layer exceeds its flag table");
                        Expert meta;
                        for(int p=0;p<3;++p) {
                            const auto& h=*item.sources[p];auto& m=meta.projection[p];
                            m.format=k::GlmPackedFormat::Nvfp4;m.rows=h.rows;m.columns=h.columns;m.global=h.global;
                        }
                        // Narrow experts use the skinny packed kernel, as in the
                        // original path. Kept experts retain its cache layout.
                        const bool wide=routes.size()>8;
                        if(admit||!wide) {
                            packed=std::make_unique<Expert>();
                            for(int p=0;p<3;++p) {
                                const auto& h=*item.sources[p];auto& q=packed->projection[p];
                                q.format=k::GlmPackedFormat::Nvfp4;q.rows=h.rows;q.columns=h.columns;q.global=h.global;
                                q.scale_offset=(h.bytes[0]+15)/16*16;
                                const size_t bytes=q.scale_offset+size_t(q.rows)*q.columns/16;
                                c.room(bytes);q.data=std::make_unique<Device>(bytes);packed->bytes+=bytes;
                            }
                        }
                        const size_t slot=s.operand;s.operand=(s.operand+1)%PrefillStream::operand_slots;
                        check(cudaStreamWaitEvent(s.prep,s.full[item.frame],0));
                        check(cudaStreamWaitEvent(s.prep,s.operand_free[slot],0));
                        const auto trace=c.trace_start("stream_operands",s.prep);
                        for(int p=0;p<3;++p) {
                            const auto& h=*item.sources[p];const auto& at=item.offsets[p];
                            auto* frame=s.frames[item.frame][p];auto* out=s.operands[slot][p];
                            // Exact fused reconstruction: the same FP16 operand bits
                            // as prepack, scale decoding and fp4_sm70_operands.
                            if(wide&&h.csf)k::glm_fp4_csf32_dequant_f16(frame+at[0],frame+at[1],frame+at[2],
                                reinterpret_cast<const uint16_t*>(frame+at[3]),reinterpret_cast<const uint32_t*>(frame+at[4]),
                                reinterpret_cast<const uint16_t*>(frame+at[5]),h.global,out,s.flags+position*3+p,h.rows,h.columns,s.prep);
                            else if(wide)k::glm_fp4_raw_dequant_f16(frame+at[0],frame+at[1],h.global,out,h.rows,h.columns,false,s.prep);
                            if(packed) {
                                auto& q=packed->projection[p];auto* output=static_cast<uint8_t*>(q.data->p);
                                k::glm_fp4_prepack(frame+at[0],h.csf?nullptr:frame+at[1],output,h.csf?nullptr:output+q.scale_offset,q.rows,q.columns,s.prep);
                                if(h.csf)k::glm_fp4_csf32_scales(frame+at[1],frame+at[2],
                                    reinterpret_cast<const uint16_t*>(frame+at[3]),reinterpret_cast<const uint32_t*>(frame+at[4]),
                                    reinterpret_cast<const uint16_t*>(frame+at[5]),output+q.scale_offset,
                                    s.flags+PrefillStream::flag_count/2+position*3+p,q.rows,q.columns,s.prep);
                            }
                        }
                        check(cudaGetLastError());
                        c.trace_end(trace,s.prep);
                        {std::lock_guard lock(s.mutex);check(cudaEventRecord(s.consumed[item.frame],s.prep));s.refillable[item.frame]=1;}
                        check(cudaEventRecord(s.operand_ready[slot],s.prep));
                        check(cudaStreamWaitEvent(c.stream,s.operand_ready[slot],0));
                        if(wide)execute(meta,routes,nullptr,rows,&s.operands[slot]);
                        else execute(*packed,routes,nullptr,rows);
                        check(cudaEventRecord(s.operand_free[slot],c.stream));
                        ++c.misses;++c.stream_consumed;c.uploaded+=expert_bytes;++position;
                    }else {
                        // Evicted after planning: the original bounded upload path.
                        demand=prepare_expert(c,layer,id);demand.wait();
                        execute(*demand.value,routes,nullptr,rows);++c.stream_demand;
                    }
                    prepare_hit(j+1);
                    // finish_expert's admission, without its stream fence. The
                    // new entry stays protected while its packing kernels run.
                    if(packed&&admit) {
                        while(!c.experts.empty()&&c.cache_bytes+packed->bytes>c.storage_limit())c.evict();
                        packed->use=++c.clock;c.cache_bytes+=packed->bytes;
                        if(!c.experts.emplace(key,std::move(*packed)).second)throw std::logic_error("GLM FP4: duplicate streamed admission");
                        c.protected_experts.insert(key);++c.stream_admitted;
                    }else if(packed)narrow.push_back(std::move(packed));
                    if(demand.owned&&admit) {
                        while(!c.experts.empty()&&c.cache_bytes+demand.value->bytes>c.storage_limit())c.evict();
                        demand.value->use=++c.clock;c.cache_bytes+=demand.value->bytes;
                        c.experts.emplace(key,std::move(*demand.owned));demand.owned.reset();
                        c.protected_experts.insert(key);
                    }
                    if(demand.owned)demands.push_back(std::move(demand));
                    stream_fill(c,fill_limit[rank]);
                }
                while(!s.queued.empty()&&s.queued.front().layer==layer)stream_drop(c);
                while(!s.planned.empty()&&s.planned.front().first==layer)s.planned.pop_front();
                stream_fill(c,fill_limit[rank]);
                check(cudaStreamSynchronize(c.stream));
                check(cudaMemcpyAsync(s.flags_host->p,s.flags,PrefillStream::flag_count*4,cudaMemcpyDeviceToHost,s.prep));
                check(cudaStreamSynchronize(s.prep));
                for(auto& d:demands)d.completed=true;
                const auto* flags=static_cast<const int*>(s.flags_host->p);
                for(size_t i=0;i<PrefillStream::flag_count;++i)if(flags[i])
                    throw std::runtime_error("GLM ModelOpt: invalid FP4 group scale in streamed prefill");
                ++c.stream_layers;
                if(profile)times[rank].product+=elapsed(stream_begin);
            }else if(c.bank&&c.bank->layer==layer&&device_routing) {
                auto& bank=*c.bank;c.bank_unused_experts-=jobs.size();
                // Keep addresses stable while all experts are queued. The
                // host route vectors live until the final stream fence.
                c.buffer("expert_x_half",max_rows*4096*2);
                c.buffer("expert_gate",max_rows*2048*4);c.buffer("expert_up",max_rows*2048*4);
                c.buffer("expert_hidden_half",max_rows*2048*2);
                c.buffer("expert_result",max_rows*4096*4);c.buffer("expert_route_ids",max_rows*4);
                c.buffer("dequant_half",size_t(2048)*4096*2);
                for(int id:jobs) {
                    const size_t slot=size_t(id)/2;
                    Expert* expert=bank.cached[slot]?bank.cached[slot]:&bank.experts[slot];
                    if(bank.cached[slot])expert->use=++c.clock;
                    execute(*expert,grouped.at(id));
                }
                check(cudaStreamSynchronize(c.stream));
                // Cached kept sources stay protected through admission. An
                // uncached bank slot owns its packed bytes independently.
                const auto& kept=c.keep.at(layer);
                for(auto it=c.protected_experts.begin();it!=c.protected_experts.end();) {
                    if(it->first==layer&&!kept.count(it->second))it=c.protected_experts.erase(it);
                    else ++it;
                }
                for(int id:jobs) {
                    const auto key=std::make_pair(layer,id);
                    if(!kept.count(id)||c.experts.count(key))continue;
                    const size_t slot=size_t(id)/2;
                    if(bank.cached[slot])throw std::logic_error("GLM FP4: prefill bank lost a protected source");
                    while(c.cache_bytes+expert_bytes>c.storage_limit())c.evict();
                    Expert e;
                    struct CopyDrain {
                        cudaStream_t stream;int exceptions=std::uncaught_exceptions();
                        ~CopyDrain(){if(std::uncaught_exceptions()>exceptions)cudaStreamSynchronize(stream);}
                    } copying{c.stream};
                    for(int p=0;p<3;++p) {
                        const auto& from=bank.experts[slot].projection[p];auto& to=e.projection[p];
                        to.format=from.format;to.rows=from.rows;to.columns=from.columns;
                        to.global=from.global;to.scale_offset=from.scale_offset;
                        c.room(from.data->bytes);to.data=std::make_unique<Device>(from.data->bytes);
                        check(cudaMemcpyAsync(to.data->p,from.data->p,from.data->bytes,cudaMemcpyDeviceToDevice,c.stream));
                        e.bytes+=from.data->bytes;
                    }
                    check(cudaStreamSynchronize(c.stream));e.use=++c.clock;
                    const size_t bytes=e.bytes;
                    if(!c.experts.emplace(key,std::move(e)).second)
                        throw std::logic_error("GLM FP4: duplicate prefill bank admission");
                    c.cache_bytes+=bytes;c.protected_experts.insert(key);
                }
            }else if(prefill_operand_pipeline&&pipeline&&max_rows>8&&
                     working_bytes+96*MiB+2*expert_bytes<=256*MiB) {
                const auto queue_begin=tick();
                // Allocate before enqueuing readers: buffer growth may release
                // its old address, and room() may evict unprotected experts.
                c.buffer("expert_x_half",max_rows*4096*2);
                c.buffer("expert_gate",max_rows*2048*4);c.buffer("expert_up",max_rows*2048*4);
                c.buffer("expert_hidden_half",max_rows*2048*2);
                c.buffer("expert_result",max_rows*4096*4);c.buffer("expert_route_ids",max_rows*4);
                constexpr size_t elements=size_t(2048)*4096;
                // Dense prefixes have retired at routed()'s input fence. This
                // optional workspace can serve both phases without retaining
                // a second reconstructed-weight allocation between layers.
                auto* operands=static_cast<uint16_t*>(c.buffer(
                    prefill_operand_overlap?"dequant_half":"prefill_operands",6*elements*2).p);
                // A tiny group keeps the original skinny product and its scratch.
                c.buffer("matrix_partial",k::glm_fp4_partial_bytes(512));
                c.init_upload();
                struct OperandEvents {
                    Context& c;cudaEvent_t ready[2]{};cudaStream_t reconstruct=nullptr;
                    ~OperandEvents(){cudaStreamSynchronize(c.copy_stream);
                        if(reconstruct)cudaStreamSynchronize(reconstruct);
                        cudaStreamSynchronize(c.stream);c.async_eviction=false;
                        for(auto e:ready)if(e)cudaEventDestroy(e);
                        if(reconstruct)cudaStreamDestroy(reconstruct);}
                } events{c};
                if(prefill_operand_overlap)check(cudaStreamCreateWithFlags(&events.reconstruct,cudaStreamNonBlocking));
                const auto reader=events.reconstruct?events.reconstruct:c.copy_stream;
                for(auto& event:events.ready)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                c.async_eviction=true;
                std::deque<PendingExpert> retired;
                auto retire=[&] {
                    auto& p=retired.front();p.wait();
                    check(cudaEventSynchronize(c.consumed[p.done_slot]));p.completed=true;
                    finish_expert(c,p,c.keep.at(layer).count(p.key.second));retired.pop_front();
                };
                auto stage=[&](size_t j) {
                    const int slot=int(j%2);auto p=prepare_expert(c,layer,jobs[j]);
                    c.protected_experts.insert(p.key);p.done_slot=slot;
                    // An owned source must survive an exception after a
                    // reconstruction enqueue, before its compute consumer.
                    p.reader_stream=events.reconstruct;
                    if(events.reconstruct&&p.slot>=0)
                        check(cudaStreamWaitEvent(reader,c.upload_ready[p.slot],0));
                    if(grouped.at(jobs[j]).size()>8)for(int projection=0;projection<3;++projection) {
                        const auto& w=p.value->projection[projection];
                        if(w.rows*w.columns!=int(elements)||w.format!=k::GlmPackedFormat::Nvfp4)
                            throw std::logic_error("GLM FP4: unexpected expert operand shape/format");
                        const auto trace=c.trace_start("reconstruct",reader);
                        k::glm_fp4_dequant_f16(w.format,w.data->p,w.scales(),w.global,
                            operands+(slot*3+projection)*elements,w.rows,w.columns,reader,sm70);
                        c.trace_end(trace,reader);
                    }
                    check(cudaGetLastError());check(cudaEventRecord(events.ready[slot],reader));
                    return p;
                };
                for(size_t j=0;j<jobs.size();++j) {
                    if(retired.size()==2)retire();
                    auto p=stage(j);
                    check(cudaStreamWaitEvent(c.stream,events.ready[p.done_slot],0));
                    execute(*p.value,grouped.at(jobs[j]),operands+size_t(p.done_slot)*3*elements);
                    ++c.operand_groups;c.operand_rows+=grouped.at(jobs[j]).size();
                    check(cudaEventRecord(c.consumed[p.done_slot],c.stream));
                    retired.push_back(std::move(p));
                }
                while(!retired.empty())retire();
                if(profile)times[rank].product+=elapsed(queue_begin);
            }else if(batch_limit>1) {
                const auto batch_begin=tick();c.init_upload();
                struct EvictionMode {Context& c;~EvictionMode(){c.async_eviction=false;}} eviction_mode{c};
                c.async_eviction=true;
                struct Batch {
                    std::vector<PendingExpert> experts;std::vector<void*> pointers;int slot;
                    Context* context=nullptr;bool completed=false;
                    Batch()=default;Batch(Batch&&)=default;
                    ~Batch(){if(context&&!completed&&!pointers.empty())cudaStreamSynchronize(context->stream);}
                };
                std::deque<Batch> retired;
                auto retire=[&] {
                    auto& b=retired.front();
                    for(auto& p:b.experts)p.wait();
                    check(cudaEventSynchronize(c.consumed[b.slot]));
                    for(auto& p:b.experts){p.completed=true;finish_expert(c,p,c.keep.at(layer).count(p.key.second));}
                    b.completed=true;retired.pop_front();
                };
                size_t serial=0;
                for(size_t j=0;j<jobs.size();) {
                    if(retired.size()==batch_window)retire();
                    size_t n=std::min(batch_limit,jobs.size()-j);
                    while(n>1) {
                        size_t total=0,largest=0,smallest=SIZE_MAX;
                        for(size_t b=0;b<n;++b) {const auto count=grouped.at(jobs[j+b]).size();total+=count;largest=std::max(largest,count);smallest=std::min(smallest,count);}
                        const size_t padded=(largest+15)/16*16;
                        if(smallest>8&&padded*n*4<=total*5&&
                           n*(32*MiB+padded*45060)+4*expert_bytes+4096<=256*MiB)break;
                        n/=2;
                    }
                    Batch batch;batch.context=&c;batch.slot=int(serial++%2);batch.experts.reserve(n);
                    std::vector<int> ids(jobs.begin()+j,jobs.begin()+j+n);
                    for(int id:ids) {
                        auto p=prepare_expert(c,layer,id);c.protected_experts.insert(p.key);
                        if(p.slot>=0)check(cudaStreamWaitEvent(c.stream,c.upload_ready[p.slot],0));
                        batch.experts.push_back(std::move(p));
                    }
                    if(n==1)execute(*batch.experts[0].value,grouped.at(ids[0]));
                    else batch.pointers=execute_batch(batch.experts,ids);
                    check(cudaEventRecord(c.consumed[batch.slot],c.stream));
                    retired.push_back(std::move(batch));j+=n;
                }
                while(!retired.empty())retire();
                if(profile)times[rank].product+=elapsed(batch_begin);
            }else if(queue_depth>1) {
                const auto queue_begin=tick();
                c.init_upload();
                struct EvictionMode {Context& c;~EvictionMode(){c.async_eviction=false;}} eviction_mode{c};
                c.async_eviction=true;
                std::deque<PendingExpert> retired;
                auto retire=[&] {
                    auto& p=retired.front();p.wait();
                    check(cudaEventSynchronize(c.consumed[p.done_slot]));p.completed=true;
                    finish_expert(c,p,c.keep.at(layer).count(p.key.second));
                    retired.pop_front();
                };
                for(size_t j=0;j<jobs.size();++j) {
                    if(retired.size()==queue_depth)retire();
                    auto p=prepare_expert(c,layer,jobs[j]);
                    c.protected_experts.insert(p.key);
                    if(p.slot>=0)check(cudaStreamWaitEvent(c.stream,c.upload_ready[p.slot],0));
                    execute(*p.value,grouped.at(jobs[j]));
                    p.done_slot=int(j%queue_depth);
                    check(cudaEventRecord(c.consumed[p.done_slot],c.stream));
                    retired.push_back(std::move(p));
                }
                while(!retired.empty())retire();
                if(profile)times[rank].product+=elapsed(queue_begin);
            }else {
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
                execute(e,routes,nullptr,batched_device_routes?batched_device_routes+route_offsets[id]:nullptr);
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
            }
            if(device_routing) {
                const auto begin=tick();
                if(gpu_reduce) {
                    if(rank&&!peer_enabled) {
                        if(!c.result_host||c.result_host_bytes<sum_count*4){c.result_host=std::make_unique<Pinned>(sum_count*4,true);c.result_host_bytes=sum_count*4;}
                        check(cudaMemcpyAsync(c.result_host->p,device_sum,sum_count*4,cudaMemcpyDeviceToHost,c.stream));
                    }else if(rank&&pushed_sum) {
                        // Send the partial sum as soon as this device finishes,
                        // while the primary may still be running its experts.
                        check(cudaMemcpyPeerAsync(pushed_sum,contexts.front()->device,device_sum,c.device,sum_count*4,c.stream));
                    }
                }else check(cudaMemcpyAsync(sum.data(),device_sum,sum_count*4,cudaMemcpyDeviceToHost,c.stream));
                check(cudaStreamSynchronize(c.stream));
                if(profile)times[rank].product+=elapsed(begin);
            }
        };
        // Independent device streams operate concurrently; shared model pages
        // and bounded CPU staging are owned by this executor.
        if(stream_layer&&layer<44)plan_stream_layer(layer+1);
        std::vector<std::future<void>> work;
        try {
            for(size_t rank=0;rank<contexts.size();++rank)work.push_back(contexts[rank]->submit([&,rank]{run(rank);}));
            run_cpu();for(auto& f:work)f.get();
            // The secondary fenced its pushed sum; release the held uploads.
            if(streaming&&fill_limit[0]!=INT_MAX) {
                auto& first=*contexts.front();
                held_fill=first.submit([this,&first]{DeviceScope scope(first.device);stream_fill(first);});
            }
        }catch(...){for(auto& f:work)if(f.valid())f.wait();throw;}
        auto& c=*contexts.front();DeviceScope scope(c.device);
        if(gpu_reduce) {
            const size_t count=size_t(nt)*4096;
            float* sum=c.scratch.at("prefill_route_sum")->f();
            const float one=1;check(cublasSetStream(c.blas,caller));
            if(contexts.size()==2) {
                auto& second=*contexts[1];auto& other=c.buffer("route_other_sum",count*4);
                if(pushed_sum&&other.p!=pushed_sum)throw std::logic_error("GLM FP4: pushed partial sum moved");
                // A pushed sum was already copied and fenced by the secondary worker.
                if(!pushed_sum&&peer_enabled)check(cudaMemcpyPeerAsync(other.p,c.device,second.scratch.at("prefill_route_sum")->p,second.device,count*4,caller));
                else if(!pushed_sum)check(cudaMemcpyAsync(other.p,second.result_host->p,count*4,cudaMemcpyHostToDevice,caller));
                check(cublasSaxpy(c.blas,count,&one,other.f(),1,sum,1));
            }
            if(!cpu_jobs.empty()) {
                auto& cpu=c.buffer("route_cpu_sum",count*4);
                check(cudaMemcpyAsync(cpu.p,cpu_sum.data(),count*4,cudaMemcpyHostToDevice,caller));
                check(cublasSaxpy(c.blas,count,&one,cpu.f(),1,sum,1));
            }
            check(cublasSaxpy(c.blas,count,&one,sum,1,out,1));
            check(cublasSetStream(c.blas,c.stream));
        }else {
        if(sums.size()==2)for(size_t i=0;i<sums[0].size();++i)sums[0][i]+=sums[1][i];
        if(!cpu_jobs.empty())for(size_t i=0;i<sums[0].size();++i)sums[0][i]+=cpu_sum[i];
        auto& result=c.buffer("route_sum",sums[0].size()*4);
        check(cudaMemcpyAsync(result.p,sums[0].data(),sums[0].size()*4,cudaMemcpyHostToDevice,caller));
        const float one=1;
        check(cublasSetStream(c.blas,caller));
        check(cublasSaxpy(c.blas,sums[0].size(),&one,result.f(),1,out,1));
        check(cublasSetStream(c.blas,c.stream));
        }
        check(cudaGetLastError());
        if(profile) {
            for(auto& context:contexts){DeviceScope scope(context->device);context->trace_report(layer,nt);}
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
        if(prefill_bank_enabled&&!decode&&contexts.front()->bank) {
            if(layer<44)start_bank_layer(layer+1);
            else release_banks();
        }
        // Free the ring before batched draft priming and decode.
        if(stream_layer&&layer==44){release_prefill_stream(true);++stream_followups;}
    }
    void report(bool flush_profile=true)const {
        if(grouped_scatter)for(const auto& c:contexts)std::cerr<<"FP4_GROUPED_SCATTER_STATS device="<<c->device
            <<" calls="<<c->grouped_scatters<<'\n';
        if(resident_grouped)for(const auto& c:contexts)std::cerr<<"FP4_RESIDENT_GROUPED_STATS device="<<c->device
            <<" calls="<<c->grouped_calls<<" experts="<<c->grouped_experts<<" rows="<<c->grouped_rows<<'\n';
        if(early_readback)std::cerr<<"FP4_EARLY_READBACK_STATS calls="<<early_readback_calls<<'\n';
        if(route_id_batch)for(const auto& c:contexts)std::cerr<<"FP4_ROUTE_ID_BATCH_STATS device="<<c->device
            <<" uploads="<<c->route_id_uploads<<" groups="<<c->route_id_groups<<" ids="<<c->route_id_entries<<'\n';
        if(profile&&flush_profile)for(const auto& c:contexts) {
            DeviceScope scope(c->device);c->trace_report(-1,0);
        }
        if(prefill_bank_enabled)for(const auto& c:contexts)
            std::cerr<<"FP4_PREFILL_BANK_STATS device="<<c->device<<" layers="<<c->bank_layers
                <<" unused_experts="<<c->bank_unused_experts<<" wait_ms="<<c->bank_wait_ms<<'\n';
        if(prefill_stream_enabled)for(const auto& c:contexts)
            std::cerr<<"FP4_PREFILL_STREAM_STATS device="<<c->device<<" layers="<<c->stream_layers
                <<" prefetched="<<c->stream_prefetched<<" consumed="<<c->stream_consumed
                <<" dropped="<<c->stream_dropped<<" demand="<<c->stream_demand
                <<" admitted="<<c->stream_admitted<<" restored="<<c->stream_restored<<" lent_evicted="<<c->lent_evicted<<" lent_freed="<<c->lent_freed<<" small_direct="<<c->stream_small_direct<<" small_staged="<<c->stream_small_staged<<" skipped="<<stream_skipped
                <<" evictions="<<c->evictions
                <<" frame_bytes="<<stream_projection_bytes[0]+stream_projection_bytes[1]+stream_projection_bytes[2]<<'\n';
        // Diagnostic only: room() evictions are forced by physical memory,
        // the rest by the cache budget.
        if(eviction_stats)for(const auto& c:contexts)std::cerr<<"FP4_EVICTIONS device="<<c->device
            <<" total="<<c->evictions<<" physical="<<c->room_evictions<<" cache_bytes="<<c->cache_bytes
            <<" cache_limit="<<c->cache_limit<<'\n';
        if(head_partition)std::cerr<<"FP4_HEAD_TP_STATS calls="<<head_partition->calls<<'\n';
        if(keep_reselect_rows)std::cerr<<"FP4_KEEP_RESELECT rows="<<keep_reselect_rows<<" max_swaps="<<keep_reselect_swaps
            <<" prefetch="<<keep_reselect_prefetch<<" layers="<<reselect_layers<<" swaps="<<reselect_swaps
            <<" reclaimed_bytes="<<reselect_reclaimed_bytes<<" prefetched="<<reselect_prefetched
            <<" prefetch_skipped="<<reselect_prefetch_skipped<<" prefetch_dropped="<<reselect_prefetch_dropped<<'\n';
        if(cache_adapt_rows)std::cerr<<"FP4_CACHE_ADAPT rows="<<cache_adapt_rows
            <<" max_promotions_per_device=2 hysteresis=2 promotions="<<cache_promotions
            <<" reclaimed_bytes="<<cache_reclaimed_bytes<<'\n';
        if(decode_device_routing_enabled)std::cerr<<"FP4_DECODE_ROUTING device_calls="<<decode_device_calls
            <<" host_calls="<<decode_host_calls<<'\n';
        if(route_drop_fraction>0)std::cerr<<"FP4_ROUTE_DROP_STATS token_layers="<<route_drop_tokens
            <<" cpu_candidates="<<route_drop_candidates<<" dropped="<<route_drop_count
            <<" mean_share="<<(route_drop_tokens?route_drop_share_sum/route_drop_tokens:0)
            <<" max_share="<<route_drop_share_max<<" max_multiplier="<<route_drop_multiplier_max<<'\n';
        std::cerr<<"FP4_CPU expert_groups="<<cpu_experts<<" activation_format="
            <<(cpu_q8_enabled?"q8_block64_from_f16":"f16")<<" weight_format=native_nvfp4\n";
        std::cerr<<"FP4_CPU_SCALES hits="<<cpu_cache_hits<<" misses="<<cpu_cache_misses<<" bytes="<<cpu_cache_bytes<<'\n';
        if(cpu_q8_policy||cpu_scale_local_m1) {
            std::cerr<<"FP4_CPU_WIDTHS";
            for(int width=1;width<=8;++width)std::cerr<<" w"<<width<<'='<<cpu_width_groups[width];
            std::cerr<<'\n';
        }
        for(const auto& c:contexts)if(prefill_batch_experts>1)std::cerr<<"FP4_BATCH device="<<c->device
            <<" groups="<<c->batch_groups<<" experts="<<c->batch_experts<<" actual_rows="<<c->batch_actual_rows
            <<" padded_rows="<<c->batch_padded_rows<<'\n';
        for(const auto& c:contexts)if(prefill_operand_pipeline)std::cerr<<"FP4_PREFILL_OPERANDS device="<<c->device
            <<" groups="<<c->operand_groups<<" rows="<<c->operand_rows
            <<" overlap="<<prefill_operand_overlap<<'\n';
        for(const auto& c:contexts)if(prefill_mma_tile)std::cerr<<"FP4_FUSED_PREFILL device="<<c->device
            <<" calls="<<c->fused_prefill_calls<<'\n';
        for(const auto& c:contexts)std::cerr<<"FP4_TRANSFER device="<<c->device
            <<" startup_bytes="<<c->wire_bytes[0]<<" prefill_bytes="<<c->wire_bytes[1]
            <<" decode_bytes="<<c->wire_bytes[2]<<'\n';
        for(const auto& c:contexts)std::cerr<<"FP4_ROUTES device="<<c->device<<" hits="<<c->hits<<" misses="<<c->misses
            <<" uploaded_bytes="<<c->uploaded<<" cache_bytes="<<c->cache_bytes<<" mma="<<c->mma_calls
            <<" reference="<<c->reference_calls<<" prefill="<<c->prefill_calls<<'\n';
    }
};
#endif
