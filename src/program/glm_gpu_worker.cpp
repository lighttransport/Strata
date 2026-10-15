// Linux stateless expert service. This target links exactly one vendor runtime.
#include "strata/core/model.hpp"
#include "strata/host/glm_env.h"
#include "ggml-cpu.h"
#include <cmath>
#include "strata/artifact/json.hpp"
#include <numeric>
#include "strata/core/glm_mixed.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/canon_expert.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/glm_q23.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/partition.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>
namespace {
namespace k = strata::kernels;
namespace cpu = k::cpu;
#include "glm_gpu_buffers.hpp"
#include "glm_resident_executor.hpp"
#ifdef STRATA_NATIVE_EXPERTS
#include "ggml-cpu.h"
#endif
// CPU copies only: GPU calls remain on the worker's main thread.
class CopyPool {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::packaged_task<void()>> tasks;
    std::vector<std::thread> threads;
    bool stopping=false;
public:
    explicit CopyPool(int count) {
        for (int i=0;i<count;++i) threads.emplace_back([this] {
            for (;;) {
                std::packaged_task<void()> task;
                { std::unique_lock lock(mutex);ready.wait(lock,[&]{return stopping || !tasks.empty();});
                  if (tasks.empty()) return;
                  task=std::move(tasks.front());tasks.pop_front(); }
                task();
            }
        });
    }
    ~CopyPool() {
        {std::lock_guard lock(mutex);stopping=true;}ready.notify_all();
        for (auto &thread:threads) thread.join();
    }
    std::future<void> submit(std::function<void()> copy) {
        std::packaged_task<void()> task(std::move(copy));auto future=task.get_future();
        if (threads.empty()) task();
        else {std::lock_guard lock(mutex);tasks.push_back(std::move(task));ready.notify_one();}
        return future;
    }
};
}

int main(int argc, char **argv) {
#ifdef __linux__
    try {
        if (argc==2 && std::string(argv[1])=="--device-info") { report_gpu_devices(std::cout); return 0; }
        if (argc==2 && std::string(argv[1])=="--backend") {
#ifdef STRATA_USE_HIP
            std::cout << "hip\n";
#else
            std::cout << "cuda\n";
#endif
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--transfer-bench") {
            cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            constexpr size_t maximum=32*MiB;
            Device device(maximum); Pinned input(maximum), output(maximum);
            std::memset(input.p, 0x5a, maximum);
            std::cout << "["; bool comma=false;
            for (size_t bytes: {size_t(4096),size_t(65536),MiB,8*MiB,32*MiB}) {
                for (int direction=0; direction<2; ++direction) {
                    const auto kind=direction ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                    void *destination=direction ? output.p : device.p;
                    const void *source=direction ? device.p : input.p;
                    for (int i=0; i<10; ++i) check(cudaMemcpyAsync(destination,source,bytes,kind,stream));
                    check(cudaStreamSynchronize(stream));
                    const auto start=std::chrono::steady_clock::now();
                    const int iterations=bytes<MiB ? 1000 : 100;
                    for (int i=0; i<iterations; ++i) {
                        check(cudaMemcpyAsync(destination,source,bytes,kind,stream));
                        check(cudaStreamSynchronize(stream));
                    }
                    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/iterations;
                    if (comma) std::cout << ",";
                    comma=true;
                    std::cout << "{\"direction\":\"" << (direction ? "d2h" : "h2d") << "\",\"bytes\":" << bytes
                              << ",\"latency_us\":" << seconds*1e6 << ",\"GB_s\":" << bytes/seconds/1e9 << "}";
                }
            }
            std::cout << "]\n"; check(cudaStreamDestroy(stream)); return 0;
        }
        if (argc==2 && std::string(argv[1])=="--gemm-bench") {
            cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            strata::prefill::Gemm gemm;std::string error;
            if (!gemm.init(stream,0,error)) throw std::runtime_error(error);
            constexpr int batches=4,N=4096,K=4096,maximum=512;
            Device x(size_t(batches)*maximum*K*2),w(size_t(batches)*N*K*2),y(size_t(batches)*maximum*N*sizeof(float));
            check(cudaMemset(x.p,0x23,x.bytes));check(cudaMemset(w.p,0x23,w.bytes));
            std::cout << "[";bool comma=false;
            for (int rows: {64,128,256,512}) for (bool single: {false,true}) {
                auto run=[&] {
                    if (single) for (int group=0;group<batches;++group)
                        gemm.f16(static_cast<uint16_t *>(x.p)+size_t(group)*rows*K,
                                 static_cast<uint16_t *>(w.p)+size_t(group)*N*K,
                                 y.f()+size_t(group)*rows*N,rows,N,K);
                    else gemm.f16_batched(static_cast<uint16_t *>(x.p),static_cast<uint16_t *>(w.p),y.f(),rows,N,K,batches);
                };
                for (int i=0;i<3;++i) run();check(cudaStreamSynchronize(stream));
                cudaEvent_t first,last;check(cudaEventCreate(&first));check(cudaEventCreate(&last));
                check(cudaEventRecord(first,stream));for (int i=0;i<10;++i) run();check(cudaEventRecord(last,stream));
                check(cudaEventSynchronize(last));float ms=0;check(cudaEventElapsedTime(&ms,first,last));ms/=10;
                const auto actual=y.floats(1);const uint16_t bits=0x2323;
                const float value=ggml_fp16_to_fp32(bits),expected=value*value*K;
                const float relative=std::abs(actual[0]-expected)/expected;
                if (!std::isfinite(actual[0]) || relative>0.03f) throw std::runtime_error("mixed GEMM micro reference failed");
                if (comma) std::cout << ",";
                comma=true;
                std::cout << "{\"rows\":" << rows << ",\"single\":" << (single ? "true" : "false")
                          << ",\"gpu_ms\":" << ms << ",\"TFLOP_s\":" << (2.0*rows*N*K*batches)/(ms*1e9)
                          << ",\"relative_error\":" << relative << "}";
                check(cudaEventDestroy(first));check(cudaEventDestroy(last));
            }
            std::cout << "]\n";check(cudaStreamDestroy(stream));return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--protocol-size") { std::cout << strata::core::mixed_layout().bytes << '\n'; return 0; }
        if (argc == 2 && std::string(argv[1]) == "--protocol-layout") { strata::core::mixed_layout_report(std::cout); return 0; }
        if (argc != 6) throw std::invalid_argument("usage: strata-glm-gpu-worker MODEL MEMORY_FD SOCKET_FD CACHE_MIB GPU_BUDGET_MIB");
        const strata_env_knob cache_knobs[] = {
            {"STRATA_GLM_MIXED_ADAPT_MIN_ROUTES", STRATA_ENV_FLOAT, 0, 1000000, 0},
            {"STRATA_GLM_MIXED_ADAPT_MIN_GAIN_PCT", STRATA_ENV_FLOAT, 0, 1000, 0},
        };
        char cache_knob_error[512];
        if (strata_env_init(cache_knobs, 2, cache_knob_error, sizeof cache_knob_error)!=STRATA_ENV_OK)
            throw std::invalid_argument(cache_knob_error);
        const float adapt_min_routes=float(strata_env_get(0));
        const double adapt_min_gain=strata_env_get(1);
        const bool cache_hysteresis=adapt_min_routes>0 || adapt_min_gain>0;
        strata::core::MixedMapping transport(std::stoi(argv[2]), std::stoi(argv[3]));
        std::unique_ptr<HostMappingRegistration> registration;
        if (strata_env_truthy_one("STRATA_GLM_MIXED_REGISTER"))
            registration=std::make_unique<HostMappingRegistration>(transport.buffer,transport.layout.bytes,"worker");
        const bool registered_transport=registration && registration->registered;
        strata::core::ModelArtifact artifact(argv[1]);
        const auto &m = artifact.descriptor();
        if (m.hidden != 4096 || m.top_k != 8 || m.experts > 288 ||
            std::any_of(m.layers.begin(),m.layers.end(),[](const auto &l){return l.ffn==strata::core::FfnKind::Moe && l.intermediate!=2048;})) throw std::runtime_error("mixed worker unsupported geometry");
        auto mib_bytes = [](const char *text) {
            const std::string value(text); size_t used = 0;
            if (value.empty() || value.front() == '-') throw std::invalid_argument("mixed worker budget must be positive MiB");
            const auto n = std::stoull(value, &used);
            if (used != value.size() || !n || n > SIZE_MAX / MiB)
                throw std::invalid_argument("mixed worker budget must be positive MiB without overflow");
            return size_t(n) * MiB;
        };
        size_t budget = mib_bytes(argv[4]);
        const bool check_rows = strata_env_truthy_one("STRATA_GLM_MIXED_CHECK_ROWS");
        std::unique_ptr<cpu::ExpertPool> checker;
        if (check_rows) checker=std::make_unique<cpu::ExpertPool>(2,false,true);
        const size_t requested_gpu_bytes = mib_bytes(argv[5]);
        size_t initial_free = 0, physical_bytes = 0; check(cudaMemGetInfo(&initial_free, &physical_bytes));
        const size_t effective_gpu_bytes = std::min(requested_gpu_bytes, physical_bytes);
        if (effective_gpu_bytes <= Device::budget_reserve() || budget > effective_gpu_bytes - Device::budget_reserve())
            throw std::invalid_argument("mixed worker cache exceeds the GPU budget after the runtime reserve");
        Device::limit() = effective_gpu_bytes - Device::budget_reserve();
        std::cerr << "MIXED_BUDGET requested_gpu_MiB=" << requested_gpu_bytes/double(MiB)
                  << " effective_gpu_MiB=" << effective_gpu_bytes/double(MiB)
                  << " requested_cache_MiB=" << budget/double(MiB) << '\n';
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        ResidentExecutor executor;
        strata::prefill::Gemm gemm; std::string error;
        if (!gemm.init(stream, 8LL * 4096 * 2048, error)) throw std::runtime_error(error);
        const char *tile_value=std::getenv("STRATA_GLM_MIXED_PREFILL_TILE");
        const int panel_rows=tile_value ? std::stoi(tile_value) : 128;
        if (panel_rows!=128 && panel_rows!=256 && panel_rows!=512) throw std::invalid_argument("mixed prefill tile must be 128, 256 or 512");
        const size_t panel_capacity=size_t(panel_rows)*4;
        const bool profile_prefill=strata_env_truthy_one("STRATA_GLM_MIXED_PROFILE");
        const bool async_panels=strata_env_truthy_one("STRATA_GLM_MIXED_ASYNC_PANELS");
        const bool single_gemm=strata_env_truthy_one("STRATA_GLM_MIXED_SINGLE_GEMM");
        Device px(STRATA_MIXED_TOKENS * 4096ULL * sizeof(float)), hx(panel_capacity * 4096 * 2), gu(panel_capacity * 4096 * sizeof(float));
        Device hidden(panel_capacity * 2048 * sizeof(float)), hh(panel_capacity * 2048 * 2), py(panel_capacity * 4096 * sizeof(float));
        Pinned input(STRATA_MIXED_TOKENS * 4096ULL * sizeof(float)), output(panel_capacity * 4096 * sizeof(float));
        Device sources(STRATA_MIXED_ROUTES*sizeof(int)), bounds(5*sizeof(int)), down_dense(4ULL*4096*2048*2);
        Pinned source_host(STRATA_MIXED_ROUTES*sizeof(int)), bounds_host(5*sizeof(int));
        std::unique_ptr<Device> partial_result, partial_weights, partial_dest;
        std::unique_ptr<Device> partial_half;
        const bool half_partial=strata_env_truthy_one("STRATA_GLM_MIXED_HALF_PARTIAL");
        std::unique_ptr<Pinned> partial_weight_host, partial_dest_host;
        if (strata_env_truthy_one("STRATA_GLM_MIXED_PARTIAL_PREFILL")) {
            partial_result=std::make_unique<Device>(STRATA_MIXED_TOKENS*4096ULL*sizeof(float));
            partial_weights=std::make_unique<Device>(STRATA_MIXED_ROUTES*sizeof(float));
            partial_dest=std::make_unique<Device>(STRATA_MIXED_ROUTES*sizeof(int));
            partial_weight_host=std::make_unique<Pinned>(partial_weights->bytes);
            partial_dest_host=std::make_unique<Pinned>(partial_dest->bytes);
        }
        Device decode_sum(8*4096*sizeof(float)),decode_weights(64*sizeof(float));
        Pinned decode_weight_host(64*sizeof(float)),decode_sum_host(8*4096*sizeof(float));
        struct CopyReady {
            cudaEvent_t event=nullptr;bool pending=true;
            CopyReady() { check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming)); }
            ~CopyReady() { if (event) (void)cudaEventDestroy(event); }
        };
        struct Entry { std::unique_ptr<Device> device; uint64_t used; size_t charged; std::shared_ptr<CopyReady> ready; };
        std::map<std::pair<int,int>, Entry> cache;
        size_t occupied = 0, upload_bytes=0; uint64_t clock = 0, previous = 0, decode_rows=0, prefill_rows=0;
        const char *copy_value=std::getenv("STRATA_GLM_MIXED_UPLOAD_THREADS");
        const int copy_threads=copy_value ? std::stoi(copy_value) : 0;
        if (copy_threads!=0 && copy_threads!=2 && copy_threads!=4) throw std::invalid_argument("mixed upload threads must be 0, 2 or 4");
        CopyPool copy_pool(copy_threads);
        struct UploadSlot {
            std::unique_ptr<Pinned> host;
            std::future<void> copied;
            Device *destination=nullptr;
            size_t bytes=0;
            cudaEvent_t ready=nullptr;
            bool busy=false;
            std::shared_ptr<CopyReady> completion;
            ~UploadSlot() { if (copied.valid()) copied.wait(); if (ready) cudaEventDestroy(ready); }
        };
        std::array<UploadSlot,4> uploads;
        struct CopyChannel {
            cudaStream_t stream=nullptr;
            ~CopyChannel() { if (stream) { (void)cudaStreamSynchronize(stream);(void)cudaStreamDestroy(stream); } }
        } copy_channel;
        if (strata_env_truthy_one("STRATA_GLM_MIXED_UPLOAD_STREAM"))
            check(cudaStreamCreateWithFlags(&copy_channel.stream,cudaStreamNonBlocking));
        auto drain_upload=[&](UploadSlot &slot) {
            if (!slot.copied.valid()) return;
            slot.copied.get();
            const auto destination_stream=copy_channel.stream ? copy_channel.stream : stream;
            slot.destination->put_async(slot.host->p,slot.bytes,destination_stream);upload_bytes+=slot.bytes;
            if (slot.completion) check(cudaEventRecord(slot.completion->event,destination_stream));
            check(cudaEventRecord(slot.ready,destination_stream));slot.busy=true;
        };
        size_t upload_cursor=0;
        size_t free_bytes=0,total_bytes=0; check(cudaMemGetInfo(&free_bytes,&total_bytes));
        const size_t reserve=std::max(Device::physical_reserve(),Device::budget_reserve())+512*MiB;
        if (free_bytes<=reserve) throw std::runtime_error("mixed worker has no expert cache headroom");
        budget=std::min(budget,free_bytes-reserve);
        std::map<size_t,size_t> charges;
        size_t all_layers_bytes = 0;
        for (size_t l = 0; l < m.layers.size(); ++l) {
            if (m.layers[l].ffn != strata::core::FfnKind::Moe) continue;
            const auto p = "blk." + std::to_string(l) + ".";
            const size_t bytes=(artifact.at(p+"ffn_gate_exps.weight").bytes + artifact.at(p+"ffn_up_exps.weight").bytes + artifact.at(p+"ffn_down_exps.weight").bytes) / m.experts;
            if (!charges.count(bytes)) {
                size_t before=0,after=0,total=0; check(cudaMemGetInfo(&before,&total));
                { Device probe(bytes); check(cudaMemGetInfo(&after,&total)); charges[bytes]=std::max(bytes,before>=after ? before-after : size_t(0)); }
            }
            all_layers_bytes += charges.at(bytes);
        }
        size_t largest=0; for (const auto &[bytes,charge]:charges) largest=std::max(largest,bytes);
        for (auto &slot: uploads) {
            slot.host=std::make_unique<Pinned>(largest);
            check(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming));
        }
        const size_t per_layer = std::min(size_t(m.experts), budget / std::max(size_t(1), all_layers_bytes));
        if (!per_layer || m.layers.size() > 64) throw std::runtime_error("mixed worker cache budget too small");
        std::memset(transport.buffer->worker_owned, 0, sizeof transport.buffer->worker_owned);
        // Stable ownership, later replaced by measured prior ordering. Primary ownership always wins.
        for (size_t l = 0; l < m.layers.size(); ++l)
            for (size_t e = 0; e < per_layer; ++e) transport.buffer->worker_owned[l*288+e] = 1;
        std::vector<std::vector<int>> ranked(m.layers.size(), std::vector<int>(m.experts));
        for (auto &order: ranked) std::iota(order.begin(), order.end(), 0);
        std::vector<std::vector<double>> prior_share(m.layers.size(),std::vector<double>(m.experts,1./m.experts));
        uint64_t adapt_updates=0,adapt_replacements=0,adapt_upload_bytes=0;
        const bool adapt_metrics=strata_env_truthy_one("STRATA_GLM_MIXED_ADAPT_METRICS");
        const bool adapt_cache=strata_env_truthy_one("STRATA_GLM_MIXED_ADAPT_CACHE");
        const bool adapt_after_rows=adapt_cache && strata_env_truthy_one("STRATA_GLM_MIXED_ADAPT_AFTER_ROWS");
        const bool reuse_adaptive=strata_env_truthy_one("STRATA_GLM_MIXED_REUSE_ADAPT_SLOTS");
        if (cache_hysteresis && !adapt_cache)
            throw std::invalid_argument("mixed cache admission thresholds require STRATA_GLM_MIXED_ADAPT_CACHE=1");
        if (cache_hysteresis)
            std::cerr << "MIXED_CACHE_ADMISSION min_routes=" << adapt_min_routes
                      << " min_gain_pct=" << adapt_min_gain << " max_replacements=4\n";
        std::vector<std::vector<int>> pending_owners(m.layers.size());
        const bool cpu_ggml_reference=strata_env_equals("STRATA_GLM_MIXED_CPU_REFERENCE","ggml");
        double maximum_cpu_float_error=0,maximum_cpu_ggml_error=0;
        const bool prompt_prior=strata_env_truthy_one("STRATA_GLM_MIXED_PROMPT_PRIOR") || adapt_cache;
        if (const char *prior = std::getenv("STRATA_GLM_EXPERT_PRIOR")) {
            const auto json = strata::artifact::read_json(prior);
            for (const auto &projection: json.at("projections").array) {
                const auto &name=projection.at("name").string();
                if (!name.starts_with("blk.") || !name.ends_with(".gu")) continue;
                const size_t l=std::stoul(name.substr(4,name.size()-7));
                const auto &counts=projection.at("counts").array;
                if (l>=ranked.size() || counts.size()!=size_t(m.experts)) throw std::runtime_error("mixed prior geometry");
                if (prompt_prior) {
                    double total=0;
                    for (int e=0;e<m.experts;++e) {
                        const auto count=counts[e].integer();
                        // Json::integer rejects negative and malformed counts.
                        total+=count;
                    }
                    if (total) for (int e=0;e<m.experts;++e) prior_share[l][e]=counts[e].integer()/total;
                }
                std::stable_sort(ranked[l].begin(),ranked[l].end(),[&](int a,int b){ return counts[a].integer()>counts[b].integer(); });
            }
        }
        std::cerr << "MIXED_CACHE budget_MiB=" << budget/double(MiB) << " slots_per_layer=" << per_layer
                  << " reserved_runtime_MiB=" << reserve/double(MiB) << '\n';
        auto load_experts=[&](int layer_id,const k::NativeExpertLayout &layout,
                              const strata::core::ArtifactTensor &G,const strata::core::ArtifactTensor &U,
                              const strata::core::ArtifactTensor &D,const std::map<int,std::vector<int>> &hits,
                              std::vector<std::unique_ptr<Device>> *recycled=nullptr) {
            if (hits.size()*charges.at(layout.bytes)>budget) throw std::runtime_error("mixed worker preload exceeds cache budget");
            for (const auto &[e, indices] : hits) {
                auto key = std::make_pair(int(layer_id), e);
                if (!cache.count(key)) {
                    std::unique_ptr<Device> weights;
                    if (recycled && !recycled->empty()) {
                        weights=std::move(recycled->back());recycled->pop_back();
                        if (weights->bytes!=layout.bytes) throw std::runtime_error("mixed recycled slot geometry mismatch");
                    }
                    size_t available=0,total=0; check(cudaMemGetInfo(&available,&total));
                    if (!weights && (occupied+charges.at(layout.bytes)>budget || available<reserve+charges.at(layout.bytes))) {
                        auto victim=cache.end();
                        for (auto it=cache.begin(); it!=cache.end(); ++it) {
                            if (it->second.device->bytes!=layout.bytes || (it->first.first==int(layer_id) && hits.count(it->first.second))) continue;
                            if (victim==cache.end() || it->second.used<victim->second.used) victim=it;
                        }
                        if (victim!=cache.end()) {
                            occupied-=victim->second.charged;
                            weights=std::move(victim->second.device); cache.erase(victim);
                        }
                    }
                    auto room = [&] {
                        size_t available=0,total=0; check(cudaMemGetInfo(&available,&total));
                        return occupied+charges.at(layout.bytes)<=budget && available>=reserve+charges.at(layout.bytes);
                    };
                    while (!weights && !room()) {
                        auto victim = cache.end();
                        for (auto it = cache.begin(); it != cache.end(); ++it) {
                            if (it->first.first == int(layer_id) && hits.count(it->first.second)) continue;
                            if (victim == cache.end() || it->second.used < victim->second.used) victim = it;
                        }
                        if (victim == cache.end()) throw std::runtime_error("mixed worker no eviction candidate");
                        occupied -= victim->second.charged; cache.erase(victim);
                    }
                    if (!weights) weights=std::make_unique<Device>(layout.bytes);
                    auto &upload=uploads[upload_cursor++ % uploads.size()];
                    drain_upload(upload);
                    if (upload.busy) check(cudaEventSynchronize(upload.ready));
                    auto *host=static_cast<char *>(upload.host->p);
                    const auto *gate_source=G.data()+size_t(e)*layout.up_off;
                    const auto *up_source=U.data()+size_t(e)*layout.up_off;
                    const auto *down_source=D.data()+size_t(e)*(layout.bytes-layout.down_off);
                    upload.destination=weights.get();upload.bytes=layout.bytes;
                    auto completion=copy_channel.stream ? std::make_shared<CopyReady>() : nullptr;
                    upload.completion=completion;
                    upload.copied=copy_pool.submit([=] {
                        std::memcpy(host,gate_source,layout.up_off);
                        std::memcpy(host+layout.up_off,up_source,layout.up_off);
                        std::memcpy(host+layout.down_off,down_source,layout.bytes-layout.down_off);
                    });
                    occupied += charges.at(layout.bytes); cache.emplace(key, Entry{std::move(weights), ++clock, charges.at(layout.bytes),std::move(completion)});
                }
                cache.at(key).used = ++clock;
            }
            for (auto &slot:uploads) drain_upload(slot);
        };
        const bool prefetch_next=strata_env_truthy_one("STRATA_GLM_MIXED_PREFETCH_NEXT");
        std::map<int,std::vector<int>> next_experts;
        if (prefetch_next) {
            if (m.experts%16) throw std::invalid_argument("mixed prefetch requires whole 16-expert groups");
            const int groups=int(m.experts)/16;
            const char *value=std::getenv("STRATA_GLM_MIXED_PRIMARY_FRACTION");
            size_t used=0;const double fraction=value ? std::stod(value,&used) : 0.5;
            if (!std::isfinite(fraction) || fraction<=0 || fraction>=1 || (value && value[used]))
                throw std::invalid_argument("mixed prefetch fraction must be in (0,1)");
            const int primary_groups=std::clamp(int(std::lround(fraction*groups)),1,groups-1);
            for (int e=0;e<m.experts;++e)
                if (strata::prefill::owns_expert_group(e/16,1,2,primary_groups,groups)) next_experts[e]={};
        }
        const bool profile_decode=strata_env_truthy_one("STRATA_GLM_MIXED_PROFILE_DECODE");
        const bool scan_once=strata_env_truthy_one("STRATA_GLM_MIXED_SCAN_ONCE");
        std::vector<std::array<uint8_t,288>> scanned_masks(m.layers.size());
        std::vector<uint8_t> scanned_valid(m.layers.size());
        double decode_prologue_ms=0, decode_body_ms=0, decode_idle_ms=0; uint64_t decode_requests=0;
        std::chrono::steady_clock::time_point last_reply{};
        std::chrono::steady_clock::time_point prefill_last_end{};
        uint64_t ready = STRATA_MIXED_MAGIC; strata::core::mixed_signal(transport.socket, ready, true);
        for (;;) {
            uint64_t sequence = 0; strata::core::mixed_signal(transport.socket, sequence, false, -1);
            if (!sequence) {
                std::ostringstream report;
                if (profile_decode) report << "MIXED_WORKER_DECODE_PROFILE requests=" << decode_requests << " prologue_ms=" << decode_prologue_ms
                    << " body_ms=" << decode_body_ms << " idle_ms=" << decode_idle_ms << '\n';
                report << "MIXED_WORKER_SUMMARY requests=" << previous << " decode_rows=" << decode_rows
                          << " prefill_rows=" << prefill_rows << " upload_MiB=" << upload_bytes/double(MiB)
                       << " adapt_updates=" << adapt_updates << " adapt_replacements=" << adapt_replacements
                       << " max_cpu_float_error=" << maximum_cpu_float_error
                       << " max_cpu_ggml_error=" << maximum_cpu_ggml_error
                       << " worker_graphs=" << executor.graphs.size()+executor.device_graphs.size();
                if (adapt_metrics) report << " adapt_upload_MiB=" << adapt_upload_bytes/double(MiB);
                report << '\n';
                std::cerr << report.str();
                check(cudaStreamDestroy(stream)); return 0;
            }
            auto &b = *transport.buffer;
            if (sequence <= previous || b.sequence != sequence || b.generation != sequence ||
                !transport.valid(m.layers.size(), m.experts)) throw std::runtime_error("mixed worker invalid request");
            previous = sequence;
            const auto request_start = std::chrono::steady_clock::now();
            if (profile_decode && last_reply.time_since_epoch().count())
                decode_idle_ms += std::chrono::duration<double,std::milli>(request_start-last_reply).count();
            const std::string p = "blk." + std::to_string(b.layer) + ".";
            const auto &G = artifact.at(p + "ffn_gate_exps.weight");
            const auto &U = artifact.at(p + "ffn_up_exps.weight");
            const auto &D = artifact.at(p + "ffn_down_exps.weight");
            auto layout = k::native_expert_layout(G.tensor->type, D.tensor->type, m.hidden, m.layers.at(b.layer).intermediate);
            layout.swiglu_limit = b.swiglu_limit; layout.gate_skip = b.gate_skip;
            if (b.canon && !((layout.gu_type==10 || layout.gu_type==11) && (layout.d_type==10 || layout.d_type==11)))
                throw std::runtime_error("mixed worker canonical type mismatch");
            if (!k::native_expert_supported(G.tensor->type, D.tensor->type, m.hidden, m.layers.at(b.layer).intermediate)) throw std::runtime_error("mixed worker unsupported expert types");
            // Evict primary-owned entries only when this layer's primary mask changed since the last scan
            // (STRATA_GLM_MIXED_SCAN_ONCE=1, opt-in); the scan visits every cached expert.
            if (scan_once && b.prefill != 0 && b.prefill != STRATA_MIXED_DECODE_PARTIAL) std::fill(scanned_valid.begin(), scanned_valid.end(), 0);
            const bool scan = !scan_once || b.layer >= scanned_masks.size() || !scanned_valid[b.layer] ||
                              std::memcmp(scanned_masks[b.layer].data(), b.primary_owned, 288);
            if (scan) {
                for (auto it = cache.begin(); it != cache.end();) {
                    if (it->first.first == int(b.layer) && b.primary_owned[it->first.second]) {
                        occupied -= it->second.charged; it = cache.erase(it);
                    } else ++it;
                }
                if (scan_once && b.layer < scanned_masks.size()) { std::memcpy(scanned_masks[b.layer].data(), b.primary_owned, 288); scanned_valid[b.layer] = 1; }
            }
            std::map<int, std::vector<int>> hits;
            for (uint32_t i = 0; i < b.routes; ++i) hits[b.expert[i]].push_back(b.destination[i]);
            const auto prologue_end = std::chrono::steady_clock::now();
            if (b.prefill == 2) {
                std::array<uint8_t,288> previous_owned{};
                std::memcpy(previous_owned.data(),b.worker_owned+b.layer*288,288);
                std::memset(b.worker_owned+b.layer*288, 0, 288);
                auto order=ranked.at(b.layer);
                if (prompt_prior) {
                    if (!strata_mixed_prompt_valid(&b,m.experts)) throw std::runtime_error("mixed prompt counts invalid");
                    const double prior_routes=adapt_cache && b.chunk_tag==1 ? 512. : 2048.;
                    std::stable_sort(order.begin(),order.end(),[&](int a,int c) {
                        return double(b.route_weight[a])+prior_routes*prior_share[b.layer][a] >
                               double(b.route_weight[c])+prior_routes*prior_share[b.layer][c];
                    });
                    if (adapt_cache && b.chunk_tag==1) {
                        // Retain the best old owners, allowing at most four
                        // replacements unless primary ownership displaced them.
                        const size_t keep=per_layer>4 ? per_layer-4 : 0;
                        for (int e:order) {
                            if (hits.size()==keep) break;
                            if (previous_owned[e] && !b.primary_owned[e]) hits[e]={};
                        }
                    }
                }
                for (int e: order) {
                    if (hits.size()==per_layer) break;
                    if (b.primary_owned[e]) continue;
                    hits[e] = {};
                }
                if (cache_hysteresis && adapt_cache && b.chunk_tag==1) {
                    std::array<double,288> score{};
                    std::array<uint8_t,288> admitted{};
                    for (int e=0;e<m.experts;++e)
                        score[e]=double(b.route_weight[e])+512.*prior_share[b.layer][e];
                    if (!strata_mixed_select_cache(m.experts,uint32_t(per_layer),order.data(),score.data(),
                            b.route_weight,previous_owned.data(),b.primary_owned,4,
                            adapt_min_routes,adapt_min_gain,admitted.data()))
                        throw std::runtime_error("mixed cache admission policy rejected invalid input");
                    hits.clear();
                    for (int e=0;e<m.experts;++e) if (admitted[e]) hits[e]={};
                }
                for (const auto &[e,unused]:hits) b.worker_owned[b.layer*288+e]=1;
                if (adapt_after_rows && b.chunk_tag==1) {
                    // This request's router already used the old ownership.
                    // Execute its rows first, then upload the next cache set.
                    auto &pending=pending_owners[b.layer];pending.clear();
                    for (const auto &[e,unused]:hits) pending.push_back(e);
                    hits.clear();
                    std::memset(b.worker_owned+b.layer*288,0,288);
                    for (int e=0;e<m.experts;++e) if (previous_owned[e] && !b.primary_owned[e]) {
                        hits[e]={};b.worker_owned[b.layer*288+e]=1;
                    }
                } else if (adapt_cache && b.chunk_tag==1) {
                    ++adapt_updates;
                    for (int e=0;e<m.experts;++e)
                        adapt_replacements+=b.worker_owned[b.layer*288+e] && !previous_owned[e];
                } else pending_owners[b.layer].clear();
            }
            if (b.prefill==2 && strata_env_truthy_one("STRATA_GLM_MIXED_STABLE_OWNERSHIP")) {
                // Extra prefill entries must not displace declared decode
                // residents elsewhere while the complementary cache loads.
                for (auto it=cache.begin();it!=cache.end();) {
                    if (it->first.first==int(b.layer) && !hits.count(it->first.second)) {
                        occupied-=it->second.charged;it=cache.erase(it);
                    } else ++it;
                }
            }
            size_t required = hits.size() * charges.at(layout.bytes);
            if (required > budget) throw std::runtime_error("mixed worker request exceeds cache budget");
            const uint64_t uploads_before_request=upload_bytes;
            load_experts(b.layer,layout,G,U,D,hits);
            if (adapt_cache && b.prefill==2 && b.chunk_tag==1 && !adapt_after_rows)
                adapt_upload_bytes+=upload_bytes-uploads_before_request;
            if (b.prefill!=2) for (const auto &[e,unused]:hits) {
                auto &ready=cache.at({int(b.layer),e}).ready;
                if (ready && ready->pending) {
                    // Every subsequent use stays on this execution stream.
                    check(cudaStreamWaitEvent(stream,ready->event,0));ready->pending=false;
                }
            }
            const double cache_prepare_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count();
            double ack_ms=0, products_ms=0;
            uint32_t panels=0;
            uint32_t chunk_counter=0;
            std::array<float,4*4096> checked_output{};
            const bool decode_partial=b.prefill==STRATA_MIXED_DECODE_PARTIAL;
            std::vector<float> checked_decode;
            const bool partial=b.prefill==STRATA_MIXED_PREFILL_PARTIAL;
            const bool is_prefill=b.prefill==1 || partial;
            if (is_prefill) {
                if (partial && !partial_result) throw std::runtime_error("mixed partial prefill workspace not enabled");
                const char *lossy_min=std::getenv("STRATA_PREFILL_LOSSY_MIN_TOKENS");
                strata::prefill::Gemm::lossy_allowed=int(b.nt)>=(lossy_min ? std::max(1,std::atoi(lossy_min)) : 4096);
                std::array<int, STRATA_MIXED_ROUTES> output_index;
                output_index.fill(-1);
                for (uint32_t i=0; i<b.routes; ++i) output_index[b.destination[i]]=i;
                const bool half_activation=transport.activation_half();
                const size_t activation_bytes=size_t(b.nt)*4096*(half_activation ? sizeof(uint16_t) : sizeof(float));
                if (registered_transport) px.put_async(b.activation,activation_bytes,stream);
                else { std::memcpy(input.p,b.activation,activation_bytes); px.put_async(input.p,activation_bytes,stream); }
                if (partial) {
                    check(cudaMemsetAsync(partial_result->p,0,size_t(b.nt)*4096*sizeof(float),stream));
                    std::memcpy(partial_weight_host->p,b.route_weight,size_t(b.nt)*8*sizeof(float));
                    partial_weights->put_async(partial_weight_host->p,size_t(b.nt)*8*sizeof(float),stream);
                }
                std::map<int,std::vector<int>> buckets;
                for (const auto &[e,indices]: hits)
                    buckets[(int(indices.size())+31)/32*32].push_back(e);
                auto flush_chunk=[&] {
                    if (!b.chunk_rows) return;
                    b.chunk_tag=++chunk_counter;
                    strata_mixed_message packet{sequence,b.chunk_tag,STRATA_MIXED_CHUNK};
                    const auto start=std::chrono::steady_clock::now();
                    strata::core::mixed_packet(transport.socket,packet,true);
                    strata::core::mixed_packet(transport.socket,packet,false);
                    ack_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    if (packet.sequence!=sequence || packet.tag!=chunk_counter || packet.kind!=STRATA_MIXED_ACK)
                        throw std::runtime_error("mixed prefill stale acknowledgement");
                    b.chunk_rows=0;
                };
                const int F=layout.n_ff;
                for (const auto &[bucket, experts]: buckets) {
                    for (size_t first=0; first<experts.size(); first+=4) {
                        const int batches=std::min(size_t(4),experts.size()-first);
                        auto *starts=static_cast<int *>(bounds_host.p);
                        auto *tokens=static_cast<int *>(source_host.p);
                        starts[0]=0;
                        for (int group=0; group<batches; ++group) {
                            const auto &indices=hits.at(experts[first+group]);
                            starts[group+1]=starts[group]+int(indices.size());
                            for (size_t row=0;row<indices.size();++row) {
                                tokens[starts[group]+row]=indices[row]/8;
                                if (partial) static_cast<int *>(partial_dest_host->p)[starts[group]+row]=indices[row];
                            }
                            void *w=cache.at({int(b.layer),experts[first+group]}).device->p;
                            k::dequant_f16(layout.gu_type,w,0,2*F,4096,gemm.scratch()+size_t(group)*2*F*4096,stream);
                            k::dequant_f16(layout.d_type,static_cast<char *>(w)+layout.down_off,0,4096,F,
                                          static_cast<uint16_t *>(down_dense.p)+size_t(group)*4096*F,stream);
                        }
                        bounds.put_async(starts,size_t(batches+1)*sizeof(int),stream);
                        sources.put_async(tokens,size_t(starts[batches])*sizeof(int),stream);
                        if (partial) partial_dest->put_async(partial_dest_host->p,size_t(starts[batches])*sizeof(int),stream);
                        for (int offset=0; offset<bucket; offset+=panel_rows) {
                            const auto product_start=std::chrono::steady_clock::now();
                            ++panels;
                            const int rows=std::min(panel_rows,bucket-offset),all_rows=rows*batches;
                            if (half_activation)
                                k::glm_gather_expert_f16_from_half(static_cast<const uint16_t *>(px.p),static_cast<uint16_t *>(hx.p),static_cast<int *>(bounds.p),
                                    static_cast<int *>(sources.p),0,batches,offset,rows,4096,stream);
                            else k::glm_gather_expert_f16(px.f(),static_cast<uint16_t *>(hx.p),static_cast<int *>(bounds.p),
                                static_cast<int *>(sources.p),0,batches,offset,rows,4096,stream);
                            if (single_gemm) for (int group=0;group<batches;++group)
                                gemm.f16(static_cast<uint16_t *>(hx.p)+size_t(group)*rows*4096,
                                         gemm.scratch()+size_t(group)*2*F*4096,gu.f()+size_t(group)*rows*2*F,rows,2*F,4096);
                            else gemm.f16_batched(static_cast<uint16_t *>(hx.p),gemm.scratch(),gu.f(),rows,2*F,4096,batches);
                            strata::prefill::mmq::swiglu(gu.f(),hidden.f(),all_rows,F,false,stream,layout.swiglu_limit);
                            k::glm_f16(hidden.f(),static_cast<uint16_t *>(hh.p),int64_t(all_rows)*F,stream);
                            if (single_gemm) for (int group=0;group<batches;++group)
                                gemm.f16(static_cast<uint16_t *>(hh.p)+size_t(group)*rows*F,
                                         static_cast<uint16_t *>(down_dense.p)+size_t(group)*4096*F,
                                         py.f()+size_t(group)*rows*4096,rows,4096,F);
                            else gemm.f16_batched(static_cast<uint16_t *>(hh.p),static_cast<uint16_t *>(down_dense.p),py.f(),rows,4096,F,batches);
                            if (partial) k::glm_scatter_expert_weighted(py.f(),partial_result->f(),partial_weights->f(),
                                static_cast<int *>(bounds.p),static_cast<int *>(partial_dest->p),0,batches,offset,rows,4096,8,stream);
                            if (!partial || check_rows)
                                check(cudaMemcpyAsync(output.p,py.p,size_t(all_rows)*4096*sizeof(float),cudaMemcpyDeviceToHost,stream));
                            if (!partial || check_rows || !async_panels) check(cudaStreamSynchronize(stream));
                            products_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-product_start).count();
                            if (partial && !check_rows) continue;
                            b.chunk_rows=0;
                            for (int group=0; group<batches; ++group) {
                                const auto &indices=hits.at(experts[first+group]);
                                for (int row=0; row<rows && offset+row<int(indices.size()); ++row) {
                                    const int destination=indices[offset+row],j=output_index[destination];
                                    if (j<0) throw std::runtime_error("mixed prefill output destination missing");
                                    const auto *value=static_cast<float *>(output.p)+size_t(group*rows+row)*4096;
                                    if (j<4) std::memcpy(checked_output.data()+size_t(j)*4096,value,4096*sizeof(float));
                                    if (partial) continue;
                                    transport.destinations()[b.chunk_rows]=destination;
                                    std::memcpy(transport.output()+size_t(b.chunk_rows++)*4096,value,4096*sizeof(float));
                                    if (b.chunk_rows==transport.layout.output_rows) flush_chunk();
                                }
                            }
                            flush_chunk();
                        }
                        // Pinned bounds and token lists are reused by the next
                        // expert group. All panels share one stream, so their
                        // device workspaces can be reused without a panel wait.
                        if (partial && !check_rows && async_panels) {
                            const auto wait_start=std::chrono::steady_clock::now();
                            check(cudaStreamSynchronize(stream));
                            products_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wait_start).count();
                        }
                    }
                }
                if (partial) {
                    // Activation staging is free after its upload; reuse its
                    // bounded pinned buffer for one FP32 partial vector/token.
                    // STRATA_GLM_MIXED_HALF_PARTIAL=1 (opt-in, lossy; registered protocol 5): the partial returns as
                    // FP16, one chunk, halving the reply bytes on the link.
                    if (half_partial) {
                        if (!registered_transport || b.nt>transport.layout.output_rows)
                            throw std::runtime_error("mixed half partial needs a registered protocol 5 mapping");
                        if (!partial_half) partial_half=std::make_unique<Device>(STRATA_MIXED_TOKENS*4096ULL*sizeof(uint16_t));
                        k::glm_f16(partial_result->f(),static_cast<uint16_t *>(partial_half->p),int64_t(b.nt)*4096,stream);
                        check(cudaMemcpyAsync(transport.output(),partial_half->p,size_t(b.nt)*4096*sizeof(uint16_t),cudaMemcpyDeviceToHost,stream));
                        check(cudaStreamSynchronize(stream));
                        b.chunk_rows=b.nt;
                        for (uint32_t row=0;row<b.nt;++row) transport.destinations()[row]=row;
                        flush_chunk();
                    }
                    if (!registered_transport)
                        check(cudaMemcpyAsync(input.p,partial_result->p,size_t(b.nt)*4096*sizeof(float),cudaMemcpyDeviceToHost,stream));
                    if (!half_partial) check(cudaStreamSynchronize(stream));
                    for (uint32_t start=0;start<b.nt && !half_partial;start+=transport.layout.output_rows) {
                        b.chunk_rows=std::min(transport.layout.output_rows,b.nt-start);
                        if (registered_transport) {
                            check(cudaMemcpyAsync(transport.output(),partial_result->f(size_t(start)*4096),
                                size_t(b.chunk_rows)*4096*sizeof(float),cudaMemcpyDeviceToHost,stream));
                            check(cudaStreamSynchronize(stream));
                        } else std::memcpy(transport.output(),static_cast<float *>(input.p)+size_t(start)*4096,
                            size_t(b.chunk_rows)*4096*sizeof(float));
                        for (uint32_t row=0;row<b.chunk_rows;++row) transport.destinations()[row]=start+row;
                        flush_chunk();
                    }
                }
            } else if (b.prefill == 0 || decode_partial) {
            const bool fused_return=decode_partial && strata_env_truthy_one("STRATA_GLM_MIXED_FUSED_RETURN");
            executor.run(layout, hits, [&](int e) { return cache.at({int(b.layer),e}).device->p; }, b.activation, b.nt, stream,
                         strata_env_truthy_one("STRATA_GLM_MIXED_DECODE_GRAPHS"), b.canon, true, !decode_partial || check_rows,
                         fused_return ? b.route_weight : nullptr);
            check(cudaStreamSynchronize(stream));
            if (decode_partial) {
                if (check_rows) {
                    checked_decode.resize(size_t(b.routes)*4096);
                    for (uint32_t i=0;i<b.routes;++i) std::memcpy(checked_decode.data()+size_t(i)*4096,
                        static_cast<float *>(executor.host_output->p)+size_t(b.destination[i])*4096,4096*sizeof(float));
                }
                if (fused_return) {
                    std::memcpy(transport.output(),executor.host_partial_sum->p,size_t(b.nt)*4096*sizeof(float));
                } else {
                    std::memcpy(decode_weight_host.p,b.route_weight,size_t(b.nt)*8*sizeof(float));
                    decode_weights.put_async(decode_weight_host.p,size_t(b.nt)*8*sizeof(float),stream);
                    k::glm_sum_selected_rows(executor.output->f(),decode_weights.f(),decode_sum.f(),4096,8,b.nt,stream);
                    check(cudaMemcpyAsync(decode_sum_host.p,decode_sum.p,size_t(b.nt)*4096*sizeof(float),cudaMemcpyDeviceToHost,stream));
                    check(cudaStreamSynchronize(stream));
                    std::memcpy(transport.output(),decode_sum_host.p,size_t(b.nt)*4096*sizeof(float));
                }
                if (check_rows) {
                    const float *rows=static_cast<float *>(executor.host_output->p);
                    for (uint32_t t=0;t<b.nt;++t) for (int h=0;h<4096;++h) {
                        float expected=0;
                        for (int r=7;r>=0;--r) {
                            const float weight=b.route_weight[t*8+r];
                            if (weight!=0) expected=std::fma(rows[(size_t(t)*8+r)*4096+h],weight,expected);
                        }
                        if (std::memcmp(&expected,transport.output()+size_t(t)*4096+h,sizeof(float)))
                            throw std::runtime_error("mixed worker decode partial reduction parity failed layer="+std::to_string(b.layer));
                    }
                }
            } else for (uint32_t i = 0; i < b.routes; ++i)
                std::memcpy(transport.output() + size_t(i) * 4096, static_cast<float *>(executor.host_output->p) + size_t(b.destination[i]) * 4096, 4096 * sizeof(float));
            }
            check(cudaStreamSynchronize(stream));
            const float *check_output=is_prefill ? checked_output.data() : decode_partial ? checked_decode.data() : transport.output();
            if (check_rows && b.prefill != 2) {
                cpu::NativeFmt f; std::string error;
                if (!cpu::native_fmt(G.tensor->type,D.tensor->type,4096,layout.n_ff,f,error)) throw std::runtime_error(error);
                f.canon=b.canon; f.swiglu_limit=b.swiglu_limit; f.gate_skip=b.gate_skip;
                std::vector<uint8_t> quant(f.act_bytes);
                std::vector<float> reference(4096);
                for (uint32_t i=0; i<(is_prefill ? std::min(b.routes,uint32_t(4)) : b.routes); ++i) {
                    cpu::native_quant_act(f,b.activation+size_t(b.destination[i]/8)*4096,quant.data());
                    cpu::ExpertJobMulti job{}; const int e=b.expert[i];
                    job.expert_id=e; job.nt=1; job.nact[0]=quant.data(); job.out[0]=reference.data();
                    job.blob=G.data()+size_t(e)*f.up_off; job.native_up=U.data()+size_t(e)*f.up_off;
                    job.native_down=D.data()+size_t(e)*(f.bytes-f.down_off);
                    checker->run_split_multi_native(f,&job,1);
                    if (b.canon) {
                        if (std::memcmp(reference.data(),check_output+size_t(i)*4096,4096*sizeof(float)))
                            throw std::runtime_error("mixed worker canonical CPU parity failed layer="+std::to_string(b.layer));
                    } else {
                        // Keep native_expert_parity's existing 3% comparisons
                        // against dequantized float weights. CPU and GPU round
                        // activations differently; CPU/GPU relative error is
                        // not that test's qualification threshold. Prefill does not
                        // execute CPU experts, so only its GPU/float gate applies.
                        const int F=layout.n_ff;
                        std::vector<float> gate(size_t(F)*4096),up(gate.size()),down(size_t(F)*4096),hidden(F),exact(4096);
                        const auto *gu=ggml_get_type_traits(static_cast<ggml_type>(layout.gu_type));
                        const auto *dt=ggml_get_type_traits(static_cast<ggml_type>(layout.d_type));
                        for (int r=0;r<F;++r) {
                            gu->to_float(job.blob+size_t(r)*layout.gu_row,gate.data()+size_t(r)*4096,4096);
                            gu->to_float(job.native_up+size_t(r)*layout.gu_row,up.data()+size_t(r)*4096,4096);
                        }
                        for (int r=0;r<4096;++r) dt->to_float(job.native_down+size_t(r)*layout.d_row,down.data()+size_t(r)*F,F);
                        const float *x=b.activation+size_t(b.destination[i]/8)*4096;
                        for (int r=0;r<F;++r) {
                            double g=0,u=0;
                            for (int c=0;c<4096;++c) {g+=double(gate[size_t(r)*4096+c])*x[c];u+=double(up[size_t(r)*4096+c])*x[c];}
                            if (b.swiglu_limit>0) {g=std::min(g,double(b.swiglu_limit));u=std::clamp(u,-double(b.swiglu_limit),double(b.swiglu_limit));}
                            hidden[r]=float(g/(1+std::exp(-g))*u);
                        }
                        double cpu_error=0,gpu_error=0,exact_norm=0;
                        for (int r=0;r<4096;++r) {
                            double value=0;for (int c=0;c<F;++c) value+=double(down[size_t(r)*F+c])*hidden[c];
                            exact[r]=float(value);exact_norm+=std::abs(exact[r]);
                            cpu_error+=std::abs(reference[r]-exact[r]);gpu_error+=std::abs(check_output[size_t(i)*4096+r]-exact[r]);
                        }
                        maximum_cpu_float_error=std::max(maximum_cpu_float_error,cpu_error/(exact_norm+1e-30));
                        if (cpu_ggml_reference && (b.prefill==0 || decode_partial)) {
#ifdef STRATA_NATIVE_EXPERTS
                            // Independent ggml dots with the CPU path's actual
                            // quantized activations separate arithmetic errors
                            // from the intrinsic Q8_K rounding against float.
                            const auto *gc=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.gu_type));
                            const auto *dc=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.d_type));
                            std::vector<float> gh(F);std::vector<uint8_t> hq(f.h_bytes);
                            for (int r=0;r<F;++r) {
                                float g=0,u=0;
                                gc->vec_dot(4096,&g,0,job.blob+size_t(r)*f.gu_row,0,quant.data(),0,1);
                                gc->vec_dot(4096,&u,0,job.native_up+size_t(r)*f.gu_row,0,quant.data(),0,1);
                                if (f.swiglu_limit>0) {
                                    g=std::fmin(g,f.swiglu_limit);
                                    u=std::fmax(-f.swiglu_limit,std::fmin(u,f.swiglu_limit));
                                }
                                gh[r]=g/(1.f+std::exp(-g))*u;
                            }
                            // Validate GU before hidden quantization: tiny
                            // legal float-order differences can cross a Q8_K
                            // rounding boundary, so the down-stage reference
                            // must use the CPU path's actual hidden values.
                            std::vector<float> actual_hidden(F);
                            const void *qa=quant.data();float *ha=actual_hidden.data();
                            cpu::native_gu_rows(f,job.blob,&qa,1,&ha,0,F,job.native_up);
                            double gu_error=0,gu_norm=0;
                            for (int r=0;r<F;++r) {
                                gu_error+=std::abs(double(actual_hidden[r])-gh[r]);gu_norm+=std::abs(gh[r]);
                            }
                            const double gu_relative=gu_error/(gu_norm+1e-30);
                            maximum_cpu_ggml_error=std::max(maximum_cpu_ggml_error,gu_relative);
                            if (!std::isfinite(gu_relative) || gu_relative>=1e-5)
                                throw std::runtime_error("mixed worker CPU ggml GU parity failed layer="+
                                                         std::to_string(b.layer)+" relative="+std::to_string(gu_relative));
                            cpu::native_quant_h(f,actual_hidden.data(),hq.data());
                            double error=0,norm=0;
                            for (int r=0;r<4096;++r) {
                                float expected=0;
                                dc->vec_dot(F,&expected,0,job.native_down+size_t(r)*f.d_row,0,hq.data(),0,1);
                                error+=std::abs(double(reference[r])-expected);norm+=std::abs(expected);
                            }
                            const double relative=error/(norm+1e-30);
                            maximum_cpu_ggml_error=std::max(maximum_cpu_ggml_error,relative);
                            if (!std::isfinite(relative) || relative>=1e-5)
                                throw std::runtime_error("mixed worker CPU ggml arithmetic parity failed layer="+
                                                         std::to_string(b.layer)+" relative="+std::to_string(relative));
#else
                            throw std::runtime_error("mixed CPU ggml reference requires native experts");
#endif
                        }
                        if (!std::isfinite(cpu_error) || !std::isfinite(gpu_error) ||
                            (!cpu_ggml_reference && (b.prefill==0 || decode_partial) && cpu_error/(exact_norm+1e-30)>=0.03) || gpu_error/(exact_norm+1e-30)>=0.03)
                            throw std::runtime_error("mixed worker native float parity failed layer="+std::to_string(b.layer)+
                                                     " cpu="+std::to_string(cpu_error/(exact_norm+1e-30))+" gpu="+std::to_string(gpu_error/(exact_norm+1e-30)));
                    }
                }
            }
            if (adapt_after_rows && (b.prefill==0 || decode_partial) && !pending_owners[b.layer].empty()) {
                // All old rows and their diagnostics are complete. The lease
                // still belongs to this worker, so ownership changes cannot
                // change the parent's current request's route mask or rows.
                std::array<uint8_t,288> previous_owned{};
                std::memcpy(previous_owned.data(),b.worker_owned+b.layer*288,288);
                std::map<int,std::vector<int>> desired;
                std::vector<std::unique_ptr<Device>> recycled;
                for (int e:pending_owners[b.layer]) if (!b.primary_owned[e]) desired[e]={};
                for (auto it=cache.begin();it!=cache.end();) {
                    if (it->first.first==int(b.layer) && !desired.count(it->first.second)) {
                        occupied-=it->second.charged;
                        if (reuse_adaptive) recycled.push_back(std::move(it->second.device));
                        it=cache.erase(it);
                    } else ++it;
                }
                // Uploads remain ordered on the execution stream. The next
                // request cannot execute weights before their upload; the
                // current rows were already copied to the response buffer.
                const uint64_t uploads_before_adapt=upload_bytes;
                load_experts(b.layer,layout,G,U,D,desired,reuse_adaptive ? &recycled : nullptr);
                adapt_upload_bytes+=upload_bytes-uploads_before_adapt;
                std::memset(b.worker_owned+b.layer*288,0,288);
                for (const auto &[e,unused]:desired) {
                    b.worker_owned[b.layer*288+e]=1;
                    adapt_replacements+=!previous_owned[e];
                }
                ++adapt_updates;pending_owners[b.layer].clear();
            }
            if (profile_prefill && is_prefill)
                std::cerr << "MIXED_PREFILL_PROFILE layer=" << b.layer << " tile=" << panel_rows << " panels=" << panels
                          << " cache_prepare_ms=" << cache_prepare_ms << " products_copy_ms=" << products_ms
                          << " acknowledgement_ms=" << ack_ms << " chunks=" << chunk_counter
                          << " wall_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count()
                          << " gap_ms=" << (prefill_last_end.time_since_epoch().count() ?
                                 std::chrono::duration<double,std::milli>(request_start-prefill_last_end).count() : 0.0) << '\n';
            if (profile_prefill && is_prefill) prefill_last_end=std::chrono::steady_clock::now();
            if (b.prefill==0 || decode_partial) {
                decode_rows+=b.routes;
                if (profile_decode) {
                    const auto now = std::chrono::steady_clock::now();
                    decode_prologue_ms += std::chrono::duration<double,std::milli>(prologue_end-request_start).count();
                    decode_body_ms += std::chrono::duration<double,std::milli>(now-prologue_end).count();
                    ++decode_requests;
                }
            }
            if (is_prefill) prefill_rows+=b.routes;
            if (sequence % 1000 == 0 || check_rows)
                std::cerr << "MIXED_WORKER sequence=" << sequence << " layer=" << b.layer << " mode=" << b.prefill
                          << " routes=" << b.routes << " cache_MiB=" << occupied/double(MiB)
                          << " elapsed_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count() << '\n';
            const int next_layer=prefetch_next && is_prefill && b.nt>=4096 ? int(b.layer)+1 : -1;
            if (is_prefill) {
                b.chunk_rows=0; b.chunk_tag=(++chunk_counter)|STRATA_MIXED_FINAL;
                strata_mixed_message packet{sequence,b.chunk_tag,STRATA_MIXED_CHUNK};
                strata::core::mixed_packet(transport.socket,packet,true);
            } else strata::core::mixed_signal(transport.socket, sequence, true);
            if (profile_decode) last_reply = std::chrono::steady_clock::now();
            if (next_layer>=0 && size_t(next_layer)<m.layers.size() && m.layers[next_layer].ffn==strata::core::FfnKind::Moe) {
                const auto start=std::chrono::steady_clock::now();
                const std::string prefix="blk."+std::to_string(next_layer)+".";
                const auto &ng=artifact.at(prefix+"ffn_gate_exps.weight"),&nu=artifact.at(prefix+"ffn_up_exps.weight"),&nd=artifact.at(prefix+"ffn_down_exps.weight");
                const auto format=k::native_expert_layout(ng.tensor->type,nd.tensor->type,m.hidden,m.layers[next_layer].intermediate);
                load_experts(next_layer,format,ng,nu,nd,next_experts);
                if (profile_prefill) std::cerr << "MIXED_NEXT_PREFETCH layer=" << next_layer << " experts=" << next_experts.size()
                    << " prepare_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() << '\n';
            }
        }
    } catch (const std::exception &e) { std::cerr << "mixed worker: " << e.what() << '\n'; return 1; }
#else
    std::cerr << "mixed worker requires Linux\n"; return 1;
#endif
}
