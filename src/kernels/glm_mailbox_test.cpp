// The decode-step mailbox: publication, CPU completion, graph replay across generations, and abort.
#include "strata/kernels/glm_mailbox.hpp"
#include "strata/kernels/glm.hpp"
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void check_fused_publish() {
    constexpr int max_tokens=8,top_k=8,hidden=4096;
    void *host=nullptr,*mapped=nullptr;
    const size_t bytes=k::glm_mailbox_bytes(2,max_tokens,top_k,hidden,false,true);
    ck(cudaHostAlloc(&host,bytes,cudaHostAllocMapped|cudaHostAllocPortable));
    ck(cudaHostGetDevicePointer(&mapped,host,0));std::memset(host,0,bytes);
    auto hv=k::glm_mailbox_view(host,2,max_tokens,top_k,hidden,false,true);
    auto dv=k::glm_mailbox_view(mapped,2,max_tokens,top_k,hidden,false,true);
    float *logits=nullptr,*bias=nullptr,*bonus=nullptr,*weights=nullptr,*unused=nullptr,*x=nullptr;
    int *ids=nullptr,*wanted=nullptr;unsigned *generation=nullptr;
    ck(cudaMalloc(&logits,max_tokens*512*4));ck(cudaMalloc(&bias,512*4));ck(cudaMalloc(&bonus,512*4));
    ck(cudaMalloc(&weights,max_tokens*top_k*4));ck(cudaMalloc(&unused,max_tokens*top_k*4));
    ck(cudaMalloc(&ids,max_tokens*top_k*4));ck(cudaMalloc(&wanted,max_tokens*top_k*4));
    ck(cudaMalloc(&x,max_tokens*hidden*4));ck(cudaMalloc(&generation,4));
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    unsigned step=0;
    for (int ne:{224,288,512}) for (int nt:{1,2,4,8}) for (bool with_bonus:{false,true}) {
        std::vector<float> hl(nt*ne),hb(ne),ha(ne),hx(nt*hidden);
        const auto enqueue=[&] {
            k::glm_mailbox_begin(dv,generation,stream);
            k::glm_router_tokens(logits,bias,ids,weights,ne,top_k,1.f,nt,stream,with_bonus?bonus:nullptr);
            k::glm_router_tokens(logits,bias,wanted,unused,ne,top_k,1.f,nt,stream);
            k::glm_mailbox_publish(dv,0,ids,weights,x,nt,generation,stream,wanted);
            k::glm_router_publish_tokens(logits,bias,ids,weights,ne,top_k,1.f,with_bonus?bonus:nullptr,
                                         dv,1,x,nt,generation,stream,wanted);
        };
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        for (int replay=0;replay<3;++replay) {
            for (int i=0;i<nt*ne;++i) hl[i]=replay ? float((i%ne)%7-3)*.125f : 0.f;
            for (int i=0;i<ne;++i) {hb[i]=replay ? float(i%3)*.01f : 0.f;ha[i]=float(i%13)*.02f;}
            for (int i=0;i<nt*hidden;++i) hx[i]=float(i%101-50)*.125f;
            ck(cudaMemcpyAsync(logits,hl.data(),hl.size()*4,cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(bias,hb.data(),hb.size()*4,cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(bonus,ha.data(),ha.size()*4,cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,stream));
            __atomic_store_n(hv.control,++step,__ATOMIC_SEQ_CST);
            if (!exec) {
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));enqueue();
                ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            }
            ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
            const size_t routes=max_tokens*top_k;
            if (std::memcmp(hv.ids,hv.ids+routes,nt*top_k*4) ||
                std::memcmp(hv.weights,hv.weights+routes,nt*top_k*4) ||
                std::memcmp(hv.wanted,hv.wanted+routes,nt*top_k*4) ||
                std::memcmp(hv.act,hv.act+max_tokens*hidden,nt*hidden*4) ||
                hv.flags[0]!=step || hv.flags[32]!=step)
                throw std::runtime_error("batched fused router/mailbox parity failed");
        }
        ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    }
    ck(cudaStreamDestroy(stream));
    cudaFree(logits);cudaFree(bias);cudaFree(bonus);cudaFree(weights);cudaFree(unused);
    cudaFree(x);cudaFree(ids);cudaFree(wanted);cudaFree(generation);cudaFreeHost(host);
}
int main() {
    try {
        check_fused_publish();
        const int slots = 46, max_tokens = 8, top_k = 8, hidden = 4096;
        const int layers[] = {3, 7, 44};
        const size_t bytes = k::glm_mailbox_bytes(slots, max_tokens, top_k, hidden);
        void *host = nullptr, *mapped = nullptr;
        ck(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(host, 0, bytes);
        ck(cudaHostGetDevicePointer(&mapped, host, 0));
        const auto hv = k::glm_mailbox_view(host, slots, max_tokens, top_k, hidden);
        const auto dv = k::glm_mailbox_view(mapped, slots, max_tokens, top_k, hidden);
        unsigned *generation = nullptr;
        int *ids = nullptr;
        float *weights = nullptr, *x = nullptr, *out = nullptr;
        ck(cudaMalloc(&generation, 4));
        ck(cudaMalloc(&ids, max_tokens * top_k * 4));
        ck(cudaMalloc(&weights, max_tokens * top_k * 4));
        ck(cudaMalloc(&x, max_tokens * hidden * 4));
        ck(cudaMalloc(&out, max_tokens * hidden * 4));
        cudaStream_t stream;
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto enqueue = [&](int nt) {
            for (int l : layers) {
                k::glm_mailbox_publish(dv, l, ids, weights, x, nt, generation, stream);
                k::glm_mailbox_wait_add(dv, l, out, nt, generation, stream);
            }
        };
        cudaGraphExec_t graphs[2] = {};
        int failures = 0;
        for (unsigned step = 1; step <= 6; ++step) {
            const int nt = step % 2 ? 1 : 4;
            const bool replay = step > 2;
            std::vector<int> hid(nt * top_k);
            std::vector<float> hw(nt * top_k), hx(nt * hidden), hout(nt * hidden, 1.f);
            for (int i = 0; i < nt * top_k; ++i) { hid[i] = (i * 37 + step) % 288; hw[i] = 0.1f * i + step; }
            for (int i = 0; i < nt * hidden; ++i) hx[i] = i * 0.5f - step;
            ck(cudaMemcpyAsync(ids, hid.data(), nt * top_k * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(weights, hw.data(), nt * top_k * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(x, hx.data(), nt * hidden * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(out, hout.data(), nt * hidden * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaStreamSynchronize(stream));
            __atomic_store_n(hv.control, step, __ATOMIC_SEQ_CST);
            k::glm_mailbox_begin(dv, generation, stream);
            if (!replay) enqueue(nt);
            else {
                auto &exec = graphs[nt == 4];
                if (!exec) {
                    cudaGraph_t graph;
                    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    enqueue(nt);
                    ck(cudaStreamEndCapture(stream, &graph));
                    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                }
                ck(cudaGraphLaunch(exec, stream));
            }
            for (int l : layers) {
                while (__atomic_load_n(hv.flags + l * 32, __ATOMIC_ACQUIRE) != step) _mm_pause();
                const int *pi = hv.ids + (size_t)l * max_tokens * top_k;
                const float *pw = hv.weights + (size_t)l * max_tokens * top_k;
                const float *pa = hv.act + (size_t)l * max_tokens * hidden;
                float *ps = hv.sum + (size_t)l * max_tokens * hidden;
                if (std::memcmp(pi, hid.data(), nt * top_k * 4) || std::memcmp(pw, hw.data(), nt * top_k * 4) ||
                    std::memcmp(pa, hx.data(), (size_t)nt * hidden * 4)) {
                    std::cerr << "published data differs at step " << step << " layer " << l << '\n';
                    ++failures;
                }
                for (int i = 0; i < nt * hidden; ++i) ps[i] = (float)l + 0.25f * i;
                __atomic_store_n(hv.flags + l * 32 + 16, step, __ATOMIC_RELEASE);
            }
            ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(hout.data(), out, nt * hidden * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < nt * hidden; ++i) {
                float expected = 1.f;
                for (int l : layers) expected += (float)l + 0.25f * i;
                if (hout[i] != expected) {
                    std::cerr << "sum differs at step " << step << " index " << i << '\n';
                    ++failures;
                    break;
                }
            }
            if (hv.control[16]) { std::cerr << "unexpected GPU timeout\n"; ++failures; }
        }
        // An abort releases a wait whose CPU completion never comes, and leaves the output unchanged.
        float before = 0;
        ck(cudaMemcpy(&before, out, 4, cudaMemcpyDeviceToHost));
        __atomic_store_n(hv.control, 99u, __ATOMIC_SEQ_CST);
        k::glm_mailbox_begin(dv, generation, stream);
        k::glm_mailbox_wait_add(dv, 5, out, 1, generation, stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (cudaStreamQuery(stream) != cudaErrorNotReady) { std::cerr << "wait did not block\n"; ++failures; }
        __atomic_store_n(hv.control + 32, 1u, __ATOMIC_SEQ_CST);
        ck(cudaStreamSynchronize(stream));
        float after = 0;
        ck(cudaMemcpy(&after, out, 4, cudaMemcpyDeviceToHost));
        if (after != before) { std::cerr << "aborted wait changed the output\n"; ++failures; }
        std::cout << (failures ? "FAIL" : "PASS") << " glm_mailbox_test\n";
        return failures != 0;
    } catch (const std::exception &e) {
        std::cerr << "glm_mailbox_test: " << e.what() << '\n';
        return 1;
    }
}
