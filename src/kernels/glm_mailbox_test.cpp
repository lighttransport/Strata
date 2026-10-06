// The decode-step mailbox: publication, CPU completion, graph replay across generations, and abort.
#include "strata/kernels/glm_mailbox.hpp"
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
int main() {
    try {
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
