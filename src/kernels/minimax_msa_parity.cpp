#include "strata/host/minimax_msa.h"
#include "strata/kernels/minimax_msa.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "MiniMax MSA parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) ck(cudaFree(p), "cudaFree"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

template<class T> void upload(Device<T>& d, const std::vector<T>& h) {
    ck(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
}

int main() {
    int devices = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&devices);
    if (count_status != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA/HIP device available");
        return 77;
    }
    const int queries = 2, heads = 2, dim = 2, keys = 8, block = 2, nblocks = 4;
    const size_t score_stride = 6, id_stride = 4;
    const std::vector<float> q = {1, 0, 0, 1, 1, 0, 0, 1};
    const std::vector<float> k = {1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0, 8};
    const std::vector<int32_t> qpos = {3, 5};
    const std::vector<int32_t> kpos = {0, 1, 2, 3, 4, 5, 6, 7};
    const size_t score_n = (size_t)queries * heads * score_stride;
    const size_t id_n = (size_t)queries * heads * id_stride;
    std::vector<float> want_scores(score_n, 123.0f), got_scores(score_n, 123.0f);
    std::vector<int32_t> want_ids(id_n, -9), got_ids(id_n, -9);
    if (strata_minimax_msa_block_scores_f32(q.data(), k.data(), qpos.data(), kpos.data(),
            queries, heads, dim, keys, block, nblocks, want_scores.data(), score_stride) != STRATA_MINIMAX_MSA_OK ||
        strata_minimax_msa_select_blocks(want_scores.data(), qpos.data(), queries, heads, nblocks,
            block, 2, 1, score_stride, want_ids.data(), id_stride) != STRATA_MINIMAX_MSA_OK) {
        std::fprintf(stderr, "MiniMax MSA parity: host reference rejected fixture\n");
        return 1;
    }

    Device<float> dq(q.size()), dk(k.size()), ds(score_n);
    Device<int32_t> dqp(qpos.size()), dkp(kpos.size()), did(id_n);
    upload(dq, q); upload(dk, k); upload(dqp, qpos); upload(dkp, kpos);
    ck(cudaMemcpy(ds.p, got_scores.data(), score_n * sizeof(float), cudaMemcpyHostToDevice), "initialize scores");
    ck(cudaMemcpy(did.p, got_ids.data(), id_n * sizeof(int32_t), cudaMemcpyHostToDevice), "initialize IDs");
    strata::kernels::minimax_msa_block_scores(dq.p, dk.p, dqp.p, dkp.p, queries, heads, dim,
        keys, block, nblocks, ds.p, score_stride, nullptr);
    strata::kernels::minimax_msa_select_blocks(ds.p, dqp.p, queries, heads, nblocks, block,
        2, 1, score_stride, did.p, id_stride, nullptr);
    ck(cudaDeviceSynchronize(), "synchronize");
    ck(cudaMemcpy(got_scores.data(), ds.p, score_n * sizeof(float), cudaMemcpyDeviceToHost), "download scores");
    ck(cudaMemcpy(got_ids.data(), did.p, id_n * sizeof(int32_t), cudaMemcpyDeviceToHost), "download IDs");

    for (size_t i = 0; i < score_n; ++i) {
        const float a = want_scores[i], b = got_scores[i];
        if ((std::isinf(a) && a != b) || (!std::isinf(a) && std::fabs(a - b) > 1e-6f)) {
            std::fprintf(stderr, "score[%zu]: host %.9g device %.9g\n", i, a, b);
            return 1;
        }
    }
    for (size_t row = 0; row < (size_t)queries * heads; ++row) {
        for (int i = 0; i < 2; ++i) if (want_ids[row * id_stride + i] != got_ids[row * id_stride + i]) {
            std::fprintf(stderr, "row %zu id[%d]: host %d device %d\n", row, i,
                want_ids[row * id_stride + i], got_ids[row * id_stride + i]);
            return 1;
        }
        if (want_ids[row * id_stride + 2] != -9 || got_ids[row * id_stride + 2] != -9) {
            std::fprintf(stderr, "MiniMax MSA parity: kernel overwrote id row padding\n");
            return 1;
        }
    }

    const std::vector<float> aq = {1, 0, 0, 1};
    const std::vector<float> ak = {1, 0, 0, 1, 100, 100, 100, 100};
    const std::vector<float> av = {1, 0, 0, 2, 50, 50, 50, 50};
    const std::vector<int32_t> aqpos = {1}, akpos = {0, 1, 2, 3}, selected = {0, 1};
    std::vector<float> want_attn(4, 0.0f), got_attn(4, 0.0f);
    if (strata_minimax_msa_attention_f32(aq.data(), ak.data(), av.data(), aqpos.data(), akpos.data(),
            selected.data(), 1, 2, 1, 2, 4, 2, 2, 2, 2, want_attn.data()) != STRATA_MINIMAX_MSA_OK) {
        std::fprintf(stderr, "MiniMax MSA parity: host attention reference rejected fixture\n");
        return 1;
    }
    Device<float> daq(aq.size()), dak(ak.size()), dav(av.size()), dout(got_attn.size());
    Device<int32_t> daqpos(aqpos.size()), dakpos(akpos.size()), dselected(selected.size());
    upload(daq, aq); upload(dak, ak); upload(dav, av);
    upload(daqpos, aqpos); upload(dakpos, akpos); upload(dselected, selected);
    strata::kernels::minimax_msa_attention_f32(daq.p, dak.p, dav.p, daqpos.p, dakpos.p,
        dselected.p, 1, 2, 1, 2, 4, 2, 2, 2, dout.p, nullptr);
    ck(cudaDeviceSynchronize(), "attention synchronize");
    ck(cudaMemcpy(got_attn.data(), dout.p, got_attn.size() * sizeof(float), cudaMemcpyDeviceToHost),
       "download attention");
    for (size_t i = 0; i < got_attn.size(); ++i) {
        if (std::fabs(got_attn[i] - want_attn[i]) > 1e-6f) {
            std::fprintf(stderr, "attention[%zu]: host %.9g device %.9g\n", i, want_attn[i], got_attn[i]);
            return 1;
        }
    }

    // ---- the real geometry: 128-wide heads, 128-token blocks, 4 index / kv heads, 64 query heads, top-4 of 8 blocks
    {
        const int Q = 3, IH = 4, D = 128, KEYS = 1000, BS = 128, NB = 8, TOPK = 4, LOCAL = 1, QH = 64;
        const size_t sstride = NB, istride = TOPK;
        std::vector<float> rq((size_t)Q * IH * D), rk((size_t)KEYS * D), rv((size_t)KEYS * IH * D), rqa((size_t)Q * QH * D),
            rka((size_t)KEYS * IH * D);
        uint32_t seed = 12345;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (float& x : rq) x = rnd();
        for (float& x : rk) x = rnd();
        for (float& x : rv) x = rnd();
        for (float& x : rqa) x = rnd();
        for (float& x : rka) x = rnd();
        const std::vector<int32_t> rqpos = {900, 640, 999};
        std::vector<int32_t> rkpos(KEYS);
        for (int i = 0; i < KEYS; ++i) rkpos[i] = i;
        std::vector<float> w_scores((size_t)Q * IH * sstride, 0.f), g_scores((size_t)Q * IH * sstride, 0.f);
        std::vector<int32_t> w_ids((size_t)Q * IH * istride, -1), g_ids((size_t)Q * IH * istride, -1);
        if (strata_minimax_msa_block_scores_f32(rq.data(), rk.data(), rqpos.data(), rkpos.data(), Q, IH, D, KEYS, BS, NB,
                w_scores.data(), sstride) != STRATA_MINIMAX_MSA_OK ||
            strata_minimax_msa_select_blocks(w_scores.data(), rqpos.data(), Q, IH, NB, BS, TOPK, LOCAL, sstride,
                w_ids.data(), istride) != STRATA_MINIMAX_MSA_OK) {
            std::fprintf(stderr, "MiniMax MSA parity: host reference rejected the 128-wide fixture\n");
            return 1;
        }
        std::vector<float> w_out((size_t)Q * QH * D, 0.f), g_out((size_t)Q * QH * D, 0.f);
        if (strata_minimax_msa_attention_f32(rqa.data(), rka.data(), rv.data(), rqpos.data(), rkpos.data(), w_ids.data(),
                Q, QH, IH, D, KEYS, BS, NB, TOPK, istride, w_out.data()) != STRATA_MINIMAX_MSA_OK) {
            std::fprintf(stderr, "MiniMax MSA parity: host attention rejected the 128-wide fixture\n");
            return 1;
        }
        Device<float> Dq(rq.size()), Dk(rk.size()), Dv(rv.size()), Dqa(rqa.size()), Dka(rka.size()), Ds(w_scores.size()),
            Dout(g_out.size());
        Device<int32_t> Dqp(rqpos.size()), Dkp(rkpos.size()), Did(w_ids.size());
        upload(Dq, rq); upload(Dk, rk); upload(Dv, rv); upload(Dqa, rqa); upload(Dka, rka); upload(Dqp, rqpos);
        upload(Dkp, rkpos);
        strata::kernels::minimax_msa_block_scores(Dq.p, Dk.p, Dqp.p, Dkp.p, Q, IH, D, KEYS, BS, NB, Ds.p, sstride, nullptr);
        strata::kernels::minimax_msa_select_blocks(Ds.p, Dqp.p, Q, IH, NB, BS, TOPK, LOCAL, sstride, Did.p, istride, nullptr);
        strata::kernels::minimax_msa_attention_f32(Dqa.p, Dka.p, Dv.p, Dqp.p, Dkp.p, Did.p, Q, QH, IH, D, KEYS, BS, TOPK,
            istride, Dout.p, nullptr);
        ck(cudaDeviceSynchronize(), "128-wide synchronize");
        ck(cudaMemcpy(g_scores.data(), Ds.p, g_scores.size() * sizeof(float), cudaMemcpyDeviceToHost), "download scores");
        ck(cudaMemcpy(g_ids.data(), Did.p, g_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download ids");
        ck(cudaMemcpy(g_out.data(), Dout.p, g_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "download attention");
        float worst_score = 0.f, worst_out = 0.f;
        for (size_t i = 0; i < w_scores.size(); ++i) {
            if (std::isinf(w_scores[i])) { if (w_scores[i] != g_scores[i]) { std::fprintf(stderr, "128-wide score[%zu] inf mismatch\n", i); return 1; } continue; }
            worst_score = std::fmax(worst_score, std::fabs(w_scores[i] - g_scores[i]));
        }
        for (size_t i = 0; i < w_ids.size(); ++i)
            if (w_ids[i] != g_ids[i]) { std::fprintf(stderr, "128-wide id[%zu]: host %d device %d\n", i, w_ids[i], g_ids[i]); return 1; }
        for (size_t i = 0; i < w_out.size(); ++i) worst_out = std::fmax(worst_out, std::fabs(w_out[i] - g_out[i]));
        std::printf("128-wide case: max |score diff| %.3g, max |attention diff| %.3g\n", worst_score, worst_out);
        if (worst_score > 2e-4f || worst_out > 2e-4f) {
            std::fprintf(stderr, "MiniMax MSA parity: 128-wide case exceeds the 2e-4 float-order tolerance\n");
            return 1;
        }
    }
    // ---- the same geometry with FP16 caches: the device reads the half bits, the host reference the same rounded values
    {
        const int Q = 3, IH = 4, D = 128, KEYS = 1000, BS = 128, NB = 8, TOPK = 4, LOCAL = 1, QH = 64;
        const size_t sstride = NB, istride = TOPK;
        auto f2h = [](float f) -> uint16_t {   // round to nearest even; the data is in [-1, 1], subnormals flush to zero
            uint32_t x;
            std::memcpy(&x, &f, 4);
            const uint32_t sign = (x >> 16) & 0x8000u;
            const int e = (int) ((x >> 23) & 0xff) - 127 + 15;
            if (e <= 0) return (uint16_t) sign;
            if (e >= 31) return (uint16_t) (sign | 0x7bffu);
            uint32_t mant = x & 0x7fffffu;
            uint32_t h = sign | ((uint32_t) e << 10) | (mant >> 13);
            const uint32_t rest = mant & 0x1fffu;
            if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
            return (uint16_t) h;
        };
        auto h2f = [](uint16_t h) -> float {
            const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
            const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu;
            uint32_t x;
            if (e == 0) x = sign;   // zero (the data has no subnormals after the flush above)
            else x = sign | ((e + 127 - 15) << 23) | (m << 13);
            float f;
            std::memcpy(&f, &x, 4);
            return f;
        };
        std::vector<float> rq((size_t) Q * IH * D), rk((size_t) KEYS * D), rv((size_t) KEYS * IH * D), rqa((size_t) Q * QH * D),
            rka((size_t) KEYS * IH * D);
        uint32_t seed = 987;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float) ((seed >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (float& x : rq) x = rnd();
        for (float& x : rqa) x = rnd();
        std::vector<uint16_t> hk(rk.size()), hv(rv.size()), hka(rka.size());
        for (size_t i = 0; i < rk.size(); ++i) { hk[i] = f2h(rnd()); rk[i] = h2f(hk[i]); }
        for (size_t i = 0; i < rv.size(); ++i) { hv[i] = f2h(rnd()); rv[i] = h2f(hv[i]); }
        for (size_t i = 0; i < rka.size(); ++i) { hka[i] = f2h(rnd()); rka[i] = h2f(hka[i]); }
        const std::vector<int32_t> rqpos = {900, 640, 999};
        std::vector<int32_t> rkpos(KEYS);
        for (int i = 0; i < KEYS; ++i) rkpos[i] = i;
        std::vector<float> w_scores((size_t) Q * IH * sstride, 0.f), g_scores((size_t) Q * IH * sstride, 0.f);
        std::vector<int32_t> w_ids((size_t) Q * IH * istride, -1), g_ids((size_t) Q * IH * istride, -1);
        if (strata_minimax_msa_block_scores_f32(rq.data(), rk.data(), rqpos.data(), rkpos.data(), Q, IH, D, KEYS, BS, NB,
                w_scores.data(), sstride) != STRATA_MINIMAX_MSA_OK ||
            strata_minimax_msa_select_blocks(w_scores.data(), rqpos.data(), Q, IH, NB, BS, TOPK, LOCAL, sstride,
                w_ids.data(), istride) != STRATA_MINIMAX_MSA_OK) {
            std::fprintf(stderr, "MiniMax MSA parity: host reference rejected the half fixture\n");
            return 1;
        }
        std::vector<float> w_out((size_t) Q * QH * D, 0.f), g_out((size_t) Q * QH * D, 0.f);
        if (strata_minimax_msa_attention_f32(rqa.data(), rka.data(), rv.data(), rqpos.data(), rkpos.data(), w_ids.data(),
                Q, QH, IH, D, KEYS, BS, NB, TOPK, istride, w_out.data()) != STRATA_MINIMAX_MSA_OK) {
            std::fprintf(stderr, "MiniMax MSA parity: host attention rejected the half fixture\n");
            return 1;
        }
        Device<float> Dq(rq.size()), Dqa(rqa.size()), Ds(w_scores.size()), Dout(g_out.size());
        Device<uint16_t> Dk(hk.size()), Dka(hka.size()), Dv(hv.size());
        Device<int32_t> Dqp(rqpos.size()), Dkp(rkpos.size()), Did(w_ids.size());
        upload(Dq, rq); upload(Dqa, rqa); upload(Dk, hk); upload(Dka, hka); upload(Dv, hv); upload(Dqp, rqpos); upload(Dkp, rkpos);
        strata::kernels::minimax_msa_block_scores_h(Dq.p, Dk.p, Dqp.p, Dkp.p, Q, IH, D, KEYS, BS, NB, Ds.p, sstride, nullptr);
        strata::kernels::minimax_msa_select_blocks(Ds.p, Dqp.p, Q, IH, NB, BS, TOPK, LOCAL, sstride, Did.p, istride, nullptr);
        strata::kernels::minimax_msa_attention_h(Dqa.p, Dka.p, Dv.p, Dqp.p, Dkp.p, Did.p, Q, QH, IH, D, KEYS, BS, TOPK, istride,
            Dout.p, nullptr);
        ck(cudaDeviceSynchronize(), "half synchronize");
        ck(cudaMemcpy(g_scores.data(), Ds.p, g_scores.size() * sizeof(float), cudaMemcpyDeviceToHost), "download scores");
        ck(cudaMemcpy(g_ids.data(), Did.p, g_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download ids");
        ck(cudaMemcpy(g_out.data(), Dout.p, g_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "download attention");
        float worst_score = 0.f, worst_out = 0.f;
        for (size_t i = 0; i < w_scores.size(); ++i)
            if (!std::isinf(w_scores[i])) worst_score = std::fmax(worst_score, std::fabs(w_scores[i] - g_scores[i]));
        for (size_t i = 0; i < w_ids.size(); ++i)
            if (w_ids[i] != g_ids[i]) { std::fprintf(stderr, "half id[%zu]: host %d device %d\n", i, w_ids[i], g_ids[i]); return 1; }
        for (size_t i = 0; i < w_out.size(); ++i) worst_out = std::fmax(worst_out, std::fabs(w_out[i] - g_out[i]));
        std::printf("FP16-cache case: max |score diff| %.3g, max |attention diff| %.3g\n", worst_score, worst_out);
        if (worst_score > 2e-4f || worst_out > 2e-4f) {
            std::fprintf(stderr, "MiniMax MSA parity: FP16-cache case exceeds the 2e-4 float-order tolerance\n");
            return 1;
        }
    }
    // ---- selection at long-context sizes (the register-resident kernel above 64 blocks, the block-barrier kernel above 8192): many tied
    // scores, blocks no query can see (-inf), padded score rows, several local blocks; ids must equal the host reference exactly
    for (const int NB : {65, 128, 129, 300, 1024, 1025, 1056, 4096, 4097, 8192, 8193})
    for (const int Q : {1, 4}) {
        const int IH = 4, BS = 128, TOPK = 16, LOCAL = 3;
        const size_t sstride = (size_t)NB + 7, istride = TOPK + 2;
        uint32_t seed = 777u + (uint32_t)NB * 3u + (uint32_t)Q;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        std::vector<int32_t> qp(Q);
        for (int q = 0; q < Q; ++q) qp[q] = (int32_t)(NB * BS - 1 - (int)(rnd() % (uint32_t)(NB * BS / 2)));
        std::vector<float> sc((size_t)Q * IH * sstride, 0.f);
        for (int r = 0; r < Q * IH; ++r)
            for (int b = 0; b < NB; ++b) {
                const uint32_t x = rnd();
                sc[(size_t)r * sstride + b] = (x & 15) == 0 ? -INFINITY : (float)(x % 64u) * 0.25f - 8.f;   // 64 levels: ties
            }
        std::vector<int32_t> want((size_t)Q * IH * istride, -1), got((size_t)Q * IH * istride, -1);
        if (strata_minimax_msa_select_blocks(sc.data(), qp.data(), Q, IH, NB, BS, TOPK, LOCAL, sstride, want.data(), istride) !=
            STRATA_MINIMAX_MSA_OK) {
            std::fprintf(stderr, "MiniMax MSA parity: host selection rejected n_blocks %d\n", NB);
            return 1;
        }
        Device<float> Ds(sc.size());
        Device<int32_t> Dq(qp.size()), Di(got.size());
        upload(Ds, sc); upload(Dq, qp); upload(Di, got);
        strata::kernels::minimax_msa_select_blocks(Ds.p, Dq.p, Q, IH, NB, BS, TOPK, LOCAL, sstride, Di.p, istride, nullptr);
        ck(cudaDeviceSynchronize(), "selection synchronize");
        ck(cudaMemcpy(got.data(), Di.p, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download selection");
        for (int r = 0; r < Q * IH; ++r)
            for (int i = 0; i < TOPK; ++i)
                if (want[(size_t)r * istride + i] != got[(size_t)r * istride + i]) {
                    std::fprintf(stderr, "selection n_blocks %d row %d id[%d]: host %d device %d\n", NB, r, i, want[(size_t)r * istride + i],
                                 got[(size_t)r * istride + i]);
                    return 1;
                }
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
        ck(cudaEventRecord(e0, nullptr), "record");
        for (int it = 0; it < 50; ++it)
            strata::kernels::minimax_msa_select_blocks(Ds.p, Dq.p, Q, IH, NB, BS, TOPK, LOCAL, sstride, Di.p, istride, nullptr);
        ck(cudaEventRecord(e1, nullptr), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        std::printf("selection n_blocks %4d, %d rows, top-%d: ids match, %.1f us a launch\n", NB, Q * IH, TOPK, ms * 1000.f / 50.f);
    }
    // ---- decode-size block scores (all-heads kernels, >= 128 blocks): a hash of the score bits, to compare kernels bitwise across runs
    // (STRATA_MSA_SCORES_OLD=1, STRATA_MSA_SCORES_U=1/2/4/8), and the time a launch
    for (const int NB : {1024, 4096})
        for (const int Q : {1, 5}) {
            const int IH = 4, BS = 128, KEYS = NB * BS - 37;
            const size_t sstride = (size_t)NB + 3;
            uint32_t seed = 4242u + (uint32_t)NB * 7u + (uint32_t)Q;
            auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
            std::vector<float> hq((size_t)Q * IH * 128), hks(KEYS);
            std::vector<uint16_t> hk((size_t)KEYS * 128);
            std::vector<uint8_t> hk8((size_t)KEYS * 128);
            std::vector<int32_t> hqp(Q), hkp(KEYS);
            for (float& x : hq) x = (float)(rnd() & 0xffff) / 32768.f - 1.f;
            for (uint16_t& x : hk) { const uint32_t r = rnd(); x = (uint16_t)(((r & 1) << 15) | ((10 + (r >> 1) % 8) << 10) | ((r >> 4) & 1023)); }
            for (uint8_t& x : hk8) x = (uint8_t)(rnd() & 0xff);
            for (float& x : hks) x = 0.001f + (float)(rnd() & 1023) * 1e-5f;
            for (int i = 0; i < KEYS; ++i) hkp[i] = (i % 1000) == 999 ? -1 : i;   // a few empty slots
            for (int i = 0; i < Q; ++i) hqp[i] = KEYS - 1 - (Q - 1 - i) * 61;     // a window whose queries see different key counts
            Device<float> Dq(hq.size()), Dks(hks.size()), Ds((size_t)Q * IH * sstride);
            Device<uint16_t> Dk(hk.size());
            Device<uint8_t> Dk8(hk8.size());
            Device<int32_t> Dqp(hqp.size()), Dkp(hkp.size());
            upload(Dq, hq); upload(Dks, hks); upload(Dk, hk); upload(Dk8, hk8); upload(Dqp, hqp); upload(Dkp, hkp);
            for (int path = 0; path < 2; ++path) {
                auto run = [&]() {
                    if (path == 0)
                        strata::kernels::minimax_msa_block_scores_h(Dq.p, Dk.p, Dqp.p, Dkp.p, Q, IH, 128, KEYS, BS, NB, Ds.p, sstride, nullptr);
                    else
                        strata::kernels::minimax_msa_block_scores_q8(Dq.p, Dk8.p, Dks.p, Dqp.p, Dkp.p, Q, IH, 128, KEYS, BS, NB, Ds.p, sstride, nullptr);
                };
                run();
                ck(cudaDeviceSynchronize(), "scores synchronize");
                std::vector<float> got((size_t)Q * IH * sstride);
                ck(cudaMemcpy(got.data(), Ds.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "download scores");
                uint64_t hsh = 1469598103934665603ull;
                for (int r = 0; r < Q * IH; ++r)
                    for (int b = 0; b < NB; ++b) {
                        uint32_t bits;
                        std::memcpy(&bits, &got[(size_t)r * sstride + b], 4);
                        hsh = (hsh ^ bits) * 1099511628211ull;
                    }
                cudaEvent_t e0, e1;
                ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
                ck(cudaEventRecord(e0, nullptr), "record");
                for (int it = 0; it < 50; ++it) run();
                ck(cudaEventRecord(e1, nullptr), "record");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                cudaEventDestroy(e0); cudaEventDestroy(e1);
                std::printf("scores %s n_blocks %4d queries %d: hash %016llx, %.1f us a launch\n", path == 0 ? "fp16" : "fp8 ", NB, Q,
                            (unsigned long long)hsh, ms * 1000.f / 50.f);
            }
        }
    std::puts("MiniMax MSA scores, selection and sparse attention match the host reference");
    return 0;
}
