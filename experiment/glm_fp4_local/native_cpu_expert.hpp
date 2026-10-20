#pragma once
// Experimental adapter for bounded partial-model runners. Uses native NVFP4/Q8
// arithmetic, which differs from the existing BF16-input W4A16 tree backend.
#include "strata/artifact/glm_fp4.hpp"
#include "strata/host/glm_fp4.h"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include <memory>
#include "native_hidden_scale.hpp"
#include "native_finite.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::experiment {
class NativeFp4CpuExpert {
    kernels::cpu::NativeFmt format_{};
    std::vector<uint8_t> packed_;
    bool preserve_small_hidden_;
    bool numa_only_ = false;
    alignas(16) std::array<float,8> compact_scales_{};
    std::array<std::unique_ptr<kernels::cpu::NumaTensor>,3> numa_;
    std::array<int,2> numa_cores_{{-1,-1}};
public:
    explicit NativeFp4CpuExpert(const std::array<artifact::GlmFp4Matrix,3>& weights,
                                bool preserve_small_hidden = false, bool packed_k32 = false)
        : preserve_small_hidden_(preserve_small_hidden) {
        using namespace kernels::cpu;
        const auto& g = weights[0];
        if (g.cols != 4096 || g.rows != 2048 ||
            weights[1].rows != g.rows || weights[1].cols != g.cols ||
            weights[2].rows != g.cols || weights[2].cols != g.rows)
            throw std::invalid_argument("native partial expert requires H4096/F2048");
        std::string error;
        if (!native_fmt(kNvfp4Type, kNvfp4Type, g.cols, g.rows, format_, error))
            throw std::invalid_argument(error);
        format_.swiglu_limit = 10.f;
        packed_.resize(format_.bytes);
        const size_t offsets[] = {0, format_.up_off, format_.down_off};
        for (int p = 0; p < 3; ++p) {
            const auto& w = weights[p];
            if (!std::isfinite(w.global) || strata_glm_fp4_pack_nvfp4(
                    w.rows, w.cols, w.codes.data(), w.codes.size(),
                    w.scales.data(), w.scales.size(), packed_.data() + offsets[p],
                    w.rows * w.cols / 64 * 36))
                throw std::invalid_argument("invalid native partial FP4 weights");
        }
        // Explicit CPU layout opt-in: repack in place once, before NUMA preparation.
        // Global scales and persistent weight size are unchanged.
        if (packed_k32) {
            for (int p = 0; p < 3; ++p) {
                auto* projection = packed_.data() + offsets[p];
                const size_t bytes = weights[p].rows * weights[p].cols / 64 * 36;
                if (strata_glm_nvfp4_repack_k32(projection, projection, bytes))
                    throw std::invalid_argument("invalid native K32 repack");
            }
            format_.nvfp4_k32 = true;
        }
        const float scales[] = {weights[0].global, weights[1].global, weights[2].global, 0};
        std::memcpy(packed_.data() + format_.tail_off, scales, sizeof(scales));
    }
    // Serialize preparation with run(), while the pool is idle.
    // Explicit preparation is outside timed service. Retains the complete blob
    // for baseline comparisons and NVFP4 global scales, so this partial-model
    // experiment uses an additional copy until release_original_blob() is called.
    bool prepare_numa(kernels::cpu::ExpertPool& pool) {
        if (!pool.numa_rows_available()) return false;
        const auto cores=pool.numa_cores();
        if (numa_[0] && cores==numa_cores_) return true;
        if (numa_only_) throw std::invalid_argument("released NUMA expert cannot change topology");
        std::array<std::unique_ptr<kernels::cpu::NumaTensor>,3> prepared;
        const size_t offsets[]={0,format_.up_off,format_.down_off};
        for (int p=0;p<3;++p)
            prepared[p]=std::make_unique<kernels::cpu::NumaTensor>(
                packed_.data()+offsets[p],p==2?format_.d_row:format_.gu_row,
                p==2?format_.n_embd:format_.n_ff,1,cores,false);
        numa_=std::move(prepared);numa_cores_=cores;
        return true;
    }
    // After preparation, optionally release the duplicate complete weight blob.
    // Owned-row kernels use job.blob only as a scale base. A local format copy
    // points its nonzero tail offset into this compact buffer; every weight row
    // still comes from the explicitly supplied NUMA projections.
    // This is irreversible: subsequent run() calls must select use_numa=true.
    void release_original_blob() {
        if (numa_only_) return;
        if (!numa_[0]) throw std::invalid_argument("prepare NUMA rows before releasing weights");
        std::memcpy(compact_scales_.data()+4,packed_.data()+format_.tail_off,4*sizeof(float));
        std::vector<uint8_t>().swap(packed_);
        numa_only_=true;
    }
    size_t retained_blob_bytes() const { return packed_.capacity(); }
    // The caller owns one persistent pool and serializes use of it. No pool per
    // expert: reference experts must not create extra spinning worker teams.
    // Up to sixteen tokens are split into balanced native windows of at most 8.
    std::vector<float> run(kernels::cpu::ExpertPool& pool,
                           const std::vector<float>& input, int tokens, bool use_numa = false) const {
        std::vector<float> output;
        std::vector<uint8_t> act;
        run_into(pool, input, tokens, output, act, use_numa);
        return output;
    }
    // Opt-in common-input NT1/NT2 batch, two or three experts. Caller serializes pool
    // and workspace use. Prepared NUMA rows may use original or compact scale blobs.
    // Validate all arguments before resizing or writing any caller-owned buffer.
    template<size_t Count>
    static void run_batch_into(kernels::cpu::ExpertPool& pool,
                              const std::array<const NativeFp4CpuExpert*,Count>& experts,
                              const std::vector<float>& input,
                              std::array<std::vector<float>,Count>& output,
                              std::vector<uint8_t>& act, bool use_numa = false, int tokens = 1) {
        static_assert(Count>=2 && Count<=4, "native batch supports two through four experts");
        using namespace kernels::cpu;
        if (tokens<1 || tokens>2) throw std::invalid_argument("native batch supports one or two tokens");
        if (input.size()!=size_t(tokens)*4096 || !native_all_finite(input.data(),input.size()))
            throw std::invalid_argument("invalid native multi-expert input");
        for (size_t e=0;e<experts.size();++e) {
            if (&input==&output[e]) throw std::invalid_argument("multi-expert input/output alias");
            const auto* expert=experts[e];
            if (!expert) throw std::invalid_argument("null native batch expert");
            if (expert->format_.nvfp4_k32)
                throw std::invalid_argument("multi-expert batch requires original FP4 layout");
            if (expert->numa_only_ && !use_numa)
                throw std::invalid_argument("batch original weights released; select NUMA rows");
            if (use_numa && (!expert->numa_[0] || !pool.numa_rows_available() ||
                             expert->numa_cores_!=pool.numa_cores()))
                throw std::invalid_argument("batch NUMA rows not prepared for this pool");
        }
        for (const auto* expert:experts) {
            if (expert->preserve_small_hidden_!=experts[0]->preserve_small_hidden_)
                throw std::invalid_argument("inconsistent batch hidden-rescale policy");
            if (expert->numa_only_!=experts[0]->numa_only_)
                throw std::invalid_argument("inconsistent batch compact storage");
        }
        MultiExpertHiddenScale<Count> scaler;
        NativeFmt fmt=experts[0]->format_;
        if (experts[0]->numa_only_) fmt.tail_off=4*sizeof(float);
        if (experts[0]->preserve_small_hidden_) fmt.hidden_transform=&scaler;
        act.resize(size_t(tokens)*fmt.act_bytes);
        for(int t=0;t<tokens;++t)native_quant_act(fmt,input.data()+size_t(t)*4096,act.data()+size_t(t)*fmt.act_bytes);
        std::array<ExpertJobMulti,Count> jobs{};
        for (size_t e=0;e<Count;++e) {
            const auto& expert=*experts[e];
            output[e].resize(input.size());
            jobs[e].expert_id=e; jobs[e].nt=tokens; jobs[e].blob=expert.numa_only_?reinterpret_cast<const uint8_t*>(expert.compact_scales_.data()):expert.packed_.data();
            for(int t=0;t<tokens;++t){jobs[e].nact[t]=act.data()+size_t(t)*fmt.act_bytes;jobs[e].out[t]=output[e].data()+size_t(t)*4096;}
            if (use_numa) for (int node=0;node<2;++node)
                jobs[e].numa[node]={expert.numa_[0]->shard(0,node),
                                   expert.numa_[1]->shard(0,node),expert.numa_[2]->shard(0,node)};
        }
        pool.run_split_multi_native(fmt,jobs.data(),int(Count));
        for (size_t e=0;e<Count;++e) {
            if (experts[0]->preserve_small_hidden_) for(int t=0;t<tokens;++t)scaler.restore(e,t,output[e].data()+size_t(t)*4096,4096);
            if (!native_all_finite(output[e].data(),output[e].size()))
                throw std::runtime_error("nonfinite native batch output");
        }
    }
    static void run_three_into(kernels::cpu::ExpertPool& pool,
                              const std::array<const NativeFp4CpuExpert*,3>& experts,
                              const std::vector<float>& input,
                              std::array<std::vector<float>,3>& output,
                              std::vector<uint8_t>& act, bool use_numa = false, int tokens = 1) {
        run_batch_into(pool, experts, input, output, act, use_numa, tokens);
    }
    // Explicit caller-owned storage. Reuse across serialized calls/experts;
    // never share these buffers between concurrent pool submissions. The output
    // must not alias input. Buffers may retain capacity after a smaller call.
    void run_into(kernels::cpu::ExpertPool& pool,
                           const std::vector<float>& input, int tokens, std::vector<float>& output, std::vector<uint8_t>& act, bool use_numa = false) const {
        if (&input == &output) throw std::invalid_argument("input/output workspace alias");
        using namespace kernels::cpu;
        if (tokens < 1 || tokens > 16 || input.size() != size_t(tokens) * format_.n_embd)
            throw std::invalid_argument("invalid native partial activation extent");
        if (!native_all_finite(input.data(),input.size()))
            throw std::invalid_argument("nonfinite native partial activation");
        if (use_numa && (!numa_[0] || !pool.numa_rows_available() || pool.numa_cores()!=numa_cores_))
            throw std::invalid_argument("native NUMA rows not prepared for this pool topology");
        if (numa_only_ && !use_numa) throw std::invalid_argument("original weights released; select NUMA rows");
        const int windows = (tokens + MAXT - 1) / MAXT;
        const int window_capacity = (tokens + windows - 1) / windows;
        output.resize(input.size());
        // Reuse caller capacity, sized for the largest balanced window.
        // Quantization writes every active byte before the pool reads it.
        act.resize(size_t(window_capacity) * format_.act_bytes);
        SmallHiddenScale scaler;
        NativeFmt selected_format = format_;
        if (numa_only_) selected_format.tail_off=4*sizeof(float);
        if (preserve_small_hidden_) selected_format.hidden_transform = &scaler;
        for (int i = 0, first = 0; i < windows; ++i) {
            const int count = tokens / windows + int(i < tokens % windows);
            ExpertJobMulti job;
            job.nt = count;
            job.blob = numa_only_?reinterpret_cast<const uint8_t*>(compact_scales_.data()):packed_.data();
            if (use_numa) for (int node=0;node<2;++node)
                job.numa[node]={numa_[0]->shard(0,node),numa_[1]->shard(0,node),numa_[2]->shard(0,node)};
            for (int t = 0; t < count; ++t) {
                native_quant_act(format_, input.data() + size_t(first + t) * format_.n_embd, act.data() + size_t(t) * format_.act_bytes);
                job.nact[t] = act.data() + size_t(t) * format_.act_bytes;
                job.out[t] = output.data() + size_t(first + t) * format_.n_embd;
            }
            scaler.reset();
            pool.run_split_multi_native(selected_format, &job, 1);
            if (preserve_small_hidden_)
                for (int t = 0; t < count; ++t) scaler.restore(t, job.out[t], int(format_.n_embd));
            first += count;
        }
        if (!native_all_finite(output.data(),output.size()))
            throw std::runtime_error("nonfinite native partial output");
        return;
    }
};
}
