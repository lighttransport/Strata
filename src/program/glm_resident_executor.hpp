// Shared numerical executor for local residents and isolated GPU workers.
#pragma once
struct ResidentExecutor {
    struct Graph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        const float *input = nullptr;
        const int *ids = nullptr;
        ~Graph() {
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
        }
    };
    static constexpr size_t entries = cpu::MAXT * 8;
    std::unique_ptr<Device> metadata, activation, quantized, scratch, output;
    std::unique_ptr<Pinned> host_metadata, host_activation, host_output;
    // Resident geometry is fixed; expert addresses and route rows are data in
    // pinned metadata, so replacements do not invalidate these graphs.
    using GraphKey = std::tuple<int, int, float, float, int, size_t, size_t, bool, bool>;
    std::map<GraphKey, std::unique_ptr<Graph>> graphs;
    std::map<std::tuple<int, bool, int>, std::unique_ptr<Graph>> device_graphs;
    std::unique_ptr<Device> lookup, route_ids;
    std::unique_ptr<Pinned> host_lookup;
    std::unique_ptr<Pinned> host_reduction;
    std::unique_ptr<Device> partial_weights, partial_sum;
    std::unique_ptr<Pinned> host_partial_weights, host_partial_sum;
    bool lookup_preloaded = false;
    struct ReaderEvent {
        cudaEvent_t event = nullptr;
        ReaderEvent() { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        ~ReaderEvent() { if (event) cudaEventDestroy(event); }
    };
    std::map<int, std::unique_ptr<ReaderEvent>> readers;
    std::map<int, std::unique_ptr<Pinned>> reductions;
    cudaEvent_t reader_done(int layer) {
        auto &entry = readers[layer];
        if (!entry) entry = std::make_unique<ReaderEvent>();
        return entry->event;
    }
    void record_reader(int layer, cudaStream_t stream) {
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(stream, &status));
        check(cudaEventRecordWithFlags(reader_done(layer), stream,
              status == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
    }
    Pinned &reduction_for(int layer) {
        auto &entry = reductions[layer];
        if (!entry) entry = std::make_unique<Pinned>(entries * 4096 * sizeof(float) + entries * sizeof(int));
        return *entry;
    }
    ResidentExecutor() {
        DeviceTag tag("resident_exec");
        const size_t meta_bytes = entries * 8 + (3 * entries + 2) * 4;
        metadata = std::make_unique<Device>(meta_bytes);
        activation = std::make_unique<Device>(cpu::MAXT * 4096 * 4);
        quantized = std::make_unique<Device>(k::native_q8_1_bytes(4096, cpu::MAXT));
        auto canonical = k::native_expert_layout(10, 11, 4096, 2048);
        scratch = std::make_unique<Device>(std::max(k::native_expert_scratch_bytes(entries, 2048),
                                                    k::glm_q23_scratch_bytes(canonical, cpu::MAXT, entries)));
        output = std::make_unique<Device>(entries * 4096 * 4);
        host_metadata = std::make_unique<Pinned>(meta_bytes);
        host_activation = std::make_unique<Pinned>(cpu::MAXT * 4096 * 4);
        host_output = std::make_unique<Pinned>(output->bytes);
        lookup = std::make_unique<Device>(64 * 288 * sizeof(unsigned long long));
        host_lookup = std::make_unique<Pinned>(lookup->bytes);
        std::memset(host_lookup->p, 0, lookup->bytes);
        route_ids = std::make_unique<Device>(entries * sizeof(int));
        host_reduction = std::make_unique<Pinned>(entries * 4096 * sizeof(float) + entries * sizeof(int));
        check(cudaMemset(scratch->p, 0, scratch->bytes));
    }
    void update_lookup(int layer, int expert, void *address, cudaStream_t stream) {
        const size_t offset = (size_t(layer) * 288 + expert) * sizeof(unsigned long long);
        auto *cell = (unsigned long long *)((char *)host_lookup->p + offset);
        *cell = (unsigned long long)address;
        // Readers finish before slot replacement updates pinned lookup cells.
        (void)stream;
    }
    // Canonical layers (canon_expert.hpp) take the Q2_K/Q3_K kernels whose rows equal the CPU pool's bit for bit.
    void expert_rows(const k::NativeExpertLayout &layout, bool canon, const unsigned long long *ptr, const int *starts,
                     const int *count, const int *dest, const int *tokens, size_t groups, size_t rows, const float *x,
                     int nt, cudaStream_t stream) {
        if (canon) {
            k::glm_q23_expert_grouped(layout, ptr, starts, count, dest, tokens, groups, rows, x, nt, scratch->p,
                                      output->f(), stream);
            return;
        }
        k::native_quantize_q8_1(x, quantized->p, 4096, nt, stream);
        k::native_expert_grouped(layout, ptr, starts, count, dest, tokens, groups, rows, quantized->p, scratch->p,
                                 output->f(), stream, 0, nt);
    }
    void run_device(const k::NativeExpertLayout &layout, const float *x, const int *ids,
                    int layer, cudaStream_t stream, bool use_graphs, bool readback = true, int nt = 1,
                    bool canon = false) {
        if (nt < 1 || nt > cpu::MAXT) throw std::runtime_error("GLM: invalid resident batch");
        auto *ptr = (unsigned long long *)metadata->p;
        auto *starts = (int *)(ptr + entries);
        auto *dest = starts + entries + 1, *tokens = dest + entries, *count = tokens + entries;
        auto enqueue = [&] {
            const size_t offset = (size_t)layer * 288 * sizeof(unsigned long long);
            const void *source = (char *)host_lookup->p + offset;
            if (!lookup_preloaded)
                check(cudaMemcpyAsync((char *)lookup->p + offset, source,
                                      288 * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
            if (nt == 1)
                k::glm_resident_routes(ids, (const unsigned long long *)lookup->p + layer * 288,
                                       ptr, starts, dest, tokens, count, stream);
            else k::glm_resident_routes_batch(ids, (const unsigned long long *)lookup->p + layer * 288,
                                              ptr, starts, dest, tokens, count, nt, stream);
            expert_rows(layout, canon, ptr, starts, count, dest, tokens, nt * 8, nt * 8, x, nt, stream);
            if (readback)
                check(cudaMemcpyAsync(host_output->p, output->p, (size_t)nt * 8 * 4096 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        };
        if (!use_graphs) { enqueue(); return; }
        auto &graph = device_graphs[{layer, readback, nt + (canon ? 100 : 0)}];
        if (graph && (graph->input != x || graph->ids != ids)) graph.reset();
        if (!graph) {
            auto capture = std::make_unique<Graph>();
            capture->input = x; capture->ids = ids;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                throw;
            }
            check(cudaStreamEndCapture(stream, &capture->graph));
            check(cudaGraphInstantiate(&capture->executable, capture->graph, nullptr, nullptr, 0));
            graph = std::move(capture);
        }
        check(cudaGraphLaunch(graph->executable, stream));
    }
    void run(const k::NativeExpertLayout &layout, const std::map<int, std::vector<int>> &hits,
             const std::function<void *(int)> &address, const float *x, int nt, cudaStream_t stream,
             bool use_graphs = false, bool canon = false, bool batched_graphs = false, bool readback = true,
             bool fixed_geometry = false, const float *reduction_weights = nullptr) {
        if (reduction_weights) {
            if (!partial_weights) {
                DeviceTag tag("resident_exec");
                partial_weights = std::make_unique<Device>(entries * sizeof(float));
                partial_sum = std::make_unique<Device>(cpu::MAXT * 4096 * sizeof(float));
                host_partial_weights = std::make_unique<Pinned>(partial_weights->bytes);
                host_partial_sum = std::make_unique<Pinned>(partial_sum->bytes);
            }
            if (nt < 1 || nt > cpu::MAXT) throw std::runtime_error("GLM: invalid resident reduction batch");
            std::memcpy(host_partial_weights->p, reduction_weights, size_t(nt)*8*sizeof(float));
        }
        auto *ptr = (unsigned long long *)host_metadata->p;
        auto *starts = (int *)(ptr + entries);
        auto *dest = starts + entries + 1;
        auto *tokens = dest + entries;
        auto *count = tokens + entries;
        size_t groups = 0, rows = 0;
        starts[0] = 0;
        for (const auto &[expert, indices] : hits) {
            ptr[groups] = (unsigned long long)address(expert);
            for (int j : indices) {
                dest[rows] = j; tokens[rows++] = j / 8;
            }
            starts[++groups] = rows;
        }
        if (!groups || groups > entries || rows > entries || nt > cpu::MAXT)
            throw std::runtime_error("GLM: resident decode workspace geometry exceeded");
        *count = groups;
        // Counts remain request data. Kernel capacities can stay fixed so a
        // long worker session does not capture every groups/rows combination.
        const size_t launch_groups=use_graphs && batched_graphs && fixed_geometry ? size_t(nt)*8 : groups;
        const size_t launch_rows=use_graphs && batched_graphs && fixed_geometry ? size_t(nt)*8 : rows;
        std::memcpy(host_activation->p, x, (size_t)nt * 4096 * 4);
        const auto offset = [&](const void *p) { return (const char *)p - (const char *)host_metadata->p; };
        auto *base = (char *)metadata->p;
        auto enqueue = [&] {
            metadata->put_async(host_metadata->p, metadata->bytes, stream);
            activation->put_async(host_activation->p, (size_t)nt * 4096 * 4, stream);
            if (reduction_weights)
                partial_weights->put_async(host_partial_weights->p, size_t(nt)*8*sizeof(float), stream);
            expert_rows(layout, canon, (const unsigned long long *)base,
                (const int *)(base + offset(starts)), (const int *)(base + offset(count)),
                (const int *)(base + offset(dest)), (const int *)(base + offset(tokens)),
                launch_groups, launch_rows, activation->f(), nt, stream);
            if (readback) check(cudaMemcpyAsync(host_output->p, output->p, (size_t)nt * 8 * 4096 * 4,
                                                 cudaMemcpyDeviceToHost, stream));
            if (reduction_weights) {
                k::glm_sum_selected_rows(output->f(), partial_weights->f(), partial_sum->f(),4096,8,nt,stream);
                check(cudaMemcpyAsync(host_partial_sum->p, partial_sum->p, size_t(nt)*4096*sizeof(float),
                                      cudaMemcpyDeviceToHost,stream));
            }
        };
        if (!use_graphs || (!batched_graphs && (nt != 1 || rows != groups))) {
            enqueue(); return;
        }
        auto &graph = graphs[{layout.gu_type + (canon ? 1000 : 0), layout.d_type, layout.swiglu_limit, layout.gate_skip, nt, launch_groups, launch_rows, readback, reduction_weights != nullptr}];
        if (!graph) {
            auto captured = std::make_unique<Graph>();
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
                cudaGraph_t invalid = nullptr;
                cudaStreamEndCapture(stream, &invalid);
                if (invalid) cudaGraphDestroy(invalid);
                throw;
            }
            check(cudaStreamEndCapture(stream, &captured->graph));
            check(cudaGraphInstantiate(&captured->executable, captured->graph, nullptr, nullptr, 0));
            graph = std::move(captured);
        }
        check(cudaGraphLaunch(graph->executable, stream));
    }
    void collect(const std::map<int, std::vector<int>> &hits, std::vector<float> &results) const {
        for (const auto &[expert, indices] : hits)
            for (int j : indices)
                std::copy_n((const float *)host_output->p + j * 4096, 4096, results.data() + j * 4096);
    }
};
