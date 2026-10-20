# Run from repository root: make -f experiment/glm_fp4_local/Makefile
BUILD ?= build-fp4-local
HIPCC ?= hipcc
CXX ?= c++
CC ?= cc
CPPFLAGS += -Iinclude
.PHONY: all
all: $(BUILD)/glm-fp4-rdna4-test
$(BUILD):
	mkdir -p $@
$(BUILD)/glm_fp4.o: src/host/glm_fp4.c include/strata/host/glm_fp4.h | $(BUILD)
	$(CC) -std=c11 -O2 $(CPPFLAGS) -c $< -o $@
$(BUILD)/gpu-hip.o: tools/glm_fp4_gpu.cu include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(HIPCC) -O2 -std=c++20 --offload-arch=gfx1201 $(CPPFLAGS) -c $< -o $@
$(BUILD)/rdna4-test.o: tools/glm_fp4_rdna4_test.cpp include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(CXX) -O2 -std=c++20 $(CPPFLAGS) -c $< -o $@
$(BUILD)/glm-fp4-rdna4-test: $(BUILD)/rdna4-test.o $(BUILD)/gpu-hip.o $(BUILD)/glm_fp4.o
	$(HIPCC) -x none $^ -o $@

NVCC ?= nvcc
UCOMM = tools/ucomm/build/libucomm.a
.PHONY: nodes
nodes: $(BUILD)/glm-fp4-moe-node-hip $(BUILD)/glm-fp4-moe-node-cuda
$(UCOMM):
	$(MAKE) -C tools/ucomm build/libucomm.a
$(BUILD)/moe-node.o: tools/glm_fp4_moe_node.cpp include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(CXX) -O2 -std=c++20 $(CPPFLAGS) -Itools/ucomm/include -c $< -o $@
$(BUILD)/gpu-cuda.o: tools/glm_fp4_gpu.cu include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(NVCC) -O2 -std=c++20 -arch=sm_120 $(CPPFLAGS) -c $< -o $@
$(BUILD)/glm-fp4-moe-node-hip: $(BUILD)/moe-node.o $(BUILD)/gpu-hip.o $(BUILD)/glm_fp4.o $(UCOMM)
	$(HIPCC) -x none $^ -ldl -pthread -o $@
$(BUILD)/glm-fp4-moe-node-cuda: $(BUILD)/moe-node.o $(BUILD)/gpu-cuda.o $(BUILD)/glm_fp4.o $(UCOMM)
	$(NVCC) $^ -ldl -Xcompiler -pthread -o $@

.PHONY: tree
tree: $(BUILD)/glm-fp4-tree-hip $(BUILD)/glm-fp4-tree-cuda
$(BUILD)/tree-node.o: tools/glm_fp4_tree_node.cpp tools/glm_fp4_service_lane.hpp tools/glm_fp4_registered_staging.hpp include/strata/host/glm_fp4_protocol.h include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(CXX) -O2 -std=c++20 $(CPPFLAGS) -Itools/ucomm/include -c $< -o $@
$(BUILD)/gpu-cuda-tree.o: tools/glm_fp4_gpu.cu include/strata/kernels/fp4_local.hpp include/strata/artifact/glm_fp4.hpp | $(BUILD)
	$(NVCC) -O2 -std=c++20 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 $(CPPFLAGS) -c $< -o $@
$(BUILD)/glm-fp4-tree-hip: $(BUILD)/glm_fp4_protocol.o $(BUILD)/tree-node.o $(BUILD)/gpu-hip.o $(BUILD)/glm_fp4.o $(UCOMM)
	$(HIPCC) -x none $^ -ldl -pthread -o $@
$(BUILD)/glm-fp4-tree-cuda: $(BUILD)/glm_fp4_protocol.o $(BUILD)/tree-node.o $(BUILD)/gpu-cuda-tree.o $(BUILD)/glm_fp4.o $(UCOMM)
	$(NVCC) $^ -ldl -Xcompiler -pthread -o $@

$(BUILD)/ib-probe: tools/glm_fp4_ib_probe.c $(UCOMM) | $(BUILD)
	$(CC) -O2 -std=c11 -Wall -Wextra -Itools/ucomm/include $^ -ldl -pthread -o $@

$(BUILD)/glm_fp4_protocol.o: src/host/glm_fp4_protocol.c include/strata/host/glm_fp4_protocol.h | $(BUILD)
	$(CC) -O2 -std=c11 -Wall -Wextra $(CPPFLAGS) -c $< -o $@

# Opt-in CPU-only NVFP4/Q8 experiment; no engine default changes.
.PHONY: cpu-q8
cpu-q8: $(BUILD)/glm-fp4-q8-cpu-test
$(BUILD)/glm-fp4-q8-cpu-test: tools/glm_fp4_q8_cpu_test.cpp src/kernels/cpu/fp4_local_q8.cpp include/strata/kernels/cpu/fp4_local_q8.hpp $(BUILD)/glm_fp4.o
	$(CXX) -O3 -mavx2 -mfma -std=c++20 -Wall -Wextra -fopenmp $(CPPFLAGS) tools/glm_fp4_q8_cpu_test.cpp src/kernels/cpu/fp4_local_q8.cpp $(BUILD)/glm_fp4.o -o $@

.PHONY: cpu-chain
cpu-chain: $(BUILD)/glm-fp4-cpu-chain
$(BUILD)/glm-fp4-cpu-chain: include/strata/kernels/cpu/fp4_local_w4a16.hpp tools/glm_fp4_cpu_chain.cpp src/kernels/cpu/fp4_local_q8.cpp src/kernels/cpu/fp4_local_w4a16.cpp include/strata/kernels/cpu/fp4_local_q8.hpp $(BUILD)/glm_fp4.o
	$(CXX) -O3 -mavx2 -mfma -std=c++20 -Wall -Wextra -fopenmp $(CPPFLAGS) tools/glm_fp4_cpu_chain.cpp src/kernels/cpu/fp4_local_q8.cpp src/kernels/cpu/fp4_local_w4a16.cpp $(BUILD)/glm_fp4.o -o $@

.PHONY: cpu-w4a16
cpu-w4a16: $(BUILD)/glm-fp4-w4a16-cpu-test
$(BUILD)/glm-fp4-w4a16-cpu-test: tools/glm_fp4_w4a16_cpu_test.cpp src/kernels/cpu/fp4_local_w4a16.cpp include/strata/kernels/cpu/fp4_local_w4a16.hpp $(BUILD)/glm_fp4.o
	$(CXX) -O3 -std=c++20 -Wall -Wextra $(CPPFLAGS) tools/glm_fp4_w4a16_cpu_test.cpp src/kernels/cpu/fp4_local_w4a16.cpp $(BUILD)/glm_fp4.o -o $@

.PHONY: cpu-projection
cpu-projection: $(BUILD)/glm-fp4-cpu-projection-test
$(BUILD)/glm-fp4-cpu-projection-test: tools/glm_fp4_cpu_projection_test.cpp src/kernels/cpu/fp4_local.cpp include/strata/kernels/cpu/fp4_local.hpp src/kernels/cpu/fp4_local_w4a16.cpp include/strata/kernels/cpu/fp4_local_w4a16.hpp $(BUILD)/glm_fp4.o
	$(CXX) -O3 -std=c++20 -Wall -Wextra -fopenmp $(CPPFLAGS) tools/glm_fp4_cpu_projection_test.cpp src/kernels/cpu/fp4_local.cpp src/kernels/cpu/fp4_local_w4a16.cpp $(BUILD)/glm_fp4.o -o $@

# CPU-only backend for the same bounded three-role protocol; no GPU runtime linked.
.PHONY: tree-cpu
tree-cpu: $(BUILD)/glm-fp4-tree-cpu
$(BUILD)/tree-cpu.o: tools/glm_fp4_tree_node.cpp tools/glm_fp4_service_lane.hpp tools/glm_fp4_registered_staging.hpp include/strata/kernels/cpu/fp4_local.hpp include/strata/host/glm_fp4_protocol.h | $(BUILD)
	$(CXX) -O2 -std=c++20 -DSTRATA_FP4_CPU_ONLY $(CPPFLAGS) -Itools/ucomm/include -c $< -o $@
$(BUILD)/cpu-projection.o: src/kernels/cpu/fp4_local.cpp include/strata/kernels/cpu/fp4_local.hpp include/strata/kernels/cpu/fp4_local_w4a16.hpp | $(BUILD)
	$(CXX) -O3 -std=c++20 -Wall -Wextra -fopenmp $(CPPFLAGS) -c $< -o $@
$(BUILD)/cpu-w4a16.o: src/kernels/cpu/fp4_local_w4a16.cpp include/strata/kernels/cpu/fp4_local_w4a16.hpp | $(BUILD)
	$(CXX) -O3 -std=c++20 -Wall -Wextra $(CPPFLAGS) -c $< -o $@
$(BUILD)/glm-fp4-tree-cpu: $(BUILD)/tree-cpu.o $(BUILD)/cpu-projection.o $(BUILD)/cpu-w4a16.o $(BUILD)/glm_fp4_protocol.o $(BUILD)/glm_fp4.o $(UCOMM)
	$(CXX) $^ -fopenmp -ldl -pthread -o $@

# Separate native-Q8 adapter test; intentionally not part of the W4A16 tree target.
# Supply a configured native CPU archive build and its GGML include directory.
.PHONY: cpu-native-adapter
cpu-native-adapter: | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O2 -std=c++20 $(CPPFLAGS) -I. -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_expert_test.cpp $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-adapter-test

.PHONY: cpu-native-batch
cpu-native-batch: | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O2 -std=c++20 $(CPPFLAGS) -I. -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_batch_test.cpp $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-batch-test

.PHONY: cpu-native-batch-tree
cpu-native-batch-tree: $(UCOMM) | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) -I. -Itools -Itools/ucomm/include -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_batch_tree.cpp $(UCOMM) $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-batch-tree

.PHONY: cpu-native-chain
cpu-native-chain: $(UCOMM) | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) -I. -Itools/ucomm/include -Ithird_party/ggml -I$(GGML_INCLUDE) tools/glm_fp4_native_chain.cpp $(UCOMM) $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-chain

.PHONY: cpu-native-quality
cpu-native-quality: $(BUILD)/cpu-projection.o $(BUILD)/cpu-w4a16.o | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 -fopenmp $(CPPFLAGS) -I. -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_quality.cpp $(BUILD)/cpu-projection.o $(BUILD)/cpu-w4a16.o $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-quality

.PHONY: cpu-native-scale-test
cpu-native-scale-test: | $(BUILD)
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) experiment/glm_fp4_local/native_hidden_scale_test.cpp -o $(BUILD)/glm-fp4-native-scale-test
	$(BUILD)/glm-fp4-native-scale-test

# Explicit native FP4/Q8 tree build; W4A16 and GPU targets are unchanged.
.PHONY: tree-native-cpu
tree-native-cpu: $(BUILD)/glm_fp4_protocol.o $(UCOMM) | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 -DSTRATA_FP4_CPU_ONLY -DSTRATA_FP4_NATIVE_CPU $(CPPFLAGS) -I. -Itools/ucomm/include -Ithird_party/ggml -I$(GGML_INCLUDE) tools/glm_fp4_tree_node.cpp $(BUILD)/glm_fp4_protocol.o $(UCOMM) $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-tree-native-cpu

.PHONY: tree-dispatch-test
tree-dispatch-test: | $(BUILD)
	$(CXX) -O3 -std=c++20 -pthread -I. experiment/glm_fp4_local/service_lane_test.cpp -o $(BUILD)/glm-fp4-service-lane-test
	$(BUILD)/glm-fp4-service-lane-test

.PHONY: cpu-native-numa
cpu-native-numa: | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) -I. -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_numa_test.cpp $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-numa-test

.PHONY: cpu-native-compact
cpu-native-compact: | $(BUILD)
	@test -n "$(NATIVE_BUILD)" -a -n "$(GGML_INCLUDE)" || { echo "Set NATIVE_BUILD and GGML_INCLUDE"; exit 1; }
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) -I. -Ithird_party/ggml -I$(GGML_INCLUDE) experiment/glm_fp4_local/native_cpu_compact_test.cpp $(NATIVE_BUILD)/libstrata_kernels_cpu.a $(NATIVE_BUILD)/libstrata_host.a $(NATIVE_BUILD)/ggml/src/libggml-cpu.a $(NATIVE_BUILD)/ggml/src/libggml-base.a -lpthread -ldl -lm -o $(BUILD)/glm-fp4-native-compact-test

.PHONY: cpu-native-finite-test
cpu-native-finite-test: | $(BUILD)
	$(CXX) -O3 -std=c++20 $(CPPFLAGS) experiment/glm_fp4_local/native_finite_test.cpp -o $(BUILD)/glm-fp4-native-finite-test
	$(BUILD)/glm-fp4-native-finite-test

.PHONY: registered-staging-test
registered-staging-test: | $(BUILD)
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -I. -Itools/ucomm/include experiment/glm_fp4_local/registered_staging_test.cpp -o $(BUILD)/registered-staging-test
	$(BUILD)/registered-staging-test
