HIPCC ?= hipcc
CXX ?= g++
ARCH ?= gfx1201
BUILD := build
HIPFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH) -Wall -Wextra -Wpedantic

.PHONY: all cpu test bench isa
all: $(BUILD)/nvfp4_rdna4 $(BUILD)/test_numeric

$(BUILD):
	mkdir -p $@

$(BUILD)/nvfp4_rdna4: bench.hip numeric.hpp kernels.hpp Makefile | $(BUILD)
	$(HIPCC) $(HIPFLAGS) bench.hip -o $@

$(BUILD)/test_numeric: bench.hip numeric.hpp kernels.hpp Makefile | $(BUILD)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -DCPU_ONLY -x c++ bench.hip -o $@

cpu: $(BUILD)/test_numeric
	./$(BUILD)/test_numeric

test: all
	./$(BUILD)/test_numeric
	./$(BUILD)/nvfp4_rdna4 --test

bench: all
	./$(BUILD)/nvfp4_rdna4 --bench --output $(BUILD)/timings.csv

isa: | $(BUILD)
	$(HIPCC) $(HIPFLAGS) --cuda-device-only -S bench.hip -o $(BUILD)/kernels.s
	rg 'v_wmma_.*(iu4|iu8|fp8|bf16)' $(BUILD)/kernels.s
