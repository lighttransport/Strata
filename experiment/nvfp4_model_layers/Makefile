HIPCC ?= hipcc
FLAGS := -O3 -std=c++17 --offload-arch=gfx1201 -Wall -Wextra -Wpedantic
.PHONY: all fetch bench isa
all: build/layers
build:
	mkdir -p $@
build/layers: layers.hip ../nvfp4_rdna4/w4a16.hip ../nvfp4_rdna4/numeric.hpp Makefile | build
	$(HIPCC) $(FLAGS) layers.hip -o $@
fetch: | build
	python3 fetch.py --output build/model
bench: all
	./build/layers build/model/layers.bin build/timings
isa: | build
	$(HIPCC) $(FLAGS) --cuda-device-only -S layers.hip -o build/layers.s
	rg 'v_wmma_f32_.*(f16|bf16)' build/layers.s
