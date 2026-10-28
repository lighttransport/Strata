#!/usr/bin/env bash
cd "$(dirname "$0")"; R=/mnt/nvme02/work/Strata
out=$(/opt/rocm/core-10.1/bin/hipcc -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1201 -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 \
  -include $R/include/strata/hip_compat/cuda_runtime.h -I$R/third_party/ggml -I$R/include -I$R/include/strata/hip_compat -I. -DCR=8 -DCE=2 -DRW=4 \
  -o kbench kbench.hip 2>&1)
if echo "$out" | grep -q ' error'; then echo "$out" | grep -A3 ' error' | head -20; exit 1; fi
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ./kbench
