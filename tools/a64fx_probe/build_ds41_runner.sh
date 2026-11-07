#!/usr/bin/env bash
# Standalone Linux/A64FX build; site configuration belongs in the environment.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/build/ds41-cpu"}
: "${UTOFU_DIR:?Set UTOFU_DIR to the directory containing tp_allreduce.h}"
cc=${DS41_CC:-mpiclang}
cxx=${DS41_CXX:-mpiclang++}
read -r -a cflags <<< "${DS41_CFLAGS:--O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp -D_GNU_SOURCE}"
read -r -a cxxflags <<< "${DS41_CXXFLAGS:--std=c++20 -stdlib=libc++}"
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$repo"
"$cc" "${cflags[@]}" -Iinclude -c src/kernels/cpu/ds41_sve.c -o "$out/kernel.o"
for unit in glm_fp4 glm_names glm_env glm_expert_pack; do
    "$cc" "${cflags[@]}" -Iinclude -c "src/host/$unit.c" -o "$out/$unit.o"
done
"$cc" "${cflags[@]}" -Iinclude -I"$UTOFU_DIR" -c src/cpu_runner/comm_utofu.c -o "$out/comm.o"
"$cxx" "${cflags[@]}" "${cxxflags[@]}" -Iinclude \
    -DSTRATA_CPU_RUNNER_MPI -DSTRATA_CPU_RUNNER_UTOFU \
    -c src/cpu_runner/ds41_cpu_runner.cpp -o "$out/runner.o"
"$cxx" "${cflags[@]}" "${cxxflags[@]}" "$out/runner.o" "$out/kernel.o" \
    "$out/comm.o" "$out/glm_fp4.o" "$out/glm_names.o" "$out/glm_env.o" \
    "$out/glm_expert_pack.o" -ltofucom -o "$out/ds41_cpu_runner"
"$cxx" "${cflags[@]}" "${cxxflags[@]}" -Iinclude -pthread \
    tools/a64fx_probe/row_prefetch_test.cpp -o "$out/row_prefetch_test"
"$out/row_prefetch_test"
"$cxx" "${cflags[@]}" "${cxxflags[@]}" -Iinclude \
    -DSTRATA_CPU_RUNNER_MPI -DSTRATA_CPU_RUNNER_UTOFU \
    tools/a64fx_probe/ds41_a2a_test.cpp "$out/comm.o" "$out/glm_env.o" \
    -ltofucom -o "$out/ds41_a2a_test"
for probe in ds41_kernel_test ds41_iq2r_test ds41_prefill_test; do
    "$cc" "${cflags[@]}" -Iinclude "tools/a64fx_probe/$probe.c" \
        "$out/kernel.o" -lm -lpthread -o "$out/$probe"
done
printf 'Built %s\n' "$out/ds41_cpu_runner"
