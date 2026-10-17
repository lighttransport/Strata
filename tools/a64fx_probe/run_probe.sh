#!/bin/bash
# tools/a64fx_probe/run_probe.sh - build and run the A64FX probes inside an allocation (compute node, native fcc/FCC).
#
#   LLAMA_SRC=$HOME/work/llama.cpp STRATA=$HOME/work/Strata tools/a64fx_probe/run_probe.sh [bench args...]
#
# 1. builds ggml (ggml-base, ggml-cpu) from LLAMA_SRC with the gemm a64fx recipe into /local (node-local; falls back
#    to $STRATA/build-a64fx-probe when /local is absent) unless LLAMA_BUILD already has the libraries;
# 2. compiles and runs the C11 strata_host tests (names, env switches, expert-pack rule, stage) natively;
# 3. compiles q23_sve_bench and runs the correctness check and the bandwidth sweep at 1, 12 and 48 threads on the
#    GLM-5.3-Flash expert shapes (gate/up 2048 x 4096, down 4096 x 2048) and the DS4.1F shapes.
# Every result line starts with A64FX_ so a log can be grepped; the script never writes outside LLAMA_BUILD and $OUT.
set -uo pipefail
STRATA=${STRATA:-$HOME/work/Strata}
LLAMA_SRC=${LLAMA_SRC:-$HOME/work/llama.cpp}
if [ -d /local ]; then ROOT=/local/strata-a64fx-probe-${PJM_JOBID:-nojob}; else ROOT=$STRATA/build-a64fx-probe; fi
LLAMA_BUILD=${LLAMA_BUILD:-$ROOT/ggml}
OUT=${OUT:-$STRATA/tmp/a64fx-probe-${PJM_JOBID:-nojob}}
JOBS=${JOBS:-16}
mkdir -p "$ROOT/tmp" "$OUT"; export TMPDIR=$ROOT/tmp
CC=${A64FX_CC:-fcc}; CXX=${A64FX_CXX:-FCC}
CFLAGS="-Nclang -O3 -fopenmp -march=armv8.2-a+sve"
# FCC 4.12 (LLVM 7) crashes on auto-vectorized FP16 code in ggml-cpu's C++ files (gemm uses `#pragma clang optimize off`
# for its diagnostic llama build). The quantized dot kernels this probe measures are C with explicit SVE intrinsics, so
# only the C++ side gets the workaround: no auto-vectorization by default, or the gemm pragma with GGML_CXX_WORKAROUND=pragma.
GGML_C_FLAGS="-Nclang -O3 -fopenmp"
case "${GGML_CXX_WORKAROUND:-novec}" in
  novec)  GGML_CXX_FLAGS="-Nclang -O3 -fopenmp -fno-vectorize -fno-slp-vectorize" ;;
  pragma) GGML_CXX_FLAGS="-Nclang -O3 -fopenmp -include $HOME/work/gemm/glm53f/a64fx/glm5/llama_a64fx_disable_vectorization.h" ;;
  none)   GGML_CXX_FLAGS="-Nclang -O3 -fopenmp" ;;
esac
echo "A64FX_PROBE host=$(hostname) job=${PJM_JOBID:-none} cpus=$(nproc) strata=$STRATA llama=$LLAMA_SRC build=$LLAMA_BUILD out=$OUT"
command -v $CC >/dev/null || { echo "A64FX_PROBE FAIL: $CC not found (run on a compute node)"; exit 1; }

# ---- 1. ggml
if [ ! -f "$LLAMA_BUILD/bin/libggml-cpu.so" ] && [ ! -f "$LLAMA_BUILD/bin/libggml-cpu.a" ]; then
  echo "A64FX_PROBE building ggml from $LLAMA_SRC"
  cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DGIT_EXECUTABLE=/bin/false \
    -DCMAKE_C_FLAGS="$GGML_C_FLAGS" -DCMAKE_CXX_FLAGS="$GGML_CXX_FLAGS" \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH='armv8.2-a+sve' -DGGML_CPU=ON -DGGML_OPENMP=ON -DGGML_CCACHE=OFF \
    -DGGML_CPU_HBM=OFF -DGGML_CPU_KLEIDIAI=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_LLGUIDANCE=OFF -DLLAMA_ALL_WARNINGS=OFF > "$OUT/ggml-cmake.log" 2>&1 \
    || { tail -20 "$OUT/ggml-cmake.log"; echo "A64FX_PROBE FAIL ggml cmake"; exit 1; }
  ninja -C "$LLAMA_BUILD" -j "$JOBS" ggml-base ggml-cpu > "$OUT/ggml-build.log" 2>&1 || { tail -30 "$OUT/ggml-build.log"; echo "A64FX_PROBE FAIL ggml build"; exit 1; }
fi
echo "A64FX_GGML PASS $(ls "$LLAMA_BUILD"/bin/libggml-*.so* 2>/dev/null | tr '\n' ' ')"
FJLIB=$(dirname "$(command -v $CC)")/../lib64
export LD_LIBRARY_PATH=$LLAMA_BUILD/bin:$FJLIB:${LD_LIBRARY_PATH:-}   # libfjomphk.so lives next to the compiler

# ---- 2. strata_host (C11) and its tests
cd "$STRATA"
HOST_SRC="src/host/glm_names.c src/host/glm_env.c src/host/glm_expert_pack.c src/host/glm_mixed.c src/host/glm_layer.c src/host/ds41f_names.c"
for t in tests/host/glm_names_test.c tests/host/glm_env_test.c tests/host/glm_expert_pack_test.c; do
  [ -f "$t" ] || { echo "A64FX_HOST SKIP $t (absent)"; continue; }
  n=$(basename "$t" .c)
  if $CC -Nclang -O2 -std=c11 -Iinclude $HOST_SRC "$t" -o "$ROOT/$n" > "$OUT/$n.build.log" 2>&1 && "$ROOT/$n" > "$OUT/$n.log" 2>&1; then
    echo "A64FX_HOST PASS $n"; else echo "A64FX_HOST FAIL $n"; tail -5 "$OUT/$n.build.log" "$OUT/$n.log" 2>/dev/null; fi
done

# ---- 3. the kernel bench
$CXX $CFLAGS -std=c++17 -I"$LLAMA_SRC/ggml/include" tools/a64fx_probe/q23_sve_bench.cpp -L"$LLAMA_BUILD/bin" -lggml-cpu -lggml-base -o "$ROOT/q23_sve_bench" > "$OUT/bench.build.log" 2>&1 \
  || { tail -20 "$OUT/bench.build.log"; echo "A64FX_PROBE FAIL bench build"; exit 1; }
echo "A64FX_BENCH_BUILD PASS"
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
B="$ROOT/q23_sve_bench"
"$B" --check --threads 1 --mb 64 --reps 1 --rows 64 --cols 4096 "$@" | tee "$OUT/check.log"
for th in 1 12 48; do
  "$B" --threads $th --mb $((th == 1 ? 256 : 2048)) --reps 3 --rows 2048 --cols 4096 "$@" | tee -a "$OUT/bench-glm-gateup.log"
  "$B" --threads $th --mb $((th == 1 ? 256 : 2048)) --reps 3 --rows 4096 --cols 2048 --types q2_k,q3_k,q4_k,iq2_s "$@" | tee -a "$OUT/bench-glm-down.log"
done
"$B" --threads 48 --mb 2048 --reps 3 --rows 2304 --cols 5120 --types q2_k,iq2_xxs,iq1_s --tokens 1 | tee "$OUT/bench-ds41f.log"
"$B" --threads 48 --mb 2048 --reps 3 --rows 2304 --cols 5120 --types q2_k,iq2_xxs,iq1_s --tokens 4 | tee -a "$OUT/bench-ds41f.log"
echo "A64FX_PROBE DONE out=$OUT"
