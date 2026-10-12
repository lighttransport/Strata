cmake_minimum_required(VERSION 3.24)
# Standalone MXFP4 decode/GEMM benchmarks for MiMo-V2.6-Flash. Not part of the Strata build.
#   cmake -S experiment/fp4 -B experiment/fp4/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
#   cmake --build experiment/fp4/build -j
project(fp4_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(bench_cpu cpu/bench_cpu.cpp)
target_compile_options(bench_cpu PRIVATE -O3 -mavx2 -mfma -mf16c)
target_link_libraries(bench_cpu PRIVATE OpenMP::OpenMP_CXX)

option(FP4_CUDA "Build the GPU benchmark" ON)
if(FP4_CUDA)
  # sm_120a: the software paths only need sm_80 (int8 mma) / sm_89 (fp8 mma); the native
  # block-scaled FP4 instructions in gpu/mma_native.cu need the arch-specific 120a target.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120a)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_executable(bench_gpu gpu/bench_gpu.cu gpu/mma_native.cu gpu/mma_peak.cu)
  target_compile_options(bench_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr -Xcompiler=-fopenmp>)
  target_link_libraries(bench_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()
