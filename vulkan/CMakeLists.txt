find_package(Vulkan REQUIRED COMPONENTS glslc)
find_package(SPIRV-Headers CONFIG REQUIRED)
# The pinned GGML checkout discovers SPIR-V headers but does not propagate their
# include path to its Vulkan target when the SDK is outside the system prefix.
target_link_libraries(ggml-vulkan PRIVATE SPIRV-Headers::SPIRV-Headers)
set(_glm_vk_shaders kda kda_fused mla mhc expert_dot mla_batch index_scores mla_batch_wide index_scores_wide)
foreach(_shader IN LISTS _glm_vk_shaders)
  set(_shader_source ${_shader})
  set(_shader_flags)
  if(_shader MATCHES "_wide$")
    string(REGEX REPLACE "_wide$" "" _shader_source ${_shader})
    set(_shader_flags -DSTRATA_WIDE_DISPATCH=1)
  endif()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_shader}.spv ${CMAKE_CURRENT_BINARY_DIR}/${_shader}_spv.hpp
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -O ${_shader_flags} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_shader_source}.comp -o ${CMAKE_CURRENT_BINARY_DIR}/${_shader}.spv
    COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_BINARY_DIR}/${_shader}.spv -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${_shader}_spv.hpp -DNAME=${_shader} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_shader_source}.comp ${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake VERBATIM)
endforeach()
add_custom_target(glm_vulkan_shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/index_scores_wide.spv ${CMAKE_CURRENT_BINARY_DIR}/mla_batch_wide.spv ${CMAKE_CURRENT_BINARY_DIR}/index_scores.spv ${CMAKE_CURRENT_BINARY_DIR}/mla_batch.spv ${CMAKE_CURRENT_BINARY_DIR}/expert_dot.spv ${CMAKE_CURRENT_BINARY_DIR}/kda_fused.spv ${CMAKE_CURRENT_BINARY_DIR}/kda.spv ${CMAKE_CURRENT_BINARY_DIR}/mla.spv ${CMAKE_CURRENT_BINARY_DIR}/mhc.spv)
add_library(strata_glm_vulkan runtime.cpp projections.cpp)
target_include_directories(strata_glm_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(strata_glm_vulkan PUBLIC Vulkan::Vulkan ggml ggml-vulkan strata_artifact)
add_dependencies(strata_glm_vulkan glm_vulkan_shaders)
add_executable(strata-glm-vulkan decode.cpp)
target_link_libraries(strata-glm-vulkan PRIVATE strata_glm_vulkan strata_kernels_cpu)
add_executable(glm_vulkan_parity parity.cpp)
target_link_libraries(glm_vulkan_parity PRIVATE strata_glm_vulkan strata_kernels_cpu)
add_executable(glm_vulkan_bench bench.cpp)
target_link_libraries(glm_vulkan_bench PRIVATE strata_glm_vulkan strata_kernels_cpu)
if(STRATA_BUILD_TESTS)
  add_test(NAME glm_vulkan_parity COMMAND glm_vulkan_parity)
  set_tests_properties(glm_vulkan_parity PROPERTIES LABELS "glm;vulkan" TIMEOUT 120)
endif()

option(STRATA_VULKAN_NATIVE_HOST "Compile Vulkan host loops for the build CPU" OFF)
if(STRATA_VULKAN_NATIVE_HOST)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
    target_compile_options(strata_glm_vulkan PRIVATE -march=native -ffp-contract=off -fno-math-errno)
    target_compile_options(strata-glm-vulkan PRIVATE -march=native -ffp-contract=off -fno-math-errno)
  else()
    message(FATAL_ERROR "STRATA_VULKAN_NATIVE_HOST currently requires GCC or Clang")
  endif()
endif()

# This check submits no GPU work and is safe before a device precheck.
add_executable(glm_vulkan_executor_check executor_check.cpp)
target_link_libraries(glm_vulkan_executor_check PRIVATE strata_kernels_cpu)

# Opt-in fast engine (--engine fast). Shaders are listed as name|source|defines
# and embedded into one registry header; the strict path above is unchanged.
set(_glm_vk_fast_shaders
  "hw_memory|hw_diag|-DHW_MEMORY" "hw_alu|hw_diag|-DHW_ALU" "hw_gemm|hw_diag|-DHW_GEMM"
  "gemm_q4k|gemm_tiled|-DTYPE_Q4K" "gemm_q6k|gemm_tiled|-DTYPE_Q6K" "gemm_q50|gemm_tiled|-DTYPE_Q50"
  "gemm_q2k|gemm_tiled|-DTYPE_Q2K" "gemm_q3k|gemm_tiled|-DTYPE_Q3K" "gemm_f32|gemm_tiled|-DTYPE_F32"
  "gemm_q4k_wide|gemm_tiled|-DTYPE_Q4K -DNTILE=128" "gemm_q6k_wide|gemm_tiled|-DTYPE_Q6K -DNTILE=128" "gemm_q50_wide|gemm_tiled|-DTYPE_Q50 -DNTILE=128"
  "gemm_q2k_wide|gemm_tiled|-DTYPE_Q2K -DNTILE=128" "gemm_q3k_wide|gemm_tiled|-DTYPE_Q3K -DNTILE=128" "gemm_f32_wide|gemm_tiled|-DTYPE_F32 -DNTILE=128"
  "gemm_q4k_wider|gemm_tiled|-DTYPE_Q4K -DNTILE=256" "gemm_q6k_wider|gemm_tiled|-DTYPE_Q6K -DNTILE=256" "gemm_q50_wider|gemm_tiled|-DTYPE_Q50 -DNTILE=256"
  "gemm_q2k_wider|gemm_tiled|-DTYPE_Q2K -DNTILE=256" "gemm_q3k_wider|gemm_tiled|-DTYPE_Q3K -DNTILE=256" "gemm_f32_wider|gemm_tiled|-DTYPE_F32 -DNTILE=256"
  "q8k_round|q8k_round|" "expert_mix_canon|expert_mix|-DCANON_SWIGLU"
  "q8k_round_portable|q8k_round|-DPORTABLE_REDUCTION"
  "q8k_codes|q8k_round|-DCODES_AND_SUMS" "q8k_codes_portable|q8k_round|-DCODES_AND_SUMS -DPORTABLE_REDUCTION"
  "q8k_unpack|q8k_unpack|"
  "repack_expert_q2k|repack_expert|-DTYPE_Q2K" "repack_expert_q3k|repack_expert|-DTYPE_Q3K"
  "repack_expert_q2k_checked|repack_expert|-DTYPE_Q2K -DCHECK_SOURCE_BOUNDS"
  "repack_expert_q3k_checked|repack_expert|-DTYPE_Q3K -DCHECK_SOURCE_BOUNDS"
  "repack_expert_q3k_checked_legacy|repack_expert|-DTYPE_Q3K -DCHECK_SOURCE_BOUNDS -DLEGACY_WIDE_SIDE"
  "fma_oracle|fma_oracle|"
  "expert_gemm_q2k_canon|expert_gemm_canon|-DTYPE_Q2K" "expert_gemm_q3k_canon|expert_gemm_canon|-DTYPE_Q3K"
  "expert_gemm_q2k_canon_vector|expert_gemm_canon|-DTYPE_Q2K -DVECTOR_WEIGHT_LOADS"
  "expert_gemm_q3k_canon_vector|expert_gemm_canon|-DTYPE_Q3K -DVECTOR_WEIGHT_LOADS"
  "expert_reduce_canon|expert_reduce_canon|"
  "trace_window|trace_window|"
  "route_merge|route_merge|"
  "gemv_rows_q4k|gemv_rows|-DTYPE_Q4K" "gemv_rows_q6k|gemv_rows|-DTYPE_Q6K" "gemv_rows_q50|gemv_rows|-DTYPE_Q50"
  "gemv_f32|gemv_f32|" "reduce_parts|reduce_parts|" "touch|touch|" "rmsnorm|rmsnorm|" "mhc|mhc|"
  "kda_step|kda_step|" "swiglu|swiglu|" "router|router|" "index_key|index_key|"
  "index_scores_step|index_scores_step|" "select_pools|select_pools|" "mla_step|mla_step|" "argmax|argmax|" "q8_round|q8_round|" "mhc_read|mhc_read|" "mhc_write|mhc_write|"
  "gemv_expert_q2k|gemv_expert|-DTYPE_Q2K" "gemv_expert_q3k|gemv_expert|-DTYPE_Q3K" "expert_mix|expert_mix|" "xsum16|xsum16|" "hits_union|hits_union|" "residual_add|residual_add|" "hidden_store|hidden_store|")
foreach(_type IN ITEMS q2k q3k)
  string(TOUPPER "${_type}" _type_define)
  foreach(_columns IN ITEMS 16 32)
    list(APPEND _glm_vk_fast_shaders
      "expert_gemm_${_type}_canon_${_columns}|expert_gemm_canon|-DTYPE_${_type_define} -DNTILE=${_columns}"
      "expert_gemm_${_type}_canon_vector_${_columns}|expert_gemm_canon|-DTYPE_${_type_define} -DVECTOR_WEIGHT_LOADS -DNTILE=${_columns}")
  endforeach()
endforeach()
foreach(_type IN ITEMS q2k q3k)
  string(TOUPPER "${_type}" _type_define)
  foreach(_columns IN ITEMS 16 32 64)
    if(_columns EQUAL 64)
      set(_column_suffix "")
    else()
      set(_column_suffix "_${_columns}")
    endif()
    list(APPEND _glm_vk_fast_shaders
      "expert_gemm_${_type}_canon${_column_suffix}_m32|expert_gemm_canon|-DTYPE_${_type_define} -DNTILE=${_columns} -DMTILE=32"
      "expert_gemm_${_type}_canon_vector${_column_suffix}_m32|expert_gemm_canon|-DTYPE_${_type_define} -DVECTOR_WEIGHT_LOADS -DNTILE=${_columns} -DMTILE=32")
  endforeach()
endforeach()
option(STRATA_VULKAN_NATIVE_FMA "Build optional VK_KHR_shader_fma expert variants" OFF)
if(STRATA_VULKAN_NATIVE_FMA)
  list(APPEND _glm_vk_fast_shaders
    "expert_mix_fp32_math_native_fma|expert_mix|-DCANON_SWIGLU -DFP32_MATH"
    "q8k_codes_fp32_math|q8k_round|-DCODES_AND_SUMS -DFP32_MATH"
    "q8k_codes_portable_fp32_math|q8k_round|-DCODES_AND_SUMS -DPORTABLE_REDUCTION -DFP32_MATH"
    "fma_oracle_fp64_scale_native_fma|fma_oracle|-DFP64_SCALE -DNATIVE_FMA"
    "expert_mix_fp64_math_native_fma|expert_mix|-DCANON_SWIGLU -DFP64_MATH -DNATIVE_FMA")
  foreach(_type IN ITEMS q2k q3k)
    string(TOUPPER "${_type}" _type_define)
    foreach(_loader IN ITEMS "" "_vector")
      set(_loader_define "")
      if(_loader STREQUAL "_vector")
        set(_loader_define "-DVECTOR_WEIGHT_LOADS")
      endif()
      list(APPEND _glm_vk_fast_shaders
        "expert_gemm_${_type}_canon${_loader}_fp32_math_native_fma_expanded|expert_gemm_canon|-DTYPE_${_type_define} -DFP32_MATH -DEXPANDED_WEIGHTS ${_loader_define}"
        "expert_gemm_${_type}_canon${_loader}_fp32_math_native_fma|expert_gemm_canon|-DTYPE_${_type_define} -DFP32_MATH ${_loader_define}"
        "expert_gemm_${_type}_canon${_loader}_fp64_scale_native_fma|expert_gemm_canon|-DTYPE_${_type_define} -DFP64_SCALE -DNATIVE_FMA ${_loader_define}")
      if(_type STREQUAL "q2k")
        list(APPEND _glm_vk_fast_shaders
          "expert_gemm_${_type}_canon${_loader}_fp64_scale_native_fma_trace|expert_gemm_canon|-DTYPE_${_type_define} -DFP64_SCALE -DNATIVE_FMA -DTRACE_BLOCK_SCALE ${_loader_define}")
      endif()
    endforeach()
  endforeach()
endif()
foreach(_columns IN ITEMS 8 16)
  list(APPEND _glm_vk_fast_shaders "gemv_f32_n${_columns}|gemv_f32|-DNMAX=${_columns} -DRSMAX=2")
  foreach(_type IN ITEMS q4k q6k q50)
    string(TOUPPER "${_type}" _type_define)
    list(APPEND _glm_vk_fast_shaders
      "gemv_rows_${_type}_n${_columns}|gemv_rows|-DTYPE_${_type_define} -DNMAX=${_columns}")
  endforeach()
endforeach()
list(APPEND _glm_vk_fast_shaders "fma_oracle_fp64_scale|fma_oracle|-DFP64_SCALE")
list(APPEND _glm_vk_fast_shaders "round_fp64_oracle|round_fp64_oracle|")
list(APPEND _glm_vk_fast_shaders
  "expert_mix_fp64_math|expert_mix|-DCANON_SWIGLU -DFP64_MATH"
  "q8k_codes_fp64_math|q8k_round|-DCODES_AND_SUMS -DFP64_MATH"
  "q8k_codes_portable_fp64_math|q8k_round|-DCODES_AND_SUMS -DPORTABLE_REDUCTION -DFP64_MATH")
list(APPEND _glm_vk_fast_shaders
  "expert_gemm_q2k_canon_fp64_scale_trace|expert_gemm_canon|-DTYPE_Q2K -DFP64_SCALE -DTRACE_BLOCK_SCALE"
  "expert_gemm_q2k_canon_vector_fp64_scale_trace|expert_gemm_canon|-DTYPE_Q2K -DVECTOR_WEIGHT_LOADS -DFP64_SCALE -DTRACE_BLOCK_SCALE")
foreach(_type IN ITEMS q2k q3k)
  string(TOUPPER "${_type}" _type_define)
  list(APPEND _glm_vk_fast_shaders
    "expert_gemm_${_type}_canon_fp64_scale|expert_gemm_canon|-DTYPE_${_type_define} -DFP64_SCALE"
    "expert_gemm_${_type}_canon_vector_fp64_scale|expert_gemm_canon|-DTYPE_${_type_define} -DVECTOR_WEIGHT_LOADS -DFP64_SCALE")
endforeach()
foreach(_type IN ITEMS q4k q6k q50)
  string(TOUPPER "${_type}" _define)
  list(APPEND _glm_vk_fast_shaders "expand_${_type}|gemm_tiled|-DTYPE_${_define} -DEXPAND_WEIGHTS")
endforeach()
list(APPEND _glm_vk_fast_shaders "mla_prefill_tiled|mla_prefill_tiled|")
list(APPEND _glm_vk_fast_shaders "mla_gather|mla_gather|" "mla_softmax|mla_softmax|")
list(APPEND _glm_vk_fast_shaders
  "expert_expand_q2k|expert_gemm_canon|-DTYPE_Q2K -DEXPAND_INTEGER"
  "expert_expand_q3k|expert_gemm_canon|-DTYPE_Q3K -DEXPAND_INTEGER")
list(APPEND _glm_vk_fast_shaders
  "gemm_f32_coalesced|gemm_tiled|-DTYPE_F32 -DCOALESCED_F32"
  "gemm_f32_wide_coalesced|gemm_tiled|-DTYPE_F32 -DCOALESCED_F32 -DNTILE=128"
  "gemm_f32_wider_coalesced|gemm_tiled|-DTYPE_F32 -DCOALESCED_F32 -DNTILE=256"
  "index_key_panel|index_key|-DTOKEN_OFFSET"
  "index_scores_step_panel|index_scores_step|-DTOKEN_OFFSET"
  "gemm_f32_transposed|gemm_tiled|-DTYPE_F32 -DTRANSPOSED_F32"
  "gemm_f32_wide_transposed|gemm_tiled|-DTYPE_F32 -DTRANSPOSED_F32 -DNTILE=128"
  "gemm_f32_wider_transposed|gemm_tiled|-DTYPE_F32 -DTRANSPOSED_F32 -DNTILE=256")
set(_fast_outputs)
set(_fast_registry "#pragma once\n#include <span>\n#include <string>\n")
set(_fast_lookup "inline std::span<const unsigned char> fast_shader(const std::string &name) {\n")
foreach(_entry IN LISTS _glm_vk_fast_shaders)
  string(REPLACE "|" ";" _parts "${_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _source)
  list(GET _parts 2 _defines)
  separate_arguments(_defines UNIX_COMMAND "${_defines}")
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/fast_${_name}.spv)
  set(_hdr ${CMAKE_CURRENT_BINARY_DIR}/fast_${_name}_spv.hpp)
  add_custom_command(OUTPUT ${_spv} ${_hdr}
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -O ${_defines} ${CMAKE_CURRENT_SOURCE_DIR}/fast/shaders/${_source}.comp -o ${_spv}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${_spv} -DOUTPUT=${_hdr} -DNAME=fast_${_name} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fast/shaders/${_source}.comp ${CMAKE_CURRENT_SOURCE_DIR}/fast/shaders/common.glsl
            ${CMAKE_CURRENT_SOURCE_DIR}/fast/shaders/fp64_round.glsl
            ${CMAKE_CURRENT_SOURCE_DIR}/fast/shaders/fp32_native.glsl
            ${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake VERBATIM)
  list(APPEND _fast_outputs ${_hdr})
  string(APPEND _fast_registry "#include \"fast_${_name}_spv.hpp\"\n")
  string(APPEND _fast_lookup "  if (name == \"${_name}\") return fast_${_name}_spv;\n")
endforeach()
string(APPEND _fast_lookup "  return {};\n}\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fast_shaders.hpp.tmp "${_fast_registry}${_fast_lookup}")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/fast_shaders.hpp.tmp ${CMAKE_CURRENT_BINARY_DIR}/fast_shaders.hpp COPYONLY)
add_custom_target(glm_vulkan_fast_shaders DEPENDS ${_fast_outputs})
add_library(strata_glm_vulkan_fast fast/device.cpp fast/kernels.cpp fast/expert_batch.cpp fast/engine.cpp)
if(STRATA_VULKAN_NATIVE_FMA)
  target_compile_definitions(strata_glm_vulkan_fast PRIVATE STRATA_VULKAN_NATIVE_FMA=1)
endif()
target_include_directories(strata_glm_vulkan_fast PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(strata_glm_vulkan_fast PUBLIC Vulkan::Vulkan strata_artifact ggml strata_kernels_cpu)
add_dependencies(strata_glm_vulkan_fast glm_vulkan_fast_shaders)
target_link_libraries(strata-glm-vulkan PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_bench fast/bench.cpp)
target_link_libraries(glm_vulkan_fast_bench PRIVATE strata_glm_vulkan_fast ggml strata_kernels_cpu)
add_executable(glm_vulkan_hw_diag hw_diag.cpp)
target_link_libraries(glm_vulkan_hw_diag PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_gemm_bench fast/gemm_bench.cpp)
target_link_libraries(glm_vulkan_fast_gemm_bench PRIVATE strata_glm_vulkan_fast ggml strata_kernels_cpu)
add_executable(glm_vulkan_fast_gemm_check fast/gemm_check.cpp)
target_link_libraries(glm_vulkan_fast_gemm_check PRIVATE strata_glm_vulkan_fast ggml)
add_executable(glm_vulkan_fast_repack_check fast/repack_check.cpp)
add_executable(glm_vulkan_fast_gemv_panel_check fast/gemv_panel_check.cpp)
target_link_libraries(glm_vulkan_fast_gemv_panel_check PRIVATE strata_glm_vulkan_fast ggml)
target_link_libraries(glm_vulkan_fast_repack_check PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_repack_gpu_check fast/repack_gpu_check.cpp)
target_link_libraries(glm_vulkan_fast_repack_gpu_check PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_q8_check fast/q8_check.cpp)
target_link_libraries(glm_vulkan_fast_q8_check PRIVATE strata_glm_vulkan_fast ggml strata_kernels_cpu)
add_executable(glm_vulkan_fast_swiglu_check fast/swiglu_check.cpp)
target_link_libraries(glm_vulkan_fast_swiglu_check PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_expert_quality fast/expert_quality.cpp)
target_link_libraries(glm_vulkan_fast_expert_quality PRIVATE strata_glm_vulkan_fast ggml strata_kernels_cpu)
add_executable(glm_vulkan_fast_expert_batch_check fast/expert_batch_check.cpp)
target_link_libraries(glm_vulkan_fast_expert_batch_check PRIVATE strata_glm_vulkan_fast ggml strata_kernels_cpu)
add_executable(glm_vulkan_fast_window_check fast/window_check.cpp)
target_link_libraries(glm_vulkan_fast_window_check PRIVATE strata_glm_vulkan_fast)
add_executable(glm_vulkan_fast_ref fast/reference.cpp)
target_link_libraries(glm_vulkan_fast_ref PRIVATE strata_artifact ggml strata_kernels_cpu)
target_include_directories(glm_vulkan_fast_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(glm_vulkan_fast_mla_prefill_check fast/mla_prefill_check.cpp)
target_link_libraries(glm_vulkan_fast_mla_prefill_check PRIVATE strata_glm_vulkan_fast)

target_sources(strata_glm_vulkan_fast PRIVATE fast/mla_prefill.cpp)

# The rejected CPU panel prototype is available only to the numerical check.
add_executable(glm_vulkan_fast_prefill_expert_check fast/prefill_expert_check.cpp fast/cpu_prefill.cpp)
target_link_libraries(glm_vulkan_fast_prefill_expert_check PRIVATE strata_glm_vulkan_fast strata_kernels_cpu)

add_executable(glm_vulkan_fast_prefill_schedule_check fast/prefill_schedule_check.cpp)
target_include_directories(glm_vulkan_fast_prefill_schedule_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(STRATA_BUILD_TESTS)
  add_test(NAME glm_vulkan_fast_prefill_schedule_check COMMAND glm_vulkan_fast_prefill_schedule_check)
  set_tests_properties(glm_vulkan_fast_prefill_schedule_check PROPERTIES LABELS "glm;vulkan;cpu")
endif()
