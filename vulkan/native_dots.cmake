# Preserve the supplied checkout. Patch a build-local copy only for opt-in
# Vulkan integer-vector dots; CUDA/HIP and the regular Vulkan build use it as-is.
function(strata_vulkan_ggml_source source output)
  if(NOT STRATA_ENABLE_VULKAN OR NOT STRATA_VULKAN_NATIVE_DOTS)
    set(${output} "${source}" PARENT_SCOPE)
    return()
  endif()
  set(copy "${CMAKE_BINARY_DIR}/strata-vulkan-ggml")
  file(COPY "${source}/" DESTINATION "${copy}" PATTERN "mul_mat_vecq_funcs.glsl" EXCLUDE)
  set(shader "${copy}/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl")
  file(READ "${source}/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl" text)
  set(old "return FLOAT_TYPE(float(cache_b_ds.x) * float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(cache_b_ds.y / 2));")
  set(new "// Strata: K-quant minima use the integer activation sum, not a rounded half sum.
    const int32_t b_sum = dotPacked4x8EXT(0x01010101, cache_b_qs[0]) +
                          dotPacked4x8EXT(0x01010101, cache_b_qs[1]) +
                          dotPacked4x8EXT(0x01010101, cache_b_qs[2]) +
                          dotPacked4x8EXT(0x01010101, cache_b_qs[3]);
    return FLOAT_TYPE(float(cache_b_ds.x) * (float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(b_sum)));")
  string(FIND "${text}" "${old}" found)
  if(found LESS 0)
    message(FATAL_ERROR "Pinned GGML K-quant vector-dot expression changed")
  endif()
  string(REPLACE "${old}" "${new}" text "${text}")
  set(old "return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 4);")
  set(new "#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0)
    // CPU applies the native original-input-sum correction after readback.
    const int32_t b_sum = dotPacked4x8EXT(0x01010101, cache_b_qs[0]) +
                          dotPacked4x8EXT(0x01010101, cache_b_qs[1]);
    return mul_q8_1(q_sum, get_dm(ib_a), vec2(cache_b_ds.x, cache_b_ds.x * float(b_sum) * 4.0), 4);
#else
    return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 4);
#endif")
  string(FIND "${text}" "${old}" found)
  if(found LESS 0)
    message(FATAL_ERROR "Pinned GGML legacy vector-dot expression changed")
  endif()
  string(REPLACE "${old}" "${new}" text "${text}")
  if(EXISTS "${shader}")
    file(READ "${shader}" previous)
  endif()
  if(NOT previous STREQUAL text)
    file(WRITE "${shader}" "${text}")
  endif()
  set(${output} "${copy}" PARENT_SCOPE)
endfunction()
