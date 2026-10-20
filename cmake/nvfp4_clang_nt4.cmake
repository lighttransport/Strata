# Optional second compilation of the existing CPU source. No GPU compiler or
# baseline compiler flags are changed. Runtime selection also defaults off.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "STRATA_NVFP4_CLANG_NT4 supports native Linux x86-64 GNU host builds only")
endif()
find_program(STRATA_NVFP4_CLANG_COMPILER NAMES clang++ REQUIRED)
execute_process(COMMAND "${STRATA_NVFP4_CLANG_COMPILER}" --version
                OUTPUT_VARIABLE _nvfp4_clang_version RESULT_VARIABLE _nvfp4_clang_status)
if(NOT _nvfp4_clang_status EQUAL 0 OR NOT _nvfp4_clang_version MATCHES "clang version")
  message(FATAL_ERROR "STRATA_NVFP4_CLANG_COMPILER must identify a working clang++")
endif()
set(_nvfp4_clang_object "${CMAKE_CURRENT_BINARY_DIR}/nvfp4_clang_nt4.o")
add_custom_command(
  OUTPUT "${_nvfp4_clang_object}"
  COMMAND "${STRATA_NVFP4_CLANG_COMPILER}" -O3 -DNDEBUG -std=c++20 -fPIC
          -stdlib=libstdc++ -mavx2 -mfma -mf16c
          "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
          "-I${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml"
          "-I${_strata_ggml_source}/include"
          -Dnvfp4_256_fits=strata_clang_nvfp4_256_fits
          -Dnvfp4_256_rows=strata_clang_nvfp4_256_rows
          -Dnvfp4_256_gu_rows=strata_clang_nvfp4_256_gu_rows
          -Dnvfp4_256_gu_rows_clamped=strata_clang_nvfp4_256_gu_rows_clamped
          -Dnvfp4_256_k32_rows=strata_clang_nvfp4_256_k32_rows
          -Dnvfp4_256_k32_gu_rows_clamped=strata_clang_nvfp4_256_k32_gu_rows_clamped
          -MMD -MF "${_nvfp4_clang_object}.d"
          -c "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/nvfp4_avx2.cpp"
          -o "${_nvfp4_clang_object}"
  DEPENDS src/kernels/cpu/nvfp4_avx2.cpp "${STRATA_NVFP4_CLANG_COMPILER}"
  DEPFILE "${_nvfp4_clang_object}.d"
  VERBATIM)
set_source_files_properties("${_nvfp4_clang_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
target_sources(strata_kernels_cpu PRIVATE "${_nvfp4_clang_object}")
# Applied only to the baseline compilation, never the custom Clang command.
set_property(SOURCE src/kernels/cpu/nvfp4_avx2.cpp APPEND PROPERTY
             COMPILE_DEFINITIONS STRATA_HAVE_NVFP4_CLANG_NT4=1)
