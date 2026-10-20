# Optional second compilation of the existing CPU source. No GPU compiler or
# baseline compiler flags are changed. Runtime selection also defaults off.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "STRATA_Q3_CLANG_NT4/NT2 support native Linux x86-64 GNU host builds only")
endif()
find_program(STRATA_Q3_CLANG_COMPILER NAMES clang++ REQUIRED)
execute_process(COMMAND "${STRATA_Q3_CLANG_COMPILER}" --version
                OUTPUT_VARIABLE _q3_clang_version RESULT_VARIABLE _q3_clang_status)
if(NOT _q3_clang_status EQUAL 0 OR NOT _q3_clang_version MATCHES "clang version")
  message(FATAL_ERROR "STRATA_Q3_CLANG_COMPILER must identify a working clang++")
endif()
set(_q3_clang_object "${CMAKE_CURRENT_BINARY_DIR}/q3_clang_nt4.o")
add_custom_command(
  OUTPUT "${_q3_clang_object}"
  COMMAND "${STRATA_Q3_CLANG_COMPILER}" -O3 -DNDEBUG -std=c++20 -fPIC
          -stdlib=libstdc++ -mavx2 -mfma -mf16c
          "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
          "-I${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml"
          "-I${_strata_ggml_source}/include"
          -Dq23_rows=strata_q3_clang_rows
          -Dq23_gu_rows=strata_q3_clang_gu_rows
          -Dq3k_expanded_rows=strata_q3_clang_expanded_rows
          -Dq3k_expanded_gu_rows=strata_q3_clang_expanded_gu_rows
          -MMD -MF "${_q3_clang_object}.d"
          -c "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/q23_avx2.cpp"
          -o "${_q3_clang_object}"
  DEPENDS src/kernels/cpu/q23_avx2.cpp "${STRATA_Q3_CLANG_COMPILER}"
  DEPFILE "${_q3_clang_object}.d"
  VERBATIM)
set_source_files_properties("${_q3_clang_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
target_sources(strata_kernels_cpu PRIVATE "${_q3_clang_object}")
# Applied only to the baseline compilation, never the custom Clang command.
# Both widths share one Clang object; each baseline dispatch is independently opt-in.
foreach(_q3_width IN ITEMS 2 4)
  if(STRATA_Q3_CLANG_NT${_q3_width})
    set_property(SOURCE src/kernels/cpu/q23_avx2.cpp APPEND PROPERTY
                 COMPILE_DEFINITIONS STRATA_HAVE_Q3_CLANG_NT${_q3_width}=1)
  endif()
endforeach()
