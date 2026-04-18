cmake_minimum_required(VERSION 3.26)
project(libcudf_tutorial LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Use ccache if available to cache compiler outputs across rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER  ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
endif()

# ---- find packages ----
find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)
find_package(Arrow REQUIRED)

# ---- executable ----
add_executable(libcudf_tpch_orders_groupby libcudf_tpch_orders_groupby.cu)

target_link_libraries(libcudf_tpch_orders_groupby
    PRIVATE
        cudf::cudf
        Arrow::arrow_shared
        ${CMAKE_DL_LIBS}   # dladdr() used by rmm_backtrace_resource_adaptor.hpp
)

set_target_properties(libcudf_tpch_orders_groupby PROPERTIES
    CUDA_ARCHITECTURES native
)

# Preserve source file/line info in the binary for nsys/ncu backtraces.
# --generate-line-info is lightweight (no full debug, keeps optimizations).
target_compile_options(libcudf_tpch_orders_groupby PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<COMPILE_LANGUAGE:CXX>:-g>
)
