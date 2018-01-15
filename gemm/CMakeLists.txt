# get sources
file(GLOB tmp "*.cc")
set(gemm_lib_srcs ${tmp})

# file(GLOB ${tmp} "*_benchmark.cc")
# remove(gemm_lib_srcs ${gemm_lib_srcs} ${tmp})

# create gemm library
set(gemm_lib "gemm")
add_library(${gemm_lib} SHARED ${gemm_lib_srcs})

target_include_directories(${gemm_lib} BEFORE PUBLIC "${gemm_includes}")
target_link_libraries(${gemm_lib} ${gemm_dependencies})

