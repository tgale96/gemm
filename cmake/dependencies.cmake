# For CUDA
find_package(CUDA)
list(APPEND gemm_includes ${CUDA_INCLUDE_DIRS})
list(APPEND gemm_dependencies ${CUDA_LIBRARIES})

# For Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable GBench testing")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
list(APPEND gemm_includes ${BENCHMARK_INCLUDE_DIRS})
list(APPEND gemm_dependencies ${BENCHMARK_LIBRARIES})

# For BLAS on osx
find_library(ACCELERATE Accelerate)
list(APPEND gemm_dependencies ${ACCELERATE})

message("poop")
message(${ACCELERATE})
