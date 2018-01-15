# For CUDA
find_package(CUDA)
list(APPEND gemm_includes ${CUDA_INCLUDE_DIRS})
list(APPEND gemm_dependencies ${CUDA_LIBRARIES})

# For Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable GBench testing")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
list(APPEND benchmark_includes "${PROJECT_SOURCES_DIR}/third_party/benchmark/include")
list(APPEND benchmark_dependencies "benchmark")

# For BLAS on Apple
find_library(ACCELERATE Accelerate)
list(APPEND benchmark_dependencies ${ACCELERATE})
