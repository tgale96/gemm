# TODO(tgale): Only one i've tested with
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(gemm CXX C)

# Helper functions
function (remove OUTPUT INPUT)
set(REMOVES ${ARGN})
foreach(REMOVE ${REMOVES})
        list(REMOVE_ITEM INPUT "${REMOVE}")
endforeach()
set(${OUTPUT} ${INPUT} PARENT_SCOPE)
endfunction(remove)

if(NOT CMAKE_BUILD_TYPE)
  message("default build type: Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Built type." FORCE)
endif()
  
# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

# Vars for tracking libs
set(gemm_dependencies)
set(gemm_includes)
set(benchmark_dependencies)
set(benchmark_includes)

# Include the project root
list(APPEND gemm_includes ${PROJECT_SOURCE_DIR})
list(APPEND benchmark_includes ${PROJECT_SOURCE_DIR})

# Dependencies
include(cmake/dependencies.cmake)

# Sources tree
add_subdirectory(gemm)
