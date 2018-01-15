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

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

# Vars for tracking libs
set(gemm_dependencies)
set(gemm_includes)

# Dependencies
include(cmake/dependencies.cmake)

# Sources tree
add_subdirectory(gemm)
