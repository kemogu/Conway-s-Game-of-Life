cmake_minimum_required(VERSION 3.10)
project(CGOL CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(include)

file(GLOB SOURCES "src/*.cu")

add_executable(CGOL ${SOURCES})

set_target_properties(CGOL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

