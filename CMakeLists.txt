cmake_minimum_required(VERSION 3.10)

project(ReduceMaxWithIndex)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA QUIET REQUIRED)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)

enable_language(CUDA)

set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

file(GLOB CuHeaders src/*.h)

add_executable(ReduceMaxWithIndex src/main.cu ${CuHeaders})

add_executable(ReduceMaxWithIndexxCpu src/MaxElement.cpp)


