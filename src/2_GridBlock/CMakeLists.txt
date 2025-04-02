cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

set(This1 CheckDimension)
set(This2 defineGridBlock)
set(This3 CheckThreadIndex)

project(${This1} LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Source file
add_executable(${This1} ${This1}.cu)
target_compile_options(${This1} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(${This1} PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(${This1} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Source file
add_executable(${This2} ${This2}.cu)
target_compile_options(${This2} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(${This2} PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(${This2} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Source file
add_executable(${This3} ${This3}.cu)
target_compile_options(${This3} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(${This3} PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(${This3} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
