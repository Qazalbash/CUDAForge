cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

set(This ArrSum)

project(${This} LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

enable_testing()
add_executable(test_${This} ${This}.cu test_${This}.cu)

target_link_libraries(test_${This} PRIVATE GTest::GTest GTest::Main)

include(GoogleTest)
gtest_discover_tests(test_${This})