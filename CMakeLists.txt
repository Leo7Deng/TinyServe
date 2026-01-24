cmake_minimum_required(VERSION 3.18)
project(tinyserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


# 1. Point CMake to the pybind11 source in your extern folder
add_subdirectory(extern/pybind11)

# 2. Define your Python module
pybind11_add_module(tinyserve_ext src/bindings.cpp)

install(TARGETS tinyserve_ext DESTINATION .)