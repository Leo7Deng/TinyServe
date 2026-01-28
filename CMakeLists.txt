cmake_minimum_required(VERSION 3.18)
project(tinyserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)


# Point CMake to the pybind11 source in your extern folder
add_subdirectory(extern/pybind11)

# Define your Python module
pybind11_add_module(tinyserve_ext src/bindings.cu)

# Link CUDA runtime
target_link_libraries(tinyserve_ext PRIVATE CUDA::cudart)

install(TARGETS tinyserve_ext DESTINATION .)