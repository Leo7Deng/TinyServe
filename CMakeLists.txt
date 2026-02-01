include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)

FetchContent_MakeAvailable(pybind11)

cmake_minimum_required(VERSION 3.18)
project(tinyserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

# Define your Python module
pybind11_add_module(tinyserve_ext 
    src/bindings.cu 
    src/memory.h
    src/kernels/attention.cu
)

# Include headers inside src/
target_include_directories(tinyserve_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link CUDA runtime
target_link_libraries(tinyserve_ext PRIVATE CUDA::cudart)

install(TARGETS tinyserve_ext DESTINATION .)