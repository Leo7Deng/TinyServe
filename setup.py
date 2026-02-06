import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

global_includes = include_paths()

try:
    import pybind11
    pybind_inc = pybind11.get_include()
    # Check if the PyTorch paths actually contain pybind11
    has_bundled = any("pybind11" in p for p in global_includes)
    if not has_bundled and pybind_inc not in global_includes:
        print(f"Injecting explicit pybind11 include: {pybind_inc}")
        global_includes.append(pybind_inc)
except ImportError:
    pass

setup(
    name='tinyserve',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='tinyserve_ext', 
            sources=[
                'src/bindings.cu',
                'src/kernels/reshape_and_cache.cu',
                'src/kernels/attention_v1.cu',
                'src/kernels/attention_v2.cu',
            ],
            include_dirs=global_includes,
            extra_compile_args={
                'cxx': ['-O3'], 
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)