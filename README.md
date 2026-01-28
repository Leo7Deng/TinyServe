# TinyServe

### Dev Notes
`pip install .` reads `pyproject.toml` to create a temporary, hidden virtual environment. This forces a full recompile every time.

Instead, during devlopment, `pip install` libraries manually. Then use `pip install --no-build-isolation -ve .` to get incremental builds by reusing the `build/` folder.

`pip install .` uses pyproject.toml and `build-backend = "scikit_build_core.build"` tells Python to use CMake to build a shared library.

`ninja` is used to compile C++ incrementally and parallely

100x100 matrix times 100x100 matrix test: CPU execution time: 0.6925301551818848, GPU execution time: 0.21920156478881836, Results are equal: True
At 1000x1000 matrix times 1000x1000 matrix test, the CPU takes > 10 minutes while the GPU execution time is 0.22040009498596191, hardly an increase