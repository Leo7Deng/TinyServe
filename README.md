# TinyServe

### Dev Notes
`pip install .` reads `pyproject.toml` to create a temporary, hidden virtual environment. This forces a full recompile every time.

Instead, during devlopment, `pip install` libraries manually. Then use `pip install --no-build-isolation -ve .` to get incremental builds by reusing the `build/` folder.

`pip install .` uses pyproject.toml and `build-backend = "scikit_build_core.build"` tells Python to use CMake to build a shared library.

`ninja` is used to compile C++ incrementally and parallely