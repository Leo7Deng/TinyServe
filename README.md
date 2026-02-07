# TinyServe

### Dev Notes
`pip install .` reads `pyproject.toml` to create a temporary, hidden virtual environment. This forces a full recompile every time.

During development, use `make install` on each new GPU instance to install libraries manually. Then use `make build` to pip run `pip install -e . --no-build-isolation` to get incremental builds by reusing the `build/` folder.

`ninja` is used to compile C++ incrementally and in parallel.

PyTorch contains the PyBind11 headers (C++ files), but it does not install the PyBind11 Python package.