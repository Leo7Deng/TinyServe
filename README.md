# TinyServe

### Dev Notes
`pip install .` reads `pyproject.toml` to create a temporary, hidden virtual environment. This forces a full recompile every time.

Instead, during devlopment, `pip install scikit-build-core pybind11 cmake ninja packaging` libraries manually. Then use `pip install --no-build-isolation -ve .` to get incremental builds by reusing the `build/` folder.

`pip install .` uses pyproject.toml and `build-backend = "scikit_build_core.build"` tells Python to use CMake to build a shared library.

`ninja` is used to compile C++ incrementally and parallely

100x100 matrix times 100x100 matrix: CPU execution time: 0.6925301551818848, GPU execution time: 0.21920156478881836, Results are equal: True
200x100 matrix times 200x100 matrix: CPU execution time: 5.019192218780518, GPU execution time: 0.21590900421142578, Results are equal: True
1000x1000 matrix times 1000x1000 matrix test: CPU execution time: >11 min, GPU execution time: 0.22040009498596191, Results are equal: True

Use FetchContent in `CMakeLists.txt` to remove need for extern and git submodule.

test_fragmentation:
```
Free Blocks: 2
Freeing users A and C
Free Blocks: 6
Allocating sequence E (needs 5 blocks)
User E successfully allocated blocks: [5, 4, 1, 0, 8]
```
