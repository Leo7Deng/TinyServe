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

For the Paged Attention Kernel, I will need to write a CUDA kernel that computes the attention mechanism:
```
Output = Softmax(Q * K^T) * V
```
In standard PyTorch, K and V are continuous rectangles in memory. When context scales up, K and V sizes increase linearly, and takes up more and more memory. In TinyServe, K and V will be broken down into tiny blocks of memory scattered randomly across the GPU.
This kernel is essentially acting like a Memory Management Unit (MMU). For example, on token 50, it must calculate token 50 is on page 50//16 = 3. Page 3 maps to physical block 99. Read the data at block 99 with offset 2.

```
python tests/test_attention.py 
Config: 2 Seqs, 32 Heads, Block Size 16
Running Paged Attention Kernel
Running Reference PyTorch Attention
Pass: Kernel output matched PyTorch reference!
Sample Kernel (User 0, Head 0): [0.28064659237861633, 0.09345307946205139, -0.1507822871208191]
Sample Ref    (User 0, Head 0): [0.28064659237861633, 0.093453049659729, -0.1507822722196579]
```

test_benchmark is used to test the latency of my attention kernel vs standard PyTorch. We expect PyTorch to be faster because with my attention kernel using PagedAttention, the GPU needs to query for where the next block is from the block table, as opposed to linearly allocated data in PyTorch where the GPU can "predict" where the next piece of data will be. test_benchmark is purely a test of latency, where I want to reduce the latency to be as close to PyTorch as possible. PagedAttention will perform better when the amount of requests increase to where standard PyTorch will run into "out of memory" errors, and PagedAttention will not.
```
python tests/test_benchmark.py 
Config: Batch=50, Context=1000-1500, Heads=32
Max Blocks Needed: 5724
VRAM Used for KV Cache: 0.75 GB
Allocating and scattering blocks randomly.
Sequence Lengths (Sample): [1477, 1308, 1242, 1091, 1424]
Fragmentation: User 0's first 5 blocks are at: [3143, 4200, 1683, 1207, 4580]
Running 100 iterations for average latency
Average Latency: 12.982 ms
Memory Bandwidth: 79.19 GB/s
PyTorch Latency: 1.034 ms
Slowdown Factor: 12.6x slower than PyTorch
```