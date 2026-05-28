# TinyServe
TinyServe is a high throughput LLM inference engine.

I built this to deepen my understanding of LLM inference internals, ML systems engineering, and GPU memory optimization, applying what I learned from reading Programming Massively Parallel Processors.

The core optimization in this project is PagedAttention (originally introduced by [vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/#inputs)), which takes Operating Systems concepts of virtual memory and paging, and applies them to the tensors required for LLM inference. 

## PagedAttention
To implement PagedAttention, the system is split between a Python based control plane and a C++/CUDA data plane.

### Control Plane (memory_manager.py, scheduler.py)
This acts as a virtual memory manager. It creates a KVCache object which allocates a fixed pool of GPU VRAM at startup to hold the entire memory grid for the serving engine. Instead of allocating memory per user, it maintains a free_blocks list. As tokens are generated, it dynamically maps tokens to physical VRAM blocks, stored in the Sequence object.

> In real systems, to maximize concurrent users, the system should calculate the exact amount of free VRAM remaining after the model weights are loaded. This determines exactly how many physical memory blocks to allocate to the KVCache pool, allowing us to safely utilize ~100% of the GPU without OOM crashes.

### Data Plane (reshape_and_cache.cu)
To actually store and move the data in VRAM, we interface using a custom CUDA kernel. During the LLM's forward pass, the Python control plane passes the virtual to physical memory mapping to this custom kernel. reshape_and_cache is called for every new token, bypassing PyTorch to physically insert the newly generated Key/Value tensors into the scattered, non-contiguous physical memory blocks.

## Continuous Batching (scheduler.py)
To complement PagedAttention, I created a custom scheduler to implement continuous batching. Instead of static batching where the GPU sits idle waiting for the longest sequence to finish, the scheduler ejects finished sequences immediately and inserts new requests from the queue.

This design has some similarities to an OS thread scheduler. Each request is treated as a discrete task with its own saved context (its block table mapping and generation state), allowing the engine to efficiently multiplex sequences at every token generation cycle and keep GPU utilization at its peak.

## Custom Kernels
To compute attention across scattered memory pages, standard PyTorch functions no longer work. I wrote custom CUDA attention kernels designed specifically to intake the block table mappings to traverse a non-contiguous KV Cache. The final iteration of my custom kernel adds FlashAttention-style online softmax on top of the earlier optimizations: block-level parallelism through shared memory, parallel tree reduction, and vectorized memory access patterns.

## Results

By managing memory in fixed-size pages, TinyServe minimizes external fragmentation using purposeful data structures.

Without PagedAttention, standard PyTorch forces you to allocate a massive, contiguous rectangular tensor based on the maximum possible sequence length. This wastes huge amounts of VRAM, since most users only generate short responses, while a few generate large ones.

On the current A10 test run, `tests/test_max_concurrency.py` reached 2500 concurrent users for the best TinyServe kernels versus 300 for the PyTorch baseline, or about `8.3x` higher capacity.

On the decode throughput benchmark, `tests/test_throughput.py` now compares against PyTorch SDPA with separate MHA and GQA baselines. In that run, `attention_v7` was `10.31x` faster than the PyTorch GQA SDPA baseline for the one-token decode workload being measured.

### Dev Notes
`pip install .` reads `pyproject.toml` to create a temporary, hidden virtual environment. This forces a full recompile every time.

During development, use `make install` on each new GPU instance to install libraries manually. Then use `make build` to pip run `pip install -e . --no-build-isolation` to get incremental builds by reusing the `build/` folder.

`ninja` is used to compile C++ incrementally and in parallel.

PyTorch contains the PyBind11 headers (C++ files), but it does not install the PyBind11 Python package.
