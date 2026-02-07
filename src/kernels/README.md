# Kernels

### `attention_draft.py`
A Python-based reference implementation acting as pseudocode. This establishes the tensor shapes, memory layout, and mathematical logic required for the PagedAttention mechanism without the complexity of CUDA memory management.

### `attention_v1.cu`
The most bare bones implementation for functional correctness. It processes each attention head using a single thread, iterating sequentially over all tokens. While extremely slow, this kernel was written for simplicity to ensure the math is correct.

### `attention_v2.cu`
This kernel introduces block level parallelism. We launch a grid of (num_heads, num_seqs) where each block manages one head sequence. It utilizes 128 threads per block to process tokens in parallel and stores intermediate results in shared memory. However, the final reduction (finding the max score and sum of exponentials) is performed linearly by a single thread (Thread 0), creating a serialization bottleneck.

### `attention_v3.cu`
This kernel optimized the bottleneck in V2 by implementing a Parallel Tree Reduction in shared memory. Instead of a linear scan (O(N)), active threads cooperate to reduce values in logarithmic steps (O(logN)). This maximizes GPU occupancy and significantly reduces the time threads spend waiting at the __syncthreads() barrier.

### `attention_v4.cu`
This kernel keeps the parallel tree reduction from V3 but optimizes the Memory Access Patterns. In previous versions, threads loaded data 1 float (4 bytes) at a time, which underutilizes the GPU's memory bus. In V4, we use reinterpret_cast<float4*> to load 128 bits (16 bytes) in a single instruction. This reduces the total number of memory transactions by 4x, significantly increasing effective bandwidth and throughput without changing the core math.