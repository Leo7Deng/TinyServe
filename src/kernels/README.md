# Kernels

### `attention_draft.py`
A Python-based reference implementation acting as pseudocode. This establishes the tensor shapes, memory layout, and mathematical logic required for the PagedAttention mechanism without the complexity of CUDA memory management.

### `attention_v1.cu`
The most bare bones implementation for functional correctness. It processes each attention head using a single thread, iterating sequentially over all tokens. While extremely slow, this kernel was written for simplicity to ensure the math is correct.

### `attention_v2.cu`
This kernel introduces block level parallelism. We launch a grid of (num_heads, num_seqs) where each block manages one head sequence. It utilizes 128 threads per block to process tokens in parallel and stores intermediate results in shared memory. However, the final reduction (finding the max score and sum of exponentials) is performed linearly by a single thread (Thread 0), creating a serialization bottleneck.

### `attention_v3.cu`
This kernel optimized the bottleneck in V2 by implementing a Parallel Tree Reduction in shared memory. Instead of a linear scan (O(N)), active threads cooperate to reduce values in logarithmic steps (O(logN)). This maximizes GPU occupancy and significantly reduces the time threads spend waiting at the __syncthreads() barrier.