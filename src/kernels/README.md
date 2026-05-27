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

### `attention_v5.cu`
While integrating TinyLlama as a benchmark, the attention kernel needed to support Grouped Query Attention (GQA). In previous versions, the kernel assumed a 1:1 mapping between Q heads and KV heads. However, TinyLlama 1.1B uses 32 Q heads but only 4 KV heads, where groups of 8 Q heads share the same KV head. KV caches are relatively redundant across heads, so this reduces KV memory cost by 8x without significantly decreasing performance. The kernel correctly maps KV cache indexes taking num Q heads:num KV heads into account.

### `attention_v6.cu`
This kernel keeps the PagedAttention block table lookup and GQA mapping from V5, but replaces the two-pass softmax with online softmax. Each thread streams its assigned K/V tokens once, maintaining a running max, running softmax denominator, and running weighted-value accumulator. Threads then merge those online-softmax states in shared memory. This is the first FlashAttention-style version: correctness-oriented first, with fewer K-cache reads than V5, but not yet fully tiled or tuned.
