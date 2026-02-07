# Tests

### `test_attention.py`
Validates numerical accuracy by generating random Q, K, V tensors, scattering them into non-contiguous physical blocks, and asserting that the custom paged attention kernel output matches a standard PyTorch reference implementation. This ensures that memory optimizations (PagedAttention) do not degrade model quality, verifying FP32 precision stability within a 1e-3 tolerance.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Attention Config: 2 Seqs, 32 Heads, Block Size 16 ---

    Computing PyTorch Reference

    Testing Attention Kernel V1 Pass: Kernel output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.28064659237861633, 0.09345307946205139, -0.1507822871208191]
    Sample Ref    (User 0, Head 0): [0.28064653277397156, 0.09345296770334244, -0.1507822722196579]

    Testing Attention Kernel V2 Pass: Kernel output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.28064659237861633, 0.09345307946205139, -0.1507822722196579]
    Sample Ref    (User 0, Head 0): [0.28064653277397156, 0.09345296770334244, -0.1507822722196579]

### `test_benchmark.py`
Measures the effective Memory Bandwidth (GB/s) and kernel latency (ms) of the engine, comparing TinyServe (V1 & V2) against naive PyTorch implementations. This quantifies the overhead introduced by memory indirection. While the non-contiguous memory access prevents us from beating PyTorch's contiguous baseline in raw latency, this test verifies that the speed remains competitive. This ensures the latency cost is minimal and well worth the tradeoff for the gain in concurrent user capacity.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Benchmark Config: Batch=64, Context=1024-4096, VRAM Reserved: 2.68 GB ---

    Running Attention Kernel V1
    Latency: 124.839 ms | Effective Bandwidth: 22.71 GB/s

    Running Attention Kernel V2
    Latency: 53.589 ms | Effective Bandwidth: 51.83 GB/s

    Running: PyTorch Baseline
    Latency: 9.226 ms | Effective Bandwidth: 284.24 GB/s

    --- Results ---
    Attention Kernel V1: 13.53x slower than PyTorch
    Attention Kernel V2: 5.81x slower than PyTorch

### `test_max_concurrency.py`
Stress tests the memory manager by simulating irregular sequence lengths (Zipfian distribution) and incrementally increasing batch size until the GPU hits Out Of Memory (OOM). This demonstrates the reduction in KV Cache Fragmentation, showing exactly how many more concurrent users TinyServe can handle compared to contiguous allocation.

_In this test, attention kernel versions should not have much affect on results, but I am running all kernel versions to make sure shared memory optimizations isn't reducing max concurrency._

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Max Concurrency Stress Test (Zipfian Distribution) ---
    Simulating traffic: 90% Short (64-512), 10% Long (2048-4096)

    PyTorch
    - Batch 100   | VRAM: 6.53 GB | Efficiency: 13.94%
    - Batch 250   | VRAM: 15.81 GB | Efficiency: 14.75%
    - Batch 500   | FAIL (OOM Crash)

    PagedAttention V1
    - Batch 100   | VRAM: 0.99 GB | Efficiency: 98.71%
    - Batch 250   | VRAM: 2.36 GB | Efficiency: 98.67%
    - Batch 500   | VRAM: 4.67 GB | Efficiency: 98.69%
    - Batch 1000  | VRAM: 9.36 GB | Efficiency: 98.69%
    - Batch 2000  | VRAM: 18.86 GB | Efficiency: 98.69%
    - Batch 3000  | FAIL (OOM Crash)

    PagedAttention V2
    - Batch 100   | VRAM: 0.93 GB | Efficiency: 98.72%
    - Batch 250   | VRAM: 2.39 GB | Efficiency: 98.72%
    - Batch 500   | VRAM: 4.73 GB | Efficiency: 98.69%
    - Batch 1000  | VRAM: 9.54 GB | Efficiency: 98.70%
    - Batch 2000  | VRAM: 18.67 GB | Efficiency: 98.66%
    - Batch 3000  | FAIL (OOM Crash)

    --- Results ---
    PyTorch Baseline:      250 users
    PagedAttention V1   : 2000 users
    PagedAttention V2   : 2000 users
    TinyServe handles 8.0x more concurrent users!

### `test_memory_manager.py`
Unit tests the KVCache and reshape_and_cache kernel. Verifies that the "Virtual Memory Manager" correctly translates logical token indices (e.g., Token 50) to physical GPU blocks (e.g., Block 12, Offset 2). This validates data integrity, ensuring that despite non-contiguous storage, every token is written to and read from the correct physical slot without corruption.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    1. Requesting 5 tokens (Prefill)
    - User Blocks: [0, 1]
    - Token Count: 5

    2. Generating 3 more tokens (should NOT add block)
    - User Blocks: [0, 1]
    - Token Count: 8

    3. Generating 1 more token (should ADD block)
    - User Blocks: [0, 1, 2]
    - Slot Mapping: tensor([3], device='cuda:0')
    - Block 0 Content (Last row (tensor([3], device='cuda:0')) row should be 9s):
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [9., 9., 9., 9.]], device='cuda:0')

    Memory manager test ran successfully! Kernel wrote data to the correct physical slot.