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

    Testing Attention Kernel V3 Pass: Kernel output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.28064653277397156, 0.0934530645608902, -0.1507822424173355]
    Sample Ref    (User 0, Head 0): [0.28064653277397156, 0.09345296770334244, -0.1507822722196579]

    Testing Attention Kernel V4 Pass: Kernel output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.28064650297164917, 0.09345297515392303, -0.15078230202198029]
    Sample Ref    (User 0, Head 0): [0.28064653277397156, 0.09345296770334244, -0.1507822722196579]


    --- GQA Attention Test for V5 (32 Q heads, 4 KV heads) ---

    Computing PyTorch GQA Reference

    Testing Attention Kernel V5 (GQA) Pass: V5 GQA output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.02329263463616371, -0.17037571966648102, 0.08549156039953232]
    Sample Ref    (User 0, Head 0): [0.023292670026421547, -0.1703757345676422, 0.08549157530069351]

### `test_benchmark.py`
Measures the effective Memory Bandwidth (GB/s) and kernel latency (ms) of the engine, comparing TinyServe (V1 & V2) against naive PyTorch implementations. This quantifies the overhead introduced by memory indirection. While the non-contiguous memory access prevents us from beating PyTorch's contiguous baseline in raw latency, this test verifies that the speed remains competitive. This ensures the latency cost is minimal and well worth the tradeoff for the gain in concurrent user capacity.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Benchmark Config: Batch=64, Context=1024-4096, VRAM Reserved: 2.68 GB ---

    Running Attention Kernel V1
    Latency: 125.526 ms | Effective Bandwidth: 22.59 GB/s

    Running Attention Kernel V2
    Latency: 53.009 ms | Effective Bandwidth: 52.40 GB/s

    Running Attention Kernel V3
    Latency: 51.940 ms | Effective Bandwidth: 51.32 GB/s

    Running Attention Kernel V4
    Latency: 26.556 ms | Effective Bandwidth: 99.73 GB/s

    Running Attention Kernel V5 (GQA)
    Latency: 18.552 ms | Effective Bandwidth: 138.52 GB/s

    Running: PyTorch Baseline
    Latency: 9.224 ms | Effective Bandwidth: 318.30 GB/s

    --- Results ---
    Attention Kernel V1: 13.61x slower than PyTorch
    Attention Kernel V2: 5.75x slower than PyTorch
    Attention Kernel V3: 5.63x slower than PyTorch
    Attention Kernel V4: 2.88x slower than PyTorch
    Attention Kernel V5: 2.01x slower than PyTorch

### `test_max_concurrency.py`
Stress tests the memory manager by simulating irregular sequence lengths (Zipfian distribution) and incrementally increasing batch size until the GPU hits Out Of Memory (OOM). This demonstrates the reduction in KV Cache Fragmentation, showing exactly how many more concurrent users TinyServe can handle compared to contiguous allocation.

_In this test, attention kernel versions should not have much affect on results, but I am running all kernel versions to make sure shared memory optimizations isn't reducing max concurrency._

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Max Concurrency Stress Test (Zipfian Distribution) ---
    Simulating traffic: 90% Short (64-512), 10% Long (2048-4096)

    PyTorch
    - Batch 100   | VRAM: 6.59 GB | Efficiency: 14.39%
    - Batch 200   | VRAM: 13.27 GB | Efficiency: 14.51%
    - Batch 300   | VRAM: 20.06 GB | Efficiency: 14.24%
    - Batch 400   | FAIL (OOM Crash)

    Attention Kernel V1
    - Batch 100   | VRAM: 0.96 GB | Efficiency: 98.78%
    - Batch 200   | VRAM: 1.87 GB | Efficiency: 98.66%
    - Batch 300   | VRAM: 2.88 GB | Efficiency: 98.73%
    - Batch 400   | VRAM: 3.76 GB | Efficiency: 98.73%
    - Batch 500   | VRAM: 4.59 GB | Efficiency: 98.64%
    - Batch 1000  | VRAM: 9.31 GB | Efficiency: 98.64%
    - Batch 2000  | VRAM: 18.72 GB | Efficiency: 98.69%
    - Batch 2100  | VRAM: 19.39 GB | Efficiency: 98.67%
    - Batch 2200  | VRAM: 20.90 GB | Efficiency: 98.69%
    - Batch 2300  | VRAM: 21.70 GB | Efficiency: 98.69%
    - Batch 2400  | VRAM: 22.52 GB | Efficiency: 98.69%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V2
    - Batch 100   | VRAM: 0.98 GB | Efficiency: 98.71%
    - Batch 200   | VRAM: 1.98 GB | Efficiency: 98.77%
    - Batch 300   | VRAM: 2.83 GB | Efficiency: 98.73%
    - Batch 400   | VRAM: 3.82 GB | Efficiency: 98.72%
    - Batch 500   | VRAM: 4.77 GB | Efficiency: 98.62%
    - Batch 1000  | VRAM: 9.16 GB | Efficiency: 98.63%
    - Batch 2000  | VRAM: 18.98 GB | Efficiency: 98.70%
    - Batch 2100  | VRAM: 19.88 GB | Efficiency: 98.70%
    - Batch 2200  | VRAM: 20.94 GB | Efficiency: 98.69%
    - Batch 2300  | VRAM: 21.54 GB | Efficiency: 98.68%
    - Batch 2400  | VRAM: 22.51 GB | Efficiency: 98.69%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V3
    - Batch 100   | VRAM: 0.92 GB | Efficiency: 98.67%
    - Batch 200   | VRAM: 1.96 GB | Efficiency: 98.67%
    - Batch 300   | VRAM: 2.90 GB | Efficiency: 98.66%
    - Batch 400   | VRAM: 3.87 GB | Efficiency: 98.81%
    - Batch 500   | VRAM: 4.63 GB | Efficiency: 98.66%
    - Batch 1000  | VRAM: 9.66 GB | Efficiency: 98.72%
    - Batch 2000  | VRAM: 18.71 GB | Efficiency: 98.71%
    - Batch 2100  | VRAM: 19.78 GB | Efficiency: 98.68%
    - Batch 2200  | VRAM: 20.41 GB | Efficiency: 98.69%
    - Batch 2300  | VRAM: 21.69 GB | Efficiency: 98.71%
    - Batch 2400  | VRAM: 22.54 GB | Efficiency: 98.69%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V4
    - Batch 100   | VRAM: 0.98 GB | Efficiency: 98.80%
    - Batch 200   | VRAM: 1.85 GB | Efficiency: 98.74%
    - Batch 300   | VRAM: 2.89 GB | Efficiency: 98.74%
    - Batch 400   | VRAM: 3.84 GB | Efficiency: 98.73%
    - Batch 500   | VRAM: 4.73 GB | Efficiency: 98.74%
    - Batch 1000  | VRAM: 9.30 GB | Efficiency: 98.65%
    - Batch 2000  | VRAM: 18.77 GB | Efficiency: 98.69%
    - Batch 2100  | VRAM: 19.94 GB | Efficiency: 98.66%
    - Batch 2200  | VRAM: 20.77 GB | Efficiency: 98.70%
    - Batch 2300  | VRAM: 21.74 GB | Efficiency: 98.68%
    - Batch 2400  | VRAM: 22.73 GB | Efficiency: 98.70%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V5
    - Batch 100   | VRAM: 0.94 GB | Efficiency: 98.68%
    - Batch 200   | VRAM: 1.97 GB | Efficiency: 98.66%
    - Batch 300   | VRAM: 2.80 GB | Efficiency: 98.79%
    - Batch 400   | VRAM: 3.74 GB | Efficiency: 98.71%
    - Batch 500   | VRAM: 4.82 GB | Efficiency: 98.64%
    - Batch 1000  | VRAM: 9.31 GB | Efficiency: 98.65%
    - Batch 2000  | VRAM: 18.79 GB | Efficiency: 98.68%
    - Batch 2100  | VRAM: 19.75 GB | Efficiency: 98.69%
    - Batch 2200  | VRAM: 20.81 GB | Efficiency: 98.71%
    - Batch 2300  | VRAM: 21.78 GB | Efficiency: 98.70%
    - Batch 2400  | VRAM: 22.71 GB | Efficiency: 98.68%
    - Batch 2500  | FAIL (OOM Crash)

    --- Results ---
    PyTorch Baseline:      300 users
    Attention Kernel V1 : 2400 users
    Attention Kernel V2 : 2400 users
    Attention Kernel V3 : 2400 users
    Attention Kernel V4 : 2400 users
    Attention Kernel V5 : 2400 users
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