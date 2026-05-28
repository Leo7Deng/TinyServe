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


    --- GQA Attention Test for V5/V6/V7 (32 Q heads, 4 KV heads) ---

    Computing PyTorch GQA Reference

    Testing Attention Kernel V5 (GQA) Pass: GQA output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.02329263463616371, -0.17037571966648102, 0.08549156039953232]
    Sample Ref    (User 0, Head 0): [0.023292670026421547, -0.1703757345676422, 0.08549157530069351]

    Testing Attention Kernel V6 (GQA + online softmax) Pass: GQA output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.023292647674679756, -0.17037570476531982, 0.08549156785011292]
    Sample Ref    (User 0, Head 0): [0.023292670026421547, -0.1703757345676422, 0.08549157530069351]

    Testing Attention Kernel V7 (GQA + bank-conflict optimized reduction) Pass: GQA output matched PyTorch reference!
    Sample Kernel (User 0, Head 0): [0.023292647674679756, -0.17037570476531982, 0.08549156785011292]
    Sample Ref    (User 0, Head 0): [0.023292670026421547, -0.1703757345676422, 0.08549157530069351]

### `test_throughput.py`
Measures the effective Memory Bandwidth (GB/s) and kernel latency (ms) of the engine, comparing TinyServe kernels against PyTorch SDPA baselines. This quantifies the overhead introduced by memory indirection while keeping the reference path closer to production inference. The test uses separate MHA and GQA baselines so `v1-v4` and `v5-v7` are compared against matching head layouts.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Throughput Config: Batch=64, Context=1024-4096, VRAM Reserved: 2.68 GB ---

    Running Attention Kernel V1
    Latency: 126.431 ms | Effective Bandwidth: 1435.04 GB/s

    Running Attention Kernel V2
    Latency: 52.684 ms | Effective Bandwidth: 3373.36 GB/s

    Running Attention Kernel V3
    Latency: 51.581 ms | Effective Bandwidth: 3306.59 GB/s

    Running Attention Kernel V4
    Latency: 26.552 ms | Effective Bandwidth: 6382.10 GB/s

    Running Attention Kernel V5
    Latency: 18.634 ms | Effective Bandwidth: 1103.04 GB/s

    Running Attention Kernel V6
    Latency: 6.254 ms | Effective Bandwidth: 3327.38 GB/s

    Running Attention Kernel V7
    Latency: 2.783 ms | Effective Bandwidth: 8334.72 GB/s

    Running: PyTorch Baseline (SDPA, MHA)
    Latency: 20.932 ms | Effective Bandwidth: 8410.63 GB/s

    Running: PyTorch Baseline (SDPA, GQA)
    Latency: 28.688 ms | Effective Bandwidth: 706.32 GB/s

    --- Results vs PyTorch SDPA ---
    Attention Kernel V1: 6.04x slower than PyTorch
    Attention Kernel V2: 2.52x slower than PyTorch
    Attention Kernel V3: 2.46x slower than PyTorch
    Attention Kernel V4: 1.27x slower than PyTorch
    Attention Kernel V5: 1.54x faster than PyTorch
    Attention Kernel V6: 4.59x faster than PyTorch
    Attention Kernel V7: 10.31x faster than PyTorch

### `test_max_concurrency.py`
Stress tests the memory manager by simulating irregular sequence lengths (Zipfian distribution) and incrementally increasing batch size until the GPU hits Out Of Memory (OOM). This demonstrates the reduction in KV Cache Fragmentation, showing exactly how many more concurrent users TinyServe can handle compared to contiguous allocation.

_In this test, attention kernel versions should not have much affect on results, but I am running all kernel versions to make sure shared memory optimizations are not reducing max concurrency._

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    --- Max Concurrency Stress Test (Zipfian Distribution) ---
    Simulating traffic: 90% Short (64-512), 10% Long (2048-4096)

    PyTorch
    - Batch 100   | VRAM: 6.51 GB | Efficiency: 14.34%
    - Batch 200   | VRAM: 13.40 GB | Efficiency: 14.05%
    - Batch 300   | VRAM: 19.74 GB | Efficiency: 13.86%
    - Batch 400   | FAIL (OOM Crash)

    Attention Kernel V1
    - Batch 100   | VRAM: 0.99 GB | Efficiency: 98.77%
    - Batch 200   | VRAM: 1.96 GB | Efficiency: 98.79%
    - Batch 300   | VRAM: 2.76 GB | Efficiency: 98.57%
    - Batch 400   | VRAM: 3.82 GB | Efficiency: 98.79%
    - Batch 500   | VRAM: 4.67 GB | Efficiency: 98.68%
    - Batch 1000  | VRAM: 9.45 GB | Efficiency: 98.72%
    - Batch 2000  | VRAM: 18.73 GB | Efficiency: 98.64%
    - Batch 2100  | VRAM: 19.83 GB | Efficiency: 98.72%
    - Batch 2200  | VRAM: 21.08 GB | Efficiency: 98.70%
    - Batch 2300  | VRAM: 21.66 GB | Efficiency: 98.70%
    - Batch 2400  | VRAM: 22.43 GB | Efficiency: 98.67%
    - Batch 2500  | VRAM: 23.32 GB | Efficiency: 98.67%

    Attention Kernel V2
    - Batch 100   | VRAM: 0.95 GB | Efficiency: 98.57%
    - Batch 200   | VRAM: 1.91 GB | Efficiency: 98.85%
    - Batch 300   | VRAM: 2.78 GB | Efficiency: 98.60%
    - Batch 400   | VRAM: 3.79 GB | Efficiency: 98.71%
    - Batch 500   | VRAM: 4.87 GB | Efficiency: 98.73%
    - Batch 1000  | VRAM: 9.40 GB | Efficiency: 98.65%
    - Batch 2000  | VRAM: 19.05 GB | Efficiency: 98.71%
    - Batch 2100  | VRAM: 19.80 GB | Efficiency: 98.69%
    - Batch 2200  | VRAM: 20.39 GB | Efficiency: 98.62%
    - Batch 2300  | VRAM: 21.64 GB | Efficiency: 98.68%
    - Batch 2400  | VRAM: 22.57 GB | Efficiency: 98.67%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V3
    - Batch 100   | VRAM: 0.97 GB | Efficiency: 98.68%
    - Batch 200   | VRAM: 1.90 GB | Efficiency: 98.63%
    - Batch 300   | VRAM: 2.69 GB | Efficiency: 98.58%
    - Batch 400   | VRAM: 3.90 GB | Efficiency: 98.67%
    - Batch 500   | VRAM: 4.67 GB | Efficiency: 98.66%
    - Batch 1000  | VRAM: 9.46 GB | Efficiency: 98.70%
    - Batch 2000  | VRAM: 18.68 GB | Efficiency: 98.69%
    - Batch 2100  | VRAM: 19.67 GB | Efficiency: 98.65%
    - Batch 2200  | VRAM: 20.58 GB | Efficiency: 98.65%
    - Batch 2300  | VRAM: 21.41 GB | Efficiency: 98.68%
    - Batch 2400  | VRAM: 22.71 GB | Efficiency: 98.69%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V4
    - Batch 100   | VRAM: 0.94 GB | Efficiency: 98.58%
    - Batch 200   | VRAM: 1.87 GB | Efficiency: 98.55%
    - Batch 300   | VRAM: 2.87 GB | Efficiency: 98.65%
    - Batch 400   | VRAM: 3.68 GB | Efficiency: 98.71%
    - Batch 500   | VRAM: 4.72 GB | Efficiency: 98.67%
    - Batch 1000  | VRAM: 9.32 GB | Efficiency: 98.66%
    - Batch 2000  | VRAM: 19.16 GB | Efficiency: 98.71%
    - Batch 2100  | VRAM: 19.84 GB | Efficiency: 98.69%
    - Batch 2200  | VRAM: 20.79 GB | Efficiency: 98.73%
    - Batch 2300  | VRAM: 21.62 GB | Efficiency: 98.67%
    - Batch 2400  | VRAM: 22.66 GB | Efficiency: 98.73%
    - Batch 2500  | VRAM: 23.27 GB | Efficiency: 98.69%

    Attention Kernel V5
    - Batch 100   | VRAM: 0.96 GB | Efficiency: 98.78%
    - Batch 200   | VRAM: 1.81 GB | Efficiency: 98.58%
    - Batch 300   | VRAM: 2.84 GB | Efficiency: 98.64%
    - Batch 400   | VRAM: 3.89 GB | Efficiency: 98.76%
    - Batch 500   | VRAM: 4.68 GB | Efficiency: 98.72%
    - Batch 1000  | VRAM: 9.59 GB | Efficiency: 98.75%
    - Batch 2000  | VRAM: 19.21 GB | Efficiency: 98.70%
    - Batch 2100  | VRAM: 19.33 GB | Efficiency: 98.65%
    - Batch 2200  | VRAM: 20.37 GB | Efficiency: 98.65%
    - Batch 2300  | VRAM: 21.49 GB | Efficiency: 98.67%
    - Batch 2400  | VRAM: 22.62 GB | Efficiency: 98.70%
    - Batch 2500  | FAIL (OOM Crash)

    Attention Kernel V6
    - Batch 100   | VRAM: 0.92 GB | Efficiency: 98.66%
    - Batch 200   | VRAM: 2.02 GB | Efficiency: 98.80%
    - Batch 300   | VRAM: 2.75 GB | Efficiency: 98.68%
    - Batch 400   | VRAM: 3.80 GB | Efficiency: 98.73%
    - Batch 500   | VRAM: 4.73 GB | Efficiency: 98.68%
    - Batch 1000  | VRAM: 9.37 GB | Efficiency: 98.67%
    - Batch 2000  | VRAM: 18.97 GB | Efficiency: 98.68%
    - Batch 2100  | VRAM: 19.65 GB | Efficiency: 98.68%
    - Batch 2200  | VRAM: 20.60 GB | Efficiency: 98.71%
    - Batch 2300  | VRAM: 21.79 GB | Efficiency: 98.68%
    - Batch 2400  | VRAM: 22.65 GB | Efficiency: 98.71%
    - Batch 2500  | VRAM: 23.34 GB | Efficiency: 98.68%

    --- Results ---
    PyTorch Baseline:      300 users
    Attention Kernel V1 : 2500 users
    Attention Kernel V2 : 2400 users
    Attention Kernel V3 : 2400 users
    Attention Kernel V4 : 2500 users
    Attention Kernel V5 : 2400 users
    Attention Kernel V6 : 2500 users
    TinyServe handles 8.3x more concurrent users!

### `test_scheduler.py`
Unit tests the scheduler logic on CPU. Verifies FIFO admission, sequence-count and token-count limits, VRAM block availability checks, cleanup of finished requests, and decode-time block growth.

#### Results
A10 (24 GB PCIe) 30vCPUs, 200 GiB RAM, 1.4 TiB SSD

    ......
    ----------------------------------------------------------------------
    Ran 6 tests in 0.001s

    OK

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
