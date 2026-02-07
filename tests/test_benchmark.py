import torch
import tinyserve_ext
import time
import math

def run_attention_kernel(
    num_seqs, min_seq_len, max_seq_len, num_heads, head_dim, block_size, 
    device, dtype, max_blocks_per_seq, total_blocks_needed, max_num_blocks, 
    paged_attention_func
):
    # Create the massive physical heap in GPU VRAM
    k_cache = torch.randn(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # 1 token query per user (Decoding Phase)
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    out = torch.empty_like(q)
    
    # Block Tables (maps sequence -> physical block, stored as a tensor in GPU, not very large)
    block_tables = torch.full((num_seqs, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    
    # Random sequence lengths
    lens = torch.randint(min_seq_len, max_seq_len + 1, (num_seqs,), dtype=torch.int32, device=device)
    
    # Scatter blocks randomly (simulate fragmentation)
    physical_block_pool = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)
    pool_idx = 0
    
    for i in range(num_seqs):
        seq_len = lens[i].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        
        allocated_blocks = physical_block_pool[pool_idx : pool_idx + num_blocks]
        pool_idx += num_blocks
        block_tables[i, :num_blocks] = allocated_blocks
        
    # Warmup
    for _ in range(10):
        paged_attention_func(
            out, q, k_cache, v_cache,
            block_tables, lens
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iterations = 100
    start_event.record()
    for _ in range(iterations):
        paged_attention_func(
            out, q, k_cache, v_cache,
            block_tables, lens
        )
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_latency = elapsed_time_ms / iterations
    
    total_tokens = lens.sum().item()
    # Cost = Read K (tokens) + Read V (tokens) + Write Out (1 token per user)
    # 4 bytes per float
    read_bytes = (total_tokens * num_heads * head_dim * 4) * 2 
    write_bytes = (num_seqs * num_heads * head_dim * 4)
    total_bytes = read_bytes + write_bytes
    
    gb_processed = total_bytes / 1e9
    bandwidth = gb_processed / (avg_latency / 1000.0)
    
    print(f"Latency: {avg_latency:.3f} ms | Effective Bandwidth: {bandwidth:.2f} GB/s")
    return avg_latency

def benchmark():
    SEED = 67
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    num_seqs = 64
    min_seq_len = 1024
    max_seq_len = 4096
    num_heads = 32
    head_dim = 64
    block_size = 16
    
    device = "cuda"
    dtype = torch.float32
    
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks_needed = num_seqs * max_blocks_per_seq
    max_num_blocks = total_blocks_needed + 4096 # Buffer
    
    vram_reserved_size = max_num_blocks * block_size * num_heads * head_dim * 4 / 1e9
    print(f"--- Benchmark Config: Batch={num_seqs}, Context={min_seq_len}-{max_seq_len}, VRAM Reserved: {vram_reserved_size:.2f} GB ---")

    # 1. Measure Kernels
    kernels = [
        ("Attention Kernel V1", tinyserve_ext.paged_attention_v1),
        ("Attention Kernel V2", tinyserve_ext.paged_attention_v2),
    ]
    
    kernel_latencies = {}
    for name, kernel_func in kernels:
        print(f"\nRunning {name}")
        try:
            latency = run_attention_kernel(
                num_seqs, min_seq_len, max_seq_len, num_heads, head_dim, block_size, 
                device, dtype, max_blocks_per_seq, total_blocks_needed, max_num_blocks, 
                kernel_func
            )
            kernel_latencies[name] = latency
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            kernel_latencies[name] = float('inf')

    # 2. Measure PyTorch Baseline
    print("\nRunning: PyTorch Baseline")
    # To be mathematically equivalent, we must simulate the masking.
    # PyTorch allocates a rectangular block [Batch, MaxSeq].
    # But real PagedAttention only processes valid tokens.
    
    k_contig = torch.randn(num_seqs, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    v_contig = torch.randn(num_seqs, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    
    # Create a mask to ensure PyTorch ignores the "padding" area
    # This adds slight overhead but is "correct"
    lens = torch.randint(min_seq_len, max_seq_len + 1, (num_seqs,), device=device)
    mask = torch.arange(max_seq_len, device=device).expand(num_seqs, max_seq_len) < lens.unsqueeze(1)
    mask = mask.view(num_seqs, 1, 1, max_seq_len) # [Batch, 1, 1, Seq] for broadcasting
    
    q_expanded = q.unsqueeze(2) # [Batch, Heads, 1, Dim]
    scale = 1.0 / math.sqrt(head_dim)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iterations = 100
    
    # Warmup
    for _ in range(10):
        scores = torch.matmul(q_expanded, k_contig.transpose(-1, -2)) * scale
        # Mask out padding with -inf so softmax ignores them
        scores = scores.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_contig)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iterations):
        scores = torch.matmul(q_expanded, k_contig.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_contig)
    end_event.record()
    torch.cuda.synchronize()
    
    torch_latency = start_event.elapsed_time(end_event) / iterations
    
    total_tokens = lens.sum().item()
    read_bytes = (total_tokens * num_heads * head_dim * 4) * 2 
    write_bytes = (num_seqs * num_heads * head_dim * 4)
    total_bytes = read_bytes + write_bytes
    
    gb_processed = total_bytes / 1e9
    pytorch_bandwidth = gb_processed / (torch_latency / 1000.0)
    
    print(f"Latency: {torch_latency:.3f} ms | Effective Bandwidth: {pytorch_bandwidth:.2f} GB/s")
    
    print("\n--- Results ---")
    for name, latency in kernel_latencies.items():
        if latency == float('inf'):
            print(f"{name}: Failed")
        else:
            speedup = torch_latency / latency
            if speedup > 1.0:
                print(f"{name}: {speedup:.2f}x faster than PyTorch")
            else:
                print(f"{name}: {1.0/speedup:.2f}x slower than PyTorch")

if __name__ == "__main__":
    benchmark()