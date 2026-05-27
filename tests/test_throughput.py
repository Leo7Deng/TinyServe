import torch
import torch.nn.functional as F
import tinyserve_ext


def calculate_effective_bandwidth_gbps(total_tokens, num_q_heads, num_kv_heads, head_dim, avg_latency_ms, dtype):
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    read_bytes = (total_tokens * num_kv_heads * head_dim * bytes_per_element) * 2
    write_bytes = num_q_heads * head_dim * bytes_per_element
    total_bytes = read_bytes + write_bytes
    return (total_bytes / 1e9) / (avg_latency_ms / 1000.0)

def run_attention_kernel(
    num_seqs, min_seq_len, max_seq_len, num_heads, head_dim, block_size, 
    device, dtype, max_blocks_per_seq, total_blocks_needed, max_num_blocks, 
    paged_attention_func, num_kv_heads=None
):
    cache_heads = num_kv_heads if num_kv_heads is not None else num_heads
    k_cache = torch.randn(max_num_blocks, block_size, cache_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(max_num_blocks, block_size, cache_heads, head_dim, device=device, dtype=dtype)
    
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
    
    # Throughput measurement
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
    bandwidth = calculate_effective_bandwidth_gbps(
        total_tokens,
        num_seqs * num_heads,
        num_seqs * cache_heads,
        head_dim,
        avg_latency,
        dtype,
    )
    
    print(f"Latency: {avg_latency:.3f} ms | Effective Bandwidth: {bandwidth:.2f} GB/s")
    return avg_latency

def run_sdpa_baseline(
    name,
    num_seqs,
    min_seq_len,
    max_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    device,
    dtype,
):
    print(f"\nRunning: {name}")

    k_contig = torch.randn(num_seqs, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    v_contig = torch.randn(num_seqs, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    q = torch.randn(num_seqs, num_q_heads, head_dim, device=device, dtype=dtype)

    # SDPA still sees a rectangular cache here, so we mask away padded tokens.
    lens = torch.randint(min_seq_len, max_seq_len + 1, (num_seqs,), device=device)
    mask = torch.arange(max_seq_len, device=device).expand(num_seqs, max_seq_len) < lens.unsqueeze(1)
    mask = mask.view(num_seqs, 1, 1, max_seq_len)

    q_expanded = q.unsqueeze(2)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iterations = 100

    for _ in range(10):
        F.scaled_dot_product_attention(
            q_expanded,
            k_contig,
            v_contig,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=(num_q_heads != num_kv_heads),
        )
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iterations):
        F.scaled_dot_product_attention(
            q_expanded,
            k_contig,
            v_contig,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=(num_q_heads != num_kv_heads),
        )
    end_event.record()
    torch.cuda.synchronize()

    torch_latency = start_event.elapsed_time(end_event) / iterations
    total_tokens = lens.sum().item()
    bandwidth = calculate_effective_bandwidth_gbps(
        total_tokens,
        num_seqs * num_q_heads,
        num_seqs * num_kv_heads,
        head_dim,
        torch_latency,
        dtype,
    )

    print(f"Latency: {torch_latency:.3f} ms | Effective Bandwidth: {bandwidth:.2f} GB/s")
    return torch_latency


def throughput():
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
    print(f"--- Throughput Config: Batch={num_seqs}, Context={min_seq_len}-{max_seq_len}, VRAM Reserved: {vram_reserved_size:.2f} GB ---")

    # 1. Measure paged attention kernels
    kernels = [
        ("Attention Kernel V1", tinyserve_ext.paged_attention_v1),
        ("Attention Kernel V2", tinyserve_ext.paged_attention_v2),
        ("Attention Kernel V3", tinyserve_ext.paged_attention_v3),
        ("Attention Kernel V4", tinyserve_ext.paged_attention_v4),
    ]
    
    # V5 and V6 use GQA (4 KV heads instead of 32), so they need separate caches.
    gqa_kernels = [
        ("Attention Kernel V5", tinyserve_ext.paged_attention_v5),
        ("Attention Kernel V6", tinyserve_ext.paged_attention_v6),
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
    
    for name, kernel_func in gqa_kernels:
        print(f"\nRunning {name}")
        try:
            latency = run_attention_kernel(
                num_seqs, min_seq_len, max_seq_len, num_heads, head_dim, block_size, 
                device, dtype, max_blocks_per_seq, total_blocks_needed, max_num_blocks, 
                kernel_func, num_kv_heads=4
            )
            kernel_latencies[name] = latency
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            kernel_latencies[name] = float('inf')

    torch_mha_latency = run_sdpa_baseline(
        "PyTorch Baseline (SDPA, MHA)",
        num_seqs,
        min_seq_len,
        max_seq_len,
        num_heads,
        num_heads,
        head_dim,
        device,
        dtype,
    )
    torch_gqa_latency = run_sdpa_baseline(
        "PyTorch Baseline (SDPA, GQA)",
        num_seqs,
        min_seq_len,
        max_seq_len,
        num_heads,
        4,
        head_dim,
        device,
        dtype,
    )

    print("\n--- Results vs PyTorch SDPA ---")
    for name, latency in kernel_latencies.items():
        if latency == float('inf'):
            print(f"{name}: Failed")
        else:
            baseline_latency = torch_gqa_latency if name in {"Attention Kernel V5", "Attention Kernel V6"} else torch_mha_latency
            speedup = baseline_latency / latency
            if speedup > 1.0:
                print(f"{name}: {speedup:.2f}x faster than PyTorch")
            else:
                print(f"{name}: {1.0/speedup:.2f}x slower than PyTorch")

if __name__ == "__main__":
    throughput()
