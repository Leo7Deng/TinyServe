import torch
import tinyserve_ext
import time

def benchmark():
    SEED = 67
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Let's use a reasonable load to stress GPU
    num_seqs = 50
    min_seq_len = 1000
    max_seq_len = 1500
    num_heads = 32
    head_dim = 64
    block_size = 16
    
    device = "cuda"
    dtype = torch.float32
    
    # Calculate how many max blocks we need per sequence
    max_blocks_per_seq  = (max_seq_len + block_size - 1) // block_size
    total_blocks_needed = num_seqs * max_blocks_per_seq
    # Physical heap size (total slots available on GPU), add a buffer to be safe
    max_num_blocks = total_blocks_needed + 1024
    
    print(f"Config: Batch={num_seqs}, Context={min_seq_len}-{max_seq_len}, Heads={num_heads}")
    print(f"Max Blocks Needed: {max_num_blocks}")
    print(f"VRAM Used for KV Cache: {max_num_blocks * block_size * num_heads * head_dim * 4 / 1e9:.2f} GB")
    print("Allocating and scattering blocks randomly.")
    
    k_cache = torch.randn(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # 1 token per user
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    out = torch.empty_like(q)
    
    block_tables = torch.full((num_seqs, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    
    # This introduced thread divergence where some threads may finish earlier than others
    lens = torch.randint(min_seq_len, max_seq_len + 1, (num_seqs,), dtype=torch.int32, device=device)
    print(f"Sequence Lengths (Sample): {lens[:5].tolist()}")
    
    # Randomize order of physical blocks
    physical_block_pool = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)
    pool_idx = 0
    for i in range(num_seqs):
        seq_len = lens[i].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        
        allocated_blocks = physical_block_pool[pool_idx : pool_idx + num_blocks]
        pool_idx += num_blocks
        
        # Write physical blocks to the table
        block_tables[i, :num_blocks] = allocated_blocks
        
    print(f"Fragmentation: User 0's first 5 blocks are at: {block_tables[0, :5].tolist()}")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        tinyserve_ext.paged_attention_v1(
            out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
            block_tables.data_ptr(), lens.data_ptr(),
            num_seqs, num_heads, head_dim, max_blocks_per_seq, block_size
        )
    torch.cuda.synchronize()
    
    iterations = 100
    print(f"Running {iterations} iterations for average latency")
    
    start_event.record()
    for _ in range(iterations):
        tinyserve_ext.paged_attention_v1(
            out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
            block_tables.data_ptr(), lens.data_ptr(),
            num_seqs, num_heads, head_dim, max_blocks_per_seq, block_size
        )
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_latency = elapsed_time_ms / iterations
    print(f"Average Latency: {avg_latency:.3f} ms")
    
    # Bandwith Calculation
    total_tokens = lens.sum().item()
    total_bytes = (total_tokens * num_heads * head_dim * 4) * 2
    gb_processed = total_bytes / 1e9
    
    bandwidth = gb_processed / (avg_latency / 1000.0)
    print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")
    
    # PyTorch Baseline
    k_contig = torch.randn(num_seqs, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    q_expanded = q.unsqueeze(2)
    
    start_event.record()
    for _ in range(iterations):
        torch.matmul(q_expanded, k_contig.transpose(-1, -2))
    end_event.record()
    torch.cuda.synchronize()
    
    torch_latency = start_event.elapsed_time(end_event) / iterations
    print(f"PyTorch Latency: {torch_latency:.3f} ms")
    print(f"Slowdown Factor: {avg_latency / torch_latency:.1f}x slower than PyTorch")
    
if __name__ == "__main__":
    benchmark()
    