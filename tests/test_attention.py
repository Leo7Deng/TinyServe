import torch
import tinyserve_ext
import math

def test_attention():
    SEED = 67
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    num_seqs = 2            # User A and User B
    num_heads = 32
    head_dim = 64
    block_size = 16         # blocks are chunks of memory created for PagedAttention. Each block holds 16 time steps, and for each of those time steps, it holds all 32 heads
    max_num_blocks = 100    # size of physical heap
    dtype = torch.float32
    device = "cuda"
    
    print(f"--- Attention Config: {num_seqs} Seqs, {num_heads} Heads, Block Size {block_size} ---\n")
    
    # Create random sequences
    # User A: 30 tokens (spans 2 blocks)
    # User B: 55 tokens (spans 4 blocks)
    seq_lengths = [30, 55]
    
    # store the "true" history for validation later
    true_keys = []
    true_values = []
    current_queries = []
    
    # To generate the "correct" answer, we create k, v, and q
    # We are using dimension [seq_len, num_heads, head_dim] because we seq_len acts as iterating through time
    # At each seq_len, we have 32 heads with each head_dim to find different attention scores for the query
    for seq_len in seq_lengths:
        # history (past tokens)
        k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
        # current query
        q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
        
        true_keys.append(k)
        true_values.append(v)
        current_queries.append(q)
        
    # Simulate the paged cache
    # This creates x number of blocks, with y block size, each block holds 
    # This creates a pre allocated "Heap" of 'max_num_blocks' physical blocks on the GPU.
    # Each block holds 'block_size' tokens (timesteps).
    # For every single token slot, we store the vectors for ALL 'num_heads' heads.
    k_cache = torch.zeros(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(max_num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # block table with shape [num_seqs, max_blocks_per_seq]
    max_blocks_per_seq = 10
    block_tables = torch.full((num_seqs, max_blocks_per_seq), -1, device=device, dtype=torch.int32)
    
    # scatter the data into the cache into random physical blocks
    pool_idx = 0
    physical_blocks_pool = torch.randperm(max_num_blocks)
    for seq_idx, (k_seq, v_seq) in enumerate(zip(true_keys, true_values)):
        seq_len = k_seq.shape[0]
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        for logical_idx in range(num_blocks_needed):
            # assign a physical block
            physical_block = physical_blocks_pool[pool_idx]
            pool_idx += 1
            
            # write to block table
            block_tables[seq_idx, logical_idx] = physical_block
            
            # copy data to cache
            start_token = logical_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            chunk_size = end_token - start_token
            
            # fill the physical blocks k_cache and v_cache
            # match the shape [num_blocks, block_size, num_heads, head_dim]
            k_cache[physical_block, :chunk_size, :, :] = k_seq[start_token:end_token, :, :]
            v_cache[physical_block, :chunk_size, :, :] = v_seq[start_token:end_token, :, :]
            
    # flatten queries into [num_seqs, num_heads, head_dim]
    q_tensor = torch.cat(current_queries, dim=0)
    lens_tensor = torch.tensor(seq_lengths, device=device, dtype=torch.int32)
    
    max_context_len = max(seq_lengths)

    print("Computing PyTorch Reference")
    ref_outputs = []
    scale = 1.0 / math.sqrt(head_dim)
    
    for i in range(num_seqs):
        q = current_queries[i]
        k = true_keys[i]
        v = true_values[i]
        
        # Reshape for Matmul, shape into [Batch, Head, Sequence, Dim]
        # We move num_heads to the front so we can process all 4 heads in parallel
        
        # Q: [1, 32, 64] -> [32, 1, 64]
        q_ref = q.permute(1, 0, 2)
        
        # K: [30, 4, 64] -> [32, 30, 64]
        k_ref = k.permute(1, 0, 2)
        
        # V: [30, 4, 64] -> [32, 30, 64]
        v_ref = v.permute(1, 0, 2)
        
        # Compute scores (Q * K_transpose)
        # k_ref transpose: [32, 30, 64] -> [32, 64, 30]
        # Matmul: [32, 1, 64] @ [32, 64, 30] -> [32, 1, 30]
        # The result [32, 1, 30] means: for each of the 32 heads, we have 1 query matching 30 past tokens.
        scores = torch.matmul(q_ref, k_ref.transpose(1, 2)) * scale
        
        probs = torch.softmax(scores, dim=-1)
        
        # Compute output (probs * V)
        # Matmul: [32, 1, 30] @ [32, 30, 64] = [32, 1, 64]
        # This means: for each of the 32 heads, we get a weighted sum vector of size 64
        out_ref = torch.matmul(probs, v_ref)
        
        # Permute back to [1, 32, 64] so it matches the q_tensor shape
        out_ref = out_ref.permute(1, 0, 2)

        ref_outputs.append(out_ref)
        
    ref_tensor = torch.cat(ref_outputs, dim=0)
    
    mha_kernels = [
        ("Attention Kernel V1", tinyserve_ext.paged_attention_v1),
        ("Attention Kernel V2", tinyserve_ext.paged_attention_v2),
        ("Attention Kernel V3", tinyserve_ext.paged_attention_v3),
        ("Attention Kernel V4", tinyserve_ext.paged_attention_v4),
    ]

    for name, kernel_func in mha_kernels:
        print(f"\nTesting {name}", end=" ")
        
        out_tensor = torch.empty_like(q_tensor)
        
        try:
            kernel_func(
                out_tensor,
                q_tensor,
                k_cache,
                v_cache,
                block_tables,
                lens_tensor
            )
            
            if torch.allclose(out_tensor, ref_tensor, atol=1e-3, rtol=1e-3):
                print("Pass: Kernel output matched PyTorch reference!")
                print(f"Sample Kernel (User 0, Head 0): {out_tensor[0,0,:3].tolist()}")
                print(f"Sample Ref    (User 0, Head 0): {ref_tensor[0,0,:3].tolist()}")
            else:
                print("Fail: Outputs do not match.")
                diff = (out_tensor - ref_tensor).abs().max().item()
                print(f"Max Difference: {diff}")
                print("Kernel Output:\n", out_tensor[0,0,:5])
                print("Reference Output:\n", ref_tensor[0,0,:5])
                
        except Exception as e:
            print(f"CRASH: {e}")

    # --- GQA Test for V5 ---
    # V5 uses separate KV head count (4 KV heads for 32 Q heads)
    # We need a separate cache, reference, and block table for this
    print("\n\n--- GQA Attention Test for V5 (32 Q heads, 4 KV heads) ---")
    
    num_kv_heads = 4
    num_q_per_kv = num_heads // num_kv_heads  # 8
    
    # Generate KV data with only 4 heads
    gqa_true_keys = []
    gqa_true_values = []
    gqa_queries = []
    
    for seq_len in seq_lengths:
        k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
        gqa_true_keys.append(k)
        gqa_true_values.append(v)
        gqa_queries.append(q)
    
    # Build GQA cache with num_kv_heads
    gqa_k_cache = torch.zeros(max_num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    gqa_v_cache = torch.zeros(max_num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    gqa_block_tables = torch.full((num_seqs, max_blocks_per_seq), -1, device=device, dtype=torch.int32)
    
    pool_idx = 0
    physical_blocks_pool = torch.randperm(max_num_blocks)
    for seq_idx, (k_seq, v_seq) in enumerate(zip(gqa_true_keys, gqa_true_values)):
        seq_len = k_seq.shape[0]
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        for logical_idx in range(num_blocks_needed):
            physical_block = physical_blocks_pool[pool_idx]
            pool_idx += 1
            gqa_block_tables[seq_idx, logical_idx] = physical_block
            
            start_token = logical_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            chunk_size = end_token - start_token
            
            gqa_k_cache[physical_block, :chunk_size, :, :] = k_seq[start_token:end_token, :, :]
            gqa_v_cache[physical_block, :chunk_size, :, :] = v_seq[start_token:end_token, :, :]
    
    gqa_q_tensor = torch.cat(gqa_queries, dim=0)
    
    # Compute GQA reference: expand 4 KV heads -> 32 by repeating each 8 times
    print("\nComputing PyTorch GQA Reference")
    gqa_ref_outputs = []
    
    for i in range(num_seqs):
        q = gqa_queries[i]
        k = gqa_true_keys[i]
        v = gqa_true_values[i]
        
        # Q: [1, 32, 64] -> [32, 1, 64]
        q_ref = q.permute(1, 0, 2)
        
        # K: [seq_len, 4, 64] -> expand to [seq_len, 32, 64]
        # Each KV head is repeated 8 times to match the 32 Q heads
        k_expanded = k.unsqueeze(2).expand(-1, -1, num_q_per_kv, -1)
        k_expanded = k_expanded.reshape(k.shape[0], num_heads, head_dim)
        k_ref = k_expanded.permute(1, 0, 2)  # [32, seq_len, 64]
        
        v_expanded = v.unsqueeze(2).expand(-1, -1, num_q_per_kv, -1)
        v_expanded = v_expanded.reshape(v.shape[0], num_heads, head_dim)
        v_ref = v_expanded.permute(1, 0, 2)
        
        scores = torch.matmul(q_ref, k_ref.transpose(1, 2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(probs, v_ref)
        out_ref = out_ref.permute(1, 0, 2)
        gqa_ref_outputs.append(out_ref)
    
    gqa_ref_tensor = torch.cat(gqa_ref_outputs, dim=0)
    
    gqa_kernels = [
        ("Attention Kernel V5 (GQA)", tinyserve_ext.paged_attention_v5),
        ("Attention Kernel V6 (GQA + online softmax)", tinyserve_ext.paged_attention_v6),
    ]

    for name, kernel_func in gqa_kernels:
        print(f"\nTesting {name}", end=" ")
        gqa_out_tensor = torch.empty_like(gqa_q_tensor)

        try:
            kernel_func(
                gqa_out_tensor,
                gqa_q_tensor,
                gqa_k_cache,
                gqa_v_cache,
                gqa_block_tables,
                lens_tensor
            )

            if torch.allclose(gqa_out_tensor, gqa_ref_tensor, atol=1e-3, rtol=1e-3):
                print("Pass: GQA output matched PyTorch reference!")
                print(f"Sample Kernel (User 0, Head 0): {gqa_out_tensor[0,0,:3].tolist()}")
                print(f"Sample Ref    (User 0, Head 0): {gqa_ref_tensor[0,0,:3].tolist()}")
            else:
                print("Fail: Outputs do not match.")
                diff = (gqa_out_tensor - gqa_ref_tensor).abs().max().item()
                print(f"Max Difference: {diff}")

        except Exception as e:
            print(f"CRASH: {e}")
        
if __name__ == "__main__":
    test_attention()
