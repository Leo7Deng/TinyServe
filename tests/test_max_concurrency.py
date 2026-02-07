import torch
import tinyserve_ext
import sys
import gc

def get_vram_usage_gb():
    return torch.cuda.memory_allocated() / 1e9

def cleanup():
    # Forces Python to release all GPU memory references
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def generate_zipfian_lengths(bs, min_seq_len, max_seq_len, device):
    short_users = int(bs * 0.90)
    long_users = bs - short_users
    
    # 90% are short (64 to 512 tokens)
    lens_short = torch.randint(min_seq_len, 512, (short_users,), dtype=torch.int32, device=device)
    
    # 10% are long (2048 to max_seq_len)
    lens_long = torch.randint(2048, max_seq_len, (long_users,), dtype=torch.int32, device=device)
    
    # Combine and shuffle
    lens = torch.cat([lens_short, lens_long])
    lens = lens[torch.randperm(bs)]
    
    return lens

def run_pytorch_test(batch_sizes, min_seq_len, max_seq_len, num_heads, head_dim, device, dtype):
    print("\nPyTorch")
    cleanup() 
    
    max_users = 0
            
    for bs in batch_sizes:
        try:
            # Generate Skewed Lenghts
            lens = generate_zipfian_lengths(bs, min_seq_len, max_seq_len, device)
            current_max = lens.max().item()
            
            # the rectangle: batch * max_len
            total_slots_allocated = bs * current_max
            # the actual data: sum of all lengths
            total_tokens_used = lens.sum().item()
            
            efficiency = (total_tokens_used / total_slots_allocated) * 100
            
            # PyTorch allocates for the LONGEST sequence in the batch because PyTorch tensors must be perfect rectangles
            # In PagedAttention however, this large rectangle gets broken down into small cubes
            # Even though 90% of users are < 512 tokens, they all pay for 4096.
            # Allocate, K and V are massive conitiguous blocks
            k = torch.empty(bs, num_heads, current_max, head_dim, device=device, dtype=dtype)
            v = torch.empty(bs, num_heads, current_max, head_dim, device=device, dtype=dtype)
            q = torch.randn(bs, num_heads, head_dim, device=device, dtype=dtype)
            
            # Run math
            torch.matmul(q.unsqueeze(2), k.transpose(-1, -2))
            
            # Check memory to see if it ran successfully
            mem = get_vram_usage_gb()
            print(f"- Batch {bs:<5} | VRAM: {mem:.2f} GB | Efficiency: {efficiency:.2f}%")
            max_users = bs
            
            del k, v, q, lens
            cleanup()
            
        except torch.cuda.OutOfMemoryError:
            print(f"- Batch {bs:<5} | FAIL (OOM Crash)")
            break
        except Exception as e:
            print(f"- Batch {bs:<5} | FAIL ({e})")
            break
            
    return max_users

def run_paged_test(batch_sizes, min_seq_len, max_seq_len, num_heads, head_dim, block_size, device, dtype, kernels):
    # We test all kernels to ensure optimizations don't regress capacity (e.g. using too much shared mem)
    
    results = {}
    
    for name, kernel_func in kernels:
        print(f"\n{name}")
        cleanup() 
        max_users = 0
        for bs in batch_sizes:
            try:
                lens = generate_zipfian_lengths(bs, min_seq_len, max_seq_len, device)
                
                # Calculate internal fragmentation
                blocks_per_seq = (lens + block_size - 1) // block_size
                num_total_blocks = blocks_per_seq.sum().item()
                
                total_slots_allocated = num_total_blocks * block_size
                total_tokens_used = lens.sum().item()
                efficiency = (total_tokens_used / total_slots_allocated) * 100
                
                k_cache = torch.empty(num_total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
                v_cache = torch.empty(num_total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
                
                max_blocks_per_seq = (max_seq_len // block_size) + 1
                block_tables = torch.zeros(bs, max_blocks_per_seq, dtype=torch.int32, device=device)
                
                q = torch.randn(bs, num_heads, head_dim, device=device, dtype=dtype)
                out = torch.empty_like(q)
                
                # Run Kernel
                kernel_func(
                    out, q, k_cache, v_cache,
                    block_tables, lens
                )
                
                mem = get_vram_usage_gb()
                print(f"- Batch {bs:<5} | VRAM: {mem:.2f} GB | Efficiency: {efficiency:.2f}%")
                
                max_users = bs
                
                del k_cache, v_cache, q, out, block_tables, lens
                cleanup()
                
            except torch.cuda.OutOfMemoryError:
                print(f"- Batch {bs:<5} | FAIL (OOM Crash)")
                break
            except Exception as e:
                print(f"- Batch {bs:<5} | FAIL ({e})")
                break
        
        results[name] = max_users
            
    return results

def main():
    print("--- Max Concurrency Stress Test (Zipfian Distribution) ---")
    print("Simulating traffic: 90% Short (64-512), 10% Long (2048-4096)")
    
    device = "cuda"
    dtype = torch.float32
    
    num_heads = 32
    head_dim = 64
    block_size = 16
    min_seq_len = 64
    max_seq_len = 4096 
    
    batch_sizes = [100, 200, 300, 400, 500, 1000, 2000, 2100, 2200, 2300, 2400, 2500]
    
    max_pytorch = run_pytorch_test(batch_sizes, min_seq_len, max_seq_len, num_heads, head_dim, device, dtype)

    cleanup()
    
    kernels = [
        ("Attention Kernel V1", tinyserve_ext.paged_attention_v1),
        ("Attention Kernel V2", tinyserve_ext.paged_attention_v2),
        ("Attention Kernel V3", tinyserve_ext.paged_attention_v3),
        ("Attention Kernel V4", tinyserve_ext.paged_attention_v4),
    ]
    
    paged_results = run_paged_test(batch_sizes, min_seq_len, max_seq_len, num_heads, head_dim, block_size, device, dtype, kernels)

    print("\n--- Results ---")
    print(f"PyTorch Baseline:      {max_pytorch} users")
    
    best_paged = 0
    for name, score in paged_results.items():
        print(f"{name:<20}: {score} users")
        best_paged = max(best_paged, score)
    
    if best_paged > max_pytorch:
        ratio = best_paged / max_pytorch
        print(f"TinyServe handles {ratio:.1f}x more concurrent users!")
    else:
        print("Test inconclusive: No concurrency gain observed.")

if __name__ == "__main__":
    main()