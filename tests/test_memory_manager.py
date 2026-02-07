import torch
import tinyserve_ext
from tinyserve.memory_manager import KVCache, Sequence

def test_allocation_logic():
    block_size = 4
    # Create small cache with 4 blocks total (capacity would be 16 tokens)
    cache = KVCache(num_blocks=4, block_size=block_size, num_heads=2, head_dim=4, device="cuda")
    user = Sequence(seq_id=0)
    
    # 1. Test Prefill Allocation
    # Request 5 tokens. Should need 2 blocks (size 4 + size 4)
    print("1. Requesting 5 tokens (Prefill)")
    cache.allocate_for_prefill(user, num_prompt_tokens=5)
    
    print(f"   - User Blocks: {user.block_table}")
    print(f"   - Token Count: {user.num_tokens}")
    
    assert len(user.block_table) == 2, f"Expected 2 blocks, got {len(user.block_table)}"
    assert user.num_tokens == 5, "Token count mismatch"
    
    # 2. Test Incremental Allocation (Decoding)
    # We have 5 tokens. Capacity of 2 blocks is 8.
    # Test to add 3 more tokens without needing a new block
    print("\n2. Generating 3 more tokens (should NOT add block)")
    
    for _ in range(3):
        cache.allocate_slot_for_next_token(user)
        
    print(f"   - User Blocks: {user.block_table}")
    print(f"   - Token Count: {user.num_tokens}")
    assert len(user.block_table) == 2, "Should still have 2 blocks"
    assert user.num_tokens == 8, "Should have 8 tokens"
    
    # 3. Test Boundary Crossing
    # We have 8 tokens, the next one (9th) will require a 3rd block
    print("\n3. Generating 1 more token (should ADD block)")
    cache.allocate_slot_for_next_token(user)
    
    print(f"   - User Blocks: {user.block_table}")
    assert len(user.block_table) == 3, "Should have allocated 3rd block"
    
    
def test_kernel_write():
    device = "cuda"
    block_size = 4
    num_heads = 2
    head_dim = 4
    
    cache = KVCache(num_blocks=4, block_size=block_size, 
                    num_heads=num_heads, head_dim=head_dim, device=device)
    
    # User has 3 tokens. They sit in block 0. We are generating the 4th token.
    user = Sequence(seq_id=0)
    cache.allocate_for_prefill(user, num_prompt_tokens=3)
    
    # 1. Prepare slot for 4th token
    cache.allocate_slot_for_next_token(user)
    
    # 2. Get Physical Address
    # returns a tensor
    slot_mapping = cache.get_slot_mapping([user])
    print(f"   - Slot Mapping: {slot_mapping}")
    
    # 3. Create Fake Data for just 1 token (all 9.0)
    k = torch.full((1, num_heads, head_dim), 9.0, device=device)
    v = torch.full((1, num_heads, head_dim), 9.0, device=device)
    
    # 4. Run Kernel
    # this adds the k, v, to slot_mapping location
    tinyserve_ext.reshape_and_cache(
        k, v,
        cache.k_cache, cache.v_cache,
        slot_mapping
    )
    
    # 5. Check GPU Memory
    # Look at Block 0 in the cache
    # Slot 3 should be 9.0. Slots 0-2 should be 0.0
    block_0 = cache.k_cache[user.block_table[0]]
    print(f"   - Block 0 Content (Last row ({slot_mapping}) row should be 9s):")
    print(block_0[:, 0, :]) # Print first head, this would show all time steps for block 0, head 0
    
    expected_value = block_0[3, 0, 0].item()
    if expected_value == 9.0:
        print("\nMemory manager test ran successfully! Kernel wrote data to the correct physical slot.")
    else:
        print(f"Expected 9.0, got {expected_value}")

if __name__ == "__main__":
    test_allocation_logic()
    test_kernel_write()