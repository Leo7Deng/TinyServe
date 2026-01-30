import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinyserve import MemoryManager

def test_fragmentation():
    # Setup a tiny GPU that can only hold 10 * 16 tokens
    mgr = MemoryManager(total_blocks=10, block_size=16)
    
    # Fill 80% of the GPU with sequences
    mgr.allocate_request("A", 32)
    mgr.allocate_request("B", 32)
    mgr.allocate_request("C", 32)
    mgr.allocate_request("D", 32)
    
    print(f"Free Blocks: {mgr.allocator.get_free_block_count()}")
    assert mgr.allocator.get_free_block_count() == 2
    
    # Create memory fragmenation
    print("Freeing users A and C")
    mgr.free_request("A")
    mgr.free_request("C")
    
    print(f"Free Blocks: {mgr.allocator.get_free_block_count()}")
    assert mgr.allocator.get_free_block_count() == 6
    
    # Allocate sequence id E who neesd 5 blocks
    # In a contiguous memory system like PyTorch, this should fail.
    # In PagedAttention, this should succeed by getting blocks from non contiguous memory.
    print("Allocating sequence E (needs 5 blocks)")
    try:
        blocks = mgr.allocate_request("E", 80)
        print(f"User E successfully allocated blocks: {blocks}")
    except Exception as e:
        print(f"Failed: {e}")
        exit(1)
        
if __name__ == "__main__":
    test_fragmentation()
        
    
