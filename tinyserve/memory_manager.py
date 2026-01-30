import tinyserve_ext

class BlockTable:
    """Represents the 'Page Table' for each sequence."""
    def __init__(self, seq_id, block_size):
        self.seq_id = seq_id
        self.block_size = block_size
        self.blocks = []
        
    def num_blocks(self):
        return len(self.blocks)
    
    def append_blocks(self, block_ids):
        self.blocks.extend(block_ids)
        
class MemoryManager:
    def __init__(self, total_blocks, block_size):
        self.allocator = tinyserve_ext.BlockAllocator(total_blocks, block_size)
        self.block_tables = {} # maps seq_id -> BlockTable
        
    def allocate_request(self, seq_id, prompt_len):
        # Ex. 30 tokens / 16 block_size = 2 blocks
        blocks_needed = (prompt_len + self.allocator.get_block_size() - 1) // self.allocator.get_block_size()
        
        block_ids = self.allocator.allocate(blocks_needed)
        
        table = BlockTable(seq_id, self.allocator.get_block_size())
        table.append_blocks(block_ids)
        self.block_tables[seq_id] = table
        
        return block_ids
    
    def free_request(self, seq_id):
        if seq_id not in self.block_tables:
            return
        
        table = self.block_tables[seq_id]
        self.allocator.free(table.blocks)
        del self.block_tables[seq_id]
        
        