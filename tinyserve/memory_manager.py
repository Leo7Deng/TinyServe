import torch

class Sequence:
    """Represents one user's conversation."""
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.block_table = []   # list of physical blocks used
        self.num_tokens = 0
        
    def is_current_block_full(self, block_size):
        return self.num_tokens % block_size == 0
    
class KVCache:
    """Represents the Physical GPU Memory."""
    def __init__(self, num_blocks, block_size, num_heads, head_dim, device="cuda"):
        self.block_size = block_size
        self.device = device
        
        # torch.zeros with device="cuda" calls cudaMalloc if more GPU memory is needed
        # This data is stored on the GPU VRAM 
        self.k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device)
        self.v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device)
        
        # Initialize free list
        self.free_blocks = list(range(num_blocks))
        
    def allocate_block(self):
        """Pop one free block ID from the list."""
        if not self.free_blocks:
            raise RuntimeError("OOM: GPU Out of Memory")
        return self.free_blocks.pop(0)
    
    def free_sequence(self, sequence):
        """Recycle a user's blocks back to the free list"""
        for block_id in sequence.block_table:
            self.free_blocks.append(block_id)
        sequence.block_table = []
        
    def allocate_for_prefill(self, sequence, num_prompt_tokens):
        """Called once at the start. Allocates enough blocks to hold the user's prompt.

        Args:
            sequence (Sequence): User's sequence object
            num_prompt_tokens (int): Number of tokens from the user's prompt
        """
        sequence.num_tokens = num_prompt_tokens
        blocks_needed = (num_prompt_tokens + self.block_size - 1) // self.block_size
        
        for _ in range(blocks_needed):
            block_id = self.allocate_block()
            sequence.block_table.append(block_id)
            
    def allocate_slot_for_next_token(self, sequence: Sequence):
        """Called every step. Checks if the user needs a new block for the new token."""
        # if the last block is full, we need a new block before we write
        if sequence.is_current_block_full(self.block_size):
            new_block = self.allocate_block()
            sequence.block_table.append(new_block)
            
        # We don't write any data here, just make sure there is enough space in our "virtual memory"
        sequence.num_tokens += 1
        
    def get_slot_mapping(self, sequences):
        """The bridge from User X to Physical Address. Used by the reshape kernel."""
        slot_mapping = []
        for seq in sequences:
            # get most recent block id
            physical_block_id = seq.block_table[-1]
            
            # token 17 (index 16) -> 16 % 16 = slot 0
            block_offset = (seq.num_tokens - 1) % self.block_size
            
            # calculate flat address which is what the CUDA kernel needs
            flat_slot_id = physical_block_id * self.block_size + block_offset
            slot_mapping.append(flat_slot_id)
            
        # returns a tensor of physical locations per sequence, copying them over to GPU
        return torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
    
    def get_block_table_tensor(self, sequences, max_len):
        """Converts Python lists -> GPU tensor for the attention kernel.
        Shape: [num_seqs, max_blocks_per_seq]
        """
        # create a tensor filled with -1 (padding)
        tensor = torch.full((len(sequences), max_len), -1, dtype=torch.int32, device=self.device)
        
        for i, seq in enumerate(sequences):
            blocks = seq.block_table
            # copy the block IDs into the row
            tensor[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
            
        return tensor