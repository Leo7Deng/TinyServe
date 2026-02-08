import torch
from typing import List

class KVCache:
    """Represents the Physical GPU Memory."""
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        self.block_size = block_size
        self.device = device
        self.num_layers = num_layers
        
        # Initialize free list
        self.free_blocks = list(range(num_blocks))
        
        # List of tensors, one for each layer
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        
        for _ in range(num_layers):
            k = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
            v = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
            self.k_cache.append(k)
            self.v_cache.append(v)
        
    def allocate_block(self):
        """Pop one free block ID from the list."""
        if not self.free_blocks:
            raise RuntimeError("OOM: GPU Out of Memory")
        return self.free_blocks.pop(0)
    
    def free_sequence(self, sequence: "Sequence"):
        """Recycle a user's blocks back to the free list"""
        for block_id in sequence.block_table:
            self.free_blocks.append(block_id)
        sequence.block_table = []
        
    def allocate_for_prefill(self, sequence: "Sequence", num_prompt_tokens: int):
        """Called once at the start. Allocates enough blocks to hold the user's prompt."""
        sequence.num_tokens = num_prompt_tokens
        blocks_needed = (num_prompt_tokens + self.block_size - 1) // self.block_size
        
        for _ in range(blocks_needed):
            block_id = self.allocate_block()
            sequence.block_table.append(block_id)
            
    def allocate_slot_for_next_token(self, sequence: "Sequence"):
        """Called every step. Checks if the user needs a new block for the new token."""
        # Check if the current block is full
        # We check BEFORE incrementing num_tokens.
        # Example: Size 16. Tokens 0-15 fit. 
        # When num_tokens=16 (meaning we have 16 items), we need a new block for the 17th item (index 16).
        if sequence.num_tokens % self.block_size == 0:
            new_block = self.allocate_block()
            sequence.block_table.append(new_block)
            
        sequence.num_tokens += 1
        
    def get_slot_mapping(self, sequences: List["Sequence"]) -> torch.Tensor:
        """The bridge from User X to Physical Address."""
        slot_mapping = []
        for seq in sequences:
            # get most recent block id
            physical_block_id = seq.block_table[-1]
            
            # token 17 (index 16) -> 16 % 16 = slot 0
            # We use (num_tokens - 1) because we just allocated space for it
            block_offset = (seq.num_tokens - 1) % self.block_size
            
            # calculate flat address which is what the CUDA kernel needs
            flat_slot_id = physical_block_id * self.block_size + block_offset
            slot_mapping.append(flat_slot_id)
            
        return torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
    
    def get_block_table_tensor(self, sequences: List["Sequence"], max_len: int) -> torch.Tensor:
        """Converts Python lists -> GPU tensor for the attention kernel."""
        tensor = torch.full((len(sequences), max_len), -1, dtype=torch.int32, device=self.device)
        
        for i, seq in enumerate(sequences):
            blocks = seq.block_table
            tensor[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
            
        return tensor