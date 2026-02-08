from .memory_manager import KVCache
from collections import deque
from typing import List, Optional
import enum
import time

class SequenceStatus(enum.Enum):
    WAITING = 0     # waiting in queue for "prefill"
    RUNNING = 1     # currently executing tokens
    FINISHED = 2    # done (EOS or max len)

class Sequence:
    """Represents a "fork" of user's conversation."""
    def __init__(self, seq_id, prompt, prompt_token_ids: List[int], block_size):
        self.seq_id = seq_id
        self.prompt = prompt
        # Make a copy of the list so we don't modify the original input
        self.data = list(prompt_token_ids)
        self.block_size = block_size
        
        self.status = SequenceStatus.WAITING
        self.output_text = ""
        
        self.prompt_len = len(prompt_token_ids)
        self.block_table = []   # list of physical blocks used  
        
    def append_token_id(self, token_id, status: SequenceStatus):
        self.data.append(token_id)
        self.status = status
        
    def get_len(self):
        return len(self.data)
    
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    def get_last_token_id(self):
        return self.data[-1]
    
class SequenceGroup:
    """
    Sequence group will only hold one sequence most of the time (like ChatGPT). 
    owever, in benchmarking and special use cases, beam search is needed, and 
    SequenceGroup will significantly decrease the amount of memory used, with 
    little overhead when it is not needed.
    """
    def __init__(self, request_id, seqs: List[Sequence]):
        self.request_id = request_id
        self.seqs = seqs
        self.arrival_time = time.time()
        
    def get_seqs(self, status: Optional[SequenceStatus] = None):
        if status is None:
            return self.seqs
        return [s for s in self.seqs if s.status == status]
    
    def is_finished(self):
        return all(seq.is_finished() for seq in self.seqs)

# class to schedule continuous sequences
class Scheduler:
    def __init__(
        self,
        kv_cache: KVCache,
        max_num_seqs: int = 256,            # max batch size, this limit is for the CPU, too many users to handle will cause too much scheduling overhead
        max_num_batched_tokens: int = 4096  # max tokens per step, this limit is for the GPU, too many tokens will cause OOM error
    ):
        self.kv_cache = kv_cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        
        self.waiting = deque()
        self.running = []

    def add_request(self, request: SequenceGroup):
        self.waiting.append(request)
        
    def schedule(self):
        """
        1. Free finished sequences
        2. Make sure running sequences have space for the next token
        3. Add new sequences from waiting queue if possible
        
        Returns: The batch of groups to run this step
        """
        # keep track of which sequence groups are still running
        active_groups = []
        
        for group in self.running:
            if group.is_finished():
                # sequence group is done, release the blocks held by the sequence
                for seq in group.get_seqs():
                    self.kv_cache.free_sequence(seq)
            else:
                # create space for the next token generation if needed
                # catch OOM errors
                try:
                    for seq in group.get_seqs(status=SequenceStatus.RUNNING):
                        self.kv_cache.allocate_slot_for_next_token(seq)
                    active_groups.append(group)
                except Exception:
                    print(f"OOM during decoding for requst {group.request_id}")
                    raise
                
        self.running = active_groups
        
        # add new groups and prefill
        while self.waiting:
            candidate_group = self.waiting[0]
            
            # check for sequence limit (too much CPU latency)
            num_new_seqs = len(candidate_group.get_seqs())
            if len(self.running) + num_new_seqs > self.max_num_seqs:
                break
            
            # check token limit (too much GPU computation)
            current_tokens = sum(s.get_len() for g in self.running for s in g.get_seqs())
            new_tokens = sum(s.get_len() for s in candidate_group.get_seqs())
            
            if current_tokens + new_tokens > self.max_num_batched_tokens:
                break
            
            # check memory limit (too much VRAM use)
            # do we have enough free blocks for the prompt?
            blocks_needed = 0
            for seq in candidate_group.get_seqs():
                blocks_needed += (seq.get_len() + seq.block_size - 1) // seq.block_size
                
            if len(self.kv_cache.free_blocks) < blocks_needed:
                break
            
            group = self.waiting.popleft()
            # allocate blocks for the prompt (prefill)
            for seq in group.get_seqs():
                self.kv_cache.allocate_for_prefill(seq, seq.get_len())
                seq.status = SequenceStatus.RUNNING
                
            self.running.append(group)
            
        return self.running