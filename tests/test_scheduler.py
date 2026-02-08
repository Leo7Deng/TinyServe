import unittest
import torch
from tinyserve.scheduler import Scheduler, Sequence, SequenceGroup, SequenceStatus
from tinyserve.memory_manager import KVCache

class TestScheduler(unittest.TestCase):
    def setUp(self):
        """
        Setup a Scheduler and KVCache for each test.
        We use device='cpu' to make tests fast and GPU independent.
        """
        self.block_size = 4
        self.total_blocks = 10
        
        # A tiny cache: 10 blocks * 4 tokens = 40 tokens total capacity
        self.kv_cache = KVCache(
            num_blocks=self.total_blocks,
            block_size=self.block_size,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            device="cpu"
        )
        
        # A strict scheduler
        self.scheduler = Scheduler(
            self.kv_cache,
            max_num_seqs=4,           # Max 4 users
            max_num_batched_tokens=50 # Max 50 tokens processing at once
        )

    def create_dummy_request(self, req_id, prompt_len):
        """Helper to create a request with dummy tokens."""
        prompt = [1] * prompt_len
        seq = Sequence(
            seq_id=int(req_id),
            prompt="test",
            prompt_token_ids=prompt,
            block_size=self.block_size
        )
        return SequenceGroup(request_id=str(req_id), seqs=[seq])

    def test_basic_scheduling(self):
        """Test 1: Simple FIFO scheduling."""
        # Add User 1 (Len 4)
        req1 = self.create_dummy_request(1, 4)
        self.scheduler.add_request(req1)
        
        # Schedule
        batch = self.scheduler.schedule()
        
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].request_id, "1")
        self.assertEqual(batch[0].get_seqs()[0].status, SequenceStatus.RUNNING)
        # 4 tokens = 1 block used
        self.assertEqual(len(self.kv_cache.free_blocks), self.total_blocks - 1)

    def test_sequence_limit(self):
        """Test 2: Enforce max_num_seqs."""
        # Add 5 users (Limit is 4)
        for i in range(5):
            self.scheduler.add_request(self.create_dummy_request(i, 2))
            
        batch = self.scheduler.schedule()
        
        # Should only schedule 4, leave 1 waiting
        self.assertEqual(len(batch), 4)
        self.assertEqual(len(self.scheduler.waiting), 1)

    def test_token_limit(self):
        """Test 3: Enforce max_num_batched_tokens (Weight Limit)."""
        # Limit is 50 tokens.
        
        # User 1: 35 tokens
        self.scheduler.add_request(self.create_dummy_request(1, 35))
        # User 2: 20 tokens
        self.scheduler.add_request(self.create_dummy_request(2, 20))
        
        batch = self.scheduler.schedule()
        
        # Should only schedule User 1 (35 < 50). 
        # User 2 (35 + 20 = 55 > 50) must wait.
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].request_id, "1")
        self.assertEqual(len(self.scheduler.waiting), 1)

    def test_memory_limit_oom(self):
        """Test 4: Enforce Physical Memory limits (VRAM)."""
        # Cache capacity: 10 blocks.
        # User 1 needs 8 blocks (32 tokens).
        req1 = self.create_dummy_request(1, 32)
        self.scheduler.add_request(req1)
        
        self.scheduler.schedule()
        if len(self.scheduler.running) != 1:
            raise
        self.assertEqual(len(self.scheduler.running), 1)
        
        # Remaining free blocks: 2
        
        # User 2 needs 3 blocks (12 tokens).
        req2 = self.create_dummy_request(2, 12)
        self.scheduler.add_request(req2)
        
        batch = self.scheduler.schedule()
        
        # User 2 should NOT run because 3 > 2 (Free blocks)
        self.assertEqual(len(batch), 1) # Only User 1 running
        self.assertEqual(batch[0].request_id, "1")
        self.assertEqual(len(self.scheduler.waiting), 1) # User 2 still waiting

    def test_free_finished_sequences(self):
        """Test 5: Memory is freed when user finishes."""
        # Add User 1 (Uses 1 block)
        req = self.create_dummy_request(1, 4)
        self.scheduler.add_request(req)
        self.scheduler.schedule()
        
        # Verify memory used
        self.assertEqual(len(self.kv_cache.free_blocks), self.total_blocks - 1)
        
        # Mark finished
        req.get_seqs()[0].status = SequenceStatus.FINISHED
        
        # Schedule again (should cleanup)
        self.scheduler.schedule()
        
        # Verify memory freed
        self.assertEqual(len(self.kv_cache.free_blocks), self.total_blocks)
        self.assertEqual(len(self.scheduler.running), 0)

    def test_decode_allocation(self):
        """Test 6: Running sequences allocate new blocks eventually."""
        # Prompt: 4 tokens (Exact fit for block_size=4)
        # Block 1 is FULL.
        req = self.create_dummy_request(1, 4)
        self.scheduler.add_request(req)
        self.scheduler.schedule()
        
        # Current usage: 1 block.
        initial_free = len(self.kv_cache.free_blocks)
        
        # Run schedule again (Decode step). 
        # Sequence has 4 tokens. 4 % 4 == 0. 
        # allocate_slot_for_next_token should allocate NEW block for token 5.
        self.scheduler.schedule()
        
        # Should have consumed 1 more block
        self.assertEqual(len(self.kv_cache.free_blocks), initial_free - 1)

if __name__ == "__main__":
    unittest.main()