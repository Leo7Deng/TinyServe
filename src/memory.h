#include <vector>
#include <stack>
#include <iostream>

struct Block {
    int id;
    int size;
};

class BlockAllocator {
    private:
        int total_blocks_;
        int block_size_;
        std::stack<int> free_list_;

    public:
        BlockAllocator(int total_blocks, int block_size) : total_blocks_(total_blocks), block_size_(block_size) {
            for (int i = total_blocks - 1; i >= 0; i--) {
                free_list_.push(i);
            }
        }

        std::vector<int> allocate(int num_blocks) {
            if (num_blocks > free_list_.size()) {
                throw std::runtime_error("OOM: GPU Out of Memory");
            }

            std::vector<int> allocated_ids;
            for (int i = 0; i < num_blocks; i++) {
                int free_id = free_list_.top();
                free_list_.pop();
                allocated_ids.push_back(free_id);
            }
            return allocated_ids;
        }

        void free(const std::vector<int>& block_ids) {
            for (int id : block_ids) {
                if (id < 0 || id >= total_blocks_) {
                    std::cerr << "Warning: Attempted to free invalid block ID " << id << std::endl;
                    continue;
                }
                free_list_.push(id);
            }
        }

        int get_free_block_count() const { return free_list_.size(); }
        int get_block_size() const { return block_size_; }
};