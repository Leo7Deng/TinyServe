#pragma once
#include <cstdint>

void launch_paged_attention_v1(
    std::uintptr_t out_ptr,
    std::uintptr_t q_ptr,
    std::uintptr_t k_ptr,
    std::uintptr_t v_ptr,
    std::uintptr_t table_ptr,
    std::uintptr_t lens_ptr,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_num_blocks,
    int block_size
);

void launch_paged_attention_v2(
    std::uintptr_t out_ptr,
    std::uintptr_t q_ptr,
    std::uintptr_t k_ptr,
    std::uintptr_t v_ptr,
    std::uintptr_t table_ptr,
    std::uintptr_t lens_ptr,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_num_blocks,
    int block_size
);