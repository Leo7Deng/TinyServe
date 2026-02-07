#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <torch/extension.h>

#define MAX_HEAD_DIM 64     // mirrors head_dim

__global__ void paged_attention_kernel_v1(
    // __restrict__
    float* out,                 // output tensor
    const float* q,             // query tensor (not a bottleneck in inference because only need q of most recent token)
    const float* k_cache,       // key cache (can be very large), shape: [num_blocks, block_size, num_heads, head_dim]
    const float* v_cache,       // value cache (mirrors keys)
    const int* block_tables,    // maps [seq, block_idx] -> physical_block
    const int* context_lens,    // length of each sequence
    int max_blocks_per_seq,         // block_idx dimension of block_tables
    int block_size,             // 16
    int head_dim,               // 64
    int num_heads               // 32
) {
    // Launch grid as (num_heads, num_seqs)
    // blockIdx.x handles each head
    // blockIdx.y handles each sequence (think each user/prompt)
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    // Let's get one thread block to work first
    if (threadIdx.x > 0 || threadIdx.y > 0) return;

    // Q tensor shape is [num_seqs, num_heads, head_dim]
    // This offset is the start for this specific attention head
    int q_offset = (seq_idx * num_heads * head_dim) + (head_idx * head_dim);

    // accumulators for softmax
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;

    // standard attention scaling
    float scale = 1.0f / sqrtf((float)head_dim);
    
    int num_tokens = context_lens[seq_idx];

    // Calculate scores Q*K to see which keys have high correlation to the query
    // first loop to find max_score
    for (int i = 0; i < num_tokens; i++) {
        // get which physical block to look at
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        // k_cache is in the shape [num_blocks, block_size, num_heads, head_dim]
        int block_offset = i % block_size;
        int k_offset = (physical_block * block_size * num_heads * head_dim) 
                        + (block_offset * num_heads * head_dim)
                        + (head_idx * head_dim);

        // compute dot product
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q[q_offset + d] * k_cache[k_offset + d];
        }

        // used for softmax stability
        dot *= scale;
        if (dot > max_score) {
            max_score = dot;
        }
    }

    // same loop but softmax and accumulate V
    float out_accumulator[MAX_HEAD_DIM];
    for (int d=0; d<head_dim; d++) out_accumulator[d] = 0.0f;

    for (int i = 0; i < num_tokens; i++) {
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        int block_offset = i % block_size;
        int k_offset = (physical_block * block_size * num_heads * head_dim) 
                        + (block_offset * num_heads * head_dim)
                        + (head_idx * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q[q_offset + d] * k_cache[k_offset + d];
        }
        dot *= scale;

        // softmax and calculate prob
        float prob = expf(dot - max_score);
        sum_exp += prob;

        // accumulate output from v_cache
        int v_offset = k_offset;
        for (int d = 0; d < head_dim; d++) {
            out_accumulator[d] += prob * v_cache[v_offset + d];
        }
    }

    // q tensor and out tensor have the same dimension in this case
    // offset for: Sequence A, Head 5, Dimension 0 will be the same
    int out_offset = q_offset;
    for (int d = 0; d < head_dim; d++) {
        out[out_offset + d] = out_accumulator[d] / sum_exp;
    }
}


void launch_paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens
) {
    int num_seqs = query.size(0);
    int num_heads = query.size(1);
    int head_dim = query.size(2);
    
    // In Python: block_tables is [num_seqs, max_blocks_per_seq]
    int max_blocks_per_seq = block_tables.size(1);
    
    // In Python: key_cache is [num_blocks, block_size, num_heads, head_dim]
    int block_size = key_cache.size(1);

    dim3 grid(num_heads, num_seqs);
    dim3 block(1); // Single thread per head

    paged_attention_kernel_v1<<<grid, block>>>(
        out.data_ptr<float>(),
        query.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        max_blocks_per_seq,
        block_size,
        head_dim,
        num_heads
    );
}