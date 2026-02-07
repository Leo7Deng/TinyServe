#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <torch/extension.h>

#define MAX_HEAD_DIM 64         // mirrors head_dim
#define THREADS_PER_BLOCK 128   // mirrors blockDim.x

__global__ void paged_attention_kernel_v4(
    float* __restrict__ out,                 // output tensor
    const float* __restrict__ q,             // query tensor (not a bottleneck in inference because only need q of most recent token)
    const float* __restrict__ k_cache,       // key cache (can be very large), shape: [num_blocks, block_size, num_heads, head_dim]
    const float* __restrict__ v_cache,       // value cache (mirrors keys)
    const int* __restrict__ block_tables,    // maps [seq, block_idx] -> physical_block
    const int* __restrict__ context_lens,    // length of each sequence
    int max_blocks_per_seq,                  // block_idx dimension of block_tables
    int block_size,                          // 16
    int head_dim,                            // 64
    int num_heads                            // 32
) {
    // Launch grid as (num_heads, num_seqs)
    // blockIdx.x handles each head
    // blockIdx.y handles each sequence (think each user/prompt)
    // Later we will use threadIdx which works together to calculate for this one sequence's head
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    int tid = threadIdx.x;

    // Declare shared memory for threads to compare max and sum for softmax
    __shared__ float max_buffer[THREADS_PER_BLOCK];
    __shared__ float sum_buffer[THREADS_PER_BLOCK];

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
    for (int i = tid; i < num_tokens; i += blockDim.x) {
        // get which physical block to look at
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        // k_cache is in the shape [num_blocks, block_size, num_heads, head_dim]
        int block_offset = i % block_size;
        long long k_offset = (long long)physical_block * block_size * num_heads * head_dim
                        + (long long)block_offset * num_heads * head_dim
                        + (long long)head_idx * head_dim;

        // compute dot product
        float dot = 0.0f;
        
        // Cast the pointers to float4* to load 128 bits (4 floats) at once
        // This means less load instructions, and the memory bus is used more efficiently
        // This is also a form of thread coarsening where each thread works on more data at a time
        const float4* k_vec = reinterpret_cast<const float4*>(&k_cache[k_offset]);
        const float4* q_vec = reinterpret_cast<const float4*>(&q[q_offset]);

        // Iterate 4 items at a time (head_dim / 4)
        for (int v = 0; v < head_dim / 4; ++v) {
            float4 kv = k_vec[v];
            float4 qv = q_vec[v];
            dot += kv.w * qv.w + kv.x * qv.x + kv.y * qv.y + kv.z * qv.z;
        }

        // used for softmax stability
        dot *= scale;
        if (dot > max_score) {
            max_score = dot;
        }
    }
    // write thread's max score to shared buffer
    max_buffer[tid] = max_score;
    __syncthreads();

    // Now that we have the max scores from all threads, use thread 0 to find the actual single max
    __shared__ float global_max_score;

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_buffer[tid] = max(max_buffer[tid], max_buffer[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) global_max_score = max_buffer[0];
    __syncthreads();

    // same loop but softmax and accumulate V
    float out_accumulator[MAX_HEAD_DIM];
    for (int d=0; d<head_dim; d++) out_accumulator[d] = 0.0f;

    for (int i = tid; i < num_tokens; i += blockDim.x) {
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        int block_offset = i % block_size;
        long long k_offset = (long long)physical_block * block_size * num_heads * head_dim
                        + (long long)block_offset * num_heads * head_dim
                        + (long long)head_idx * head_dim;
        float dot = 0.0f;
        
        const float4* k_vec = reinterpret_cast<const float4*>(&k_cache[k_offset]);
        const float4* q_vec = reinterpret_cast<const float4*>(&q[q_offset]);

        for (int v = 0; v < head_dim / 4; ++v) {
            float4 kv = k_vec[v];
            float4 qv = q_vec[v];
            dot += kv.w * qv.w + kv.x * qv.x + kv.y * qv.y + kv.z * qv.z;
        }

        dot *= scale;

        // softmax and calculate prob
        float prob = expf(dot - global_max_score);
        sum_exp += prob;

        // accumulate output from v_cache
        long long v_offset = k_offset;
        
        // Cast v_cache pointer to float4* to work on more computations per load
        const float4* v_vec = reinterpret_cast<const float4*>(&v_cache[v_offset]);
        
        for (int v = 0; v < head_dim / 4; ++v) {
            float4 vv = v_vec[v];
            // Manually unroll the addition to our local accumulator
            out_accumulator[v * 4 + 0] += prob * vv.x;
            out_accumulator[v * 4 + 1] += prob * vv.y;
            out_accumulator[v * 4 + 2] += prob * vv.z;
            out_accumulator[v * 4 + 3] += prob * vv.w;
        }
    }
    sum_buffer[tid] = sum_exp;
    __syncthreads();

    // Also sum global_sum_exp for softmax using tree reduction
    __shared__ float global_sum_exp;
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_buffer[tid] = sum_buffer[tid] + sum_buffer[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_sum_exp = sum_buffer[0] + 1e-6f;
    }
    __syncthreads();

    // Use shared memory to write to have less global memory steps writing to output
    __shared__ float shared_out[MAX_HEAD_DIM];
    if (tid < head_dim) {
        shared_out[tid] = 0.0f;
    }
    __syncthreads();

    // add local accumulator to shared accumulator per head
    for (int d = 0; d < head_dim; d++) {
        atomicAdd(&shared_out[d], out_accumulator[d]);
    }
    __syncthreads();

    // Each head has an output, so only first 64 threads will write this to out
    // q tensor and out tensor have the same dimension in this case
    // offset for: Sequence A, Head 5, Dimension 0 will be the same
    int out_offset = q_offset;
    if (tid < head_dim) {
        out[out_offset + tid] = shared_out[tid] / global_sum_exp;
    }
}

void launch_paged_attention_v4(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens
) 
{
    int num_seqs = query.size(0);
    int num_heads = query.size(1);
    int head_dim = query.size(2);
    
    int max_blocks_per_seq = block_tables.size(1);
    int block_size = key_cache.size(1);

    dim3 grid(num_heads, num_seqs);
    dim3 block(THREADS_PER_BLOCK); 

    paged_attention_kernel_v4<<<grid, block>>>(
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