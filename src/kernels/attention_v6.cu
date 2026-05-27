#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <torch/extension.h>

#define MAX_HEAD_DIM 64         // mirrors head_dim
#define THREADS_PER_BLOCK 128   // mirrors blockDim.x

__global__ void paged_attention_kernel_v6(
    float* __restrict__ out,                 // output tensor
    const float* __restrict__ q,             // query tensor (not a bottleneck in inference because only need q of most recent token)
    const float* __restrict__ k_cache,       // key cache (can be very large), shape: [num_blocks, block_size, num_heads, head_dim]
    const float* __restrict__ v_cache,       // value cache (mirrors keys)
    const int* __restrict__ block_tables,    // maps [seq, block_idx] -> physical_block
    const int* __restrict__ context_lens,    // length of each sequence
    int max_blocks_per_seq,                  // block_idx dimension of block_tables
    int block_size,                          // 16
    int head_dim,                            // 64
    int num_q_heads,                         // 32
    int num_kv_heads                         // 4
) {
    // Launch grid as (num_heads, num_seqs)
    // blockIdx.x handles each head
    // blockIdx.y handles each sequence (think each user/prompt)
    // Later we will use threadIdx which works together to calculate for this one sequence's head
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Q tensor shape is [num_seqs, num_heads, head_dim]
    // This offset is the start for this specific attention head
    int q_offset = (seq_idx * num_q_heads * head_dim) + (q_head_idx * head_dim);
    int num_tokens = context_lens[seq_idx];

    // standard attention scaling
    float scale = 1.0f / sqrtf((float)head_dim);

    // online softmax accumulators for each thread
    // local_m is the running max, local_l is the running sum of exp(score - local_m)
    float local_m = -FLT_MAX;
    float local_l = 0.0f;
    float local_acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        local_acc[d] = 0.0f;
    }

    const float4* q_vec = reinterpret_cast<const float4*>(&q[q_offset]);

    // calculate scores Q*K and accumulate V in one loop using online softmax
    for (int i = tid; i < num_tokens; i += blockDim.x) {
        // get which physical block to look at
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        int block_offset = i % block_size;

        // k_cache is in the shape [num_blocks, block_size, num_heads, head_dim]
        long long cache_offset = (long long)physical_block * block_size * num_kv_heads * head_dim
                               + (long long)block_offset * num_kv_heads * head_dim
                               + (long long)kv_head_idx * head_dim;

        // compute dot product
        const float4* k_vec = reinterpret_cast<const float4*>(&k_cache[cache_offset]);
        float dot = 0.0f;

        // Cast the pointers to float4* to load 128 bits (4 floats) at once
        // This means less load instructions, and the memory bus is used more efficiently
        // This is also a form of thread coarsening where each thread works on more data at a time
        for (int v = 0; v < head_dim / 4; ++v) {
            float4 kv = k_vec[v];
            float4 qv = q_vec[v];
            dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
        dot *= scale;

        // online softmax update
        // if the new score is the max, rescale the previous accumulator to the new max
        float new_m = fmaxf(local_m, dot);
        float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - new_m);
        float new_scale = expf(dot - new_m);

        // accumulate output from v_cache
        const float4* v_vec = reinterpret_cast<const float4*>(&v_cache[cache_offset]);
        for (int v = 0; v < head_dim / 4; ++v) {
            float4 vv = v_vec[v];
            local_acc[v * 4 + 0] = local_acc[v * 4 + 0] * old_scale + vv.x * new_scale;
            local_acc[v * 4 + 1] = local_acc[v * 4 + 1] * old_scale + vv.y * new_scale;
            local_acc[v * 4 + 2] = local_acc[v * 4 + 2] * old_scale + vv.z * new_scale;
            local_acc[v * 4 + 3] = local_acc[v * 4 + 3] * old_scale + vv.w * new_scale;
        }

        local_l = local_l * old_scale + new_scale;
        local_m = new_m;
    }

    // Declare shared memory for threads to merge online softmax states
    __shared__ float shared_m[THREADS_PER_BLOCK];
    __shared__ float shared_l[THREADS_PER_BLOCK];
    __shared__ float shared_acc[THREADS_PER_BLOCK][MAX_HEAD_DIM];

    // write thread's local softmax state to shared memory
    shared_m[tid] = local_m;
    shared_l[tid] = local_l;
    for (int d = 0; d < head_dim; ++d) {
        shared_acc[tid][d] = local_acc[d];
    }
    __syncthreads();

    // use parallel tree reduction to merge the per-thread online softmax states
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float left_l = shared_l[tid];
            float right_l = shared_l[tid + stride];

            // if the right side saw no tokens, there is nothing to merge
            if (right_l != 0.0f) {
                // if the left side saw no tokens, copy the right state directly
                if (left_l == 0.0f) {
                    shared_m[tid] = shared_m[tid + stride];
                    shared_l[tid] = right_l;
                    for (int d = 0; d < head_dim; ++d) {
                        shared_acc[tid][d] = shared_acc[tid + stride][d];
                    }
                } else {
                    // both sides have tokens, so rescale them to the merged max
                    float left_m = shared_m[tid];
                    float right_m = shared_m[tid + stride];
                    float merged_m = fmaxf(left_m, right_m);
                    float left_scale = expf(left_m - merged_m);
                    float right_scale = expf(right_m - merged_m);

                    shared_l[tid] = left_l * left_scale + right_l * right_scale;
                    shared_m[tid] = merged_m;
                    for (int d = 0; d < head_dim; ++d) {
                        shared_acc[tid][d] =
                            shared_acc[tid][d] * left_scale +
                            shared_acc[tid + stride][d] * right_scale;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Each head has an output, so only first 64 threads will write this to out
    // q tensor and out tensor have the same dimension in this case
    // offset for: Sequence A, Head 5, Dimension 0 will be the same
    int out_offset = q_offset;
    float denom = shared_l[0] + 1e-6f;
    if (tid < head_dim) {
        out[out_offset + tid] = shared_acc[0][tid] / denom;
    }
}

void launch_paged_attention_v6(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens
)
{
    int num_seqs = query.size(0);
    int num_q_heads = query.size(1);
    int head_dim = query.size(2);

    int max_blocks_per_seq = block_tables.size(1);
    int block_size = key_cache.size(1);
    int num_kv_heads = key_cache.size(2);

    dim3 grid(num_q_heads, num_seqs);
    dim3 block(THREADS_PER_BLOCK);

    paged_attention_kernel_v6<<<grid, block>>>(
        out.data_ptr<float>(),
        query.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        max_blocks_per_seq,
        block_size,
        head_dim,
        num_q_heads,
        num_kv_heads
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
