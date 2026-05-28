#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <torch/extension.h>

#define HEAD_DIM_V7 64
#define HEAD_DIM_V4 16
#define THREADS_PER_BLOCK_V7 128

__global__ void paged_attention_kernel_v7(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int max_blocks_per_seq,
    int block_size,
    int num_q_heads,
    int num_kv_heads
) {
    // Launch grid as (num_heads, num_seqs)
    // blockIdx.x handles each query head
    // blockIdx.y handles each sequence (think each user/prompt)
    int q_head_idx = blockIdx.x;

    // GQA maps multiple query heads onto one KV head.
    // Example: 32 query heads and 4 KV heads means every 8 query heads share one KV head.
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Q tensor shape is [num_seqs, num_heads, head_dim]
    // This offset is the start for this specific sequence and query head.
    int q_offset = (seq_idx * num_q_heads + q_head_idx) * HEAD_DIM_V7;
    int num_tokens = context_lens[seq_idx];

    // standard attention scaling
    float scale = 1.0f / sqrtf((float)HEAD_DIM_V7);

    // online softmax accumulators for each thread
    // local_m is the running max, local_l is the running sum of exp(score - local_m)
    float local_m = -FLT_MAX;
    float local_l = 0.0f;

    // Keep the 64 output dimensions as 16 float4 values.
    // This matches the vectorized K/V loads and is more compiler friendly than a scalar float[64].
    float4 local_acc[HEAD_DIM_V4];

    // HEAD_DIM_V4 is a compile-time constant, so unrolling removes loop overhead
    // and gives the compiler more visibility into the fixed 16 float4 operations.
#pragma unroll
    for (int v = 0; v < HEAD_DIM_V4; ++v) {
        local_acc[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float4* q_vec = reinterpret_cast<const float4*>(&q[q_offset]);

    // calculate scores Q*K and accumulate V in one loop using online softmax
    for (int i = tid; i < num_tokens; i += blockDim.x) {
        // get which physical block to look at
        int block_idx = i / block_size;
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        int block_offset = i - block_idx * block_size;

        // k_cache/v_cache are in the shape [num_blocks, block_size, num_kv_heads, head_dim]
        long long cache_offset = (long long)physical_block * block_size * num_kv_heads * HEAD_DIM_V7
                               + (long long)block_offset * num_kv_heads * HEAD_DIM_V7
                               + (long long)kv_head_idx * HEAD_DIM_V7;

        // compute dot product
        const float4* k_vec = reinterpret_cast<const float4*>(&k_cache[cache_offset]);
        float dot = 0.0f;

        // Cast the pointers to float4* to load 128 bits (4 floats) at once
        // This means less load instructions, and the memory bus is used more efficiently
        // This is also a form of thread coarsening where each thread works on more data at a time
#pragma unroll
        for (int v = 0; v < HEAD_DIM_V4; ++v) {
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
#pragma unroll
        for (int v = 0; v < HEAD_DIM_V4; ++v) {
            float4 vv = v_vec[v];
            float4 acc = local_acc[v];
            local_acc[v] = make_float4(
                acc.x * old_scale + vv.x * new_scale,
                acc.y * old_scale + vv.y * new_scale,
                acc.z * old_scale + vv.z * new_scale,
                acc.w * old_scale + vv.w * new_scale
            );
        }

        local_l = local_l * old_scale + new_scale;
        local_m = new_m;
    }

    __shared__ float shared_m[THREADS_PER_BLOCK_V7];
    __shared__ float shared_l[THREADS_PER_BLOCK_V7];

    // Store shared_acc as [dim][thread] instead of [thread][dim].
    // The +1 padding breaks power-of-two row strides that cause shared memory bank conflicts.
    __shared__ float shared_acc[HEAD_DIM_V7][THREADS_PER_BLOCK_V7 + 1];

    // write thread's local softmax state to shared memory
    shared_m[tid] = local_m;
    shared_l[tid] = local_l;
#pragma unroll
    for (int v = 0; v < HEAD_DIM_V4; ++v) {
        float4 acc = local_acc[v];
        shared_acc[v * 4 + 0][tid] = acc.x;
        shared_acc[v * 4 + 1][tid] = acc.y;
        shared_acc[v * 4 + 2][tid] = acc.z;
        shared_acc[v * 4 + 3][tid] = acc.w;
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
#pragma unroll
                    for (int d = 0; d < HEAD_DIM_V7; ++d) {
                        shared_acc[d][tid] = shared_acc[d][tid + stride];
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
#pragma unroll
                    for (int d = 0; d < HEAD_DIM_V7; ++d) {
                        shared_acc[d][tid] =
                            shared_acc[d][tid] * left_scale +
                            shared_acc[d][tid + stride] * right_scale;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Each head has an output, so only first 64 threads will write this to out.
    // q tensor and out tensor have the same dimension in this case.
    float denom = shared_l[0] + 1e-6f;
    if (tid < HEAD_DIM_V7) {
        out[q_offset + tid] = shared_acc[tid][0] / denom;
    }
}

void launch_paged_attention_v7(
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

    TORCH_CHECK(head_dim == HEAD_DIM_V7, "paged_attention_v7 currently requires head_dim == 64");
    TORCH_CHECK(block_size > 0, "paged_attention_v7 requires block_size > 0");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "paged_attention_v7 requires num_q_heads divisible by num_kv_heads");

    dim3 grid(num_q_heads, num_seqs);
    dim3 block(THREADS_PER_BLOCK_V7);

    paged_attention_kernel_v7<<<grid, block>>>(
        out.data_ptr<float>(),
        query.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        max_blocks_per_seq,
        block_size,
        num_q_heads,
        num_kv_heads
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
