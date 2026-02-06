#include <torch/extension.h>

// This function is called to store the new K and V received for each sequence from inference. We need to store these in our non contiguous memory cache.
__global__ void reshape_and_cache(
    const float* __restrict__ key,          // incoming new key
    const float* __restrict__ value,
    float* __restrict__ key_cache,          // the big KV cache storage
    float* __restrict__ value_cache,
    const long* __restrict__ slot_mapping,   // the mapping from sequence -> physical block
    const int key_stride,                   // stride to get to the next user's data
    const int val_stride, 
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    // Each CUDA block will handle one sequence
    const int seq_idx = blockIdx.x;

    const long flat_slot_idx = slot_mapping[seq_idx];
    const int block_idx = flat_slot_idx / block_size;
    const int block_offset = flat_slot_idx % block_size;

    // Create pointers to the memory in the cache
    // The cache shape is [num_blocks, block_size, num_heads, head_dim]
    const int floats_per_block = block_size * num_heads * head_dim;
    float* k_block_ptr = key_cache + (long)block_idx * floats_per_block;
    float* v_block_ptr = value_cache + (long)block_idx * floats_per_block;

    const int floats_per_slot = num_heads * head_dim;
    float* k_dest_ptr = k_block_ptr + (long)block_offset * floats_per_slot;
    float* v_dest_ptr = v_block_ptr + (long)block_offset * floats_per_slot;

    // each thread will be on a different stride to copy float into destination
    const float* k_src_ptr = key + seq_idx * key_stride;
    const float* v_src_ptr = value + seq_idx * val_stride;
    for (int i = threadIdx.x; i < floats_per_slot; i+= blockDim.x) {
        k_dest_ptr[i] = k_src_ptr[i];
        v_dest_ptr[i] = v_src_ptr[i];
    }

}


// This is what Python calls, tt unpacks tensors and launches the kernel.
void launch_reshape_and_cache(
    torch::Tensor& key,        
    torch::Tensor& value,        
    torch::Tensor& key_cache,  
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping)
{
    // 1. Extract Dimensions
    int num_seqs = key.size(0);
    int num_heads = key.size(1);
    int head_dim = key.size(2);
    int block_size = key_cache.size(1);

    int key_stride = key.stride(0);
    int val_stride = value.stride(0);

    // 2. Launch Configuration
    // Grid: One block per sequence (user)
    dim3 grid(num_seqs);
    // Block: 256 threads to copy the data in parallel
    dim3 block(256);

    // 3. Launch Kernel
    reshape_and_cache<<<grid, block>>>(
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        slot_mapping.data_ptr<long>(),
        key_stride,
        val_stride,
        num_heads,
        head_dim,
        block_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}