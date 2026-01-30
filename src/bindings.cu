#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory.h>

namespace py = pybind11;

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

}

py::array_t<float> matrix_mult(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    int N = buf_A.shape[0];
    int M = buf_A.shape[1];
    int P = buf_B.shape[1];

    py::array_t<float> result({N, P});
    py::buffer_info buf_result = result.request();
    
    // Get pointers to data
    float* ptr_A = static_cast<float*>(buf_A.ptr);
    float* ptr_B = static_cast<float*>(buf_B.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = N * M * sizeof(float);
    size_t size_B = M * P * sizeof(float);
    size_t size_C = N * P * sizeof(float);
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, ptr_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, ptr_B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(ptr_result, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}


PYBIND11_MODULE(tinyserve_ext, m) {
    m.doc() = "TinyServe Low-Level Bindings";

    // Bind function
    m.def("matrix_mult", &matrix_mult, "A function that multiplies matricies");

    // Bind the class
    py::class_<BlockAllocator>(m, "BlockAllocator")
        .def(py::init<int, int>(), py::arg("total_blocks"), py::arg("block_size"))
        .def("allocate", &BlockAllocator::allocate)
        .def("free", &BlockAllocator::free)
        .def("get_free_block_count", &BlockAllocator::get_free_block_count)
        .def("get_block_size", &BlockAllocator::get_block_size);
}