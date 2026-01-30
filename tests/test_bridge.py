import tinyserve_ext
import numpy as np
import time

def matrix_mult(A, B):
    result = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result
                
    
def main():
    N = 200

    # Create arrays of size NxN
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    
    start_time = time.time()
    result = matrix_mult(A, B)
    end_time = time.time()
    
    start_time_gpu = time.time()
    result_gpu = tinyserve_ext.matrix_mult(A, B)
    end_time_gpu = time.time()
    
    print(f"CPU execution time: {end_time - start_time}, GPU execution time: {end_time_gpu - start_time_gpu}, Results are equal: {np.allclose(result, result_gpu)}")

main()