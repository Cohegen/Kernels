#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_DIM 256

__global__ void blelloch_scan(const float* input,float* output,float* partialSums,unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    __shared__ float buffer_shared[2 * BLOCK_DIM];
    unsigned int idx1 = segment + tid;
    unsigned int idx2 = segment + tid + BLOCK_DIM;

    buffer_shared[tid] = (idx1 < N) ? input[idx1] : 0.0f;
    buffer_shared[tid + BLOCK_DIM] = (idx2 < N) ? input[idx2] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1;stride <= BLOCK_DIM;stride *= 2)
    { 
        unsigned int i = (tid + 1) * 2 * stride - 1;
        if (i < 2 * BLOCK_DIM)
        {
            buffer_shared[i] += buffer_shared[i - stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialSums[blockIdx.x] = buffer_shared[2 * BLOCK_DIM - 1];
        buffer_shared[2 * BLOCK_DIM - 1] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM;stride >= 1;stride /= 2)
    { 
        unsigned int i = (tid + 1) * 2 * stride - 1;
        if (i < 2 * BLOCK_DIM)
        {
            float temp = buffer_shared[i - stride];
            buffer_shared[i - stride] = buffer_shared[i];
            buffer_shared[i] += temp;
        }
        __syncthreads();
    }

    if (idx1 < N)
    {
        output[idx1] = buffer_shared[tid];
    }
    if (idx2 < N)
    {
        output[idx2] = buffer_shared[tid + BLOCK_DIM];
    }
}

int main() {
    const int N = 512;
    float h_X[N], h_Y[N], h_sums[1];
    for (int i = 0; i < N; ++i) {
        h_X[i] = 1.0f;
    }

    float *d_X, *d_Y, *d_sums;
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));
    cudaMalloc(&d_sums, 1 * sizeof(float));

    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    belloch_scan<<<1, BLOCK_DIM>>>(d_X, d_Y, d_sums, N);

    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sums, d_sums, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Belloch (Blelloch) First 5 elements of exclusive scan:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }
    printf("Total Block Sum = %f\n", h_sums[0]);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_sums);
    return 0;
}