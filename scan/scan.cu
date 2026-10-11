#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Inline Timer class definition
struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

#define BLOCK_DIM 1024

__global__ void scan_kernel(const float* input,float* output,float* partialsums,unsigned int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    // Double-buffered shared memory
    __shared__ float buffer_shared1[BLOCK_DIM];
    __shared__ float buffer_shared2[BLOCK_DIM];

    float* inBuffer_shared = buffer_shared1;
    float* outBuffer_shared = buffer_shared2;

    // Load input into shared memory
    inBuffer_shared[tx] = (i < N) ? input[i] : 0.0f;

    __syncthreads();

    // Kogge-Stone inclusive scan
    for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2)
    {
        if (tx >= stride)
        {
            outBuffer_shared[tx] = inBuffer_shared[tx] + inBuffer_shared[tx - stride];
        }
        else
        {
            outBuffer_shared[tx] = inBuffer_shared[tx];
        }

        __syncthreads();

        // Swap input and output buffers
        float* temp = inBuffer_shared;
        inBuffer_shared = outBuffer_shared;
        outBuffer_shared = temp;
    }

    // Store the block's total sum
    if (tx == BLOCK_DIM - 1)
    {
        partialsums[blockIdx.x] = inBuffer_shared[tx];
    }

    // Store the local scan result
    if (i < N)
    {
        output[i] = inBuffer_shared[tx];
    }
}

int main() {
    const int N = 1024;
    float h_in[N], h_out[N], h_sums[1];
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out, *d_sums;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_sums, 1 * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    Timer timer;
    timer.start();
    scan_kernel<<<1, BLOCK_DIM>>>(d_in, d_out, d_sums, N);
    cudaDeviceSynchronize();
    double duration = timer.stop();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sums, d_sums, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Double-Buffered Scan execution time: %f ms\n", duration);
    printf("Buffered scan total sum: %f\n", h_sums[0]);
    printf("First 5 elements of buffered prefix sum:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Y[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sums);
    return 0;
}
