#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error at "
                  << file << ":" << line
                  << " " << func
                  << " -> " << cudaGetErrorString(err)
                  << std::endl;

        exit(EXIT_FAILURE);
    }
}

template <unsigned int block_size>
__device__ void warpReduce(volatile float* shared_arr, unsigned int tid)
{
    if (block_size >= 64)
        shared_arr[tid] += shared_arr[tid + 32];

    if (block_size >= 32)
        shared_arr[tid] += shared_arr[tid + 16];

    if (block_size >= 16)
        shared_arr[tid] += shared_arr[tid + 8];

    if (block_size >= 8)
        shared_arr[tid] += shared_arr[tid + 4];

    if (block_size >= 4)
        shared_arr[tid] += shared_arr[tid + 2];

    if (block_size >= 2)
        shared_arr[tid] += shared_arr[tid + 1];
}

template <unsigned int block_size>
__global__ void unroll_complete(
    const float* arr,
    float* arr_out,
    int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int global_index = blockIdx.x * block_size + tid;

    __shared__ float shared_arr[block_size];

    if (global_index < N)
        shared_arr[tid] = arr[global_index];
    else
        shared_arr[tid] = 0.0f;

    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
            shared_arr[tid] += shared_arr[tid + 256];

        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
            shared_arr[tid] += shared_arr[tid + 128];

        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
            shared_arr[tid] += shared_arr[tid + 64];

        __syncthreads();
    }

    if (tid < 32)
        warpReduce<block_size>(shared_arr, tid);

    if (tid == 0)
        atomicAdd(arr_out, shared_arr[0]);
}

int main()
{
    constexpr int N = 256;
    constexpr unsigned int BLOCK_SIZE = 256;

    constexpr size_t array_bytes =
        N * sizeof(float);

    constexpr size_t output_bytes =
        sizeof(float);

    float* h_arr = new float[N];
    float h_out = 0.0f;

    for (int i = 0; i < N; ++i)
        h_arr[i] = static_cast<float>(i);

    float* d_arr = nullptr;
    float* d_out = nullptr;

    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_kernel, end_kernel;
    cudaEvent_t start_d2h, end_d2h;
    cudaEvent_t start_e2e, end_e2e;

    CHECK_CUDA_ERROR(cudaEventCreate(&start_h2d));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_h2d));

    CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_kernel));

    CHECK_CUDA_ERROR(cudaEventCreate(&start_d2h));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_d2h));

    CHECK_CUDA_ERROR(cudaEventCreate(&start_e2e));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_e2e));

    CHECK_CUDA_ERROR(cudaMalloc(&d_arr, array_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, output_bytes));

    CHECK_CUDA_ERROR(
        cudaMemset(d_out, 0, output_bytes)
    );

    int num_blocks =
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_CUDA_ERROR(
        cudaEventRecord(start_e2e)
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(start_h2d)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(
            d_arr,
            h_arr,
            array_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(end_h2d)
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(start_kernel)
    );

    unroll_complete<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE>>>(
            d_arr,
            d_out,
            N
        );

    CHECK_CUDA_ERROR(
        cudaEventRecord(end_kernel)
    );

    CHECK_CUDA_ERROR(
        cudaDeviceSynchronize()
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(start_d2h)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(
            &h_out,
            d_out,
            output_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(end_d2h)
    );

    CHECK_CUDA_ERROR(
        cudaEventRecord(end_e2e)
    );

    CHECK_CUDA_ERROR(
        cudaEventSynchronize(end_h2d)
    );

    CHECK_CUDA_ERROR(
        cudaEventSynchronize(end_kernel)
    );

    CHECK_CUDA_ERROR(
        cudaEventSynchronize(end_d2h)
    );

    CHECK_CUDA_ERROR(
        cudaEventSynchronize(end_e2e)
    );

    float h2d_time_ms = 0.0f;
    float kernel_time_ms = 0.0f;
    float d2h_time_ms = 0.0f;
    float e2e_time_ms = 0.0f;

    CHECK_CUDA_ERROR(
        cudaEventElapsedTime(
            &h2d_time_ms,
            start_h2d,
            end_h2d
        )
    );

    CHECK_CUDA_ERROR(
        cudaEventElapsedTime(
            &kernel_time_ms,
            start_kernel,
            end_kernel
        )
    );

    CHECK_CUDA_ERROR(
        cudaEventElapsedTime(
            &d2h_time_ms,
            start_d2h,
            end_d2h
        )
    );

    CHECK_CUDA_ERROR(
        cudaEventElapsedTime(
            &e2e_time_ms,
            start_e2e,
            end_e2e
        )
    );

    float h2d_bandwidth_gbs =
        array_bytes / (h2d_time_ms * 1e6);

    float d2h_bandwidth_gbs =
        output_bytes / (d2h_time_ms * 1e6);

    std::cout << "\n--- Performance Metrics ---\n";

    std::cout << "Host-to-Device Transfer Time: "
              << h2d_time_ms
              << " ms ("
              << h2d_bandwidth_gbs
              << " GB/s)\n";

    std::cout << "CUDA Kernel Execution Time:   "
              << kernel_time_ms
              << " ms\n";

    std::cout << "Device-to-Host Transfer Time: "
              << d2h_time_ms
              << " ms ("
              << d2h_bandwidth_gbs
              << " GB/s)\n";

    std::cout << "End-to-End CUDA Execution Time: "
              << e2e_time_ms
              << " ms\n";

    std::cout << "\n--- Validation ---\n";

    std::cout << "GPU sum: "
              << h_out
              << "\n";

    float expected = 0.0f;

    for (int i = 0; i < N; ++i)
        expected += h_arr[i];

    std::cout << "Expected: "
              << expected
              << '\n';

    if (std::abs(h_out - expected) < 1e-3f)
        std::cout << "Result correct!\n";
    else
        std::cout << "Result incorrect!\n";

    CHECK_CUDA_ERROR(cudaFree(d_arr));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    CHECK_CUDA_ERROR(
        cudaEventDestroy(start_h2d)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(end_h2d)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(start_kernel)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(end_kernel)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(start_d2h)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(end_d2h)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(start_e2e)
    );

    CHECK_CUDA_ERROR(
        cudaEventDestroy(end_e2e)
    );

    delete[] h_arr;

    return 0;
}