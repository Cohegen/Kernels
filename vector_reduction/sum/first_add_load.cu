#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error at: "
                  << file << ":" << line
                  << " " << func
                  << " -> " << cudaGetErrorString(err)
                  << std::endl;

        exit(EXIT_FAILURE);
    }
}

#define BLOCK_DIM 256

__global__ void first_add_load(float* arr, float* arr_out, int N)
{
    int tid = threadIdx.x;
    int global_index = blockIdx.x * BLOCK_DIM + tid;

    __shared__ float shared_arr[BLOCK_DIM];

    // Each thread combines two global elements before storing into shared memory.
    if (global_index * 2 + 1 < N)
    {
        shared_arr[tid] =
            arr[global_index * 2] +
            arr[global_index * 2 + 1];
    }
    else if (global_index * 2 < N)
    {
        shared_arr[tid] = arr[global_index * 2];
    }
    else
    {
        shared_arr[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            shared_arr[tid] += shared_arr[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(arr_out, shared_arr[0]);
    }
}

int main()
{
    constexpr int N_original = 256;
    constexpr int N_kernel_elements = (N_original + 1) / 2;

    int num_blocks =
        (N_kernel_elements + BLOCK_DIM - 1) / BLOCK_DIM;

    constexpr size_t array_bytes =
        N_original * sizeof(float);

    constexpr size_t output_bytes =
        sizeof(float);

    float* h_arr = new float[N_original];
    float h_out = 0.0f;

    for (int i = 0; i < N_original; i++)
    {
        h_arr[i] = static_cast<float>(i);
    }

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
    CHECK_CUDA_ERROR(cudaMemset(d_out, 0, output_bytes));

    CHECK_CUDA_ERROR(cudaEventRecord(start_e2e));

    CHECK_CUDA_ERROR(cudaEventRecord(start_h2d));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_arr, h_arr, array_bytes, cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(cudaEventRecord(end_h2d));

    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));

    first_add_load<<<num_blocks, BLOCK_DIM>>>(
        d_arr,
        d_out,
        N_original
    );

    CHECK_CUDA_ERROR(cudaEventRecord(end_kernel));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaEventRecord(start_d2h));
    CHECK_CUDA_ERROR(
        cudaMemcpy(&h_out, d_out, output_bytes, cudaMemcpyDeviceToHost)
    );
    CHECK_CUDA_ERROR(cudaEventRecord(end_d2h));

    CHECK_CUDA_ERROR(cudaEventRecord(end_e2e));

    CHECK_CUDA_ERROR(cudaEventSynchronize(end_h2d));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_kernel));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_d2h));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_e2e));

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
              << h2d_time_ms << " ms ("
              << h2d_bandwidth_gbs << " GB/s)\n";

    std::cout << "CUDA Kernel Execution Time:   "
              << kernel_time_ms << " ms\n";

    std::cout << "Device-to-Host Transfer Time: "
              << d2h_time_ms << " ms ("
              << d2h_bandwidth_gbs << " GB/s)\n";

    std::cout << "End-to-End CUDA Execution Time: "
              << e2e_time_ms << " ms\n";

    std::cout << "\n--- Validation ---\n";

    std::cout << "GPU sum: " << h_out << "\n";

    float expected = 0.0f;

    for (int i = 0; i < N_original; ++i)
    {
        expected += h_arr[i];
    }

    std::cout << "Expected: " << expected << '\n';

    if (std::abs(h_out - expected) < 1e-3)
    {
        std::cout << "Result correct!\n";
    }
    else
    {
        std::cout << "Result incorrect!\n";
    }

    CHECK_CUDA_ERROR(cudaFree(d_arr));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    CHECK_CUDA_ERROR(cudaEventDestroy(start_h2d));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_h2d));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_d2h));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_d2h));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_e2e));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_e2e));

    delete[] h_arr;

    return 0;
}