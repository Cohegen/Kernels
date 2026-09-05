/*
An optimization aimed at minimizing latency and divergent branches.

We do this by: 
      1. loading input array into shared memory
      2. Using proper indexing strategy to minimize control divergence
*/

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 256
__global__ ConvergentSharedMem(float*arr,float*arr_out){
    //defining coordinates
    int index = threaIdx.x;

    //declaring the shared memory array
    __shared__ shared_arr[BLOCK_DIM];

    //moving array elements into shared memory
    shared_arr[index] = arr[index];

    for(stride=BLOCK_DIM/2;stride<=BLOCK_DIM;stride/=2){
        if(index <stride){
            shared_arr[index] += shared_arr[index+stride];
        }
        __synchthreads();
    }

    if(index==0){
        *output = shared_arr[index];
    }

}

int main()
{
    constexpr int N= 256; // Array size
    constexpr size_t array_bytes = N * sizeof(float);
    constexpr size_t output_bytes = sizeof(float);

    // Host arrays
    float *h_arr = new float[N];
    float h_out = 0.0f;
    
    // Initializing the host array
    for(int i=0;i<N;i++){
        h_arr[i] = static_cast<float>(i);
    }

    // Device memory
    float* d_arr = nullptr;
    float* d_out = nullptr;

    // CUDA events for timing
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

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_arr, array_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, output_bytes));

    // Start End-to-End Timer
    CHECK_CUDA_ERROR(cudaEventRecord(start_e2e, 0));

    //H2D Transfer (measuring time taken to transfer data from host to device)
    CHECK_CUDA_ERROR(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(d_arr, h_arr, array_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(end_h2d, 0));

    //Kernel Launch 
    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel, 0));
    ConvergentSharedMem<<<1, N>>>(d_arr, d_out); // Launched the correct kernel
    CHECK_CUDA_ERROR(cudaEventRecord(end_kernel, 0));
    
    // Waiting for kernel to finish (important for accurate timing of kernel and subsequent D2H)
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // D2H Transfer (time taken measuring device to host )
    CHECK_CUDA_ERROR(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_out, d_out, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(end_d2h, 0));

    // End End-to-End Timer 
    CHECK_CUDA_ERROR(cudaEventRecord(end_e2e, 0));

    // Synchronize all events and calculate elapsed times
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_h2d));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_kernel));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_d2h));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_e2e));

    float h2d_time_ms = 0.0f;
    float kernel_time_ms = 0.0f;
    float d2h_time_ms = 0.0f;
    float e2e_time_ms = 0.0f;

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d_time_ms, start_h2d, end_h2d));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_time_ms, start_kernel, end_kernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2h_time_ms, start_d2h, end_d2h));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&e2e_time_ms, start_e2e, end_e2e));

    // Calculate bandwidth
    // Bandwidth in GB/s = (bytes / 1e9) / (time_ms / 1e3)
    //                    = (bytes * 1e3) / (time_ms * 1e9)
    //                    = bytes / (time_ms * 1e6)
    float h2d_bandwidth_gbs = (array_bytes / (h2d_time_ms * 1e6)); // Convert to GB/s
    float d2h_bandwidth_gbs = (output_bytes / (d2h_time_ms * 1e6)); // Convert to GB/s

    // Printing results
    std::cout << "\n--- Performance Metrics ---\n";
    std::cout << "Host-to-Device Transfer Time: " << h2d_time_ms << " ms (" << h2d_bandwidth_gbs << " GB/s)\n";
    std::cout << "CUDA Kernel Execution Time:   " << kernel_time_ms << " ms\n";
    std::cout << "Device-to-Host Transfer Time: " << d2h_time_ms << " ms (" << d2h_bandwidth_gbs << " GB/s)\n";
    std::cout << "End-to-End CUDA Execution Time: " << e2e_time_ms << " ms\n";

    std::cout << "\n--- Validation ---\n";
    std::cout << "GPU sum: " << h_out << "\n";

    float expected = 0.0f;
    for (int i=0; i<N; ++i) // Recalculate sum from initial h_arr for correctness
    {
        expected += h_arr[i];
    }

    std::cout << "Expected: " << expected << '\n';

    if (h_out == expected)
    {
        std::cout << "Result correct!\n";
    }
    else
    {
        std::cout << "Result incorrect!\n";
    }

    // Freeing GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_arr));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start_h2d));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_h2d));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_d2h));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_d2h));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_e2e));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_e2e));

    delete[] h_arr; // Free host memory

    return 0;
}