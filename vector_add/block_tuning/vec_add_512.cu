/*
A vector addition kernel with 512 threads per block
*/
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vector_add(const float* A,const float* B,float* C, int N){
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index<N){
        C[index] = A[index] + B[index];
    }
}

int main(){
    const int N = 100'000'000; 
    
    size_t bytes = N*sizeof(float);

    //host vectors
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    //intializing the vectors
    for(int i=0;i<N;i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2*i);
    }

    //device vectors
    float* d_A;
    float* d_B;
    float* d_C;

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event, memcpy_h2d_start, memcpy_h2d_stop, kernel_start, kernel_stop, memcpy_d2h_start, memcpy_d2h_stop;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&memcpy_h2d_start);
    cudaEventCreate(&memcpy_h2d_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&memcpy_d2h_start);
    cudaEventCreate(&memcpy_d2h_stop);

    // Record start of end-to-end time
    cudaEventRecord(start_event);

    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    //copying host vectors into device and measure time
    cudaEventRecord(memcpy_h2d_start);
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice); 
    cudaEventRecord(memcpy_h2d_stop);

    //kernel configuration
    int blockSize = 512;
    int gridSize = (N+blockSize -1) / blockSize;

    // Vector addition kernel launch and measure time
    cudaEventRecord(kernel_start);
    vector_add<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);
    cudaEventRecord(kernel_stop);
    cudaDeviceSynchronize(); 

    //copying results to host and measure time
    cudaEventRecord(memcpy_d2h_start);
    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(memcpy_d2h_stop);

    // Record end of end-to-end time
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float h2d_time, kernel_time, d2h_time, total_cuda_time;
    cudaEventElapsedTime(&h2d_time, memcpy_h2d_start, memcpy_h2d_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    cudaEventElapsedTime(&d2h_time, memcpy_d2h_start, memcpy_d2h_stop);
    cudaEventElapsedTime(&total_cuda_time, start_event, stop_event);

    // Calculate effective memory bandwidth
    // Each element is a float (4 bytes). Total elements = N.
    // Two vectors A and B are read, one vector C is written.
    // So, total data moved = N * sizeof(float) * 3
    double total_data_transfer_gb = (double)bytes * 3 / (1024 * 1024 * 1024); // in GB
    double total_memcpy_time_sec = (h2d_time + d2h_time) / 1000.0; // in seconds
    double effective_bandwidth_gbps = (total_memcpy_time_sec > 0) ? (total_data_transfer_gb / total_memcpy_time_sec) : 0.0;


    std::cout << "\n--- CUDA Benchmarking Results ---\n";
    std::cout << "CUDA H2D Memory Transfer Time: " << h2d_time << " ms\n";
    std::cout << "CUDA Kernel Execution Time: " << kernel_time << " ms\n";
    std::cout << "CUDA D2H Memory Transfer Time: " << d2h_time << " ms\n";
    std::cout << "Total CUDA End-to-End Time: " << total_cuda_time << " ms\n";
    std::cout << "Effective Memory Bandwidth (including H2D and D2H): " << effective_bandwidth_gbps << " GB/s\n";


    //printing some results (optional, can be removed for pure benchmarking)
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i]
                  << " + "
                  << h_B[i]
                  << " = "
                  << h_C[i]
                  << '\n';
    }

    //cleaning up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(memcpy_h2d_start);
    cudaEventDestroy(memcpy_h2d_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(memcpy_d2h_start);
    cudaEventDestroy(memcpy_d2h_stop);

    return 0;
}