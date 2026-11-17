#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>



__device__ void exclusive_scan_block(unsigned int* bits, unsigned int* scanned, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        // Simple sequential scan on shared memory or global memory for demonstration (for 1 block)
        if (idx == 0) {
            unsigned int sum = 0;
            for (int j = 0; j < N; j++) {
                scanned[j] = sum;
                sum += bits[j];
            }
            // Store total sum in the N-th element
            scanned[N] = sum;
        }
    }
    __syncthreads();
}

__global__ void radix_sort_iter(
    unsigned int*  input,
    unsigned int* output,
    unsigned int* bits,
    unsigned int* scanned_bits,
    unsigned int N,
    unsigned int iter
)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int key = 0;
    unsigned int bit = 0;
    if(i < N)
    { 
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }
    __syncthreads();

    // Perform the scan
    exclusive_scan_block(bits, scanned_bits, N);

    if(i < N)
    {
        unsigned int numOnesBefore = scanned_bits[i];
        unsigned int numOnesTotal = scanned_bits[N];
        unsigned int numZerosTotal = N - numOnesTotal;
        
        // Destination index calculation for stable partitioning:
        // Zeros go to the front, Ones go to the back
        unsigned int dst;
        if (bit == 0) {
            // Index among zeros is: total index 'i' minus number of ones before 'i'
            dst = i - numOnesBefore;
        } else {
            // Index among ones is: offset by total zeros + number of ones before 'i'
            dst = numZerosTotal + numOnesBefore;
        }

        output[dst] = key;
    }
}

int main() {
    const int N = 16;
    unsigned int host_input[N] = {15, 2, 8, 4, 1, 0, 9, 3, 7, 5, 11, 14, 12, 13, 6, 10};
    unsigned int host_output[N] = {0};

    unsigned int *d_input, *d_output, *d_bits, *d_scanned_bits;
    cudaMalloc(&d_input, N * sizeof(unsigned int));
    cudaMalloc(&d_output, N * sizeof(unsigned int));
    cudaMalloc(&d_bits, N * sizeof(unsigned int));
    cudaMalloc(&d_scanned_bits, (N + 1) * sizeof(unsigned int));

    cudaMemcpy(d_input, host_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    std::cout << "Input data: ";
    for(int i = 0; i < N; i++) std::cout << host_input[i] << " ";
    std::cout << "\n";

    // Let's sort bit-by-bit from LSB (0) to MSB (3) for values < 16
    for (int iter = 0; iter < 4; iter++) {
        radix_sort_iter<<<1, N>>>(d_input, d_output, d_bits, d_scanned_bits, N, iter);
        cudaDeviceSynchronize();
        // Copy output back to input for the next iteration
        cudaMemcpy(d_input, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(host_output, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted data: ";
    for(int i = 0; i < N; i++) std::cout << host_output[i] << " ";
    std::cout << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bits);
    cudaFree(d_scanned_bits);
    return 0;
}
