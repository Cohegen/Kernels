#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "naive_merge.cu"
#include "tiled_merge.cu"
#include "circular_buffer_merge.cu"

int main() {
    const int m = 1 << 20;
    const int n = 1 << 20;

    std::vector<int> h_A(m);
    std::vector<int> h_B(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10000000);

    for (int i = 0; i < m; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < n; ++i) h_B[i] = dis(gen);

    std::sort(h_A.begin(), h_A.end());
    std::sort(h_B.begin(), h_B.end());

    std::vector<int> h_C_cpu(m + n);
    std::merge(h_A.begin(), h_A.end(), h_B.begin(), h_B.end(), h_C_cpu.begin());

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, (m + n) * sizeof(int));

    cudaMemcpy(d_A, h_A.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // 1. Test Naive Merge Kernel
    cudaMemset(d_C, 0, (m + n) * sizeof(int));
    int threadsPerBlock = 256;
    int blocksPerGrid = 128;

    merge_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, m, d_B, n, d_C);
    cudaDeviceSynchronize();

    std::vector<int> h_C_naive(m + n);
    cudaMemcpy(h_C_naive.data(), d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

    bool naive_success = true;
    for (int i = 0; i < m + n; ++i) {
        if (h_C_naive[i] != h_C_cpu[i]) {
            naive_success = false;
            break;
        }
    }
    if (naive_success) {
        std::cout << "SUCCESS: Basic parallel merge verified correctly!\n";
    } else {
        std::cout << "FAILURE: Basic parallel merge produced incorrect results.\n";
    }

    // 2. Test Tiled Merge Kernel
    cudaMemset(d_C, 0, (m + n) * sizeof(int));
    int tile_size = 512;
    threadsPerBlock = 128;
    blocksPerGrid = 64;
    size_t shared_mem_size = 2 * tile_size * sizeof(int);

    tiled_merge<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_A, m, d_B, n, d_C, tile_size);
    cudaDeviceSynchronize();

    std::vector<int> h_C_tiled(m + n);
    cudaMemcpy(h_C_tiled.data(), d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

    bool tiled_success = true;
    for (int i = 0; i < m + n; ++i) {
        if (h_C_tiled[i] != h_C_cpu[i]) {
            tiled_success = false;
            break;
        }
    }
    if (tiled_success) {
        std::cout << "SUCCESS: Tiled parallel merge verified correctly!\n";
    } else {
        std::cout << "FAILURE: Tiled parallel merge produced incorrect results.\n";
    }

    // 3. Test Circular Buffer Merge Kernel
    cudaMemset(d_C, 0, (m + n) * sizeof(int));
    merge_circular_buffer_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_A, m, d_B, n, d_C, tile_size);
    cudaDeviceSynchronize();

    std::vector<int> h_C_circular(m + n);
    cudaMemcpy(h_C_circular.data(), d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

    bool circular_success = true;
    for (int i = 0; i < m + n; ++i) {
        if (h_C_circular[i] != h_C_cpu[i]) {
            circular_success = false;
            break;
        }
    }
    if (circular_success) {
        std::cout << "SUCCESS: Circular Buffer parallel merge verified correctly!\n";
    } else {
        std::cout << "FAILURE: Circular Buffer parallel merge produced incorrect results.\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
