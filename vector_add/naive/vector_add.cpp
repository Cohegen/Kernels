/*
Naive Vector addition in C++
*/
#include <iostream>
#include <cstddef>
#include <chrono>

void vector_add(
    const float* A,
    const float* B,
    float* C,
    std::size_t N
) {
    for (std::size_t i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const std::size_t N = 100'000'000; 

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    // Initialize vectors
    for (std::size_t i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    // Start timing for CPU execution
    auto start_time = std::chrono::high_resolution_clock::now();

    // Vector addition
    vector_add(A, B, C, N);

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "\n--- CPU Benchmarking Results ---\n";
    std::cout << "CPU Execution Time: " << duration.count() << " microseconds\n";

    // Print first 10 results
    for (std::size_t i = 0; i < 10; i++) {
        std::cout << A[i]
                  << " + "
                  << B[i]
                  << " = "
                  << C[i]
                  << '\n';
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}