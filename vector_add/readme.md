# Vector Addition

* In this folder, element-wise vector addition is implemented on both the CPU and GPU.

* Given two vectors **A** and **B**, the operation is:

```text
C[i] = A[i] + B[i]
```

for every element **i** in the vectors.

## Implementation

The vector addition has two variants:

* A vector addition program that runs on the CPU.
* A vector addition kernel that runs on the GPU.

The CPU implementation iterates over the elements of the vectors, while the GPU uses massive thread-level parallelism, with each thread computing one element of the output vector.

### CPU

```cpp
for (std::size_t i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

### CUDA

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < N) {
    C[index] = A[index] + B[index];
}
```

The CUDA implementation uses the global thread index to map each GPU thread to an element of the vector.

## Benchmarks

The implementations are compared using the following metrics:

1. CPU execution time
2. CUDA kernel execution time
3. Host-to-device (H2D) transfer time
4. Device-to-host (D2H) transfer time
5. End-to-end CUDA execution time
6. Effective memory bandwidth
7. CPU/GPU speedup

### Results

#### CPU

```text
--- CPU Benchmarking Results ---
CPU Execution Time: 484941 microseconds
```

#### CUDA

```text
--- CUDA Benchmarking Results ---
CUDA H2D Memory Transfer Time: 175.889 ms
CUDA Kernel Execution Time: 4.84806 ms
CUDA D2H Memory Transfer Time: 283.234 ms
Total CUDA End-to-End Time: 464.391 ms
Effective Memory Bandwidth (including H2D and D2H): 2.43418 GB/s
```

## Benchmarks
### 1. Testing different blocksizes
- In the **block_tuning** folder there are two cuda files which have blocksizes of 512 and 1024 threads respectively.
- After compiling them, I compared their metrics with our naive vector add kernel.
- The following results were obtained

| Kernel | H2D Transfer (ms) | Kernel Execution (ms) | D2H Transfer (ms) | Total End-to-End (ms) | Effective Bandwidth (GB/s) |
|:------------------------|--------------------:|----------------------:|------------------:|------------------------:|---------------------------|
| Naive (Block Size 1024) |             170.280 |                 4.76640 |           269.089 |                 444.505 |                     2.54361 |
| Block Size 512          |             168.334 |                 4.72291 |           265.990 |                 439.399 |                     2.57316 |
| Block Size 1024         |             167.794 |                 4.76867 |           261.112 |                 434.065 |                     2.60567 |

## Observations

* Vector addition performs very little computation per element compared with the amount of data that must be read and written.
* Each element requires reading **A[i]**, reading **B[i]**, performing one addition, and writing **C[i]**.
* Therefore, vector addition is primarily a **memory-bandwidth-bound** workload.
* The CUDA kernel is significantly faster than the CPU implementation when considering kernel execution alone.
* However, the CUDA implementation incurs significant host-to-device and device-to-host memory-transfer overhead.
* For this benchmark, the total CUDA execution time is currently lower than the CPU execution time, so the cost of transferring data between the CPU and GPU dominates the overall execution time.

### Future Work

* Benchmark different vector sizes.
* Benchmark different CUDA block sizes.
* Compare kernel-only performance against CPU execution time.
* Investigate global memory access efficiency.
* Measure achievable GPU memory bandwidth.
* Compare against an optimized CPU implementation.
* Investigate the point at which GPU execution becomes advantageous over CPU execution.
