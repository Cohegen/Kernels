# Parallel Vector Sum Reduction

In this directory, parallel sum reduction is implemented in CUDA across several progressive optimization stages.

Given an input array $A$ of $N$ elements, the reduction computes the scalar sum:

$$S = \sum_{i=0}^{N-1} A[i]$$

Sum reduction uses a binary tree structure where pairs of elements are added in parallel across logarithmic steps ($\mathcal{O}(\log N)$ time complexity).

---

## Optimization Stages & Kernels

### 1. Naive Interleaved Addressing (`naive_sum_reduction.cu`)
- **Memory**: Global Memory
- **Addressing**: Interleaved with modulo arithmetic (`if (index % (2 * stride) == 0)`).
- **Bottleneck**: Severe **warp divergence**. Because participating threads are spaced out across the warp, threads within the same 32-thread warp take conflicting execution paths, serializing warp execution.

```cpp
for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (index % (2 * stride) == 0) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

### 2. Non-Divergent Interleaved Addressing (`naive_sum2.cu`)
- **Memory**: Global Memory
- **Addressing**: Contiguous thread ID mapping (`int index = 2 * stride * tid`).
- **Improvement**: Active threads are grouped consecutively (`tid`). Threads in lower-numbered warps remain active together while higher-numbered warps exit cleanly without intra-warp divergence.

```cpp
for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

### 3. Convergent Tree Reduction (`convergent_sum_naive.cu`)
- **Memory**: Global Memory
- **Addressing**: Reversed stride halving (`stride = BLOCK_DIM / 2; stride > 0; stride /= 2`).
- **Improvement**: Threads read and accumulate pairs separated by half the active block dimension (`index + stride`). Threads are kept contiguous (`index < stride`).

```cpp
for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (index < stride) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

### 4. Shared Memory Interleaved Reduction (`shared_mem_reduction.cu`)
- **Memory**: On-chip Shared Memory (`__shared__`)
- **Addressing**: Interleaved modulo addressing.
- **Improvement**: Stages data from off-chip DRAM into on-chip shared memory (SRAM) before reduction, reducing memory access latency from hundreds of cycles to a few cycles.

```cpp
__shared__ float shared_arr[BLOCK_DIM];
shared_arr[index] = arr[index];
__syncthreads();

for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
    if (index % (2 * stride) == 0) {
        shared_arr[index] += shared_arr[index + stride];
    }
    __syncthreads();
}
```

### 5. Shared Memory Non-Divergent Reduction (`shared_mem_sum2.cu`)
- **Memory**: On-chip Shared Memory
- **Addressing**: Contiguous thread indexing (`2 * stride * tid`).
- **Improvement**: Combines fast on-chip shared memory access with contiguous thread indexing to prevent warp divergence.

```cpp
__shared__ float shared_arr[BLOCK_DIM];
shared_arr[tid] = arr[tid];
__syncthreads();

for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < BLOCK_DIM) {
        shared_arr[index] += shared_arr[index + stride];
    }
    __syncthreads();
}
```

### 6. Convergent Shared Memory Reduction (`convergent_shared_reductio.cu`)
- **Memory**: On-chip Shared Memory
- **Addressing**: Reversed stride halving (`stride = BLOCK_DIM / 2; stride > 0; stride /= 2`).
- **Improvement**: Combines shared memory speed with stride halving, maximizing warp convergence and mitigating bank conflicts.

```cpp
__shared__ float shared_arr[BLOCK_DIM];
shared_arr[index] = arr[index];
__syncthreads();

for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (index < stride) {
        shared_arr[index] += shared_arr[index + stride];
    }
    __syncthreads();
}
```

---

## Benchmarks ($N = 256$)

### Performance Comparison Table

| Kernel Name | Memory Level | Addressing Strategy | Kernel Time (ms) | End-to-End Time (ms) |
|:---|:---|:---|---:|---:|
| `naive_sum_reduction` | Global | Interleaved (Modulo `%`) | 0.2447 ms | 0.2905 ms |
| `naive_sum_reduction_noDiverge` | Global | Interleaved (Contiguous `2*stride*tid`) | 0.1624 ms | 0.2397 ms |
| `convergent_sum_reduction` | Global | Reversed Halving (`stride /= 2`) | 0.2314 ms | 0.3118 ms |
| `shared_mem_sum_reduction` | Shared | Interleaved (Modulo `%`) | 0.1700 ms | 0.2353 ms |
| `shared_mem_noDiverge` | Shared | Interleaved (Contiguous `2*stride*tid`) | 0.1710 ms | 0.2899 ms |
| `convergent_shared_mem` | Shared | Reversed Halving (`stride /= 2`) | 0.1721 ms | 0.2620 ms |

---

### Raw Benchmark Outputs

<details>
<summary>Click to expand individual kernel metrics</summary>

#### Naive Sum Reduction (`naive_sum_reduction.cu`)
```text
Host-to-Device Transfer Time:   0.022528 ms (0.0454545 GB/s)
CUDA Kernel Execution Time:     0.244736 ms
Device-to-Host Transfer Time:   0.027456 ms (0.0001457 GB/s)
End-to-End CUDA Execution Time: 0.313216 ms
```

#### Naive Sum Reduction (No Divergence) (`naive_sum2.cu`)
```text
Host-to-Device Transfer Time:   0.019520 ms (0.0524590 GB/s)
CUDA Kernel Execution Time:     0.162368 ms
Device-to-Host Transfer Time:   0.025376 ms (0.0001576 GB/s)
End-to-End CUDA Execution Time: 0.223808 ms
```

#### Convergent Sum Reduction (`convergent_sum_naive.cu`)
```text
Host-to-Device Transfer Time:   0.016352 ms (0.0626223 GB/s)
CUDA Kernel Execution Time:     0.231424 ms
Device-to-Host Transfer Time:   0.027072 ms (0.0001478 GB/s)
End-to-End CUDA Execution Time: 0.289760 ms
```

#### Shared Memory Sum Reduction (`shared_mem_reduction.cu`)
```text
Host-to-Device Transfer Time:   0.023392 ms (0.0437757 GB/s)
CUDA Kernel Execution Time:     0.169984 ms
Device-to-Host Transfer Time:   0.028512 ms (0.0001403 GB/s)
End-to-End CUDA Execution Time: 0.239872 ms
```

#### Shared Memory (No Divergence) (`shared_mem_sum2.cu`)
```text
Host-to-Device Transfer Time:   0.016896 ms (0.0606061 GB/s)
CUDA Kernel Execution Time:     0.170976 ms
Device-to-Host Transfer Time:   0.028224 ms (0.0001417 GB/s)
End-to-End CUDA Execution Time: 0.233504 ms
```

#### Convergent Shared Memory Reduction (`convergent_shared_reductio.cu`)
```text
Host-to-Device Transfer Time:   0.022272 ms (0.0459770 GB/s)
CUDA Kernel Execution Time:     0.172064 ms
Device-to-Host Transfer Time:   0.028704 ms (0.0001394 GB/s)
End-to-End CUDA Execution Time: 0.239232 ms
```

</details>

---

## Observations & Analysis

1. **Warp Divergence Elimination**:
   - The naive implementation suffers from severe thread divergence due to `if (index % (2 * stride) == 0)`.
   - By restructuring thread indices to `int index = 2 * stride * tid`, active threads remain packed within the lowest thread IDs. This allows entire warps to become inactive early rather than diverging, resulting in a ~34% reduction in kernel execution time (0.245 ms down to 0.162 ms).

2. **Global vs. Shared Memory Latency**:
   - Intermediate tree reduction writes to global memory require expensive DRAM transactions.
   - Copying the block into `__shared__` memory keeps intermediate accumulations inside on-chip SRAM, cutting memory access latency from hundreds of cycles to a few clock cycles.

3. **Data Transfer & Launch Overhead**:
   - At $N = 256$, PCIe transfer time and kernel launch overhead account for the majority of the total end-to-end execution time.
   - The algorithmic advantages of shared memory, conflict-free banking, and warp-synchronous reduction will become dramatically more visible at larger scales ($N \ge 10^5$).

---

## Future Work & Next Optimizations

- [ ] **Warp Shuffle Primitives**: Use `__shfl_down_sync()` to exchange data directly across registers within a warp without shared memory or `__syncthreads()`.
- [ ] **Loop Unrolling**: Unroll the final warp (last 32 iterations) to eliminate loop branch instructions and barriers.
- [ ] **Grid-Stride Multi-Element Processing**: Have each thread aggregate multiple elements sequentially before reduction to increase arithmetic intensity.
- [ ] **Multi-Block Hierarchical Reduction**: Implement multi-block reduction with atomic operations or two-pass kernels to support arbitrary input vector lengths ($N > 1024$).