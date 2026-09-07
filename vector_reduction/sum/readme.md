# Vector Sum Reduction

In this folder, I implemented parallel sum reduction in CUDA, testing several kernels to see how different memory and thread indexing strategies impact performance.

Given an array `arr` of size `N`, the goal is to compute the sum of all its elements:

```text
sum = arr[0] + arr[1] + ... + arr[N-1]
```

Instead of summing sequentially on the CPU in O(N), a parallel reduction uses a tree-based approach where pairs of elements are added at each step, finishing in O(log N) steps.

---

## Implementations

I went through a few iterations to optimize the reduction:

### 1. Naive Sum Reduction (`naive_sum_reduction.cu`)
This is the baseline implementation using global memory and interleaved addressing.

```cpp
for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (index % (2 * stride) == 0) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

- **Problem**: The `index % (2 * stride) == 0` check causes heavy warp divergence. Threads within the same 32-thread warp take different branch paths, so execution gets serialized within each warp.

### 2. Naive Reduction without Divergence (`naive_sum2.cu`)
To fix warp divergence, I changed how thread indices are calculated. Instead of using modulo on thread IDs, active threads are grouped together:

```cpp
for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

- **Why this helps**: Consecutive threads (`tid = 0, 1, 2...`) do the work. This keeps threads in the same warp executing together, while idle warps can remain inactive without stalling active ones.

### 3. Convergent Sum Reduction (`convergent_sum_naive.cu`)
Another way to avoid divergence is to reverse the stride direction—starting with half the block and halving it each step:

```cpp
for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (index < stride) {
        arr[index] += arr[index + stride];
    }
    __syncthreads();
}
```

- Active threads are always `0` to `stride - 1`, keeping memory accesses and thread execution convergent.

### 4. Shared Memory Reduction (`shared_mem_reduction.cu`)
In the previous versions, every addition step reads and writes back to global memory (DRAM). In this version, the input array is loaded once into on-chip shared memory (`__shared__`):

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

- Reading from shared memory is much faster than round-tripping to global memory on every step.

### 5. Shared Memory without Divergence (`shared_mem_sum2.cu`)
This kernel combines shared memory with the non-divergent indexing from kernel 2 (`2 * stride * tid`):

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

### 6. Convergent Shared Memory (`convergent_shared_reductio.cu`)
Combines shared memory with reversed stride halving (`stride = BLOCK_DIM / 2; stride /= 2`). This helps minimize both warp divergence and shared memory bank conflicts.

---

## Benchmarks ($N = 256$)

I benchmarked each kernel on an array of size $N = 256$ using CUDA events to record transfer times and kernel execution time.

| Kernel | Memory Used | Addressing | Kernel Time (ms) | Total End-to-End (ms) |
|:---|:---|:---|---:|---:|
| `naive_sum_reduction` | Global | Interleaved (modulo) | 0.2447 ms | 0.2905 ms |
| `naive_sum_reduction_noDiverge` | Global | Interleaved (`2*stride*tid`) | 0.1624 ms | 0.2397 ms |
| `convergent_sum_reduction` | Global | Convergent (`stride /= 2`) | 0.2314 ms | 0.3118 ms |
| `shared_mem_sum_reduction` | Shared | Interleaved (modulo) | 0.1700 ms | 0.2353 ms |
| `shared_mem_noDiverge` | Shared | Interleaved (`2*stride*tid`) | 0.1710 ms | 0.2899 ms |
| `convergent_shared_mem` | Shared | Convergent (`stride /= 2`) | 0.1721 ms | 0.2620 ms |

### Detailed Timing Breakdown

| Kernel Name                    | H2D Time (ms) | H2D BW (GB/s) | Kernel Time (ms) | D2H Time (ms) | D2H BW (GB/s) | E2E Time (ms) |
|--------------------------------|--------------:|--------------:|-----------------:|--------------:|--------------:|--------------:|
| `naive_sum_reduction`          | 0.046304      | 0.176918      | 0.202464         | 0.031520      | 0.000127      | 0.299392      |
| `shared_mem_sum_reduction`     | 0.021184      | 0.386707      | 0.163712         | 0.024832      | 0.000161      | 0.225824      |
| `convergent_sum_reduction`     | 0.018432      | 0.444444      | 0.217120         | 0.026432      | 0.000151      | 0.279552      |
| `convergent_shared_mem`        | 0.071136      | 0.115160      | 0.212768         | 0.026176      | 0.000153      | 0.327072      |
| `naive_sum_reduction_noDiverge` | 0.022528     | 0.363636      | 0.151520         | 0.025248      | 0.000158      | 0.215040      |
| `shared_mem_noDiverge`         | 0.023904      | 0.342704      | 0.169504         | 0.029184      | 0.000137      | 0.249088      |
| `fast_add_load`               | 0.014528      | 0.070485      | 0.162720         | 0.027296      | 0.000147      | 0.222560      |
| `unrolling_last_warp`         | 0.016352      | 0.062622      | 0.164864         | 0.026112      | 0.000153      | 0.225312      |

---

## Observations

* **Warp divergence matters a lot**: In the naive kernel, using modulo (`index % (2 * stride) == 0`) forced threads in the same warp to branch differently. Switching to contiguous indexing (`2 * stride * tid`) dropped kernel execution time from ~0.245 ms to ~0.162 ms (around 34% faster).
* **Shared memory vs. Global memory**: Loading the data into shared memory once avoids repeated round trips to global memory during each reduction pass.
* **Transfer overhead at small array sizes**: With $N = 256$, the array is small, so the host-to-device and device-to-host transfers take up a noticeable portion of the total time. The real optimization differences show up primarily in the kernel execution time.
* For larger inputs (e.g. $N > 1024$), multi-block reductions and warp shuffle instructions will be needed to scale further.

## Next Steps

* [ ] Benchmark with much larger array sizes ($N = 10^5, 10^6$) to see shared memory benefits at scale.
* [ ] Use warp shuffle primitives (`__shfl_down_sync`) to reduce within warps without shared memory overhead.
* [ ] Unroll the last warp to eliminate unnecessary `__syncthreads()` calls.
* [ ] Implement a multi-block reduction with atomic additions or a two-pass reduction kernel.