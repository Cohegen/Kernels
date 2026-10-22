#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void singlePassScan(const int* input, 
                               int* output, 
                               int* scan_value, 
                               int* flags, 
                               unsigned int* blockCounter, 
                                std::size_t N)
{
    __shared__ int temp[BLOCK_SIZE];
    __shared__ unsigned int bid_s;
    __shared__ int previous_sum;

    unsigned int tid = threadIdx.x;

    //phase 0: dynamic block index assignment
    if(tid == 0)
    {
        bid_s = atomicAdd(blockCounter, 1u);
    }

    __syncthreads();

    unsigned int bid = bid_s;
    std::size_t gid = static_cast<std::size_t>(bid) * BLOCK_SIZE + tid;

    //phase 1: loading input to shared memory
    temp[tid] = (gid < N) ? input[gid] : 0;

    __syncthreads();

    //phase 2: Brent-Kung upsweep
    for(unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2)
    {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if(index < BLOCK_SIZE)
        {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    //phase 3: brent-kung downsweep
    for(unsigned int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2)
    {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if(index + stride < BLOCK_SIZE)
        {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    //phase 4: adjacent block synchronization
    if(tid == 0)
    {
        int offset = 0;
        if(bid > 0)
        {
            /*
            Device-scope atomic references to the
            predecessor's readiness flag
            */
            cuda::atomic_ref<int, cuda::thread_scope_device> ready(flags[bid]);

            //waiting until predecessor publishes its sum
            while(ready.load(cuda::memory_order_acquire) == 0)
            {
                // Busy wait
            }
            offset = scan_value[bid];
        }

        previous_sum = offset;

        //last element contains the local block sum
        int local_sum = temp[BLOCK_SIZE - 1];
        int cumulative_sum = offset + local_sum;

        //publishing cumulative sum for next block
        scan_value[bid + 1] = cumulative_sum;

        cuda::atomic_ref<int, cuda::thread_scope_device> next_ready(flags[bid + 1]);
        next_ready.store(1, cuda::memory_order_release);
    }
    __syncthreads();

    //phase 5: add block offset and write final results
    if(gid < N)
    {
        output[gid] = temp[tid] + previous_sum;
    }
}

void launchSinglePassScan(const int* d_input, int* d_output, std::size_t N)
{
    if(N == 0)
    {
        return;
    }

    std::size_t numBlocks = (N - 1) / BLOCK_SIZE + 1;

    if(numBlocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::invalid_argument("Too many blocks");
    }

    int *d_scan_value = nullptr;
    int *d_flags = nullptr;
    unsigned int *d_blockCounter = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_scan_value), (numBlocks + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_flags), (numBlocks + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_blockCounter), sizeof(unsigned int));

    //all flags initially zero
    cudaMemset(d_flags, 0, (numBlocks + 1) * sizeof(int));
    //dynamic block index starts at zero
    cudaMemset(d_blockCounter, 0, sizeof(unsigned int));

    // Initialise the first scan value and flag to make the first block pass through
    int initial_ready = 1;
    cudaMemcpy(d_flags, &initial_ready, sizeof(int), cudaMemcpyHostToDevice);

    singlePassScan<<<static_cast<unsigned int>(numBlocks), BLOCK_SIZE>>>(d_input, 
                                                                         d_output, 
                                                                         d_scan_value, 
                                                                         d_flags, 
                                                                         d_blockCounter, 
                                                                         N);

    //waiting for the kernel before it releases memory
    cudaDeviceSynchronize();

    cudaFree(d_scan_value);
    cudaFree(d_flags);
    cudaFree(d_blockCounter);
}

int main() {
    const std::size_t N = 512;
    int h_X[N], h_Y[N];
    for (std::size_t i = 0; i < N; ++i) {
        h_X[i] = 1;
    }

    int *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(int));
    cudaMalloc(&d_Y, N * sizeof(int));

    cudaMemcpy(d_X, h_X, N * sizeof(int), cudaMemcpyHostToDevice);

    launchSinglePassScan(d_X, d_Y, N);

    cudaMemcpy(h_Y, d_Y, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Single-Pass Scan First 10 elements:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d] = %d\n", i, h_Y[i]);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    return 0;
}
