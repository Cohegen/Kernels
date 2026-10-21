#include <cuda_runtime.h>
#include <stdio.h>

constexpr unsigned int BLOCK_DIM = 4;
constexpr unsigned int ELEMENTS_PER_THREAD = 4;
constexpr unsigned int SECTION_SIZE = BLOCK_DIM * ELEMENTS_PER_THREAD;

__global__ void coarsened_scan(const int* input, int* output, unsigned int N)
{
    //shared memory array
    __shared__ int shared_arr[SECTION_SIZE];

    //shared memory array for subsection totals
    __shared__ int totals_arr[BLOCK_DIM];

    //defining coordinates
    const unsigned int tid = threadIdx.x;

    //first input element process by this block
    const unsigned int block_start = blockIdx.x * SECTION_SIZE;

    //phase 1: cooperatively loading input into shared memory
    for(unsigned int i=tid; i<SECTION_SIZE; i+= BLOCK_DIM)
    {
        shared_arr[i] = (block_start + i < N) ? input[block_start+i] : 0;
    }

    __syncthreads();

    //each thread's subsection begins here
    const unsigned int local_start = tid * ELEMENTS_PER_THREAD;

    //Sequential inclusive scan per thread
    int running_sum = 0;
    for(unsigned int i=0; i<ELEMENTS_PER_THREAD; ++i)
    {
        const unsigned int index = local_start + i;
        running_sum += shared_arr[index];
        shared_arr[index] = running_sum;
    }

    //saving the total of this thread's subsection
    totals_arr[tid] = running_sum;

    __syncthreads();

    //Phase 2: Kogge-Stone scan for subsection total
    for(unsigned int stride=1; stride<BLOCK_DIM; stride*=2)
    {
        /*
        Each thread first reads the previous iteration's
        values before any thread overwrites them
        */
        const int current = totals_arr[tid];
        int predecessor = 0;
        if(tid >= stride)
        {
            predecessor = totals_arr[tid-stride];
        }

        __syncthreads();

         //update the current subsection total
        if(tid >= stride)
        {
            totals_arr[tid] = current + predecessor;
        }
        else{
            totals_arr[tid] = current;
        }

        __syncthreads();
    }

    //phase 3: Add preceding subsections totals
    int prefix = 0;
    if(tid > 0)
    {
        prefix = totals_arr[tid-1];
    }

    for(unsigned int i=0; i<ELEMENTS_PER_THREAD; ++i)
    {
        const unsigned int index = local_start + i;
        int value = shared_arr[index];

        /*
        The final element already has its global prefix
        sum after phase 2
        */
        if(i == ELEMENTS_PER_THREAD-1)
        {
            value = totals_arr[tid];
        }
        else{
            value += prefix;
        }
        if (block_start + index < N) {
            output[block_start+index] = value;
        }
    }
}

int main() {
    const int N = SECTION_SIZE;
    int h_X[N], h_Y[N];
    for (int i = 0; i < N; ++i) {
        h_X[i] = 1;
    }

    int *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(int));
    cudaMalloc(&d_Y, N * sizeof(int));

    cudaMemcpy(d_X, h_X, N * sizeof(int), cudaMemcpyHostToDevice);

    coarsened_scan<<<1, BLOCK_DIM>>>(d_X, d_Y, N);

    cudaMemcpy(h_Y, d_Y, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Coarsened Scan First 16 elements:\n");
    for (int i = 0; i < 16; ++i) {
        printf("Y[%d] = %d\n", i, h_Y[i]);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    return 0;
}
