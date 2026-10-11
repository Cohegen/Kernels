#include "timer.h"

#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input,float* output,float*partialsums,unsigned int N)
{
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float buffer_shared1[BLOCK_DIM];
    __shared__ float buffer_shared2[BLOCK_DIM];
    float *inBuffer_shared = buffer_shared1;
    float* outBuffer_shared = buffer_shared2;
    buffer_shared[threadIdx.x] = input[i];
    for(unsigned int stride=1;stride<BLOCK_DIM/2;stride *=2)
    {
        if(threadIdx.x>= stride)
        {
            float temp
            outBuffer_shared[threadIdx.x]= inBuffer_shared[threadIdx.x] + inBuffer_shared[threadIdx.x - stride];
        }else{
            outBuffer_shared[threadIdx.x] = inBuffer_shared[threadIdx.x];
        }
        __syncthreads();
       float* temp = inBuffer_shared;
       inBuffer_shared = outBuffer_shared;
       outBuffer_shared = temp;
      
        if(threadIdx.x==BLOCK_DIM-1)
        {
            partialsums[blockIdx.x] = buffer_shared[threadIdx.x];
        }
        output[i] = inBuffer_shared[threadIdx.x];
       
    }
}