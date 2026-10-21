#include <cuda_runtime.h>

#define SECTION_SIZE 256

__global__ void block_scan(const float*X,float*Y,float* block_sums,unsigned int N)
{
   __shared__ float XY[SECTION_SIZE];
   
   unsigned int tx = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x+tx;

   //loading elements into shared memory
   XY[tx] = (i<N) ? X[i] : 0.0f;

   __synchthreads();

   //koggle-stone inclusive scan
   for(unsigned int stride=1;stride<blockDim.x;stride*=2)
   {
    float temp = 0.0f;
    if(tx>= stride)
    {
        temp = XY[tx] + XY[tx-stride];
    }
    __syncthreads();

    if(tx>= stride)
    {
        XY[tx] = temp;
    }

    __syncthreads();
   }

   //writing block-local scan results
   if(i<N)
   {
    Y[i] = XY[tx];
   }

   //storing the total sums of each block
   if(tx ==blockDim.x -1)
   {
    block_sums[blockIdx.x] = XY[tx];
   }
}

//kernel 2: scans the block sums
__global__ void scan_block_sums(float* block_sums,unsigned int num_blocks)
{
    extern __shared__  float XY[];

    unsigned int tx = threadIdx.x;

    XY[tx] = (tx<num_blocks) ? block_sums[tx] : 0.0f;

    __syncthreads();

    for(unsigned int stride=1;stride<blockDim.x;stride*=2)
    {
        float temp =0.0f;

        if(tx>= stride)
        {
            temp = XY[tx] + XY[tx-stride];
        }

        __syncthreads();

        if(tx >= stride)
        {
            XY[tx] = temp;
        }
    }

    if(tx <num_blocks)
    {
        block_sums[tx] = XY[tx];
    }
}

//kernel 3: Adding the preceding block sus

__global__ void add_block_sums(float* Y,const float* block_sums,unsigned int N)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<N && blockIdx.x >0)
    {
        Y[i] += block_sums[blockIdx.x-1];
    }
}

//launching the kernels
void hierarchial_scan(const float* d_X,float* d_Y,unsigned int N)
{
    if(N==0)
    {
        return;
    }

    unsigned int num_blocks = (N+SECTION_SIZE) /SECTION_SIZE;

    float* d_block_sums;

    cudaMalloc(&d_block_sums,num_blocks*sizeof(float));

    //phase 1: scanning each block
    block_scan <<< num_blocks,SECTION_SIZE>>>(d_X,d_Y,d_block_sums,N);

    //phase 2: scan block sums
    if(num_blocks >1)
    {
        scan_block_sums<<<1,num_blocks,num_blocks*sizeof(float)>>>(d_block_sums,num_blocks);

        //phase 3: adding preceding sums
        add_block_sums<<<num_blocks,SECTION_SIZE>>>(d_Y,d_block_sums,N);
    }
    
    cudaFree(d_block_sums);
}