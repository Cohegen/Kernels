#include <cuda_runtime.h>
#include <stdio.h>

#define SECTION_SIZE 256
__global__ void brent_kung_scan(float*X,float*Y,unsigned int N)
{
    //array in shared memory which stores partial sums
    __shared__ float XY[SECTION_SIZE];

    //defining coordinates
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;

    //loading elements in shared memory
    if(i<N)
    {
        XY[threadIdx.x] = X[i];
    }
    if(i+blockDim.x <N)
    {
        XY[threadIdx.x +blockDim.x] = X[i+blockDim.x];
    }

    // reduction tree
    for(unsigned int stride=1;stride <= blockDim.x;stride*=2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2* stride -1;
        if(index <SECTION_SIZE)
        { 
            XY[index] += XY[index - stride];
        }
    }

    //reverse tree
    for(int stride = SECTION_SIZE/4;stride >0;stride /=2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x +1)*stride*2 -1;
        if(index + stride <SECTION_SIZE)
        {
            XY[index+stride] += XY[index]; 
        }
    }

    __syncthreads();
    if(i<N)
    {
        Y[i] = XY[threadIdx.x];
    }
    if(i+blockDim.x <N)
    {
        Y[i+blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}

int main() {
    const int N = 256;
    float h_X[N], h_Y[N];
    for (int i = 0; i < N; ++i) {
        h_X[i] = 1.0f;
    }

    float *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));

    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    brent_kung_scan<<<1, SECTION_SIZE/2>>>(d_X, d_Y, N);

    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Brent-Kung First 5 elements:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    return 0;
}