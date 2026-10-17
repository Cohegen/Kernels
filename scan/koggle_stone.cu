#include <cuda_runtime.h>
#include <stdio.h>

#define SECTION_SIZE 256

__global__ void Koggle_Stone_scan(float* X, float* Y, unsigned int N)
{
    //shared memory partial sums
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //loading elements into the shared memory array
    if (i < N)
    {
        XY[threadIdx.x] = X[i];
    }
    else
    {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp = 0.0f;
        if (threadIdx.x >= stride)
        {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride)
        {
            XY[threadIdx.x] = temp;
        }
    }

    if (i < N)
    {
        Y[i] = XY[threadIdx.x];
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

    Koggle_Stone_scan<<<1, SECTION_SIZE>>>(d_X, d_Y, N);

    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 5 elements of prefix sum:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    return 0;
}