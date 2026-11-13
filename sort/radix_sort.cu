#include <cuda_runtime.h>

__global__ void radix_sort_iter(
    unsigned int*  input,
    unsigned int* output,
    unsigned int* bits,
    unsigned int N,
    unsigned int iter
)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int key,bit;
    if(i<N)
    {
        key = input[i];
        bit = [key >> iter]&1;
        bits[i] = bit;
        
    }
    exclusiveScan(bits,N);
    if(i<N)
    {
        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit==0) ? (i-numOnesBefore):(N-numOnesTotal-numOnesBefore);

        output[dst] = key;
    }
}