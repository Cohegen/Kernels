# Introduction Inclusive Scan
- An inclusive scan is a parallel algorithm that calculates the prefix sum of an array, including the current element in each sum.

## Example
Say we have an input array:
```
X = [2,4,6,8,10]
```
- An inclusive scan produces
```
Y = [2,6,12,20,30]
```
- So get various inclusive sums in our array we do:
  ```
  Y[0] = X[0]
  Y[1] = X[0] + X[1]
  Y[2] = X[0] + X[1] + X[2]
  Y[3] = X[0] + X[1] + X[2] + X[3]
  Y[4] = X[0] + X[1] + X[2] + X[3] + X[4]
  ```
- To implement sequential scan we use the following algorithm in the C++ code below:
```
void sequential_scan(float* x, float* y,unsigned int N)
{
    y[0] = x[0];
    for(unsigned int i=1;i<N;++i)
    {
        y[i] = y[i-1] + x[i];
    }
}
```
- From the code above we can see that each position contains the current element and the inclusive of the previous element.
- Take for example **Y[3]** it contains the sum of **X[3]** and **Y[2]**, where **Y[2]** contains the sum of **X[0],X[1],X[2]**.

What if instead of using a naive sequential scan algorithm, we could parallel algorithms through kernels run a GPU?. Well this is possibe.

In this directory the following kernels have been implemented and we're going to go deep into them.
These kernels include:
           - The Koggle-Stone algorithm
           - The Brent-Kung algorithm
           - The Blelloch scan algorith (this is an exclusive scan algorithm)
           - A coarsened scan .

# Koggle-Stone Algorithm
- This algorithm uses an approach whereby we compute the prefix results of an array in a tree-like structure.
- We implement it using the kernel in **koggle_stone.cu**.
- Let's take out main skeleton/algorithm of the kernel.

```
__global__ void Koggle_Stone_scan(float* X, float* Y, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
```
