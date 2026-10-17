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
From the kernel above we first created an array **XY** which is located in the shared memory. It's main role will be storing the partial sums after each stage completes their respective sum.
But what are these stages and where do they come from? Well, assume we have a **blockDim** of say 8, meaning our array has eight elements. From the for loop we can  that the stride doubles after every round/stage and since we set a condition that stride must be less than **blockDim**. This means that stride will go from 1 to 2 then to 4 counting the amount of the strides we see we have 3 stages.

Given an array:
```
X = [1,2,3,4,5,6,7,8]
```
when stride = 1
```
XY[0] = X[0]
XY[1] = X[0]+X[1]
XY[2] = X[1] + X[2]
.....
XY[7] = X[6] + X[7]
```

stride = 2
```
XY[2] = X[0] + X[2]
XY[3] = X[1] + X[3]
...
XY[7] = X[5] + x[7]
```
stride = 4
```
XY[4] =X[0] + X[4]
XY[5] = X[1] + X[5]
XY[6] = X[2] + X[6]
XY[7] = X[3] + X[7] 
```

From above we can see that when stride=1(stage 1) we have 7 additions , when stride=2(stage 2) we have 6 additions and finally when stride=4 we have 4 additions brings the total number of additions to 7+6+4 = 17. 
So we conclude that work done $W$ by Koggle-Stone scan is 17 additions and for us to finish this work we need 3 stages  meaning Koggle-Stone dependent stage is $log_{2} N$ where $N$ is the number of stages that are required to deem the work to be complete.
