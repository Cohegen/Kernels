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
