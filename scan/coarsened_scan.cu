#include <iostream>

#define BLOCK_DIM 256

__global__ void coarsened_scan(float*input,float*output,float*partialSums)
