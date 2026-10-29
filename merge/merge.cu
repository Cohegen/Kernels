
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

// Sequentially merge two sorted subarrays.
void merge_sequential(const int* A, int m,const int* B, int n,int* C)
{
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < m && j < n)
    {
        if (A[i] <= B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }

    while (i < m)
    {
        C[k++] = A[i++];
    }

    while (j < n)
    {
        C[k++] = B[j++];
    }
}

// Find the co-rank for the first k elements
// of the merged output.
__device__ int co_rank(nt k,const int* A, int m,const int* B, int n)
{
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    while (true)
    {
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            int delta = (i - i_low + 1) >> 1;

            j_low = j;
            i -= delta;
            j += delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            int delta = (j - j_low + 1) >> 1;

            i_low = i;
            i += delta;
            j -= delta;
        }
        else
        {
            return i;
        }
    }
}

// Basic parallel merge kernel.
__global__ void merge_basic_kernel(const int* A, int m,const int* B, int n,int* C)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int totalElements = m + n;
    int totalThreads = blockDim.x * gridDim.x;

    // Ceiling division.
    int elementsPerThread =
        (totalElements + totalThreads - 1) / totalThreads;

    int k_curr = index * elementsPerThread;

    // No output elements remain for this thread.
    if (k_curr >= totalElements)
        return;

    int k_next = min(
        k_curr + elementsPerThread,
        totalElements
    );

    // Find input boundaries for this thread.
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    // Merge the assigned input subarrays.
    merge_sequential(
        A + i_curr, i_next - i_curr,
        B + j_curr, j_next - j_curr,
        C + k_curr
    );
}

