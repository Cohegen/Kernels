#include <iostream>
#include <cuda_runtime.h>

__device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C)
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

__device__ int co_rank(int k, const int* A, int m, const int* B, int n)
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
__global__ void tiled_merge(int* A,int m,int*B,int n,int* C,int tile_size)
{
    /*
    shared memory allocation
    */
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0]; //A_S is first half of shareAB
    int* B_S = sharedAB[tile_size]; //B_S is second half of shareAB
    int C_curr = blockIdx.x*ceil((m+n)/gridDim.x); // start point of block's C subarray
    int C_next = min((blockIdx.x+1)*ceil((m+n)/gridDim.x),(m+n));//ending point

    if(threadIdx.x == 0)
    {
        A_S[0] = co_rank(C_curr,A,m,B,n); // making block-level co-rank values visible
        A_S[1] = co_rank(C_next,A,m,B,n); // to other threads in the block
    }

    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr-A_curr;
    int B_next = C_next - A_next;

    __syncthreads();

    int  counter =0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iters = ceil((C_length)/tile_size);
    int C_completed  =0;
    int A_consumed = 0;
    int B_consumed = 0;
    while(counter <total_iters)
    {
        /*
        loading tile-size A and B elements into shared memory
        */
        for(int i=0;i<tile_size;i+=blockDim.x)
        {
            if(i+threadIdx.x < A_length-A_consumed)
            {
                A_S[i+threadIdx.x] = A[A_curr+A_consumed+i+threadIdx.x];
            }
        }
        for(int i=0;i<tile_size;i+=blockDim.x)
        {
            if(i+threadIdx.x <B_length-B_consumed)
            {
                B_S[i+threadIdx.x] = B[B_curr+B_consumed+i+threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x* (tile_size/blockDim.x);
        int c_next = (threadIdx.x)*(tile_size/blockDim.x);
        c_curr = (c_curr <= C_length-C_completed) ? c_curr :C_length-C_completed;
        c_next = (c_next <= C_length-C_completed) ? c_next: C_length - C_completed;

        //finding co-rank for c_curr and c_next
        int a_curr = co_rank(c_curr,A_S,min(tile_size,A_length-A_consumed),B_S,min(tile_size,B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        
        int a_next = co_rank(c_next,A_S,min(tile_size,A_length-A_consumed),B_S,min(tile_size,B_length-B_consumed));
        int b_next = c_next - a_next;

        /*
        All threads call sequential merge function
        */
        merge_sequential(A_S+a_curr,a_next-a_curr,B_S+b_curr,b_next-b_curr,C+C_curr+C_completed+c_curr);

        /*
        updating the number of A and B elements that habe been consumed thus far
        */
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size,B_S,tile_size);
        B_consumed = C_completed - A_consumed;

        __syncthreads();
    }   
    
}