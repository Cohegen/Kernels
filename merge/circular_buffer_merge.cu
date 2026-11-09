#include <iostream>
#include <cuda_runtime.h>


// Helper functions
__device__ int min_int(int a, int b)
{
    return (a < b) ? a : b;
}

__device__ int max_int(int a, int b)
{
    return (a > b) ? a : b;
}



// Sequential merge
__device__ void merge_sequential_circular(const int* A,int m,const int* B,int n,int* C,int A_S_start,int B_S_start,int tile_size)
{
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < m && j < n)
    {
        int a_index = (A_S_start + i) % tile_size;
        int b_index = (B_S_start + j) % tile_size;

        if (A[a_index] <= B[b_index])
        {
            C[k++] = A[a_index++];
            i++;
        }
        else
        {
            C[k++] = B[b_index++];
            j++;
        }
    }

    while (i < m)
    {
        int a_index = (A_S_start + i) % tile_size;
        C[k++] = A[a_index];
        i++;
    }

    while (j < n)
    {
        int b_index = (B_S_start + j) % tile_size;
        C[k++] = B[b_index];
        j++;
    }
}



// Global memory co-rank

__device__ int co_rank(int k,const int* A,int m,const int* B,int n)
{
    int i = min_int(k, m);
    int j = k - i;

    int i_low = max_int(0, k - n);
    int j_low = max_int(0, k - m);

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



// Circular buffer co-rank
__device__ int co_rank_circular(int k,const int* A_S,int m,const int* B_S,int n,int A_S_start,int B_S_start,int tile_size)
{
    int i = min_int(k, m);
    int j = k - i;

    int i_low = max_int(0, k - n);
    int j_low = max_int(0, k - m);

    while (true)
    {
        int i_cir = (A_S_start + i) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;

        int i_m_1_cir =
            (A_S_start + i - 1 + tile_size) % tile_size;

        int j_m_1_cir =
            (B_S_start + j - 1 + tile_size) % tile_size;

        if (i > 0 && j < n &&
            A_S[i_m_1_cir] > B_S[j_cir])
        {
            int delta = (i - i_low + 1) >> 1;

            j_low = j;
            i -= delta;
            j += delta;
        }
        else if (j > 0 && i < m &&
                 B_S[j_m_1_cir] >= A_S[i_cir])
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



// Circular buffer merge kernel


__global__ void merge_circular_buffer_kernel(const int* A, int m,const int* B,int n,int* C,int tile_size)
{
    extern __shared__ int shareAB[];

    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];

    int total = m + n;

  
    // Part 1: Partition the output array among thread blocks
    
    int block_output_size =
        (total + gridDim.x - 1) / gridDim.x;

    int C_curr = blockIdx.x * block_output_size;

    int C_next = min_int(
        C_curr + block_output_size,
        total
    );

    if (C_curr >= total)
        return;

    // --------------------------------------------------------
    // Part 2: Determine input partitions using co-ranking
    // --------------------------------------------------------

    int A_curr = co_rank(C_curr, A, m, B, n);
    int A_next = co_rank(C_next, A, m, B, n);

    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int C_length = C_next - C_curr;

    // --------------------------------------------------------
    // Part 3: Initialize circular buffer bookkeeping
    // --------------------------------------------------------

    int A_S_start = 0;
    int B_S_start = 0;

    int A_active = min_int(tile_size, A_length);
    int B_active = min_int(tile_size, B_length);

    int A_loaded = A_active;
    int B_loaded = B_active;

    int C_completed = 0;

    // --------------------------------------------------------
    // Part 4: Initial loading of shared memory
    // --------------------------------------------------------

    for (int i = threadIdx.x; i < A_active; i += blockDim.x)
    {
        A_S[i] = A[A_curr + i];
    }

    for (int i = threadIdx.x; i < B_active; i += blockDim.x)
    {
        B_S[i] = B[B_curr + i];
    }

    __syncthreads();

    // --------------------------------------------------------
    // Part 5: Iterative tiled merge
    // --------------------------------------------------------

    while (C_completed < C_length)
    {
        int current_tile_merged_size = min_int(
            tile_size,
            C_length - C_completed
        );

        // ----------------------------------------------------
        // Part 5A: Partition current output tile among threads
        // ----------------------------------------------------

        int c_curr =
            (current_tile_merged_size * threadIdx.x) / blockDim.x;

        int c_next =
            (current_tile_merged_size * (threadIdx.x + 1)) / blockDim.x;

        // ----------------------------------------------------
        // Part 5B: Calculate thread-level co-ranks
        // ----------------------------------------------------

        int a_curr = co_rank_circular(
            c_curr,
            A_S,
            A_active,
            B_S,
            B_active,
            A_S_start,
            B_S_start,
            tile_size
        );

        int b_curr = c_curr - a_curr;

        int a_next = co_rank_circular(
            c_next,
            A_S,
            A_active,
            B_S,
            B_active,
            A_S_start,
            B_S_start,
            tile_size
        );

        int b_next = c_next - a_next;

        // ----------------------------------------------------
        // Part 5C: Sequential merge by individual threads
        // ----------------------------------------------------

        merge_sequential_circular(
            A_S,
            a_next - a_curr,
            B_S,
            b_next - b_curr,
            C + C_curr + C_completed + c_curr,
            (A_S_start + a_curr) % tile_size,
            (B_S_start + b_curr) % tile_size,
            tile_size
        );

        __syncthreads();

        // ----------------------------------------------------
        // Part 5D: Determine consumed elements
        // ----------------------------------------------------

        int a_consumed = co_rank_circular(
            current_tile_merged_size,
            A_S,
            A_active,
            B_S,
            B_active,
            A_S_start,
            B_S_start,
            tile_size
        );

        int b_consumed =
            current_tile_merged_size - a_consumed;

        // ----------------------------------------------------
        // Part 5E: Advance circular buffer starting positions
        // ----------------------------------------------------

        A_S_start =
            (A_S_start + a_consumed) % tile_size;

        B_S_start =
            (B_S_start + b_consumed) % tile_size;

        // ----------------------------------------------------
        // Part 5F: Update active elements
        // ----------------------------------------------------

        A_active -= a_consumed;
        B_active -= b_consumed;

        // ----------------------------------------------------
        // Part 5G: Determine the number of elements to reload
        // ----------------------------------------------------

        int A_to_load = min_int(
            a_consumed,
            A_length - A_loaded
        );

        int B_to_load = min_int(
            b_consumed,
            B_length - B_loaded
        );

        // ----------------------------------------------------
        // Part 5H: Refill A circular buffer
        // ----------------------------------------------------

        for (int i = threadIdx.x; i < A_to_load; i += blockDim.x)
        {
            int index =
                (A_S_start + A_active + i) % tile_size;

            A_S[index] = A[A_curr + A_loaded + i];
        }

        // ----------------------------------------------------
        // Part 5I: Refill B circular buffer
        // ----------------------------------------------------

        for (int i = threadIdx.x; i < B_to_load; i += blockDim.x)
        {
            int index =
                (B_S_start + B_active + i) % tile_size;

            B_S[index] = B[B_curr + B_loaded + i];
        }

        __syncthreads();

        // ----------------------------------------------------
        // Part 5J: Update bookkeeping
        // ----------------------------------------------------

        A_active += A_to_load;
        B_active += B_to_load;

        A_loaded += A_to_load;
        B_loaded += B_to_load;

        C_completed += current_tile_merged_size;

        __syncthreads();
    }
}