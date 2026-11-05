
#include <cuda_runtime.h>

__device__ void merge_sequential_tiled(const int* A, int m, const int* B, int n, int* C)
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

__device__ int co_rank_tiled(int k, const int* A, int m, const int* B, int n)
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

__global__ void tiled_merge(int* A, int m, int* B, int n, int* C, int tile_size)
{
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];
    int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x), (m + n));

    if (threadIdx.x == 0)
    {
        A_S[0] = co_rank_tiled(C_curr, A, m, B, n);
        A_S[1] = co_rank_tiled(C_next, A, m, B, n);
    }

    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iters = (C_length + tile_size - 1) / tile_size;
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (counter < total_iters)
    {
        for (int i = 0; i < tile_size; i += blockDim.x)
        {
            if (i + threadIdx.x < A_length - A_consumed)
            {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < tile_size; i += blockDim.x)
        {
            if (i + threadIdx.x < B_length - B_consumed)
            {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        int a_curr = co_rank_tiled(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;

        int a_next = co_rank_tiled(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential_tiled(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);

        __syncthreads();

        int active_A_tile = min(tile_size, A_length - A_consumed);
        int active_B_tile = min(tile_size, B_length - B_consumed);
        int current_tile_merged_size = min(tile_size, C_length - C_completed);

        int a_consumed_this_step = co_rank_tiled(current_tile_merged_size, A_S, active_A_tile, B_S, active_B_tile);
        int b_consumed_this_step = current_tile_merged_size - a_consumed_this_step;

        C_completed += current_tile_merged_size;
        A_consumed += a_consumed_this_step;
        B_consumed += b_consumed_this_step;
        counter++;
        __syncthreads();
    }
}
