# Merge Algorithm

* The merge algorithm works as follows:

  1. It assumes that the two input arrays, **A** and **B**, are already sorted.
  2. If this condition is satisfied, it merges the two arrays into a single sorted output array **C**.

## Sorting Deep Dive

* Given two input arrays:

```cpp
A = [1, 3, 4, 5, 8]
B = [2, 5, 9, 34]
```

* We pass these two arrays into our sequential merge algorithm, shown below:

```cpp
void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0; // index into A
    int j = 0; // index into B
    int k = 0; // index into C

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    if (i == m) {
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}
```

* From the code above, we can see that the variables `i`, `j`, and `k` are used as indices into our input arrays **A** and **B**, and our output array **C**, respectively. Before any iteration takes place, they are all initialized to zero.

* At the core of the algorithm, during each iteration, we compare the elements at `A[i]` and `B[j]`. If `A[i]` is less than or equal to `B[j]`, the element `A[i]` is copied to `C`, and both `i` and `k` are incremented. Otherwise, `B[j]` is copied to `C`, and both `j` and `k` are incremented.

* For example:

```cpp
A[0] = 1
B[0] = 2
```

* Since `1 < 2`, the element `A[0]` (which is `1`) is copied to `C[0]`. Since `A[i]` is the element being copied to `C`, we increment `i` to point to the next element in **A**. We also increment `k` so that it points to `C[1]`, the position where the next element will be inserted.

* This means that we will no longer compare `A[0]` with elements from **B**. Instead, in the next iteration, we will compare `A[1]` with `B[0]`. This process continues until one of the input arrays has been fully processed.

* Suppose we reach a point where array **B** has been fully processed. Our remaining task will be to copy the remaining elements of **A** into **C**. This work is done by the final `while` loop in the `else` block.

* Since `n = 4`, which is the length of **B**, if `j == n`, it means array **B** has been fully processed. There are no more elements in **B** to compare with elements in **A**, so we copy the remaining elements of **A** into **C**. The same principle applies if **A** is fully processed first: we copy the remaining elements of **B** into **C**.

# Parallel Merge Approach

* Say we have two sorted arrays, as shown below:

```cpp
A = [1, 2, 3, 4, 5, 6]
B = [7, 8, 9, 10, 11, 12]
```

* We're tasked with merging these two arrays into a fully sorted array **C**.

* Say we have two threads, $T_{0}$ and $T_{1}$. We partition our input arrays such that each thread has a dedicated region it's supposed to handle. In this scenario, we can assign each thread 6 output elements in array **C**, meaning each thread will merge 3 elements from each input array.

* $T_{0}$ will handle A[0] to A[2] and B[0] to B[2], while $T_{1}$ will handle A[3] to A[5] and B[3] to B[5].

* From $T_{0}$, the first 6 output elements are:

```cpp
C = [1, 2, 3, 7, 8, 9]
```

* From $T_{1}$, the next 6 output elements are:

```cpp
C = [4, 5, 6, 10, 11, 12]
```

* If we were to concatenate these subarrays, we would get:

```cpp
C = [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]
```

* As you can see, the final array **C** isn't fully sorted. Therefore, partitioning our input arrays into equal-sized regions isn't a reliable strategy.

* We need a different strategy for merging the input arrays. A good approach is to partition the output array instead.

* To identify which regions of the input arrays correspond to each output partition, we use something known as a **co-rank function**.

## Co-Rank Function

* Given an array **A** containing `m` sorted elements.

* Array **B** contains `n` sorted elements.

* Array **C** will contain `m + n` merged elements.

* For any output boundary `k`, we want to find two indices: $i$ and $j$.

* You can think of $k$ as the number of output elements preceding a particular boundary in **C**. The values $i$ and $j$ tell us how many elements from **A** and **B**, respectively, are needed to fill those first $k$ positions.

* From this, we can deduce that:

$$
k = i + j
$$

* Given a rank `k = 5`, we're tasked with finding the co-ranks $i$ and $j$. In this example, we determine $k$ by dividing the total number of elements in our expected output array **C** by the number of threads, which is 2.

```cpp
k = 10 / 2 = 5
```

* Given the arrays below:

```cpp
A = [2, 5, 8, 12, 15]
B = [1, 3, 7, 10, 14]
```

* We're tasked with producing:

```cpp
C = [1, 2, 3, 5, 7, 8, 10, 12, 14, 15]
```

* Since `k = 5`, $T_{0}$ is responsible for producing C[0] to C[4], while $T_{1}$ is tasked with producing C[5] to C[9].

* Looking at the first five elements of **C**, we can see that we need the first two elements of **A** and the first three elements of **B**. This means that for $T_{0}$, `i = 2` and `j = 3`.

$$
k = i + j = 2 + 3 = 5
$$

* Therefore, for rank `k = 5`, the co-ranks are:

$$
(i,j) = (2,3)
$$

* But how do we determine the values of $i$ and $j$ in the first place?

* For a pair $(i,j)$ to be considered valid co-ranks, they must satisfy the following conditions:

```cpp
A[i - 1] <= B[j]
B[j - 1] < A[i]
```

* These conditions apply when the referenced indices are within bounds. The strict inequality in the second condition ensures consistent handling of duplicate elements by prioritizing elements from **A**.

* Say we're given `k = 5`. If we were to compute the possible co-rank combinations, we would get:

```cpp
(i,j) = (0,5), (1,4), (2,3), (3,2), (4,1), (5,0)
```

* Let's get our hands dirty and work through these combinations.

* For simplicity, we'll start with combinations where both indices are greater than zero. However, combinations containing zero are also valid candidates and must be handled by a complete co-rank algorithm.

### 1. Checking (i,j) = (1,4)

Using the conditions above:

```cpp
A[0] <= B[4]
2 <= 14 // true

B[3] < A[1]
10 < 5 // false
```

* Since the second condition is false, we eliminate `(1,4)`.

### 2. Checking (i,j) = (2,3)

```cpp
A[1] <= B[3]
5 <= 10 // true

B[2] < A[2]
7 < 8 // true
```

* Both conditions are satisfied, so we can use `(2,3)` as our co-ranks.

* There's no need to proceed with the remaining combinations since we've found the unique pair of co-ranks that satisfies both conditions.

* Therefore, for `k = 5`:

$$
\boxed{(i,j) = (2,3)}
$$

# Co-Rank Function

```cpp
int co_rank(int k, const int* A, int m, const int* B, int n)
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
```

```cpp
__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int totalElements = m + n;
    int totalThreads = blockDim.x * gridDim.x;

    // Ceiling division.
    int elementsPerThread = (totalElements + totalThreads - 1) / totalThreads;

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
```

* Given two arrays, as shown below:

```
A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 10]
```

* We're tasked with merging both arrays into our output array **C**.

* Assuming our block has 2 threads, meaning `blockDim.x = 2`, and our grid contains only one block, meaning `gridDim.x = 1`.

* The total number of threads will be `2 * 1 = 2`, whereas the total number of elements will be `m + n = 5 + 5 = 10`. Since both input arrays contain five elements, array **C** will contain 10 elements in total.

* Our `elementsPerThread` will be:

```
elementsPerThread = (10 + 2 - 1) / 2 = 5
```

* Therefore, `index` can either be `0` or `1`, since we only have two threads in our block.

* Let's use thread 0 ($T_{0}$) as our example to find the remaining variables.

### For $T_{0}$

```
k_curr = 0 * 5 = 0
k_next = min(0 + 5, 10) = 5
```

* We then call the `co_rank` function using `k_curr` and `k_next` as arguments to determine the values of `i_curr` and `i_next`, respectively.

```
i_curr = co_rank(0, A, 5, B, 5)
i_next = co_rank(5, A, 5, B, 5)
```

## Finding `i_curr`

* For this call:

```
k = 0
m = 5
n = 5
```

* Using our `co_rank` function above:

```
i = min(0, 5) = 0
j = 0 - 0 = 0
i_low = max(0, 0 - 5) = 0
j_low = max(0, 0 - 5) = 0
```

* Within the body of the `while` loop, neither the `if` nor the `else if` condition is satisfied. Therefore, we exit the loop and return `i`, which is `0`.

* Hence, `i_curr = 0`.

## Finding `i_next`

* For this call:

```
k = 5
m = 5
n = 5
```

* Based on how we've called the `co_rank` function:

```
i = min(5, 5) = 5
j = 5 - 5 = 0
i_low = max(0, 5 - 5) = 0
j_low = max(0, 5 - 5) = 0
```

* Within the `while` loop, the conditions in the `if` block are satisfied.

```
i = 5 > 0 -> True
j = 0 < 5 -> True
A[4] > B[0] -> 9 > 2 -> True

delta = (5 - 0 + 1) / 2 = 3

j_low = j = 0
j = j + delta = 0 + 3 = 3
i = i - delta = 5 - 3 = 2
```

* From the calculations above:

```
i = 2
j = 3
i_low = 0
j_low = 0
```

* In the second iteration, the conditions in the `else if` block are satisfied.

```
j = 3 > 0 -> True
i = 2 < 5 -> True
B[2] >= A[2] -> 6 >= 5 -> True

delta = (3 - 0 + 1) / 2 = 2

i_low = i = 2
i = i + delta = 2 + 2 = 4
j = j - delta = 3 - 2 = 1
```

* From the calculations above:

```
i = 4
j = 1
i_low = 2
j_low = 0
```

* In the third iteration, the conditions in the `if` block are satisfied.

```
delta = (4 - 2 + 1) / 2 = 1

j_low = j = 1
j = j + delta = 1 + 1 = 2
i = i - delta = 4 - 1 = 3
```

* From the calculations above:

```
i = 3
j = 2
i_low = 2
j_low = 1
```

* In the fourth iteration, neither the `if` nor the `else if` conditions are satisfied. Therefore, we exit the `while` loop.

* We can now conclude that `i_next = 3`.

## Finding `j_curr`

* From the code in the kernel:

```cpp
j_curr = k_curr - i_curr;
j_curr = 0 - 0 = 0;
```

## Finding `j_next`

* From the code in the kernel:

```cpp
j_next = k_next - i_next;
j_next = 5 - 3 = 2;
```

## Calling the Merge Function

* After obtaining all the necessary information for $T_{0}$, we call the `merge_sequential` function:

```cpp
merge_sequential(
    &A[i_curr],
    i_next - i_curr,
    &B[j_curr],
    j_next - j_curr,
    &C[k_curr]
);
```

* Which is equivalent to:

```cpp
merge_sequential(&A[0], 3, &B[0], 2, &C[0]);
```

* `&A[0]` is a pointer to the first element of array **A**, while `3` indicates that we're taking three elements from **A**.

* Similarly, `&B[0]` is a pointer to the first element of array **B**, while `2` indicates that we're taking the first two elements from **B**.

* Finally, `&C[0]` is a pointer to the first element of our output array **C**, where the merged elements will be written.

# Tiled Merge

- The basic merge kernel has a downside. When calling the co-rank and merge functions, threads may access data in global memory in an uncoalesced manner.
- To address this issue, we load the data into shared memory before performing the merge operations.
- However, let's say both of our arrays **A** and **B** contain 6000 elements each. Shared memory has limited capacity, meaning we might not be able to load all 12,000 elements into it at once.
- To overcome this limitation, we perform **tile-based loading**, where we load smaller chunks of data into shared memory and process them iteratively.

## Deep Dive

- Given the following metrics:

```
m = 6000
n = 6000
num_blocks = 4
num_threads_per_block = 128
tile_size = 1024
```

- We first declare our shared memory array, as shown below:

```cpp
extern __shared__ int shareAB[];
```

- Notice that we haven't explicitly specified its size. This is because we're using dynamically allocated shared memory. Its size is specified when launching the kernel and must not exceed the available shared memory limit per block.
- We also define some pointers, as shown below:

```cpp
int* A_S = &shareAB[0];
int* B_S = &shareAB[tile_size];
```

- The statements above point to the memory locations where we'll store elements from arrays **A** and **B**, respectively.
- Since `tile_size = 1024`, our shared memory array is logically divided into two regions:
  - `shareAB[0]` to `shareAB[1023]` will hold a tile of 1024 elements from **A**.
  - `shareAB[1024]` to `shareAB[2047]` will hold a tile of 1024 elements from **B**.
- This means we'll need `2048 * sizeof(int)` bytes of dynamically allocated shared memory per block.

## Block-Level Merge

- Each block is assigned a region in the output array **C** that it will be responsible for producing.
- So, for $block_{0}$ to determine which region of output array **C** it will be responsible for, we need the following statements:

```cpp
int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x);

int C_next = min(
    (blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x),
    (m + n)
);
```

- Let's get our hands dirty and calculate the values above.
- Since we're dealing with $block_{0}$:

```
blockIdx.x = 0
gridDim.x = 4 // Since we have 4 blocks.
```

- Therefore:

```
C_curr = 0 * ((6000 + 6000 + 4 - 1) / 4) = 0

C_next = min(
    (0 + 1) * ((6000 + 6000 + 4 - 1) / 4),
    12000
) = 3000
```

- So, `C_curr = 0` and `C_next = 3000`, meaning $block_{0}$ is responsible for writing the first 3000 elements of array **C**, corresponding to the range `[0, 3000)`.

- Now we need to determine which regions of the input arrays **A** and **B** $block_{0}$ will be responsible for merging.
- The code below helps us do that:

```cpp
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
```

- The thread with index 0, i.e., $T_{0}$, is responsible for calculating the co-rank values. Specifically, we're interested in finding out how many elements from **A** and **B** are needed to produce the output region assigned to our block.
- Notice that we're only storing the co-rank values from **A** in shared memory. We can obtain the corresponding co-rank values from **B** using the relationship:

$$
k = i + j
$$

- Let's get our hands dirty and do the maths.

For this example, let's assume the co-rank calculations produce the following results:

```
A_S[0] = 0
A_S[1] = 1000
```

- Note that these are hypothetical values. The actual co-ranks depend on the contents of arrays **A** and **B**.

- With the information above:

```
A_curr = A_S[0] = 0
A_next = A_S[1] = 1000

B_curr = C_curr - A_curr = 0 - 0 = 0
B_next = C_next - A_next = 3000 - 1000 = 2000
```

- From the calculations above, we conclude that $block_{0}$ needs the first 1000 elements from array **A** and the first 2000 elements from array **B**.
- In other words, our block has been assigned 3000 output elements, which will be produced by merging these 3000 input elements.

## How Many Iterations Will $block_{0}$ Need to Fully Write Its Output Elements?

- From the previous section, we established that $block_{0}$ is responsible for the output region `[0, 3000)`.
- However, the block can't load all 3000 elements into shared memory in one go.
- In our case, we have space for 1024 elements from **A** and another 1024 elements from **B** in shared memory.
- Since the chunk of **A** assigned to our block contains 1000 elements, we can load the entire chunk in one round because its size is less than `tile_size = 1024`.
- However, the chunk from **B** contains 2000 elements, which is greater than our tile size. Therefore:
  - Round 1 will load 1024 elements from **B**.
  - Round 2 will load the remaining 976 elements from **B**.
- But how do we determine the number of iterations needed in code?
- We use the following code snippet:

```cpp
int counter = 0;

int C_length = C_next - C_curr;
int A_length = A_next - A_curr;
int B_length = B_next - B_curr;

int total_iters = (C_length + tile_size - 1) / tile_size;

int C_completed = 0;
int A_consumed = 0;
int B_consumed = 0;
```

- The `counter` variable keeps track of the number of iterations we've completed so far.
- Now let's calculate the remaining variables:

```
C_length = C_next - C_curr
         = 3000 - 0
         = 3000

A_length = A_next - A_curr
         = 1000 - 0
         = 1000

B_length = B_next - B_curr
         = 2000 - 0
         = 2000

total_iters = (C_length + tile_size - 1) / tile_size

            = (3000 + 1024 - 1) / 1024

            = 4023 / 1024

            = 3
```

- This means we'll need three block-level iterations to produce all 3000 output elements.
- But what exactly do `C_completed`, `A_consumed`, and `B_consumed` represent?
  - `C_completed` keeps track of how many output elements we've written to **C** so far.
  - `A_consumed` keeps track of how many elements we've consumed from the assigned region of **A**.
  - `B_consumed` does the same for **B**.
- These variables help us keep track of our progress across iterations so that we don't process the same input elements repeatedly.

## Thread-Level Loading

- We've established that $block_{0}$ needs three iterations to produce its assigned output elements.
- Given the code below:

```cpp
while (counter < total_iters)
{
    for (int i = 0; i < tile_size; i += blockDim.x)
    {
        if (i + threadIdx.x < A_length - A_consumed)
        {
            A_S[i + threadIdx.x] =
                A[A_curr + A_consumed + i + threadIdx.x];
        }
    }

    for (int i = 0; i < tile_size; i += blockDim.x)
    {
        if (i + threadIdx.x < B_length - B_consumed)
        {
            B_S[i + threadIdx.x] =
                B[B_curr + B_consumed + i + threadIdx.x];
        }
    }

    __syncthreads();

    // Divide the output tile among threads.
    int c_curr = threadIdx.x * (tile_size / blockDim.x);
    int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

    c_curr = (c_curr <= C_length - C_completed)
        ? c_curr : C_length - C_completed;

    c_next = (c_next <= C_length - C_completed)
        ? c_next : C_length - C_completed;

    int a_curr = co_rank_tiled(
        c_curr,
        A_S,
        min(tile_size, A_length - A_consumed),
        B_S,
        min(tile_size, B_length - B_consumed)
    );

    int b_curr = c_curr - a_curr;

    int a_next = co_rank_tiled(
        c_next,
        A_S,
        min(tile_size, A_length - A_consumed),
        B_S,
        min(tile_size, B_length - B_consumed)
    );

    int b_next = c_next - a_next;

    merge_sequential_tiled(
        A_S + a_curr, a_next - a_curr,
        B_S + b_curr, b_next - b_curr,
        C + C_curr + C_completed + c_curr
    );

    __syncthreads();

    int active_A_tile = min(tile_size, A_length - A_consumed);
    int active_B_tile = min(tile_size, B_length - B_consumed);

    int current_tile_merged_size = min(
        tile_size,
        C_length - C_completed
    );

    int a_consumed_this_step = co_rank_tiled(
        current_tile_merged_size,
        A_S, active_A_tile,
        B_S, active_B_tile
    );

    int b_consumed_this_step =
        current_tile_merged_size - a_consumed_this_step;

    C_completed += current_tile_merged_size;
    A_consumed += a_consumed_this_step;
    B_consumed += b_consumed_this_step;

    counter++;

    __syncthreads();
}
```

- This is where we shift the responsibility of loading data into shared memory to the threads within our block.
- Recall that each block has 128 threads.
- Let's revisit the loading section:

```cpp
for (int i = 0; i < tile_size; i += blockDim.x)
{
    if (i + threadIdx.x < A_length - A_consumed)
    {
        A_S[i + threadIdx.x] =
            A[A_curr + A_consumed + i + threadIdx.x];
    }
}

for (int i = 0; i < tile_size; i += blockDim.x)
{
    if (i + threadIdx.x < B_length - B_consumed)
    {
        B_S[i + threadIdx.x] =
            B[B_curr + B_consumed + i + threadIdx.x];
    }
}

__syncthreads();
```

- Since `tile_size = 1024` and `blockDim.x = 128`, each thread can load up to:

$$
\frac{1024}{128} = 8
$$

- In other words, each thread performs up to eight loads for each input array, meaning we have eight rounds of loading at the thread level.
- Let's see how this works.

### Round 1

```
i = 0

A_length - A_consumed = 1000
```

- During this round:

```
Thread 0 loads A[0] into A_S[0]

...

Thread 127 loads A[127] into A_S[127]
```

- Notice that each thread is responsible for loading a different element into shared memory.
- During the second round, `i = 128`:

```
Thread 0 loads A[128] into A_S[128]

...

Thread 127 loads A[255] into A_S[255]
```

- This strategy continues until we've completed all eight rounds.
- However, remember that our **A** chunk only contains 1000 elements. Therefore, during the final round, some threads won't load anything because their corresponding indices exceed the number of remaining elements.
- The same process happens when loading elements from **B**.
- Finally, `__syncthreads()` ensures that all threads have finished loading their respective elements into shared memory before any thread starts accessing the tiles for merging.

## Thread-Level Merge

- Now that we've loaded our input tiles into shared memory, we need to divide the output tile among the threads in our block.
- For each thread, we assign a region in the output array **C** that it will be responsible for producing.
- We achieve this using the following code:

```cpp
// Divide the output tile among threads.
int c_curr = threadIdx.x * (tile_size / blockDim.x);
int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

c_curr = (c_curr <= C_length - C_completed)
    ? c_curr : C_length - C_completed;

c_next = (c_next <= C_length - C_completed)
    ? c_next : C_length - C_completed;
```

- So, for $T_{0}$:

```cpp
c_curr = 0 * (1024 / 128) = 0

c_next = (0 + 1) * (1024 / 128) = 8
```

- This means $T_{0}$ is responsible for producing 8 elements in the current output tile, corresponding to the range `[0, 8)`, or C[0] through C[7].
- After determining the region of array **C** that $T_{0}$ will be responsible for, we need to know which portions of **A** and **B** are required to produce those elements.
- To do so, we use the co-rank function again:

```cpp
int a_curr = co_rank_tiled(
    c_curr,
    A_S,
    min(tile_size, A_length - A_consumed),
    B_S,
    min(tile_size, B_length - B_consumed)
);

int b_curr = c_curr - a_curr;

int a_next = co_rank_tiled(
    c_next,
    A_S,
    min(tile_size, A_length - A_consumed),
    B_S,
    min(tile_size, B_length - B_consumed)
);

int b_next = c_next - a_next;
```

- From the calculations above, let's assume we obtain:

```
a_curr = 0
b_curr = 0 - 0 = 0

a_next = 3

b_next = 8 - 3 = 5
```

- Note that `a_next = 3` is an illustrative assumption. The actual value depends on the elements in the loaded tiles.
- We finally call the merge function as follows:

```cpp
merge_sequential_tiled(
    A_S + a_curr,
    a_next - a_curr,
    B_S + b_curr,
    b_next - b_curr,
    C + C_curr + C_completed + c_curr
);
```

- Which translates to:

```cpp
merge_sequential_tiled(
    shareAB + 0,
    3,
    shareAB + 1024,
    5,
    C + 0 + 0 + 0
);
```

- Meaning that $T_{0}$ will merge the first three elements of the **A** tile, `shareAB[0]` through `shareAB[2]`, with the first five elements of the **B** tile, `shareAB[1024]` through `shareAB[1028]`.
- The resulting eight elements will be written to C[0] through C[7].
- Note that I've used $T_{0}$ as my example. The other threads, from $T_{1}$ all the way to $T_{127}$, follow the same procedure for their respective output regions.

## Tracking the Consumed Elements

- Next, we need to know how many elements of **A** and **B** have been consumed during the current iteration.
- Why is this important?
- Remember that our block has to process 3000 output elements, but we're only loading tiles of up to 1024 elements from each input array into shared memory.
- Once we've merged the current tiles, we need to update our counters so that the next iteration loads the correct portions of **A** and **B**.
- To do so, we use:

```cpp
int active_A_tile = min(tile_size, A_length - A_consumed);
int active_B_tile = min(tile_size, B_length - B_consumed);

int current_tile_merged_size = min(
    tile_size,
    C_length - C_completed
);

int a_consumed_this_step = co_rank_tiled(
    current_tile_merged_size,
    A_S,
    active_A_tile,
    B_S,
    active_B_tile
);

int b_consumed_this_step =
    current_tile_merged_size - a_consumed_this_step;
```

- Let's substitute our current values:

```
active_A_tile = min(1024, 1000 - 0)
               = 1000

active_B_tile = min(1024, 2000 - 0)
               = 1024

current_tile_merged_size = min(1024, 3000 - 0)
                         = 1024
```

- Notice that we're determining how many elements from each input tile are available for merging, as well as how many output elements we can produce during the current iteration.
- We then use the co-rank function to determine how many elements from **A** are needed to produce the current output tile.
- Since the merged output tile contains 1024 elements, the remaining elements must come from **B**:

$$
b_{\text{consumed}} =
1024 - a_{\text{consumed}}
$$

- Finally, we update our counters:

```cpp
C_completed += current_tile_merged_size;

A_consumed += a_consumed_this_step;

B_consumed += b_consumed_this_step;
```

- This allows us to keep track of how many elements we've processed from each input array and how many output elements we've written so far.
- We then increment `counter` and proceed to the next iteration.

- The process continues until we've produced all 3000 output elements assigned to $block_{0}$.

# Circular Buffer Merge

- The tiled merge kernel has an inefficiency. Consider, for example, the shared memory arrays for $block_0$, `A_S` and `B_S`, shown below:

```cpp
A_S = {1, 3, 5, 7, 9, 11, 13, 17, 19}
B_S = {2, 4, 8, 10, 12, 14, 16, 18, 20}
```

- Suppose that $block_0$ has 2 threads, and each thread is responsible for merging 5 output elements.
- For $T_0$, the co-rank values `i` and `j` at the end of its output partition are 3 and 2, respectively. This means that it consumes 3 elements from `A_S` and 2 from `B_S`.
- For $T_1$, the co-rank values at the end of its output partition are 6 and 4, respectively. Since $T_0$ has already consumed 3 elements from `A_S` and 2 from `B_S`, $T_1$ consumes another 3 elements from `A_S` and 2 from `B_S`.
- Consequently, after the merging operation in round 1, we expect `C` to contain:

```cpp
C = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11}
```

- Notice that we have consumed 6 elements from `A_S` and 4 from `B_S`. Therefore, the remaining elements in our shared memory arrays are:

```cpp
A_S = {x, x, x, x, x, x, 13, 17, 19}
B_S = {x, x, x, x, 12, 14, 16, 18, 20}
```

- In a conventional tiled merge kernel, the next iteration loads another set of input elements into shared memory.
- However, notice that some of the elements that were loaded into shared memory during the previous iteration were never consumed. These elements are still available in the original input arrays in global memory.
- Consequently, when the next iteration loads its input tiles, it may load these elements into shared memory again, even though they were already loaded during the previous iteration.
- This results in unnecessary global memory traffic.

### Introducing Circular Buffers

- We need a way to solve this issue, and this is where **circular buffers** come in.
- Instead of discarding the unconsumed elements and loading them again, what if we could preserve them in shared memory?
- We can achieve this by keeping track of the number of elements consumed from `A_S` and `B_S`.
- We then use these consumption counts to advance the logical starting positions of the respective buffers.
- The positions occupied by consumed elements become available for new elements loaded from global memory.
- This allows us to load only the new elements we need while preserving the elements that were already loaded but haven't been consumed.
- This process of consuming elements, advancing the buffer positions, and refilling the available slots continues throughout the merge operation.

To achieve this, we use the following code:

```cpp
A_S_start =
    (A_S_start + a_consumed) % tile_size;

B_S_start =
    (B_S_start + b_consumed) % tile_size;
```

- In our example, we use a tile size of 9 for both `A_S` and `B_S`.
- Initially, both buffer starting positions are 0.
- After the first iteration, we have:

```cpp
a_consumed = 6;
b_consumed = 4;
```

- Therefore, to obtain the new starting positions, we do:

```cpp
A_S_start = (0 + 6) % 9 = 6;
B_S_start = (0 + 4) % 9 = 4;
```

- Notice that we are not physically shifting the elements inside our shared memory arrays. Instead, we are changing the logical starting positions. This is what makes the buffers circular.

### Refilling the Buffers

- After advancing the starting positions, we need to determine how many new elements we can load into the available slots.
- We use the following code:

```cpp
int A_to_load = min_int(
    a_consumed,
    A_length - A_loaded
);

int B_to_load = min_int(
    b_consumed,
    B_length - B_loaded
);
```

- Here:
  - `A_length` and `B_length` represent the total number of elements in the original input arrays.
  - `A_loaded` and `B_loaded` represent the cumulative number of elements loaded from global memory.
  - `a_consumed` and `b_consumed` represent the number of slots that have become available in the respective shared memory buffers.
- So far, our example has used arrays containing only 9 elements each. This means that we have already loaded all their elements into shared memory.
- However, real-world merge operations can involve much larger arrays. Let's consider what happens when our original arrays contain 30 elements each.

Suppose the original arrays are:

```cpp
A = {1, 3, 5, 7, 9, 11, 13, 17, 19,
     21, 23, 25, 27, 29, 31, 33, 35, 37,
     39, 41, 43, 45, 47, 49, 51, 53, 55,
     57, 59, 61};

B = {2, 4, 8, 10, 12, 14, 16, 18, 20,
     22, 24, 26, 28, 30, 32, 34, 36, 38,
     40, 42, 44, 46, 48, 50, 52, 54,
     56, 58, 60, 62};
```

- Since our tile size is 9, we initially load only the first 9 elements from each array into shared memory.
- After the first merge iteration, we have consumed 6 elements from `A_S` and 4 from `B_S`.
- This leaves us with 3 unconsumed elements in `A_S` and 5 unconsumed elements in `B_S`.
- Consequently, we have 6 available slots in `A_S` and 4 available slots in `B_S`.
- We can now load new elements from global memory into these available slots.

Using our previous code:

```cpp
A_length = 30;
B_length = 30;

A_loaded = 9;
B_loaded = 9;

A_to_load = min(6, 30 - 9) = 6;
B_to_load = min(4, 30 - 9) = 4;
```

- This means that we load 6 new elements from `A` and 4 new elements from `B`.
- These elements are loaded into the slots that were previously occupied by consumed elements.
- Notice that we do not reload the elements that were already present in shared memory but weren't consumed.

Our shared memory buffers now logically contain:

```cpp
A_S = {13, 17, 19, 21, 23, 25, 27, 29, 31};

B_S = {12, 14, 16, 18, 20, 22, 24, 26, 28};
```

- We now have 9 elements available from each input array for the next merge iteration.
- The next iteration can therefore continue merging from where the previous iteration stopped, without having to reload the unconsumed elements.
- We repeat this process of merging, advancing the buffer starting positions, and refilling available slots until the block has produced all its assigned output elements.

