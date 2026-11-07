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

# Tiled merge
- The merge kernel has it's downside namely when calling the co-rank and merge functions, we're accessing data in memory in an uncoalesced manner.
- To fix this issue we loading the data into shared memory.
- However, let's say both of our arrays **A** and **B** have  6000 elements. The shared memory may have a limited memory capacity such that it can accomodate say at most 2048 elements. Facing this obstacle we perform tile-based loading of data into shared memory

## Deep dive
- Given the following metrics:
```
m=6000
n=6000
num_blocks = 4
num_threads_per_block = 128
tile_size=1024
```
- We first declare out shared memory array which as shown below:
```cpp
  extern __shared__ int shareAB[];
```
- Note we haven't specified its size meaning it can be of any size but it's size must be less or equal to the maximum amount not more .
- We also define some statements like shown below:
```cpp
int* A_S = &shareAB[0];
int* B_S = &shareAB[tile_size];
```
- The statements above are points to addresses of locations of where the first elements of arrays **A** and  **B** are.
- So from `shareAB[0]-sharedAB[1023]` a chunk of 1024 elements from A are copied there whereas from `shareAB[tile_size]-shareAB[N]` a chunk of 1024 elements from B are copied there.

## Block-level merge
- Each block is assigned a region in the output array **C** it will responsible for.
- So for $block_{0}$ to find out which region of output array **C** it will responsible for, we need these statements:
```cpp
int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x);
int C_next = min((blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x), (m + n));
```
- Let's get our hands dirty and calculate the values above.
- Since for $block_{0}$
```
blockIdx.x = 0
gridDim.x = 4 .. since we have 4 threadblocks
```
- So:
```
C_curr = 0*((6000+6000)+4-1)/4) = 0
C_next = min((0+1)*(6000+6000+4-1)/gridDim.x),(6000+6000) = 3000
```
- So `C_curr=0` and `C_next=3000`, meaning $block_{0}$ is responsible for writing the first 3000 elements of array **C**.

- Now we need to know which regions of shared memory subarrays `A_S` and `B_S`, $block_{0}$ will be responsible for:
- The code below helps us to do so:
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
- The thread with index 0 i.e $T_{0}$ in the if block will responsible for calculating the co-rank values $i$ and $j$ so that we'll know what amount of elements from **A**, we'll use:
- Let's get our hands dirty and do the maths
```
A_S[0] = 0
A_S[1] = 1000 .. Note that this is a wild guess since A_S[1] is dependant on how the array A's elements look like
```
- With the information above:
```
A_curr = A_S[0] = 0
A_next = A_S[1] = 1000
B_curr = C_curr- A_curr = 0-0= 0
B_next = C_next-A_next = 3000-1000=2000
```
- From the data above we conclude that $block_{0}$ needs the first 1000 elements from array **A** and  needs the first 3000 elements from array **B**.

## How many Iteration will $block_0$ need to fully write it's output elements?
