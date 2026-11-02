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

