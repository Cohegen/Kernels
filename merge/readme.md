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

# Parralize approach
- Say we have two sorted shown below:
```
A = [1, 2, 3, 4, 5, 6]
B = [7, 8, 9, 10, 11, 12]
```
- And we're tasked to merge these two arrays in a fully sorted array **C**.

- Say we have threads $T_{0}$ and $T_{1}$.We partition our input arrays such that each thread has a dedicated region it's suppose to look after .In these scenario we can each thread 6 output elements in array **C**, meaning each thread will compare 3 elements from both arrays.
- $T_{0}$ will handle A[0]-A[2] and B[0] - B[2] while $T_{1}$ will handle A[3] to A[5] and B[3] to B[5]
- From $T_{0}$ C[0:2] is:
```
C = [1,2,3,7,8,9]
```
- From $T_{1}$ C[3-5] is:
```
C = [4,5,6,10,11,12]
```

- If we were to concatenate the sub-arrays of C we would get:
```
C = [1,2,3,7,8,9,4,5,6,10,11,12]
```

- As you can see the final array **C** isn't fully sorted. So partitioning our input arrays isn't a good strategy.
- We need a different strategy of merging the input arrays a good way to do this is to partition the output array.

- To identify which regions of the output array to partition we use something known as a co-rank function

## Co-rank function
- Given an array A containing `m` sorted elements.
- Array B containing `n` sorted elements.
- Array C will contain `m+n` merged elements.
- For any output boundary `k` we want to find two indices: $i$ and $j$.
- You can think of $k$ as the number which tells us how many output elements in C will each thread handle. $i$ and $j$ tells us how many elements of A and B respectively are needed to fill in $k$ spaces in array C.
- So from this we can deduce that $k=i+j$
