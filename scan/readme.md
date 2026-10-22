# Introduction: Inclusive Scan

* An inclusive scan is a parallel algorithm that calculates the prefix sum of an array, including the current element in each sum.

## Example

Say we have an input array:

```
X = [2,4,6,8,10]
```

* An inclusive scan produces:

```
Y = [2,6,12,20,30]
```

* To get the various inclusive sums in our array, we do:

  ```
  Y[0] = X[0]
  Y[1] = X[0] + X[1]
  Y[2] = X[0] + X[1] + X[2]
  Y[3] = X[0] + X[1] + X[2] + X[3]
  Y[4] = X[0] + X[1] + X[2] + X[3] + X[4]
  ```

* To implement a sequential scan, we use the following algorithm in the C++ code below:

```
void sequential_scan(float* x, float* y, unsigned int N)
{
    if (N == 0) return;

    y[0] = x[0];

    for (unsigned int i = 1; i < N; ++i)
    {
        y[i] = y[i-1] + x[i];
    }
}
```

* From the code above, we can see that each position contains the current element and the inclusive sum of the previous elements.
* Take **Y[3]**, for example. It contains the sum of **X[3]** and **Y[2]**, where **Y[2]** contains the sum of **X[0], X[1], and X[2]**.

What if, instead of using a naive sequential scan algorithm, we could use parallel algorithms implemented through kernels running on a GPU? Well, this is possible.

In this directory, the following kernels have been implemented, and we're going to explore them in depth.

These kernels include:

* The Kogge–Stone algorithm
* The Brent–Kung algorithm
* The Blelloch scan algorithm (this is commonly used for exclusive scans)
* A coarsened scan

# Kogge–Stone Algorithm

* This algorithm uses an approach whereby we compute the prefix sums of an array in a tree-like structure.
* We implement it using the kernel in **kogge_stone.cu**.
* Let's look at the main skeleton/algorithm of the kernel.

```
__global__ void Kogge_Stone_scan(float* X, float* Y, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        XY[threadIdx.x] = X[i];
    }
    else
    {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp = 0.0f;

        if (threadIdx.x >= stride)
        {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }

        __syncthreads();

        if (threadIdx.x >= stride)
        {
            XY[threadIdx.x] = temp;
        }
    }

    if (i < N)
    {
        Y[i] = XY[threadIdx.x];
    }
}
```

From the kernel above, we first create an array called **XY**, which is located in shared memory. Its main role is to store the partial sums after each stage completes its respective additions.

But what are these stages, and where do they come from? Well, assume we have a **blockDim.x** of 8, meaning our thread block contains eight threads. From the `for` loop, we can see that the stride doubles after every round/stage. Since we set the condition that the stride must be less than **blockDim.x**, the stride will go from 1 to 2 and then to 4. Counting the stages, we have three.

Given an array:

```
X = [1,2,3,4,5,6,7,8]
```

When stride = 1:

```
XY[0] = X[0]
XY[1] = X[0] + X[1]
XY[2] = X[1] + X[2]
...
XY[7] = X[6] + X[7]
```

When stride = 2:

```
XY[2] = XY[2] + XY[0]
XY[3] = XY[3] + XY[1]
...
XY[7] = XY[7] + XY[5]
```

When stride = 4:

```
XY[4] = XY[4] + XY[0]
XY[5] = XY[5] + XY[1]
XY[6] = XY[6] + XY[2]
XY[7] = XY[7] + XY[3]
```

From the above, we can see that when stride = 1 (stage 1), we have 7 additions; when stride = 2 (stage 2), we have 6 additions; and finally, when stride = 4 (stage 3), we have 4 additions. This brings the total number of additions to:

$$
7 + 6 + 4 = 17
$$

So, we conclude that the work done, \(W\), by the Kogge–Stone scan for eight elements is 17 additions. To complete this work, we need three stages, meaning the parallel depth is:

$$
D(N) = \log_2 N
$$

where \(N\) is the number of elements, assuming \(N\) is a power of two.

We can clearly see that the Kogge–Stone approach does a lot of work. What if there were another approach we could use to perform an inclusive scan with less work?

This is where the Brent–Kung scan algorithm comes in.

# Brent–Kung Scan Algorithm

* The Brent–Kung algorithm differs from the Kogge–Stone algorithm in that it first performs a sum reduction of our input array and then performs a reverse tree traversal to calculate the remaining prefix sums.

## How Does It Work?

* Given the input array below:

```
X = [1,2,3,4,5,6,7,8]
```

* We first perform the sum reduction as follows.

### Round 1

```
X[1] = X[0] + X[1]
X[3] = X[2] + X[3]
X[5] = X[4] + X[5]
X[7] = X[6] + X[7]
```

### Round 2

```
X[3] = X[1] + X[3]
X[7] = X[5] + X[7]
```

### Round 3

```
X[7] = X[3] + X[7]
```

* Then, we perform the reverse tree traversal.
* To do this, we use the algorithm below:

```cpp
// Reverse tree
for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
{
    __syncthreads();

    unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;

    if (index + stride < SECTION_SIZE)
    {
        XY[index + stride] += XY[index];
    }
}
```

* Since we have eight elements, our **SECTION_SIZE** will be 8, meaning that in our first iteration, **stride** will be 2. The following elements will therefore be handled by certain threads.
* Let's go deeper to see which elements our respective threads will be responsible for.

### Iteration 1: stride = 2

```
Thread index = 0:
Element index = (0 + 1) * 2 * 2 - 1 = 3

Therefore:
XY[5] = XY[5] + XY[3]
XY[5] = 11 + 10 = 21
```

```
Thread index = 1:
Element index = (1 + 1) * 2 * 2 - 1 = 7

Since index + stride = 7 + 2 = 9 >= SECTION_SIZE,
the condition is false, so no addition is performed.
```

### Iteration 2: stride = 1

```
Thread index = 0:
Element index = (0 + 1) * 1 * 2 - 1 = 1

Therefore:
XY[2] = XY[2] + XY[1]
XY[2] = 3 + 3 = 6
```

```
Thread index = 1:
Element index = (1 + 1) * 1 * 2 - 1 = 3

Therefore:
XY[4] = XY[4] + XY[3]
XY[4] = 5 + 10 = 15
```

```
Thread index = 2:
Element index = (2 + 1) * 1 * 2 - 1 = 5

Therefore:
XY[6] = XY[6] + XY[5]
XY[6] = 7 + 21 = 28
```

```
Thread index = 3:
Element index = (3 + 1) * 1 * 2 - 1 = 7

Since index + stride = 7 + 1 = 8 >= SECTION_SIZE,
the condition is false, so no addition is performed.
```

The resulting array is:

```
XY = [1,3,6,10,15,21,28,36]
```
- From above we can than in the down sweep (reduction phase) we perform 7 additions while in the upsweep we perform 4 additions thus bring our total additions work done ($W$) to 7+4 = 11 additions.
- Thus we can see that the Brent-Kung addition performs much less work than the Koggle-Stone.
- For an array of (N) elements, where (N) is a power of two, Brent–Kung performs:


$W(N)$ = $2N - \log_2 N - 2$


Therefore:


$W(N) = O(N)$


The number of parallel stages is:


$D(N)$ = $2\log_2 N - 1$

# Coarsened Scan

* Coarsened scan is divided into three phases.

## Phase 1

* Say we're given the array below:

```
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
```

* Here, we have each thread perform a sequential scan on its contiguous subsection.
* In the case of our array, let's have four threads in our block. We partition our 16-element array into 4 subsections with 4 elements each.
* Thread 0 will perform a scan on its section **(1,2,3,4)** and generate **(1,3,6,10)**.
* Thread 1 will perform a scan on its section **(5,6,7,8)** and generate **(5,11,18,26)**.
* Thread 2 will perform a scan on its section **(9,10,11,12)** and generate **(9,19,30,42)**.
* Thread 3 will perform a scan on its section **(13,14,15,16)** and generate **(13,27,42,58)**.
* If each thread directly performs a scan by accessing the input elements from global memory, their accesses would not be coalesced. Say thread 0 accesses input element 0, thread 1 accesses input element 4, and so on. These accesses are not coalesced since the elements being accessed initially are not contiguous.
* To solve this problem of uncoalesced accesses, instead of directly accessing elements from global memory, we have all threads collaboratively load the input elements into shared memory. This is done in a coalesced manner.
* Once all input elements are in shared memory, the threads access their own subsections from shared memory and perform a sequential scan on them.

## Phase 2

* During this phase, all threads in each block collaborate to perform a scan operation on a logical array consisting of the last element of each subsection.
* This can be done using either the Kogge–Stone or Brent–Kung algorithm.
* After doing this, our subsections become:

  ```
  a = [1,3,6,10]
  b = [5,11,18,36]
  c = [9,19,30,78]
  d = [13,27,42,136]
  ```
* We can see that the last elements of subsections b to d are cumulative sums of the last elements of the preceding subsections, including their own last elements.

## Phase 3

* In this phase, each thread adds the new value of the last element of its predecessor's subsection to its elements.

* The last elements of each subsection do not need to be updated during this phase.

* So our subsections become:

  ```
  a = [1,3,6,10]
  b = [15,21,28,36]
  c = [45,55,66,78]
  d = [91,105,120,136]
  ```

* With this three-phase approach, we can use a much smaller number of threads than the number of elements in a section. In our case, we've used 4 threads instead of 16 threads.

# Segmented Parallel Scan for Arbitrary-Length Inputs

* In cases where we have a large dataset to scan, we need a different strategy from the ones we've used above.

* We first partition the input into sections so that each section can fit into the shared memory of a streaming multiprocessor and be processed by a single thread block.

* By calling the Kogge–Stone or Brent–Kung kernels on this large dataset, at the end of the grid execution, the Y array (output array) will contain the scan results for individual sections, called **scan blocks**.

* Each element in a scan block only contains the accumulated values of all preceding elements in the same scan block, including itself. These scan blocks need to be combined to produce the final results.

* To do this, we can use another kernel that adds the sums of all elements in the preceding scan blocks to each element of the current scan block.

## Example

* Given a 16-element input array:

```
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
```

* We divide X into four scan blocks.

* We can use either the Kogge–Stone, Brent–Kung, or coarsened scan kernels to process the individual scan blocks.

* These kernels treat the four scan blocks as independent input datasets.

* After the chosen kernel terminates, each Y element contains the scan result within its scan block. For example, scan block 1 contains **(5,11,18,26)**. These elements do not include any contribution from the preceding scan block 0. To produce the final results for this scan block, the sum of all input elements in block 0, which is 1+2+3+4=10, is added to every result element in scan block 1.

* In the case of scan block 2, it needs the sum of all elements in block 0 (1+2+3+4=10) and block 1 (5+6+7+8=26), which is 10+26=36. This value is then added to every element in scan block 2.

* It's important to know that the last output element of each scan block gives the sum of all input elements in that scan block. In our case, these values are **(10,26,42,58)**. This brings us to the second step of the segmented scan algorithm, which gathers the last result elements from each scan block into an array and performs a scan on these elements.

* After this, the new array becomes **(10,36,78,136)**.

* The last step of the segmented scan is to add the appropriate accumulated sums to the elements of their corresponding scan blocks. In this case, **36 is added to every element in scan block 2 except the last element if it has already been updated**, thus giving us **(45,55,66,78)**.

# Single pass scan
- Instead of launching three seperate kernels like in the case of the segmented scan kernel, we can perform the entire operation using a single kernel.
- The main challenge here is that, CUDA thread blocks cannot normally synchronize with one another using **__syncthreads()** like in the case of how individual threads synchronize with other threads in the block they are in.

## Recap of the three-kernel Segmented Scan
- Given the following array below:
```
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
```
- Assuming each thread block processes four elements.
- The array will be partitioned into four sections
```
block 0 : [1,2,3,4]
block 1: [5,6,7,8]
block 2: [9,10,11,12]
block 3: [13,14,15,16]
```
- The main goal is to end up with the array below after inclusive scan:
```
Y = [1,3,6,10,15,21,28,28,36,45,55,66,78,91,105,120,136]
```
### Phase 1 :Local scan
- Each block independently performs a parallel scan on its assigned section.
- By using either Koggle-Stone or Brent-Kung
```
block 0 : [1,3,6,10]
block 1: [5,11,18,26]
block 2: [9,19,30,42]
block 3: [13,27,42,58]
```
- The sums are stored in a seperate array:
  ```
  S = [10,26,42,58]
  ```
- The problem here is that the partially scanned results must be written to global memory because the first kernel is about to finish

### Phase 2:Global scan
- The second kernel performs a scan on the block sums.
  ```
  block_sums = S[10,26,42,58]
  becomes:
    S = [10,36,78,136]
  ```
  - After this the we have the cumulative sums of all preceding blocks
  ```
  block 0 needs an offset of 0 since it doesn't a preceding block
  block 1 needs an offset of 10 from block 0
  block 2 needs an offset of 36 from block 1
  ```
### Phase 3: Add the offsets
- A third kernel reads the partially scanned results from the global memory.
- Each bock adds it corresponding offset
  ```
  block 0 : [1,3,6,10]+0 = [1,3,6,10]
  block 1 : [5,11,18,26]+10 = [15,21,28,36]
  block 2: [9,19,30,42] + 36 = [45,55,66,78]
  block 3: [13,27,42,58] + 78 = [91,105,120,136]
  ```

- After doing this we end up with the desired output inclusive scan

##  Performance consideration
- Kernel 1 follows this approach
```
Global Memory
      |
      | Read input
      v
   Thread Block
      |
      | Local scan
      v
Global Memory
      |
      | Write partial results
      v
```
- Here we see that there both 1 phase global memory reads and writes.

- Kernel 2 follows this approach :
 ```
 Read block sums
       |
       v
Scan block sums
       |
       v
Write scanned block sums
  ```
- There is 1 global memory read and write.

- Kernel 3 follows this approach:
 ```
  Global Memory
      |
      | Read partial results again
      v
   Thread Block
      |
      | Add offsets
      v
Global Memory
      |
      | Write final results
      v
  ```
- it also the same number of reads and writes as kernel one and two.
- This whole process of accessing global memory can disadvatageus since these reads and write take time due to global memory traffic and it's expensive.
- Also launching these three kernels results to **kernel-launch** overhead.

## Solution
To solve these whole issue of global memory accesses so as to improve performance we do the use a **single-pass scan**. Where instead of launching three kernels we can just lauch one kernel , we can launch a kernel whereby each thread block receives the cumulative sum its predecessor and passes an updated sum to its succcessor.

### Step 1: All blocks perform their local sums
- The blocks perform their can independetly resulting to:
```
Block 0             Block 1             Block 2

[1, 3, 6, 10]      [5, 11, 18, 26]     [9, 19, 30, 42]

Local sum = 10      Local sum = 26      Local sum = 42
```
- Here each block uses shared memory to compute its local scan.
- The main difference here is that the partially scan results remain in the shared memory while the blocks determine their offsets.

### Step 2: Block i passes its sum to block j
- Block 0 doesn't need an offset because no elements precede it.
- Thus:
```
Block 0:

Offset = 0

Local sum = 10

Cumulative sum = 0 + 10 = 10
```
- Block 0 passes it cumulative sum to block 1
  ```
  block0 ----> 10 ---> block 1
  ```
- Block 1 passes it cumulative sum to block 2.
- It first receives 10 from block 0
- It uses this value in two ways as shown below:
```
previous sum =10
local sum = 26
cumulative sum = 10+26 = 36

```
- It passes 36 to block 2
```
block0 --->10-----> block 1 ----> 36 ----> block 2
```
- It adds the received offset to it's local scan resukts
```
Block 0:

Offset = 0

Local sum = 10

Cumulative sum = 0 + 10 = 10
```
- continue with the rest

## Communication among threadblocks
- In CUDA threads synchronize with **__syncthreads()**. Thus threads within block 0 can sychronize with each other.
- However, how can threads in block 1 synchronize with threads in block 0?
- This is done **adjacent synchronization**.
- This technique uses global memory and atomic operations to establish communication between neighboring blocks.

### Adjacent synchronization using flags
- Here two global arrays are needed, namely
```
int* scan_value
int* flags
```
- **scan_value** stores cumulative sums passes between blocks
```
 scan_value[0] = 0
scan_value[1] = 10
scan_value[2] = 36
scan_value[3] = 78
scan_value[4] = 136
 ```

- **flags** - indicates whether a cumulative sum is ready to consumed by the successor block
  ```
  flags = [1, 0, 0, 0]
  ```
- **flags[0]** is intialized to 1 since block 0's initial offset is zero and is already available.
- The remaining flags are zero because their corresponding values haven't been produced.
