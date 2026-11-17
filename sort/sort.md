# Intro to Sorting

- In this directory, we'll dive into sorting algorithms. More specifically, we'll explore Radix Sort.

## Deep Dive into Naive Radix Sort

- To understand how Radix Sort works, let's use a small example.
- Given the array below:

```
A = [4,3,2,7,6,1]
```

- If we were using Bubble Sort, we would compare two adjacent elements.
- For example, in our first pass, 4 is greater than 3, so we would need to swap them, giving us the array below:

```
A = [3,4,2,7,6,1]
```

- The array above is obtained after the first comparison and swap.
- However, Radix Sort works differently. Instead of comparing adjacent elements, we sort our array based on keys and values.
- But first, let's represent our array elements in their binary forms:

```
A = [100,011,010,111,110,001]
```

- Let's introduce two keys, known as **Bucket 0** and **Bucket 1**.
- Bucket 0 represents the category containing elements whose selected bit is 0, while Bucket 1 contains elements whose selected bit is 1.

## Sorting Process

- Let's begin our sorting process.

### Round 1

- In this round, we will examine the bit at position 0 of each binary number.
- Bucket 0 will contain elements whose bit at position 0 is 0:

```
bucket 0 = [100,010,110]
```

- On the other hand, Bucket 1 will contain elements whose bit at position 0 is 1:

```
bucket 1 = [011,111,001]
```

- Our final sorted array will be:

```
out = [100,010,110,011,111,001]
```

- Notice that we maintain the relative order of the elements within each bucket. Elements belonging to Bucket 0 also come before those belonging to Bucket 1.

### Round 2

- This round, like all subsequent rounds, depends on the output of the preceding round to determine its input.
- In this scenario, Round 2 depends on the output of Round 1.
- Therefore, our input is:

```
out = [100,010,110,011,111,001]
```

- However, in this round, we will examine the bit at position 1 of every binary number and classify the elements into their respective buckets.
- Bucket 0:

```
bucket 0 = [100,001]
```

- And Bucket 1:

```
bucket 1 = [010,110,011,111]
```

- Finally, our output in this round is:

```
out = [100,001,010,110,011,111]
```

- We maintain the same ordering principle: elements belonging to Bucket 0 come before those belonging to Bucket 1, while preserving their relative order within each bucket.

### Round 3

- The input of this round is the output of the previous round.
- Therefore, our input is:

```
input = [100,001,010,110,011,111]
```

- In this round, we use the bit at position 2 as our sorting factor.
- Bucket 0:

```
bucket 0 = [001,010,011]
```

- Bucket 1:

```
bucket 1 = [100,110,111]
```

- Our output therefore becomes:

```
out = [001,010,011,100,110,111]
```

- Finally, after converting the binary numbers back into their decimal representations, we get:

```
out = [1,2,3,4,6,7]
```
## Parallel Radix Sort

- Is there a way we can parallelize the work performed during Radix Sort?
- Yes, there is! In this section, we'll dive deep into Parallel Radix Sort.

### Deep Dive

- Given our input array:

```cpp
A = [4,3,2,7,6,1]
```

- Our input array in binary form becomes:

```cpp
A = [100,011,010,111,110,001]
```

- Assuming our thread block has 6 threads, i.e., the same number of threads as the number of input elements.
- We assign each thread to a respective input element.

- Therefore:

```cpp
thread 0 -> 4
thread 1 -> 3
thread 2 -> 2
thread 3 -> 7
thread 4 -> 6
thread 5 -> 1
```

- Each thread will be responsible for the following:

  1. Loading its assigned key.
  2. Extracting the relevant bit.
  3. Determining which bucket the key belongs to.
  4. Determining its destination index.
  5. Writing the key to the output array.

- The first three steps seem straightforward.
- However, how does an individual thread know where its destination index is?
- For example, consider the element at index 0, which is 4. We want to determine the index at which it will be written in the output array after the first round.

- Since the LSB (Least Significant Bit) of 4 is 0, its destination index is calculated using the following formula:

```cpp
Dest_idx(0) = index - OnesBefore
```

- In our case, the index of 4 is 0, and there are no elements with an LSB of 1 before it. Therefore, its destination index in the output array will be:

```cpp
Dest_idx(0) = 0 - 0 = 0
```

- But what about elements whose LSB is 1? What will their destination indices be in the output array?
- To determine this, we use the formula below:

```cpp
Dest_idx(1) = ZeroTotal + OnesBefore

where:
ZeroTotal = inputSize - OnesTotal
```

- For example, consider element 7, which is at index 3.
- There are 3 elements whose LSB is 0, meaning `ZeroTotal = 3`.
- Also, element 3 is the only element before 7 whose LSB is 1, meaning `OnesBefore = 1`.
- Therefore, our destination index is:

```cpp
Dest_idx(1) = 3 + 1 = 4
```

- This whole process might look sequential, which isn't what we expected when we set out to parallelize Radix Sort.
- Now, let's get into the parallel part! :)

## Parallelization Process

- From the previous section, we established that each thread needs to perform the following operations:

  - Load its assigned key.
  - Extract the relevant bit.
  - Determine which bucket the key belongs to.
  - Determine its destination index.
  - Write the key to the output array.

- But how does each thread know, for example, the number of ones preceding its assigned element?
- To solve this problem, we perform an **exclusive scan** on the extracted bits.

- In the first round, the extracted bits are:

```cpp
bits = [0,1,0,1,0,1]
```

- We then perform an exclusive scan on these bits to obtain:

```cpp
scan = [0,0,1,1,2,2]
```

- The values in the `scan` array represent the number of ones preceding each element.
- From this, we can deduce the total number of ones using the following formula:

```cpp
OnesTotal = scan[n-1] + bits[n-1]

where n is the number of elements in our input array.

OnesTotal = 2 + 1 = 3
```

- Now, let's calculate the destination indices for each element.

**Element at index 0:**

```cpp
key = 4
bit = 0
OnesBefore = 0

Dest = 0 - 0 = 0
```

**Element at index 1:**

```cpp
key = 3
bit = 1
OnesBefore = 0

Dest = 6 - 3 + 0 = 3
```

**Element at index 2:**

```cpp
key = 2
bit = 0
OnesBefore = 1

Dest = 2 - 1 = 1
```

**Element at index 3:**

```cpp
key = 7
bit = 1
OnesBefore = 1

Dest = 3 + 1 = 4
```

**Element at index 4:**

```cpp
key = 6
bit = 0
OnesBefore = 2

Dest = 4 - 2 = 2
```

**Element at index 5:**

```cpp
key = 1
bit = 1
OnesBefore = 2

Dest = 6 - 3 + 2 = 5
```
