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
## Parallel Radix
- Is there a way in which we parallel like the work done in Radix sort?
- Yes there is, here we'll dive deep into parallel radix sort.

### Deep dive
- Given our input array:
```
A= [4,3,2,7,6,1]
```
- Our input array in binary form becomes:
```
A = [100,011,010,111,110,001]

```
- Assuming that our threadblock has 6 threads i.e the same number as our input element.
- We assign each thread to a respective input element.
- So:
```
thread 0 -> 4
thread 1 -> 3
thread 2-> 2
thread 3 -> 7
thread 4-> 6
thread 5->1
```
- Each thread will be responsible for:
      1. Loading its assigned key.
      2. Extracts the relevant bit.
      3. Determines which bucket the key belongs to.
      4. Determines its destination index
      5. Writes the key to the output array

- The first three steps seem straighforward.
- However, how does an individual thread know where the destination index is?
- For example for element at index 0 which is 4. we want to know at which index after the first round it will be written to in the output array.
- Since the LSB of 4 is zero, for any element in round 1 whose LSB is zero. It's destination index is:
```
Dest_idx(0) = index - OnesBefore
```
- In our case the index of 4 is 0 and there are no ones before it so its index in the output array will be:
```
Dest_idx(0) = 0-0 =0
```

- But what about the elements whose LSB is 1? What will their indices be in the output array?
- To find out we use the formular below
```
dest_idx(1) = ZeroTotal+onesBefore
where Zerototal = inputSize-OnesTotal
```
- For example element 7 which is at index 3.
- There are 3 elements with zero as their LSB meaning `zeroTotals=3`.
- Also element 3 is the only element that comes before element 7 whose LSB is one meaning `OneBefore=1`.
- So from this our index is:
```
dest_idx(1) = 3+1=4
```
- This whole thing looks sequential and not parallel like we expected now let's get into the parallel territory :) .

## Parallize rush
- From the previous section we knew that each of our threads had to the follow:
                - Loads its assigned key.
                - Extracts the relevant bit.
                - Determines which bucket the key belongs to.
                 - Determines its destination index.
                 - Writes the key to the output array
