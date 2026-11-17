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
