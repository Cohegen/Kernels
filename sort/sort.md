# Intro to Sorting
- In this directory we'll dive into sorting algorithms. But specifically we'll dive into radix sort.

## Deep dive into Radix sort Naive
- To understand how radix sort works, let's use a small example:
- Given the array below:

```
A = [4,3,2,7,6,1]
```
- If we using Bubble sort, we would be comparing two adjacent elements.
- Like in our first round 4 is greater than 3 so we would need to swap them giving us the array below:
```
A = [3,4,2,7,6,1]
```
- The array above is obtained in the first round.
- However radix sort differs such that we sort our array in terms of keys and values.
- But first let's write our array elements in their binary forms:
```
A = [100,011,010,111,110,001]
```

- Let's introduce two keys now was **bucket 0** and **bucket 1**.
- Bucket 0 will be sort of the category where the select bit 0 while Bucket 1 will be the category where the select bit is 1.

## Sorting process
- Let's begin our sorting process.

### Round 1
- In this round we will look at the bit in position 0 of each binary number.
- So bucket 0 will contains elements whose bit at position 0 is 0, so it will be:
```
bucket 0 = [100,010,110]
```
- On the other hand bucket 1 will contain elements whose bit at position 1 is 1, so it will be:
```
bucket 1 = [011,111,001]
```
- Our final sorted array will be:
```
out = [100,010,110,011,111,001]
```
- Notice that we maintain the order of the elements that belong to bucket 0 come before elements in bucket 1.

### Round 2
- This round like other subsequent rounds depend on their preceding round to give them their respective input array.
- In this scenario round 2 depends on the output of round 1.
- So input is:
```
out = [100,010,110,011,111,001]
```
- However in this round we will looking at the bit in position 1 of every single binary number and classifying them into their respective buckets.
- So bucket 0:
```
bucket 0 = [100,001]
```
- And bucket 1:
```
bucket 1 = [010,110,011,111]
```
- Finally our output in this round is:
```
out = [100,001,010,110,011,111]
```

- We maintain the same order of bucket 0 elements coming before bucket 1 elements.

### Round 3
- The input of this round is the output of the previous rouns.
- So input:
```
input = [100,001,010,110,011,111]
```
- In this round we use bit at position 2 as our factor:
- Bucket 0:
```
bucket 0 = [001,010,011]
```
- Bucket 1:
```
bucket 1 = [100,110,111]
```
- So our output becomes:
```
out = [001,010,011,100,110,111]
```

- And after converting them into decimal we get:
```
out = [1,2,3,4,6,7]
```
