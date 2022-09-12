# Problem 2
## Code:

```c++
int i,j, sqrt_N;

char *array = malloc(N); // N is a predefined very large integer
array[0] = array[1] = 0;
#pragma omp parallel for //simple parallel for loop
for (i=2; i<N; i++)
    array[i] = 1;

sqrt_N = (int)(sqrt(N)); // square root of N
#pragma omp parallel private(j)
for (i=2; i<=sqrt_N; i++) {
    if (array[i]) {
        #pragma omp for
        for (j=i*i; j<N; j+=i)
            array[j] = 0;
    }
    #pragma omp barrier
}

free (array);
```

## Explanation

The first for-loop can easily be parallelized by using `#pragma omp parallel for`.
The second nested for-loop is harder. Each iteration of $i$ is dependent on the previous one, and so the inner loop must be completed before a new iteration starts. To avoid a lot of overhead we initialize the region outside of the nested for-loop. Since no work is done outside of the nested for-loop and still inside the outer for-loop no time is lost due to several tasks doing the same work. $j$ must be kept private and therefore the `private(j)` is included. Since the same element in `array` can never be updated at the same time in this program we dont need to include a `reduction` pragma. The `barrier` pragma stops faster threads getting ahead before the array is complete. The speedup will decrease the more threads are used. This has to do with `array` having to be updated for each of the the threads which has a lot of overhead when the number of threads increase. Additionally, the nested for-loop, which is actually parallelized, is incrisingly made smaller. Meaning that there not only is a lot of overhead updating `array`, but an unneccesary amount as a large amount of workers may be active at once.