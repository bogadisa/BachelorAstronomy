#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Bandwidth = data volume / (time usage - latency), when time usage >> latency -> Bandwidth = data volume / time usage

void kernel_loop(int A, int B, int s, int n);

int main() {
    int N, n;
    int *B;

    N = 10000;
    n = 100;

    int data_volume = N * n* sizeof(*B);
    *B = malloc(data_volume);


    for (int i=0 ; i<N ; i++) {
        int random_value, *y = malloc(n * sizeof(*y));
        for (int j=0 ; j<n; j++) {
            random_value = (int)rand()/RAND_MAX*2.0-1.0;
            y[j] = random_value;
        }
        B[i] = y;
    }

    int s = 2;

    for (int i=0 ; i<N ; i++) {
        kernel_loop(B[i*(n - 1)], B, s, n);
    }


    return 0;
}

void kernel_loop(int A, int B, int s, int n) {
    for (int j=0; j<n ; j++) {
        A[j] = s*B[j];
    }
}