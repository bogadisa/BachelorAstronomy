#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int nargs, char **args) {
    double *a, *b;
    int x, N, i;

    x = 16;
    N = 100;

    a = (double *)malloc(N*sizeof(double));
    b = (double *)malloc(N*sizeof(double));

    #pragma omp parallel for
        for (i=0; i < (int) sqrt(x); i++) {
            a[i] = 2.3 * x;
            if (i < 10) b[i] = a[i];
    }

    return 0;
}