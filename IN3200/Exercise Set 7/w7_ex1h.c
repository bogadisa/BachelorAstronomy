#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int nargs, char **args) {
    double *a;
    int i, k, n, b;

    k = 100;
    n = 2*k;
    b = 1;

    a = (double *)malloc(n*sizeof(double));

    #pragma omp parallel for
    for (i=k; i < 2*k; i++) {
        a[i] = b * a[i-k];
    }

    return 0;
}