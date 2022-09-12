#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int nargs, char **args) {
    double *a, *b, dotp;
    int n, i;

    n = 100;

    a = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(n*sizeof(double));

    #pragma omp parallel for reduction(+:dotp)
    for (i=0; i < n; i++) {
        dotp += a[i] * b[i];
    }

    return 0;
}