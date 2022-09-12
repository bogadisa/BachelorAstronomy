#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int nargs, char **args) {
    double *a;
    int i, k;

    k = 100;

    a = (double *)malloc(2*k*sizeof(double));

    #pragma omp parallel for
    for (i=k; i < 2*k; i++) {
        a[i] = a[i] + a[i-k];
    }

    return 0;
}