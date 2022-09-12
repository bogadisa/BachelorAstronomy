#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double foo(i){
    return 0.0;
}

int main(int nargs, char **args) {
    double *a;
    int n, i;

    n = 100;

    a = (double *)malloc(n*sizeof(double));

    #pragma omp parallel for 
    for (i=0; i < n; i++) {
        a[i] = foo(i);
    }

    return 0;
}