#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main() {
    double *a, *b;
    int n, i;
    int flag;

    n = 100;

    a = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(n*sizeof(double));

    
    for (i=0; (i < n) & (!flag); i++) {
        a[i] = 2.3 * i;
        if (a[i] < b[i]) flag = 1;
    }
    // Not suitable due to each iteration is dependent on previous one

    return 0;
}