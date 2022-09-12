#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double foo(i){
    return 0.0;
}


int main() {
    double *a, *b;
    int n, i;
    int flag;

    n = 100;

    a = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(n*sizeof(double));

    for (i=0; i < n; i++) {
        a[i] = foo(i);
        if (a[i] < b[i]) break;
    }
    // Not suitable due to each iteration is dependent on previous one, an element of a might
    // be changed when the loop shouldve already been broken

    return 0;
}