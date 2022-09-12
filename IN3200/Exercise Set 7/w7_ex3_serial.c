#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void dense_mat_vec(int m, int n, double *x, double *A, double *y);


int main(int nargs, char **args) {
    int m, n;
    double *x, *y, *A;

    printf("Enter integer value of m:\n");
    scanf("%d", &m);
    printf("Enter integer value of n:\n");
    scanf("%d", &n);

    x = (double *)malloc(m*sizeof(double));
    y = (double *)malloc(n*sizeof(double));
    A = (double *)malloc(n*m*sizeof(double));

    double factor = 1./n;
    for (size_t i = 0; i < n; i++) {
        y[i] = i*factor;
    }
    for (size_t i =0; i<m*n; i++) {
        A[i] = i*factor;
    }

    dense_mat_vec(m, n, x, A, y);

    free(x);
    free(y);
    free(A);
    return 0;
}



void dense_mat_vec(int m, int n, double *x, double *A, double *y) {
    int i, j;
    for (i=0; i<m; i++) {
        double tmp = 0.;
        for (j=0; j<n; j++) {
            tmp += A[i*n+j]*y[j];
        }
        x[i] = tmp;
    }
}


