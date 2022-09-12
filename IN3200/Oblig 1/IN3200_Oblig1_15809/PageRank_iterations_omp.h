#include"PageRank_iterations.h"
#include<omp.h>

void PageRank_iterations_omp(int N, int *row_ptr, int *col_idx, double *val, double d, double epsilon, double *scores);

void PageRank_iterations_omp(int N, int *row_ptr, int *col_idx, double *val, double d, double epsilon, double *scores) {
    int i, j, condition;
    double *tmp, W, scalar, epsilon_max;
    tmp = (double *)malloc(N*sizeof(double));

    #pragma omp parallel for
    for (i=0; i<N; i++) {
        tmp[i] = (double) 1/N;
        //adds a dangling webpage
        if (row_ptr[i] == row_ptr[i+1]) {
            W += scores[i];
        }
    }
    condition = 1;
    j = 0;
    while(condition){
        #pragma omp parallel for reduction(+:W)
        for (i=0; i<N; i++) {
            scores[i] = tmp[i];
        }
        epsilon_max = 0.0;
        j ++;
        scalar = (1 - d + d*W)/N;
        #pragma omp parallel for schedule(dynamic)
        for (i=0; i<N; i++) {
            tmp[i] = scalar + d*vector_product_sparse(i, row_ptr, col_idx, val, scores);
            #pragma omp critical
            epsilon_max = fmax(epsilon_max, fabs(tmp[i]-scores[i]));
        }
        //Testing
        //print_PageRank_iterations(N, tmp, j);
        
        if (epsilon_max < epsilon) {
            scores = tmp;
            condition = 0;
        }
    }
}