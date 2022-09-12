#include"read_graph_from_file.h"
#include<math.h>

void PageRank_iterations(int N, int *row_ptr, int *col_idx, double *val, double d, double epsilon, double *scores);
double vector_product_sparse(int i, int *row_ptr, int *col_idx, double *val, double *scores);
void print_PageRank_iterations(int N, double *tmp, int j);
int printvec_d(double *a, int n){
    printf("[%f,", a[0]);
    for (size_t i = 1; i < n-1; i++) {
        printf(" %f,", a[i]);
    }
    printf(" %f]\n", a[n-1]);
    return 0;
}

void PageRank_iterations(int N, int *row_ptr, int *col_idx, double *val, double d, double epsilon, double *scores) {
    int i, j, condition;
    double *tmp, W, scalar, epsilon_max;
    tmp = (double *)malloc(N*sizeof(double));
    
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
        //updates scores
        for (i=0; i<N; i++) {
            scores[i] = tmp[i];
        }
        epsilon_max = 0.0;
        //only used to keep track of iteration
        j ++;
        scalar = (1 - d + d*W)/N;
        for (i=0; i<N; i++) {
            tmp[i] = scalar + d*vector_product_sparse(i, row_ptr, col_idx, val, scores);
            epsilon_max = fmax(epsilon_max, fabs(tmp[i]-scores[i]));
        }
        //Testing
        //print_PageRank_iterations(N, tmp, j);
        
        //checks if we are happy with current scores
        if (epsilon_max < epsilon) {
            scores = tmp;
            condition = 0;
        }
        
        printf("scores= ");
        printvec_d(scores, N);
    }
}

//self explanitory
double vector_product_sparse(int i, int *row_ptr, int *col_idx, double *val, double *scores) {
    int row_start, row_end;

    row_start = row_ptr[i];
    row_end = row_ptr[i+1];

    double product = 0.0;
    for (int j=row_start; j<row_end; j++) {
        product += val[j] * scores[col_idx[j]];
    }
    return product;
}



//prints the scores for every 10 iteration, so that you can see them converging
void print_PageRank_iterations(int N, double *tmp, int j) {
    int i;
    //ensures not every iteration is printed
    if ((j-1) % 10 != 0) {
        return;
    } else {
        printf("Iteration %d:\n[", j);
        for (i=0; i<N; i++) {
            printf("%lf, ", tmp[i]);
        }
        printf("]\n");
    }
}   