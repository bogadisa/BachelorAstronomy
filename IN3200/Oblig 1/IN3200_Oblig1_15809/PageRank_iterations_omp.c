#include"PageRank_iterations_omp.h"

void test(int N, int *row_ptr, int *col_idx, double *val);

int main(int nargs, char **args) {
    int *row_ptr, *col_idx, N;
    double *val;
    
    //Reads the file and makes the sparse matrix
    printf("Output from `read_graph_from_file`:\n");
    read_graph_from_file("100nodes_graph.txt", &N, &row_ptr, &col_idx, &val);
    
    //Due to how row_ptr, col_idx, val and N are stored in memory, I cannot allocate 
    //within the same scope, and forced to make the function below, unsure why this is 
    test(N, row_ptr, col_idx, val);

    free(row_ptr);
    free(col_idx);
    free(val);
    
    return 0;
}

void test(int N, int *row_ptr, int *col_idx, double *val) {
    double epsilon, d;
    double *scores;

    scores = (double *)malloc(N*sizeof(double));

    d = 1.0;
    epsilon = 0.00001;
    
    double start, end;
    double cpu_time_used_parallel;
    //Rank the websites
    printf("Output from `PageRank_iterations`:\n");

    //Parallel version
    start = omp_get_wtime();
    PageRank_iterations_omp(N, row_ptr, col_idx, val, d, epsilon, scores);
    end = omp_get_wtime();
    cpu_time_used_parallel = ((double) (end - start)) * 1000.0;
    printf("With parallelization it takes %lf milliseconds.\n", cpu_time_used_parallel);

    //Non-parallelized
    start = omp_get_wtime();
    PageRank_iterations(N, row_ptr, col_idx, val, d, epsilon, scores);
    end = omp_get_wtime();
    cpu_time_used_parallel = ((double) (end - start)) * 1000.0;
    printf("Without parallelization it takes %lf milliseconds.\n", cpu_time_used_parallel);

    free(scores);
}