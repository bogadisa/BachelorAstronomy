#include"PageRank_iterations.h"

void test(int N, int *row_ptr, int *col_idx, double *val);

int main() {
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
    
    //Rank the websites
    printf("Output from `PageRank_iterations`:\n");
    PageRank_iterations(N, row_ptr, col_idx, val, d, epsilon, scores);

    free(scores);
}