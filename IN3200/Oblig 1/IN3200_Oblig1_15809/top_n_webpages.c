#include"top_n_webpages.h"

void test(int N, int *row_ptr, int *col_idx, double *val);

int main() {
    int *row_ptr, *col_idx, N;
    double *val;
    
    //Reads the file and makes the sparse matrix
    printf("Output from `read_graph_from_file`:\n");
    read_graph_from_file("100nodes_graph.txt", &N, &row_ptr, &col_idx, &val);
    
    //test now calls top_n_webpages
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

    d = 0.85;
    epsilon = 0.00001;
    
    //Rank the websites
    printf("Output from `PageRank_iterations`:\n");
    PageRank_iterations(N, row_ptr, col_idx, val, d, epsilon, scores);

    int n = 8;
    printf("Output from `top_n_webpages`:\n");
    top_n_webpages(N, scores, n);
    free(scores);
}