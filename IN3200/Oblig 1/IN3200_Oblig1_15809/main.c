#include"top_n_webpages.h"

void test(int N, int *row_ptr, int *col_idx, double *val, char *argv[]);

int main(int narg, char **argv) {
    if (narg < 2) {
        printf("Requires a file\n");
        exit(0);
    } else if (narg < 3) {
        printf("Requires dampning constant d");
        exit(0);
    } else if (narg < 4) {
        printf("Requires convergence threshold value epsilon");
        exit(0);
    } else if (narg < 5) {
        printf("Requires value of n top web pages");
        exit(0);
    }
    
    
    int *row_ptr, *col_idx, N;
    double *val;
    
    //Reads the file and makes the sparse matrix
    printf("Output from `read_graph_from_file`:\n");
    read_graph_from_file(argv[1], &N, &row_ptr, &col_idx, &val);

    //test now calls top_n_webpages too
    test(N, row_ptr, col_idx, val, argv);

    free(row_ptr);
    free(col_idx);
    free(val);
    
    return 0;
}

void test(int N, int *row_ptr, int *col_idx, double *val, char *argv[]) {
    double epsilon, d;
    double *scores;

    scores = (double *)malloc(N*sizeof(double));
    
    //converts string to double
    d = strtof(argv[2], NULL);
    epsilon = strtof(argv[3], NULL);
    
    //Rank the websites
    printf("Output from `PageRank_iterations`:\n");
    PageRank_iterations(N, row_ptr, col_idx, val, d, epsilon, scores);

    int n = strtol(argv[4], NULL, 10);
    //find the top n pages
    top_n_webpages(N, scores, n);
    free(scores);
}