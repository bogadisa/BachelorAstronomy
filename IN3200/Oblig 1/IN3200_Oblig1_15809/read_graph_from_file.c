#include"read_graph_from_file.h"

void testing(int *col_idx, int N);

int main() {
    int c;
    int *row_ptr, *col_idx, N;
    double *val;

    //notes: I am allocating something wrong, assigning any value crashes the program, e.g int i = 10
    //       this is not however the case if I define something in another scope
    //read_graph_from_file("simple_webgraph.txt", &N, &row_ptr, &col_idx, &val);
    //read_graph_from_file("100nodes_graph.txt", &N, &row_ptr, &col_idx, &val);
    read_graph_from_file("web-stanford.txt", &N, &row_ptr, &col_idx, &val);
    

    //testing(col_idx, N);

    //int c = 10;
    free(row_ptr);
    free(col_idx);
    free(val);
    return 0;
}

void testing(int *col_idx, int N) {
    printf("[");
    for (int j=0; j < N; j++) {
        printf("%d, ", col_idx[j]);
    }
    printf("]\n");
}