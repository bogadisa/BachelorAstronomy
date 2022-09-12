#include<stdlib.h>
#include<stdio.h>
#include<string.h>

typedef struct organized{
    int n_nodes;
    int n_edges;
    int self;
    int *overviewFrom;
    int *overviewTo;
    int *refrences;
} organized_webgraph;

void read_graph_from_file (char *filename, int *N, int **row_ptr, int **col_idx, double **val);
void print_graph_from_file(organized_webgraph organized, int *row_ptr, int *col_idx, double *val);



void read_graph_from_file (char *filename, int *N, int **row_ptr, int **col_idx, double **val) {
    int i, m, err;
    FILE *fp = fopen(filename, "r") ;

    err = fscanf(fp, "%*[^\n]\n");  //ignores 1st line
    err = fscanf(fp, "%*[^\n]\n");  //ignores 2nd line
    // I initially get N and m mixed up, but fix it second time I read the values (didnt wanna risk breaking something)
    err = fscanf(fp, "# Nodes: %d   Edges: %d %*[^\n]", &m, N); 
    err = fscanf(fp, "%*[^\n]\n");  //ignores 4th line

    *row_ptr = malloc((m+2)*sizeof(*row_ptr));
    *col_idx = malloc(*N*sizeof(*col_idx));
    *val = malloc(*N*sizeof(*val));

    //Tracks temporarily stored info to be easily sent to other functions (printing)
    organized_webgraph organized;
    organized.n_nodes = m+2;
    organized.n_edges = *N;
    
    organized.overviewFrom = calloc(organized.n_nodes, sizeof *organized.overviewFrom);
    organized.overviewTo = calloc(organized.n_nodes, sizeof *organized.overviewTo);
    organized.refrences = calloc(organized.n_nodes, sizeof *organized.refrences);
    
    //Getting an overview
    int From, To;
    while (fscanf(fp, "%d %d", &From, &To) == 2) {
        if (From != To) {
            organized.overviewTo[To]++;
        }
        else {
            organized.self++;
        }
    }
    
    //Sorting into row_ptr
    (*row_ptr)[0] = 0;
    for (i=1; i<(organized.n_nodes-organized.self); i++) {
        (*row_ptr)[i] = organized.overviewTo[i-1] + (*row_ptr)[i-1];
    }
    rewind(fp);

    err = fscanf(fp, "%*[^\n]\n");  //ignores 1st line
    err = fscanf(fp, "%*[^\n]\n");  //ignores 2nd line
    err = fscanf(fp, "# Nodes: %d   Edges: %d %*[^\n]", N, &m);
    err = fscanf(fp, "%*[^\n]\n");  //ignores 4th line

    //Making col_idx
    while (fscanf(fp, "%d %d", &From, &To) != EOF) {
        if (From != To) {
            (*col_idx)[organized.overviewFrom[To] + (*row_ptr)[To]] = From;
            organized.overviewFrom[To]++;
        }
    }

    //Making values
    for (i=0; i < organized.n_edges-organized.self; i++) {
        organized.refrences[(*col_idx)[i]]++;
    }
    for (i=0; i < organized.n_edges-organized.self; i++) {
        (*val)[i] = (double)1/organized.refrences[(*col_idx)[i]];
    }
    
    //Checking results
    //print_graph_from_file(organized, *row_ptr, *col_idx, *val);
    
    fclose(fp);
    free(organized.overviewFrom);
    free(organized.overviewTo);
    free(organized.refrences);
}

//prints first the row_ptr array, then the col_idx and last the values
void print_graph_from_file(organized_webgraph organized, int *row_ptr, int *col_idx, double *val) {
    int i;
    printf("Row_ptr:\n[");
    for (i=0; i < organized.n_nodes-organized.self; i++) {
        printf("%d, ", row_ptr[i]);
    }
    printf("]\n Col_idx:\n[");
    for (i=0; i < organized.n_edges-organized.self; i++) {
        printf("%d, ", col_idx[i]);
    }
    printf("]\n Val:\n[");
    for (i=0; i < organized.n_edges-organized.self; i++) {
        printf("%lf, ", val[i]);
    }
    printf("]\n");
}