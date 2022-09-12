#include"PageRank_iterations.h"

void top_n_webpages(int N, double *scores, int n);
void print_top_n_webpages(int *top_indices, double *top_scores, int n);

void top_n_webpages(int N, double *scores, int n) {
    int i, j;

    double *top_scores;
    int *top_indices;

    top_scores = (double *)malloc(n*sizeof(double));
    top_indices = (int *)malloc(n*sizeof(int));

    double max_score = 2.0, top_score = 0.0;
    double top_indice;

    for (i = 0; i<n; i++){
        for (j=0; j<N; j++) {
            //this excludes previous higher scores
            if (scores[j] < max_score) {
                if (scores[j] > top_score) {
                    top_score = scores[j];
                    top_indice = j;
                }
            }
        }
        //only looks for lower scores than those already found
        max_score = top_score;
        top_score = 0.0;

        top_scores[i] = max_score;
        top_indices[i] = top_indice;
    }
    print_top_n_webpages(top_indices, top_scores, n);
    free(top_indices);
    free(top_scores);
}


//Prints the top score and their corresponding indice in the val array, in a very easy to read table
void print_top_n_webpages(int *top_indices, double *top_scores, int n) {
    int i;
    printf("+-----------+-----------+\n");
    printf("|Top scores | Web indice|\n");
    for (i=0; i<n; i++) {
        printf("|%8.8lf | %8d  |\n", top_scores[i], top_indices[i]);
    }
    printf("+-----------+-----------+\n");
}