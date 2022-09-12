int printvec_d(double *a, int n){
    printf("[%f,", a[0]);
    for (size_t i = 1; i < n-1; i++) {
        printf(" %f,", a[i]);
    }
    printf(" %f]\n", a[n-1]);
    return 0;
}

int printvec_i(int *a, int n){
    printf("[%d,", a[0]);
    for (size_t i = 1; i < n-1; i++) {
        printf(" %d,", a[i]);
    }
    printf(" %d]\n", a[n-1]);
    return 0;
}


// In read_graph_from_file

printf("row_ptr = ");
printvec_i(*row_ptr, nodes+1);
printf("col_idx = ");
printvec_i(*col_idx, edges);
printf("val= ");
printvec_d(*val, edges);
