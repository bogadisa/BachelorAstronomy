#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
     


int main(int nargs, char **args) {
    double start, end;
    double cpu_time_used_parallel, cpu_time_used_serial, cpu_time_used_parallel_chunk;
    
    double dotp, *a, *b;
    int i, n = 10000000, chunksize = 1000;

    a = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(n*sizeof(double));

    double factor = 1./n;
    for (size_t i = 0; i < n; i++) {
        a[i] = i*factor;
        b[i] = i*factor;
    }

    dotp = 0.;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:dotp)
    for (i=0; i<n; i++) {
        dotp += a[i]*b[i];
    }
    end = omp_get_wtime();
    cpu_time_used_parallel = ((double) (end - start)) * 1000;

    dotp = 0.;
    start = omp_get_wtime();
    for (i=0; i<n; i++) {
        dotp += a[i]*b[i];
    }
    end = omp_get_wtime();
    cpu_time_used_serial = ((double) (end - start)) * 1000;

    dotp = 0.;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:dotp) schedule(static, chunksize)
    for (i=0; i<n; i++) {
        dotp += a[i]*b[i];
    }
    end = omp_get_wtime();
    cpu_time_used_parallel_chunk = ((double) (end - start)) * 1000;

    printf("The serial version uses %lf milliseconds, while the parallel version uses %lf milliseconds. Meaning a difference of %lf\n", 
            cpu_time_used_serial, 
            cpu_time_used_parallel, 
            fabs(cpu_time_used_parallel - cpu_time_used_serial));

    printf("With parallel using a chuncksize %d it takes %lf milliseconds.\n", chunksize, cpu_time_used_parallel_chunk);

    return 0;
}