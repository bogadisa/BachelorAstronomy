#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

double numerical_integration (double x_min, double x_max, int slices);

int main() {
    double *result;
    int x_min, x_max, N, step;
    x_min = 0;
    x_max = 1;
    N = 10;
    step = 10;
    
    result = (double *)malloc((N - N%step)*sizeof(double));

    for (int i=0 ; i<N ; i++) {
        result[i] = numerical_integration(x_min, x_max, (i+1)*step);
        printf("With %d slices we get %lf \n", (i + 1)*step, result[i]);
    }

    free(result);
    return 0;
}

double numerical_integration (double x_min, double x_max, int slices) {
    double delta_x = (x_max - x_min)/slices;
    double x, sum = 0.;
    for (int i=0 ; i<slices; i++) {
        x = x_min + (i+0.5)*delta_x;
        sum = sum + 4.0/(1.0 + x*x);
    }
    return sum*delta_x;
}