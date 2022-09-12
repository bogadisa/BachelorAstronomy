#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float limit(int N);

int main() {
    double actual, real, accuracy, tol;
    int N;

    N = 30.0;

    actual = limit(N);
    real = 4./5.;

    tol = 1e-12;

    accuracy = fabs(actual - real);
    if ( accuracy >= tol ) {
        printf("\n The sequence does not approach 4/5, but reaches %lf", actual);
        printf(" after %d iterations.\n", N);
        printf("Accuracy %f", tol);
    }
    else {
        printf("\n The sequence approaches 4/5, and reaches %lf", actual);
        printf(" after %d iterations.\n", N);
    }

    return 0;
}

float limit(int N) {
    float sum = 1;

    for (int i = 1; i < N; ++i) {
        double sign, denom;
        sign =   pow(-1, i);
        denom = pow(2, 2*i);
        sum += sign / denom;
    }
    return sum;
}