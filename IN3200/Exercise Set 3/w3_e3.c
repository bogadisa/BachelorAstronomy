#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

double pow_mine(double x, int y);

int main() {
    int y1;
    double x, y2, x_min, x_std;

    y1 = 100;
    y2 = 100.0;
    x = 10.0;

    clock_t start_min, end_min, start_std, end_std;
    start_min = clock();
    x_min = pow_mine(x, y1);
    end_min = clock();
    start_std = clock();
    x_std = pow(x, y2);
    end_std = clock();

    double time_mine, time_std;
    time_mine = (end_min - start_min);
    time_std = (end_std - start_std);
    printf("Mine took %lfms, while the standard one took %lfms", time_mine, time_std);

    return 0;
}

double pow_mine(double x, int y) {
    for (int i=0 ; i<y ; i++) {
        x *= x;
    }
    return x;
}