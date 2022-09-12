#include "w1_e2_functions.h"

int main(int narg, char **argv) {
    if (narg < 2) {
        printf("Requires a file\n");
        exit(0);
    }

    double avg, dev;
    int minidx, maxidx;
    temperature_data *data;

    data = (temperature_data *)malloc(sizeof(temperature_data));

    readfile(argv[1], data);

    avg = average(data->temps, data->n);
    dev = stdev(data->temps, data->n, avg);
    minidx = findminidx(data->temps, data->n);
    maxidx = findmaxidx(data->temps, data->n);

    printf("The average temperature was %lf with a standard deviation of %lf\n", avg, dev);
    printf("The highest temperature recorded was %lf at %.5s\n", data->temps[maxidx], &data->times[5*maxidx]);
    printf("The lowest temperature recorded was %lf at %.5s\n", data->temps[minidx], &data->times[5*minidx]);

    free(data->times);
    free(data->temps);
    free(data);

    return 0;
}