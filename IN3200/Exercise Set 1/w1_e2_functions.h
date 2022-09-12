#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "w1_e2_common.h"

double average(double *arr, int n) {
    double sum;

    for (int i = 0 ; i < n ; i++) {
        sum += arr[i];
    }
    return sum / (double)n;
}

double stdev(double *arr, int n, double average) {
    double dev = 0.;
    double term;

    for (int i = 0 ; i < n ; i++) {
        term = arr[i] - average;
        dev += term*term;
    }
    dev = sqrt(dev/(double)(n-1));

    return dev;
}

int findminidx(double *arr, int n) {
    double min;
    int minidx;

    min = arr[0];
    for (int i = 1 ; i < n ; i++) {
        if (min > arr[i]) {
            min = arr[i];
            minidx = i;
        }
    }
    return minidx;
}

int findmaxidx(double *arr, int n) {
    double max;
    int maxidx;

    max = arr[0];
    for (int i = 1 ; i < n ; i++) {
        if (max < arr[i]) {
            max = arr[i];
            maxidx = i;
        }
    }
    return maxidx;
}

void readfile(char *filename, temperature_data *data) {
    FILE *datafile;
    int err;

    datafile = fopen(filename, "r");

    if (datafile == NULL) {
        printf("No file found");
        exit(0);
    }

    err = fscanf(datafile, "%d", &data->n);
    //reading the first line and finding how many data points we have

    data->times = (char *)malloc(5 * data->n * sizeof(char));
    data->temps = (double *)malloc(data->n * sizeof(double));
    //allocating memory for times and temperatures, times is a flat 1D array

    //go through the rest of the lines
    for (int i = 0 ; i < data->n; i++) {
        err = fscanf(datafile, "%5s %lf", &data->times[5 * i], &data->temps[i]);
        //look for 5 strings (%5s), then a float (%fl)
        //remember times is flat and we are looking for five at the time
    }

    fclose(datafile);
}