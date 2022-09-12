#include <stdio.h>
#include <float.h>

int main() {
    printf("Storage size of float-point %lu \n", sizeof(float));
    printf("Min float positive value: %E \n",  FLT_MIN);
    printf("Max float positive value: %E \n",  FLT_MAX);
    printf("Precision: %d \n", FLT_DIG);

    return 0;
}