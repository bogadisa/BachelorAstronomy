#include<stdlib.h>

typedef struct{
    int immune = 0; // 0 if not immune, 1 if immune
    int infected = 0; // 0 if not infected, 1 if infected
    int days_infected = 0;
}person;