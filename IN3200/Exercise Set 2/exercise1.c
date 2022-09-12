#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define CLOCKS_TO_MILLISEC(t) (t*1000)/CLOCKS_PER_SEC

#define idx(i,j, k) n*o*i + o*j + k