#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define CLOCKS_TO_MILLISEC(t) (t*1000)/CLOCKS_PER_SEC

#define idx(i, j, k) n*o*i + o*j + k
// kommer fra math

/* 
Hele arrayen er i en linje (ikke tredimensjonal slik som i python)
n*o*i: which row (gives you a two dimensional array sort of)
o*j: what column (gives you a one-dimensional array)
k: index in that row

so the array A has the shape (m x n x o)
A[i=1, j=2, k=3] in python is the same as, if we were to flatten A, A_flat[n*o*1 + o*2 + 3]

Du slipper å stresse med å lage en 3D array
*/


