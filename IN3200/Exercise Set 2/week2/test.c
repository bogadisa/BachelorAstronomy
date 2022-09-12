#include <stdlib.h>

int main (int nargs, char** args)
{
  double *a;
  int n = 10000;
  a = malloc (n*sizeof(double));
  for (int i=0; i<n; i++)
    a[i] = 1.*i;

  printf("sizeof(double)=%d\n",sizeof(double));
  printf("sizeof(double*)=%d\n",sizeof(a));

  return 0;
}
