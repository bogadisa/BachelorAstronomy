# Problem 3

## 3a

### Code:
```c++
void sweep (int N, double **table1, int n, double **mask double**table2) {
    int i,j,ii,jj;
    double temp;
    #pragma omp parallel for private(j, ii, jj, temp)
    for (i=0; i<=N-n; i++)
        for (j=0; j<=N-n; j++) {
            temp = 0.0;
            for (ii=0; ii<n; ii++)
                for (jj=0; jj<n; jj++)
                    temp += table1[i+ii][j+jj]*mask[ii][jj];
            table2[i][j] = temp;
        }
}
```

### Explenation:
Each iteration is independent of eachother. $i=0$ can be completed at the same time as $i=1$ and so on. Therefore we can already initialize the parallel region on the outermost for-loop if we remember to privatize all variables the `for` pragma does´nt already handle.

## 3b
Hmm. Usikker. 

Forsøk: There are $(N-n)\cdot (N-n)\cdot n \cdot n$ number of floating point operations. The theoretical minimum computing time would then be $\frac{(N-n)\cdot (N-n)\cdot n \cdot n}{2\cdot (Max flops)}$.