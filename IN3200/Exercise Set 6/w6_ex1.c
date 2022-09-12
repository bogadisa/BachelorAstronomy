#include <stdio.h>
#include <omp.h>

int main (int nargs, char **args) {
    int i, j, k, imax, jstart, jend, jmax, kmax;
    float phi[10][10][10];


    #pragma omp parallel private(k,j,i)
    {
        int numthreads, threadID, jstart, jend, m;
        numthreads = omp_get_num_threads();
        threadID = omp_get_thread_num();
        jstart = ((jmax-2)*threadID)/numthreads + 1;
        jend = ((jmax-2)*(threadID+1))/numthreads;
        for (m=1; m<=kmax+numthreads-3; m++) { // the wavefronts
            k = m - threadID;
            if (k>=1 && k<=kmax-2) {
                for (j=jstart; j<=jend; j++)
                    for (i=1; i<imax-1; i++)
                        phi[k][j][i] = (phi[k-1][j][i] + phi[k][j-1][i]
                        +phi[k][j][i-1] + phi[k][j][i+1]
                        +phi[k][j+1][i] + phi[k+1][j][i])/6.0;
        }
        #pragma omp barrier
        }
        } // end of the parallel region


    return 0;
}
