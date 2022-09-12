#include<stdio.h>
#include<mpi.h>

int main(int nargs, char** args) {
    int size, my_rank;
    MPI_Init(&nargs, &args);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("Hello world, Ive rank %d out of %d procs. \n", my_rank, size);
    MPI_finalizae();
    return 0;
}