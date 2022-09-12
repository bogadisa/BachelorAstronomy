#include<string.h>
#include<stdlib.h>
#import<omp.h>

int main(int argc, char *argv[]) {
    const char text1[50] = "ATTTGCGCAGACCTAAGCA";
    const char text2[50] = "AAABCDEFGAAGEREAANMT";
    const char text3[50] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char pattern1[50] = "GCA";
    const char pattern2[50] = "AA";
    const char pattern3[50] = "BTTT";
    printf("Text 1 returns: %d\n", count_occurence(text1, pattern1));
    printf("Text 2 returns: %d\n", count_occurence(text2, pattern2));
    printf("Text 3 returns: %d\n", count_occurence(text3, pattern3));
}

int count_occurence(const char *text_string, const char *pattern) {
    int N = strlen(text_string);
    int n = strlen(pattern);
    int occurences = 0;
    #pragma omp parallel for reduction(+:occurences)
    for (int i=0; i<N-n+1; i++) occurences += (strncmp(&(text_string[i]), pattern, n) == 0);
    return occurences;
}

int count_occurence(const char *text_string, const char *pattern) {
    int N, n, occurences;
    N = strlen(text_string);
    n = strlen(pattern);
    occurences = 0;

    int size, rank;
    MPI_Status recv_status;
    MPI_Comm_size( MPI_COMM_WORLD , &size);
    MPI_Comm_rank( MPI_COMM_WORLD , &rank);
    MPI_Bcast( &text_string , N , MPI_CONST_CHAR , 0 , MPI_COMM_WORLD);
    MPI_Bcast( &pattern , n , MPI_CONST_CHAR , 0 , MPI_COMM_WORLD);

    for (int i=rank*(N/size+N%size); i<(rank+1)*N/size+N%size; i++) occurences += (strncmp(&(text_string[i]), pattern, n) == 0);
    int total = 0;
    if (rank==0) {
        total += occurences;
        for (int i=1; i<size; i++)
            MPI_IRecv( &occurences , 1 , MPI_INT , i , i , MPI_COMM_WORLD , &recv_status);
            total += occurences;
    } else {
        MPI_Send(&occurences, 1, MPI_INT, rank, rank, MPI_COMM_WORLD)
    }
    return total;
}