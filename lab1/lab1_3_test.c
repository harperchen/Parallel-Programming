#include <stdio.h>
#include <stdlib.h> 
#include "mpi.h"   

int main(int argc, char **argv)
{
    int *sendBuf;
    int *recvBuf;
    
    int i, size, rank;
    int vector_a[4], vector_b[4], vector_result[4];
	
    for (i = 0; i < 4; i++) {
		vector_a[i] = rand() % 100;
		vector_b[i] = rand() % 100;
	}
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*0号进程为发送缓存填充数据*/
    if (rank == 0) {
        printf("The Vector_a is:");
        for (i = 0; i < 4; i++) {
            printf(" %d", vector_a[i]);
        }
        printf("\nThe Vector_b is:");
        for (i = 0; i < 4; i++) {
            printf(" %d", vector_b[i]);
        }
        printf("\n");
    }

    /*结果输出*/
    printf("------------------------------------------------\nrank=%d\n", rank);
    printf("vector_a[%d] = %d  vector_b[%d] = %d\n", rank, vector_a[rank], rank, vector_b[rank]);
    vector_result[rank] = vector_a[rank] + vector_a[rank];
    printf("vector_result[%d] = %d\n", rank, vector_result[rank]);
    printf("------------------------------------------------\n");

    MPI_Finalize();  
    return 0;  
}  