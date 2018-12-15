#include <stdio.h>
#include <stdlib.h> 
#include "mpi.h"   

int main(int argc, char **argv)
{
    int *sendBuf;
    int *recvBuf;
    
    int i, size, rank;
    int vector_a[4], vector_b[4], vector_result[4];
	
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*0号进程为发送缓存填充数据*/
    if (rank == 0) {
        for (i = 0; i < 4; i++) {
		    vector_a[i] = rand() % 100;
		    vector_b[i] = rand() % 100;
	    }
        /*创建消息发送和接收缓冲区*/
        sendBuf = (int *)malloc(size * 2 * sizeof(int));
        recvBuf = (int *)malloc(2 * sizeof(int));
        for (i = 0; i < size * 2; i++) {
            if(i % 2 == 0){
                sendBuf[i] = vector_a[i / 2];
            }
            else{
                sendBuf[i] = vector_b[i / 2];
            }
        }
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

    /*---------------------------------------MPI_Scatter函数详解--------------------------------------*/
    //功能：
    //  通过根进程向同一个通信域中的所有进程发送数据，将数据发送缓冲区的数据分割成长度相等的
    //  段，然后分段发送数据给每个进程，如果每段包含N个数据，则向进程i发送的段为[send[i*N],int[i*N+N])
    //函数参数：
    //  MPI_Scatter(待发送数据缓冲区地址，数据个数，数据类型，接收缓冲区地址，数据个数，
    //  数据类型，发送消息的进程的标识，通信域）
    MPI_Scatter(sendBuf, 2, MPI_INT, recvBuf, 2, MPI_INT, 0, MPI_COMM_WORLD);

    /*结果输出*/
    printf("------------------------------------------------\nrank=%d\n", rank);
    printf("vector_a[%d] = %d  vector_b[%d] = %d\n", rank, recvBuf[0], rank, recvBuf[1]);
    vector_result[rank] = recvBuf[0] + recvBuf[1];
    printf("vector_result[%d] = %d\n", rank, vector_result[rank]);
    printf("------------------------------------------------\n");

    MPI_Finalize();  
    return 0;  
}  