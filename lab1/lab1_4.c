#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>

#define RANDOM(x) (rand() % x)

#define MAX 100000

#define BLOCKSIZE 4
#define BLOCKSIZE 4


__global__ void add(const int *a, const int *b, int *c, int n) {
    //threadIdx.x 0~15 blockDim.x = 16
    //threadIdx.y 0~4 blockDim.y = 4
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
     + threadIdx.y * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


int main(int argc, char **argv) {
    int i, n = 512;
    struct timeval start, end;

    if (argc == 2) {
        n = atoi(argv[1]);
    }
    //vector_a
    int *host_a = (int *)malloc(sizeof(int) * n);
    //vector_b
    int *host_b = (int *)malloc(sizeof(int) * n);
    //vector_c_gpu
    int *host_c = (int *)malloc(sizeof(int) * n);
    //vector_c_cpu
    int *host_c2 = (int *)malloc(sizeof(int) * n);

    for (i = 0; i < n; i++) {
        host_a[i] = RANDOM(MAX);
        host_b[i] = RANDOM(MAX);
    }

    cudaError_t error = cudaSuccess;

    int *device_a, *device_b, *device_c;
    error = cudaMalloc((void **)&device_a, sizeof(int) * n);
    error = cudaMalloc((void **)&device_b, sizeof(int) * n);
    error = cudaMalloc((void **)&device_c, sizeof(int) * n);

    if (error != cudaSuccess) {
        printf("Fail to cudaMalloc on GPU");
        return 1;
    }

//GPU parallel start
    gettimeofday(&start, NULL);

    cudaMemcpy(device_a, host_a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(int) * n, cudaMemcpyHostToDevice);

    int gridsize = (int)ceil(sqrt(ceil(n / (BLOCKSIZE_x * BLOCKSIZE_y))));

    dim3 dimBlock(BLOCKSIZE_x, BLOCKSIZE_y, 1);
    dim3 dimGrid(gridsize, gridsize, 1);

    add<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, n);
    cudaThreadSynchronize();

    cudaMemcpy(host_c, device_c, sizeof(int) * n, cudaMemcpyDeviceToHost);

    gettimeofday(&end, NULL);

    double t = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Cuda const time is %lf ms\n", t / 1000);
//GPU parallel end


//CPU serial start
    gettimeofday(&start, NULL);
    for (i = 0; i < n; i++) {
        host_c2[i] = host_a[i] + host_b[i];
    }
    gettimeofday(&end, NULL);
    t = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Serial const time is %f ms\n", t / 1000);
//CPU serial start

//check
    int errorNum = 0;
    for (int i = 0; i < n; i++) {
        if (host_c[i] != host_c2[i]) {
            errorNum ++;
            printf("Error occurs at index: %d: a + b = %d + %d = %d, but c = %d, c2 = %d\n", i, host_a[i], host_b[i], host_a[i] + host_b[i], host_c[i], host_c2[i]);
        }
    }
    if (errorNum == 0) {
        printf("Successfully run on GPU and CPU!\n");
    } else {
        printf("%d error(s) occurs!\n", errorNum);
    }

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_c2);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}