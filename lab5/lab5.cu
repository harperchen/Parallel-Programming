#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cv.h>
#include <highgui.h>

#define BLOCKSIZE 4

using namespace cv;
using namespace std;
int get_pixel(Mat &img, int i, int j) {
	return *(img.data + img.cols * i + j);
}

void set_pixel(Mat &img, int i, int j, int color) { 
    *(img.data + img.cols * i + j) = color;
}

__global__ void erosion(char *a, char *b, int rows, int cols) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
     + threadIdx.y * blockDim.x + threadIdx.x;
    if(i < rows){
        for (int j = 0; j < cols; j++) {
            if((i == 0) || (i == rows - 1) || (j == 0) || (j == cols - 1)){
                *(b + i * cols + j) = 0;
                continue;
            }
            int origin = *(a + i * cols + j);
            int upper = *(a + i * cols + j - 1);
            int left = *(a + (i - 1) * cols + j);
            int lower = *(a + i * cols + j + 1);
            int right = *(a + (i + 1) * cols + j);
            if (upper && origin && left && lower && right) {
                *(b + i * cols + j) = 255;
            }
            else {
               *(b + i * cols + j) = 0;
            }
        }
    }
}


int main(int argc, char **argv) {
    struct timeval start, end;
    double cuda_timeuse, serial_timeuse;

    Mat src, erosion_dst;
    Mat erosion_dst_cuda, binary_dst;

    src = imread("test.png", 0);
	binary_dst = src.clone();
	threshold(src, binary_dst, 150, 255, CV_THRESH_BINARY);
    imwrite("binary.png", binary_dst);
    erosion_dst = binary_dst.clone();
	erosion_dst_cuda = binary_dst.clone();
	memset(erosion_dst_cuda.data, 0, binary_dst.rows * binary_dst.cols);

    cudaError_t error = cudaSuccess;

    char *device_a, *device_b;
    error = cudaMalloc((void **)&device_a, binary_dst.rows * binary_dst.cols);
    error = cudaMalloc((void **)&device_b, binary_dst.rows * binary_dst.cols);

    if (error != cudaSuccess) {
        printf("Fail to cudaMalloc on GPU");
        return 1;
    }

//GPU parallel start
    gettimeofday(&start, NULL);
    cudaMemcpy(device_a, binary_dst.data, binary_dst.rows * binary_dst.cols, cudaMemcpyHostToDevice);
    int gridsize = (int)ceil(sqrt(ceil(binary_dst.rows / (BLOCKSIZE * BLOCKSIZE))));

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(gridsize, gridsize, 1);

    erosion<<<dimGrid, dimBlock>>>(device_a, device_b, binary_dst.rows, binary_dst.cols);
    cudaThreadSynchronize();

    cudaMemcpy(erosion_dst_cuda.data, device_b, binary_dst.rows * binary_dst.cols, cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);

    cuda_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Cuda const time is %lf ms\n", cuda_timeuse / 1000);
//GPU parallel end

//CPU serial start
    gettimeofday(&start, NULL);
    for (int i = 0; i < binary_dst.rows; i++) {
		for (int j = 0; j < binary_dst.cols; j++) {
			if((i == 0) || (i == binary_dst.rows - 1) || (j == 0) || (j == binary_dst.cols - 1)){
				set_pixel(erosion_dst, i, j, 0);
				continue;
			}
			int origin = get_pixel(binary_dst, i, j);
			int upper = get_pixel(binary_dst, i, j - 1);
			int left = get_pixel(binary_dst, i - 1, j);
			int lower = get_pixel(binary_dst, i, j + 1);
			int right = get_pixel(binary_dst, i + 1, j);
			if (upper && origin && left && lower && right) {
				set_pixel(erosion_dst, i, j, 255);
			}
			else {
				set_pixel(erosion_dst, i, j, 0);
			}
		}
	}
    gettimeofday(&end, NULL);
    serial_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Serial const time is %f ms\n", serial_timeuse / 1000);
//CPU serial end
     imwrite("serial.png", erosion_dst);
     imwrite("cuda.png", erosion_dst_cuda);
//check
    int errorNum = 0;
   for (int i = 1; i < binary_dst.rows - 1; i++) {
		for (int j = 1; j < binary_dst.cols - 1; j++) {
			if(get_pixel(erosion_dst, i, j) != get_pixel(erosion_dst_cuda, i, j)) {
                errorNum ++;
                printf("%d %d\n", get_pixel(erosion_dst, i, j), get_pixel(erosion_dst_cuda, i, j));
            }
        }
    }
    if (errorNum == 0) {
        printf("Successfully run on GPU and CPU!\n");
    } else {
        printf("%d error(s) occurs!\n", errorNum);
    }

    waitKey();
    cudaFree(device_a);
    cudaFree(device_b);
    return 0;
}
