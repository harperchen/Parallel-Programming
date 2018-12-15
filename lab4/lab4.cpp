#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "mpi.h"

using namespace cv;

int get_pixel(Mat &img, int i, int j) {
	return *(img.data + img.step[0] * i + img.step[1] * j);
}

void set_pixel(Mat &img, int i, int j, int color) { 
	*(img.data + img.step[0] * i + img.step[1] * j) = color;
}


int main(int argc, char** argv)
{
	int size, rank;
	struct timeval start, end;
	double ompi_timeuse, serial_timeuse;
	Mat src, erosion_dst, binary_dst;

	src = imread("test.png", 0);
	binary_dst = src.clone();
	threshold(src, binary_dst, 150, 255, CV_THRESH_BINARY);
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0){
		namedWindow("二值图", 1);
		imshow("二值图", binary_dst);
		gettimeofday(&start, NULL);
		erosion_dst = binary_dst.clone();
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
		serial_timeuse /= 1000;
		printf("The Serial const time is %f ms\n", serial_timeuse);
		namedWindow("腐蚀(serial)", 1);
		imshow("腐蚀(serial)", erosion_dst);
	}
	
	if(rank == 0){
		gettimeofday(&start, NULL);
	}
	int step = ceil(binary_dst.rows / size);
	char *sendBuf = new char[binary_dst.cols * step];
	memset(sendBuf, 0, binary_dst.cols * step);
	for(int i = step * rank; (i < (step * rank + step)) && (i < binary_dst.rows); i++){
		for (int j = 1; j < binary_dst.cols - 1; j++) {
			int origin, upper, left, lower, right;
			origin = get_pixel(binary_dst, i, j);
			upper = get_pixel(binary_dst, i, j - 1);
			lower = get_pixel(binary_dst, i, j + 1);
			if((i >= 1) && (i < binary_dst.rows - 1)){
				left = get_pixel(binary_dst, i - 1, j);
				right = get_pixel(binary_dst, i + 1, j);
			}
			else{
				continue;
			}
			if (upper && origin && left && lower && right) {
				sendBuf[(i - step * rank) * binary_dst.cols + j] = 255;
			}
		}
	}
	
	MPI_Send(sendBuf, step * binary_dst.cols, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
	if(rank == 0){
		int count = 0;
		MPI_Status status;
		bool wrong = false;
		Mat erosion_dst_ompi = binary_dst.clone();
		memset(erosion_dst_ompi.data, 0, binary_dst.rows * binary_dst.cols);
		char *recvBuf = new char[binary_dst.rows * binary_dst.cols];
		for(int i = 0; i < size; i++){
			MPI_Recv(recvBuf + i * step * binary_dst.cols, step * binary_dst.cols, MPI_CHAR, i, i, MPI_COMM_WORLD, &status);
		}
		memcpy(erosion_dst_ompi.data, recvBuf, binary_dst.rows * binary_dst.cols);
		gettimeofday(&end, NULL);
		ompi_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
		ompi_timeuse /= 1000;
		printf("The Openmpi const time is %f ms\n", ompi_timeuse);
		for (int i = 0; i < binary_dst.rows; i++) {
			for (int j = 0; j < binary_dst.cols; j++) {
				if(get_pixel(erosion_dst, i, j) != get_pixel(erosion_dst_ompi, i, j)) {
					wrong = true;	
					count++;
				}
			}
		}
		if(wrong){
			printf("Error!count is %d\n", count);
		}
		else{
			printf("Openmpi compute successfully\n");
		}
		namedWindow("腐蚀(openmpi)", 1);
		imshow("腐蚀(openmpi)", erosion_dst_ompi);
	}
	MPI_Finalize(); 
//	waitKey();
	return 0;
}
