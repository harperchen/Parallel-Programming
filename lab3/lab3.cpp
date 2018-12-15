#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "omp.h"
#include "opencv2/opencv.hpp"

using namespace cv;

/// 全局变量
Mat src, erosion_dst, erosion_dst_omp, binary_dst;

int get_pixel(Mat &img, int i, int j) {
	return *(img.data + img.step[0] * i + img.step[1] * j);
}

void set_pixel(Mat &img, int i, int j, int color) { 
	*(img.data + img.step[0] * i + img.step[1] * j) = color;
}


int main(int argc, char** argv)
{
	int n = atoi(argv[1]);
	struct timeval start, end;
	double omp_timeuse, serial_timeuse;
	src = imread("test.png", 0);

	// binary
	binary_dst = src.clone();
	threshold(src, binary_dst, 150, 255, CV_THRESH_BINARY);
	//show the binary image
	namedWindow("二值图", 1);
	imshow("二值图", binary_dst);
	
	erosion_dst_omp = binary_dst.clone();
	memset(erosion_dst_omp.data, 0, binary_dst.rows * binary_dst.cols);
	gettimeofday(&start, NULL);
#pragma omp parallel for num_threads(n)
	for (int i = 0; i < binary_dst.rows; i++) {
		for (int j = 0; j < binary_dst.cols; j++) {
			if((i == 0) || (i == binary_dst.rows - 1) || (j == 0) || (j == binary_dst.cols - 1)){
				set_pixel(erosion_dst_omp, i, j, 0);
				continue;
			}
			int origin = get_pixel(binary_dst, i, j);
			int upper = get_pixel(binary_dst, i, j - 1);
			int left = get_pixel(binary_dst, i - 1, j);
			int lower = get_pixel(binary_dst, i, j + 1);
			int right = get_pixel(binary_dst, i + 1, j);
			if (upper && origin && left && lower && right) {
				set_pixel(erosion_dst_omp, i, j, 255);
			}
			else {
				set_pixel(erosion_dst_omp, i, j, 0);
			}
		}
	}
	gettimeofday(&end, NULL);
	omp_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	omp_timeuse /= 1000;
	printf("The Openmp const time is %f ms\n", omp_timeuse);

	erosion_dst = binary_dst.clone();
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
	serial_timeuse /= 1000;
	printf("The Serial const time is %f ms\n", serial_timeuse);

	namedWindow("腐蚀(serial)", 1);
	imshow("腐蚀(serial)", erosion_dst);

	namedWindow("腐蚀(openmp)", 1);
	imshow("腐蚀(openmp)", erosion_dst_omp);

	for (int i = 1; i < binary_dst.rows - 1; i++) {
		for (int j = 1; j < binary_dst.cols - 1; j++) {
			if(get_pixel(erosion_dst, i, j) != get_pixel(erosion_dst_omp, i, j)) {
				printf("Error!\n");
			}
		}
	}
	printf("Openmp compute successfully\n");
	
//	waitKey();
	return 0;
}
