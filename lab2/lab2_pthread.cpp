#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

using namespace cv;
int n; 
/// 全局变量
Mat src, erosion_dst, erosion_dst_pthread, binary_dst;

int get_pixel(Mat &img, int i, int j) {
	return *(img.data + img.step[0] * i + img.step[1] * j);
}

void set_pixel(Mat &img, int i, int j, int color) { 
	*(img.data + img.step[0] * i + img.step[1] * j) = color;
}

void *Erosion(void *args) {
	int k = *(int *)args;
	int step = ceil(binary_dst.rows / n);
	for (int i = k * step; (i < binary_dst.rows) && (i < (k * step + step)); i++) {
		for (int j = 0; j < binary_dst.cols; j++) {
			if((i == 0) || (i == binary_dst.rows - 1) || (j == 0) || (j == binary_dst.cols - 1)){
				set_pixel(erosion_dst_pthread, i, j, 0);
				continue;
			}
			int origin = get_pixel(binary_dst, i, j);
			int upper = get_pixel(binary_dst, i, j - 1);
			int left = get_pixel(binary_dst, i - 1, j);
			int lower = get_pixel(binary_dst, i, j + 1);
			int right = get_pixel(binary_dst, i + 1, j);
			if (upper && origin && left && lower && right) {
				set_pixel(erosion_dst_pthread, i, j, 255);
			}
			else {
				set_pixel(erosion_dst_pthread, i, j, 0);
			}
		}
	}
}

int main(int argc, char** argv)
{
	n = atoi(argv[1]);
	struct timeval start, end;
	double pthread_timeuse, serial_timeuse;
	    
	int *attr = (int *)malloc(sizeof(int) * n);
	pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * n);
	
	src = imread("test.png", 0);
	binary_dst = src.clone();
	threshold(src, binary_dst, 150, 255, CV_THRESH_BINARY);
	namedWindow("二值图", 1);
	imshow("二值图", binary_dst);
	
	erosion_dst_pthread = binary_dst.clone();
	memset(erosion_dst_pthread.data, 0, binary_dst.rows * binary_dst.cols);
	gettimeofday(&start, NULL);
	for (int i = 0; i < n; i++) {
		attr[i] = i;
		pthread_create(&tid[i], NULL, &Erosion, &attr[i]);
	}
	for(int i = 0; i < n; i++){
		pthread_join(tid[i], NULL);
	}
	gettimeofday(&end, NULL);
	pthread_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	pthread_timeuse /= 1000;
	printf("The Pthread const time is %f ms\n", pthread_timeuse);

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

	namedWindow("腐蚀(pthread)", 1);
	imshow("腐蚀(pthread)", erosion_dst_pthread);

	for (int i = 1; i < binary_dst.rows - 1; i++) {
		for (int j = 1; j < binary_dst.cols - 1; j++) {
			if(get_pixel(erosion_dst, i, j) != get_pixel(erosion_dst_pthread, i, j)) {
				printf("Error!\n");
			}
		}
	}
	printf("Pthread compute successfully\n");
	
	waitKey();
	free(tid);
	free(attr);
	return 0;
}
