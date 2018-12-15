#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// 全局变量
Mat src, erosion_dst, binary_dst;


int get_pixel(Mat &img, int i, int j) {
	return *(img.data + img.step[0] * i + img.step[1] * j);
}

void set_pixel(Mat &img, int i, int j, int color) { 
	*(img.data + img.step[0] * i + img.step[1] * j) = color;
}

int main(int argc, char** argv)
{
	src = imread("test.bmp", 0);
	if (!src.data) {
		printf("Unable to read image!\n");
		return -1;
	}
	namedWindow("原图", 1);
	imshow("原图", src);
	binary_dst = src.clone();
	threshold(src, binary_dst, 150, 255, CV_THRESH_BINARY);
	//show the binary image
	namedWindow("二值图", 1);
	imshow("二值图", binary_dst);
	
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
	//show the erosion image
	namedWindow("腐蚀", 1);
	imshow("腐蚀", erosion_dst);
	
//	waitKey();
	return 0;
}
