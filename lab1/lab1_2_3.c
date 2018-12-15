#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "omp.h"
int main() {
	int i;
	struct timeval start, end;
	int vector_a[100000], vector_b[100000], vector_result[100000];
	int vector_result_serial[100000];
	double openmp_timeuse, serial_timeuse;
	for (i = 0; i < 100000; i++) {
		vector_a[i] = rand() % 100;
		vector_b[i] = rand() % 100;
	}
    
	gettimeofday(&start, NULL);
	for (i = 0; i < 100000; i++) {
		vector_result_serial[i] = vector_a[i] + vector_b[i];
	}
	gettimeofday(&end, NULL);
	serial_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	serial_timeuse /= 1000;

	gettimeofday(&start, NULL);
#pragma omp parallel for
	for (i = 0; i < 100000; i++) {
		vector_result[i] = vector_a[i] + vector_b[i];
	}
	gettimeofday(&end, NULL);
	openmp_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	openmp_timeuse /= 1000;
	printf("\nThe Openmp const time is %f ms", openmp_timeuse);
	printf("\nThe Serial const time is %f ms", serial_timeuse);
	return 0;
}