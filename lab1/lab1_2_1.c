#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
int main() {
	int i;
	int vector_a[10], vector_b[10], vector_result[10];

	for (i = 0; i < 10; i++) {
		vector_a[i] = rand() % 100;
		vector_b[i] = rand() % 100;
	}

#pragma omp parallel for
	for (i = 0; i < 10; i++) {
		vector_result[i] = vector_a[i] + vector_b[i];
		printf("Process %d: vector_result[%d] = vector_a[%d] + vector_b[%d]\n", omp_get_thread_num(), i, i, i);
	}
	printf("The Vector_a is:");
	for (i = 0; i < 10; i++) {
		printf(" %d", vector_a[i]);
	}
	printf("\nThe Vector_b is:");
	for (i = 0; i < 10; i++) {
		printf(" %d", vector_b[i]);
	}
	printf("\nThe Vector_result is:");
	for (i = 0; i < 10; i++) {
		printf(" %d", vector_result[i]);
	}
	printf("\n");
	return 0;
}