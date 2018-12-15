#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int vector_a[10];
int vector_b[10];
int vector_result[10];

void *plus_pthread(void *arg) {
	int i = *(int *)arg;
	vector_result[i] = vector_a[i] + vector_b[i];
	printf("Process %d: vector_result[%d] = vector_a[%d] + vector_b[%d]\n", i, i, i, i);
}

int main() {
	int ret, i;
	pthread_t tid[10];
	int attr[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	for (i = 0; i < 10; i++) {
		vector_a[i] = rand() % 100;
		vector_b[i] = rand() % 100;
	}
	for (i = 0; i < 10; i++) {
		ret = pthread_create(&tid[i], NULL, &plus_pthread, &attr[i]);
	}
	for (i = 0; i < 10; i++) {
		pthread_join(tid[i], NULL);
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
