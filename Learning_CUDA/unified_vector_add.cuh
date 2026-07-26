#pragma once
#include <iostream>
#include <chrono>
#include <cmath>


#define N 10000000
#define THREADS 128

using namespace std;

namespace uva {
	__global__ void gpuPerformAdd(int* da, int* db, int* dc);

	void matrixInit(int* arr);
	void cpuAdd(int* ha, int* hb, int* hc);
	void gpuAdd(int* ha, int* hb, int* hc);
	bool verifyArrays(int* a, int* b);
	void vector_add();
}