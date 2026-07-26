#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

#define N 10000

namespace tsort {
	__global__ void oddEvenKernel(float* a, int n, int phase);

	void initArray(float* a, int n);
	void sort();
}