#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

#define M 4096
#define K 4096
#define N 4096

constexpr int tilesize = 16;

using namespace std;

namespace tcore_bmat {
	__global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n);

	void matmul();
	void printMatrix(half* m, int w, int h);
	void initMatrix(half* m_half, float* m_float, int r, int c);
}