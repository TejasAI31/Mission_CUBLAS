#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define M 1000
#define K 1000
#define N 1000

#define BLOCKSIZE 32
#define TILESIZE BLOCKSIZE
#define SHMEM_SIZE TILESIZE*TILESIZE*4

using namespace std;

namespace tmat {
	__global__ void gpuMult(int* da, int* db, int* dc, int m, int k, int n);

	void matmul();
	void printMatrix(int* m, int w, int h);
	void initMatrix(int* m, int w, int h);
}