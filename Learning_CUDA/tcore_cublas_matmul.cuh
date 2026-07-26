#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

#define M 1024
#define K 1024
#define N 1024
#define MARGIN 1e-4

#define BLOCKSIZE 32

using namespace std;

namespace cublas_tcore_mat {

	void matmul();
	void printMatrix(float* m, int w, int h);
	void initMatrix(float* m, int w, int h);
}