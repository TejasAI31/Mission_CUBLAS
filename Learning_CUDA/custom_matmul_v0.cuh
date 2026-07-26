#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define M 4096
#define K 4096
#define N 4096

#define BLOCKSIZE 32
#define TILESIZE BLOCKSIZE
#define SHMEM_SIZE TILESIZE*TILESIZE

namespace custmat_v0 {
    __global__ void gpuMult(int* da, int* db, int* dc, int m, int k, int n);
    void matmul();
    void printMatrix(int* m, int w, int h);
    void initMatrix(int* m, int w, int h);
}