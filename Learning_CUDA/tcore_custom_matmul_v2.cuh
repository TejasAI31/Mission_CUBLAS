#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define M 4096
#define K 4096
#define N 4096

using namespace std;

namespace tcore_custmat_v2 {
    constexpr int tilesize = 16;
    constexpr int blcksize = 128; // Scaled up for 128x128 block tiles

    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n);

    void matmul();
    void printMatrix(half* m, int r, int c);
    void printMatrix(float* m, int r, int c);
    void initMatrix(half* m_half, float* m_float, int r, int c);
    bool verify(const float* custom_res, const float* cublas_res, int m, int n, int k);
}