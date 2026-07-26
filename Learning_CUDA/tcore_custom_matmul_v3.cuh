#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define MDIM 4096
#define KDIM 4096
#define NDIM 4096

using namespace std;

namespace tcore_custmat_v3 {
    constexpr int tilesize = 16;
    constexpr int blcksize = 128; // 128x128 block tile size

    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n);

    void matmul();
    void printMatrix(half* m, int r, int c);
    void printMatrix(float* m, int r, int c);
    void initMatrix(half* m_half, float* m_float, int r, int c);
    bool verify(const float* custom_res, const float* cublas_res, int m, int n, int k);
}