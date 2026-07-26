#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define MDIM 4096
#define KDIM 4096
#define NDIM 4096

namespace tcore_custmat_v4 {
    void matmul();
    void initMatrix(half* m_half, float* m_float, int r, int c);
    bool verify(const float* custom_res, const float* cublas_res, int m, int n, int k);
}