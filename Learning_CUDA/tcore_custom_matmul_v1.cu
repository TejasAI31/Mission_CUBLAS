#include "tcore_custom_matmul_v1.cuh"
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mma.h>
#include <cublas_v2.h>

namespace tcore_custmat_v1 {

    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n)
    {
        // 128 threads = 4 warps arranged in a 2x2 spatial warp grid
        int warpid = threadIdx.x / 32;
        int warpx = warpid % 2;
        int warpy = warpid / 2;

        // Shared memory workspace
        __shared__ half As[blcksize][tilesize]; // 32 x 16
        __shared__ half Bs[tilesize][blcksize]; // 16 x 32

        int cRow = blockIdx.y * blcksize + (warpy * tilesize);
        int cCol = blockIdx.x * blcksize + (warpx * tilesize);

        if (cRow >= m || cCol >= n) return;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tilesize, tilesize, tilesize, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tilesize, tilesize, tilesize, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tilesize, tilesize, tilesize, float> c_frag;

        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        int tid = threadIdx.x; // 0..127

        for (int i = 0; i < k; i += tilesize)
        {
            int blockRowOffset = blockIdx.y * blcksize;
            int blockColOffset = blockIdx.x * blcksize;

            // 1. Cooperative Load Matrix A tile [32 x 16] using 128 threads
            int aRow = blockRowOffset + (tid / 4);
            int aCol = i + (tid % 4) * 4;
            if (aRow < m && aCol < k) {
                *reinterpret_cast<int2*>(&As[tid / 4][(tid % 4) * 4]) =
                    *reinterpret_cast<const int2*>(&da[aRow * k + aCol]);
            }

            // 2. Cooperative Load Matrix B tile [16 x 32] using 128 threads
            int bRow = i + (tid / 8);
            int bCol = blockColOffset + (tid % 8) * 4;
            if (bRow < k && bCol < n) {
                *reinterpret_cast<int2*>(&Bs[tid / 8][(tid % 8) * 4]) =
                    *reinterpret_cast<const int2*>(&db[bRow * n + bCol]);
            }

            __syncthreads();

            // 3. Load fragments from Shared Memory & Perform Tensor Core MMA
            nvcuda::wmma::load_matrix_sync(a_frag, &As[warpy * tilesize][0], tilesize);
            nvcuda::wmma::load_matrix_sync(b_frag, &Bs[0][warpx * tilesize], blcksize);

            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        nvcuda::wmma::store_matrix_sync(&dc[cRow * n + cCol], c_frag, n, nvcuda::wmma::mem_row_major);
    }

    void initMatrix(half* m_half, float* m_float, int r, int c)
    {
        for (int i = 0; i < r * c; ++i) {
            float val = static_cast<float>(rand() % 3);
            m_half[i] = __float2half(val);
            m_float[i] = val;
        }
    }

    void printMatrix(half* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++) {
            std::cout << __half2float(m[i]) << " ";
            if ((i + 1) % c == 0) std::cout << '\n';
        }
        std::cout << '\n';
    }

    void printMatrix(float* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++) {
            std::cout << m[i] << " ";
            if ((i + 1) % c == 0) std::cout << '\n';
        }
        std::cout << '\n';
    }

    // Verification against cuBLAS Reference Matrix Output
    bool verify(const float* custom_res, const float* cublas_res, int m, int n, int k)
    {
        float eps = 1e-1f * static_cast<float>(k);
        for (int i = 0; i < m * n; i++) {
            if (std::fabs(custom_res[i] - cublas_res[i]) > eps)
                return false;
        }
        return true;
    }

    void matmul()
    {
        // Pinned Host Memory Allocation
        half* a_half, * b_half;
        float* a_float, * b_float, * c_custom, * c_cublas;

        cudaMallocHost((void**)&a_half, sizeof(half) * M * K);
        cudaMallocHost((void**)&b_half, sizeof(half) * K * N);
        cudaMallocHost((void**)&a_float, sizeof(float) * M * K);
        cudaMallocHost((void**)&b_float, sizeof(float) * K * N);
        cudaMallocHost((void**)&c_custom, sizeof(float) * M * N);
        cudaMallocHost((void**)&c_cublas, sizeof(float) * M * N);

        initMatrix(a_half, a_float, M, K);
        initMatrix(b_half, b_float, K, N);

        // Device Memory Allocation
        half* da, * db;
        float* dc_custom, * dc_cublas;
        cudaMalloc(&da, sizeof(half) * M * K);
        cudaMalloc(&db, sizeof(half) * K * N);
        cudaMalloc(&dc_custom, sizeof(float) * M * N);
        cudaMalloc(&dc_cublas, sizeof(float) * M * N);

        // cuBLAS Setup
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        void* workspace = nullptr;
        size_t workspaceSize = 32 * 1024 * 1024;
        cudaMalloc(&workspace, workspaceSize);
        cublasSetWorkspace(handle, workspace, workspaceSize);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Launch Configuration
        dim3 blocksize(128);
        dim3 gridsize((N + blcksize - 1) / blcksize, (M + blcksize - 1) / blcksize);

        // Warmup Custom Kernel and cuBLAS (10 rounds)
        cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);

        for (int i = 0; i < 10; ++i) {
            gpuMult << <gridsize, blocksize >> > (da, db, dc_custom, M, K, N);
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                db, CUDA_R_16F, N,
                da, CUDA_R_16F, K,
                &beta,
                dc_cublas, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
        cudaDeviceSynchronize();

        // Compute cuBLAS Reference Output for Verification
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            db, CUDA_R_16F, N,
            da, CUDA_R_16F, K,
            &beta,
            dc_cublas, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT
        );
        cudaMemcpy(c_cublas, dc_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        // CUDA Events Setup
        cudaEvent_t startH2D, stopH2D, startCustom, stopCustom, startCublas, stopCublas, startD2H, stopD2H;
        cudaEventCreate(&startH2D);    cudaEventCreate(&stopH2D);
        cudaEventCreate(&startCustom); cudaEventCreate(&stopCustom);
        cudaEventCreate(&startCublas); cudaEventCreate(&stopCublas);
        cudaEventCreate(&startD2H);    cudaEventCreate(&stopD2H);

        // Host to Device
        cudaEventRecord(startH2D);
        cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);
        cudaEventSynchronize(stopH2D);

        // Custom Kernel Execution Benchmark (100 iterations)
        cudaEventRecord(startCustom);
        for (int i = 0; i < 100; ++i) {
            gpuMult << <gridsize, blocksize >> > (da, db, dc_custom, M, K, N);
        }
        cudaEventRecord(stopCustom);
        cudaEventSynchronize(stopCustom);

        // cuBLAS Execution Benchmark (100 iterations)
        cudaEventRecord(startCublas);
        for (int i = 0; i < 100; ++i) {
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                db, CUDA_R_16F, N,
                da, CUDA_R_16F, K,
                &beta,
                dc_cublas, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
        cudaEventRecord(stopCublas);
        cudaEventSynchronize(stopCublas);

        // Device to Host
        cudaEventRecord(startD2H);
        cudaMemcpy(c_custom, dc_custom, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopD2H);
        cudaEventSynchronize(stopD2H);

        // Time Calculations
        float h2dTime = 0.0f, customKernelTime = 0.0f, cublasKernelTime = 0.0f, d2hTime = 0.0f;
        cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
        cudaEventElapsedTime(&customKernelTime, startCustom, stopCustom);
        cudaEventElapsedTime(&cublasKernelTime, startCublas, stopCublas);
        cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);

        customKernelTime /= 100.0f;
        cublasKernelTime /= 100.0f;

        // Verify Custom Output against cuBLAS Reference
        bool correct = verify(c_custom, c_cublas, M, N, K);

        // Performance Metrics
        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(half);
        double d2hBytes = static_cast<double>(M) * N * sizeof(float);
        double totalBytes = h2dBytes + d2hBytes;

        float customTotalTime = h2dTime + customKernelTime + d2hTime;
        float cublasTotalTime = h2dTime + cublasKernelTime + d2hTime;

        double customTFLOPS = (customKernelTime > 0.0f) ? (totalFLOP / (customKernelTime * 1e-3) / 1e12) : 0.0;
        double cublasTFLOPS = (cublasKernelTime > 0.0f) ? (totalFLOP / (cublasKernelTime * 1e-3) / 1e12) : 0.0;

        double customTotalTFLOPS = (customTotalTime > 0.0f) ? (totalFLOP / (customTotalTime * 1e-3) / 1e12) : 0.0;
        double cublasTotalTFLOPS = (cublasTotalTime > 0.0f) ? (totalFLOP / (cublasTotalTime * 1e-3) / 1e12) : 0.0;

        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double customTotalBW = (customTotalTime > 0.0f) ? (totalBytes / (customTotalTime * 1e-3) / 1e9) : 0.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=========================================\n";
        std::cout << "GPU H2D Copy          : " << h2dTime << " ms\n";
        std::cout << "Custom Kernel GEMM    : " << customKernelTime << " ms\n";
        std::cout << "cuBLAS Tensor GEMM    : " << cublasKernelTime << " ms\n";
        std::cout << "GPU D2H Copy          : " << d2hTime << " ms\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "Custom Total Pipeline : " << customTotalTime << " ms\n";
        std::cout << "cuBLAS Total Pipeline : " << cublasTotalTime << " ms\n";
        std::cout << "=========================================\n";
        std::cout << "Verification vs cuBLAS: " << (correct ? "PASSED" : "FAILED") << "\n";
        std::cout << "=========================================\n";
        std::cout << "         THROUGHPUT & BANDWIDTH          \n";
        std::cout << "=========================================\n";
        std::cout << "GPU H2D Bandwidth     : " << h2dBW << " GB/s\n";
        std::cout << "GPU D2H Bandwidth     : " << d2hBW << " GB/s\n";
        std::cout << "Custom Kernel Compute : " << customTFLOPS << " TFLOPS\n";
        std::cout << "cuBLAS Kernel Compute : " << cublasTFLOPS << " TFLOPS\n";
        std::cout << "Custom Efficiency     : " << (cublasTFLOPS > 0.0 ? (customTFLOPS / cublasTFLOPS * 100.0) : 0.0) << " % of cuBLAS\n";
        std::cout << "Custom Total Compute  : " << customTotalTFLOPS << " TFLOPS\n";
        std::cout << "cuBLAS Total Compute  : " << cublasTotalTFLOPS << " TFLOPS\n";
        std::cout << "Custom Total Pipeline : " << customTotalBW << " GB/s\n";
        std::cout << "=========================================\n";

        // Cleanup
        cudaEventDestroy(startH2D);    cudaEventDestroy(stopH2D);
        cudaEventDestroy(startCustom); cudaEventDestroy(stopCustom);
        cudaEventDestroy(startCublas); cudaEventDestroy(stopCublas);
        cudaEventDestroy(startD2H);    cudaEventDestroy(stopD2H);

        cublasDestroy(handle);
        cudaFree(workspace);

        cudaFree(da); cudaFree(db);
        cudaFree(dc_custom); cudaFree(dc_cublas);

        cudaFreeHost(a_half); cudaFreeHost(b_half);
        cudaFreeHost(a_float); cudaFreeHost(b_float);
        cudaFreeHost(c_custom); cudaFreeHost(c_cublas);
    }

} // namespace tcore_custmat_v1