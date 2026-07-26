#include "custom_matmul_v0.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>

namespace custmat_v0 {

    void initMatrix(int* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++)
        {
            m[i] = rand() % 3;
        }
    }

    __global__ void gpuMult(int* da, int* db, int* dc, int m, int k, int n)
    {
        __shared__ int A[SHMEM_SIZE];
        __shared__ int B[SHMEM_SIZE];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;

        int sum = 0;
        for (int i = 0; i < (k + TILESIZE - 1) / TILESIZE; i++)
        {
            int tiledCol = i * TILESIZE + tx;
            int tiledRow = i * TILESIZE + ty;

            if (row < m && tiledCol < k)
                A[ty * TILESIZE + tx] = da[row * k + tiledCol];
            else
                A[ty * TILESIZE + tx] = 0;

            if (tiledRow < k && col < n)
                B[ty * TILESIZE + tx] = db[tiledRow * n + col];
            else
                B[ty * TILESIZE + tx] = 0;

            __syncthreads();

            for (int j = 0; j < TILESIZE; j++)
            {
                sum += A[ty * TILESIZE + j] * B[tx + j * TILESIZE];
            }

            __syncthreads();
        }
        if (row < m && col < n)
            dc[row * n + col] = sum;
    }

    void cpuMult(int* a, int* b, int* c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    bool verify(const float* custom_res, const float* cublas_res, int m, int n)
    {
        float eps = 1e-1f;
        for (int i = 0; i < m * n; i++)
        {
            if (std::fabs(custom_res[i] - cublas_res[i]) > eps)
                return false;
        }
        return true;
    }

    void matmul()
    {
        int* a, * b;
        float* c_custom, * c_cublas;

        a = (int*)malloc(sizeof(int) * M * K);
        b = (int*)malloc(sizeof(int) * K * N);
        c_custom = (float*)malloc(sizeof(float) * M * N);
        c_cublas = (float*)malloc(sizeof(float) * M * N);

        initMatrix(a, M, K);
        initMatrix(b, K, N);

        // Convert int to float for GPU
        float* a_float, * b_float;
        a_float = (float*)malloc(sizeof(float) * M * K);
        b_float = (float*)malloc(sizeof(float) * K * N);
        for (int i = 0; i < M * K; i++) a_float[i] = (float)a[i];
        for (int i = 0; i < K * N; i++) b_float[i] = (float)b[i];

        //---------------- GPU Setup ----------------//
        int* da_int, * db_int;
        float* dc_custom_gpu, * da_float_gpu, * db_float_gpu, * dc_cublas_gpu;

        dim3 blocksize(BLOCKSIZE, BLOCKSIZE);
        dim3 gridsize(
            (N + BLOCKSIZE - 1) / BLOCKSIZE,
            (M + BLOCKSIZE - 1) / BLOCKSIZE);

        cudaEvent_t startH2D, stopH2D, startCustom, stopCustom,
            startCublas, stopCublas, startD2H, stopD2H;
        cudaEventCreate(&startH2D);    cudaEventCreate(&stopH2D);
        cudaEventCreate(&startCustom); cudaEventCreate(&stopCustom);
        cudaEventCreate(&startCublas); cudaEventCreate(&stopCublas);
        cudaEventCreate(&startD2H);    cudaEventCreate(&stopD2H);

        //---------------- Memory Allocation ----------------//
        cudaMalloc(&da_int, sizeof(int) * M * K);
        cudaMalloc(&db_int, sizeof(int) * K * N);
        cudaMalloc(&dc_custom_gpu, sizeof(float) * M * N);
        cudaMalloc(&da_float_gpu, sizeof(float) * M * K);
        cudaMalloc(&db_float_gpu, sizeof(float) * K * N);
        cudaMalloc(&dc_cublas_gpu, sizeof(float) * M * N);

        // cuBLAS setup with Tensor Cores
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        float alpha = 1.0f;
        float beta = 0.0f;

        //---------------- Host -> Device ----------------//
        cudaEventRecord(startH2D);
        cudaMemcpy(da_int, a, sizeof(int) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db_int, b, sizeof(int) * K * N, cudaMemcpyHostToDevice);
        cudaMemcpy(da_float_gpu, a_float, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db_float_gpu, b_float, sizeof(float) * K * N, cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);
        cudaEventSynchronize(stopH2D);

        //---------------- Warm-up ----------------//
        for (int i = 0; i < 10; i++)
        {
            gpuMult << <gridsize, blocksize >> > (da_int, db_int, (int*)dc_custom_gpu, M, K, N);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, db_float_gpu, CUDA_R_32F, N,
                da_float_gpu, CUDA_R_32F, K,
                &beta, dc_cublas_gpu, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();

        //---------------- Custom Kernel Timing ----------------//
        cudaEventRecord(startCustom);
        for (int i = 0; i < 100; i++)
        {
            gpuMult << <gridsize, blocksize >> > (da_int, db_int, (int*)dc_custom_gpu, M, K, N);
        }
        cudaEventRecord(stopCustom);
        cudaEventSynchronize(stopCustom);

        //---------------- cuBLAS Tensor Core Timing ----------------//
        cudaEventRecord(startCublas);
        for (int i = 0; i < 100; i++)
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, db_float_gpu, CUDA_R_32F, N,
                da_float_gpu, CUDA_R_32F, K,
                &beta, dc_cublas_gpu, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(stopCublas);
        cudaEventSynchronize(stopCublas);

        //---------------- Device -> Host ----------------//
        cudaEventRecord(startD2H);
        cudaMemcpy(c_custom, dc_custom_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(c_cublas, dc_cublas_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopD2H);
        cudaEventSynchronize(stopD2H);

        float h2dTime, customKernelTime, cublasKernelTime, d2hTime;
        cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
        cudaEventElapsedTime(&customKernelTime, startCustom, stopCustom);
        cudaEventElapsedTime(&cublasKernelTime, startCublas, stopCublas);
        cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);

        customKernelTime /= 100.0f;
        cublasKernelTime /= 100.0f;

        //---------------- Results & Metrics ----------------//
        bool correct = verify(c_custom, c_cublas, M, N);

        float customTotalTime = h2dTime + customKernelTime + d2hTime;
        float cublasTotalTime = h2dTime + cublasKernelTime + d2hTime;

        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(float);
        double d2hBytes = static_cast<double>(M) * N * sizeof(float);
        double totalBytes = h2dBytes + d2hBytes;

        double customGFLOPS = (customKernelTime > 0.0f) ? (totalFLOP / (customKernelTime * 1e-3) / 1e9) : 0.0;
        double cublasGFLOPS = (cublasKernelTime > 0.0f) ? (totalFLOP / (cublasKernelTime * 1e-3) / 1e9) : 0.0;
        double customTotalGFLOPS = (customTotalTime > 0.0f) ? (totalFLOP / (customTotalTime * 1e-3) / 1e9) : 0.0;
        double cublasTotalGFLOPS = (cublasTotalTime > 0.0f) ? (totalFLOP / (cublasTotalTime * 1e-3) / 1e9) : 0.0;

        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double customTotalBW = (customTotalTime > 0.0f) ? (totalBytes / (customTotalTime * 1e-3) / 1e9) : 0.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=========================================\n";
        std::cout << "GPU H2D Copy         : " << h2dTime << " ms\n";
        std::cout << "Custom Kernel GEMM   : " << customKernelTime << " ms\n";
        std::cout << "cuBLAS Tensor GEMM   : " << cublasKernelTime << " ms\n";
        std::cout << "GPU D2H Copy         : " << d2hTime << " ms\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "Custom Total Pipeline: " << customTotalTime << " ms\n";
        std::cout << "cuBLAS Total Pipeline: " << cublasTotalTime << " ms\n";
        std::cout << "=========================================\n";
        std::cout << "Verification vs cuBLAS: " << (correct ? "PASSED" : "FAILED") << "\n";
        std::cout << "=========================================\n";
        std::cout << "          THROUGHPUT & BANDWIDTH          \n";
        std::cout << "=========================================\n";
        std::cout << "GPU H2D Bandwidth    : " << h2dBW << " GB/s\n";
        std::cout << "GPU D2H Bandwidth    : " << d2hBW << " GB/s\n";
        std::cout << "Custom Kernel Compute: " << customGFLOPS << " GFLOPS\n";
        std::cout << "cuBLAS Kernel Compute: " << cublasGFLOPS << " GFLOPS\n";
        std::cout << "Custom Efficiency    : " << (cublasGFLOPS > 0.0 ? (customGFLOPS / cublasGFLOPS * 100.0) : 0.0) << " % of cuBLAS\n";
        std::cout << "Custom Total Compute : " << customTotalGFLOPS << " GFLOPS\n";
        std::cout << "cuBLAS Total Compute : " << cublasTotalGFLOPS << " GFLOPS\n";
        std::cout << "Custom Total Pipeline: " << customTotalBW << " GB/s\n";
        std::cout << "=========================================\n";

        //---------------- Cleanup ----------------//
        cudaEventDestroy(startH2D);    cudaEventDestroy(stopH2D);
        cudaEventDestroy(startCustom); cudaEventDestroy(stopCustom);
        cudaEventDestroy(startCublas); cudaEventDestroy(stopCublas);
        cudaEventDestroy(startD2H);    cudaEventDestroy(stopD2H);

        cublasDestroy(handle);
        cudaFree(da_int); cudaFree(db_int); cudaFree(dc_custom_gpu);
        cudaFree(da_float_gpu); cudaFree(db_float_gpu); cudaFree(dc_cublas_gpu);

        free(a); free(b); free(c_custom); free(c_cublas);
        free(a_float); free(b_float);
    }

}