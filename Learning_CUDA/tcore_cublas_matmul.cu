#include "cublas_matmul.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace cublas_tcore_mat
{

    void initMatrix(half* m_half, float* m_float, int r, int c)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                float val = static_cast<float>(rand() % 3);
                m_half[i * c + j] = __float2half(val);
                m_float[i * c + j] = val;
            }
        }
    }

    void cpuMult(const float* a, const float* b, float* c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    bool verify(const float* cpu, const float* gpu)
    {
        for (int i = 0; i < M * N; i++)
        {
            if (fabsf(cpu[i] - gpu[i]) > MARGIN)
                return false;
        }
        return true;
    }

    void matmul()
    {
        //---------------- Pinned Host Memory (Fast DMA Transfers) ----------------//

        half* a_half, * b_half;
        float* a_float, * b_float, * c_cpu, * d_gpu;

        cudaMallocHost((void**)&a_half, sizeof(half) * M * K);
        cudaMallocHost((void**)&b_half, sizeof(half) * K * N);
        cudaMallocHost((void**)&a_float, sizeof(float) * M * K);
        cudaMallocHost((void**)&b_float, sizeof(float) * K * N);
        cudaMallocHost((void**)&c_cpu, sizeof(float) * M * N);
        cudaMallocHost((void**)&d_gpu, sizeof(float) * M * N);

        initMatrix(a_half, a_float, M, K);
        initMatrix(b_half, b_float, K, N);

        //---------------- Device Memory Allocation ----------------//

        half* da, * db;
        float* dd;

        cudaMalloc(&da, sizeof(half) * M * K);
        cudaMalloc(&db, sizeof(half) * K * N);
        cudaMalloc(&dd, sizeof(float) * M * N);

        //---------------- cuBLAS Handle & Tensor Core Setup ----------------//

        cublasHandle_t handle;
        cublasCreate(&handle);

        // Force Tensor Core execution mode explicitly (Overrides CUDA 12/13 pedantic defaults)
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        // Allocate 32 MB workspace buffer for high-performance Tensor Core heuristics
        void* workspace = nullptr;
        size_t workspaceSize = 32 * 1024 * 1024;
        cudaMalloc(&workspace, workspaceSize);
        cublasSetWorkspace(handle, workspace, workspaceSize);

        float alpha = 1.0f;
        float beta = 0.0f;

        //---------------- GPU Warmup (Exactly 10 Rounds) ----------------//

        cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);

        for (int i = 0; i < 10; i++)
        {
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                db, CUDA_R_16F, N,
                da, CUDA_R_16F, K,
                &beta,
                dd, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }

        cudaDeviceSynchronize();

        //---------------- CPU Calculation for Verification (No Warmup) ----------------//

        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMult(a_float, b_float, c_cpu, M, K, N);
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        //---------------- CUDA Events Setup ----------------//

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float h2dTime = 0.0f;
        float kernelTime = 0.0f;
        float d2hTime = 0.0f;

        //---------------- Host -> Device Transfer ----------------//

        cudaEventRecord(start);

        cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&h2dTime, start, stop);

        //---------------- High-Speed Tensor Core GEMM Execution ----------------//

        cudaEventRecord(start);

        for (int i = 0; i < 100; i++)
        {
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                db, CUDA_R_16F, N,
                da, CUDA_R_16F, K,
                &beta,
                dd, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&kernelTime, start, stop);
        kernelTime /= 100.0f;

        //---------------- Device -> Host Transfer ----------------//

        cudaEventRecord(start);

        cudaMemcpy(d_gpu, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&d2hTime, start, stop);

        //---------------- Throughput & Bandwidth Calculations ----------------//

        float totalGPU = h2dTime + kernelTime + d2hTime;

        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(half);
        double d2hBytes = static_cast<double>(M) * N * sizeof(float);
        double totalBytes = h2dBytes + d2hBytes;

        double kernelGFLOPS = (kernelTime > 0.0f) ? (totalFLOP / (kernelTime * 1e-3) / 1e9) : 0.0;
        double kernelTFLOPS = kernelGFLOPS / 1000.0;
        double gpuTotalTFLOPS = (totalGPU > 0.0f) ? (totalFLOP / (totalGPU * 1e-3) / 1e12) : 0.0;
        double cpuGFLOPS = (cpuTime > 0.0) ? (totalFLOP / (cpuTime * 1e-3) / 1e9) : 0.0;

        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double totalBW = (totalGPU > 0.0f) ? (totalBytes / (totalGPU * 1e-3) / 1e9) : 0.0;

        bool correct = verify(c_cpu, d_gpu);

        //---------------- Results Output ----------------//

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n============== Performance ==============\n";
        std::cout << "CPU Total           : " << cpuTime << " ms\n\n";

        std::cout << "GPU H2D Copy        : " << h2dTime << " ms\n";
        std::cout << "GPU cuBLAS (Tensor) : " << kernelTime << " ms\n";
        std::cout << "GPU D2H Copy        : " << d2hTime << " ms\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "GPU Total           : " << totalGPU << " ms\n";
        std::cout << "=========================================\n";

        std::cout << "\nVerification        : "
            << std::boolalpha << correct << "\n";

        std::cout << "=========================================\n";
        std::cout << "         THROUGHPUT & BANDWIDTH          \n";
        std::cout << "=========================================\n";
        std::cout << "GPU H2D Bandwidth   : " << h2dBW << " GB/s\n";
        std::cout << "GPU D2H Bandwidth   : " << d2hBW << " GB/s\n";
        std::cout << "GPU Kernel Compute  : " << kernelTFLOPS << " TFLOPS\n";
        std::cout << "GPU Total Compute   : " << gpuTotalTFLOPS << " TFLOPS\n";
        std::cout << "GPU Overall Pipeline: " << totalBW << " GB/s\n";
        std::cout << "CPU Compute         : " << cpuGFLOPS << " GFLOPS\n";
        std::cout << "=========================================\n";

        //---------------- Cleanup ----------------//

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cublasDestroy(handle);

        cudaFree(workspace);
        cudaFree(da);
        cudaFree(db);
        cudaFree(dd);

        cudaFreeHost(a_half);
        cudaFreeHost(b_half);
        cudaFreeHost(a_float);
        cudaFreeHost(b_float);
        cudaFreeHost(c_cpu);
        cudaFreeHost(d_gpu);
    }

} // namespace cublas_tcore_mat