#include "cublas_matmul.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace cublasmat
{

    void initMatrix(float* m, int r, int c)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                m[i * c + j] = static_cast<float>(rand() % 3);
            }
        }
    }

    void printMatrix(float* m, int r, int c)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                std::cout << m[i * c + j] << " ";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    void cpuMult(float* a, float* b, float* c, int m, int k, int n)
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

    bool verify(float* cpu, float* gpu)
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
        //---------------- Host Memory ----------------//

        float* a, * b, * c_cpu, * d_gpu;

        a = (float*)malloc(sizeof(float) * M * K);
        b = (float*)malloc(sizeof(float) * K * N);
        c_cpu = (float*)malloc(sizeof(float) * M * N);
        d_gpu = (float*)malloc(sizeof(float) * M * N);

        initMatrix(a, M, K);
        initMatrix(b, K, N);

        //---------------- CPU Warmup & Timing ----------------//

        for (int i = 0; i < 10; ++i) {
            cpuMult(a, b, c_cpu, M, K, N);
        }

        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMult(a, b, c_cpu, M, K, N);
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        //---------------- Device Memory ----------------//

        float* da, * db, * dd;

        cudaMalloc(&da, sizeof(float) * M * K);
        cudaMalloc(&db, sizeof(float) * K * N);
        cudaMalloc(&dd, sizeof(float) * M * N);

        //---------------- cuBLAS Setup ----------------//

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        //---------------- Warmup ----------------//

        cudaMemcpy(da, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        for (int i = 0; i < 100; i++)
        {
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                db,
                N,
                da,
                K,
                &beta,
                dd,
                N);
        }

        cudaDeviceSynchronize();

        //---------------- CUDA Events ----------------//

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float h2dTime = 0.0f;
        float kernelTime = 0.0f;
        float d2hTime = 0.0f;

        //---------------- Host -> Device ----------------//

        cudaEventRecord(start);

        cudaMemcpy(da, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&h2dTime, start, stop);

        //---------------- cuBLAS SGEMM Execution ----------------//
        // Row-major multiplication:
        // C = A * B
        //
        // cuBLAS assumes column-major, so compute:
        // C^T = B^T * A^T

        cudaEventRecord(start);

        for (int i = 0; i < 100; i++)
        {
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                db,
                N,
                da,
                K,
                &beta,
                dd,
                N);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&kernelTime, start, stop);
        kernelTime /= 100.0f;

        //---------------- Device -> Host ----------------//

        cudaEventRecord(start);

        cudaMemcpy(d_gpu, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&d2hTime, start, stop);

        //---------------- Throughput & Bandwidth Calculations ----------------//

        float totalGPU = h2dTime + kernelTime + d2hTime;

        // Total Floating-Point Operations (2 * M * N * K)
        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

        // Memory Bytes Transferred (FP32 = 4 bytes per element)
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(float);
        double d2hBytes = static_cast<double>(M) * N * sizeof(float);
        double totalBytes = h2dBytes + d2hBytes;

        // Compute Throughput
        double kernelGFLOPS = (kernelTime > 0.0f) ? (totalFLOP / (kernelTime * 1e-3) / 1e9) : 0.0;
        double kernelTFLOPS = kernelGFLOPS / 1000.0;
        double gpuTotalTFLOPS = (totalGPU > 0.0f) ? (totalFLOP / (totalGPU * 1e-3) / 1e12) : 0.0;
        double cpuGFLOPS = (cpuTime > 0.0) ? (totalFLOP / (cpuTime * 1e-3) / 1e9) : 0.0;

        // Memory Bandwidth (GB/s)
        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double totalBW = (totalGPU > 0.0f) ? (totalBytes / (totalGPU * 1e-3) / 1e9) : 0.0;

        // Verification check
        bool correct = verify(c_cpu, d_gpu);

        //---------------- Results Output ----------------//

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n============== Performance ==============\n";
        std::cout << "CPU Total           : " << cpuTime << " ms\n\n";

        std::cout << "GPU H2D Copy        : " << h2dTime << " ms\n";
        std::cout << "GPU SGEMM (cuBLAS)  : " << kernelTime << " ms\n";
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
        std::cout << "GPU Kernel Compute  : " << kernelGFLOPS << " GFLOPS (" << kernelTFLOPS << " TFLOPS)\n";
        std::cout << "GPU Total Compute   : " << gpuTotalTFLOPS << " TFLOPS\n";
        std::cout << "GPU Overall Pipeline: " << totalBW << " GB/s\n";
        std::cout << "CPU Compute         : " << cpuGFLOPS << " GFLOPS\n";
        std::cout << "=========================================\n";

        //---------------- Cleanup ----------------//

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cublasDestroy(handle);

        cudaFree(da);
        cudaFree(db);
        cudaFree(dd);

        free(a);
        free(b);
        free(c_cpu);
        free(d_gpu);
    }

}