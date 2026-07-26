#include "tcore_basic_matmul.cuh"
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>

namespace tcore_bmat {

    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n)
    {
        int warpx = blockIdx.x;
        int warpy = blockIdx.y;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tilesize, tilesize, tilesize, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tilesize, tilesize, tilesize, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tilesize, tilesize, tilesize, float> c_frag;

        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        for (int i = 0; i < k; i += tilesize)
        {
            int aRow = warpy * tilesize;
            int aCol = i;
            int bRow = i;
            int bCol = warpx * tilesize;

            nvcuda::wmma::load_matrix_sync(a_frag, &da[aRow * k + aCol], k);
            nvcuda::wmma::load_matrix_sync(b_frag, &db[bRow * n + bCol], n);

            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        int cRow = warpy * tilesize;
        int cCol = warpx * tilesize;

        nvcuda::wmma::store_matrix_sync(&dc[cRow * n + cCol], c_frag, n, nvcuda::wmma::mem_row_major);
    }

    void initMatrix(half* m_half, float* m_float, int r, int c)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                float val = (float)(rand() % 3);
                m_half[i * c + j] = __float2half(val);
                m_float[i * c + j] = val;
            }
        }
    }

    void printMatrix(half* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++)
        {
            std::cout << __half2float(m[i]) << " ";
            if ((i + 1) % c == 0)
                std::cout << '\n';
        }
        std::cout << '\n';
    }

    void printMatrix(float* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++)
        {
            std::cout << m[i] << " ";
            if ((i + 1) % c == 0)
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

    bool verify(float* cpu, float* gpu, int m, int n)
    {
        for (int i = 0; i < m * n; i++)
        {
            if (std::fabs(cpu[i] - gpu[i]) > 1e-3f)
                return false;
        }
        return true;
    }

    void matmul()
    {
        half* a_half = (half*)malloc(sizeof(half) * M * K);
        half* b_half = (half*)malloc(sizeof(half) * K * N);

        float* a_float = (float*)malloc(sizeof(float) * M * K);
        float* b_float = (float*)malloc(sizeof(float) * K * N);

        float* c_gpu = (float*)malloc(sizeof(float) * M * N);
        float* c_cpu = (float*)malloc(sizeof(float) * M * N);

        initMatrix(a_half, a_float, M, K);
        initMatrix(b_half, b_float, K, N);

        half* da, * db;
        float* dc;
        cudaMalloc(&da, sizeof(half) * M * K);
        cudaMalloc(&db, sizeof(half) * K * N);
        cudaMalloc(&dc, sizeof(float) * M * N);

        dim3 blocksize(32);
        dim3 gridsize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);

        // ==================== WARMUP ROUNDS ====================
        for (int i = 0; i < 10; ++i) {
            cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
            cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);
            gpuMult << <gridsize, blocksize >> > (da, db, dc, M, K, N);
            cudaMemcpy(c_gpu, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();

        // ========================================================

        // CUDA Timing Events setup
        cudaEvent_t startH2D, stopH2D;
        cudaEvent_t startKernel, stopKernel;
        cudaEvent_t startD2H, stopD2H;

        cudaEventCreate(&startH2D);    cudaEventCreate(&stopH2D);
        cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
        cudaEventCreate(&startD2H);    cudaEventCreate(&stopD2H);

        // 1. Host to Device Transfer
        cudaEventRecord(startH2D);
        cudaMemcpy(da, a_half, sizeof(half) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * K * N, cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);

        // 2. Kernel Execution
        cudaEventRecord(startKernel);
        gpuMult << <gridsize, blocksize >> > (da, db, dc, M, K, N);
        cudaEventRecord(stopKernel);

        // 3. Device to Host Transfer
        cudaEventRecord(startD2H);
        cudaMemcpy(c_gpu, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopD2H);

        cudaEventSynchronize(stopD2H);

        // Calculate GPU timings in milliseconds
        float h2dTime = 0.0f, kernelTime = 0.0f, d2hTime = 0.0f;
        cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
        cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
        cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);

        // CPU Execution timing
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMult(a_float, b_float, c_cpu, M, K, N);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

        // Verification check
        bool correct = verify(c_cpu, c_gpu, M, N);

        // ==================== THROUGHPUT CALCULATIONS ====================
        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(half);
        double d2hBytes = static_cast<double>(M) * N * sizeof(float);
        double totalBytes = h2dBytes + d2hBytes;

        float gpuTotalTime = h2dTime + kernelTime + d2hTime;

        // TFLOPS / GFLOPS (Compute)
        double kernelTFLOPS = (kernelTime > 0.0f) ? (totalFLOP / (kernelTime * 1e-3) / 1e12) : 0.0;
        double gpuTotalTFLOPS = (gpuTotalTime > 0.0f) ? (totalFLOP / (gpuTotalTime * 1e-3) / 1e12) : 0.0;
        double cpuGFLOPS = (cpuTime > 0.0f) ? (totalFLOP / (cpuTime * 1e-3) / 1e9) : 0.0;

        // Bandwidth (Memory)
        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double totalBW = (gpuTotalTime > 0.0f) ? (totalBytes / (gpuTotalTime * 1e-3) / 1e9) : 0.0;

        // ==================== OUTPUT SUMMARY ====================
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=========================================\n";
        std::cout << "GPU H2D Copy        : " << h2dTime << " ms\n";
        std::cout << "GPU SGEMM           : " << kernelTime << " ms\n";
        std::cout << "GPU D2H Copy        : " << d2hTime << " ms\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "GPU Total           : " << gpuTotalTime << " ms\n";
        std::cout << "=========================================\n";
        std::cout << "CPU Time            : " << cpuTime << " ms\n";
        std::cout << "Verification        : " << (correct ? "PASSED" : "FAILED") << "\n";
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

        // Cleanup
        cudaEventDestroy(startH2D);    cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startD2H);    cudaEventDestroy(stopD2H);

        cudaFree(da); cudaFree(db); cudaFree(dc);
        free(a_half); free(b_half);
        free(a_float); free(b_float);
        free(c_gpu); free(c_cpu);
    }

}