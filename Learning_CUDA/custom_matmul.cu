#include "custom_matmul.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>

namespace custmat {

    void initMatrix(int* m, int r, int c)
    {
        for (int i = 0; i < c; i++)
        {
            for (int j = 0; j < r; j++)
            {
                m[i * r + j] = rand() % 3;
            }
        }
    }

    void printMatrix(int* m, int r, int c)
    {
        for (int i = 0; i < r * c; i++)
        {
            std::cout << m[i] << " ";
            if ((i + 1) % c == 0)
                std::cout << '\n';
        }
        std::cout << '\n';
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
            //Copying Tile
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

            //Calculation
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

    bool verify(int* cpu, int* gpu)
    {
        for (int i = 0; i < M * N; i++)
        {
            if (cpu[i] != gpu[i])
                return false;
        }
        return true;
    }

    void matmul()
    {
        int* a, * b, * c, * d;

        a = (int*)malloc(sizeof(int) * M * K);
        b = (int*)malloc(sizeof(int) * K * N);
        c = (int*)malloc(sizeof(int) * M * N);
        d = (int*)malloc(sizeof(int) * M * N);

        initMatrix(a, M, K);
        initMatrix(b, K, N);

        //---------------- CPU Timing ----------------//

        auto cpuStart = std::chrono::high_resolution_clock::now();

        cpuMult(a, b, c, M, K, N);

        auto cpuEnd = std::chrono::high_resolution_clock::now();

        double cpuTime =
            std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        //---------------- GPU Setup ----------------//

        int* da, * db, * dd;

        dim3 blocksize(BLOCKSIZE, BLOCKSIZE);
        dim3 gridsize(
            (N + BLOCKSIZE - 1) / BLOCKSIZE,
            (M + BLOCKSIZE - 1) / BLOCKSIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float h2dTime = 0.0f;
        float kernelTime = 0.0f;
        float d2hTime = 0.0f;

        //---------------- Memory Allocation ----------------//

        cudaMalloc(&da, sizeof(int) * M * K);
        cudaMalloc(&db, sizeof(int) * K * N);
        cudaMalloc(&dd, sizeof(int) * M * N);

        //---------------- Host -> Device ----------------//

        cudaEventRecord(start);

        cudaMemcpy(da, a, sizeof(int) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, sizeof(int) * K * N, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&h2dTime, start, stop);

        //---------------- Warm-up ----------------//

        for (int i = 0; i < 100; i++)
        {
            gpuMult << <gridsize, blocksize >> > (da, db, dd, M, K, N);
        }

        cudaDeviceSynchronize();

        //---------------- Kernel Timing ----------------//

        cudaEventRecord(start);

        for (int i = 0; i < 100; i++)
        {
            gpuMult << <gridsize, blocksize >> > (da, db, dd, M, K, N);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&kernelTime, start, stop);

        kernelTime /= 100.0f;

        //---------------- Device -> Host ----------------//

        cudaEventRecord(start);

        cudaMemcpy(d, dd, sizeof(int) * M * N, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&d2hTime, start, stop);

        //---------------- Results & Metrics ----------------//

        float totalGPU = h2dTime + kernelTime + d2hTime;

        double totalFLOP = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double h2dBytes = (static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(int);
        double d2hBytes = static_cast<double>(M) * N * sizeof(int);
        double totalBytes = h2dBytes + d2hBytes;

        double kernelGFLOPS = (kernelTime > 0.0f) ? (totalFLOP / (kernelTime * 1e-3) / 1e9) : 0.0;
        double gpuTotalGFLOPS = (totalGPU > 0.0f) ? (totalFLOP / (totalGPU * 1e-3) / 1e9) : 0.0;
        double cpuGFLOPS = (cpuTime > 0.0f) ? (totalFLOP / (cpuTime * 1e-3) / 1e9) : 0.0;

        double h2dBW = (h2dTime > 0.0f) ? (h2dBytes / (h2dTime * 1e-3) / 1e9) : 0.0;
        double d2hBW = (d2hTime > 0.0f) ? (d2hBytes / (d2hTime * 1e-3) / 1e9) : 0.0;
        double totalBW = (totalGPU > 0.0f) ? (totalBytes / (totalGPU * 1e-3) / 1e9) : 0.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=========================================\n";
        std::cout << "CPU Total           : " << cpuTime << " ms\n\n";

        std::cout << "GPU H2D Copy        : " << h2dTime << " ms\n";
        std::cout << "GPU Custom Kernel   : " << kernelTime << " ms\n";
        std::cout << "GPU D2H Copy        : " << d2hTime << " ms\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "GPU Total           : " << totalGPU << " ms\n";
        std::cout << "=========================================\n";
        std::cout << "Verification        : " << std::boolalpha << verify(c, d) << "\n";
        std::cout << "=========================================\n";
        std::cout << "         THROUGHPUT & BANDWIDTH          \n";
        std::cout << "=========================================\n";
        std::cout << "GPU H2D Bandwidth   : " << h2dBW << " GB/s\n";
        std::cout << "GPU D2H Bandwidth   : " << d2hBW << " GB/s\n";
        std::cout << "GPU Kernel Compute  : " << kernelGFLOPS << " GFLOPS\n";
        std::cout << "GPU Total Compute   : " << gpuTotalGFLOPS << " GFLOPS\n";
        std::cout << "GPU Overall Pipeline: " << totalBW << " GB/s\n";
        std::cout << "CPU Compute         : " << cpuGFLOPS << " GFLOPS\n";
        std::cout << "=========================================\n";

        //---------------- Cleanup ----------------//

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(da);
        cudaFree(db);
        cudaFree(dd);

        free(a);
        free(b);
        free(c);
        free(d);
    }

}