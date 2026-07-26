#include "tcore_custom_matmul_v2.cuh"
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mma.h>
#include <cublas_v2.h>

namespace tcore_custmat_v2 {

    using namespace nvcuda;

    // Clean tile definitions for 2D shared memory indexing
    struct TileA { half data[128][16]; };
    struct TileB { half data[16][128]; };

    struct StageBuffer {
        TileA A;
        TileB B;
    };

    // 128-bit Vectorized Cooperative Loads (Each thread loads two int4 vectors = 16 halves)
    __device__ __forceinline__ void load_A_tile(TileA& tile, const half* da, int blockRow, int k_curr, int m, int k, int tid) {
        // 128 rows, 16 cols -> 128 threads load 1 row of 16 halves per thread
        int row = tid;
        int gRow = blockRow + row;

        int4* smem_ptr0 = reinterpret_cast<int4*>(&tile.data[row][0]);
        int4* smem_ptr1 = reinterpret_cast<int4*>(&tile.data[row][8]);

        if (gRow < m && (k_curr + 15) < k) {
            *smem_ptr0 = *reinterpret_cast<const int4*>(&da[gRow * k + k_curr + 0]);
            *smem_ptr1 = *reinterpret_cast<const int4*>(&da[gRow * k + k_curr + 8]);
        }
        else {
            *smem_ptr0 = make_int4(0, 0, 0, 0);
            *smem_ptr1 = make_int4(0, 0, 0, 0);
        }
    }

    __device__ __forceinline__ void load_B_tile(TileB& tile, const half* db, int blockCol, int k_curr, int n, int k, int tid) {
        // 16 rows, 128 cols -> 128 threads need 2 loads per thread to cover all 16 rows
        int row0 = tid / 16;        // Covers rows 0..7
        int col = (tid % 16) * 8;   // Covers columns 0, 8, 16, ..., 120
        int gRow0 = k_curr + row0;
        int gCol = blockCol + col;

        // Load 1: Upper 8 rows (rows 0..7)
        int4* smem_ptr0 = reinterpret_cast<int4*>(&tile.data[row0][col]);
        if (gRow0 < k && (gCol + 7) < n) {
            *smem_ptr0 = *reinterpret_cast<const int4*>(&db[gRow0 * n + gCol]);
        }
        else {
            *smem_ptr0 = make_int4(0, 0, 0, 0);
        }

        // Load 2: Lower 8 rows (rows 8..15)
        int row1 = row0 + 8;
        int gRow1 = k_curr + row1;
        int4* smem_ptr1 = reinterpret_cast<int4*>(&tile.data[row1][col]);
        if (gRow1 < k && (gCol + 7) < n) {
            *smem_ptr1 = *reinterpret_cast<const int4*>(&db[gRow1 * n + gCol]);
        }
        else {
            *smem_ptr1 = make_int4(0, 0, 0, 0);
        }
    }

    // Optimized Kernel: 128 threads (2x2 Warps), 128x128 Tile, Double-Buffered
    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n)
    {
        // 128 threads = 4 warps organized in a 2x2 spatial warp grid
        const int tid = threadIdx.x;
        const int warpid = tid / 32;
        const int warpx = warpid % 2; // 0, 1
        const int warpy = warpid / 2; // 0, 1

        // Double-buffered Shared Memory
        __shared__ StageBuffer stages[2];

        // Fragments for 4x4 grid per warp (64x64 computation area per warp)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>             c_frag[4][4];

#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(c_frag[i][j], 0.0f);
            }
        }

        const int blockRow = blockIdx.y * blcksize;
        const int blockCol = blockIdx.x * blcksize;

        // --- PROLOGUE: Load Stage 0 ---
        int write_stage = 0;
        load_A_tile(stages[write_stage].A, da, blockRow, 0, m, k, tid);
        load_B_tile(stages[write_stage].B, db, blockCol, 0, n, k, tid);
        __syncthreads();

        // --- MAIN PIPELINED K-LOOP ---
        for (int k_idx = tilesize; k_idx < k; k_idx += tilesize)
        {
            int read_stage = write_stage;
            write_stage ^= 1; // Toggle active buffer stage

            // 1. Prefetch NEXT tile into write_stage buffer
            load_A_tile(stages[write_stage].A, da, blockRow, k_idx, m, k, tid);
            load_B_tile(stages[write_stage].B, db, blockCol, k_idx, n, k, tid);

            // 2. Load Matrix A fragments for this warp (4 sub-tiles vertically = 64 rows)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int a_row_offset = (warpy * 4 + i) * tilesize;
                wmma::load_matrix_sync(a_frag[i], &stages[read_stage].A.data[a_row_offset][0], tilesize);
            }

            // 3. Load Matrix B fragments for this warp (4 sub-tiles horizontally = 64 cols)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int b_col_offset = (warpx * 4 + j) * tilesize;
                wmma::load_matrix_sync(b_frag[j], &stages[read_stage].B.data[0][b_col_offset], blcksize);
            }

            // 4. Compute 4x4 Tensor Core Math (16 WMMA Ops)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }

            __syncthreads();
        }

        // --- EPILOGUE: Compute Last Stage ---
        int final_stage = write_stage;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int a_row_offset = (warpy * 4 + i) * tilesize;
            wmma::load_matrix_sync(a_frag[i], &stages[final_stage].A.data[a_row_offset][0], tilesize);
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int b_col_offset = (warpx * 4 + j) * tilesize;
            wmma::load_matrix_sync(b_frag[j], &stages[final_stage].B.data[0][b_col_offset], blcksize);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        // --- STORE ACCUMULATORS TO GLOBAL MEMORY ---
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int cRow = blockRow + (warpy * 4 + i) * tilesize;
                int cCol = blockCol + (warpx * 4 + j) * tilesize;

                if (cRow < m && cCol < n) {
                    wmma::store_matrix_sync(&dc[cRow * n + cCol], c_frag[i][j], n, wmma::mem_row_major);
                }
            }
        }
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
        constexpr int blcksize = 128;

        // Pinned Host Memory Allocation for fast DMA transfers
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

        // Launch Configuration: 128 threads (2x2 spatial warp grid)
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

} // namespace tcore_custmat_v2