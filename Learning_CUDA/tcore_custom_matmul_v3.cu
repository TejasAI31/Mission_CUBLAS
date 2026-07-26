#include "tcore_custom_matmul_v3.cuh"
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

namespace tcore_custmat_v3 {

    using namespace nvcuda;

    // --- Conflict-Free Padded Shared Memory Layout ---
    constexpr int STAGES = 3;
    constexpr int PADDING = 8;

    struct alignas(16) TileA { half data[128][16 + PADDING]; };
    struct alignas(16) TileB { half data[16][128 + PADDING]; };

    struct StageBuffer {
        TileA A;
        TileB B;
    };

    // --- Pipeline Helpers ---
    __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
        __pipeline_memcpy_async(smem, gmem, 16);
    }

    // Vectorized Asynchronous Tile Loads with Padding Layout
    __device__ __forceinline__ void load_A_tile_async(TileA& tile, const half* da, int blockRow, int k_curr, int m, int k, int tid) {
        int gRow = blockRow + tid;
        half* a0 = &tile.data[tid][0];
        half* a1 = &tile.data[tid][8];

        if (gRow < m && (k_curr + 15) < k) {
            cp_async_16(a0, &da[gRow * k + k_curr]);
            cp_async_16(a1, &da[gRow * k + k_curr + 8]);
        }
        else {
            *reinterpret_cast<int4*>(a0) = *reinterpret_cast<int4*>(a1) = make_int4(0, 0, 0, 0);
        }
    }

    __device__ __forceinline__ void load_B_tile_async(TileB& tile, const half* db, int blockCol, int k_curr, int n, int k, int tid) {
        int r0 = tid / 16;
        int col = (tid % 16) * 8;
        int gR0 = k_curr + r0;
        int gR1 = k_curr + r0 + 8;
        int gCol = blockCol + col;

        half* b0 = &tile.data[r0][col];
        half* b1 = &tile.data[r0 + 8][col];

        if (gR0 < k && (gCol + 7) < n) cp_async_16(b0, &db[gR0 * n + gCol]);
        else *reinterpret_cast<int4*>(b0) = make_int4(0, 0, 0, 0);

        if (gR1 < k && (gCol + 7) < n) cp_async_16(b1, &db[gR1 * n + gCol]);
        else *reinterpret_cast<int4*>(b1) = make_int4(0, 0, 0, 0);
    }

    // --- Main Kernel: 3-Stage Ring Pipeline with Padded SMEM ---
    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n) {
        const int tid = threadIdx.x;
        const int warpid = tid / 32;
        const int warpx = warpid % 2;
        const int warpy = warpid / 2;

        const int blockRow = blockIdx.y * 128;
        const int blockCol = blockIdx.x * 128;

        __shared__ StageBuffer stages[STAGES];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][4];

#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(c_frag[i][j], 0.0f);
            }
        }

        int write_stage = 0;
        int read_stage = 0;

        // --- PROLOGUE ---
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            int k_curr = s * 16;
            if (k_curr < k) {
                load_A_tile_async(stages[write_stage].A, da, blockRow, k_curr, m, k, tid);
                load_B_tile_async(stages[write_stage].B, db, blockCol, k_curr, n, k, tid);
                __pipeline_commit();
                write_stage = (write_stage + 1) % STAGES;
            }
        }

        // --- MAIN PIPELINED K-LOOP ---
        for (int k_idx = (STAGES - 1) * 16; k_idx < k; k_idx += 16) {
            load_A_tile_async(stages[write_stage].A, da, blockRow, k_idx, m, k, tid);
            load_B_tile_async(stages[write_stage].B, db, blockCol, k_idx, n, k, tid);
            __pipeline_commit();

            __pipeline_wait_prior(STAGES - 1);
            __syncthreads();

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(a_frag[i], &stages[read_stage].A.data[(warpy * 4 + i) * 16][0], 16 + PADDING);
            }

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &stages[read_stage].B.data[0][(warpx * 4 + j) * 16], 128 + PADDING);
            }

#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }

            write_stage = (write_stage + 1) % STAGES;
            read_stage = (read_stage + 1) % STAGES;

            __syncthreads();
        }

        // --- EPILOGUE ---
        for (int s = 0; s < STAGES - 1; ++s) {
            __pipeline_wait_prior(STAGES - 2 - s);
            __syncthreads();

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(a_frag[i], &stages[read_stage].A.data[(warpy * 4 + i) * 16][0], 16 + PADDING);
            }

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &stages[read_stage].B.data[0][(warpx * 4 + j) * 16], 128 + PADDING);
            }

#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }

            read_stage = (read_stage + 1) % STAGES;
            __syncthreads();
        }

        // --- STORE ACCUMULATORS ---
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int cRow = blockRow + (warpy * 4 + i) * 16;
                int cCol = blockCol + (warpx * 4 + j) * 16;

                if (cRow < m && cCol < n) {
                    wmma::store_matrix_sync(&dc[cRow * n + cCol], c_frag[i][j], n, wmma::mem_row_major);
                }
            }
        }
    }

    // --- Helpers ---
    void initMatrix(half* m_half, float* m_float, int r, int c) {
        for (int i = 0; i < r * c; ++i) {
            float val = static_cast<float>(rand() % 3);
            m_half[i] = __float2half(val);
            m_float[i] = val;
        }
    }

    // Verification against cuBLAS Reference Matrix Output
    bool verify(const float* custom_res, const float* cublas_res, int m, int n, int k) {
        float eps = 1e-1f * static_cast<float>(k);
        for (int i = 0; i < m * n; i++) {
            if (std::fabs(custom_res[i] - cublas_res[i]) > eps)
                return false;
        }
        return true;
    }

    void matmul() {
        constexpr int blcksize = 128;

        // Pinned Host Memory Allocation for fast DMA transfers
        half* a_half, * b_half;
        float* a_float, * b_float, * c_custom, * c_cublas;

        cudaMallocHost((void**)&a_half, sizeof(half) * MDIM * KDIM);
        cudaMallocHost((void**)&b_half, sizeof(half) * KDIM * NDIM);
        cudaMallocHost((void**)&a_float, sizeof(float) * MDIM * KDIM);
        cudaMallocHost((void**)&b_float, sizeof(float) * KDIM * NDIM);
        cudaMallocHost((void**)&c_custom, sizeof(float) * MDIM * NDIM);
        cudaMallocHost((void**)&c_cublas, sizeof(float) * MDIM * NDIM);

        initMatrix(a_half, a_float, MDIM, KDIM);
        initMatrix(b_half, b_float, KDIM, NDIM);

        // Device Memory Allocation
        half* da, * db;
        float* dc_custom, * dc_cublas;
        cudaMalloc(&da, sizeof(half) * MDIM * KDIM);
        cudaMalloc(&db, sizeof(half) * KDIM * NDIM);
        cudaMalloc(&dc_custom, sizeof(float) * MDIM * NDIM);
        cudaMalloc(&dc_cublas, sizeof(float) * MDIM * NDIM);

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

        dim3 blocksize(128);
        dim3 gridsize((NDIM + blcksize - 1) / blcksize, (MDIM + blcksize - 1) / blcksize);

        // Warmup Custom Kernel and cuBLAS (10 rounds)
        cudaMemcpy(da, a_half, sizeof(half) * MDIM * KDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * KDIM * NDIM, cudaMemcpyHostToDevice);

        for (int i = 0; i < 10; ++i) {
            gpuMult << <gridsize, blocksize >> > (da, db, dc_custom, MDIM, KDIM, NDIM);
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                NDIM, MDIM, KDIM,
                &alpha,
                db, CUDA_R_16F, NDIM,
                da, CUDA_R_16F, KDIM,
                &beta,
                dc_cublas, CUDA_R_32F, NDIM,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
        cudaDeviceSynchronize();

        // Compute cuBLAS Reference Output for Verification
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            NDIM, MDIM, KDIM,
            &alpha,
            db, CUDA_R_16F, NDIM,
            da, CUDA_R_16F, KDIM,
            &beta,
            dc_cublas, CUDA_R_32F, NDIM,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT
        );
        cudaMemcpy(c_cublas, dc_cublas, sizeof(float) * MDIM * NDIM, cudaMemcpyDeviceToHost);

        // Benchmarking Events Setup
        cudaEvent_t startH2D, stopH2D, startCustom, stopCustom, startCublas, stopCublas, startD2H, stopD2H;
        cudaEventCreate(&startH2D);    cudaEventCreate(&stopH2D);
        cudaEventCreate(&startCustom); cudaEventCreate(&stopCustom);
        cudaEventCreate(&startCublas); cudaEventCreate(&stopCublas);
        cudaEventCreate(&startD2H);    cudaEventCreate(&stopD2H);

        // Host -> Device Transfers
        cudaEventRecord(startH2D);
        cudaMemcpy(da, a_half, sizeof(half) * MDIM * KDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * KDIM * NDIM, cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);
        cudaEventSynchronize(stopH2D);

        // Measure Custom Kernel Performance (100 iterations)
        cudaEventRecord(startCustom);
        for (int i = 0; i < 100; ++i) {
            gpuMult << <gridsize, blocksize >> > (da, db, dc_custom, MDIM, KDIM, NDIM);
        }
        cudaEventRecord(stopCustom);
        cudaEventSynchronize(stopCustom);

        // Measure cuBLAS Performance (100 iterations)
        cudaEventRecord(startCublas);
        for (int i = 0; i < 100; ++i) {
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                NDIM, MDIM, KDIM,
                &alpha,
                db, CUDA_R_16F, NDIM,
                da, CUDA_R_16F, KDIM,
                &beta,
                dc_cublas, CUDA_R_32F, NDIM,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
        cudaEventRecord(stopCublas);
        cudaEventSynchronize(stopCublas);

        // Device -> Host Transfers
        cudaEventRecord(startD2H);
        cudaMemcpy(c_custom, dc_custom, sizeof(float) * MDIM * NDIM, cudaMemcpyDeviceToHost);
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

        // Verify custom output against cuBLAS
        bool correct = verify(c_custom, c_cublas, MDIM, NDIM, KDIM);

        // Throughput Metrics
        double totalFLOP = 2.0 * static_cast<double>(MDIM) * static_cast<double>(NDIM) * static_cast<double>(KDIM);
        double h2dBytes = (static_cast<double>(MDIM) * KDIM + static_cast<double>(KDIM) * NDIM) * sizeof(half);
        double d2hBytes = static_cast<double>(MDIM) * NDIM * sizeof(float);
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

} // namespace tcore_custmat_v3