#include "tcore_custom_matmul_v4.cuh"
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

namespace tcore_custmat_v4 {

    using namespace nvcuda;

    // Tuning Parameters
    constexpr int STAGES = 3;
    constexpr int M_TILE = 128;
    constexpr int N_TILE = 128;
    constexpr int K_TILE = 16;
    constexpr int PADDING = 8;

    struct alignas(16) TileA { half data[M_TILE][K_TILE + PADDING]; };
    struct alignas(16) TileB { half data[K_TILE][N_TILE + PADDING]; };

    struct StageBuffer {
        TileA A;
        TileB B;
    };

    __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
        __pipeline_memcpy_async(smem, gmem, 16);
    }

    // Coalesced 128-bit Async Global -> Shared Load for Tile A
    __device__ __forceinline__ void load_A_tile_async(TileA& tile, const half* da, int blockRow, int k_curr, int m, int k, int tid) {
        // 128 threads load 128 x 16 elements (2048 halfs = 4096 bytes = 256 int4 transfers)
        // Each thread loads two int4 (16 bytes = 8 half elements)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = tid + i * 128; // 0 to 255
            int row = load_idx / 2;       // 0 to 127
            int col = (load_idx % 2) * 8; // 0 or 8

            int gRow = blockRow + row;
            int gCol = k_curr + col;

            half* smem_ptr = &tile.data[row][col];

            if (gRow < m && (gCol + 7) < k) {
                cp_async_16(smem_ptr, &da[gRow * k + gCol]);
            }
            else {
                *reinterpret_cast<int4*>(smem_ptr) = make_int4(0, 0, 0, 0);
            }
        }
    }

    // Coalesced 128-bit Async Global -> Shared Load for Tile B
    __device__ __forceinline__ void load_B_tile_async(TileB& tile, const half* db, int blockCol, int k_curr, int n, int k, int tid) {
        // 128 threads load 16 x 128 elements (2048 halfs = 4096 bytes = 256 int4 transfers)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = tid + i * 128;  // 0 to 255
            int row = load_idx / 16;       // 0 to 15
            int col = (load_idx % 16) * 8; // 0, 8, 16, ..., 120

            int gRow = k_curr + row;
            int gCol = blockCol + col;

            half* smem_ptr = &tile.data[row][col];

            if (gRow < k && (gCol + 7) < n) {
                cp_async_16(smem_ptr, &db[gRow * n + gCol]);
            }
            else {
                *reinterpret_cast<int4*>(smem_ptr) = make_int4(0, 0, 0, 0);
            }
        }
    }

    // Optimized MatMul Kernel with Ring-Buffer Multistage Async Pipeline
    __global__ void gpuMult(half* da, half* db, float* dc, int m, int k, int n) {
        const int tid = threadIdx.x;
        const int warpid = tid / 32;
        const int warpx = warpid % 2; // 0 or 1
        const int warpy = warpid / 2; // 0, 1, 2, or 3

        const int blockRow = blockIdx.y * M_TILE;
        const int blockCol = blockIdx.x * N_TILE;

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
            int k_curr = s * K_TILE;
            if (k_curr < k) {
                load_A_tile_async(stages[write_stage].A, da, blockRow, k_curr, m, k, tid);
                load_B_tile_async(stages[write_stage].B, db, blockCol, k_curr, n, k, tid);
                __pipeline_commit();
                write_stage = (write_stage + 1) % STAGES;
            }
        }

        // --- MAIN PIPELINED K-LOOP ---
        for (int k_idx = (STAGES - 1) * K_TILE; k_idx < k; k_idx += K_TILE) {
            load_A_tile_async(stages[write_stage].A, da, blockRow, k_idx, m, k, tid);
            load_B_tile_async(stages[write_stage].B, db, blockCol, k_idx, n, k, tid);
            __pipeline_commit();

            __pipeline_wait_prior(STAGES - 1);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(a_frag[i], &stages[read_stage].A.data[(warpy * 4 + i) * 16][0], K_TILE + PADDING);
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &stages[read_stage].B.data[0][(warpx * 4 + j) * 16], N_TILE + PADDING);
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

        // --- EPILOGUE PIPELINE DRAIN ---
        for (int s = 0; s < STAGES - 1; ++s) {
            __pipeline_wait_prior(STAGES - 2 - s);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(a_frag[i], &stages[read_stage].A.data[(warpy * 4 + i) * 16][0], K_TILE + PADDING);
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &stages[read_stage].B.data[0][(warpx * 4 + j) * 16], N_TILE + PADDING);
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

        // --- ACCUMULATOR STORE BACK TO GLOBAL MEMORY ---
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

    void initMatrix(half* m_half, float* m_float, int r, int c) {
        for (int i = 0; i < r * c; ++i) {
            float val = static_cast<float>(rand() % 3);
            m_half[i] = __float2half(val);
            m_float[i] = val;
        }
    }

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

        half* da, * db;
        float* dc_custom, * dc_cublas;
        cudaMalloc(&da, sizeof(half) * MDIM * KDIM);
        cudaMalloc(&db, sizeof(half) * KDIM * NDIM);
        cudaMalloc(&dc_custom, sizeof(float) * MDIM * NDIM);
        cudaMalloc(&dc_cublas, sizeof(float) * MDIM * NDIM);

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

        // Warmup Custom Kernel and cuBLAS
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

        // Verification Output Generation
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

        cudaEvent_t startH2D, stopH2D, startCustom, stopCustom, startCublas, stopCublas, startD2H, stopD2H;
        cudaEventCreate(&startH2D);    cudaEventCreate(&stopH2D);
        cudaEventCreate(&startCustom); cudaEventCreate(&stopCustom);
        cudaEventCreate(&startCublas); cudaEventCreate(&stopCublas);
        cudaEventCreate(&startD2H);    cudaEventCreate(&stopD2H);

        cudaEventRecord(startH2D);
        cudaMemcpy(da, a_half, sizeof(half) * MDIM * KDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b_half, sizeof(half) * KDIM * NDIM, cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);
        cudaEventSynchronize(stopH2D);

        // Benchmark Custom Kernel
        cudaEventRecord(startCustom);
        for (int i = 0; i < 100; ++i) {
            gpuMult << <gridsize, blocksize >> > (da, db, dc_custom, MDIM, KDIM, NDIM);
        }
        cudaEventRecord(stopCustom);
        cudaEventSynchronize(stopCustom);

        // Benchmark cuBLAS
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

        cudaEventRecord(startD2H);
        cudaMemcpy(c_custom, dc_custom, sizeof(float) * MDIM * NDIM, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopD2H);
        cudaEventSynchronize(stopD2H);

        float h2dTime = 0.0f, customKernelTime = 0.0f, cublasKernelTime = 0.0f, d2hTime = 0.0f;
        cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
        cudaEventElapsedTime(&customKernelTime, startCustom, stopCustom);
        cudaEventElapsedTime(&cublasKernelTime, startCublas, stopCublas);
        cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);

        customKernelTime /= 100.0f;
        cublasKernelTime /= 100.0f;

        bool correct = verify(c_custom, c_cublas, MDIM, NDIM, KDIM);

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
        std::cout << "          THROUGHPUT & BANDWIDTH          \n";
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