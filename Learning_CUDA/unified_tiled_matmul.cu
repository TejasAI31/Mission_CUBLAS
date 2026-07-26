#include "unified_tiled_matmul.cuh"
#include <chrono>

namespace utmat {

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
            cout << m[i] << " ";
            if ((i + 1) % c == 0)
                cout << '\n';
        }
        cout << '\n';
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

        float mallocTime = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cudaMallocManaged(&a, sizeof(int) * M * K);
        cudaMallocManaged(&b, sizeof(int) * K * N);
        cudaMallocManaged(&c, sizeof(int) * M * N);
        cudaMallocManaged(&d, sizeof(int) * M * N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&mallocTime, start, stop);


        initMatrix(a, M, K);
        initMatrix(b, K, N);

        //---------------- CPU Timing ----------------//

        auto cpuStart = std::chrono::high_resolution_clock::now();

        cpuMult(a, b, c, M, K, N);

        auto cpuEnd = std::chrono::high_resolution_clock::now();

        double cpuTime =
            std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        //---------------- GPU Setup ----------------//

        dim3 blocksize(BLOCKSIZE, BLOCKSIZE);
        dim3 gridsize(
            (N + BLOCKSIZE - 1) / BLOCKSIZE,
            (M + BLOCKSIZE - 1) / BLOCKSIZE);

        float kernelTime = 0.0f;

        //---------------- Warm-up ----------------//

        for (int i = 0; i < 10; i++)
        {
            gpuMult << <gridsize, blocksize >> > (a, b, d, M, K, N);
        }

        cudaDeviceSynchronize();

        //---------------- Kernel Timing ----------------//

        cudaEventRecord(start);

        for (int i = 0; i < 100; i++)
        {
            gpuMult << <gridsize, blocksize >> > (a, b, d, M, K, N);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&kernelTime, start, stop);

        kernelTime /= 100.0f;

        //---------------- Results ----------------//

        float totalGPU = kernelTime;

        cout << "\n================ Performance ================\n";
        cout << "Unified Memory Allocation : " << mallocTime << " ms\n\n";

        cout << "CPU Computation Time      : " << cpuTime << " ms\n";
        cout << "GPU Computation Time      : " << totalGPU << " ms\n";
        cout << "=============================================\n";

        cout << "\nVerification : "
            << boolalpha << verify(c, d) << endl;

        //---------------- Cleanup ----------------//

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaFree(d);
    }

}