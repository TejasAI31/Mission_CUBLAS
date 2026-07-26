#include "tiled_matmul.cuh"
#include <chrono>

namespace tmat {

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
                sum += A[ty*TILESIZE+j] * B[tx+j*TILESIZE];
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

        float mallocTime = 0.0f;
        float h2dTime = 0.0f;
        float kernelTime = 0.0f;
        float d2hTime = 0.0f;

        //---------------- Memory Allocation ----------------//

        cudaEventRecord(start);

        cudaMalloc(&da, sizeof(int) * M * K);
        cudaMalloc(&db, sizeof(int) * K * N);
        cudaMalloc(&dd, sizeof(int) * M * N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&mallocTime, start, stop);

        //---------------- Host -> Device ----------------//

        cudaEventRecord(start);

        cudaMemcpy(da, a, sizeof(int) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, sizeof(int) * K * N, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&h2dTime, start, stop);

        //---------------- Warm-up ----------------//

        for (int i = 0; i < 10; i++)
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

        //---------------- Results ----------------//

        float totalGPU = mallocTime + h2dTime + kernelTime + d2hTime;

        cout << "\n================ Performance ================\n";
        cout << "CPU Computation Time      : " << cpuTime << " ms\n\n";

        cout << "GPU Memory Allocation     : " << mallocTime << " ms\n";
        cout << "GPU Host -> Device Copy   : " << h2dTime << " ms\n";
        cout << "GPU Kernel (Average)      : " << kernelTime << " ms\n";
        cout << "GPU Device -> Host Copy   : " << d2hTime << " ms\n";
        cout << "---------------------------------------------\n";
        cout << "GPU Total Time            : " << totalGPU << " ms\n";
        cout << "=============================================\n";

        cout << "\nVerification : "
            << boolalpha << verify(c, d) << endl;

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