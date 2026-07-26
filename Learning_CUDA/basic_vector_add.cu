#include "basic_vector_add.cuh"

namespace bva {
    __global__ void gpuPerformAdd(int* da, int* db, int* dc)
    {
        int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (idx < N)
        {
            dc[idx] = da[idx] + db[idx];
        }
    }

    void matrixInit(int* arr)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] = rand() % 100;
        }
    }

    void cpuAdd(int* ha, int* hb, int* hc)
    {
        for (int i = 0; i < N; i++)
        {
            hc[i] = ha[i] + hb[i];
        }
    }

    void gpuAdd(int* ha, int* hb, int* hc)
    {
        size_t size = N * sizeof(int);

        int* da, * db, * dc;

        cudaMalloc(&da, size);
        cudaMalloc(&db, size);
        cudaMalloc(&dc, size);

        cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

        int threads = THREADS;
        int blocks = (int)ceil(N / (float)threads);

        // ---------------- Warm-up (10 rounds) ----------------
        for (int i = 0; i < 10; i++)
        {
            gpuPerformAdd << <blocks, threads >> > (da, db, dc);
        }
        cudaDeviceSynchronize();

        // ---------------- Timing ----------------
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Run kernel 100 times for stable timing
        for (int i = 0; i < 100; i++)
        {
            gpuPerformAdd << <blocks, threads >> > (da, db, dc);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime = 0.0f;
        cudaEventElapsedTime(&gpuTime, start, stop);

        cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

        cout << "Average GPU Kernel Time: " << gpuTime / 100.0f << " ms" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }

    bool verifyArrays(int* a, int* b)
    {
        for (int i = 0; i < N; i++)
        {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

    void vector_add()
    {
        int* ha, * hb, * hc, * hg;

        ha = (int*)malloc(sizeof(int) * N);
        hb = (int*)malloc(sizeof(int) * N);
        hc = (int*)malloc(sizeof(int) * N);
        hg = (int*)malloc(sizeof(int) * N);

        matrixInit(ha);
        matrixInit(hb);

        cout << N << " Element Arrays Initialized" << endl;

        // ---------------- CPU Timing ----------------
        auto cpuStart = chrono::high_resolution_clock::now();

        cpuAdd(ha, hb, hc);

        auto cpuEnd = chrono::high_resolution_clock::now();

        double cpuTime = chrono::duration<double, milli>(cpuEnd - cpuStart).count();

        cout << "CPU Addition Performed" << endl;
        cout << "CPU Time: " << cpuTime << " ms" << endl;

        // ---------------- GPU ----------------
        gpuAdd(ha, hb, hg);

        cout << "GPU Addition Performed" << endl;

        cout << "Correct Status: " << boolalpha << verifyArrays(hc, hg) << endl;

        free(ha);
        free(hb);
        free(hc);
        free(hg);
    }
}