#include "basic_vector_add.cuh"

namespace uva {
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

        int threads = THREADS;
        int blocks = (int)ceil(N / (float)threads);

        // ---------------- Warm-up (10 rounds) ----------------
        for (int i = 0; i < 10; i++)
        {
            gpuPerformAdd << <blocks, threads >> > (ha, hb, hc);
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
            gpuPerformAdd << <blocks, threads >> > (ha, hb, hc);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime = 0.0f;
        cudaEventElapsedTime(&gpuTime, start, stop);

        cout << "Average GPU Kernel Time: " << gpuTime / 100.0f << " ms" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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

        cudaMallocManaged(&ha, N * sizeof(int));
        cudaMallocManaged(&hb, N * sizeof(int));
        cudaMallocManaged(&hc, N * sizeof(int));
        cudaMallocManaged(&hg, N * sizeof(int));

        matrixInit(ha);
        matrixInit(hb);

        cout << N << " Element Arrays Initialized" << endl;

        // ---------------- CPU Timing ----------------
        cudaMemLocation cpu_loc;
        cpu_loc.id = 0;
        cpu_loc.type = cudaMemLocationTypeHostNuma;
        cudaMemPrefetchAsync(ha, sizeof(int) * N, cpu_loc, 0);
        cudaMemPrefetchAsync(hb, sizeof(int) * N, cpu_loc, 0);
        cudaMemPrefetchAsync(hc, sizeof(int) * N, cpu_loc, 0);

        auto cpuStart = chrono::high_resolution_clock::now();

        cpuAdd(ha, hb, hc);

        auto cpuEnd = chrono::high_resolution_clock::now();

        double cpuTime = chrono::duration<double, milli>(cpuEnd - cpuStart).count();

        cout << "CPU Addition Performed" << endl;
        cout << "CPU Time: " << cpuTime << " ms" << endl;

        // ---------------- GPU ----------------
        int gpu_id;
        cudaGetDevice(&gpu_id);
        cudaMemLocation gpu_loc;
        gpu_loc.id = gpu_id;
        gpu_loc.type = cudaMemLocationTypeDevice;
        cudaMemPrefetchAsync(ha, sizeof(int) * N, gpu_loc, 0);
        cudaMemPrefetchAsync(hb, sizeof(int) * N, gpu_loc, 0);
        cudaMemPrefetchAsync(hg, sizeof(int) * N, gpu_loc, 0);

        gpuAdd(ha, hb, hg);

        cout << "GPU Addition Performed" << endl;

        cout << "Correct Status: " << boolalpha << verifyArrays(hc, hg) << endl;

        cudaFree(ha);
        cudaFree(hb);
        cudaFree(hc);
        cudaFree(hg);
    }
}