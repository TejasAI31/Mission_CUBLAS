#include "transposition_sort.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

namespace tsort {

    __global__ void oddEvenKernel(float* a, int n, int phase)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int i = 2 * tid + (phase & 1);
        if (i + 1 < n && a[i] > a[i + 1]) {
            float t = a[i];
            a[i] = a[i + 1];
            a[i + 1] = t;
        }
    }

    void initArray(float* a, int n)
    {
        for (int i = 0; i < n; i++) a[i] = static_cast<float>(rand() % n);
    }

    void sort()
    {
        float* cpuArray = (float*)malloc(sizeof(float) * N);
        float* gpuArray = (float*)malloc(sizeof(float) * N);

        initArray(cpuArray, N);
        memcpy(gpuArray, cpuArray, sizeof(float) * N);

        auto c0 = std::chrono::high_resolution_clock::now();
        std::sort(cpuArray, cpuArray + N);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(c1 - c0).count();

        float* da;
        cudaMalloc(&da, sizeof(float) * N);

        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);

        float h2dMs, kernelMs, d2hMs, totalMs;

        cudaEventRecord(e0);
        cudaMemcpy(da, gpuArray, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);

        int block = 256;
        int grid = ((N / 2) + block - 1) / block;

        for (int phase = 0; phase < N; ++phase)
            oddEvenKernel << <grid, block >> > (da, N, phase);

        cudaDeviceSynchronize();
        cudaEventRecord(e2);

        cudaMemcpy(gpuArray, da, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);

        cudaEventElapsedTime(&h2dMs, e0, e1);
        cudaEventElapsedTime(&kernelMs, e1, e2);
        cudaEventElapsedTime(&d2hMs, e2, e3);
        cudaEventElapsedTime(&totalMs, e0, e3);

        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (cpuArray[i] != gpuArray[i]) {
                correct = false;
                std::cout << "Mismatch at " << i << ": " << cpuArray[i] << " " << gpuArray[i] << "\n";
                break;
            }
        }

        std::cout << "CPU std::sort : " << cpuMs << " ms\n";
        std::cout << "H2D          : " << h2dMs << " ms\n";
        std::cout << "Kernel       : " << kernelMs << " ms\n";
        std::cout << "D2H          : " << d2hMs << " ms\n";
        std::cout << "GPU Total    : " << totalMs << " ms\n";
        std::cout << "Correct      : " << (correct ? "YES" : "NO") << "\n";

        cudaFree(da);
        free(cpuArray);
        free(gpuArray);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e3);
    }

}
