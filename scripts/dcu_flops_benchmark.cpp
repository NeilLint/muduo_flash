#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define SIZE 4096  // 可调，越大测得越准

__global__ void fp32_gemm(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void fp64_gemm(double* a, double* b, double* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

template<typename T>
void run_gemm_test(bool is_fp64) {
    int N = SIZE;
    size_t bytes = N * N * sizeof(T);
    T *a, *b, *c;
    hipMalloc(&a, bytes);
    hipMalloc(&b, bytes);
    hipMalloc(&c, bytes);

    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;

    // warmup
    if (is_fp64)
        hipLaunchKernelGGL(fp64_gemm, dim3(blocks), dim3(threads), 0, 0, (double*)a, (double*)b, (double*)c, N);
    else
        hipLaunchKernelGGL(fp32_gemm, dim3(blocks), dim3(threads), 0, 0, (float*)a, (float*)b, (float*)c, N);

    hipDeviceSynchronize();

    // timing
    auto start = std::chrono::high_resolution_clock::now();
    if (is_fp64)
        hipLaunchKernelGGL(fp64_gemm, dim3(blocks), dim3(threads), 0, 0, (double*)a, (double*)b, (double*)c, N);
    else
        hipLaunchKernelGGL(fp32_gemm, dim3(blocks), dim3(threads), 0, 0, (float*)a, (float*)b, (float*)c, N);

    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    double total_flops = 2.0 * pow(N, 3);  // 2*N^3 ops per GEMM
    double gflops = total_flops / elapsed_sec / 1e9;

    if (is_fp64)
        std::cout << "FP64 GEMM GFLOPS: " << gflops << std::endl;
    else
        std::cout << "FP32 GEMM GFLOPS: " << gflops << std::endl;

    hipFree(a);
    hipFree(b);
    hipFree(c);
}

int main() {
    run_gemm_test<float>(false);
    run_gemm_test<double>(true);
    return 0;
}
