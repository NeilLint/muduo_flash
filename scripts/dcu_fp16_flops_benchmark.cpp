#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define SIZE 4096  // 越大越准，8GB显存上可改成8192

__global__ void fp16_gemm(half* a, half* b, half* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        half sum = __float2half(0.0f);
        for (int k = 0; k < N; ++k) {
            sum = __hadd(sum, __hmul(a[row * N + k], b[k * N + col]));
        }
        c[row * N + col] = sum;
    }
}

void run_fp16_gemm_test() {
    int N = SIZE;
    size_t bytes = N * N * sizeof(half);
    half *a, *b, *c;
    hipMalloc(&a, bytes);
    hipMalloc(&b, bytes);
    hipMalloc(&c, bytes);

    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;

    // warmup
    hipLaunchKernelGGL(fp16_gemm, dim3(blocks), dim3(threads), 0, 0, a, b, c, N);
    hipDeviceSynchronize();

    // timing
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(fp16_gemm, dim3(blocks), dim3(threads), 0, 0, a, b, c, N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    double total_flops = 2.0 * pow(N, 3);  // 2*N^3 ops per GEMM
    double gflops = total_flops / elapsed_sec / 1e9;

    std::cout << "FP16 GEMM GFLOPS: " << gflops << std::endl;

    hipFree(a);
    hipFree(b);
    hipFree(c);
}

int main() {
    run_fp16_gemm_test();
    return 0;
}
