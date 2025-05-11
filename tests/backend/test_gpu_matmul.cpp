#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm> // For std::min
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// 用于HIP错误检查的辅助宏
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define HIPBLAS_CHECK(command) { \
    hipblasStatus_t status = command; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n", \
                status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// 定义CPU版本的matmul函数
void cpu_matmul(float* o, const float* x, const float* w, int n, int d) {
    // o[d] = w[d][0] * x[0] + w[d][1] * x[1] + ... + w[d][n-1] * x[n-1]
    // w 是 d x n (行主序)
    // x 是 n x 1
    // o 是 d x 1
    for (int i = 0; i < d; ++i) { // 遍历输出向量 o 的每个元素，对应 w 的每一行
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) { // 遍历 w 的当前行 和 x 的所有元素
            sum += w[i * n + j] * x[j];  // w[i][j] 对应的内存位置是 w[i * n + j]
        }
        o[i] = sum;
    }
}

// ----------------------------------------------------------------------------
// THE FUNCTION TO BE TESTED
// ----------------------------------------------------------------------------
void gpu_backend_matmul(float* o_d,           // 指向 GPU 上的输出向量 o (d x 1) 的指针
                      const float* x_d,     // 指向 GPU 上的输入向量 x (n x 1) 的指针
                      const float* w_d,     // 指向 GPU 上的输入矩阵 w (d x n, Row-Major) 的指针
                      int n,                // 矩阵 w 的列数 / 向量 x 的行数
                      int d,                // 矩阵 w 的行数 / 向量 o 的行数
                      hipStream_t stream = nullptr) { // 指定的流，可选
    // 输入参数检查
    if (!o_d || !x_d || !w_d || n <= 0 || d <= 0) {
        fprintf(stderr, "GPU Matmul Error: Invalid input pointers or dimensions.\n");
        return;
    }

    // 初始化hipBLAS
    hipblasHandle_t blas_handle = nullptr;
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));
    // 如果传入了非默认流，则设置流
    if (stream != nullptr) {
        HIPBLAS_CHECK(hipblasSetStream(blas_handle, stream));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // o = w * x
    // C = alpha * op(A) * op(B) + beta * C
    // op(A) is W, op(B) is X, C is O
    // W is d x n (row-major). hipBLAS expects column-major.
    // If A is W (d x n, row-major), it's seen by BLAS as A_blas (n x d, col-major).
    // We want op(A) to be d x n. So we use HIPBLAS_OP_T on A_blas.
    // (A_blas (n x d, col-major))^T  => d x n. This effectively uses W as is.
    // X is n x 1 (col-vector). hipBLAS sees it as n x 1 (col-major). Use HIPBLAS_OP_N.
    // Result O is d x 1.

    int M = d;      // 行数 of op(A) and C. op(A) is d x n.
    int N_gemm = 1; // 列数 of op(B) and C. op(B) is n x 1.
    int K = n;      // 列数 of op(A) and 行数 of op(B).

    // lda: leading dimension of A (W).
    // W is d x n (row-major). hipBLAS interprets it as A_blas (n x d, col-major).
    // Leading dimension for A_blas (col-major) is its number of rows, which is n.
    int lda = n;

    // ldb: leading dimension of B (X).
    // X is n x 1 (col-major). Leading dimension is its number of rows, which is n.
    int ldb = n;

    // ldc: leading dimension of C (O).
    // O is d x 1 (col-major). Leading dimension is its number of rows, which is d.
    int ldc = d;

    HIPBLAS_CHECK(hipblasSgemm(blas_handle,
                               HIPBLAS_OP_T, HIPBLAS_OP_N,
                               M, N_gemm, K,
                               &alpha,
                               w_d, lda,  // A (W_d), lda = n (K)
                               x_d, ldb,  // B (X_d), ldb = n (K)
                               &beta,
                               o_d, ldc   // C (O_d), ldc = d (M)
                               ));

}
// ----------------------------------------------------------------------------

// 生成随机矩阵和向量
void generateRandomData(std::vector<float>& w, std::vector<float>& x, int n, int d) {
    w.resize(d * n);
    x.resize(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 生成随机矩阵 w (d x n)
    for (int i = 0; i < d * n; ++i) {
        w[i] = dist(gen);
    }

    // 生成随机向量 x (n)
    for (int i = 0; i < n; ++i) {
        x[i] = dist(gen);
    }
}

// 检查两个结果是否匹配
bool compareResults(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 1e-4) {
    if (a.size() != b.size()) {
        fprintf(stderr, "Size mismatch: a.size() = %zu, b.size() = %zu\n", a.size(), b.size());
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tolerance) {
            // 如果需要更详细的错误，可以取消下面这行注释
            // fprintf(stderr, "Mismatch at index %zu: a=%.6f, b=%.6f, diff=%.6f\n", i, a[i], b[i], std::fabs(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

// 打印部分结果
void printResults(const std::vector<float>& data, const char* name, int max_print = 5) {
    printf("%s (前%d个): ", name, max_print);
    int count = std::min(max_print, (int)data.size());
    for (int i = 0; i < count; ++i) {
        printf("%.6f ", data[i]);
    }
    if (data.size() > (size_t)max_print) printf("...");
    printf("\n");
}


int main() {
    struct TestSize {
        int n;  // 矩阵列数/向量长度
        int d;  // 矩阵行数/输出向量长度
    };

    std::vector<TestSize> test_sizes = {
        {1, 1},
        {10, 1},
        {1, 10},
        {10, 10},
        {128, 64},
        {64, 128},
        {768, 1024}, // d > n
        {1024, 768}, // n > d
        {2048, 2048}
    };

    printf("开始测试 gpu_backend_matmul 函数的正确性...\n");
    printf("================================================\n");

    for (const auto& size : test_sizes) {
        int n = size.n;
        int d = size.d;

        printf("\n测试尺寸: W(%d x %d), X(%d x 1) => O(%d x 1)\n", d, n, n, d);

        // 1. 主机端数据分配和初始化
        std::vector<float> w_host;
        std::vector<float> x_host;
        std::vector<float> o_cpu_ref(d);
        std::vector<float> o_gpu_host(d); // 用于存储从GPU拷贝回来的结果

        generateRandomData(w_host, x_host, n, d);

        // 2. 计算CPU参考结果
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(o_cpu_ref.data(), x_host.data(), w_host.data(), n, d);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        printf("CPU 计算时间: %.4f ms\n", cpu_time.count());

        // 3. GPU端内存分配
        float *w_device = nullptr, *x_device = nullptr, *o_device = nullptr;
        size_t w_bytes = w_host.size() * sizeof(float);
        size_t x_bytes = x_host.size() * sizeof(float);
        size_t o_bytes = o_cpu_ref.size() * sizeof(float); // d * sizeof(float)

        HIP_CHECK(hipMalloc(&w_device, w_bytes));
        HIP_CHECK(hipMalloc(&x_device, x_bytes));
        HIP_CHECK(hipMalloc(&o_device, o_bytes));

        // 4. 数据从主机拷贝到设备
        HIP_CHECK(hipMemcpy(w_device, w_host.data(), w_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(x_device, x_host.data(), x_bytes, hipMemcpyHostToDevice));
        // 输出缓冲区 o_device 不需要预先清零，因为 hipblasSgemm 中的 beta=0 会覆盖它

        // 5. 执行 gpu_backend_matmul
        hipStream_t stream = nullptr; // 或者创建一个流: hipStreamCreate(&stream);
        // HIP_CHECK(hipStreamCreate(&stream)); // 如果要测试特定流

        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_backend_matmul(o_device, x_device, w_device, n, d, stream);
        HIP_CHECK(hipDeviceSynchronize()); // 等待GPU计算完成
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        printf("GPU 计算时间 (gpu_backend_matmul): %.4f ms\n", gpu_time.count());

        // 6. 结果从设备拷贝回主机
        HIP_CHECK(hipMemcpy(o_gpu_host.data(), o_device, o_bytes, hipMemcpyDeviceToHost));

        // 7. 比较结果
        bool match = compareResults(o_cpu_ref, o_gpu_host, 1e-3f); // 稍微放宽容差以应对浮点精度
        if (match) {
            printf("结果匹配: CPU 和 GPU 计算结果一致。\n");
        } else {
            printf("错误: CPU 和 GPU 计算结果不一致!\n");
        }

        // 打印部分结果进行目视检查
        if (d <= 10) { // 如果输出向量较小，全部打印
            printResults(o_cpu_ref, "CPU ref", d);
            printResults(o_gpu_host, "GPU out", d);
        } else {
            printResults(o_cpu_ref, "CPU ref");
            printResults(o_gpu_host, "GPU out");
        }

        // 8. 释放GPU内存和流
        HIP_CHECK(hipFree(w_device));
        HIP_CHECK(hipFree(x_device));
        HIP_CHECK(hipFree(o_device));
        // if (stream != nullptr) { HIP_CHECK(hipStreamDestroy(stream)); }

        printf("------------------------------------------------\n");
    }

    printf("所有测试完成。\n");
    return 0;
}