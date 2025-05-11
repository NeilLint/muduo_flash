#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

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

// CPU版本的矩阵-向量乘法
/*  TODO 矩阵向量乘法加速执行  
    matmul: 矩阵-向量乘 
        o[d][1] = w[d][n] X x[n][1]
*/
void matmul(float* o, float* x, float* w, int n, int d) {
    // o[d] = w[d][0] * x[0] + w[d][1] * x[1] + ... + w[d][n-1] * x[n-1]
    
    // 遍历矩阵的每一行 d
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        
        // 对矩阵的每一列 n 进行求和
        for (int j = 0; j < n; ++j) {
            sum += w[i * n + j] * x[j];  // w[i][j] 对应的内存位置是 w[i * n + j]
        }
        
        // 将求和结果存入输出向量 o[i]
        o[i] = sum;
    }
}

// GPU内核函数 - 每个线程计算输出向量的一个元素
__global__ void matmul_gemv_kernel_v2(float* o, const float* x, const float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < d) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += w[i * n + j] * x[j];
        }
        o[i] = sum;
    }
}

// 使用手动HIP内核实现的GPU矩阵-向量乘法
void matmul_gpu(float* o, float* x, float* w, int n, int d) {
    // 输入参数检查
    if (!o || !x || !w || n <= 0 || d <= 0) {
        fprintf(stderr, "Matmul Error: Invalid input pointers or dimensions.\n");
        return;
    }

    // --- rocBLAS 推荐 ---
    // 对于标准的 GEMV 操作，强烈推荐使用 rocBLAS 库中的 hipblasSgemv 函数。
    // 它通常比手动编写的内核性能更好，因为它经过了针对特定硬件的深度优化。
    // 如果使用 rocBLAS，这里的代码会是调用 hipblasSgemv，而不是下面的手动内核流程。
    // --------------------

    // --- 手动 HIP 内核实现 ---

    // 1. 定义设备端指针
    float *d_o = nullptr, *d_x = nullptr, *d_w = nullptr;

    // 2. 计算内存大小
    size_t o_size = d * sizeof(float);
    size_t x_size = n * sizeof(float);
    size_t w_size = (size_t)d * n * sizeof(float); // 使用 size_t 避免大矩阵时溢出

    // 3. 分配设备内存
    HIP_CHECK(hipMalloc(&d_o, o_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_w, w_size));

    // 4. 将输入数据从主机内存复制到设备内存
    // o 是输出，不需要从主机拷贝到设备
    HIP_CHECK(hipMemcpy(d_x, x, x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w, w, w_size, hipMemcpyHostToDevice));

    // 5. 配置内核启动参数
    int block_size = 256; // 每个块的线程数 (可调整的参数)
    // 计算需要的块数，确保覆盖所有 'd' 个输出元素
    int grid_size = (d + block_size - 1) / block_size;

    dim3 gridDim(grid_size); // 网格维度
    dim3 blockDim(block_size); // 块维度

    // 6. 启动 HIP 内核
    hipLaunchKernelGGL(matmul_gemv_kernel_v2, gridDim, blockDim, 0, 0, d_o, d_x, d_w, n, d);
    HIP_CHECK(hipGetLastError()); // 检查内核启动异步错误

    // 可选：同步设备以确保内核完成 (通常 hipMemcpyDeviceToHost 会隐式同步)
    // HIP_CHECK(hipDeviceSynchronize());

    // 7. 将计算结果从设备内存复制回主机内存
    HIP_CHECK(hipMemcpy(o, d_o, o_size, hipMemcpyDeviceToHost));

    // 8. 释放设备内存
    HIP_CHECK(hipFree(d_o));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_w));
}

// 使用hipBLAS库实现的GPU矩阵-向量乘法
void matmul_gpu_hipblas(float* o, float* x, float* w, int n, int d) {
    // 输入参数检查
    if (!o || !x || !w || n <= 0 || d <= 0) {
        fprintf(stderr, "Matmul Error: Invalid input pointers or dimensions.\n");
        return;
    }

    // 1. 初始化hipBLAS
    hipblasHandle_t blas_handle;
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));

    // 2. 定义设备端指针
    float *d_o = nullptr, *d_x = nullptr, *d_w = nullptr;

    // 3. 计算内存大小
    size_t o_size = d * sizeof(float);
    size_t x_size = n * sizeof(float);
    size_t w_size = (size_t)d * n * sizeof(float);

    // 4. 分配设备内存
    HIP_CHECK(hipMalloc(&d_o, o_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_w, w_size));

    // 5. 将输入数据从主机内存复制到设备内存
    HIP_CHECK(hipMemcpy(d_x, x, x_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w, w, w_size, hipMemcpyHostToDevice));

    // 6. 设置alpha和beta系数
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 7. 调用hipBLAS函数
    // 行主元矩阵W(d×n)，使用HIPBLAS_OP_N
    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_N,  // 不转置矩阵
        d,             // 矩阵行数
        n,             // 矩阵列数
        &alpha,        // alpha系数
        d_w,           // 矩阵指针
        n,             // 行间距
        d_x,           // 向量x指针
        1,             // x的增量
        &beta,         // beta系数
        d_o,           // 输出向量o指针
        1              // o的增量
    ));

    // 8. 将结果从设备内存复制回主机内存
    HIP_CHECK(hipMemcpy(o, d_o, o_size, hipMemcpyDeviceToHost));

    // 9. 释放资源
    HIP_CHECK(hipFree(d_o));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_w));
    HIPBLAS_CHECK(hipblasDestroy(blas_handle));
}

// 生成随机矩阵和向量
void generateRandomData(float* w, float* x, int n, int d) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // 生成随机矩阵 w (d x n)
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < n; ++j) {
            w[i * n + j] = dist(gen);
        }
    }
    
    // 生成随机向量 x (n)
    for (int i = 0; i < n; ++i) {
        x[i] = dist(gen);
    }
}

// 验证两个结果是否近似相等
bool validateResults(float* o1, float* o2, int d, float tolerance) {
    for (int i = 0; i < d; ++i) {
        if (fabs(o1[i] - o2[i]) > tolerance) {
            printf("结果不匹配 在索引 %d: CPU = %f, GPU = %f, 差值 = %f\n",
                   i, o1[i], o2[i], fabs(o1[i] - o2[i]));
            return false;
        }
    }
    return true;
}

// 打印矩阵和向量（用于调试）
void printData(float* w, float* x, float* o, int n, int d, bool full = false) {
    // 打印部分矩阵 w
    printf("矩阵 W (前5行5列):\n");
    for (int i = 0; i < std::min(5, d); ++i) {
        for (int j = 0; j < std::min(5, n); ++j) {
            printf("%8.4f ", w[i * n + j]);
        }
        printf("...\n");
    }
    printf("...\n\n");
    
    // 打印向量 x
    printf("向量 X (前5个元素):");
    for (int i = 0; i < std::min(5, n); ++i) {
        printf(" %8.4f", x[i]);
    }
    if (n > 5) printf(" ...");
    printf("\n\n");
    
    // 打印输出向量 o
    printf("输出向量 O (前5个元素):");
    for (int i = 0; i < std::min(5, d); ++i) {
        printf(" %8.4f", o[i]);
    }
    if (d > 5) printf(" ...");
    printf("\n\n");
}

int main() {
    // 设置矩阵和向量尺寸
    const int n = 1024;  // 矩阵列数，向量长度
    const int d = 768;   // 矩阵行数，输出向量长度
    
    // 分配主机内存
    float* w = (float*)malloc(d * n * sizeof(float));  // 矩阵
    float* x = (float*)malloc(n * sizeof(float));      // 输入向量
    float* o_cpu = (float*)malloc(d * sizeof(float));  // CPU结果
    float* o_gpu = (float*)malloc(d * sizeof(float));  // 自定义GPU内核结果
    float* o_hipblas = (float*)malloc(d * sizeof(float)); // hipBLAS结果
    
    if (!w || !x || !o_cpu || !o_gpu || !o_hipblas) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    
    // 生成随机测试数据
    generateRandomData(w, x, n, d);
    
    // 1. 执行CPU版本并测量时间
    printf("执行CPU版本的矩阵-向量乘法 (%dx%d)...\n", d, n);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul(o_cpu, x, w, n, d);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    
    // 2. 执行自定义GPU内核版本并测量时间
    printf("执行自定义GPU内核版本的矩阵-向量乘法...\n");
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matmul_gpu(o_gpu, x, w, n, d);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    
    // 3. 执行hipBLAS版本并测量时间
    printf("执行hipBLAS版本的矩阵-向量乘法...\n");
    auto start_hipblas = std::chrono::high_resolution_clock::now();
    matmul_gpu_hipblas(o_hipblas, x, w, n, d);
    auto end_hipblas = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hipblas_time = end_hipblas - start_hipblas;
    
    // 4. 验证结果
    printf("\n验证结果：\n");
    const float tolerance = 1e-4;  // 误差容限
    bool cpu_gpu_match = validateResults(o_cpu, o_gpu, d, tolerance);
    bool cpu_hipblas_match = validateResults(o_cpu, o_hipblas, d, tolerance);
    bool gpu_hipblas_match = validateResults(o_gpu, o_hipblas, d, tolerance);
    
    printf("CPU 和 自定义GPU结果%s匹配\n", cpu_gpu_match ? "" : "不");
    printf("CPU 和 hipBLAS结果%s匹配\n", cpu_hipblas_match ? "" : "不");
    printf("自定义GPU 和 hipBLAS结果%s匹配\n", gpu_hipblas_match ? "" : "不");
    
    // 5. 打印性能比较
    printf("\n性能比较：\n");
    printf("CPU 时间: %.4f ms\n", cpu_time.count());
    printf("自定义GPU 时间: %.4f ms (加速比: %.2fx)\n", 
           gpu_time.count(), cpu_time.count() / gpu_time.count());
    printf("hipBLAS 时间: %.4f ms (加速比: %.2fx)\n", 
           hipblas_time.count(), cpu_time.count() / hipblas_time.count());
    
    // 释放主机内存
    free(w);
    free(x);
    free(o_cpu);
    free(o_gpu);
    free(o_hipblas);
    
    return 0;
}