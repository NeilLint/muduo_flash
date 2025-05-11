#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// 用于HIP错误检查的辅助宏
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP 错误: %s (%d) 在 %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/*
 * Softmax 计算内核 (原地修改版本)
 * 从 data 读取原始值，计算后将结果写回 data
 * 假设一个块处理整个向量。
 * blockDim.x 最好是 2 的幂。
 */
__global__ void softmax_kernel_inplace(float* data, int size) {
    // 用于规约的共享内存 (最大值和指数和)
    extern __shared__ float s_data[];

    int tid = threadIdx.x;          // 块内线程 ID
    int block_size = blockDim.x;    // 块中的线程数

    // --- 阶段 1: 查找最大值 ---
    float thread_max = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        thread_max = fmaxf(thread_max, data[i]); // 从 data 读取
    }
    s_data[tid] = thread_max;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float block_max = s_data[0];
    __syncthreads();

    // --- 阶段 2: 计算指数和 ---
    float thread_sum_exp = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        thread_sum_exp += expf(data[i] - block_max); // 从 data 读取
    }
    s_data[tid] = thread_sum_exp;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float block_sum_exp = s_data[0];
    __syncthreads();

    // --- 阶段 3: 归一化并写回输出 (原地修改) ---
    for (int i = tid; i < size; i += block_size) {
        float exp_val = expf(data[i] - block_max); // 从 data 读取原始值相关的计算
        if (block_sum_exp > 0.0f) {
             data[i] = exp_val / block_sum_exp; // 将结果写回 data
        } else {
             data[i] = 0.0f; // 处理 sum 为 0 的情况
             // 或者 data[i] = 1.0f / (float)size;
        }
    }
}

// 主机端函数，符合 void softmax(float* x, int size) 签名
// 在 GPU 上执行计算，并将结果写回 x 指向的内存
void softmax(float* x, int size) {
    if (x == nullptr || size <= 0) {
        fprintf(stderr, "错误：输入指针为空或大小无效。\n");
        return;
    }

    float *d_data = nullptr; // 设备端缓冲区指针
    size_t data_size = size * sizeof(float);

    // 1. 分配设备内存
    HIP_CHECK(hipMalloc(&d_data, data_size));

    // 2. 将输入数据从主机 x 复制到设备 d_data
    HIP_CHECK(hipMemcpy(d_data, x, data_size, hipMemcpyHostToDevice));

    // 3. 配置内核启动参数
    int block_size = 256; // 可以根据需要调整
    // 确保 block_size 不小于 size (如果 size 很小) 或不超过设备限制
    // 这里简单处理：如果 size 小于 block_size，多余的线程在内核中不会执行有效工作
    // block_size = (size < block_size) ? nextPowerOf2(size) : block_size; // 更精细的调整

    dim3 gridDim(1); // 仍然使用一个块处理
    dim3 blockDim(block_size);
    size_t shared_mem_size = block_size * sizeof(float); // 共享内存大小

    // 4. 启动原地修改内核
    hipLaunchKernelGGL(softmax_kernel_inplace, gridDim, blockDim, shared_mem_size, 0, d_data, size);
    HIP_CHECK(hipGetLastError()); // 检查内核启动错误

    // 5. 将计算结果从设备 d_data 复制回主机 x
    HIP_CHECK(hipMemcpy(x, d_data, data_size, hipMemcpyDeviceToHost));

    // 6. 释放设备内存
    HIP_CHECK(hipFree(d_data));
}

// --- 用于测试的主函数 ---
int main() {
    int n = 10; // 示例向量大小
    float h_data[n]; // 现在只有一个主机数组
    float h_out_cpu[n]; // 用于 CPU 验证

    // 初始化输入数据 (示例)
    printf("输入向量 (n=%d):\n", n);
    for (int i = 0; i < n; ++i) {
        h_data[i] = (float)(i + 1); // 示例: 1.0, 2.0, ..., 10.0
        printf("%.2f ", h_data[i]);
    }
    printf("\n");

    // --- CPU Softmax 用于验证 ---
    // (CPU 验证部分保持不变，但使用 h_data 作为输入)
    double cpu_max = -DBL_MAX;
    for(int i = 0; i < n; ++i) {
        if ((double)h_data[i] > cpu_max) {
            cpu_max = (double)h_data[i];
        }
    }
    double cpu_sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_sum_exp += exp((double)h_data[i] - cpu_max);
    }
    printf("\nCPU 验证 (用于比较):\n");
    // printf("CPU 最大值: %.6f\n", cpu_max);
    // printf("CPU 指数和: %.6f\n", cpu_sum_exp);
    printf("CPU Softmax 预期输出:\n");
    double cpu_check_sum = 0.0;
     for (int i = 0; i < n; ++i) {
        h_out_cpu[i] = (float)(exp((double)h_data[i] - cpu_max) / cpu_sum_exp);
        printf("%.6f ", h_out_cpu[i]);
        cpu_check_sum += h_out_cpu[i];
    }
    printf("\nCPU 概率和: %.6f\n", cpu_check_sum);
    // --- CPU 验证结束 ---

    // --- GPU Softmax 计算 (调用新的接口) ---
    printf("\n执行 HIP Softmax (原地修改 h_data)...\n");
    softmax(h_data, n); // 调用新的函数签名
    // --- GPU 计算结束 ---

    // --- 打印 GPU 结果 (现在直接打印 h_data) ---
    printf("\nHIP Softmax 输出 (修改后的 h_data):\n");
    double gpu_check_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        printf("%.6f ", h_data[i]); // 直接打印修改后的 h_data
         gpu_check_sum += h_data[i];
    }
    printf("\nGPU 概率和: %.6f\n", gpu_check_sum);

    // --- 比较 CPU 和 GPU 结果 ---
    printf("\n比较 CPU 预期结果和 GPU 修改后的 h_data...\n");
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        double error = fabs((double)h_data[i] - (double)h_out_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("最大绝对误差: %.8f\n", max_error);
    if (max_error < 1e-6) {
        printf("结果: 通过\n");
    } else {
        printf("结果: 失败\n");
    }

    return 0;
}