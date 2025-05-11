#include <hip/hip_runtime.h> // HIP运行时头文件
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // 使用 expf, fmaxf
#include <float.h>  // 使用 -FLT_MAX

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
 * Softmax 计算内核，使用共享内存进行规约。
 * 假设一个块处理整个向量。
 * 为了这个规约实现，blockDim.x 最好是 2 的幂。
 */
__global__ void softmax_kernel(float* out, const float* in, int n) {
    // 用于规约的共享内存 (最大值和指数和)
    // 我们为两种规约复用同一块共享内存空间。
    extern __shared__ float s_data[];

    int tid = threadIdx.x;          // 块内线程 ID
    int block_size = blockDim.x;    // 块中的线程数

    // --- 阶段 1: 查找最大值 ---
    float thread_max = -FLT_MAX; // 初始化此线程的最大值

    // 每个线程负责处理一部分元素，查找这些元素中的最大值
    // 使用 grid-stride 循环，这样即使 n > block_size 也能正确处理
    for (int i = tid; i < n; i += block_size) {
        thread_max = fmaxf(thread_max, in[i]);
    }

    // 将线程的局部最大值存储在共享内存中
    s_data[tid] = thread_max;
    __syncthreads(); // 同步，确保所有线程都已写入其局部最大值

    // 在共享内存中执行并行规约（求最大值）
    // 假设 block_size 是 2 的幂
    for (int s = block_size / 2; s > 0; s >>= 1) {
        // 只让前半部分的线程参与合并
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads(); // 每次规约步骤后同步
    }

    // 整个向量（块处理范围）的最大值现在位于 s_data[0]
    float block_max = s_data[0];
    __syncthreads(); // 确保所有线程在继续之前都读取了正确的 block_max

    // --- 阶段 2: 计算指数和 ---
    float thread_sum_exp = 0.0f; // 初始化此线程的指数和

    // 每个线程为其元素计算 exp(x_i - max)，并将它们加起来
    for (int i = tid; i < n; i += block_size) {
        thread_sum_exp += expf(in[i] - block_max);
    }

    // 将线程的局部和存储在共享内存中 (复用 s_data)
    s_data[tid] = thread_sum_exp;
    __syncthreads(); // 确保所有线程都已写入其和

    // 在共享内存中执行并行规约（求和）
    // 假设 block_size 是 2 的幂
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads(); // 每次规约步骤后同步
    }

    // 指数的总和现在位于 s_data[0]
    float block_sum_exp = s_data[0];
     __syncthreads(); // 确保所有线程在继续之前都读取了正确的 block_sum_exp

    // --- 阶段 3: 归一化并写入输出 ---
    // 每个线程计算其元素的最终 softmax 值
    for (int i = tid; i < n; i += block_size) {
        // 防止 block_sum_exp 为 0 导致除零错误 (虽然在 softmax 中不太可能，但可以加个保护)
        if (block_sum_exp > 0.0f) {
             out[i] = expf(in[i] - block_max) / block_sum_exp;
        } else {
             // 如果 sum 为 0，则所有 exp(x_i - max) 都为 0
             // 这意味着所有 x_i 都非常小（趋近负无穷）或相等且非常小
             // 均匀分布或者设为0可能是合理的处理，这里设为0
             out[i] = 0.0f;
             // 或者可以根据需要分配均匀概率 1.0f / n
             // out[i] = 1.0f / (float)n;
        }
    }
}

// 主机端函数，用于调用 softmax 内核
void hip_softmax(float* h_out, const float* h_in, int n) {
    float *d_in = nullptr, *d_out = nullptr;
    size_t data_size = n * sizeof(float); // 数据大小（字节）

    // 1. 分配设备内存 (GPU显存)
    HIP_CHECK(hipMalloc(&d_in, data_size));
    HIP_CHECK(hipMalloc(&d_out, data_size));

    // 2. 将输入数据从主机内存复制到设备内存
    HIP_CHECK(hipMemcpy(d_in, h_in, data_size, hipMemcpyHostToDevice));

    // 3. 配置内核启动参数
    // 选择一个块大小 (推荐为 2 的幂，以优化规约)
    // 常见的选择是 128, 256, 512, 1024，取决于 GPU 能力和问题规模
    int block_size = 256;
    // 如果 n 太小，可以调整 block_size，但下面的内核实现假设 block_size >= n
    // 或者更简单的做法是，即使 n < block_size，也启动 block_size 个线程，
    // 内核中的判断 (tid < n 或循环条件 i < n) 会确保只有需要的线程工作。
    // 此处保持 block_size 为 256，让内核中的 grid-stride 循环处理。
    // 注意：block_size 不能超过设备的限制 (通常是 1024)。

    // 我们使用一个块来处理整个向量 'n'
    dim3 gridDim(1); // 1个块
    dim3 blockDim(block_size); // 每个块 block_size 个线程

    // 计算所需的共享内存大小：需要 block_size 个 float 用于规约
    size_t shared_mem_size = block_size * sizeof(float);

    // 4. 启动内核
    // hipLaunchKernelGGL(kernel_name, gridDim, blockDim, sharedMemBytes, stream, args...)
    hipLaunchKernelGGL(softmax_kernel, gridDim, blockDim, shared_mem_size, 0, d_out, d_in, n);
    HIP_CHECK(hipGetLastError()); // 检查内核启动是否有错误

    // 可选：同步设备，确保内核执行完毕再拷贝回结果
    // 对于阻塞型的 hipMemcpyDeviceToHost，这通常是隐式完成的。
    // HIP_CHECK(hipDeviceSynchronize());

    // 5. 将输出数据从设备内存复制回主机内存
    HIP_CHECK(hipMemcpy(h_out, d_out, data_size, hipMemcpyDeviceToHost));

    // 6. 释放设备内存
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}

// --- 用于测试的主函数 ---
int main() {
    int n = 10; // 示例向量大小
    float h_in[n];
    float h_out[n];
    float h_out_cpu[n]; // 用于 CPU 验证

    // 初始化输入数据 (示例)
    printf("输入向量 (n=%d):\n", n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = (float)(i + 1); // 示例: 1.0, 2.0, ..., 10.0
        printf("%.2f ", h_in[i]);
    }
    printf("\n");

    // --- CPU Softmax 用于验证 ---
    // 使用 double 以提高 CPU 计算的精度
    double cpu_max = -DBL_MAX;
    for(int i = 0; i < n; ++i) {
        if ((double)h_in[i] > cpu_max) {
            cpu_max = (double)h_in[i];
        }
    }
    double cpu_sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_sum_exp += exp((double)h_in[i] - cpu_max);
    }
    printf("\nCPU 验证:\n");
    printf("CPU 最大值: %.6f\n", cpu_max);
    printf("CPU 指数和: %.6f\n", cpu_sum_exp);
    printf("CPU Softmax 输出:\n");
    double cpu_check_sum = 0.0;
     for (int i = 0; i < n; ++i) {
        h_out_cpu[i] = (float)(exp((double)h_in[i] - cpu_max) / cpu_sum_exp);
        printf("%.6f ", h_out_cpu[i]);
        cpu_check_sum += h_out_cpu[i];
    }
    printf("\nCPU 概率和: %.6f\n", cpu_check_sum); // 应接近 1.0
    // --- CPU 验证结束 ---


    // --- GPU Softmax 计算 ---
    printf("\n执行 HIP Softmax...\n");
    hip_softmax(h_out, h_in, n);
    // --- GPU 计算结束 ---


    // --- 打印 GPU 结果 ---
    printf("\nHIP Softmax 输出:\n");
    double gpu_check_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        printf("%.6f ", h_out[i]);
         gpu_check_sum += h_out[i];
    }
    printf("\nGPU 概率和: %.6f\n", gpu_check_sum); // 应接近 1.0

    // --- 比较 CPU 和 GPU 结果 ---
    printf("\n比较 CPU 和 GPU 结果...\n");
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        double error = fabs((double)h_out[i] - (double)h_out_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("最大绝对误差: %.8f\n", max_error);
    // 设置一个小的容忍度，因为浮点计算可能存在微小差异
    if (max_error < 1e-6) {
        printf("结果: 通过\n");
    } else {
        printf("结果: 失败\n");
    }

    return 0;
}