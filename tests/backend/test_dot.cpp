#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "../../src/backend/gpu_backend.hpp"

// 定义CPU版本的dot函数
void cpu_dot(float *y, const float *x1, const float* x2, int dim) {
    if (dim <= 0 || x1 == nullptr || x2 == nullptr || y == nullptr) return;

    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
        result += x1[i] * x2[i];
    }
    *y = result;
}

// 生成随机数据
void generateRandomData(std::vector<float>& x1, std::vector<float>& x2, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        x1[i] = dist(gen);
        x2[i] = dist(gen);
    }
}

// 验证结果
bool compareResults(float a, float b, float tolerance = 1e-4) {
    if (std::fabs(a - b) > tolerance) {
        printf("不匹配: CPU: %f, GPU: %f, 差值: %f\n", 
              a, b, std::fabs(a - b));
        return false;
    }
    return true;
}

int main() {
    // 测试不同大小
    std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000, 1000000};
    
    // 创建GPU后端
    GPU_Backend gpu_backend;
    
    printf("测试 DOT 函数 (点积) - CPU vs GPU 实现\n");
    printf("------------------------------------\n");
    
    for (int size : sizes) {
        printf("\n测试向量大小: %d\n", size);
        
        // 分配内存并生成数据
        std::vector<float> x1(size);      // 第一个输入向量
        std::vector<float> x2(size);      // 第二个输入向量
        float cpu_result = 0.0f;          // CPU 结果
        float gpu_result = 0.0f;          // GPU 结果
        
        // 生成随机数据
        generateRandomData(x1, x2, size);
        
        // 打印部分输入（仅当向量较小时）
        if (size <= 10) {
            printf("x1: ");
            for (int i = 0; i < size; ++i) {
                printf("%.4f ", x1[i]);
            }
            printf("\n");
            
            printf("x2: ");
            for (int i = 0; i < size; ++i) {
                printf("%.4f ", x2[i]);
            }
            printf("\n");
        }
        
        // CPU实现计时
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_dot(&cpu_result, x1.data(), x2.data(), size);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        
        // 为GPU分配设备内存
        float *d_x1 = nullptr, *d_x2 = nullptr, *d_result = nullptr;
        hipMalloc(&d_x1, size * sizeof(float));
        hipMalloc(&d_x2, size * sizeof(float));
        hipMalloc(&d_result, sizeof(float));
        
        // 将数据传输到GPU
        hipMemcpy(d_x1, x1.data(), size * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_x2, x2.data(), size * sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_result, 0, sizeof(float));  // 初始化为0
        
        // GPU实现计时
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_backend.dot(d_result, d_x1, d_x2, size);
        gpu_backend.synchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        
        // 将结果拷贝回主机
        hipMemcpy(&gpu_result, d_result, sizeof(float), hipMemcpyDeviceToHost);
        
        // 释放设备内存
        hipFree(d_x1);
        hipFree(d_x2);
        hipFree(d_result);
        
        // 验证结果
        bool results_match = compareResults(cpu_result, gpu_result);
        
        // 输出性能对比
        printf("CPU 结果: %.6f\n", cpu_result);
        printf("GPU 结果: %.6f\n", gpu_result);
        printf("CPU 时间: %.4f ms\n", cpu_time.count());
        printf("GPU 时间: %.4f ms\n", gpu_time.count());
        printf("加速比: %.2fx\n", cpu_time.count() / gpu_time.count());
        printf("结果%s匹配\n", results_match ? "" : "不");
    }
    
    return 0;
} 