#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "../../src/backend/gpu_backend.hpp"

// 定义CPU后端的softmax函数实现
void cpu_softmax(float* x, int n) {
    if (n <= 0 || x == nullptr) return;

    // 为了数值稳定性，先减去最大值
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    // 计算 e^(x[i] - max_val) 和 sum
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // 归一化
    for (int i = 0; i < n; ++i) {
        x[i] /= sum;
    }
}

// 生成随机数据
void generateRandomData(std::vector<float>& data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

// 验证结果
bool compareResults(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 1e-4) {
    if (a.size() != b.size()) return false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tolerance) {
            printf("不匹配: 索引 %zu, CPU: %f, GPU: %f, 差值: %f\n", 
                  i, a[i], b[i], std::fabs(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

int main() {
    // 测试不同大小
    std::vector<int> sizes = {1, 10, 100, 1000, 10000};
    
    // 创建GPU后端
    GPU_Backend gpu_backend;
    
    printf("测试 Softmax 函数 - CPU vs GPU 实现\n");
    printf("-----------------------------------\n");
    
    for (int size : sizes) {
        printf("\n测试向量大小: %d\n", size);
        
        // 分配内存并生成数据
        std::vector<float> cpu_data(size);
        std::vector<float> gpu_data(size);
        
        // 生成相同的随机数据
        generateRandomData(cpu_data, size);
        gpu_data = cpu_data; // 复制数据以确保两者输入相同
        
        // CPU实现计时
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_softmax(cpu_data.data(), size);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        
        // GPU实现计时，需要先将数据拷贝到设备
        float* gpu_device_data = nullptr;
        size_t data_size = size * sizeof(float);
        hipMalloc(&gpu_device_data, data_size);
        hipMemcpy(gpu_device_data, gpu_data.data(), data_size, hipMemcpyHostToDevice);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_backend.softmax(gpu_device_data, size);
        gpu_backend.synchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        
        // 将结果拷贝回主机
        hipMemcpy(gpu_data.data(), gpu_device_data, data_size, hipMemcpyDeviceToHost);
        hipFree(gpu_device_data);
        
        // 验证结果
        bool results_match = compareResults(cpu_data, gpu_data);
        
        // 输出性能对比
        printf("CPU 时间: %.4f ms\n", cpu_time.count());
        printf("GPU 时间: %.4f ms\n", gpu_time.count());
        printf("加速比: %.2fx\n", cpu_time.count() / gpu_time.count());
        printf("结果%s匹配\n", results_match ? "" : "不");
        
        // 打印部分结果作为参考
        if (size <= 10) {
            printf("CPU结果: ");
            for (int i = 0; i < size; ++i) {
                printf("%.6f ", cpu_data[i]);
            }
            printf("\n");
            
            printf("GPU结果: ");
            for (int i = 0; i < size; ++i) {
                printf("%.6f ", gpu_data[i]);
            }
            printf("\n");
        } else {
            printf("CPU结果(前5个): ");
            for (int i = 0; i < 5; ++i) {
                printf("%.6f ", cpu_data[i]);
            }
            printf("...\n");
            
            printf("GPU结果(前5个): ");
            for (int i = 0; i < 5; ++i) {
                printf("%.6f ", gpu_data[i]);
            }
            printf("...\n");
        }
    }
    
    return 0;
} 