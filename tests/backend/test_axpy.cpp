#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "../../src/backend/gpu_backend.hpp"

// 定义CPU版本的axpy函数
void cpu_axpy(float *y, const float *x, float factor, int dim) {
    if (dim <= 0 || x == nullptr || y == nullptr) return;

    for (int i = 0; i < dim; ++i) {
        y[i] += x[i] * factor;
    }
}

// 生成随机数据
void generateRandomData(std::vector<float>& x, std::vector<float>& y, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
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

// 打印部分结果
void printResults(const std::vector<float>& data, const char* name, int max_print = 5) {
    printf("%s (前%d个): ", name, max_print);
    for (int i = 0; i < std::min(max_print, (int)data.size()); ++i) {
        printf("%.6f ", data[i]);
    }
    if (data.size() > max_print) printf("...");
    printf("\n");
}

int main() {
    // 测试不同大小和不同因子
    std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000};
    std::vector<float> factors = {0.5f, 1.0f, 2.0f, -1.0f};
    
    // 创建GPU后端
    GPU_Backend gpu_backend;
    
    printf("测试 AXPY 函数 (y += factor * x) - CPU vs GPU 实现\n");
    printf("----------------------------------------------\n");
    
    for (int size : sizes) {
        for (float factor : factors) {
            printf("\n测试向量大小: %d, 因子: %.2f\n", size, factor);
            
            // 分配内存并生成数据
            std::vector<float> x(size);          // 输入向量 x
            std::vector<float> cpu_y(size);      // CPU 输出向量 y (将被更新)
            std::vector<float> gpu_y(size);      // GPU 输出向量 y (将被更新)
            
            // 生成随机数据
            generateRandomData(x, cpu_y, size);
            gpu_y = cpu_y;  // 确保初始值相同
            
            // 打印部分输入
            if (size <= 10) {
                printResults(x, "输入 x", size);
                printResults(cpu_y, "初始 y", size);
            } else {
                printResults(x, "输入 x");
                printResults(cpu_y, "初始 y");
            }
            
            // CPU实现计时
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_axpy(cpu_y.data(), x.data(), factor, size);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
            
            // 为GPU分配设备内存
            float *d_x = nullptr, *d_y = nullptr;
            size_t data_size = size * sizeof(float);
            hipMalloc(&d_x, data_size);
            hipMalloc(&d_y, data_size);
            
            // 将数据传输到GPU
            hipMemcpy(d_x, x.data(), data_size, hipMemcpyHostToDevice);
            hipMemcpy(d_y, gpu_y.data(), data_size, hipMemcpyHostToDevice);
            
            // GPU实现计时
            auto gpu_start = std::chrono::high_resolution_clock::now();
            gpu_backend.axpy(d_y, d_x, factor, size);
            gpu_backend.synchronize();
            auto gpu_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
            
            // 将结果拷贝回主机
            hipMemcpy(gpu_y.data(), d_y, data_size, hipMemcpyDeviceToHost);
            
            // 释放设备内存
            hipFree(d_x);
            hipFree(d_y);
            
            // 验证结果
            bool results_match = compareResults(cpu_y, gpu_y);
            
            // 输出性能对比
            printf("CPU 时间: %.4f ms\n", cpu_time.count());
            printf("GPU 时间: %.4f ms\n", gpu_time.count());
            printf("加速比: %.2fx\n", cpu_time.count() / gpu_time.count());
            printf("结果%s匹配\n", results_match ? "" : "不");
            
            // 打印部分结果
            if (size <= 10) {
                printResults(cpu_y, "CPU输出", size);
                printResults(gpu_y, "GPU输出", size);
            } else {
                printResults(cpu_y, "CPU输出");
                printResults(gpu_y, "GPU输出");
            }
        }
    }
    
    return 0;
} 