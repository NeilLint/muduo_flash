#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "../../src/backend/gpu_backend.hpp"

// 定义CPU版本的gemvQkSeq函数
void cpu_gemvQkSeq(float *q, float *key, float *attentionScores, int pos, int kvDim, int headSize) {
    for (int timestep = 0; timestep <= pos; timestep++) {
        float* k = key + timestep * kvDim;
        float score = 0.0f;
        
        // 点积
        for (int i = 0; i < headSize; ++i) {
            score += q[i] * k[i];
        }
        
        // 缩放
        score /= sqrtf(headSize);
        attentionScores[timestep] = score;
    }
}

// 生成随机数据
void generateRandomData(std::vector<float>& q, std::vector<float>& key, int pos, int kvDim, int headSize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // 生成随机q向量
    for (int i = 0; i < headSize; ++i) {
        q[i] = dist(gen);
    }
    
    // 生成随机key矩阵（所有时间步）
    for (int t = 0; t <= pos; ++t) {
        for (int i = 0; i < kvDim; ++i) {
            key[t * kvDim + i] = dist(gen);
        }
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
    // 测试不同的配置
    struct TestConfig {
        int pos;      // 当前序列位置
        int kvDim;    // key/value向量的维度
        int headSize; // 注意力头的大小
    };
    
    std::vector<TestConfig> configs = {
        {0, 64, 64},     // 单个token
        {9, 768, 64},    // 10个token，常见模型维度
        {49, 768, 64},   // 50个token
        {99, 1024, 64},  // 100个token，更大的维度
    };
    
    // 创建GPU后端
    GPU_Backend gpu_backend;
    
    printf("测试 GemvQkSeq 函数 - CPU vs GPU 实现\n");
    printf("--------------------------------------\n");
    
    for (const auto& config : configs) {
        int pos = config.pos;
        int kvDim = config.kvDim;
        int headSize = config.headSize;
        int num = pos + 1; // 计算序列长度
        
        printf("\n测试配置: pos=%d, kvDim=%d, headSize=%d (序列长度=%d)\n", 
               pos, kvDim, headSize, num);
        
        // 分配内存
        std::vector<float> q(kvDim, 0);                 // 查询向量
        std::vector<float> key(num * kvDim, 0);         // key矩阵，每个时间步一行
        std::vector<float> cpu_scores(num, 0);          // CPU结果
        std::vector<float> gpu_scores(num, 0);          // GPU结果
        
        // 生成随机数据
        generateRandomData(q, key, pos, kvDim, headSize);
        
        // CPU实现计时
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemvQkSeq(q.data(), key.data(), cpu_scores.data(), pos, kvDim, headSize);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        
        // 为GPU分配设备内存
        float *d_q = nullptr, *d_key = nullptr, *d_scores = nullptr;
        hipMalloc(&d_q, q.size() * sizeof(float));
        hipMalloc(&d_key, key.size() * sizeof(float));
        hipMalloc(&d_scores, num * sizeof(float));
        
        // 将数据传输到GPU
        hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_key, key.data(), key.size() * sizeof(float), hipMemcpyHostToDevice);
        
        // GPU实现计时
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_backend.gemvQkSeq(d_q, d_key, d_scores, pos, kvDim, headSize);
        gpu_backend.synchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        
        // 将结果拷贝回主机
        hipMemcpy(gpu_scores.data(), d_scores, num * sizeof(float), hipMemcpyDeviceToHost);
        
        // 释放设备内存
        hipFree(d_q);
        hipFree(d_key);
        hipFree(d_scores);
        
        // 验证结果
        bool results_match = compareResults(cpu_scores, gpu_scores);
        
        // 输出性能对比
        printf("CPU 时间: %.4f ms\n", cpu_time.count());
        printf("GPU 时间: %.4f ms\n", gpu_time.count());
        printf("加速比: %.2fx\n", cpu_time.count() / gpu_time.count());
        printf("结果%s匹配\n", results_match ? "" : "不");
        
        // 打印部分结果
        printResults(cpu_scores, "CPU结果", std::min(num, 10));
        printResults(gpu_scores, "GPU结果", std::min(num, 10));
    }
    
    return 0;
} 