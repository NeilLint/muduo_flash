#include "../src/model/modelConfig.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

struct WeightInfo {
    std::string name;
    uint64_t offsetInBytes;
    uint64_t sizeInBytes;
    double sizeInMB;
    float percentOfTotal;
};

void printModelInfo(const CModelConfig& config) {
    std::cout << "===== 模型配置信息 =====" << std::endl;
    std::cout << "dim: " << config.dim << std::endl;
    std::cout << "feedForwardDim: " << config.feedForwardDim << std::endl;
    std::cout << "numLayers: " << config.numLayers << std::endl;
    std::cout << "numHeads: " << config.numHeads << std::endl;
    std::cout << "numKvHeads: " << config.numKvHeads << std::endl;
    std::cout << "vocabSize: " << std::abs(config.vocabSize) << std::endl;
    std::cout << "maxSeqLen: " << config.maxSeqLen << std::endl;
    std::cout << "sharedWeights: " << (config.vocabSize > 0 ? "是" : "否") << std::endl;
    std::cout << std::endl;
}

void printWeightInfo(const std::vector<WeightInfo>& weights, uint64_t totalSize) {
    std::cout << std::left << std::fixed << std::setprecision(2);
    std::cout << std::setw(25) << "权重名称" 
              << std::setw(20) << "偏移量(字节)"
              << std::setw(20) << "大小(字节)" 
              << std::setw(15) << "大小(MB)"
              << std::setw(15) << "占比(%)" << std::endl;
    
    std::cout << std::string(95, '-') << std::endl;
    
    for (const auto& w : weights) {
        std::cout << std::setw(25) << w.name 
                  << std::setw(20) << w.offsetInBytes
                  << std::setw(20) << w.sizeInBytes
                  << std::setw(15) << w.sizeInMB
                  << std::setw(15) << w.percentOfTotal << std::endl;
    }
    
    // 打印总计
    std::cout << std::string(95, '-') << std::endl;
    std::cout << std::setw(25) << "总计" 
              << std::setw(20) << ""
              << std::setw(20) << totalSize
              << std::setw(15) << totalSize / (1024.0 * 1024.0)
              << std::setw(15) << 100.0 << std::endl;
}

int main() {
    const std::string modelPath = "../data/stories110M.bin";
    CModelConfig config;
    
    // 打开文件并获取文件大小
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "错误: 无法打开模型文件 " << modelPath << std::endl;
        return 1;
    }
    
    // 获取文件总大小
    uint64_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取CModelConfig结构
    file.read(reinterpret_cast<char*>(&config), sizeof(CModelConfig));
    if (!file) {
        std::cerr << "错误: 无法读取模型配置信息" << std::endl;
        return 1;
    }
    
    // 配置参数
    const int headSize = config.dim / config.numHeads;
    const uint64_t numLayers = config.numLayers;
    const bool sharedWeights = config.vocabSize > 0;
    config.vocabSize = std::abs(config.vocabSize);
    const int kvDim = (config.dim * config.numKvHeads) / config.numHeads;
    
    // 计算各部分大小
    std::vector<WeightInfo> weights;
    uint64_t currentOffset = 0;
    
    // 配置信息
    uint64_t configSize = sizeof(CModelConfig);
    weights.push_back({"CModelConfig", 0, configSize, 
                      configSize / (1024.0 * 1024.0),
                      static_cast<float>(configSize * 100.0 / fileSize)});
    currentOffset += configSize;
    
    // tokenEmbeddingTable
    uint64_t size = config.vocabSize * config.dim * sizeof(float);
    weights.push_back({"tokenEmbeddingTable", currentOffset, size, 
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // rmsAttWeight
    size = numLayers * config.dim * sizeof(float);
    weights.push_back({"rmsAttWeight", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // wq
    size = numLayers * config.dim * config.dim * sizeof(float);
    weights.push_back({"wq", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // wk
    size = numLayers * config.dim * kvDim * sizeof(float);
    weights.push_back({"wk", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // wv
    size = numLayers * config.dim * kvDim * sizeof(float);
    weights.push_back({"wv", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // wo
    size = numLayers * config.dim * config.dim * sizeof(float);
    weights.push_back({"wo", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // rmsFfnWeight
    size = numLayers * config.dim * sizeof(float);
    weights.push_back({"rmsFfnWeight", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // w1
    size = numLayers * config.dim * config.feedForwardDim * sizeof(float);
    weights.push_back({"w1", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // w2
    size = numLayers * config.feedForwardDim * config.dim * sizeof(float);
    weights.push_back({"w2", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // w3
    size = numLayers * config.dim * config.feedForwardDim * sizeof(float);
    weights.push_back({"w3", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // rmsFinalWeight
    size = config.dim * sizeof(float);
    weights.push_back({"rmsFinalWeight", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // 位置编码
    size = config.maxSeqLen * headSize * sizeof(float);
    weights.push_back({"positionEncoding", currentOffset, size,
                      size / (1024.0 * 1024.0), 
                      static_cast<float>(size * 100.0 / fileSize)});
    currentOffset += size;
    
    // wcls (如果不共享)
    if (!sharedWeights) {
        size = config.vocabSize * config.dim * sizeof(float);
        weights.push_back({"wcls", currentOffset, size,
                          size / (1024.0 * 1024.0), 
                          static_cast<float>(size * 100.0 / fileSize)});
        currentOffset += size;
    } else {
        weights.push_back({"wcls (共享)", weights[1].offsetInBytes, 0,
                          0.0, 0.0f});
    }
    
    // 计算参数总量
    uint64_t totalParams = 0;
    for (size_t i = 1; i < weights.size(); ++i) {
        if (weights[i].name != "wcls (共享)") {
            totalParams += weights[i].sizeInBytes / sizeof(float);
        }
    }
    
    // 打印模型信息
    printModelInfo(config);
    std::cout << "文件总大小: " << fileSize << " 字节 (" << fileSize / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "参数总量: " << totalParams << " (约 " << totalParams / 1000000.0 << " M)" << std::endl;
    std::cout << std::endl;
    
    // 打印权重信息表
    printWeightInfo(weights, fileSize);
    
    // 检查计算是否准确
    if (currentOffset != fileSize) {
        std::cout << std::endl << "警告: 计算的总大小(" << currentOffset 
                  << "字节)与文件实际大小(" << fileSize << "字节)不符!" << std::endl;
        std::cout << "差距: " << (int64_t)fileSize - (int64_t)currentOffset << " 字节" << std::endl;
    }
    
    return 0;
}
