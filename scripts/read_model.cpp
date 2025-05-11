#include "../src/model/modelConfig.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

void printModelConfig(const CModelConfig& config) {
    std::cout << "===== 模型配置信息 =====" << std::endl;
    std::cout << "dim: " << config.dim << std::endl;
    std::cout << "feedForwardDim: " << config.feedForwardDim << std::endl;
    std::cout << "numLayers: " << config.numLayers << std::endl;
    std::cout << "numHeads: " << config.numHeads << std::endl;
    std::cout << "numKvHeads: " << config.numKvHeads << std::endl;
    std::cout << "vocabSize: " << std::abs(config.vocabSize) << std::endl;
    std::cout << "maxSeqLen: " << config.maxSeqLen << std::endl;
    std::cout << "sharedWeights: " << (config.vocabSize > 0 ? "是" : "否") << std::endl;
}

int main() {
    const std::string modelPath = "../data/stories110M.bin";
    CModelConfig modelConfig;
    
    // 以二进制模式打开文件
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        std::cerr << "错误: 无法打开模型文件 " << modelPath << std::endl;
        return 1;
    }
    
    // 读取CModelConfig结构
    file.read(reinterpret_cast<char*>(&modelConfig), sizeof(CModelConfig));
    if (!file) {
        std::cerr << "错误: 无法读取模型配置信息" << std::endl;
        return 1;
    }
    
    // 打印模型配置信息
    printModelConfig(modelConfig);
    
    return 0;
}