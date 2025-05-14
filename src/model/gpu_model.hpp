#ifndef GPU_MODEL_HPP
#define GPU_MODEL_HPP

#include "tokenizer.hpp"
#include "modelConfig.hpp"
#include <cstdint>
#include <unistd.h>
#include <string>

#include "../infer/gpu_runState.hpp"
#include "../backend/gpu_backend.hpp"

class GPU_Model
{
public:
    CModelConfig config;
    GPU_RunState state;   
    GPU_Backend* backend; // GPU后端
    int fd;             
    float* data;         
    ssize_t fileSize;    

    struct cpu_weights {
        // 主机端权重
        float* tokenEmbeddingTable;
        float* rmsAttWeight;         // 注意力层的 RMS 权重
        float* rmsFfnWeight;         // 前馈网络层的 RMS 权重
        float* wq;
        float* wk;
        float* wv;
        float* wo;
        float* w1;
        float* w2;
        float* w3;
        float* rmsFinalWeight;
        float* wcls;
    }h_w;
    struct gpu_weights{    
        // 设备端权重
        float* d_tokenEmbeddingTable;
        float* d_rmsAttWeight;
        float* d_rmsFfnWeight;
        float* d_wqkv; // 排列顺序 q,k,v，按层排列
        float* d_wo;
        float* d_w1;
        float* d_w2;
        float* d_w3;
        float* d_rmsFinalWeight;
        float* d_wcls;
    }d_w; // 模型权重信息
    
    // 用于QKV投影的设备指针数组
    float **d_A_array;    // 权重矩阵(W_q, W_k, W_v)指针数组
    float **d_B_array;    // 输入激活值指针数组
    float **d_C_array;    // 输出(Q, K, V)指针数组
    
    GPU_Model();
    ~GPU_Model();
    
    void load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize);
    void encode(CTokenizer* t, std::string text, int8_t bos, int8_t eos, int *tokens, int *nTokens);
    float* forward(int token, int pos, GPU_Backend *backend,float *logits);
    char* decode(CTokenizer* t, int prevToken, int token);
    void initializeModel(const std::string checkpointPath);
    void mapWeightsToMemory(CModelConfig* modelConfig, float* ptr, int sharedWeights);
    void transferWeightsToDevice(); // 将权重从主机复制到设备
    void freeModel();
};

#endif