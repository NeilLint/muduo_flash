#include "gpu_model.hpp"
#include "tokenizer.hpp"
#include "modelConfig.hpp"
// #include "../backend/backend.hpp"
#include "../backend/gpu_backend.hpp"

#include <cmath>
#include <iostream>
#include <fstream> 
#include <fcntl.h>
#include <sys/mman.h>

GPU_Model::GPU_Model() : fd(-1), data(nullptr), fileSize(0)
{
    // 置主机指针为空指针
    h_w.tokenEmbeddingTable = nullptr;
    h_w.rmsAttWeight = nullptr;
    h_w.rmsFfnWeight = nullptr;
    h_w.wq = nullptr;
    h_w.wk = nullptr;
    h_w.wv = nullptr;
    h_w.wo = nullptr;
    h_w.w1 = nullptr;
    h_w.w2 = nullptr;
    h_w.w3 = nullptr;
    h_w.rmsFinalWeight = nullptr;
    h_w.wcls = nullptr;

    // 置设备指针为空指针
    d_w.d_tokenEmbeddingTable = nullptr;
    d_w.d_rmsAttWeight = nullptr;
    d_w.d_rmsFfnWeight = nullptr;
    d_w.d_wo = nullptr;
    d_w.d_w2 = nullptr;
    d_w.d_w1_w3 = nullptr;
    d_w.d_rmsFinalWeight = nullptr;
    d_w.d_wcls = nullptr;
    
}

GPU_Model::~GPU_Model() {
    freeModel();
}



int compareTokens(const void *a, const void *b) {
    return strcmp(static_cast<const CTokenIndex*>(a)->token, static_cast<const CTokenIndex*>(b)->token);
}

int getTokenIndex(const char* str, CTokenIndex* vocabSortedList, int vocabSize) {
    CTokenIndex tok{ str };
    CTokenIndex* res = static_cast<CTokenIndex*>(bsearch(&tok, vocabSortedList, vocabSize, sizeof(CTokenIndex), compareTokens));
    return res != nullptr ? res->id : -1;
}
/* GPU 模型的CPU函数，无需进行 GPU 加速 */
void GPU_Model::encode(CTokenizer* tokenizer, std::string text, int8_t bos, int8_t eos, int* tokens, int* numTokens) {
    if (text.empty()) {
        std::cerr<<"[ERROR:] Text input is empty and cannot be processed.\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (tokenizer == nullptr || tokens == nullptr || numTokens == nullptr) {
        std::cerr<<"[ERROR:] Invalid input arguments detected. Ensure tokenizer, tokens, and numTokens are properly initialized.\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (tokenizer->vocabSortedList == nullptr) {
        tokenizer->vocabSortedList = static_cast<CTokenIndex*>(malloc(tokenizer->vocabSize * sizeof(CTokenIndex)));
        for (int i = 0; i < tokenizer->vocabSize; i++) {
            tokenizer->vocabSortedList[i].token = tokenizer->vocab[i];
            tokenizer->vocabSortedList[i].id = i;
        }
        qsort(tokenizer->vocabSortedList, tokenizer->vocabSize, sizeof(CTokenIndex), compareTokens);
    }
    
    char* strBuffer = static_cast<char*>(malloc((tokenizer->maxTokenLength * 2 + 1 + 2) * sizeof(char)));
    size_t strLen = 0;

    *numTokens = 0;

    if (bos) tokens[(*numTokens)++] = 1;
    
    if (text[0] != '\0') {
        int dummyPrefix = getTokenIndex(" ", tokenizer->vocabSortedList, tokenizer->vocabSize);
        tokens[(*numTokens)++] = dummyPrefix;
    }

    for (size_t i = 0; i < text.size(); ++i) {
        char currentChar = text[i];
        if ((currentChar & 0xC0) != 0x80) {
            strLen = 0;
        }

        strBuffer[strLen++] = currentChar;
        strBuffer[strLen] = '\0';

        if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && strLen < 4) {
            continue;
        }

        int id = getTokenIndex(strBuffer, tokenizer->vocabSortedList, tokenizer->vocabSize);

        if (id != -1) {
            tokens[(*numTokens)++] = id;
        } else {

            // +3: <unk>, <s>, </s>
            for (size_t i = 0; i < strLen; i++) {
                tokens[(*numTokens)++] = static_cast<unsigned char>(strBuffer[i]) + 3;
            }
        }
        strLen = 0;
    }

    while (true) {
        float bestScore = -1e10;
        int bestId = -1;
        int bestIdx = -1;

        for (int i = 0; i < (*numTokens - 1); i++) {
            sprintf(strBuffer, "%s%s", tokenizer->vocab[tokens[i]], tokenizer->vocab[tokens[i + 1]]);
            int id = getTokenIndex(strBuffer, tokenizer->vocabSortedList, tokenizer->vocabSize);
            if (id != -1 && tokenizer->vocabScores[id] > bestScore) {
                bestScore = tokenizer->vocabScores[id];
                bestId = id;
                bestIdx = i;
            }
        }

        if (bestIdx == -1) {
            break; 
        }

        tokens[bestIdx] = bestId;
        for (int i = bestIdx + 1; i < (*numTokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*numTokens)--; 
    }

    if (eos) tokens[(*numTokens)++] = 2;

    delete []strBuffer;
}
/* GPU的CPU函数，无需进行 GPU 加速 */
char* GPU_Model::decode(CTokenizer* tokenizer, int previousToken, int token) {
    char* piece = tokenizer->vocab[token];

    if (previousToken == 1 && piece[0] == ' ') {
        piece++;  }
    unsigned char byteVal;
    if (sscanf(piece, "<0x%02hhX>", &byteVal) == 1) {
        piece = reinterpret_cast<char*>(tokenizer->bytePieces) + byteVal * 2;
    }
    return piece;  
}

/* 将权重映射到主机内存 */
void GPU_Model::mapWeightsToMemory(CModelConfig* modelConfig, float* ptr, int sharedWeights){
    const int headSize = modelConfig->dim / modelConfig->numHeads;
    const uint64_t numLayers = modelConfig->numLayers;
    float* currentPtr = ptr; 

    h_w.tokenEmbeddingTable = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;

    h_w.rmsAttWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;

    h_w.wq = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);

    h_w.wk = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);

    h_w.wv = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    
    h_w.wo = currentPtr;
    currentPtr += numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;

    h_w.rmsFfnWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;

    h_w.w1 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    h_w.w2 = currentPtr;
    currentPtr += numLayers * modelConfig->feedForwardDim * modelConfig->dim;
    
    h_w.w3 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    h_w.rmsFinalWeight = currentPtr;
    currentPtr += modelConfig->dim;

    currentPtr += modelConfig->maxSeqLen * headSize / 2; 
    currentPtr += modelConfig->maxSeqLen * headSize / 2; 
    
    h_w.wcls = sharedWeights ? h_w.tokenEmbeddingTable : currentPtr;
}


void GPU_Model::load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize) {
    

    std::ifstream fileStream(checkpointPath, std::ios::binary | std::ios::ate);
    if (!fileStream) {
        std::cerr<<"[ERROR:] Unable to open checkpoint file: " << checkpointPath <<std::endl;
    }


    *totalFileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);


    fileStream.read(reinterpret_cast<char*>(modelConfig), sizeof(CModelConfig));
    if (!fileStream) {
        std::cerr<< "[ERROR:] Unable to read model config"<<std::endl;
    }


    bool sharedWeights = modelConfig->vocabSize > 0;
    modelConfig->vocabSize = std::abs(modelConfig->vocabSize);

    fileStream.close();


    *fileDescriptor = open(checkpointPath.c_str(), O_RDONLY);
    if (*fileDescriptor == -1) {
         std::cerr<< "[ERROR:] Unable to open file descriptor for: " << checkpointPath<<std::endl;
    }


    *data = static_cast<float*>(mmap(nullptr, *totalFileSize, PROT_READ, MAP_PRIVATE, *fileDescriptor, 0));
    if (*data == MAP_FAILED) {
         std::cerr<< "[ERROR:] Unable to memory-map the file: " << checkpointPath<<std::endl;
    }

    constexpr uint64_t configSizeInFloats = sizeof(CModelConfig) / sizeof(float);
    float* weightsPtr = *data + configSizeInFloats;


    mapWeightsToMemory(modelConfig, weightsPtr, sharedWeights);
}


void GPU_Model::initializeModel(const std::string checkpointPath) {
    load(checkpointPath, &config, &fd, &data, &fileSize);
    state.allocateGPUMemory(&config);
    transferWeightsToDevice();
}


void GPU_Model::freeModel() {
    if (data != MAP_FAILED && data != nullptr) {
        munmap(data, fileSize);
        data = nullptr;
    }
    
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
    // 释放GPU内存
    // 释放保存在GPU上的权重
    HIP_CHECK(hipFree(d_w.d_tokenEmbeddingTable));
    HIP_CHECK(hipFree(d_w.d_rmsAttWeight));    
    HIP_CHECK(hipFree(d_w.d_wo));
    HIP_CHECK(hipFree(d_w.d_w1_w3));
    HIP_CHECK(hipFree(d_w.d_w2));
    HIP_CHECK(hipFree(d_w.d_rmsFinalWeight));
    HIP_CHECK(hipFree(d_w.d_wcls));
    HIP_CHECK(hipFree(d_w.d_wqkv));
    // 释放保存在GPU上的运行状态内存
    state.deallocateGPUMemory();
}


float* GPU_Model::forward(int token, int pos, GPU_Backend *backend,float *logits) {
    CModelConfig* config = &this->config;
    GPU_RunState* state = &this->state;
    
    float* inputVec = state->d_currentActivation;
    const int embeddingDim = config->dim;
    const int kvDim = config->dim; // 当numKvHeads == numHeads时，kvDim就等于dim
    const int headSize = embeddingDim / config->numHeads;
    const int ffnHiddenDim = config->feedForwardDim;
    
    // tokenEmbeddingTable 在GPU上，所以tokenEmbedding 在GPU上
    float* tokenEmbedding = d_w.d_tokenEmbeddingTable + token * embeddingDim;
    HIP_CHECK(hipMemcpyAsync(inputVec, tokenEmbedding, embeddingDim * sizeof(float), hipMemcpyDeviceToDevice, backend->getStream()));

    for (uint64_t layer = 0; layer < config->numLayers; ++layer) {
        backend->rmsnorm(state->d_branchActivation, inputVec, d_w.d_rmsAttWeight + layer * embeddingDim, embeddingDim, backend->getStream());

        const int kvCacheOffset = layer * config->maxSeqLen * kvDim;
        state->d_k = state->d_keyCache + kvCacheOffset + pos * kvDim;
        state->d_v = state->d_valueCache + kvCacheOffset + pos * kvDim;
        
        // 计算QKV权重矩阵在设备内存中的偏移
        size_t layerOffset = layer * 3 * embeddingDim * embeddingDim;
        
        // 使用一次矩阵乘法同时计算Q、K、V投影
        // 对输入向量乘以堆叠的QKV权重矩阵，得到堆叠的QKV结果
        backend->matmul(state->d_qkv, state->d_branchActivation, d_w.d_wqkv + layerOffset, embeddingDim, embeddingDim * 3, backend->getStream());
        
        // 从计算结果中分别提取Q、K、V
        HIP_CHECK(hipMemcpyAsync(state->d_q, state->d_qkv, config->dim * sizeof(float), hipMemcpyDeviceToDevice, backend->getStream()));
        HIP_CHECK(hipMemcpyAsync(state->d_k, state->d_qkv + config->dim, config->dim * sizeof(float), hipMemcpyDeviceToDevice, backend->getStream()));
        HIP_CHECK(hipMemcpyAsync(state->d_v, state->d_qkv + 2 * config->dim, config->dim * sizeof(float), hipMemcpyDeviceToDevice, backend->getStream()));
        
        backend->ropeEncoding(state->d_q, state->d_k, headSize, pos, embeddingDim, kvDim, backend->getStream());

        
        // 自注意力机制后同步，确保Q、K已准备好
        // 
        
        // 使用flash attention实现
        backend->flash_attention_gpu_step(
            state->d_q,                    // 查询向量
            state->d_keyCache + kvCacheOffset,  // 键缓存
            state->d_valueCache + kvCacheOffset, // 值缓存
            state->d_branchActivation,     // 输出
            state->d_scores,               // 分数
            state->d_attn,                 // 注意力权重
            pos + 1,                       // 序列长度
            backend->getStream()           // 流
        );

        // 使用融合算子替换单独的matmul和axpy操作
        backend->matmul_axpy(
            inputVec,                                         // 输出向量(同时是axpy的目标)
            state->d_branchActivation,                        // 输入向量
            d_w.d_wo + layer * embeddingDim * embeddingDim,  // 权重矩阵
            nullptr,                                          // 不使用偏置
            1.0f,                                             // axpy的缩放因子
            embeddingDim,                                     // 输入维度
            embeddingDim,                                     // 输出维度
            backend->getStream()                              // 流
        );
        
        backend->rmsnorm(state->d_branchActivation, inputVec, d_w.d_rmsFfnWeight + layer * embeddingDim, embeddingDim, backend->getStream());
        
        // backend->matmul(state->d_hiddenBuffer, state->d_branchActivation, d_w.d_w1 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim, backend->getStream());
        // backend->matmul(state->d_extraHiddenBuffer, state->d_branchActivation, d_w.d_w3 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim, backend->getStream());
        // 使用一次矩阵乘法，同时计算两个矩阵
        backend->matmul(state->d_hiddenBuffer_extraHiddenBuffer, state->d_branchActivation, d_w.d_w1_w3 + layer * 2 * embeddingDim * ffnHiddenDim, embeddingDim, 2 * ffnHiddenDim, backend->getStream());

        // 使用后端的swiGLLUFunc实现
        backend->swiGLLUFunc(state->d_hiddenBuffer, state->d_extraHiddenBuffer, ffnHiddenDim, backend->getStream());
        
        // 使用融合算子替换单独的matmul和axpy操作
        backend->matmul_axpy(
            inputVec,                                         // 输出向量(同时是axpy的目标)
            state->d_hiddenBuffer,                            // 输入向量
            d_w.d_w2 + layer * ffnHiddenDim * embeddingDim,  // 权重矩阵
            nullptr,                                          // 不使用偏置
            1.0f,                                             // axpy的缩放因子
            ffnHiddenDim,                                     // 输入维度
            embeddingDim,                                     // 输出维度
            backend->getStream()                              // 流
        );
        
        // 原先的代码被注释掉：
        // backend->matmul(state->d_branchActivation, state->d_hiddenBuffer, d_w.d_w2 + layer * ffnHiddenDim * embeddingDim, ffnHiddenDim, embeddingDim, backend->getStream());
        // backend->axpy(inputVec, state->d_branchActivation, 1.f, embeddingDim, backend->getStream());
    }

    backend->rmsnorm(inputVec, inputVec, d_w.d_rmsFinalWeight, embeddingDim, backend->getStream());
    backend->matmul(state->d_logits, inputVec, d_w.d_tokenEmbeddingTable, embeddingDim, config->vocabSize, backend->getStream());

    HIP_CHECK(hipMemcpy(logits, state->d_logits, config->vocabSize * sizeof(float), hipMemcpyDeviceToHost));

    return logits;
}

void GPU_Model::transferWeightsToDevice() {
    const int headSize = config.dim / config.numHeads;
    const uint64_t numLayers = config.numLayers;

    // 为GPU分配内存
    HIP_CHECK(hipMalloc(&d_w.d_tokenEmbeddingTable, config.vocabSize * config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_rmsAttWeight, numLayers * config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_rmsFfnWeight, numLayers * config.dim * sizeof(float)));
    // 为QKV权重分配一个大矩阵，垂直堆叠Q、K、V
    HIP_CHECK(hipMalloc(&d_w.d_wqkv, numLayers * 3 * config.dim * config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_wo, numLayers * config.dim * config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_w2, numLayers * config.feedForwardDim * config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_rmsFinalWeight, config.dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_wcls, config.vocabSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w.d_w1_w3, numLayers * 2 *config.dim * config.feedForwardDim * sizeof(float)));
    // 将CPU权重复制到GPU
    HIP_CHECK(hipMemcpy(d_w.d_tokenEmbeddingTable, h_w.tokenEmbeddingTable, config.vocabSize * config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_rmsAttWeight, h_w.rmsAttWeight, numLayers * config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_rmsFfnWeight, h_w.rmsFfnWeight, numLayers * config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_wo, h_w.wo, numLayers * config.dim * config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_w2, h_w.w2, numLayers * config.feedForwardDim * config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_rmsFinalWeight, h_w.rmsFinalWeight, config.dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.d_wcls, h_w.wcls, config.vocabSize * sizeof(float), hipMemcpyHostToDevice));
    // 将QKV权重矩阵，按照【层，q,k,v】的顺序，垂直堆叠复制到设备
    float* h_wq_ptr = nullptr, *h_wk_ptr = nullptr, *h_wv_ptr = nullptr;
    const size_t matrixSize = config.dim * config.dim; // 每个矩阵的大小
    
    for (uint64_t layer = 0; layer < numLayers; ++layer) {
        // 计算本层在QKV矩阵中的偏移位置
        size_t layerOffset = layer * 3 * matrixSize;
        
        // 获取Q、K、V在CPU上的地址
        h_wq_ptr = h_w.wq + layer * matrixSize;
        h_wk_ptr = h_w.wk + layer * matrixSize;
        h_wv_ptr = h_w.wv + layer * matrixSize;
        
        // 依次复制Q、K、V到堆叠的QKV矩阵
        HIP_CHECK(hipMemcpy(d_w.d_wqkv + layerOffset, h_wq_ptr, matrixSize * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w.d_wqkv + layerOffset + matrixSize, h_wk_ptr, matrixSize * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w.d_wqkv + layerOffset + 2 * matrixSize, h_wv_ptr, matrixSize * sizeof(float), hipMemcpyHostToDevice));
    }
    const size_t w1_w3_size = config.dim * config.feedForwardDim;
    for (uint64_t layer = 0; layer < numLayers; ++layer) {
        // 计算本层在w1_w3矩阵中的偏移位置
        size_t layerOffset = layer * 2 * w1_w3_size;
        
        // 获取w1、w3在CPU上的地址
        float* h_w1_ptr = h_w.w1 + layer * w1_w3_size;
        float* h_w3_ptr = h_w.w3 + layer * w1_w3_size;
        
        // 依次复制w1、w3到堆叠的w1_w3矩阵
        HIP_CHECK(hipMemcpy(d_w.d_w1_w3 + layerOffset, h_w1_ptr, w1_w3_size * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w.d_w1_w3 + layerOffset + w1_w3_size, h_w3_ptr, w1_w3_size * sizeof(float), hipMemcpyHostToDevice));
    }

}

