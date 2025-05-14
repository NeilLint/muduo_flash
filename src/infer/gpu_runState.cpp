#include "gpu_runState.hpp"
#include <iostream>


GPU_RunState::GPU_RunState() 
    : d_currentActivation(nullptr), d_branchActivation(nullptr), d_extraBuffer(nullptr),
      d_hiddenBuffer(nullptr), d_extraHiddenBuffer(nullptr), d_q(nullptr), d_k(nullptr),
      d_v(nullptr), d_attentionScores(nullptr), d_logits(nullptr),
      d_keyCache(nullptr), d_valueCache(nullptr), d_scores(nullptr), d_attn(nullptr) {
    
}

GPU_RunState::~GPU_RunState() {
    // 确保所有GPU内存都已释放
    deallocateGPUMemory();
}

void GPU_RunState::allocateGPUMemory(CModelConfig* config) {
    // 先释放可能已分配的内存
    deallocateGPUMemory();
    // printf("AllocatingGPUMemory\n");
    int kvDim = (config->dim * config->numKvHeads) / config->numKvHeads;
    // printf("config->dim = %d\n",config->dim);
    // printf("config->feedForwardDim = %d\n",config->feedForwardDim);
    // printf("config->numLayers = %d\n",config->numLayers);
    // printf("config->heads = %d\n",config->numHeads);
    // printf("Config->maxSeqLen = %d\n",config->maxSeqLen);
    // printf("Config->numKVHeads = %d\n",config->numKvHeads);
    // 分配设备内存
    HIP_CHECK(hipMalloc((void**)&d_currentActivation, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_branchActivation, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_extraBuffer, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_hiddenBuffer, config->feedForwardDim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_extraHiddenBuffer, config->feedForwardDim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_q, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_k, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_v, config->dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_keyCache, config->numLayers * config->maxSeqLen * kvDim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_valueCache, config->numLayers * config->maxSeqLen * kvDim * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_attentionScores, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_logits, config->vocabSize * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_scores, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_attn, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_qkv, 3 * config->dim * sizeof(float)));

    // 初始化GPU内存为0
    HIP_CHECK(hipMemset(d_currentActivation, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_branchActivation, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_extraBuffer, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_hiddenBuffer, 0, config->feedForwardDim * sizeof(float)));
    HIP_CHECK(hipMemset(d_extraHiddenBuffer, 0, config->feedForwardDim * sizeof(float)));
    HIP_CHECK(hipMemset(d_q, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_k, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_v, 0, config->dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_keyCache, 0, config->numLayers * config->maxSeqLen * kvDim * sizeof(float)));
    HIP_CHECK(hipMemset(d_valueCache, 0, config->numLayers * config->maxSeqLen * kvDim * sizeof(float)));
    HIP_CHECK(hipMemset(d_attentionScores, 0, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMemset(d_logits, 0, config->vocabSize * sizeof(float)));
    HIP_CHECK(hipMemset(d_scores, 0, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMemset(d_attn, 0, config->numHeads * config->maxSeqLen * sizeof(float)));
    HIP_CHECK(hipMemset(d_qkv, 0, 3 * config->dim * sizeof(float)));
    // std::cout << "[INFO:] GPU memory allocation successful!" << std::endl;
}

void GPU_RunState::deallocateGPUMemory() {
    if (d_currentActivation) { HIP_CHECK(hipFree(d_currentActivation)); d_currentActivation = nullptr; }
    if (d_branchActivation) { HIP_CHECK(hipFree(d_branchActivation)); d_branchActivation = nullptr; }
    if (d_extraBuffer) { HIP_CHECK(hipFree(d_extraBuffer)); d_extraBuffer = nullptr; }
    if (d_hiddenBuffer) { HIP_CHECK(hipFree(d_hiddenBuffer)); d_hiddenBuffer = nullptr; }
    if (d_extraHiddenBuffer) { HIP_CHECK(hipFree(d_extraHiddenBuffer)); d_extraHiddenBuffer = nullptr; }
    if (d_q) { HIP_CHECK(hipFree(d_q)); d_q = nullptr; }
    if (d_k) { HIP_CHECK(hipFree(d_k)); d_k = nullptr; }
    if (d_v) { HIP_CHECK(hipFree(d_v)); d_v = nullptr; }
    if (d_attentionScores) { HIP_CHECK(hipFree(d_attentionScores)); d_attentionScores = nullptr; }
    if (d_logits) { HIP_CHECK(hipFree(d_logits)); d_logits = nullptr; }
    if (d_keyCache) { hipError_t error = hipFree(d_keyCache); d_keyCache = nullptr; }
    if (d_valueCache) { hipError_t error = hipFree(d_valueCache); d_valueCache = nullptr; }
    if (d_scores) { HIP_CHECK(hipFree(d_scores)); d_scores = nullptr; }
    if (d_attn) { HIP_CHECK(hipFree(d_attn)); d_attn = nullptr; }
    if (d_qkv) { HIP_CHECK(hipFree(d_qkv)); d_qkv = nullptr; }
    // std::cout << "[INFO:] GPU memory deallocation successful!" << std::endl;
}

void GPU_RunState::copyToGPU(CRunState* cpuState, CModelConfig* config) {
    if (!cpuState) {
        std::cerr << "[ERROR:] CPU RunState is null when copying to GPU!" << std::endl;
        return;
    }
    
    int kvDim = (config->dim * config->numKvHeads) / config->numKvHeads;
    
    // 将CPU内存数据复制到GPU
    if (cpuState->currentActivation && d_currentActivation) {
        HIP_CHECK(hipMemcpy(d_currentActivation, cpuState->currentActivation, 
                            config->dim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->branchActivation && d_branchActivation) {
        HIP_CHECK(hipMemcpy(d_branchActivation, cpuState->branchActivation, 
                            config->dim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->extraBuffer && d_extraBuffer) {
        HIP_CHECK(hipMemcpy(d_extraBuffer, cpuState->extraBuffer, 
                            config->dim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->hiddenBuffer && d_hiddenBuffer) {
        HIP_CHECK(hipMemcpy(d_hiddenBuffer, cpuState->hiddenBuffer, 
                            config->feedForwardDim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->extraHiddenBuffer && d_extraHiddenBuffer) {
        HIP_CHECK(hipMemcpy(d_extraHiddenBuffer, cpuState->extraHiddenBuffer, 
                            config->feedForwardDim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->q && d_q) {
        HIP_CHECK(hipMemcpy(d_q, cpuState->q, 
                            config->dim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->keyCache && d_keyCache) {
        HIP_CHECK(hipMemcpy(d_keyCache, cpuState->keyCache, 
                            config->numLayers * config->maxSeqLen * kvDim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->valueCache && d_valueCache) {
        HIP_CHECK(hipMemcpy(d_valueCache, cpuState->valueCache, 
                            config->numLayers * config->maxSeqLen * kvDim * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->attentionScores && d_attentionScores) {
        HIP_CHECK(hipMemcpy(d_attentionScores, cpuState->attentionScores, 
                            config->numHeads * config->maxSeqLen * sizeof(float), hipMemcpyHostToDevice));
    }
    
    if (cpuState->logits && d_logits) {
        HIP_CHECK(hipMemcpy(d_logits, cpuState->logits, 
                            config->vocabSize * sizeof(float), hipMemcpyHostToDevice));
    }
}

void GPU_RunState::copyFromGPU(CRunState* cpuState, CModelConfig* config) {
    if (!cpuState) {
        std::cerr << "[ERROR:] CPU RunState is null when copying from GPU!" << std::endl;
        return;
    }
    
    int kvDim = (config->dim * config->numKvHeads) / config->numKvHeads;
    
    // 将GPU数据复制回CPU内存
    if (cpuState->currentActivation && d_currentActivation) {
        HIP_CHECK(hipMemcpy(cpuState->currentActivation, d_currentActivation, 
                            config->dim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->branchActivation && d_branchActivation) {
        HIP_CHECK(hipMemcpy(cpuState->branchActivation, d_branchActivation, 
                            config->dim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->extraBuffer && d_extraBuffer) {
        HIP_CHECK(hipMemcpy(cpuState->extraBuffer, d_extraBuffer, 
                            config->dim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->hiddenBuffer && d_hiddenBuffer) {
        HIP_CHECK(hipMemcpy(cpuState->hiddenBuffer, d_hiddenBuffer, 
                            config->feedForwardDim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->extraHiddenBuffer && d_extraHiddenBuffer) {
        HIP_CHECK(hipMemcpy(cpuState->extraHiddenBuffer, d_extraHiddenBuffer, 
                            config->feedForwardDim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->q && d_q) {
        HIP_CHECK(hipMemcpy(cpuState->q, d_q, 
                            config->dim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->keyCache && d_keyCache) {
        HIP_CHECK(hipMemcpy(cpuState->keyCache, d_keyCache, 
                            config->numLayers * config->maxSeqLen * kvDim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->valueCache && d_valueCache) {
        HIP_CHECK(hipMemcpy(cpuState->valueCache, d_valueCache, 
                            config->numLayers * config->maxSeqLen * kvDim * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->attentionScores && d_attentionScores) {
        HIP_CHECK(hipMemcpy(cpuState->attentionScores, d_attentionScores, 
                            config->numHeads * config->maxSeqLen * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    if (cpuState->logits && d_logits) {
        HIP_CHECK(hipMemcpy(cpuState->logits, d_logits, 
                            config->vocabSize * sizeof(float), hipMemcpyDeviceToHost));
    }
}
