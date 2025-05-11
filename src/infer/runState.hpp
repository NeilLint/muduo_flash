#ifndef RUNSTATE_HPP
#define RUNSTATE_HPP

#include "../model/modelConfig.hpp"
#include "../util.hpp"

class CRunState {
public:
    CRunState();
    ~CRunState();
    // CPU内存指针
    float* currentActivation;  // 当前时间步的激活值(dim)
    float* branchActivation;   // 残差分支内部当前时间步的激活值(dim)
    float* extraBuffer;        // 额外的缓冲区(dim)
    float* hiddenBuffer;       // FFN 中隐藏维度的缓冲区(dim)
    float* extraHiddenBuffer;  // FFN 中额外隐藏维度的缓冲区(dim)

    //注意力机制
    float* q;              // (dim)
    float* k;              // (dim)
    float* v;              // (dim)
    float* attentionScores;    // (numHeads, seqLen)
    float* logits;             

    //KVCache
    float* keyCache;           // KCache (layer, seqLen, dim)
    float* valueCache;         // VCache (layer, seqLen, dim)

    void allocateMemory(CModelConfig* config);
    void deallocateMemory();
};

#endif 