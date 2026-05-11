#include "modelConfig.hpp"
CModelConfig::CModelConfig()
{
    this->attentionPattern = AttentionPattern::MHA;
    this->attentionKernel = AttentionKernel::FLASH;
}

CModelConfig::CModelConfig(int dim, int feedForwardDim, int numLayers, int numHeads,
                           int numKvHeads, int vocabSize, int maxSeqLen)
{
    this->dim = dim;
    this->feedForwardDim = feedForwardDim;
    this->numLayers = numLayers;
    this->numHeads = numHeads;
    this->numKvHeads = numKvHeads;
    this->vocabSize = vocabSize;
    this->maxSeqLen = maxSeqLen;
    this->attentionPattern = (numKvHeads == numHeads) ? AttentionPattern::MHA : AttentionPattern::GQA;
    this->attentionKernel = AttentionKernel::FLASH;
}

CModelConfig::~CModelConfig()
{
}
