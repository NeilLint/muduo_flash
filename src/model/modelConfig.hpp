#ifndef CMODELCONFIG_HPP
#define CMODELCONFIG_HPP

class CModelConfig
{
public:
    enum class AttentionPattern
    {
        MHA,
        GQA
    };

    enum class AttentionKernel
    {
        CLASSIC,
        FLASH
    };

    int dim;            // Transformer 维度
    int feedForwardDim; // FFN 维度
    int numLayers;      // Transformer层数
    int numHeads;       // query头数
    int numKvHeads;     // 键/值头数
    int vocabSize;      // 词汇表大小
    int maxSeqLen;      // 最大序列长度

    AttentionPattern attentionPattern;
    AttentionKernel attentionKernel;

    int headSize() const { return dim / numHeads; }
    int kvDim() const { return headSize() * numKvHeads; }

    CModelConfig();
    CModelConfig(int dim, int feedForwardDim, int numLayers, int numHeads,
                 int numKvHeads, int vocabSize, int maxSeqLen);
    ~CModelConfig();
};

#endif
