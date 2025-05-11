 rmsFfnWeight之后 | numLayers × dim × feedForwardDim |
| w2 | 前馈网络第二个投影矩阵 | w1之后 | numLayers × feedForwardDim × dim |
| w3 | 前馈网络第三个投影矩阵(Swi# 模型权重内存映射

## 模型文件结构

模型文件(`stories110M.bin`)的结构如下：

1. 文件开头：`CModelConfig`结构体（模型配置信息）
2. 后续内容：按特定顺序排列的模型权重数据

## 权重偏移量映射表

```cpp
struct weights {
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
    } w; // 模型权重信息
```

`mapWeightsToMemory`函数将模型文件中的权重数据映射到内存中的特定位置。以下是各权重在内存中的偏移计算逻辑：

| 权重名称 | 描述 | 偏移计算 | 大小 (float数量) |
|---------|------|---------|-----------------|
| tokenEmbeddingTable | 词嵌入表 | 起始位置 | vocabSize × dim |
| rmsAttWeight | 注意力层前的RMS归一化权重 | 词嵌入表之后 | numLayers × dim |
| wq | 查询(Q)投影矩阵 | rmsAttWeight之后 | numLayers × dim × (numHeads × headSize) |
| wk | 键(K)投影矩阵 | wq之后 | numLayers × dim × (numKvHeads × headSize) |
| wv | 值(V)投影矩阵 | wk之后 | numLayers × dim × (numKvHeads × headSize) |
| wo | 输出投影矩阵 | wv之后 | numLayers × (numHeads × headSize) × dim |
| rmsFfnWeight | 前馈网络层前的RMS归一化权重 | wo之后 | numLayers × dim |
| w1 | 前馈网络第一个投影矩阵 |GLU激活) | w2之后 | numLayers × dim × feedForwardDim |
| rmsFinalWeight | 最终输出前的RMS归一化权重 | w3之后 | dim |
| 位置编码 | 旋转位置编码(RoPE) | rmsFinalWeight之后 | maxSeqLen × headSize |
| wcls | 输出层权重（词汇表投影） | 如果sharedWeights为true，与tokenEmbeddingTable共享；否则位于位置编码之后 | 共享时不占额外空间 |

## 关键参数说明

- `headSize = dim / numHeads`: 每个注意力头的维度
- `sharedWeights`: 决定输出层权重是否与词嵌入表共享
- `numKvHeads`: 键值头数量，用于分组查询注意力(GQA)或多查询注意力(MQA)

## stories110M模型参数

从读取的配置信息可知，`stories110M.bin`的模型参数为：

- dim: 768
- feedForwardDim: 2048
- numLayers: 12
- numHeads: 12
- numKvHeads: 12
- vocabSize: 32000
- maxSeqLen: 1024
- sharedWeights: 是

这是一个类似于小型GPT-2的语言模型，使用了参数共享以减少模型大小。 