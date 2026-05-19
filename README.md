# muduo

## abstract
这是一个在海光DCU上运行的大语言模型推理引擎，基于ROCM/HIP。当前工程为“模型结构解耦”的推理框架：

- 通过 `GPU_Infer + GPU_Model` 的抽象分层，解耦“模型加载/推理调度”和“具体注意力实现”；
- 注意力机制支持 **MHA（多头注意力）** 与 **GQA（分组注意力）**，依据 `numHeads` 与 `numKvHeads` 自动判定；
- 后端提供 **Flash Attention** 执行路径，并支持通过参数切换注意力内核策略（`flash`/`classic`，两者均支持 MHA/GQA 索引布局）。

但请注意该模型目前仅仅支持fp32推理。

## 编译：  
```bash  
make clean  
make  
```

## 运行
1. Usage
```bash
./muduo 模型路径 分词器路径 提示词/提示词文件路径 [modelType] [attentionKernel]
```

2. 示例 
```bash
./muduo data/stories110M.bin data/tokenizer.bin "once upon a time," llama flash > output.txt 2>&1
./muduo data/stories110M.bin data/tokenizer.bin data/input_prompt.txt > output.txt 2>&1
```

3. 任务提交式运行：最终的评分根据此方式运行输出的吞吐量为准
```bash
sbatch muduo.sh
```

# 
