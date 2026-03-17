# muduo

## abstract
这是一个在海光DCU上运行的大语言模型推理引擎，基于ROCM/HIP。模型可以加速stories110M推理。该模型可以加速大语言模型推理，实现较高的GPU 利用率，在海光K100GPGPU 上该推理引擎可以达到 510tokens/s的推理速度。

但请注意该模型目前仅仅支持fp32推理，而且实现的多头注意力机制kernel目前还不能称得上flash attention。

## 编译：  
```bash  
make clean  
make  
```

## 运行
1. Usage
```bash
./muduo 模型路径 分词器路径 提示词/提示词文件路径
```

2. 示例 
```bash
./muduo data/stories110M.bin data/tokenizer.bin "once upon a time," > output.txt 2>&1
./muduo data/stories110M.bin data/tokenizer.bin data/input_prompt.txt > output.txt 2>&1
```

3. 任务提交式运行：最终的评分根据此方式运行输出的吞吐量为准
```bash
sbatch muduo.sh
```

# 
