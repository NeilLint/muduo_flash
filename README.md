# muduo

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