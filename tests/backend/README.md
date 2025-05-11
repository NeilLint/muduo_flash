# GPU vs CPU 后端性能对比测试

本目录包含了一系列测试文件，用于对比CPU和GPU实现的各种计算函数在性能和正确性上的差异。

## 测试内容

共包含以下测试：

1. **Softmax测试** (`test_softmax.cpp`): 测试softmax函数在不同向量大小下的性能对比。
2. **矩阵-向量乘法测试** (`test_matmul.cpp`): 测试矩阵-向量乘法在不同矩阵大小下的性能对比。
3. **RMSNorm测试** (`test_rmsnorm.cpp`): 测试RMS归一化操作在不同向量大小下的性能对比。
4. **AXPY测试** (`test_axpy.cpp`): 测试AXPY操作(`y += factor * x`)在不同向量大小和不同因子下的性能对比。
5. **点积测试** (`test_dot.cpp`): 测试向量点积操作在不同向量大小下的性能对比。
6. **GemvQkSeq测试** (`test_gemvQkSeq.cpp`): 测试GPU函数`gemvQkSeq`(注意力机制中的Q-K操作)的性能和正确性。

## 编译和运行

### 编译所有测试

```bash
make
```

### 运行所有测试

```bash
make run
```

### 单独运行特定测试

```bash
# 运行Softmax测试
make run_softmax

# 运行矩阵-向量乘法测试
make run_matmul

# 运行RMSNorm测试
make run_rmsnorm

# 运行AXPY测试
make run_axpy

# 运行点积测试
make run_dot

# 运行GemvQkSeq测试
make run_gemvQkSeq
```

### 清理编译产物

```bash
make clean
```

## 测试输出说明

每个测试程序会输出以下信息：

1. 测试向量/矩阵大小
2. CPU和GPU实现的执行时间
3. GPU相对CPU的加速比
4. 结果是否匹配
5. 部分输入和输出数据的内容

## 依赖项

- ROCM/HIP
- hipBLAS
- C++11或更高版本的编译器

## 注意事项

1. 这些测试假设系统上已正确安装ROCM和hipBLAS库。
2. 部分测试在大规模数据上可能运行时间较长。
3. 小规模数据可能因为调用开销，使得GPU实现没有明显优势，甚至比CPU慢。 