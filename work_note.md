# 开始新分支的开发
新分支期望以更高效的方式重排QKV矩阵，使用这种方式，我们可以使用一次矩阵乘法解决kqv projection的问题。

# 算子合并（将两个matmul合并成一个）
        backend->matmul(state->d_hiddenBuffer, state->d_branchActivation, d_w.d_w1 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim, backend->getStream());
        backend->matmul(state->d_extraHiddenBuffer, state->d_branchActivation, d_w.d_w3 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim, backend->getStream());

需要将w1和w3合并成一个矩阵，然后使用一个矩阵乘法完成kqv projection。
将d_w.d_w1和d_w.d_w3合并成一个矩阵，然后使用一个矩阵乘法完成kqv projection。
然后再把state->d_hiddenBuffer和state->d_extraHiddenBuffer合并成一个向量。
[d_hiddenBuffer, d_extraHiddenBuffer] 
