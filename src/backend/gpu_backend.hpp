#ifndef GPU_BACKEND_HPP
#define GPU_BACKEND_HPP

#include <hipblas.h>

class GPU_Backend
{
private:
    hipblasHandle_t blas_handle; // hipBLAS handle
    hipStream_t stream;          // HIP流，用于管理内核执行顺序

public:
    static const int BLOCK_SIZE = 256; // CUDA块大小
    static const int NUM_HEADS = 12;   // 注意力头数
    static const int HEAD_SIZE = 64;   // 每个注意力头的大小

    GPU_Backend();
    ~GPU_Backend();

    // 获取当前流
    hipStream_t getStream() { return stream; }

    // GPU后端方法
    void matmul(float *o, const float *x, const float *w, int n, int d, hipStream_t stream = nullptr);

    void ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim, hipStream_t stream = nullptr);
    void swiGLLUFunc(float *hb, float *hb2, int hiddenDim, hipStream_t stream = nullptr);
    void flash_attention(float *q, float *k_cache, float *v_cache, float *output, float *scores, float *attn, int seq_len, hipStream_t stream = nullptr);
    // 融合算子：执行矩阵乘法后立即执行axpy操作
    void matmul_axpy(float *out, const float *x, const float *w, float factor, int n, int d, hipStream_t stream = nullptr);

    // 融合的QKV分离：直接从QKV结果中提取Q、K、V，避免内存拷贝
    void extract_qkv(const float *qkv_result, float *q, float *k, float *v, int dim, hipStream_t stream = nullptr);

    // 优化的RMSNorm：使用更好的内存访问模式和warp-level归约
    void rmsnorm(float *o, const float *x, const float *weight, int size, hipStream_t stream = nullptr);
};

#endif // GPU_BACKEND_HPP
