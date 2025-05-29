#ifndef GPU_BACKEND_HPP
#define GPU_BACKEND_HPP

#include <hipblas.h>

class GPU_Backend {
private:
    hipblasHandle_t blas_handle;     // hipBLAS handle
    hipStream_t stream;         // HIP流，用于管理内核执行顺序

public:
    static const int NUM_STREAMS = 4;  // 固定使用4个流
    static const int BLOCK_SIZE = 256;  // CUDA块大小
    static const int NUM_HEADS = 12;    // 注意力头数
    static const int HEAD_SIZE = 64;    // 每个注意力头的大小
    static const int MAX_SEQ_LEN = 1024; // 最大序列长度
    
    GPU_Backend();
    ~GPU_Backend();

    // 获取当前流
    hipStream_t getStream() { return stream; }
    // 获取BLAS句柄
    hipblasHandle_t getBLASHandle() { return blas_handle; }
    // 等待所有操作完成
    void synchronize();

    // GPU后端方法
    void matmul(float* o, const float* x, const float* w, int n, int d, hipStream_t stream = nullptr);
    void rmsnorm(float* o, const float* x, const float* weight, int size, hipStream_t stream = nullptr);
    void axpy(float *y, const float *x, float factor, int dim, hipStream_t stream = nullptr);
    void dot(float *y, const float *x1, const float *x2, int dim, hipStream_t stream = nullptr);
    void ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim, hipStream_t stream = nullptr);
    void swiGLLUFunc(float *hb, float *hb2, int hiddenDim, hipStream_t stream = nullptr);
    void flash_attention_gpu_step(float* q, float* k_cache, float* v_cache, float* output, float* scores, float* attn, int seq_len, hipStream_t stream = nullptr);
    // 融合版本的Flash Attention：减少内存访问
    void flash_attention_fused(float* q, float* k_cache, float* v_cache, float* output, int seq_len, hipStream_t stream = nullptr);
    // 融合算子：执行矩阵乘法后立即执行axpy操作
    void matmul_axpy(float* out, const float* x, const float* w, float factor, int n, int d, hipStream_t stream = nullptr);
    
    // 融合的QKV分离：直接从QKV结果中提取Q、K、V，避免内存拷贝
    void extract_qkv(const float* qkv_result, float* q, float* k, float* v, int dim, hipStream_t stream = nullptr);
    
    // 优化的RMSNorm：使用更好的内存访问模式和warp-level归约
    void rmsnorm_optimized(float* o, const float* x, const float* weight, int size, hipStream_t stream = nullptr);
    
    // 优化的logits计算：只计算部分logits以减少计算量（可选优化）
    void matmul_partial_logits(float* logits, const float* x, const float* w, int input_dim, int vocab_size, int top_k = 1000, hipStream_t stream = nullptr);
    
    // 智能采样优化：基于上下文预测最可能的token集合
    void smart_sampling_logits(float* logits, const float* x, const float* w, const int* context_tokens, int context_len, int input_dim, int vocab_size, hipStream_t stream = nullptr);
    
    // 自适应Top-K：根据采样参数自动调整K值
    void adaptive_logits(float* logits, const float* x, const float* w, int input_dim, int vocab_size, float temperature, float top_p, hipStream_t stream = nullptr);
    
    // 安全的采样优化：在完整logits基础上进行Top-K筛选
    void optimize_logits_for_sampling(float* logits, int vocab_size, int top_k, hipStream_t stream = nullptr);
};

#endif // GPU_BACKEND_HPP




