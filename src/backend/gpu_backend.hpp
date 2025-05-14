#ifndef GPU_BACKEND_HPP
#define GPU_BACKEND_HPP

#include <hipblas.h>

// Kernel函数声明
__global__ void optimized_flash_qk_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    float* __restrict__ scores,
    float* __restrict__ attn,
    int seq_len,
    int head_size,
    int num_heads
);

__global__ void optimized_flash_output_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ v_cache,
    float* __restrict__ output,
    int seq_len,
    int head_size,
    int num_heads
);

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
    
    // 修改后的QKV投影批处理方法，支持使用预分配的设备指针数组
    void qkvProjectionBatched(
        float* q, float* k, float* v,
        const float* x,
        const float* wq, const float* wk, const float* wv,
        float** d_A_array,float ** d_B_array,float** d_C_array,
        int embeddingDim, int kvDim,
        int layer,
        hipStream_t stream
    );
};

#endif // GPU_BACKEND_HPP




