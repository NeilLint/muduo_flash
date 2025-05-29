#include "gpu_backend.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>
#include <cmath>
#include <cfloat>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// 用于HIP错误检查的辅助宏
#define HIP_CHECK(command)                                                  \
    {                                                                       \
        hipError_t status = command;                                        \
        if (status != hipSuccess)                                           \
        {                                                                   \
            fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n",                \
                    hipGetErrorString(status), status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#define HIPBLAS_CHECK(command)                                             \
    {                                                                      \
        hipblasStatus_t status = command;                                  \
        if (status != HIPBLAS_STATUS_SUCCESS)                              \
        {                                                                  \
            fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n",         \
                    status, __FILE__, __LINE__);                           \
            /* You might want a function to map status codes to strings */ \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// 优化的RMSNorm内核 - 使用warp-level归约
__global__ void __launch_bounds__(512) rmsnorm_kernel(
    float *__restrict__ o,
    const float *__restrict__ x,
    const float *__restrict__ weight,
    int size,
    float epsilon)
{
    int tid = threadIdx.x;
    int warp_id = tid / 64;  // 改为64
    int lane_id = tid % 64;  // 改为64

    // 使用共享内存进行warp间归约
    extern __shared__ float s_sum[];

    // 计算线程局部平方和
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        float val = x[i];
        thread_sum_sq += val * val;
    }

    // Warp-level归约
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
    {
        thread_sum_sq += __shfl_down(thread_sum_sq, offset);
    }

    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0)
    {
        s_sum[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    // 最后一个warp处理warp间归约
    if (warp_id == 0)
    {
        float warp_sum = (lane_id < (blockDim.x + 63) / 64) ? s_sum[lane_id] : 0.0f;  // 改为63
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
        {
            warp_sum += __shfl_down(warp_sum, offset);
        }
        if (lane_id == 0)
        {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();

    // 计算RMS归一化
    float ss = s_sum[0];
    float rms = sqrtf(ss / (float)size + epsilon);
    float inv_rms = 1.0f / rms;

    // 应用归一化和权重
    for (int i = tid; i < size; i += blockDim.x)
    {
        o[i] = (x[i] * inv_rms) * weight[i];
    }
}

GPU_Backend::GPU_Backend() : blas_handle(nullptr), stream(nullptr)
{
    // 创建HIP流
    HIP_CHECK(hipStreamCreate(&stream));

    // 创建hipBLAS句柄
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));

    // 将hipBLAS句柄与流关联
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, stream));

    // 设置指针模式为主机
    HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));

    // printf("GPU_Backend initialized with HIP stream and hipBLAS handle.\n");
}

GPU_Backend::~GPU_Backend()
{
    // 销毁hipBLAS句柄
    if (blas_handle)
    {
        HIPBLAS_CHECK(hipblasDestroy(blas_handle));
        blas_handle = nullptr;
    }

    // printf("GPU_Backend destroyed, hipBLAS handle and HIP stream released.\n");
}



// 假设 CHECK 和 HIPBLAS_CHECK 宏已定义

// --- 重构后的 matmul 函数 ---
// 计算 o = w * x (矩阵-向量乘法)
// 假设 o_d, x_d, w_d 是指向 GPU 设备内存的有效指针。
// 假设 w_d 指向的矩阵 W 是按行主元 (Row-Major) 存储的 d x n 矩阵。
// 内存管理和数据传输由调用者负责。
void GPU_Backend::matmul(float *o_d,         // 指向 GPU 上的输出向量 o (d x 1) 的指针
                         const float *x_d,   // 指向 GPU 上的输入向量 x (n x 1) 的指针
                         const float *w_d,   // 指向 GPU 上的输入矩阵 w (d x n, Row-Major) 的指针
                         int n,              // 矩阵 w 的列数 / 向量 x 的行数
                         int d,              // 矩阵 w 的行数 / 向量 o 的行数
                         hipStream_t stream) // 指定的流，可选
{                                            // 指定的流，可选
    // 输入参数检查
    if (!o_d || !x_d || !w_d || n <= 0 || d <= 0)
    {
        fprintf(stderr, "GPU Matmul Error: Invalid input pointers or dimensions.\n");
        return;
    }

    // 使用指定的流或默认流
    hipStream_t useStream = stream ? stream : this->stream;
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 对于矩阵-向量乘法，使用GEMV而不是GEMM以获得更高效率
    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_T, // 转置矩阵，因为我们的矩阵是行主序存储
        n,            // 原始矩阵的行数
        d,            // 原始矩阵的列数
        &alpha,       // 缩放因子
        w_d,          // 矩阵
        n,            // 矩阵的leading dimension
        x_d,          // 输入向量
        1,            // 输入向量的步长
        &beta,        // 累积因子
        o_d,          // 输出向量
        1             // 输出向量的步长
        ));
}

void GPU_Backend::rmsnorm(
    float *o,
    const float *x,
    const float *weight,
    int size,
    hipStream_t stream)
{
    if (size <= 0 || !o || !x || !weight)
    {
        fprintf(stderr, "RMSNorm_Optimized Error: Invalid input parameters.\n");
        return;
    }

    hipStream_t useStream = stream ? stream : this->stream;

    const float epsilon = 1e-5f;
    const int block_size = 256;  // 改回256确保稳定性

    // 使用单个块处理，共享内存大小为warp数量
    dim3 gridDim(1);
    dim3 blockDim(block_size);
    size_t shared_mem_size = ((block_size + 63) / 64) * sizeof(float);

    hipLaunchKernelGGL(
        rmsnorm_kernel,
        gridDim,
        blockDim,
        shared_mem_size,
        useStream,
        o, x, weight, size, epsilon);

    HIP_CHECK(hipGetLastError());
}

__global__ void ropeEncoding_kernel(float *q, float *k,
                                    int headSize,
                                    int position,
                                    int dim,
                                    int kvDim)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = pair_id * 2;
    if (i + 1 >= dim)
        return;

    int headDim = i % headSize;
    float freq = 1.0f / powf(10000.0f, headDim / (float)headSize);
    float angle = position * freq;
    float fcr = cosf(angle);
    float fci = sinf(angle);

    // q 旋转
    float q0 = q[i], q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // k 旋转（i < kvDim 时）
    if (i < kvDim)
    {
        float k0 = k[i], k1 = k[i + 1];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

void GPU_Backend::ropeEncoding(float *q, float *k,
                               int headSize,
                               int position,
                               int dim,
                               int kvDim,
                               hipStream_t stream)
{ // 指定的流，可选
    if (!q || !k || headSize <= 0 || dim <= 0 || kvDim < 0)
        return;

    // 如果未指定流，则使用类的默认流
    hipStream_t useStream = stream ? stream : this->stream;
    int numPairs = (dim + 1) / 2;
    const int threads = 256;
    int blocks = (numPairs + threads - 1) / threads;
    hipLaunchKernelGGL(
        ropeEncoding_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        useStream, // 使用指定的流
        q, k, headSize, position, dim, kvDim);
    HIP_CHECK(hipGetLastError());
    // 不再显式同步，由调用者决定何时同步
}

__global__ void swiGLLU_kernel(float *headOutput, const float *value, int hiddenDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hiddenDim)
    {
        float h = headOutput[idx];
        float sig = 1.0f / (1.0f + expf(-h));
        headOutput[idx] = h * sig * value[idx];
    }
}

void GPU_Backend::swiGLLUFunc(float *hb, float *hb2, int hiddenDim, hipStream_t stream)
{
    if (hiddenDim <= 0 || hb == nullptr || hb2 == nullptr)
        return;
    // 如果未指定流，则使用类的默认流
    hipStream_t useStream = stream ? stream : this->stream;

    // 1. 计算网格与线程数
    const int threads = 256;
    const int blocks = (hiddenDim + threads - 1) / threads;

    // 2. 启动 kernel，使用指定的流
    hipLaunchKernelGGL(
        swiGLLU_kernel,
        dim3(blocks),  // grid
        dim3(threads), // block
        0,             // shared mem
        useStream,     // 使用指定的流
        hb,            // kernel args...
        hb2,
        hiddenDim);
    HIP_CHECK(hipGetLastError());
    // 不再显式同步，由调用者决定何时同步
}

// 优化的QK计算和Softmax内核 - 添加warp-level优化
__global__ void __launch_bounds__(512) optimized_flash_qk_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k_cache,
    float *__restrict__ scores,
    float *__restrict__ attn,
    int seq_len,
    int head_size,
    int num_heads)
{
    // 获取头索引
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 64;  // 改为64
    int lane_id = tid % 64;  // 改为64

    // 共享内存
    extern __shared__ float s_data[];
    float *s_max = s_data;              // 最大值
    float *s_sum = &s_data[blockDim.x]; // 总和

    // 确保只有有效的线程参与计算
    if (head_idx >= num_heads)
        return;

    // 访问当前头的q向量
    const float *q_head = q + head_idx * head_size;

    // 缩放因子
    float scale = 1.0f / sqrtf((float)head_size);

    // 初始化最大值
    float thread_max = -FLT_MAX;

    // 计算 QK 点积 (批处理方式，添加循环展开优化)
    for (int i = tid; i < seq_len; i += blockDim.x)
    {
        float score = 0.0f;

        // 计算点积 - 添加循环展开以提高性能
        const float *k_vec = k_cache + i * num_heads * head_size + head_idx * head_size;

        // 展开循环以减少循环开销，并添加预取
        int d = 0;
        for (; d + 7 < head_size; d += 8)
        {
            // 8路展开以更好地利用内存带宽
            score += q_head[d] * k_vec[d] +
                     q_head[d + 1] * k_vec[d + 1] +
                     q_head[d + 2] * k_vec[d + 2] +
                     q_head[d + 3] * k_vec[d + 3] +
                     q_head[d + 4] * k_vec[d + 4] +
                     q_head[d + 5] * k_vec[d + 5] +
                     q_head[d + 6] * k_vec[d + 6] +
                     q_head[d + 7] * k_vec[d + 7];
        }
        // 处理剩余元素
        for (; d < head_size; d++)
        {
            score += q_head[d] * k_vec[d];
        }

        score *= scale;
        scores[head_idx * seq_len + i] = score;
        thread_max = fmaxf(thread_max, score);
    }

    // Warp-level归约找出最大值（更高效）
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
    {
        thread_max = fmaxf(thread_max, __shfl_down(thread_max, offset));
    }

    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0)
    {
        s_max[warp_id] = thread_max;
    }
    __syncthreads();

    // 最后一个warp处理warp级别的归约
    if (warp_id == 0)
    {
        float warp_max = (lane_id < (int)((blockDim.x + 63) / 64)) ? s_max[lane_id] : -FLT_MAX;  // 改为63和64
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
        {
            warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
        }
        if (lane_id == 0)
        {
            s_max[0] = warp_max;
        }
    }
    __syncthreads();

    float max_val = s_max[0];

    // 计算softmax - 数值稳定版本
    float thread_sum = 0.0f;

    for (int i = tid; i < seq_len; i += blockDim.x)
    {
        float score = scores[head_idx * seq_len + i];
        float exp_val = expf(score - max_val);
        attn[head_idx * seq_len + i] = exp_val;
        thread_sum += exp_val;
    }

    // Warp-level归约计算总和
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
    {
        thread_sum += __shfl_down(thread_sum, offset);
    }

    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0)
    {
        s_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // 最后一个warp处理warp级别的归约
    if (warp_id == 0)
    {
        float warp_sum = (lane_id < (int)((blockDim.x + 63) / 64)) ? s_sum[lane_id] : 0.0f;  // 改为63和64
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)  // 改为32开始
        {
            warp_sum += __shfl_down(warp_sum, offset);
        }
        if (lane_id == 0)
        {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();

    float sum = s_sum[0];

    // 归一化
    for (int i = tid; i < seq_len; i += blockDim.x)
    {
        if (sum > 0.0f)
        {
            attn[head_idx * seq_len + i] /= sum;
        }
        else
        {
            attn[head_idx * seq_len + i] = 1.0f / seq_len;
        }
    }
}

// 优化的输出计算内核 - 添加向量化访问
__global__ void optimized_flash_output_kernel(
    const float *__restrict__ attn,
    const float *__restrict__ v_cache,
    float *__restrict__ output,
    int seq_len,
    int head_size,
    int num_heads)
{
    int head_idx = blockIdx.x;
    int feat_idx = threadIdx.x;

    // 确保有效范围
    if (head_idx >= num_heads || feat_idx >= head_size)
        return;

    // 初始化输出累加器
    float acc = 0.0f;

    // 预计算基础偏移量
    const float *attn_head = attn + head_idx * seq_len;
    const float *v_base = v_cache + head_idx * head_size + feat_idx;

    // 优化的累加循环 - 展开以提高性能
    int i = 0;
    for (; i + 3 < seq_len; i += 4)
    {
        // 4路展开以更好地利用内存带宽和指令级并行
        float a0 = attn_head[i];
        float a1 = attn_head[i + 1];
        float a2 = attn_head[i + 2];
        float a3 = attn_head[i + 3];

        float v0 = v_base[i * num_heads * head_size];
        float v1 = v_base[(i + 1) * num_heads * head_size];
        float v2 = v_base[(i + 2) * num_heads * head_size];
        float v3 = v_base[(i + 3) * num_heads * head_size];

        acc += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
    }

    // 处理剩余元素
    for (; i < seq_len; i++)
    {
        float a = attn_head[i];
        float v = v_base[i * num_heads * head_size];
        acc += a * v;
    }

    // 写入最终结果
    output[head_idx * head_size + feat_idx] = acc;
}

void GPU_Backend::flash_attention_gpu_step(
    float *q,
    float *k_cache,
    float *v_cache,
    float *output,
    float *scores,
    float *attn,
    int seq_len,
    hipStream_t stream)
{
    // 使用指定的流或默认流
    hipStream_t useStream = stream ? stream : this->stream;

    // 优化的Flash Attention实现 - 针对gfx906优化
    const int BLOCK_SIZE = 256; // 改回256确保稳定性
    const int HEAD_SIZE = GPU_Backend::HEAD_SIZE;
    const int NUM_HEADS = GPU_Backend::NUM_HEADS;

    // 第一阶段：QK计算和Softmax
    dim3 grid_qk(NUM_HEADS);
    dim3 block_qk(BLOCK_SIZE);
    size_t shmem_size_qk = 2 * BLOCK_SIZE * sizeof(float); // 存储最大值和总和

    // 第二阶段：输出计算
    dim3 grid_output(NUM_HEADS);
    dim3 block_output(HEAD_SIZE);
    size_t shmem_size_output = ((HEAD_SIZE + 63) / 64) * sizeof(float);

    // 计算QK点积并应用Softmax
    optimized_flash_qk_kernel<<<grid_qk, block_qk, shmem_size_qk, useStream>>>(
        q, k_cache, scores, attn, seq_len, HEAD_SIZE, NUM_HEADS);

    // 确保QK计算完成
    HIP_CHECK(hipGetLastError());

    // 计算输出
    optimized_flash_output_kernel<<<grid_output, block_output, shmem_size_output, useStream>>>(
        attn, v_cache, output, seq_len, HEAD_SIZE, NUM_HEADS);

    // 检查错误但不强制同步
    HIP_CHECK(hipGetLastError());
}

// 融合的matmul_axpy函数实现
void GPU_Backend::matmul_axpy(
    float *out,        // 输出向量，同时是axpy的目标
    const float *x,    // 输入向量
    const float *w,    // 权重矩阵
    float factor,      // axpy的缩放因子
    int n,             // 输入维度
    int d,             // 输出维度
    hipStream_t stream // 可选的CUDA流
)
{
    // 输入验证
    if (!out || !x || !w || d <= 0 || n <= 0)
    {
        fprintf(stderr, "GPU matmul_axpy Error: Invalid input pointers or dimensions.\n");
        return;
    }

    hipStream_t useStream = stream ? stream : this->stream;
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

    // 直接使用hipBLAS库的GEMV函数实现高性能的矩阵乘法+axpy
    // 设置beta=1.0，这样就直接在输出向量上累加结果
    const float alpha = factor;
    const float beta = 1.0f; // 保留原始值并添加新结果

    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_T, // 转置矩阵，因为我们的矩阵是行主序存储
        n,            // 原始矩阵的行数
        d,            // 原始矩阵的列数
        &alpha,       // 缩放因子
        w,            // 矩阵
        n,            // 矩阵的leading dimension
        x,            // 输入向量
        1,            // 输入向量的步长
        &beta,        // 累积因子，1.0表示保留原始输出并添加结果
        out,          // 输出向量
        1             // 输出向量的步长
        ));

    HIP_CHECK(hipGetLastError());
}

// 融合的QKV分离内核 - 避免内存拷贝
__global__ void extract_qkv_kernel(
    const float *__restrict__ qkv_result,
    float *__restrict__ q,
    float *__restrict__ k,
    float *__restrict__ v,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dim)
    {
        // 直接从QKV结果中提取对应部分
        q[idx] = qkv_result[idx];           // Q部分
        k[idx] = qkv_result[idx + dim];     // K部分
        v[idx] = qkv_result[idx + 2 * dim]; // V部分
    }
}

void GPU_Backend::extract_qkv(
    const float *qkv_result,
    float *q,
    float *k,
    float *v,
    int dim,
    hipStream_t stream)
{
    // 输入验证
    if (!qkv_result || !q || !k || !v || dim <= 0)
    {
        fprintf(stderr, "GPU extract_qkv Error: Invalid input pointers or dimensions.\n");
        return;
    }

    hipStream_t useStream = stream ? stream : this->stream;

    // 计算网格和块大小
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;

    // 启动内核
    hipLaunchKernelGGL(
        extract_qkv_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        useStream,
        qkv_result, q, k, v, dim);

    HIP_CHECK(hipGetLastError());
}

// 优化版本的矩阵乘法：使用更好的线程块布局
__global__ void __launch_bounds__(256) optimized_matmul_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int input_dim,
    int output_dim
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= output_dim) return;
    
    float result = 0.0f;
    
    // 使用8路展开以提高内存带宽利用率
    int i = 0;
    for (; i + 7 < input_dim; i += 8) {
        float input_vals[8];
        float weight_vals[8];
        
        // 预加载输入值
        input_vals[0] = input[i];
        input_vals[1] = input[i+1];
        input_vals[2] = input[i+2];
        input_vals[3] = input[i+3];
        input_vals[4] = input[i+4];
        input_vals[5] = input[i+5];
        input_vals[6] = input[i+6];
        input_vals[7] = input[i+7];
        
        // 预加载权重值（转置访问）
        weight_vals[0] = weight[output_idx * input_dim + i];
        weight_vals[1] = weight[output_idx * input_dim + i + 1];
        weight_vals[2] = weight[output_idx * input_dim + i + 2];
        weight_vals[3] = weight[output_idx * input_dim + i + 3];
        weight_vals[4] = weight[output_idx * input_dim + i + 4];
        weight_vals[5] = weight[output_idx * input_dim + i + 5];
        weight_vals[6] = weight[output_idx * input_dim + i + 6];
        weight_vals[7] = weight[output_idx * input_dim + i + 7];
        
        // 计算点积
        result += input_vals[0] * weight_vals[0] +
                  input_vals[1] * weight_vals[1] +
                  input_vals[2] * weight_vals[2] +
                  input_vals[3] * weight_vals[3] +
                  input_vals[4] * weight_vals[4] +
                  input_vals[5] * weight_vals[5] +
                  input_vals[6] * weight_vals[6] +
                  input_vals[7] * weight_vals[7];
    }
    
    // 处理剩余元素
    for (; i < input_dim; i++) {
        result += input[i] * weight[output_idx * input_dim + i];
    }
    
    output[output_idx] = result;
}

void GPU_Backend::matmul_optimized(
    float* output,
    const float* input,
    const float* weight,
    int input_dim,
    int output_dim,
    hipStream_t stream
) {
    if (!output || !input || !weight || input_dim <= 0 || output_dim <= 0) {
        fprintf(stderr, "GPU matmul_optimized Error: Invalid input parameters.\n");
        return;
    }
    
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 使用更大的线程块以提高并行度
    const int block_size = 256;
    const int grid_size = (output_dim + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(
        optimized_matmul_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        useStream,
        output, input, weight, input_dim, output_dim
    );
    
    HIP_CHECK(hipGetLastError());
}
