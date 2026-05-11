#include "gpu_backend.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>
#include "../hip_utils.hpp"
#include <cmath>
#include <cfloat>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// RMSNorm内核
__global__ void __launch_bounds__(512) rmsnorm_kernel(
    float *__restrict__ o,
    const float *__restrict__ x,
    const float *__restrict__ weight,
    int size,
    float epsilon)
{
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    unsigned int lane_id = tid % 64;

    extern __shared__ float s_sum[];

    // 计算线程局部平方和
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        float val = x[i];
        thread_sum_sq += val * val;
    }

#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)
    {
        thread_sum_sq += __shfl_down(thread_sum_sq, offset);
    }

    if (lane_id == 0)
    {
        s_sum[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float warp_sum = (lane_id < (blockDim.x + 63) / 64) ? s_sum[lane_id] : 0.0f;
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)
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
}

GPU_Backend::~GPU_Backend()
{
    // 销毁hipBLAS句柄
    if (blas_handle)
    {
        HIPBLAS_CHECK(hipblasDestroy(blas_handle));
        blas_handle = nullptr;
    }

    if (stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
        stream = nullptr;
    }
}

void GPU_Backend::matmul(float *o_d,
                         const float *x_d,
                         const float *w_d,
                         int n,
                         int d,
                         hipStream_t stream)
{
    hipStream_t useStream = stream ? stream : this->stream;
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 对于矩阵-向量乘法，使用GEMV而不是GEMM以获得更高效率
    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_T, 
        n,            
        d,            
        &alpha,       
        w_d,          
        n,            
        x_d,          
        1,            
        &beta,        
        o_d,          
        1             
        ));
}

void GPU_Backend::rmsnorm(
    float *o,
    const float *x,
    const float *weight,
    int size,
    hipStream_t stream)
{
    hipStream_t useStream = stream ? stream : this->stream;

    const float epsilon = 1e-5f;
    const int block_size = 256;

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
{
    hipStream_t useStream = stream ? stream : this->stream;
    int numPairs = (dim + 1) / 2;
    const int threads = 256;
    int blocks = (numPairs + threads - 1) / threads;
    hipLaunchKernelGGL(
        ropeEncoding_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        useStream, 
        q, k, headSize, position, dim, kvDim);
    HIP_CHECK(hipGetLastError());
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
    hipStream_t useStream = stream ? stream : this->stream;

    const int threads = 256;
    const int blocks = (hiddenDim + threads - 1) / threads;
    hipLaunchKernelGGL(
        swiGLLU_kernel,
        dim3(blocks),  
        dim3(threads), 
        0,             
        useStream,     
        hb,            
        hb2,
        hiddenDim);
    HIP_CHECK(hipGetLastError());
}

// QK计算和Softmax内核
__global__ void __launch_bounds__(512) flash_qk_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k_cache,
    float *__restrict__ scores,
    float *__restrict__ attn,
    int seq_len,
    int head_size,
    int num_heads,
    int num_kv_heads)
{
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    unsigned int lane_id = tid % 64;

    extern __shared__ float s_data[];
    float *s_max = s_data;
    float *s_sum = &s_data[blockDim.x];

    if (head_idx >= num_heads)
        return;

    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_group_size;
    const float *q_head = q + head_idx * head_size;
    float scale = 1.0f / sqrtf((float)head_size);
    float thread_max = -FLT_MAX;

    for (int i = tid; i < seq_len; i += blockDim.x)
    {
        float score = 0.0f;
        const float *k_vec = k_cache + i * num_kv_heads * head_size + kv_head_idx * head_size;

        int d = 0;
        for (; d + 7 < head_size; d += 8)
        {
            score += q_head[d] * k_vec[d] +
                     q_head[d + 1] * k_vec[d + 1] +
                     q_head[d + 2] * k_vec[d + 2] +
                     q_head[d + 3] * k_vec[d + 3] +
                     q_head[d + 4] * k_vec[d + 4] +
                     q_head[d + 5] * k_vec[d + 5] +
                     q_head[d + 6] * k_vec[d + 6] +
                     q_head[d + 7] * k_vec[d + 7];
        }
        for (; d < head_size; d++)
        {
            score += q_head[d] * k_vec[d];
        }

        score *= scale;
        scores[head_idx * seq_len + i] = score;
        thread_max = fmaxf(thread_max, score);
    }

#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)
    {
        thread_max = fmaxf(thread_max, __shfl_down(thread_max, offset));
    }

    if (lane_id == 0)
    {
        s_max[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float warp_max = (lane_id < (blockDim.x + 63) / 64) ? s_max[lane_id] : -FLT_MAX;
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)
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
    float thread_sum = 0.0f;

    for (int i = tid; i < seq_len; i += blockDim.x)
    {
        float score = scores[head_idx * seq_len + i];
        float exp_val = expf(score - max_val);
        attn[head_idx * seq_len + i] = exp_val;
        thread_sum += exp_val;
    }

#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)
    {
        thread_sum += __shfl_down(thread_sum, offset);
    }

    if (lane_id == 0)
    {
        s_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float warp_sum = ((unsigned)lane_id < (blockDim.x + 63) / 64) ? s_sum[lane_id] : 0.0f;
#pragma unroll
        for (int offset = 32; offset > 0; offset /= 2)
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

// 输出计算内核
__global__ void flash_output_kernel(
    const float *__restrict__ attn,
    const float *__restrict__ v_cache,
    float *__restrict__ output,
    int seq_len,
    int head_size,
    int num_heads,
    int num_kv_heads)
{
    int head_idx = blockIdx.x;
    int feat_idx = threadIdx.x;

    if (head_idx >= num_heads || feat_idx >= head_size)
        return;

    float acc = 0.0f;
    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_group_size;
    const float *attn_head = attn + head_idx * seq_len;
    const float *v_base = v_cache + kv_head_idx * head_size + feat_idx;

    int i = 0;
    for (; i + 3 < seq_len; i += 4)
    {
        float a0 = attn_head[i];
        float a1 = attn_head[i + 1];
        float a2 = attn_head[i + 2];
        float a3 = attn_head[i + 3];

        float v0 = v_base[i * num_kv_heads * head_size];
        float v1 = v_base[(i + 1) * num_kv_heads * head_size];
        float v2 = v_base[(i + 2) * num_kv_heads * head_size];
        float v3 = v_base[(i + 3) * num_kv_heads * head_size];

        acc += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
    }

    for (; i < seq_len; i++)
    {
        float a = attn_head[i];
        float v = v_base[i * num_kv_heads * head_size];
        acc += a * v;
    }

    output[head_idx * head_size + feat_idx] = acc;
}



__global__ void flash_attention_online_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k_cache,
    const float *__restrict__ v_cache,
    float *__restrict__ out,
    int seq_len,
    int head_size,
    int num_heads,
    int num_kv_heads)
{
    constexpr int TILE_N = 64;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= num_heads)
        return;

    extern __shared__ float smem[];
    float *red = smem; // blockDim.x
    float *tile_scores = smem + blockDim.x; // TILE_N

    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_head = h / kv_group_size;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f; // each thread accumulates one dim lane
    const float scale = rsqrtf((float)head_size);

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_N)
    {
        const int tile_len = min(TILE_N, seq_len - tile_start);
        float tile_m = -INFINITY;

        // pass1: compute tile scores and tile max
        for (int j = 0; j < tile_len; ++j)
        {
            int t = tile_start + j;
            float partial = 0.0f;
            if (tid < head_size)
            {
                partial = q[h * head_size + tid] * k_cache[t * num_kv_heads * head_size + kv_head * head_size + tid];
            }

            red[tid] = partial;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                    red[tid] += red[tid + stride];
                __syncthreads();
            }

            if (tid == 0)
            {
                float s = red[0] * scale;
                tile_scores[j] = s;
                tile_m = fmaxf(tile_m, s);
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            red[0] = tile_m;
        }
        __syncthreads();
        tile_m = red[0];

        float m_new = fmaxf(m, tile_m);
        float alpha = expf(m - m_new);

        // pass2: sum exp and update accumulator
        float tile_l = 0.0f;
        for (int j = tid; j < tile_len; j += blockDim.x)
        {
            tile_l += expf(tile_scores[j] - m_new);
        }
        red[tid] = tile_l;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
                red[tid] += red[tid + stride];
            __syncthreads();
        }
        float l_new = alpha * l + red[0];

        if (tid < head_size)
        {
            float weighted = 0.0f;
            for (int j = 0; j < tile_len; ++j)
            {
                int t = tile_start + j;
                float p = expf(tile_scores[j] - m_new);
                weighted += p * v_cache[t * num_kv_heads * head_size + kv_head * head_size + tid];
            }
            float prev = acc;
            acc = (alpha * l * prev + weighted) / (l_new + 1e-12f);
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    if (tid < head_size)
    {
        out[h * head_size + tid] = acc;
    }
}
void GPU_Backend::flash_attention(
    float *q,
    float *k_cache,
    float *v_cache,
    float *output,
    float *scores,
    float *attn,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_size,
    hipStream_t stream)
{
    hipStream_t useStream = stream ? stream : this->stream;
    const int launch_threads = ((head_size + 63) / 64) * 64;
    if (launch_threads > 1024)
    {
        // 超出单个block线程上限时回退到classic路径
        classic_attention(q, k_cache, v_cache, output, scores, attn, seq_len, num_heads, num_kv_heads, head_size, useStream);
        return;
    }

    // 标准FlashAttention核心思想：在线softmax + 不落地完整attention矩阵
    // 每个block处理一个query head，通过m/l在线归一化累计输出
    size_t shmem = (launch_threads + 64) * sizeof(float);
    hipLaunchKernelGGL(flash_attention_online_kernel, dim3(num_heads), dim3(launch_threads), shmem, useStream,
                       q, k_cache, v_cache, output, seq_len, head_size, num_heads, num_kv_heads);
    HIP_CHECK(hipGetLastError());
}

void GPU_Backend::classic_attention(
    float *q,
    float *k_cache,
    float *v_cache,
    float *output,
    float *scores,
    float *attn,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_size,
    hipStream_t stream)
{
    hipStream_t useStream = stream ? stream : this->stream;
    const int block_size = 256;

    dim3 grid_qk(num_heads);
    dim3 block_qk(block_size);
    size_t shmem_size_qk = 2 * block_size * sizeof(float);

    dim3 grid_output(num_heads);
    dim3 block_output(head_size);

    flash_qk_kernel<<<grid_qk, block_qk, shmem_size_qk, useStream>>>(
        q, k_cache, scores, attn, seq_len, head_size, num_heads, num_kv_heads);
    HIP_CHECK(hipGetLastError());

    flash_output_kernel<<<grid_output, block_output, 0, useStream>>>(
        attn, v_cache, output, seq_len, head_size, num_heads, num_kv_heads);
    HIP_CHECK(hipGetLastError());
}

// matmul_axpy函数实现
void GPU_Backend::matmul_axpy(
    float *out,        
    const float *x,    
    const float *w,    
    float factor,      
    int n,             
    int d,             
    hipStream_t stream 
)
{
    hipStream_t useStream = stream ? stream : this->stream;
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

    const float alpha = factor;
    const float beta = 1.0f;

    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_T,
        n,
        d,
        &alpha,
        w,
        n,
        x,
        1,
        &beta,
        out,
        1));

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
        q[idx] = qkv_result[idx];
        k[idx] = qkv_result[idx + dim];
        v[idx] = qkv_result[idx + 2 * dim];
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
    hipStream_t useStream = stream ? stream : this->stream;

    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;

    hipLaunchKernelGGL(
        extract_qkv_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        useStream,
        qkv_result, q, k, v, dim);

    HIP_CHECK(hipGetLastError());
}
    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_group_size;
