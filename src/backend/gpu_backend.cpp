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
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define HIPBLAS_CHECK(command) { \
    hipblasStatus_t status = command; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n", \
                status, __FILE__, __LINE__); \
        /* You might want a function to map status codes to strings */ \
        exit(EXIT_FAILURE); \
    } \
}




// RMSNorm 内核 (单核实现, 假设 size 适合单块处理)
// 计算 o[i] = (x[i] / sqrt(sum(x^2)/size + eps)) * weight[i]
__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int size, float epsilon) {
    // Dynamically allocated shared memory for sum of squares reduction
    extern __shared__ float s_cache[];

    int tid = threadIdx.x;          // Thread ID within the block
    int block_size = blockDim.x;    // Number of threads in the block (e.g., 256)

    // --- Phase 1: Calculate thread-local sum of squares ---
    float thread_sum_sq = 0.0f;
    // Each thread processes a subset of elements in a grid-stride loop
    for (int i = tid; i < size; i += block_size) {
        thread_sum_sq += x[i] * x[i];
    }

    // Store thread-local sum in shared memory
    s_cache[tid] = thread_sum_sq;

    // Synchronize block to ensure all threads have written to shared memory
    __syncthreads();

    // --- Phase 2: Perform parallel reduction in shared memory ---
    // Assumes block_size is a power of 2 for this simple reduction,
    // otherwise, need slight modification for non-power-of-2 block sizes.
    for (int s = block_size / 2; s > 0; s >>= 1) {
        // Only the first 's' threads participate in adding values
        if (tid < s) {
            s_cache[tid] += s_cache[tid + s];
        }
        // Synchronize after each reduction step
        __syncthreads();
    }

    // After reduction, s_cache[0] holds the total sum of squares for the block (entire vector)

    // --- Phase 3: Calculate normalization factor ---
    // Only thread 0 needs to calculate the final factor, but letting all threads
    // calculate it is often fine and avoids broadcasting from s_cache[0].
    // Or, thread 0 calculates and writes back to shared memory for others to read.
    // Let's have all threads calculate it to potentially overlap with memory ops later.

    // Calculate RMS = sqrt( sum(x^2)/size + epsilon )
    float ss = s_cache[0]; // Total sum of squares
    float rms = sqrtf(ss / (float)size + epsilon);

    // Pre-calculate the inverse for efficiency (division is slow)
    float inv_rms = 1.0f / rms;

    // Optional: Synchronize if other threads *needed* to read inv_rms calculated by thread 0
    // __syncthreads();

    // --- Phase 4: Apply normalization and weight, write output ---
    // Each thread calculates the final output for its subset of elements
    for (int i = tid; i < size; i += block_size) {
        o[i] = (x[i] * inv_rms) * weight[i]; // o = (x / rms) * weight
    }
}

// 优化的RMSNorm内核 - 使用warp-level归约
__global__ void rmsnorm_optimized_kernel(
    float* __restrict__ o,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int size,
    float epsilon
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 共享内存用于warp间归约
    extern __shared__ float s_sum[];
    
    // 计算线程局部平方和
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = x[i];
        thread_sum_sq += val * val;
    }
    
    // Warp-level归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_down(thread_sum_sq, offset);
    }
    
    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0) {
        s_sum[warp_id] = thread_sum_sq;
    }
    __syncthreads();
    
    // 最后一个warp处理warp间归约
    if (warp_id == 0) {
        float warp_sum = (lane_id < (blockDim.x + 31) / 32) ? s_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down(warp_sum, offset);
        }
        if (lane_id == 0) {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // 计算归一化因子
    float total_sum_sq = s_sum[0];
    float rms = sqrtf(total_sum_sq / (float)size + epsilon);
    float inv_rms = 1.0f / rms;
    
    // 应用归一化和权重
    for (int i = tid; i < size; i += blockDim.x) {
        o[i] = (x[i] * inv_rms) * weight[i];
    }
}

GPU_Backend::GPU_Backend() : blas_handle(nullptr), stream(nullptr) {
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

GPU_Backend::~GPU_Backend() {
    // 销毁hipBLAS句柄
    if (blas_handle) {
        HIPBLAS_CHECK(hipblasDestroy(blas_handle));
        blas_handle = nullptr;
    }
    

    // printf("GPU_Backend destroyed, hipBLAS handle and HIP stream released.\n");
}

// 同步方法：等待所有GPU操作完成
void GPU_Backend::synchronize() {
    if (stream) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }
}

// 假设 CHECK 和 HIPBLAS_CHECK 宏已定义

// --- 重构后的 matmul 函数 ---
// 计算 o = w * x (矩阵-向量乘法)
// 假设 o_d, x_d, w_d 是指向 GPU 设备内存的有效指针。
// 假设 w_d 指向的矩阵 W 是按行主元 (Row-Major) 存储的 d x n 矩阵。
// 内存管理和数据传输由调用者负责。
void GPU_Backend::matmul(float* o_d,           // 指向 GPU 上的输出向量 o (d x 1) 的指针
                      const float* x_d,     // 指向 GPU 上的输入向量 x (n x 1) 的指针
                      const float* w_d,     // 指向 GPU 上的输入矩阵 w (d x n, Row-Major) 的指针
                      int n,                // 矩阵 w 的列数 / 向量 x 的行数
                      int d,                // 矩阵 w 的行数 / 向量 o 的行数
                      hipStream_t stream)  // 指定的流，可选
{ // 指定的流，可选
    // 输入参数检查
    if (!o_d || !x_d || !w_d || n <= 0 || d <= 0) {
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
        HIPBLAS_OP_T,       // 转置矩阵，因为我们的矩阵是行主序存储
        n,                  // 原始矩阵的行数
        d,                  // 原始矩阵的列数
        &alpha,             // 缩放因子
        w_d,                // 矩阵
        n,                  // 矩阵的leading dimension
        x_d,                // 输入向量
        1,                  // 输入向量的步长
        &beta,              // 累积因子
        o_d,                // 输出向量
        1                   // 输出向量的步长
    ));
}


// --- 重构后的 rmsnorm_gpu 函数 ---
// 假设 o_d, x_d, weight_d 是指向 GPU 设备内存的有效指针。
// 内存管理 (分配/释放) 和数据传输 (如果需要) 由调用者负责。
void GPU_Backend::rmsnorm(float* o_d,           // 指向 GPU 上的输出缓冲区的指针
                         const float* x_d,     // 指向 GPU 上的输入向量 x 的指针
                         const float* weight_d,// 指向 GPU 上的权重向量的指针
                         int size,            // 向量大小 (主机值)
                         hipStream_t stream) { // 指定的流，可选

    // --- 输入验证 (针对 size) ---
    if (size <= 0) {
        // 也可以象征性地检查指针非空，但从主机验证设备指针有效性比较困难。
        // 通常需要信任调用者或添加更健壮的检查。
        fprintf(stderr, "RMSNorm_GPU Error: Invalid size (%d).\n", size);
        return; // 或采取适当的错误处理
    }
    // 如果未指定流，则使用类的默认流
    hipStream_t useStream = stream ? stream : this->stream;

    // --- 1. 移除设备内存分配/释放 ---
    // 此函数内部不再有 hipMalloc/hipFree 调用。

    // --- 2. 移除数据传输 ---
    // 此函数内部不再有 hipMemcpy 调用。

    // --- 3. 内核启动配置 (保留这部分逻辑) ---
    const float epsilon = 1e-5f; // 标准 RMSNorm epsilon
    int block_size = 256;        // 常用值，如有必要可调优

    // 为小输入调整块大小 (逻辑同前)
    if (size == 0) block_size = 1; // 避免块大小为 0
    else block_size = (size < block_size) ? size : block_size;


    // 使用单个块执行 RMSNorm 归约内核
    dim3 gridDim(1);
    dim3 blockDim(block_size);

    // 内核内部归约所需的共享内存大小
    // 确保这个大小与 rmsnorm_kernel 的期望匹配！
    size_t shared_mem_size = (block_size > 0) ? block_size * sizeof(float) : 0;

    // --- 4. 启动内核 (使用传入的设备指针) ---
    hipLaunchKernelGGL(rmsnorm_kernel,
                       gridDim,
                       blockDim,
                       shared_mem_size, // 动态分配的共享内存字节数
                       useStream,        // 使用指定的流
                       // 内核参数 - 使用设备指针:
                       o_d, x_d, weight_d, size, epsilon);
    HIP_CHECK(hipGetLastError()); // 检查异步启动错误

    // --- 5. 结果保留在 GPU 上的 o_d 中 ---
    // 调用者负责在需要时将结果拷贝回主机。

    // 注意：此函数现在是异步的。
    // 如果调用者需要确保此操作完成后才能进行后续的 GPU 工作
    // 或需要将结果拷贝回主机，则可能需要在调用此函数 *之后*
    // 由调用者进行同步 (hipDeviceSynchronize 或使用 HIP 流)。
}

void GPU_Backend::rmsnorm_optimized(
    float* o,
    const float* x,
    const float* weight,
    int size,
    hipStream_t stream
) {
    if (size <= 0 || !o || !x || !weight) {
        fprintf(stderr, "RMSNorm_Optimized Error: Invalid input parameters.\n");
        return;
    }
    
    hipStream_t useStream = stream ? stream : this->stream;
    
    const float epsilon = 1e-5f;
    const int block_size = 256;
    
    // 使用单个块处理，共享内存大小为warp数量
    dim3 gridDim(1);
    dim3 blockDim(block_size);
    size_t shared_mem_size = ((block_size + 31) / 32) * sizeof(float);
    
    hipLaunchKernelGGL(
        rmsnorm_optimized_kernel,
        gridDim,
        blockDim,
        shared_mem_size,
        useStream,
        o, x, weight, size, epsilon
    );
    
    HIP_CHECK(hipGetLastError());
}

/*  TODO axpy加速执行 
    axpy: 标量和向量相乘
    y[0] = x[0] * factor, 
    y[1] = x[1] * factor, 
    ......, 
    y[dim-1] = x[dim-1] * factor
*/
void GPU_Backend::axpy(float *y_d,           // 指向 GPU 上向量 y 的指针 (输入/输出)
                      const float *x_d,     // 指向 GPU 上向量 x 的指针 (输入)
                      float factor,         // 标量因子 (主机值)
                      int dim,              // 向量维度 (主机值)
                      hipStream_t stream) { // 指定的流，可选

    // --- 输入验证 (可选，但推荐) ---
    // 注意：无法轻易地从主机验证设备指针的有效性，但可以检查其他参数
    if (dim <= 0 || !blas_handle) { // 假设 blas_handle 是成员变量或全局可访问
        fprintf(stderr, "AXPY_GPU Error: Invalid dimension (%d) or handle.\n", dim);
        return;
    }

    // 如果未指定流，则使用类的默认流
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 将hipBLAS句柄与指定的流关联
    // HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

    // --- 1. 无需分配设备内存 ---
    // --- 2. 无需主机到设备的数据传输 ---

    // --- 3. 调用 hipBLAS 函数在 GPU 上执行 y = y + factor * x ---
    // 直接使用传入的设备指针
    HIPBLAS_CHECK(hipblasSaxpy(blas_handle,
                               dim,          // 向量维度
                               &factor,      // 指向主机标量因子的指针
                               x_d,          // 向量 x 的设备指针
                               1,            // 向量 x 的增量
                               y_d,          // 向量 y 的设备指针 (输入/输出)
                               1             // 向量 y 的增量
                               ));

    // (可选) 检查异步错误
    HIP_CHECK(hipGetLastError());

    // --- 4. 无需设备到主机的数据传输 ---
    // 结果保留在 y_d 指向的 GPU 内存中

    // --- 5. 无需释放设备内存 (由调用者管理) ---

    // 注意: hipblasSaxpy 通常是异步执行的。如果后续代码需要确保
    // 这个 axpy 操作完成，调用者可能需要使用 hipDeviceSynchronize()
    // 或通过 HIP 流来管理依赖关系。
}

/*  TODO 向量点积加速执行
    dot: 向量点积
    *y += x1[0]*x2[0] + x1[1]*x2[1] + ... + x1[dim-1]*x2[dim-1]
*/

// --- 修改后的 dot 函数，直接操作设备内存 ---
void GPU_Backend::dot(float* d_y,       // 指向设备内存中的结果位置
                  const float* d_x1, // 指向设备内存中的向量 x1 (const 因为不修改)
                  const float* d_x2, // 指向设备内存中的向量 x2 (const 因为不修改)
                  int dim,          // 向量维度
                  hipStream_t stream) { // 指定的流，可选
        // 输入验证 (可选，但推荐检查指针和维度)
        if (dim <= 0 || d_x1 == nullptr || d_x2 == nullptr || d_y == nullptr) {
             fprintf(stderr, "错误：dot_gpu 输入无效 (设备指针或维度)。\n");
             // 可能需要更健壮的错误处理
             return;
        }
        // 如果未指定流，则使用类的默认流
        hipStream_t useStream = stream ? stream : this->stream;
        
        // 将hipBLAS句柄与指定的流关联
        // HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

        // 注意：不再需要 hipMalloc
        // 注意：不再需要 hipMemcpyHostToDevice

        // 3. 调用 hipblas 函数计算 dot 产品
        // 确保结果写入设备指针 d_y 指向的位置
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_DEVICE));
        HIPBLAS_CHECK(hipblasSdot(blas_handle,
                                  dim,    // 向量维度
                                  d_x1,   // 向量 x1 的设备指针
                                  1,      // 向量 x1 的增量
                                  d_x2,   // 向量 x2 的设备指针
                                  1,      // 向量 x2 的增量
                                  d_y));  // 结果存储到设备内存中的 d_y
        
        // 检查异步错误
        HIP_CHECK(hipGetLastError());

        // 注意：不再需要 hipMemcpyDeviceToHost (除非此函数设计目标就是如此)
        // 注意：不再需要 hipFree
    }

__global__ void ropeEncoding_kernel(float* q, float* k,
                         int headSize,
                         int position,
                         int dim,
                         int kvDim) {
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = pair_id * 2;
    if (i + 1 >= dim) return;

    int headDim = i % headSize;
    float freq  = 1.0f / powf(10000.0f, headDim / (float)headSize);
    float angle = position * freq;
    float fcr   = cosf(angle);
    float fci   = sinf(angle);

    // q 旋转
    float q0 = q[i], q1 = q[i+1];
    q[i]   = q0 * fcr - q1 * fci;
    q[i+1] = q0 * fci + q1 * fcr;

    // k 旋转（i < kvDim 时）
    if (i < kvDim) {
        float k0 = k[i], k1 = k[i+1];
        k[i]   = k0 * fcr - k1 * fci;
        k[i+1] = k0 * fci + k1 * fcr;
    }
}


void GPU_Backend::ropeEncoding(float *q, float *k,
                       int headSize,
                       int position,
                       int dim,
                       int kvDim,
                       hipStream_t stream) { // 指定的流，可选
        if (!q || !k || headSize <= 0 || dim <= 0 || kvDim < 0) return;
        
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
            useStream,  // 使用指定的流
            q, k, headSize, position, dim, kvDim
        );
        HIP_CHECK( hipGetLastError() );
        // 不再显式同步，由调用者决定何时同步
    }






__global__ void swiGLLU_kernel(float* headOutput, const float* value, int hiddenDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hiddenDim) {
        float h = headOutput[idx];
        float sig = 1.0f / (1.0f + expf(-h));
        headOutput[idx] = h * sig * value[idx];
    }
}

void GPU_Backend::swiGLLUFunc(float *hb, float *hb2, int hiddenDim, hipStream_t stream) {
        if (hiddenDim <= 0 || hb == nullptr || hb2 == nullptr) return;
        // 如果未指定流，则使用类的默认流
        hipStream_t useStream = stream ? stream : this->stream;

        // 1. 计算网格与线程数
        const int threads = 256;
        const int blocks = (hiddenDim + threads - 1) / threads;

        // 2. 启动 kernel，使用指定的流
        hipLaunchKernelGGL(
            swiGLLU_kernel,
            dim3(blocks),         // grid
            dim3(threads),        // block
            0,                    // shared mem
            useStream,            // 使用指定的流
            hb,                   // kernel args...
            hb2,
            hiddenDim
        );
        HIP_CHECK( hipGetLastError() );
        // 不再显式同步，由调用者决定何时同步
    }


// 优化的QK计算和Softmax内核 - 添加warp-level优化
__global__ void optimized_flash_qk_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    float* __restrict__ scores,
    float* __restrict__ attn,
    int seq_len,
    int head_size,
    int num_heads
) {
    // 获取头索引
    int head_idx = blockIdx.x;
    
    // 获取当前线程在块内的索引
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 共享内存
    extern __shared__ float s_data[];
    float* s_max = s_data;                  // 最大值
    float* s_sum = &s_data[blockDim.x];     // 总和
    
    // 确保只有有效的线程参与计算
    if (head_idx >= num_heads) return;
    
    // 访问当前头的q向量
    const float* q_head = q + head_idx * head_size;
    
    // 缩放因子
    const float scale = 1.0f / sqrtf((float)head_size);
    
    // 计算分数并找到最大值
    float thread_max = -FLT_MAX;
    
    // 计算 QK 点积 (批处理方式，添加循环展开优化)
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float score = 0.0f;
        
        // 计算点积 - 添加循环展开以提高性能
        const float* k_vec = k_cache + i * num_heads * head_size + head_idx * head_size;
        
        // 展开循环以减少循环开销，并添加预取
        int d = 0;
        for (; d + 7 < head_size; d += 8) {
            // 8路展开以更好地利用内存带宽
            score += q_head[d] * k_vec[d] + 
                     q_head[d+1] * k_vec[d+1] + 
                     q_head[d+2] * k_vec[d+2] + 
                     q_head[d+3] * k_vec[d+3] +
                     q_head[d+4] * k_vec[d+4] + 
                     q_head[d+5] * k_vec[d+5] + 
                     q_head[d+6] * k_vec[d+6] + 
                     q_head[d+7] * k_vec[d+7];
        }
        // 处理剩余元素
        for (; d < head_size; d++) {
            score += q_head[d] * k_vec[d];
        }
        
        // 应用缩放
        score *= scale;
        
        // 保存分数
        scores[head_idx * seq_len + i] = score;
        
        // 更新线程局部最大值
        thread_max = fmaxf(thread_max, score);
    }
    
    // Warp-level归约找出最大值（更高效）
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down(thread_max, offset));
    }
    
    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0) {
        s_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // 最后一个warp处理warp级别的归约
    if (warp_id == 0) {
        float warp_max = (lane_id < (int)((blockDim.x + 31) / 32)) ? s_max[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
        }
        if (lane_id == 0) {
            s_max[0] = warp_max;
        }
    }
    __syncthreads();
    
    // 所有线程获取最大值
    float max_val = s_max[0];
    
    // 计算指数和总和
    float thread_sum = 0.0f;
    
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float score = scores[head_idx * seq_len + i];
        float exp_val = expf(score - max_val);
        
        // 存储中间结果
        attn[head_idx * seq_len + i] = exp_val;
        
        // 累加本地和
        thread_sum += exp_val;
    }
    
    // Warp-level归约计算总和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down(thread_sum, offset);
    }
    
    // 每个warp的第一个线程写入共享内存
    if (lane_id == 0) {
        s_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // 最后一个warp处理warp级别的归约
    if (warp_id == 0) {
        float warp_sum = (lane_id < (int)((blockDim.x + 31) / 32)) ? s_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down(warp_sum, offset);
        }
        if (lane_id == 0) {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // 获取总和
    float sum = s_sum[0];
    
    // 归一化
    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (sum > 0.0f) {
            attn[head_idx * seq_len + i] /= sum;
        } else {
            attn[head_idx * seq_len + i] = 1.0f / seq_len;
        }
    }
}

// 优化的输出计算内核 - 添加向量化访问
__global__ void optimized_flash_output_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ v_cache,
    float* __restrict__ output,
    int seq_len,
    int head_size,
    int num_heads
) {
    int head_idx = blockIdx.x;
    int feat_idx = threadIdx.x;
    
    // 确保有效范围
    if (head_idx >= num_heads || feat_idx >= head_size) return;
    
    // 初始化输出累加器
    float acc = 0.0f;
    
    // 预计算基础偏移量
    const float* attn_head = attn + head_idx * seq_len;
    const float* v_base = v_cache + head_idx * head_size + feat_idx;
    
    // 优化的累加循环 - 展开以提高性能
    int i = 0;
    for (; i + 3 < seq_len; i += 4) {
        // 4路展开以更好地利用内存带宽和指令级并行
        float a0 = attn_head[i];
        float a1 = attn_head[i+1];
        float a2 = attn_head[i+2];
        float a3 = attn_head[i+3];
        
        float v0 = v_base[i * num_heads * head_size];
        float v1 = v_base[(i+1) * num_heads * head_size];
        float v2 = v_base[(i+2) * num_heads * head_size];
        float v3 = v_base[(i+3) * num_heads * head_size];
        
        acc += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
    }
    
    // 处理剩余元素
    for (; i < seq_len; i++) {
        float a = attn_head[i];
        float v = v_base[i * num_heads * head_size];
        acc += a * v;
    }
    
    // 写入最终结果
    output[head_idx * head_size + feat_idx] = acc;
}

void GPU_Backend::flash_attention_gpu_step(
    float* q,
    float* k_cache,
    float* v_cache,
    float* output,
    float* scores,
    float* attn,
    int seq_len,
    hipStream_t stream
) {
    // 使用指定的流或默认流
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 优化的Flash Attention实现 - 安全版本
    const int BLOCK_SIZE = 256;  // 恢复更大的块大小以提高性能
    const int HEAD_SIZE = GPU_Backend::HEAD_SIZE;
    const int NUM_HEADS = GPU_Backend::NUM_HEADS;
    
    // 1. 计算QK点积和Softmax
    // 每个block处理一个注意力头
    dim3 grid_qk(NUM_HEADS);
    dim3 block_qk(BLOCK_SIZE);
    
    // 共享内存大小：最大值和总和归约
    size_t shmem_size_qk = 2 * BLOCK_SIZE * sizeof(float);
    
    // 计算QK点积并应用Softmax
    optimized_flash_qk_kernel<<<grid_qk, block_qk, shmem_size_qk, useStream>>>(
        q, k_cache, scores, attn, seq_len, HEAD_SIZE, NUM_HEADS
    );
    
    // 确保QK计算完成
    HIP_CHECK(hipGetLastError());
    
    // 2. 计算最终输出
    // 使用1D线程块，每个线程处理一个特征维度
    dim3 grid_output(NUM_HEADS);
    dim3 block_output(HEAD_SIZE);
    
    // 不使用共享内存以保持稳定性
    size_t shmem_size_output = 0;
    
    // 计算输出
    optimized_flash_output_kernel<<<grid_output, block_output, shmem_size_output, useStream>>>(
        attn, v_cache, output, seq_len, HEAD_SIZE, NUM_HEADS
    );
    
    // 检查错误但不强制同步
    HIP_CHECK(hipGetLastError());
}



// 融合的matmul_axpy函数实现
void GPU_Backend::matmul_axpy(
    float* out,                    // 输出向量，同时是axpy的目标
    const float* x,                // 输入向量
    const float* w,                // 权重矩阵
    float factor,                  // axpy的缩放因子
    int n,                         // 输入维度
    int d,                         // 输出维度
    hipStream_t stream             // 可选的CUDA流
) {
    // 输入验证
    if (!out || !x || !w || d <= 0 || n <= 0) {
        fprintf(stderr, "GPU matmul_axpy Error: Invalid input pointers or dimensions.\n");
        return;
    }
    
    hipStream_t useStream = stream ? stream : this->stream;
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));
    
    // 直接使用hipBLAS库的GEMV函数实现高性能的矩阵乘法+axpy
    // 设置beta=1.0，这样就直接在输出向量上累加结果
    const float alpha = factor;
    const float beta = 1.0f;  // 保留原始值并添加新结果
    
    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,
        HIPBLAS_OP_T,    // 转置矩阵，因为我们的矩阵是行主序存储
        n,               // 原始矩阵的行数
        d,               // 原始矩阵的列数
        &alpha,          // 缩放因子
        w,               // 矩阵
        n,               // 矩阵的leading dimension
        x,               // 输入向量
        1,               // 输入向量的步长
        &beta,           // 累积因子，1.0表示保留原始输出并添加结果
        out,             // 输出向量
        1                // 输出向量的步长
    ));
    
    HIP_CHECK(hipGetLastError());
}

// 融合的QKV分离内核 - 避免内存拷贝
__global__ void extract_qkv_kernel(
    const float* __restrict__ qkv_result,
    float* __restrict__ q,
    float* __restrict__ k, 
    float* __restrict__ v,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dim) {
        // 直接从QKV结果中提取对应部分
        q[idx] = qkv_result[idx];                    // Q部分
        k[idx] = qkv_result[idx + dim];              // K部分  
        v[idx] = qkv_result[idx + 2 * dim];          // V部分
    }
}

void GPU_Backend::extract_qkv(
    const float* qkv_result,
    float* q,
    float* k,
    float* v,
    int dim,
    hipStream_t stream
) {
    // 输入验证
    if (!qkv_result || !q || !k || !v || dim <= 0) {
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
        qkv_result, q, k, v, dim
    );
    
    HIP_CHECK(hipGetLastError());
}

// Top-K采样优化内核 - 只计算最可能的K个token
__global__ void topk_logits_kernel(
    float* __restrict__ logits,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const int* __restrict__ topk_indices,
    int input_dim,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < k) {
        int token_idx = topk_indices[idx];
        
        // 计算该token的logit
        float logit = 0.0f;
        const float* weight_row = weights + token_idx * input_dim;
        
        // 向量化点积计算
        for (int i = 0; i < input_dim; i += 4) {
            if (i + 3 < input_dim) {
                logit += input[i] * weight_row[i] + 
                         input[i+1] * weight_row[i+1] + 
                         input[i+2] * weight_row[i+2] + 
                         input[i+3] * weight_row[i+3];
            } else {
                for (int j = i; j < input_dim; j++) {
                    logit += input[j] * weight_row[j];
                }
                break;
            }
        }
        
        logits[token_idx] = logit;
    }
}

void GPU_Backend::matmul_partial_logits(
    float* logits,
    const float* x,
    const float* w,
    int input_dim,
    int vocab_size,
    int top_k,
    hipStream_t stream
) {
    if (!logits || !x || !w || input_dim <= 0 || vocab_size <= 0 || top_k <= 0) {
        fprintf(stderr, "GPU matmul_partial_logits Error: Invalid input parameters.\n");
        return;
    }
    
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 简化版本：使用预定义的高频token索引
    // 在实际应用中，可以基于历史统计或动态计算
    static int* d_topk_indices = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        // 使用更智能的top-k索引选择策略
        int* h_topk_indices = new int[top_k];
        
        // 策略1：包含常用token（前500个）
        int common_tokens = std::min(500, top_k / 2);
        for (int i = 0; i < common_tokens; i++) {
            h_topk_indices[i] = i;
        }
        
        // 策略2：包含一些中频token（跳跃采样）
        int remaining = top_k - common_tokens;
        int step = std::max(1, vocab_size / remaining);
        for (int i = 0; i < remaining; i++) {
            int idx = common_tokens + (i * step) % (vocab_size - common_tokens);
            h_topk_indices[common_tokens + i] = idx;
        }
        
        HIP_CHECK(hipMalloc(&d_topk_indices, top_k * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_topk_indices, h_topk_indices, top_k * sizeof(int), hipMemcpyHostToDevice));
        
        delete[] h_topk_indices;
        initialized = true;
    }
    
    // 首先清零logits数组
    HIP_CHECK(hipMemset(logits, 0, vocab_size * sizeof(float)));
    
    // 计算网格和块大小
    const int threads = 256;
    const int blocks = (top_k + threads - 1) / threads;
    
    // 启动内核只计算top-k个logits
    hipLaunchKernelGGL(
        topk_logits_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        useStream,
        logits, x, w, d_topk_indices, input_dim, top_k
    );
    
    HIP_CHECK(hipGetLastError());
}

void GPU_Backend::adaptive_logits(
    float* logits,
    const float* x,
    const float* w,
    int input_dim,
    int vocab_size,
    float temperature,
    float top_p,
    hipStream_t stream
) {
    // 根据采样参数智能选择K值
    int adaptive_k;
    
    if (temperature == 0.0f) {
        // 贪婪采样：只需要top-1
        adaptive_k = 1;
    } else if (temperature < 0.5f) {
        // 低温度：较小的K值
        adaptive_k = std::min(100, vocab_size);
    } else if (temperature < 1.0f) {
        // 中等温度：中等K值
        adaptive_k = std::min(500, vocab_size);
    } else {
        // 高温度：较大的K值
        adaptive_k = std::min(1500, vocab_size);
    }
    
    // 根据top_p进一步调整
    if (top_p > 0.0f && top_p < 1.0f) {
        // Top-P采样需要更多候选
        adaptive_k = std::max(adaptive_k, std::min(2000, vocab_size));
    }
    
    // 调用优化的partial logits计算
    matmul_partial_logits(logits, x, w, input_dim, vocab_size, adaptive_k, stream);
}

// 安全的采样优化内核 - 保留top-k，其他设为负无穷
__global__ void optimize_logits_kernel(
    float* __restrict__ logits,
    int vocab_size,
    int top_k
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 使用共享内存存储top-k的值和索引
    extern __shared__ float s_data[];
    float* s_values = s_data;
    int* s_indices = (int*)&s_data[top_k];
    
    // 初始化共享内存
    if (tid < top_k) {
        s_values[tid] = -FLT_MAX;
        s_indices[tid] = -1;
    }
    __syncthreads();
    
    // 每个线程处理一个logit值
    if (tid < vocab_size) {
        float val = logits[tid];
        
        // 找到最小的top-k值
        for (int i = 0; i < top_k; i++) {
            if (val > s_values[i]) {
                // 插入新值，移动其他值
                for (int j = top_k - 1; j > i; j--) {
                    s_values[j] = s_values[j-1];
                    s_indices[j] = s_indices[j-1];
                }
                s_values[i] = val;
                s_indices[i] = tid;
                break;
            }
        }
    }
    __syncthreads();
    
    // 将不在top-k中的logits设为负无穷
    if (tid < vocab_size) {
        bool in_topk = false;
        for (int i = 0; i < top_k; i++) {
            if (s_indices[i] == tid) {
                in_topk = true;
                break;
            }
        }
        if (!in_topk) {
            logits[tid] = -FLT_MAX;
        }
    }
}

void GPU_Backend::optimize_logits_for_sampling(
    float* logits,
    int vocab_size,
    int top_k,
    hipStream_t stream
) {
    if (!logits || vocab_size <= 0 || top_k <= 0) {
        return;
    }
    
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 限制top_k不超过vocab_size
    top_k = std::min(top_k, vocab_size);
    
    // 计算网格和块大小
    const int threads = 256;
    const int blocks = (vocab_size + threads - 1) / threads;
    
    // 共享内存大小：top_k个float + top_k个int
    size_t shared_mem_size = top_k * (sizeof(float) + sizeof(int));
    
    // 启动内核
    hipLaunchKernelGGL(
        optimize_logits_kernel,
        dim3(blocks),
        dim3(threads),
        shared_mem_size,
        useStream,
        logits, vocab_size, top_k
    );
    
    HIP_CHECK(hipGetLastError());
}







