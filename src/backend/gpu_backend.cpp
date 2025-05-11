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


/*
 * Softmax 计算内核 (原地修改版本)
 * 从 data 读取原始值，计算后将结果写回 data
 * 假设一个块处理整个向量。
 * blockDim.x 最好是 2 的幂。
 */
__global__ void softmax_kernel_inplace(float* data, int size) {
    // 用于规约的共享内存 (最大值和指数和)
    extern __shared__ float s_data[];

    int tid = threadIdx.x;          // 块内线程 ID
    int block_size = blockDim.x;    // 块中的线程数

    // --- 阶段 1: 查找最大值 ---
    float thread_max = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        thread_max = fmaxf(thread_max, data[i]); // 从 data 读取
    }
    s_data[tid] = thread_max;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float block_max = s_data[0];
    __syncthreads();

    // --- 阶段 2: 计算指数和 ---
    float thread_sum_exp = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        thread_sum_exp += expf(data[i] - block_max); // 从 data 读取
    }
    s_data[tid] = thread_sum_exp;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float block_sum_exp = s_data[0];
    __syncthreads();

    // --- 阶段 3: 归一化并写回输出 (原地修改) ---
    for (int i = tid; i < size; i += block_size) {
        float exp_val = expf(data[i] - block_max); // 从 data 读取原始值相关的计算
        if (block_sum_exp > 0.0f) {
             data[i] = exp_val / block_sum_exp; // 将结果写回 data
        } else {
             data[i] = 0.0f; // 处理 sum 为 0 的情况
             // 或者 data[i] = 1.0f / (float)size;
        }
    }
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
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }
}

GPU_Backend::~GPU_Backend() {
    // 销毁hipBLAS句柄
    if (blas_handle) {
        HIPBLAS_CHECK(hipblasDestroy(blas_handle));
        blas_handle = nullptr;
    }
    // printf("GPU_Backend destroyed, hipBLAS handle and HIP stream released.\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipStreamDestroy(streams[i]));
    }
}

// 同步方法：等待所有GPU操作完成
void GPU_Backend::synchronize() {
    if (stream) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }
}

/*  TODO Softmax加速执行
    softmax: 将实数向量转换为概率分布
    (1) sum = e^(x[0]) + e^(x[1]) + ... + e^(x[n-1])
    (2) x[0] = (e^x[0]) / sum, 
        x[1] = (e^x[1]) / sum, 
        ......,
        x[n-1] = (e^x[n-1]) / sum
*/

void GPU_Backend::softmax(float* d_data, int size, hipStream_t stream) {
        // 输入验证 (可选，但推荐)
        if (d_data == nullptr || size <= 0) {
            fprintf(stderr, "错误：设备指针为空或大小无效。\n");
            // 可能需要更健壮的错误处理，例如返回错误码
            return;
        }
        // 如果未指定流，则使用类的默认流
        hipStream_t useStream = stream ? stream : this->stream;

        // 3. 配置内核启动参数 (与之前类似)
        int block_size = 256;
        // block_size = (size < block_size) ? nextPowerOf2(size) : block_size; // 精细调整

        dim3 gridDim(1); // 仍然使用一个块处理
        dim3 blockDim(block_size);
        // 共享内存大小，取决于 softmax_kernel_inplace 的实现
        size_t shared_mem_size = block_size * sizeof(float);

        // 4. 启动原地修改内核，使用指定的流
        hipLaunchKernelGGL(softmax_kernel_inplace, gridDim, blockDim, shared_mem_size, useStream, d_data, size);
        HIP_CHECK(hipGetLastError()); // 检查内核启动错误
        // 不再显式同步，由调用者决定何时同步
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
    // 如果传入了非默认流，则设置流
    if (stream != nullptr) {
        HIPBLAS_CHECK(hipblasSetStream(blas_handle, stream));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;


    int M = d;      // 行数 of op(A) and C. op(A) is d x n.
    int N_gemm = 1; // 列数 of op(B) and C. op(B) is n x 1.
    int K = n;      // 列数 of op(A) and 行数 of op(B).

    int lda = n;

    int ldb = n;

    int ldc = d;

    HIPBLAS_CHECK(hipblasSgemm(blas_handle,
                               HIPBLAS_OP_T, HIPBLAS_OP_N,
                               M, N_gemm, K,
                               &alpha,
                               w_d, lda,  // A (W_d), lda = n (K)
                               x_d, ldb,  // B (X_d), ldb = n (K)
                               &beta,
                               o_d, ldc   // C (O_d), ldc = d (M)
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
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

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
        HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));

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



void GPU_Backend::gemvQkSeq(float *q, float *key, float *scores, int pos, int kvDim, int headSize, hipStream_t stream) {
        if (pos < 0 || !q || !key || !scores || kvDim <= 0 || headSize <= 0 || !blas_handle) {
            fprintf(stderr, "gemvQkSeq Error: Invalid inputs (pos=%d, kvDim=%d, headSize=%d, handles=%p)\n", 
                    pos, kvDim, headSize, blas_handle);
            return;
        }
        // 确保pos >= 0，num至少为1
        int num = pos + 1;
        
        // 如果未指定流，则使用类的默认流
        hipStream_t useStream = stream ? stream : this->stream;
        
        // 将hipBLAS句柄与指定的流关联
        HIPBLAS_CHECK(hipblasSetStream(blas_handle, useStream));
        
        const float alpha = 1.0f / std::sqrt((float)headSize);
        const float beta  = 0.0f;
        
        // 检查指针有效性
        if (!q || !key || !scores) {
            fprintf(stderr, "gemvQkSeq Error: Null pointers detected (q=%p, key=%p, scores=%p)\n", q, key, scores);
            return;
        }
        
        // 检查LDA是否符合要求
        // 对于HIPBLAS_OP_T操作，主维度lda必须 >= max(1, num)，即原矩阵的行数
        if (kvDim < headSize) {
            fprintf(stderr, "gemvQkSeq Error: Invalid matrix dimensions (kvDim=%d < headSize=%d)\n", kvDim, headSize);
            return;
        }
        
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));
        
        // 打印调试信息
        // printf("gemvQkSeq: headSize=%d, num=%d, alpha=%f, key=%p, kvDim=%d, q=%p, scores=%p\n", 
        //        headSize, num, alpha, key, kvDim, q, scores);
        
        
        // 对于行主元key矩阵(num x headSize)，视为矩阵的转置
        HIPBLAS_CHECK(hipblasSgemv(
            blas_handle,
            HIPBLAS_OP_T,       // 转置操作
            headSize,           // 原矩阵列数 = 转置后行数
            num,                // 原矩阵行数 = 转置后列数
            &alpha,             // 缩放因子
            key,                // key矩阵指针
            kvDim,              // 行间距
            q,                  // 查询向量指针
            1,                  // 增量
            &beta,              // 累加因子
            scores,             // 输出分数指针
            1                   // 增量
        ));
         
        
        
        // 检查异步错误
        HIP_CHECK(hipGetLastError());
        // 不再显式同步，由调用者决定何时同步
    }



__global__ void weightedVKernel_reduction(float *headOutput, const float *value, const float *attentionScores,
                                          int pos, int kvDim, int headSize) {
    int tid = blockIdx.x;
    int lane = threadIdx.x;

    if (tid >= headSize) return;

    extern __shared__ float partialSum[];

    float sum = 0.0f;

    for (int t = lane; t <= pos; t += blockDim.x) {
        float v = value[t * kvDim + tid];
        float a = attentionScores[t];
        sum += a * v;
    }

    partialSum[lane] = sum;
    __syncthreads();

    // Block内归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            partialSum[lane] += partialSum[lane + stride];
        __syncthreads();
    }

    if (lane == 0)
        headOutput[tid] = partialSum[0];
}



// --- 修改后的函数 ---
// 注意：参数名已修改，明确表示它们是设备指针或值
// headOutput_d, value_d, attentionScores_d 现在必须是有效的 GPU 设备指针
void GPU_Backend::weightedV(float *headOutput_d,     // 指向 GPU 上已分配的输出缓冲区的指针
                         float *value_d,         // 指向 GPU 上已存在的 Value 缓存的指针
                         float *attentionScores_d, // 指向 GPU 上已存在的注意力分数数组的指针
                         int pos,                // 当前位置 (值)
                         int kvDim,              // Key/Value 维度 (值)
                         int headSize,           // Head 维度 (值)
                         hipStream_t stream)     // 指定的流，可选
{
    // --- 1. 移除内部 GPU 内存分配 ---
    // 不再需要: hipMalloc(&d_value, valueSize);
    // 不再需要: hipMalloc(&d_attentionScores, scoreSize);
    // 不再需要: hipMalloc(&d_headOutput, outputSize);
    // 我们假设传入的 headOutput_d, value_d, attentionScores_d 已经是有效的设备指针
    // 如果未指定流，则使用类的默认流
    hipStream_t useStream = stream ? stream : this->stream;

    // --- 2. 移除主机到设备的数据传输 ---
    // 不再需要: hipMemcpy(d_value, value, valueSize, hipMemcpyHostToDevice);
    // 不再需要: hipMemcpy(d_attentionScores, attentionScores, scoreSize, hipMemcpyHostToDevice);
    // 假设数据已在 GPU 上

    // --- 3. 配置并启动 GPU 内核 (保持这部分逻辑) ---
    int blockSize = 256; // 或者根据你的 GPU 和内核进行优化调整
    // 这个网格大小的计算 (numBlocks = headSize) 看起来特定于你的 reduction 内核
    // 如果你的内核是每个线程计算输出的一个元素，则计算方式不同
    // 保持原始逻辑，但需确保它与 weightedVKernel_reduction 的实现匹配
    int numBlocks = headSize;
    // 如果内核需要共享内存进行归约，则需要计算
    size_t sharedMemSize = (blockSize > 0) ? blockSize * sizeof(float) : 0;

    // 启动 GPU 内核，使用指定的流
    hipLaunchKernelGGL(weightedVKernel_reduction, // 内核函数名
                       dim3(numBlocks),           // 网格维度
                       dim3(blockSize),           // 块维度
                       sharedMemSize,             // 每个块的共享内存大小
                       useStream,                 // 使用指定的流
                       // 内核参数:
                       headOutput_d,         // 使用传入的设备指针
                       value_d,              // 使用传入的设备指针
                       attentionScores_d,    // 使用传入的设备指针
                       pos,                  // 传递值
                       kvDim,                // 传递值
                       headSize);            // 传递值

    // 检查内核启动是否有错误 (推荐保留)
    HIP_CHECK(hipGetLastError());
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
    // 检查内存大小
    size_t q_size = GPU_Backend::NUM_HEADS * GPU_Backend::HEAD_SIZE * sizeof(float);
    size_t k_cache_size = GPU_Backend::MAX_SEQ_LEN * GPU_Backend::NUM_HEADS * GPU_Backend::HEAD_SIZE * sizeof(float);
    size_t scores_size = GPU_Backend::NUM_HEADS * GPU_Backend::MAX_SEQ_LEN * sizeof(float);
    
    // 检查内存分配是否足够
    if (q_size > 3072 || k_cache_size > 37748736 || scores_size > 49152) {
        fprintf(stderr, "Error: Memory allocation too small for the operation\n");
        return;
    }
    
    // 使用指定的流或默认流
    hipStream_t useStream = stream ? stream : this->stream;
    
    // 计算QK点积 - 使用改进的内核
    dim3 block_qk(GPU_Backend::BLOCK_SIZE);
    dim3 grid_qk((GPU_Backend::NUM_HEADS * seq_len + GPU_Backend::BLOCK_SIZE - 1) / GPU_Backend::BLOCK_SIZE);
    compute_qk_kernel<<<grid_qk, block_qk, 0, useStream>>>(q, k_cache, scores, seq_len, GPU_Backend::HEAD_SIZE);
    
    // 添加显式同步点，确保QK计算完成
    HIP_CHECK(hipStreamSynchronize(useStream));
    
    // 计算softmax - 使用改进的内核，确保结果一致性
    dim3 block_softmax(GPU_Backend::BLOCK_SIZE);
    dim3 grid_softmax(GPU_Backend::NUM_HEADS);
    // 需要为两个数组分配共享内存：最大值和总和
    size_t softmax_shared_mem = 2 * GPU_Backend::BLOCK_SIZE * sizeof(float);
    compute_softmax_kernel<<<grid_softmax, block_softmax, softmax_shared_mem, useStream>>>(scores, attn, seq_len);
    
    // 添加显式同步点，确保softmax计算完成
    HIP_CHECK(hipStreamSynchronize(useStream));
    
    // 计算最终输出 - 使用改进的内核
    dim3 block_output(GPU_Backend::BLOCK_SIZE);
    dim3 grid_output(GPU_Backend::NUM_HEADS);
    // 不需要共享内存
    compute_output_kernel<<<grid_output, block_output, 0, useStream>>>(attn, v_cache, output, seq_len, GPU_Backend::HEAD_SIZE);
    
    // 等待所有操作完成
    HIP_CHECK(hipStreamSynchronize(useStream));
}

__global__ void compute_qk_kernel(float* q, float* k_cache, float* scores, int seq_len, int head_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = tid / seq_len;
    int pos = tid % seq_len;
    
    if (head_idx >= GPU_Backend::NUM_HEADS || pos >= seq_len) return;
    
    // 使用double精度进行计算，减少累计误差
    double score = 0.0;
    for (int i = 0; i < head_size; i++) {
        score += (double)q[head_idx * head_size + i] * (double)k_cache[pos * GPU_Backend::NUM_HEADS * head_size + head_idx * head_size + i];
    }
    // 使用double精度的sqrt，然后再转回float
    score /= sqrt((double)head_size);
    scores[head_idx * seq_len + pos] = (float)score;
}

__global__ void compute_softmax_kernel(float* scores, float* attn, int seq_len) {
    // 使用共享内存和同步原语改进的softmax计算
    extern __shared__ float s_data[];
    float* s_max = s_data;                   // 用于存储最大值
    float* s_sum = &s_data[blockDim.x];      // 用于存储和
    
    int head_idx = blockIdx.x;
    if (head_idx >= GPU_Backend::NUM_HEADS) return;
    
    // 每个线程初始化为最小值
    float thread_max = -INFINITY;
    
    // 第一阶段：找出最大值
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        thread_max = max(thread_max, scores[head_idx * seq_len + i]);
    }
    
    // 共享内存中存储线程的最大值
    s_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // 规约找出块内最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // 现在s_max[0]包含块内的最大值
    float max_val = s_max[0];
    __syncthreads();
    
    // 第二阶段：计算指数和总和
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float exp_val = expf(scores[head_idx * seq_len + i] - max_val);
        attn[head_idx * seq_len + i] = exp_val; // 存储中间结果
        thread_sum += exp_val;
    }
    
    // 在共享内存中存储线程的部分和
    s_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // 规约计算总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 现在s_sum[0]包含总和
    float sum = s_sum[0];
    __syncthreads();
    
    // 第三阶段：归一化
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        // 确保和不为零
        if (sum > 0.0f) {
            attn[head_idx * seq_len + i] /= sum;
        } else {
            // 如果和为零，均匀分布注意力
            attn[head_idx * seq_len + i] = 1.0f / seq_len;
        }
    }
}

__global__ void compute_output_kernel(float* attn, float* v_cache, float* output, int seq_len, int head_size) {
    // 不需要共享内存，直接使用寄存器进行累加
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (head_idx >= GPU_Backend::NUM_HEADS || tid >= head_size) return;
    
    // 每个线程计算一个输出元素
    double sum = 0.0;  // 使用double提高精度
    
    for (int i = 0; i < seq_len; i++) {
        sum += (double)attn[head_idx * seq_len + i] * 
               (double)v_cache[i * GPU_Backend::NUM_HEADS * head_size + head_idx * head_size + tid];
    }
    
    // 直接写入输出
    output[head_idx * head_size + tid] = (float)sum;
}


