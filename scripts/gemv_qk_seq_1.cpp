#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h> // for exit

// --- 检查宏 (请确保它们已正确定义或替换为你的错误检查机制) ---
#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Failed: HIP error %s:%d '%s'\n", \
                __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef HIPBLAS_CHECK
#define HIPBLAS_CHECK(cmd) do { \
    hipblasStatus_t status = cmd; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "Failed: hipBLAS error %s:%d, status = %d\n", \
                __FILE__, __LINE__, status); \
        /* Consider adding hipblasGetErrorString if available */ \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// --- GPU 后端类定义 ---
class GPU_Backend {
private:
    hipblasHandle_t blas_handle;
public:
    GPU_Backend() {
        HIPBLAS_CHECK(hipblasCreate(&blas_handle));
        std::cout << "hipBLAS handle created." << std::endl;
    }
    ~GPU_Backend() {
        if (blas_handle) {
            HIPBLAS_CHECK(hipblasDestroy(blas_handle));
            std::cout << "hipBLAS handle destroyed." << std::endl;
        }
    }

    // --- 高效 GPU 版本 gemvQkSeq (使用 hipblasSgemv) ---
    // !!! 假设: query, key, attentionScores 都是有效的设备指针 !!!
    void gemvQkSeq_gpu_efficient(
            float* d_attentionScores, // 输出: 设备指针
            const float* d_q,         // 输入: 设备指针
            const float* d_key,       // 输入: 设备指针 (指向相关 key 序列的开始)
            int pos,                  // 当前位置索引 (0-based)
            int kvDim,                // 内存中 key 向量之间的步长 (lda)
            int headSize)             // Q/K 向量维度 (n)
    {
        if (pos < 0) return; // 没有历史记录
        if (d_attentionScores == nullptr || d_q == nullptr || d_key == nullptr || headSize <= 0 || kvDim <= 0) {
             fprintf(stderr, "错误：gemvQkSeq_gpu_efficient 输入无效。\n");
             // 添加更健壮的错误处理
             return;
        }

        int num_keys = pos + 1; // 矩阵 A 的行数 (m)
        const float alpha = 1.0f / sqrtf((float)headSize); // 缩放因子
        const float beta = 0.0f;                         // 覆盖输出

        // 为主机标量 alpha, beta 设置指针模式
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));


        HIPBLAS_CHECK(hipblasSgemv(
                                blas_handle,
                                HIPBLAS_OP_T,          // transpose A
                                headSize,        // number of rows in A^T == headSize
                                num_keys,        // number of cols in A^T == num_keys
                                &alpha,
                                d_key,                 // pointer to your row‑major blob
                                kvDim,                 // leading dimension still = distance between rows in memory
                                d_q,                   // x
                                1,                     // incx
                                &beta,
                                d_attentionScores,     // y
                                1                      // incy
                            ) );
    }
}; // end class GPU_Backend


// --- CPU 版本 gemvQkSeq (模拟 hipblasSgemv 计算) ---
// !!! 假设: query, key, attentionScores 都是有效的主机指针 !!!
void gemvQkSeq_cpu_gemv(
        float* h_attentionScores, // 输出: 主机指针
        const float* h_q,         // 输入: 主机指针
        const float* h_key,       // 输入: 主机指针 (指向相关 key 序列的开始)
        int pos,                  // 当前位置索引 (0-based)
        int kvDim,                // 内存中 key 向量之间的步长 (lda)
        int headSize)             // Q/K 向量维度 (n)
{
    if (pos < 0) return;
    if (h_attentionScores == nullptr || h_q == nullptr || h_key == nullptr || headSize <= 0 || kvDim <= 0) {
         fprintf(stderr, "错误：gemvQkSeq_cpu_gemv 输入无效。\n");
         return;
    }

    int num_keys = pos + 1; // 结果向量 y 的大小，也是矩阵 A 的行数 m
    const float alpha = 1.0f / sqrtf((float)headSize);
    // beta = 0.0f，所以我们直接计算 alpha * A * x

    // 循环遍历输出向量 y (attentionScores) 的每一个元素 (对应矩阵 A 的每一行)
    for (int t = 0; t < num_keys; ++t) { // t 从 0 到 pos
        // 定位到矩阵 A 的第 t 行 (即第 t 个 key 向量) 的起始位置
        const float* k_t = h_key + (size_t)t * kvDim; // 使用 size_t 避免溢出

        // 计算矩阵 A 的第 t 行与向量 x (h_q) 的点积
        // 注意：只使用 k_t 的前 headSize (n) 个元素
        double dot_sum = 0.0; // 使用 double 提高精度
        for (int j = 0; j < headSize; ++j) {
            dot_sum += (double)k_t[j] * h_q[j];
        }

        // 计算最终结果: y[t] = alpha * dot_product(A[t,:], x) + beta * y[t] (beta=0)
        h_attentionScores[t] = alpha * (float)dot_sum;
    }
}


// --- 比较函数 ---
// 返回 false 如果任何元素差异大于 tolerance
bool compare_vectors(const float* ref, const float* test, size_t n, float tolerance = 1e-5f, bool verbose = true) {
    if (n == 0) return true;
    double max_diff = 0.0;
    size_t diff_count = 0;
    size_t first_mismatch_idx = 0;
    bool match = true;

    for (size_t i = 0; i < n; ++i) {
        bool ref_is_finite = std::isfinite(ref[i]);
        bool test_is_finite = std::isfinite(test[i]);

        if (ref_is_finite && test_is_finite) {
            double diff = std::abs((double)ref[i] - (double)test[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tolerance) {
                if (match && verbose) {
                   std::cerr << "不匹配! Index: " << i << ", CPU ref: " << ref[i]
                             << ", GPU test: " << test[i] << ", Diff: " << diff
                             << ", Tolerance: " << tolerance << std::endl;
                   first_mismatch_idx = i;
                }
                match = false;
                diff_count++;
            }
        } else if (ref[i] != test[i]) {
             if (match && verbose) {
                std::cerr << "不匹配! Index: " << i << ", CPU ref: " << ref[i]
                          << ", GPU test: " << test[i] << " (非有限数值不匹配)" << std::endl;
                first_mismatch_idx = i;
             }
             match = false;
             diff_count++;
        }
    }

    if (verbose) {
        std::cout << "比较完成. 总元素: " << n << ". 最大误差: " << max_diff;
        if (!match) {
            std::cout << ". 不匹配数量: " << diff_count << " (第一个在 index " << first_mismatch_idx << ")";
        }
         std::cout << std::endl;
    }
    return match;
}


// --- 测试 Demo ---
int main() {
    srand(1234); // 固定随机种子以便复现

    // --- 参数定义 ---
    int headSize = 64;       // 计算维度 (n)
    int kvDim = 64*12;          // Key Cache 内存步长 (lda), 特意设为不等于 headSize
    int pos = 50;            // 当前位置 (最大时间步索引)
    int num_scores = pos + 1; // 输出分数数量 (m)
    std::cout << "测试参数: headSize=" << headSize << ", kvDim=" << kvDim << ", pos=" << pos << std::endl;

    // --- 实例化后端 ---
    GPU_Backend backend;

    // --- 主机内存分配 ---
    size_t q_size = headSize;
    // 注意：key 缓存需要能容纳到 pos 位置的所有 key，每个 key 在内存中占据 kvDim 空间
    size_t key_cache_size = (size_t)(pos + 1) * kvDim;
    size_t scores_size = num_scores;

    std::vector<float> h_q(q_size);
    std::vector<float> h_key(key_cache_size);
    std::vector<float> h_scores_cpu(scores_size); // CPU 版本结果
    std::vector<float> h_scores_gpu(scores_size); // 用于接收 GPU 结果

    // --- 生成主机随机数据 ---
    std::cout << "生成主机数据..." << std::endl;
    for(size_t i = 0; i < q_size; ++i) h_q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for(size_t i = 0; i < key_cache_size; ++i) h_key[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    // 将 GPU 结果向量初始化为 NaN，以便检查是否所有值都被写入
    std::fill(h_scores_gpu.begin(), h_scores_gpu.end(), std::numeric_limits<float>::quiet_NaN());

    // --- 设备内存分配 ---
    std::cout << "分配设备内存..." << std::endl;
    float *d_q = nullptr, *d_key = nullptr, *d_scores_gpu = nullptr;
    size_t qBytes = q_size * sizeof(float);
    size_t keyBytes = key_cache_size * sizeof(float);
    size_t scoresBytes = scores_size * sizeof(float);

    HIP_CHECK(hipMalloc(&d_q, qBytes));
    HIP_CHECK(hipMalloc(&d_key, keyBytes));
    HIP_CHECK(hipMalloc(&d_scores_gpu, scoresBytes));

    // --- 拷贝数据 H -> D ---
    std::cout << "拷贝数据 Host -> Device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_q, h_q.data(), qBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_key, h_key.data(), keyBytes, hipMemcpyHostToDevice));

    // --- 执行 CPU 参考版本 ---
    std::cout << "执行 CPU 版本 (gemvQkSeq_cpu_gemv)..." << std::endl;
    gemvQkSeq_cpu_gemv(h_scores_cpu.data(), h_q.data(), h_key.data(), pos, kvDim, headSize);
    std::cout << "CPU 版本完成." << std::endl;

    // --- 执行 GPU 版本 ---
    std::cout << "执行 GPU 版本 (gemvQkSeq_gpu_efficient)..." << std::endl;
    // !!! 传递设备指针给 GPU 函数 !!!
    backend.gemvQkSeq_gpu_efficient(d_scores_gpu, d_q, d_key, pos, kvDim, headSize);
    HIP_CHECK(hipDeviceSynchronize()); // 确保 GPU 计算完成
    std::cout << "GPU 版本完成." << std::endl;

    // --- 拷贝结果 D -> H ---
    std::cout << "拷贝结果 Device -> Host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_scores_gpu.data(), d_scores_gpu, scoresBytes, hipMemcpyDeviceToHost));

    // --- 比较结果 ---
    std::cout << "比较 CPU 和 GPU 结果..." << std::endl;
    // 由于浮点计算差异，通常需要设置一个容差
    // 对于 SGEMV，1e-5f 或 1e-4f 通常是合理的起点
    float tolerance = 1e-4f;
    bool match = compare_vectors(h_scores_cpu.data(), h_scores_gpu.data(), scores_size, tolerance);

    std::cout << "验证结果: " << (match ? "通过" : "失败") << " (使用容差: " << tolerance << ")" << std::endl;

    // --- 打印部分结果 (可选) ---
    if (!match) {
        std::cout << "打印部分结果以供检查 (最多 10 个):" << std::endl;
        int print_count = std::min((int)scores_size, 10);
        printf("Index |   CPU Result   |   GPU Result   |    Difference\n");
        printf("------|----------------|----------------|----------------\n");
        for(int i = 0; i < print_count; ++i) {
            printf("%5d | %14.7f | %14.7f | %14.7e\n",
                   i, h_scores_cpu[i], h_scores_gpu[i], fabsf(h_scores_cpu[i] - h_scores_gpu[i]));
        }
    }


    // --- 清理设备内存 ---
    std::cout << "释放设备内存..." << std::endl;
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_key));
    HIP_CHECK(hipFree(d_scores_gpu));

    std::cout << "Demo 完成." << std::endl;

    return 0;
}