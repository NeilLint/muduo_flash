#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric> // for std::iota, std::inner_product
#include <limits>  // for numeric_limits
#include <cassert>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// --- 检查宏 (假设已定义) ---
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
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif





// --- GPU 后端类 (假设已定义，包含 dot_gpu, dot_gpu_to_host, gemvQkSeq_gpu) ---
class GPU_Backend {
private:
    hipblasHandle_t blas_handle;
public:
    GPU_Backend() { HIPBLAS_CHECK(hipblasCreate(&blas_handle)); }
    ~GPU_Backend() { if (blas_handle) HIPBLAS_CHECK(hipblasDestroy(blas_handle)); }

    // --- Softmax (GPU) - 需要内核实现 ---
    // 假设 softmax_kernel_inplace 内核定义在别处
    // __global__ void softmax_kernel_inplace(float* data, int size) { /* ... kernel logic ... */ }

    void softmax_gpu(float* d_data, int size) {
        if (d_data == nullptr || size <= 0) { /* ... error handling ... */ return; }
        int block_size = 256;
        dim3 gridDim(1);
        dim3 blockDim(block_size);
        size_t shared_mem_size = block_size * sizeof(float); // 根据内核需要调整
        // *** 注意：需要实际的 softmax_kernel_inplace 内核才能运行 ***
        // hipLaunchKernelGGL(softmax_kernel_inplace, gridDim, blockDim, shared_mem_size, 0, d_data, size);
        // HIP_CHECK(hipGetLastError());
        printf("警告: softmax_gpu 未执行，需要 softmax_kernel_inplace 内核定义。\n");
    }

    // --- Dot Product (GPU) ---
    void dot_gpu(float* d_y, const float* d_x1, const float* d_x2, int dim) {
        if (dim <= 0 || d_x1 == nullptr || d_x2 == nullptr || d_y == nullptr) { /* ... error handling ... */ return; }
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_DEVICE));
        HIPBLAS_CHECK(hipblasSdot(blas_handle, dim, d_x1, 1, d_x2, 1, d_y));
    }

    void dot_gpu_to_host(float* h_y, const float* d_x1, const float* d_x2, int dim) {
        if (dim <= 0 || d_x1 == nullptr || d_x2 == nullptr || h_y == nullptr) { /* ... error handling ... */ return; }
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));
        HIPBLAS_CHECK(hipblasSdot(blas_handle, dim, d_x1, 1, d_x2, 1, h_y));
    }

    // --- GEMV Q*K (GPU) ---
     void gemvQkSeq_gpu(float* d_attentionScores, const float* d_q, const float* d_key,
                       int pos, int kvDim, int headSize) {
        if (pos < 0) return;
        if (d_attentionScores == nullptr || d_q == nullptr || d_key == nullptr || headSize <= 0 || kvDim <= 0) { /* ... error handling ... */ return; }

        int num_keys = pos + 1;
        const float alpha = 1.0f / sqrtf((float)headSize);
        const float beta = 0.0f;

        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));
        HIPBLAS_CHECK(hipblasSgemv(blas_handle, HIPBLAS_OP_N,
                                   num_keys, headSize,
                                   &alpha, d_key, kvDim,
                                   d_q, 1,
                                   &beta, d_attentionScores, 1));
    }
};

// --- CPU 参考实现 ---

// CPU Softmax
void softmax_cpu(float* x, int size) {
    if (x == nullptr || size <= 0) return;
    // 找到最大值以提高数值稳定性
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // 计算 exp(x_i - max_val) 并求和
    float sum = 0.0f;
    std::vector<float> exp_values(size);
    for (int i = 0; i < size; ++i) {
        exp_values[i] = expf(x[i] - max_val);
        sum += exp_values[i];
    }
    // 归一化
    if (sum == 0) sum = 1.0f; // 避免除以零
    for (int i = 0; i < size; ++i) {
        x[i] = exp_values[i] / sum;
    }
}

// CPU Dot Product
void dot_cpu(float *y, const float *x1, const float *x2, int dim) {
    if (dim <= 0 || x1 == nullptr || x2 == nullptr || y == nullptr) {
        if(y) *y = 0.0f; // 或者设定一个错误值
        return;
    }
    double result = 0.0; // 使用 double 提高精度
    for (int i = 0; i < dim; ++i) {
        result += (double)x1[i] * x2[i];
    }
    *y = (float)result;
}

// CPU GEMV Q*K (使用 dot_cpu)
void gemvQkSeq_cpu_loop(float *h_attentionScores, const float *h_q, const float *h_key,
                       int pos, int kvDim, int headSize)
{
    if (pos < 0) return;
    for (int timestep = 0; timestep <= pos; timestep++) {
        const float* k = h_key + timestep * kvDim; // 指向当前 key
        float score = 0.0f;
        dot_cpu(&score, h_q, k, headSize); // 调用 CPU 点积
        score /= sqrtf((float)headSize);
        h_attentionScores[timestep] = score;
    }
}

// --- 比较函数 ---
bool compare_vectors(const float* ref, const float* test, size_t n, float tolerance = 1e-5f) {
    double max_diff = 0.0;
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs((double)ref[i] - (double)test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        // 可以添加相对误差检查，但对于这些操作，绝对误差通常足够
        if (diff > tolerance) {
            std::cerr << "不匹配! Index: " << i << ", CPU ref: " << ref[i]
                      << ", GPU test: " << test[i] << ", Diff: " << diff << std::endl;
            match = false;
            // 可以选择在这里提前返回 return false;
        }
    }
    std::cout << "最大误差: " << max_diff << std::endl;
    return match;
}


// --- 主验证函数 ---
int main() {
    srand(time(NULL)); // 初始化随机种子

    GPU_Backend backend; // 创建后端实例 (包含 hipBLAS handle)

    // --- 1. 验证 Dot Product ---
    std::cout << "--- 验证 Dot Product ---" << std::endl;
    {
        int dim = 1024 * 1024; // 向量维度
        size_t vectorBytes = dim * sizeof(float);
        size_t resultBytes = sizeof(float);

        // 分配主机内存
        std::vector<float> h_x1(dim), h_x2(dim);
        float h_y_cpu = 0.0f;
        float h_y_gpu = 0.0f;         // 用于接收 d_y_gpu 的结果
        float h_y_gpu_direct = 0.0f; // 用于接收 dot_gpu_to_host 的结果

        // 生成随机数据
        for(int i=0; i<dim; ++i) {
            h_x1[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
            h_x2[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        // 分配设备内存
        float *d_x1 = nullptr, *d_x2 = nullptr, *d_y_gpu = nullptr;
        HIP_CHECK(hipMalloc(&d_x1, vectorBytes));
        HIP_CHECK(hipMalloc(&d_x2, vectorBytes));
        HIP_CHECK(hipMalloc(&d_y_gpu, resultBytes)); // 给 GPU 结果分配空间

        // 拷贝输入数据到设备
        HIP_CHECK(hipMemcpy(d_x1, h_x1.data(), vectorBytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_x2, h_x2.data(), vectorBytes, hipMemcpyHostToDevice));

        // 计算 CPU 参考结果
        dot_cpu(&h_y_cpu, h_x1.data(), h_x2.data(), dim);

        // 计算 GPU 结果 (写入设备内存)
        backend.dot_gpu(d_y_gpu, d_x1, d_x2, dim);
        // 将 GPU 结果拷贝回主机
        HIP_CHECK(hipMemcpy(&h_y_gpu, d_y_gpu, resultBytes, hipMemcpyDeviceToHost));

        // 计算 GPU 结果 (直接写入主机内存)
        backend.dot_gpu_to_host(&h_y_gpu_direct, d_x1, d_x2, dim);

        // 比较结果
        std::cout << "CPU 参考结果: " << h_y_cpu << std::endl;
        std::cout << "GPU (Device->Host Copy) 结果: " << h_y_gpu << std::endl;
        std::cout << "GPU (Direct to Host) 结果: " << h_y_gpu_direct << std::endl;

        bool match1 = compare_vectors(&h_y_cpu, &h_y_gpu, 1,1e-4);
        bool match2 = compare_vectors(&h_y_cpu, &h_y_gpu_direct, 1,1e-4);
        std::cout << "Dot Product (Device->Host) 验证: " << (match1 ? "通过" : "失败") << std::endl;
        std::cout << "Dot Product (Direct to Host) 验证: " << (match2 ? "通过" : "失败") << std::endl;

        // 释放设备内存
        HIP_CHECK(hipFree(d_x1));
        HIP_CHECK(hipFree(d_x2));
        HIP_CHECK(hipFree(d_y_gpu));
    }
    std::cout << std::endl;


    // --- 2. 验证 GEMV Q*K ---
    std::cout << "--- 验证 GEMV Q*K ---" << std::endl;
    {
        int headSize = 64;  // K/Q 向量维度
        int kvDim = 64;    // Key Cache 中键之间的步长 (可能大于 headSize)
        int maxPos = 50;    // 模拟的最大位置 (0 到 maxPos)
        int num_scores = maxPos + 1;
        size_t qBytes = headSize * sizeof(float);
        size_t keyCacheBytes = (maxPos + 1) * kvDim * sizeof(float); // 缓存大小
        size_t scoresBytes = num_scores * sizeof(float);

        // 分配主机内存
        std::vector<float> h_q(headSize);
        std::vector<float> h_key( (maxPos + 1) * kvDim ); // 分配足够大的 key cache
        std::vector<float> h_scores_cpu(num_scores);
        std::vector<float> h_scores_gpu(num_scores);

        // 生成随机数据
        for(int i=0; i<headSize; ++i) h_q[i] = (float)rand() / RAND_MAX;
        for(size_t i=0; i<h_key.size(); ++i) h_key[i] = (float)rand() / RAND_MAX;

        // 分配设备内存
        float *d_q = nullptr, *d_key = nullptr, *d_scores_gpu = nullptr;
        HIP_CHECK(hipMalloc(&d_q, qBytes));
        HIP_CHECK(hipMalloc(&d_key, keyCacheBytes));
        HIP_CHECK(hipMalloc(&d_scores_gpu, scoresBytes));

        // 拷贝输入数据到设备
        HIP_CHECK(hipMemcpy(d_q, h_q.data(), qBytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_key, h_key.data(), keyCacheBytes, hipMemcpyHostToDevice));

        // 计算 CPU 参考结果
        gemvQkSeq_cpu_loop(h_scores_cpu.data(), h_q.data(), h_key.data(), maxPos, kvDim, headSize);

        // 计算 GPU 结果
        backend.gemvQkSeq_gpu(d_scores_gpu, d_q, d_key, maxPos, kvDim, headSize);
        // 将 GPU 结果拷贝回主机
        HIP_CHECK(hipMemcpy(h_scores_gpu.data(), d_scores_gpu, scoresBytes, hipMemcpyDeviceToHost));

        // 比较结果
        std::cout << "比较 CPU 和 GPU 的 GEMV Q*K 结果 (比较前 " << num_scores << " 个分数):" << std::endl;
        bool gemv_match = compare_vectors(h_scores_cpu.data(), h_scores_gpu.data(), num_scores, 1e-4f); // 可能需要稍大的容差
        std::cout << "GEMV Q*K 验证: " << (gemv_match ? "通过" : "失败") << std::endl;

        // 释放设备内存
        HIP_CHECK(hipFree(d_q));
        HIP_CHECK(hipFree(d_key));
        HIP_CHECK(hipFree(d_scores_gpu));
    }
     std::cout << std::endl;


    // --- 3. 验证 Softmax (需要内核实现) ---
    std::cout << "--- 验证 Softmax (需要 softmax_kernel_inplace 内核) ---" << std::endl;
    {
        int size = 1024;
        size_t dataBytes = size * sizeof(float);

        // 分配主机内存
        std::vector<float> h_data_cpu(size);
        std::vector<float> h_data_gpu(size); // 用于 GPU 结果
        std::vector<float> h_data_orig(size); // 保存原始数据用于 GPU 输入

        // 生成随机数据
        for(int i=0; i<size; ++i) {
             h_data_cpu[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // [-5, 5]
        }
        h_data_orig = h_data_cpu; // 复制一份用于 GPU

        // 计算 CPU 参考结果 (原地修改 h_data_cpu)
        softmax_cpu(h_data_cpu.data(), size);

        // *** GPU 部分需要内核才能运行 ***
        /*
        // 分配设备内存
        float *d_data = nullptr;
        HIP_CHECK(hipMalloc(&d_data, dataBytes));

        // 拷贝原始数据到设备
        HIP_CHECK(hipMemcpy(d_data, h_data_orig.data(), dataBytes, hipMemcpyHostToDevice));

        // 调用 GPU Softmax (原地修改 d_data)
        backend.softmax_gpu(d_data, size);

        // 将 GPU 结果拷贝回主机
        HIP_CHECK(hipMemcpy(h_data_gpu.data(), d_data, dataBytes, hipMemcpyDeviceToHost));

        // 比较结果
        std::cout << "比较 CPU 和 GPU 的 Softmax 结果:" << std::endl;
        bool softmax_match = compare_vectors(h_data_cpu.data(), h_data_gpu.data(), size, 1e-5f);
        std::cout << "Softmax 验证: " << (softmax_match ? "通过" : "失败") << std::endl;

        // 释放设备内存
        HIP_CHECK(hipFree(d_data));
        */
        std::cout << "Softmax 验证部分已跳过，需要提供 softmax_kernel_inplace 内核实现。" << std::endl;

    }
    std::cout << std::endl;

    std::cout << "验证完成。" << std::endl;

    return 0;
}