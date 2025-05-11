#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// --- 错误检查宏 ---
#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "HIP error " << hipGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef HIPBLAS_CHECK
#define HIPBLAS_CHECK(cmd) do { \
    hipblasStatus_t s = cmd; \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLAS error " << s \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// --- CBackend GPU 实现 (同上) ---
class CBackend {
private:
    hipblasHandle_t handle;
public:
    CBackend() {
        HIPBLAS_CHECK( hipblasCreate(&handle) );
    }
    ~CBackend() {
        HIPBLAS_CHECK( hipblasDestroy(handle) );
    }
    void gemvQkSeq(float *q, float *key, float *scores, int pos, int kvDim, int headSize) {
        if (pos < 0 || !q || !key || !scores || kvDim<=0 || headSize<=0) return;
        int num = pos + 1;
        const float alpha = 1.0f / std::sqrt((float)headSize);
        const float beta  = 0.0f;
        HIPBLAS_CHECK( hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST) );
        HIPBLAS_CHECK( hipblasSgemv(
            handle,
            HIPBLAS_OP_T,
            headSize,
            num,
            &alpha,
            key,
            kvDim,
            q,
            1,
            &beta,
            scores,
            1
        ));
    }
};

// --- CPU 参考实现 (同上) ---
void gemvQkSeq_cpu(const float *q, const float *key, float *scores,
                   int pos, int kvDim, int headSize) {
    int num = pos + 1;
    float scale = 1.0f / std::sqrt((float)headSize);
    for (int t = 0; t < num; t++) {
        const float *k = key + (size_t)t * kvDim;
        double sum = 0.0;
        for (int j = 0; j < headSize; j++) {
            sum += (double)q[j] * k[j];
        }
        scores[t] = scale * (float)sum;
    }
}

// 比较并打印
void compare_print(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    int num = cpu.size();
    double max_diff = 0.0;
    int    bad = 0;
    for (int i = 0; i < num; i++) {
        double d = std::abs(cpu[i] - gpu[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-4) bad++;
    }
    std::cout << "  max error = " << max_diff
              << ", mismatches (>1e-4) = " << bad << "/" << num << std::endl;
    if (bad > 0) {
        std::cout << "Index |   CPU    |   GPU    | Diff\n"
                  << "----------------------------------\n";
        for (int i = 0; i < std::min(num, 10); i++) {
            std::printf("%5d | %8.5f | %8.5f | %8.2e\n",
                        i, cpu[i], gpu[i], std::abs(cpu[i] - gpu[i]));
        }
    }
}

int main() {
    const int headSize = 64;
    const int kvDim     = headSize * 12;
    const int pos       = 50;
    const int num       = pos + 1;

    std::cout << "Test parameters: headSize=" << headSize
              << ", kvDim=" << kvDim
              << ", pos=" << pos << "\n\n";

    // 扩大 h_key：在前面预留 headSize 空间用于“偏移”
    std::vector<float> h_q(headSize);
    std::vector<float> h_key(headSize + (size_t)num * kvDim);
    std::vector<float> h_scores_cpu(num), h_scores_gpu(num);

    // 初始化随机数据
    srand(2025);
    for (auto &v : h_q)  v = (float(rand())/RAND_MAX)*2 - 1;
    for (size_t i = 0; i < (size_t)num * kvDim; i++) {
        // 写入从 offset=headSize 开始的位置
        h_key[headSize + i] = (float(rand())/RAND_MAX)*2 - 1;
    }

    // 设备内存
    float *d_q=nullptr, *d_key=nullptr, *d_scores=nullptr;
    HIP_CHECK( hipMalloc(&d_q,    headSize * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_key,  (size_t)num*kvDim * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_scores, num * sizeof(float)) );

    // 拷贝 q
    HIP_CHECK( hipMemcpy(d_q, h_q.data(), headSize*sizeof(float), hipMemcpyHostToDevice) );
    // 拷贝 key：从 &h_key[headSize]
    HIP_CHECK( hipMemcpy(d_key,
                         h_key.data() + headSize,
                         (size_t)num*kvDim*sizeof(float),
                         hipMemcpyHostToDevice) );

    // --- 第一轮：使用未偏移的视图 (等同于基地址0) ---
    std::cout << "[Round 1] no extra offset:\n";
    // CPU
    gemvQkSeq_cpu(h_q.data(), h_key.data() + headSize, h_scores_cpu.data(),
                  pos, kvDim, headSize);
    // GPU
    CBackend backend;
    backend.gemvQkSeq(d_q, d_key, d_scores, pos, kvDim, headSize);
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipMemcpy(h_scores_gpu.data(), d_scores, num*sizeof(float), hipMemcpyDeviceToHost) );
    compare_print(h_scores_cpu, h_scores_gpu);

    // --- 第二轮：再向后偏移一个“头”距离（headSize floats） ---
    std::cout << "\n[Round 2] extra offset by headSize:\n";
    // 新的 base 地址在 host: h_key.data() + headSize*2
    const float* h_key_off = h_key.data() + headSize + headSize;
    // 在 device 上指针也偏移 headSize 元素：
    float* d_key_off = d_key + headSize;

    // CPU
    gemvQkSeq_cpu(h_q.data(), h_key_off, h_scores_cpu.data(),
                  pos, kvDim, headSize);
    // GPU
    backend.gemvQkSeq(d_q, d_key_off, d_scores, pos, kvDim, headSize);
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipMemcpy(h_scores_gpu.data(), d_scores, num*sizeof(float), hipMemcpyDeviceToHost) );
    compare_print(h_scores_cpu, h_scores_gpu);

    // 释放
    HIP_CHECK( hipFree(d_q) );
    HIP_CHECK( hipFree(d_key) );
    HIP_CHECK( hipFree(d_scores) );

    return 0;
}
