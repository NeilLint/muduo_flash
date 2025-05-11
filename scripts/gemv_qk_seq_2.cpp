#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cassert>
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

// --- CBackend GPU 实现 ---
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

    // GPU 版 gemvQkSeq
    void gemvQkSeq(float *q, float *key, float *scores, int pos, int kvDim, int headSize) {
        if (pos < 0 || !q || !key || !scores || kvDim<=0 || headSize<=0) return;
        int num = pos + 1;
        const float alpha = 1.0f / std::sqrt((float)headSize);
        const float beta  = 0.0f;
        HIPBLAS_CHECK( hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST) );
        HIPBLAS_CHECK( hipblasSgemv(
            handle,
            HIPBLAS_OP_T,        // 视 row-major (num×headSize) 为转置
            headSize,            // rows of A^T
            num,                 // cols of A^T
            &alpha,
            key,                 // device pointer
            kvDim,               // row stride
            q,
            1,
            &beta,
            scores,
            1
        ));
    }
};

// --- CPU 参考实现 ---
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

int main() {
    // 参数
    const int headSize = 64;
    const int kvDim     = headSize * 12;  // 故意不等于 headSize
    const int pos       = 50;
    const int num      = pos + 1;

    std::cout << "Test parameters: headSize=" << headSize
              << ", kvDim=" << kvDim
              << ", pos=" << pos << std::endl;

    // 主机内存
    std::vector<float> h_q(headSize), h_key((size_t)num * kvDim);
    std::vector<float> h_scores_cpu(num), h_scores_gpu(num, std::numeric_limits<float>::quiet_NaN());

    // 随机初始化
    srand(2025);
    for (auto &v : h_q)  v = (float(rand())/RAND_MAX)*2 - 1;
    for (auto &v : h_key)v = (float(rand())/RAND_MAX)*2 - 1;

    // 分配设备内存
    float *d_q=nullptr, *d_key=nullptr, *d_scores=nullptr;
    HIP_CHECK( hipMalloc(&d_q,   headSize * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_key, (size_t)num*kvDim * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_scores, num * sizeof(float)) );
    
    // 拷贝到设备
    HIP_CHECK( hipMemcpy(d_q,    h_q.data(),   headSize*sizeof(float), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_key,  h_key.data(), (size_t)num*kvDim*sizeof(float), hipMemcpyHostToDevice) );

    // CPU 计算
    gemvQkSeq_cpu(h_q.data(), h_key.data(), h_scores_cpu.data(), pos, kvDim, headSize);

    // GPU 计算
    CBackend backend;
    backend.gemvQkSeq(d_q, d_key, d_scores, pos, kvDim, headSize);
    HIP_CHECK( hipDeviceSynchronize() );

    // 拷贝回主机
    HIP_CHECK( hipMemcpy(h_scores_gpu.data(), d_scores, num*sizeof(float), hipMemcpyDeviceToHost) );

    // 对比结果
    double max_diff = 0.0;
    int    bad = 0;
    for (int i = 0; i < num; i++) {
        double d = std::abs(h_scores_cpu[i] - h_scores_gpu[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-4) bad++;
    }
    std::cout << "Comparison: max error = " << max_diff
              << ", mismatches (>1e-4) = " << bad << "/" << num << std::endl;

    if (bad > 0) {
        std::cout << "Index |   CPU    |   GPU    | Diff\n"
                  << "----------------------------------\n";
        for (int i = 0; i < std::min(num, 10); i++) {
            std::printf("%5d | %8.5f | %8.5f | %8.2e\n",
                        i, h_scores_cpu[i], h_scores_gpu[i],
                        std::abs(h_scores_cpu[i] - h_scores_gpu[i]));
        }
    } else {
        std::cout << "✅ GPU result matches CPU reference within tolerance.\n";
    }

    // 释放
    HIP_CHECK( hipFree(d_q) );
    HIP_CHECK( hipFree(d_key) );
    HIP_CHECK( hipFree(d_scores) );
    return 0;
}
