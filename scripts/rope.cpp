#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <hip/hip_runtime.h>

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

// --- GPU kernel（同实现） ---
__global__
void ropeEncoding_kernel(float* q, float* k,
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

// --- GPU Backend 声明 ---
class CBackend {
public:
    void ropeEncoding(float *q, float *k,
                      int headSize,
                      int position,
                      int dim,
                      int kvDim) {
        if (!q || !k || headSize <= 0 || dim <= 0 || kvDim < 0) return;
        int numPairs = (dim + 1) / 2;
        const int threads = 256;
        int blocks = (numPairs + threads - 1) / threads;
        hipLaunchKernelGGL(
            ropeEncoding_kernel,
            dim3(blocks),
            dim3(threads),
            0, 0,
            q, k, headSize, position, dim, kvDim
        );
        HIP_CHECK( hipGetLastError() );
        HIP_CHECK( hipDeviceSynchronize() );
    }
};

// --- CPU 串行参考实现 ---
void ropeEncoding_cpu(float *q, float *k,
                      int headSize,
                      int position,
                      int dim,
                      int kvDim) {
    for (int i = 0; i < dim; i+=2) {
        int headDim = i % headSize;
        float freq = 1.0f / powf(10000.0f, headDim / (float)headSize);
        float val = position * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kvDim ? 2 : 1; 
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? q : k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

int main() {
    const int headSize = 16;
    const int position = 7;
    const int dim      = 64;     // 向量总维度（偶数）
    const int kvDim    = 48;     // k 旋转截止位置

    std::cout << "Test ropeEncoding: headSize=" << headSize
              << ", position=" << position
              << ", dim=" << dim
              << ", kvDim=" << kvDim << std::endl;

    // 1. 主机数据初始化
    std::vector<float> h_q(dim), h_k(dim);
    srand(2025);
    for (int i = 0; i < dim; ++i) {
        h_q[i] = (float(rand())/RAND_MAX)*2 - 1;
        h_k[i] = (float(rand())/RAND_MAX)*2 - 1;
    }
    // 备份一份做 CPU 参考
    std::vector<float> q_cpu = h_q, k_cpu = h_k;

    // 2. 分配并拷贝到设备
    float *d_q = nullptr, *d_k = nullptr;
    HIP_CHECK( hipMalloc(&d_q, dim * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_k, dim * sizeof(float)) );
    HIP_CHECK( hipMemcpy(d_q, h_q.data(), dim * sizeof(float), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_k, h_k.data(), dim * sizeof(float), hipMemcpyHostToDevice) );

    // 3. CPU 计算
    ropeEncoding_cpu(q_cpu.data(), k_cpu.data(), headSize, position, dim, kvDim);

    // 4. GPU 计算
    CBackend backend;
    backend.ropeEncoding(d_q, d_k, headSize, position, dim, kvDim);

    // 5. 拷回并对比
    std::vector<float> q_gpu(dim), k_gpu(dim);
    HIP_CHECK( hipMemcpy(q_gpu.data(), d_q, dim * sizeof(float), hipMemcpyDeviceToHost) );
    HIP_CHECK( hipMemcpy(k_gpu.data(), d_k, dim * sizeof(float), hipMemcpyDeviceToHost) );

    // 验证
    double max_err = 0.0;
    int mismatches = 0;
    const float tol = 1e-6f;
    for (int i = 0; i < dim; ++i) {
        double err_q = std::abs(q_cpu[i] - q_gpu[i]);
        double err_k = std::abs(k_cpu[i] - k_gpu[i]);
        max_err = std::max({max_err, err_q, err_k});
        if (err_q > tol || err_k > tol) {
            if (mismatches < 10) {
                std::printf("Mis @%2d: CPU_q=%.6f, GPU_q=%.6f | CPU_k=%.6f, GPU_k=%.6f\n",
                            i, q_cpu[i], q_gpu[i], k_cpu[i], k_gpu[i]);
            }
            ++mismatches;
        }
    }

    std::cout << "Max error = " << max_err
              << ", mismatches (> " << tol << ") = "
              << mismatches << " / " << (dim*2) << std::endl;
    if (mismatches == 0) {
        std::cout << "✅ ropeEncoding GPU 正确性验证通过！" << std::endl;
    } else {
        std::cout << "❌ 存在不匹配，请检查实现。" << std::endl;
    }

    // 6. 释放
    HIP_CHECK( hipFree(d_q) );
    HIP_CHECK( hipFree(d_k) );

    return 0;
}
