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

// --- GPU 后端（包含 kernel） ---
__global__
void swiGLLU_kernel(float* headOutput, const float* value, int hiddenDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hiddenDim) {
        float h = headOutput[idx];
        float sig = 1.0f / (1.0f + expf(-h));
        headOutput[idx] = h * sig * value[idx];
    }
}

class GPU_Backend {
public:
    void swiGLLUFunc(float *headOutput, const float *value, int hiddenDim) {
        if (hiddenDim <= 0 || headOutput == nullptr || value == nullptr) return;
        const int threads = 256;
        const int blocks = (hiddenDim + threads - 1) / threads;
        hipLaunchKernelGGL(
            swiGLLU_kernel,
            dim3(blocks),
            dim3(threads),
            0, 0,
            headOutput, value, hiddenDim
        );
        HIP_CHECK( hipGetLastError() );
        HIP_CHECK( hipDeviceSynchronize() );
    }
};

// --- CPU 参考实现 ---
void swiGLLU_cpu(std::vector<float> &headOutput, const std::vector<float> &value) {
    int hiddenDim = headOutput.size();
    for (int i = 0; i < hiddenDim; i++) {
        float h = headOutput[i];
        float sig = 1.0f / (1.0f + std::exp(-h));
        headOutput[i] = h * sig * value[i];
    }
}

int main() {
    const int hiddenDim = 1 << 16;  // 示例长度（64K 元素）
    std::cout << "Testing swiGLLUFunc with hiddenDim=" << hiddenDim << std::endl;
    const int hostDim = hiddenDim;
    // 1. 主机数据初始化
    std::vector<float> h_head(hostDim), h_val(hiddenDim);
    srand(2025);
    for (int i = 0; i < hiddenDim; ++i) {
        h_head[i] = (float(rand())/RAND_MAX)*4 - 2;  // [-2,2]
        h_val[i]  = (float(rand())/RAND_MAX)*2 - 1;  // [-1,1]
    }

    // 保存一份给 CPU 用
    std::vector<float> cpu_head = h_head;

    // 2. 分配并拷贝到设备
    float *d_head = nullptr, *d_val = nullptr;
    HIP_CHECK( hipMalloc(&d_head, hiddenDim * sizeof(float)) );
    HIP_CHECK( hipMalloc(&d_val,  hiddenDim * sizeof(float)) );
    HIP_CHECK( hipMemcpy(d_head, h_head.data(), hiddenDim * sizeof(float),
                         hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_val,  h_val.data(),  hiddenDim * sizeof(float),
                         hipMemcpyHostToDevice) );

    // 3. 调用 GPU 版本
    GPU_Backend backend;
    backend.swiGLLUFunc(d_head, d_val, hiddenDim);

    // 4. 拷回并对比
    std::vector<float> gpu_head(hiddenDim);
    HIP_CHECK( hipMemcpy(gpu_head.data(), d_head,
                         hiddenDim * sizeof(float),
                         hipMemcpyDeviceToHost) );

    // CPU 计算
    swiGLLU_cpu(cpu_head, h_val);

    // 5. 验证结果
    double max_err = 0.0;
    int mismatches = 0;
    const float tol = 1e-6f;
    for (int i = 0; i < hiddenDim; ++i) {
        double err = std::abs(cpu_head[i] - gpu_head[i]);
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (mismatches < 10) {
                std::printf("Mismatch @%d: CPU=%.7f, GPU=%.7f, diff=%.2e\n",
                            i, cpu_head[i], gpu_head[i], err);
            }
            mismatches++;
        }
    }

    std::cout << "Max error = " << max_err
              << ", mismatches (> " << tol << ") = "
              << mismatches << " / " << hiddenDim << std::endl;

    if (mismatches == 0) {
        std::cout << "✅ swiGLLUFunc GPU 正确性验证通过！" << std::endl;
    } else {
        std::cout << "❌ 存在 " << mismatches
                  << " 个不匹配，检查实现或数值精度。" << std::endl;
    }

    // 清理
    HIP_CHECK( hipFree(d_head) );
    HIP_CHECK( hipFree(d_val) );
    return 0;
}
