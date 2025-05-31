#ifndef GPU_RUNSTATE_HPP
#define GPU_RUNSTATE_HPP

#include "../model/modelConfig.hpp"
#include "../infer/runState.hpp"
#include <hip/hip_runtime.h>

// 错误检查宏
#define HIP_CHECK(cmd)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t error = cmd;                                              \
        if (error != hipSuccess)                                             \
        {                                                                    \
            std::cerr << "[ERROR:] HIP error " << hipGetErrorString(error)   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

class GPU_RunState
{
public:
    GPU_RunState();
    ~GPU_RunState();
    // GPU设备内存指针
    float *d_currentActivation;              // 设备上的当前激活值
    float *d_branchActivation;               // 设备上的分支激活值
    float *d_extraBuffer;                    // 设备上的额外缓冲区
    float *d_hiddenBuffer_extraHiddenBuffer; // hiddenBuffer和extraHiddenBuffer的合并
    float *d_hiddenBuffer;                   // 设备上的隐藏缓冲区
    float *d_extraHiddenBuffer;              // 设备上的额外隐藏缓冲区
    float *d_q;                              // 设备上的查询向量
    float *d_k;                              // 设备上的键向量
    float *d_v;                              // 设备上的值向量
    float *d_attentionScores;                // 设备上的注意力分数
    float *d_logits;                         // 设备上的logits
    float *h_logits;                         // 主机上的logits
    float *d_keyCache;                       // 设备上的键缓存
    float *d_valueCache;                     // 设备上的值缓存
    float *d_scores;                         // 设备上的flash attention分数
    float *d_attn;                           // 设备上的flash attention权重
    float *d_qkv;                            // 设备上的QKV

    // GPU内存管理方法
    void allocateGPUMemory(CModelConfig *config);
    void deallocateGPUMemory();

    // 数据传输方法
    void copyToGPU(CRunState *cpuState, CModelConfig *config);   // 将主机内存数据复制到GPU
    void copyFromGPU(CRunState *cpuState, CModelConfig *config); // 将GPU数据复制回主机内存

private:
};

#endif // GPU_RUNSTATE_HPP