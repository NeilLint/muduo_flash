#include <iostream>
#include <hip/hip_runtime.h>

int main() {
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(error) << std::endl;
        std::cerr << "GPU不可用或HIP环境未正确设置" << std::endl;
        return 1;
    }
    
    std::cout << "找到 " << deviceCount << " 个GPU设备" << std::endl;
    
    // 打印每个设备的信息
    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        
        std::cout << "设备 #" << i << ": " << props.name << std::endl;
        std::cout << "  计算能力: " << props.major << "." << props.minor << std::endl;
        std::cout << "  总内存: " << props.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  多处理器数量: " << props.multiProcessorCount << std::endl;
        std::cout << "  时钟频率: " << props.clockRate / 1000 << " MHz" << std::endl;
    }
    
    return 0;
} 