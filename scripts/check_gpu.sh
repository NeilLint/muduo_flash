#!/bin/bash

# 设置HIP路径（如果需要）
export HIP_PATH=${HIP_PATH:-/opt/rocm}

# 编译检测程序
TMP_DIR=$(mktemp -d)
TMP_SRC="$TMP_DIR/check_gpu.cpp"
TMP_BIN="$TMP_DIR/check_gpu"

cat > "$TMP_SRC" << 'EOF'
#include <iostream>
#include <hip/hip_runtime.h>

int main() {
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount <= 0) {
        std::cerr << "未找到GPU设备" << std::endl;
        return 1;
    }
    
    std::cout << "找到 " << deviceCount << " 个GPU设备" << std::endl;
    return 0;
}
EOF

# 编译检测程序
hipcc -I${HIP_PATH}/include -L${HIP_PATH}/lib "$TMP_SRC" -o "$TMP_BIN" 2>/dev/null

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败！HIP环境可能未正确设置"
    rm -rf "$TMP_DIR"
    exit 1
fi

# 运行检测程序
"$TMP_BIN"
RESULT=$?

# 清理临时文件
rm -rf "$TMP_DIR"

# 返回检测结果
exit $RESULT 