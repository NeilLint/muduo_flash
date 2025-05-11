#!/bin/bash

# 设置HIP路径（如果需要）
export HIP_PATH=${HIP_PATH:-/opt/rocm}

# 编译GPU检测程序
echo "编译GPU检测程序..."
hipcc -I${HIP_PATH}/include -L${HIP_PATH}/lib scripts/check_gpu.cpp -o check_gpu

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败！请检查HIP环境设置"
    exit 1
fi

# 运行检测程序
echo "运行GPU检测程序..."
./check_gpu

# 检查是否运行成功
if [ $? -ne 0 ]; then
    echo "GPU检测失败！GPU可能不可用或不支持HIP"
else
    echo "GPU检测成功！"
fi

# 清理
rm -f check_gpu 