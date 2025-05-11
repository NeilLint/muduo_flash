#!/bin/bash

g++ -std=c++17 read_bin_part_size.cpp ../src/model/modelConfig.cpp -o read_bin_part_size -I../src

echo "编译完成，使用 ./read_bin_part_size 来运行程序" 