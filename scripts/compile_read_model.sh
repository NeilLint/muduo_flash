#!/bin/bash

g++ -std=c++17 read_model.cpp ../src/model/modelConfig.cpp -o read_model -I../src

echo "编译完成，使用 ./read_model 来运行程序" 