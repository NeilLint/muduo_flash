CC=hipcc
CFLAGS=-std=c++11 -O3 -I/opt/rocm/include
LDFLAGS=-L/opt/rocm/lib -lhipblas

SRC_DIR=../../src/backend
BUILD_DIR=./build
TEST_TARGETS=$(BUILD_DIR)/test_softmax $(BUILD_DIR)/test_matmul $(BUILD_DIR)/test_rmsnorm $(BUILD_DIR)/test_axpy $(BUILD_DIR)/test_dot $(BUILD_DIR)/test_gemvQkSeq

.PHONY: all clean

all: prepare $(TEST_TARGETS)

prepare:
	mkdir -p $(BUILD_DIR)

# 编译各个测试
$(BUILD_DIR)/test_softmax: test_softmax.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_matmul: test_matmul.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_rmsnorm: test_rmsnorm.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_axpy: test_axpy.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_dot: test_dot.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_gemvQkSeq: test_gemvQkSeq.cpp $(SRC_DIR)/gpu_backend.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 运行所有测试
run: all
	@echo "运行所有测试..."
	@for test in $(TEST_TARGETS); do \
		echo "\n=== 运行 $$test ==="; \
		$$test; \
	done

# 单独运行各个测试
run_softmax: $(BUILD_DIR)/test_softmax
	$(BUILD_DIR)/test_softmax

run_matmul: $(BUILD_DIR)/test_matmul
	$(BUILD_DIR)/test_matmul

run_rmsnorm: $(BUILD_DIR)/test_rmsnorm
	$(BUILD_DIR)/test_rmsnorm

run_axpy: $(BUILD_DIR)/test_axpy
	$(BUILD_DIR)/test_axpy

run_dot: $(BUILD_DIR)/test_dot
	$(BUILD_DIR)/test_dot

run_gemvQkSeq: $(BUILD_DIR)/test_gemvQkSeq
	$(BUILD_DIR)/test_gemvQkSeq

clean:
	rm -rf $(BUILD_DIR) 