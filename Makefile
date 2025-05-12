# Source directory
SRC_DIR = ./src
# Build directory for object files
BUILD_DIR = build

# Enable GPU backend by default (can be overridden by make USE_GPU=0)
USE_GPU ?= 1

# HIP安装路径
HIP_PATH ?= $(shell hipconfig --path 2>/dev/null || echo /opt/rocm)

# Find all .cpp source files in specified subdirectories
SRCS = $(wildcard $(SRC_DIR)/*.cpp) \
       $(wildcard $(SRC_DIR)/backend/*.cpp) \
       $(wildcard $(SRC_DIR)/model/*.cpp) \
       $(wildcard $(SRC_DIR)/infer/*.cpp)

# Create corresponding object file paths in the build directory
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Create dependency file paths (for automatic header dependency tracking)
DEPS = $(OBJS:.o=.d)

# Compiler
HIPCC = hipcc

# 基础编译参数
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra \
           -I. \
           -I$(SRC_DIR)/model \
           -I$(SRC_DIR)/backend \
           -I$(SRC_DIR)/infer \
           -MMD -MP

# GPU支持条件编译
ifeq ($(USE_GPU), 1)
    CXXFLAGS += -DUSE_GPU_BACKEND -DHAS_HIP
    # 添加HIP包含路径
    CXXFLAGS += -I$(HIP_PATH)/include
    $(info Building with GPU support (HIP_PATH: $(HIP_PATH)))
else
    $(info Building without GPU support)
endif

# Libraries to link against
# -lhipblas: Link the hipBLAS library
# -lm: Link the standard math library
LDLIBS = -lhipblas -lm
# 添加库搜索路径
LDFLAGS = -L$(HIP_PATH)/lib

# Target executable name
TARGET = muduo

# --- Rules ---

# Default rule: Build the target executable
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(OBJS)
	@echo "Linking $@..."
	$(HIPCC) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build finished: $@"

# Pattern rule to compile .cpp files into .o files in the build directory
# $<: The first prerequisite (the .cpp file)
# $@: The target (the .o file)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@) # Create the build subdirectory if it doesn't exist
	@echo "Compiling $< -> $@..."
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Rule to clean build artifacts
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR) $(TARGET)

# Include the generated dependency files.
# The '-' sign tells make to ignore errors if the .d file doesn't exist (e.g., on first build or after clean)
-include $(DEPS)

# Phony targets are targets that don't represent actual files
.PHONY: all clean