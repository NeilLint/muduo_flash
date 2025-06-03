# Project Configuration
SRC_DIR = ./src
BUILD_DIR = build
TARGET = muduo

# GPU Configuration
USE_GPU ?= 1
HIP_PATH ?= $(shell hipconfig --path 2>/dev/null || echo /opt/rocm)

# Source Files
SRCS = $(wildcard $(SRC_DIR)/*.cpp) \
       $(wildcard $(SRC_DIR)/backend/*.cpp) \
       $(wildcard $(SRC_DIR)/model/*.cpp) \
       $(wildcard $(SRC_DIR)/infer/*.cpp)

# Object Files
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
DEPS = $(OBJS:.o=.d)

# Compiler Settings
HIPCC = hipcc
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra \
           -I. \
           -I$(SRC_DIR)/model \
           -I$(SRC_DIR)/backend \
           -I$(SRC_DIR)/infer \
           -MMD -MP

# GPU Support
ifeq ($(USE_GPU), 1)
    CXXFLAGS += -DUSE_GPU_BACKEND -DHAS_HIP -I$(HIP_PATH)/include
    $(info Building with GPU support (HIP_PATH: $(HIP_PATH)))
else
    $(info Building without GPU support)
endif

# Linker Settings
LDFLAGS = -L$(HIP_PATH)/lib
LDLIBS = -lhipblas -lm

# Build Rules
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	$(HIPCC) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build completed: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling $< -> $@..."
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)