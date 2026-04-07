CXX  := g++
NVCC := nvcc

CXXFLAGS  := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead
NVCCFLAGS := -std=c++17 -O3 -Ihead -Xcompiler="-Wall -Wextra"

CUDA_ARCH := -arch=sm_86
NVCCFLAGS += $(CUDA_ARCH)

BUILD_DIR := build
CPU_BUILD_DIR := $(BUILD_DIR)/cpu
GPU_BUILD_DIR := $(BUILD_DIR)/gpu

CPU_TARGET := main_cpu
GPU_TARGET := main_gpu

CPU_MAIN := scripts/cpu/main_cpu.cpp
GPU_MAIN := scripts/gpu/main_gpu.cu

COMMON_CPP := src/test_cases.cpp src/init.cpp
CPU_CPP := src/cpu/solver_cpu.cpp
GPU_CU := src/gpu/solver_gpu.cu src/gpu/boundary_gpu.cu

CPU_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu.o \
	$(CPU_BUILD_DIR)/test_cases.o \
	$(CPU_BUILD_DIR)/init.o \
	$(CPU_BUILD_DIR)/solver_cpu.o

GPU_OBJS := \
	$(GPU_BUILD_DIR)/main_gpu.o \
	$(GPU_BUILD_DIR)/test_cases.o \
	$(GPU_BUILD_DIR)/init.o \
	$(GPU_BUILD_DIR)/solver_gpu.o \
	$(GPU_BUILD_DIR)/boundary_gpu.o

.PHONY: all
all: cpu gpu

.PHONY: cpu
cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_OBJS)
	$(CXX) $(CPU_OBJS) -o $@

$(CPU_BUILD_DIR)/main_cpu.o: $(CPU_MAIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/solver_cpu.o: src/cpu/solver_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: gpu
gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) -o $@

$(GPU_BUILD_DIR)/main_gpu.o: $(GPU_MAIN)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(GPU_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(GPU_BUILD_DIR)/solver_gpu.o: src/gpu/solver_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/boundary_gpu.o: src/gpu/boundary_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: run_cpu
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

.PHONY: run_gpu
run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(CPU_TARGET) $(GPU_TARGET)