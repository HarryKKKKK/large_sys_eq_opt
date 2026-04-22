CXX  := g++
NVCC := nvcc

# =========================
# Common flags
# =========================
CXXFLAGS_BASE  := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead
NVCCFLAGS_BASE := -std=c++17 -O3 -Ihead -Xcompiler="-Wall -Wextra"

CUDA_ARCH := -arch=sm_80
NVCCFLAGS_BASE += $(CUDA_ARCH)

# =========================
# OpenMP flags
# =========================
OMPFLAGS := -fopenmp

# By default, build CPU with OpenMP enabled.
# If you want a pure serial baseline:
#   make cpu_serial
CXXFLAGS  := $(CXXFLAGS_BASE) $(OMPFLAGS)
NVCCFLAGS := $(NVCCFLAGS_BASE)

# =========================
# Directories / targets
# =========================
BUILD_DIR := build
CPU_BUILD_DIR := $(BUILD_DIR)/cpu
GPU_BUILD_DIR := $(BUILD_DIR)/gpu

CPU_TARGET := main_cpu
CPU_SERIAL_TARGET := main_cpu_serial
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

CPU_SERIAL_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu_serial.o \
	$(CPU_BUILD_DIR)/test_cases_serial.o \
	$(CPU_BUILD_DIR)/init_serial.o \
	$(CPU_BUILD_DIR)/solver_cpu_serial.o

GPU_OBJS := \
	$(GPU_BUILD_DIR)/main_gpu.o \
	$(GPU_BUILD_DIR)/test_cases.o \
	$(GPU_BUILD_DIR)/init.o \
	$(GPU_BUILD_DIR)/solver_gpu.o \
	$(GPU_BUILD_DIR)/boundary_gpu.o

.PHONY: all
all: cpu gpu

# =========================
# CPU with OpenMP
# =========================
.PHONY: cpu
cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_OBJS)
	$(CXX) $(CXXFLAGS) $(CPU_OBJS) -o $@ -lstdc++fs

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

# =========================
# CPU serial baseline
# =========================
.PHONY: cpu_serial
cpu_serial: $(CPU_SERIAL_TARGET)

$(CPU_SERIAL_TARGET): $(CPU_SERIAL_OBJS)
	$(CXX) $(CXXFLAGS_BASE) $(CPU_SERIAL_OBJS) -o $@ -lstdc++fs

$(CPU_BUILD_DIR)/main_cpu_serial.o: $(CPU_MAIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/test_cases_serial.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/init_serial.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/solver_cpu_serial.o: src/cpu/solver_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

# =========================
# GPU
# =========================
.PHONY: gpu
gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) -o $@ -lstdc++fs

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

# =========================
# Run helpers
# =========================
.PHONY: run_cpu
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

.PHONY: run_cpu_serial
run_cpu_serial: $(CPU_SERIAL_TARGET)
	./$(CPU_SERIAL_TARGET)

.PHONY: run_cpu_omp
run_cpu_omp: $(CPU_TARGET)
	OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores ./$(CPU_TARGET)

.PHONY: run_gpu
run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET)

# =========================
# Clean
# =========================
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(CPU_TARGET) $(CPU_SERIAL_TARGET) $(GPU_TARGET)