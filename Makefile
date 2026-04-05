# =========================
# Compilers
# =========================
CXX  := g++
NVCC := nvcc

# =========================
# Flags
# =========================
CXXFLAGS       := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead -Ihead/cpu -Ihead/gpu
NVCC_STD       := -std=c++17
NVCC_ARCH      := -arch=sm_80
NVCC_WARN      := -Xcompiler="-Wall -Wextra"
NVCC_INC       := -Ihead -Ihead/cpu -Ihead/gpu
NVCCFLAGS      := $(NVCC_STD) -O3 $(NVCC_ARCH) $(NVCC_WARN) $(NVCC_INC)

# =========================
# Targets
# =========================
CPU_TARGET := main_cpu
GPU_TARGET := main_gpu

# =========================
# Sources
# =========================
CPU_MAIN   := scripts/cpu/main_cpu.cpp
GPU_MAIN   := scripts/gpu/main_gpu.cu

COMMON_CPP := src/test_cases.cpp
CPU_CPP    := src/cpu/solver_cpu.cpp
GPU_CU     := src/gpu/solver_gpu.cu src/gpu/boundary_gpu.cu

# =========================
# Build directories
# =========================
BUILD_DIR      := build
CPU_BUILD_DIR  := $(BUILD_DIR)/cpu
GPU_BUILD_DIR  := $(BUILD_DIR)/gpu

CPU_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu.o \
	$(CPU_BUILD_DIR)/test_cases.o \
	$(CPU_BUILD_DIR)/solver_cpu.o

GPU_OBJS := \
	$(GPU_BUILD_DIR)/main_gpu.o \
	$(GPU_BUILD_DIR)/test_cases.o \
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

$(CPU_BUILD_DIR)/test_cases.o: $(COMMON_CPP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/solver_cpu.o: $(CPU_CPP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: gpu
gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) -o $@

$(GPU_BUILD_DIR)/main_gpu.o: $(GPU_MAIN)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/test_cases.o: $(COMMON_CPP)
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