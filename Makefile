CXX    := g++
NVCC   := nvcc
MPICXX := mpicxx

# =========================
# Common flags
# =========================
CXXFLAGS_BASE     := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead
NVCCFLAGS_BASE    := -std=c++17 -O3 -Ihead -Xcompiler="-Wall -Wextra"
MPICXXFLAGS_BASE  := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead -DOMPI_SKIP_MPICXX

CUDA_ARCH := -arch=sm_80
NVCCFLAGS_BASE += $(CUDA_ARCH)

# =========================
# OpenMP flags
# =========================
OMPFLAGS := -fopenmp

# CPU OpenMP build
CXXFLAGS := $(CXXFLAGS_BASE) $(OMPFLAGS)

# GPU build
NVCCFLAGS := $(NVCCFLAGS_BASE)

# Pure MPI build: deliberately no OpenMP
MPICXXFLAGS := $(MPICXXFLAGS_BASE)

# Optional hybrid MPI + OpenMP build
MPICXXFLAGS_OMP := $(MPICXXFLAGS_BASE) $(OMPFLAGS)

# =========================
# Directories / targets
# =========================
BUILD_DIR := build
CPU_BUILD_DIR := $(BUILD_DIR)/cpu
GPU_BUILD_DIR := $(BUILD_DIR)/gpu
MPI_BUILD_DIR := $(BUILD_DIR)/mpi
MPI_OMP_BUILD_DIR := $(BUILD_DIR)/mpi_omp

CPU_TARGET := main_cpu
CPU_SERIAL_TARGET := main_cpu_serial
GPU_TARGET := main_gpu
MPI_TARGET := main_mpi
MPI_OMP_TARGET := main_mpi_omp

CPU_MAIN := scripts/cpu/main_cpu.cpp
GPU_MAIN := scripts/gpu/main_gpu.cu
MPI_MAIN := scripts/cpu/main_mpi.cpp

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

# Pure MPI: do not link solver_cpu.o
MPI_OBJS := \
	$(MPI_BUILD_DIR)/main_mpi.o \
	$(MPI_BUILD_DIR)/test_cases.o \
	$(MPI_BUILD_DIR)/init.o \
	$(MPI_BUILD_DIR)/solver_mpi.o

# Optional hybrid target: still only links solver_mpi.o.
# Use this only if solver_mpi.cpp itself contains OpenMP pragmas later.
MPI_OMP_OBJS := \
	$(MPI_OMP_BUILD_DIR)/main_mpi.o \
	$(MPI_OMP_BUILD_DIR)/test_cases.o \
	$(MPI_OMP_BUILD_DIR)/init.o \
	$(MPI_OMP_BUILD_DIR)/solver_mpi.o

# Safer default: do not build GPU unless explicitly requested
.PHONY: all
all: cpu cpu_serial mpi

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
# Pure MPI
# =========================
.PHONY: mpi
mpi: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_OBJS)
	$(MPICXX) $(MPICXXFLAGS) $(MPI_OBJS) -o $@ -lstdc++fs

$(MPI_BUILD_DIR)/main_mpi.o: $(MPI_MAIN)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/solver_mpi.o: src/cpu/solver_mpi.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# =========================
# Optional MPI + OpenMP build
# =========================
.PHONY: mpi_omp
mpi_omp: $(MPI_OMP_TARGET)

$(MPI_OMP_TARGET): $(MPI_OMP_OBJS)
	$(MPICXX) $(MPICXXFLAGS_OMP) $(MPI_OMP_OBJS) -o $@ -lstdc++fs

$(MPI_OMP_BUILD_DIR)/main_mpi.o: $(MPI_MAIN)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/solver_mpi.o: src/cpu/solver_mpi.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

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

.PHONY: run_mpi
run_mpi: $(MPI_TARGET)
	OMP_NUM_THREADS=1 mpirun -np 4 ./$(MPI_TARGET) 1 --timing-only

.PHONY: run_mpi_output
run_mpi_output: $(MPI_TARGET)
	OMP_NUM_THREADS=1 mpirun -np 4 ./$(MPI_TARGET) 1 --output

.PHONY: run_mpi_omp
run_mpi_omp: $(MPI_OMP_TARGET)
	OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores mpirun -np 4 ./$(MPI_OMP_TARGET) 1 --timing-only

# =========================
# Clean
# =========================
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(CPU_TARGET) $(CPU_SERIAL_TARGET) $(GPU_TARGET) $(MPI_TARGET) $(MPI_OMP_TARGET)