#!/bin/bash
#SBATCH -J sys_eq_gpu_base
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORKDIR="${SLURM_SUBMIT_DIR}"
cd "$WORKDIR"

mkdir -p logs outputs scaling validation

echo "===== JOB INFO ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo ""

module purge
module load rhel8/default-amp

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "===== BUILD ====="
make clean
make cpu
make gpu
echo ""

N=1

CASES=("shock_bubble" "blast_wave")
# SOLVERS=("exact" "force")
SOLVERS=("hll" "hllc" "exact" "force")

# ============================================================
# Part 1: GPU output runs for result validation
# 2 cases × 2 solvers × GPU = 4 runs
# These generate snapshot CSV files in outputs/
# ============================================================

VALIDATION_SUMMARY="validation/gpu_validation_output_${SLURM_JOB_ID}.csv"

echo "arch,case,solver,n,nx,ny,total_cells,total_steps,main_loop_time,boundary_time,compute_dt_time,advance_time,total_program_time,avg_time_per_step,log_file" > "$VALIDATION_SUMMARY"

echo "===== PART 1: GPU VALIDATION OUTPUT RUNS ====="

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do
        echo ""
        echo "===== GPU OUTPUT RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="

        RUN_LOG="logs/gpu_output_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

        ./main_gpu "$N" --case "$CASE" --solver "$SOLVER" --output 2>&1 | tee "$RUN_LOG"
        ./main_cpu "$N" --case "$CASE" --solver "$SOLVER" --output 2>&1 | tee "$RUN_LOG"

        NX=$(awk -F ':' '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        NY=$(awk -F ':' '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        CELLS=$(awk -F ':' '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        STEPS=$(awk -F '=' '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")

        MAIN_LOOP=$(awk -F ':' '/main_loop_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        BOUNDARY=$(awk -F ':' '/boundary_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        COMPUTE_DT=$(awk -F ':' '/compute_dt_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        ADVANCE=$(awk -F ':' '/advance_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        TOTAL_PROGRAM=$(awk -F ':' '/total_program_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")
        AVG_STEP=$(awk -F ':' '/avg_time_per_step/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$RUN_LOG")

        echo "gpu,${CASE},${SOLVER},${N},${NX},${NY},${CELLS},${STEPS},${MAIN_LOOP},${BOUNDARY},${COMPUTE_DT},${ADVANCE},${TOTAL_PROGRAM},${AVG_STEP},${RUN_LOG}" >> "$VALIDATION_SUMMARY"

        echo "Saved GPU validation log: $RUN_LOG"
    done
done

echo ""
echo "GPU validation output summary saved to:"
echo "$VALIDATION_SUMMARY"

# ============================================================
# Part 2: Pure timing runs, no snapshot output
# 2 cases × 2 solvers × 2 architectures = 8 runs
# These do NOT use --output.
# ============================================================

TIMING_SUMMARY="validation/cpu_gpu_timing_no_output_${SLURM_JOB_ID}.csv"

echo "arch,case,solver,n,nx,ny,total_cells,total_steps,main_loop_time,boundary_time,compute_dt_time,advance_time,total_program_time,avg_time_per_step,log_file" > "$TIMING_SUMMARY"

echo ""
echo "===== PART 2: CPU/GPU PURE TIMING RUNS ====="

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do

        # -----------------------------
        # CPU timing run
        # -----------------------------
        echo ""
        echo "===== CPU TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="

        CPU_LOG="logs/cpu_timing_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

        ./main_cpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$CPU_LOG"

        NX=$(awk -F ':' '/\[CPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        NY=$(awk -F ':' '/\[CPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        CELLS=$(awk -F ':' '/\[CPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        STEPS=$(awk -F '=' '/\[CPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")

        MAIN_LOOP=$(awk -F ':' '/main_loop_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        BOUNDARY=$(awk -F ':' '/boundary_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        COMPUTE_DT=$(awk -F ':' '/compute_dt_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        ADVANCE=$(awk -F ':' '/advance_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        TOTAL_PROGRAM=$(awk -F ':' '/total_program_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")
        AVG_STEP=$(awk -F ':' '/avg_time_per_step/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$CPU_LOG")

        echo "cpu,${CASE},${SOLVER},${N},${NX},${NY},${CELLS},${STEPS},${MAIN_LOOP},${BOUNDARY},${COMPUTE_DT},${ADVANCE},${TOTAL_PROGRAM},${AVG_STEP},${CPU_LOG}" >> "$TIMING_SUMMARY"

        echo "Saved CPU timing log: $CPU_LOG"

        # -----------------------------
        # GPU timing run
        # -----------------------------
        echo ""
        echo "===== GPU TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="

        GPU_LOG="logs/gpu_timing_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

        ./main_gpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$GPU_LOG"

        NX=$(awk -F ':' '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        NY=$(awk -F ':' '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        CELLS=$(awk -F ':' '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        STEPS=$(awk -F '=' '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")

        MAIN_LOOP=$(awk -F ':' '/main_loop_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        BOUNDARY=$(awk -F ':' '/boundary_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        COMPUTE_DT=$(awk -F ':' '/compute_dt_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        ADVANCE=$(awk -F ':' '/advance_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        TOTAL_PROGRAM=$(awk -F ':' '/total_program_time/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")
        AVG_STEP=$(awk -F ':' '/avg_time_per_step/{gsub(/s/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$GPU_LOG")

        echo "gpu,${CASE},${SOLVER},${N},${NX},${NY},${CELLS},${STEPS},${MAIN_LOOP},${BOUNDARY},${COMPUTE_DT},${ADVANCE},${TOTAL_PROGRAM},${AVG_STEP},${GPU_LOG}" >> "$TIMING_SUMMARY"

        echo "Saved GPU timing log: $GPU_LOG"

    done
done

echo ""
echo "===== GPU VALIDATION OUTPUT SUMMARY ====="
cat "$VALIDATION_SUMMARY"

echo ""
echo "===== CPU/GPU TIMING SUMMARY ====="
cat "$TIMING_SUMMARY"

echo ""
echo "Validation output summary:"
echo "$VALIDATION_SUMMARY"

echo ""
echo "Timing summary:"
echo "$TIMING_SUMMARY"

echo ""
echo "Snapshot CSV files should be in:"
echo "outputs/"

echo ""
echo "===== END ====="
date