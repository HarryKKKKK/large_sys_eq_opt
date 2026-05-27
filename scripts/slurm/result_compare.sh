#!/bin/bash -l
#SBATCH -J sys_eq_part2_scale
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Part 2 only: CPU OpenMP / GPU / pure MPI timing comparison
#
# Runs:
#   scales  : N = 1, 2, 4, 8, 16
#   cases   : shock_bubble, blast_wave
#   solvers : hll, hllc, exact, force
#   arch    : cpu_omp, gpu, mpi
#
# Total runs:
#   5 scales x 2 cases x 4 solvers x 3 arch = 120 runs
#
# Default parallel settings:
#   CPU OpenMP : 1 process x 8 threads
#   MPI        : 8 MPI ranks x 1 thread per rank
#   GPU        : 1 GPU
#
# Usage:
#   sbatch result_compare_part2_scales.sh
#
# Optional overrides:
#   MPI_RANKS=8 OMP_THREADS_CPU=8 OMP_THREADS_MPI=1 sbatch result_compare_part2_scales.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

WORKDIR="${SLURM_SUBMIT_DIR}"
cd "$WORKDIR"

mkdir -p logs outputs scaling validation

SCALES=(1 2 4 8 16)

MPI_RANKS="${MPI_RANKS:-8}"
OMP_THREADS_MPI="${OMP_THREADS_MPI:-1}"
OMP_THREADS_CPU="${OMP_THREADS_CPU:-8}"

CASES=("shock_bubble" "blast_wave")
SOLVERS=("hll" "hllc" "exact" "force")

# For a quick test:
# SCALES=(1)
# CASES=("shock_bubble")
# SOLVERS=("hll")

echo "===== JOB INFO ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "Workdir: ${WORKDIR}"
echo "SCALES: ${SCALES[*]}"
echo "CASES: ${CASES[*]}"
echo "SOLVERS: ${SOLVERS[*]}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "OMP_THREADS_CPU: ${OMP_THREADS_CPU}"
echo "OMP_THREADS_MPI: ${OMP_THREADS_MPI}"
echo "MPI total CPU units: $((MPI_RANKS * OMP_THREADS_MPI))"
echo ""

echo "===== MODULE SETUP ====="

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -f /usr/local/Modules/init/bash ]; then
        source /usr/local/Modules/init/bash
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
    echo "[INFO] Loaded module: rhel8/default-amp"
else
    echo "[WARN] module command is still not available."
    echo "[WARN] Skipping module purge/load and continuing with current PATH."
fi

echo "which g++   : $(which g++ || true)"
echo "which nvcc  : $(which nvcc || true)"
echo "which mpicxx: $(which mpicxx || true)"
echo "which mpirun: $(which mpirun || true)"
echo ""

export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "===== BUILD ====="
make clean
make cpu
make gpu
make mpi
echo ""

run_mpi_command() {
    local log_file="$1"
    shift

    # This cluster's OpenMPI cannot be launched directly by srun because
    # it was not built with Slurm PMI support. Use mpirun inside the Slurm
    # allocation instead.
    mpirun -np "${MPI_RANKS}" "$@" 2>&1 | tee "${log_file}"
}

extract_value_colon() {
    local needle="$1"
    local log_file="$2"

    awk -F ':' -v needle="$needle" '
        index($0, needle) > 0 {
            gsub(/s/, "", $2);
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            print $2;
            exit
        }
    ' "$log_file"
}

extract_steps() {
    local prefix="$1"
    local log_file="$2"
    local needle="[${prefix}] Total steps"

    awk -F '=' -v needle="$needle" '
        index($0, needle) > 0 {
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            print $2;
            exit
        }
    ' "$log_file"
}

append_summary_row() {
    local summary_file="$1"
    local arch="$2"
    local prefix="$3"
    local case_name="$4"
    local solver_name="$5"
    local n_scale="$6"
    local log_file="$7"

    local nx ny cells steps main_loop boundary compute_dt advance total_program avg_step

    nx=$(extract_value_colon "[${prefix}] nx" "$log_file")
    ny=$(extract_value_colon "[${prefix}] ny" "$log_file")
    cells=$(extract_value_colon "[${prefix}] total_cells" "$log_file")
    steps=$(extract_steps "$prefix" "$log_file")

    main_loop=$(extract_value_colon "main_loop_time" "$log_file")
    boundary=$(extract_value_colon "boundary_time" "$log_file")
    compute_dt=$(extract_value_colon "compute_dt_time" "$log_file")
    advance=$(extract_value_colon "advance_time" "$log_file")
    total_program=$(extract_value_colon "total_program_time" "$log_file")
    avg_step=$(extract_value_colon "avg_time_per_step" "$log_file")

    echo "${arch},${case_name},${solver_name},${n_scale},${nx},${ny},${cells},${steps},${main_loop},${boundary},${compute_dt},${advance},${total_program},${avg_step},${log_file}" >> "$summary_file"
}

write_summary_header() {
    local summary_file="$1"
    echo "arch,case,solver,n,nx,ny,total_cells,total_steps,main_loop_time,boundary_time,compute_dt_time,advance_time,total_program_time,avg_time_per_step,log_file" > "$summary_file"
}

TIMING_SUMMARY="validation/cpu_gpu_mpi_timing_scales_${SLURM_JOB_ID}.csv"
write_summary_header "$TIMING_SUMMARY"

echo ""
echo "===== PART 2: CPU/GPU/MPI PURE TIMING RUNS ACROSS SCALES ====="

for N in "${SCALES[@]}"; do
    echo ""
    echo "============================================================"
    echo "===== STARTING SCALE N=${N} ====="
    echo "============================================================"

    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do

            echo ""
            echo "===== CPU OMP TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N}, threads=${OMP_THREADS_CPU} ====="
            CPU_LOG="logs/cpu_timing_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_CPU}"
            ./main_cpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$CPU_LOG"
            append_summary_row "$TIMING_SUMMARY" "cpu_omp" "CPU" "$CASE" "$SOLVER" "$N" "$CPU_LOG"

            echo "Saved CPU timing log: $CPU_LOG"

            echo ""
            echo "===== GPU TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="
            GPU_LOG="logs/gpu_timing_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

            ./main_gpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$GPU_LOG"
            append_summary_row "$TIMING_SUMMARY" "gpu" "GPU" "$CASE" "$SOLVER" "$N" "$GPU_LOG"

            echo "Saved GPU timing log: $GPU_LOG"

            echo ""
            echo "===== MPI TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_timing_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            run_mpi_command "$MPI_LOG" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER"
            append_summary_row "$TIMING_SUMMARY" "mpi" "MPI" "$CASE" "$SOLVER" "$N" "$MPI_LOG"

            echo "Saved MPI timing log: $MPI_LOG"

            echo ""
            echo "Current timing summary tail:"
            tail -n 6 "$TIMING_SUMMARY" || true

        done
    done
done

echo ""
echo "===== CPU/GPU/MPI TIMING SUMMARY ACROSS SCALES ====="
cat "$TIMING_SUMMARY"

echo ""
echo "Timing summary:"
echo "$TIMING_SUMMARY"

echo ""
echo "===== END ====="
date
