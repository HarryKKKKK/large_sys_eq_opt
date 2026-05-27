#!/bin/bash -l
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J sys_eq_clean_fullrank
#SBATCH -o /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.out
#SBATCH -e /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.err

# ============================================================
# Clean Part 2 run script for CSD3
#
# Purpose:
#   Run CPU OpenMP / GPU / pure MPI using clean main files with
#   no internal timing code.
#
# Timing method:
#   Uses /usr/bin/time outside the program. This avoids modifying
#   the solver/main code and has much lower impact than per-step
#   internal timing.
#
# Default allocation:
#   CPU OpenMP : 1 process x all allocated CPU cores
#   MPI        : one MPI rank per allocated CPU core
#   GPU        : 1 GPU
#
# Optional overrides:
#   MPI_RANKS=32 OMP_THREADS_CPU=32 sbatch result_compare_clean_fullrank.sh
#   SCALES_STR="1 4 8 16" CASES_STR="shock_bubble" SOLVERS_STR="hll hllc" sbatch result_compare_clean_fullrank.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

WORKDIR="${SLURM_SUBMIT_DIR}"
cd "$WORKDIR"

mkdir -p logs outputs scaling validation

# -------------------------
# Benchmark matrix
# -------------------------
read -r -a SCALES  <<< "${SCALES_STR:-1 4 8 16}"
read -r -a CASES   <<< "${CASES_STR:-shock_bubble}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll}"

# For quick test:
#   SCALES_STR="1" CASES_STR="shock_bubble" SOLVERS_STR="hll" sbatch result_compare_clean_fullrank.sh

# -------------------------
# Module setup
# -------------------------
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
    echo "[WARN] module command is not available. Continuing with current environment."
fi

# -------------------------
# Detect usable CPU cores
# -------------------------
detect_cpus_on_node() {
    if [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
        echo "${SLURM_CPUS_ON_NODE}"
        return
    fi

    if [ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]; then
        echo "${SLURM_JOB_CPUS_PER_NODE}" | sed -E 's/\(x[0-9]+\)//'
        return
    fi

    nproc
}

CPUS_ON_NODE="$(detect_cpus_on_node)"

MPI_RANKS="${MPI_RANKS:-${CPUS_ON_NODE}}"
OMP_THREADS_CPU="${OMP_THREADS_CPU:-${CPUS_ON_NODE}}"
OMP_THREADS_MPI="${OMP_THREADS_MPI:-1}"

MPI_TOTAL_CPU_UNITS=$((MPI_RANKS * OMP_THREADS_MPI))

if [ "${MPI_TOTAL_CPU_UNITS}" -gt "${CPUS_ON_NODE}" ]; then
    echo "[ERROR] MPI_RANKS * OMP_THREADS_MPI = ${MPI_TOTAL_CPU_UNITS}, but detected only ${CPUS_ON_NODE} CPUs."
    exit 1
fi

if [ "${OMP_THREADS_CPU}" -gt "${CPUS_ON_NODE}" ]; then
    echo "[ERROR] OMP_THREADS_CPU = ${OMP_THREADS_CPU}, but detected only ${CPUS_ON_NODE} CPUs."
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo ""
echo "===== JOB INFO ====="
echo "JobID                : ${SLURM_JOB_ID}"
echo "Host                 : $(hostname)"
echo "Start                : $(date)"
echo "Workdir              : ${WORKDIR}"
echo "Partition            : ${SLURM_JOB_PARTITION:-unknown}"
echo "Nodes                : ${SLURM_JOB_NUM_NODES:-1}"
echo "SLURM_CPUS_ON_NODE   : ${SLURM_CPUS_ON_NODE:-unset}"
echo "Detected CPUs/node   : ${CPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCALES               : ${SCALES[*]}"
echo "CASES                : ${CASES[*]}"
echo "SOLVERS              : ${SOLVERS[*]}"
echo "MPI_RANKS            : ${MPI_RANKS}"
echo "OMP_THREADS_CPU      : ${OMP_THREADS_CPU}"
echo "OMP_THREADS_MPI      : ${OMP_THREADS_MPI}"
echo "MPI total CPU units  : ${MPI_TOTAL_CPU_UNITS}"
echo ""

echo "===== CPU/GPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
nvidia-smi -L || true
echo ""

echo "which g++    : $(which g++ || true)"
echo "which nvcc   : $(which nvcc || true)"
echo "which mpicxx : $(which mpicxx || true)"
echo "which mpirun : $(which mpirun || true)"
echo ""

# -------------------------
# Build
# -------------------------
echo "===== BUILD ====="
make clean
make cpu
make gpu
make mpi
echo ""

# -------------------------
# Helpers
# -------------------------
run_with_time() {
    local log_file="$1"
    shift

    /usr/bin/time \
        -f "[TIME] real_seconds=%e\n[TIME] user_seconds=%U\n[TIME] sys_seconds=%S\n[TIME] max_rss_kb=%M" \
        "$@" 2>&1 | tee "${log_file}"
}

run_mpi_with_time() {
    local log_file="$1"
    shift

    /usr/bin/time \
        -f "[TIME] real_seconds=%e\n[TIME] user_seconds=%U\n[TIME] sys_seconds=%S\n[TIME] max_rss_kb=%M" \
        mpirun \
            --host "$(hostname):${MPI_RANKS}" \
            -np "${MPI_RANKS}" \
            --map-by core \
            --bind-to core \
            "$@" 2>&1 | tee "${log_file}"
}

extract_value_colon() {
    local needle="$1"
    local log_file="$2"

    awk -F ':' -v needle="$needle" '
        index($0, needle) > 0 {
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

extract_time_field() {
    local field="$1"
    local log_file="$2"

    awk -F '=' -v field="[TIME] ${field}" '
        index($0, field) > 0 {
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
    local ranks="$7"
    local threads="$8"
    local log_file="$9"

    local nx ny cells steps real_s user_s sys_s rss_kb

    nx=$(extract_value_colon "[${prefix}] nx" "$log_file")
    ny=$(extract_value_colon "[${prefix}] ny" "$log_file")
    cells=$(extract_value_colon "[${prefix}] total_cells" "$log_file")
    steps=$(extract_steps "$prefix" "$log_file")

    real_s=$(extract_time_field "real_seconds" "$log_file")
    user_s=$(extract_time_field "user_seconds" "$log_file")
    sys_s=$(extract_time_field "sys_seconds" "$log_file")
    rss_kb=$(extract_time_field "max_rss_kb" "$log_file")

    echo "${arch},${case_name},${solver_name},${n_scale},${ranks},${threads},${nx},${ny},${cells},${steps},${real_s},${user_s},${sys_s},${rss_kb},${log_file}" >> "$summary_file"
}

write_summary_header() {
    local summary_file="$1"
    echo "arch,case,solver,n,ranks,threads_per_rank,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,log_file" > "$summary_file"
}

SUMMARY_FILE="validation/cpu_gpu_mpi_clean_fullrank_${SLURM_JOB_ID}.csv"
write_summary_header "$SUMMARY_FILE"

echo ""
echo "===== CLEAN CPU/GPU/MPI RUNS ACROSS SCALES ====="

for N in "${SCALES[@]}"; do
    echo ""
    echo "============================================================"
    echo "===== STARTING SCALE N=${N} ====="
    echo "============================================================"

    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do

            echo ""
            echo "===== CPU OMP RUN: case=${CASE}, solver=${SOLVER}, n=${N}, threads=${OMP_THREADS_CPU} ====="
            CPU_LOG="logs/cpu_clean_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_CPU}"
            run_with_time "$CPU_LOG" ./main_cpu "$N" --case "$CASE" --solver "$SOLVER"
            append_summary_row "$SUMMARY_FILE" "cpu_omp" "CPU" "$CASE" "$SOLVER" "$N" 1 "$OMP_THREADS_CPU" "$CPU_LOG"

            echo "Saved CPU log: $CPU_LOG"

            echo ""
            echo "===== GPU RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="
            GPU_LOG="logs/gpu_clean_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS=1
            run_with_time "$GPU_LOG" ./main_gpu "$N" --case "$CASE" --solver "$SOLVER"
            append_summary_row "$SUMMARY_FILE" "gpu" "GPU" "$CASE" "$SOLVER" "$N" 1 1 "$GPU_LOG"

            echo "Saved GPU log: $GPU_LOG"

            echo ""
            echo "===== MPI RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_clean_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            run_mpi_with_time "$MPI_LOG" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER"
            append_summary_row "$SUMMARY_FILE" "mpi" "MPI" "$CASE" "$SOLVER" "$N" "$MPI_RANKS" "$OMP_THREADS_MPI" "$MPI_LOG"

            echo "Saved MPI log: $MPI_LOG"

            echo ""
            echo "Current summary tail:"
            tail -n 6 "$SUMMARY_FILE" || true

        done
    done
done

echo ""
echo "===== CLEAN CPU/GPU/MPI SUMMARY ====="
cat "$SUMMARY_FILE"

echo ""
echo "Summary file:"
echo "$SUMMARY_FILE"

echo ""
echo "===== END ====="
date
