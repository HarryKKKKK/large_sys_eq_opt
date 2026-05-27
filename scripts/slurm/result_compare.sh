#!/bin/bash -l
#SBATCH -J sys_eq_fullnode_compare
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# CPU OpenMP / GPU / pure MPI comparison, clean-main version
#
# IMPORTANT FOR CSD3 CSC GPU PARTITION:
#   CSD3 rejects: --exclusive + --gres=gpu:1
#   Therefore, to get the whole GPU node, this script requests:
#       --exclusive + --gres=gpu:4
#
# This gives CPU OpenMP and MPI access to the full node CPU resources.
# The GPU run still uses one executable instance; CUDA will usually choose
# device 0 from CUDA_VISIBLE_DEVICES unless your code selects otherwise.
#
# Clean-main timing:
#   The C++ programs do not need internal timers. This script measures
#   external wall time using /usr/bin/time.
#
# Runs by default:
#   scales  : N = 1, 2, 4, 8, 16
#   cases   : shock_bubble, blast_wave
#   solvers : hll, hllc, exact, force
#   arch    : cpu_omp, gpu, mpi
#
# Optional overrides:
#   SCALES_STR="1 2 4" CASES_STR="shock_bubble" SOLVERS_STR="hll hllc" sbatch this_script.sh
#   MPI_RANKS=32 OMP_THREADS_CPU=32 sbatch this_script.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

mkdir -p logs outputs scaling validation

# -------------------------
# Benchmark matrix
# -------------------------
read -r -a SCALES <<< "${SCALES_STR:-1 2 4 8 16}"
read -r -a CASES  <<< "${CASES_STR:-shock_bubble blast_wave}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc exact force}"

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

echo ""

# -------------------------
# Detect full-node CPU resources
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

# Full-node defaults:
#   cpu_omp: one process using all CPU cores
#   mpi: one MPI rank per CPU core
MPI_RANKS="${MPI_RANKS:-${CPUS_ON_NODE}}"
OMP_THREADS_CPU="${OMP_THREADS_CPU:-${CPUS_ON_NODE}}"
OMP_THREADS_MPI="${OMP_THREADS_MPI:-1}"

MPI_TOTAL_CPU_UNITS=$((MPI_RANKS * OMP_THREADS_MPI))

if [ "${MPI_TOTAL_CPU_UNITS}" -gt "${CPUS_ON_NODE}" ]; then
    echo "[ERROR] MPI_RANKS * OMP_THREADS_MPI = ${MPI_TOTAL_CPU_UNITS}, but detected only ${CPUS_ON_NODE} CPUs on this node."
    echo "[ERROR] Reduce MPI_RANKS or OMP_THREADS_MPI."
    exit 1
fi

if [ "${OMP_THREADS_CPU}" -gt "${CPUS_ON_NODE}" ]; then
    echo "[ERROR] OMP_THREADS_CPU = ${OMP_THREADS_CPU}, but detected only ${CPUS_ON_NODE} CPUs on this node."
    echo "[ERROR] Reduce OMP_THREADS_CPU."
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# -------------------------
# Job info
# -------------------------
echo "===== JOB INFO ====="
echo "JobID                 : ${SLURM_JOB_ID}"
echo "Host                  : $(hostname)"
echo "Start                 : $(date)"
echo "Workdir               : ${WORKDIR}"
echo "Partition             : ${SLURM_JOB_PARTITION:-unknown}"
echo "Nodes                 : ${SLURM_JOB_NUM_NODES:-unknown}"
echo "SLURM_NTASKS          : ${SLURM_NTASKS:-unset}"
echo "SLURM_CPUS_PER_TASK   : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE    : ${SLURM_CPUS_ON_NODE:-unset}"
echo "SLURM_JOB_CPUS_PER_NODE: ${SLURM_JOB_CPUS_PER_NODE:-unset}"
echo "Detected CPUs/node    : ${CPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCALES                : ${SCALES[*]}"
echo "CASES                 : ${CASES[*]}"
echo "SOLVERS               : ${SOLVERS[*]}"
echo "MPI_RANKS             : ${MPI_RANKS}"
echo "OMP_THREADS_CPU       : ${OMP_THREADS_CPU}"
echo "OMP_THREADS_MPI       : ${OMP_THREADS_MPI}"
echo "MPI total CPU units   : ${MPI_TOTAL_CPU_UNITS}"
echo "OMP_PROC_BIND         : ${OMP_PROC_BIND}"
echo "OMP_PLACES            : ${OMP_PLACES}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
echo ""

echo "===== GPU INFO ====="
nvidia-smi || true
echo ""

echo "===== COMPILER INFO ====="
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

read_time_value() {
    local key="$1"
    local time_file="$2"

    awk -F '=' -v key="$key" '
        $1 == key {
            print $2;
            exit
        }
    ' "$time_file"
}

run_with_time() {
    local time_file="$1"
    shift

    /usr/bin/time \
        -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$time_file" \
        "$@"
}

run_mpi_command() {
    local time_file="$1"
    shift

    # Use mpirun inside the Slurm allocation. The explicit host slot count
    # prevents OpenMPI from thinking the allocation has too few slots.
    /usr/bin/time \
        -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$time_file" \
        mpirun \
            --host "$(hostname):${MPI_RANKS}" \
            -np "${MPI_RANKS}" \
            --map-by core \
            --bind-to core \
            "$@"
}

append_summary_row() {
    local summary_file="$1"
    local arch="$2"
    local prefix="$3"
    local case_name="$4"
    local solver_name="$5"
    local n_scale="$6"
    local ranks="$7"
    local threads_per_rank="$8"
    local run_log="$9"
    local time_log="${10}"

    local nx ny cells steps real user sys rss

    nx=$(extract_value_colon "[${prefix}] nx" "$run_log")
    ny=$(extract_value_colon "[${prefix}] ny" "$run_log")
    cells=$(extract_value_colon "[${prefix}] total_cells" "$run_log")
    steps=$(extract_steps "$prefix" "$run_log")

    real=$(read_time_value "real_seconds" "$time_log")
    user=$(read_time_value "user_seconds" "$time_log")
    sys=$(read_time_value "sys_seconds" "$time_log")
    rss=$(read_time_value "max_rss_kb" "$time_log")

    echo "${arch},${case_name},${solver_name},${n_scale},${ranks},${threads_per_rank},${nx},${ny},${cells},${steps},${real},${user},${sys},${rss},${run_log},${time_log}" >> "$summary_file"
}

write_summary_header() {
    local summary_file="$1"
    echo "arch,case,solver,n,ranks,threads_per_rank,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,run_log,time_log" > "$summary_file"
}

SUMMARY="validation/cpu_gpu_mpi_fullnode_external_timing_${SLURM_JOB_ID}.csv"
write_summary_header "$SUMMARY"

# -------------------------
# Runs
# -------------------------
echo ""
echo "===== CPU/GPU/MPI FULL-NODE EXTERNAL TIMING RUNS ====="

for N in "${SCALES[@]}"; do
    echo ""
    echo "============================================================"
    echo "===== STARTING SCALE N=${N} ====="
    echo "============================================================"

    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do

            echo ""
            echo "===== CPU OMP RUN: case=${CASE}, solver=${SOLVER}, n=${N}, threads=${OMP_THREADS_CPU} ====="
            CPU_LOG="logs/cpu_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.log"
            CPU_TIME="logs/cpu_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.time"

            export OMP_NUM_THREADS="${OMP_THREADS_CPU}"
            run_with_time "$CPU_TIME" ./main_cpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$CPU_LOG"
            append_summary_row "$SUMMARY" "cpu_omp" "CPU" "$CASE" "$SOLVER" "$N" 1 "$OMP_THREADS_CPU" "$CPU_LOG" "$CPU_TIME"

            echo "Saved CPU run log : $CPU_LOG"
            echo "Saved CPU time log: $CPU_TIME"

            echo ""
            echo "===== GPU RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="
            GPU_LOG="logs/gpu_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"
            GPU_TIME="logs/gpu_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.time"

            export OMP_NUM_THREADS=1
            run_with_time "$GPU_TIME" ./main_gpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$GPU_LOG"
            append_summary_row "$SUMMARY" "gpu" "GPU" "$CASE" "$SOLVER" "$N" 1 1 "$GPU_LOG" "$GPU_TIME"

            echo "Saved GPU run log : $GPU_LOG"
            echo "Saved GPU time log: $GPU_TIME"

            echo ""
            echo "===== MPI RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"
            MPI_TIME="logs/mpi_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.time"

            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            run_mpi_command "$MPI_TIME" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$MPI_LOG"
            append_summary_row "$SUMMARY" "mpi" "MPI" "$CASE" "$SOLVER" "$N" "$MPI_RANKS" "$OMP_THREADS_MPI" "$MPI_LOG" "$MPI_TIME"

            echo "Saved MPI run log : $MPI_LOG"
            echo "Saved MPI time log: $MPI_TIME"

            echo ""
            echo "Current summary tail:"
            tail -n 6 "$SUMMARY" || true

        done
    done
done

echo ""
echo "===== SUMMARY CSV ====="
cat "$SUMMARY"

echo ""
echo "Timing summary saved to:"
echo "$SUMMARY"

echo ""
echo "===== END ====="
date
