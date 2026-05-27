#!/bin/bash -l
#SBATCH -J sys_eq_part2_fullrank
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Part 2: CPU OpenMP / GPU / pure MPI timing comparison
#
# This version is designed to maximise CPU/MPI usage inside the
# allocated node:
#   CPU OpenMP : 1 process x all allocated physical CPU cores
#   MPI        : all allocated physical CPU cores x 1 thread/rank
#   GPU        : 1 GPU
#
# IMPORTANT:
#   --exclusive is used so the job can see/use the whole node.
#   MPI_RANKS defaults to the detected number of CPUs on the node.
#   OMP_THREADS_CPU defaults to the detected number of CPUs on the node.
#
# Optional overrides:
#   MPI_RANKS=76 OMP_THREADS_CPU=76 sbatch result_compare_part2_fullrank.sh
#   MPI_RANKS=32 OMP_THREADS_CPU=32 sbatch result_compare_part2_fullrank.sh
#
# For a true CPU-cluster full-power run, submit a CPU-only version on
# icelake/sapphire instead of csc-mphil-gpu.
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
SCALES=(1 2 4 8 16)
CASES=("shock_bubble" "blast_wave")
SOLVERS=("hll" "hllc" "exact" "force")

# For a quick test, uncomment:
# SCALES=(1)
# CASES=("shock_bubble")
# SOLVERS=("hll")

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
    # SLURM_CPUS_ON_NODE is usually the best value inside an allocation.
    if [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
        echo "${SLURM_CPUS_ON_NODE}"
        return
    fi

    # SLURM_JOB_CPUS_PER_NODE may look like "76" or "76(x2)".
    if [ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]; then
        echo "${SLURM_JOB_CPUS_PER_NODE}" | sed -E 's/\(x[0-9]+\)//'
        return
    fi

    # Fallback for manual/local runs.
    nproc
}

CPUS_ON_NODE="$(detect_cpus_on_node)"

# Default: pure MPI uses one rank per CPU core.
MPI_RANKS="${MPI_RANKS:-${CPUS_ON_NODE}}"

# Default: OpenMP CPU uses all CPU cores.
OMP_THREADS_CPU="${OMP_THREADS_CPU:-${CPUS_ON_NODE}}"

# Default: pure MPI uses one OpenMP thread per MPI rank.
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

# Binding choices:
#   OpenMP: bind threads to cores.
#   MPI: bind ranks to cores.
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
echo "SCALES               : ${SCALES[*]}"
echo "CASES                : ${CASES[*]}"
echo "SOLVERS              : ${SOLVERS[*]}"
echo "MPI_RANKS            : ${MPI_RANKS}"
echo "OMP_THREADS_CPU      : ${OMP_THREADS_CPU}"
echo "OMP_THREADS_MPI      : ${OMP_THREADS_MPI}"
echo "MPI total CPU units  : ${MPI_TOTAL_CPU_UNITS}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
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
run_mpi_command() {
    local log_file="$1"
    shift

    # Some OpenMPI builds on this cluster cannot be launched directly by srun
    # because they were not built with Slurm PMI support. We therefore use
    # mpirun inside the Slurm allocation.
    #
    # --host "$(hostname):${MPI_RANKS}" tells OpenMPI that the current node has
    # exactly MPI_RANKS available slots. This avoids OpenMPI incorrectly seeing
    # only one slot when the Slurm allocation was obtained with --exclusive.
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
    local ranks="$7"
    local threads="$8"
    local log_file="$9"

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

    echo "${arch},${case_name},${solver_name},${n_scale},${ranks},${threads},${nx},${ny},${cells},${steps},${main_loop},${boundary},${compute_dt},${advance},${total_program},${avg_step},${log_file}" >> "$summary_file"
}

write_summary_header() {
    local summary_file="$1"
    echo "arch,case,solver,n,ranks,threads_per_rank,nx,ny,total_cells,total_steps,main_loop_time,boundary_time,compute_dt_time,advance_time,total_program_time,avg_time_per_step,log_file" > "$summary_file"
}

TIMING_SUMMARY="validation/cpu_gpu_mpi_timing_scales_fullrank_${SLURM_JOB_ID}.csv"
write_summary_header "$TIMING_SUMMARY"

echo ""
echo "===== PART 2: CPU/GPU/MPI TIMING RUNS ACROSS SCALES ====="

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
            append_summary_row "$TIMING_SUMMARY" "cpu_omp" "CPU" "$CASE" "$SOLVER" "$N" 1 "$OMP_THREADS_CPU" "$CPU_LOG"

            echo "Saved CPU timing log: $CPU_LOG"

            echo ""
            echo "===== GPU TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="
            GPU_LOG="logs/gpu_timing_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS=1
            ./main_gpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$GPU_LOG"
            append_summary_row "$TIMING_SUMMARY" "gpu" "GPU" "$CASE" "$SOLVER" "$N" 1 1 "$GPU_LOG"

            echo "Saved GPU timing log: $GPU_LOG"

            echo ""
            echo "===== MPI TIMING RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_timing_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            run_mpi_command "$MPI_LOG" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER"
            append_summary_row "$TIMING_SUMMARY" "mpi" "MPI" "$CASE" "$SOLVER" "$N" "$MPI_RANKS" "$OMP_THREADS_MPI" "$MPI_LOG"

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
