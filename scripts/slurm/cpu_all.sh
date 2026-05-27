#!/bin/bash -l
#SBATCH -J sys_eq_cpu_mpi_clean
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# CPU OpenMP + pure MPI clean benchmark script for CSD3 CPU nodes.
#
# Runs all combinations of:
#   SCALES  : 1 2 4 8 16 by default
#   CASES   : shock_bubble blast_wave
#   SOLVERS : hll hllc exact force
#
# This script assumes main_cpu and main_mpi are clean/no-internal-timing binaries.
# Wall time is measured externally using /usr/bin/time.
#
# Default behavior on one exclusive CPU node:
#   cpu_omp : 1 process x all detected cores
#   mpi     : all detected cores x 1 MPI rank/core
#
# Usage:
#   sbatch -A YOUR_CPU_ACCOUNT scripts/slurm/run_cpu_mpi_clean_compare.sh
#
# Optional overrides:
#   CPU_PARTITION: edit #SBATCH -p above, e.g. sapphire or icelake.
#   SCALES_STR="1 4 8 16" sbatch -A YOUR_CPU_ACCOUNT scripts/slurm/run_cpu_mpi_clean_compare.sh
#   MPI_RANKS=56 OMP_THREADS_CPU=56 sbatch -A YOUR_CPU_ACCOUNT scripts/slurm/run_cpu_mpi_clean_compare.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${SLURM_SUBMIT_DIR}"
cd "${WORKDIR}"

mkdir -p logs outputs validation scaling

# -------------------------
# Benchmark matrix
# -------------------------
SCALES_STR="${SCALES_STR:-1 2 4 8 16}"
CASES_STR="${CASES_STR:-shock_bubble blast_wave}"
SOLVERS_STR="${SOLVERS_STR:-hll hllc exact force}"

read -r -a SCALES <<< "${SCALES_STR}"
read -r -a CASES <<< "${CASES_STR}"
read -r -a SOLVERS <<< "${SOLVERS_STR}"

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

# -------------------------
# Job info
# -------------------------
echo ""
echo "===== JOB INFO ====="
echo "JobID               : ${SLURM_JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date)"
echo "Workdir             : ${WORKDIR}"
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "Nodes               : ${SLURM_JOB_NUM_NODES:-1}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "Detected CPUs/node  : ${CPUS_ON_NODE}"
echo "MPI_RANKS           : ${MPI_RANKS}"
echo "OMP_THREADS_CPU     : ${OMP_THREADS_CPU}"
echo "OMP_THREADS_MPI     : ${OMP_THREADS_MPI}"
echo "MPI total CPU units : ${MPI_TOTAL_CPU_UNITS}"
echo "SCALES              : ${SCALES[*]}"
echo "CASES               : ${CASES[*]}"
echo "SOLVERS             : ${SOLVERS[*]}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
echo ""

echo "which g++    : $(which g++ || true)"
echo "which mpicxx : $(which mpicxx || true)"
echo "which mpirun : $(which mpirun || true)"
echo ""

# -------------------------
# Build
# -------------------------
echo "===== BUILD ====="
if [ "${MAKE_CLEAN:-0}" = "1" ]; then
    make clean
fi
make cpu
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

extract_time_value() {
    local key="$1"
    local log_file="$2"
    awk -F '=' -v key="$key" '
        index($0, "[TIME] " key "=") > 0 {
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            print $2;
            exit
        }
    ' "$log_file"
}

write_summary_header() {
    local summary_file="$1"
    echo "arch,case,solver,n,ranks,threads_per_rank,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,exit_status,log_file" > "$summary_file"
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
    local exit_status="$9"
    local log_file="${10}"

    local nx ny cells steps real_s user_s sys_s rss
    nx=$(extract_value_colon "[${prefix}] nx" "$log_file" || true)
    ny=$(extract_value_colon "[${prefix}] ny" "$log_file" || true)
    cells=$(extract_value_colon "[${prefix}] total_cells" "$log_file" || true)
    steps=$(extract_steps "$prefix" "$log_file" || true)
    real_s=$(extract_time_value "real_seconds" "$log_file" || true)
    user_s=$(extract_time_value "user_seconds" "$log_file" || true)
    sys_s=$(extract_time_value "sys_seconds" "$log_file" || true)
    rss=$(extract_time_value "max_rss_kb" "$log_file" || true)

    echo "${arch},${case_name},${solver_name},${n_scale},${ranks},${threads},${nx},${ny},${cells},${steps},${real_s},${user_s},${sys_s},${rss},${exit_status},${log_file}" >> "$summary_file"
}

run_and_log() {
    local log_file="$1"
    shift

    set +e
    /usr/bin/time \
        -f "[TIME] real_seconds=%e\n[TIME] user_seconds=%U\n[TIME] sys_seconds=%S\n[TIME] max_rss_kb=%M" \
        "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e

    return "$status"
}

run_mpi_and_log() {
    local log_file="$1"
    shift

    set +e
    /usr/bin/time \
        -f "[TIME] real_seconds=%e\n[TIME] user_seconds=%U\n[TIME] sys_seconds=%S\n[TIME] max_rss_kb=%M" \
        mpirun \
            --host "$(hostname):${MPI_RANKS}" \
            -np "${MPI_RANKS}" \
            --map-by core \
            --bind-to core \
            "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e

    return "$status"
}

SUMMARY="validation/cpu_mpi_clean_timing_${SLURM_JOB_ID}.csv"
write_summary_header "$SUMMARY"

# -------------------------
# Runs
# -------------------------
echo ""
echo "===== CPU OMP + MPI CLEAN RUNS ====="

for N in "${SCALES[@]}"; do
    echo ""
    echo "============================================================"
    echo "===== STARTING SCALE N=${N} ====="
    echo "============================================================"

    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            echo ""
            echo "===== CPU OMP RUN: case=${CASE}, solver=${SOLVER}, n=${N}, threads=${OMP_THREADS_CPU} ====="
            CPU_LOG="logs/cpu_omp_clean_${CASE}_${SOLVER}_n${N}_t${OMP_THREADS_CPU}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_CPU}"
            set +e
            run_and_log "$CPU_LOG" ./main_cpu "$N" --case "$CASE" --solver "$SOLVER"
            STATUS=$?
            set -e

            append_summary_row "$SUMMARY" "cpu_omp" "CPU" "$CASE" "$SOLVER" "$N" 1 "$OMP_THREADS_CPU" "$STATUS" "$CPU_LOG"

            if [ "$STATUS" -ne 0 ]; then
                echo "[ERROR] CPU OMP run failed with status ${STATUS}. See ${CPU_LOG}."
                exit "$STATUS"
            fi

            echo "Saved CPU OMP log: ${CPU_LOG}"

            echo ""
            echo "===== MPI RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_clean_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"

            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            set +e
            run_mpi_and_log "$MPI_LOG" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER"
            STATUS=$?
            set -e

            append_summary_row "$SUMMARY" "mpi" "MPI" "$CASE" "$SOLVER" "$N" "$MPI_RANKS" "$OMP_THREADS_MPI" "$STATUS" "$MPI_LOG"

            if [ "$STATUS" -ne 0 ]; then
                echo "[ERROR] MPI run failed with status ${STATUS}. See ${MPI_LOG}."
                exit "$STATUS"
            fi

            echo "Saved MPI log: ${MPI_LOG}"
            tail -n 6 "$SUMMARY" || true
        done
    done
done

echo ""
echo "===== CPU/MPI SUMMARY ====="
cat "$SUMMARY"
echo ""
echo "Summary CSV: $SUMMARY"
echo "===== END ====="
date
