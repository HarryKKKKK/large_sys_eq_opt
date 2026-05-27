#!/bin/bash -l
#SBATCH -J sys_eq_gpu_clean
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# GPU-only clean benchmark script for CSD3 Ampere GPU nodes.
#
# Runs all combinations of:
#   SCALES  : 1 2 4 8 16 by default
#   CASES   : shock_bubble blast_wave
#   SOLVERS : hll hllc exact force
#
# This script assumes main_gpu is a clean/no-internal-timing binary.
# Wall time is measured externally using /usr/bin/time.
#
# Usage:
#   sbatch scripts/slurm/run_gpu_clean_compare.sh
#
# Optional overrides:
#   SCALES_STR="1 4 8 16" sbatch scripts/slurm/run_gpu_clean_compare.sh
#   CASES_STR="shock_bubble" SOLVERS_STR="hll hllc" sbatch scripts/slurm/run_gpu_clean_compare.sh
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

# Keep CPU side light for GPU runs.
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# -------------------------
# Job info
# -------------------------
echo ""
echo "===== JOB INFO ====="
echo "JobID              : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "Start              : $(date)"
echo "Workdir            : ${WORKDIR}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_CPUS_ON_NODE : ${SLURM_CPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCALES             : ${SCALES[*]}"
echo "CASES              : ${CASES[*]}"
echo "SOLVERS            : ${SOLVERS[*]}"
echo ""

echo "===== CPU/GPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
nvidia-smi || true
echo ""

echo "which g++  : $(which g++ || true)"
echo "which nvcc : $(which nvcc || true)"
echo ""

# -------------------------
# Build
# -------------------------
echo "===== BUILD ====="
if [ "${MAKE_CLEAN:-0}" = "1" ]; then
    make clean
fi
make gpu
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
    echo "arch,case,solver,n,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,exit_status,log_file" > "$summary_file"
}

append_summary_row() {
    local summary_file="$1"
    local arch="$2"
    local prefix="$3"
    local case_name="$4"
    local solver_name="$5"
    local n_scale="$6"
    local exit_status="$7"
    local log_file="$8"

    local nx ny cells steps real_s user_s sys_s rss
    nx=$(extract_value_colon "[${prefix}] nx" "$log_file" || true)
    ny=$(extract_value_colon "[${prefix}] ny" "$log_file" || true)
    cells=$(extract_value_colon "[${prefix}] total_cells" "$log_file" || true)
    steps=$(extract_steps "$prefix" "$log_file" || true)
    real_s=$(extract_time_value "real_seconds" "$log_file" || true)
    user_s=$(extract_time_value "user_seconds" "$log_file" || true)
    sys_s=$(extract_time_value "sys_seconds" "$log_file" || true)
    rss=$(extract_time_value "max_rss_kb" "$log_file" || true)

    echo "${arch},${case_name},${solver_name},${n_scale},${nx},${ny},${cells},${steps},${real_s},${user_s},${sys_s},${rss},${exit_status},${log_file}" >> "$summary_file"
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

SUMMARY="validation/gpu_clean_timing_${SLURM_JOB_ID}.csv"
write_summary_header "$SUMMARY"

# -------------------------
# Runs
# -------------------------
echo ""
echo "===== GPU CLEAN RUNS ====="

for N in "${SCALES[@]}"; do
    echo ""
    echo "============================================================"
    echo "===== STARTING SCALE N=${N} ====="
    echo "============================================================"

    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            echo ""
            echo "===== GPU RUN: case=${CASE}, solver=${SOLVER}, n=${N} ====="
            LOG="logs/gpu_clean_${CASE}_${SOLVER}_n${N}_${SLURM_JOB_ID}.log"

            set +e
            run_and_log "$LOG" ./main_gpu "$N" --case "$CASE" --solver "$SOLVER"
            STATUS=$?
            set -e

            append_summary_row "$SUMMARY" "gpu" "GPU" "$CASE" "$SOLVER" "$N" "$STATUS" "$LOG"

            if [ "$STATUS" -ne 0 ]; then
                echo "[ERROR] GPU run failed with status ${STATUS}. See ${LOG}."
                exit "$STATUS"
            fi

            echo "Saved GPU log: ${LOG}"
            tail -n 5 "$SUMMARY" || true
        done
    done
done

echo ""
echo "===== GPU SUMMARY ====="
cat "$SUMMARY"
echo ""
echo "Summary CSV: $SUMMARY"
echo "===== END ====="
date
