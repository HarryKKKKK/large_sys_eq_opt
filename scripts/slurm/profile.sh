#!/bin/bash -l
#SBATCH -J gpu_nsight_profile
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

mkdir -p logs profiles

CASE="shock_bubble"
SOLVER="hll"
N=4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID               : ${SLURM_JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date)"
echo "Workdir             : ${WORKDIR}"
echo "Git Branch          : ${GIT_BRANCH}"
echo "Git Commit          : ${GIT_COMMIT}"
echo "Case                : ${CASE}"
echo "Solver              : ${SOLVER}"
echo "Scale               : ${N}"
echo ""

echo "===== GPU INFO ====="
nvidia-smi || true

echo ""
echo "===== BUILD ====="
make clean
make gpu

PROFILE_BASE="profiles/${GIT_BRANCH}_${SLURM_JOB_ID}"

echo ""
echo "===== KERNEL SYMBOLS ====="
cuobjdump -symbols main_gpu | grep -i "advance\|compute_block\|speed"

echo ""
echo "===== NSIGHT SYSTEMS ====="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --output="${PROFILE_BASE}_nsys" \
    --force-overwrite=true \
    ./main_gpu "$N" --case "$CASE" --solver "$SOLVER"

echo ""
echo "===== NSIGHT COMPUTE ====="
NCU_TMPDIR="${WORKDIR}/profiles/ncu_tmp"
mkdir -p "$NCU_TMPDIR"

for KERNEL in \
    "compute_block_max_speed_kernel" \
    "advance_x_reconstruct_smem_fused_kernel" \
    "advance_y_reconstruct_smem_fused_kernel"
do
    echo "--- Profiling kernel: ${KERNEL} ---"
    TMPDIR="$NCU_TMPDIR" ncu \
        --set full \
        --kernel-name-base function \
        --kernel-name "${KERNEL}" \
        --launch-count 3 \
        -o "${PROFILE_BASE}_ncu_${KERNEL}" \
        --force-overwrite \
        ./main_gpu "$N" --case "$CASE" --solver "$SOLVER"
done

echo ""
echo "===== PROFILE FILES ====="
ls -lh profiles/
echo ""
echo "===== END ====="
date