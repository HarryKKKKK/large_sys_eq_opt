#!/bin/bash
#SBATCH --exclude=gpu-q-43
#SBATCH -J sys_eq_gpu_nsys_check
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.out
#SBATCH --error=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.err

set -euo pipefail

cd /home/hk597/rds/hpc-work/large_sys_eq_opt

mkdir -p logs outputs profiles

NSYS="/usr/local/software/cuda/11.4/bin/nsys"
JOB_TAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

PREFLIGHT_STDOUT="logs/${JOB_TAG}_preflight_stdout.txt"
PREFLIGHT_STDERR="logs/${JOB_TAG}_preflight_stderr.txt"

NSYS_STDOUT="logs/${JOB_TAG}_nsys_stdout.txt"
NSYS_STDERR="logs/${JOB_TAG}_nsys_stderr.txt"

PROFILE_PREFIX="profiles/${JOB_TAG}"

echo "=========================================="
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Job name            : ${SLURM_JOB_NAME}"
echo "Node                : ${SLURMD_NODENAME}"
echo "Start time          : $(date)"
echo "Working directory   : $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "=========================================="

echo
echo "[1/5] Checking GPU allocation..."
nvidia-smi

echo
echo "[2/5] Checking Nsight Systems..."
if [ ! -x "$NSYS" ]; then
    echo "ERROR: nsys not found at $NSYS"
    exit 1
fi

echo "Using nsys: $NSYS"
"$NSYS" --version

echo
echo "[3/5] Cleaning old profile files for this job..."
rm -f "${PROFILE_PREFIX}"*

echo
echo "[4/5] Running preflight without Nsight..."
if ./main_gpu > "$PREFLIGHT_STDOUT" 2> "$PREFLIGHT_STDERR"; then
    echo "Preflight run: SUCCESS"
else
    echo "Preflight run: FAILED"
    echo "See:"
    echo "  $PREFLIGHT_STDOUT"
    echo "  $PREFLIGHT_STDERR"
    exit 1
fi

echo
echo "[5/5] Running with Nsight Systems..."
if "$NSYS" profile \
    --trace=cuda \
    --sample=none \
    --stats=true \
    --force-overwrite=true \
    -o "$PROFILE_PREFIX" \
    ./main_gpu \
    > "$NSYS_STDOUT" \
    2> "$NSYS_STDERR"; then
    echo "Nsight run: SUCCESS"
else
    echo "Nsight run: FAILED"
    echo "See:"
    echo "  $NSYS_STDOUT"
    echo "  $NSYS_STDERR"
    exit 1
fi

echo
echo "=========================================="
echo "Generated logs:"
echo "  Preflight stdout : $PREFLIGHT_STDOUT"
echo "  Preflight stderr : $PREFLIGHT_STDERR"
echo "  Nsight stdout    : $NSYS_STDOUT"
echo "  Nsight stderr    : $NSYS_STDERR"

echo
echo "Generated profile files:"
ls -lh "${PROFILE_PREFIX}"* || true


echo
echo "End time: $(date)"
echo "=========================================="