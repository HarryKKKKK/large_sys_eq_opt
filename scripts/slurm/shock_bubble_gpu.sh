#!/bin/bash
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J sys_eq_nsys_qdrep
#SBATCH -o /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.log
#SBATCH -e /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.log

set -euo pipefail

WORKDIR=/rds/user/hk597/hpc-work/large_sys_eq_opt
cd "$WORKDIR"

mkdir -p logs outputs profiles tmp

PROFDIR="$WORKDIR/profiles"
TMPDIR_JOB="$WORKDIR/tmp/nsys_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR_JOB"
export TMPDIR="$TMPDIR_JOB"

echo "===== JOB INFO ====="
echo "JobID:    ${SLURM_JOB_ID}"
echo "JobName:  ${SLURM_JOB_NAME}"
echo "Host:     $(hostname)"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Start:    $(date)"
echo "Workdir:  $(pwd)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "TMPDIR:   ${TMPDIR}"

echo ""
echo "===== MODULES ====="
module purge
module load rhel8/default-amp
module list 2>&1 || true

echo ""
echo "===== TOOL CHECK ====="
which nvcc || true
nvcc --version || true
which nsys || true
nsys --version || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo ""
echo "===== BUILD ====="
make clean
make

echo ""
echo "===== CLEAN OLD NSYS OUTPUTS FOR THIS JOB ====="
NSYS_OUT="${PROFDIR}/nsys_gpu_${SLURM_JOB_ID}"
rm -f "${NSYS_OUT}"*

echo ""
echo "===== NSYS PROFILE: QDREP-FIRST MODE ====="
echo "Goal: generate .qdrep or .nsys-rep"
echo "Important: stats disabled to avoid extra post-processing failure."

nsys profile \
    --stats=false \
    --sample=none \
    --trace=cuda,nvtx,osrt \
    --force-overwrite true \
    -o "${NSYS_OUT}" \
    ./main_gpu

echo ""
echo "===== NSYS OUTPUT FILES ====="
ls -lh "${PROFDIR}"/nsys_gpu_"${SLURM_JOB_ID}"* || true

echo ""
echo "===== NSYS REPORT CHECK ====="

if [ -f "${NSYS_OUT}.qdrep" ]; then
    echo "[OK] qdrep generated:"
    ls -lh "${NSYS_OUT}.qdrep"

elif [ -f "${NSYS_OUT}.nsys-rep" ]; then
    echo "[OK] nsys-rep generated:"
    ls -lh "${NSYS_OUT}.nsys-rep"

elif [ -f "${NSYS_OUT}.qdstrm" ]; then
    echo "[WARNING] Only qdstrm generated:"
    ls -lh "${NSYS_OUT}.qdstrm"
    echo ""
    echo "Trying manual import with QdstrmImporter if available..."

    NSYS_BIN="$(which nsys || true)"
    NSYS_DIR="$(dirname "$NSYS_BIN")"

    echo "nsys path: ${NSYS_BIN}"
    echo "nsys dir : ${NSYS_DIR}"

    echo ""
    echo "Searching for QdstrmImporter..."
    find "$(dirname "$NSYS_DIR")" -name "QdstrmImporter" -type f 2>/dev/null || true

    IMPORTER="$(find "$(dirname "$NSYS_DIR")" -name "QdstrmImporter" -type f 2>/dev/null | head -n 1 || true)"

    if [ -n "${IMPORTER}" ]; then
        echo "Found importer: ${IMPORTER}"
        echo "Running manual qdstrm import..."

        "${IMPORTER}" \
            -i "${NSYS_OUT}.qdstrm" \
            -o "${NSYS_OUT}.qdrep" || true

        echo ""
        echo "Files after manual import:"
        ls -lh "${PROFDIR}"/nsys_gpu_"${SLURM_JOB_ID}"* || true
    else
        echo "QdstrmImporter was not found in the Nsight Systems installation."
    fi

else
    echo "[ERROR] No Nsight Systems output found."
fi

echo ""
echo "===== FINAL PROFILE FILE SEARCH ====="
find "$PROFDIR" \
    \( -name "*.qdrep" -o -name "*.nsys-rep" -o -name "*.qdstrm" \) \
    -type f \
    -ls || true

echo ""
echo "===== END ====="
date