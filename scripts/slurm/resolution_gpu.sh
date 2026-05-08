#!/bin/bash
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J sys_eq_gpu_scaling
#SBATCH -o /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.log
#SBATCH -e /rds/user/hk597/hpc-work/large_sys_eq_opt/logs/%x_%j.log

set -euo pipefail

WORKDIR=/rds/user/hk597/hpc-work/large_sys_eq_opt
cd "$WORKDIR"

mkdir -p logs scaling

echo "===== JOB INFO ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo ""

module purge
module load rhel8/default-amp

echo "===== BUILD ====="
make clean
make gpu
echo ""

SUMMARY="scaling/gpu_scaling_timing_${SLURM_JOB_ID}.csv"

echo "n,nx,ny,total_cells,total_steps,main_loop_time,boundary_time,compute_dt_time,advance_time,total_program_time,avg_time_per_step" > "$SUMMARY"

for N in 1 2 4 8; do
    echo "===== RUN n=${N} ====="

    RUN_LOG="logs/gpu_scaling_n${N}_${SLURM_JOB_ID}.log"

    ./main_gpu "$N" 2>&1 | tee "$RUN_LOG"

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

    echo "${N},${NX},${NY},${CELLS},${STEPS},${MAIN_LOOP},${BOUNDARY},${COMPUTE_DT},${ADVANCE},${TOTAL_PROGRAM},${AVG_STEP}" >> "$SUMMARY"

    echo "Saved run log: $RUN_LOG"
    echo ""
done

echo "===== SUMMARY CSV ====="
cat "$SUMMARY"

echo ""
echo "Timing summary saved to:"
echo "$SUMMARY"

echo ""
echo "===== END ====="
date