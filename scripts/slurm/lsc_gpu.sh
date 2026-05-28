#!/bin/bash -l
#SBATCH -J sys_eq_gpu_clean_fused_kernel_compute_dt_with_tiling
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

# 提取 Git 信息，方便后续溯源
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

mkdir -p logs validation scaling outputs

# Override like:
# SCALES_STR="1" CASES_STR="shock_bubble" SOLVERS_STR="hll" sbatch lsc_gpu_clean_compare.sh
read -r -a SCALES <<< "${SCALES_STR:-1 2 4 8}"
read -r -a CASES  <<< "${CASES_STR:-shock_bubble blast_wave}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc exact force}"

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
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_NTASKS        : ${SLURM_NTASKS:-unset}"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS     : ${OMP_NUM_THREADS}"
echo "SCALES              : ${SCALES[*]}"
echo "CASES               : ${CASES[*]}"
echo "SOLVERS             : ${SOLVERS[*]}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true

echo ""
echo "===== GPU INFO ====="
nvidia-smi || true

echo ""
echo "===== BUILD ====="
make clean
make gpu

SUMMARY="validation/gpu_clean_external_timing_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,git_branch,git_commit" > "$SUMMARY"

run_and_record() {
    local case_name="$1"
    local solver_name="$2"
    local n_scale="$3"

    echo ""
    echo "===== GPU RUN: case=${case_name}, solver=${solver_name}, n=${n_scale} ====="

    # 创建隐藏的临时文件（存储在系统的 /tmp 目录下），运行完即焚
    local temp_log=$(mktemp)
    local temp_time=$(mktemp)

    # 用 tee 保证 main_gpu 的输出能实时显示在 SLURM 的 out 文件里，同时存入临时文件供 awk 解析
    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./main_gpu "$n_scale" --case "$case_name" --solver "$solver_name" 2>&1 | tee "$temp_log"

    # 从临时文件中提取数据
    local nx ny cells steps real user sys rss
    nx=$(awk -F ':' '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    ny=$(awk -F ':' '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    cells=$(awk -F ':' '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    steps=$(awk -F '=' '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    
    real=$(awk -F '=' '/real_seconds/{print $2; exit}' "$temp_time")
    user=$(awk -F '=' '/user_seconds/{print $2; exit}' "$temp_time")
    sys=$(awk -F '=' '/sys_seconds/{print $2; exit}' "$temp_time")
    rss=$(awk -F '=' '/max_rss_kb/{print $2; exit}' "$temp_time")

    # 每次运行结束后，立刻把提取出来的 Timing 打印到主 log 里
    echo "------------------------------------------------------------"
    echo "[TIMING RECORDED] Real: ${real}s | User: ${user}s | Sys: ${sys}s | Max RSS: ${rss} KB"
    echo "------------------------------------------------------------"

    # 追加到 CSV
    echo "gpu,${case_name},${solver_name},${n_scale},${nx},${ny},${cells},${steps},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT}" >> "$SUMMARY"

    # 清理临时文件，你的文件夹依然干干净净
    rm -f "$temp_log" "$temp_time"
}

for N in "${SCALES[@]}"; do
    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            run_and_record "$CASE" "$SOLVER" "$N"
        done
    done
done

echo ""
echo "===== SUMMARY CSV ====="
cat "$SUMMARY"
echo ""
echo "Saved summary: $SUMMARY"
echo "===== END ====="
date