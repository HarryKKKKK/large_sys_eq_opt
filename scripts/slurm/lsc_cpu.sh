#!/bin/bash -l
#SBATCH -J sys_eq_cpu_mpi_full_1
#SBATCH -A hk597
#SBATCH -p csc-mphil
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

mkdir -p logs validation scaling outputs

# Override like:
# SCALES_STR="1" CASES_STR="shock_bubble" SOLVERS_STR="hll" sbatch lsc_cpu_mpi_fullnode_clean_compare.sh
read -r -a SCALES <<< "${SCALES_STR:-1 2 4 8}"
read -r -a CASES  <<< "${CASES_STR:-shock_bubble blast_wave}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc exact force}"

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
OMP_THREADS_CPU="${OMP_THREADS_CPU:-${CPUS_ON_NODE}}"
MPI_RANKS="${MPI_RANKS:-${CPUS_ON_NODE}}"
OMP_THREADS_MPI="${OMP_THREADS_MPI:-1}"
MPI_TOTAL_CPU_UNITS=$((MPI_RANKS * OMP_THREADS_MPI))

if [ "$OMP_THREADS_CPU" -gt "$CPUS_ON_NODE" ]; then
    echo "[ERROR] OMP_THREADS_CPU=${OMP_THREADS_CPU} > CPUS_ON_NODE=${CPUS_ON_NODE}"
    exit 1
fi
if [ "$MPI_TOTAL_CPU_UNITS" -gt "$CPUS_ON_NODE" ]; then
    echo "[ERROR] MPI_RANKS*OMP_THREADS_MPI=${MPI_TOTAL_CPU_UNITS} > CPUS_ON_NODE=${CPUS_ON_NODE}"
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID              : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "Start              : $(date)"
echo "Workdir            : ${WORKDIR}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_CPUS_ON_NODE : ${SLURM_CPUS_ON_NODE:-unset}"
echo "Detected CPUs/node : ${CPUS_ON_NODE}"
echo "OMP_THREADS_CPU    : ${OMP_THREADS_CPU}"
echo "MPI_RANKS          : ${MPI_RANKS}"
echo "OMP_THREADS_MPI    : ${OMP_THREADS_MPI}"
echo "MPI total CPU units: ${MPI_TOTAL_CPU_UNITS}"
echo "SCALES             : ${SCALES[*]}"
echo "CASES              : ${CASES[*]}"
echo "SOLVERS            : ${SOLVERS[*]}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true

echo ""
echo "===== ENV CHECK ====="
echo "which g++   : $(which g++ || true)"
echo "which mpicxx: $(which mpicxx || true)"
echo "which mpirun: $(which mpirun || true)"
echo "which time  : $(which time || true)"
if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found in PATH."
    exit 1
fi
if ! command -v mpicxx >/dev/null 2>&1; then
    echo "[ERROR] mpicxx not found in PATH."
    exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] mpirun not found in PATH."
    exit 1
fi

# Keep these for reproducible OpenMP placement.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo ""
echo "===== BUILD ====="
make clean
make cpu
make mpi

SUMMARY="validation/cpu_mpi_fullnode_external_timing_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,ranks,threads_per_rank,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,run_log,time_log" > "$SUMMARY"

extract_common() {
    local prefix="$1"
    local run_log="$2"
    local nx ny cells steps
    nx=$(awk -F ':' -v p="[$prefix]" 'index($0,p" nx")>0 {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$run_log")
    ny=$(awk -F ':' -v p="[$prefix]" 'index($0,p" ny")>0 {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$run_log")
    cells=$(awk -F ':' -v p="[$prefix]" 'index($0,p" total_cells")>0 {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$run_log")
    steps=$(awk -F '=' -v p="[$prefix]" 'index($0,p" Total steps")>0 {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$run_log")
    echo "${nx},${ny},${cells},${steps}"
}

append_row() {
    local arch="$1" case_name="$2" solver_name="$3" n_scale="$4" ranks="$5" threads="$6" prefix="$7" run_log="$8" time_log="$9"
    local common real user sys rss
    common=$(extract_common "$prefix" "$run_log")
    real=$(awk -F '=' '/real_seconds/{print $2; exit}' "$time_log")
    user=$(awk -F '=' '/user_seconds/{print $2; exit}' "$time_log")
    sys=$(awk -F '=' '/sys_seconds/{print $2; exit}' "$time_log")
    rss=$(awk -F '=' '/max_rss_kb/{print $2; exit}' "$time_log")
    echo "${arch},${case_name},${solver_name},${n_scale},${ranks},${threads},${common},${real},${user},${sys},${rss},${run_log},${time_log}" >> "$SUMMARY"
}

run_mpi_command() {
    local time_log="$1"
    local run_log="$2"
    shift 2

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$time_log" \
        mpirun --host "$(hostname):${MPI_RANKS}" -np "${MPI_RANKS}" --map-by core --bind-to core "$@" \
        2>&1 | tee "$run_log"
}

for N in "${SCALES[@]}"; do
    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            echo ""
            echo "===== CPU OMP RUN: case=${CASE}, solver=${SOLVER}, n=${N}, threads=${OMP_THREADS_CPU} ====="
            # CPU_LOG="logs/cpu_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.log"
            # CPU_TIME="logs/cpu_${CASE}_${SOLVER}_n${N}_omp${OMP_THREADS_CPU}_${SLURM_JOB_ID}.time"
            # export OMP_NUM_THREADS="${OMP_THREADS_CPU}"
            # /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
            #     -o "$CPU_TIME" \
            #     ./main_cpu "$N" --case "$CASE" --solver "$SOLVER" 2>&1 | tee "$CPU_LOG"
            # echo "----- CPU TIME -----"
            # cat "$CPU_TIME"
            # append_row "cpu_omp" "$CASE" "$SOLVER" "$N" 1 "$OMP_THREADS_CPU" "CPU" "$CPU_LOG" "$CPU_TIME"

            echo ""
            echo "===== MPI RUN: case=${CASE}, solver=${SOLVER}, n=${N}, ranks=${MPI_RANKS}, threads_per_rank=${OMP_THREADS_MPI} ====="
            MPI_LOG="logs/mpi_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.log"
            MPI_TIME="logs/mpi_${CASE}_${SOLVER}_n${N}_r${MPI_RANKS}_t${OMP_THREADS_MPI}_${SLURM_JOB_ID}.time"
            export OMP_NUM_THREADS="${OMP_THREADS_MPI}"
            run_mpi_command "$MPI_TIME" "$MPI_LOG" ./main_mpi "$N" --case "$CASE" --solver "$SOLVER"
            echo "----- MPI TIME -----"
            cat "$MPI_TIME"
            append_row "mpi" "$CASE" "$SOLVER" "$N" "$MPI_RANKS" "$OMP_THREADS_MPI" "MPI" "$MPI_LOG" "$MPI_TIME"
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
