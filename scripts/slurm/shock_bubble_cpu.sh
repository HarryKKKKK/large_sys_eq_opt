#!/bin/bash
#SBATCH -J sys_eq_cpu_omp8
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=00:20:00
#SBATCH --output=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.out
#SBATCH --error=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/large_sys_eq_opt
mkdir -p logs outputs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true
export OMP_PLACES=cores

./main_cpu | tee outputs/cpu_stdout.txt