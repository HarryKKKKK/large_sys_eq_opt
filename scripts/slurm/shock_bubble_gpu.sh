#!/bin/bash
#SBATCH -J sys_eq_gpu_base
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.out
#SBATCH --error=/home/hk597/rds/hpc-work/large_sys_eq_opt/logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/large_sys_eq_opt
mkdir -p logs outputs

./main_gpu | tee outputs/gpu_stdout.txt