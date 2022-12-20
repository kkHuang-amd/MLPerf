#!/bin/bash

#SBATCH --job-name="bert"
#SBATCH --output="bert_n4_g32_%j.out"
#SBATCH --partition=MI250-x4-IB
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00

ulimit -n 1000000
export DATA_DIR=/global/scratch/mlperf_datasets/wiki_20200101
export PATH=~/.local/bin:$PATH
source config_ACP-MI250_4x8x56x1.sh

srun -N 4 $@
