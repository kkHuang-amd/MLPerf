#!/bin/bash

#SBATCH --job-name="bert"
#SBATCH --output="bert_n2_g16_%j.out"
#SBATCH --partition=MI250-x4-IB
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:29:00

ulimit -n 1000000
export DATA_DIR=/global/scratch/mlperf_datasets/wiki_20200101
export PATH=~/.local/bin:$PATH
source config_ACP-MI250_2x8x56x1.sh

srun -N 2 $@
