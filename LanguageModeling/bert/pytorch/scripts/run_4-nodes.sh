#!/bin/bash

#SBATCH --job-name="bert"
#SBATCH --output="bert_n4_g32_%j.out"
#SBATCH --partition=MI250-x4-IB
#SBATCH --nodes=4
#SBATCH --partition=MI250-x4-IB
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00

ulimit -n 100000
export DATA_DIR=/global/scratch/mlperf_datasets/wiki_20200101
srun -N 4 $@ 
