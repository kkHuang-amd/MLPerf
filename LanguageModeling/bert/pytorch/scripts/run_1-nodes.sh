#!/bin/bash

#SBATCH --job-name="bert"
#SBATCH --output="bert_n1_g8_%j.out"
#SBATCH --partition=4CN512C32G4H_4IB_MI250_Ubuntu20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00

ulimit -n 1000000
export DATA_DIR=/mnt/beegfs/scratch/kk/wiki_20200101
export PATH=~/.local/bin:$PATH
source config_ACP-MI250_1x8x56x1.sh

if [ $# -eq 4 ]
then
    export SEED=$4
    echo "$1 $2 $3 $4"
    srun -N 1 $1 $2 $3
else
    srun -N 1 $@
fi
