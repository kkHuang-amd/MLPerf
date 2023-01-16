#!/bin/bash

#SBATCH --job-name="bert"
#SBATCH --output="bert_n8_g128_%j.out"
#SBATCH --partition=4CN512C32G4H_4IB_MI250_Ubuntu20 ##need to modify for hyabusa system
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=128                         ##need to modify for hyabusa system
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=16
#SBATCH --time=00:40:00

ulimit -n 1000000
export DATA_DIR=/mnt/beegfs/scratch/kk/wiki_20200101 ##need to modify for hyabusa system
export PATH=~/.local/bin:$PATH
source config_HYABUSA-MI250X_8x128x48x1.sh

if [ $# -eq 4 ]
then
    export SEED=$4
    echo "$1 $2 $3 $4"
    srun -N 8 $1 $2 $3
else
    srun -N 8 $@
fi
