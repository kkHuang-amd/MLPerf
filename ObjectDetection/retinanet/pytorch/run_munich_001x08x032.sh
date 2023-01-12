#!/bin/bash

#SBATCH --job-name=retinanet
#SBATCH --output=retinanet-%j_001x08x032.out
##SBATCH --partition=4CN512C32G4H_4IB_MI250_Ubuntu20
#SBATCH --partition=5CN512C40G5H_1IB_MI250_Ubuntu20
##SBATCH --partition=1CN96C8G1H_MI250_Ubuntu20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --exclusive

export DATASET_DIR="/mnt/beegfs/scratch/yuyun/datasets/retinanet"
export TORCH_HOME="$PWD/torch-home"
export LOG_DIR="$PWD/results"

module load rocm/5.4.0
module load pytorch/1.12_rocm_5.4

source config_MI250_001x08x032.sh

srun ./run_and_time.sh
