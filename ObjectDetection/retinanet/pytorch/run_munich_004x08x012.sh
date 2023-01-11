#!/bin/bash

#SBATCH --job-name=retinanet
#SBATCH --output=retinanet-%j_004x08x012.out
#SBATCH --partition=4CN512C32G4H_4IB_MI250_Ubuntu20
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=05:00:00
#SBATCH --exclusive

export DATASET_DIR="/mnt/beegfs/scratch/yuyun/datasets/retinanet"
export TORCH_HOME="$PWD/torch-home"
export LOG_DIR="$PWD/results"

source config_MI250_004x08x012.sh

# Determine master node
master_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
export MASTER_PORT=23456
echo "master_node: $master_node"
echo "MASTER_ADDR: $MASTER_ADDR"
export LOGLEVEL=INFO

module load rocm/5.4.0
module load pytorch/1.12_rocm_5.4

srun ./run_and_time.sh
