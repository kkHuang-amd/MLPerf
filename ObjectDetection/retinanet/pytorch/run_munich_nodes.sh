#!/bin/bash

#SBATCH --job-name=retinanet
#SBATCH --partition=1CN96C8G1H_MI250_Ubuntu20
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --output=retinanet-%j.out
#SBATCH --time=08:00:00

export DATASET_DIR="/mnt/beegfs/scratch/yuyun/datasets/retinanet"
export TORCH_HOME="$PWD/torch-home"
export LOG_DIR="$PWD/results"
#export DEBUG=1

# Determine master node
candidates=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export master_node=${candidates[0]}
export master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo "Master node: $master_node"
echo "Master IP: $master_node_ip"
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

srun -N 2 ./run_with_slurm.sh