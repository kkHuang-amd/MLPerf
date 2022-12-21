#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J resnet50
#Output and error (also --output, --error):
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#Initial working directory (also --chdir):
#SBATCH -D ./
# Wall clock limit:
#SBATCH --no-requeue
# Setup of execution environment
#SBATCH --get-user-env
#SBATCH --exclusive
# Resource configuration
#SBATCH --nodes=4
#SBATCH --partition=MI250
#SBATCH --partition=MI250-x4-IB
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:59:00
#SBATCH --gpus-per-node=8
$@
