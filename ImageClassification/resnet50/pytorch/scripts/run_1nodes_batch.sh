#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J test
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
#SBATCH --partition=MI250-ubuntu20
# Resource configuration
#SBATCH --nodes=1
#SBATCH --partition=MI250-x4-IB
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:59:00
srun -N 1  $@
