#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 1
#SBATCH -o batch_outputs/slurm-%j.out
#SBATCH -e batch_outputs/slurm-%j.out
filename=$1
shift
python $filename  $@
