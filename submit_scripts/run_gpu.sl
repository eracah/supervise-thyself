#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH -t 960
#SBATCH -o batch_outputs/slurm-%j.out
#SBATCH -e batch_outputs/slurm-%j.out
filename=$1
shift
python $filename  $@
