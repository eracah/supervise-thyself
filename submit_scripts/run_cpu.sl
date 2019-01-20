#!/bin/bash -l
#SBATCH -c 1
#SBATCH -t 30
#SBATCH -o batch_outputs/slurm-%j.out
#SBATCH -e batch_outputs/slurm-%j.out
filename=$1
shift
python $filename  $@
