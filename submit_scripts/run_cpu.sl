#!/bin/bash -l
#SBATCH -c 4
#SBATCH -t 960
#SBATCH -o batch_outputs/slurm-%j.out
filename=$1
shift
python $filename  $@
