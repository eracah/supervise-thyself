#!/bin/bash -l
#SBATCH -c 1
#SBATCH -t 120
#SBATCH -o batch_outputs/slurm-%j.out
#SBATCH -e batch_outputs/slurm-%j.out
filename=$1
shift
xvfb-run -s "-screen 0 1400x900x24" python $filename  $@
