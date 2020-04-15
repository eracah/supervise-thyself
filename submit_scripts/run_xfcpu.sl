#!/bin/bash -l
<<<<<<< HEAD
#SBATCH -c 1
#SBATCH -t 120
=======
#SBATCH -c 2
#SBATCH -t 960
>>>>>>> parent of 0e2938f... fixed bug with pipes in flappybird
#SBATCH -o batch_outputs/slurm-%j.out
#SBATCH -e batch_outputs/slurm-%j.out
filename=$1
shift
xvfb-run -s "-screen 0 1400x900x24" python $filename  $@
