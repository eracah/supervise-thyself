#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

games = ["Pitfall-v0", "PrivateEye-v0"] 
encoders = ["inv_model", "vae"]
lrs = [0.0001, 0.00001]
main_file = "main.py"
mode= "train"
seed = 4
for game in games:
    for lr in lrs:
        for enc in encoders:
            args = ["sbatch", "./submit_scripts/run_gpu.sl","%s --model_name %s --env_name %s --mode %s"%(main_file,enc,game,mode),"--tr_size %i --val_size %i --batch_size %i"%(10000,1000,64), "--lr %f"%(lr),"--seed %i"%(seed)]
            print(" ".join(args))
            subprocess.run(args)
        

