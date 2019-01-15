#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

games = ["Pitfall-v0"] 
encoders = ["rand_cnn"]
label_names = ["x_coord"]
lrs = [0.01]
main_file = "main.py"
mode = "eval"
seed = 6 
for game in games:
    for label_name in label_names:
            for lr in lrs:
                for enc in encoders:
                    args = ["sbatch",
                            "./submit_scripts/run_gpu.sl",
                            "%s --model_name %s --env_name %s --label_name %s --mode %s"%(main_file,enc,game,label_name, mode),
                            "--tr_size %i --val_size %i --batch_size %i"%(10000,1000,64), "--lr %f"%(lr), "--seed %i"%(seed)]
                    print(" ".join(args))
                    subprocess.run(args)
                    
                    
                    

        

