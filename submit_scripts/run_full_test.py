#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

games = ["Pitfall-v0", "PrivateEye-v0"]
encoders = ["rand_cnn" ]
label_names = ["x_coord"] #, "y_coord", "on_ladder"]
#eval_modes = ["infer","predict"]
main_file = "main.py"
mode = "test"
seed = 7 
for game in games:
    for label_name in label_names:
                for enc in encoders:
                    args = ["sbatch",
                            "./submit_scripts/run_gpu.sl",
                            "%s --model_name %s --env_name %s --label_name %s --mode %s"%(main_file,enc,game,label_name, mode),
                            "--test_size %i --batch_size %i"%(1000,64),"--seed %i"%(seed)]
                    print(" ".join(args))
                    subprocess.run(args)