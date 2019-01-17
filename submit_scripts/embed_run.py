#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
games = ["originalGame-v0"]#["FlappyBirdDay-v0"]#, "originalGame-v0"]#,"PrivateEye-v0"
encoders = ["vae","inv_model","tdc","snl"]
lrs = [0.001]
main_file = "main.py"
mode= "train"
task="embed"
comet_mode = "online"
seed = 4
for game in games:
    for lr in lrs:
        for enc in encoders:
            args = ["sbatch",
                    #"-w %s"%("eos3"),
                    "./submit_scripts/run_gpu.sl",
                    "%s"%(main_file),
                    "--embedder_name %s"%enc,
                    "--embed_env %s"%game,
                    "--mode %s"%mode,
                    "--task %s"%task,
                    "--embed_env %s"%(game),
                    "--comet_mode %s"%(comet_mode),
                    "--tr_size %i"%10000,
                    "--val_size %i"%1000,
                    "--batch_size %i"%64,
                    "--lr %f"%(lr),
                    "--seed %i"%(seed)]
            print(" ".join(args))
            subprocess.run(args)
        
