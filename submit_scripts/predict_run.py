#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
games = ["FlappyBirdDay-v0"]
levels=[None]#["FlappyBirdDay-v0"]#, "originalGame-v0"]#,"PrivateEye-v0"
encoders =  ["rand_cnn","tdc","vae","inv_model"]
lrs = [0.001]
labels=["y_coord"]
main_file = "main.py"
mode= "train"
task="predict"
comet_mode = "online"
seed = 4
nodes = ["leto28","leto31","kepler2","leto20"]
node_dict = dict(zip(encoders,nodes))
for game in games:
    for level in levels:
        for lr in lrs:
            for label in labels:
                for enc in encoders:
                    if game == "LunarLander-v2":
                        script = "./submit_scripts/run_xfgpu.sl"
                    else:
                        script = "./submit_scripts/run_gpu.sl"
                    args = ["sbatch",
                    "-w %s"%(node_dict[enc]),
                    script,
                    "%s"%(main_file),
                    "--embedder_name %s"%enc,
                    "--embed_env %s"%game,
                    "--embed_level %s"%level,
                    "--mode %s"%mode,
                    "--task %s"%task,
                    "--label_name %s"%label,
                    "--comet_mode %s"%(comet_mode),
                    "--tr_size %i"%10000,
                    "--val_size %i"%1000,
                    "--batch_size %i"%64,
                    "--lr %f"%(lr),
                    "--seed %i"%(seed)]
                    print(" ".join(args))
                    subprocess.run(args)
        
