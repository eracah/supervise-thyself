#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
embed_env = "FlappyBirdDay-v0"
transfer_env = "FlappyBirdDay-v0"
embed_level = None
transfer_level = None
encoders =  ["rand_cnn"]#["tdc","vae","inv_model"]
lrs = [0.001]
labels=["y_coord"]
main_file = "main.py"
mode= "train"
task="infer"
comet_mode = "online"
seed = 4
#nodes = ["leto20","leto17","leto07"]
#node_dict = dict(zip(encoders,nodes))
for lr in lrs:
    for label in labels:
        for enc in encoders:
            if game == "LunarLander-v2":
                script = "./submit_scripts/run_xfgpu.sl"
            else:
                script = "./submit_scripts/run_gpu.sl"
            args = ["sbatch",
                    "-w %s"%("leto16"),
                    script,
                    "%s"%(main_file),
                    "--embedder_name %s"%enc,
                    "--embed_env %s"%embed_env,
                    "--embed_level %s"%embed_level,
                    "--transfer_env %s"%transfer_env,
                    "--transfer_level %s"%trasnfer_level,
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
        
