#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
# embed_env = "SonicTheHedgehog-Genesis"
# transfer_env = "SonicTheHedgehog-Genesis"
# embed_level = 'GreenHillZone.Act1'
# transfer_level = 'GreenHillZone.Act1'
embed_env = "PrivateEye-v0"
transfer_env = "PrivateEye-v0"
embed_level = None
transfer_level = None
encoders =  ["snl", "inv_model","rand_cnn"]#,"tdc","vae"]
lrs = [0.001]
labels=["x_coord"]
main_file = "main.py"
mode= "train"
task="infer"
comet_mode = "online"
seed = 4
nodes = ["leto12"]*3
node_dict = dict(zip(encoders,nodes))
for lr in lrs:
    for label in labels:
        for enc in encoders:
            if transfer_env == "LunarLander-v2":
                script = "./submit_scripts/run_xfgpu.sl"
            else:
                script = "./submit_scripts/run_gpu.sl"
            args = ["sbatch",
                    "-w %s"%(node_dict[enc]),
                    script,
                    "%s"%(main_file),
                    "--embedder_name %s"%enc,
                    "--embed_env %s"%embed_env,
                    "--embed_level %s"%embed_level,
                    "--transfer_env %s"%transfer_env,
                    "--transfer_level %s"%transfer_level,
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
        
