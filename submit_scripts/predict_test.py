#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
# embed_env = "FlappyBirdDay-v0"
# transfer_env = "FlappyBirdDay-v0"
# test_envs = ["FlappyBirdDay-v0", "FlappyBirdNight-v0"]
# test_levels = [None]
# embed_level = None
# transfer_level = None
embed_env = "SonicTheHedgehog-Genesis"
transfer_env = "SonicTheHedgehog-Genesis"
embed_level = 'GreenHillZone.Act1'
transfer_level = 'GreenHillZone.Act1'
test_envs = ["SonicTheHedgehog-Genesis"]
test_levels = ['GreenHillZone.Act1'] #, 'GreenHillZone.Act2'] #,"LabyrinthZone.Act1"]
encoders =  ["rand_cnn"]
lrs = [0.001]
labels=["y_coord"]
main_file = "main.py"
mode= "test"
task="predict"
comet_mode = "online"
seed = 4
nodes = ["kepler2"]
node_dict = dict(zip(encoders,nodes))
for test_env in test_envs:
    for test_level in test_levels:
        for lr in lrs:
            for label in labels:
                for enc in encoders:
                    if test_env == "LunarLander-v2":
                        script = "./submit_scripts/run_xfcpu.sl"
                    else:
                        script = "./submit_scripts/run_cpu.sl"
                    args = [#"sbatch",
                            #"-w %s"%("kepler2"),
                            #script,
                            "python",
                            "%s"%(main_file),
                            "--embedder_name %s"%enc,
                            "--embed_env %s"%embed_env,
                            "--embed_level %s"%embed_level,
                            "--transfer_env %s"%transfer_env,
                            "--transfer_level %s"%transfer_level,
                            "--test_env %s"%test_env,
                            "--test_level %s"%test_level,
                            "--mode %s"%mode,
                            "--task %s"%task,
                            "--label_name %s"%label,
                            "--comet_mode %s"%(comet_mode),
                            "--test_size %i" % 5000,
                            "--batch_size %i"%64,
                            "--seed %i"%(seed)]
                    print(" ".join(args))
                    #subprocess.run(args)
        
