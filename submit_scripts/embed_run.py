#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

games = ["Pitfall-v0"] 
encoders = ["inv_model"]
lrs = [0.0001]
main_file = "main.py"
mode= "train"
task="embed"
comet_mode = "offline"
seed = 4
for game in games:
    for lr in lrs:
        for enc in encoders:
            args = ["sbatch",
                    "./submit_scripts/run_gpu.sl",
                    "%s --embedder_name %s --env_name %s --mode %s --task %s"%(main_file,enc,game,mode, task),
                    "--embed_env %s"%(game),
                    "--comet_mode %s"%(comet_mode),
                    "--tr_size %i --val_size %i --batch_size %i"%(64,32,8),
                    "--lr %f"%(lr),
                    "--seed %i"%(seed)]
            print(" ".join(args))
            subprocess.run(args)
        

# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode train  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode train  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size = 64 --batch_size 8
# python main.py --tr_size  64 --batch_size 8
# python main.py --tr_size  64 --val size 32 --batch_size 8
# python main.py --tr_size  64 --val_size 32 --batch_size 8
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer --label_name y_coord
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict  --label_name y_coord
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer mode test  --label_name y_coord
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer --mode test  --label_name y_coord
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task infer --mode test  --label_name y_coord --test_size 64
# python main.py --tr_size  64 --val_size 32 --batch_size 8 --task predict --mode test  --label_name y_coord --test