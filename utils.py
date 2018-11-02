import sys
from torch import nn
import torch
import numpy as np
import json
import numpy as np
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
from collections import namedtuple
from itertools import product
import random
import argparse
from comet_ml import Experiment
import copy
model_names = ['inv_model', 'vae', 'raw_pixel', 'lin_proj', 'rand_cnn', 'linv_model']
model_names = model_names + ["forward_" + model_name for model_name in model_names ]

def setup_args():
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default="PrivateEye-v0"),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--model_name",choices=model_names,default="inv_model")
    parser.add_argument("--beta",type=float,default=2.0)
    parser.add_argument("--tr_size",type=int,default=10000)
    parser.add_argument("--val_size",type=int,default=1000)
    parser.add_argument("--test_size",type=int,default=1000)
    parser.add_argument('--mode', choices=['train','train_forward', 'eval', 'test'], default="train")
    parser.add_argument("--buckets",type=int,default=20)
    parser.add_argument("--label_name",type=str,default="x_coord")
    parser.add_argument("--frames_per_trans",type=int,default=2)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--model_type",type=str,default="classifier")
    #parser.add_argument("--eval_mode",type=str,default="infer")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.test_notebook = test_notebook
    if args.test_notebook:
        args.workers=1
        args.batch_size = 8  
        args.tr_size = 16
        args.test_size= 8
        args.val_size = 16
        args.resize_to = (128,128)
        args.mode="train"
        args.model_name = "vae"

    return args




def get_env_nickname(args):
    env_nickname = args.env_name.split("-")[0]
    exp_nickname = str(args.resize_to[0]) + env_nickname
    return exp_nickname

def get_hyp_str(args):
    hyp_str = ("lr%f"%args.lr).rstrip('0').rstrip('.')
    if args.model_name == "beta_vae":
        hyp_str += ("beta=%f"%args.beta).rstrip('0').rstrip('.')
    return hyp_str 


def get_child_dir(args, mode):
    env_nn = get_env_nickname(args)
    
    
    child_dir = Path(mode)
    if mode == "eval" or mode == "test":
        child_dir = child_dir / Path(args.label_name)
    
    child_dir = child_dir / Path(args.model_name) / Path(env_nn) / Path(("nb_" if args.test_notebook else "") + ("" if mode == "test" else get_hyp_str(args) ))
    
    
    
    return child_dir



def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)


