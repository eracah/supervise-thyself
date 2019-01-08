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
import copy
import uuid
import retro

def setup_exp(args):
    if args.use_comet == False:
        id = uuid.uuid4().hex
        return None, id
    from comet_ml import Experiment
    exp_name = ("nb_" if args.test_notebook else "") + "_".join([args.mode, args.embedder_name, get_hyp_str(args)])
    experiment = Experiment(api_key="kH9YI2iv3Ks9Hva5tyPW9FAbx",
                            project_name="self-supervised-survey",
                            workspace="eracah")
    experiment.set_name(exp_name)
    experiment.log_multiple_params(args.__dict__)
    return experiment, experiment.id



def setup_dir(args,exp_id,basename=".models"):
    dir_ = Path(basename) / get_child_dir(args,task=args.task) / Path(exp_id)
    dir_.mkdir(exist_ok=True,parents=True)
    return dir_



def setup_args():
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    
    #env params
    parser.add_argument("--embed_env",type=str, default="FlappyBirdDay-v0")
    parser.add_argument("--transfer_env",type=str, default="FlappyBirdDay-v0")
    parser.add_argument("--test_env",type=str, default="FlappyBirdDay-v0")
    
    #level params
    parser.add_argument("--embed_level",type=str, default="None")
    parser.add_argument("--transfer_level",type=str, default="None")
    parser.add_argument("--test_level",type=str, default="None")
    
    
    #mode params
    parser.add_argument('--mode', choices=["embed","transfer","test"],default="embed")
    parser.add_argument("--task", choices=["embed","infer","predict","control"], default="embed")
    
    
    #embed params
    parser.add_argument("--base_enc_name",type=str,default="world_models")
    embedder_names = ['inv_model', 'vae','rand_cnn',"tdc", "snl"]
    parser.add_argument("--embedder_name",choices=embedder_names,default="inv_model")
    parser.add_argument("--embed_len",type=int,default=32)

    # embedder specific args
    parser.add_argument("--num_time_dist_buckets",default=4)
    
    
    
    #data params
    parser.add_argument("--resize_to",type=int, nargs=2, default=[128, 128])
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--frames_per_example",type=int,default=2)
    parser.add_argument("--tr_size",type=int,default=10000)
    parser.add_argument("--val_size",type=int,default=1000)
    parser.add_argument("--test_size",type=int,default=1000)
    
    #general params
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--no_actions",action="store_true")
 
    
    
    # inference (non-control) params   
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--buckets",type=int,default=16)
    parser.add_argument("--label_name",type=str,default="x_coord")
  

    # control args
    parser.add_argument("--rollouts",type=str,default=10)
    parser.add_argument("--val_rollouts",type=str,default=5)
    parser.add_argument("--eval_best_freq",type=int,default=5)
    
    #unused?
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--model_type",type=str,default="classifier")

    
    args = parser.parse_args()
    
    
    args.env_name = getattr(args, args.mode + "_env")
    
    if args.env_name in ["Snake-v0", "FlappyBird-v0","FlappyBirdDay-v0","FlappyBirdNight-v0", "WaterWorld-v0", 'Catcher-v0']:
        args.ple =True
    else:
        args.ple = False
        
    args.retro = True if args.env_name in retro.data.list_games() else False
    if args.retro:
        args.level = getattr(args, args.mode + "_level")
        assert args.level != "None"
    else:
        args.level = "None"

    

    args.resize_to = tuple(args.resize_to)
    args.there_are_actions = not args.no_actions
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.test_notebook = test_notebook
    
    if args.test_notebook:
        args.workers=1
        args.batch_size = 8  
        args.tr_size = 64
        args.test_size= 64
        args.val_size = 48
        args.resize_to = (128,128)
        args.mode="test"
        args.task="infer"
        args.label_name="y_coord"
        args.use_comet = False
        args.frames_per_example = 10
        
        
    assert not (args.mode == "test" and args.task == "embed"), "no testing for embed!"
    args.needs_labels = True if args.task == "infer" or (args.mode == "test" and args.task == "predict") else False

    return args


def convert_to1hot(a,n_actions):
    dims = a.size()
    batch_size = dims[0]
    if len(dims) < 2:
        a = a[:,None]
    
    a = a.long()
    a_1hot = torch.zeros((batch_size,n_actions)).long().to(a.device)

    src = torch.ones_like(a).to(a.device)

    a_1hot = a_1hot.scatter_(dim=1,index=a,src=src)
    return a_1hot

def get_env_nickname(args):
    env_nickname = args.env_name.split("-")[0]
    exp_nickname = str(args.resize_to[0]) + env_nickname
    return exp_nickname

def get_hyp_str(args):
    hyp_str = ("lr%f"%args.lr).rstrip('0').rstrip('.')
    if args.embedder_name == "beta_vae":
        hyp_str += ("beta=%f"%args.beta).rstrip('0').rstrip('.')
    return hyp_str 


def get_child_dir(args, task):
    env_nn = get_env_nickname(args)
    
    mode = args.mode
    child_dir = Path(task)
    if task == "infer":
        child_dir = child_dir / Path(args.label_name)
    
    child_dir = child_dir / Path(args.embedder_name) / Path(env_nn) / Path(("nb_" if args.test_notebook else "") + ("" if mode == "test" else get_hyp_str(args) ))
    
    
    
    return child_dir

def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)


