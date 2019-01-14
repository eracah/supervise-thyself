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
    exp_name = ("nb_" if args.test_notebook else "") + "_".join([args.task, args.mode, args.embedder_name, get_hyp_str(args)])
    exp_kwargs = dict(project_name="self-supervised-survey",
                            workspace="eracah")
    if args.comet_mode == "online":
        from comet_ml import Experiment
        exp_kwargs.update(api_key="kH9YI2iv3Ks9Hva5tyPW9FAbx")
    elif args.comet_mode == "offline":
        from comet_ml.offline import OfflineExperiment as Experiment
        offline_directory = Path(".logs")
        exp_kwargs.update(offline_directory=str(offline_directory))
    
    experiment = Experiment(**exp_kwargs)
    experiment.set_name(exp_name)
    experiment.log_parameters (args.__dict__)
    return experiment, experiment.id



def setup_dir(args,exp_id,basename=".models"):
    dir_ = Path(basename) / get_child_dir(args,task=args.task,env_name=args.env_name,level=args.level) / Path(exp_id)
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
    parser.add_argument('--mode', choices=["train","test"],default="train")
    parser.add_argument("--task", choices=["embed","infer","predict","control"], default="embed")
    
    
    #embed params
    parser.add_argument("--base_enc_name",type=str,default="world_models")
    embedder_names = ['inv_model', 'vae','rand_cnn',"tdc", "snl"]
    parser.add_argument("--embedder_name",choices=embedder_names,default="inv_model")
    parser.add_argument("--embed_len",type=int,default=32)

    # embedder specific args
    parser.add_argument("--num_time_dist_buckets",default=4)
    parser.add_argument("--seq_tasks_num_frames",default=10)
    
    
    
    #data params
    parser.add_argument("--resize_to",type=int, nargs=2, default=[128, 128])
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--frames_per_example",type=int)
    parser.add_argument("--tr_size",type=int,default=10000)
    parser.add_argument("--val_size",type=int,default=1000)
    parser.add_argument("--test_size",type=int,default=1000)
    parser.add_argument("--there_are_actions",type=bool,default=False)
    parser.add_argument("--there_are_rewards",type=bool,default=False)
    
    #general params
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--no_actions",action="store_true")
    parser.add_argument("--comet_mode",type=str, choices=["online", "offline"],default="online")
    
    
    # inference (non-control) params   
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--buckets",type=int,default=16)
    parser.add_argument("--label_name",type=str,default="x_coord")
  

    # prediction parameters
    parser.add_argument("--pred_num_params", type=int, default=10)

    # control args
    parser.add_argument("--rollouts",type=str,default=10)
    parser.add_argument("--val_rollouts",type=str,default=5)
    parser.add_argument("--eval_best_freq",type=int,default=5)
    
    #unused?
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--model_type",type=str,default="classifier")

    
    args = parser.parse_args()
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
        args.embedder_name = "snl"
        args.embed_env=args.transfer_env=args.test_env="Pitfall-v0"
#         args.embed_env="SonicAndKnuckles3-Genesis"
#         args.transfer_env="SonicAndKnuckles3-Genesis"
#         args.transfer_level="CarnivalNightZone.Act1"
#         args.embed_level = "AngelIslandZone.Act1"
        args.label_name="x_coord"
        args.comet_mode = "online"

    
    if args.mode == "train":
        if args.task == "embed":
            args.regime = "embed"
            assert args.embedder_name != "rand_cnn", "Random CNN needs no training!"
        elif args.task in ["predict","infer","control"]:
            args.regime = "transfer"
        else:
            assert False, "what task did you pick???!!  %s"%(args.task)
    if args.mode == "test":
        if args.task == "embed":
            print("no testing  for embed!")
        else:
            args.regime = "test"
            
            
    args.env_name = getattr(args, args.regime + "_env")
    
    if args.env_name in ["Snake-v0", "FlappyBird-v0","FlappyBirdDay-v0","FlappyBirdNight-v0", "WaterWorld-v0", 'Catcher-v0']:
        args.ple =True
    else:
        args.ple = False
        
    args.retro = True if args.env_name in retro.data.list_games() else False
    if args.retro:
        args.level = getattr(args, args.regime + "_level")
        assert args.level != "None", "must specify a level!"
    else:
        args.level = "None"

    

    args.resize_to = tuple(args.resize_to)
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    

    args.needs_labels = True if args.task == "infer" or (args.mode == "test" and args.task == "predict") else False
    if args.task == "infer":
        args.frames_per_example = 1
    if args.task == "predict":
        args.frames_per_example = args.pred_num_params
        args.there_are_actions = True
        
    if args.task == "embed":
        if args.embedder_name in ['vae','rand_cnn']:
            args.frames_per_example = 1
        elif args.embedder_name == "inv_model":
            args.frames_per_example = 2
            args.there_are_actions = True
        elif args.embedder_name in ["snl", "tdc"]:
            args.frames_per_example = args.seq_tasks_num_frames
    
    print("num_frames_per_example",args.frames_per_example)
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

def get_env_nickname(env_name,level, resize_to):
    level = "" if level is None else "_" + level
    return str(resize_to[0]) + env_name.split("-")[0] + level

def get_hyp_str(args):
    hyp_str = ("lr%f"%args.lr).rstrip('0').rstrip('.')
    if args.embedder_name == "beta_vae":
        hyp_str += ("beta=%f"%args.beta).rstrip('0').rstrip('.')
    return hyp_str 


def get_child_dir(args, task, env_name, level):
    env_nn = get_env_nickname(env_name,level, args.resize_to)
    child_dir = Path(task)
    if task=="infer":
        child_dir = child_dir / Path(args.label_name)
    
    child_dir = child_dir / Path(args.embedder_name) / Path(env_nn) / Path(("nb_" if args.test_notebook else "") + get_hyp_str(args))
    
    
    
    return child_dir

def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)


