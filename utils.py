import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import sys
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np
import json


from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
from collections import namedtuple
from itertools import product
import random

from comet_ml import Experiment


def get_upper(st):
    ret = "".join([s for s in st if s.isupper()])
    return ret

def get_exp_nickname(args):
    return str(args.resize_to[0]) + args.env_nickname + (parse_minigrid_env_name(args.env_name) if "MiniGrid" in args.env_name else "")

def setup_exp_dir(args, base_name="eval"):
    exp_nn = get_exp_nickname(args)
    exp_name = Path(exp_nn)
    base_dir = Path(base_name)
    return base_dir / exp_name

def setup_exp(args,project_name, exp_name):
    experiment = Experiment(api_key="kH9YI2iv3Ks9Hva5tyPW9FAbx",
                            project_name=project_name,
                            workspace="eracah")
    experiment.set_name(exp_name)
    return experiment
    
def setup_model_dir(child_dir):
    model_dir = Path(".models") / child_dir
    model_dir.mkdir(exist_ok=True,parents=True)
    return model_dir

def parse_minigrid_env_name(name):
    return name.split("-")[2].split("x")[0]

def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)


