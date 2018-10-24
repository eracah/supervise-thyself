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


