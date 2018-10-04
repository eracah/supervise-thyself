#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch
from torch import nn
import torch.functional as F
from torch.optim import Adam, RMSprop
import argparse
import sys
import copy
from copy import deepcopy
from torchvision.utils import make_grid
import numpy as np
import time
from pathlib import Path
import json
from functools import partial
from data.tr_val_test_splitter import setup_tr_val_val_test
from models.base_encoder import Encoder
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN, VAE, BetaVAE
from models.inverse_model import InverseModel
from utils import mkstr,write_to_config_file,convert_frame,classification_acc,setup_env, setup_exp,parse_minigrid_env_name
from evaluations.quant_evaluation import QuantEvals
import os
from comet_ml import Experiment

# In[2]:


def setup_exp_dir(args):
    mstr = partial(mkstr,args=args)
    prefix = ("nb_" if args.test_notebook else "")
    exp_name = Path(prefix  + str(args.resize_to[0]) + "_" + args.env_name)
    base_dir = Path("eval")
    return base_dir / exp_name





def get_weights_path(enc_name, args):
    best_loss = np.inf
    weights_path = None
    base_path = Path(".models") / Path(enc_name)
    suffix = "_" + str(args.grid_size+2) + "_r" + str(args.resize_to[0])
    if not base_path.exists():
        return None
    for model_dir in base_path.iterdir():
        if suffix in model_dir.name:
            if args.test_notebook:
                if "nb" not in model_dir.name:
                    continue
            else:
                if "nb" in model_dir.name:
                    continue
            model_path = list(model_dir.glob("best_model*"))[0]
            loss = float(str(model_path).split("_")[-1].split(".pt")[0])
            if loss < best_loss:
                best_loss = copy.deepcopy(loss)
                weights_path = copy.deepcopy(model_path)
            
    return weights_path


# In[7]:


def setup_args():
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False 
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]

    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lasso_coeff", type=float, default=0.1)
    parser.add_argument("--max_quant_eval_epochs", type=int, default=50)
    parser.add_argument("--gen_loss_alpha", type=float, default=0.4)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--val_batch_size",type=int,default=32)
    parser.add_argument("--num_episodes",type=int,default=100)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[96, 96])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=50000)
    parser.add_argument("--eval_init_buffer_size",type=int,default=1000)
    parser.add_argument("--eval_trials",type=int,default=5)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["forward", "left", "right"])
    parser.add_argument("--decoder_batches", type=int, default=1000)
    parser.add_argument("--collect_data",action="store_true")
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--encoders_to_eval",type=str, nargs='+', default=["inv_model","vae","beta_vae"])
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.batch_size = 5
        args.max_quant_eval_epochs = 200
        args.test_notebook=True
    else:
        args.test_notebook = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def setup_models(action_space, args):
    model_table = {"inv_model":InverseModel, "vae":VAE, "beta_vae": BetaVAE}
    encoder_kwargs = dict(in_ch=3,im_wh=args.resize_to,h_ch=args.hidden_width, embed_len=args.embed_len,  
                          batch_norm=args.batch_norm)
    
    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(args.device).eval()
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(args.device).eval()
    
    rand_cnn = Encoder(**encoder_kwargs).to(args.device).eval()
    encoder_table = dict(raw_pixel_enc=raw_pixel_enc, rand_lin_proj=rand_lin_proj, rand_cnn=rand_cnn)
    
    for enc_name in args.encoders_to_eval:
        model = model_table[enc_name](**encoder_kwargs).to(args.device).eval()
        model_path = get_weights_path(enc_name, args)
        if model_path:
            model.load_state_dict(torch.load(str(model_path)))
        else:
            print("No weights available for %s. Using randomly initialized %s"%(enc_name,enc_name))
        encoder_table[enc_name] = copy.deepcopy(model.encoder)

    return encoder_table


# In[3]:


# In[2]:


if __name__ == "__main__":
           
    args = setup_args()

    exp_dir = setup_exp_dir(args)
    

    env, action_space, grid_size, num_directions, tot_examples, random_policy = setup_env(args.env_name, args.seed)
    args.grid_size = grid_size
    args.num_directions = num_directions
    args.tot_examples = tot_examples

    enc_dict = setup_models(action_space, args)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    tr_buf, val_buf, eval_tr_buf, eval_val_buf, test_buf = setup_tr_val_val_test(env, random_policy,
                                                                                 convert_fxn, tot_examples, args.batch_size)

    qevs = QuantEvals(eval_tr_buf, eval_val_buf, test_buf, args, exp_dir)

    eval_dict = qevs.run_evals(enc_dict)

        


# In[ ]:




