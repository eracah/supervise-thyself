
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
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
import time
from pathlib import Path
import json
from functools import partial
from data.tr_val_test_splitter import setup_tr_val_val_test
from models.base_encoder import Encoder
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from models.inverse_model import InverseModel
from utils import mkstr,write_to_config_file,convert_frame, classification_acc,setup_env, setup_dirs_logs,parse_minigrid_env_name
from evaluations.quant_evaluation import QuantEvals


# In[2]:


def setup_exp_name(test_notebook, args):
    mstr = partial(mkstr,args=args)
    prefix = ("nb_" if test_notebook else "")
    exp_name = Path(prefix  + "_".join(["%s"%parse_minigrid_env_name(args.env_name), "r%i"%(args.resize_to[0])]))
    base_dir = Path("eval")
    return base_dir / exp_name


# In[3]:


def setup_args(test_notebook):
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
    parser.add_argument("--encoders_to_eval",type=str, nargs='+', default=["inv_model"])
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.batch_size = 5
        args.max_quant_eval_epochs = 2
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args



# In[6]:


def setup_models(action_space):

    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(args.device)
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(args.device)
    rand_cnn = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(args.device)
    if "inv_model" in args.encoders_to_eval:
        encoder = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(args.device)
        
        inv_model = InverseModel(encoder=encoder,num_actions=len(action_space)).to(args.device)
        # cutomize this for the size of images and the size of the env
        inv_model.load_state_dict(torch.load(".models/inv_model/lr0.00005_64_r96/cur_model.pt"))
        

    return raw_pixel_enc, rand_lin_proj, rand_cnn, inv_model.encoder #,  q_net, target_q_net, encoder,inv_model, 
    


# In[7]:


#train
if __name__ == "__main__":
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False        
    args = setup_args(test_notebook)

    exp_dir = setup_exp_name(test_notebook, args)
    writer, models_dir = setup_dirs_logs(args, exp_dir)
    



    env, action_space, grid_size, num_directions, tot_examples, random_policy = setup_env(args.env_name, args.seed)
    raw_pixel_enc, rand_lin_proj, rand_cnn, inv_model = setup_models(action_space)
    
    

    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    tr_buf, val_buf, eval_tr_buf, eval_val_buf, test_buf = setup_tr_val_val_test(env, random_policy,
                                                                                 convert_fxn, tot_examples, args.batch_size)

    

 
    
    

    enc_dict = {"rand_cnn":rand_cnn, "rand_proj":rand_lin_proj, "inv_model":inv_model} #"raw_pix":raw_pixel_enc,"inv_model":encoder }
    
    
    
    qevs = QuantEvals(eval_tr_buf, eval_val_buf, test_buf, writer,
               grid_size,num_directions, args)

    eval_dict = qevs.run_evals(enc_dict)

        

