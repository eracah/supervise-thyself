
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
from data.replay_buffer import BufferFiller
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
    parser.add_argument("--resize_to",type=int, nargs=2, default=[84, 84])
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
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.batch_size = 5
        args.max_quant_eval_epochs = 2
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args



# In[4]:


def setup_models():

    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(args.device)
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(args.device)
    rand_cnn = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(args.device)

    return raw_pixel_enc, rand_lin_proj, rand_cnn #,  q_net, target_q_net, encoder,inv_model, 
    


# In[5]:


def setup_tr_val_val_test(env, policy, convert_fxn, tot_examples):

    
    
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy,
                      batch_size=args.batch_size)
    print("creating tr_buf")
    size = int(0.7*tot_examples)
    tr_buf = bf.fill(size=size)
    print(len(tr_buf))
    assert size == len(tr_buf)
    
    
    print("creating val1_buf")
    size=int(0.1*tot_examples)
    val1_buf = bf.fill_with_unvisited_states(visited_buffer=tr_buf, size=size)
    print(len(val1_buf))
    assert size == len(val1_buf)
    
    print("creating val2_buf")
    size=int(0.1*tot_examples)
    val2_buf = bf.fill_with_unvisited_states(visited_buffer=tr_buf + val1_buf, size=size)
    
    print(len(val2_buf))
    assert size == len(val2_buf)
    print("creating test_buf")
    
    size=int(0.1*tot_examples)
    test_buf = bf.fill_with_unvisited_states(visited_buffer=tr_buf + val1_buf + val2_buf, size=size)
    print(len(test_buf))
    assert size == len(test_buf)
    return tr_buf, val1_buf, val2_buf, test_buf
    
 


# In[6]:


#train
if __name__ == "__main__":
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False        
    args = setup_args(test_notebook)
    exp_dir = setup_exp_name(test_notebook, args)
    writer, models_dir = setup_dirs_logs(args, exp_dir)
    

    env, action_space, grid_size, num_directions, tot_examples = setup_env(args.env_name)
    if test_notebook:
        tot_examples = 50
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    policy=lambda x0: np.random.choice(action_space)
    tr_buf, val1_buf, val2_buf, test_buf = setup_tr_val_val_test(env, policy, convert_fxn, tot_examples)

    
    raw_pixel_enc, rand_lin_proj, rand_cnn = setup_models()
    enc_dict = {"rand_cnn":rand_cnn, "rand_proj":rand_lin_proj} #"raw_pix":raw_pixel_enc,"inv_model":encoder }
    qevs = QuantEvals(val1_buf, val2_buf, test_buf, writer,
               grid_size,num_directions, args)
    #train_inv_model()


    eval_dict = qevs.run_evals(enc_dict)

        

