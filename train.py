#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from models.base_encoder import Encoder
from utils import setup_env, setup_exp_dir
from data.replay_buffer import BufferFiller
import argparse
from models.inverse_model import InverseModel
from models.baselines import VAE, BetaVAE
from utils import convert_frame, classification_acc, mkstr, parse_minigrid_env_name, setup_model_dir, get_upper, setup_exp
import argparse
import sys
import copy
import torch
from functools import partial
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np
from pathlib import Path
import time
from data.tr_val_test_splitter import setup_tr_val_val_test
import os


# In[8]:


def setup_args(test_notebook):
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-8x8-v0'),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[96, 96])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--model",type=str,default="beta_vae")
    parser.add_argument("--beta",type=float,default=2.0)
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    args.env_nickname = get_upper(args.env_name)
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mstr = partial(mkstr,args=args)
    if test_notebook:
        args.test_notebook=True
    else:
        args.test_notebook = False
#     output_dirname = ("nb_" if test_notebook else "") + "_".join(["%s_e%s"%(args.model,parse_minigrid_env_name(args.env_name)),
#                                                                         "r%i"%(args.resize_to[0])
#                                                                        ])
#     args.output_dirname = output_dirname
    return args


# In[9]:


class Trainer(object):
    def __init__(self, model, tr_buf, val_buf, model_dir, args, experiment):
        self.model = model
        self.val_buf = val_buf
        self.tr_buf =tr_buf
        self.args = args
        self.model_dir = model_dir
        self.model_name = self.args.model
        self.experiment = experiment
        
        #self.opt_template = partial(Adam,params=self.model.parameters())
        
        #self.opt = None
        self.opt = Adam(params=self.model.parameters(),lr=args.lr)
        self.epoch=0
        self.max_epochs = 10000

    def one_iter(self, trans, update_weights=True):
        if update_weights:
            self.opt.zero_grad()
        loss, acc = self.model.loss_acc(trans)
        if update_weights:
            loss.backward()
            self.opt.step()
        return float(loss.data),acc
    
    def one_epoch(self, buffer,mode="train"):
        update_weights = True if mode=="train" else False
        losses, accs = [], []
        for trans in buffer:
            loss,acc = self.one_iter(trans,update_weights=update_weights)
            losses.append(loss)
            accs.append(acc)
        
        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        experiment.log_multiple_metrics(dict(loss=avg_loss,acc=avg_acc), prefix=mode, step=self.epoch)
        if mode == "train":
            print("Epoch %i: "%self.epoch)
        print("\t%s"%mode)
        print("\t\tLoss: %8.4f \n\t\tAccuracy: %9.3f%%"%(avg_loss, 100*avg_acc))
        return avg_loss, avg_acc
    
        
    def train(self):
        state_dict = self.model.state_dict()
        val_acc = -np.inf
        best_val_loss = np.inf
        while val_acc < 95. or self.epoch < self.max_epochs:
            self.epoch+=1
            self.model.train()
            tr_loss,tr_acc = self.one_epoch(self.tr_buf,mode="train")
            
            
            torch.save(self.model.state_dict(), self.model_dir / "cur_model.pt")
            self.model.eval()
            val_loss, val_acc = self.one_epoch(self.val_buf,mode="val")
            if self.epoch == 1:
                torch.save(self.model.state_dict(), self.model_dir / Path(  ("best_model_%f.pt"%val_loss).rstrip('0').rstrip('.')))
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                for f in self.model_dir.glob("best_model*"):
                    os.remove(str(f))
                torch.save(self.model.state_dict(), self.model_dir / Path(  ("best_model_%f.pt"%best_val_loss).rstrip('0').rstrip('.')))


def setup_model(args, action_space):
    model_table = {"inv_model":InverseModel, "vae":VAE, "beta_vae": BetaVAE}
    encoder_kwargs = dict(in_ch=3,im_wh=args.resize_to,h_ch=args.hidden_width, embed_len=args.embed_len,  
                        num_actions=len(action_space), beta=args.beta)

    model = model_table[args.model](**encoder_kwargs ).to(args.device)
    return model    

def get_project_name(args, exp_dir):
    return "_".join([str(exp_dir.parent), args.model])
    
def get_hyp_str(args):
    hyp_str = ("lr%f"%args.lr).rstrip('0').rstrip('.')
    if args.model == "beta_vae":
        hyp_str += ("beta=%f"%args.beta).rstrip('0').rstrip('.')
    return hyp_str   

def get_exp_name(args, exp_dir):
    exp_name = ("nb_" if args.test_notebook else "") + exp_dir.name
    exp_name += "_" + get_hyp_str(args)
    return exp_name
    


# In[2]:


def get_experiment(args,exp_dir):
    project_name = get_project_name(args, exp_dir)
    exp_name = get_exp_name(args, exp_dir)
    experiment = setup_exp(args,project_name, exp_name)
    return experiment


# In[3]:


if __name__ == "__main__":
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False        
    args = setup_args(test_notebook)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)

    exp_dir = setup_exp_dir(args=args,base_name="train")
    model_dir = setup_model_dir(exp_dir.parent / Path(exp_dir.name) / Path(args.model) / Path(("nb_" if args.test_notebook else "") + get_hyp_str(args)) )
    experiment = get_experiment(args,exp_dir)
    experiment.log_multiple_params(args.__dict__)

    env, action_space, grid_size, num_directions, tot_examples, random_policy = setup_env(args.env_name,args.seed)
    model = setup_model(args, action_space)
    
    

    print("starting to load buffers")
    tr_buf, val_buf = setup_tr_val_val_test(env, random_policy, 
                                 convert_fxn, tot_examples,args.batch_size,just_train=True, frames_per_trans=3)

    print("done loading buffers")
    

    trainer = Trainer(model, tr_buf, val_buf, model_dir, args, experiment)
    trainer.train()


# In[ ]:





# In[ ]:




