
# coding: utf-8

# In[5]:


import random
import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from models.base_encoder import Encoder
from utils import setup_env
from data.replay_buffer import BufferFiller
import argparse
from models.inverse_model import InverseModel
from utils import convert_frame, classification_acc, mkstr, setup_dirs_logs, parse_minigrid_env_name
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


# In[2]:


def setup_args(test_notebook):
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[96, 96])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--seed",type=int,default=4)
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mstr = partial(mkstr,args=args)
    output_dirname = ("nb_" if test_notebook else "") + "_".join(["inv_model_e%s"%parse_minigrid_env_name(args.env_name),
                                                                        "r%i"%(args.resize_to[0])
                                                                       ])
    args.output_dirname = output_dirname
    return args


# In[3]:


class Trainer(object):
    def __init__(self, model, tr_buf, val_buf, model_dir, args, writer):
        self.model = model
        self.val_buf = val_buf
        self.tr_buf =tr_buf
        self.args = args
        self.model_dir = model_dir

        self.writer=writer
        
        #self.opt_template = partial(Adam,params=self.model.parameters())
        
        #self.opt = None
        self.opt = Adam(params=self.model.parameters(),lr=args.lr)
        self.epoch=0
        self.max_epochs = 10000

    def one_iter(self, trans, update_weights=True):
        if update_weights:
            self.opt.zero_grad()
        pred = self.model(trans.x0,trans.x1)
        true = trans.a
        loss =  nn.CrossEntropyLoss()(pred,true)
            
            
        acc = classification_acc(logits=pred,true=true)
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
        writer.add_scalar("inv_model/%s_loss"%mode,avg_loss,global_step=self.epoch)
        writer.add_scalar("inv_model/%s_acc"%mode,avg_acc,global_step=self.epoch)
        if mode == "train":
            print("Epoch %i: "%self.epoch)
        print("\t%s"%mode)
        print("\t\tLoss: %8.4f \n\t\tAccuracy: %9.3f%%"%(avg_loss, 100*avg_acc))
        return avg_loss, avg_acc
    
        
    def train(self):
        #self.opt = self.opt_template(lr=lr)
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
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                for f in self.model_dir.glob("best_model*"):
                    os.remove(str(f))
                torch.save(self.model.state_dict(), self.model_dir / Path(  ("best_model_%f.pt"%best_val_loss).rstrip('0').rstrip('.')))

def split_tr_set(tr_buf, fraction=0.8):
    t1 = time.time()
    print("doing split")
    tr_buf,val_buf = bf.split(tr_buf,fraction)
    t2 = time.time()
    print("%8.4f seconds"%(t2-t1))
    

    return tr_buf, val_buf

def setup_model(args, action_space):
    encoder = Encoder(in_ch=3,
    im_wh=args.resize_to,
    h_ch=args.hidden_width,
    embed_len=args.embed_len).to(args.device)

    inv_model = InverseModel(encoder=encoder,num_actions=len(action_space)).to(args.device)
    return inv_model    

def setup_exp_name(test_notebook, args):
    prefix = ("nb_" if test_notebook else "")
    exp_name = Path(prefix  + "_".join([ ("lr%f"%args.lr).rstrip('0').rstrip('.'),"%s"%parse_minigrid_env_name(args.env_name), "r%i"%(args.resize_to[0])]))
    base_dir = Path("inv_model")
    return base_dir / exp_name
    


# In[4]:


if __name__ == "__main__":
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False        
    args = setup_args(test_notebook)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)


    exp_name = setup_exp_name(test_notebook,args)
    writer, model_dir = setup_dirs_logs(args,exp_name)
    env, action_space, grid_size, num_directions, tot_examples, random_policy = setup_env(args.env_name,args.seed)
    inv_model = setup_model(args, action_space)

    print("starting to load buffers")
    tr_buf, val_buf = setup_tr_val_val_test(env, random_policy, 
                                convert_fxn, tot_examples,args.batch_size,just_train=True)
    print("done loading buffers")
    

    trainer = Trainer(inv_model, tr_buf, val_buf, model_dir, args, writer)
    trainer.train()


# In[ ]:




