
# coding: utf-8

# In[1]:


import random
import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid

from models.base_encoder import Encoder
from utils import setup_exp_dir
from data.utils import setup_env,convert_frame
from data.replay_buffer import BufferFiller
import argparse
from models.inverse_model import InverseModel
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN, VAE, BetaVAE
from evaluations.eval_models import EvalModel, LinearClassifier
from utils import mkstr, parse_minigrid_env_name, setup_model_dir, get_upper, setup_exp,setup_exp_dir,                  parse_minigrid_env_name, get_upper, get_exp_nickname
                    
from evaluations.utils import classification_acc
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
from data.tr_val_test_splitter import setup_tr_val_test
import os


# In[2]:


def setup_args(test_notebook):
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default='originalGame-v0'),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--model_name",type=str,default="vae")
    parser.add_argument("--beta",type=float,default=2.0)
    parser.add_argument("--tr_size",type=int,default=96)
    parser.add_argument("--val_size",type=int,default=32)
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default="train")
    parser.add_argument("--label_name",type=str,default="direction")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    args.env_nickname = get_upper(args.env_name) if "MiniGrid" in args.env_name else args.env_name.split("-")[0]
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mstr = partial(mkstr,args=args)
    if test_notebook:
        args.test_notebook=True
        args.batch_size =4 
        args.tr_size = 8
        args.val_size = 8
        args.resize_to = (96,96)
    else:
        args.test_notebook = False

    return args

class Trainer(object):
    def __init__(self, model, args, experiment):
        self.model = model
        self.val_buf = val_buf
        self.tr_buf =tr_buf
        self.args = args
        self.model_name = self.args.model_name
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
    
    def test(self,test_set):
        self.model.eval()
        test_loss, test_acc = self.one_epoch(test_set,mode="test")
        return test_acc
        
    def train(self, tr_buf, val_buf, model_dir):
        state_dict = self.model.state_dict()
        val_acc = -np.inf
        best_val_loss = np.inf
        while self.epoch < self.max_epochs:
            self.epoch+=1
            self.model.train()
            tr_loss,tr_acc = self.one_epoch(tr_buf,mode="train")
            torch.save(self.model.state_dict(), model_dir / "cur_model.pt")
            
            self.model.eval()
            val_loss, val_acc = self.one_epoch(val_buf,mode="val")
            if self.epoch == 1:
                torch.save(self.model.state_dict(), model_dir / Path(  ("best_model_%f.pt"%val_loss).rstrip('0').rstrip('.')))
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                for f in model_dir.glob("best_model*"):
                    os.remove(str(f))
                torch.save(self.model.state_dict(), model_dir / Path(  ("best_model_%f.pt"%best_val_loss).rstrip('0').rstrip('.')))

                
def get_weights_path(enc_name, args):
    best_loss = np.inf
    weights_path = None
    base_path = Path(".models/train") / Path(get_exp_nickname(args)) / Path(enc_name)
    #suffix = "_" + str(args.grid_size+2) + "_r" + str(args.resize_to[0])
    if not base_path.exists():
        return None
    for model_dir in base_path.iterdir():

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

def setup_model(args, env):
    model_name = args.model_name
    model_table = model_table = {"inv_model":InverseModel, "vae":VAE, "raw_pixel": RawPixelsEncoder,
                                 "lin_proj": RandomLinearProjection,
                                 "rand_cnn": RandomWeightCNN,  "beta_vae": BetaVAE }
    encoder_kwargs = dict(in_ch=3,im_wh=args.resize_to,h_ch=args.hidden_width, embed_len=args.embed_len,  
                        num_actions=env.action_space.n, beta=args.beta)

    base_model = model_table[model_name](**encoder_kwargs).to(args.device)
    if args.mode == "eval" or args.mode == "test":
        weights_path = get_weights_path(args.model_name, args)
        if weights_path:
            base_model.load_state_dict(torch.load(str(weights_path)))
        else:
            print("No weights available for %s. Using randomly initialized %s"%(model_name,model_name))
        
       
        eval_model = EvalModel(encoder=base_model.encoder,
                               num_classes=env.nclasses_table[args.label_name],
                               label_name=args.label_name).to(args.device)
        model = eval_model
        
    else: # if args.mode=train
        model = base_model
        
        
    return model    

def get_project_name(args, exp_dir):
    return "_".join([str(exp_dir.parent), args.model_name])
    
def get_hyp_str(args):
    hyp_str = ("lr%f"%args.lr).rstrip('0').rstrip('.')
    if args.model_name == "beta_vae":
        hyp_str += ("beta=%f"%args.beta).rstrip('0').rstrip('.')
    return hyp_str   

def get_exp_name(args, exp_dir):
    exp_name = ("nb_" if args.test_notebook else "") + exp_dir.name
    exp_name += "_" + get_hyp_str(args)
    return exp_name
    


# In[2]:


def get_experiment(args,exp_dir):
    project_name = get_project_name(args, exp_dir)
    print(project_name)
    exp_name = get_exp_name(args, exp_dir)
    experiment = setup_exp(args,project_name, exp_name)
    return experiment


# In[3]:


# In[3]:


if __name__ == "__main__":
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False        
    args = setup_args(test_notebook)
    #args.mode = "eval"
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)

    
    exp_dir = setup_exp_dir(args=args,base_name=args.mode)
    model_dir = setup_model_dir(exp_dir.parent / Path(exp_dir.name) / Path(args.model_name) / Path(("nb_" if args.test_notebook else "") + get_hyp_str(args)) )
    experiment = get_experiment(args,exp_dir)
    experiment.log_multiple_params(args.__dict__)
    
    env, random_policy = setup_env(env_name=args.env_name,seed=args.seed)
    print("starting to load buffers")
    tr_buf, val_buf = setup_tr_val_test(env=env,sizes=[args.tr_size,args.val_size], policy=random_policy, 
                                 convert_fxn=convert_fxn,batch_size=args.batch_size,just_train=True, frames_per_trans=3)
    if args.resize_to[0] == -1:
        args.resize_to = tr_buf.memory[0].xs[0].shape[:2]
    print(args.resize_to)
    

    model = setup_model(args, env)

    trainer = Trainer(model, args, experiment)
    trainer.train(tr_buf, val_buf,model_dir)

