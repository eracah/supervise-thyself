
# coding: utf-8

# In[1]:


import random
import data.custom_envs
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid

from models.base_encoder import Encoder
from data.utils import setup_env,convert_frame
from data.replay_buffer import BufferFiller
import argparse
from models.inverse_model import InverseModel
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN, VAE, BetaVAE
from models.utils import get_weights_path
from evaluations.eval_models import EvalModel
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
from comet_ml import Experiment
import os
from utils import get_child_dir, get_hyp_str

def setup_args():
    test_notebook= True if "ipykernel_launcher" in sys.argv[0] else False
    tmp_argv = copy.deepcopy(sys.argv)
    if test_notebook:
        sys.argv = [""]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default="originalGame-v0"),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--seed",type=int,default=4)
    parser.add_argument("--model_name",choices=['inv_model', 'vae', 'raw_pixel', 'lin_proj', 'rand_cnn', 'beta_vae'],default="inv_model")
    parser.add_argument("--beta",type=float,default=2.0)
    parser.add_argument("--tr_size",type=int,default=60000)
    parser.add_argument("--val_size",type=int,default=10000)
    parser.add_argument("--test_size",type=int,default=10000)
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default="train")
    parser.add_argument("--buckets",type=int,default=20)
    parser.add_argument("--label_name",type=str,default="y_coord")
    parser.add_argument("--frames_per_trans",type=int,default=2)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if test_notebook:
        args.test_notebook=True
        args.batch_size =32 
        args.tr_size = 1000
        args.test_size=8
        args.val_size = 1000
        args.resize_to = (96,96)
    else:
        args.test_notebook = False

    return args

class Trainer(object):
    def __init__(self, model, args, experiment):
        self.model = model
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
        self.experiment.log_multiple_metrics(dict(loss=avg_loss,acc=avg_acc),prefix=mode, step=self.epoch)
        if mode == "train":
            print("Epoch %i: "%self.epoch)
        print("\t%s"%mode)
        if args.mode == "eval" or args.mode == "test":
            print("\t%s"%args.label_name)
        print("\t\tLoss: %8.4f \n\t\tAccuracy: %9.3f%%"%(avg_loss, 100*avg_acc))
        return avg_loss, avg_acc
    
    def test(self,test_set):
        self.model.eval()
        test_loss, test_acc = self.one_epoch(test_set,mode="test")
        self.experiment.log_metric("test_acc",test_acc)
        return test_acc
        
    def train(self, tr_buf, val_buf, model_dir):
        state_dict = self.model.state_dict()
        val_acc = -np.inf
        best_val_loss = np.inf
        while self.epoch < self.max_epochs:
            self.epoch+=1
            self.model.train()
            self.experiment.train()
            tr_loss,tr_acc = self.one_epoch(tr_buf,mode="train")
            torch.save(self.model.state_dict(), model_dir / "cur_model.pt")
            
            self.model.eval()
            self.experiment.validate()
            val_loss, val_acc = self.one_epoch(val_buf,mode="val")
            if self.epoch == 1:
                torch.save(self.model.state_dict(), model_dir / Path(  ("best_model_%f.pt"%val_loss).rstrip('0').rstrip('.')))
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                for f in model_dir.glob("best_model*"):
                    os.remove(str(f))
                torch.save(self.model.state_dict(), model_dir / Path(  ("best_model_%f.pt"%best_val_loss).rstrip('0').rstrip('.')))



def setup_model(args, env):
    model_table = {"inv_model":InverseModel, "vae":VAE, "raw_pixel": RawPixelsEncoder,
                                 "lin_proj": RandomLinearProjection,
                                 "rand_cnn": RandomWeightCNN,  "beta_vae": BetaVAE }
    encoder_kwargs = dict(in_ch=3,im_wh=args.resize_to,h_ch=args.hidden_width, embed_len=args.embed_len,  
                        num_actions=env.action_space.n, beta=args.beta)

    base_model = model_table[args.model_name](**encoder_kwargs).to(args.device)
    if args.mode == "eval" or args.mode == "test":
        
        eval_model = EvalModel(encoder=base_model.encoder,
                       num_classes=env.nclasses_table[args.label_name],
                       label_name=args.label_name, model_type="classifier").to(args.device)
        
        weights_path = get_weights_path(args)
        if args.mode == "eval":
            if weights_path:
                eval_model.encoder.load_state_dict(torch.load(str(weights_path)))
            else:
                print("No weights available for %s. Using randomly initialized %s"%(args.model_name,args.model_name))
        if args.mode == "test":
            if weights_path:
                eval_model.load_state_dict(torch.load(str(weights_path)))
            else:
                print("No weights available for best eval model")
            
        
       

        model = eval_model
        
    else: # if args.mode=train
        model = base_model
        
        
    return model    


# In[3]:

def setup_exp(args):
    exp_name = ("nb_" if args.test_notebook else "") + "_".join([args.mode, args.model_name, get_hyp_str(args)])
    experiment = Experiment(api_key="kH9YI2iv3Ks9Hva5tyPW9FAbx",
                            project_name="self-supervised-survey",
                            workspace="eracah")
    experiment.set_name(exp_name)
    experiment.log_multiple_params(args.__dict__)
    return experiment



def setup_dir(args,exp_id,basename=".models"):
    dir_ = Path(basename) / get_child_dir(args,mode=args.mode) / Path(exp_id)
    dir_.mkdir(exist_ok=True,parents=True)
    return dir_


# In[2]:


if __name__ == "__main__":
    args = setup_args()
    experiment = setup_exp(args)
    env = setup_env(args)
    
    print("starting to load buffers")
    bufs = setup_tr_val_test(args)
    
    model_dir = setup_dir(basename=".models",args=args,exp_id=experiment.id)
    ims_dir = setup_dir(basename=".images",args=args,exp_id=experiment.id)
    
    model = setup_model(args, env)
    #update params
    experiment.log_multiple_params(args.__dict__)
    
    trainer = Trainer(model, args, experiment)
    if args.mode == "test":
        trainer.test(bufs[0])
    else:
        trainer.train(*bufs,model_dir)

