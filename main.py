#!/usr/bin/env python
# coding: utf-8

# In[1]:


from comet_ml import Experiment # comet must come before any torch modules. I don't know why?
import random
from models.setup import setup_model
from data.env_utils.env_setup import setup_env
import argparse
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
from data.splitter import setup_tr_val_test
import os
from utils import get_child_dir, get_hyp_str, setup_args, setup_dir, setup_exp


# In[1]:


def setup_all(args):        
    experiment, exp_id = setup_exp(args)
    print(exp_id)
    env = setup_env(args)
    experiment.log_parameters(args.__dict__)
    data = setup_tr_val_test(args)
    model = setup_model(args)
    model_dir = setup_dir(basename=".models",args=args,exp_id=exp_id)
    ims_dir = setup_dir(basename=".images",args=args,exp_id=exp_id)
    return data, model, experiment, model_dir, ims_dir
    

if __name__ == "__main__":
    args = setup_args()
    print(args.mode,args.task)
    data, model, experiment, model_dir, ims_dir = setup_all(args)
   

    if args.task == "embed":
        from training.embed_trainer import EmbedTrainer
        trainer = EmbedTrainer(model, args, experiment)
    
    elif args.task == "infer":
        from training.inference_trainer import InferenceTrainer
        trainer = InferenceTrainer(model, args, experiment)
        
    elif args.task == "predict":
        from training.prediction_trainer import PredictionTrainer
        trainer = PredictionTrainer(model, args, experiment)
        
    elif args.task == "control":
        from training.control_trainer import ControlTrainer
        trainer = ControlTrainer(model, args, experiment)
    else:
        assert False, "no other type of Trainer"

    if args.mode == "train":
        tr, test = data
        tr_kwargs = dict(model_dir=model_dir,tr_buf=tr, val_buf=test)
        trainer.train(**tr_kwargs)
        
    elif args.mode == "test":
        test, = data
        test_kwargs = dict(test_set=test)
        trainer.test(**test_kwargs)


# In[ ]:




