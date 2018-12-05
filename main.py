
# coding: utf-8

# In[1]:


from comet_ml import Experiment # comet must come before any torch modules. I don't know why?
import random
from models.setup import setup_model
from data.utils import setup_env
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
from data.tr_val_test_splitter import setup_tr_val_test
import os
from utils import get_child_dir, get_hyp_str, setup_args, setup_dir, setup_exp
from training.inference_trainer import InferenceTrainer
from training.prediction_trainer import PredictionTrainer
from training.control_trainer import ControlTrainer

if __name__ == "__main__":
    args = setup_args()
#     args.model_name = "rand_cnn"
#     args.base_enc_name = "world_models"
#     args.mode = "eval_ctl"
#     args.env_name = "WaterWorld-v0"
#     args.resize_to = (128,128)
#     args.lr = 0.1
#     args.frames_per_trans = 5
    #args.tr_size = 1000
    
    experiment = setup_exp(args)
    env = setup_env(args)
    print("starting to load buffers")
    bufs = setup_tr_val_test(args)
    
    # setup models before dirs because some args get changed in this fxn
    model = setup_model(args, env)
    

    model_dir = setup_dir(basename=".models",args=args,exp_id=experiment.id)
    print(model_dir)
    ims_dir = setup_dir(basename=".images",args=args,exp_id=experiment.id)
    
    #update params
    experiment.log_multiple_params(args.__dict__)
    
    if "ctl" in args.mode:
        trainer = ControlTrainer(model, args, experiment)
        tr_kwargs = dict(model_dir=model_dir)
        test_kwargs = {}
    else:
        trainer = InferenceTrainer(model, args, experiment)
        tr_kwargs = dict(model_dir=model_dir,tr_buf=bufs[0], val_buf=bufs[1])
        test_kwargs = dict(test_set=bufs[0])
    
    if "test" in args.mode:
        trainer.test(**test_kwargs)
    else:
        trainer.train(**tr_kwargs)

