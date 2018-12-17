
# coding: utf-8

# In[2]:


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
from training.inference_trainer import InferenceTrainer
from training.prediction_trainer import PredictionTrainer


if __name__ == "__main__":
    args = setup_args()
    print(args.device)
#     args.model_name = "tdc"
#     args.mode = "train"
    args.use_comet = False
    args.frames_per_example = 10
    args.tr_size = 60
    args.mode = "eval"
    args.model_name = "forward_rand_cnn"
#     args.env_name = "Pong-v0"
#     args.lr = 0.00001

    
    experiment, exp_id = setup_exp(args)
    env = setup_env(args)
    print("starting to load buffers")
    bufs = setup_tr_val_test(args)
    
    # setup models before dirs because some args get changed in this fxn
    model = setup_model(args)

    model_dir = setup_dir(basename=".models",args=args,exp_id=exp_id)
    print(model_dir)
    ims_dir = setup_dir(basename=".images",args=args,exp_id=exp_id)

    #update params
    try:
        experiment.log_multiple_params(args.__dict__)
    except:
        pass


    if "ctl" in args.mode:
        from training.control_trainer import ControlTrainer
        trainer = ControlTrainer(model, args, experiment)
        tr_kwargs = dict(model_dir=model_dir)
        test_kwargs = {}
    else:
        trainer = InferenceTrainer(model, args, experiment)


    if "test" in args.mode:
        test_kwargs = dict(test_set=bufs[0])
        trainer.test(**test_kwargs)
    else:
        tr_kwargs = dict(model_dir=model_dir,tr_buf=bufs[0], val_buf=bufs[1])
        trainer.train(**tr_kwargs)

