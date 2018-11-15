
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
from training.trainer import Trainer


# In[3]:


if __name__ == "__main__":
    args = setup_args()
    args.model_name = "rand_cnn"
    args.base_enc_name = "world_models"
    
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
    trainer = Trainer(model, args, experiment)
    if args.mode == "test":
        trainer.test(bufs[0])
    else:
        trainer.train(*bufs,model_dir)

