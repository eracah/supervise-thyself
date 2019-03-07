
# coding: utf-8

# In[1]:


from comet_ml import Experiment # comet must come before any torch modules. I don't know why?
import random
from models.setup import setup_model
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


# In[2]:


if __name__ == "__main__":
    args = setup_args()
    experiment = setup_exp(args)
    model = setup_model(args)
    data = setup_tr_val_test(args)

 

  

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
    elif args.task == "viz":
        pass
    else:
        assert False, "no other type of Trainer"

    if args.mode == "train":
        tr, test = data
        tr_kwargs = dict(model_dir=args.models_dir,tr_buf=tr, val_buf=test)
        trainer.train(**tr_kwargs)
        
    elif args.mode == "test":
        test, = data
        test_kwargs = dict(test_set=test)
        trainer.test(**test_kwargs)
        
    elif args.mode == "viz":
        test, = data
        get_ipython().run_line_magic('matplotlib', 'inline')
        from evaluations.fmap_superimpose import superimpose_seq_frames
        emb_fmap_dict = {'inv_model':67, 'vae':95,"tdc":212, "snl":54, "rand_cnn":8}
        for embedder_name, fmap_index in emb_fmap_dict.items():
            args.embedder_name = embedder_name
            model = setup_model(args)
            encoder = model.encoder if embedder_name is not "rand_cnn" else model
            model_name = model.__class__.__name__ if embedder_name is not "rand_cnn" else "RandCNN"
            superimpose_seq_frames(encoder,model_name,test,fmap_index=fmap_index)
        


# In[ ]:

