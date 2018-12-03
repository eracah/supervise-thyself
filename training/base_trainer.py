import copy
import torch
from torch.optim import Adam
import numpy as np
from pathlib import Path
import os

class BaseTrainer(object):
    def __init__(self, model, args, experiment):
        self.model = model
        self.args = args
        self.model_name = self.args.model_name
        self.experiment = experiment
        self.epoch=0
        self.max_epochs = 10000

    def one_iter(self, trans, update_weights=True):
        raise NotImplementedError
    
    def one_epoch(self, buffer,mode="train"):
        raise NotImplementedError
    
    def test(self,test_set):
        raise NotImplementedError
    
    def save_model(self,model, model_dir, name):
        state_dict = model.encoder.state_dict() if self.args.mode == "train" else model.state_dict()
        save_path = model_dir / Path((name).rstrip('0').rstrip('.'))
        torch.save(state_dict, save_path )
        
    def replace_best_model(self, model_dir):
        old = [f for f in model_dir.glob("best_model*")]
        for f in old:
            os.remove(str(f))
        
    def train(self, tr_buf, val_buf, model_dir):
        raise NotImplementedError