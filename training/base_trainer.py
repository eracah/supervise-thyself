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
        self.experiment = experiment
        self.epoch=0
        self.max_epochs = 10000
        self.last_epoch_logged = 0
        self.label_name = self.args.label_name
        
        print("%s, %s"%(args.mode, args.task))
        if self.args.needs_labels:
            print("\t %s"%(self.label_name))

    def one_iter(self, trans, update_weights=True):
        raise NotImplementedError
    
    def one_epoch(self, buffer,mode="train"):
        raise NotImplementedError
    
    def test(self,test_set):
        raise NotImplementedError
    
    def save_model(self,model, model_dir, name):
        # save as cpu cuz its easy during loading to switch weights from cpu to gpu and not other way around
        state_dict = model.encoder.cpu().state_dict() if self.args.task == "embed" else model.cpu().state_dict()
        save_path = model_dir / Path((name).rstrip('0').rstrip('.'))
        torch.save(state_dict, save_path )
        # put it back to device
        model.to(self.args.device)
        
    def log_metric(self,key,value):
        if value is None:
            return
        if self.args.mode == "train" and self.last_epoch_logged < self.epoch:
            print("Epoch %i: "%self.epoch)
            self.last_epoch_logged = copy.deepcopy(self.epoch)
            
            
        if isinstance(value,dict):
            self.experiment.log_metrics(dic=value,prefix=key, step=self.epoch)
            print("\t\t%s: "%(key))
            for k,v in value.items():
                if v is None:
                    continue
                self.printkv(k,v)
                
                
        else:
            self.printkv(key,value)
            self.experiment.log_metric(name=key,value=value, step=self.epoch)

            
        
    def printkv(self,k,v):
        perc = "%\n" if "acc" in k else ""
        print("\t\t\t%s: %8.4f%s"%(k,v,perc))
        
        
        
        
    def replace_best_model(self, model_dir):
        old = [f for f in model_dir.glob("best_model*")]
        for f in old:
            os.remove(str(f))
        
    def train(self, tr_buf, val_buf, model_dir):
        raise NotImplementedError
        
    def collect_embeddings_megabatch(self,test_set, encoder):
        raise NotImplementedError
        
        