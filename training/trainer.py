import copy
import torch
from torch.optim import Adam
import numpy as np
from pathlib import Path
import os

class Trainer(object):
    def __init__(self, model, args, experiment):
        self.model = model
        self.args = args
        self.model_name = self.args.model_name
        self.experiment = experiment
        self.opt = Adam(params=self.model.parameters(),lr=self.args.lr)
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
        
        if mode == "train":
            print("Epoch %i: "%self.epoch)
        print("\t%s"%mode)
        if self.args.mode == "eval" or self.args.mode == "test":
            print("\t %s"%(self.args.label_name))
        
        avg_loss = np.mean(losses)
        self.experiment.log_metric(avg_loss, mode + "_loss", step=self.epoch)
        print("\t\tLoss: %8.4f"%(avg_loss))
        if None in accs:
            avg_acc =None
        else:
            avg_acc = np.mean(accs)
            self.experiment.log_metric(avg_acc, mode + "_acc", step=self.epoch)
            print("\t\tAccuracy: %9.3f%%"%(100*avg_acc))
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
            state_dict = self.model.encoder.state_dict() if self.args.mode == "train" else self.model.state_dict()
            torch.save(state_dict, model_dir / "cur_model.pt")
            
            self.model.eval()
            self.experiment.validate()
            val_loss, val_acc = self.one_epoch(val_buf,mode="val")
            
            if self.epoch == 1 or val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                old = [f for f in model_dir.glob("best_model*")]
                for f in old:
                    os.remove(str(f))
                #print("hey")
                save_path = model_dir / Path(("best_model_%f.pt"%best_val_loss).rstrip('0').rstrip('.'))
                #print(save_path)
                torch.save(state_dict,save_path )