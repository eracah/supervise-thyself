import copy
import torch
from torch.optim import Adam
import numpy as np
from pathlib import Path
import os
from training.base_trainer import BaseTrainer
from evaluations.pca_corr_model import PCACorr

class InferenceTrainer(BaseTrainer):
    def __init__(self, model, args, experiment):
        super(InferenceTrainer, self).__init__(model, args, experiment)
        self.opt = Adam(params=self.model.parameters(),lr=self.args.lr)


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
        
        self.log_metric(key = mode + "_loss",value=avg_loss)

#         self.experiment.log_metric(name=mode + "_loss",value=avg_loss, step=self.epoch)
#         print("\t\tLoss: %8.4f"%(avg_loss))
        avg_acc = np.mean(accs) if None not in accs else None
        if avg_acc:
            self.log_metric(key=mode + "_acc",value=100*avg_acc)
#             self.experiment.log_metric(name=mode + "_acc",value=avg_acc, step=self.epoch)
#             print("\t\tAccuracy: %9.3f%%"%(100*avg_acc))
        return avg_loss, avg_acc
    
    def test(self,test_set):
        pcc = PCACorr(self.model.encoder,test_set)
        pearson_corr, evr = pcc.run()
        self.model.eval()
        test_loss, test_acc = self.one_epoch(test_set,mode="test")
        self.log_metric(key="evr_pc1",value=evr)
        self.log_metric(key="pearson",value=pearson_corr)
        return test_acc,pearson_corr,evr
        
    def train(self, model_dir, tr_buf, val_buf):
        val_acc = -np.inf
        best_val_loss = np.inf
        while self.epoch < self.max_epochs:
            self.epoch+=1
            self.model.train()
            tr_loss,tr_acc = self.one_epoch(tr_buf,mode="train")
            self.save_model(self.model, model_dir, "cur_model.pt" )
            self.model.eval()
            val_loss, val_acc = self.one_epoch(val_buf,mode="val")
            
            if self.epoch == 1 or val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                self.replace_best_model(model_dir)
                self.save_model(self.model, model_dir, "best_model_%f.pt"%best_val_loss)