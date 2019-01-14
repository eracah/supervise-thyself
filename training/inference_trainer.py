import copy
import torch
from torch.optim import Adam
import numpy as np
from pathlib import Path
import os
from training.base_trainer import BaseTrainer
from evaluations.pca_corr_model import compute_pca_corr
from evaluations.fmap_superimpose import superimpose_fmaps

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
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        update_weights = True if mode=="train" else False
        losses, accs = [], []
        for trans in buffer:
            loss,acc = self.one_iter(trans,update_weights=update_weights)
            
            losses.append(loss)
            accs.append(acc)
        

        avg_loss = np.mean(losses)
        
        self.log_metric(key = mode + "_loss",value=avg_loss)
        avg_acc = np.mean(accs) if None not in accs else None
        if None not in accs:
            self.log_metric(key=mode + "_acc",value=100*avg_acc)
        return avg_loss, avg_acc

    def do_pca_corr(self,test_set, encoder):
        all_fs, all_ys = self.collect_embeddings_megabatch(test_set, encoder)
        sp_corr, evr = compute_pca_corr(embeddings=all_fs, labels=all_ys)
        
        self.log_metric(key="evr",value=evr)
        self.log_metric(key="spearman_corr",value=sp_corr)
        
    
    def test(self,test_set):
        self.one_epoch(test_set,mode="test")
        self.do_pca_corr(test_set, self.model.encoder)
        superimpose_fmaps(self.model.encoder, test_set, self.experiment)

        
    def train(self, model_dir, tr_buf, val_buf):
        best_val_loss = np.inf
        while self.epoch < self.max_epochs:
            self.epoch+=1
            self.one_epoch(tr_buf,mode="train")
            self.save_model(self.model, model_dir, "cur_model.pt" )
            val_loss, _ = self.one_epoch(val_buf,mode="val")
            
            if self.epoch == 1 or val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                self.replace_best_model(model_dir)
                self.save_model(self.model, model_dir, "best_model_%f.pt"%best_val_loss)
                
    def collect_embeddings_megabatch(self,test_set, encoder):
        fs = []
        ys = []
        for trans in test_set:
            x1 = trans.xs[:,0]
            f = encoder(x1)
            fs.append(f)

            y = trans.state_param_dict[self.label_name]
            ys.append(copy.deepcopy(y[:,0]))

        f = torch.cat(fs)
        y = torch.cat(ys).squeeze()
        return f.detach(), y
                
