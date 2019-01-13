import torch
import numpy as np
from torch import nn
from evaluations.linear_model import LinearModel
import torch.functional as F
import numpy as np
from utils import convert_to1hot
import copy

class PredictModel(nn.Module):
    def __init__(self, encoder, args):
        super(PredictModel,self).__init__()
        self.args = args
        self.encoder = encoder
        self.embed_len = self.encoder.embed_len
        self.num_actions = self.args.num_actions
        self.predictor = OneStepForwardModel(self.encoder, self.num_actions)
    
    def embed_state(self,trans,index):
        t = index
        xt = trans.xs[:,t]
        ft = self.encoder(xt).detach()
        return ft
    
    def get_all_embeddings(self,trans):
        fs = [self.embed_state(trans,i) for i in range(trans.xs.shape[1])]
        return fs
            
    def forward(self, trans):

        f_trues = self.get_all_embeddings(trans)
        f_preds = []

        # predict n steps forward where n = frames_per_example - 1
        for t in range(trans.actions.shape[1]):
            if self.args.mode == "train": # teacher forcing
                ft = f_trues[t]
            elif self.args.mode == "test": #non teacher forcing
                ft = f_trues[t] if t == 0 else ftp1_pred
            at = trans.actions[:,t]
            ftp1_pred = self.predictor(ft,at)
            f_preds.append(ftp1_pred)
           
        f_preds = torch.stack(f_preds,dim=1)
        
        return f_preds
        
    
    def loss_acc(self,trans):
        f_trues = torch.stack(self.get_all_embeddings(trans), dim=1)
        f_preds = self.forward(trans)

        
        losses = []
        for i in range(f_preds.shape[1]):
            f_pred = f_preds[:,i]
            f_true = f_trues[:,i+1] # we don't predict f0
            loss = self.predictor.loss(f_true, f_pred)
            losses.append(loss)

        loss = torch.mean(torch.stack(losses,dim=0))
        acc = None
        return loss,acc
    
    @property
    def importance_matrix(self):
        return self.predcitor.fc.weight.abs().transpose(1,0).data
        
    

class OneStepForwardModel(nn.Module):
    """Takes embedding of state and one-hot encoded action and predicts embedding of next state"""
    def __init__(self,encoder, num_actions):
        super(OneStepForwardModel,self).__init__()
        self.encoder = encoder
        self.embed_len = self.encoder.embed_len
        self.num_actions = num_actions
        self.fc = LinearModel(input_len=self.embed_len + self.num_actions, output_len=self.embed_len)
        
        
    def forward(self,ft, a):

        a = convert_to1hot(a,self.num_actions).float()
        #double check make sure embedding is detached
        if ft.requires_grad:
            #print("eeek")
            ft = ft.detach()
        inp = torch.cat((ft,a),dim=1)
        ftp1_pred = self.fc(inp)
        return ftp1_pred
           
    def loss(self,f_true, f_pred):    
        loss = nn.MSELoss()(f_true,f_pred)
        return loss


            

        