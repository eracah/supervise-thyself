import torch
import numpy as np
from torch import nn
from evaluations.linear_model import LinearModel
import torch.functional as F
import numpy as np
from utils import convert_to1hot


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
        
    
    def loss_acc(self,trans):
        fs = [self.embed_state(trans,i) for i in range(trans.xs.shape[1])]
        
        losses = []
        for i in range(len(fs) - 1 ):
            loss = self.predictor.loss(fs[i],
                                       trans.actions[:,i],
                                       fs[i+1])
            losses.append(loss)

        loss = torch.mean(torch.stack(losses,dim=0))
        acc = None
        return loss,acc
        
    

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
           
    def loss(self,ft, a, ftp1):
        f_pred = self.forward(ft, a)
        f_true = ftp1
       
        loss = nn.MSELoss()(f_true,f_pred)
        return loss


class PredictTestModel(nn.Module):
    """Takes embedding of state and one-hot encoded action and predicts embedding of next state"""
    def __init__(self,encoder, n_actions):
        super(ForwardModel,self).__init__()
        self.encoder = encoder
        self.embed_len = self.encoder.embed_len
        self.n_actions = n_actions
        self.fc = nn.Linear(in_features=self.embed_len + n_actions, out_features=self.embed_len)
        
    def forward(self, trans):
        """f1 is embedding of a frame and a is one-hot encoded action"""
        x0 = trans.xs[:,0]
        f0 = self.encoder(x0).detach()
        ft = f0
        f_preds = []
        # predict n steps forward where n = frames_per_example - 1
        for t in range(trans.actions.shape[1]):
            at = trans.actions[:,t]
            ftp1_pred = self.forward_one_step(ft,at)
            f_preds.append(ftp1_pred)
            ft = ftp1_pred
            
            
        f_preds = torch.stack(f_preds,dim=1)
        
        return f_preds

            
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data
        