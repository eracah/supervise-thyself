import torch
from torch import nn
import torch.functional as F
import numpy as np

def convert_to1hot(a,n_actions):
    dims = a.size()
    batch_size = dims[0]
    if len(dims) < 2:
        a = a[:,None]
    
    a = a.long()
    a_1hot = torch.zeros((batch_size,n_actions)).long().to(a.device)

    src = torch.ones_like(a).to(a.device)

    a_1hot = a_1hot.scatter_(dim=1,index=a,src=src)
    return a_1hot

class ForwardModel(nn.Module):
    """Takes embedding of state and one-hot encoded action and predicts embedding of next state"""
    def __init__(self,encoder, n_actions):
        super(ForwardModel,self).__init__()
        self.encoder = encoder
        self.embed_len = self.encoder.embed_len
        self.n_actions = n_actions
        self.fc = nn.Linear(in_features=self.embed_len + n_actions, out_features=self.embed_len)
        
    def forward(self, trans):
        a = trans.actions[:,0]
        a = convert_to1hot(a,self.n_actions).float()
        x1 = trans.xs[:,0]
        """f1 is embedding of a frame and a is one-hot encoded action"""
        f1 = self.encoder(x1).detach()
        #make sure embedding is detached
        if f1.requires_grad:
            print("eeek")
            f1 = f1.detach()
        inp = torch.cat((f1,a),dim=1)
        f2_pred = self.fc(inp)
        return f2_pred

    def loss_acc(self,trans):
        x2 = trans.xs[:,1]
        f2 = self.encoder(x2).detach()
        f2_pred = self.forward(trans)
        
        loss = nn.MSELoss()(f2,f2_pred)
        acc = None # no accuracy
        return loss, acc
            
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data
    
    
    
    

