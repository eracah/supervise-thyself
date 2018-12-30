import torch
from torch import nn
import torch.functional as F
import numpy as np
from evaluations.linear_model import LinearModel
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

class PredictEvalModel(nn.Module):
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

    def forward_one_step(self,ft, a):

        a = convert_to1hot(a,self.n_actions).float()
        #double check make sure embedding is detached
        if ft.requires_grad:
            #print("eeek")
            ft = ft.detach()
        inp = torch.cat((ft,a),dim=1)
        ftp1_pred = self.fc(inp)
        return ftp1_pred
    
    def get_f_trues(self,trans):
        f_trues = [self.encoder(trans.xs[:,i]).detach() 
                    for i in range(1, trans.actions.shape[1] + 1) ]
        f_trues = torch.stack(f_trues,dim=1)
        return f_trues
        
    
    def _loss(self,trans):
        f_preds = self.forward(trans)
        f_trues = self.get_f_trues(trans)
       
        loss = nn.MSELoss()(f_trues,f_preds)
        acc = None # no accuracy
        return loss, acc
            
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data
    
    
    

