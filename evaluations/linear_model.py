import torch
from torch import nn
import torch.functional as F
import numpy as np

# used for prediction or inference just depends on whether y is the next state and if the embedding    
class LinearModel(nn.Module):
    def __init__(self, input_len,output_len): #,lasso_coeff=0.):
        super(LinearModel,self).__init__()
        self.fc = nn.Linear(in_features=input_len, out_features=output_len)        
    
    def forward(self, embeddings):
        #make sure embedding is detached
        if embeddings.requires_grad:
            print("eeek")
            embeddings = embeddings.detach()
        out = self.fc(embeddings)
        return out
    
     
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data
    