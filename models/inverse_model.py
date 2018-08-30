
# coding: utf-8

# In[10]:


import torch
from torch import nn
import torch.functional as F


# In[11]:


class ActionPredictor(nn.Module):
    def __init__(self, num_actions, in_ch, h_ch=256):
        super(ActionPredictor,self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=in_ch,out_features=h_ch),
            nn.ReLU(),
            nn.Linear(in_features=h_ch,out_features=num_actions)
        )
    def forward(self,x):
        return self.predictor(x)
        

class InverseModel(nn.Module):
    def __init__(self,encoder, num_actions=3):
        super(InverseModel,self).__init__()
        self.encoder = encoder
        self.ap = ActionPredictor(num_actions=num_actions,in_ch=2*self.encoder.embed_len)
    def forward(self,x0,x1):
        f0 = self.encoder(x0)
        f1 = self.encoder(x1)
        fboth = torch.cat([f0,f1],dim=-1)
        return self.ap(fboth)

