
# coding: utf-8

# In[1]:


import torch

from torch import nn

import torch.functional as F
import numpy as np


# In[59]:


class Gate(nn.Module):
    def __init__(self,embed_len=32, init_gamma=1.0, max_gamma=100, sharpen_rate=10):
        super(Gate,self).__init__()
        self.embed_len = embed_len
        self.fc = nn.Linear(in_features=2*self.embed_len, out_features=self.embed_len)
        self.noise_dist = torch.distributions.Normal(0,0.01)
        self.init_gamma = init_gamma
        self.gamma = init_gamma
        self.max_gamma = max_gamma
        self.sharpen_rate = 10
        self.iteration = 0
    
    def update_gamma(self):
        self.iteration += 1
        self.gamma = min(self.init_gamma + (self.iteration / 10000) * self.sharpen_rate, self.max_gamma)
        
    def sharpen(self, logits):
        relu_noise = nn.ReLU()(self.noise_dist.sample(logits.size()))
        relu_logits = nn.ReLU()(logits)
        numerator = (relu_logits + relu_noise)**self.gamma 
        denominator = torch.sum(relu_logits**self.gamma)
        weights = numerator / denominator
        weights = torch.clamp(weights,0,1)
        return weights
        
    
    def forward(self,f0,f1):
        fboth = torch.cat([f0,f1],dim=-1)
        logits = self.fc(fboth)
        weights = self.sharpen(logits)
        new_f1 = f0*(1 - weights) + f1*weights
        self.update_gamma()
        return f0, new_f1
        
        
    


# In[60]:


class ContinuationLearningEncoder(nn.Module):
    def __init__(self,base_encoder, embed_len=32, init_gamma=1.0, max_gamma=100, sharpen_rate=10):
        super(ContinuationLearningEncoder,self).__init__()
        self.base_encoder = base_encoder
        self.gate = Gate(embed_len, init_gamma, max_gamma, sharpen_rate)
    def forward(self,x0,x1):
        f0 = self.base_encoder(x0)
        f1 = self.base_encoder(x1)
        return self.gate(f0,f1)
    

