
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam
import argparse
import sys
import copy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from load_data import get_tensor_data_loaders
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer


# In[4]:


class Encoder(nn.Module):
    def __init__(self,in_ch=3,h_ch=32, batch_norm=False):
        super(Encoder,self).__init__()
        bias= False if batch_norm else True
            
        layers = [nn.Conv2d(in_channels=in_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU()
                 ]
        if not batch_norm:
            for layer in layers:
                if "BatchNorm" in str(layer):
                    layers.remove(layer)
        self.encoder = nn.Sequential(*layers)
                    
        self.fc = nn.Linear(in_features=h_ch, out_features=h_ch)

    def get_output_shape(self,inp_shape):
        a = torch.randn(inp_shape)
        return self.forward(a).size(1)
    
#     def get_feature_maps(self):
#         return self.fmaps

    def forward(self,x):
        fmaps = self.encoder(x)
        vec = fmaps.view(fmaps.size(0),-1)
        self.fmaps = fmaps
        return vec


# In[6]:


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
        


# In[8]:


class InverseModel(nn.Module):
    def __init__(self,in_ch,im_wh, h_ch, num_actions, batch_norm):
        super(InverseModel,self).__init__()
        self.enc = Encoder(in_ch=in_ch,h_ch=h_ch, batch_norm=batch_norm)
        
        embed_len = self.enc.get_output_shape((1, in_ch, *im_wh))
        self.ap = ActionPredictor(num_actions=num_actions,in_ch=2*embed_len)
    def forward(self,x0,x1):
        f0 = self.enc(x0)
        f1 = self.enc(x1)
        fboth = torch.cat([f0,f1],dim=-1)
        return self.ap(fboth)


# In[5]:


# enc = Encoder(batch_norm=True)

# x = torch.randn(8,3,64,64)

# vec = enc(x)

# print(vec.size())
# print(enc.get_output_shape((8,3,64,64)))

# enc = Encoder(batch_norm=True)

# x1 = torch.randn(8,3,64,64)
# x2 = torch.randn(8,3,64,64)
# vec1 = enc(x1)
# vec2 = enc(x2)
# vec = torch.cat((vec1,vec2),dim=-1)
# ap = ActionPredictor(3,1024)

# logits = ap(vec)
# print(logits.size())

# prd = InverseModel(in_ch=3,im_wh=(64,64),h_ch=32,num_actions=4,batch_norm=False)

# x1 = torch.randn(8,3,64,64)
# x2 = torch.randn(8,3,64,64)

# prd(x1,x2)

# prd.parameters()

# nn.CrossEntropyLoss?


