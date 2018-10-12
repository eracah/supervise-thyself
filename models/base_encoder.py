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
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
#from replay_buffer import ReplayMemory, fill_replay_buffer


# In[10]:


class Encoder(nn.Module):
    def __init__(self,im_wh=(64,64),in_ch=3,
                 h_ch=32,embed_len=32, 
                 batch_norm=False):
        super(Encoder,self).__init__()
        self.bias= False if batch_norm else True
        self.im_wh = im_wh 
        self.in_ch = in_ch
        self.h_ch = h_ch
        self.embed_len = embed_len
        self.vec = None
        layers = [nn.Conv2d(in_channels=self.in_ch, out_channels=self.h_ch,
                      kernel_size=3, stride=2, padding=1,bias=self.bias),
            nn.BatchNorm2d(self.h_ch),
            nn.ELU(),
            
            nn.Conv2d(in_channels=self.h_ch, out_channels=self.h_ch,
                      kernel_size=3, stride=2, padding=1,bias=self.bias),
            nn.BatchNorm2d(self.h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=self.h_ch, out_channels=self.h_ch,
                      kernel_size=3, stride=2, padding=1,bias=self.bias),
            nn.BatchNorm2d(self.h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=self.h_ch, out_channels=self.h_ch,
                      kernel_size=3, stride=2, padding=1,bias=self.bias),
            nn.BatchNorm2d(self.h_ch),
            nn.ELU()
                 ]
        if not batch_norm:
            for layer in layers:
                if "BatchNorm" in str(layer):
                    layers.remove(layer)
        self.encoder = nn.Sequential(*layers)
                    
        self.fc = nn.Linear(in_features=self.enc_out_shape,
                            out_features=self.embed_len)



    @property
    def enc_out_shape(self):
        return int(np.prod(self.last_im_shape))
    
    @property
    def last_im_shape(self):
        inp_shape = (1,self.in_ch,*self.im_wh)
        a = torch.randn(inp_shape)
        return self.encoder(a).size()[1:]
    

    def forward(self,x):
        fmaps = self.encoder(x)
        self.vec = fmaps.view(fmaps.size(0),-1)
        embedding = self.fc(self.vec)
        return embedding