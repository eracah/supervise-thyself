
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
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
#from replay_buffer import ReplayMemory, fill_replay_buffer
from base_encoder import Encoder


# In[6]:


class RawPixelsEncoder(nn.Module):
    def __init__(self, im_wh=(64,64),in_ch=3):
        super(RawPixelsEncoder,self).__init__()
        self.embed_len = np.prod(im_wh) * in_ch
    def forward(self,x):
        return x.view(x.size(0),-1)

class RandomLinearProjection(nn.Module):
    def __init__(self,im_wh=(64,64),in_ch=3, embed_len=32 ):
        super(RandomLinearProjection,self).__init__()
        self.embed_len = embed_len
        self.input_len = np.prod(im_wh) * in_ch
        self.fc = nn.Linear(in_features=self.input_len,out_features=self.embed_len)
    def forward(self,x):
        vec = x.view(x.size(0),-1)
        return self.fc(vec)
        

class RandomWeightCNN(Encoder):
    def __init__(self,im_wh=(64,64),in_ch=3,
                 h_ch=32,embed_len=32, 
                 batch_norm=False):
        super(RandomWeightCNN,self).__init__(im_wh=im_wh,in_ch=in_ch,
                 h_ch=h_ch,embed_len=embed_len, 
                 batch_norm=batch_norm)
        

