
# coding: utf-8

# In[9]:


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
from replay_buffer import ReplayMemory, fill_replay_buffer


# In[10]:


class Encoder(nn.Module):
    def __init__(self,im_wh=(64,64),in_ch=3,
                 h_ch=32,embed_len=32, 
                 batch_norm=False):
        super(Encoder,self).__init__()
        bias= False if batch_norm else True
        self.im_wh = im_wh 
        self.in_ch = in_ch
        self.embed_len = embed_len
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
                    
        self.fc = nn.Linear(in_features=self.enc_out_shape,
                            out_features=self.embed_len)

    @property
    def enc_out_shape(self):
        inp_shape = (1,self.in_ch,*self.im_wh)
        a = torch.randn(inp_shape)
        return np.prod(self.encoder(a).size()[1:])
    
#     def get_feature_maps(self):
#         return self.fmaps

    def forward(self,x):
        fmaps = self.encoder(x)
        vec = fmaps.view(fmaps.size(0),-1)
        embedding = self.fc(vec)
        return embedding


# In[11]:


class RawPixelsEncoder(nn.Module):
    def __init__(self, im_wh=(64,64),in_ch=3):
        super(RawPixelsEncoder,self).__init__()
        self.embed_len = np.prod(im_wh) * in_ch
    def forward(self,x):
        return x.view(x.size(0),-1)


# In[12]:


class RandomLinearProjection(nn.Module):
    def __init__(self,im_wh=(64,64),in_ch=3, embed_len=32 ):
        super(RandomLinearProjection,self).__init__()
        self.embed_len = embed_len
        self.input_len = np.prod(im_wh) * in_ch
        self.fc = nn.Linear(in_features=self.input_len,out_features=self.embed_len)
    def forward(self,x):
        vec = x.view(x.size(0),-1)
        return self.fc(vec)
        


# In[13]:


class RandomWeightCNN(Encoder):
    def __init__(im_wh=(64,64),in_ch=3,
                 h_ch=32,embed_len=32, 
                 batch_norm=False):
        super(RandomWeightCNN,self).__init__(im_wh=im_wh,in_ch=in_ch,
                 h_ch=h_ch,embed_len=embed_len, 
                 batch_norm=batch_norm)
        


# In[6]:


# if __name__ == "__main__":
#     rpe = RawPixelsEncoder()
#     x = torch.randn((10,3,64,64))
#     f = rpe(x)
#     print(f.size())
    


# In[3]:


# if __name__ == "__main__":
#     x = torch.randn((10,3,64,64))
#     x0 = x[0][None,:]
#     enc = Encoder(embed_len=32)
#     q = QNet(enc)
#     print(q(x0))
#     print(list(dict(q.named_parameters()).keys()))
    


# In[23]:


# if __name__ == "__main__":
#     x = torch.randn((10,3,64,64))
#     enc = Encoder(embed_len=64)
#     print(enc(x).size())
if __name__ == "__main__":
    x0 = torch.randn((10,3,64,64))
    x1 = torch.randn((10,3,64,64))
    enc = Encoder(embed_len=64)
    im = InverseModel(encoder=enc)

    print(im(x0,x1).size())
    print(list(dict(im.named_parameters()).keys()))

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


