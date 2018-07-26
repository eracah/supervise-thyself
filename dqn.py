
# coding: utf-8

# In[ ]:


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
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam, RMSprop
import argparse
import sys
import copy
from copy import deepcopy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from utils import convert_frame
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer
from models import InverseModel, QNet, Encoder
from utils import mkstr, initialize_weights, write_ims, write_to_config_file,collect_one_data_point


# In[ ]:


def get_q_loss(x0,x1,a,r, dones, gamma, q_net, target_q_net,):
    qbootstrap = gamma * torch.max(target_q_net(x1).detach(),dim=1)[0]
    # zero out bootstraps for states that are the last state
    qbootsrap = (1-torch.tensor(dones)).cuda().float() * qbootstrap
    y = r + qbootstrap
    #print(dones)
    q_vals = torch.gather(q_net(x0),1,a[:,None])[:,0]
    error = y - q_vals
    error = torch.clamp(error,-1.0,1.0)
    #print(error)
    q_loss = torch.sum(error**2)
    return q_loss


# In[ ]:


def e_greedy(q_values,epsilon=0.1):
    r = np.random.uniform(0,1)
    if r < epsilon:
        action = np.random.choice(len(q_values))
    else:
        action = np.argmax(q_values)
    return action


# In[ ]:


class QNet(nn.Module):
    def __init__(self,encoder,num_actions=3, h_ch=32):
        super(QNet,self).__init__()
        self.encoder = encoder
        self.q_enc = nn.Sequential(
            nn.Linear(in_features=self.encoder.embed_len,out_features=h_ch),
            nn.ELU(),
            nn.Linear(in_features=h_ch,out_features=h_ch),
            nn.ELU(),
            nn.Linear(in_features=h_ch,out_features=num_actions)
        )
    def forward(self,x):
        h = self.encoder(x)
        return self.q_enc(h)


# In[ ]:


def qpolicy(x0,q_net,epsilon=0.1):
    q_values = q_net(x0[None,:])[0].cpu().data.numpy()
    action = e_greedy(q_values,epsilon=epsilon)
    return int(action)

