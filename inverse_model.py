
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


# In[ ]:


def get_im_loss_acc(x0,x1,a):
    y = a
    a_pred = inv_model(x0,x1)
    im_loss = im_criterion(a_pred,y)
    action_guess = torch.argmax(a_pred,dim=1)
    acc = (float(torch.sum(torch.eq(y,action_guess)).data) / y.size(0))*100
    return im_loss, acc

