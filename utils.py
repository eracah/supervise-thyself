
# coding: utf-8

# In[1]:


import sys
import copy
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np
import time
import json
from pathlib import Path


# In[2]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[3]:


def initialize_weights(self):
    # Official init from torch repo.
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


# In[4]:


def write_ims(index,rows,ims,name, iter_):
    num_ims = rows**2
    ims_grid = make_grid((ims.data[index] + 1) / 2, rows)
    writer.add_image(name, ims_grid, iter_)
    


# In[5]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[6]:


def save_incorrect_examples(y,action_guess,x0,x1,iter_):
    wrong_actions = y[torch.ne(action_guess,y)].long()
    num_wrong = wrong_actions.size(0)
    right_actions = y[torch.eq(action_guess,y)].long()
    num_right = right_actions.size(0)
    
    if iter_ % 50 == 0:
        try:
            write_ims(ims=x0,index=wrong_actions,rows=int(np.ceil(np.sqrt(num_wrong))),name=mode +"/debug/x0_wrong", iter_=iter_)
            write_ims(ims=x1,index=wrong_actions,rows=int(np.ceil(np.sqrt(num_wrong))),name=mode +"/debug/x1_wrong", iter_=iter_)
        except:
            print("Num wrong and right: ",num_wrong,num_right)

