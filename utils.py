
# coding: utf-8

# In[2]:


import sys
import copy
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np
import time
import json
from pathlib import Path
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from matplotlib import pyplot as plt
#%matplotlib inline

from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Compose,Normalize,Resize,ToTensor, Grayscale
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import time
import torch
from copy import deepcopy

from pathlib import Path
import time
import argparse
import sys
from functools import partial
import math
import sys
from torch.utils.data import TensorDataset


# In[14]:


def convert_frame(obs, resize_to=(84,84),to_tensor=False):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if to_tensor:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        frame= frame.to(DEVICE)
    else:
        frame = np.asarray(frame)


    
    return frame

def convert_frames(frames,resize_to=(64,64),to_tensor=False):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor)
    return torch.stack([convert(frame) for frame in frames])
        


# In[12]:


def collect_one_data_point(env, policy, convert_fxn):
    x0 = env.render("rgb_array")
    action = policy(convert_fxn(x0,to_tensor=True))
    x0 = convert_fxn(x0,to_tensor=False)
    _, reward, done, _ = env.step(action)
    x1 = convert_fxn(env.render("rgb_array"))
    return x0,x1,action,reward,done

def rollout_iterator(env=gym.make("MiniGrid-Empty-6x6-v0"),
                     policy=lambda x0: np.random.choice(3),
                     convert_fxn=convert_frame):
    _ = env.reset()
    done = False
    while not done:
        x0,x1,a,reward, done = collect_one_data_point(env, policy, convert_fxn)
        yield x0,x1,a,reward, done


# In[5]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[6]:


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


# In[7]:


def write_ims(index,rows,ims,name, iter_):
    num_ims = rows**2
    ims_grid = make_grid((ims.data[index] + 1) / 2, rows)
    writer.add_image(name, ims_grid, iter_)
    


# In[8]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[9]:


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


# In[11]:


def plot_test(x0s,x1s,ys,rs, label_list):
    def plot(x0,x1,y,i):
        from matplotlib import pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure(i)
        plt.clf()
        sp1 = plt.subplot(1,2,1)
        sp1.imshow(x0[0])
        sp2 = plt.subplot(1,2,2)
        sp2.imshow(x1[0])
        plt.title(label_list[y])
    if len(x0s.size()) == 3:
        plot(x0s,x1s,ys,i=1)
    else:
        for i in range(x0s.size()[0]):
            plot(x0s[i],x1s[i],ys[i],i)
    plt.show()

