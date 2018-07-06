
# coding: utf-8

# In[1]:


import gym

from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from matplotlib import pyplot as plt
#%matplotlib inline

from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Grayscale, Compose,Normalize,Resize,ToTensor
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
from data import convert_frame


# In[138]:


from torch.utils.data import TensorDataset


# In[33]:


def convert_frame(obs, new_shape=(64,64), pt_tensor=False, grayscale=False):
    pil_image = Image.fromarray(obs, 'RGB')
#     if grayscale:
#         state = state.convert("L")
    if new_shape != (-1,-1):
        transforms = [Resize(new_shape)]
    else:
        transforms = []
    if pt_tensor:
        if grayscale:
            transforms.append(Grayscale())
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    
    return frame


# In[118]:


def collect_one_data_point(env,obs, convert_fxn, action_space):
    x0 = deepcopy(obs)
    action = np.random.choice(action_space)
    obs, reward, done, info = env.step(action)
    obs = convert_fxn(env.render("rgb_array"))
    a = torch.from_numpy(np.asarray(action))
    x1 = deepcopy(obs)
    return x0,x1,a
    


# In[124]:


def do_rollout(env_name="MiniGrid-Empty-6x6-v0",
               resize_to = (64,64),
               rollout_size=128,
               action_space = [0,1,2],
               grayscale=False):
    
    xs = []
    ys = []
    convert_fxn = partial(convert_frame,new_shape=resize_to, pt_tensor=True, grayscale=grayscale)
    env = gym.make(env_name)
    state = env.reset()
    obs = convert_fxn(env.render('rgb_array'))
    for i in range(rollout_size):
        x0,x1,a = collect_one_data_point(env,obs,convert_fxn,action_space)
        obs = deepcopy(x1)
        x0x1 = torch.cat((x0,x1),dim=1)
        xs.append(x0x1[None,:])
        ys.append(a)
    x = torch.cat(xs)
    y = torch.stack(ys,dim=0)
    return x,y


        


# In[154]:


def create_tensor_dataset(size,
                       env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       rollout_size=50,
                       action_space = [0,1,2],
                       grayscale=False):
    
    xs = []
    ys = []
    num_rollouts = int(np.ceil(size / rollout_size))
    for rollout in range(num_rollouts):
        x,y = do_rollout(env_name, resize_to, rollout_size,
                         action_space,grayscale)
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs)
    y = torch.cat(ys)
    return TensorDataset(x,y)
    
        

