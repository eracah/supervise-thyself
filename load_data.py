
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
from create_data import create_tensor_dataset


# In[12]:


# LEFT, RIGHT, FORWARD, PICK_UP, DROP, TOGGLE, WAIT = range(7)

# action_space = [LEFT, RIGHT, FORWARD]


# In[3]:


def get_dirs(data_dir, dataset):
    basedir = Path(data_dir) / dataset
    tr_dir = basedir / "train"
    val_dir = basedir / "val"
    test_dir = basedir / "test"
    return tr_dir, val_dir, test_dir


# In[4]:


def get_data_loaders(data_dir = "../data",dataset = "MiniGrid-Empty-6x6-v0",
                     batch_size = 128,num_workers =4, grayscale=False):

    tr_dir, val_dir, test_dir = get_dirs(data_dir, dataset)
    transforms = [ToTensor()]
    if grayscale:
        transforms.insert(0,Grayscale())
    transforms = Compose([*transforms,Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    train_set = ImageFolder(tr_dir,transform=transforms)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 

    val_set = ImageFolder(val_dir,transform=transforms)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 
    
    test_set = ImageFolder(test_dir,transform=transforms)
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 
    return train_loader, val_loader, test_loader


# In[65]:


def get_tensor_data_loaders(env_name="MiniGrid-Empty-6x6-v0", resize_to = (64,64),
                            batch_size = 64, total_examples=1024):
    data_loader = partial(DataLoader,batch_size=batch_size,shuffle=True,num_workers=4)
    create_td = partial(create_tensor_dataset,env_name=env_name, resize_to=resize_to)
    tr_size = int(0.8*total_examples)
    
    val_size = test_size = int(0.1*total_examples)
    tr,val,test = create_td(tr_size),                create_td(val_size),                create_td(test_size)
    trl, vall, testl = data_loader(tr), data_loader(val), data_loader(test)
    return trl, vall, testl
    


# In[72]:


# tr,v,te = get_tensor_data_loaders(env_name="MiniGrid-Empty-8x8-v0",resize_to=(-1,-1),total_examples=1024)

# x,y = next(tr.__iter__())

# x.size()

# plt.imshow(x[1][0].data)
# plt.title(y[1])


