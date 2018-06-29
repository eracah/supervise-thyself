
# coding: utf-8

# In[19]:


import gym

from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from matplotlib import pyplot as plt
#%matplotlib inline

from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
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


# In[20]:


# LEFT, RIGHT, FORWARD, PICK_UP, DROP, TOGGLE, WAIT = range(7)

# action_space = [LEFT, RIGHT, FORWARD]


# In[21]:


def convert_frame(state, new_shape=(64,64), pt_tensor=False):
    state = Image.fromarray(state, 'RGB')

    if new_shape != (-1,-1):
        transforms = [Resize(new_shape)]
    else:
        transforms = []
    if pt_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    state = transforms(state)

    return state #Image


# In[22]:


def get_dirs(data_dir, dataset):
    basedir = Path(data_dir) / dataset
    tr_dir = basedir / "train"
    val_dir = basedir / "val"
    test_dir = basedir / "test"
    return tr_dir, val_dir, test_dir


# In[23]:


def get_data_loaders(data_dir = "../data",dataset = "MiniGrid-Empty-6x6-v0",batch_size = 128,num_workers =4):

    tr_dir, val_dir, test_dir = get_dirs(data_dir, dataset)
    save_out_frames = False

    for dir_ in [tr_dir, val_dir, test_dir]:
        if not dir_.exists():
            save_out_frames = True
    if save_out_frames:
        print("saving the raw frames. you gotta wait, dood")
        print("but actually this is not implemented yet, so save them manually dood")
        #save_frames(dataset,tr_dir,val_dir,test_dir, num_frames=num_frames,resize_to=resize_to)
    else:
        print("Already Saved")

    transforms = Compose([ToTensor(),Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    train_set = ImageFolder(tr_dir,transform=transforms)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 

    val_set = ImageFolder(val_dir,transform=transforms)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 
    
    test_set = ImageFolder(test_dir,transform=transforms)
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=num_workers) 
    return train_loader, val_loader, test_loader


# In[5]:


def make_batch(env_name="MiniGrid-Empty-6x6-v0",batch_size=128,
               resize_to=(-1,-1),device="cpu", burn_in=0, shuffle=True):
    #burn in is how many steps to take before saving frames
    x0 = []
    x1 = []
    a = []
    env = gym.make(env_name)
    action_space = [env.actions.left, env.actions.right, env.actions.forward ]
    state = env.reset()
    obs = convert_frame(env.render('rgb_array'),new_shape=resize_to,pt_tensor=True)
    
    for i in range(batch_size + burn_in):
        if i >= burn_in:
            x0.append(obs[None,:])
        action = np.random.choice(action_space)
        action = torch.tensor(int(action))[None]
        obs, reward, done, info = env.step(action)
        obs = convert_frame(env.render("rgb_array"),new_shape=resize_to,pt_tensor=True)
        if i >= burn_in:
            a.append(action)
            x1.append(obs[None,:])

    if shuffle:
        inds = np.arange(len(x0))
        np.random.shuffle(inds)
    return torch.cat(tuple(x0)).to(device)[inds],            torch.cat(tuple(x1)).to(device)[inds],            torch.cat(tuple(a)).to(device)[inds]
        
    
    
        


# In[6]:


def save_dataset( rollout_offset, data_dir = "../data",env_name="MiniGrid-Empty-6x6-v0",mode="train", num_rollouts=400, rollout_size=128, resize_to=(-1,-1), burn_in=0):
    ds_basedir = Path(data_dir) / env_name / mode
    if not ds_basedir.exists():
        try:
            ds_basedir.mkdir(parents=True)
        except:
            pass
    env = gym.make(env_name)
    action_space = [env.actions.left.real, env.actions.right.real, env.actions.forward.real ]
    for rollout in range(num_rollouts):
        state = env.reset()
        obs = convert_frame(env.render('rgb_array'),new_shape=resize_to)

        for i in range(rollout_size + burn_in):
            if i >= burn_in:
                x0 = deepcopy(obs)
            action = np.random.choice(action_space)
            #action = torch.tensor(int(action))[None]
            obs, reward, done, info = env.step(action)
            obs = convert_frame(env.render("rgb_array"),new_shape=resize_to)
            if i >= burn_in:
                a = deepcopy(action)
                x1 = deepcopy(obs)
                # save jpg of concantenated x0 and x1 with directory name as action
                savedir = ds_basedir / str(a) 
                if not savedir.exists():
                    try:
                        savedir.mkdir(parents=True)
                    except:
                        pass
                ind = i - burn_in
                ind += (rollout + rollout_offset)  * rollout_size
                savepath = savedir / (str(ind) + ".jpg")


                # concat images
                xc = Image.fromarray(np.concatenate((np.asarray(x0),np.asarray(x1))))
                xc.save(savepath)
        
        
        
        
        
    


# In[7]:


if __name__ == "__main__":
    tmp_argv = deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--rollout_size",type=int,default=10)
    parser.add_argument("--num_frames",type=int,default=400)
    parser.add_argument("--burn_in",type=int,default=50)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[64,64])
    parser.add_argument("--mode",type=str, default='train')
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="../data")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    
    sys.argv = tmp_argv

    from multiprocessing import Pool

    p = Pool(args.processes)

    kwargs = {k:v for k,v in args.__dict__.items() if k != "processes" and k!="num_frames"}
    
    num_rollouts = math.ceil(args.num_frames / args.rollout_size)
    num_rollouts_per_process = math.ceil(num_rollouts / args.processes)
    kwargs["num_rollouts"] = num_rollouts_per_process

    f = partial(save_dataset,**kwargs)

    p.map(f,range(0,num_rollouts,num_rollouts_per_process))

