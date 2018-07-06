
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
from create_data import convert_frame, collect_one_data_point


# In[2]:


def save_dataset( rollout_offset, data_dir = "../data",env_name="MiniGrid-Empty-6x6-v0",mode="train", 
                 num_rollouts=400, rollout_size=128, resize_to=(-1,-1), burn_in=0):
    ds_basedir = Path(data_dir) / env_name / mode
    if not ds_basedir.exists():
        try:
            ds_basedir.mkdir(parents=True)
        except:
            pass
    convert_fxn = partial(convert_frame,new_shape=resize_to, pt_tensor=False)
    env = gym.make(env_name)
    action_space = [env.actions.left.real, env.actions.right.real, env.actions.forward.real ]
    for rollout in range(num_rollouts):
        state = env.reset()
        obs = convert_fxn(env.render('rgb_array'))
        for i in range(rollout_size):
            x0,x1,a = collect_one_data_point(env,obs,convert_fxn,action_space)
            obs = deepcopy(x1)
            savedir = ds_basedir / str(a) 
            if not savedir.exists():
                try:
                    savedir.mkdir(parents=True)
                except:
                    pass
            ind += i + (rollout + rollout_offset)  * rollout_size
            savepath = savedir / (str(ind) + ".jpg")
            xc = Image.fromarray(np.concatenate((np.asarray(x0),np.asarray(x1))))
            xc.save(savepath)


# In[ ]:


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
    #parser.add_argument("--grayscale",action="store_true")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    
    sys.argv = tmp_argv
    #print(args)
    from multiprocessing import Pool

    p = Pool(args.processes)

    kwargs = {k:v for k,v in args.__dict__.items() if k != "processes" and k!="num_frames"}
    
    num_rollouts = math.ceil(args.num_frames / args.rollout_size)
    num_rollouts_per_process = math.ceil(num_rollouts / args.processes)
    kwargs["num_rollouts"] = num_rollouts_per_process

    f = partial(save_dataset,**kwargs)

    p.map(f,range(0,num_rollouts,num_rollouts_per_process))

