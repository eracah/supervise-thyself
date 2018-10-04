
# coding: utf-8

# In[1]:



# coding: utf-8

# In[1]:


import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid



import sys
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np
import json


from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
from collections import namedtuple
from itertools import product
import random

from comet_ml import Experiment


def setup_exp(args,exp_dir, exp_name):
    experiment = Experiment(api_key="kH9YI2iv3Ks9Hva5tyPW9FAbx",
                            project_name=str(exp_dir.parent) + "_" + exp_dir.name,
                            workspace="eracah")
    experiment.set_name(exp_name)
    model_dir = Path(".models") / exp_dir
    model_dir.mkdir(exist_ok=True,parents=True)
    return experiment, model_dir


# In[4]:


def parse_minigrid_env_name(name):
    return name.split("-")[2].split("x")[0]


# In[5]:


def setup_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    if "MiniGrid" in env_name:
        action_space = range(3)
        grid_size = env.grid_size - 2
        num_directions = 4
        tot_examples = grid_size**2 * num_directions
    else:
        action_space = list(range(env.action_space.n))
        grid_size = None
        num_directions = None
        tot_exampls = None
    num_actions = len(action_space)
    
    rng = np.random.RandomState(seed)
    random_policy = lambda x0: rng.randint(num_actions)
    return env, action_space, grid_size, num_directions, tot_examples, random_policy


# In[8]:


def classification_acc(logits,true):
    guess = torch.argmax(logits,dim=1)
    acc = (float(torch.sum(torch.eq(true,guess)).data) / true.size(0))
    return acc


# In[9]:



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
        


# In[18]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[19]:


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


    


# In[21]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[22]:


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


# In[19]:


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


