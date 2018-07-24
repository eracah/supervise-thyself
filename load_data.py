
# coding: utf-8

# In[177]:


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


# In[225]:


# env = gym.make("MsPacman-v0")
# env.reset()
# for i in range(100):
#     obs,_,_,_ = env.step(0)

# a=convert_frame(obs,resize_to=(84,84))

# plt.imshow(a,cmap="gray")


# In[226]:


def convert_frame(obs, resize_to=(84,84),to_tensor=False):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5],[0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if not to_tensor:
        frame = np.asarray(frame)
    
    return frame


# In[135]:


def convert_frames(frames,resize_to=(64,64),to_tensor=False):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor)
    return torch.stack([convert(frame) for frame in frames])
        


# In[136]:


class DataCreator(object):
    def __init__(self,to_tensor=False,env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       rollout_size=128,
                       action_space=range(3)):
        self.env_name = env_name
        tmp_env = gym.make(self.env_name)
        self.resize_to = resize_to
        self.rollout_size = rollout_size
        
        self.action_space = action_space
        self.to_tensor = to_tensor
        self.convert = partial(convert_frame, resize_to = self.resize_to,to_tensor=self.to_tensor)

    
    def collect_one_data_point(self,env,obs):
        x0 = deepcopy(obs)
        action = np.random.choice(self.action_space)
        obs, reward, done, info = env.step(action)
        obs = env.render("rgb_array")
        obs = self.convert(obs)
        a = torch.tensor([int(action)])
        reward = torch.tensor([reward])
        x1 = deepcopy(obs)
        return x0,x1,a,reward, done

    def rollout_iterator(self):
        env = gym.make(self.env_name)
        state = env.reset()
        obs = env.render('rgb_array')
        obs = self.convert(obs)
        done = False
        while not done:
            x0,x1,a,reward, done= self.collect_one_data_point(env,obs)
            obs = deepcopy(x1)

            yield x0,x1,a,reward, done
    
        


# In[8]:


def plot_test(x0,x1,y,r, label_list ):
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    for i,(im0,im1,y) in enumerate(zip(x0,x1,y)):
        plt.figure(i)
        plt.clf()
        sp1 = plt.subplot(1,2,1)
        sp1.imshow(im0[0])
        sp2 = plt.subplot(1,2,2)
        sp2.imshow(im1[0])
        plt.title(label_list[y])
    


# In[9]:


if __name__ == "__main__":
    action_strings = ["move_up", "move_down","move_right","move_left"]
    dc = DataCreator(rollout_size=20,to_tensor=False)
    x0s,x1s,ys, rs = dc.do_rollout()
    plot_test(np.transpose(x0s,axes=(0,3,1,2)),np.transpose(x1s,axes=(0,3,1,2)),ys, rs, label_list=["left","right","forward"])
    


# In[10]:


if __name__ == "__main__":    
    dc = DataCreator(rollout_size=20,to_tensor=True)
    x0s,x1s,ys, rs = dc.do_rollout()
    plot_test(x0s,x1s,ys, rs, label_list=["left","right","forward"])
    
    
    #x0 = convert_frames(np.asarray(x0s),to_tensor=True,resize_to=(-1,-1))
    #x1 = convert_frames(np.asarray(x1s),to_tensor=True,resize_to=(-1,-1))
    #from matplotlib import pyplot as plt


    


# In[11]:


def get_tensor_data_loaders(env_name="MiniGrid-Empty-6x6-v0", resize_to = (64,64),
                            batch_size = 64, total_examples=1024,
                            action_strings=["move_left","move_right","move_down", "move_up"],
                            rollout_size=128):
    if total_examples < 10:
        sys.stderr.write("You cannot have fewer than 10 total examples\n")
        sys.stderr.write("because that would result in a test and val set of 0 examples\n")
        sys.stderr.write("Proceeding with 10 examples...\n")
        total_examples = 10
    data_loader = partial(DataLoader,batch_size=batch_size,shuffle=True,num_workers=4)
    dc = DataCreator(env_name=env_name,
                     resize_to = resize_to,
                     action_strings=action_strings,
                     rollout_size=rollout_size, to_tensor=True)
    
    tr_size = int(0.8*total_examples)
    
    val_size = test_size = int(0.1*total_examples)
    tr,val,test = dc.create_tensor_dataset(tr_size),                dc.create_tensor_dataset(val_size),                dc.create_tensor_dataset(test_size)
    trl, vall, testl = data_loader(tr), data_loader(val), data_loader(test)
    return trl, vall, testl, dc.label_list
    


# In[1]:


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    tr,v,te,ll = get_tensor_data_loaders(env_name="MiniGrid-Empty-6x6-v0",resize_to=(64,64),total_examples=1000)

    x0,x1,y,r = next(tr.__iter__())
    print(x0.size())
    plot_test(x0,x1,y, r, ll)

