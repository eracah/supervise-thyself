
# coding: utf-8

# In[32]:


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
import sys
from torch.utils.data import TensorDataset


# In[33]:


env= gym.make("MiniGrid-Empty-6x6-v0")


# In[34]:


def convert_frame(obs, resize_to=(64,64)):
    pil_image = Image.fromarray(obs, 'RGB')
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    
    return frame

def env_dot_multistep(env,actions):
    '''generalizes gym function "step" that takes an action to one that takes a list of actions'''
    if not (isinstance(actions,list) or isinstance(actions,tuple)):
        actions = [actions]
            
    for action in actions:
        obs, reward, done, info = env.step(action)
    return obs, reward, done, info


            
def create_action_space_minigrid(env,list_of_action_strings=["left", "right", "forward"]):
    actions_dict = {k:env.actions[k].real for k in ["forward", "right", "left"]}
    # these move agent in that direction without rotating them at all
    #actions_dict["backward"] = [actions_dict[k] for k in ["right","right","forward","right","right"]]
    actions_dict["move_right"] = [actions_dict[k] for k in ["forward"]]
    actions_dict["move_left"] = [actions_dict[k] for k in ["right","right","forward","right","right"]]
    actions_dict["move_down"] = [actions_dict[k] for k in ["right","forward", "left"]]
    actions_dict["move_up"] = [actions_dict[k] for k in ["left","forward", "right"]]
    action_space = [actions_dict[k] for k in list_of_action_strings]
    return action_space

# env = gym.make("MiniGrid-Empty-6x6-v0")
# action_strings = ["move_" + k for k in ["left","right","up","down"]]
# a = create_action_space(env,action_strings)

def check_for_corner(env):
    grid_size = env.grid_size
    agent_pos = tuple(env.agent_pos)
    in_corner = False
    new_label_name = ""
    if agent_pos in [(1, 1), (grid_size - 2, 1), (1, grid_size - 2), (grid_size - 2, grid_size - 2)]:
        in_corner = True
        if agent_pos == (1,1):
            new_label_name = "left_or_up"
        elif agent_pos == (grid_size - 2,1):
            new_label_name = "right_or_up"
        elif agent_pos == (1, grid_size - 2):
            new_label_name = "left_or_down"
        else:
            new_label_name = "right_or_down"
    return in_corner, new_label_name
            
        
    


# In[35]:


class DataCreator(object):
    def __init__(self,env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       rollout_size=128,
                       action_strings = ["move_up", "move_down","move_right","move_left"]):
        self.env_name = env_name
        tmp_env = gym.make(self.env_name)
        self.resize_to = resize_to
        self.rollout_size = rollout_size
        self.action_strings = action_strings
        self.action_space = create_action_space_minigrid(env=tmp_env,
                                                         list_of_action_strings=self.action_strings)
        if "MiniGrid" in env_name:
            corner_actions = ["left_or_up", "right_or_up", "left_or_down", "right_or_down"]
        else:
            corner_actions = []
        
        self.label_list = deepcopy(action_strings) + corner_actions
        print(self.label_list)
        self.convert_fxn = partial(convert_frame, resize_to = self.resize_to)

    
    def collect_one_data_point(self,env,obs):
        x0 = deepcopy(obs)
        action = self.action_space[np.random.choice(len(self.action_space))]
        obs, reward, done, info = env_dot_multistep(env,action)
        obs = self.convert_fxn(env.render("rgb_array"))
        a = torch.tensor([self.action_space.index(action)])
        reward = torch.tensor([reward])
        x1 = deepcopy(obs)
        return x0,x1,a,reward

    def rollout_iterator(self):
        env = gym.make(self.env_name)
        state = env.reset()
        obs = self.convert_fxn(env.render('rgb_array'))
        for i in range(self.rollout_size):
            x0,x1,a,reward = self.collect_one_data_point(env,obs)
            obs = deepcopy(x1)
            if torch.allclose(torch.eq(x0,x1).float(), torch.ones_like(x0)):
                in_corner, label_name = check_for_corner(env)
                if in_corner:
                    a = torch.tensor([self.label_list.index(label_name)])
            yield x0,x1,a,reward

    

    def do_rollout(self):      
        rollouts = [(x0[None,:],x1[None,:],a,reward) for x0,x1,a,reward in self.rollout_iterator()]
        x0,x1,y,r = [torch.cat(arr) for arr in zip(*rollouts)]

        return x0, x1, y, r

    def create_tensor_dataset(self,size):
        num_rollouts = int(np.ceil(size / self.rollout_size))
        rollouts = [self.do_rollout() for rollout in range(num_rollouts)]
        x0,x1,y,r = [torch.cat(list_)[:size] for list_ in zip(*rollouts)]

        return TensorDataset(x0, x1, y,r)
    
        


# In[36]:


def plot_test(x0,x1,y,r, label_list ):
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    for i,(im0,im1,y) in enumerate(zip(x0.data,x1.data,y)):
        plt.figure(i)
        plt.clf()
        sp1 = plt.subplot(1,2,1)
        sp1.imshow(im0[0])
        sp2 = plt.subplot(1,2,2)
        sp2.imshow(im1[0])
        plt.title(label_list[y])
    


# In[37]:


# if __name__ == "__main__":
#     action_strings = ["move_up", "move_down","move_right","move_left"]
#     dc = DataCreator(rollout_size=10)
#     x0,x1,y, r = dc.do_rollout()
#     plot_test(x0,x1,y, r, dc.label_list)

    


# In[38]:


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
                     rollout_size=rollout_size)
    
    tr_size = int(0.8*total_examples)
    
    val_size = test_size = int(0.1*total_examples)
    tr,val,test = dc.create_tensor_dataset(tr_size),                dc.create_tensor_dataset(val_size),                dc.create_tensor_dataset(test_size)
    trl, vall, testl = data_loader(tr), data_loader(val), data_loader(test)
    return trl, vall, testl, dc.label_list
    


# In[39]:


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     %matplotlib inline
#     tr,v,te,ll = get_tensor_data_loaders(env_name="MiniGrid-Empty-6x6-v0",resize_to=(-1,-1),total_examples=1000)

#     x0,x1,y = next(tr.__iter__())

#     x0.size()

#     for i in range(64):
#         plt.figure(i)
#         plt.imshow(x0[i][0].data)
#         plt.title(ll[int(y[i].data)])


# In[40]:


def fill_replay_buffer(buffer,size, rollout_size=128,
                       env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       action_strings = ["move_left","move_right","move_down", "move_up"]):
    #fills replay buffer with size examples
    num_rollouts = int(np.ceil(size / rollout_size))
    dc = DataCreator(env_name=env_name,
                     resize_to = resize_to,
                     action_strings=action_strings,
                     rollout_size=rollout_size)
    for rollout in range(num_rollouts):
        for i, (x0,x1,a,r) in enumerate(dc.rollout_iterator()):
            if i > size:
                break
            buffer.push(state=x0,action=a,next_state=x1,reward=r)
    return buffer, dc.label_list
    

    
    
    


# In[41]:


# if __name__ == "__main__":
#     from replay_buffer import ReplayMemory

#     rm  = ReplayMemory(batch_size=10,capacity=20)

#     rm, label_list = fill_replay_buffer(rm,21)

#     x0,a,x1,r = rm.sample(batch_size=10)

#     plot_test(x0,x1,a,r, label_list )

