
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
import sys
from torch.utils.data import TensorDataset


# In[2]:


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

def env_dot_multistep(env,actions):
    '''generalizes gym function "step" that takes an action to one that takes a list of actions'''
    if not (isinstance(actions,list) or isinstance(actions,tuple)):
        actions = [actions]
            
    for action in actions:
        obs, reward, done, info = env.step(action)
    return obs, reward, done, info
    

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     %matplotlib inline

#     env = gym.make("MiniGrid-Empty-6x6-v0")
#     state = env.reset()
#     env_dot_multistep(env,actions=[2,2,1,2,2,0,2,1])
#     frame = convert_frame(env.render("rgb_array"))
#     plt.figure(1)
#     plt.imshow(frame)
#     obs, reward, done, info = env_dot_multistep(env,actions=[1,1,2,1,1])
#     frame = convert_frame(env.render("rgb_array"))
#     plt.figure(2)
#     plt.imshow(frame)



def collect_one_data_point(env,obs, convert_fxn, action_space):
    x0 = deepcopy(obs)
    action = action_space[np.random.choice(len(action_space))]
    obs, reward, done, info = env_dot_multistep(env,action)
    obs = convert_fxn(env.render("rgb_array"))
    a = torch.from_numpy(np.asarray(action_space.index(action)))
    x1 = deepcopy(obs)
    return x0,x1,a
    

# if __name__ == "__main__":
#     env = gym.make("MiniGrid-Empty-6x6-v0")
#     from matplotlib import pyplot as plt
#     %matplotlib inline
#     action_strings = ["move_up", "move_down","move_right","move_left"]
#     action_space = create_action_space(env,action_strings)
#     obs = env.reset()
            
        

def create_action_space(env,list_of_action_strings=["left", "right", "forward"]):
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
            
        
    



def do_rollout(env_name="MiniGrid-Empty-6x6-v0",
               resize_to = (64,64),
               rollout_size=128,
               action_strings = ["left","right","forward"],
               grayscale=False):
    
    corner_actions = ["left_or_up", "right_or_up", "left_or_down", "right_or_down"]
    xs = []
    ys = []
    convert_fxn = partial(convert_frame,new_shape=resize_to, pt_tensor=True, grayscale=grayscale)
    env = gym.make(env_name)
    # get rid of duplicates
    action_strings = list(set(action_strings))
    action_space = create_action_space(env,action_strings)
    label_list = action_strings + corner_actions
    state = env.reset()
    obs = convert_fxn(env.render('rgb_array'))
    for i in range(rollout_size):
        x0,x1,a = collect_one_data_point(env,obs,convert_fxn,action_space)
        obs = deepcopy(x1)
        x0x1 = torch.cat((x0,x1),dim=1)
        if torch.allclose(torch.eq(x0,x1).float(), torch.ones_like(x0)):
            in_corner, label_name = check_for_corner(env)
            if in_corner:
                a = torch.from_numpy(np.asarray(label_list.index(label_name)))
                

        xs.append(x0x1[None,:])
        ys.append(a)
    x = torch.cat(xs)
    y = torch.stack(ys,dim=0)
    return x,y, label_list


        

def create_tensor_dataset(size,
                       env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       rollout_size=128,
                       action_strings = ["left","right","forward"],
                       grayscale=False):
    
    xs = []
    ys = []
    if size < rollout_size:
        rollout_size = size
    num_rollouts = int(np.ceil(size / rollout_size))
    for rollout in range(num_rollouts):
        x,y, label_list = do_rollout(env_name, resize_to, rollout_size,
                         action_strings,grayscale)
        xs.append(x)
        ys.append(y)
    
    x = torch.cat(xs)[:size]
    y = torch.cat(ys)[:size]
    return TensorDataset(x,y), label_list
    
        

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     %matplotlib inline
#     action_strings = ["move_up", "move_down","move_right","move_left"]
#     x,y, label_list = do_rollout(rollout_size=10000,action_strings=action_strings, env_name="MiniGrid-Empty-8x8-v0")
#     print(x.size())
#     for i,im in enumerate(x.data):
#         plt.figure(i)
#         plt.imshow(im[0])
#         plt.title(label_list[y.data[i]])
    


# In[3]:


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
    create_td = partial(create_tensor_dataset,env_name=env_name, resize_to=resize_to,
                        action_strings=action_strings, rollout_size=rollout_size)
    tr_size = int(0.8*total_examples)
    
    val_size = test_size = int(0.1*total_examples)
    (tr,label_list),(val,_),(test,_) = create_td(tr_size),                create_td(val_size),                create_td(test_size)
    trl, vall, testl = data_loader(tr), data_loader(val), data_loader(test)
    return trl, vall, testl, label_list
    


# In[4]:


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    tr,v,te,ll = get_tensor_data_loaders(env_name="MiniGrid-Empty-6x6-v0",resize_to=(-1,-1),total_examples=1000)

    x,y = next(tr.__iter__())

    x.size()

    for i in range(64):
        plt.figure(i)
        plt.imshow(x[i][0].data)
        plt.title(ll[int(y[i].data)])

